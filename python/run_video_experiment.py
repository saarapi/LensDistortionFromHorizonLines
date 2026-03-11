#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import shutil

import cv2

# ============================================================
#  IMPORTS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.horizon_lens_distortion.video.io import (
    get_videos_to_process,
    get_calib_dir_for_video,
)

from src.horizon_lens_distortion.video.sampling import (
    extract_sampled_frames_ffmpeg,
)

from src.horizon_lens_distortion.video.calibration_cache import (
    precompute_calibration_estimations_for_level,
    precompute_level_geometry_for_video,
)

from src.horizon_lens_distortion.distortion.division_model import (
    distort_image_division_model,
)

from src.horizon_lens_distortion.undistortion.opencv_adapter import (
    match_and_undistort_with_opencv,
)

from src.horizon_lens_distortion.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_cwssim_from_arrays,
)

from src.horizon_lens_distortion.evaluation.plots import (
    make_curves_from_results,
    save_multiline_plot,
)

from src.horizon_lens_distortion.utils.third_party import (
    add_lens_distortion_to_path,
)

# Añadir el módulo externo al path antes de importarlo
add_lens_distortion_to_path()

from lens_distortion_module import process_image_file  # type: ignore  # noqa: E402


# ============================================================
# CONFIGURACIÓN
# ============================================================

VIDEO_DIR = (PROJECT_ROOT / "data" / "raw" / "videos").resolve()
CALIB_DIR = (PROJECT_ROOT / "data" / "calibration").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "results" / "output_videos").resolve()

# Para pruebas: un solo vídeo
INPUT_VIDEO_FILES = [
    "MVI_0788_VIS_OB.avi",
]

# Para procesar todos, deja INPUT_VIDEO_FILES = []
VIDEO_GLOB = "MVI_*_VIS_OB.avi"

SAMPLE_EVERY_N_FRAMES = 20
NUM_SAMPLED_FRAMES = 15

DISTORTION_LEVELS = ["low", "mid", "high"]

GT_PARAMS_BY_LEVEL = {
    "low":  {"d1": -2e-07, "d2": 0.0, "cx": None, "cy": None},
    "mid":  {"d1": -5e-07, "d2": 0.0, "cx": None, "cy": None},
    "high": {"d1": -1e-06, "d2": 0.0, "cx": None, "cy": None},
}

WRITE_INTERMEDIATES = False
WRITE_OUTPUT = False
DISTANCE_POINT_LINE_MAX_HOUGH = 10.0


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = get_videos_to_process(
        VIDEO_DIR,
        INPUT_VIDEO_FILES,
        VIDEO_GLOB,
    )

    print("Vídeos a procesar:")
    for v in videos:
        print(f" - {v.name}")

    k_cache = {}

    for video_path in videos:
        video_id = video_path.stem
        calib_dir = get_calib_dir_for_video(video_path, CALIB_DIR)

        print(f"\nVIDEO: {video_id}")
        print(f"Carpeta de calibración: {calib_dir}")

        out_dir_video = OUTPUT_DIR / video_id
        out_dir_video.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # EXTRAER FRAMES
        # =====================================================

        extracted_dir = out_dir_video / "_extracted_frames"

        extracted = extract_sampled_frames_ffmpeg(
            video_path,
            extracted_dir,
            every_n=SAMPLE_EVERY_N_FRAMES,
            max_frames=NUM_SAMPLED_FRAMES,
        )

        if len(extracted) == 0:
            raise RuntimeError(f"No se extrajeron frames para {video_id}")

        first_frame = cv2.imread(str(extracted[0][0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise RuntimeError(f"No se pudo leer el primer frame extraído: {extracted[0][0]}")

        h0, w0 = first_frame.shape[:2]

        # =====================================================
        # PRECOMPUTAR GEOMETRÍA POR NIVEL (1 vez por vídeo)
        # =====================================================

        geometry_by_level = {}

        for level in DISTORTION_LEVELS:
            print(f"[{video_id}] Precalculando geometría GT para nivel '{level}'...")

            geometry_by_level[level] = precompute_level_geometry_for_video(
                frame_h=h0,
                frame_w=w0,
                level=level,
                gt_params_by_level=GT_PARAMS_BY_LEVEL,
                k_cache=k_cache,
            )

        # =====================================================
        # PRECOMPUTAR ESTIMACIONES C++ (1 vez por vídeo y nivel)
        # =====================================================

        estimations_by_level = {}

        for level in DISTORTION_LEVELS:
            print(f"[{video_id}] Precalculando estimaciones C++ para nivel '{level}'...")

            estimations_by_level[level] = precompute_calibration_estimations_for_level(
                calib_dir,
                level,
                out_dir_video,
                k_cache,
                process_image_file=process_image_file,
                write_intermediates=WRITE_INTERMEDIATES,
                write_output=WRITE_OUTPUT,
                distance_point_line_max_hough=DISTANCE_POINT_LINE_MAX_HOUGH,
            )

        # =====================================================
        # PROCESAR FRAMES
        # =====================================================

        for png_path, sample_idx, approx_frame_idx in extracted:
            frame = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"No se pudo leer frame extraído: {png_path}")

            frame_tag = f"frame_{sample_idx:03d}"

            out_dir_frame = out_dir_video / frame_tag
            out_dir_frame.mkdir(parents=True, exist_ok=True)

            shutil.copy2(png_path, out_dir_frame / f"{frame_tag}_original.png")

            all_results_by_level = {}

            for level in DISTORTION_LEVELS:
                geometry = geometry_by_level[level]
                estimations = estimations_by_level[level]

                # =========================
                # DISTORSIÓN GT
                # =========================

                distorted, map_x, map_y, _, _ = distort_image_division_model(
                    frame,
                    geometry["d1_gt"],
                    geometry["d2_gt"],
                    cx=geometry["cx_full"],
                    cy=geometry["cy_full"],
                )

                xmin = geometry["bbox"]["xmin"]
                ymin = geometry["bbox"]["ymin"]
                xmax = geometry["bbox"]["xmax"]
                ymax = geometry["bbox"]["ymax"]

                distorted_cropped = distorted[ymin:ymax + 1, xmin:xmax + 1]

                # =========================
                # CORRECCIÓN GT
                # =========================

                corrected_gt, _, _ = match_and_undistort_with_opencv(
                    distorted_cropped,
                    geometry["d1_gt"],
                    geometry["d2_gt"],
                    geometry["cx_cropped"],
                    geometry["cy_cropped"],
                    geometry["max_r_gt"],
                )

                results = []

                for est in estimations:
                    cx_scaled = est["cx"] * geometry["cropped_size"]["w"] / est["w"]
                    cy_scaled = est["cy"] * geometry["cropped_size"]["h"] / est["h"]

                    corrected_est, _, _ = match_and_undistort_with_opencv(
                        distorted_cropped,
                        est["d1"],
                        est["d2"],
                        cx_scaled,
                        cy_scaled,
                        est["max_r"],
                    )

                    psnr = compute_psnr(corrected_gt, corrected_est)
                    ssim = compute_ssim(corrected_gt, corrected_est)
                    cw = compute_cwssim_from_arrays(corrected_gt, corrected_est)

                    results.append(
                        {
                            "frames": est["frames"],
                            "psnr": psnr,
                            "ssim": ssim,
                            "cwssim": cw,
                        }
                    )

                all_results_by_level[level] = results

            # =========================
            # PLOTS
            # =========================

            psnr_curves = make_curves_from_results(all_results_by_level, "psnr")
            ssim_curves = make_curves_from_results(all_results_by_level, "ssim")
            cw_curves = make_curves_from_results(all_results_by_level, "cwssim")

            title = f"{video_id} | {frame_tag}"

            save_multiline_plot(
                psnr_curves,
                y_label="PSNR",
                title=title,
                out_path=out_dir_frame / "plot_psnr.png",
                gt_params_by_level=GT_PARAMS_BY_LEVEL,
            )

            save_multiline_plot(
                ssim_curves,
                y_label="SSIM",
                title=title,
                out_path=out_dir_frame / "plot_ssim.png",
                gt_params_by_level=GT_PARAMS_BY_LEVEL,
            )

            save_multiline_plot(
                cw_curves,
                y_label="CWSSIM",
                title=title,
                out_path=out_dir_frame / "plot_cwssim.png",
                gt_params_by_level=GT_PARAMS_BY_LEVEL,
            )

            print(f"{video_id} {frame_tag} OK")


if __name__ == "__main__":
    main()