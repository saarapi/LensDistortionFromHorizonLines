"""
Este script estima los parámetros del modelo de división (d1, d2, cx, cy)
a partir de una imagen de calibración formada por horizontes acumulados.

Después:
1. Convierte esos parámetros al modelo fisheye de OpenCV mediante un ajuste numérico.
2. Corrige la propia imagen de calibración.
3. Corrige otra imagen distinta tomada con la misma cámara.
4. Adapta cx, cy si cambia la resolución entre ambas imágenes.
5. Guarda imágenes, comparativas visuales y un JSON reutilizable con los parámetros.
"""

from pathlib import Path
import sys
import json

import cv2
import numpy as np


# Añadir la raíz del proyecto al path para poder importar desde src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.horizon_lens_distortion.utils.third_party import (  # noqa: E402
    add_lens_distortion_to_path,
)
from src.horizon_lens_distortion.undistortion.opencv_adapter import (  # noqa: E402
    generate_opencv_distortion_coefs,
    undistort_with_matching_opencv_direct_centered,
    match_and_undistort_with_opencv_direct_centered,
    scale_cx_cy,
    DEFAULT_OPENCV_FOCAL_LENGTH,
)
from src.horizon_lens_distortion.evaluation.visualization import (  # noqa: E402
    show_three_images,
    show_two_images,
)

add_lens_distortion_to_path()

from lens_distortion_module import process_image_file  # type: ignore  # noqa: E402


# ==========================
# CONFIGURACIÓN
# ==========================

# Imagen DISTORSIONADA usada para aprender los parámetros
CALIB_IMAGE_NAME = PROJECT_ROOT / "data" / "raw" / "images" / "horizontes_GT_submuestreo_10_dist.png"

# Imagen GT opcional para comparar la calibración
CALIB_GT_IMAGE_NAME = ""

# Imagen DISTORSIONADA diferente, tomada con la misma cámara
APPLY_IMAGE_NAME = PROJECT_ROOT / "data" / "raw" / "images" / "frame_001_ejemplo_dist.png"

# Control del módulo C++
WRITE_INTERMEDIATES = False
WRITE_OUTPUT = False

# Parámetro del estimador
DISTANCE_POINT_LINE_MAX_HOUGH = 10.0


def main():
    # ==========================
    # RUTAS DE RESULTADOS
    # ==========================
    images_dir = PROJECT_ROOT / "results" / "images" / "learn_and_apply"
    metrics_dir = PROJECT_ROOT / "results" / "metrics" / "learn_and_apply"
    plots_dir = PROJECT_ROOT / "results" / "plots" / "learn_and_apply"

    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # CARGAR IMAGEN DE CALIBRACIÓN
    # ==========================
    calib_img = cv2.imread(str(CALIB_IMAGE_NAME), cv2.IMREAD_COLOR)
    if calib_img is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen de calibración: {CALIB_IMAGE_NAME}"
        )

    h0, w0 = calib_img.shape[:2]
    max_r0 = w0 / 2.0 + 10.0

    # ==========================
    # ESTIMAR PARÁMETROS CON C++
    # ==========================
    _, res_dict = process_image_file(
        str(CALIB_IMAGE_NAME),
        int(w0),
        int(h0),
        str(metrics_dir.resolve()),
        write_intermediates=WRITE_INTERMEDIATES,
        write_output=WRITE_OUTPUT,
        distance_point_line_max_hough=DISTANCE_POINT_LINE_MAX_HOUGH,
    )

    d1 = float(res_dict["d1"])
    d2 = float(res_dict["d2"])
    cx = float(res_dict["cx"])
    cy = float(res_dict["cy"])

    # ==========================
    # CORREGIR IMAGEN DE CALIBRACIÓN
    # ==========================
    calib_corrected, k_array, k_base = match_and_undistort_with_opencv_direct_centered(
        img=calib_img,
        division_coef_d1=d1,
        division_coef_d2=d2,
        cx=cx,
        cy=cy,
        max_r=max_r0,
    )

    # ==========================
    # CARGAR IMAGEN A APLICAR
    # ==========================
    apply_img = cv2.imread(str(APPLY_IMAGE_NAME), cv2.IMREAD_COLOR)
    if apply_img is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen a corregir: {APPLY_IMAGE_NAME}"
        )

    h1, w1 = apply_img.shape[:2]

    # ==========================
    # ESCALAR CX, CY SI CAMBIA LA RESOLUCIÓN
    # ==========================
    cx1, cy1 = scale_cx_cy(cx, cy, w0, h0, w1, h1)

    # ==========================
    # REUTILIZAR LOS k APRENDIDOS EN LA SEGUNDA IMAGEN
    # ==========================
    k_apply_matrix = np.array(
        [
            [DEFAULT_OPENCV_FOCAL_LENGTH, 0.0, cx1],
            [0.0, DEFAULT_OPENCV_FOCAL_LENGTH, cy1],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    apply_corrected, k_used = undistort_with_matching_opencv_direct_centered(
        img=apply_img,
        camera_matrix=k_apply_matrix,
        opencv_k1=float(k_array[0]),
        opencv_k2=float(k_array[1]),
        opencv_k3=float(k_array[2]),
        opencv_k4=float(k_array[3]),
    )

    # ==========================
    # NOMBRES DE SALIDA
    # ==========================
    calib_tag = CALIB_IMAGE_NAME.stem
    apply_tag = APPLY_IMAGE_NAME.stem

    calib_distorted_out = images_dir / f"{calib_tag}_distorsionada.png"
    calib_corrected_out = images_dir / f"{calib_tag}_corregida.png"

    apply_distorted_out = images_dir / f"{apply_tag}_distorsionada.png"
    apply_corrected_out = images_dir / f"{apply_tag}_corregida.png"

    learned_params_json = metrics_dir / f"learn_{calib_tag}__apply_{apply_tag}.json"

    calib_plot_name = f"{calib_tag}_comparison.png"
    apply_plot_name = f"{apply_tag}_comparison.png"

    # ==========================
    # GUARDAR IMÁGENES
    # ==========================
    cv2.imwrite(str(calib_distorted_out), calib_img)
    cv2.imwrite(str(calib_corrected_out), calib_corrected)
    cv2.imwrite(str(apply_distorted_out), apply_img)
    cv2.imwrite(str(apply_corrected_out), apply_corrected)

    # ==========================
    # PLOTS
    # ==========================
    calib_gt_ok = False
    calib_gt_resolved = ""

    if CALIB_GT_IMAGE_NAME:
        calib_gt_path = Path(CALIB_GT_IMAGE_NAME)
        if not calib_gt_path.is_absolute():
            calib_gt_path = PROJECT_ROOT / calib_gt_path

        if calib_gt_path.exists():
            gt = cv2.imread(str(calib_gt_path), cv2.IMREAD_COLOR)
            if gt is None:
                raise FileNotFoundError(
                    f"No se pudo cargar la GT de calibración: {calib_gt_path}"
                )

            show_three_images(
                img_distorted_bgr=calib_img,
                img_gt_bgr=gt,
                img_corrected_bgr=calib_corrected,
                output_path=plots_dir,
                filename=calib_plot_name,
            )
            calib_gt_ok = True
            calib_gt_resolved = str(calib_gt_path.resolve())

    if not calib_gt_ok:
        show_two_images(
            img_distorted_bgr=calib_img,
            img_corrected_bgr=calib_corrected,
            output_path=plots_dir,
            filename=calib_plot_name,
        )

    show_two_images(
        img_distorted_bgr=apply_img,
        img_corrected_bgr=apply_corrected,
        output_path=plots_dir,
        filename=apply_plot_name,
    )

    # ==========================
    # JSON FINAL
    # ==========================
    learned_out = {
        "calibration_image": str(CALIB_IMAGE_NAME.resolve()),
        "apply_image": str(APPLY_IMAGE_NAME.resolve()),
        "calib_size": {
            "w": int(w0),
            "h": int(h0),
        },
        "apply_size": {
            "w": int(w1),
            "h": int(h1),
        },
        "division": {
            "d1": float(d1),
            "d2": float(d2),
            "cx": float(cx),
            "cy": float(cy),
        },
        "opencv": {
            "k_array": [float(x) for x in k_array.tolist()],
            "K_base": k_base.tolist(),
        },
        "apply_used_K": {
            "K_apply": k_apply_matrix.tolist(),
            "output_K": k_used.tolist(),
        },
        "outputs": {
            "calib_distorted": str(calib_distorted_out.resolve()),
            "calib_corrected": str(calib_corrected_out.resolve()),
            "apply_distorted": str(apply_distorted_out.resolve()),
            "apply_corrected": str(apply_corrected_out.resolve()),
            "calib_plot": str((plots_dir / calib_plot_name).resolve()),
            "apply_plot": str((plots_dir / apply_plot_name).resolve()),
            "calib_gt": calib_gt_resolved,
        },
    }

    with open(learned_params_json, "w", encoding="utf-8") as f:
        json.dump(learned_out, f, indent=2, ensure_ascii=False)

    # ==========================
    # INFO POR CONSOLA
    # ==========================
    print(f"Salida imágenes: {images_dir}")
    print(f"Salida métricas: {metrics_dir}")
    print(f"Salida plots: {plots_dir}")
    print(f"Parámetros guardados: {learned_params_json}")
    print(f"Imagen corregida (apply): {apply_corrected_out}")


if __name__ == "__main__":
    main()