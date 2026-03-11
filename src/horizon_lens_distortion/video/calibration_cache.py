import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.horizon_lens_distortion.distortion.division_model import (
    distort_image_division_model,
)
from src.horizon_lens_distortion.distortion.cropping import (
    crop_largest_valid_rect_no_resize,
)
from src.horizon_lens_distortion.undistortion.opencv_adapter import (
    match_opencv_distortion_to_undistortion_model,
)


def maybe_write_json(path: Path, data: Any) -> None:
    """
    Guarda un JSON creando antes las carpetas necesarias.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_calib_images_for_level(calib_dir: Path, level: str) -> List[Tuple[Path, int]]:
    """
    Lee calibraciones con formato:
      <NUM>_calib_<N>_frames_<level>.png
    dentro de calib_dir.
    """
    calib_dir = calib_dir.resolve()
    if not calib_dir.exists():
        raise FileNotFoundError(f"No existe calib_dir: {calib_dir}")

    pattern = f"*calib_*_frames_{level}.png"
    files = sorted(calib_dir.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No se encontraron calibraciones con patrón {pattern} en {calib_dir}"
        )

    out: List[Tuple[Path, int]] = []
    for f in files:
        m = re.search(r"_calib_(\d+)_frames_", f.name)
        if m is None:
            continue
        out.append((f, int(m.group(1))))

    if len(out) == 0:
        raise FileNotFoundError(
            f"Se encontraron {pattern}, pero ninguno encajó con regex _calib_(N)_frames_."
        )

    out.sort(key=lambda x: x[1])
    return out


def make_division_to_opencv_cache_key(
    d1: float,
    d2: float,
    max_r: float,
) -> Tuple[float, float, float]:
    """
    Construye una clave robusta para cachear la conversión
    division model -> coeficientes OpenCV.
    """
    return (
        round(float(d1), 16),
        round(float(d2), 16),
        round(float(max_r), 8),
    )


def get_or_compute_k_array(
    d1: float,
    d2: float,
    max_r: float,
    k_cache: Dict[Tuple[float, float, float], np.ndarray],
) -> np.ndarray:
    """
    Devuelve k_array desde caché o lo calcula si aún no existe.
    """
    key = make_division_to_opencv_cache_key(d1, d2, max_r)
    if key not in k_cache:
        k_cache[key] = match_opencv_distortion_to_undistortion_model(
            lambda r: 1.0 / (1.0 + d1 * r * r + d2 * r * r * r * r),
            max_r,
        )
    return k_cache[key]


def precompute_calibration_estimations_for_level(
    calib_dir: Path,
    level: str,
    output_dir: Path,
    k_cache: Dict[Tuple[float, float, float], np.ndarray],
    *,
    process_image_file,
    write_intermediates: bool,
    write_output: bool,
    distance_point_line_max_hough: float,
) -> List[Dict[str, Any]]:
    """
    Estima UNA SOLA VEZ los parámetros de todas las calibraciones de un vídeo
    para un nivel dado, y además precalcula su k_array de OpenCV.

    Devuelve una lista ordenada por número de frames:
      {
        "frames": ...,
        "d1": ...,
        "d2": ...,
        "cx": ...,
        "cy": ...,
        "w": ...,
        "h": ...,
        "max_r": ...,
        "k_array": [...]
      }
    """
    calib_list = list_calib_images_for_level(calib_dir, level)
    estimations: List[Dict[str, Any]] = []

    for calib_path, frames_n in calib_list:
        calib_img = cv2.imread(str(calib_path), cv2.IMREAD_COLOR)
        if calib_img is None:
            raise FileNotFoundError(f"No se pudo cargar calibración: {calib_path.resolve()}")

        hC, wC = calib_img.shape[:2]
        max_rC = wC / 2.0 + 10.0

        _, res_dict = process_image_file(
            str(calib_path),
            int(wC),
            int(hC),
            str(output_dir),
            write_intermediates=write_intermediates,
            write_output=write_output,
            distance_point_line_max_hough=distance_point_line_max_hough,
        )

        d1_est = float(res_dict["d1"])
        d2_est = float(res_dict["d2"])
        cx_est = float(res_dict["cx"])
        cy_est = float(res_dict["cy"])

        k_array_est = get_or_compute_k_array(
            d1=d1_est,
            d2=d2_est,
            max_r=float(max_rC),
            k_cache=k_cache,
        )

        estimations.append(
            {
                "frames": float(frames_n),
                "d1": d1_est,
                "d2": d2_est,
                "cx": cx_est,
                "cy": cy_est,
                "w": int(wC),
                "h": int(hC),
                "max_r": float(max_rC),
                "k_array": [float(x) for x in k_array_est.tolist()],
            }
        )

    estimations.sort(key=lambda d: d["frames"])
    return estimations


def precompute_level_geometry_for_video(
    frame_h: int,
    frame_w: int,
    level: str,
    gt_params_by_level: Dict[str, Dict[str, Optional[float]]],
    k_cache: Dict[Tuple[float, float, float], np.ndarray],
) -> Dict[str, Any]:
    """
    Precalcula una sola vez para un vídeo y un nivel:
    - bbox del recorte válido
    - tamaño recortado
    - centro recortado
    - k_array GT para la corrección de referencia
    - mapas map_x y map_y de distorsión GT
    """
    gt = gt_params_by_level[level]
    d1_gt = float(gt["d1"])
    d2_gt = float(gt["d2"])

    cx0 = gt["cx"] if gt["cx"] is not None else frame_w / 2.0
    cy0 = gt["cy"] if gt["cy"] is not None else frame_h / 2.0
    cx0 = float(cx0)
    cy0 = float(cy0)

    # Usamos la función ya modularizada para calcular la distorsión GT
    dummy_img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    _, map_x, map_y, cx_used, cy_used = distort_image_division_model(
        dummy_img,
        d1_gt,
        d2_gt,
        cx=cx0,
        cy=cy0,
    )

    distorted_dummy = np.zeros_like(dummy_img)
    _, bbox = crop_largest_valid_rect_no_resize(
        distorted_dummy,
        map_x,
        map_y,
    )

    xmin, ymin, xmax, ymax = bbox
    cropped_w = int(xmax - xmin + 1)
    cropped_h = int(ymax - ymin + 1)

    cx_cropped = cx_used - xmin
    cy_cropped = cy_used - ymin

    max_r_gt = cropped_w / 2.0 + 10.0
    k_array_gt = get_or_compute_k_array(
        d1=d1_gt,
        d2=d2_gt,
        max_r=float(max_r_gt),
        k_cache=k_cache,
    )

    return {
        "level": level,
        "d1_gt": float(d1_gt),
        "d2_gt": float(d2_gt),
        "cx_full": float(cx_used),
        "cy_full": float(cy_used),
        "cx_cropped": float(cx_cropped),
        "cy_cropped": float(cy_cropped),
        "bbox": {
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax),
        },
        "cropped_size": {
            "w": int(cropped_w),
            "h": int(cropped_h),
        },
        "max_r_gt": float(max_r_gt),
        "k_array_gt": [float(x) for x in k_array_gt.tolist()],
        "map_x": map_x,
        "map_y": map_y,
    }