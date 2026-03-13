from typing import Callable

import cv2
import numpy as np
from scipy.optimize import minimize

from src.horizon_lens_distortion.utils.third_party import add_lens_distortion_to_path

# Añadimos el módulo externo al path antes de importarlo
add_lens_distortion_to_path()

from lens_distortion_module import (  # type: ignore
    opencv_fisheye_polynomial as opencv_fisheye_polynomial_pybind,
)


DEFAULT_OPENCV_FOCAL_LENGTH = 1000.0


def opencv_fisheye_polynomial(
    r: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    opencv_focal_length: float = DEFAULT_OPENCV_FOCAL_LENGTH,
) -> float:
    """
    Wrapper al polinomio fisheye implementado en el módulo externo.
    """
    return opencv_fisheye_polynomial_pybind(
        r, k1, k2, k3, k4, 0.0, 0.0, opencv_focal_length
    )


def division_model_polynomial(r: float, d1: float, d2: float) -> float:
    """
    Division model de undistorsión:
        d(r) = 1 / (1 + d1*r^2 + d2*r^4)
    """
    return 1.0 / (1.0 + d1 * r * r + d2 * r * r * r * r)


def match_opencv_distortion_to_undistortion_model(
    undistortion_model: Callable[[float], float],
    max_r: float,
) -> np.ndarray:
    """
    Ajusta los coeficientes k1..k4 de OpenCV para aproximar
    la undistorsión definida por el division model.
    """
    r_array = np.linspace(1.0, max_r, 4000)
    d = np.array([undistortion_model(r) for r in r_array], dtype=np.float64)

    def cost_function(k_array: np.ndarray) -> float:
        k1, k2, k3, k4 = k_array
        sumout = 0.0

        for i, r in enumerate(r_array):
            r_theta = d[i] * r / DEFAULT_OPENCV_FOCAL_LENGTH
            sumout += (
                1.0 / d[i]
                - opencv_fisheye_polynomial(
                    r_theta,
                    k1,
                    k2,
                    k3,
                    k4,
                    opencv_focal_length=1.0,
                )
            ) ** 2

        return float(sumout)

    k0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    res = minimize(
        cost_function,
        k0,
        method="l-bfgs-b",
        options={
            "ftol": 1e-10,
            "disp": False,
            "maxiter": 10000,
            "pgtol": 1e-8,
        },
    )

    return np.asarray(res.x, dtype=np.float64)


def generate_opencv_distortion_coefs(
    division_coef_d1: float,
    division_coef_d2: float,
    cx: float,
    cy: float,
    max_r: float,
):
    """
    Convierte d1,d2 del division model en coeficientes OpenCV fisheye
    y genera una matriz K con focal ficticia fija.
    """
    k_array = match_opencv_distortion_to_undistortion_model(
        lambda r: division_model_polynomial(r, division_coef_d1, division_coef_d2),
        max_r,
    )

    camera_matrix = np.array(
        [
            [DEFAULT_OPENCV_FOCAL_LENGTH, 0.0, cx],
            [0.0, DEFAULT_OPENCV_FOCAL_LENGTH, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return k_array, camera_matrix


def undistort_with_matching_opencv(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    opencv_k1: float,
    opencv_k2: float,
    opencv_k3: float,
    opencv_k4: float,
):
    """
    Variante simple: ajusta K al centro de la imagen y aplica OpenCV fisheye.
    """
    h, w = img.shape[:2]

    camera_matrix_centered = camera_matrix.copy()
    camera_matrix_centered[0, 2] = w / 2.0
    camera_matrix_centered[1, 2] = h / 2.0

    dist_coeffs = np.array(
        [opencv_k1, opencv_k2, opencv_k3, opencv_k4],
        dtype=np.float64,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix_centered,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        camera_matrix_centered,
        (w, h),
        cv2.CV_16SC2,
    )

    img_res = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_DEFAULT,
    )

    return img_res, camera_matrix_centered


def undistort_with_matching_opencv_direct_centered(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    opencv_k1: float,
    opencv_k2: float,
    opencv_k3: float,
    opencv_k4: float,
):
    """
    Variante DIRECTA con recentrado del punto principal:
    1. Hace padding.
    2. Desplaza la imagen para centrar el punto principal.
    3. Recorta al tamaño original.
    4. Ajusta K al centro.
    5. Aplica undistorsión OpenCV fisheye.
    """
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    h, w = img.shape[:2]
    diff_cx = cx - w / 2.0
    diff_cy = cy - h / 2.0

    pad = int(max(abs(diff_cx), abs(diff_cy)))

    img_padded = cv2.copyMakeBorder(
        img,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    img_padded = np.roll(img_padded, -int(diff_cx), axis=1)
    img_padded = np.roll(img_padded, int(diff_cy), axis=0)

    img_centered = img_padded[pad: pad + h, pad: pad + w]

    camera_matrix_centered = camera_matrix.copy()
    camera_matrix_centered[0, 2] = w / 2.0
    camera_matrix_centered[1, 2] = h / 2.0

    dist_coeffs = np.array(
        [opencv_k1, opencv_k2, opencv_k3, opencv_k4],
        dtype=np.float64,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix_centered,
        dist_coeffs,
        np.eye(3, dtype=np.float64),
        camera_matrix_centered,
        (w, h),
        cv2.CV_16SC2,
    )

    img_res = cv2.remap(
        img_centered,
        map1,
        map2,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_DEFAULT,
    )

    return img_res, camera_matrix_centered


def match_and_undistort_with_opencv(
    img: np.ndarray,
    division_coef_d1: float,
    division_coef_d2: float,
    cx: float,
    cy: float,
    max_r: float,
):
    """
    Pipeline completo sencillo:
    1. Ajusta coeficientes OpenCV a partir del division model.
    2. Aplica la undistorsión con OpenCV.
    """
    k_array, camera_matrix = generate_opencv_distortion_coefs(
        division_coef_d1,
        division_coef_d2,
        cx,
        cy,
        max_r,
    )

    img_res, camera_matrix_new = undistort_with_matching_opencv(
        img,
        camera_matrix,
        float(k_array[0]),
        float(k_array[1]),
        float(k_array[2]),
        float(k_array[3]),
    )

    return img_res, k_array, camera_matrix_new


def match_and_undistort_with_opencv_direct_centered(
    img: np.ndarray,
    division_coef_d1: float,
    division_coef_d2: float,
    cx: float,
    cy: float,
    max_r: float,
):
    """
    Pipeline completo DIRECTO con recentrado:
    1. Ajusta coeficientes OpenCV a partir del division model.
    2. Recentra la imagen respecto al punto principal.
    3. Aplica la undistorsión con OpenCV.
    """
    k_array, camera_matrix = generate_opencv_distortion_coefs(
        division_coef_d1,
        division_coef_d2,
        cx,
        cy,
        max_r,
    )

    img_res, camera_matrix_new = undistort_with_matching_opencv_direct_centered(
        img,
        camera_matrix,
        float(k_array[0]),
        float(k_array[1]),
        float(k_array[2]),
        float(k_array[3]),
    )

    return img_res, k_array, camera_matrix_new


def scale_cx_cy(cx: float, cy: float, w0: int, h0: int, w1: int, h1: int):
    """
    Escala el centro de distorsión proporcionalmente cuando cambia el tamaño.
    """
    sx = w1 / float(w0)
    sy = h1 / float(h0)
    return cx * sx, cy * sy