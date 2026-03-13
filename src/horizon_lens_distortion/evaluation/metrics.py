from typing import List

import cv2
import numpy as np
import pywt
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim


DEFAULT_CWSSIM_WIDTH = 30
DEFAULT_CWSSIM_K = 0.01


def safe_mean(values: List[float]) -> float:
    """
    Media robusta:
    - ignora NaN
    - ignora inf (si hay finitos, usa finitos; si no hay finitos pero hay inf, devuelve inf)
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)

    if np.any(finite):
        return float(np.mean(arr[finite]))
    if np.any(np.isinf(arr)):
        return float("inf")
    return float("nan")


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula PSNR entre dos imágenes.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula SSIM entre dos imágenes.
    Si son BGR, primero convierte a escala de grises.
    """
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2

    return float(ssim(img1_gray, img2_gray, data_range=255))


def _np_bgr_to_pil_gray(img_bgr: np.ndarray) -> Image.Image:
    """
    Convierte una imagen numpy BGR o gris a PIL en escala de grises.
    """
    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(img_rgb)
    return ImageOps.grayscale(pil)


def compute_cwssim_from_arrays(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    width: int = DEFAULT_CWSSIM_WIDTH,
    k: float = DEFAULT_CWSSIM_K,
) -> float:
    """
    Calcula CW-SSIM entre dos imágenes.
    Requiere que ambas tengan el mismo tamaño.
    """
    pil1 = _np_bgr_to_pil_gray(img1)
    pil2 = _np_bgr_to_pil_gray(img2)

    if pil1.size != pil2.size:
        raise ValueError(f"CW-SSIM requiere mismo tamaño: {pil1.size} vs {pil2.size}")

    widths = np.arange(1, int(width) + 1)
    sig1 = np.asarray(list(pil1.getdata()), dtype=np.float64)
    sig2 = np.asarray(list(pil2.getdata()), dtype=np.float64)

    cwtmatr1, _ = pywt.cwt(sig1, widths, "mexh")
    cwtmatr2, _ = pywt.cwt(sig2, widths, "mexh")

    c1c2 = np.multiply(np.abs(cwtmatr1), np.abs(cwtmatr2))
    c1_2 = np.square(np.abs(cwtmatr1))
    c2_2 = np.square(np.abs(cwtmatr2))

    num_ssim_1 = 2.0 * np.sum(c1c2, axis=0) + k
    den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + k

    c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
    num_ssim_2 = 2.0 * np.abs(np.sum(c1c2_conj, axis=0)) + k
    den_ssim_2 = 2.0 * np.sum(np.abs(c1c2_conj), axis=0) + k

    ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
    return float(np.average(ssim_map))