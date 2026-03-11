import cv2
import numpy as np


def distort_image_division_model(
    img: np.ndarray,
    d1: float,
    d2: float,
    cx: float | None = None,
    cy: float | None = None,
    interpolation: int = cv2.INTER_CUBIC,
    border_mode: int = cv2.BORDER_CONSTANT,
):
    """
    Aplica distorsión radial usando el division model.
    Devuelve la imagen distorsionada, los mapas de remapeo y el centro usado.
    """
    h, w = img.shape[:2]

    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys)

    dx = x_grid - cx
    dy = y_grid - cy
    r2 = dx * dx + dy * dy

    a = 1.0 + d1 * r2 + d2 * (r2 * r2)

    map_x = (cx + dx / a).astype(np.float32)
    map_y = (cy + dy / a).astype(np.float32)

    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
    )

    return distorted, map_x, map_y, float(cx), float(cy)