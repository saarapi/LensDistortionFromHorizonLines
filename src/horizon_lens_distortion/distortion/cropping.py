import cv2
import numpy as np


def crop_largest_valid_rect_no_resize(
    distorted: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
):
    """
    Encuentra y recorta el mayor rectángulo completamente válido
    (sin píxeles fuera de la imagen original).
    """
    h, w = distorted.shape[:2]

    valid = (map_x >= 0) & (map_x <= (w - 1)) & (map_y >= 0) & (map_y <= (h - 1))
    mask = (valid.astype(np.uint8) * 255)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    t_max = float(dist.max())
    if t_max <= 0.0:
        raise RuntimeError("No hay región válida para recortar.")

    def bbox_from_threshold(t: float):
        safe = dist >= t
        ys, xs = np.where(safe)
        if ys.size == 0 or xs.size == 0:
            return None

        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())

        if valid[ymin:ymax + 1, xmin:xmax + 1].all():
            return xmin, ymin, xmax, ymax
        return None

    lo, hi = 0.0, t_max
    best = None

    for _ in range(30):
        mid = (lo + hi) / 2.0
        bbox = bbox_from_threshold(mid)
        if bbox is not None:
            best = bbox
            hi = mid
        else:
            lo = mid

    if best is None:
        raise RuntimeError("No se pudo encontrar un rectángulo válido.")

    xmin, ymin, xmax, ymax = best
    cropped = distorted[ymin:ymax + 1, xmin:xmax + 1]

    return cropped, (xmin, ymin, xmax, ymax)