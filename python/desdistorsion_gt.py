"""
Este script corrige una imagen distorsionada usando parámetros GT conocidos.

Características:
- El nivel de distorsión se elige con DISTORTION_LEVEL: "low", "mid" o "high".
- d1 se asigna automáticamente según ese nivel.
- d2 se fija a 0.0.
- cx y cy se toman siempre como el centro de la imagen.
- La salida se guarda separando:
    - imágenes en results/images
    - métricas/JSON en results/metrics
    - plots en results/plots
- La imagen corregida se guarda como:
    <nombre_entrada>_corregida_gt.png
"""

from pathlib import Path
import sys
import json

import cv2


# Añadir la raíz del proyecto al path para poder importar desde src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.horizon_lens_distortion.undistortion.opencv_adapter import (  # noqa: E402
    match_and_undistort_with_opencv,
    DEFAULT_OPENCV_FOCAL_LENGTH,
)
from src.horizon_lens_distortion.evaluation.visualization import (  # noqa: E402
    show_three_images,
    show_two_images,
)


# ==========================
# CONFIGURACIÓN
# ==========================

# Imagen distorsionada de entrada
TEST_IMAGE_NAME = PROJECT_ROOT / "data" / "raw" / "images" / "horizontes_GT_submuestreo_10_dist.png"

# Imagen GT opcional para comparación visual.
# Déjala como "" si no quieres usarla.
# Ejemplo:
# ORIGINAL_IMAGE_NAME = PROJECT_ROOT / "data" / "raw" / "images" / "horizontes_GT_submuestreo_10.png"
ORIGINAL_IMAGE_NAME = ""

# Nivel de distorsión GT: "low", "mid" o "high"
DISTORTION_LEVEL = "high"

# Radio máximo opcional. Si es None, se calcula automáticamente.
MAX_R = None


# ==========================
# PARÁMETROS GT POR NIVEL
# ==========================

D1_BY_LEVEL = {
    "low": -2e-07,
    "mid": -5e-07,
    "high": -1e-06,
}


def main():
    # ==========================
    # VALIDAR NIVEL
    # ==========================
    level = DISTORTION_LEVEL.strip().lower()
    if level not in D1_BY_LEVEL:
        raise ValueError(
            f"DISTORTION_LEVEL debe ser 'low', 'mid' o 'high'. Recibido: {DISTORTION_LEVEL}"
        )

    d1 = float(D1_BY_LEVEL[level])
    d2 = 0.0

    # ==========================
    # CARGA DE IMAGEN
    # ==========================
    img_distorted = cv2.imread(str(TEST_IMAGE_NAME), cv2.IMREAD_COLOR)
    if img_distorted is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen distorsionada: {TEST_IMAGE_NAME}"
        )

    h, w = img_distorted.shape[:2]

    # cx y cy siempre en el centro de la imagen
    cx = w / 2.0
    cy = h / 2.0

    if MAX_R is None:
        max_r_used = w / 2.0 + 10.0
    else:
        max_r_used = float(MAX_R)

    # ==========================
    # CORRECCIÓN
    # ==========================
    img_corrected, k_array, k_new = match_and_undistort_with_opencv(
        img=img_distorted,
        division_coef_d1=d1,
        division_coef_d2=d2,
        cx=float(cx),
        cy=float(cy),
        max_r=float(max_r_used),
    )

    # ==========================
    # RUTAS DE SALIDA
    # ==========================
    input_stem = TEST_IMAGE_NAME.stem

    results_dir = PROJECT_ROOT / "results"
    images_dir = results_dir / "images"
    metrics_dir = results_dir / "metrics"
    plots_dir = results_dir / "plots"

    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    corrected_image_path = images_dir / f"{input_stem}_corregida_gt.png"
    distorted_copy_path = images_dir / f"{input_stem}_entrada_distorsionada.png"
    results_json_path = metrics_dir / f"{input_stem}_corregida_gt.json"
    comparison_path = plots_dir / f"{input_stem}_comparison.png"

    # ==========================
    # GUARDAR IMÁGENES
    # ==========================
    cv2.imwrite(str(distorted_copy_path), img_distorted)
    cv2.imwrite(str(corrected_image_path), img_corrected)

    # ==========================
    # VISUALIZACIÓN
    # ==========================
    gt_ok = False
    gt_image_resolved = ""

    if ORIGINAL_IMAGE_NAME:
        original_image_path = Path(ORIGINAL_IMAGE_NAME)
        if not original_image_path.is_absolute():
            original_image_path = PROJECT_ROOT / original_image_path

        if original_image_path.exists():
            img_gt = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
            if img_gt is not None:
                show_three_images(
                    img_distorted_bgr=img_distorted,
                    img_gt_bgr=img_gt,
                    img_corrected_bgr=img_corrected,
                    output_path=plots_dir,
                    filename=comparison_path.name,
                )
                gt_ok = True
                gt_image_resolved = str(original_image_path.resolve())

    if not gt_ok:
        show_two_images(
            img_distorted_bgr=img_distorted,
            img_corrected_bgr=img_corrected,
            output_path=plots_dir,
            filename=comparison_path.name,
        )

    # ==========================
    # GUARDAR JSON
    # ==========================
    res_dict = {
        "distortion_level": level,
        "division": {
            "d1": float(d1),
            "d2": float(d2),
            "cx": float(cx),
            "cy": float(cy),
        },
        "opencv_direct": {
            "k_array": [float(x) for x in k_array.tolist()],
            "K": k_new.tolist(),
        },
        "image_size": {
            "w": int(w),
            "h": int(h),
        },
        "max_r": float(max_r_used),
        "DEFAULT_OPENCV_FOCAL_LENGTH": float(DEFAULT_OPENCV_FOCAL_LENGTH),
        "input_image": str(TEST_IMAGE_NAME.resolve()),
        "output_image": str(corrected_image_path.resolve()),
        "distorted_copy_image": str(distorted_copy_path.resolve()),
        "comparison_plot": str(comparison_path.resolve()),
        "gt_image": gt_image_resolved,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)

    # ==========================
    # INFO POR CONSOLA
    # ==========================
    print(f"Nivel GT usado: {level}")
    print(f"Centro usado (cx, cy): ({cx:.2f}, {cy:.2f})")
    print(f"Imagen corregida: {corrected_image_path}")
    print(f"Copia distorsionada: {distorted_copy_path}")
    print(f"JSON: {results_json_path}")
    print(f"Comparación: {comparison_path}")


if __name__ == "__main__":
    main()