"""
Este script hace lo mismo que match_opencv_fisheye, pero en vez de aplicar
el division model, aplica SOLO el modelo directo de OpenCV, adaptando antes
los parámetros de distorsión del division model al modelo fisheye directo.

Flujo:
1. Carga una imagen distorsionada y su GT sin distorsión.
2. Estima d1, d2, cx, cy con process_image_file del módulo C++.
3. Convierte esos parámetros al modelo directo fisheye de OpenCV.
4. Corrige la imagen con recentrado del punto principal.
5. Guarda imágenes, figura comparativa y JSONs en results/.
"""

from pathlib import Path
import sys
import json

import click
import cv2


# Añadir la raíz del proyecto al path para poder importar desde src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.horizon_lens_distortion.utils.third_party import (  # noqa: E402
    add_lens_distortion_to_path,
)
from src.horizon_lens_distortion.undistortion.opencv_adapter import (  # noqa: E402
    match_and_undistort_with_opencv_direct_centered,
)
from src.horizon_lens_distortion.evaluation.visualization import (  # noqa: E402
    show_three_images,
)

add_lens_distortion_to_path()

from lens_distortion_module import process_image_file  # type: ignore  # noqa: E402


@click.command()
@click.option(
    "--test_image_name",
    type=str,
    default="data/raw/images/horizontes_GT_submuestreo_10_dist.png",
)
@click.option(
    "--original_image_name",
    type=str,
    default="data/raw/images/horizontes_GT_submuestreo_10.png",
)
@click.option(
    "--write_intermediates",
    type=bool,
    default=False,
)
@click.option(
    "--write_output",
    type=bool,
    default=False,
)
def cli(
    test_image_name: str,
    original_image_name: str,
    write_intermediates: bool,
    write_output: bool,
):
    # ==========================
    # RUTAS
    # ==========================
    test_image_path = Path(test_image_name)
    if not test_image_path.is_absolute():
        test_image_path = PROJECT_ROOT / test_image_path

    original_image_path = Path(original_image_name)
    if not original_image_path.is_absolute():
        original_image_path = PROJECT_ROOT / original_image_path

    results_dir = PROJECT_ROOT / "results"
    images_dir = results_dir / "images" / "opencv_direct_match"
    metrics_dir = results_dir / "metrics" / "opencv_direct_match"
    plots_dir = results_dir / "plots" / "opencv_direct_match"

    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # CARGA DE IMÁGENES
    # ==========================
    img_distorted = cv2.imread(str(test_image_path), cv2.IMREAD_COLOR)
    if img_distorted is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen distorsionada: {test_image_path}"
        )

    img_gt = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
    if img_gt is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen GT: {original_image_path}"
        )

    h, w = img_distorted.shape[:2]
    max_r = w / 2.0 + 10.0

    # ==========================
    # ESTIMACIÓN CON C++
    # ==========================
    _, res_dict = process_image_file(
        str(test_image_path),
        int(w),
        int(h),
        str(metrics_dir.resolve()),
        write_intermediates=write_intermediates,
        write_output=write_output,
        distance_point_line_max_hough=10.0,
    )

    print("Parámetros estimados por el C++ (division model):")
    print(res_dict)

    # ==========================
    # CORRECCIÓN OPENCV DIRECTA
    # ==========================
    img_opencv_direct, k_array, k_new = match_and_undistort_with_opencv_direct_centered(
        img=img_distorted,
        division_coef_d1=float(res_dict["d1"]),
        division_coef_d2=float(res_dict["d2"]),
        cx=float(res_dict["cx"]),
        cy=float(res_dict["cy"]),
        max_r=float(max_r),
    )

    print("Coeficientes OpenCV (direct):", k_array)
    print("Matriz K usada:\n", k_new)

    # ==========================
    # NOMBRES DE SALIDA
    # ==========================
    input_stem = test_image_path.stem

    distorted_out = images_dir / f"{input_stem}_entrada_distorsionada.png"
    gt_out = images_dir / f"{input_stem}_gt.png"
    corrected_out = images_dir / f"{input_stem}_opencv_direct_corregida.png"

    estimated_params_json = metrics_dir / f"{input_stem}_division_estimado.json"
    opencv_results_json = metrics_dir / f"{input_stem}_opencv_direct.json"

    comparison_plot_name = f"{input_stem}_comparison.png"

    # ==========================
    # GUARDAR FIGURA COMPARATIVA
    # ==========================
    show_three_images(
        img_distorted_bgr=img_distorted,
        img_gt_bgr=img_gt,
        img_corrected_bgr=img_opencv_direct,
        output_path=plots_dir,
        filename=comparison_plot_name,
    )

    # ==========================
    # GUARDAR IMÁGENES
    # ==========================
    cv2.imwrite(str(distorted_out), img_distorted)
    cv2.imwrite(str(gt_out), img_gt)
    cv2.imwrite(str(corrected_out), img_opencv_direct)

    # ==========================
    # GUARDAR JSONS
    # ==========================
    with open(estimated_params_json, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)

    with open(opencv_results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_image": str(test_image_path.resolve()),
                "gt_image": str(original_image_path.resolve()),
                "output_image": str(corrected_out.resolve()),
                "comparison_plot": str((plots_dir / comparison_plot_name).resolve()),
                "division_estimated": {
                    "d1": float(res_dict["d1"]),
                    "d2": float(res_dict["d2"]),
                    "cx": float(res_dict["cx"]),
                    "cy": float(res_dict["cy"]),
                },
                "opencv_direct": {
                    "k_array": [float(x) for x in k_array.tolist()],
                    "K": k_new.tolist(),
                },
                "image_size": {
                    "w": int(w),
                    "h": int(h),
                },
                "max_r": float(max_r),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ==========================
    # INFO POR CONSOLA
    # ==========================
    print(f"Imagen corregida: {corrected_out}")
    print(f"JSON división estimada: {estimated_params_json}")
    print(f"JSON OpenCV direct: {opencv_results_json}")
    print(f"Comparación: {plots_dir / comparison_plot_name}")


if __name__ == "__main__":
    cli()