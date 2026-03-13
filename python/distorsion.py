"""
Este script simula una distorsión radial de lente usando el division model y
después recorta automáticamente la zona válida máxima sin píxeles negros.

Flujo:
1. Lee una imagen ideal.
2. Aplica una distorsión sintética con el division model.
3. Calcula la mayor región completamente válida.
4. Recorta esa región sin reescalar.
5. Ajusta el centro de distorsión al sistema de coordenadas del recorte.
6. Guarda la imagen recortada y, opcionalmente, un JSON con los parámetros.
"""

from pathlib import Path
import sys
import json

import cv2


# Añadir la raíz del proyecto al path para poder importar desde src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.horizon_lens_distortion.distortion.division_model import (  # noqa: E402
    distort_image_division_model,
)
from src.horizon_lens_distortion.distortion.cropping import (  # noqa: E402
    crop_largest_valid_rect_no_resize,
)


def main():
    # ==========================
    # CONFIGURACIÓN
    # ==========================
    input_path = PROJECT_ROOT / "data" / "raw" / "images" / "horizontes_GT_submuestreo_10.png"
    # Si quieres guardar también la versión distorsionada sin recortar, descomenta y usa esta ruta
    # output_full_path = "./results/horizontes_GT_submuestreo_10_dist_full.png"

    output_crop_path = PROJECT_ROOT / "data" / "raw" / "images" / "horizontes_GT_submuestreo_10_dist.png"

    # Si quieres guardar también los parámetros GT del recorte, activa esta ruta
    # output_json_path = "./data/horizontes_GT_submuestreo_10_dist_gt.json"

    # Distorsión mínima: d1 = 0.0
    # Distorsión alta típica: d1 = -1e-06
    # Baja:  d1 = -2e-07
    # Media: d1 = -5e-07
    # Alta:  d1 = -1e-06
    d1 = -1e-06
    d2 = 0.0

    # ==========================
    # CARGA DE IMAGEN
    # ==========================
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {input_path}")

    # ==========================
    # DISTORSIÓN SINTÉTICA
    # ==========================
    distorted, map_x, map_y, cx, cy = distort_image_division_model(
        img=img,
        d1=d1,
        d2=d2,
    )

    # ==========================
    # RECORTE DE REGIÓN VÁLIDA
    # ==========================
    cropped, (xmin, ymin, xmax, ymax) = crop_largest_valid_rect_no_resize(
        distorted=distorted,
        map_x=map_x,
        map_y=map_y,
    )

    # Ajustar cx, cy al nuevo sistema de coordenadas del recorte
    cx_cropped = cx - xmin
    cy_cropped = cy - ymin

    h_c, w_c = cropped.shape[:2]

    # ==========================
    # GUARDADO
    # ==========================
    output_crop_path_obj = Path(output_crop_path)
    output_crop_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Si quieres guardar la imagen completa distorsionada:
    # output_full_path_obj = Path(output_full_path)
    # output_full_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(output_full_path_obj), distorted)

    cv2.imwrite(str(output_crop_path_obj), cropped)

    data = {
        "image": Path(input_path).name,
        "cx": float(cx_cropped),
        "cy": float(cy_cropped),
        "d1": float(d1),
        "d2": float(d2),
        "w": int(w_c),
        "h": int(h_c),
        "crop_bbox": {
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax),
        },
    }

    # Si quieres guardar JSON, descomenta:
    # output_json_path_obj = Path(output_json_path)
    # output_json_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_json_path_obj, "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)

    # ==========================
    # INFO POR CONSOLA
    # ==========================
    print("Imagen distorsionada recortada (sin reescalar) guardada en:", output_crop_path)
    print("cx, cy del recorte:", cx_cropped, cy_cropped)
    print("Distorsión aplicada, d1, d2:", d1, d2)
    print("Tamaño imagen recortada:", w_c, h_c)
    print("Bounding box del recorte:", (xmin, ymin, xmax, ymax))


if __name__ == "__main__":
    main()