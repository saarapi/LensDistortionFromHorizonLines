from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def legend_label_for_level(level: str, gt_params_by_level: Dict[str, Dict[str, float]]) -> str:
    """
    Construye la etiqueta de leyenda para un nivel usando su d1.
    """
    d1 = float(gt_params_by_level[level]["d1"])
    return f"{level} = {d1:.2e}"


def make_curves_from_results(
    results_by_level: Dict[str, List[Dict[str, float]]],
    metric_key: str,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Convierte resultados por nivel en curvas (xs, ys) para plotting.
    """
    curves: Dict[str, Tuple[List[float], List[float]]] = {}

    for level, res in results_by_level.items():
        xs = [r["frames"] for r in res]
        ys = [r[metric_key] for r in res]
        curves[level] = (xs, ys)

    return curves


def save_multiline_plot(
    curves: Dict[str, Tuple[List[float], List[float]]],
    *,
    y_label: str,
    title: str,
    out_path: Path,
    gt_params_by_level: Dict[str, Dict[str, float]],
) -> None:
    """
    Guarda una gráfica con varias curvas, una por nivel.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for level, (xs, ys) in curves.items():
        if len(xs) == 0:
            continue
        plt.plot(
            xs,
            ys,
            marker="o",
            label=legend_label_for_level(level, gt_params_by_level),
        )

    plt.xlabel("Nº de frames (calibración)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()