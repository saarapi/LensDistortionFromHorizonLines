from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_three_images(
    img_distorted_bgr: np.ndarray,
    img_gt_bgr: np.ndarray,
    img_corrected_bgr: np.ndarray,
    output_path: Path,
    filename: str = "comparison.png",
) -> Path:
    """
    Muestra y guarda una figura con:
    1) imagen distorsionada
    2) imagen GT
    3) imagen corregida
    """
    h_gt, w_gt = img_gt_bgr.shape[:2]

    img_dist_resized = cv2.resize(
        img_distorted_bgr,
        (w_gt, h_gt),
        interpolation=cv2.INTER_LINEAR,
    )
    img_corr_resized = cv2.resize(
        img_corrected_bgr,
        (w_gt, h_gt),
        interpolation=cv2.INTER_LINEAR,
    )

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(img_dist_resized, cv2.COLOR_BGR2RGB))
    ax1.set_title("Distorsionada")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(img_gt_bgr, cv2.COLOR_BGR2RGB))
    ax2.set_title("Original (GT)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(cv2.cvtColor(img_corr_resized, cv2.COLOR_BGR2RGB))
    ax3.set_title("Corregida")
    ax3.axis("off")

    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / filename
    fig.savefig(str(comparison_path), dpi=200)
    plt.show()

    return comparison_path


def show_two_images(
    img_distorted_bgr: np.ndarray,
    img_corrected_bgr: np.ndarray,
    output_path: Path,
    filename: str = "comparison.png",
) -> Path:
    """
    Muestra y guarda una figura con:
    1) imagen distorsionada
    2) imagen corregida
    """
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(img_distorted_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Distorsionada")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(img_corrected_bgr, cv2.COLOR_BGR2RGB))
    ax2.set_title("Corregida")
    ax2.axis("off")

    fig.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / filename
    fig.savefig(str(comparison_path), dpi=200)
    plt.show()

    return comparison_path