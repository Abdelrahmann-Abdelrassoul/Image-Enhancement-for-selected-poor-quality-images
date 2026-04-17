import os
import cv2
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, apply_clahe, adaptive_threshold, morph_open, morph_close,
    apply_gamma, unsharp_mask
)
from visualization import save_comparison


def process_newspaper(image_path: str) -> None:
    out_dir = get_output_dir("visualEnhancement", "newsPaper")

    img = read_image(image_path)
    gray = to_gray(img)
    clahe = apply_clahe(gray, 2.5, (8, 8))
    thresh = adaptive_threshold(clahe, 17, 7)
    inv = cv2.bitwise_not(thresh)
    opened = morph_open(inv, 2, 1)
    closed = morph_close(opened, 2, 1)
    final = cv2.bitwise_not(closed)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_clahe.png"), clahe)
    save_image(os.path.join(out_dir, "04_adaptive_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "05_opened.png"), opened)
    save_image(os.path.join(out_dir, "06_closed.png"), closed)
    save_image(os.path.join(out_dir, "07_final.png"), final)

    save_comparison(
        [img, gray, clahe, thresh, final],
        ["Original", "Gray", "CLAHE", "Adaptive Threshold", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_name_plate(image_path: str) -> None:
    out_dir = get_output_dir("visualEnhancement", "namePlate")

    img = read_image(image_path)
    gray = to_gray(img)
    gamma = apply_gamma(gray, 1.5)
    clahe = apply_clahe(gamma, 3.0, (8, 8))
    sharp = unsharp_mask(clahe, ksize=5, sigma=1.0, amount=1.4)
    thresh = adaptive_threshold(sharp, 15, 6)

    final = sharp

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_gamma.png"), gamma)
    save_image(os.path.join(out_dir, "04_clahe.png"), clahe)
    save_image(os.path.join(out_dir, "05_sharp.png"), sharp)
    save_image(os.path.join(out_dir, "06_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "07_final.png"), final)

    save_comparison(
        [img, gray, gamma, clahe, sharp, thresh, final],
        ["Original", "Gray", "Gamma", "CLAHE", "Sharpened", "Threshold", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )