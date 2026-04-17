import os
import cv2
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, median_filter, bilateral_filter, non_local_means_denoise,
    otsu_threshold, adaptive_threshold, morph_open, morph_close,
    apply_clahe
)
from visualization import save_comparison


def process_text(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "text")

    img = read_image(image_path)
    gray = to_gray(img)
    median = median_filter(gray, 3)
    thresh = adaptive_threshold(median, 15, 8)
    inv = cv2.bitwise_not(thresh)
    opened = morph_open(inv, 2, 1)
    closed = morph_close(opened, 2, 1)
    final = cv2.bitwise_not(closed)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median.png"), median)
    save_image(os.path.join(out_dir, "04_adaptive_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "05_inverted.png"), inv)
    save_image(os.path.join(out_dir, "06_opened.png"), opened)
    save_image(os.path.join(out_dir, "07_closed.png"), closed)
    save_image(os.path.join(out_dir, "08_final.png"), final)

    save_comparison(
        [img, gray, median, thresh, final],
        ["Original", "Gray", "Median", "Threshold", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_rocket(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "rocket")

    img = read_image(image_path)
    gray = to_gray(img)
    median = median_filter(gray, 3)
    bilateral = bilateral_filter(gray, 9, 75, 75)
    nlm = non_local_means_denoise(gray)

    final = bilateral

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median.png"), median)
    save_image(os.path.join(out_dir, "04_bilateral.png"), bilateral)
    save_image(os.path.join(out_dir, "05_nlm.png"), nlm)
    save_image(os.path.join(out_dir, "06_final.png"), final)

    save_comparison(
        [img, gray, median, bilateral, nlm, final],
        ["Original", "Gray", "Median", "Bilateral", "NLM", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_wind_chart(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "windChart")

    img = read_image(image_path)
    gray = to_gray(img)
    median = median_filter(gray, 3)
    bilateral = bilateral_filter(median, 7, 60, 60)
    clahe = apply_clahe(bilateral, 2.0, (8, 8))
    thresh = otsu_threshold(clahe)
    final = thresh

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median.png"), median)
    save_image(os.path.join(out_dir, "04_bilateral.png"), bilateral)
    save_image(os.path.join(out_dir, "05_clahe.png"), clahe)
    save_image(os.path.join(out_dir, "06_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "07_final.png"), final)

    save_comparison(
        [img, gray, median, bilateral, clahe, thresh, final],
        ["Original", "Gray", "Median", "Bilateral", "CLAHE", "Threshold", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )