import os
import cv2
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, gaussian_blur, otsu_threshold, adaptive_threshold,
    morph_open, morph_close, canny_edges, apply_clahe
)
from visualization import save_comparison


def extract_largest_circular_components(binary_img: np.ndarray, original_bgr: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = original_bgr.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.6:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    return output


def process_7circles(image_path: str) -> None:
    out_dir = get_output_dir("componentExtraction", "7circles")

    img = read_image(image_path)
    gray = to_gray(img)
    blur = gaussian_blur(gray, 5)
    thresh = otsu_threshold(blur)

    # invert if background dominates as white object extraction
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.7:
        thresh = cv2.bitwise_not(thresh)

    opened = morph_open(thresh, 3, 1)
    closed = morph_close(opened, 5, 1)
    edges = canny_edges(closed, 50, 150)
    detected = extract_largest_circular_components(closed, img)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_blur.png"), blur)
    save_image(os.path.join(out_dir, "04_otsu_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "05_opened.png"), opened)
    save_image(os.path.join(out_dir, "06_closed.png"), closed)
    save_image(os.path.join(out_dir, "07_edges.png"), edges)
    save_image(os.path.join(out_dir, "08_final.png"), detected)

    save_comparison(
        [img, gray, thresh, closed, edges, detected],
        ["Original", "Gray", "Threshold", "Morphology", "Edges", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_covid_chart(image_path: str) -> None:
    out_dir = get_output_dir("componentExtraction", "COVID-19Chart")

    img = read_image(image_path)
    gray = to_gray(img)
    clahe = apply_clahe(gray, 2.5, (8, 8))
    blur = gaussian_blur(clahe, 3)
    thresh = adaptive_threshold(blur, 15, 4)
    inv = cv2.bitwise_not(thresh)
    opened = morph_open(inv, 2, 1)
    closed = morph_close(opened, 3, 1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 40:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_clahe.png"), clahe)
    save_image(os.path.join(out_dir, "04_blur.png"), blur)
    save_image(os.path.join(out_dir, "05_adaptive_threshold.png"), thresh)
    save_image(os.path.join(out_dir, "06_inverted.png"), inv)
    save_image(os.path.join(out_dir, "07_opened.png"), opened)
    save_image(os.path.join(out_dir, "08_closed.png"), closed)
    save_image(os.path.join(out_dir, "09_final.png"), contour_overlay)

    save_comparison(
        [img, clahe, thresh, inv, closed, contour_overlay],
        ["Original", "CLAHE", "Threshold", "Inverted", "Refined", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )