import os
import cv2
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    unsharp_mask, laplacian_sharpen, bilateral_filter,
    wiener_like_sharpen, to_gray
)
from visualization import save_comparison


def mild_detail_boost(img: np.ndarray) -> np.ndarray:
    smooth = bilateral_filter(img, d=9, sigma_color=60, sigma_space=60)
    boosted = unsharp_mask(smooth, ksize=5, sigma=1.0, amount=1.2)
    return boosted


def process_building(image_path: str) -> None:
    out_dir = get_output_dir("blurEnhancement", "building")

    img = read_image(image_path)
    unsharp = unsharp_mask(img, ksize=5, sigma=1.0, amount=1.8)
    lap = laplacian_sharpen(img)

    gray = to_gray(img)
    wiener_like = wiener_like_sharpen(gray)
    wiener_like_bgr = cv2.cvtColor(wiener_like, cv2.COLOR_GRAY2BGR)

    final = unsharp

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_unsharp.png"), unsharp)
    save_image(os.path.join(out_dir, "03_laplacian.png"), lap)
    save_image(os.path.join(out_dir, "04_wiener_like.png"), wiener_like)
    save_image(os.path.join(out_dir, "05_final.png"), final)

    save_comparison(
        [img, unsharp, lap, wiener_like_bgr, final],
        ["Original", "Unsharp", "Laplacian", "Wiener-like", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_dog(image_path: str) -> None:
    out_dir = get_output_dir("blurEnhancement", "dog")

    img = read_image(image_path)
    unsharp = unsharp_mask(img, ksize=5, sigma=1.0, amount=1.0)
    bilateral_sharp = mild_detail_boost(img)
    lap = laplacian_sharpen(img)

    final = bilateral_sharp

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_unsharp.png"), unsharp)
    save_image(os.path.join(out_dir, "03_bilateral_sharpen.png"), bilateral_sharp)
    save_image(os.path.join(out_dir, "04_laplacian.png"), lap)
    save_image(os.path.join(out_dir, "05_final.png"), final)

    save_comparison(
        [img, unsharp, bilateral_sharp, lap, final],
        ["Original", "Unsharp", "Bilateral+Sharpen", "Laplacian", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )