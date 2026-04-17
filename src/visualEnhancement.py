import os

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, apply_clahe,
    apply_gamma, unsharp_mask, remove_thin_vertical_noise,
    otsu_threshold, remove_small_black_noise
)
from visualization import save_comparison


def process_newspaper(image_path: str) -> None:
    out_dir = get_output_dir("visualEnhancement", "newsPaper")

    img = read_image(image_path)
    gray = to_gray(img)

    # Threshold first to preserve tiny text strokes
    otsu = otsu_threshold(gray)

    # Remove tiny isolated black specks
    clean_small_4 = remove_small_black_noise(otsu, min_area=4)
    clean_small_6 = remove_small_black_noise(otsu, min_area=6)

    # Optional thin vertical-noise cleanup
    vertical_clean_1 = remove_thin_vertical_noise(clean_small_4, max_width=1, min_height=8)
    vertical_clean_2 = remove_thin_vertical_noise(clean_small_4, max_width=2, min_height=8)

    
    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_otsu.png"), otsu)
    save_image(os.path.join(out_dir, "04_clean_small_4.png"), clean_small_4)
    save_image(os.path.join(out_dir, "05_clean_small_6.png"), clean_small_6)
    save_image(os.path.join(out_dir, "06_vertical_clean_1.png"), vertical_clean_1)
    save_image(os.path.join(out_dir, "07_vertical_clean_2.png"), vertical_clean_2)

    save_comparison(
        [
            img,
            gray,
            otsu,
            clean_small_4,
            clean_small_6,
            vertical_clean_1,
            vertical_clean_2
        ],
        [
            "Original",
            "Gray",
            "Otsu",
            "Remove Small Noise (4)",
            "Remove Small Noise (6)",
            "Remove Thin Vertical Noise (1px)",
            "Remove Thin Vertical Noise (2px)"
        ],
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


    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_gamma.png"), gamma)
    save_image(os.path.join(out_dir, "04_clahe.png"), clahe)
    save_image(os.path.join(out_dir, "05_sharp.png"), sharp)

    save_comparison(
        [img, gray, gamma, clahe, sharp],
        ["Original", "Gray", "Gamma", "CLAHE", "Sharpened"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )