import os

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, median_filter, bilateral_filter, non_local_means_denoise,
    otsu_threshold, morph_close, gaussian_blur, 
    remove_small_connected_components, keep_text_polarity, binary_threshold,
    apply_clahe
)
from visualization import save_comparison


def process_text(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "text")

    img = read_image(image_path)
    gray = to_gray(img)

    # Method 1: Median filtering (best for TV-like salt/pepper/static noise)
    median_3 = median_filter(gray, 3)
    median_5 = median_filter(gray, 5)

    # Method 2: Slight Gaussian blur as backup for grainy noise
    gauss_3 = gaussian_blur(gray, 3)

    # Threshold bright white text
    thresh_median_3 = binary_threshold(median_3, thresh_value=180)
    thresh_median_5 = binary_threshold(median_5, thresh_value=180)
    thresh_gauss_3 = binary_threshold(gauss_3, thresh_value=180)

    # Keep correct polarity
    thresh_median_3 = keep_text_polarity(thresh_median_3)
    thresh_median_5 = keep_text_polarity(thresh_median_5)
    thresh_gauss_3 = keep_text_polarity(thresh_gauss_3)

    # Reconnect text strokes lightly
    closed_median_3 = morph_close(thresh_median_3, 2, 1)
    closed_median_5 = morph_close(thresh_median_5, 2, 1)
    closed_gauss_3 = morph_close(thresh_gauss_3, 2, 1)

    # Remove tiny leftover white noise blobs
    clean_median_3 = remove_small_connected_components(closed_median_3, min_area=15)
    clean_median_5 = remove_small_connected_components(closed_median_5, min_area=15)
    clean_gauss_3 = remove_small_connected_components(closed_gauss_3, min_area=15)


    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median_3.png"), median_3)
    save_image(os.path.join(out_dir, "04_median_5.png"), median_5)
    save_image(os.path.join(out_dir, "05_gauss_3.png"), gauss_3)
    save_image(os.path.join(out_dir, "06_thresh_median_3.png"), thresh_median_3)
    save_image(os.path.join(out_dir, "07_thresh_median_5.png"), thresh_median_5)
    save_image(os.path.join(out_dir, "08_thresh_gauss_3.png"), thresh_gauss_3)
    save_image(os.path.join(out_dir, "09_clean_median_3.png"), clean_median_3)
    save_image(os.path.join(out_dir, "10_clean_median_5.png"), clean_median_5)
    save_image(os.path.join(out_dir, "11_clean_gauss_3.png"), clean_gauss_3)

    save_comparison(
        [
            img,
            gray,
            median_3,
            thresh_median_3,
            clean_median_3,
            median_5,
            thresh_median_5,
            clean_median_5,
            gauss_3,
            thresh_gauss_3,
            clean_gauss_3
        ],
        [
            "Original",
            "Gray",
            "Median 3x3",
            "Thresh Median 3x3",
            "Clean Median 3x3",
            "Thresh Median 5x5",
            "Median 5x5",
            "Clean Median 5x5",
            "Gaussian 3x3",
            "Thresh Gaussian 3x3",
            "Clean Gaussian 3x3"
        ],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_rocket(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "rocket")

    img = read_image(image_path)
    gray = to_gray(img)
    median = median_filter(gray, 3)
    bilateral = bilateral_filter(gray, d=9, sigma_color=75, sigma_space=75)
    nlm = non_local_means_denoise(gray, patch_size=5, patch_distance=6)


    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median.png"), median)
    save_image(os.path.join(out_dir, "04_bilateral.png"), bilateral)
    save_image(os.path.join(out_dir, "05_nlm.png"), nlm)

    save_comparison(
        [img, gray, median, bilateral, nlm],
        ["Original", "Gray", "Median", "Bilateral", "NLM"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_wind_chart(image_path: str) -> None:
    out_dir = get_output_dir("noiseRemoval", "windChart")

    img = read_image(image_path)
    gray = to_gray(img)

    # Main denoising candidates
    median_3 = median_filter(gray, 3)
    median_5 = median_filter(gray, 5)

    # Very light contrast enhancement only after median
    clahe_median_3 = apply_clahe(median_3, 1.2, (8, 8))
    clahe_median_5 = apply_clahe(median_5, 1.2, (8, 8))

    # Optional threshold only for comparison, not necessarily final
    thresh_median_3 = otsu_threshold(median_3)
    thresh_median_5 = otsu_threshold(median_5)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_gray.png"), gray)
    save_image(os.path.join(out_dir, "03_median_3.png"), median_3)
    save_image(os.path.join(out_dir, "04_median_5.png"), median_5)
    save_image(os.path.join(out_dir, "05_clahe_median_3.png"), clahe_median_3)
    save_image(os.path.join(out_dir, "06_clahe_median_5.png"), clahe_median_5)
    save_image(os.path.join(out_dir, "07_thresh_median_3.png"), thresh_median_3)
    save_image(os.path.join(out_dir, "08_thresh_median_5.png"), thresh_median_5)

    save_comparison(
        [
            img,
            gray,
            median_3,
            median_5,
            clahe_median_3,
            clahe_median_5,
            thresh_median_3,
            thresh_median_5
        ],
        [
            "Original",
            "Gray",
            "Median 3x3",
            "Median 5x5",
            "CLAHE after Median 3x3",
            "CLAHE after Median 5x5",
            "Threshold after Median 3x3",
            "Threshold after Median 5x5"
        ],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )