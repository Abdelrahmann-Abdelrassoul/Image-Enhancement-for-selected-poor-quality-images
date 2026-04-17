import os
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    unsharp_mask, laplacian_sharpen, bilateral_filter,
    disk_psf, motion_psf, richardson_lucy_deblur_bgr
)
from visualization import save_comparison


def mild_detail_boost(img: np.ndarray) -> np.ndarray:
    smooth = bilateral_filter(img, d=9, sigma_color=60, sigma_space=60)
    boosted = unsharp_mask(smooth, ksize=5, sigma=1.0, amount=1.2)
    return boosted


def process_building(image_path: str) -> None:
    out_dir = get_output_dir("blurEnhancement", "building")

    img = read_image(image_path)

    # Method 1: Mild unsharp mask
    unsharp = unsharp_mask(img, ksize=5, sigma=1.2, amount=2)

    # Method 2: Bilateral filter + unsharp mask
    bilateral = bilateral_filter(img, d=7, sigma_color=50, sigma_space=50)
    bilateral_unsharp = unsharp_mask(bilateral, ksize=5, sigma=1.2, amount=2)

    # Method 3A: Richardson-Lucy with mild circular blur PSF
    psf_disk = disk_psf(radius=3)
    rl_disk = richardson_lucy_deblur_bgr(img, psf_disk, iterations=20)

    # Method 3B: Richardson-Lucy with horizontal/near-horizontal motion PSF
    psf_motion = motion_psf(length=7, angle=0.0) 
    rl_motion = richardson_lucy_deblur_bgr(img, psf_motion, iterations=20)

    # Optional slight sharpen after RL if needed
    rl_disk_sharp = unsharp_mask(rl_disk, ksize=3, sigma=1.0, amount=0.6)
    rl_motion_sharp = unsharp_mask(rl_motion, ksize=3, sigma=1.0, amount=0.6)

    # Choose best default final candidate for buildings
    final = rl_disk_sharp

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_unsharp.png"), unsharp)
    save_image(os.path.join(out_dir, "03_bilateral.png"), bilateral)
    save_image(os.path.join(out_dir, "04_bilateral_unsharp.png"), bilateral_unsharp)
    save_image(os.path.join(out_dir, "05_rl_disk.png"), rl_disk)
    save_image(os.path.join(out_dir, "06_rl_motion.png"), rl_motion)
    save_image(os.path.join(out_dir, "07_rl_disk_sharp.png"), rl_disk_sharp)
    save_image(os.path.join(out_dir, "08_rl_motion_sharp.png"), rl_motion_sharp)

    save_comparison(
        [
            img,
            unsharp,
            bilateral_unsharp,
            rl_disk,
            rl_motion,
            rl_disk_sharp,
            rl_motion_sharp
        ],
        [
            "Original",
            "Unsharp",
            "Bilateral + Unsharp",
            "RL Disk",
            "RL Motion",
            "RL Disk + Sharp",
            "RL Motion + Sharp"
        ],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )

def process_dog(image_path: str) -> None:
    out_dir = get_output_dir("blurEnhancement", "dog")

    img = read_image(image_path)
    unsharp = unsharp_mask(img, ksize=5, sigma=1.0, amount=1.0)
    bilateral_sharp = mild_detail_boost(img)
    lap = laplacian_sharpen(img)

    save_image(os.path.join(out_dir, "01_original.png"), img)
    save_image(os.path.join(out_dir, "02_unsharp.png"), unsharp)
    save_image(os.path.join(out_dir, "03_bilateral_sharpen.png"), bilateral_sharp)
    save_image(os.path.join(out_dir, "04_laplacian.png"), lap)

    save_comparison(
        [img, unsharp, bilateral_sharp, lap],
        ["Original", "Unsharp", "Bilateral+Sharpen", "Laplacian"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )