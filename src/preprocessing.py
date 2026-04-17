import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def adaptive_threshold(gray: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def morph_open(img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morph_erode(img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def morph_dilate(img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def canny_edges(gray: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
    return cv2.Canny(gray, t1, t2)


def unsharp_mask(img: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        gray = img
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = gray - 0.7 * lap
        return np.clip(sharp, 0, 255).astype(np.uint8)

    channels = cv2.split(img)
    out_channels = []
    for ch in channels:
        lap = cv2.Laplacian(ch, cv2.CV_64F)
        sharp = ch - 0.7 * lap
        out_channels.append(np.clip(sharp, 0, 255).astype(np.uint8))
    return cv2.merge(out_channels)


def wiener_like_sharpen(gray: np.ndarray) -> np.ndarray:
    # Practical simplified frequency-domain sharpening approximation
    # Used as a safe alternative to full kernel-based Wiener when blur kernel is unknown
    gray_f = np.fft.fft2(gray.astype(np.float32))
    gray_shift = np.fft.fftshift(gray_f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.float32)
    radius = min(rows, cols) // 10
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask[dist < radius] = 0.6

    enhanced_shift = gray_shift * (1.4 - mask * 0.4)
    enhanced = np.fft.ifft2(np.fft.ifftshift(enhanced_shift))
    enhanced = np.abs(enhanced)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)


def non_local_means_denoise(gray: np.ndarray) -> np.ndarray:
    gray_float = gray.astype(np.float32) / 255.0
    sigma_est = np.mean(estimate_sigma(gray_float, channel_axis=None))
    denoised = denoise_nl_means(
        gray_float,
        h=1.15 * sigma_est if sigma_est > 0 else 0.08,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None
    )
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)