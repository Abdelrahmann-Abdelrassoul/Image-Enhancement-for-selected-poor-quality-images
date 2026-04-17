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

def get_border_connected_background_mask(
    hsv_img: np.ndarray,
    lower_blue=(85, 25, 40),
    upper_blue=(135, 255, 255)
) -> np.ndarray:
    """
    Detect only the blue background that is connected to the image borders.
    This avoids treating internal blue-ish pixels as background.
    """
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))

    background_mask = np.zeros_like(blue_mask)
    for label_id in border_labels:
        background_mask[labels == label_id] = 255

    return background_mask


def get_foreground_from_background(background_mask: np.ndarray) -> np.ndarray:
    """
    Invert background mask to obtain foreground candidate mask.
    """
    return cv2.bitwise_not(background_mask)


def clean_foreground_mask(fg_mask: np.ndarray) -> np.ndarray:
    """
    Light cleanup only.
    Avoid aggressive morphology that merges nearby cards.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def watershed_split_foreground(original_bgr: np.ndarray, fg_mask: np.ndarray, dist_threshold: float = 0.28):
    """
    Split merged foreground objects using distance transform + watershed.

    Returns:
    - watershed label matrix
    - sure foreground mask
    - unknown region mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, dist_threshold, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    sure_bg = cv2.dilate(fg_mask, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_bgr.copy(), markers)

    return markers, sure_fg, unknown


def extract_boxes_from_markers(
    markers: np.ndarray,
    min_area: int = 350,
    min_width: int = 30,
    min_height: int = 20,
    aspect_ratio_range: tuple = (0.6, 3.5)
):
    """
    Convert watershed markers into bounding boxes.
    """
    boxes = []

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        region_mask = np.zeros(markers.shape, dtype=np.uint8)
        region_mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_width or h < min_height:
            continue

        aspect_ratio = w / float(h)
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        boxes.append((x, y, w, h))

    return boxes


def draw_boxes(image: np.ndarray, boxes: list, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    output = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def boxes_to_mask(image_shape: tuple, boxes: list) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    return mask


from skimage.restoration import richardson_lucy


def motion_psf(length: int = 9, angle: float = 0.0) -> np.ndarray:
    """
    Create a simple motion blur PSF.
    angle in degrees.
    """
    length = max(3, length)
    if length % 2 == 0:
        length += 1

    psf = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    psf[center, :] = 1.0

    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    psf = cv2.warpAffine(psf, rotation_matrix, (length, length))
    psf_sum = psf.sum()
    if psf_sum != 0:
        psf /= psf_sum

    return psf


def disk_psf(radius: int = 3) -> np.ndarray:
    """
    Create a small circular blur PSF for mild out-of-focus blur.
    """
    radius = max(1, radius)
    size = radius * 2 + 1
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (radius, radius), radius, 1, -1)
    psf_sum = psf.sum()
    if psf_sum != 0:
        psf /= psf_sum
    return psf


def richardson_lucy_deblur_gray(gray: np.ndarray, psf: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Richardson-Lucy deblurring for grayscale image.
    """
    gray_float = gray.astype(np.float32) / 255.0
    restored = richardson_lucy(gray_float, psf, num_iter=iterations, clip=False)
    restored = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def richardson_lucy_deblur_bgr(img: np.ndarray, psf: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Apply Richardson-Lucy deblurring channel-wise on BGR image.
    """
    channels = cv2.split(img)
    restored_channels = []
    for ch in channels:
        restored = richardson_lucy_deblur_gray(ch, psf, iterations)
        restored_channels.append(restored)
    return cv2.merge(restored_channels)