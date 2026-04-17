import cv2
import numpy as np


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


def convert_bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def is_grayscale(img: np.ndarray) -> bool:
    return len(img.shape) == 2


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if is_grayscale(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)