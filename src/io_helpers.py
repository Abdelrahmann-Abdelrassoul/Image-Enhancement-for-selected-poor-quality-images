import os
import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")

    return image


def save_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Failed to save image to path: {path}")


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_output_dir(task_name: str, image_name: str) -> str:
    root = get_project_root()
    out_dir = os.path.join(root, "data", "processed", task_name, image_name)
    ensure_dir(out_dir)
    return out_dir