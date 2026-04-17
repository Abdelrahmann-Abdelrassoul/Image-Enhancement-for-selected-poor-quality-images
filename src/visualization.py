import os
import math
import matplotlib.pyplot as plt
from utils import convert_bgr_to_rgb
from io_helpers import ensure_dir


def save_comparison(images, titles, save_path, cols=3, figsize=(14, 8)):
    ensure_dir(os.path.dirname(save_path))

    n = len(images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=figsize)

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(rows, cols, i)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(convert_bgr_to_rgb(img))
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()