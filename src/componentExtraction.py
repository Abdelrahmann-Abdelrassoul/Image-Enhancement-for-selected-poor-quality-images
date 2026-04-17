import os
import cv2
import numpy as np

from io_helpers import read_image, save_image, get_output_dir
from preprocessing import (
    to_gray, gaussian_blur, otsu_threshold, get_border_connected_background_mask, 
    get_foreground_from_background, clean_foreground_mask, watershed_split_foreground,
    morph_open, morph_close, canny_edges, extract_boxes_from_markers, boxes_to_mask, draw_boxes
)
from visualization import save_comparison


def extract_largest_circular_components(binary_img: np.ndarray, original_bgr: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = original_bgr.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0:
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
        [img, gray, thresh, opened, closed, edges, detected],
        ["Original", "Gray", "Threshold", "Opened", "Closed", "Edges", "Final"],
        os.path.join(out_dir, "comparison.png"),
        cols=3
    )


def process_covid_chart(image_path: str) -> None: 
    out_dir = get_output_dir("componentExtraction", "COVID-19Chart") 
    img = read_image(image_path) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    # 1) Detect only the border-connected blue background 
    
    background_mask = get_border_connected_background_mask( hsv, lower_blue=(85, 25, 40), upper_blue=(135, 255, 255) ) 
    # 2) Invert to get all cards as foreground candidates 

    foreground_mask = get_foreground_from_background(background_mask) 
    # 3) Light cleanup only 
    
    cleaned_foreground = clean_foreground_mask(foreground_mask) 
    # 4) Split merged cards using watershed 

    markers, sure_fg, unknown = watershed_split_foreground( img, cleaned_foreground, dist_threshold=0.28 ) 
    # 5) Extract final boxes 

    boxes = extract_boxes_from_markers( markers, min_area=350, min_width=30, min_height=20, aspect_ratio_range=(0.6, 3.5) ) 
    rectangle_mask = boxes_to_mask(img.shape, boxes) 
    final_overlay = draw_boxes(img, boxes, color=(0, 255, 0), thickness=2) 
    # Save outputs 
    save_image(os.path.join(out_dir, "01_original.png"), img) 
    save_image(os.path.join(out_dir, "02_background_mask.png"), background_mask) 
    save_image(os.path.join(out_dir, "03_foreground_mask.png"), foreground_mask) 
    save_image(os.path.join(out_dir, "04_cleaned_foreground.png"), cleaned_foreground) 
    save_image(os.path.join(out_dir, "05_sure_foreground.png"), sure_fg) 
    save_image(os.path.join(out_dir, "06_unknown.png"), unknown) 
    save_image(os.path.join(out_dir, "07_rectangle_mask.png"), rectangle_mask) 
    save_image(os.path.join(out_dir, "08_final.png"), final_overlay) 
    save_comparison( [ img, background_mask, foreground_mask, cleaned_foreground, sure_fg, unknown, rectangle_mask, final_overlay ], 
                    [ "Original", "Background Mask", "Foreground Mask", "Cleaned Foreground", "Sure Foreground", "Unknown", "Rectangle Mask", "Final" ], 
                    os.path.join(out_dir, "comparison.png"), cols=3 )