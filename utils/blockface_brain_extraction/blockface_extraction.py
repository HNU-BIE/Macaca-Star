#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import cv2
import numpy as np

def extract_image_with_mask_smooth(image_path, mask_path, output_path,
                                   blur_kernel=21, blur_sigma=5):
    """
    Extract a specific region of interest (ROI) from an image with smoothed/anti-aliased mask boundaries.

    :param image_path: Path to the source RGB image file.
    :param mask_path: Path to the binary ROI mask file.
    :param output_path: Destination path for the smoothly extracted image.
    :param blur_kernel: Gaussian blur kernel size (must be an odd integer; larger values produce softer edges).
    :param blur_sigma: Gaussian blur standard deviation controlling boundary transition width.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Validate image and mask loading
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return

    # Binarize mask to floating-point values {0.0, 1.0}
    mask_binary = (mask > 128).astype(np.float32)

    # Apply Gaussian blur to create a soft-edged transition boundary
    mask_smooth = cv2.GaussianBlur(mask_binary, (blur_kernel, blur_kernel), blur_sigma)
    mask_3channel = mask_smooth[..., np.newaxis]
    background = np.zeros_like(image, dtype=np.float32)
    extracted = image.astype(np.float32) * mask_3channel + background * (1 - mask_3channel)
    extracted = np.clip(extracted, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, extracted)
    print(f"Saved smoothed result: {output_path}")


def process_images(image_folder, mask_folder, output_folder, blur_kernel=21, blur_sigma=5):
    """
    Batch process and extract foreground image slices with smooth mask borders.

    :param image_folder: Directory containing source image slices.
    :param mask_folder: Directory containing corresponding binary masks.
    :param output_folder: Destination directory for extracted images.
    :param blur_kernel: Gaussian blur kernel size for boundary smoothing.
    :param blur_sigma: Gaussian blur standard deviation.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all PNG files in the image directory
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)
            output_path = os.path.join(output_folder, filename)

            if not os.path.exists(mask_path):
                print(f"Mask file not found: {mask_path}")
                continue

            # Extract foreground with smoothed boundaries
            extract_image_with_mask_smooth(
                image_path, mask_path, output_path,
                blur_kernel=blur_kernel,
                blur_sigma=blur_sigma
            )
