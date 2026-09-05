#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from PIL import Image
import ants

def load_images_from_folder(folder_path):
    """
    Load and stack all 2D PNG slice images from a folder into a 3D NumPy volume.

    :param folder_path: Path to the directory containing slice PNG images.
    :return: 3D NumPy array of shape [num_slices, height, width].
    """
    # 1. Discover and sort all PNG image files alphabetically by filename
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')],
                   key=lambda x: os.path.splitext(x)[0])
    images = []
    for file in files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('L')  # 转换为灰度图像
        img_array = np.array(img)
        images.append(img_array)
    return np.stack(images, axis=0)


def combine_folders_to_nifti(folder_a, folder_p, output_path):
    """
    Process, align, downsample, and merge two serial section folders (Anterior/Posterior)
    into a unified 3D NIfTI (.nii.gz) volume.

    :param folder_a: Path to folder A containing anterior/first set of slice images.
    :param folder_p: Path to folder P containing posterior/second set of slice images.
    :param output_path: Destination path for the assembled NIfTI volume (.nii.gz).
    """
    images_a = load_images_from_folder(folder_a)
    images_p = load_images_from_folder(folder_p)
    # Reverse slice sequence of folder A and subsample every 2nd slice
    images_a = np.flip(images_a, axis=0)
    images_a = images_a[::2, :, :].copy()
    images_p = np.flip(images_p, axis=1)
    images_p = images_p[::3, :, :].copy()
    if images_a.shape[1:] != images_p.shape[1:]:
        raise ValueError("Image slice dimensions do not match between the two folders.")

    # Concatenate both volumetric image stacks along slice axis (axis 0)
    combined_images = np.concatenate((images_p,images_a), axis=0)
    combined_ants_image = ants.from_numpy(combined_images, origin=(0, 0, 0), spacing=(1, 1, 1), direction=np.eye(3))
    ants.image_write(combined_ants_image, output_path)
