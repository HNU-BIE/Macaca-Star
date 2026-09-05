#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：PI_MRI.py
@Author  ：Zauber
"""
from utils.CycWaveM3D.test import PI_to_T1_cyclegan, PI_to_T1_cyclegan
from utils.fMOST_PI_preproc import tif_to_nii, normalize_to_8bit, denoise_img, remove_artifact, mas_cerebellum, \
    clahe_image, intensity_c, PI_alignNMT, correct_T1like, fMOST_PI_3Dreg, upsample_toOrigin, repair_atlas, \
    seg_byt1pi


def PI_preproc():
    """
    Modular and customizable pipeline for PI data preprocessing, modality synthesis,
    and 3D anatomical registration. Each processing module is designed to be fully
    decoupled to support customized analysis workflows.
    """

    # tif_to_nii()                 # Step 1: Convert raw multi-slice TIFF files into a 3D NIfTI (.nii.gz) volume
    # normalize_to_8bit()          # Step 2: Rescale and normalize image intensities to 8-bit dynamic range [0, 255]
    # mas_cerebellum()             # Step 3: Segment and apply cerebellum mask to isolate target brain regions
    # remove_artifact()            # Step 4: Detect and remove imaging stripe artifacts (destriping)
    # denoise_img()                # Step 5: Apply image denoising filters to improve signal-to-noise ratio (SNR)
    # intensity_c()                # Step 6: Perform intensity non-uniformity and bias field correction
    # clahe_image()                # Step 7: Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # PI_alignNMT()                # Step 8: Perform initial rigid/affine spatial alignment from PI to NMT atlas space
    PI_to_T1_cyclegan()          # Step 9: Synthesize T1-like volume from PI using 3D CycWaveMamba model
    # correct_T1like()             # Step 10: Mask the synthesized T1-like volume to remove non-brain background
    # fMOST_PI_3Dreg()             # Step 11: 3D non-linear registration (MRI-guided via in vivo MRI or MRI-free to NMT)
    # upsample_toOrigin()          # Step 12: Upsample atlas back to the original full-resolution space