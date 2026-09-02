#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : Macaca-Star
@File    : blockface_MRI.py
@Desc    : Preprocessing, modal synthesis, and 3D registration pipeline for
           serial block-face imaging data.
"""

from utils.CycWaveM3D.test import b_to_T1_cyclegan
from utils.blockface_preproc import intensity_c, b_alignMRI, correct_t1like, blockface_3Dreg, b_invetalignMRI, \
    repair_blockface, repair_seg_inBlockface, recon_blockface, align_Bcenter, oc_blockface_toNMT, seg_byt1pi, \
    repair_atlas


def blockface_preproc():
    """
    Modular and customizable pipeline for block-face data preprocessing,
    cross-modal synthesis (Block-face -> T1 MRI), and 3D anatomical registration.
    Each module is fully decoupled to support customized workflow execution.
    """
    # align_Bcenter()                  # Step 1: Align centroid coordinates of consecutive serial block-face slice images
    # oc_blockface_toNMT()             # Step 2: Correct spatial orientation and align reconstructed block-face volume to NMT space
    # intensity_c()                    # Step 3: Perform intensity non-uniformity and illumination bias correction
    # b_alignMRI()                     # Step 4: Perform initial spatial alignment from block-face volume to reference MRI
    b_to_T1_cyclegan()                 # Step 5: Synthesize pseudo-T1 MRI from block-face volume using 3D CycWaveMamba
    correct_t1like()                 # Step 6: Mask the synthesized T1-like volume to remove non-brain background noise
    blockface_3Dreg()                  # Step 7: Perform 3D non-linear anatomical registration to standard template
    # b_invetalignMRI()                # Step 8: Invert transformation fields to map reference templates/MRI into native block-face space
    # repair_blockface()               # Step 9: Reconstruct and repair volumetric artifacts in block-face native space
    # seg_byt1pi()                     # Step 10: [Optional] Segment tissue structures on synthetic T1 (for severe deformation cases)
    # repair_atlas()                   # Step 11: [Optional] Repair/refine atlas boundaries using segmentation priors (for severe deformation cases)
    # repair_seg_inBlockface()         # Step 12: [Optional] Map the refined segmentation and atlas parcellations back to native block-face space