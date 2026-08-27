#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：PI_MRI.py
@Author  ：Zauber
@Date    ：2024/6/3
"""
from utils.CycleGan_3D.test import PI_to_T1_cyclegan, PI_to_T1_cyclegan_, GFP_to_T2_cyclegan
from utils.fMOST_PI_preproc import tif_to_nii, normalize_to_8bit, denoise_img, remove_artifact, mas_cerebellum, \
    clahe_image, intensity_c, PI_alignNMT, correct_T1like, fMOST_PI_3Dreg, upsample_toOrigin, mergePI, repair_atlas, \
    seg_byt1pi


def PI_preproc():
    # tif_to_nii()
    # normalize_to_8bit()
    # mas_cerebellum()
    # remove_artifact()
    # denoise_img()
    # intensity_c()
    # clahe_image()
    # PI_alignNMT()
    PI_to_T1_cyclegan()
    correct_T1like()
    # mergePI()
    fMOST_PI_3Dreg()
    # seg_byt1pi()
    # repair_atlas()
    # upsample_toOrigin()