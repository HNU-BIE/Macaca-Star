#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：MRI_brain_preproc.py
@Author  ：Zauber
@Date    ：2024/6/3
"""
import os
import shutil

import numpy as np
import yaml
import ants
import utils.Logger as loggerz
from utils.util import reset_img

MRI_CONFIG = yaml.safe_load(open(os.getcwd() + '/config/MRI_config.yaml', 'r'))

OUTPUT_DIR=None
CONFIG=None

def set_output_dir(type):
    global OUTPUT_DIR, CONFIG
    if type==0:
        YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
        fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
        CONFIG=fMOST_PI_CONFIG
        OUTPUT_DIR = fMOST_PI_CONFIG['output_dir']
    elif type==1:
        YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
        fluor_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
        CONFIG=fluor_CONFIG
        OUTPUT_DIR = fluor_CONFIG['output_dir']
    elif type==2:
        YAML_PATH = os.getcwd() + '/config/fMOST_GFP_config.yaml'
        fMOST_GFP_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
        CONFIG=fMOST_GFP_CONFIG
        OUTPUT_DIR = fMOST_GFP_CONFIG['output_dir']
    return OUTPUT_DIR

def brain_orientation_correction():
    global OUTPUT_DIR
    logger = loggerz.get_logger()
    logger.info('MRI orientation correction')
    mri=ants.image_read(MRI_CONFIG['MRI_file'])
    if MRI_CONFIG['MRI_oc']:
        nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
        logger.info('MRI oc')
        img_data = mri.numpy()
        img_data = np.transpose(img_data, MRI_CONFIG['MRI_transpose'])
        if MRI_CONFIG['MRI_flip'][0] == 1:
            img_data = np.flip(img_data, 0)
        if MRI_CONFIG['MRI_flip'][1] == 1:
            img_data = np.flip(img_data, 1)
        if MRI_CONFIG['MRI_flip'][2] == 1:
            img_data = np.flip(img_data, 2)
        img_ = ants.from_numpy(img_data)
        img_.set_direction(nmt.direction)
        img_.set_origin(nmt.origin)
        img_.set_spacing(mri.spacing)
        # mri.set_direction(MRI_CONFIG['metrix'])
        # mri.set_spacing(mri.spacing)
        img_.to_file(OUTPUT_DIR+'/MRI/MRI_oc.nii.gz')
    else:
        shutil.copy(MRI_CONFIG['MRI_file'], OUTPUT_DIR + '/MRI/MRI_oc.nii.gz')

def rm_neck():
    global OUTPUT_DIR
    mri = ants.image_read(OUTPUT_DIR+'/MRI/MRI_oc.nii.gz')
    mri_data=mri.numpy()
    # mri_data_=mri_data[:,0:mri_data.shape[1]-int(MRI_CONFIG['rm_neck']),:].copy()
    if MRI_CONFIG['rm_background']==0:
        rm_b=mri_data.shape[2]
    else:
        rm_b=MRI_CONFIG['rm_background']
    mri_data_ = mri_data[:, :,int(MRI_CONFIG['rm_neck']):rm_b].copy()
    mri_=ants.from_numpy(mri_data_)
    mri_.set_direction(mri.direction)
    mri_.set_origin(mri.origin)
    mri_.set_spacing(mri.spacing)
    mri_.to_file(OUTPUT_DIR+'/MRI/MRI_oc_.nii.gz')


def make_brain_mask():
    logger = loggerz.get_logger()
    logger.info('skull brain')
    nmt=ants.image_read('template/NMT/NMT_fh/NMT_v2.0_sym_fh.nii.gz')
    nmt_mask = ants.image_read('template/NMT/NMT_fh/NMT_v2.0_sym_fh_brainmask.nii.gz')
    mri=ants.image_read(OUTPUT_DIR+'/MRI/MRI_oc_.nii.gz')
    mri_=ants.resample_image(mri,nmt.spacing,interp_type=4)
    mri_.to_file(OUTPUT_DIR+'/MRI/MRI_oc_0.25mm.nii.gz')
    t = ants.registration(mri_,nmt, type_of_transform='SyN',syn_metric="mattes")
    mri_mask=ants.apply_transforms(mri_,nmt_mask,t['fwdtransforms'],'multiLabel')
    mri_mask.to_file(OUTPUT_DIR+'/MRI/MRI_oc_mask.nii.gz')

def extract_brainByMask():
    logger = loggerz.get_logger()
    logger.info('extract MRI brain')
    mri=ants.image_read(OUTPUT_DIR+'/MRI/MRI_oc_0.25mm.nii.gz')
    mask=ants.image_read(OUTPUT_DIR+'/MRI/MRI_oc_mask.nii.gz')
    mask=ants.morphology(mask,'dilate',radius=3)
    mask = ants.morphology(mask, 'erode', radius=3)
    brain=ants.mask_image(mri,mask)
    brain.to_file(OUTPUT_DIR+'/MRI/MRI_brain.nii.gz')


def MRI_preprocess():
    logger = loggerz.get_logger()
    logger.info('MRI denoise and bias correction')
    mri=ants.image_read(OUTPUT_DIR+'/MRI/MRI_brain.nii.gz')
    mask=ants.get_mask(mri,1)
    mri_ = ants.n4_bias_field_correction(mri, mask,shrink_factor=8)
    mri_ = ants.n4_bias_field_correction(mri_,mask, shrink_factor=4)
    mri_ = ants.n4_bias_field_correction(mri_, mask,shrink_factor=2)
    mri_=ants.denoise_image(mri_,mask, noise_model="Rician",p=2, r=3)
    # mri_ = ants.iMath_truncate_intensity(mri_, 0.01, 0.98)
    mri_.to_file(OUTPUT_DIR+'/MRI/MRI_brain_bc_dn.nii.gz')


def crop_brain():
    mri=ants.image_read(OUTPUT_DIR+'/MRI/MRI_brain_bc_dn.nii.gz')
    nmt=ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    nmt_,mri_=reset_img([nmt,mri])
    t=ants.registration(nmt_,mri_,'Similarity')
    mri_=ants.apply_transforms(nmt_,mri_,t['fwdtransforms'],'bSpline')
    mri_=ants.copy_image_info(nmt,mri_)
    mri_.to_file(OUTPUT_DIR+'/MRI/MRI_brain_bc_dn.nii.gz')
    if not CONFIG['wholeBrain']:
        if CONFIG['LR'] == 'L':
            mri_[int(mri_.shape[0] / 2):mri_.shape[0], 0:mri_.shape[1], 0:mri_.shape[2]] = 0
        elif CONFIG['LR'] == 'R':
            mri_[0:int(mri_.shape[0] / 2), 0:mri_.shape[1], 0:mri_.shape[2]] = 0
        mri_.to_file(OUTPUT_DIR+'/MRI/MRI_brain_bc_dn_.nii.gz')
    else:
        mri_.to_file(OUTPUT_DIR+'/MRI/MRI_brain_bc_dn_.nii.gz')

def get_logger():
    global OUTPUT_DIR, CONFIG
    return OUTPUT_DIR, CONFIG