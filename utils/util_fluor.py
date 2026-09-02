#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：util.py
@Author  ：Zauber
@Date    ：2025/3/1
"""
import os
import matplotlib.pyplot as plt
import ants
import cv2
import numpy as np
import yaml
from skimage import util
import imutils
from utils.util import reset_img, touint8
from scipy.spatial.distance import cdist
from scipy.ndimage import label, binary_dilation, generate_binary_structure
import time

blockface_YAML_PATH = os.getcwd() + '/config/blockface_config.yaml'
blockface_CONFIG = yaml.safe_load(open(blockface_YAML_PATH, 'r'))
fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))


def atlas_reg_ByT1w():
    """
    Perform MRI-guided 3D non-linear registration (SyN) connecting the standard NMT template,
    subject-specific in vivo T1w MRI, and synthetic T1-like block-face optical data.

    Workflow:
      1. Prepare output directories for registration outputs and transformation matrices.
      2. Load subject in vivo T1w MRI, synthetic T1-like volume, block-face volume, and NMT atlases.
      3. Standardize spatial origins across all volumes (reset_img).
      4. Reg Iter 1: Deformable registration from NMT template to in vivo T1w MRI space (SyN with CC metric).
      5. Reg Iter 2: Deformable registration from synthetic T1-like block-face to in vivo T1w MRI (SyN with Mattes metric).
      6. Reg Iter 3: Non-linear refinement between warped NMT and warped synthetic T1-like volume.
      7. Invert and compose transformations to map NMT template into native block-face space (NMT_inblockface.nii.gz).
    """
    method = ''
    atlas_level = 6  # Hierarchical parcellation level for CHARM (cortical) and SARM (subcortical) atlases

    # 1. Ensure required output directories exist (atlas/ and xfms/)
    output_reg_dir = os.path.join(fluor_CONFIG['output_dir'], 'reg3D', method)
    os.makedirs(os.path.join(output_reg_dir, 'atlas'), exist_ok=True)
    os.makedirs(os.path.join(output_reg_dir, 'xfms'), exist_ok=True)

    # 2. Load subject in vivo T1w MRI, synthetic T1-like volume, aligned block-face volume, and mask
    print('MRI: '+fluor_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_.nii.gz')
    t1=ants.image_read(fluor_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    b_mask = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/atlas/B_mask.nii.gz')

    # 3. Load standard NMT template brain and anatomical parcellation atlases
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/CHARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/SARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    atlas6 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')

    # 4. Standardize spatial origins across all volumes
    t1, tsfer, blockface, tmp, atlas1, atlas2,atlas3,atlas4,atlas5,atlas6,b_mask = reset_img([t1, tsfer, blockface, tmp_origin, atlas1, atlas2,atlas3,atlas4,atlas5,atlas6,b_mask])

    # =========================================================================
    # Reg Iter 1: Deformable registration from NMT Template -> In vivo T1w MRI
    # =========================================================================
    print('Reg iter1: MRI <--> NMT')
    tf1 = ants.registration(
        fixed=t1,
        moving=tmp,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        reg_iterations=(1200, 1200, 40),
        flow_sigma=3,
        total_sigma=0.5,
        outprefix=fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/atlas_NMTtoT1w_'
    )
    # Save NMT template warped into T1w MRI space
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inT1w.nii.gz')

    # =========================================================================
    # Reg Iter 2: Deformable registration from Synthetic T1 -> In vivo T1w MRI
    # =========================================================================
    print('Reg iter2: T1like <--> MRI')
    # Apply brain foreground mask to optical data
    tsfer=ants.mask_image(tsfer,b_mask)
    blockface = ants.mask_image(blockface, b_mask)
    tf3 = ants.registration(
        fixed=t1,
        moving=tsfer,
        type_of_transform='SyN',
        syn_metric='mattes',
        syn_sampling=32,
        reg_iterations=(1200, 1200, 40),
        flow_sigma=3,
        total_sigma=0.7,
        outprefix=fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/atlas_PItoT1w_'
    )
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tf3['warpedmovout']))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1PI_inT1w.nii.gz')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf3['fwdtransforms'], 'bSpline')

    # =========================================================================
    # Reg Iter 3: Non-linear refinement between warped T1-like volume and warped NMT
    # =========================================================================
    print('Reg iter3: T1like_ <--> NMT_')
    tf2 = ants.registration(
        fixed=tsfer_,
        moving=tmp_,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=(1200, 1200, 40),
        flow_sigma=5,
        total_sigma=0.7,
        outprefix=fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/atlas_NMTtoPIinT1w_'
    )
    # 5. Invert and compose transforms to map NMT template back into native block-face space
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')

    atlas_save_dir = fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas'

    # =========================================================================
    # Transform anatomical atlases into T1w MRI and Blockface spaces
    # =========================================================================
    # Mapping: {filename_prefix: (atlas_image_object, interpolator_type)}
    atlases_dict = {
        'CHARM4': (atlas1, 'multiLabel'),
        'SARM4': (atlas2, 'multiLabel'),
        'segmentation_edit': (atlas3, 'genericLabel'),
        'cerebellum_mask': (atlas4, 'genericLabel'),
        'segmentation': (atlas5, 'genericLabel'),
        'SARM6': (atlas6, 'multiLabel'),
    }

    for name, (cur_atlas, interp) in atlases_dict.items():
        # Step A: Transform to T1w MRI space and save
        atlas_in_t1 = ants.apply_transforms(fixed=t1, moving=cur_atlas, transformlist=tf1['fwdtransforms'],
                                            interpolator=interp)
        img_out = ants.copy_image_info(tmp_origin, ants.image_clone(atlas_in_t1))
        img_out.to_file(f"{atlas_save_dir}/{name}_inT1w.nii.gz")

        # Step B: Transform through intermediate space to native block-face space and save
        atlas_in_tsfer = ants.apply_transforms(fixed=tsfer_, moving=atlas_in_t1, transformlist=tf2['fwdtransforms'],
                                               interpolator=interp)
        atlas_in_bf = ants.apply_transforms(fixed=tsfer, moving=atlas_in_tsfer, transformlist=tf3['invtransforms'],
                                            interpolator=interp)
        img_out = ants.copy_image_info(tmp_origin, ants.image_clone(atlas_in_bf))
        img_out.to_file(f"{atlas_save_dir}/{name}_inblockface.nii.gz")

    # =========================================================================
    # Block-face / Synthetic T1 -> T1w MRI -> NMT
    # =========================================================================
    # Mapping: {filename_prefix: intensity_image_object}
    intensity_images_dict = {
        'T1PI': tsfer,  # Synthetic T1-like volume
        'blockface': blockface  # Aligned optical block-face volume
    }

    for name, cur_img in intensity_images_dict.items():
        # Step A: Warp into T1w MRI space and save
        img_in_t1 = ants.apply_transforms(fixed=t1, moving=cur_img, transformlist=tf3['fwdtransforms'],
                                          interpolator='bSpline')
        img_out = ants.copy_image_info(tmp_origin, ants.image_clone(img_in_t1))
        img_out.to_file(f"{atlas_save_dir}/{name}_inT1w.nii.gz")

        # Step B: Warp further into standard NMT template space and save
        img_in_nmt = ants.apply_transforms(fixed=tmp, moving=img_in_t1, transformlist=tf2['invtransforms'],
                                           interpolator='bSpline')
        img_in_nmt = ants.apply_transforms(fixed=tmp, moving=img_in_nmt, transformlist=tf1['invtransforms'],
                                           interpolator='bSpline')
        img_out = ants.copy_image_info(tmp_origin, ants.image_clone(img_in_nmt))
        img_out.to_file(f"{atlas_save_dir}/{name}_inNMT.nii.gz")

    # =========================================================================
    # 3. Warp in vivo T1w MRI into standard NMT template space
    # =========================================================================
    t1_in_nmt = ants.apply_transforms(fixed=tmp, moving=t1, transformlist=tf1['invtransforms'], interpolator='bSpline')
    t1_in_nmt = ants.copy_image_info(tmp_origin, t1_in_nmt)
    t1_in_nmt.to_file(f"{atlas_save_dir}/T1w_inNMT.nii.gz")


def atlas_reg_noT1w():
    """
    Perform direct 3D non-linear anatomical registration (SyN) between the synthetic
    T1-like block-face volume and the standard NMT template without in vivo MRI guidance.

    Workflow:
      1. Prepare output directories for registration outputs and transformation matrices.
      2. Load synthetic T1-like volume, block-face volume, NMT template, and anatomical atlases/masks.
      3. Standardize spatial coordinate origins across all volumes (reset_img).
      4. Perform direct SyN deformable registration from NMT template (moving) to synthetic T1 (fixed).
      5. Transform anatomical atlases into native block-face space using forward transforms (multiLabel/genericLabel).
      6. Warp optical intensity volumes into standard NMT template space using inverse transforms (bSpline).
    """
    print('no MRI')
    method = ''

    # Ensure required output directories exist (atlas/ and xfms/)
    output_reg_dir = os.path.join(fluor_CONFIG['output_dir'], 'reg3D', method)
    os.makedirs(os.path.join(output_reg_dir, 'atlas'), exist_ok=True)
    os.makedirs(os.path.join(output_reg_dir, 'xfms'), exist_ok=True)

    # Load synthetic T1-like volume, aligned block-face volume, and reference NMT template
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeBlockface_origin.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')

    # 3. Load anatomical parcellation atlases and tissue masks
    atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    atlas6 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')

    # Standardize spatial origins across all volumes
    tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3, atlas4, atlas5,atlas6 = reset_img([tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3, atlas4, atlas5,atlas6])

    # =========================================================================
    # Direct Deformable Registration: NMT Template -> Synthetic T1 Block-face
    # =========================================================================
    tf1 = ants.registration(
        fixed=tsfer,
        moving=tmp,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        reg_iterations=(2400, 1200, 40),
        flow_sigma=3,
        total_sigma=0.1,
        outprefix=fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/atlas_PItoNMT_',
        redius=4
    )
    # Save NMT template warped to native block-face space
    img_ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')

    # =========================================================================
    # 1. Loop over Atlases & Masks: Warp NMT -> Native Block-face Space
    # =========================================================================
    # Mapping: {filename_prefix: (atlas_image_object, interpolator_type)}
    atlas_save_dir = fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas'
    atlases_dict = {
        'D99': (atlas, 'multiLabel'),
        'CHARM4': (atlas1, 'multiLabel'),
        'SARM4': (atlas2, 'multiLabel'),
        'SARM6': (atlas6, 'multiLabel'),
        'segmentation_edit': (atlas3, 'multiLabel'),
        'cerebellum_mask': (atlas4, 'genericLabel'),
        'segmentation': (atlas5, 'multiLabel'),
    }

    for name, (cur_atlas, interp) in atlases_dict.items():
        atlas_in_bf = ants.apply_transforms(fixed=tsfer, moving=cur_atlas, transformlist=tf1['fwdtransforms'], interpolator=interp)
        atlas_in_bf.to_file(f"{atlas_save_dir}/{name}_inblockface.nii.gz")

    # =========================================================================
    # 2. Loop over Intensity Images: Warp Block-face / Synthetic T1 -> NMT Space
    # =========================================================================
    # Mapping: {filename_prefix: intensity_image_object}
    intensity_images_dict = {
        'T1PI': tsfer,          # Synthetic T1-like volume
        'blockface': blockface  # Aligned optical block-face volume
    }

    for name, cur_img in intensity_images_dict.items():
        img_in_nmt = ants.apply_transforms(fixed=tmp, moving=cur_img, transformlist=tf1['invtransforms'], interpolator='bSpline')
        img_in_nmt = ants.copy_image_info(tmp_origin, img_in_nmt)
        img_in_nmt.to_file(f"{atlas_save_dir}/{name}_inNMT.nii.gz")


def get_fslice_mask(img,hole_size=50):
    isPlot=False
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    img = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    # gray = util.invert(img)
    gray=img
    ret, mask = cv2.threshold(gray, 10, 255,  cv2.THRESH_BINARY)
    threshod_image_erode = cv2.erode(mask, kernel2, iterations=1)

    contours, _ = cv2.findContours(threshod_image_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= hole_size:
            cv_contours.append(contour)
        else:
            continue
    threshod_image_erode=cv2.fillPoly(threshod_image_erode, cv_contours, 255)
    plot_show(gray, threshod_image_erode, isPlot)
    ret, threshod_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    threshod_image=threshod_image_erode+threshod_image
    plot_show(gray, threshod_image, isPlot)
    threshod_image[threshod_image>0]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshod_image, cv2.MORPH_OPEN, kernel, iterations=2)
    plot_show(img, opening, isPlot)
    mask=opening
    mask[mask>0]=1
    return mask


def plot_show(image,image2,isPlot=False):
    if isPlot:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image,cmap='gray')
        ax1.set_title('image')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image2,cmap='gray')
        ax2.set_title('segmentation')
        plt.show()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def get_maskBywatershed(img):
    isPlot=False
    gray = util.invert(img)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, threshod_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, threshod_image = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
    plot_show(img, threshod_image, isPlot)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5 3,3
    threshod_image_erode = cv2.erode(threshod_image, kernel2, iterations=1)
    threshod_image = fill_hole(threshod_image_erode)+threshod_image
    threshod_image[threshod_image>0]=255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshod_image, cv2.MORPH_OPEN, kernel, iterations=2)
    print('MORPH_OPEN')
    plot_show(img, opening, isPlot)
    # opening=threshod_image
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    dist_transform_threshold_image = dist_transform.copy()
    print('dist_transform_threshold_image')
    plot_show(img, dist_transform_threshold_image, isPlot)
    # dist_transform_threshold_image[dist_transform_threshold_image < 0.1] = 0
    # dist_transform_threshold_image[dist_transform_threshold_image >= 0.1] = 255
    dist_transform_threshold_image[dist_transform_threshold_image < 0.03] = 0
    dist_transform_threshold_image[dist_transform_threshold_image >= 0.03] = 255
    dist_transform_threshold_image = touint8(dist_transform_threshold_image)

    dilate_image = cv2.dilate(opening, kernel, iterations=2)
    unknown = cv2.subtract(dilate_image, dist_transform_threshold_image)

    ret2, markers = cv2.connectedComponents(dist_transform_threshold_image)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers_copy = markers.copy()
    markers_copy[markers == 0] = 150
    markers_copy[markers == 1] = 0
    markers_copy[markers > 1] = 255
    markers = cv2.watershed(image, markers)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for obj_id in np.unique(markers):
        if obj_id == 0:
            continue
        if obj_id == -1:
            mask[markers == obj_id] = 1
            continue
        mask[markers == obj_id] = obj_id
    plot_show(img, mask,isPlot)
    return mask


def fill_hole(img):

    mask = 255 - img
    marker = np.zeros_like(img)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
    count = 0
    while True:
        count += 1
        marker_pre = marker
        dilation = cv2.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    return dst

def centerxy_img(image):
    image[image < 0] = 0
    image = image.astype(np.uint8)
    thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    xs = []
    ys = []
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if not M["m00"] == 0:
            cX = int(M["m10"] / M["m00"])
            xs.append(cX)
            cY = int(M["m01"] / M["m00"])
            ys.append(cY)
    try:
        x = int(np.mean(xs))
        y = int(np.mean(ys))
    except:
        x=0
        y=0
    return x, y

def repair_mask(bf_mask,b_mask,b_seg):
    for i in np.unique(bf_mask):
        area=bf_mask[bf_mask==i]
        if len(area)<fluor_CONFIG['ex_min_blikef_area']:
            bf_mask[bf_mask == i]=1
    for i in np.unique(b_mask):
        area=b_mask[b_mask==i]
        if len(area)<fluor_CONFIG['ex_min_b_area']:
            b_mask[b_mask == i]=1
    b_seg=get_maskBywatershed(touint8(b_seg*35))
    b_seg[b_seg<=1]=0
    for i in np.unique(b_seg):
        area=b_seg[b_seg==i]
        if len(area)<fluor_CONFIG['ex_min_b_area']:
            b_seg[b_seg == i]=0

    if len(np.unique(bf_mask))!=len(np.unique(b_mask)):
        if len(np.unique(bf_mask))==len(np.unique(b_seg)):
            b_mask_tmp=b_seg
            n=1
            for i in np.unique(b_mask_tmp):
                b_mask_tmp[b_mask_tmp==i]=n
                n=n+1
        elif len(np.unique(bf_mask))> len(np.unique(b_seg)):
            bf_clist = []
            b_clist = []
            bf_flag=[]
            b_flag=[]
            for i in np.unique(bf_mask):
                if i !=1:
                    bf_mask_ = bf_mask.copy()
                    bf_mask_[bf_mask != i] = 0
                    bfx, bfy = centerxy_img(bf_mask_ * 35)
                    if not (bfx == 0 and bfy == 0):
                        if bfy>=350 and bfx>200 and bfx<300:
                            # bf_mask=center_editmask(bf_mask, bfx, bfy, 1)
                            bf_flag.append((bfx, bfy))
                        else:
                            bf_clist.append((bfx, bfy))

            for i in np.unique(b_seg):
                if i !=0:
                    b_seg_ = b_seg.copy()
                    b_seg_[b_seg != i] = 0
                    bx, by = centerxy_img(b_seg_ * 35)
                    if not (bfx == 0 and bfy == 0):
                        if by>=350 and bx>200 and bx<300:
                            # b_mask=center_editmask(bf_mask, bx, by, 1)
                            b_flag.append((bfx, bfy))
                        else:
                            b_clist.append((bfx, bfy))
            if len(b_flag)==len(bf_flag) and len(b_flag)>=1 and len(bf_clist)==2:
                if bf_clist[0][1]+10>bf_clist[1][1] and bf_clist[0][1]-10<bf_clist[1][1]:
                    bf_mask = center_editmask(bf_mask, bf_clist[0][0], bf_clist[0][1], 6)
                    bf_mask = center_editmask(bf_mask, bf_clist[1][0], bf_clist[1][1], 6)
                    n = 1
                    for i in np.unique(bf_mask):
                        bf_mask[bf_mask == i] = n
                        n = n + 1
            elif  len(bf_flag)==1 and len(b_flag)==0:
                bf_mask = center_editmask(bf_mask, bf_flag[0][0], bf_flag[0][1], 1)
            if len(np.unique(bf_mask))> len(np.unique(b_seg)) and len(np.unique(b_seg))>2:
                bf_distances = cdist(bf_clist, bf_clist)
                np.fill_diagonal(bf_distances, np.inf)
                min_distance = np.min(bf_distances)
                min_indices = np.unravel_index(np.argmin(bf_distances), bf_distances.shape)
                point1 = bf_clist[min_indices[0]]
                point2 = bf_clist[min_indices[1]]
                bf_mask = center_editmask(bf_mask, point2[0], point2[1], 6)
                bf_mask = center_editmask(bf_mask, point1[0], point1[1], 6)
                n = 1
                for i in np.unique(bf_mask):
                    bf_mask[bf_mask == i] = n
                    n = n + 1
                # bf_mask[bf_mask==bf_mask[point2[1],point2[0]]]=bf_mask[point1[1],point1[0]]
                if len(np.unique(bf_mask))==len(np.unique(b_mask)):
                    b_mask_tmp = b_mask
                else:
                    b_mask_tmp = b_seg
                    n = 1
                    for i in np.unique(b_mask_tmp):
                        b_mask_tmp[b_mask_tmp == i] = n
                        n = n + 1
            elif len(np.unique(b_seg))==2:
                b_mask_tmp = b_seg
                n = 1
                for i in np.unique(b_mask_tmp):
                    b_mask_tmp[b_mask_tmp == i] = n
                    n = n + 1
                bf_mask[bf_mask>1]=2
            else:
                b_mask_tmp=b_seg
                n = 1
                for i in np.unique(b_mask_tmp):
                    b_mask_tmp[b_mask_tmp == i] = n
                    n = n + 1
        elif len(np.unique(bf_mask)) < len(np.unique(b_seg)):
            b_mask_tmp = b_mask
            b_mask_tmp[b_mask_tmp>1]=2
            bf_mask[bf_mask>1]=2
    else:
        bf_clist = []
        b_clist = []
        bf_flag = []
        b_flag = []
        for i in np.unique(bf_mask):
            if i != 1:
                bf_mask_ = bf_mask.copy()
                bf_mask_[bf_mask != i] = 0
                bfx, bfy = centerxy_img(bf_mask_ * 35)
                if not (bfx == 0 and bfy == 0):
                    if bfy >= 350 and bfx > 200 and bfx < 300:
                        bf_flag.append((bfx, bfy))
                    else:
                        bf_clist.append((bfx, bfy))

        for i in np.unique(b_seg):
            if i != 0:
                b_seg_ = b_seg.copy()
                b_seg_[b_seg != i] = 0
                bx, by = centerxy_img(b_seg_ * 35)
                if not (bfx == 0 and bfy == 0):
                    if by >= 350 and bx > 200 and bx < 300:
                        b_flag.append((bfx, bfy))
                    else:
                        b_clist.append((bfx, bfy))
        if len(b_flag) != len(bf_flag):
            b_mask_tmp = b_mask
            b_mask_tmp[b_mask_tmp>1]=2
            bf_mask[bf_mask>1]=2
        else:
            b_mask_tmp=b_mask
            n = 1
            for i in np.unique(b_mask_tmp):
                b_mask_tmp[b_mask_tmp == i] = n
                n = n + 1

    return bf_mask,b_mask_tmp

def syn_toB_bySeg(b_img,b_mask,bf_img,bf_mask,f_img,b_seg,index):
    isPlot=False
    dis=fluor_CONFIG['max_dis_centers']
    b_clist = []
    f_clist = []
    matched_pairs = {}
    bf_mask,b_mask_tmp=repair_mask(bf_mask,b_mask,b_seg)
    # bf_mask, b_mask_tmp = repair_mask(bf_mask, b_seg, b_seg)
    plot_show(bf_mask, b_mask_tmp, isPlot)
    for i in np.unique(bf_mask):
        if i != 1:
            b_mask_ = b_mask_tmp.copy()
            f_mask_ = bf_mask.copy()
            b_mask_[b_mask_tmp != i] = 0
            f_mask_[bf_mask != i] = 0
            if len(b_mask_[b_mask_>0])>0:
                bx, by = centerxy_img(b_mask_ * 35)
                fx, fy = centerxy_img(f_mask_ * 35)
                if not (bx == 0 and by == 0):
                    b_clist.append((bx, by))
                if not (fx == 0 and fy == 0):
                    f_clist.append((fx, fy))
            else:
                continue
    b_clist = np.array(b_clist)
    f_clist = np.array(f_clist)
    if len(f_clist)==len(b_clist) and len(f_clist)==1:
        bf_mask[bf_mask>1]=2
        b_mask_tmp[b_mask_tmp>1]=2
    else:
        try:
            distances = cdist(f_clist,b_clist)
            matched_f_points = {}
            for i, f_point in enumerate(f_clist):
                for j, b_point in enumerate(b_clist):
                    if distances[i, j] <= dis:
                        f_point_tuple = tuple(f_point)
                        if f_point_tuple not in matched_f_points or distances[i, j] < matched_f_points[f_point_tuple]:
                            matched_pairs[f_point_tuple] = b_point
                            matched_f_points[f_point_tuple] = distances[i, j]
            keys=list(matched_pairs.keys())
            # bf的两个组织都离b的一个组织最近，则不按距离进行mask赋值
            if (matched_pairs[keys[0]]==matched_pairs[keys[1]])[0] and (matched_pairs[keys[0]]==matched_pairs[keys[1]])[1]:
                matched_pairs={}
                for i, f_point in enumerate(f_clist):
                    f_point_tuple = tuple(f_point)
                    matched_pairs[f_point_tuple]=b_clist[i]
                keys = list(matched_pairs.keys())
            label=22
            for i in range(len(keys)):
                bf_mask=center_editmask(bf_mask,keys[i][0],keys[i][1],label)
                b_mask_tmp = center_editmask(b_mask_tmp, matched_pairs[keys[i]][0], matched_pairs[keys[i]][1], label)
                label=label+10
        except:
            print('center match error')
            bf_mask[bf_mask>1]=2
            b_mask_tmp[b_mask_tmp>1]=2
    plot_show(bf_mask,b_mask_tmp, isPlot)
    cv2.imwrite(fluor_CONFIG['output_dir']+ '/reg2D/xfms/masks/bslice_mask' + str(index) + '.tif', b_mask_tmp)
    cv2.imwrite(fluor_CONFIG['output_dir']+ '/reg2D/xfms/masks/tfslice_mask' + str(index) + '.tif', bf_mask)
    newbf_data = reg_byimgdata(bf_img, bf_mask,f_img, b_img, b_mask_tmp,index)
    return newbf_data


def center_editmask(mask,x,y,value):
    if mask[y,x]==1:
        tmp = np.zeros((500, 500))
        tmp[y - 10:y + 10, x - 10: x + 10] = 1
        tmp = tmp * mask
        tmplist = np.unique(tmp)
        tmplist.sort()
        if len(tmplist) == 3:
            mask[mask == tmplist[2]] = value
        else:
            tmp = np.zeros((500, 500))
            tmp[y - 10:y + 10, x - 10: x + 10] = 1
            tmp = tmp * mask
            tmplist = np.unique(tmp)
            tmplist.sort()
            if len(tmplist) == 3:
                mask[mask == tmplist[2]] = value
    else:
        mask[mask == mask[y, x]] = value
    return mask

def reg_byimgdata(bf_img, bf_mask,f_img, b_img, b_mask,index):
    newbf_data=np.zeros_like(b_mask)
    newf_data = np.zeros_like(b_mask)
    f_img_data=f_img.numpy()
    bf_img_data = bf_img.numpy()
    b_img_data = b_img.numpy()
    mask = b_mask.copy()
    mask[mask <= 1] = 0
    mask[mask > 1] = 1
    for i in np.unique(bf_mask):
        if i !=1:
            bf_img_data_=bf_img_data.copy()
            f_img_data_ = f_img_data.copy()
            b_img_data_ = b_img_data.copy()
            bf_img_data_[bf_mask != i] = 0
            f_img_data_[bf_mask != i] = 0
            b_img_data_[b_mask != i] = 0
            bf_slice = ants.from_numpy(bf_img_data_)
            f_slice = ants.from_numpy(f_img_data_)
            b_slice = ants.from_numpy(b_img_data_)

            # bf_mask[bf_mask != i] = 0
            # bf_mask[bf_mask>0]=1
            # b_mask[b_mask != i] = 0
            # b_mask[b_mask>0]=1
            # bf_slice_mask = ants.from_numpy(bf_mask)
            # b_slice_mask = ants.from_numpy(b_mask)


            try:

                result = ants.registration(b_slice, bf_slice, 'SyN', syn_metric='mattes', aff_metric='GC',
                                           aff_iterations=(2100, 2100, 2100,2100),aff_sampling=100,aff_random_sampling_rate=1,aff_shrink_factors=(1, 1, 1, 1),
                                           reg_iterations=(2100, 1200, 1200, 20), flow_sigma=3,total_sigma=1,
                                           outprefix=fluor_CONFIG['output_dir']+ '/reg2D/xfms/FtoB_iter1_' + str(
                                               index) + '_part' + str(i) + '_')

                bf_slice_ = ants.apply_transforms(b_slice, bf_slice, result['fwdtransforms'],'linear')
                f_slice_ = ants.apply_transforms(b_slice, f_slice, result['fwdtransforms'], 'linear')
                newbf_data = newbf_data + bf_slice_.numpy().copy()
                newf_data = newf_data + f_slice_.numpy().copy()
                # newbf_data=newbf_data*mask
            except:
                print('reg error')
                newbf_data = bf_slice.numpy().copy()
                newf_data = f_slice.numpy().copy()
                # newbf_data = newbf_data * mask
    return newbf_data,newf_data

def translate_bycenter(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_ = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),flags=cv2.INTER_CUBIC)
    return img_

def get_bmask(img):
    img = touint8(img)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 2. 连通组件（8-连通）
    num, labels = cv2.connectedComponents(bw, connectivity=8)
    # 3. 给每个区域建 mask
    masks = [(labels == i).astype(np.uint8) * 255 for i in range(1, num)]  # 跳过背景 0
    whole_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        whole_mask |= m
    return whole_mask,masks

def reassign_anomalous_pixels(mask, valid_values=(150, 200), anomaly=100):
    """
    把每个独立的 anomaly 连通块改为与它接壤的 valid 区域的标签值。

    Parameters
    ----------
    mask : np.ndarray
        单通道灰度图（2D 或 3D），包含 valid_values、anomaly、背景0。
    valid_values : tuple
        合法的标签集合。
    anomaly : int
        需要重新分配的异常标签值。

    Returns
    -------
    np.ndarray
        修改后的 mask。
    """
    img = mask.copy()
    structure = generate_binary_structure(img.ndim, 1)  # 4/6连通
    anomaly_mask = (img == anomaly)

    # 计算 anomaly 区域的连通块
    labeled, n = label(anomaly_mask, structure=structure)
    if n == 0:
        return img

    for i in range(1, n+1):
        region = (labeled == i)

        # 膨胀一次得到邻域
        dilated = binary_dilation(region, structure=structure)

        # 邻域标签集合（去掉 anomaly 自身和背景）
        border_vals = np.unique(img[dilated & ~region])
        border_vals = [v for v in border_vals if v in valid_values]

        if len(border_vals) == 1:
            # 仅接壤一种合法标签
            img[region] = border_vals[0]
        elif len(border_vals) > 1:
            # 若接壤多种合法标签，选邻域面积最大的那种
            counts = [(v, np.sum(dilated & (img == v))) for v in border_vals]
            best_label = max(counts, key=lambda x: x[1])[0]
            img[region] = best_label
        # 如果完全不接壤任何 valid 区域（孤立），则保持 anomaly

    return img