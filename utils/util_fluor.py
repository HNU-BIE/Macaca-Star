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
    method = 'Method A (CC)'
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method)
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/')
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/')
    print('MRI: '+fluor_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_edit.nii.gz')
    t1=ants.image_read(fluor_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_edit.nii.gz')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    b_mask = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/atlas/B_mask.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    atlas6 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    t1, tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3,atlas4,atlas5,atlas6,b_mask = reset_img([t1, tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3,atlas4,atlas5,atlas6,b_mask])
    print('Reg iter1: MRI <--> NMT')
    tf1 = ants.registration(t1,tmp, 'SyN',
                            syn_metric='CC',syn_sampling=4,
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.5,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_NMTtoT1w_')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inT1w.nii.gz')
    print('Reg iter2: T1like <--> MRI')
    tsfer=ants.mask_image(tsfer,b_mask)
    blockface = ants.mask_image(blockface, b_mask)
    tf3 = ants.registration(t1,tsfer, 'SyN',
                            syn_metric='mattes',syn_sampling=32,
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.7,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_PItoT1w_')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tf3['warpedmovout']))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1PI_inT1w.nii.gz')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf3['fwdtransforms'], 'bSpline')
    print('Reg iter3: T1like_ <--> NMT_')
    tf2 = ants.registration(tsfer_,tmp_, 'SyN',
                            syn_metric='CC',syn_sampling=2,
                            reg_iterations=(2400,1200,40),flow_sigma=5,total_sigma=0.7,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_NMTtoPIinT1w_')
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')

    ####################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inT1w.nii.gz')
    atlas_ = ants.apply_transforms(tsfer_, atlas_, tf2['fwdtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf3['invtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inblockface.nii.gz')

    atlas1_ = ants.apply_transforms(t1, atlas1, tf1['fwdtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM4_inT1w.nii.gz')
    atlas1_ = ants.apply_transforms(tsfer_, atlas1_, tf2['fwdtransforms'], 'multiLabel')
    atlas1_ = ants.apply_transforms(tsfer, atlas1_, tf3['invtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM4_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM4_inT1w.nii.gz')
    atlas2_ = ants.apply_transforms(tsfer_, atlas2_, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf3['invtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM4_inblockface.nii.gz')

    atlas3_ = ants.apply_transforms(t1, atlas3, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_edit_inT1w.nii.gz')
    atlas3_ = ants.apply_transforms(tsfer_, atlas3_, tf2['fwdtransforms'], 'genericLabel')
    atlas3_ = ants.apply_transforms(tsfer, atlas3_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_edit_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas4_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/cerebellum_mask_inT1w.nii.gz')
    atlas4_ = ants.apply_transforms(tsfer_, atlas4_, tf2['fwdtransforms'], 'genericLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas4_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/cerebellum_mask_inblockface.nii.gz')

    atlas5_ = ants.apply_transforms(t1, atlas5, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas5_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_inT1w.nii.gz')
    atlas5_ = ants.apply_transforms(tsfer_, atlas5_, tf2['fwdtransforms'], 'genericLabel')
    atlas5_ = ants.apply_transforms(tsfer, atlas5_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas5_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_inblockface.nii.gz')

    atlas6_ = ants.apply_transforms(t1, atlas6, tf1['fwdtransforms'], 'multiLabel')
    atlas6_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM6_inT1w.nii.gz')
    atlas6_ = ants.apply_transforms(tsfer_, atlas6_, tf2['fwdtransforms'], 'multiLabel')
    atlas6_ = ants.apply_transforms(tsfer, atlas6_, tf3['invtransforms'], 'multiLabel')
    atlas6_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM6_inblockface.nii.gz')

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1w_inNMT.nii.gz')

    img_ = ants.apply_transforms(t1, tsfer, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1PI_inT1w.nii.gz')
    img_ = ants.apply_transforms(tmp, img_, tf2['invtransforms'], 'bSpline')
    img_ = ants.apply_transforms(tmp, img_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1PI_inNMT.nii.gz')

    blockface_ = ants.apply_transforms(t1, blockface, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/blockface_inT1w.nii.gz')
    blockface_ = ants.apply_transforms(tmp, blockface_, tf2['invtransforms'], 'bSpline')
    blockface_ = ants.apply_transforms(tmp, blockface_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/blockface_inNMT.nii.gz')

def atlas_reg_ByT1w_v2():
    t1=ants.image_read(fluor_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/CHARM_1_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    t1, tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3,atlas4,atlas5 = reset_img([t1, tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3,atlas4,atlas5])
    print('Reg iter1: T1like <--> MRI')
    tf1 = ants.registration(t1,tsfer, 'SyN',
                            syn_metric='meansquares',syn_sampling=32,
                            reg_iterations=(2100,1200,1200,20),flow_sigma=3,total_sigma=1,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_btoT1w_v2_')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf1['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tsfer_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1B_inT1w.nii.gz')
    print('Reg iter2: T1like in MRI <--> NMT')
    tf2 = ants.registration(tmp,tsfer_, 'SyN',
                            syn_metric='meansquares',syn_sampling=32,
                            reg_iterations=(2100,1200,1200,20),flow_sigma=3,total_sigma=0.15,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_btoNMT_v2_',redius=4)

    tsfer_ = ants.apply_transforms(tmp,tsfer_, tf2['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tsfer_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1PI_inNMT.nii.gz')



    ####################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf2['invtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf1['invtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/D99_inblockface.nii.gz')

    atlas1_ = ants.apply_transforms(t1, atlas1, tf2['invtransforms'], 'multiLabel')
    atlas1_ = ants.apply_transforms(tsfer, atlas1_, tf1['invtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/CHARM1_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf2['invtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf1['invtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/SARM6_inblockface.nii.gz')

    atlas3_ = ants.apply_transforms(t1, atlas3, tf2['invtransforms'], 'genericLabel')
    atlas3_ = ants.apply_transforms(tsfer, atlas3_, tf1['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_edit_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf2['invtransforms'], 'genericLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf1['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas4_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/cerebellum_mask_inblockface.nii.gz')

    atlas5_ = ants.apply_transforms(t1, atlas5, tf2['invtransforms'], 'genericLabel')
    atlas5_ = ants.apply_transforms(tsfer, atlas5_, tf1['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas5_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_inblockface.nii.gz')

    blockface_ = ants.apply_transforms(tmp, blockface, tf1['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/blockface_inMRI.nii.gz')
    blockface_ = ants.apply_transforms(tmp, blockface_, tf2['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/blockface_inNMT.nii.gz')


def atlas_reg_noT1w():
    print('no MRI')
    method = 'Method D (CC) withOriginCycle'
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method)
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/')
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeBlockface_origin.nii.gz')
    # tsfer = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    atlas6 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3, atlas4, atlas5,atlas6 = reset_img([tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3, atlas4, atlas5,atlas6])
    tf1 = ants.registration(tsfer,tmp, 'SyN',
                            syn_metric='CC',syn_sampling=4,
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.1,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_PItoNMT_',redius=4)
    img_ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')
    ###################################################################

    atlas_ = ants.apply_transforms(tsfer, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inblockface.nii.gz')

    atlas1_ = ants.apply_transforms(tsfer, atlas1, tf1['fwdtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM4_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(tsfer, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM4_inblockface.nii.gz')

    atlas6_ = ants.apply_transforms(tsfer, atlas6, tf1['fwdtransforms'], 'multiLabel')
    atlas6_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM6_inblockface.nii.gz')

    atlas3_ = ants.apply_transforms(tsfer, atlas3, tf1['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_edit_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(tsfer, atlas4, tf1['fwdtransforms'], 'genericLabel')
    atlas4_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/cerebellum_mask_inblockface.nii.gz')

    atlas5_ = ants.apply_transforms(tsfer, atlas5, tf1['fwdtransforms'], 'multiLabel')
    atlas5_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_inblockface.nii.gz')

    img_ = ants.apply_transforms(tmp, tsfer, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin, img_)
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/T1PI_inNMT.nii.gz')

    blockface_ = ants.apply_transforms(tmp, blockface, tf1['invtransforms'], 'bSpline')
    blockface_ = ants.copy_image_info(tmp_origin, blockface_)
    blockface_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/blockface_inNMT.nii.gz')


def atlas_reg_ByT1w_missing(subject,method='Method A (CC)_Lesion',p=0,LR=None):
    fluor_CONFIG['output_dir']=subject
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method)
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/')
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/')
    t1 = ants.image_read(fluor_CONFIG['output_dir'] + '/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    pi=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    mask = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/atlas/B_mask.nii.gz')
    mask_data=mask.numpy().copy()
    les=int(mask_data.shape[1] * p)
    mask_data[:,0:les,:]=0
    mask_data[:, mask_data.shape[1]-les:mask_data.shape[1], :] = 0
    mask[:,:,:]=mask_data
    pi = ants.mask_image(pi, mask)
    tsfer = ants.mask_image(tsfer, mask)
    pi.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/B_Lesion_'+str(p)+'.nii.gz')
    tsfer.to_file(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/T1B_Lesion'+str(p)+'.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    tmp_mask = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_brainmask.nii.gz')
    tmp_origin=ants.mask_image(tmp_origin, tmp_mask)
    atlas2 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    # atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    # atlas4 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    atlas=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2_=ants.mask_image(atlas2,atlas2,5)
    atlas2_data=atlas2_.numpy()
    atlas2_data[atlas2_data>0]=1
    atlas2_[:,:,:]=atlas2_data
    tmp_origin=tmp_origin-ants.mask_image(tmp_origin, atlas2_)
    tmp_origin.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/NMT.nii.gz')
    atlas.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/D99.nii.gz')
    t1,tsfer,pi,tmp,atlas,atlas2,atlas4,mask=reset_img([t1,tsfer,pi,tmp_origin,atlas,atlas2,atlas4,mask])
    print('Reg iter1: T1like in MRI <--> NMT')
    start_time = time.time()
    # syn_sampling 4  total_sigma 0.5 reg3D_iterations=(2400,1200,40)
    tf1 = ants.registration(t1,tmp, 'SyN',syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg3D_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.5,singleprecision=True,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_NMTtoT1w_',verbose=False)
    img__ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/TMP_inT1w.nii.gz')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline' )
    print('reg3D iter2: T1like <--> MRI')
    # syn_sampling 4  total_sigma 0.7 reg3D_iterations=(1200,1200,40)
    if p>=0.3:
        type_reg='Similarity'
    elif p >= 0.2:
        type_reg = 'Affine'
    else:
        type_reg='SyN'
    print(type_reg)
    tf_mask = ants.registration(t1,tsfer, type_reg,aff_metric='mattes',
                            reg_iterations=(40,20,0),outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/mask_PItoT1w_Affine_',verbose=False)
    tsfer_ = ants.apply_transforms(t1, tsfer, tf_mask['fwdtransforms'], 'bSpline')
    tsfer_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/tsfer_affinetoT1w.nii.gz')
    mask_ = ants.apply_transforms(t1, mask, tf_mask['fwdtransforms'], 'multiLabel')
    mask_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/mask_affinetoT1w.nii.gz')
    # mask_=mask
    t1_=ants.mask_image(t1, mask_)
    t1_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/T1w.nii.gz')
    tsfer_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/tsfer_syntoT1w.nii.gz')

    tf3 = ants.registration(t1_,tsfer, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.1,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_PItoT1w_',verbose=False)

    tsfer_ = ants.apply_transforms(t1_,tsfer, tf3['fwdtransforms'],whichtoinvert=[False,False], interpolator='bSpline')
    mask_ = ants.apply_transforms(t1_, mask, tf3['fwdtransforms'], 'multiLabel')
    tf3['warpedmovout'].to_file(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/T1B_inT1w2.nii.gz')
    tsfer_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/T1B_inT1w.nii.gz')
    mask_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/mask_affinetoT1w2.nii.gz')

    print('reg3D iter3: T1like_ <--> NMT_')
    tmp_=ants.mask_image(tmp_, mask_)
    ## syn_sampling 2  total_sigma 0.7 reg_iterations=(2400,1200,40)
    tf2 = ants.registration(tsfer_,tmp_, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.2,
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.0,singleprecision=True,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_NMTtoPIinT1w_',verbose=False)
    end_time = time.time()
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, tmp_)
    img__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/TMP_inT1B.nii.gz')
    ###################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/Charm4_inT1w.nii.gz')
    atlas_ = ants.apply_transforms(tsfer_, atlas_, tf2['fwdtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf3['invtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/Charm4_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/Seg_inT1w.nii.gz')
    atlas2_ = ants.apply_transforms(tsfer_, atlas2_, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf3['invtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/Seg_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf1['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/SARM4_inT1w.nii.gz')
    atlas4_ = ants.apply_transforms(tsfer_, atlas4_, tf2['fwdtransforms'], 'multiLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf3['invtransforms'], 'multiLabel')
    atlas4_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/SARM4_inblockface.nii.gz')

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/T1w_inNMT.nii.gz')

    img_ = ants.apply_transforms(t1, tsfer, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/T1B_inT1w.nii.gz')
    img_ = ants.apply_transforms(t1, img_, tf2['invtransforms'], 'bSpline')
    img_ = ants.apply_transforms(tmp, img_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, img_)
    img__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/T1B_inNMT.nii.gz')

    pi_ = ants.apply_transforms(t1, pi, tf3['fwdtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, ants.image_clone(pi_))
    pi__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/blockface_inT1w.nii.gz')
    pi_ = ants.apply_transforms(t1, pi_, tf2['invtransforms'], 'bSpline')
    pi_ = ants.apply_transforms(tmp, pi_, tf1['invtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, pi_)
    pi__.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/blockface_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

def atlas_reg_noT1w_missing(subject,method='Method A (CC)_Lesion',p=0,LR=None):
    print('no MRI')
    fluor_CONFIG['output_dir']=subject
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method)
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/')
    if not os.path.exists(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/'):
        os.mkdir(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/')
    tsfer = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/T1likeB_c.nii.gz')
    pi=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    mask = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/atlas/B_mask.nii.gz')
    mask_data=mask.numpy().copy()
    les=int(mask_data.shape[1] * p)
    mask_data[:,0:les,:]=0
    mask_data[:, mask_data.shape[1]-les:mask_data.shape[1], :] = 0
    mask[:,:,:]=mask_data
    pi = ants.mask_image(pi, mask)
    tsfer = ants.mask_image(tsfer, mask)
    pi.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/B_Lesion_'+str(p)+'.nii.gz')
    tsfer.to_file(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/T1B_Lesion'+str(p)+'.nii.gz')
    tmp_ = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas3=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    tsfer, pi, tmp, atlas3, atlas4 = reset_img([tsfer, pi, tmp_, atlas3, atlas4])
    start_time = time.time()
    if p>=0.3:
        type_reg='Similarity'
    elif p>=0.2:
        type_reg='Affine'
    else:
        type_reg='SyN'
    print(type_reg)
    tf_mask = ants.registration(tsfer,tmp, type_of_transform=type_reg,aff_metric='mattes',
                            reg_iterations=(40,20,0),outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/mask_PItoT1w_Affine_',verbose=False)
    mask_ = ants.apply_transforms(tmp, mask, tf_mask['invtransforms'], 'multiLabel')
    mask_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/mask_affinetoNMT.nii.gz')
    # mask_=mask
    tmp_=ants.mask_image(tmp, mask_)
    tmp_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/NMT.nii.gz')

    tf = ants.registration(tsfer,tmp_, 'SyN',syn_metric='mattes',syn_sampling=32,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_PItoNMT_',
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0)
    end_time = time.time()
    tf['warpedmovout'].to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/TMP_inT1B.nii.gz')
    ####################################################################
    atlas3_ = ants.apply_transforms(pi, atlas3, tf['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/CHARM4_inB.nii.gz')
    atlas4_ = ants.apply_transforms(pi, atlas4, tf['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/SARM4_inB.nii.gz')
    img_ = ants.apply_transforms(tmp, pi, tf['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_, img_)
    img_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/blockface_inNMT.nii.gz')
    tsfer_ = ants.apply_transforms(tmp, tsfer, tf['invtransforms'], 'bSpline')
    tsfer_ = ants.copy_image_info(tmp_, tsfer_)
    tsfer_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/T1B_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

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