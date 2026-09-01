import os
import re
import shutil
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import albumentations as A
import ants
import cv2
import numpy as np
import utils.Logger as loggerz
from utils.util import touint8, reset_img
import yaml
from utils.util_fluor import atlas_reg_ByT1w, atlas_reg_noT1w, get_maskBywatershed, centerxy_img, atlas_reg_ByT1w_v2, \
    translate_bycenter, get_bmask, reassign_anomalous_pixels, atlas_reg_ByT1w_missing
from memory_profiler import memory_usage
import time

blockface_YAML_PATH = os.getcwd() + '/config/blockface_config.yaml'
blockface_CONFIG = yaml.safe_load(open(blockface_YAML_PATH, 'r'))
fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))

def recon_blockface():
    logger = loggerz.get_logger()
    logger.info('3D recon blockface')
    imgs_path = Path(blockface_CONFIG['subject_dir']+'/2Dblockface/')
    files = sorted(
        imgs_path.glob('Section*_qualified_b.png'),
        key=lambda p: int(re.search(r'Section(\d+)', p.stem).group(1))
    )
    b_data=np.zeros((500,len(files),500))
    # 要完整路径字符串
    files = [str(f) for f in files]
    for i,img_path in enumerate(files):
        slice_tmp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        b_data[:, i, :] = slice_tmp.copy()
    b=ants.from_numpy(b_data)
    b.to_file(blockface_CONFIG['subject_dir']+'/b.nii.gz')

def align_Bcenter():
    # img=ants.image_read(blockface_CONFIG['subject_dir']+'/b.nii.gz')
    img = ants.image_read(blockface_CONFIG['subject_dir'] + '/b_recon.nii.gz')
    img_data=img.numpy()[:,:,:].copy()
    tmp=[]
    for i in range(img_data.shape[1]-1):
        slice1=img_data[:,i,:]
        slice2 = img_data[:, i+1, :]
        mse=mean_squared_error(slice1, slice2)
        tmp.append(mse)
    plt.plot(tmp)
    plt.show()
    tmp=np.array(tmp)
    t=900
    indexs=np.where(tmp>t)[0]
    indexs=np.insert(indexs, 0, 0)
    # indexs = np.array([0,122])
    for i in range(len(indexs)-1,-1,-1):
        mov_data=img_data[:, indexs[i], :].copy()
        mov=ants.from_numpy(mov_data)
        fix_data=img_data[:, indexs[i] + 1, :].copy()
        fix = ants.from_numpy(fix_data)
        if i+1>len(indexs)-1:
            end=indexs[len(indexs)-2]
        else:
            end = indexs[i-1]
        basex, basey = ants.get_center_of_mass(fix)
        movx, movy = ants.get_center_of_mass(mov)
        tx = basex - movx
        ty = basey - movy
        tx=ty=0
        mov[:, :] = translate_bycenter(mov[:, :].numpy(), tx, ty)
        # Translation
        t = ants.registration(fix, mov,type_of_transform='Affine', aff_metric='GC', aff_sampling=32)
        # for n in range(indexs[i], end-1, -1):
        for n in range(indexs[i],-1, -1):
            mov = ants.from_numpy(img_data[:, n, :])
            mov[:, :] = translate_bycenter(mov[:, :].numpy(), tx, ty)
            img_data[:, n, :]=ants.apply_transforms(fix,mov,t['fwdtransforms'],'bSpline')[:,:].numpy().copy()

    img_data[img_data<0]=0
    img[:,:,:]=img_data
    img.to_file(blockface_CONFIG['subject_dir']+'/b_recon.nii.gz')

def oc_blockface_toNMT():
    img=ants.image_read(blockface_CONFIG['subject_dir']+'/b_recon.nii.gz')
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    img_data=img.numpy()
    tmp = np.transpose(img_data, [2, 0, 1])
    # tmp=img_data
    tmp=np.flip(tmp,1)
    tmp = np.flip(tmp, 2)
    tmp = np.flip(tmp, 0)
    img=ants.from_numpy(tmp)
    img.set_origin(nmt.origin)
    img.set_direction(nmt.direction)
    img.to_file(blockface_CONFIG['subject_dir']+'/b_recon_oc.nii.gz')


def intensity_c():
    logger = loggerz.get_logger()
    logger.info('Intensity correction')
    CLAHE = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(10, 10), always_apply=True)
    img=ants.image_read(blockface_CONFIG['subject_dir']+'/b_recon_oc.nii.gz')
    for i in range(img.shape[1]):
        fslice=img[:,i,:].numpy()
        fslice=touint8(fslice)
        fslice=CLAHE.apply(fslice,clip_limit=2)
        img[:, i, :]=fslice
    mask=ants.get_mask(img)
    img=ants.denoise_image(img,mask)
    img=ants.n4_bias_field_correction(img,mask,shrink_factor=2)
    img=ants.mask_image(img,mask)
    img.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')

def b_alignMRI():
    logger = loggerz.get_logger()
    logger.info('blockface align to MRI')
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')
    # b = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/b_recon_oc_scale.nii.gz')
    b=ants.from_numpy(b.numpy())
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    # nmt_=nmt
    nmt_=ants.from_numpy(nmt.numpy())
    t2 = ants.registration(nmt_, b, 'Affine', aff_metric='GC',aff_sampling=32)
    b_ = ants.apply_transforms(nmt_, b, t2['fwdtransforms'],'bSpline')
    shutil.copyfile(t2['fwdtransforms'][0], fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat')
    mask=ants.get_mask(b_,10)
    b_=ants.mask_image(b_,mask)
    # b_=ants.denoise_image(b_,mask)
    b_=ants.n4_bias_field_correction(b_,mask,shrink_factor=4)
    b_ = ants.n4_bias_field_correction(b_, mask, shrink_factor=2)
    b_=ants.copy_image_info(nmt,b_)
    b_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/b_recon_oc_scale_alignMRI.nii.gz')

def correct_t1like():
    tsfer=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB.nii.gz')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    mask=ants.get_mask(b,10)
    img = ants.mask_image(tsfer, mask)
    # t = ants.registration(b, img, 'SyN',syn_metric='CC',reg_iterations=(40,20,20),flow_sigma=3,total_sigma=0.6,syn_sampling=4)
    # img=ants.apply_transforms(b,img,t['fwdtransforms'],'bSpline')
    img.to_file(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    mask.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/B_mask.nii.gz')

def blockface_3Dreg():
    logger = loggerz.get_logger()
    logger.info('fMOST PI register to NMT')
    if MRI_CONFIG['MRI-guided']:
        logger.warning('MRI-guided registration')
        start_time = time.time()
        memory_usage_data = memory_usage(atlas_reg_ByT1w, interval=5.0)
        # memory_usage_data = memory_usage(atlas_reg_ByT1w_v2, interval=5.0)
        # memory_usage_data = memory_usage(atlas_reg_ByT1w_missing, interval=5.0)
        end_time = time.time()
        average_memory_usage = sum(memory_usage_data) / len(memory_usage_data)
        print(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning(f"average_memory : {average_memory_usage:.2f}MB")
        total_time = end_time - start_time
        print(f"total time：{total_time:.2f}s")
        logger.warning(f"total time : {total_time:.2f}s")
        logger.warning('MRI-guided registration end')
    else:
        logger.warning('no MRI-guided registration')
        atlas_reg_noT1w()
        logger.warning('no MRI-guided registration end')


def b_invetalignMRI():
    logger = loggerz.get_logger()
    logger.info('blockface inverse align to MRI')
    method=''
    isAffine=True
    b_origin = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/b_recon_oc_scale_clahe.nii.gz')
    b=ants.from_numpy(b_origin.numpy())
    b_alignMRI = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    b_ = ants.from_numpy(b_alignMRI.numpy())
    atlas=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inblockface.nii.gz')
    atlas1 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_inblockface.nii.gz')
    atlas1 = ants.from_numpy(atlas1.numpy())
    atlas2 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM2_inblockface.nii.gz')
    atlas3 = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/SARM6_inblockface.nii.gz')
    atlas4 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/cerebellum_mask_inblockface.nii.gz')
    atlas4 = ants.from_numpy(atlas4.numpy())
    atlas6 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM1_inblockface.nii.gz')
    atlas7 = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/segmentation_edit_inblockface.nii.gz')
    atlas7 = ants.from_numpy(atlas7.numpy())
    nmt = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')
    blikef = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')
    if isAffine:
        t = ants.registration(b, b_, type_of_transform='Affine')
        atlas_ = ants.apply_transforms(b, atlas, t['fwdtransforms'],'multiLabel')
        atlas1_ = ants.apply_transforms(b, atlas1, t['fwdtransforms'], 'multiLabel')
        atlas2_ = ants.apply_transforms(b, atlas2, t['fwdtransforms'], 'multiLabel')
        atlas3_ = ants.apply_transforms(b, atlas3, t['fwdtransforms'], 'multiLabel')
        atlas4_ = ants.apply_transforms(b, atlas4, t['fwdtransforms'], 'multiLabel')
        nmt_ = ants.apply_transforms(b, nmt, t['fwdtransforms'], 'bSpline')
        atlas6_ = ants.apply_transforms(b, atlas6, t['fwdtransforms'], 'multiLabel')
        atlas7_ = ants.apply_transforms(b, atlas7, t['fwdtransforms'], 'multiLabel')
        blikef_ = ants.apply_transforms(b, blikef, t['fwdtransforms'], 'bSpline')
    else:
        atlas_ = ants.apply_transforms(b, atlas, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'],'multiLabel',whichtoinvert=[True])
        atlas1_ = ants.apply_transforms(b, atlas1, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        atlas2_ = ants.apply_transforms(b, atlas2, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'multiLabel',whichtoinvert=[True])
        atlas3_ = ants.apply_transforms(b, atlas3,[fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/b_regt1.mat'],'multiLabel', whichtoinvert=[True])
        atlas4_ = ants.apply_transforms(b, atlas4, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        nmt_ = ants.apply_transforms(b, nmt, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'bSpline', whichtoinvert=[True])
        atlas6_ = ants.apply_transforms(b, atlas6, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        atlas7_ = ants.apply_transforms(b, atlas7, [fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/xfms/b_regt1.mat'],'multiLabel', whichtoinvert=[True])
        blikef_ = ants.apply_transforms(b, blikef, [fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/b_regt1.mat'], 'bSpline', whichtoinvert=[True])
    atlas_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/D99_inOriginB'+method+'.nii.gz')
    atlas1_ = ants.copy_image_info(b_origin, atlas1_)
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_inOriginB'+method+'.nii.gz')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/SARM2_inOriginB'+method+'.nii.gz')
    atlas3_.to_file(fluor_CONFIG['output_dir'] + '/blockface/atlas/SARM6_inOriginB' + method + '.nii.gz')
    atlas4_ = ants.copy_image_info(b_origin, atlas4_)
    atlas4_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/cerebellum_mask_inOriginB'+method+'.nii.gz')
    nmt_=ants.copy_image_info(b_origin,nmt_)
    nmt_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/TMP_inOriginB'+method+'.nii.gz')
    atlas6_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/CHARM1_inOriginB'+method+'.nii.gz')
    atlas7_.to_file(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB'+method+'.nii.gz')
    blikef_ = ants.copy_image_info(b_origin, blikef_)
    blikef_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/T1wlikeB_inOriginB'+method+'.nii.gz')



def repair_blockface():
    logger = loggerz.get_logger()
    logger.info('repair blockface with segmentation')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')
    seg = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_inOriginB.nii.gz')
    cere_mask = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/atlas/cerebellum_mask_inOriginB.nii.gz')
    b=ants.copy_image_info(seg,b)
    b_rmc=b-ants.mask_image(b,cere_mask,1)
    b_rmc_mask=ants.get_mask(b_rmc)
    b_rmc_mask=ants.morphology(b_rmc_mask,'erode',2)
    b_rmc_mask = ants.morphology(b_rmc_mask, 'dilate',2)
    b_rmc=ants.mask_image(b_rmc,b_rmc_mask,1)
    b_rmc.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc.nii.gz')
    seg = ants.morphology(seg, 'erode', 1)
    seg_data=seg.numpy()
    seg[:,:,:]=seg_data
    b_rmc_repair=b_rmc-ants.mask_image(b_rmc,seg,[1,5])
    b_rmc_repair_mask = ants.get_mask(b_rmc_repair)
    b_rmc_repair_mask=ants.morphology(b_rmc_repair_mask,'erode',3)
    b_rmc_repair_mask = ants.morphology(b_rmc_repair_mask, 'dilate',3)
    b_rmc_repair = ants.mask_image(b_rmc_repair, b_rmc_repair_mask, 1)
    b_rmc_repair.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')

def seg_byt1pi():
    method = 'Method A (CC)'
    MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
    MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))
    if fluor_CONFIG['atlas_repair_bySeg']:
        path = fluor_CONFIG['output_dir']
        if not os.path.exists(path+'/reg3D/seg/'):
            os.mkdir(path+'/reg3D/seg/')
        if not os.path.exists(path + '/reg3D/seg/tmp/'):
            os.mkdir(path + '/reg3D/seg/tmp/')
        subcortex = ants.image_read('template/NMT/NMT_brain/subcortex.nii.gz')
        subcortex_, = reset_img([subcortex])
        img_tmp = ants.image_read(path + '/reg3D/T1likeB_c.nii.gz')
        if MRI_CONFIG['MRI-guided']:
            print('MRI-guided')
            subcortex_ = ants.apply_transforms(subcortex_, subcortex_, [path + '/reg3D/'+method+'/xfms/atlas_NMTtoT1w_1Warp.nii.gz',
                                                                        path + '/reg3D/'+method+'/xfms/atlas_NMTtoT1w_0GenericAffine.mat'],
                                               'multiLabel', whichtoinvert=[False, False])
            subcortex_ = ants.apply_transforms(subcortex_, subcortex_, [path + '/reg3D/'+method+'/xfms/atlas_NMTtoPIinT1w_1Warp.nii.gz',
                                                                        path + '/reg3D/'+method+'/xfms/atlas_NMTtoPIinT1w_0GenericAffine.mat'],
                                               'multiLabel', whichtoinvert=[False, False])
            subcortex_ = ants.apply_transforms(subcortex_, subcortex_, [path + '/reg3D/'+method+'/xfms/atlas_PItoT1w_0GenericAffine.mat',
                                                                        path + '/reg3D/'+method+'/xfms/atlas_PItoT1w_1InverseWarp.nii.gz'],
                                               'multiLabel', whichtoinvert=[True, False])
            subcortex_ = ants.copy_image_info(subcortex, subcortex_)
        elif not MRI_CONFIG['MRI-guided'] and os.path.exists(path + '/reg3D/'+method+'/xfms/atlas_PItoNMT_1Warp.nii.gz'):
            print('no MRI')
            subcortex_ = ants.apply_transforms(subcortex_, subcortex_, [path + '/reg3D/'+method+'/xfms/atlas_PItoNMT_1Warp.nii.gz',
                                                                        path + '/reg3D/'+method+'/xfms/atlas_PItoNMT_0GenericAffine.mat'],'multiLabel',whichtoinvert=[False, False])
        else:
            print('reg')
            tmp=ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
            tmp_,img_tmp_ = reset_img([tmp,img_tmp])
            t=ants.registration(img_tmp_,tmp_,'SyN')
            subcortex_=ants.apply_transforms(img_tmp_,subcortex_,t['fwdtransforms'],'multiLabel')
        subcortex_.to_file(path + '/reg3D/atlas/subcortex_inB.nii.gz')

        img = ants.image_read(path+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
        subcortex = ants.image_read(path+'/reg3D/atlas/subcortex_inB.nii.gz')
        cortex = ants.image_read(path + '/reg3D/'+method+'/atlas/CHARM4_inblockface.nii.gz')
        cortex_data=cortex.numpy()
        cortex_data[cortex_data>0]=1
        cortex[:,:,:]=cortex_data
        cortex=ants.copy_image_info(subcortex, cortex)

        subcortex=subcortex-ants.mask_image(subcortex, cortex, level=[1])
        subcortex=ants.copy_image_info(img_tmp, subcortex)
        # cortex.to_file(path + '/reg/seg/subcortex.nii.gz')
        img_subcortex = ants.mask_image(img, subcortex, level=[1])
        img = img - img_subcortex
        subcortex=ants.copy_image_info(img_tmp,subcortex)
        subcortex = ants.mask_image(img_tmp, subcortex, level=[1])
        img_tmp = img_tmp - subcortex
        mask = ants.get_mask(img, cleanup=2)
        mask = ants.copy_image_info(img_tmp, mask)
        img_tmp = ants.mask_image(img_tmp, mask)
        seg = ants.kmeans_segmentation(img_tmp, 2, kmask=mask, mrf=0.1)
        ants.image_write(seg['segmentation'], path + '/reg3D/seg/tmp/seg0.nii.gz')
        ants.image_write(seg['probabilityimages'][0], path + '/reg3D/seg/tmp/p0_iter0.nii.gz')
        ants.image_write(seg['probabilityimages'][1], path + '/reg3D/seg/tmp/p1_iter0.nii.gz')
        img=ants.copy_image_info(img_tmp,img)
        priorseg = ants.prior_based_segmentation(img, seg['probabilityimages'], mask, 0.15,mrf=0.2)
        ants.image_write(priorseg['segmentation'], path + '/reg3D/seg/tmp/seg1.nii.gz')
        ants.image_write(priorseg['probabilityimages'][0], path + '/reg3D/seg/tmp/p1_iter1.nii.gz')
        ants.image_write(priorseg['probabilityimages'][1], path + '/reg3D/seg/tmp/p2_iter1.nii.gz')
        gm_mask = ants.mask_image(priorseg['segmentation'], priorseg['segmentation'], level=[1])
        ants.image_write(gm_mask, path + '/reg3D/seg/gm_mask.nii.gz')

def repair_atlas():
    logger = loggerz.get_logger()
    logger.info('repair atlas')
    method='Method A (CC)'
    pi=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tipi=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    tmp=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')
    atlas=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inblockface.nii.gz')
    atlas2 = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/atlas/SARM6_inblockface.nii.gz')
    atlas,atlas2,tmp,pi_=reset_img([atlas,atlas2,tmp,pi])
    mask=ants.get_mask(tipi)
    atlas_mask = atlas.clone()
    img_data = atlas.numpy().copy() + atlas2.numpy().copy()
    img_data[img_data > 0] = 100
    atlas_mask[:, :, :] = img_data
    tf2 = ants.registration(pi_,atlas_mask, 'SyNOnly',
                            syn_metric='mattes',
                            reg_iterations=(400, 200, 100),flow_sigma=3,outprefix=fluor_CONFIG['output_dir']+'/reg3D/'+method+'/xfms/atlas_repair_')
    atlas_ = ants.apply_transforms(pi_, atlas, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(pi_, atlas2, tf2['fwdtransforms'], 'multiLabel')
    atlas2_=ants.copy_image_info(pi,atlas2_)
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM6_inPI_repair.nii.gz')
    tmp_ = ants.apply_transforms(pi_, tmp, tf2['fwdtransforms'], 'bSpline')

    if fluor_CONFIG['atlas_repair_bySeg']:
        logger.info('atlas_repair_bySeg')
        gm=ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/seg/gm_mask.nii.gz')
        gm,=reset_img([gm])
        atlas__ = atlas_.clone()
        img_data = atlas__.numpy()
        img_data[img_data > 0] = 100
        atlas__[:, :, :] = img_data
        pi_affine_transform1 = ants.registration(gm, atlas__, 'SyNOnly', syn_metric="mattes", reg_iterations=(400, 200,100),
                                                 flow_sigma=3, total_sigma=0,
                                                 outprefix=fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/xfms/gm_constrain_')
        atlas_ = ants.apply_transforms(gm, atlas_, pi_affine_transform1['fwdtransforms'], 'multiLabel')
        atlas_ = ants.mask_image(atlas_, gm)
    atlas_=ants.copy_image_info(pi,atlas_)
    mask = ants.copy_image_info(pi, mask)
    atlas_=ants.mask_image(atlas_,mask)

    tmp_ = ants.copy_image_info(pi, tmp_)
    if not fluor_CONFIG['atlas_repair_bySeg']:
        atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/D99_inPI_repair_.nii.gz')
        tmp_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/TMP_inT1PI_repair_.nii.gz')
    else:
        atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM6_inPI_repair_byGM.nii.gz')
        tmp_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/TMP_inT1PI_repair_byGM.nii.gz')

def repair_seg_inBlockface():
    path=fluor_CONFIG['output_dir']
    seg = ants.image_read(path + '/blockface/atlas/segmentation_edit_inOriginB.nii.gz')
    b   = ants.image_read(path + '/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    seg=ants.copy_image_info(b,seg)
    mask=ants.get_mask(b)
    seg=ants.mask_image(seg, mask)
    seg_data=seg.numpy()
    b_data=b.numpy()
    first_2LRlabel = False
    first_temlabel = False
    first_wholelabel = False
    first_stemlabel = False
    for i in range(b_data.shape[1]-1,-1,-1):
        seg_slice=seg_data[:,i,:]
        b_slice=b_data[:,i,:]
        seg_labels=np.unique(seg_slice)
        seg_labels=seg_labels[seg_labels!=0]
        whole_mask, masks = get_bmask(b_slice)
        seg_slice = seg_slice * (whole_mask // 255).astype(np.uint8)
        if i==67:
            1+1
            print(i)
        if 160 in seg_labels or 210 in seg_labels:
            first_2LRlabel=True
        if len(seg_labels) <= 1 and not first_2LRlabel:
            continue
        if len(seg_labels) > 2 and not first_2LRlabel :
            if len(masks)==2:
                seg_slice = reassign_anomalous_pixels(seg_slice)
            elif len(masks) < 2 and 0<len(masks):
                seg_slice=reassign_anomalous_pixels(seg_slice)
        elif len(seg_labels) > 2 and not first_temlabel:
            if len(masks)==1:
                first_temlabel=True
                seg_slice[seg_slice>0]=2

            whole_ratio=len(seg_slice[seg_slice==2])/len(seg_slice[seg_slice>0])
            if len(masks) == 2 and whole_ratio<0.4:
                for idx, mask in enumerate(masks, 2):
                    seg_slice_=(mask//255).astype(np.uint8)*seg_slice
                    labels = np.unique(seg_slice_)
                    if 150 in labels or  100 in labels or 200 in labels:
                        seg_slice[mask > 0] = 100
            elif len(masks) == 2 and whole_ratio>0.4:
                first_temlabel = True
                seg_slice[seg_slice > 0] = 2
            elif len(masks) == 3:
                seg_slice = reassign_anomalous_pixels(seg_slice,valid_values=(100,160, 210), anomaly=2)
                if 210 in np.unique(seg_slice) and  160 in np.unique(seg_slice):
                    seg_slice[seg_slice==150]=100
                    seg_slice[seg_slice == 200] = 100
            elif len(masks) == 4:
                seg_slice = reassign_anomalous_pixels(seg_slice,valid_values=(100,160, 210), anomaly=2)
                if 210 in np.unique(seg_slice) and  160 in np.unique(seg_slice):
                    seg_slice[seg_slice==150]=100
                    seg_slice[seg_slice == 200] = 100

            if len(seg_slice[seg_slice==2])>10:
                seg_slice[seg_slice>0]=2

        elif not first_wholelabel and first_temlabel:
            if len(masks) >= 3 and not 160 in seg_labels and not 210 in seg_labels and len(seg_labels)>1:
                first_wholelabel=True
                seg_slice[seg_slice > 0] = 2
            elif len(masks) >= 3:
                seg_slice[seg_slice > 0] = 2
            elif len(masks) < 3:
                seg_slice[seg_slice>0]=2
        elif not first_stemlabel and first_wholelabel:
            if len(masks) == 2:
                centroids=[]
                for idx, mask in enumerate(masks, 2):
                    h_idx, w_idx = np.where(mask > 0)  # 所有前景像素坐标
                    centroid = ( w_idx.mean(),h_idx.mean())  # (row, col)
                    centroids.append(centroid)
                min_h, min_w = min(centroids)
                if min_h<150:
                    if mask[int(min_w), int(min_h)] > 0:
                        seg_slice[mask > 0] = 88
                else:
                    first_stemlabel=True
            seg_slice = reassign_anomalous_pixels(seg_slice, valid_values=(150, 200), anomaly=2)
            if len(masks) == 3:
                centroids=[]
                for idx, mask in enumerate(masks, 2):
                    h_idx, w_idx = np.where(mask > 0)  # 所有前景像素坐标
                    centroid = ( w_idx.mean(),h_idx.mean())  # (row, col)
                    centroids.append(centroid)
                min_h, min_w = min(centroids)
                for idx, mask in enumerate(masks, 2):
                    if mask[int(min_w),int(min_h)]>0:
                        seg_slice[mask > 0] = 88
            if len(masks) > 3:
                centroids = []
                for idx, mask in enumerate(masks, 2):
                    area=len(mask[mask>0])
                    if area<20:
                        continue
                    h_idx, w_idx = np.where(mask > 0)  # 所有前景像素坐标
                    centroid = ( w_idx.mean(),h_idx.mean())  # (row, col)
                    centroids.append(centroid)
                min_h, min_w = min(centroids)
                for idx, mask in enumerate(masks, 2):
                    if mask[int(min_w),int(min_h)]>0:
                        seg_slice[mask > 0] = 88

        seg_data[:, i, :]=seg_slice
    seg[:,:,:]=seg_data
    seg.to_file(path+'/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')


# def repair_seg_inBlockface():
#     img = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB.nii.gz')
#     b=ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
#     mask=ants.get_mask(b)
#     img=ants.copy_image_info(b,img)
#     img_=ants.mask_image(img,mask)
#     img_.to_file(fluor_CONFIG['output_dir']+'/blockface/tttt.nii.gz')
#     img_data=img_.numpy()
#     index=None
#     # index = 154
#     for i in range(img.shape[1]-1,0,-1):
#         slice=img_data[:,i,:]
#         # print(set(np.unique(slice).astype(int)))
#         if set(np.unique(slice).astype(int)) == {0,160, 210, 100} or set(np.unique(slice).astype(int)) == {0,160, 210, 100,150,200}:
#             print(i)
#             index=i
#             print(np.unique(slice).astype(int))
#             break
#     # index=152
#     for ii in range(index,img.shape[1]):
#         slice = img_data[:, ii, :]
#         if 160 in np.unique(slice).astype(int) or 210 in np.unique(slice).astype(int):
#             remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,160, 210, 100]))
#             print(remaining_values)
#             tmp=img_data[:, ii, :]
#             tmp[tmp==150]=100
#             tmp[tmp == 200] = 100
#     index=None
#     for i in range(img.shape[1]-1,0,-1):
#         slice=img_data[:,i,:]
#         if set(np.unique(slice).astype(int)) == {0,2}:
#             print(i)
#             index=i
#             print(np.unique(slice).astype(int))
#             break
#     for ii in range(index,img.shape[1]):
#         slice = img_data[:, ii, :]
#         if 2 in np.unique(slice).astype(int) :
#             remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,2]))
#             print(remaining_values)
#             tmp=img_data[:, ii, :]
#             for t in remaining_values:
#                 tmp[tmp==t]=2
#     index=None
#     for i in range(0,img.shape[1]):
#         slice=img_data[:,i,:]
#         if set(np.unique(slice).astype(int)) == {0,2}:
#             print(i)
#             index=i
#             print(np.unique(slice).astype(int))
#             break
#     for ii in range(index,0,-1):
#         slice = img_data[:, ii, :]
#         if 2 in np.unique(slice).astype(int) :
#             remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,2]))
#             print(remaining_values)
#             tmp=img_data[:, ii, :]
#             for t in remaining_values:
#                 tmp[tmp==t]=2
#     index = None
#     for i in range(0, img.shape[1]):
#         slice = img_data[:, i, :]
#         if set(np.unique(slice).astype(int)) == {0, 2}:
#             print(i)
#             index = i
#             print(np.unique(slice).astype(int))
#             break
#     count = 0
#     for ii in range(index - 1, 0, -1):
#         slice_origin = np.rot90(img_data[:, ii, :].copy()).copy()
#         slice = slice_origin.copy()
#         slice[slice > 0] = 100
#         slice = touint8(slice)
#         w = get_maskBywatershed(slice)
#         for i in np.unique(w):
#             area = w[w == i]
#             if len(area) < 300:
#                 w[w == i] = 1
#         # plot_show(slice[:, :], w, True)
#         c_list = []
#         for iii in np.unique(w):
#             if iii != 1:
#                 w_ = w.copy()
#                 w_[w != iii] = 0
#                 sx, sy = centerxy_img(w_ * 35)
#                 if not (sx == 0 and sy == 0):
#                     c_list.append((sx, sy))
#         if len(c_list) == 3:
#             max_y_point = max(c_list, key=lambda point: point[1])
#             w[w != w[max_y_point[1], max_y_point[0]]] = 0
#             w[w > 0] = 1
#             slice_ = slice * w
#             slice_[slice_ > 0] = 88
#             img_data[:, ii, :] = np.rot90(slice_origin - slice_origin * w + slice_, 3).copy()
#         else:
#             count = count + 1
#         if count > 10:
#             break
#
#     img[:, :, :] = img_data
#     img.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')