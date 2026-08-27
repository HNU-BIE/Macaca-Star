import os
import ants
import cv2
import numpy as np
import yaml
from skimage import util
import albumentations as A
from utils.util import touint8, reset_img
from utils.util_fluor import get_fslice_mask, plot_show, normalization, get_maskBywatershed, syn_toB_bySeg
from visdom import Visdom

fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))

def oc_fluor_toNMT():
    img=ants.image_read(fluor_CONFIG['output_dir']+'/fluor/fluor.nii.gz')
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    img_data=img.numpy()
    tmp = np.transpose(img_data, [2, 0, 1])
    img=ants.from_numpy(tmp)
    img.set_origin(nmt.origin)
    img.set_direction(nmt.direction)
    img.to_file(fluor_CONFIG['output_dir']+'/fluor/fluor_oc.nii.gz')
    img=ants.image_read(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')
    img_data=img.numpy()
    tmp=img_data
    tmp=np.flip(tmp,1)
    tmp = np.flip(tmp, 2)
    img=ants.from_numpy(tmp)
    img.set_origin(nmt.origin)
    img.set_direction(nmt.direction)
    img.to_file(fluor_CONFIG['output_dir']+'/fluor/blike_f_oc.nii.gz')

def repaire_blikefluo():
    is_ZBackground=True
    f = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/fluor.nii.gz')
    tf = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')
    f=ants.iMath_normalize(f)*255
    tf = ants.iMath_normalize(tf) * 255
    tf_ = ants.from_numpy(np.zeros((f.shape[0],f.shape[1],f.shape[2])))
    tf_.set_spacing(f.spacing)
    tf_.set_direction(f.direction)
    tf_.set_origin(f.origin)
    tf_aug=ants.image_clone(tf_)

    CLAHE2 = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), always_apply=True)

    for i in range(0, f.shape[1]):
        print(str(i) + ' ' + str(f.shape[1]))
        f_slice_data = f[:, i, :].numpy()
        tf_slice_data=tf[:, i, :].numpy()
        tf_slice_data = tf_slice_data.astype(np.uint8)
        f_slice_data = f_slice_data.astype(np.uint8)
        tf_slice = ants.from_numpy(tf_slice_data)
        f_slice = ants.from_numpy(f_slice_data)
        #########################################################
        mask_slice= ants.image_clone(f_slice)
        mask_slice[:,:]=0
        inf=util.invert(f_slice[:,:].numpy())
        # inf=cv2.GaussianBlur(inf, (3, 3), 1.5)
        inf = cv2.medianBlur(inf, 3)
        infimg=ants.from_numpy(inf)
        mask2=get_fslice_mask(f_slice.numpy().copy(),1000)
        result = ants.registration(infimg, tf_slice, 'SyNOnly',syn_sampling=32,
                                   reg_iterations=(40, 20, 0),flow_sigma=3,total_sigma=0)
        tf_slice = ants.apply_transforms(f_slice, tf_slice, result['fwdtransforms'])
        plot_show(f_slice.numpy(),mask2,False)
        tf_slice_=tf_slice
        tftmp=tf_slice_.numpy()
        tf_augtmp=normalization(tftmp.copy()*(1+normalization(inf)*0.2)*mask2[:, :])*255
        if is_ZBackground:
            maskb=f_slice.numpy().copy()
            maskb[maskb>0]=1
            tf_augtmp=tf_augtmp*maskb[:,:]
            kernel = np.ones((2, 2), np.uint8)
            tf_augtmp = cv2.morphologyEx(tf_augtmp, cv2.MORPH_CLOSE, kernel)*mask2

        tf_augtmp = touint8(tf_augtmp)
        # tf_augtmp = CLAHE2.apply(tf_augtmp, clip_limit=1)
        tf_[:, i, :]=tf_slice_.numpy()
        tf_aug[:, i, :] = tf_augtmp.copy()
    ants.image_write(tf_aug,fluor_CONFIG['output_dir']+'/fluor/blikef_repair2D_aug.nii.gz')

# def repaire_blikefluo():
#     is_ZBackground=False
#     f = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/fluor.nii.gz')
#     tf = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')
#     tf_ = ants.from_numpy(np.zeros((f.shape[0],f.shape[1],f.shape[2])))
#     tf_=ants.copy_image_info(f,tf_)
#     for i in range(0, f.shape[1]):
#         print(str(i) + ' ' + str(f.shape[1]))
#         f_slice_data = f[:, i, :].numpy()
#         tf_slice_data=tf[:, i, :].numpy()
#         tf_slice = ants.from_numpy(tf_slice_data)
#         f_slice = ants.from_numpy(f_slice_data)
#         #########################################################
#         mask_slice= ants.image_clone(f_slice)
#         mask_slice_data=mask_slice.numpy()
#         mask_slice_data[mask_slice_data>0]=1
#
#         inf=util.invert(f_slice[:,:].numpy())
#         mask2=get_fslice_mask(f_slice.numpy().astype(np.uint8),1000)
#         f_slice[:, :] = inf*mask2
#         result = ants.registration(f_slice, tf_slice, 'SyNOnly',syn_sampling=32,
#                                    reg_iterations=(10, 5, 0),flow_sigma=3,total_sigma=3)
#         tf_slice = ants.apply_transforms(f_slice, tf_slice, result['fwdtransforms'],interpolator='bSpline')
#         tf_slice[:,:]=tf_slice.numpy()*mask2
#
#         kernel = np.ones((2, 2), np.uint8)
#         tf_slice[:,:] = cv2.morphologyEx(tf_slice.numpy(), cv2.MORPH_CLOSE, kernel)*mask2
#         # plot_show(tf_slice.numpy(), mask2, True)
#         tf_[:, i, :]=tf_slice.numpy().copy()
#     ants.image_write(tf_,fluor_CONFIG['output_dir']+'/fluor/blikef_repair2D_aug.nii.gz')

def fluor_SyNtoB_bySeg():
    isPlot=False
    viz = Visdom(env='slice2D_fluo_affinetoB')
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    tf = ants.image_read(fluor_CONFIG['output_dir']+ '/fluor/blikef_repair2D_aug.nii.gz')
    f = ants.image_read(fluor_CONFIG['output_dir']+ '/fluor/fluor.nii.gz')
    b_seg = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')
    tf_=ants.new_image_like(tf,tf.numpy())
    tf_[:, :, :] = 0
    f_=ants.new_image_like(tf,tf.numpy())
    f_[:, :, :] = 0
    tf_=ants.image_read(fluor_CONFIG['output_dir']+ '/fluor/blikef_repair2D_aug.nii.gz')
    f_=ants.image_read(fluor_CONFIG['output_dir'] + '/fluor/fluor.nii.gz')
    for index in range(0, tf.shape[1]):
    # for index in range(152,153):
        if set(np.unique(b_seg[:,index,:].numpy().astype(int)))=={0}:
            continue
        print(index)
        tf_[:,index,:]=0
        bslice = touint8(b.numpy()[:, index, :].copy())
        tfslice = touint8(tf.numpy()[:, index, :].copy())
        fslice = touint8(f.numpy()[:, index, :].copy())
        bslice = np.rot90(bslice).copy()
        tfslice = np.rot90(tfslice).copy()
        fslice = np.rot90(fslice).copy()
        bsliceimg = ants.from_numpy(bslice)
        tfsliceimg = ants.from_numpy(tfslice)
        fsliceimg = ants.from_numpy(fslice)
        result1 = ants.registration(bsliceimg, tfsliceimg, 'Similarity', aff_metric='GC',outprefix=fluor_CONFIG['output_dir']+ '/reg2D/xfms/ftob_affine_iter1_'+str(index)+'_')
        tfsliceimg = ants.apply_transforms(bsliceimg, tfsliceimg, result1['fwdtransforms'],'linear')
        fsliceimg = ants.apply_transforms(bsliceimg, fsliceimg, result1['fwdtransforms'], 'linear')
        tfslice = tfsliceimg.numpy().copy()
        bmask = bslice.copy()
        tfmask = tfslice.copy()
        bmask[bmask > 50] = 100
        tfmask[tfmask > 50] = 100
        bslice_mask = get_maskBywatershed(bmask)
        tfslice_mask = get_maskBywatershed(tfmask)
        bslice_mask[bslice_mask == 0] = 1
        tfslice_mask[tfslice_mask == 0] = 1
        plot_show(tfslice_mask, bslice_mask, isPlot)
        newbf_data,newf_data=syn_toB_bySeg(bsliceimg,bslice_mask,tfsliceimg,tfslice_mask,fsliceimg,np.rot90(b_seg[:,index,:].numpy()),index)
        tf_[:, index, :]=np.rot90(newbf_data.copy(),3)
        f_[:, index, :] = np.rot90(newf_data.copy(), 3)
        viz.image(bslice[:, :], win='1')
        viz.image(newbf_data[:, :], win='2')
        viz.image(newf_data[:, :], win='3')
        viz.image(np.rot90(touint8(tf.numpy()[:, index, :].copy())), win='4')
    tf_=ants.copy_image_info(b,tf_)
    f_ = ants.copy_image_info(b, f_)
    tf_.to_file(fluor_CONFIG['output_dir']+ '/reg2D/blikef_SyNtoB_iter1.nii.gz')
    f_.to_file(fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter1.nii.gz')


def fluor_SyNtoB_bySections():
    viz = Visdom(env='slice2D_fluo_SyNtoB')
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    tf = ants.image_read(fluor_CONFIG['output_dir']+ '/reg2D/blikef_SyNtoB_iter1.nii.gz')
    f = ants.image_read(fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter1.nii.gz')
    b_seg = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')
    tf_ = ants.new_image_like(tf, tf.numpy())
    f_ = ants.new_image_like(f, f.numpy())
    # tf_=ants.image_read(fluor_CONFIG['output_dir']+ '/reg2D/blikef_SyNtoB_iter2.nii.gz')
    # f_=ants.image_read(fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter2.nii.gz')
    for i in range(0, tf.shape[1]):
    # for i in range(17, 18):
        if set(np.unique(b_seg[:,i,:].numpy().astype(int)))=={0}:
            continue
        print(str(i) + ' ' + str(tf.shape[1]))
        b_slice_data = b.numpy()[:, i, :].copy()
        tf_slice_data=tf.numpy()[:, i, :]
        f_slice_data = f.numpy()[:, i, :]
        tf_slice_data = normalization(tf_slice_data) * 255
        tf_slice_data = tf_slice_data.astype(np.uint8)

        b_slice_data = normalization(b_slice_data) * 255
        b_slice_data = b_slice_data.astype(np.uint8)
        tf_slice = ants.from_numpy(tf_slice_data)
        b_slice = ants.from_numpy(b_slice_data)
        f_slice = ants.from_numpy(f_slice_data)


        result_ = ants.registration(b_slice, tf_slice, 'SyNOnly',syn_metric='mattes',reg_iterations=(400,400,400),flow_sigma=1,
                                   total_sigma=0.5,outprefix= fluor_CONFIG['output_dir']+ '/reg2D/xfms/FtoB_iter2_'+ str(i) + '_FtoB_iter3_')
        result = ants.registration(b_slice, result_['warpedmovout'], 'SyNOnly',syn_metric='CC',reg_iterations=(2400,1200,1200, 40),flow_sigma=1,syn_sampling=4,
                                   total_sigma=0.5,outprefix= fluor_CONFIG['output_dir']+ '/reg2D/xfms/FtoB_iter2_'+ str(i) + '_FtoB_iter2_')

        mask=b_slice_data.copy()
        mask[mask>0]=1
        tf_slice_ = ants.apply_transforms(b_slice, tf_slice, result_['fwdtransforms'])
        f_slice_ = ants.apply_transforms(b_slice, f_slice, result_['fwdtransforms'])
        tf_slice_ = ants.apply_transforms(b_slice, tf_slice_, result['fwdtransforms'])
        f_slice_ = ants.apply_transforms(b_slice, f_slice_, result['fwdtransforms'])

        tf_[:, i, :]=0
        tf_[:,i,:]=tf_slice_.numpy()
        f_[:, i, :]=0
        f_[:,i,:]=f_slice_.numpy()
        #################################################
        viz.image(b_slice.numpy()[:,:], win='1')
        viz.image(tf_slice.numpy()[:, :], win='2')
        viz.image(tf_slice_.numpy(), win='3')
        viz.image(f_slice_.numpy(), win='4')
        #################################################
    ants.image_write(tf_, fluor_CONFIG['output_dir']+ '/reg2D/blikef_SyNtoB_iter2.nii.gz')
    f_=ants.copy_image_info(b,f_)
    ants.image_write(f_, fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter2.nii.gz')

def fluor_SyNtoB_byVolume():
    tf=ants.image_read(fluor_CONFIG['output_dir'] + '/reg2D/blikef_SyNtoB_iter2.nii.gz')
    f = ants.image_read(fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter2.nii.gz')
    b = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    tf, f, b_ = reset_img([tf, f, b])
    t = ants.registration(b_, tf, 'SyNOnly',reg_iterations=(2100,1200,1200,20),syn_sampling=4,syn_metric='CC',flow_sigma=3,total_sigma=1,outprefix=fluor_CONFIG['output_dir']+ '/reg2D/xfms/FtoB_iter3_')
    tf_=ants.apply_transforms(b_, tf, t['fwdtransforms'],interpolator='bSpline')
    f_ = ants.apply_transforms(b_, f, t['fwdtransforms'], interpolator='bSpline')
    f_ = ants.copy_image_info(b, f_)
    tf_ = ants.copy_image_info(b, tf_)
    ants.image_write(tf_, fluor_CONFIG['output_dir']+ '/reg2D/blikef_SyNtoB_iter3.nii.gz')
    ants.image_write(f_, fluor_CONFIG['output_dir'] + '/reg2D/f_SyNtoB_iter3.nii.gz')


def atlas_regtoAffinAFSlices():
    global path
    b_alignf = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    for n in ['D99','SARM2','SARM6','CHARM1']:
        atlas = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/'+n+'_inOriginB.nii.gz')
        atlas_tmp = ants.new_image_like(b_alignf, b_alignf.numpy())
        atlas_tmp[:, :, :] = 0
        for i in range(0, b_alignf.shape[1]):
            print(n+': '+str(i))
            bslice_af_data = b_alignf[:, i, :].numpy()
            bslice_af_data = touint8(bslice_af_data)
            atlas_slice_data = atlas[:, i, :].numpy()
            atlas_slice = ants.from_numpy(atlas_slice_data)
            bslice_af = ants.from_numpy(bslice_af_data)
            atlas_slice_ = ants.apply_transforms(bslice_af, atlas_slice,
                                                 [fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter2_' + str(i) + '_FtoB_iter2_0GenericAffine.mat',fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter2_' + str(i) + '_FtoB_iter2_1InverseWarp.nii.gz'],
                                                 'multiLabel')
            atlas_tmp[:, i, :] = atlas_slice_[:, :]
        atlas_tmp.to_file(fluor_CONFIG['output_dir'] + '/reg2D/atlas/'+n+'_in_FaffinetoB.nii.gz')

def atlas_regtoRawFluoSlices():
    f = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/fluor.nii.gz')
    for n in ['D99','SARM2','SARM6','CHARM1']:
        atlas = ants.image_read(fluor_CONFIG['output_dir'] + '/reg2D/atlas/'+n+'_in_FaffinetoB.nii.gz')
        atlas_ = ants.new_image_like(atlas, atlas.numpy())
        for i in range(0, f.shape[1]):
            print(n+': '+str(i))
            fslice_data = f[:, i, :].numpy()
            atlas_data = atlas_[:, i, :].numpy().copy()
            fslice_data = np.rot90(fslice_data).copy()
            atlas_data = np.rot90(atlas_data).copy()
            atlas_data_tmp = np.zeros_like(atlas_data, np.uint8)
            if os.path.exists(fluor_CONFIG['output_dir'] + '/reg2D/xfms/masks/tfslice_mask' + str(i) + '.tif'):
                fslice_mask_data = cv2.imread(fluor_CONFIG['output_dir'] + '/reg2D/xfms/masks/tfslice_mask' + str(i) + '.tif', 0)
                for label in np.unique(fslice_mask_data):
                    fslice_data_tmp = fslice_mask_data.copy()
                    atlasslice_data_tmp = atlas_data.copy()
                    if label != 1 and label != 88:
                        fslice_data_tmp[fslice_mask_data != label] = 0
                        fslice = ants.from_numpy(fslice_data_tmp)
                        if os.path.exists(fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter1_' + str(i) + '_part' + str(label) + '_0GenericAffine.mat'):
                            # 将不同组织的mask 映射至对应blockface切片上
                            fslice_part = ants.apply_transforms(fslice, fslice,
                                                                [ fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter1_' + str(i) + '_part' + str(label) + '_0GenericAffine.mat'],
                                                                interpolator='multiLabel')
                            fslice_part = ants.apply_transforms(fslice, fslice_part,
                                                                [fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter1_' + str(i) + '_part' + str(label) + '_1Warp.nii.gz'],
                                                                interpolator='multiLabel')
                            fslice_part_data = fslice_part[:, :].numpy().copy()
                            # 取出mask对应的atlas
                            fslice_part_data[fslice_part_data > 0] = 1
                            atlasslice_step1_tmp_part = atlasslice_data_tmp * fslice_part_data[:, :]
                            atlasslice_step1_tmp_part_img = ants.from_numpy(atlasslice_step1_tmp_part)
                            # 将不同组织的atlas映射到荧光切片上
                            atlasslice_step1_tmp_part_ = ants.apply_transforms(fslice, atlasslice_step1_tmp_part_img, [
                                fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter1_' + str(i) + '_part' + str(label) + '_0GenericAffine.mat'], interpolator='multiLabel', whichtoinvert=[True])
                            atlasslice_step1_tmp_part_ = ants.apply_transforms(fslice, atlasslice_step1_tmp_part_, [
                                fluor_CONFIG['output_dir'] + '/reg2D/xfms/FtoB_iter1_' + str(i) + '_part' + str(label) + '_1InverseWarp.nii.gz'],interpolator='multiLabel')
                            tmp = atlasslice_step1_tmp_part_[:, :].numpy().copy()
                            atlas_data_tmp = atlas_data_tmp + tmp
                        else:
                            atlas_data_tmp = atlas_data.copy()
                            break
            else:
                print('step2 mask is non')
                atlas_data_tmp = atlas_data.copy()
            if os.path.exists(fluor_CONFIG['output_dir'] + '/reg2D/xfms/ftob_affine_iter1_' + str(i) + '_0GenericAffine.mat'):
                fslice=ants.from_numpy(fslice_data)
                tmp_img = ants.from_numpy(atlas_data_tmp)
                tmp_img_ = ants.apply_transforms(fslice, tmp_img,[fluor_CONFIG['output_dir'] + '/reg2D/xfms/ftob_affine_iter1_' + str(i) + '_0GenericAffine.mat'],
                                                 whichtoinvert=[True],interpolator='multiLabel')
                atlas_data_tmp = tmp_img_[:, :].numpy().copy()
            atlas_[:, i, :] = 0
            atlas_[:, i, :] = np.rot90(atlas_data_tmp.copy(),3)
        atlas_=ants.copy_image_info(f,atlas_)
        f_data=f.numpy()
        f_data[f_data>0]=1
        atlas_[:,:,:]=atlas_.numpy()*f_data
        atlas_.to_file(fluor_CONFIG['output_dir'] + '/fluor/atlas/'+n+'_inRawFluo.nii.gz')


