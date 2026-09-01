#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : Macaca-Star
@File    : fMOST_PI_preproc.py
@Desc    : Core preprocessing, artifact removal, intensity correction,
           and multi-modal registration routines for fMOST-PI optical imaging data.
"""
import os
import numpy as np
import utils.Logger as loggerz
import tifffile
import ants
import yaml
import albumentations as A
from utils.util import horizontal, sagittal, crop_brain, log, atlas_reg_ByT1w, atlas_reg_noT1w, reset_img
from utils.util_fluor import normalization
from memory_profiler import memory_usage

# 1. Load fMOST-PI pipeline configuration parameters
YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))

# 2. Load structural MRI preprocessing and registration configuration parameters
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))

def tif_to_nii():
    """
    Convert raw multi-slice TIFF image stack into a standardized NIfTI (.nii.gz) volume.

    Workflow:
      1. Load raw TIFF image stack from the configured subject path.
      2. Load reference NMT template to retrieve standard spatial metadata (origin and direction).
      3. Flip image axes (axis 2 and axis 1) to match standard anatomical orientation.
      4. Convert NumPy array to ANTsImage, assign physical voxel spacing, and synchronize
         origin/direction matrix with the NMT reference template.
      5. Save the converted NIfTI volume (PI___.nii.gz).
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('tif to nii.gz')

    # 2. Load raw TIFF stack and reference NMT template
    tif_file = tifffile.imread(fMOST_PI_CONFIG['subject_dir'])
    # tif_file = ants.image_read(fMOST_PI_CONFIG['subject_dir']).numpy()
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')

    # 3. Flip axes to align coordinate space
    tmp = tif_file
    tmp=np.flip(tmp,2)
    tmp = np.flip(tmp, 1)

    # 4. Convert NumPy array to an ANTsImage object
    tif = ants.from_numpy(tmp)

    # Set physical voxel resolution (spacing) from configuration
    tif.set_spacing(fMOST_PI_CONFIG['spacing'])

    # Synchronize coordinate origin and direction with reference NMT template
    tif.set_origin(nmt.origin)
    tif.set_direction(nmt.direction)

    # 5. Export converted volume to NIfTI format
    tif.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI.nii.gz')


def normalize_to_8bit():
    logger = loggerz.get_logger()
    logger.info('normalize to 8bit')
    img = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI.nii.gz')
    img_data = img.numpy()
    # Exclude abnormal intensities
    percent_upper = np.percentile(img_data, 95)
    img_ = img_data[img_data < percent_upper]
    if np.max(img_) > 255.0:
        space = np.max(img_) - np.min(img_)
        norm = (img - np.min(img.numpy()[:, :, :].all())) * 255 / (space + 1E-6)
    else:
        # If the maximum value is already less than 255, there is no need to normalize to 8-bit
        print('max < 255')
        norm = img
    ants.image_write(norm, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')

def mas_cerebellum():
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('Remove cerebellum')

    # Load 8-bit PI volume and resample to 0.25 mm isotropic resolution for efficient registration
    fix = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
    fix_ = ants.resample_image(fix, (0.25, 0.25, 0.25), interp_type=4)
    fix_.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_0.25mm.nii.gz')

    # Load standard NMT template brain, cerebellum mask, and tissue segmentation
    move = ants.image_read(os.getcwd() + '/template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read(os.getcwd() + '/template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas2 = ants.image_read(os.getcwd() + '/template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')

    # Crop template and atlas masks to target hemisphere (Left / Right) or whole brain
    move=crop_brain(move)
    atlas = crop_brain(atlas)
    atlas2 = crop_brain(atlas2)

    # Perform SyN deformable registration from NMT template (moving) to downsampled PI (fixed)
    pi_affine_transform = ants.registration(fix_, move, type_of_transform='SyN', reg_iterations=(40, 20, 0))

    # Apply forward transforms to map cerebellum mask and segmentation into downsampled PI space
    atlas_ = ants.apply_transforms(fix_, atlas, pi_affine_transform['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(fix_, atlas2, pi_affine_transform['fwdtransforms'], 'multiLabel')

    # Dilate cerebellum binary mask (radius=3 ball kernel) to ensure complete coverage of boundary voxels
    atlas_ = ants.morphology(atlas_, operation='dilate', radius=3, mtype='binary', shape='ball')
    ants.image_write(atlas_,fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas2_, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/Segmentation_in_PI_0.25mm.nii.gz')

    # Compute spatial mapping between downsampled PI (0.25 mm) and original-resolution PI
    pi_affine_transform = ants.registration(fix, fix_, type_of_transform='Affine', reg_iterations=(40, 20, 0))

    # Map dilated cerebellum mask and segmentation back to original-resolution PI space
    atlas_h = ants.apply_transforms(fix, atlas_, pi_affine_transform['fwdtransforms'], 'multiLabel')
    atlas2_h = ants.apply_transforms(fix, atlas2_, pi_affine_transform['fwdtransforms'], 'multiLabel')

    # Mask out cerebellum tissue from the original-resolution PI volume
    mas = ants.mask_image(fix, atlas_h, 0)

    # Save cerebellum-removed volume and high-resolution masks
    ants.image_write(mas, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')
    ants.image_write(atlas_h, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask.nii.gz')
    ants.image_write(atlas2_h, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/Segmentation_mask.nii.gz')


def denoise_img():
    logger = loggerz.get_logger()
    logger.info('denoise the fMOST PI')
    pi = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm.nii.gz')
    mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/PI_mask_0.05mm.nii.gz')
    mask_data=mask.numpy()
    mask_data[mask_data>0]=1
    mask[:,:,:]=mask_data
    pi=ants.mask_image(pi, mask)
    pi=ants.iMath_truncate_intensity(pi,0.01,upper_q=0.95)
    # Denoise the fMOST PI (using a simple Gaussian filter) 0.5
    pi_denoise = ants.smooth_image(pi, sigma=0.1, max_kernel_width=0.5)
    # pi_denoise=ants.denoise_image(pi,p=2, r=3)
    pi_denoise.to_filename(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn.nii.gz')


def remove_artifact():
    """
    Remove optical imaging stripe artifacts (destriping) across orthogonal planes.

    Workflow:
      1. Check if the cerebellum-removed intermediate volume (PI_rmc.nii.gz) exists;
         if not, load the 8-bit PI volume and mask out the cerebellum region.
      2. Execute horizontal destriping to remove horizontal stripe artifacts.
      3. Execute sagittal destriping to eliminate residual sagittal stripe artifacts.
    """
    logger = loggerz.get_logger()
    logger.info('remove striping artifact')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz'):
        img=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
        mask=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask.nii.gz')

        # Subtract cerebellum tissue to avoid interference during stripe artifact removal
        img=img-ants.mask_image(img, mask)
        img.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')

    # 3. Execute destriping procedures along horizontal and sagittal planes
    horizontal()
    sagittal()

def intensity_c():
    """
    Perform intensity correction and illumination normalization on the PI volume.

    Workflow:
      1. Load denoised PI volume and normalize intensities to [0, 255].
      2. If intensity_correction is False:
         - Apply a custom logarithmic dynamic range compression curve on high-intensity
           voxels (threshold n=60) while preserving lower-intensity signals.
      3. If intensity_correction is True:
         - Load and binarize the foreground brain mask.
         - Perform a 3-stage multi-resolution N4 bias field correction (shrink factors: 8 -> 4 -> 2).
      4. Save the intensity-corrected volume.
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('Intensity correction')

    # Load denoised PI volume and normalize intensity range to [0, 255]
    img = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn.nii.gz')
    img=ants.iMath_normalize(img)*255
    img_data=img.numpy()

    # =========================================================================
    # Branch A: Logarithmic dynamic range compression
    # =========================================================================
    if not fMOST_PI_CONFIG['intensity_correction']:
        n = 60
        tmp = img.numpy() + 2.0
        matrix = np.where(tmp < n)

        # Apply logarithmic transformation slice-by-slice along z-axis
        for i in range(0, img.shape[2]):
            tmp_ = tmp[:, :, i]
            tmp[:, :, i] = log(n, tmp_) * n

        # Preserve original intensities for lower-intensity voxels (< n)
        tmp[matrix] = img_data[matrix]

        # Convert back to ANTsImage and restore original spatial metadata
        tmp_morm = ants.from_numpy(tmp)
        tmp_morm.set_spacing(img.spacing)
        tmp_morm.set_direction(img.direction)
        tmp_morm.set_origin(img.origin)

        # Save the log-compressed output volume
        ants.image_write(tmp_morm, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')

    # =========================================================================
    # Branch B: Hierarchical multi-scale N4 bias field correction
    # =========================================================================
    else:
        print('read mask')
        # Load and binarize foreground brain mask
        mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/PI_mask_0.05mm.nii.gz')
        mask = ants.copy_image_info(img, mask)
        mask_data = mask.numpy()
        mask_data[mask_data > 0] = 1
        mask[:, :, :] = mask_data

        # Stage 1: Coarse-scale N4 bias field correction (shrink factor = 8)
        fm1 = ants.n4_bias_field_correction(img, mask=mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                            shrink_factor=8,
                                            return_bias_field=False)

        # Stage 2: Medium-scale N4 bias field correction (shrink factor = 4)
        fm2 = ants.n4_bias_field_correction(fm1, mask=mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                            shrink_factor=4,
                                            return_bias_field=False)

        # Stage 3: Fine-scale N4 bias field correction (shrink factor = 2)
        fm3 = ants.n4_bias_field_correction(fm2, mask=mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                            shrink_factor=2,
                                            return_bias_field=False)

        # Save the N4 bias-corrected output volume
        ants.image_write(fm3, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')

def clahe_image():
    """
    Perform local contrast enhancement on the PI fluorescence volume using
    slice-wise Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Workflow:
      1. Load intensity-corrected PI volume and foreground brain mask, then apply masking.
      2. If CLAHE is enabled in configuration:
         a. Normalize intensity values to 8-bit dynamic range [0, 255] uint8.
         b. Apply 2D CLAHE slice-by-slice along axis 1 (tile grid size: 10x10, clip limit: 1.0).
         c. Wrap the processed array back into an ANTsImage maintaining original spatial metadata.
      3. Save the enhanced volume (PI_8bit_rm_dn_ic_r.nii.gz).
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('Image Enhancement')

    # Load preprocessed PI volume and corresponding foreground brain mask
    pi_origin = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')
    mask=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/PI_mask_0.05mm.nii.gz')

    # Mask out non-brain background
    pi_origin=ants.mask_image(pi_origin, mask)

    # Apply slice-wise CLAHE if enabled in config
    if fMOST_PI_CONFIG['clahe']:
        logger.info('START PI clahe')
        # Initialize 2D CLAHE operator from Albumentations
        CLAHE = A.CLAHE(tile_grid_size=(10,10), always_apply=True)
        pi = pi_origin.numpy()

        # Min-max normalize intensities to [0, 255] uint8 for histogram equalization
        space = np.max(pi) - np.min(pi)
        norm = (pi - np.min(pi)) * 255 / space
        pi = norm.astype(np.uint8)

        # Apply CLAHE slice-by-slice along axis 1
        for i in range(0, pi.shape[1]):
            pi_splice = pi[:, i, :]
            pi_splice = CLAHE.apply(pi_splice,clip_limit= 1.0) # 2
            pi[:, i, :] = pi_splice

        # Convert NumPy array back to ANTsImage with original spatial metadata
        pi_=ants.new_image_like(pi_origin,pi)
    else:
        pi_ = pi_origin

    # 4. Save the contrast-enhanced output volume
    ants.image_write(pi_, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic_r.nii.gz')


def PI_alignNMT():
    """
    Perform initial spatial alignment (Similarity registration) from fMOST-PI volume
    to the standard NMT template space.

    Workflow:
      1. Load preprocessed fMOST PI volume and its high-resolution mask.
      2. Downsample PI volume and mask to isotropic resolution (0.2 mm) for efficient alignment.
      3. Crop NMT template to target hemisphere/whole brain and standardize origins.
      4. Perform Similarity registration (rigid + isotropic scaling) to align PI to NMT.
      5. Apply forward transforms to both PI image and mask.
      6. Truncate intensity outliers (1% - 99%), apply Gaussian smoothing, and perform
         N4 bias field correction to generate normalized aligned outputs.
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('fMOST PI align to NMT')

    # 2. Load preprocessed fMOST PI volume and corresponding brain mask
    img=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic_r.nii.gz')
    mask=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/PI_mask_0.05mm.nii.gz')

    # 3. Resample image to isotropic resolution (0.2mm) and resample mask to match target
    img_=ants.resample_image(img,(0.2,0.2,0.2),use_voxels=False,interp_type=4)
    mask_=ants.resample_image_to_target(mask,img_,'multiLabel')

    # Save downsampled intermediate volumes
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_0.2mm.nii.gz')
    mask_.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_mask_0.2mm.nii.gz')

    # 4. Load standard NMT template, crop to target region, and standardize coordinate origins
    nmt=ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    nmt=crop_brain(nmt)
    nmt_,img_,mask_=reset_img([nmt,img_,mask_])

    # 5. Perform Similarity registration (Rigid + Isotropic scaling) from PI to NMT space
    t = ants.registration(
        fixed=nmt_,
        moving=img_,
        type_of_transform='Similarity',
        aff_metric='GC',
        outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_'
    )

    # 6. Apply forward transformation to PI image and brain mask
    img_=ants.apply_transforms(nmt_,img_,t['fwdtransforms'],'bSpline')
    mask_ = ants.apply_transforms(nmt_, mask_, t['fwdtransforms'], 'multiLabel')

    # Synchronize image header geometry with standard NMT space
    img_=ants.copy_image_info(nmt,img_)
    mask_=ants.copy_image_info(nmt, mask_)
    mask_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')

    # 7. Truncate intensity outliers (1% to 99% quantiles) and save primary aligned image
    img_ = ants.iMath_truncate_intensity(img_, 0.01, 0.99)
    img_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')

    # 8. Post-processing: Smooth and apply N4 bias field correction within mask
    img_=ants.smooth_image(img_,0.4)
    img_=ants.n4_bias_field_correction(img_,mask_,shrink_factor=2)
    img_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')

def correct_T1like():
    """
    Mask the synthesized T1-like PI volume to remove non-brain background noise.

    Loads the aligned PI image and synthesized T1-like volume, generates or loads
    a foreground brain mask, and applies it to output a background-cleaned
    synthetic T1-like image (T1likePI_c.nii.gz).
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('correct T1like')

    # 2. Load aligned PI volume and synthesized T1-like volume
    img=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')
    t1like = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI.nii.gz')

    # 3. Check if the brain foreground mask exists; if not, generate and save it
    if not os.path.exists(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz'):
        # Generate binary brain mask from the aligned PI volume
        mask=ants.get_mask(img,1)
        mask.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')
    else:
        # Load existing brain mask
        mask=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')

    # 4. Apply mask to the synthesized T1-like volume to remove non-brain background
    t1like_ = ants.mask_image(t1like, mask)

    # 5. Save the background-cleaned synthetic T1-like volume
    t1like_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_c.nii.gz')


def fMOST_PI_3Dreg():
    """
    Perform 3D anatomical registration of fMOST-PI to the NMT atlas space.

    Dynamically executes either MRI-guided registration (via subject-specific T1w MRI)
    or MRI-free registration (direct template alignment) based on configuration,
    while monitoring runtime memory consumption.
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('fMOST PI register to NMT')

    if MRI_CONFIG['MRI-guided']:
        # 2. Branch: MRI-guided registration pipeline
        logger.warning('MRI-guided registration')

        # Profile memory usage during T1w-guided atlas registration (sample every 5.0s)
        memory_usage_data=memory_usage(atlas_reg_ByT1w,interval=5.0)

        # Calculate and log average memory consumption
        average_memory_usage = sum(memory_usage_data) / len(memory_usage_data)
        print(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning('MRI-guided registration end')

    else:
        # 3. Branch: MRI-free registration pipeline (direct alignment without subject MRI)
        logger.warning('no MRI-guided registration')

        # Profile memory usage during direct (MRI-free) atlas registration
        memory_usage_data=memory_usage(atlas_reg_noT1w,interval=5.0)

        # Calculate and log average memory consumption
        average_memory_usage = sum(memory_usage_data) / len(memory_usage_data)
        print(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning('MRI-guided registration end')

def repair_atlas():
    logger = loggerz.get_logger()
    logger.info('repair atlas')
    method=''
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')
    tipi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePIw.nii.gz')
    tmp=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')
    atlas=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/Charm6_inPI.nii.gz')
    atlas2 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/SARM6_inPI.nii.gz')
    atlas,atlas2,tmp,pi_=reset_img([atlas,atlas2,tmp,pi])
    mask=ants.get_mask(tipi)
    atlas_mask = atlas.clone()
    img_data = atlas_mask.numpy()+atlas2.numpy()
    img_data[img_data > 0] = 100
    atlas_mask[:, :, :] = img_data
    tf2 = ants.registration(
        pi_,atlas_mask, 'SyNOnly',
        syn_metric='mattes',
        reg_iterations=(400, 200, 100),
        flow_sigma=3,
        total_sigma=0.1,
        outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_repair_')
    atlas_ = ants.apply_transforms(pi_, atlas, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(pi_, atlas2, tf2['fwdtransforms'], 'multiLabel')
    tmp_ = ants.apply_transforms(pi_, tmp, tf2['fwdtransforms'], 'bSpline')
    if fMOST_PI_CONFIG['atlas_repair_bySeg']:
        logger.info('atlas_repair_bySeg')
        gm=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/seg/gm_mask.nii.gz')
        gm,=reset_img([gm])
        atlas__ = atlas_.clone()
        img_data = atlas__.numpy()
        img_data[img_data > 0] = 100
        atlas__[:, :, :] = img_data
        pi_affine_transform1 = ants.registration(gm, atlas__, 'SyNOnly', syn_metric="mattes", reg_iterations=(400, 200,100),
                                                 flow_sigma=3, total_sigma=0,
                                                 outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/xfms/gm_constrain_')
        atlas_ = ants.apply_transforms(gm, atlas_, pi_affine_transform1['fwdtransforms'], 'multiLabel')
        atlas_ = ants.mask_image(atlas_, gm)
    atlas_=ants.copy_image_info(pi,atlas_)
    mask = ants.copy_image_info(pi, mask)
    atlas_=ants.mask_image(atlas_,mask)
    tmp_ = ants.copy_image_info(pi, tmp_)
    if not fMOST_PI_CONFIG['atlas_repair_bySeg']:
        atlas_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/Charm6_inPI_repair.nii.gz')
        tmp_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI_repair.nii.gz')
    else:
        atlas_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/Charm6_inPI_repair_byGM.nii.gz')
        tmp_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI_repair_byGM.nii.gz')


def seg_byt1pi():
    method = ''
    # 1. Load MRI configuration
    MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
    MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))

    # 2. Check if segmentation-based atlas refinement is enabled
    if fMOST_PI_CONFIG['atlas_repair_bySeg']:
        path = fMOST_PI_CONFIG['output_dir']

        # Ensure segmentation output directories exist
        if not os.path.exists(path+'/reg/seg/'):
            os.mkdir(path+'/reg/seg/')
        if not os.path.exists(path + '/reg/seg/tmp/'):
            os.mkdir(path + '/reg/seg/tmp/')

        # 3. Load and preprocess standard subcortical template mask
        subcortex = ants.image_read('template/NMT/NMT_brain/subcortex.nii.gz')
        subcortex_, = reset_img([subcortex])

        # Crop to target hemisphere (Left / Right) or whole brain
        subcortex_=crop_brain(subcortex_)

        # =====================================================================
        # 4. Warp subcortical template into native PI space
        # =====================================================================
        if MRI_CONFIG['MRI-guided']:
            print('MRI-guided')
            # Chain transformations: NMT -> In vivo T1w MRI -> Intermediate Space -> Native PI
            subcortex_ = ants.apply_transforms(
                subcortex_, subcortex_,
                [path + '/reg/'+method+'/xfms/atlas_NMTtoT1w_1Warp.nii.gz',
                path + '/reg/'+method+'/xfms/atlas_NMTtoT1w_0GenericAffine.mat'],
                'multiLabel', whichtoinvert=[False, False])
            subcortex_ = ants.apply_transforms(
                subcortex_, subcortex_,
                [path + '/reg/'+method+'/xfms/atlas_T1toGFP_1Warp.nii.gz',
                path + '/reg/'+method+'/xfms/atlas_T1toGFP_0GenericAffine.mat'],
                'multiLabel', whichtoinvert=[False, False])
            subcortex_ = ants.apply_transforms(
                subcortex_, subcortex_,
                [path + '/reg/'+method+'/xfms/atlas_PItoT1w_0GenericAffine.mat',
                path + '/reg/'+method+'/xfms/atlas_PItoT1w_1InverseWarp.nii.gz'],
                'multiLabel',
                whichtoinvert=[True, False])
            subcortex_ = ants.copy_image_info(subcortex, subcortex_)
        else:
            print('no MRI')
            # Direct transform: NMT -> Native PI
            subcortex_ = ants.apply_transforms(
                subcortex_, subcortex_,
                [path + '/reg/'+method+'/xfms/atlas_PItoNMT_1Warp.nii.gz',
                            path + '/reg/'+method+'/xfms/atlas_PItoNMT_0GenericAffine.mat'],
                'multiLabel',
                whichtoinvert=[False, False])

        # Save warped subcortex mask in PI space
        subcortex_.to_file(path + '/reg/atlas/subcortex_inPI.nii.gz')
        img_tmp=ants.image_read(path+'/reg/T1likePIw.nii.gz')
        img = ants.image_read(path+'/reg/PI_alignNMT.nii.gz')
        subcortex = ants.image_read(path+'/reg/atlas/subcortex_inPI.nii.gz')
        cortex = ants.image_read(path + '/reg/'+method+'/atlas/CHARM4_inPI.nii.gz')
        cortex_data=cortex.numpy()
        cortex_data[cortex_data>0]=1
        cortex[:,:,:]=cortex_data
        cortex=ants.copy_image_info(subcortex, cortex)

        subcortex=subcortex-ants.mask_image(subcortex, cortex, level=[1])
        subcortex=ants.copy_image_info(img_tmp, subcortex)
        img_subcortex = ants.mask_image(img, subcortex, level=[1])
        img = img - img_subcortex
        subcortex=ants.copy_image_info(img_tmp,subcortex)
        subcortex = ants.mask_image(img_tmp, subcortex, level=[1])
        img_tmp = img_tmp - subcortex
        mask = ants.get_mask(img, cleanup=2)
        mask = ants.copy_image_info(img_tmp, mask)
        img_tmp = ants.mask_image(img_tmp, mask)
        seg = ants.kmeans_segmentation(img_tmp, 2, kmask=mask, mrf=0.1)
        ants.image_write(seg['segmentation'], path + '/reg/seg/tmp/seg0.nii.gz')
        ants.image_write(seg['probabilityimages'][0], path + '/reg/seg/tmp/p0_iter0.nii.gz')
        ants.image_write(seg['probabilityimages'][1], path + '/reg/seg/tmp/p1_iter0.nii.gz')
        img=ants.copy_image_info(img_tmp,img)
        priorseg = ants.prior_based_segmentation(img, seg['probabilityimages'], mask, 0.15,mrf=0.2)
        ants.image_write(priorseg['segmentation'], path + '/reg/seg/tmp/seg1.nii.gz')
        ants.image_write(priorseg['probabilityimages'][0], path + '/reg/seg/tmp/p1_iter1.nii.gz')
        ants.image_write(priorseg['probabilityimages'][1], path + '/reg/seg/tmp/p2_iter1.nii.gz')
        gm_mask = ants.mask_image(priorseg['segmentation'], priorseg['segmentation'], level=[1])
        ants.image_write(gm_mask, path + '/reg/seg/gm_mask.nii.gz')

def upsample_toOrigin():
    """
    Upsample registered anatomical atlases (CHARM, SARM) and synthetic T1-like volumes
    back to the original high-resolution space.

    Workflow:
      1. Load original resolution PI , downsampled PI, synthetic T1, and registered atlases.
      2. Invert initial affine alignment transforms to map images back to intermediate space (0.2 mm).
      3. Export intermediate 0.2 mm atlases and synthetic T1 volumes.
      4. Resample the atlases to the original high-resolution target (0.05 mm) using multiLabel interpolation.
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('upsample to original resolution')
    method=''

    # 2. Load original high-resolution PI, aligned PI, synthetic T1, and registered atlases
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/fMOST_PI/PI_8bit_rm_dn_ic_r.nii.gz')
    pi_ = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/PI_alignNMT_.nii.gz')
    t1pi=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI_c.nii.gz')
    atlas = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/CHARM6_inPI.nii.gz')
    atlas3 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/SARM6_inPI.nii.gz')
    atlas3=ants.copy_image_info(pi_,atlas3)

    # Synchronize header geometry and load intermediate resolution target (0.2 mm)
    target=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_0.1mm.nii.gz')

    # Standardize image spatial origins
    pi_reset,target_rest,atlas_reset,atlas3_reset,t1pi=reset_img([pi_,target, atlas,atlas3,t1pi])

    # 3. Check or compute Affine transformation if missing
    if not os.path.exists(fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_0GenericAffine.mat'):
        print('not xfms.mat')
        t=ants.registration(pi_reset,target_rest,'Affine',outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_')

    # =========================================================================
    # Step A: Invert affine transform to map images to intermediate 0.2 mm space
    # =========================================================================
    atlas_reset = ants.apply_transforms(
        target_rest, atlas_reset,
        [fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_0GenericAffine.mat'],
        whichtoinvert=[True],
        interpolator='multiLabel')
    atlas3_reset = ants.apply_transforms(
        target_rest, atlas3_reset,
        [fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_0GenericAffine.mat'],
        whichtoinvert=[True],
        interpolator='multiLabel')
    t1pi = ants.apply_transforms(
        target_rest, t1pi,
        [fMOST_PI_CONFIG['output_dir'] + '/reg/xfms/PItoNMT_0GenericAffine.mat'],
        whichtoinvert=[True],
        interpolator='bSpline')

    # Synchronize header information with target geometry
    atlas_=ants.copy_image_info(target,atlas_reset)
    atlas3_ = ants.copy_image_info(target, atlas3_reset)
    t1pi = ants.copy_image_info(target, t1pi)

    # Save intermediate 0.2 mm resolution results
    atlas_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/CHARM6_0.2mm.nii.gz')
    atlas3_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/SARM6_0.2mm.nii.gz')
    t1pi.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/T1PI_0.2mm.nii.gz')

    # =========================================================================
    # Step B: Resample atlases to original high-resolution target
    # =========================================================================
    atlas_, atlas3_,pi_50,t1pi=reset_img([atlas_,atlas3_, pi,t1pi])

    # Explicitly set voxel spacing
    atlas_.set_spacing((0.2,0.2,0.2))
    atlas3_.set_spacing((0.2, 0.2, 0.2))
    t1pi.set_spacing((0.2, 0.2, 0.2))
    pi_50.set_spacing((0.05, 0.05, 0.05))

    # Resample CHARM and SARM atlas to original resolution and save
    atlas__ = ants.resample_image_to_target(atlas_, pi_50, interp_type='multiLabel')
    atlas__=ants.copy_image_info(pi,atlas__)
    atlas__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/Charm6_0.05mm.nii.gz')
    atlas3__ = ants.resample_image_to_target(atlas3_, pi_50, interp_type='multiLabel')
    atlas3__ = ants.copy_image_info(pi, atlas3__)
    atlas3__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/SARM6_0.05mm.nii.gz')

