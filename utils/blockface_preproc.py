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
from utils.util_fluor import atlas_reg_ByT1w, atlas_reg_noT1w, get_maskBywatershed, centerxy_img, \
    translate_bycenter, get_bmask, reassign_anomalous_pixels
from memory_profiler import memory_usage
import time

# ==================== Load YAML Configuration Files ====================
# 1. Load block-face volume reconstruction and alignment configuration
blockface_YAML_PATH = os.path.join(os.getcwd(), 'config', 'blockface_config.yaml')
with open(blockface_YAML_PATH, 'r', encoding='utf-8') as f:
    blockface_CONFIG = yaml.safe_load(f)

# 2. Load 2D fluorescence section processing and cross-modal translation configuration
fluor_YAML_PATH = os.path.join(os.getcwd(), 'config', 'fluor_sections_config.yaml')
with open(fluor_YAML_PATH, 'r', encoding='utf-8') as f:
    fluor_CONFIG = yaml.safe_load(f)

# 3. Load reference structural MRI preprocessing and registration configuration
MRI_YAML_PATH = os.path.join(os.getcwd(), 'config', 'MRI_config.yaml')
with open(MRI_YAML_PATH, 'r', encoding='utf-8') as f:
    MRI_CONFIG = yaml.safe_load(f)

def recon_blockface():
    """
    Reconstruct a 3D block-face NIfTI volume from sequential 2D serial slice images.

    Workflow:
      1. Locate the 2D block-face image directory (2Dblockface/).
      2. Discover and sort all qualified slice PNG files numerically based on section index numbers.
      3. Allocate an empty 3D NumPy array [H, num_slices, W] for volume assembly.
      4. Read each 2D slice as a grayscale image and stack sequentially along axis 1.
      5. Convert the stacked 3D NumPy array into an ANTsImage object and export as a NIfTI volume (b.nii.gz).
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('3D recon blockface')

    #  Locate 2D slice directory and sort files by numerical section index
    imgs_path = Path(blockface_CONFIG['subject_dir']+'/2Dblockface/')
    files = sorted(
        imgs_path.glob('Section*_qualified_b.png'),
        key=lambda p: int(re.search(r'Section(\d+)', p.stem).group(1))
    )

    # Initialize 3D volumetric array: [Height=500, Slices=len(files), Width=500]
    b_data=np.zeros((500,len(files),500))
    # Convert Path objects to full string path representations
    files = [str(f) for f in files]

    # Load 2D grayscale slices and stack them sequentially along axis 1
    for i,img_path in enumerate(files):
        slice_tmp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        b_data[:, i, :] = slice_tmp.copy()

    # Convert NumPy array to ANTsImage and save as NIfTI volume
    b=ants.from_numpy(b_data)
    b.to_file(blockface_CONFIG['subject_dir']+'/b.nii.gz')

def align_Bcenter():
    """
    Detect and correct slice-to-slice cutting shifts (jitter) in serial block-face imaging.

    Workflow:
      1. Compute Mean Squared Error (MSE) between consecutive adjacent 2D slices along axis 1.
      2. Identify abrupt spatial shift boundaries where MSE exceeds a predefined threshold (t = 900).
      3. Iterate backwards across detected shift points:
         a. Compute centers of mass for adjacent reference and moving slices.
         b. Pre-align slice centroids via translational offset (translate_bycenter).
         c. Refine spatial alignment using Affine registration with Global Correlation metric.
         d. Propagate cumulative transformations backward to all preceding slices to maintain global continuity.
      4. Clamp negative interpolation artifacts and export the realigned block-face volume (b_recon.nii.gz).
    """
    # Load raw reconstructed block-face volume
    img=ants.image_read(blockface_CONFIG['subject_dir']+'/b.nii.gz')
    img_data=img.numpy()[:,:,:].copy()

    # =========================================================================
    # Compute adjacent slice dissimilarity profile (MSE)
    # =========================================================================
    tmp=[]
    for i in range(img_data.shape[1]-1):
        slice1=img_data[:,i,:]
        slice2 = img_data[:, i+1, :]
        mse=mean_squared_error(slice1, slice2)
        tmp.append(mse)

    # Plot slice-to-slice MSE profile for visual inspection
    plt.plot(tmp)
    plt.show()

    # =========================================================================
    # Detect significant cutting shift indices
    # =========================================================================
    tmp=np.array(tmp)
    t=900
    indexs=np.where(tmp>t)[0]
    indexs=np.insert(indexs, 0, 0)

    # =========================================================================
    # Backward iterative centroid translation and Affine alignment
    # =========================================================================
    for i in range(len(indexs)-1,-1,-1):
        mov_data=img_data[:, indexs[i], :].copy()
        mov=ants.from_numpy(mov_data)
        fix_data=img_data[:, indexs[i] + 1, :].copy()
        fix = ants.from_numpy(fix_data)

        # Calculate center-of-mass translational offsets (tx, ty)
        basex, basey = ants.get_center_of_mass(fix)
        movx, movy = ants.get_center_of_mass(mov)
        tx = basex - movx
        ty = basey - movy

        # Apply centroid translation to the moving slice
        mov[:, :] = translate_bycenter(mov[:, :].numpy(), tx, ty)
        # Compute fine-scale Affine registration transform
        t = ants.registration(fix, mov,type_of_transform='Affine', aff_metric='GC', aff_sampling=32)

        # Propagate translation and affine transforms backward to all preceding slices
        for n in range(indexs[i],-1, -1):
            mov = ants.from_numpy(img_data[:, n, :])
            mov[:, :] = translate_bycenter(mov[:, :].numpy(), tx, ty)
            img_data[:, n, :]=ants.apply_transforms(fix,mov,t['fwdtransforms'],'bSpline')[:,:].numpy().copy()

    # =========================================================================
    # Clean interpolation artifacts and save realigned volume
    # =========================================================================
    img_data[img_data<0]=0
    img[:,:,:]=img_data
    img.to_file(blockface_CONFIG['subject_dir']+'/b_recon.nii.gz')

def oc_blockface_toNMT():
    """
    Perform orientation correction and spatial reference alignment on the reconstructed
    block-face volume to match standard NMT template coordinate space.
    """
    # Load reconstructed block-face volume and reference NMT template
    img=ants.image_read(blockface_CONFIG['subject_dir']+'/b_recon.nii.gz')
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')

    # Extract voxel array and permute axes to match anatomical plane order
    img_data=img.numpy()
    tmp = np.transpose(img_data, [2, 0, 1])

    # Flip axes along dimensions 1, 2, and 0 to standardize anatomical directions
    tmp=np.flip(tmp,1)
    tmp = np.flip(tmp, 2)
    tmp = np.flip(tmp, 0)

    # Convert re-oriented NumPy array to an ANTsImage object
    img=ants.from_numpy(tmp)

    # Synchronize spatial origin and direction cosines with the NMT reference template
    img.set_origin(nmt.origin)
    img.set_direction(nmt.direction)

    # Save the orientation-corrected block-face volume
    img.to_file(blockface_CONFIG['subject_dir']+'/b_recon_oc.nii.gz')


def intensity_c():
    """
    Perform intensity enhancement, denoising, and bias field correction on the block-face volume.
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('Intensity correction')

    # Initialize 2D CLAHE operator from Albumentations
    CLAHE = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(10, 10), always_apply=True)

    # Load orientation-corrected block-face volume
    img=ants.image_read(blockface_CONFIG['subject_dir']+'/b_recon_oc.nii.gz')

    # Apply 2D CLAHE slice-by-slice along axis 1 for local contrast enhancement
    for i in range(img.shape[1]):
        fslice=img[:,i,:].numpy()
        fslice=touint8(fslice)
        fslice=CLAHE.apply(fslice,clip_limit=2)
        img[:, i, :]=fslice

    # enerate binary foreground mask
    mask=ants.get_mask(img)

    # Apply volumetric spatial denoising within the mask
    img=ants.denoise_image(img,mask)

    # Perform N4 bias field correction to eliminate illumination inhomogeneities
    img=ants.n4_bias_field_correction(img,mask,shrink_factor=2)
    img=ants.mask_image(img,mask)

    # Save the intensity-corrected and contrast-enhanced block-face volume
    img.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_ic.nii.gz')

def b_alignMRI():
    """
    Perform 3D Affine spatial alignment from the preprocessed block-face volume
    to the reference NMT template (or MRI) space.
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('blockface align to MRI')

    # Load preprocessed block-face volume and reference NMT template
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_ic.nii.gz')
    b=ants.from_numpy(b.numpy())
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    nmt_=ants.from_numpy(nmt.numpy())

    # Perform 3D Affine registration from block-face (moving) to NMT space (fixed)
    t2 = ants.registration(nmt_, b, 'Affine', aff_metric='GC',aff_sampling=32)

    # Apply forward transformation using b-spline interpolation
    b_ = ants.apply_transforms(nmt_, b, t2['fwdtransforms'],'bSpline')

    # Save the affine transformation matrix for downstream inverse mapping
    shutil.copyfile(t2['fwdtransforms'][0], fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat')

    # Generate foreground mask and remove background voxels
    mask=ants.get_mask(b_,10)
    b_=ants.mask_image(b_,mask)

    # Apply two-stage N4 bias field correction to eliminate residual illumination gradients
    b_=ants.n4_bias_field_correction(b_,mask,shrink_factor=4)
    b_ = ants.n4_bias_field_correction(b_, mask, shrink_factor=2)

    # Synchronize spatial header metadata with the reference NMT template
    b_=ants.copy_image_info(nmt,b_)

    # Export the affine-aligned block-face volume
    b_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/b_recon_oc_scale_alignMRI.nii.gz')

def correct_t1like():
    """
    Mask the synthesized T1-like block-face volume to remove non-brain background noise.
    """
    tsfer=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeB.nii.gz')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    mask=ants.get_mask(b,10)
    img = ants.mask_image(tsfer, mask)
    img.to_file(fluor_CONFIG['output_dir']+'/reg3D/T1likeB_c.nii.gz')
    mask.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/B_mask.nii.gz')

def blockface_3Dreg():
    """
    Execute 3D anatomical registration of the block-face volume to the NMT atlas space.

    Dynamically selects between:
      1. MRI-guided registration (via subject-specific T1w MRI) with memory and runtime profiling.
      2. MRI-free registration (direct template alignment) when subject MRI is unavailable.
    """
    # 1. Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('fMOST PI register to NMT')

    # =========================================================================
    # Branch A: MRI-guided registration pipeline
    # =========================================================================
    if MRI_CONFIG['MRI-guided']:
        logger.warning('MRI-guided registration')
        start_time = time.time()

        # Profile memory consumption during T1w-guided registration (sample interval: 5.0s)
        memory_usage_data = memory_usage(atlas_reg_ByT1w, interval=5.0)
        end_time = time.time()

        # Compute and log average memory consumption
        average_memory_usage = sum(memory_usage_data) / len(memory_usage_data)
        print(f"average_memory : {average_memory_usage:.2f}MB")
        logger.warning(f"average_memory : {average_memory_usage:.2f}MB")

        # Compute and log total execution time
        total_time = end_time - start_time
        print(f"total time：{total_time:.2f}s")
        logger.warning(f"total time : {total_time:.2f}s")
        logger.warning('MRI-guided registration end')

    # =========================================================================
    # Branch B: MRI-free registration pipeline (direct template alignment)
    # =========================================================================
    else:
        logger.warning('no MRI-guided registration')

        # Execute direct registration without in vivo MRI prior
        atlas_reg_noT1w()
        logger.warning('no MRI-guided registration end')


def b_invetalignMRI():
    """
    Invert spatial transformations to map registered NMT templates, anatomical atlases,
    and synthetic volumes back into the native block-face coordinate space (OriginB).

    Workflow:
      1. Load native block-face origin volume (b_recon_oc_scale_clahe.nii.gz) and aligned volume.
      2. Load all registered atlases, masks, and intensity volumes in block-face aligned space.
      3. Determine inverse transformation:
         - If isAffine=True: Compute fresh Affine registration from aligned to native block-face space.
         - If isAffine=False: Invert the precomputed affine matrix (b_regt1.mat).
      4. Loop through all atlases, masks, and intensity images, apply inverse transforms,
         synchronize image headers with native geometry, and export to blockface/atlas/.
    """
    # Initialize pipeline logger
    logger = loggerz.get_logger()
    logger.info('blockface inverse align to MRI')
    method=''
    isAffine=True

    # Ensure output directory exists
    atlas_save_dir = os.path.join(fluor_CONFIG['output_dir'], 'blockface', 'atlas')
    os.makedirs(atlas_save_dir, exist_ok=True)

    # Load native block-face origin volume and aligned block-face volume
    b_origin = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/b_recon_oc_scale_clahe.nii.gz')
    b=ants.from_numpy(b_origin.numpy())
    b_alignMRI = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    b_ = ants.from_numpy(b_alignMRI.numpy())

    # Load registered atlases and masks in block-face aligned space
    atlas=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/CHARM6_inblockface.nii.gz')
    atlas1 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/segmentation_inblockface.nii.gz')
    atlas1 = ants.from_numpy(atlas1.numpy())
    atlas2 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/SARM6_inblockface.nii.gz')
    atlas4 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/cerebellum_mask_inblockface.nii.gz')
    atlas4 = ants.from_numpy(atlas4.numpy())
    atlas7 = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/'+method+'/atlas/segmentation_edit_inblockface.nii.gz')
    atlas7 = ants.from_numpy(atlas7.numpy())

    # Load registered template and synthetic T1 volume
    nmt = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/'+method+'/atlas/NMT_inblockface.nii.gz')
    blikef = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')

    # =========================================================================
    # Step 6: Determine Transformation Fields (Compute Affine or Invert Matrix)
    # =========================================================================
    if isAffine:
        # Register aligned volume back to native volume using Affine
        t = ants.registration(fixed=b, moving=b_, type_of_transform='Affine')
        transform_list = t['fwdtransforms']
        invert_flags = [False]
    else:
        # Invert the precomputed affine transformation matrix
        transform_list = [fluor_CONFIG['output_dir'] + '/reg3D/' + method + '/xfms/b_regt1.mat']
        invert_flags = [True]

    # =========================================================================
    # Step 7: Loop over Atlases, Masks, and Images (Apply Inverse Transforms)
    # =========================================================================
    # Mapping: {filename_prefix: (image_object, interpolator_type)}
    items_to_transform = {
        f'CHARM6_inOriginB{method}': (atlas, 'multiLabel'),
        f'segmentation_inOriginB{method}': (atlas1, 'multiLabel'),
        f'SARM6_inOriginB{method}': (atlas2, 'multiLabel'),
        f'cerebellum_mask_inOriginB{method}': (atlas4, 'multiLabel'),
        f'TMP_inOriginB{method}': (nmt, 'bSpline'),
        f'segmentation_edit_inOriginB{method}': (atlas7, 'multiLabel'),
        f'T1wlikeB_inOriginB{method}': (blikef, 'bSpline'),
    }

    for filename, (cur_img, interp) in items_to_transform.items():
        # Apply inverse transform to map into native block-face space
        img_out = ants.apply_transforms(
            fixed=b,
            moving=cur_img,
            transformlist=transform_list,
            interpolator=interp,
            whichtoinvert=invert_flags
        )
        # Synchronize spatial header information with native block-face geometry
        img_out = ants.copy_image_info(b_origin, ants.image_clone(img_out))
        img_out.to_file(os.path.join(atlas_save_dir, f"{filename}.nii.gz"))


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
    method = ''
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
    method=''
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
                    h_idx, w_idx = np.where(mask > 0)
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
                    h_idx, w_idx = np.where(mask > 0)
                    centroid = ( w_idx.mean(),h_idx.mean())
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
                    h_idx, w_idx = np.where(mask > 0)
                    centroid = ( w_idx.mean(),h_idx.mean())
                    centroids.append(centroid)
                min_h, min_w = min(centroids)
                for idx, mask in enumerate(masks, 2):
                    if mask[int(min_w),int(min_h)]>0:
                        seg_slice[mask > 0] = 88

        seg_data[:, i, :]=seg_slice
    seg[:,:,:]=seg_data
    seg.to_file(path+'/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')
