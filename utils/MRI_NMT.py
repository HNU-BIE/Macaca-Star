#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import utils.Logger as loggerz
import yaml

from utils.MRI_brain_preproc import brain_orientation_correction, make_brain_mask, MRI_preprocess, \
    rm_neck, extract_brainByMask, crop_brain, set_output_dir

# ==================== Load MRI Configuration ====================
# Load configuration parameters for MRI preprocessing and registration
mri_config_path = os.path.join(os.getcwd(), 'config', 'MRI_config.yaml')
with open(mri_config_path, 'r', encoding='utf-8') as f:
    MRI_CONFIG = yaml.safe_load(f)


def MRI_preproc(type):
    """
    Execute the in vivo structural MRI preprocessing pipeline.

    Processes the raw structural MRI volume through orientation correction,
    neck removal, skull-stripping (brain extraction), intensity correction,
    and region cropping to serve as an anatomical prior for multimodal registration.

    :param type: Pipeline type/mode index used to configure output directory hierarchy.
    """
    # 1. Initialize pipeline logger
    logger=loggerz.get_logger()

    # 2. Check if MRI-guided pipeline is enabled
    if MRI_CONFIG['MRI-guided']:
        logger.warning('MRI-guided registration')

        # Check if the input MRI file exists on disk
        if os.path.exists(MRI_CONFIG['MRI_file']):
            # Step 1: Initialize and set up output directory structure for current mode
            set_output_dir(type)

            # Step 2: Correct spatial orientation and optical center alignment
            brain_orientation_correction()

            # Step 3: Remove neck and non-brain spinal tissues
            rm_neck()

            # Step 4: Generate whole-brain binary mask (brain extraction/skull-stripping)
            make_brain_mask()

            # Step 5: Extract brain tissue by applying the brain mask
            extract_brainByMask()

            # Step 6: Perform N4 bias field correction and Rician denoising
            MRI_preprocess()

            # Step 7: Crop to target hemisphere (Left / Right) or whole brain, and standardize spatial origins
            crop_brain()
        else:
            logger.error('No MRI image; Next step to run PI <--> NMT')
        logger.info('END MRI<-->NMT')
    else:
        logger.warning('no MRI-guided registration')
