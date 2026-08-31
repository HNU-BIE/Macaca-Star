#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Macaca-Star Pipeline Entry Script (Blockface & Fluorescence Workflow).

This script validates the required directory structure for blockface and fluorescence
imaging data, initializes logging, and executes blockface reconstruction,
fluorescence section processing, and multimodal registration pipelines.

Output Directory Structure:
---------------------------
blockface/
    Contains the processed block-face data and the results of 3D reconstruction.

fluor/
    Contains the processed 2D fluorescence sections and the results of cross-modal translation.

MRI/
    Contains the processed MRI data and intermediate results generated
    during MRI preprocessing.

reg2D/
    Contains the reconstructed fluorescence section data resulting from 2D registration.

reg3D/
    Contains the data required for 3D anatomical registration and the resulting
    registration outputs.
    ├── atlas/ : Contains various registration results and outputs generated
    │            by the registration workflow.
    └── xfms/  : Contains the transformation matrices and warp field files
                 generated during registration.
"""
import os
import warnings

# ==================== Environment & Warning Suppression ====================
# 1. Disable Albumentations automatic update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# 2. Suppress PyTorch/Mamba FutureWarning and general warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== Project Utilities & Pipelines ====================
import utils.Logger as loggerz
from utils.MRI_NMT import MRI_preproc
from utils.PI_MRI import PI_preproc
from utils.blockface_MRI import blockface_preproc
from utils.check_file_structure import check_file_structure
from utils.fluor_blockface import fluor_preproc


def main():
    """
    Main execution function for blockface and fluorescence imaging pipelines.
    Modules can be executed selectively based on project requirements.
    """
    # 1. Initialize and retrieve the pipeline logger instance
    logger=loggerz.get_logger()
    logger.info('Macaca-Star pipeline start')

    # 2. Run MRI preprocessing and template registration
    # MRI_preproc(1)

    # 3. Run blockface volume reconstruction and alignment to MRI/template
    blockface_preproc()

    # 4. Run fluorescence serial sections preprocessing and 2D-to-3D registration
    fluor_preproc()


if __name__ == "__main__":
    # Validate input directories and files for blockface/fluorescence workflow (mode=2)
    check_file_structure(2)

    # Launch the main processing pipeline
    main()