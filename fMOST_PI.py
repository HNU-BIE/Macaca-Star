#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Macaca-Star Pipeline Entry Script.

This script performs initial environment setup, validates the project's directory
structure, initializes logging, and executes the preprocessing/registration pipeline
for Macaca brain imaging data (PI / MRI).

Output Directory Structure:
---------------------------
fMOST_PI/
    Contains the processed fMOST-PI data and intermediate results generated
    during the fMOST-PI processing workflow.

MRI/
    Contains the processed MRI data and intermediate results generated
    during MRI preprocessing.

reg/
    Contains the data required for anatomical registration and the resulting
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
from utils.check_file_structure import check_file_structure


def main():
    """Main execution function for the Macaca-Star pipeline."""
    # 1. Initialize and retrieve the logger instance
    logger=loggerz.get_logger()
    logger.info('Macaca-Star pipeline start')

    # 2. Run MRI preprocessing  (optional / commented out)
    # MRI_preproc(0)

    # 3. Run Propidium Iodide (PI) fluorescence preprocessing and registration
    PI_preproc()


if __name__ == "__main__":
    # Step 1: Check and validate the required directory hierarchy and data files
    check_file_structure(1)

    # Step 2: Launch the main processing pipeline
    main()