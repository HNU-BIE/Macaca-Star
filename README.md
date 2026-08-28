# Macaca-Star: A Deep Learning-Assisted Pipeline for mesoscale connectivity mapping in the macaque brain

# Contents

- [Overview](#overview)
- [Macaca-Star Pipeline](#macaca-star-pipeline)
- [Cross-modal Translation with CycWave-Mamba](#cross-modal-translation-with-cycwave-mamba)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)


# Overview

**Macaca-Star** is an open-source, automated, and modular framework for multimodal whole-brain mapping in macaques. It is designed to integrate large-scale optical imaging data with subject-specific in vivo MRI and standard macaque brain atlases within a common anatomical space.

Macaca-Star supports multiple imaging streams, including **fMOST PI/GFP data**, **serial fluorescence sections**, **block-face images**, and **MRI data**. The framework combines modality-specific image preprocessing, 2D-to-3D reconstruction, deep learning-based cross-modal translation, and nonlinear registration to address substantial tissue deformation, cross-modal appearance discrepancies, and inter-individual anatomical variability.

A key component of Macaca-Star is **CycWave-Mamba**, a deep learning model for cross-modal image translation in both 2D and 3D. In the section-based pipeline, 2D fluorescence images are translated into block-face-like representations to facilitate correspondence with their associated block-face images and subsequent 3D reconstruction. In the volumetric pipeline, reconstructed block-face and fMOST PI volumes are translated into synthetic T1-weighted MRI-like volumes for registration with MRI-based reference spaces.

Macaca-Star supports both direct registration to the **NIMH Macaque Template (NMT)** and registration guided by **subject-specific in vivo MRI**. The resulting spatial transformations can be applied to map axonal projections, soma locations, and other extracted biological features into the standard atlas space for cross-subject integration and atlas-based analysis.

The [`example`](https://github.com/HNU-BIE/Macaca-Star/tree/main/example) folder contains example data for testing the pipeline.


# Macaca-Star Pipeline

Macaca-Star provides an integrated processing and mapping framework for different types of macaque brain imaging data.

### 1. Optical Image Processing

Acquired optical images undergo modality-specific preprocessing, including intensity correction (with artifact removal), tissue segmentation, and 3D reconstruction. A SAM2-based method is used for block-face tissue segmentation. The resulting **block-face** and **fMOST PI** volumes support cross-modal translation, while **fMOST GFP** and **2D fluorescence sections** support axon tracing and soma localization, respectively.

### 2. Cross-modal translation

Macaca-Star uses **CycWave-Mamba** to reduce the appearance gap between optical imaging and anatomical reference images.

Two cross-modal translation settings are supported:

- **2D translation:** fluorescence sections → block-face-like images
- **3D translation:** reconstructed block-face or fMOST PI volumes → synthetic T1w MRI-like volumes

### 3. Spatial registration

Synthetic T1w MRI-like volumes can be registered to the standard NMT space through two pathways:

- **MRI-guided registration:** subject-specific in vivo MRI serves as a global anatomical reference between ex vivo optical data and the standard atlas.
- **MRI-free registration:** synthetic T1w MRI-like volumes are registered directly to the NMT template when subject-specific in vivo MRI is unavailable.

### 4. Mapping and atlas-based analysis

The resulting spatial transformations can be applied to map extracted biological information, including:

- Axonal projections
- Soma locations
- Other extracted biological features

into the standard NMT space, enabling multimodal integration, cross-subject comparison, and atlas-based analysis.


# Cross-modal Translation with CycWave-Mamba

**CycWave-Mamba** is the cross-modal translation module incorporated into Macaca-Star. It supports both 2D and 3D image translation for different imaging scenarios.

For **2D fluorescence sections**, CycWave-Mamba generates block-face-like images that facilitate registration with corresponding block-face images and support subsequent 3D reconstruction.

For **3D optical data**, CycWave-Mamba translates reconstructed fMOST PI and block-face volumes into synthetic T1w MRI-like volumes. These synthetic volumes provide an MRI-compatible anatomical representation for subsequent registration to subject-specific in vivo MRI or directly to the NMT template.

Pretrained model checkpoints are provided in the [`checkpoints`](https://github.com/HNU-BIE/Macaca-Star/tree/main/checkpoints) folder and can be used directly for inference without additional retraining.


# System Requirements

Macaca-Star has been installed and tested on:

- Windows
- Ubuntu 20.04
- Ubuntu 22.04

A minimum of **32 GB RAM** is recommended for general processing. Memory requirements may increase substantially when processing large whole-brain optical datasets.

GPU acceleration and a compatible CUDA environment are required for deep learning-based cross-modal translation.

For inference, pretrained CycWave-Mamba checkpoints are provided in the [`checkpoints`](https://github.com/HNU-BIE/Macaca-Star/tree/main/checkpoints) folder.

For model training, our development environment used two NVIDIA RTX A6000 GPUs (48 GB each).


# Installation
```
git clone https://github.com/HNU-BIE/Macaca-Star.git

pip install -r requirements.txt
or
conda env create -f enviroment.yml
``` 

# Getting Started

Example scripts and test data are available in the [`example`](https://github.com/HNU-BIE/Macaca-Star/tree/main/example) folder.

The main processing stages include:

1. Optical image preprocessing and reconstruction
2. 2D or 3D cross-modal translation using CycWave-Mamba
3. MRI-guided or MRI-free registration
4. Mapping of extracted biological features to the NMT space
5. Atlas-based analysis


# License

This work is licensed under a Creative Commons Attribution 4.0 International License.
