#  Macaca-Star: A Deep Learning-Assisted Pipeline for mesoscale connectivity mapping in the macaque brain

# Contents

- [Overview](#overview)
- [Macaca-Star Pipeline](#macaca-star-pipeline)
- [Cross-modal Translation with CycWave-Mamba](#cross-modal-translation-with-cycwave-mamba)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)


## ✨ Overview

**Macaca-Star** is an open-source, automated, and modular pipeline for multimodal whole-brain mapping in macaques. It integrates large-scale optical imaging data with standard macaque brain atlases in a unified stereotaxic space, with optional guidance from subject-specific *in vivo* MRI.

The pipeline supports diverse imaging data, which are organized into two complementary categories:

* **Anatomical Imaging** — used to establish and refine the spatial correspondence between datasets:

  * **3D optical volumes:** high-resolution volumetric datasets with anatomical contrast, such as fMOST-PI and serial block-face imaging.
  * **2D serial sections:** section-based histological datasets, such as serial fluorescence sections.
  * **Structural MRI:** optional subject-specific *in vivo* MRI data for anatomical guidance.

* **Biological Feature Imaging** — used for spatial mapping and downstream analysis:

  * **Axonal projections:** continuous 3D fluorescence volumes, such as fMOST-GFP axon-tracing data.
  * **Soma distributions:** microscopic cellular markers and spatial distributions of labeled neurons.

By integrating modality-specific preprocessing, automated 2D-to-3D reconstruction, deep learning–based cross-modal translation, and nonlinear registration, Macaca-Star establishes a unified framework for mapping heterogeneous optical imaging datasets into a common macaque brain space. This design facilitates the integration of high-resolution anatomical and biological information while accommodating tissue distortion, substantial cross-modal appearance differences, and inter-individual anatomical variability.



# ⭐ Macaca-Star Pipeline

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

# 🚀 Getting Started

Example scripts and test data are available in the [`example`](https://github.com/HNU-BIE/Macaca-Star/tree/main/example) folder.

The main processing stages include:

1. Optical image preprocessing and reconstruction
2. 2D or 3D cross-modal translation using CycWave-Mamba
3. MRI-guided or MRI-free registration
4. Mapping of extracted biological features to the NMT space
5. Atlas-based analysis


# 📜 License

This work is licensed under a Creative Commons Attribution 4.0 International License.
