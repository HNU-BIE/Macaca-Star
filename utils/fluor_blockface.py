from utils.CycWaveM2D.test import fluor_toB_cyclegan
from utils.fluor_preproc import repaire_blikefluo, fluor_SyNtoB_bySeg, fluor_SyNtoB_bySections, oc_fluor_toNMT


def fluor_preproc():
    """
    Modular and customizable pipeline for serial 2D fluorescence section preprocessing,
    cross-modal synthesis (Fluorescence -> Pseudo-Blockface), and multi-stage registration.

    Workflow:
      1. Synthesize pseudo-blockface images from 2D fluorescence sections using 2D CycWave-Mamba / CycleGAN.
      2. Perform orientation correction and align reconstructed fluorescence data to NMT coordinate space.
      3. Repair and refine synthesized pseudo-blockface volumes/slices.
      4. Execute segmentation-guided non-linear SyN registration between fluorescence and block-face.
      5. Perform section-wise (slice-by-slice) 2D SyN deformable registration to map fluorescence into block-face space.
    """
    fluor_toB_cyclegan()               # Step 1: Synthesize pseudo-blockface images from 2D fluorescence sections using 2D CycWaveM2D
    oc_fluor_toNMT()                   # Step 2: Correct spatial orientation and align fluorescence volume to standard NMT coordinate space
    repaire_blikefluo()                # Step 3: Post-process and repair artifactual regions in the synthesized blockface-like fluorescence volume
    fluor_SyNtoB_bySeg()               # Step 4: Perform segmentation-guided SyN registration between fluorescence and block-face slices
    fluor_SyNtoB_bySections()          # Step 5: Perform section-by-section 2D non-linear deformable SyN registration to target block-face slices