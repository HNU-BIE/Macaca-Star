import utils.Logger as loggerz
from utils.CycleGan_3D.test import b_to_T1_cyclegan
from utils.blockface_preproc import intensity_c, b_alignMRI, correct_t1like, blockface_3Dreg, b_invetalignMRI, \
    repair_blockface, repair_seg_inBlockface, recon_blockface, align_Bcenter, oc_blockface_toNMT, seg_byt1pi, \
    repair_atlas


def blockface_preproc():
    # align_Bcenter()
    # oc_blockface_toNMT()
    # intensity_c()
    # b_alignMRI()
    b_to_T1_cyclegan()
    # correct_t1like()
    blockface_3Dreg()
    # b_invetalignMRI()
    # repair_blockface()
    # seg_byt1pi()
    # repair_atlas()
    # repair_seg_inBlockface()