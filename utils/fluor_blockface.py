from utils.CycleGan_2D.test import fluor_toB_cyclegan
from utils.fluor_preproc import repaire_blikefluo, fluor_SyNtoB_bySeg, fluor_SyNtoB_bySections, \
    atlas_regtoAffinAFSlices, atlas_regtoRawFluoSlices, fluor_SyNtoB_byVolume, oc_fluor_toNMT


def fluor_preproc():
    # fluor_toB_cyclegan()
    # oc_fluor_toNMT()
    # repaire_blikefluo()
    # fluor_SyNtoB_bySeg()
    fluor_SyNtoB_bySections()
    ## fluor_SyNtoB_byVolume()
    # atlas_regtoAffinAFSlices()
    # atlas_regtoRawFluoSlices()