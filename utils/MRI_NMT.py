import os
import utils.Logger as loggerz
import yaml

from utils.MRI_brain_preproc import brain_orientation_correction, make_brain_mask, MRI_preprocess, \
    rm_neck, extract_brainByMask, crop_brain, set_output_dir

MRI_CONFIG = yaml.safe_load(open(os.getcwd() + '/config/MRI_config.yaml', 'r'))


def MRI_preproc(type):
    logger=loggerz.get_logger()
    if MRI_CONFIG['MRI-guided']:
        logger.warning('MRI-guided registration')
        if os.path.exists(MRI_CONFIG['MRI_file']):
            set_output_dir(type)
            brain_orientation_correction()
            rm_neck()
            make_brain_mask()
            extract_brainByMask()
            MRI_preprocess()
            crop_brain()
        else:
            logger.error('No MRI image; Next step to run PI <--> NMT')
        logger.info('END MRI<-->NMT')
    else:
        logger.warning('no MRI-guided registration')
