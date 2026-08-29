import os
import numpy as np
from utils.CycWaveM2D.models import create_model
from utils.CycWaveM2D.data.base_dataset import get_transform
import ants
import yaml
from PIL import Image
import utils.Logger as loggerz
from utils.CycWaveM2D.utils.config_loader import load_config

fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))


def fluor_toB_cyclegan():
    logger=loggerz.get_logger()
    logger.info('fluor to Blockface by 2d cyclegan')
    fluor=ants.image_read(fluor_CONFIG['subject_dir'])
    fluor_data=fluor.numpy()
    blikef=ants.image_clone(fluor)
    blikef[:,:,:]=0
    opt = load_config("config/CycWave-Mamba3D_config.yaml", mode="test")
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.checkpoints_dir = os.getcwd() + '/checkpoints'
    opt.name = 'fluor2Blockface'
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1
    opt.no_dropout = True # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    transform=get_transform(opt, grayscale=True)
    for i in range(0,fluor.shape[1]):
        img=Image.fromarray(fluor_data[:,i,:]).convert('RGB')
        img_tensor=transform(img)
        model.set_input({'A': img_tensor, 'A_paths': []})  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()
        img_data=visuals['fake'][0].cpu().float().numpy()*255
        blikef[:,i,:]=img_data
        logger.info('section :'+str(i))
    blikef.to_file(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')






