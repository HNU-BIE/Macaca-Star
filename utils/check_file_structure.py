#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import utils.Logger as loggerz
from datetime import datetime

def check_file_structure(type):
    formatted_time = datetime.now().strftime("%y%m%d_%H%M%S")
    loggerz.init_logger('Macaca-Star'+str(formatted_time))
    logger = loggerz.get_logger()
    logger.info('check file structure')
    error = False
    if type==1:
        fMOST_YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
        config = yaml.safe_load(open(fMOST_YAML_PATH, 'r'))
        # if not os.path.exists(config['subject_dir']):
        #     logger.error('please create subject_dir or set correct subject_dir')
        #     error = True
        if not os.path.exists(config['output_dir']):
            os.mkdir(config['output_dir'])
        if not os.path.exists(config['output_dir']+'/MRI'):
            os.mkdir(config['output_dir']+'/MRI')
        if not os.path.exists(config['output_dir']+'/fMOST_PI'):
            os.mkdir(config['output_dir']+'/fMOST_PI')
        if not os.path.exists(config['output_dir']+'/fMOST_PI/tmp'):
            os.mkdir(config['output_dir']+'/fMOST_PI/tmp')
        if not os.path.exists(config['output_dir']+'/fMOST_PI/atlas'):
            os.mkdir(config['output_dir']+'/fMOST_PI/atlas')
        if not os.path.exists(config['output_dir']+'/reg/'):
            os.mkdir(config['output_dir']+'/reg/')
        if not os.path.exists(config['output_dir']+'/reg/xfms'):
            os.mkdir(config['output_dir']+'/reg/xfms')
        if not os.path.exists(config['output_dir'] + '/reg/atlas'):
            os.mkdir(config['output_dir'] + '/reg/atlas')
        if error:
            sys.exit(1)
    elif type==2:
        fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
        fluor_config = yaml.safe_load(open(fluor_YAML_PATH, 'r'))
        if not os.path.exists(fluor_config['subject_dir']):
            logger.error('please create subject_dir or set correct subject_dir')
            error = True
        if not os.path.exists(fluor_config['output_dir']):
            os.mkdir(fluor_config['output_dir'])
        if not os.path.exists(fluor_config['output_dir']+'/MRI'):
            os.mkdir(fluor_config['output_dir']+'/MRI')
        if not os.path.exists(fluor_config['output_dir']+'/blockface'):
            os.mkdir(fluor_config['output_dir']+'/blockface')
        if not os.path.exists(fluor_config['output_dir']+'/reg3D'):
            os.mkdir(fluor_config['output_dir']+'/reg3D')
        if not os.path.exists(fluor_config['output_dir']+'/reg3D/xfms'):
            os.mkdir(fluor_config['output_dir']+'/reg3D/xfms')
        if not os.path.exists(fluor_config['output_dir']+'/reg3D/atlas'):
            os.mkdir(fluor_config['output_dir']+'/reg3D/atlas')
        if not os.path.exists(fluor_config['output_dir']+'/blockface/atlas'):
            os.mkdir(fluor_config['output_dir']+'/blockface/atlas')
        if not os.path.exists(fluor_config['output_dir']+'/fluor'):
            os.mkdir(fluor_config['output_dir']+'/fluor')
        if not os.path.exists(fluor_config['output_dir']+'/fluor/atlas'):
            os.mkdir(fluor_config['output_dir']+'/fluor/atlas')
        if not os.path.exists(fluor_config['output_dir']+'/reg2D'):
            os.mkdir(fluor_config['output_dir']+'/reg2D')
        if not os.path.exists(fluor_config['output_dir']+'/reg2D/atlas'):
            os.mkdir(fluor_config['output_dir']+'/reg2D/atlas')
        if not os.path.exists(fluor_config['output_dir']+'/reg2D/xfms'):
            os.mkdir(fluor_config['output_dir']+'/reg2D/xfms')
        if not os.path.exists(fluor_config['output_dir']+'/reg2D/xfms/masks'):
            os.mkdir(fluor_config['output_dir']+'/reg2D/xfms/masks')
        if error:
            sys.exit(1)
    elif type==3:
        fMOST_GFP_YAML_PATH = os.getcwd() + '/config/fMOST_GFP_config.yaml'
        config = yaml.safe_load(open(fMOST_GFP_YAML_PATH, 'r'))
        if not os.path.exists(config['subject_dir']):
            logger.error('please create subject_dir or set correct subject_dir')
            error = True
        if not os.path.exists(config['output_dir']):
            os.mkdir(config['output_dir'])
        if not os.path.exists(config['output_dir']+'/MRI'):
            os.mkdir(config['output_dir']+'/MRI')
        if not os.path.exists(config['output_dir']+'/fMOST_GFP'):
            os.mkdir(config['output_dir']+'/fMOST_GFP')
        if not os.path.exists(config['output_dir']+'/fMOST_GFP/tmp'):
            os.mkdir(config['output_dir']+'/fMOST_GFP/tmp')
        if not os.path.exists(config['output_dir']+'/fMOST_GFP/atlas'):
            os.mkdir(config['output_dir']+'/fMOST_GFP/atlas')
        if not os.path.exists(config['output_dir']+'/reg/'):
            os.mkdir(config['output_dir']+'/reg/')
        if not os.path.exists(config['output_dir']+'/reg/xfms'):
            os.mkdir(config['output_dir']+'/reg/xfms')
        if not os.path.exists(config['output_dir'] + '/reg/atlas'):
            os.mkdir(config['output_dir'] + '/reg/atlas')
        if error:
            sys.exit(1)