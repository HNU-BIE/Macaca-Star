import logging
import os
import colorlog
import yaml

LOGGER = None
YAML_PATH = 'config/fMOST_PI_config.yaml'
config = yaml.safe_load(open(YAML_PATH, 'r'))

def init_logger(name='macaque_pipeline'):
    global LOGGER
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )

    file_handler = logging.FileHandler(config['output_dir']+'/'+name+'.log', mode='w', encoding='utf8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    LOGGER = logger
    LOGGER.info('init logger')


def get_logger():
    global LOGGER
    return LOGGER
