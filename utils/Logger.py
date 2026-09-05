import logging
import colorlog
import yaml

# ==================== Global Logger & Configuration ====================
LOGGER = None
YAML_PATH = 'config/fMOST_PI_config.yaml'
config = yaml.safe_load(open(YAML_PATH, 'r'))

def init_logger(name='Macaca-Star'):
    """
    Initialize and configure the global pipeline logger with file and colored console handlers.
    :param name: Logger and log file name identifier
    """
    global LOGGER

    # Define console color mapping for different log levels
    log_colors_config = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    # Configure formatters for file and console outputs
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )

    # Set up FileHandler for persistent log file writing
    file_handler = logging.FileHandler(config['output_dir']+'/'+name+'.log', mode='w', encoding='utf8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Set up StreamHandler for real-time colored console logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(console_formatter)
    # Register handlers with logger instance
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Assign to global instance and log initialization confirmation
    LOGGER = logger
    LOGGER.info('init logger')


def get_logger():
    """
    Retrieve the globally initialized logger instance.

    :return: Configured logging.Logger instance.
    """
    global LOGGER
    return LOGGER
