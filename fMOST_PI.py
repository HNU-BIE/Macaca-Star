import utils.Logger as loggerz
from utils.MRI_NMT import MRI_preproc
from utils.PI_MRI import PI_preproc
from utils.check_file_structure import check_file_structure



def main():
    logger=loggerz.get_logger()
    logger.info('Macaca-Star pipeline start')
    # MRI_preproc(0)
    PI_preproc()



if __name__ == "__main__":
    check_file_structure(1)
    main()