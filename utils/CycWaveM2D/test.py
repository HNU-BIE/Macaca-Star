import os
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
    """
    Perform slice-by-slice 2D cross-modal synthesis from fluorescence sections to pseudo-blockface
    images using the trained 2D CycWave-Mamba / CycleGAN model.

    Workflow:
      1. Load input fluorescence volume (subject_dir) and create an empty output container with identical geometry.
      2. Load 2D model test options (CycWave-Mamba2D_config.yaml) and configure test parameters.
      3. Instantiate and initialize the generator network (fluor2Blockface).
      4. Retrieve standard dataset preprocessing transforms (e.g., resizing, normalization).
      5. Iterate slice-by-slice along axis 1:
         a. Convert 2D NumPy slice to PIL RGB Image and transform into a tensor.
         b. Feed tensor into the generator and run forward inference (model.test()).
         c. Extract synthetic output ('fake'), rescale intensities to [0, 255], and store in the output volume.
      6. Export the assembled synthetic pseudo-blockface volume (blike_f.nii.gz).
    """
    # 1. Initialize pipeline logger
    logger=loggerz.get_logger()
    logger.info('fluor to Blockface by 2d cyclegan')

    # 2. Load input fluorescence volume and allocate output container with identical spatial metadata
    fluor=ants.image_read(fluor_CONFIG['subject_dir'])
    fluor_data=fluor.numpy()
    blikef=ants.image_clone(fluor)
    blikef[:,:,:]=0

    # 3. Load model options and set test runtime configurations
    opt = load_config("config/CycWave-Mamba2D_config.yaml", mode="test")
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.checkpoints_dir = os.getcwd() + '/checkpoints'
    opt.name = 'fluor2Blockface'
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1
    opt.no_dropout = True # no visdom display; the test code saves the results to a HTML file.

    # 4. Instantiate and initialize generator model
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    transform=get_transform(opt, grayscale=True)

    # =========================================================================
    # Slice-by-slice 2D Cross-Modal Inference Loop
    # =========================================================================
    for i in range(0,fluor.shape[1]):
        # Convert 2D NumPy array slice to PIL RGB image
        img=Image.fromarray(fluor_data[:,i,:]).convert('RGB')
        img_tensor=transform(img)

        # Feed tensor into model and run forward inference
        model.set_input({'A': img_tensor, 'A_paths': []})  # unpack data from data loader
        model.test()

        # Assign synthesized slice back to 3D volume container
        visuals = model.get_current_visuals()
        img_data=visuals['fake'][0].cpu().float().numpy()*255

        # Assign synthesized slice back to 3D volume container
        blikef[:,i,:]=img_data
        logger.info('section :'+str(i))
    blikef.to_file(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')
