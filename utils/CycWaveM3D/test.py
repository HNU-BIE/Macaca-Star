import yaml
from utils.CycWaveM3D.utils.NiftiDataset import *
import utils.CycWaveM3D.utils.NiftiDataset as NiftiDataset_testing
from utils.CycWaveM3D.models import create_model
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
from utils.CycWaveM3D.utils.config_loader import load_config

def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        image_batches.append(image_batch)

    return image_batches


# inference single image
def inference(model, image_path, result_path, resample, resolution, patch_size_x,
              patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size=1):
    # create transformations to image and labels
    transforms1 = [
        NiftiDataset_testing.Resample(resolution, resample)
    ]

    transforms2 = [
        NiftiDataset_testing.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    # normalize the image
    image = Normalization(image)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())

    sample = {'image': image, 'label': label_tfm}

    for transform in transforms1:
        sample = transform(sample)

    # keeping track on how much padding will be performed before the inference
    image_array = sitk.GetArrayFromImage(sample['image'])
    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_y - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']

    for transform in transforms2:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)

    label_np = np.asarray(label_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # ----------------- Padding the image if the z dimension still is not even ----------------------

    if (image_np.shape[2] % 2) == 0:
        Padding = False
    else:
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Padding = True

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)

    for i in tqdm(range(len(batches))):
        batch = batches[i]
        batch = np.transpose(batch, (0, 2, 1, 3))
        batch = (batch - 127.5) / 127.5

        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        batch = Variable(batch.cuda())

        # pred = model(batch)
        model.set_input(batch)
        model.test()
        pred = model.get_current_visuals()
        pred = pred['fake_B']
        pred = pred.squeeze().data.cpu().numpy()
        if pred.ndim == 3:
            pred = np.transpose(pred, (1, 0, 2))
        elif pred.ndim == 4:
            pred = np.transpose(pred, (0, 2, 1, 3))
        pred = (pred * 127.5) + 127.5

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))

    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    # removed the 1 pad on z
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2] - 1)]

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y, :pad_z]

    # convert back to sitk space
    label = from_numpy_to_itk(label_np, image_pre_pad)
    # ---------------------------------------------------------------------------------------------

    # save label
    writer = sitk.ImageFileWriter()

    if resample is True:
        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        # label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='bspline')   # keep this commented

        label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkLinear)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())
    else:
        label = label

    writer.SetFileName(result_path)
    writer.Execute(label)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))





def PI_to_T1_cyclegan():
    YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
    fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
    opt = load_config("config/CycWave-Mamba3D_config.yaml", mode="test")
    opt.image = fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz'
    opt.result =fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI.nii.gz'
    opt.checkpoints_dir = os.getcwd() + '/checkpoints'
    opt.name = 'fMOSTPI2NMT'
    opt.phase = 'test'
    opt.model = 'test'
    model = create_model(opt)
    model.setup(opt)

    inference(model, opt.image, opt.result, opt.resample, opt.new_resolution, opt.patch_size[0],
              opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, 1)

def b_to_T1_cyclegan():
    YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
    fluor_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
    opt = load_config("config/config.yaml", mode="test")
    opt.name = 'Blockface2NMT'
    opt.netG = 'resnet_7blocks'
    opt.gpu_ids = ''
    opt.ngf = 48
    opt.ndf = 48
    opt.image = fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz'
    opt.result =fluor_CONFIG['output_dir']+'/reg3D/T1likeBlockface_origin.nii.gz'
    opt.checkpoints_dir = os.getcwd() + '/checkpoints'
    model = create_model(opt)
    model.setup(opt)
    inference(model, opt.image, opt.result, opt.resample, opt.new_resolution, opt.patch_size[0],
              opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, 1)
