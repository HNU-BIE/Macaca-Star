import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from utils.CycWaveM2D.models import wavelet
from torch.nn.utils import spectral_norm
from mamba_ssm import Mamba
import torch.nn.functional as F

from utils.CycWaveM2D.models.wtconv2d import WTConv2d_D


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = WTResnetGenerator2D_HybridMamba(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,wavelet_name='haar')
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NWTLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_WTD(input_nc, ndf, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = WaveletTextureDiscriminator2D(input_nc, ndf)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor or list) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if isinstance(prediction, list):
            loss = 0
            for pred_i in prediction:
                if self.gan_mode in ['lsgan', 'vanilla']:
                    target_tensor = self.get_target_tensor(pred_i, target_is_real)
                    loss += self.loss(pred_i, target_tensor)
                elif self.gan_mode == 'wgangp':
                    if target_is_real:
                        loss += -pred_i.mean()
                    else:
                        loss += pred_i.mean()
            return loss / len(prediction)

        else:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            return loss



def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class WaveletMapper(nn.Module):
    """
    2D Wavelet Subband Feature Mapper.

    Processes and enhances high-frequency 2D wavelet subbands (e.g., LH, HL, HH) using
    cross-subband feature interaction, 2D depthwise spatial convolutions, channel/subband
    gating attention, and learnable residual scaling.

    Architecture Overview:
      1. Cross-Subband Mixing: 1x1 2D grouped convolution across subbands within each channel.
      2. 2D Depthwise Convolution: Spatial feature extraction and context aggregation.
      3. Pointwise Projection: 1x1 2D grouped convolution for subband feature projection.
      4. Subband Gating (SE-like attention): 2D adaptive average pooling followed by an MLP
         to compute adaptive channel-subband attention weights.
      5. Learnable Residual Combination: Scaled combination of identity and delta representations.

    Input / Output Shape:
      [B, C, K, H, W] (where K is the number of subbands)
    """
    def __init__(self, c: int, subbands: int = 3, kernel_size: int = 3, init_alpha: float = -0.5,
                 init_scale: float = 0.05, use_gate: bool = True):
        """
        :param c: Number of feature channels per subband.
        :param subbands: Number of high-frequency 2D wavelet subbands (K, default: 3).
        :param kernel_size: Spatial kernel size for depthwise 2D convolution.
        :param init_alpha: Initial value for the identity path scaling parameter.
        :param init_scale: Initial value for the residual transformation path scaling parameter.
        :param use_gate: Whether to enable adaptive channel/subband attention gating.
        """
        super().__init__()
        self.subbands = subbands
        self.use_gate = use_gate

        # Learnable scaling parameters per subband (5D tensor: [1, 1, subbands, 1, 1])
        self.alpha = nn.Parameter(torch.ones(1, 1, subbands, 1, 1) * init_alpha)
        self.scale = nn.Parameter(torch.ones(1, 1, subbands, 1, 1) * init_scale)

        # Cross-subband mixing within each channel (groups=c allows subbands of the same channel to interact)
        self.mix = nn.Conv2d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)

        # 2D Depthwise spatial convolution for local context aggregation
        self.dwconv = nn.Conv2d(c * subbands, c * subbands, kernel_size=kernel_size, padding=kernel_size // 2,
                                groups=c * subbands, bias=False)

        # 2D Pointwise convolution for subband feature projection
        self.pwconv = nn.Conv2d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)

        # 2D Channel/subband-wise gating attention network (SE-like module)
        hidden = max(c * subbands // 8, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * subbands, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c * subbands, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D wavelet subband transformation.

        :param x: High-frequency subband tensor of shape [B, C, K, H, W].
        :return: Transformed and scale-calibrated subband tensor of shape [B, C, K, H, W].
        """
        B, C, K, H, W = x.shape
        assert K == self.subbands, f"Expected {self.subbands} sub-bands, got {K}"
        identity = x

        # 1. Flatten Channel and Subband dimensions: [B, C, K, H, W] -> [B, C * K, H, W]
        x4d = x.reshape(B, C * K, H, W)

        # 2. Sequential feature transformation: Cross-subband mix -> Depthwise conv -> Pointwise conv
        delta = self.mix(x4d)
        delta = self.dwconv(delta)
        delta = self.pwconv(delta)

        # 3. Reshape back to 5D subband tensor: [B, C * K, H, W] -> [B, C, K, H, W]
        delta = delta.view(B, C, K, H, W)

        # 4. Apply adaptive attention gating if enabled
        if self.use_gate:
            gate = self.gate(x4d).view(B, C, K, 1, 1)
            delta = delta * gate

        # 5. Weighted residual combination of identity and transformed features
        return self.alpha * identity + self.scale * delta


class VisionMambaBlock2D(nn.Module):
    """
    2D Dual-Axis Shared Vision Mamba Block.

    Features:
      - Axis-Shared State Space Model (SSM): Reuses a single Mamba layer to perform
        sequential scanning along both Height (H) and Width (W) axes independently.
      - Adaptive Axis Fusion: Learnable softmax-normalized weights to dynamically blend
        directional feature representations.
      - 2D Channel Pre-LayerNorm: Stabilizes optimization by normalizing across the channel
        dimension before sequential scanning.
      - Residual Learning: Preserves input features via a standard skip connection.

    Input / Output Shape:
      [B, C, H, W]
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        """
        :param d_model: Number of feature channels (hidden dimension).
        :param d_state: SSM state expansion factor (latent state dimension).
        :param d_conv: Kernel size of the 1D local convolution inside Mamba.
        :param expand: Inner feature expansion factor within the Mamba block.
        """
        super().__init__()
        self.d_model = d_model

        # 1. Dual-axis shared Mamba layer (processes 1D sequences across H and W axes)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 2. Learnable axis fusion weights initialized to 1 for [H, W]
        self.axis_weight = nn.Parameter(torch.ones(2))

        # 3. 1x1 2D convolution for channel-wise feature mixing and projection
        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1)

        # 4. LayerNorm applied across the channel dimension
        self.norm = nn.LayerNorm(d_model)

    def ln_2d(self, x):
        """
        Apply LayerNorm over the channel dimension of a 4D tensor.

        :param x: 4D tensor of shape [B, C, H, W].
        :return: Normalized tensor of shape [B, C, H, W].
        """
        # Permute channel to the last dimension: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # Restore original dimension ordering: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        """
        Apply LayerNorm over the channel dimension of a 4D tensor.

        :param x: 4D tensor of shape [B, C, H, W].
        :return: Normalized tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        residual = x

        # 1. Pre-Normalization
        x = self.ln_2d(x)

        # =====================================================================
        # 2. Height (H-axis) Sequential Scan
        # Flatten spatial W dimension to batch dimension:
        # [B, C, H, W] -> [B * W, H, C]
        # =====================================================================
        x_h = x.permute(0, 3, 2, 1).contiguous()
        x_h = x_h.reshape(B * W, H, C)
        out_h = self.mamba(x_h)

        # Reshape and restore dimensions: [B * W, H, C] -> [B, C, H, W]
        out_h = out_h.reshape(B, W, H, C)
        out_h = out_h.permute(0, 3, 2, 1)

        # =====================================================================
        # 3. Width (W-axis) Sequential Scan
        # Flatten spatial H dimension to batch dimension:
        # [B, C, H, W] -> [B * H, W, C]
        # =====================================================================
        x_w = x.permute(0, 2, 3, 1).contiguous()
        x_w = x_w.reshape(B * H, W, C)
        out_w = self.mamba(x_w)

        # Reshape and restore dimensions: [B * H, W, C] -> [B, C, H, W]
        out_w = out_w.reshape(B, H, W, C)
        out_w = out_w.permute(0, 3, 1, 2)

        # =====================================================================
        # 4. Adaptive Dual-Axis Feature Fusion (Softmax Weighted Combination)
        # =====================================================================
        weight = torch.softmax(self.axis_weight, dim=0)

        out = (
            weight[0] * out_h +
            weight[1] * out_w
        )

        # =====================================================================
        # 5. Linear Channel Projection and Residual Addition
        # =====================================================================
        out = self.proj(out)

        return residual + out


class WTResnetGenerator2D_HybridMamba(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, padding_type='reflect', wavelet_name='haar', blocks=[4, 2, 4], ismerge=False):

        super(WTResnetGenerator2D_HybridMamba, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ismerge = ismerge
        print('WTResnetGenerator2D_HybridMamba -> ismerge: ' + str(ismerge))


        self.pad1 = nn.ReplicationPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, bias=use_bias, padding=0)
        self.norm1 = norm_layer(ngf)


        # level-1 64→128
        self.wt_p1 = WaveletMapper(ngf)
        wt_f2, iwt_f2 = wavelet.create_2d_wavelet_filter(wavelet_name, ngf, ngf, torch.float)
        wt_f2 = nn.Parameter(wt_f2, requires_grad=False)
        iwt_f2 = nn.Parameter(iwt_f2, requires_grad=False)
        self.register_buffer('wt_f2', wt_f2)
        self.register_buffer('iwt_f2', iwt_f2)


        self.conv_wt = nn.ConvTranspose2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, output_padding=1, padding=1,
                                          bias=use_bias)
        self.norm_wt = norm_layer(ngf * 1)

        self.conv2 = nn.Conv2d(ngf * 1, ngf * 2, 3, stride=2, padding=1, bias=use_bias)
        self.norm2 = norm_layer(ngf * 2)

        # level-2 128→256
        self.wt_p2 = WaveletMapper(ngf * 2)
        wt_f, iwt_f = wavelet.create_2d_wavelet_filter(wavelet_name, ngf * 2, ngf * 2, torch.float)
        wt_f = nn.Parameter(wt_f, requires_grad=False)
        iwt_f = nn.Parameter(iwt_f, requires_grad=False)
        self.register_buffer('wt_f3', wt_f)
        self.register_buffer('iwt_f3', iwt_f)

        self.conv_wt2 = nn.ConvTranspose2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, output_padding=1, padding=1,
                                           bias=use_bias)
        self.norm_wt2 = norm_layer(ngf * 2)

        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ngf * 4)


        self.bottleneck_blocks = nn.ModuleList()


        for _ in range(blocks[0]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )


        for _ in range(blocks[1]):
            self.bottleneck_blocks.append(VisionMambaBlock2D(d_model=ngf * 4))


        for _ in range(blocks[2]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )


        self.up1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up1 = norm_layer(ngf * 2)

        self.conv_iwt2 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm_iwt2 = norm_layer(ngf * 2)

        self.up2 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up2 = norm_layer(ngf * 1)

        self.conv_iwt = nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm_iwt = norm_layer(ngf * 1)

        self.iwt_up = nn.ConvTranspose2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, output_padding=1, padding=1,
                                         bias=use_bias)
        self.iwt_up_norm = norm_layer(ngf)

        self.conv_cat = nn.Conv2d(ngf * 2, ngf * 1, 3, stride=1, padding=1, bias=use_bias)
        self.conv_cat_norm = norm_layer(ngf)


        self.pad2 = nn.ReplicationPad2d(3)
        self.conv_out = nn.Conv2d(ngf, output_nc, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x, inplace=True)

        coeff_list = []

        curr_x = wavelet.wavelet_2d_transform(x, self.wt_f2)

        hh = self.wt_p1(curr_x[:, :, 1:, :, :])
        coeff_list.append(hh)

        x = curr_x[:, :, 0, :, :]
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x, inplace=True)

        curr_x = wavelet.wavelet_2d_transform(x, self.wt_f3)
        hh = self.wt_p2(curr_x[:, :, 1:, :, :])
        coeff_list.append(hh)

        x = curr_x[:, :, 0, :, :]
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.silu(x, inplace=True)

        # ---------- 🌟 Hybrid Mamba Bottleneck ----------
        for blk in self.bottleneck_blocks:
            x = blk(x)


        x = self.up1(x)
        x = self.norm_up1(x)
        x = F.silu(x, inplace=True)

        coeff = coeff_list.pop()

        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x = wavelet.inverse_wavelet_2d_transform(ll_4, self.iwt_f3)

        x = self.up2(x)
        x = self.norm_up2(x)
        x = F.silu(x, inplace=True)

        coeff = coeff_list.pop()
        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x_iwt = wavelet.inverse_wavelet_2d_transform(ll_4, self.iwt_f2)

        if self.ismerge:
            x_up = self.iwt_up(x)
            x_up = self.iwt_up_norm(x_up)
            x_up = F.silu(x_up, inplace=True)
            x = torch.cat([x_iwt, x_up], dim=1)
            x = self.conv_cat(x)
            x = self.conv_cat_norm(x)
            x = F.silu(x, inplace=True)
        else:
            x = x_iwt


        x = self.pad2(x)
        x = self.conv_out(x)
        x = self.tanh(x)

        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NWTLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NWTLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                WTConv2d_D(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class WaveletTextureDiscriminator2D(nn.Module):
    """
    WT-based 2D frequency discriminator.

    Input:
        [B, C, H, W]

    Wavelet output:
        [B, C, 4, H, W] (1 LL subband + 3 High-frequency subbands)
    """

    def __init__(self, input_nc, ndf=32, wavelet_name='haar', ll_weight=0.05):
        super().__init__()
        self.ll_weight = ll_weight


        wt_filter, _ = wavelet.create_2d_wavelet_filter(wavelet_name, input_nc, input_nc, torch.float)
        self.register_buffer('wt_filter', wt_filter)


        hf_input_channels = input_nc * 6
        ll_channels = max(ndf // 4, 8)


        self.hf_net = nn.Sequential(
            spectral_norm(nn.Conv2d(hf_input_channels, ndf, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.ll_net = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ll_channels, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out_conv = spectral_norm(nn.Conv2d(ndf * 2 + ll_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x, return_components=False):

        wt = wavelet.wavelet_2d_transform(x, self.wt_filter)


        B, C, _, H, W = wt.shape


        low_freq = wt[:, :, 0, :, :]
        high_freq = wt[:, :, 1:, :, :].contiguous().reshape(B, C * 3, H, W)


        high_freq_feature = torch.cat([high_freq, high_freq.abs()], dim=1)

        hf_feat = self.hf_net(high_freq_feature)
        ll_feat = self.ll_net(low_freq)


        feat = torch.cat([hf_feat, self.ll_weight * ll_feat], dim=1)
        pred = self.out_conv(feat)

        if return_components:
            return pred, hf_feat, ll_feat
        return pred