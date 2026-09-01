import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from mamba_ssm import Mamba
from utils.CycWaveM3D.models import wavelet
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from utils.CycWaveM3D.models.wavelet import WTConv3d_D

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], blocks=[3,2,3],ismerge=False):
    """
        Instantiate, configure, and initialize the 3D generator network (WTResnetGenerator3D_HybridMamba).
    """
    # Retrieve the normalization layer constructor based on the specified norm type
    norm_layer = get_norm_layer(norm_type=norm)

    # Instantiate the Wavelet-Transform 3D Hybrid Mamba Generator
    net = WTResnetGenerator3D_HybridMamba(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,blocks=blocks,ismerge=ismerge)

    # Initialize network weights and distribute across specified GPU device(s)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf,n_layers=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Instantiate, configure, and initialize the 3D discriminator network (WTLayerDiscriminator3D).
    """
    # Retrieve the normalization layer constructor based on the specified norm type
    norm_layer = get_norm_layer(norm_type=norm)

    # Instantiate the Wavelet-Transform 3D Layer (PatchGAN) Discriminator
    net = WTLayerDiscriminator3D(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    # Initialize network weights and distribute across specified GPU device(s)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_WTD(input_nc, ndf, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Instantiate, configure, and initialize the Wavelet Texture Discriminator network (WaveletTextureDiscriminator).
    """
    # Instantiate the Wavelet Texture Discriminator
    net = WaveletTextureDiscriminator(input_nc, ndf)

    # Initialize network weights and distribute across specified GPU device(s)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class WTResnetGenerator3D_HybridMamba(nn.Module):
    """
    3D Wavelet-Transform ResNet Generator with Hybrid Vision Mamba Bottleneck.

    Architecture Overview:
      1. Initial Convolutional Head: Feature extraction using large-kernel (7x7x7) convolution.
      2. Multi-Level Wavelet Downsampling:
         - Decomposes spatial features into low-frequency and high-frequency subbands using 3D DWT.
         - Maps high-frequency detail coefficients via WaveletMapper while downsampling low frequencies.
      3. Hybrid Mamba Bottleneck:
         - Stage 1: Local feature extraction via 3D ResNet blocks.
         - Stage 2: Global long-range volumetric dependency modeling via 3D Vision Mamba (SSM) blocks.
         - Stage 3: Feature refinement via 3D ResNet blocks.
      4. Multi-Level Wavelet Upsampling:
         - Upsamples features and reconstructs multi-scale representations using 3D Inverse Wavelet Transform (IWT).
      5. Output Head: Large-kernel convolution with Tanh activation.
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d,
                 use_dropout=False, padding_type='reflect', wavelet_name='haar',blocks=[4,2,4] ,ismerge=False):
        """
        :param input_nc: Number of channels in input 3D image volumes.
        :param output_nc: Number of channels in output 3D image volumes.
        :param ngf: Number of generator feature filters in the first convolutional layer.
        :param norm_layer: Normalization layer class (e.g., BatchNorm3d or InstanceNorm3d).
        :param use_dropout: Whether to use dropout layers in residual blocks.
        :param padding_type: Padding type for convolutional layers ('reflect', 'replicate', or 'zero').
        :param wavelet_name: Mother wavelet family name for 3D DWT/IWT (e.g., 'haar').
        :param blocks: List specifying block counts [ResNet, VisionMamba3D, ResNet] in the bottleneck.
        :param ismerge: Whether to enable feature concatenation and fusion in the upsampling path.
        """
        super(WTResnetGenerator3D_HybridMamba, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # Determine bias usage based on normalization type
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.ismerge = ismerge

        # =====================================================================
        # 1. Initial Convolutional Head (input_nc -> ngf)
        # =====================================================================
        self.pad1 = nn.ReplicationPad3d(3)
        self.conv1 = nn.Conv3d(input_nc, ngf, 7, bias=use_bias, padding=0)
        self.norm1 = norm_layer(ngf)

        # =====================================================================
        # 2. Level-1 Downsampling & 3D Wavelet Decomposition (ngf -> ngf * 2)
        # =====================================================================
        self.wt_p1 = WaveletMapper(ngf)
        wt_f2, iwt_f2 = wavelet.create_3d_wavelet_filter(wavelet_name, ngf, ngf, torch.float)
        wt_f2 = nn.Parameter(wt_f2, requires_grad=False)
        iwt_f2 = nn.Parameter(iwt_f2, requires_grad=False)
        self.register_buffer('wt_f2', wt_f2)
        self.register_buffer('iwt_f2', iwt_f2)
        self.conv_wt = nn.ConvTranspose3d(ngf * 1, ngf * 1, (3, 1, 3), stride=(2, 1, 2), output_padding=(1, 0, 1),
                                          padding=(1, 0, 1), bias=use_bias)
        self.norm_wt = norm_layer(ngf * 1)
        self.conv2 = nn.Conv3d(ngf * 1, ngf * 2, 3, stride=2, padding=1, bias=use_bias)
        self.norm2 = norm_layer(ngf * 2)

        # =====================================================================
        # 3. Level-2 Downsampling & 3D Wavelet Decomposition (ngf * 2 -> ngf * 4)
        # =====================================================================
        self.wt_p2 = WaveletMapper(ngf * 2)
        wt_f, iwt_f = wavelet.create_3d_wavelet_filter(wavelet_name, ngf * 2, ngf * 2, torch.float)
        wt_f = nn.Parameter(wt_f, requires_grad=False)
        iwt_f = nn.Parameter(iwt_f, requires_grad=False)
        self.register_buffer('wt_f3', wt_f)
        self.register_buffer('iwt_f3', iwt_f)
        self.conv_wt2 = nn.ConvTranspose3d(ngf * 2, ngf * 2, (3, 1, 3), stride=(2, 1, 2), output_padding=(1, 0, 1),
                                           padding=(1, 0, 1), bias=use_bias)
        self.norm_wt2 = norm_layer(ngf * 2)

        self.conv3 = nn.Conv3d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ngf * 4)

        # =====================================================================
        # 4. Hybrid Bottleneck (ResNet -> Vision Mamba 3D -> ResNet)
        # =====================================================================
        self.bottleneck_blocks = nn.ModuleList()

        # Initial ResNet residual blocks
        for _ in range(blocks[0]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )

        # 3D Vision Mamba blocks for global volumetric state-space modeling
        for _ in range(blocks[1]):
            self.bottleneck_blocks.append(VisionMambaBlock3D(d_model=ngf * 4))

        # Subsequent ResNet residual blocks
        for _ in range(blocks[2]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )

        # =====================================================================
        # 5. Level-2 Upsampling & Inverse Wavelet Reconstruction (ngf * 4 -> ngf * 2)
        # =====================================================================
        self.up1 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up1 = norm_layer(ngf * 2)
        self.conv_iwt2 = nn.Conv3d(ngf * 2, ngf * 2, (3, 1, 3), stride=(2, 1, 2), padding=(1, 0, 1), bias=use_bias)
        self.norm_iwt2 = norm_layer(ngf * 2)

        # =====================================================================
        # 6. Level-1 Upsampling & Inverse Wavelet Reconstruction (ngf * 2 -> ngf * 1)
        # =====================================================================
        self.up2 = nn.ConvTranspose3d(ngf * 2, ngf * 1, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up2 = norm_layer(ngf * 1)

        self.conv_iwt = nn.Conv3d(ngf * 1, ngf * 1, (3, 1, 3), stride=(2, 1, 2), padding=(1, 0, 1), bias=use_bias)
        self.norm_iwt = norm_layer(ngf * 1)

        # Optional multi-scale feature merging layers
        self.iwt_up = nn.ConvTranspose3d(ngf * 1, ngf * 1, (3, 1, 3), stride=(2, 1, 2), output_padding=(1, 0, 1),padding=(1, 0, 1), bias=use_bias)
        self.iwt_up_norm = norm_layer(ngf)

        self.conv_cat = nn.Conv3d(ngf * 2, ngf * 1, 3, stride=1, padding=1, bias=use_bias)
        self.conv_cat_norm = norm_layer(ngf)

        # =====================================================================
        # 7. Output Convolutional Head (ngf -> output_nc)
        # =====================================================================
        self.pad2 = nn.ReplicationPad3d(3)
        self.conv_out = nn.Conv3d(ngf, output_nc, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the 3D Wavelet-Hybrid Mamba generator.

        :param x: Input 3D tensor of shape [B, input_nc, D, H, W].
        :return: Synthesized 3D tensor of shape [B, output_nc, D, H, W] normalized to [-1, 1].
        """
        # ---------- Step 1: Initial Convolutional Head ----------
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x, inplace=True)

        coeff_list = []
        # ---------- Step 2: Level-1 3D Wavelet Decomposition & Downsampling ----------
        curr_x = wavelet.wavelet_3d_transform(x, self.wt_f2)
        # Extract and map high-frequency wavelet subbands
        hh = self.wt_p1(curr_x[:, :, 1:, :, :, :])
        coeff_list.append(hh)

        # Propagate low-frequency subband (LL) through convolution
        x = curr_x[:, :, 0, :, :, :]
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x, inplace=True)

        # ---------- Step 3: Level-2 3D Wavelet Decomposition & Downsampling ----------
        curr_x = wavelet.wavelet_3d_transform(x, self.wt_f3)
        # Extract and map level-2 high-frequency wavelet subbands
        hh = self.wt_p2(curr_x[:, :, 1:, :, :, :])
        coeff_list.append(hh)

        # Propagate low-frequency subband (LL) to bottleneck
        x = curr_x[:, :, 0, :, :, :]
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.silu(x, inplace=True)

        # ---------- Step 4: Hybrid Bottleneck Processing ----------
        for blk in self.bottleneck_blocks:
            x = blk(x)

        # ---------- Step 5: Level-2 Upsampling & Inverse Wavelet Reconstruction ----------
        x = self.up1(x)
        x = self.norm_up1(x)
        x = F.silu(x, inplace=True)

        # Recombine low-frequency feature map with cached high-frequency coefficients
        coeff = coeff_list.pop()
        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x = wavelet.inverse_wavelet_3d_transform(ll_4, self.iwt_f3)

        # ---------- Step 6: Level-1 Upsampling & Inverse Wavelet Reconstruction ----------
        x = self.up2(x)
        x = self.norm_up2(x)
        x = F.silu(x, inplace=True)

        # Recombine with level-1 high-frequency coefficients
        coeff = coeff_list.pop()
        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x_iwt = wavelet.inverse_wavelet_3d_transform(ll_4, self.iwt_f2)

        # Optional: Feature fusion branch
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

        # ---------- Step 7: Output Projection ----------
        x = self.pad2(x)
        x = self.conv_out(x)
        x = self.tanh(x)

        return x

class WaveletMapper(nn.Module):
    """
    Wavelet Subband Feature Mapper.

    Processes and enhances high-frequency wavelet subbands using cross-subband feature
    interaction, depthwise 3D spatial convolution, channel/subband gating attention,
    and learnable residual scaling.

    Architecture:
      1. Cross-Subband Mixing: 1x1x1 grouped convolution to exchange information across subbands.
      2. 3D Depthwise Convolution: Spatial context aggregation within each subband.
      3. Pointwise Projection: 1x1x1 grouped convolution for subband feature refinement.
      4. Subband Gating (SE-like attention): Global average pooling followed by an MLP
         to compute adaptive channel-subband attention weights.
      5. Residual Combination: Learnable alpha (identity) and scale (delta) parameters.
    """
    def __init__(self, c: int, subbands: int = 3, kernel_size: int = 3, init_alpha: float = -0.5, init_scale: float = 0.05, use_gate: bool = True):
        super().__init__()
        """
        :param c: Number of feature channels per subband.
        :param subbands: Number of high-frequency wavelet subbands (K).
        :param kernel_size: Spatial kernel size for depthwise 3D convolution.
        :param init_alpha: Initial value for the identity path scaling factor.
        :param init_scale: Initial value for the residual transformation path scaling factor.
        :param use_gate: Whether to enable adaptive channel/subband attention gating.
        """
        self.subbands = subbands
        self.use_gate = use_gate

        # Learnable scaling parameters per subband
        self.alpha = nn.Parameter(torch.ones(1, 1, subbands, 1, 1, 1) * init_alpha)
        self.scale = nn.Parameter(torch.ones(1, 1, subbands, 1, 1, 1) * init_scale)

        # Cross-subband mixing within each channel (groups=c allows subbands of the same channel to interact)
        self.mix = nn.Conv3d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)

        # 3D Depthwise spatial convolution for local context aggregation
        self.dwconv = nn.Conv3d(c * subbands, c * subbands, kernel_size=kernel_size, padding=kernel_size // 2, groups=c * subbands, bias=False)

        # 3D Pointwise convolution for subband feature projection
        self.pwconv = nn.Conv3d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)

        # Channel/subband-wise gating attention network (SE-like module)
        hidden = max(c * subbands // 8, 1)
        self.gate = nn.Sequential(GlobalAvgPool3d(), nn.Conv3d(c * subbands, hidden, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv3d(hidden, c * subbands, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for wavelet subband transformation.

        :param x: High-frequency subband tensor of shape [B, C, K, D, H, W].
        :return: Transformed and scale-calibrated subband tensor of shape [B, C, K, D, H, W].
        """
        B, C, K, D, H, W = x.shape
        assert K == self.subbands, f"Expected {self.subbands} sub-bands, got {K}"
        identity = x

        # 1. Flatten Channel and Subband dimensions: [B, C, K, D, H, W] -> [B, C * K, D, H, W]
        x5d = x.reshape(B, C * K, D, H, W)

        # 2. Sequential feature transformation: Cross-subband mix -> Depthwise conv -> Pointwise conv
        delta = self.mix(x5d)
        delta = self.dwconv(delta)
        delta = self.pwconv(delta)

        # 3. Reshape back to 6D subband tensor: [B, C * K, D, H, W] -> [B, C, K, D, H, W]
        delta = delta.view(B, C, K, D, H, W)

        # 4. Apply adaptive attention gating if enabled
        if self.use_gate:
            gate = self.gate(x5d).view(B, C, K, 1, 1, 1)
            delta = delta * gate

        # 5. Weighted residual combination of identity and transformed features
        return self.alpha * identity + self.scale * delta

class VisionMambaBlock3D(nn.Module):
    """
    3D Tri-Axis Shared Vision Mamba Block.

    Features:
      - Axis-Shared State Space Model (SSM): Reuses a single Mamba layer to perform
        sequential scanning independently along the Depth (D), Height (H), and Width (W) axes.
      - Adaptive Axis Fusion: Learnable softmax-normalized weights to dynamically balance
        directional feature representations.
      - 3D Channel Pre-LayerNorm: Stabilizes optimization by applying LayerNorm over
        the channel dimension before sequential scanning.
      - Residual Learning: Standard skip connection preserving input representations.

    Input / Output Shape:
      [B, C, D, H, W]
    """
    def __init__(self, d_model, d_state=16,d_conv=4, expand=1):
        """
        :param d_model: Number of feature channels (hidden dimension).
        :param d_state: SSM state expansion factor (latent state dimension).
        :param d_conv: Kernel size of the 1D local convolution inside Mamba.
        :param expand: Inner feature expansion factor within the Mamba block.
        """
        super().__init__()
        self.d_model = d_model

        # 1. Tri-axis shared Mamba layer (processes 1D sequences across D, H, and W axes)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 2. Learnable axis fusion weights initialized to 1 for [D, H, W]
        self.axis_weight = nn.Parameter(torch.ones(3))

        # 3. 1x1x1 3D convolution for channel-wise feature mixing and projection
        self.proj = nn.Conv3d(d_model, d_model, kernel_size=1)

        # 4. LayerNorm applied across the channel dimension
        self.norm = nn.LayerNorm(d_model)

    def ln_3d(self, x):
        """
        Apply LayerNorm over the channel dimension of a 5D tensor.

        :param x: 5D tensor of shape [B, C, D, H, W].
        :return: Normalized tensor of shape [B, C, D, H, W].
        """
        # Permute channel to the last dimension: [B, C, D, H, W] -> [B, D, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        # Restore original dimension ordering: [B, D, H, W, C] -> [B, C, D, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        """
        Forward pass for 3D tri-axis directional scanning and adaptive feature fusion.

        :param x: Input feature tensor of shape [B, C, D, H, W].
        :return: Output feature tensor of shape [B, C, D, H, W].
        """
        B, C, D, H, W = x.shape
        residual = x

        # 1. Pre-Normalization
        x = self.ln_3d(x)

        # =====================================================================
        # 2. Depth (D-axis) Sequential Scan
        # Flatten spatial H and W dimensions to batch dimension:
        # [B, C, D, H, W] -> [B * H * W, D, C]
        # =====================================================================
        x_d = x.permute(0, 3, 4, 2, 1).contiguous()
        x_d = x_d.reshape(B * H * W, D, C)
        out_d = self.mamba(x_d)

        # Reshape and restore dimensions: [B * H * W, D, C] -> [B, C, D, H, W]
        out_d = out_d.reshape(B, H, W, D, C)
        out_d = out_d.permute(0, 4, 3, 1, 2)

        # =====================================================================
        # 3. Height (H-axis) Sequential Scan
        # Flatten spatial D and W dimensions to batch dimension:
        # [B, C, D, H, W] -> [B * D * W, H, C]
        # =====================================================================
        x_h = x.permute(0, 2, 4, 3, 1).contiguous()
        x_h = x_h.reshape(B * D * W, H, C)
        out_h = self.mamba(x_h)

        # Reshape and restore dimensions: [B * D * W, H, C] -> [B, C, D, H, W]
        out_h = out_h.reshape(B, D, W, H, C)
        out_h = out_h.permute(0, 4, 1, 3, 2)

        # =====================================================================
        # 4. Width (W-axis) Sequential Scan
        # Flatten spatial D and H dimensions to batch dimension:
        # [B, C, D, H, W] -> [B * D * H, W, C]
        # =====================================================================
        x_w = x.permute(0, 2, 3, 4, 1).contiguous()
        x_w = x_w.reshape(B * D * H, W, C)
        out_w = self.mamba(x_w)

        # Reshape and restore dimensions: [B * D * H, W, C] -> [B, C, D, H, W]
        out_w = out_w.reshape(B, D, H, W, C)
        out_w = out_w.permute(0, 4, 1, 2, 3)

        # =====================================================================
        # 5. Adaptive Tri-Axis Feature Fusion (Softmax Weighted Combination)
        # =====================================================================
        weight = torch.softmax(self.axis_weight, dim=0)
        out = (
            weight[0] * out_d +
            weight[1] * out_h +
            weight[2] * out_w
        )

        # =====================================================================
        # 6. Linear Channel Projection and Residual Addition
        # =====================================================================
        out = self.proj(out)
        return residual + out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class GlobalAvgPool3d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3, 4), keepdim=True)


class WTLayerDiscriminator3D(nn.Module):
    """
    3D Wavelet-Enhanced PatchGAN Discriminator.

    Architecture Overview:
      - Multi-scale Patch-based Discrimination: Evaluates realism locally on 3D volumetric patches.
      - Standard 3D Downsampling: Strided 3D convolutions with LeakyReLU activations and normalization.
      - Wavelet Convolution Integration (WTConv3d_D): Captures multi-frequency texture and structural details
        within intermediate discriminator layers to enhance edge and boundary discrimination.
      - Output: 1-channel 3D patch-level prediction map.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """
        :param input_nc: Number of channels in the input 3D volume (e.g., 1).
        :param ndf: Number of discriminator feature filters in the first convolutional layer.
        :param n_layers: Number of intermediate downsampling layers.
        :param norm_layer: Normalization layer class (e.g., BatchNorm3d or InstanceNorm3d).
        """
        super(WTLayerDiscriminator3D, self).__init__()

        # Determine bias usage based on normalization type
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1

        # =====================================================================
        # 1. Initial Downsampling Layer (input_nc -> ndf, no normalization)
        # =====================================================================
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        # =====================================================================
        # 2. Intermediate Downsampling & Wavelet Convolution Layers
        # =====================================================================
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # Standard strided 3D convolution downsampling
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),

                # 3D Wavelet Convolution for frequency-aware feature extraction
                WTConv3d_D(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # =====================================================================
        # 3. Penultimate Convolutional Layer (stride = 1)
        # =====================================================================
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        # =====================================================================
        # 4. Final Prediction Output Layer (1-channel PatchGAN score map)
        # =====================================================================
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Forward pass of the 3D Wavelet PatchGAN discriminator.

        :param input: Input 3D volume of shape [B, input_nc, D, H, W].
        :return: 3D PatchGAN prediction map of shape [B, 1, D', H', W'].
        """
        return self.model(input)

class WaveletTextureDiscriminator(nn.Module):
    """
    3D Wavelet-based Dual-Stream Texture and Frequency Discriminator.

    Architecture Overview:
      - 3D Discrete Wavelet Transform (DWT): Decomposes input volumes into low-frequency
        (approximation) and high-frequency (detail/texture) subbands.
      - Dual-Branch Architecture:
        1. High-Frequency Branch (hf_net): Captures fine textures, structural boundaries,
           and gradient magnitudes (using raw + absolute values and dilated convolutions).
        2. Low-Frequency Branch (ll_net): Captures global intensity trends and structural layout.
      - Spectral Normalization: Applied across all convolutional layers to stabilize GAN training.
      - Weighted Feature Fusion: Blends high-frequency and low-frequency representations with
        an adjustable low-frequency weight factor (ll_weight).

    Input Shape:
      [B, C, D, H, W]
    """
    def __init__(self, input_nc, ndf=32, wavelet_name='haar', ll_weight=0.05):
        """
        :param input_nc: Number of channels in the input 3D volume (e.g., 1).
        :param ndf: Base channel filter count for the high-frequency discriminator network.
        :param wavelet_name: Mother wavelet family name for 3D DWT (e.g., 'haar').
        :param ll_weight: Weight multiplier applied to low-frequency features during fusion.
        """
        super().__init__()
        self.ll_weight = ll_weight

        # 1. Create and register 3D Discrete Wavelet Transform filter buffers
        wt_filter, _ = wavelet.create_3d_wavelet_filter(wavelet_name, input_nc, input_nc, torch.float)
        self.register_buffer('wt_filter', wt_filter)

        # 6 channels = 3 high-frequency subbands x 2 (raw subband values + absolute values)
        hf_input_channels = input_nc * 6
        ll_channels = max(ndf // 4, 8)

        # =====================================================================
        # 2. High-Frequency Stream (Texture & Detail Discrimination)
        # =====================================================================
        self.hf_net = nn.Sequential(
            # Stage 1: Initial feature extraction
            spectral_norm(nn.Conv3d(hf_input_channels, ndf, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: Downsampling convolution
            spectral_norm(nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: Dilated convolution (dilation=2) to expand receptive field for texture context
            spectral_norm(nn.Conv3d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # =====================================================================
        # 3. Low-Frequency Stream (Global Structural Discrimination)
        # =====================================================================
        self.ll_net = nn.Sequential(
            spectral_norm(nn.Conv3d(input_nc, ll_channels, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # =====================================================================
        # 4. Final Prediction Projection Head
        # =====================================================================
        self.out_conv = spectral_norm(nn.Conv3d(ndf * 2 + ll_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x, return_components=False):
        """
        Forward pass for dual-stream frequency-aware discrimination.

        :param x: Input 3D volume tensor of shape [B, C, D, H, W].
        :param return_components: Whether to return intermediate high/low-frequency feature maps.
        :return: 3D PatchGAN realism score map [B, 1, D', H', W'], or (pred, hf_feat, ll_feat).
        """
        # 1. 3D Wavelet decomposition: [B, C, D, H, W] -> [B, C, 4, D', H', W']
        wt = wavelet.wavelet_3d_transform(x, self.wt_filter)
        B, C, _, D, H, W = wt.shape

        # 2. Separate low-frequency (LL) and high-frequency (LH, HL, HH) subbands
        low_freq = wt[:, :, 0, :, :, :]
        high_freq = wt[:, :, 1:, :, :, :].contiguous().reshape(B, C * 3, D, H, W)

        # Concatenate raw high-frequency coefficients with their absolute magnitudes (energy response)
        high_freq_feature = torch.cat([high_freq, high_freq.abs()], dim=1)

        # 3. Extract features through dual frequency streams
        hf_feat = self.hf_net(high_freq_feature)
        ll_feat = self.ll_net(low_freq)

        # 4. Fuse high-frequency and low-frequency representations
        feat = torch.cat([hf_feat, self.ll_weight * ll_feat], dim=1)

        # 5. Compute final discrimination score map
        pred = self.out_conv(feat)

        if return_components:
            return pred, hf_feat, ll_feat
        return pred