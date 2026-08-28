import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import monai
from mamba_ssm import Mamba
from utils.CycWaveM3D.models import wavelet
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from utils.CycWaveM3D.models.wavelet import WTConv3d_D


###############################################################################
# Helper Functions
###############################################################################


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

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_Gdefine_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],blocks=[3,2,3],ismerge=False):
    norm_layer = get_norm_layer(norm_type=norm)
    net = WTResnetGenerator3D_HybridMamba(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,blocks=blocks,ismerge=ismerge)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf,n_layers=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = WTLayerDiscriminator3D(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_WTD(input_nc, ndf, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = WaveletTextureDiscriminator(input_nc, ndf)
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


'''
define the correlation coefficient loss
'''
def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2 # abslute constrain


class WTResnetGenerator3D_HybridMamba(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d,
                 use_dropout=False, padding_type='reflect', wavelet_name='haar',blocks=[4,2,4] ,ismerge=False):

        super(WTResnetGenerator3D_HybridMamba, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.ismerge = ismerge
        print('WTResnetGenerator3D_HybridMamba -> ismerge: ' + str(ismerge))

        # ---------- head ----------
        self.pad1 = nn.ReplicationPad3d(3)
        self.conv1 = nn.Conv3d(input_nc, ngf, 7, bias=use_bias, padding=0)
        self.norm1 = norm_layer(ngf)

        # level-1 64→128
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

        # level-2 128→256
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

        # ----------  2-4-2 Hybrid Mamba  ----------
        self.bottleneck_blocks = nn.ModuleList()

        for _ in range(blocks[0]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )

        for _ in range(blocks[1]):
            self.bottleneck_blocks.append(VisionMambaBlock3D(d_model=ngf * 4))

        for _ in range(blocks[2]):
            self.bottleneck_blocks.append(
                ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            )

        self.up1 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up1 = norm_layer(ngf * 2)
        self.conv_iwt2 = nn.Conv3d(ngf * 2, ngf * 2, (3, 1, 3), stride=(2, 1, 2), padding=(1, 0, 1), bias=use_bias)
        self.norm_iwt2 = norm_layer(ngf * 2)

        self.up2 = nn.ConvTranspose3d(ngf * 2, ngf * 1, 3, stride=2, output_padding=1, padding=1, bias=use_bias)
        self.norm_up2 = norm_layer(ngf * 1)

        self.conv_iwt = nn.Conv3d(ngf * 1, ngf * 1, (3, 1, 3), stride=(2, 1, 2), padding=(1, 0, 1), bias=use_bias)
        self.norm_iwt = norm_layer(ngf * 1)

        self.iwt_up = nn.ConvTranspose3d(ngf * 1, ngf * 1, (3, 1, 3), stride=(2, 1, 2), output_padding=(1, 0, 1),
                                         padding=(1, 0, 1), bias=use_bias)
        self.iwt_up_norm = norm_layer(ngf)

        self.conv_cat = nn.Conv3d(ngf * 2, ngf * 1, 3, stride=1, padding=1, bias=use_bias)
        self.conv_cat_norm = norm_layer(ngf)

        self.pad2 = nn.ReplicationPad3d(3)
        self.conv_out = nn.Conv3d(ngf, output_nc, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # ---------- head ----------
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x, inplace=True)

        coeff_list = []

        curr_x = wavelet.wavelet_3d_transform(x, self.wt_f2)
        hh = self.wt_p1(curr_x[:, :, 1:, :, :, :])
        coeff_list.append(hh)

        x = curr_x[:, :, 0, :, :, :]
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x, inplace=True)

        curr_x = wavelet.wavelet_3d_transform(x, self.wt_f3)
        hh = self.wt_p2(curr_x[:, :, 1:, :, :, :])
        coeff_list.append(hh)

        x = curr_x[:, :, 0, :, :, :]
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.silu(x, inplace=True)

        # ---------- Hybrid Mamba Bottleneck ----------
        for blk in self.bottleneck_blocks:
            x = blk(x)

        x = self.up1(x)
        x = self.norm_up1(x)
        x = F.silu(x, inplace=True)

        coeff = coeff_list.pop()
        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x = wavelet.inverse_wavelet_3d_transform(ll_4, self.iwt_f3)

        x = self.up2(x)
        x = self.norm_up2(x)
        x = F.silu(x, inplace=True)

        coeff = coeff_list.pop()
        ll_4 = torch.cat([x.unsqueeze(2), coeff], dim=2)
        x_iwt = wavelet.inverse_wavelet_3d_transform(ll_4, self.iwt_f2)

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

        # ---------- 输出 ----------
        x = self.pad2(x)
        x = self.conv_out(x)
        x = self.tanh(x)

        return x

class WaveletMapper(nn.Module):
    def __init__(self, c: int, subbands: int = 3, kernel_size: int = 3, init_alpha: float = -0.5, init_scale: float = 0.05, use_gate: bool = True):
        super().__init__()
        self.subbands = subbands
        self.use_gate = use_gate
        self.alpha = nn.Parameter(torch.ones(1, 1, subbands, 1, 1, 1) * init_alpha)
        self.scale = nn.Parameter(torch.ones(1, 1, subbands, 1, 1, 1) * init_scale)
        self.mix = nn.Conv3d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)
        self.dwconv = nn.Conv3d(c * subbands, c * subbands, kernel_size=kernel_size, padding=kernel_size // 2, groups=c * subbands, bias=False)
        self.pwconv = nn.Conv3d(c * subbands, c * subbands, kernel_size=1, groups=c, bias=False)
        hidden = max(c * subbands // 8, 1)
        self.gate = nn.Sequential(GlobalAvgPool3d(), nn.Conv3d(c * subbands, hidden, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv3d(hidden, c * subbands, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, K, D, H, W = x.shape
        assert K == self.subbands, f"Expected {self.subbands} sub-bands, got {K}"
        identity = x
        x5d = x.reshape(B, C * K, D, H, W)
        delta = self.mix(x5d)
        delta = self.dwconv(delta)
        delta = self.pwconv(delta)
        delta = delta.view(B, C, K, D, H, W)
        if self.use_gate:
            gate = self.gate(x5d).view(B, C, K, 1, 1, 1)
            delta = delta * gate
        return self.alpha * identity + self.scale * delta

class VisionMambaBlock3D(nn.Module):
    """
    3D Axis Shared Mamba Block

    Features:
        - Shared Mamba for D/H/W axis
        - Learnable axis fusion
        - LayerNorm pre-normalization

    Input:
        [B,C,D,H,W]
    """

    def __init__(self, d_model, d_state=16,d_conv=4, expand=1):
        super().__init__()

        self.d_model = d_model

        # 三轴共享 Mamba
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 可学习轴权重 D/H/W
        self.axis_weight = nn.Parameter(torch.ones(3))

        # 通道融合
        self.proj = nn.Conv3d(d_model, d_model, kernel_size=1)

        # LayerNorm over channel dimension
        self.norm = nn.LayerNorm(d_model)


    def ln_3d(self, x):
        """
        x:
        [B,C,D,H,W]

        LayerNorm channel:
        [B,D,H,W,C]
        """

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


    def forward(self, x):
        """
        x: [B,C,D,H,W]
        """
        B, C, D, H, W = x.shape
        residual = x

        # Pre-Norm
        x = self.ln_3d(x)

        # ==============================
        # D-axis scan
        # [B,C,D,H,W]
        # -> [B*H*W,D,C]
        # ==============================

        x_d = x.permute(0, 3, 4, 2, 1).contiguous()
        x_d = x_d.reshape(B * H * W, D, C)

        out_d = self.mamba(x_d)

        out_d = out_d.reshape(B, H, W, D, C)
        out_d = out_d.permute(0, 4, 3, 1, 2)


        # ==============================
        # H-axis scan
        # [B,C,D,H,W]
        # -> [B*D*W,H,C]
        # ==============================

        x_h = x.permute(0, 2, 4, 3, 1).contiguous()
        x_h = x_h.reshape(B * D * W, H, C)

        out_h = self.mamba(x_h)

        out_h = out_h.reshape(B, D, W, H, C)
        out_h = out_h.permute(0, 4, 1, 3, 2)


        # ==============================
        # W-axis scan
        # [B,C,D,H,W]
        # -> [B*D*H,W,C]
        # ==============================

        x_w = x.permute(0, 2, 3, 4, 1).contiguous()
        x_w = x_w.reshape(B * D * H, W, C)

        out_w = self.mamba(x_w)

        out_w = out_w.reshape(B, D, H, W, C)
        out_w = out_w.permute(0, 4, 1, 2, 3)

        # ==============================
        # Adaptive axis fusion
        # ==============================

        weight = torch.softmax(self.axis_weight, dim=0)

        out = (
            weight[0] * out_d +
            weight[1] * out_h +
            weight[2] * out_w
        )


        # ==============================
        # Projection + GELU
        # ==============================

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

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(WTLayerDiscriminator3D, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                WTConv3d_D(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class WaveletTextureDiscriminator(nn.Module):
    """
    WT-based 3D frequency discriminator.

    Input:
        [B,C,D,H,W]

    Wavelet output:
        [B,C,4,D,H,W]
    """

    def __init__(self, input_nc, ndf=32, wavelet_name='haar', ll_weight=0.05):
        super().__init__()
        self.ll_weight = ll_weight

        wt_filter, _ = wavelet.create_3d_wavelet_filter(wavelet_name, input_nc, input_nc, torch.float)
        self.register_buffer('wt_filter', wt_filter)

        hf_input_channels = input_nc * 6
        ll_channels = max(ndf // 4, 8)

        self.hf_net = nn.Sequential(
            spectral_norm(nn.Conv3d(hf_input_channels, ndf, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(ndf * 2, ndf * 2, kernel_size=3, stride=1, padding=2, dilation=2)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.ll_net = nn.Sequential(
            spectral_norm(nn.Conv3d(input_nc, ll_channels, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out_conv = spectral_norm(nn.Conv3d(ndf * 2 + ll_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x, return_components=False):
        wt = wavelet.wavelet_3d_transform(x, self.wt_filter)
        B, C, _, D, H, W = wt.shape

        low_freq = wt[:, :, 0, :, :, :]
        high_freq = wt[:, :, 1:, :, :, :].contiguous().reshape(B, C * 3, D, H, W)
        high_freq_feature = torch.cat([high_freq, high_freq.abs()], dim=1)

        hf_feat = self.hf_net(high_freq_feature)
        ll_feat = self.ll_net(low_freq)
        feat = torch.cat([hf_feat, self.ll_weight * ll_feat], dim=1)
        pred = self.out_conv(feat)

        if return_components:
            return pred, hf_feat, ll_feat
        return pred