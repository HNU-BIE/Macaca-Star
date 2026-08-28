import pywt
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn

def create_3d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_3d_transform(x5d, filters):
    target_D, target_H, target_W = x5d.shape[-3:]
    x5d = F.interpolate(x5d, size=(target_D * 2, target_H, target_W * 2), mode='trilinear', align_corners=False)
    filters = filters.to(x5d.device)
    B, C, D, H, W = x5d.shape
    filters_3d = filters.unsqueeze(3)
    pad = (filters_3d.shape[2] // 2 - 1, 0,filters_3d.shape[4] // 2 - 1)
    y = F.conv3d(x5d, filters_3d, stride=(2,1,2), padding=pad, groups=C)  # (B*D, 4*C, H//2, W//2)
    y = y.view(B, C, 4, D//2,H, W//2)
    return y

def inverse_wavelet_3d_transform(y6d, filters):
    filters = filters.to(y6d.device)
    B, C, _, D, H, W = y6d.shape
    pad = (filters.shape[2] // 2 - 1,0, filters.shape[3] // 2 - 1)
    filters_3d = filters.unsqueeze(3)
    y = y6d.view(B , 4 * C, D,H, W)
    x = F.conv_transpose3d(y, filters_3d, stride=(2,1,2), padding=pad, groups=C)
    x = x.view(B, C, D * 2,H, W * 2)
    target_D, target_H, target_W = x.shape[-3:]
    x = F.interpolate(x, size=(target_D // 2, target_H, target_W // 2), mode='trilinear', align_corners=False)
    return x


class WTConv3d_D(nn.Module):
    """3D 小波卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv3d_D, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 使用您提供的 create_3d_wavelet_filter 生成 3D 滤波器[cite: 2]
        self.wt_filter, self.iwt_filter = create_3d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 基础特征的 3D 卷积[cite: 1]
        self.base_conv = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule3D([1, in_channels, 1, 1, 1])

        # 小波特征的 3D 卷积，因您的 3D 小波依然输出 4 个子带，因此为 in_channels * 4[cite: 2]
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv3d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule3D([1, in_channels * 4, 1, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool3d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 3D 填充：顺序为 (W_left, W_right, H_top, H_bottom, D_front, D_back)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0) or (curr_shape[4] % 2 > 0):
                curr_pads = (0, curr_shape[4] % 2, 0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 执行 3D 小波变换[cite: 2]
            curr_x = wavelet_3d_transform(curr_x_ll, self.wt_filter)

            # 提取 LL 子带 (索引为0)[cite: 2]
            curr_x_ll = curr_x[:, :, 0, :, :, :]

            shape_x = curr_x.shape
            # 将 4 个子带合并到通道维度以通过卷积[cite: 1, 2]
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4], shape_x[5])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 分离卷积后的 LL 和 高频(H) 子带[cite: 1]
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            # 重新拼接 LL 和 H 子带，准备进行 3D 逆变换[cite: 1]
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_3d_transform(curr_x, self.iwt_filter)

            # 恢复原始尺寸[cite: 1]
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3], :curr_shape[4]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # 残差连接：基础卷积特征与小波卷积特征相加[cite: 1]
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule3D(nn.Module):
    """3D Scale Module: 扩展了维度以适应 [B, C, D, H, W] 的 5D 张量"""
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule3D, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

def save_wavelet_png(curr_x, path='wavelet.png', chn_per_row=4):

    with torch.no_grad():
        B, C, _, H, W = curr_x.shape
        x = curr_x[0].cpu()                      # (C, 4, H, W)
        chn_grid = torch.zeros(C, 2*H, 2*W)
        chn_grid[:, 0:H, 0:W]   = x[:, 0]        # LL
        chn_grid[:, 0:H, W:2*W] = x[:, 1]        # LH
        chn_grid[:, H:2*H, 0:W] = x[:, 2]        # HL
        chn_grid[:, H:2*H, W:2*W] = x[:, 3]      # HH
        chn_grid = (chn_grid - chn_grid.min()) / (chn_grid.max() - chn_grid.min() + 1e-8)

        rows = (C + chn_per_row - 1) // chn_per_row
        big_h = rows * 2 * H
        big_w = chn_per_row * 2 * W
        big_img = torch.zeros(big_h, big_w)
        for c in range(C):
            r = c // chn_per_row
            col = c % chn_per_row
            startr, startc = r * 2 * H, col * 2 * W
            big_img[startr:startr + 2*H, startc:startc + 2*W] = chn_grid[c]
        cv2.imwrite(path, (big_img.numpy() * 255).astype(np.uint8))

