import pywt
import torch
import torch.nn.functional as F
import cv2
import numpy as np

def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    print(wave)
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


def wavelet_2d_transform(x, filters):
    target_H, target_W = x.shape[-2:]
    x = F.interpolate(x, size=(target_H * 2, target_W * 2), mode='bilinear', align_corners=False)

    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_2d_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)

    target_H, target_W = x.shape[-2:]
    x = F.interpolate(x, size=(target_H // 2, target_W // 2), mode='bilinear', align_corners=False)
    return x

# ---------- 生成 dilation=2 的 2-D 可分离核 ----------
def create_2d_wavelet_filter_swt(wave, in_size, out_size, dtype=torch.float):
    """
    返回的核尺寸已插空（dilation=2），可直接用于 stride=1 的 conv
    """
    w = pywt.Wavelet(wave)
    # 插空（à-trous）系数，dilation=2
    def atrous(vec):
        return [v if i % 2 == 0 else 0 for i, v in enumerate(vec)] + [0]

    dec_lo_2 = torch.tensor(atrous(w.dec_lo[::-1]), dtype=dtype)
    dec_hi_2 = torch.tensor(atrous(w.dec_hi[::-1]), dtype=dtype)
    rec_lo_2 = torch.tensor(atrous(w.rec_lo), dtype=dtype)
    rec_hi_2 = torch.tensor(atrous(w.rec_hi), dtype=dtype)

    # 外积 → 4 个 2-D 核
    dec_filters = torch.stack([
        dec_lo_2.unsqueeze(0) * dec_lo_2.unsqueeze(1),
        dec_lo_2.unsqueeze(0) * dec_hi_2.unsqueeze(1),
        dec_hi_2.unsqueeze(0) * dec_lo_2.unsqueeze(1),
        dec_hi_2.unsqueeze(0) * dec_hi_2.unsqueeze(1)
    ], dim=0)[:, None]                          # (4, 1, kh, kw)

    rec_filters = torch.stack([
        rec_lo_2.unsqueeze(0) * rec_lo_2.unsqueeze(1),
        rec_lo_2.unsqueeze(0) * rec_hi_2.unsqueeze(1),
        rec_hi_2.unsqueeze(0) * rec_lo_2.unsqueeze(1),
        rec_hi_2.unsqueeze(0) * rec_hi_2.unsqueeze(1)
    ], dim=0)[:, None]

    # 复制到多通道
    dec_filters = dec_filters.repeat(in_size, 1, 1, 1)   # (4*C, 1, kh, kw)
    rec_filters = rec_filters.repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


# ---------- SWT 分析（尺寸不变） ----------
def swt_2d_transform(x, filters):
    """
    x: (B, C, H, W)   H,W 必须能被 2^level 整除（level=1 时偶数即可）
    filters: (4*C, 1, kh, kw)  dilation=2 核
    return: (B, C, 4, H, W)   同尺寸！
    """
    b, c, h, w = x.shape
    kh, kw = filters.shape[2], filters.shape[3]
    pad = (kh - 1) * 2 // 2  # 保持尺寸
    x = F.conv2d(x, filters, stride=1, dilation=2, groups=c, padding=pad)
    return x.view(b, c, 4, h, w)


# ---------- SWT 合成（尺寸不变） ----------
def iswt_2d_transform(x, filters):
    """
    x: (B, C, 4, H, W)
    filters: (4*C, 1, kh, kw)  dilation=2 合成核
    return: (B, C, H, W)   同尺寸
    """
    b, c, _, h, w = x.shape
    kh, kw = filters.shape[2], filters.shape[3]
    dilation = 2
    # 让输出尺寸 = 输入尺寸
    pad = (kh - 1) * dilation // 2
    x = x.reshape(b, c * 4, h, w)
    x = F.conv_transpose2d(x, filters, stride=1, dilation=dilation,
                           groups=c, padding=pad)
    return x

def save_wavelet_png(curr_x, path='wavelet_allChn.png', chn_per_row=4):
    """
    curr_x: Tensor (B, C, 4, H, W)
    保存所有通道的小波系数图
    """
    with torch.no_grad():
        B, C, _, H, W = curr_x.shape
        # 先取 batch 0
        x = curr_x[0].cpu()                      # (C, 4, H, W)

        # 1. 每个通道拼 2×2 网格
        chn_grid = torch.zeros(C, 2*H, 2*W)
        chn_grid[:, 0:H, 0:W]   = x[:, 0]        # LL
        chn_grid[:, 0:H, W:2*W] = x[:, 1]        # LH
        chn_grid[:, H:2*H, 0:W] = x[:, 2]        # HL
        chn_grid[:, H:2*H, W:2*W] = x[:, 3]      # HH

        # 2. 全局归一化到 [0,1]（所有通道一起，保留相对强度）
        chn_grid = (chn_grid - chn_grid.min()) / (chn_grid.max() - chn_grid.min() + 1e-8)

        # 3. 拼成大图（每行 chn_per_row 个通道）
        rows = (C + chn_per_row - 1) // chn_per_row
        big_h = rows * 2 * H
        big_w = chn_per_row * 2 * W
        big_img = torch.zeros(big_h, big_w)
        for c in range(C):
            r = c // chn_per_row
            col = c % chn_per_row
            startr, startc = r * 2 * H, col * 2 * W
            big_img[startr:startr + 2*H, startc:startc + 2*W] = chn_grid[c]

        # 4. 保存
        cv2.imwrite(path, (big_img.numpy() * 255).astype(np.uint8))

def valid_input_size(min_side=256, levels=4):
    factor = 2 ** levels          # 16
    return (min_side + factor - 1) // factor * factor   # 向上对齐到 16 倍数
