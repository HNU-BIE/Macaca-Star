import pywt
import torch
import torch.nn.functional as F
import cv2
import numpy as np

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

