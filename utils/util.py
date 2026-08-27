#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：util.py
@Author  ：Zauber
@Date    ：2024/6/3
"""
import os
import numpy as np
import ants
import scipy
import tifffile
import yaml
from matplotlib import pyplot as plt
import cv2
import time
YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))

def horizontal():
    isPlot=True
    pi = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')
    pi_origin=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    for k in range(0, pi.shape[2]):
        pi_slice = pi[:, :, k].numpy()
        for i in range(0, pi_slice.shape[0]):
            pi_1d = pi_slice[i, :]
            v5 = np.where(pi_1d > 10)
            slice_data = pi_1d[v5]
            # rm low intensity in background and brain
            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 80) #95
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 10)
                    slice_data = slice_data[slice_data > percent_lower]
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.mean(slice_data)
                else:
                    pi_avg[i, k] = 0.0001
            else:
                pi_avg[i, k] = 0.0001

    # rm area including a part of cerebellum, as the area intensity is confused
    pi_avg_ = pi_avg[:, 0:pi_avg.shape[1]]
    pi_avg_1d = np.zeros((pi_avg.shape[0], 1))
    for i in range(0, pi_avg.shape[0]):
        tmp = pi_avg_[i, :]
        if len(tmp[tmp > 10.0]) > 0:
            pi_avg_1d[i] = np.mean(tmp[tmp > 10])
        else:
            pi_avg_1d[i] = 0
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[:, 0], distance=10, height=25, width=10)  ##50
    # add the first and last point
    peaks_ = np.insert(peaks, 0, 0)
    peaks_ = np.append(peaks_, pi_avg.shape[0] - 1)
    pi_avg_1d=pi_avg_1d[:,0]
    pi_avg_1d[pi_avg_1d.shape[0] - 1] = pi_avg_1d[peaks_[len(peaks_) - 2] - 10]
    pi_avg_1d[0] = pi_avg_1d[peaks_[1] - 10]
    y = pi_avg_1d[peaks_]
    x = peaks_
    x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
    if isPlot:
        plt.plot(pi_avg_1d, label='$origin$')
        plt.legend(loc='best')
        plt.plot(x, y, 'ro')
        plt.plot(x_curve, y_curve, label='$k=0.3$')
        plt.show()
    pi_avg_1d[pi_avg_1d<=1]=1
    pi_ratio_tmp=y_curve/pi_avg_1d
    pi_ratio=np.zeros((pi_avg.shape[0],pi_avg.shape[1]))
    for k in range(0, pi_avg.shape[1]):
        pi_ratio[:,k]=pi_ratio_tmp
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    # pi_data=pi.numpy()
    pi_avg=pi_avg.astype(np.float32)
    pi_ratio = pi_ratio.astype(np.float32)
    cv2.imwrite(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_avg_h.tif', pi_avg)
    cv2.imwrite(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_ratio_h.tif', pi_ratio)
    for j in range(0, pi.shape[1]):
        pi_origin[:, j, :] = pi_origin[:, j, :].numpy() * pi_ratio[:, :]


    ants.image_write(pi_origin, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rm_h.nii.gz')


def sagittal():
    remove_edge_light = False
    rm_bias = 20
    rm_value = 10
    pi_origin = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')
    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    pi_bessel = np.zeros((pi.shape[0], pi.shape[2]))
    for i in range(0, pi.shape[0]):

        pi_slice = pi[i, :, :].numpy().copy()

        for k in range(0, pi_slice.shape[1]):
            pi_1d = pi_slice[:, k]
            v5 = np.where(pi_1d > rm_value)
            slice_data = pi_1d[v5]

            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 95)
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 85)
                    slice_data = slice_data[slice_data > percent_lower]
                else:
                    pi_avg[i, k] = 1
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.average(slice_data)
                else:
                    pi_avg[i, k] = 1
            else:
                pi_avg[i, k] = 1

    pi_avg_1d = np.zeros((1, pi_avg.shape[1]))
    for k in range(0, pi_avg.shape[1]):
        tmp = pi_avg[:, k]
        if len(tmp[tmp > 10.0]) > 5:
            pi_avg_1d[0, k] = np.mean(tmp[tmp > 10.0])
        else:
            pi_avg_1d[0, k] = 0.0

    # plt.plot(pi_avg_1d, label='pi_avg_1d')
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[0, :], distance=10, height=20, width=5)
    if len(peaks) > 10:
        center_peaks_y = int(len(peaks) / 2)
        dis = 0
        for p in range(center_peaks_y - 5, center_peaks_y + 5):
            dis = dis + peaks[p] - peaks[p - 1]
        dis = int(dis / 10)
        print('dis y: ' + str(dis))
        peaks_ = peaks_restore(peaks, dis, 0, 0)
        peaks_ = peaks_restore(peaks_, dis, pi_avg_1d.shape[1], 1)
    else:
        peaks_ = peaks
    for i in range(0, pi_avg.shape[0]):
        slice_avg = pi_avg[i, :]
        threhold=0.5
        tmp = pi_avg[:, peaks_[1]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])*threhold
        if slice_avg[peaks_[1]] - t < 0:
            for p in range(2, len(peaks_)):
                tmp = pi_avg[:, peaks_[p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])*threhold
                if slice_avg[peaks_[p]] - t < 0:
                    slice_avg[peaks_[p - 1]] = slice_avg[peaks_[p]] - 10
                    break

        tmp = pi_avg[:, peaks_[len(peaks_) - 2]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])*threhold
        if slice_avg[peaks_[len(peaks_) - 2]] - t < 0:
            for p in range(3, len(peaks_)):
                tmp = pi_avg[:, peaks_[len(peaks_) - p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])*threhold
                if slice_avg[peaks_[len(peaks_) - p]] - t < 0:
                    slice_avg[peaks_[len(peaks_) - p + 1]] = slice_avg[peaks_[len(peaks_) - p]] - 10
                    break

        y = slice_avg[peaks_]

        x = peaks_
        x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
        pi_bessel[i, :] = y_curve
        print('i: ' + str(i) + '  ' + str(pi_avg.shape[0]))

    pi_ratio = pi_bessel / pi_avg
    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1.0
    # pi_ratio[pi_ratio < 0] = 1

    if remove_edge_light:
        for k in range(0, pi_ratio.shape[1]):
            for i in range(pi_ratio.shape[0] - 1, 0, -1):
                tmp = pi_ratio[i - rm_bias:i, k]
                if len(tmp[tmp > 1.010000]) / rm_bias >= 0.80:
                    pi_ratio[i - rm_bias:pi_ratio.shape[0], k] = pi_ratio[i - rm_bias, k]
                    break
    print('ratio correction end')

    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1
    pi_ratio[pi_ratio < 0] = 1
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0

    for j in range(0, pi.shape[1]):
        pi[:, j, :] = pi_origin[:, j, :].numpy().copy() * pi_ratio[:, :]
    pi_avg=pi_avg.astype(np.float32)
    pi_ratio = pi_ratio.astype(np.float32)
    tifffile.imwrite(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_avg_s.tif', pi_avg)
    tifffile.imwrite(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_ratio_s.tif', pi_ratio)
    ants.image_write(pi, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm.nii.gz')


def peaks_restore(peaks, dis, se, status):
    if status == 0:
        peaks[3] = peaks[4] - dis
        peaks[2] = peaks[3] - dis
        peaks[1] = peaks[2] - dis
        peaks[0] = peaks[1] - dis
        n_peaks = int(peaks[0] / dis)
        for p in range(0, n_peaks):
            peaks = np.insert(peaks, 0, peaks[0] - dis)
        if peaks[0] > 0:
            peaks = np.insert(peaks, 0, 0)
    elif status == 1:
        peaks[len(peaks) - 4] = peaks[len(peaks) - 5] + dis
        peaks[len(peaks) - 3] = peaks[len(peaks) - 4] + dis
        peaks[len(peaks) - 2] = peaks[len(peaks) - 3] + dis
        peaks[len(peaks) - 1] = peaks[len(peaks) - 2] + dis
        n_peaks = int((se - 1 - peaks[len(peaks) - 1]) / dis)
        for p in range(0, n_peaks):
            peaks = np.append(peaks, peaks[len(peaks) - 1] + dis)
        if peaks[len(peaks) - 1] < se - 1:
            peaks = np.append(peaks, se - 1)
    return peaks


def bezier_curve(p0, p1, p2, p3, inserted):

    if isinstance(p0, (tuple, list)):
        p0 = np.array(p0)
    if isinstance(p1, (tuple, list)):
        p1 = np.array(p1)
    if isinstance(p2, (tuple, list)):
        p2 = np.array(p2)
    if isinstance(p3, (tuple, list)):
        p3 = np.array(p3)

    points = list()
    for t in np.linspace(0, 1, inserted + 2):
        points.append(p0 * np.power((1 - t), 3) + 3 * p1 * t * np.power((1 - t), 2) + 3 * p2 * (1 - t) * np.power(t,
                                                                                                                  2) + p3 * np.power(
            t, 3))

    return np.vstack(points)

def smoothing_base_bezier(date_x, date_y, k=0.5, inserted=10, closed=False):

    if isinstance(date_x, list) and isinstance(date_y, list):
        date_x = np.array(date_x)
        date_y = np.array(date_y)
    elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
        assert date_x.shape == date_y.shape, u'len(x)!=len(y)'
    else:
        raise Exception(u'The type of the x dataset or the y dataset is incorrect.')


    mid_points = list()
    for i in range(1, date_x.shape[0]):
        mid_points.append({
            'start': (date_x[i - 1], date_y[i - 1]),
            'end': (date_x[i], date_y[i]),
            'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
        })

    if closed:
        mid_points.append({
            'start': (date_x[-1], date_y[-1]),
            'end': (date_x[0], date_y[0]),
            'mid': ((date_x[0] + date_x[-1]) / 2.0, (date_y[0] + date_y[-1]) / 2.0)
        })


    split_points = list()
    for i in range(len(mid_points)):
        if i < (len(mid_points) - 1):
            j = i + 1
        elif closed:
            j = 0
        else:
            continue

        x00, y00 = mid_points[i]['start']
        x01, y01 = mid_points[i]['end']
        x10, y10 = mid_points[j]['start']
        x11, y11 = mid_points[j]['end']
        d0 = np.sqrt(np.power((x00 - x01), 2) + np.power((y00 - y01), 2))
        d1 = np.sqrt(np.power((x10 - x11), 2) + np.power((y10 - y11), 2))
        k_split = 1.0 * d0 / (d0 + d1)

        mx0, my0 = mid_points[i]['mid']
        mx1, my1 = mid_points[j]['mid']

        split_points.append({
            'start': (mx0, my0),
            'end': (mx1, my1),
            'split': (mx0 + (mx1 - mx0) * k_split, my0 + (my1 - my0) * k_split)
        })

    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end']
        dx = vx - split_points[i]['split'][0]
        dy = vy - split_points[i]['split'][1]

        sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy
        ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy

        cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k
        cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k

        if crt_points:
            crt_points[-1].insert(2, cp0)
        else:
            crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

        if closed:
            if i < (len(mid_points) - 1):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points[0].insert(1, cp1)
        else:
            if i < (len(mid_points) - 2):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                crt_points[0].insert(1, mid_points[0]['start'])

    out = list()
    for item in crt_points:
        inserted = item[3][0] - item[0][0] - 1
        group = bezier_curve(item[0], item[1], item[2], item[3], inserted)
        out.append(group[:-1])

    out.append(group[-1:])
    out = np.vstack(out)

    return out.T[0], out.T[1]

def crop_brain(img,LR=None):
    if not LR is None:
        fMOST_PI_CONFIG['LR']=LR
        fMOST_PI_CONFIG['wholeBrain']=False
    if not fMOST_PI_CONFIG['wholeBrain']:
        if fMOST_PI_CONFIG['LR'] == 'L':
            img[int(img.shape[0] / 2):img.shape[0], 0:img.shape[1], 0:img.shape[2]] = 0
        elif fMOST_PI_CONFIG['LR'] == 'R':
            img[0:int(img.shape[0] / 2), 0:img.shape[1], 0:img.shape[2]] = 0
        elif fMOST_PI_CONFIG['LR'] == 'LR':
            img=img
        return img
    else:
        return img

def log(base, x):
    return np.log(x, out=np.zeros_like(x)) / np.log(base, out=np.zeros_like(x))


def reset_img(imglist,fix=None):
    imglist_=[]
    for img in imglist:
        img_=ants.from_numpy(img.numpy())
        if fix is not None:
            img_=ants.copy_image_info(fix,img_)
        imglist_.append(img_)
    return imglist_
    # return imglist


def atlas_reg_ByT1w():
    method = 'Method A (MIw)'
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/')
    t1 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_wm.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')
    mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')
    pi = ants.mask_image(pi, mask)
    tsfer = ants.mask_image(tsfer, mask)
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    tmp_mask = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_brainmask.nii.gz')
    tmp_origin=ants.mask_image(tmp_origin, tmp_mask)
    atlas2 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    # atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    # atlas4 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    atlas=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2_=ants.mask_image(atlas2,atlas2,5)
    atlas2_data=atlas2_.numpy()
    atlas2_data[atlas2_data>0]=1
    atlas2_[:,:,:]=atlas2_data
    tmp_origin=tmp_origin-ants.mask_image(tmp_origin, atlas2_)
    tmp_origin.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/NMT.nii.gz')
    atlas.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/D99.nii.gz')
    tmp_origin = crop_brain(tmp_origin)
    atlas=crop_brain(atlas)
    atlas2=crop_brain(atlas2)
    atlas4 = crop_brain(atlas4)
    t1,tsfer,pi,tmp,atlas,atlas2,atlas4=reset_img([t1,tsfer,pi,tmp_origin,atlas,atlas2,atlas4])
    print('Reg iter1: T1like in MRI <--> NMT')
    start_time = time.time()
    # syn_sampling 4  total_sigma 0.5 reg_iterations=(2400,1200,40)
    tf1 = ants.registration(t1,tmp, 'SyN',syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.1,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_NMTtoT1w_',verbose=False)
    img__ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1w.nii.gz')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline' )
    print('Reg iter2: T1like <--> MRI')
    # syn_sampling 4  total_sigma 0.7 reg_iterations=(1200,1200,40)
    tf3 = ants.registration(t1,tsfer, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg_iterations=(1200,1200,40),flow_sigma=3,total_sigma=0.1,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_PItoT1w_',verbose=False)
    img__ = ants.copy_image_info(tmp_origin, tf3['warpedmovout'])
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inT1w.nii.gz')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf3['fwdtransforms'], 'bSpline')
    print('Reg iter3: T1like_ <--> NMT_')
    # syn_sampling 2  total_sigma 0.7 reg_iterations=(2400,1200,40)
    tf2 = ants.registration(tsfer_,tmp_, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.2,
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.1,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_T1toGFP_',verbose=False)
    end_time = time.time()
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, tmp_)
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')
    ####################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Charm4_inT1w.nii.gz')
    atlas_ = ants.apply_transforms(tsfer_, atlas_, tf2['fwdtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf3['invtransforms'], 'multiLabel')
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Charm4_inPI.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Seg_inT1w.nii.gz')
    atlas2_ = ants.apply_transforms(tsfer_, atlas2_, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf3['invtransforms'], 'multiLabel')
    atlas2_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Seg_inPI.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf1['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inT1w.nii.gz')
    atlas4_ = ants.apply_transforms(tsfer_, atlas4_, tf2['fwdtransforms'], 'multiLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf3['invtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inPI.nii.gz')

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1w_inNMT.nii.gz')

    img_ = ants.apply_transforms(t1, tsfer, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inT1w.nii.gz')
    img_ = ants.apply_transforms(t1, img_, tf2['invtransforms'], 'bSpline')
    img_ = ants.apply_transforms(tmp, img_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, img_)
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inNMT.nii.gz')

    pi_ = ants.apply_transforms(t1, pi, tf3['fwdtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, ants.image_clone(pi_))
    pi__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inT1w.nii.gz')
    pi_ = ants.apply_transforms(t1, pi_, tf2['invtransforms'], 'bSpline')
    pi_ = ants.apply_transforms(tmp, pi_, tf1['invtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, pi_)
    pi__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

def atlas_reg_ByT1w_s(method='Method A (MIw)_Lesion',p=0.1,LR=None):
    print('atlas_reg_ByT1w_s')
    # fMOST_PI_CONFIG['output_dir']=subject
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/')
    t1 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_wm.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')
    mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')
    mask_data=mask.numpy().copy()
    les=int(mask_data.shape[1] * p)
    mask_data[:,0:les,:]=0
    mask_data[:, mask_data.shape[1]-les:mask_data.shape[1], :] = 0
    mask[:,:,:]=mask_data
    pi = ants.mask_image(pi, mask)
    tsfer = ants.mask_image(tsfer, mask)
    pi.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_Lesion_'+str(p)+'.nii.gz')
    tsfer.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/T1PI_Lesion'+str(p)+'.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    tmp_mask = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_brainmask.nii.gz')
    tmp_origin=ants.mask_image(tmp_origin, tmp_mask)
    atlas2 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    # atlas = ants.image_read('template/NMT/NMT_brain/CHARM_6_in_NMT_v2.0_sym_D99.nii.gz')
    # atlas4 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    atlas=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas2_=ants.mask_image(atlas2,atlas2,5)
    atlas2_data=atlas2_.numpy()
    atlas2_data[atlas2_data>0]=1
    atlas2_[:,:,:]=atlas2_data
    tmp_origin=tmp_origin-ants.mask_image(tmp_origin, atlas2_)
    tmp_origin.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/NMT.nii.gz')
    atlas.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/D99.nii.gz')
    t1 = crop_brain(t1,LR)
    tmp_origin = crop_brain(tmp_origin,LR)
    atlas=crop_brain(atlas,LR)
    atlas2=crop_brain(atlas2,LR)
    atlas4 = crop_brain(atlas4,LR)
    t1,tsfer,pi,tmp,atlas,atlas2,atlas4,mask=reset_img([t1,tsfer,pi,tmp_origin,atlas,atlas2,atlas4,mask])
    print('Reg iter1: T1like in MRI <--> NMT')
    start_time = time.time()
    # syn_sampling 4  total_sigma 0.5 reg_iterations=(2400,1200,40)
    tf1 = ants.registration(t1,tmp, 'SyN',syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg_iterations=(400,200,20),flow_sigma=3,total_sigma=0.1,singleprecision=True,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_NMTtoT1w_',verbose=False)
    img__ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1w.nii.gz')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline' )
    print('Reg iter2: T1like <--> MRI')
    # syn_sampling 4  total_sigma 0.7 reg_iterations=(1200,1200,40)
    if p>=0.3:
        type_reg='Similarity'
    elif p >= 0.2:
        type_reg = 'Affine'
    else:
        type_reg='SyN'
    print(type_reg)
    tf_mask = ants.registration(t1,tsfer, type_reg,aff_metric='GC',total_sigma=5,flow_sigma=5,
                            reg_iterations=(40,20,0),outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/mask_PItoT1w_Affine_',verbose=False)
    mask_ = ants.apply_transforms(t1, mask, tf_mask['fwdtransforms'], 'multiLabel')
    mask_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/mask_affinetoT1w.nii.gz')
    t1_=ants.mask_image(t1, mask_)
    t1_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1w.nii.gz')
    #
    tf3 = ants.registration(t1_,tsfer, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.3,aff_metric='GC',
                            reg_iterations=(400,200,20),flow_sigma=3,total_sigma=0.7,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_PItoT1w_',verbose=False)

    tsfer_ = ants.apply_transforms(t1_,tsfer, tf3['fwdtransforms'],whichtoinvert=[False,False], interpolator='bSpline')
    mask_ = ants.apply_transforms(t1_, mask, tf3['fwdtransforms'], 'multiLabel')
    tf3['warpedmovout'].to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/T1PI_inT1w2.nii.gz')
    tsfer_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/T1PI_inT1w.nii.gz')
    mask_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/mask_affinetoT1w2.nii.gz')

    print('Reg iter3: T1like_ <--> NMT_')
    tmp_=ants.mask_image(tmp_, mask_)
    ## syn_sampling 2  total_sigma 0.7 reg_iterations=(2400,1200,40)
    tf2 = ants.registration(tsfer_,tmp_, 'SyN',
                            syn_metric='mattes',syn_sampling=32,grad_step=0.2,
                            reg_iterations=(400,200,20),flow_sigma=3,total_sigma=0.1,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_NMTtoPIinT1w_',verbose=False)
    end_time = time.time()
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, tmp_)
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')
    ###################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Charm4_inT1w.nii.gz')
    atlas_ = ants.apply_transforms(tsfer_, atlas_, tf2['fwdtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf3['invtransforms'], 'multiLabel')
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Charm4_inPI.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Seg_inT1w.nii.gz')
    atlas2_ = ants.apply_transforms(tsfer_, atlas2_, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf3['invtransforms'], 'multiLabel')
    atlas2_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/Seg_inPI.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf1['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inT1w.nii.gz')
    atlas4_ = ants.apply_transforms(tsfer_, atlas4_, tf2['fwdtransforms'], 'multiLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf3['invtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inPI.nii.gz')

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1w_inNMT.nii.gz')

    img_ = ants.apply_transforms(t1, tsfer, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inT1w.nii.gz')
    img_ = ants.apply_transforms(t1, img_, tf2['invtransforms'], 'bSpline')
    img_ = ants.apply_transforms(tmp, img_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, img_)
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inNMT.nii.gz')

    pi_ = ants.apply_transforms(t1, pi, tf3['fwdtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, ants.image_clone(pi_))
    pi__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inT1w.nii.gz')
    pi_ = ants.apply_transforms(t1, pi_, tf2['invtransforms'], 'bSpline')
    pi_ = ants.apply_transforms(tmp, pi_, tf1['invtransforms'], 'bSpline')
    pi__ = ants.copy_image_info(tmp_origin, pi_)
    pi__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

def atlas_reg_ByT1w_v2():
    t1 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI.nii.gz')
    # tsfer = ants.image_read(fMOST_GFP_CONFIG['output_dir'] + '/reg/T2likeGFP.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/CHARM_1_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/SARM_6_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/SARM_2_in_NMT_v2.0_sym.nii.gz')
    tmp_origin = crop_brain(tmp_origin)
    atlas=crop_brain(atlas)
    atlas1 = crop_brain(atlas1)
    atlas2 = crop_brain(atlas2)
    atlas3 = crop_brain(atlas3)
    t1,tsfer,pi,tmp,atlas,atlas1,atlas2,atlas3=reset_img([t1,tsfer,pi,tmp_origin,atlas,atlas1,atlas2,atlas3])
    print('Reg iter1: T1like <--> MRI')
    start_time = time.time()
    tf1 = ants.registration(t1,tsfer, 'SyN',
                            syn_metric='meansquares',
                            reg_iterations=(2100,1200,1200,20),flow_sigma=3,total_sigma=1,syn_sampling=4,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/xfms/atlas_PtoT1w_v2_')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf1['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tsfer_))
    img__.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/atlas/T1PI_inT1w.nii.gz')
    print('Reg iter2: T1like in MRI <--> NMT')
    tf2 = ants.registration(tmp,tsfer_, 'SyN',
                            syn_metric='meansquares',
                            reg_iterations=(2100,1200,1200,20),flow_sigma=3,total_sigma=0.15,syn_sampling=4,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/xfms/atlas_PtoNMT_v2_')
    end_time = time.time()
    tsfer_ = ants.apply_transforms(tmp,tsfer_, tf2['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tsfer_))
    img__.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/atlas/T1PI_inNMT.nii.gz')
    #################################
    atlas_ = ants.apply_transforms(tsfer_, atlas, tf2['invtransforms'], 'multiLabel')
    img_ = ants.copy_image_info(tmp_origin, atlas_.clone())
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/D99_inMRI.nii.gz')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf1['invtransforms'], 'multiLabel')
    atlas_=ants.copy_image_info(tmp_origin,atlas_)
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/D99_inPI.nii.gz')

    atlas1_ = ants.apply_transforms(t1, atlas1, tf2['invtransforms'], 'multiLabel')
    atlas1_ = ants.apply_transforms(tsfer, atlas1_, tf1['invtransforms'], 'multiLabel')
    atlas1_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/CHARM1_inPI.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf2['invtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf1['invtransforms'], 'multiLabel')
    atlas2_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/SARM6_inPI.nii.gz')

    atlas3_ = ants.apply_transforms(t1, atlas3, tf2['invtransforms'], 'multiLabel')
    img_ = ants.copy_image_info(tmp_origin, atlas3_.clone())
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/SARM2_inMRI.nii.gz')
    atlas3_ = ants.apply_transforms(tsfer, atlas3_, tf1['invtransforms'], 'multiLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/SARM2_inPI.nii.gz')

    tmp_ = ants.apply_transforms(tsfer_, tmp, tf2['invtransforms'], 'multiLabel')
    img_ = ants.copy_image_info(tmp_origin, tmp_.clone())
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/TMP_inMRI.nii.gz')
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf1['invtransforms'], 'multiLabel')
    tmp_=ants.copy_image_info(tmp_origin,tmp_)
    tmp_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/TMP_inT1PI.nii.gz')

    pi_ = ants.apply_transforms(tmp, pi, tf1['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(pi_))
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_inMRI.nii.gz')
    pi_ = ants.apply_transforms(tmp, pi_, tf2['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(pi_))
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")


def atlas_reg_noT1w():
    print('no MRI')
    method='Method D (CC) withOriginCycle_mattes'
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/')
    # tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_c_aug.nii.gz')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI_origin.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')
    tmp_ = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    # tmp_ = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/MRI/MRI_brain_bc_dn_.nii.gz')
    # tmp_=ants.image_read('/media/zzb/Raid2_block2/macaque/PI/cy_v1_0.25mm.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas3=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    tmp_ = crop_brain(tmp_)
    atlas=crop_brain(atlas)
    atlas3 = crop_brain(atlas3)
    atlas4 = crop_brain(atlas4)
    tsfer, pi, tmp, atlas, atlas3, atlas4 = reset_img([tsfer, pi, tmp_, atlas, atlas3, atlas4])
    start_time = time.time()
    tf = ants.registration(tsfer,tmp, 'SyN',syn_metric='mattes',syn_sampling=32,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_PItoNMT_',
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0.1)
    end_time = time.time()
    tf['warpedmovout'].to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')
    ####################################################################
    atlas_ = ants.apply_transforms(pi, atlas, tf['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/D99_inPI.nii.gz')
    atlas3_ = ants.apply_transforms(pi, atlas3, tf['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/CHARM4_inPI.nii.gz')
    atlas4_ = ants.apply_transforms(pi, atlas4, tf['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inPI.nii.gz')
    img_ = ants.apply_transforms(tmp, pi, tf['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_, img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inNMT.nii.gz')
    tsfer_ = ants.apply_transforms(tmp, tsfer, tf['invtransforms'], 'bSpline')
    tsfer_ = ants.copy_image_info(tmp_, tsfer_)
    tsfer_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

def atlas_reg_noT1w_s(subject,method='Method A (CC)_Lesion',p=0,LR=None):
    print('no MRI')
    fMOST_PI_CONFIG['output_dir']=subject
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI_c.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')
    mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')
    mask_data=mask.numpy().copy()
    les=int(mask_data.shape[1] * p)
    mask_data[:,0:les,:]=0
    mask_data[:, mask_data.shape[1]-les:mask_data.shape[1], :] = 0
    mask[:,:,:]=mask_data
    pi = ants.mask_image(pi, mask)
    tsfer = ants.mask_image(tsfer, mask)
    pi.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_Lesion_'+str(p)+'.nii.gz')
    tsfer.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/T1PI_Lesion'+str(p)+'.nii.gz')
    tmp_ = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas3=ants.image_read('template/NMT/NMT_brain/level4/CHARM_4_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level4/SARM_4_in_NMT_v2.0_sym.nii.gz')
    tmp_ = crop_brain(tmp_,LR)
    atlas3 = crop_brain(atlas3,LR)
    atlas4 = crop_brain(atlas4,LR)
    tsfer, pi, tmp, atlas3, atlas4 = reset_img([tsfer, pi, tmp_, atlas3, atlas4])
    start_time = time.time()
    if p>=0.4:
        type_reg='Similarity'
    else:
        type_reg='SyN'
    tf_mask = ants.registration(tsfer,tmp, type_of_transform=type_reg,aff_metric='GC',total_sigma=5,flow_sigma=3,
                            reg_iterations=(40,20,0),outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/mask_PItoT1w_Affine_',verbose=False)
    mask_ = ants.apply_transforms(tmp, mask, tf_mask['invtransforms'], 'multiLabel')
    mask_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/mask_affinetoNMT.nii.gz')
    # mask_=mask
    tmp_=ants.mask_image(tmp, mask_)
    tmp_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/NMT.nii.gz')

    tf = ants.registration(tsfer,tmp_, 'SyN',syn_metric='mattes',syn_sampling=32,outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/atlas_PItoNMT_',
                            reg_iterations=(2400,1200,40),flow_sigma=3,total_sigma=0)
    end_time = time.time()
    tf['warpedmovout'].to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')
    ####################################################################
    atlas3_ = ants.apply_transforms(pi, atlas3, tf['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/CHARM4_inPI.nii.gz')
    atlas4_ = ants.apply_transforms(pi, atlas4, tf['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM4_inPI.nii.gz')
    img_ = ants.apply_transforms(tmp, pi, tf['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_, img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inNMT.nii.gz')
    tsfer_ = ants.apply_transforms(tmp, tsfer, tf['invtransforms'], 'bSpline')
    tsfer_ = ants.copy_image_info(tmp_, tsfer_)
    tsfer_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inNMT.nii.gz')
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")

def atlas_reg_noT1w_1():
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_c_.nii.gz')
    # tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/PI_alignNMT.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')
    tmp_ = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/CHARM_1_in_NMT_v2.0_sym.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/SARM_2_in_NMT_v2.0_sym.nii.gz')
    tmp_ = crop_brain(tmp_)
    atlas=crop_brain(atlas)
    atlas3 = crop_brain(atlas3)
    atlas4 = crop_brain(atlas4)
    tsfer, pi, tmp, atlas, atlas3, atlas4 = reset_img([tsfer, pi, tmp_, atlas, atlas3, atlas4])
    for gt in [2,4,8,16]:
        os.makedirs(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt))
        tf = ants.registration(tsfer,tmp, 'SyNCC',syn_metric='CC',outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/xfms/atlas_PItoNMT_',
                                reg_iterations=(40, 20, 0),flow_sigma=3,total_sigma=0,grad_step=0.2,redius=gt)
        tf['warpedmovout'].to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/TMP_inPI.nii.gz')
        ####################################################################
        atlas_ = ants.apply_transforms(pi, atlas, tf['fwdtransforms'], 'multiLabel')
        atlas_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt)+'/D99_inPI.nii.gz')
        atlas3_ = ants.apply_transforms(pi, atlas3, tf['fwdtransforms'], 'multiLabel')
        atlas3_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt)+'/CHARM1_inPI.nii.gz')
        atlas4_ = ants.apply_transforms(pi, atlas4, tf['fwdtransforms'], 'multiLabel')
        atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt)+'/SARM2_inPI.nii.gz')
        img_ = ants.apply_transforms(tmp, pi, tf['invtransforms'], 'bSpline')
        img_=ants.copy_image_info(tmp_, img_)
        img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt)+'/PI_inNMT.nii.gz')
        tsfer_ = ants.apply_transforms(tmp, tsfer, tf['invtransforms'], 'bSpline')
        tsfer_ = ants.copy_image_info(tmp_, tsfer_)
        tsfer_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/CC_atlas'+str(gt)+'/T1PI_inNMT.nii.gz')


def touint8(image):
    image[image<0]=0
    image=(image - image.min()) / (image.max() - image.min()+1e-10) * 255
    image=image.astype(np.uint8)
    return image

def transpose_cellsxyz(transpose,x_coords,y_coords,z_coords):
    if transpose == [1, 0, 2]:
        adjusted_x = y_coords
        adjusted_y = x_coords
        adjusted_z = z_coords
    elif transpose == [0, 1, 2]:
        adjusted_x = x_coords
        adjusted_y = y_coords
        adjusted_z = z_coords
    elif transpose == [0, 2, 1]:
        adjusted_x = x_coords
        adjusted_y = z_coords
        adjusted_z = y_coords
    elif transpose == [1, 2, 0]:
        adjusted_x = y_coords
        adjusted_y = z_coords
        adjusted_z = x_coords
    elif transpose == [2, 0, 1]:
        adjusted_x = z_coords
        adjusted_y = x_coords
        adjusted_z = y_coords
    elif transpose == [2, 1, 0]:
        adjusted_x = z_coords
        adjusted_y = y_coords
        adjusted_z = x_coords
    else:
        adjusted_x = x_coords
        adjusted_y = y_coords
        adjusted_z = z_coords
    return adjusted_x,adjusted_y,adjusted_z

def flip_cellsxyz(flip,x_coords,y_coords,z_coords,dim1,dim2,dim3):
    if flip[0] == 1:
        x_coords = dim1-1-x_coords
    if flip[1] == 1:
        y_coords = dim2-1-y_coords
    if flip[2] == 1:
        z_coords = dim3 - 1 - z_coords
    return x_coords,y_coords,z_coords