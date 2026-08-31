#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：util.py
@Author  ：Zauber
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
    """
    Perform MRI-guided 3D non-linear registration (SyN) between standard NMT template,
    subject-specific in vivo T1w MRI, and synthetic T1-like PI data.

    Workflow:
      1. Prepare output directories and load/mask images and atlases.
      2. Reg Iter 1: Register NMT template to in vivo T1w MRI space.
      3. Reg Iter 2: Register synthetic T1-like PI to in vivo T1w MRI space.
      4. Reg Iter 3: Refine non-linear deformation between warped NMT and warped T1-like PI.
      5. Invert and compose transforms to map template/atlas into native PI space.
    """
    method = ''
    # 2. Hierarchical atlas level setting
    atlas_level=6    # Hierarchical parcellation level for CHARM (cortical) and SARM (subcortical) atlases

    # Ensure required output subdirectories exist (atlas/ and xfms/)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method)
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/atlas/')
    if not os.path.exists(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/'):
        os.mkdir(fMOST_PI_CONFIG['output_dir']+'/reg/'+method+'/xfms/')

    # Load subject-specific in vivo MRI, synthetic T1-like PI, and PI volumes
    t1 = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/MRI/MRI_brain_bc_dn_.nii.gz')
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_c.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')

    # Check if the pre-aligned brain foreground mask exists
    if os.path.exists(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz'):
        # Load the brain foreground mask
        mask = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/atlas/PI_alignNMT_mask.nii.gz')
        # Apply mask to optical PI and synthetic T1-like volumes to remove non-brain background
        pi = ants.mask_image(pi, mask)
        tsfer = ants.mask_image(tsfer, mask)

    # Load standard NMT template and corresponding hierarchical atlases (CHARM & SARM)
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    tmp_mask = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_brainmask.nii.gz')
    tmp_origin=ants.mask_image(tmp_origin, tmp_mask)

    atlas2 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    # Load CHARM (Cortical Hierarchy Atlas of the Rhesus Macaque) at specified level
    atlas=ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/CHARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')
    # Load SARM (Subcortical Atlas of the Rhesus Macaque) at specified level
    atlas4=ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/SARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')

    # Extract and remove vasculature (segmentation label 5) from the template
    atlas2_=ants.mask_image(atlas2,atlas2,5)
    atlas2_data=atlas2_.numpy()
    atlas2_data[atlas2_data>0]=1
    atlas2_[:,:,:]=atlas2_data
    tmp_origin=tmp_origin-ants.mask_image(tmp_origin, atlas2_)

    # Crop to target hemisphere (Left / Right) or whole brain, and standardize spatial origins
    tmp_origin = crop_brain(tmp_origin)
    atlas=crop_brain(atlas)
    atlas2=crop_brain(atlas2)
    atlas4 = crop_brain(atlas4)
    t1,tsfer,pi,tmp,atlas,atlas2,atlas4=reset_img([t1,tsfer,pi,tmp_origin,atlas,atlas2,atlas4])

    # =========================================================================
    # Reg Iter 1: Deformable registration from NMT Template -> In vivo T1w MRI
    # =========================================================================
    print('Reg iter1: T1like in MRI <--> NMT')
    start_time = time.time()
    tf1 = ants.registration(
        fixed=t1,
        moving=tmp,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        grad_step=0.2,
        aff_metric='GC',
        reg_iterations=(1200, 1200, 40),
        flow_sigma=3,
        total_sigma=0.5,
        outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/xfms/atlas_NMTtoT1w_',
        verbose=False
    )
    # Save NMT template warped to T1w MRI space
    img__ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas/TMP_inT1w.nii.gz')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline' )

    # =========================================================================
    # Reg Iter 2: Deformable registration from Synthetic T1-like PI -> In vivo T1w MRI
    # =========================================================================
    print('Reg iter2: T1like <--> MRI')
    tf3 = ants.registration(
        fixed=t1,
        moving=tsfer,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        grad_step=0.2,
        aff_metric='GC',
        reg_iterations=(1200, 1200, 40),
        flow_sigma=3,
        total_sigma=0.5,
        outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/xfms/atlas_PItoT1w_',
        verbose=False
    )
    # Save synthetic T1-like PI warped to T1w MRI space
    img__ = ants.copy_image_info(tmp_origin, tf3['warpedmovout'])
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inT1w.nii.gz')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf3['fwdtransforms'], 'bSpline')

    # =========================================================================
    # Reg Iter 3: Non-linear refinement between warped T1-like PI and warped NMT
    # =========================================================================
    print('Reg iter3: T1like_ <--> NMT_')
    tf2 = ants.registration(
        fixed=tsfer_,
        moving=tmp_,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        grad_step=0.2,
        reg_iterations=(1200, 1200, 40),
        flow_sigma=3,
        total_sigma=0.5,
        outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/xfms/atlas_T1toGFP_',
        verbose=False
    )
    end_time = time.time()

    # 5. Invert and compose transforms to map NMT template back into native T1-like PI space
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, tmp_)
    img__.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')

    # =========================================================================
    # Transform anatomical atlases into T1w MRI and PI spaces
    # =========================================================================

    # Define a key-value mapping of atlas names to their corresponding image objects
    atlases_dict = {
        f"CHARM{atlas_level}": atlas,  # CHARM cortical atlas
        "Seg": atlas2,  # NMT tissue segmentation atlas
        f"SARM{atlas_level}": atlas4,  # SARM subcortical atlas
    }

    # Base directory for saving output atlases
    atlas_save_dir = (fMOST_PI_CONFIG["output_dir"] + "/reg/" + method + "/atlas")

    # Loop through each atlas to apply transforms and export results
    for name, cur_atlas in atlases_dict.items():
        # Step 1: Transform to T1w MRI space and save
        atlas_in_t1w = ants.apply_transforms(t1, cur_atlas, tf1["fwdtransforms"], "multiLabel")
        atlas_in_t1w.to_file(f"{atlas_save_dir}/{name}_inT1w.nii.gz")

        # Step 2: Transform through intermediate space to native PI space and save
        atlas_in_tsfer = ants.apply_transforms(tsfer_, atlas_in_t1w, tf2["fwdtransforms"], "multiLabel")
        atlas_in_pi = ants.apply_transforms(tsfer, atlas_in_tsfer, tf3["invtransforms"], "multiLabel")
        atlas_in_pi.to_file(f"{atlas_save_dir}/{name}_inPI.nii.gz")

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1w_inNMT.nii.gz')

    # =========================================================================
    # 6. Transform fMOST image volumes into coordinate spaces
    # =========================================================================

    # Define a key-value mapping of fMOST images to their corresponding image objects
    fMOST_images_dict = {
        'T1PI': tsfer,  # Synthetic T1-like PI volume
        'PI': pi  # Pre-aligned PI fluorescence volume
    }

    # Base directory for saving output images
    atlas_save_dir = fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/atlas'

    # Loop through each intensity volume to apply transforms and export results
    for name, cur_img in fMOST_images_dict.items():
        # Step 1: Warp image into T1w MRI space and save
        img_in_t1w = ants.apply_transforms(t1, cur_img, tf3['fwdtransforms'], 'bSpline')
        img_in_t1w_out = ants.copy_image_info(tmp_origin, img_in_t1w)
        img_in_t1w_out.to_file(f"{atlas_save_dir}/{name}_inT1w.nii.gz")

        # Step 2: Warp image further into standard NMT template space and save
        img_in_nmt = ants.apply_transforms(t1, img_in_t1w, tf2['invtransforms'], 'bSpline')
        img_in_nmt = ants.apply_transforms(tmp, img_in_nmt, tf1['invtransforms'], 'bSpline')
        img_in_nmt_out = ants.copy_image_info(tmp_origin, img_in_nmt)
        img_in_nmt_out.to_file(f"{atlas_save_dir}/{name}_inNMT.nii.gz")

    # =========================================================================
    # 7. Registration runtime logging
    # =========================================================================
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")


def atlas_reg_noT1w():
    """
    Perform MRI-free 3D non-linear direct registration (SyN) between the synthetic
    T1-like PI volume and the standard NMT template when subject in vivo MRI is unavailable.

    Workflow:
      1. Prepare output directories for registration outputs and transformation matrices.
      2. Load synthetic T1-like PI, PI fluorescence volume, NMT template, and atlases (D99, CHARM, SARM).
      3. Crop to target hemisphere / whole brain and standardize spatial coordinates.
      4. Perform direct deformable registration (SyN with Mattes mutual information) from NMT to synthetic T1.
      5. Transform anatomical atlases into native PI space (multiLabel interpolation).
      6. Warp optical intensity volumes into standard NMT template space (bSpline interpolation).
    """
    print('no MRI')
    method = ''
    # 2. Hierarchical atlas level setting
    atlas_level = 6  # Hierarchical parcellation level for CHARM (cortical) and SARM (subcortical) atlases

    # 1. Ensure required output directories exist (atlas/ and xfms/)
    output_reg_dir = os.path.join(fMOST_PI_CONFIG['output_dir'], 'reg', method)
    os.makedirs(os.path.join(output_reg_dir, 'atlas'), exist_ok=True)
    os.makedirs(os.path.join(output_reg_dir, 'xfms'), exist_ok=True)

    # 2. Load synthetic T1-like PI and pre-aligned PI fluorescence volumes
    tsfer = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI_c.nii.gz')
    pi=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT_.nii.gz')

    # 3. Load standard NMT symmetric template brain
    tmp_ = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')

    # 4. Load anatomical parcellation atlases (D99 cortex, CHARM level 4, SARM level 4)
    atlas3=ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/CHARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')
    atlas4=ants.image_read('template/NMT/NMT_brain/level'+str(atlas_level)+'/SARM_'+str(atlas_level)+'_in_NMT_v2.0_sym.nii.gz')

    # 5. Crop to target hemisphere (Left / Right) or whole brain, and standardize spatial origins
    tmp_ = crop_brain(tmp_)
    atlas3 = crop_brain(atlas3)
    atlas4 = crop_brain(atlas4)
    tsfer, pi, tmp, atlas3, atlas4 = reset_img([tsfer, pi, tmp_, atlas3, atlas4])

    # =========================================================================
    # Direct Deformable Registration: NMT Template -> Synthetic T1-like PI
    # =========================================================================
    start_time = time.time()
    tf = ants.registration(
        fixed=tsfer,
        moving=tmp,
        type_of_transform='SyN',
        syn_metric='CC',
        syn_sampling=4,
        outprefix=fMOST_PI_CONFIG['output_dir'] + '/reg/' + method + '/xfms/atlas_PItoNMT_',
        reg_iterations=(2400, 1200, 40),
        flow_sigma=3,
        total_sigma=0.5
    )
    end_time = time.time()

    # Save NMT template warped to synthetic T1-like PI space
    tf['warpedmovout'].to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/TMP_inT1PI.nii.gz')

    # =========================================================================
    # Transform anatomical atlases into native PI space (multiLabel)
    # =========================================================================
    # Transform and save CHARM cortical atlas
    atlas3_ = ants.apply_transforms(pi, atlas3, tf['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/CHARM'+str(atlas_level)+'_inPI.nii.gz')

    # Transform and save SARM subcortical atlas
    atlas4_ = ants.apply_transforms(pi, atlas4, tf['fwdtransforms'], 'multiLabel')
    atlas4_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/SARM'+str(atlas_level)+'_inPI.nii.gz')

    # =========================================================================
    # Transform fMOST image volumes into standard NMT template space (bSpline)
    # =========================================================================
    # Warp raw PI fluorescence image into NMT space
    img_ = ants.apply_transforms(tmp, pi, tf['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_, img_)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/PI_inNMT.nii.gz')

    # Warp synthetic T1-like PI volume into NMT space
    tsfer_ = ants.apply_transforms(tmp, tsfer, tf['invtransforms'], 'bSpline')
    tsfer_ = ants.copy_image_info(tmp_, tsfer_)
    tsfer_.to_file(fMOST_PI_CONFIG['output_dir'] + '/reg/'+method+'/atlas/T1PI_inNMT.nii.gz')

    # =========================================================================
    # Registration runtime logging
    # =========================================================================
    total_time = end_time - start_time
    print(f"total time：{total_time:.2f}s")


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