#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import ast
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


@torch.no_grad()
def blockface_inference(predictor, ptsList, ptsTypeList,video_dir,result):

    """
    :param predictor: a predictor build by build_sam2_video_predictor.
    :param ptsList: the point list.(e.g.[obj1,obj2,...],and obj1 like [[502, 602],[577,618]])
    :param ptsTypeList: the points types list,which must match the length of the points list.
                         for labels, `1` means positive click and `0` means negative click(e.g.[1,1])
    :param video_dir: the blockface folder.
    :return: return a dictionary,like {frame0:{obj_id:mask},frame1:{obj_id:mask}...}
    """
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names.sort(key=lambda p: os.path.splitext(p)[0])
    # take a look the first video frame
    frame_idx = len(frame_names)-10
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    if result:plt.show()

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = len(frame_names)-10  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


    points = np.array(np.array(ptsList), dtype=np.float32)

    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(np.array(ptsTypeList), np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))


    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    if result:plt.show()

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=0):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            if result:plt.show()
    return video_segments

# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on blockface segmentation "
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="/media/zzb/Raid2_block2/Rabbit/blockface/preroc/",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    default="/media/zzb/Raid2_block2/Rabbit/blockface/mask/",
    type=str,
    help="path to the segmentation folder",
)
parser.add_argument(
    "--pts",
    type=str,
    default='[[750, 750],[750,750]]',
    help="points list of the segmentation target",
)
parser.add_argument(
    "--ptst",
    type=str,
    default='[1,1]',
    help="points' types list of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="/home/zzb/PycharmProjects/Macaca-Star/checkpoints/Blockface_brain_ex/chk_blockface_b+.pt",
    help="path to the trained model",
)
parser.add_argument(
    "-cfg",
    "--model_cfg",
    type=str,
    default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    help="path to the trained model config file",
)
parser.add_argument(
    "-s",
    "--show_result",
    type=bool,
    default=True,
    help="Whether to display the results using `plt` (Matplotlib)",
)
args = parser.parse_args()

predictor = build_sam2_video_predictor(
    config_file=args.model_cfg,
    ckpt_path=args.checkpoint,
    device=args.device
)

dict_blockface_mask=blockface_inference(
    predictor,
    ast.literal_eval(args.pts),
    ast.literal_eval(args.ptst),
    args.data_path,
    args.show_result
)


#Mask Save
frame_names = [
    p for p in os.listdir(args.data_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
frame_names.sort(key=lambda p: os.path.splitext(p)[0])

if args.seg_path is not None:
    for out_frame_idx in range(0, len(frame_names)):
        random_color = False
        for out_obj_id, out_mask in dict_blockface_mask[out_frame_idx].items():
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if out_obj_id is None else out_obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
            h, w = out_mask.shape[-2:]
            mask_image = out_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            out_dir=os.path.join("work_dirs",args.seg_path)
            if not os.path.exists(out_dir) :os.makedirs(out_dir)
            outPutPath=os.path.join(out_dir,f"{frame_names[out_frame_idx]}")
            mask_image_uint8 = (mask_image * 255).astype(np.uint8)
            Image.fromarray(mask_image_uint8).save(outPutPath)