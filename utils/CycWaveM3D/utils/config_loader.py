import argparse
import os
import torch
import yaml


def load_config(config_path="config/config.yaml", mode="train"):
    """
    Load unified YAML config and return a Namespace object for attribute access.

    :param config_path: Path to the yaml config file.
    :param mode: 'train' or 'test'.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Merge common settings with mode-specific settings
    merged_cfg = {}
    if "common" in cfg:
        merged_cfg.update(cfg["common"])
    if mode in cfg:
        merged_cfg.update(cfg[mode])

    opt = argparse.Namespace(**merged_cfg)

    # 2. Handle customized suffix
    if opt.suffix:
        suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
        opt.name = opt.name + suffix

    # 3. Setup GPU devices
    if hasattr(opt, "gpu_ids") and isinstance(opt.gpu_ids, list):
        if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] >= 0:
            torch.cuda.set_device(opt.gpu_ids[0])

    # 4. Save options to checkpoints directory (similar to CycleGAN print_options)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, f"opt_{mode}.txt")
    with open(file_name, "wt") as opt_file:
        for k, v in sorted(vars(opt).items()):
            opt_file.write(f"{str(k):>25}: {str(v):<30}\n")

    return opt