import argparse
import yaml


def load_config(config_path="config/train_config.yaml"):
    """Load YAML config and return a Namespace object for attribute access."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Convert dict to Namespace so you can still use `opt.lr`, `opt.phase`, etc.
    opt = argparse.Namespace(**config_dict)
    return opt