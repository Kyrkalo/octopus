import json
import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CONFIGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs.json"))


def load_configs(configs_path: str = _DEFAULT_CONFIGS_PATH) -> dict:
    """Load configs.json, resolve ${dataset_root} placeholders, and stamp
    the compute device onto every model's config block."""
    with open(configs_path) as f:
        configs = json.load(f)

    dataset_root = os.path.abspath(os.path.join(_PROJECT_ROOT, configs.pop("dataset_root", ".")))

    for cfg in configs.values():
        cfg["device"] = device
        for k, v in cfg.items():
            if isinstance(v, str):
                cfg[k] = v.replace("${dataset_root}", dataset_root)

    return configs
