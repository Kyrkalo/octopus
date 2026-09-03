import json

import torch

from octopus.platypus.config import load_configs


def test_load_configs_stamps_device_on_every_model(tmp_path):
    configs_path = tmp_path / "configs.json"
    configs_path.write_text(json.dumps({
        "dataset_root": "../dataset",
        "mnist": {"model_name": "mnist_model"},
        "dcgan": {"model_name": "dcgan_model"},
    }))

    configs = load_configs(str(configs_path))

    assert configs["mnist"]["device"] == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert configs["dcgan"]["device"] == configs["mnist"]["device"]


def test_load_configs_pops_dataset_root_from_result(tmp_path):
    configs_path = tmp_path / "configs.json"
    configs_path.write_text(json.dumps({
        "dataset_root": "../dataset",
        "mnist": {"model_name": "mnist_model"},
    }))

    configs = load_configs(str(configs_path))

    assert "dataset_root" not in configs
    assert set(configs.keys()) == {"mnist"}


def test_load_configs_resolves_dataset_root_placeholder(tmp_path):
    configs_path = tmp_path / "configs.json"
    configs_path.write_text(json.dumps({
        "dataset_root": "../dataset",
        "cnn14_2": {
            "dataroot": "${dataset_root}/cnn14_2sec_dataset",
            "num_classes": 12,
        },
    }))

    configs = load_configs(str(configs_path))

    resolved = configs["cnn14_2"]["dataroot"]
    assert "${dataset_root}" not in resolved
    assert resolved.endswith("cnn14_2sec_dataset")
    # non-string values must be left untouched by the placeholder substitution
    assert configs["cnn14_2"]["num_classes"] == 12
