from __future__ import annotations

from tensor_training_core.config.loader import load_experiment_config


def test_load_experiment_config() -> None:
    config = load_experiment_config("configs/experiments/dev_macos.yaml")
    assert config.runtime.target == "macos"
    assert config.runtime.experiment_id == "dev_macos_baseline"
