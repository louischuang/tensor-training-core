from __future__ import annotations

from tensor_training_core.config.loader import load_dataset_config, load_experiment_config, load_model_config


def test_load_experiment_config() -> None:
    config = load_experiment_config("configs/experiments/dev_macos.yaml")
    assert config.runtime.target == "macos"
    assert config.runtime.experiment_id == "dev_macos_baseline"


def test_load_dataset_config_with_split_settings() -> None:
    config = load_dataset_config("configs/datasets/esp32_cam_train.yaml")
    assert config.dataset.split is not None
    assert config.dataset.split.train_manifest_output.endswith("_split_train.jsonl")
    assert config.dataset.split.val_manifest_output.endswith("_split_val.jsonl")


def test_load_model_config_with_postprocess_settings() -> None:
    config = load_model_config("configs/models/ssd_mobilenet_v2_fpnlite_320.yaml")
    assert config.model.max_detections == 8
    assert len(config.model.anchors) == 8
    assert config.model.score_threshold == 0.20


def test_load_training_config_with_augmentation_settings() -> None:
    from tensor_training_core.config.loader import load_training_config

    config = load_training_config("configs/training/tensorflow_full.yaml")
    assert config.training.augmentation.enabled is True
    assert config.training.augmentation.horizontal_flip_prob == 0.5
    assert config.training.batch_size == 16
