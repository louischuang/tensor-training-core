from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tensor_training_core.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from tensor_training_core.utils.paths import resolve_repo_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in config file: {resolved}")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return DatasetConfig.model_validate(load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))
