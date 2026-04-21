from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ConfigReference(BaseModel):
    config_path: str


class RuntimeConfig(BaseModel):
    target: str
    dataset_version: str
    experiment_id: str


class ExperimentConfig(BaseModel):
    dataset: ConfigReference
    model: ConfigReference
    training: ConfigReference
    runtime: RuntimeConfig


class DatasetSettings(BaseModel):
    format: str
    dataset_root: str
    annotations: str
    images_dir: str
    label_map_output: str
    manifest_output: str
    metadata_output: str


class DatasetConfig(BaseModel):
    name: str
    dataset: DatasetSettings


class AnchorSpec(BaseModel):
    cx: float
    cy: float
    w: float
    h: float


class ModelSettings(BaseModel):
    name: str
    family: str
    image_size: list[int]
    num_classes: int
    max_detections: int = 5
    anchors: list[AnchorSpec]
    pretrained_checkpoint: str


class ModelConfig(BaseModel):
    model: ModelSettings


class TrainingSettings(BaseModel):
    backend: str
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    runtime: str
    experiment_name: str
    checkpoint_name: str
    max_samples: Optional[int] = None


class TrainingConfig(BaseModel):
    training: TrainingSettings
