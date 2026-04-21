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
    split: "DatasetSplitSettings | None" = None


class DatasetConfig(BaseModel):
    name: str
    dataset: DatasetSettings


class DatasetSplitSettings(BaseModel):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    train_manifest_output: str
    val_manifest_output: str
    test_manifest_output: str


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
    anchor_match_iou_threshold: float = 0.1
    score_threshold: float = 0.15
    nms_iou_threshold: float = 0.5
    pretrained_checkpoint: str


class ModelConfig(BaseModel):
    model: ModelSettings


class AugmentationSettings(BaseModel):
    enabled: bool = False
    horizontal_flip_prob: float = 0.0
    brightness_delta: float = 0.0
    contrast_min: float = 1.0
    contrast_max: float = 1.0


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
    resume_from_checkpoint: Optional[str] = None
    tensorboard_enabled: bool = True
    augmentation: AugmentationSettings = AugmentationSettings()


class TrainingConfig(BaseModel):
    training: TrainingSettings
