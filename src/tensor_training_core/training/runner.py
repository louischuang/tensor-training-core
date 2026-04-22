from __future__ import annotations

import json
import importlib.util
import random
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.config.schema import AugmentationSettings, ModelConfig, TrainingConfig
from tensor_training_core.data.manifest.reader import read_manifest
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.models.anchors import compute_iou, encode_box_to_anchor, load_anchor_array
from tensor_training_core.models.factory import build_keras_detection_model
from tensor_training_core.training.callbacks import build_tensorboard_callback, build_training_progress_callback
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import resolve_repo_path
from tensor_training_core.utils.seed import seed_everything


AUGMENTATION_PRESETS: dict[str, dict[str, float | bool]] = {
    "disabled": {
        "enabled": False,
        "horizontal_flip_prob": 0.0,
        "brightness_delta": 0.0,
        "contrast_min": 1.0,
        "contrast_max": 1.0,
    },
    "light": {
        "enabled": True,
        "horizontal_flip_prob": 0.25,
        "brightness_delta": 0.04,
        "contrast_min": 0.95,
        "contrast_max": 1.05,
    },
    "standard": {
        "enabled": True,
        "horizontal_flip_prob": 0.5,
        "brightness_delta": 0.08,
        "contrast_min": 0.9,
        "contrast_max": 1.15,
    },
    "aggressive": {
        "enabled": True,
        "horizontal_flip_prob": 0.5,
        "brightness_delta": 0.12,
        "contrast_min": 0.8,
        "contrast_max": 1.25,
    },
}


def resolve_augmentation_settings(augmentation: AugmentationSettings) -> AugmentationSettings:
    preset_name = augmentation.preset or "custom"
    if preset_name == "custom":
        return augmentation
    if preset_name not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unsupported augmentation preset: {preset_name}")

    defaults = AUGMENTATION_PRESETS[preset_name]
    field_defaults = AugmentationSettings()
    return AugmentationSettings(
        preset=preset_name,
        enabled=augmentation.enabled if augmentation.enabled != field_defaults.enabled else bool(defaults["enabled"]),
        horizontal_flip_prob=(
            augmentation.horizontal_flip_prob
            if augmentation.horizontal_flip_prob != field_defaults.horizontal_flip_prob
            else float(defaults["horizontal_flip_prob"])
        ),
        brightness_delta=(
            augmentation.brightness_delta
            if augmentation.brightness_delta != field_defaults.brightness_delta
            else float(defaults["brightness_delta"])
        ),
        contrast_min=(
            augmentation.contrast_min
            if augmentation.contrast_min != field_defaults.contrast_min
            else float(defaults["contrast_min"])
        ),
        contrast_max=(
            augmentation.contrast_max
            if augmentation.contrast_max != field_defaults.contrast_max
            else float(defaults["contrast_max"])
        ),
    )


def run_smoke_training(
    context: RunContext,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    manifest_path: str | Path,
) -> dict[str, str]:
    seed_everything(training_config.training.seed)

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_file}")

    records = [line for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint_dir = context.artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = context.artifact_dir / "training_metrics.jsonl"
    summary_path = context.artifact_dir / "training_summary.json"
    checkpoint_path = checkpoint_dir / training_config.training.checkpoint_name

    with metrics_path.open("w", encoding="utf-8") as handle:
        for epoch in range(1, training_config.training.epochs + 1):
            metric = {
                "epoch": epoch,
                "loss": round(1.0 / epoch, 6),
                "record_count": len(records),
                "model_name": model_config.model.name,
                "backend": training_config.training.backend,
            }
            handle.write(json.dumps(metric, ensure_ascii=True) + "\n")

    checkpoint_path.write_text("smoke checkpoint placeholder\n", encoding="utf-8")
    summary = {
        "run_id": context.run_id,
        "experiment_id": context.experiment_id,
        "dataset_version": context.dataset_version,
        "backend": training_config.training.backend,
        "model_name": model_config.model.name,
        "manifest_path": str(manifest_file),
        "checkpoint_path": str(checkpoint_path),
        "epochs": training_config.training.epochs,
        "record_count": len(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }


def _group_records_by_image(manifest_path: str | Path) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for record in read_manifest(manifest_path):
        if record.image_path not in grouped:
            grouped[record.image_path] = {
                "image_path": record.image_path,
                "width": record.width,
                "height": record.height,
                "annotations": [],
            }
        grouped[record.image_path]["annotations"].append(
            {
                "category_id": record.category_id,
                "bbox_xywh": record.bbox_xywh,
            }
        )
    return list(grouped.values())


def load_training_samples(manifest_path: str | Path) -> list[dict[str, object]]:
    return _group_records_by_image(manifest_path)


def apply_sample_limit(
    rows: list[dict[str, object]],
    max_samples: int | None,
) -> list[dict[str, object]]:
    if max_samples is None or max_samples <= 0:
        return rows
    return rows[:max_samples]


def _normalize_box(bbox_xywh: list[float], width: float, height: float) -> list[float]:
    x, y, w, h = [float(value) for value in bbox_xywh]
    return [x / width, y / height, w / width, h / height]


def _augment_image_and_boxes(
    image_array: np.ndarray,
    boxes_xywh_norm: list[np.ndarray],
    augmentation: AugmentationSettings,
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not augmentation.enabled:
        return image_array, boxes_xywh_norm

    augmented_image = np.asarray(image_array, dtype=np.float32)
    augmented_boxes = [np.asarray(box, dtype=np.float32).copy() for box in boxes_xywh_norm]

    if augmentation.horizontal_flip_prob > 0.0 and rng.random() < augmentation.horizontal_flip_prob:
        augmented_image = np.flip(augmented_image, axis=1).copy()
        for index, box in enumerate(augmented_boxes):
            x, y, w, h = [float(value) for value in box.tolist()]
            augmented_boxes[index] = np.asarray([max(0.0, min(1.0 - w, 1.0 - x - w)), y, w, h], dtype=np.float32)

    if augmentation.brightness_delta > 0.0:
        delta = rng.uniform(-augmentation.brightness_delta, augmentation.brightness_delta)
        augmented_image = np.clip(augmented_image + delta, 0.0, 1.0)

    contrast_min = float(augmentation.contrast_min)
    contrast_max = float(augmentation.contrast_max)
    if contrast_max > contrast_min and contrast_max > 0.0:
        factor = rng.uniform(contrast_min, contrast_max)
        mean = np.mean(augmented_image, axis=(0, 1), keepdims=True)
        augmented_image = np.clip((augmented_image - mean) * factor + mean, 0.0, 1.0)

    return augmented_image.astype(np.float32), augmented_boxes


def _load_single_sample(
    row: dict[str, object],
    image_size: tuple[int, int],
    anchors: np.ndarray,
    anchor_match_iou_threshold: float,
    augmentation: AugmentationSettings,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(resolve_repo_path(row["image_path"])).convert("RGB")
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    width = float(row["width"])
    height = float(row["height"])
    max_detections = anchors.shape[0]
    classes = np.zeros((max_detections,), dtype=np.int32)
    box_offsets = np.zeros((max_detections, 4), dtype=np.float32)
    bbox_weights = np.zeros((max_detections,), dtype=np.float32)
    best_ious = np.zeros((max_detections,), dtype=np.float32)
    normalized_boxes: list[np.ndarray] = []
    category_ids: list[int] = []

    for annotation in row["annotations"]:
        normalized_box = np.asarray(_normalize_box(annotation["bbox_xywh"], width, height), dtype=np.float32)
        normalized_boxes.append(normalized_box)
        category_ids.append(int(annotation["category_id"]))

    image_array, normalized_boxes = _augment_image_and_boxes(
        image_array,
        normalized_boxes,
        augmentation,
        rng,
    )

    for category_id, normalized_box in zip(category_ids, normalized_boxes, strict=False):
        box_center = np.asarray(
            [
                normalized_box[0] + normalized_box[2] / 2.0,
                normalized_box[1] + normalized_box[3] / 2.0,
                normalized_box[2],
                normalized_box[3],
            ],
            dtype=np.float32,
        )
        ious = np.asarray([compute_iou(box_center, anchor) for anchor in anchors], dtype=np.float32)
        anchor_index = int(np.argmax(ious))
        if ious[anchor_index] < anchor_match_iou_threshold:
            continue
        if ious[anchor_index] < best_ious[anchor_index]:
            continue
        best_ious[anchor_index] = ious[anchor_index]
        classes[anchor_index] = category_id
        box_offsets[anchor_index] = encode_box_to_anchor(normalized_box, anchors[anchor_index])
        bbox_weights[anchor_index] = 1.0

    return image_array, classes, box_offsets, bbox_weights


def build_manifest_sequence(
    tf,
    rows: list[dict[str, object]],
    image_size: tuple[int, int],
    batch_size: int,
    anchors: np.ndarray,
    anchor_match_iou_threshold: float,
    seed: int,
    augmentation: AugmentationSettings,
):
    class ManifestSequence(tf.keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.rows = rows
            self.image_size = image_size
            self.batch_size = max(1, batch_size)
            self.anchors = anchors
            self.anchor_match_iou_threshold = anchor_match_iou_threshold
            self.rng = random.Random(seed)
            self.augmentation = augmentation

        def __len__(self) -> int:
            return (len(self.rows) + self.batch_size - 1) // self.batch_size

        def __getitem__(self, index: int):
            start = index * self.batch_size
            end = min(start + self.batch_size, len(self.rows))
            batch_rows = self.rows[start:end]
            images: list[np.ndarray] = []
            classes: list[np.ndarray] = []
            boxes: list[np.ndarray] = []
            bbox_weights: list[np.ndarray] = []

            for row in batch_rows:
                image_array, class_targets, box_targets, box_weights = _load_single_sample(
                    row,
                    self.image_size,
                    self.anchors,
                    self.anchor_match_iou_threshold,
                    self.augmentation,
                    self.rng,
                )
                images.append(image_array)
                classes.append(class_targets)
                boxes.append(box_targets)
                bbox_weights.append(box_weights)

            class_array = np.asarray(classes, dtype=np.int32)
            bbox_array = np.asarray(boxes, dtype=np.float32)
            bbox_weight_array = np.asarray(bbox_weights, dtype=np.float32)
            return np.asarray(images), {
                "class_output": class_array,
                "bbox_output": bbox_array,
            }, {
                "class_output": np.ones_like(class_array, dtype=np.float32),
                "bbox_output": bbox_weight_array,
            }

        def on_epoch_end(self) -> None:
            self.rng.shuffle(self.rows)

    return ManifestSequence()


def run_tensorflow_training(
    context: RunContext,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    manifest_path: str | Path,
) -> dict[str, str]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow backend requested, but tensorflow is not installed in this environment. "
            "Use Docker or a Python 3.11 environment with requirements/cuda.txt."
        ) from exc

    seed_everything(training_config.training.seed)
    tf.keras.utils.set_random_seed(training_config.training.seed)
    logger = get_logger("training")
    checkpoint_dir = context.artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = context.artifact_dir / "training_metrics.jsonl"
    summary_path = context.artifact_dir / "training_summary.json"
    checkpoint_path = checkpoint_dir / training_config.training.checkpoint_name
    failure_summary_path = context.log_dir / "failure_summary.json"
    tensorboard_dir = context.log_dir / "tensorboard"

    def write_failure_summary(exc: Exception, stage: str) -> None:
        payload = {
            "run_id": context.run_id,
            "experiment_id": context.experiment_id,
            "dataset_version": context.dataset_version,
            "stage": stage,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "traceback": traceback.format_exc(),
        }
        failure_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        logger.error("training_failed stage=%s failure_summary_path=%s error=%s", stage, failure_summary_path, exc)

    try:
        image_size = tuple(model_config.model.image_size)
        anchors = load_anchor_array(model_config)
        max_detections = int(model_config.model.max_detections)
        resolved_augmentation = resolve_augmentation_settings(training_config.training.augmentation)
        rows = apply_sample_limit(load_training_samples(manifest_path), training_config.training.max_samples)
        if not rows:
            raise ValueError("Training manifest is empty; cannot start TensorFlow training.")

        inferred_num_classes = max(
            int(annotation["category_id"])
            for row in rows
            for annotation in row["annotations"]
        )
        if inferred_num_classes > int(model_config.model.num_classes):
            raise ValueError(
                "Dataset contains more classes than the configured model supports: "
                f"{inferred_num_classes} > {model_config.model.num_classes}"
            )

        resume_from_checkpoint = training_config.training.resume_from_checkpoint
        resumed_from_checkpoint: str | None = None
        pretrained_loaded_from: str | None = None

        if resume_from_checkpoint:
            resolved_resume_path = resolve_repo_path(resume_from_checkpoint)
            if not resolved_resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved_resume_path}")
            model = tf.keras.models.load_model(resolved_resume_path)
            resumed_from_checkpoint = str(resolved_resume_path)
            logger.info("training_resume_loaded checkpoint_path=%s", resolved_resume_path)
        else:
            model = build_keras_detection_model(model_config)
            pretrained_checkpoint = model_config.model.pretrained_checkpoint.strip()
            if pretrained_checkpoint and pretrained_checkpoint.lower() != "imagenet":
                resolved_pretrained_path = resolve_repo_path(pretrained_checkpoint)
                if not resolved_pretrained_path.exists():
                    raise FileNotFoundError(f"Pretrained checkpoint does not exist: {resolved_pretrained_path}")
                if resolved_pretrained_path.suffix == ".keras":
                    pretrained_model = tf.keras.models.load_model(resolved_pretrained_path)
                    model.set_weights(pretrained_model.get_weights())
                else:
                    model.load_weights(resolved_pretrained_path)
                pretrained_loaded_from = str(resolved_pretrained_path)
                logger.info("training_pretrained_loaded checkpoint_path=%s", resolved_pretrained_path)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.training.learning_rate),
            loss={
                "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "bbox_output": tf.keras.losses.Huber(),
            },
            metrics={"class_output": ["accuracy"]},
        )

        sequence = build_manifest_sequence(
            tf,
            rows,
            image_size,
            training_config.training.batch_size,
            anchors,
            model_config.model.anchor_match_iou_threshold,
            training_config.training.seed,
            resolved_augmentation,
        )
        progress_callback = build_training_progress_callback(
            tf,
            logger=logger,
            total_epochs=training_config.training.epochs,
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_weights_only=False,
            save_best_only=False,
            verbose=0,
        )
        callbacks = [progress_callback, checkpoint_callback]
        tensorboard_active = False
        if training_config.training.tensorboard_enabled and importlib.util.find_spec("tensorboard") is not None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(build_tensorboard_callback(tf, str(tensorboard_dir)))
            tensorboard_active = True
        elif training_config.training.tensorboard_enabled:
            logger.warning("tensorboard_unavailable tensorboard metrics logging will be skipped")

        logger.info(
            "training_configuration manifest_path=%s record_count=%s batch_size=%s epochs=%s max_detections=%s augmentation_enabled=%s augmentation_preset=%s tensorboard_enabled=%s resumed_from_checkpoint=%s pretrained_loaded_from=%s",
            manifest_path,
            len(rows),
            training_config.training.batch_size,
            training_config.training.epochs,
            max_detections,
            resolved_augmentation.enabled,
            resolved_augmentation.preset,
            tensorboard_active,
            resumed_from_checkpoint,
            pretrained_loaded_from or model_config.model.pretrained_checkpoint or "",
        )

        history = model.fit(
            sequence,
            epochs=training_config.training.epochs,
            verbose=0,
            callbacks=callbacks,
        )
        model.save(checkpoint_path)

        with metrics_path.open("w", encoding="utf-8") as handle:
            for epoch_idx in range(training_config.training.epochs):
                metric = {
                    "epoch": epoch_idx + 1,
                    "loss": float(history.history["loss"][epoch_idx]),
                    "class_accuracy": float(history.history["class_output_accuracy"][epoch_idx]),
                    "record_count": int(len(rows)),
                    "backend": training_config.training.backend,
                    "model_name": model_config.model.name,
                    "max_detections": max_detections,
                }
                handle.write(json.dumps(metric, ensure_ascii=True) + "\n")

        summary = {
            "run_id": context.run_id,
            "experiment_id": context.experiment_id,
            "dataset_version": context.dataset_version,
            "backend": training_config.training.backend,
            "model_name": model_config.model.name,
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "epochs": training_config.training.epochs,
            "record_count": int(len(rows)),
            "num_classes": int(model_config.model.num_classes),
            "max_detections": max_detections,
            "max_samples": training_config.training.max_samples,
            "augmentation_enabled": resolved_augmentation.enabled,
            "augmentation_preset": resolved_augmentation.preset,
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_active else "",
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "pretrained_loaded_from": pretrained_loaded_from or model_config.model.pretrained_checkpoint or "",
            "failure_summary_path": "",
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_active else "",
            "failure_summary_path": "",
        }
    except Exception as exc:
        write_failure_summary(exc, stage="train")
        raise
