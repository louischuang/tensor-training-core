from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.config.schema import ModelConfig, TrainingConfig
from tensor_training_core.data.manifest.reader import read_manifest
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.models.anchors import compute_iou, encode_box_to_anchor, load_anchor_array
from tensor_training_core.models.factory import build_keras_detection_model
from tensor_training_core.utils.seed import seed_everything


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


def _load_single_sample(
    row: dict[str, object],
    image_size: tuple[int, int],
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(row["image_path"]).convert("RGB")
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    width = float(row["width"])
    height = float(row["height"])
    max_detections = anchors.shape[0]
    classes = np.zeros((max_detections,), dtype=np.int32)
    box_offsets = np.zeros((max_detections, 4), dtype=np.float32)
    bbox_weights = np.zeros((max_detections,), dtype=np.float32)
    best_ious = np.zeros((max_detections,), dtype=np.float32)

    for annotation in row["annotations"]:
        normalized_box = np.asarray(_normalize_box(annotation["bbox_xywh"], width, height), dtype=np.float32)
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
        if ious[anchor_index] < best_ious[anchor_index]:
            continue
        best_ious[anchor_index] = ious[anchor_index]
        classes[anchor_index] = int(annotation["category_id"])
        box_offsets[anchor_index] = encode_box_to_anchor(normalized_box, anchors[anchor_index])
        bbox_weights[anchor_index] = 1.0

    return image_array, classes, box_offsets, bbox_weights


def build_manifest_sequence(
    tf,
    rows: list[dict[str, object]],
    image_size: tuple[int, int],
    batch_size: int,
    anchors: np.ndarray,
):
    class ManifestSequence(tf.keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.rows = rows
            self.image_size = image_size
            self.batch_size = max(1, batch_size)
            self.anchors = anchors

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

    image_size = tuple(model_config.model.image_size)
    anchors = load_anchor_array(model_config)
    max_detections = int(model_config.model.max_detections)
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

    model = build_keras_detection_model(model_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.training.learning_rate),
        loss={
            "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            "bbox_output": tf.keras.losses.Huber(),
        },
        metrics={"class_output": ["accuracy"]},
    )

    checkpoint_dir = context.artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = context.artifact_dir / "training_metrics.jsonl"
    summary_path = context.artifact_dir / "training_summary.json"
    checkpoint_path = checkpoint_dir / training_config.training.checkpoint_name
    sequence = build_manifest_sequence(
        tf,
        rows,
        image_size,
        training_config.training.batch_size,
        anchors,
    )

    history = model.fit(sequence, epochs=training_config.training.epochs, verbose=0)
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
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
