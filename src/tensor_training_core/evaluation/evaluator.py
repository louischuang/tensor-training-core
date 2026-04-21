from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.config.schema import ModelConfig, TrainingConfig
from tensor_training_core.evaluation.reports import compute_detection_metrics, run_classwise_nms
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.models.anchors import decode_box_from_anchor, load_anchor_array
from tensor_training_core.training.runner import apply_sample_limit, load_training_samples
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import get_latest_run_dir, resolve_repo_path


def _prepare_input_image(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(resolve_repo_path(image_path)).convert("RGB")
    image = image.resize(image_size)
    return np.asarray(image, dtype=np.float32) / 255.0


def _build_ground_truth(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        width = float(row["width"])
        height = float(row["height"])
        items: list[dict[str, object]] = []
        for annotation in row["annotations"]:
            x, y, w, h = [float(value) for value in annotation["bbox_xywh"]]
            items.append(
                {
                    "label_id": int(annotation["category_id"]),
                    "bbox_xywh_norm": [x / width, y / height, w / width, h / height],
                }
            )
        ground_truth_by_image[str(row["image_path"])] = items
    return ground_truth_by_image


def _decode_predictions(
    class_scores: np.ndarray,
    box_offsets: np.ndarray,
    anchors: np.ndarray,
    score_threshold: float,
    nms_iou_threshold: float,
) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    for anchor_index, anchor_scores in enumerate(class_scores):
        foreground_scores = anchor_scores[1:]
        label_offset = int(np.argmax(foreground_scores))
        score = float(foreground_scores[label_offset])
        if score < score_threshold:
            continue
        label_id = label_offset + 1
        decoded_box = decode_box_from_anchor(box_offsets[anchor_index], anchors[anchor_index])
        detections.append(
            {
                "anchor_index": anchor_index,
                "label_id": label_id,
                "score": score,
                "bbox_xywh_norm": [float(value) for value in decoded_box.tolist()],
            }
        )
    return run_classwise_nms(detections, iou_threshold=nms_iou_threshold)


def evaluate_model(
    context: RunContext,
    experiment_id: str,
    manifest_path: str | Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict[str, str]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required to evaluate the trained model.") from exc

    logger = get_logger("evaluation")
    latest_run_dir = get_latest_run_dir(
        experiment_id,
        required_relative_path="checkpoints/latest.keras",
        exclude_run_id=context.run_id,
    )
    checkpoint_path = latest_run_dir / "checkpoints" / "latest.keras"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    rows = apply_sample_limit(load_training_samples(manifest_path), training_config.training.max_samples)
    if not rows:
        raise ValueError("Evaluation manifest is empty; cannot run evaluation.")
    logger.info(
        "evaluation_started manifest_path=%s record_count=%s checkpoint_path=%s",
        manifest_path,
        len(rows),
        checkpoint_path,
    )

    model = tf.keras.models.load_model(checkpoint_path)
    anchors = load_anchor_array(model_config)
    image_size = tuple(model_config.model.image_size)
    predictions_by_image: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        image_array = _prepare_input_image(row["image_path"], image_size)
        outputs = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
        predictions = _decode_predictions(
            outputs["class_output"][0],
            outputs["bbox_output"][0],
            anchors,
            score_threshold=model_config.model.score_threshold,
            nms_iou_threshold=model_config.model.nms_iou_threshold,
        )
        predictions_by_image[str(row["image_path"])] = predictions
    logger.info("evaluation_predictions_completed image_count=%s", len(predictions_by_image))

    ground_truth_by_image = _build_ground_truth(rows)
    detection_metrics = compute_detection_metrics(
        predictions_by_image=predictions_by_image,
        ground_truth_by_image=ground_truth_by_image,
        class_ids=list(range(1, model_config.model.num_classes + 1)),
    )

    metrics_path = context.artifact_dir / "evaluation_metrics.json"
    summary_path = context.artifact_dir / "evaluation_summary.json"
    metrics_path.write_text(json.dumps(detection_metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    summary = {
        "run_id": context.run_id,
        "source_run_id": latest_run_dir.name,
        "experiment_id": experiment_id,
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "record_count": int(len(rows)),
        "metrics_path": str(metrics_path),
        "max_samples": training_config.training.max_samples,
        "map50": detection_metrics["map50"],
        "precision_macro": detection_metrics["precision_macro"],
        "recall_macro": detection_metrics["recall_macro"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logger.info(
        "evaluation_metrics_completed map50=%.6f precision_macro=%.6f recall_macro=%.6f",
        detection_metrics["map50"],
        detection_metrics["precision_macro"],
        detection_metrics["recall_macro"],
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }
