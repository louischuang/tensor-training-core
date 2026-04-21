from __future__ import annotations

import json
from pathlib import Path

from tensor_training_core.config.schema import DatasetConfig, ModelConfig


def build_export_metadata(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    tflite_path: str | Path,
    source_checkpoint: str | Path,
    quantization: str,
) -> dict[str, object]:
    return {
        "model_name": model_config.model.name,
        "model_family": model_config.model.family,
        "image_size": model_config.model.image_size,
        "num_classes": model_config.model.num_classes,
        "max_detections": model_config.model.max_detections,
        "anchors": [anchor.model_dump() for anchor in model_config.model.anchors],
        "dataset_name": dataset_config.name,
        "tflite_path": str(tflite_path),
        "source_checkpoint": str(source_checkpoint),
        "quantization": quantization,
        "preprocessing": {
            "color_space": "RGB",
            "value_range": [0.0, 1.0],
            "resize": model_config.model.image_size,
        },
        "postprocessing": {
            "class_output": "per-anchor softmax probabilities with background class 0",
            "bbox_output": "per-anchor offsets relative to configured anchors",
            "nms": {
                "score_threshold": model_config.model.score_threshold,
                "iou_threshold": model_config.model.nms_iou_threshold,
            },
        },
    }


def write_json_file(output_path: str | Path, payload: dict[str, object]) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_file
