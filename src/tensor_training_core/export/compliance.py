from __future__ import annotations

import json
from pathlib import Path

from tensor_training_core.config.schema import DatasetConfig, ModelConfig


def build_license_metadata(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    quantization: str,
) -> dict[str, object]:
    return {
        "model_name": model_config.model.name,
        "model_family": model_config.model.family,
        "dataset_name": dataset_config.name,
        "quantization": quantization,
        "repository_notice": (
            "This repository summary is not legal advice. Review exact dependency versions, model assets, "
            "and dataset licenses before shipping a product."
        ),
        "dependencies": [
            {
                "name": "TensorFlow",
                "license": "Apache-2.0",
                "usage_note": "Core training and export framework.",
            },
            {
                "name": "TensorFlow Lite Support",
                "license": "Apache-2.0",
                "usage_note": "Useful for mobile metadata and helper tooling.",
            },
            {
                "name": "COCO API / pycocotools",
                "license": "Simplified BSD",
                "usage_note": "Used for COCO parsing and validation workflows.",
            },
        ],
        "dataset_notice": {
            "format": dataset_config.dataset.format,
            "review_required": True,
            "notes": [
                "COCO-format import is supported, but dataset redistribution and attribution obligations must be reviewed separately.",
                "Keep raw datasets separate from project-owned manifests and generated artifacts.",
            ],
        },
    }


def build_model_card(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    quantization: str,
    metadata_path: str | Path,
) -> str:
    return "\n".join(
        [
            "# Model Card",
            "",
            "## Summary",
            "",
            f"- Model name: `{model_config.model.name}`",
            f"- Model family: `{model_config.model.family}`",
            f"- Quantization: `{quantization}`",
            f"- Dataset: `{dataset_config.name}`",
            f"- Input size: `{model_config.model.image_size}`",
            f"- Max detections: `{model_config.model.max_detections}`",
            "",
            "## Intended Use",
            "",
            "- Mobile-oriented TensorFlow Lite object detection inference.",
            "- Validation, prototyping, and downstream mobile integration testing.",
            "",
            "## Deployment Notes",
            "",
            "- Use `label.txt` for foreground label display order.",
            f"- Use `{Path(metadata_path).name}` for preprocessing and postprocessing assumptions.",
            "- Apply score threshold and NMS rules from export metadata.",
            "",
            "## Limitations",
            "",
            "- This baseline is optimized for project workflow validation, not guaranteed production accuracy.",
            "- Accuracy, latency, and memory usage should be verified on the final target device.",
            "- Dataset license review is still required before external distribution.",
            "",
            "## License Notes",
            "",
            "- See `license_metadata.json` for dependency and dataset caveats included with this export.",
        ]
    ) + "\n"


def write_license_metadata(output_path: str | Path, payload: dict[str, object]) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_file


def write_model_card(output_path: str | Path, content: str) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    return output_file
