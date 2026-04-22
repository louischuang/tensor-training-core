from __future__ import annotations

from tensor_training_core.config.loader import load_dataset_config, load_model_config
from tensor_training_core.export.compliance import build_license_metadata, build_model_card


def test_build_license_metadata_contains_dependency_and_dataset_notice() -> None:
    dataset_config = load_dataset_config("configs/datasets/coco_detection.yaml")
    model_config = load_model_config("configs/models/ssd_mobilenet_v2_fpnlite_320.yaml")

    payload = build_license_metadata(
        model_config=model_config,
        dataset_config=dataset_config,
        quantization="float32",
    )

    assert payload["model_name"] == "ssd_mobilenet_v2_fpnlite_320x320"
    assert payload["dataset_notice"]["review_required"] is True
    assert any(dep["name"] == "TensorFlow" for dep in payload["dependencies"])


def test_build_model_card_mentions_model_and_license_metadata() -> None:
    dataset_config = load_dataset_config("configs/datasets/coco_detection.yaml")
    model_config = load_model_config("configs/models/ssd_mobilenet_v2_fpnlite_320.yaml")

    model_card = build_model_card(
        model_config=model_config,
        dataset_config=dataset_config,
        quantization="float32",
        metadata_path="export_metadata_float32.json",
    )

    assert "Model Card" in model_card
    assert "ssd_mobilenet_v2_fpnlite_320x320" in model_card
    assert "license_metadata.json" in model_card
