from __future__ import annotations

import json
from pathlib import Path

from tensor_training_core.config.loader import load_dataset_config, load_model_config
from tensor_training_core.export.registry import register_model_version
from tensor_training_core.interfaces.dto import RunContext


def test_register_model_version_writes_descriptor_and_indexes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("tensor_training_core.export.registry.MODELS_DIR", tmp_path / "models")

    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True)
    export_manifest_path = export_dir / "export_manifest.json"
    export_manifest_path.write_text(
        json.dumps(
            {
                "source_run_id": "source_run_123",
                "exports": {
                    "float32": {"tflite_path": "/tmp/model_float32.tflite", "metadata_path": "/tmp/export_metadata_float32.json"},
                    "int8": {"tflite_path": "/tmp/model_int8.tflite", "metadata_path": "/tmp/export_metadata_int8.json"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    context = RunContext(
        run_id="20260422T010101Z_abcdef12",
        experiment_id="train_tensorflow_esp32_cam_dev",
        dataset_version="v1",
        experiment_dir=tmp_path / "experiments",
        artifact_dir=tmp_path / "artifacts" / "run",
        log_dir=tmp_path / "logs" / "run",
    )
    dataset_config = load_dataset_config("configs/datasets/esp32_cam_train.yaml")
    model_config = load_model_config("configs/models/ssd_mobilenet_v2_fpnlite_320.yaml")
    outputs = register_model_version(
        context=context,
        model_config=model_config,
        dataset_config=dataset_config,
        export_outputs={
            "export_manifest_path": str(export_manifest_path),
            "label_txt_path": "/tmp/label.txt",
            "model_card_path": "/tmp/MODEL_CARD.md",
            "license_metadata_path": "/tmp/license_metadata.json",
            "benchmark_report_path": "/tmp/benchmark_report.json",
        },
    )

    descriptor_path = Path(outputs["model_registry_version_path"])
    model_index_path = Path(outputs["model_registry_index_path"])
    global_index_path = Path(outputs["global_model_registry_index_path"])

    assert descriptor_path.exists()
    assert model_index_path.exists()
    assert global_index_path.exists()

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["model_key"] == "train_tensorflow_esp32_cam_dev/ssd_mobilenet_v2_fpnlite_320x320"
    assert descriptor["version_id"] == context.run_id
    assert descriptor["quantizations"] == ["float32", "int8"]

    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    assert model_index["latest_version_id"] == context.run_id
    assert model_index["version_count"] == 1

    global_index = json.loads(global_index_path.read_text(encoding="utf-8"))
    assert global_index["models"][0]["model_key"] == descriptor["model_key"]
