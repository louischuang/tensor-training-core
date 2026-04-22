from __future__ import annotations

import json

from tensor_training_core.mobile.android.bundle_writer import write_android_bundle
from tensor_training_core.mobile.ios.bundle_writer import write_ios_bundle


def _build_inputs(tmp_path):
    tflite_path = tmp_path / "model.tflite"
    label_map_path = tmp_path / "label_map.json"
    metadata_path = tmp_path / "export_metadata.json"
    model_card_path = tmp_path / "MODEL_CARD.md"
    license_metadata_path = tmp_path / "license_metadata.json"
    benchmark_report_path = tmp_path / "benchmark_report.json"
    tflite_path.write_bytes(b"tflite")
    label_map_path.write_text(json.dumps({"0": "group", "1": "cat"}), encoding="utf-8")
    metadata_path.write_text(json.dumps({"image_size": [320, 320]}), encoding="utf-8")
    model_card_path.write_text("# Model Card\n", encoding="utf-8")
    license_metadata_path.write_text(json.dumps({"dependencies": []}), encoding="utf-8")
    benchmark_report_path.write_text(json.dumps({"benchmarks": {}}), encoding="utf-8")
    return tflite_path, label_map_path, metadata_path, model_card_path, license_metadata_path, benchmark_report_path


def test_write_android_bundle_writes_assumptions_and_verification(tmp_path) -> None:
    tflite_path, label_map_path, metadata_path, model_card_path, license_metadata_path, benchmark_report_path = _build_inputs(tmp_path)

    bundle_dir = write_android_bundle(
        output_dir=tmp_path / "android",
        tflite_path=tflite_path,
        label_map_path=label_map_path,
        metadata_path=metadata_path,
        model_card_path=model_card_path,
        license_metadata_path=license_metadata_path,
        benchmark_report_path=benchmark_report_path,
    )

    assert (bundle_dir / "model.tflite").exists()
    assert (bundle_dir / "label.txt").exists()
    assert (bundle_dir / "INTEGRATION.md").exists()
    assert (bundle_dir / "MODEL_CARD.md").exists()
    assert (bundle_dir / "license_metadata.json").exists()
    assert (bundle_dir / "benchmark_report.json").exists()
    verification = json.loads((bundle_dir / "bundle_verification.json").read_text(encoding="utf-8"))
    assert verification["platform"] == "android"
    assert verification["status"] == "ready"


def test_write_ios_bundle_writes_assumptions_and_verification(tmp_path) -> None:
    tflite_path, label_map_path, metadata_path, model_card_path, license_metadata_path, benchmark_report_path = _build_inputs(tmp_path)

    bundle_dir = write_ios_bundle(
        output_dir=tmp_path / "ios",
        tflite_path=tflite_path,
        label_map_path=label_map_path,
        metadata_path=metadata_path,
        model_card_path=model_card_path,
        license_metadata_path=license_metadata_path,
        benchmark_report_path=benchmark_report_path,
    )

    assert (bundle_dir / "model.tflite").exists()
    assert (bundle_dir / "label.txt").exists()
    assert (bundle_dir / "INTEGRATION.md").exists()
    assert (bundle_dir / "MODEL_CARD.md").exists()
    assert (bundle_dir / "license_metadata.json").exists()
    assert (bundle_dir / "benchmark_report.json").exists()
    verification = json.loads((bundle_dir / "bundle_verification.json").read_text(encoding="utf-8"))
    assert verification["platform"] == "ios"
    assert verification["status"] == "ready"
