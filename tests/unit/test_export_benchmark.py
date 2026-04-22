from __future__ import annotations

from pathlib import Path

import numpy as np

from tensor_training_core.export.benchmark import build_benchmark_report


class _FakeInterpreter:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._input = {
            "index": 0,
            "shape": np.asarray([1, 320, 320, 3]),
            "dtype": np.float32,
            "quantization": (0.0, 0),
        }
        self._outputs = [
            {"index": 1, "shape": np.asarray([1, 8, 6]), "dtype": np.float32, "name": "class_output", "quantization": (0.0, 0)},
            {"index": 2, "shape": np.asarray([1, 8, 4]), "dtype": np.float32, "name": "bbox_output", "quantization": (0.0, 0)},
        ]

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self):
        return [self._input]

    def get_output_details(self):
        return self._outputs

    def set_tensor(self, index: int, value):
        self._value = value

    def invoke(self) -> None:
        return None


class _FakeLite:
    Interpreter = _FakeInterpreter


class _FakeTF:
    lite = _FakeLite()


def test_build_benchmark_report_collects_latency_and_size(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    tflite_path = export_dir / "model_float32.tflite"
    tflite_path.write_bytes(b"abcd")
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.ppm"
    image_path.write_text("P3\n1 1\n255\n255 0 0\n", encoding="utf-8")
    manifest_path.write_text(
        '{"image_id": 1, "image_path": "' + str(image_path) + '", "width": 1, "height": 1, "category_id": 1, "category_name": "object", "bbox_xywh": [0, 0, 1, 1]}\n',
        encoding="utf-8",
    )
    export_manifest_path = export_dir / "export_manifest.json"
    export_manifest_path.write_text(
        '{"source_run_id": "run_test", "exports": {"float32": {"tflite_path": "' + str(tflite_path) + '"}}}\n',
        encoding="utf-8",
    )

    report = build_benchmark_report(
        tf=_FakeTF(),
        export_manifest_path=export_manifest_path,
        manifest_path=manifest_path,
        image_size=(320, 320),
        num_runs=2,
    )

    assert report["source_run_id"] == "run_test"
    assert "float32" in report["benchmarks"]
    bench = report["benchmarks"]["float32"]
    assert bench["model_size_bytes"] == 4
    assert bench["latency_ms"]["avg"] >= 0.0
    assert bench["estimated_working_set_bytes"] > 0
