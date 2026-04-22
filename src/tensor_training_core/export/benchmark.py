from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.data.manifest.reader import read_manifest
from tensor_training_core.utils.paths import resolve_repo_path


def _prepare_input_image(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(resolve_repo_path(image_path)).convert("RGB")
    image = image.resize(image_size)
    return np.asarray(image, dtype=np.float32) / 255.0


def _quantize_input_if_needed(input_details, image_array: np.ndarray) -> np.ndarray:
    input_tensor = np.expand_dims(image_array, axis=0)
    if input_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = input_details["quantization"]
        if scale > 0:
            input_tensor = np.clip(np.round(input_tensor / scale + zero_point), 0, 255).astype(input_details["dtype"])
        else:
            input_tensor = input_tensor.astype(input_details["dtype"])
    return input_tensor.astype(input_details["dtype"])


def _tensor_bytes(detail: dict[str, object]) -> int:
    shape = detail.get("shape", [])
    dtype = detail.get("dtype")
    if dtype is None:
        return 0
    size = int(np.prod(shape)) if len(shape) > 0 else 0
    return int(size * np.dtype(dtype).itemsize)


def _plain_shape(shape: object) -> list[int]:
    return [int(value) for value in list(shape)]


def build_benchmark_report(
    tf,
    export_manifest_path: str | Path,
    manifest_path: str | Path,
    image_size: tuple[int, int],
    num_runs: int = 3,
) -> dict[str, object]:
    export_manifest = json.loads(Path(export_manifest_path).read_text(encoding="utf-8"))
    sample_record = next(iter(read_manifest(manifest_path)))
    image_array = _prepare_input_image(sample_record.image_path, image_size)

    report: dict[str, object] = {
        "source_run_id": export_manifest["source_run_id"],
        "sample_image_path": sample_record.image_path,
        "benchmarks": {},
    }

    for quantization, export_info in export_manifest["exports"].items():
        tflite_path = Path(export_info["tflite_path"])
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        input_tensor = _quantize_input_if_needed(input_details, image_array)

        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()

        durations_ms: list[float] = []
        for _ in range(num_runs):
            interpreter.set_tensor(input_details["index"], input_tensor)
            start = time.perf_counter()
            interpreter.invoke()
            end = time.perf_counter()
            durations_ms.append((end - start) * 1000.0)

        output_tensor_bytes = sum(_tensor_bytes(detail) for detail in output_details)
        report["benchmarks"][quantization] = {
            "tflite_path": str(tflite_path),
            "model_size_bytes": int(tflite_path.stat().st_size),
            "model_size_mb": round(float(tflite_path.stat().st_size) / (1024 * 1024), 4),
            "input_tensor_dtype": str(input_details["dtype"]),
            "input_tensor_shape": _plain_shape(input_details["shape"]),
            "input_tensor_bytes": _tensor_bytes(input_details),
            "output_tensor_bytes": int(output_tensor_bytes),
            "estimated_working_set_bytes": _tensor_bytes(input_details) + output_tensor_bytes,
            "latency_ms": {
                "runs": [round(value, 4) for value in durations_ms],
                "avg": round(sum(durations_ms) / len(durations_ms), 4),
                "min": round(min(durations_ms), 4),
                "max": round(max(durations_ms), 4),
            },
        }

    return report
