from __future__ import annotations

import json
import shutil
from pathlib import Path

from tensor_training_core.export.labels import write_label_txt


def write_ios_bundle(
    output_dir: str | Path,
    tflite_path: str | Path,
    label_map_path: str | Path,
    metadata_path: str | Path,
    model_card_path: str | Path,
    license_metadata_path: str | Path,
    benchmark_report_path: str | Path,
) -> Path:
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_path, bundle_dir / "model.tflite")
    shutil.copy2(label_map_path, bundle_dir / "label_map.json")
    shutil.copy2(metadata_path, bundle_dir / "export_metadata.json")
    shutil.copy2(model_card_path, bundle_dir / "MODEL_CARD.md")
    shutil.copy2(license_metadata_path, bundle_dir / "license_metadata.json")
    shutil.copy2(benchmark_report_path, bundle_dir / "benchmark_report.json")
    write_label_txt(bundle_dir / "label.txt", label_map_path)
    assumptions_path = bundle_dir / "INTEGRATION.md"
    assumptions_path.write_text(
        "\n".join(
            [
                "# iOS Integration Assumptions",
                "",
                "- Load `model.tflite` with TensorFlow Lite or LiteRT on iOS.",
                "- Use `label.txt` for foreground labels shown in the app UI.",
                "- Input tensor expects RGB image resized to the metadata image size.",
                "- Preprocessing expects normalized float values or model-specific quantization from metadata.",
                "- Postprocessing should apply the score and IoU thresholds in `export_metadata.json`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    verification_path = bundle_dir / "bundle_verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "platform": "ios",
                "required_files": [
                    "model.tflite",
                    "label.txt",
                    "label_map.json",
                    "export_metadata.json",
                    "MODEL_CARD.md",
                    "license_metadata.json",
                    "benchmark_report.json",
                    "INTEGRATION.md",
                ],
                "status": "ready",
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle_dir
