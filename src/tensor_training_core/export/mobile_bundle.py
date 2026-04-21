from __future__ import annotations

import json

from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.mobile.android.bundle_writer import write_android_bundle
from tensor_training_core.mobile.ios.bundle_writer import write_ios_bundle
from tensor_training_core.utils.paths import get_latest_run_dir


def package_mobile_bundles(
    context: RunContext,
    experiment_id: str,
    label_map_path: str,
) -> dict[str, str]:
    latest_run_dir = get_latest_run_dir(
        experiment_id,
        required_relative_path="export/export_manifest.json",
        exclude_run_id=context.run_id,
    )
    export_dir = latest_run_dir / "export"
    export_manifest_path = export_dir / "export_manifest.json"
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))

    mobile_dir = context.artifact_dir / "mobile"
    outputs: dict[str, str] = {
        "source_run_id": latest_run_dir.name,
        "export_manifest_path": str(export_manifest_path),
    }
    for quantization, export_info in export_manifest["exports"].items():
        android_dir = write_android_bundle(
            output_dir=mobile_dir / "android" / quantization,
            tflite_path=export_info["tflite_path"],
            label_map_path=label_map_path,
            metadata_path=export_info["metadata_path"],
        )
        ios_dir = write_ios_bundle(
            output_dir=mobile_dir / "ios" / quantization,
            tflite_path=export_info["tflite_path"],
            label_map_path=label_map_path,
            metadata_path=export_info["metadata_path"],
        )
        outputs[f"android_bundle_dir_{quantization}"] = str(android_dir)
        outputs[f"ios_bundle_dir_{quantization}"] = str(ios_dir)
    return outputs
