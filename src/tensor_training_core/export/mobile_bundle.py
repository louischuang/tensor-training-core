from __future__ import annotations

import json
import traceback

from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.mobile.android.bundle_writer import write_android_bundle
from tensor_training_core.mobile.ios.bundle_writer import write_ios_bundle
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import get_latest_run_dir


def package_mobile_bundles(
    context: RunContext,
    experiment_id: str,
    label_map_path: str,
) -> dict[str, str]:
    logger = get_logger("mobile")
    failure_summary_path = context.log_dir / "failure_summary.json"
    try:
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
        logger.info(
            "mobile_packaging_started source_run_id=%s export_manifest_path=%s",
            latest_run_dir.name,
            export_manifest_path,
        )
        for quantization, export_info in export_manifest["exports"].items():
            android_dir = write_android_bundle(
                output_dir=mobile_dir / "android" / quantization,
                tflite_path=export_info["tflite_path"],
                label_map_path=label_map_path,
                metadata_path=export_info["metadata_path"],
                model_card_path=export_dir / "MODEL_CARD.md",
                license_metadata_path=export_dir / "license_metadata.json",
                benchmark_report_path=export_dir / "benchmark_report.json",
            )
            ios_dir = write_ios_bundle(
                output_dir=mobile_dir / "ios" / quantization,
                tflite_path=export_info["tflite_path"],
                label_map_path=label_map_path,
                metadata_path=export_info["metadata_path"],
                model_card_path=export_dir / "MODEL_CARD.md",
                license_metadata_path=export_dir / "license_metadata.json",
                benchmark_report_path=export_dir / "benchmark_report.json",
            )
            outputs[f"android_bundle_dir_{quantization}"] = str(android_dir)
            outputs[f"ios_bundle_dir_{quantization}"] = str(ios_dir)
            logger.info(
                "mobile_bundle_completed quantization=%s android_dir=%s ios_dir=%s",
                quantization,
                android_dir,
                ios_dir,
            )
        return outputs
    except Exception as exc:
        failure_summary_path.write_text(
            json.dumps(
                {
                    "run_id": context.run_id,
                    "experiment_id": experiment_id,
                    "stage": "mobile_package",
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                    "label_map_path": str(label_map_path),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.error("mobile_packaging_failed failure_summary_path=%s error=%s", failure_summary_path, exc)
        raise
