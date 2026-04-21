from __future__ import annotations

import argparse

from tensor_training_core.interfaces.service import TrainingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-1 Python module runner.")
    parser.add_argument(
        "command",
        choices=[
            "import-coco",
            "prepare-dataset",
            "train",
            "evaluate",
            "export-tflite",
            "package-mobile",
            "verify-inference",
        ],
    )
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = TrainingService()
    handlers = {
        "import-coco": service.import_coco_dataset,
        "prepare-dataset": service.prepare_dataset,
        "train": service.train,
        "evaluate": service.evaluate,
        "export-tflite": service.export_tflite,
        "package-mobile": service.package_mobile_bundle,
        "verify-inference": service.verify_inference,
    }
    result = handlers[args.command](args.config)
    print(f"{result.name}: {result.status} - {result.message}")


if __name__ == "__main__":
    main()
