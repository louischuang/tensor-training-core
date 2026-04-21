from __future__ import annotations

import shutil
from pathlib import Path


def write_ios_bundle(
    output_dir: str | Path,
    tflite_path: str | Path,
    label_map_path: str | Path,
    metadata_path: str | Path,
) -> Path:
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_path, bundle_dir / "model.tflite")
    shutil.copy2(label_map_path, bundle_dir / "label_map.json")
    shutil.copy2(metadata_path, bundle_dir / "export_metadata.json")
    return bundle_dir
