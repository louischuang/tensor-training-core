from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tensor_training_core.data.manifest.schema import ManifestRecord


def write_manifest(records: Iterable[ManifestRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return path
