from __future__ import annotations

import json
from pathlib import Path

from tensor_training_core.data.manifest.schema import ManifestRecord


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ManifestRecord.model_validate(json.loads(line)))
    return records
