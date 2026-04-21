from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def load_coco_annotations(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
