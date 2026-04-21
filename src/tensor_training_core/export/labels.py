from __future__ import annotations

import json
from pathlib import Path


def load_label_map(label_map_path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in payload.items()}


def build_label_lines(label_map: dict[int, str]) -> list[str]:
    foreground_ids = sorted(label_id for label_id in label_map.keys() if label_id > 0)
    return [label_map[label_id] for label_id in foreground_ids]


def write_label_txt(output_path: str | Path, label_map_path: str | Path) -> Path:
    label_map = load_label_map(label_map_path)
    label_lines = build_label_lines(label_map)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return output_file
