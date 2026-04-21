from __future__ import annotations

from pathlib import Path
from typing import Any

from tensor_training_core.data.manifest.schema import ManifestRecord


def convert_coco_dict_to_manifest_records(
    coco_data: dict[str, Any],
    images_root: str | Path | None = None,
) -> list[ManifestRecord]:
    categories = {item["id"]: item["name"] for item in coco_data.get("categories", [])}
    images = {item["id"]: item for item in coco_data.get("images", [])}
    image_root_path = Path(images_root) if images_root is not None else None
    records: list[ManifestRecord] = []
    for annotation in coco_data.get("annotations", []):
        image = images.get(annotation["image_id"])
        if image is None:
            continue
        image_path = Path(image["file_name"])
        if image_root_path is not None:
            image_path = image_root_path / image_path
        records.append(
            ManifestRecord(
                image_id=image["id"],
                image_path=str(image_path),
                width=image["width"],
                height=image["height"],
                category_id=annotation["category_id"],
                category_name=categories.get(annotation["category_id"], "unknown"),
                bbox_xywh=annotation["bbox"],
            )
        )
    return records
