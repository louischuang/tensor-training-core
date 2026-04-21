from __future__ import annotations

import json

from PIL import Image

from tensor_training_core.data.validation import validate_coco_dataset


def test_validate_coco_dataset_reports_annotation_quality(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (100, 80), color=(255, 255, 255)).save(images_dir / "sample.jpg")
    coco_payload = {
        "images": [
            {"id": 1, "file_name": "sample.jpg", "width": 100, "height": 80},
        ],
        "categories": [
            {"id": 1, "name": "object"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [90, 10, 20, 20]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [5, 5, 0, 20]},
        ],
    }
    (annotations_dir / "instances.json").write_text(
        json.dumps(coco_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    result = validate_coco_dataset(
        dataset_root=dataset_root,
        annotations="annotations/instances.json",
        images_dir="images",
    )

    assert result.image_count == 1
    assert result.annotation_count == 4
    assert result.quality_report["issues"]["duplicate_annotation_count"] == 1
    assert result.quality_report["issues"]["out_of_bounds_annotation_count"] == 1
    assert result.quality_report["issues"]["invalid_bbox_dimension_count"] == 1
    assert result.quality_report["distribution"]["per_category_annotation_count"]["object"] == 4
