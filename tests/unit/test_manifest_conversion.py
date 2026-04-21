from __future__ import annotations

from tensor_training_core.data.converters.coco_to_manifest import convert_coco_dict_to_manifest_records


def test_convert_coco_dict_to_manifest_records() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 320, "height": 240}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    records = convert_coco_dict_to_manifest_records(coco)
    assert len(records) == 1
    assert records[0].category_name == "cat"
