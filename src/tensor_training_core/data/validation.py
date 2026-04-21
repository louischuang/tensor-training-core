from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensor_training_core.data.adapters.coco import load_coco_annotations
from tensor_training_core.utils.paths import resolve_repo_path


@dataclass(slots=True)
class DatasetValidationResult:
    image_count: int
    annotation_count: int
    category_count: int
    missing_images: list[str]
    quality_report: dict[str, object]


def ensure_path_exists(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def validate_coco_dataset(dataset_root: str | Path, annotations: str, images_dir: str) -> DatasetValidationResult:
    root = ensure_path_exists(dataset_root)
    annotations_path = ensure_path_exists(root / annotations)
    images_path = ensure_path_exists(root / images_dir)
    coco = load_coco_annotations(annotations_path)

    missing_images: list[str] = []
    image_lookup = {int(image["id"]): image for image in coco.get("images", [])}
    category_lookup = {int(category["id"]): str(category["name"]) for category in coco.get("categories", [])}
    for image in coco.get("images", []):
        if not (images_path / image["file_name"]).exists():
            missing_images.append(image["file_name"])

    duplicate_keys: set[tuple[int, int, tuple[float, float, float, float]]] = set()
    duplicate_annotation_count = 0
    invalid_bbox_format_count = 0
    invalid_bbox_dimension_count = 0
    out_of_bounds_annotation_count = 0
    missing_image_reference_count = 0
    unknown_category_annotation_count = 0
    empty_annotation_count = 0
    per_category_annotation_count: dict[str, int] = {}

    for annotation in coco.get("annotations", []):
        image = image_lookup.get(int(annotation.get("image_id", -1)))
        category_id = int(annotation.get("category_id", -1))
        category_name = category_lookup.get(category_id, f"unknown:{category_id}")
        per_category_annotation_count[category_name] = per_category_annotation_count.get(category_name, 0) + 1

        if image is None:
            missing_image_reference_count += 1
            continue
        if category_id not in category_lookup:
            unknown_category_annotation_count += 1

        bbox = annotation.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_bbox_format_count += 1
            continue

        x, y, width, height = [float(value) for value in bbox]
        if width <= 0.0 or height <= 0.0:
            invalid_bbox_dimension_count += 1
        if x < 0.0 or y < 0.0 or x + width > float(image["width"]) or y + height > float(image["height"]):
            out_of_bounds_annotation_count += 1
        if width == 0.0 or height == 0.0:
            empty_annotation_count += 1

        duplicate_key = (
            int(image["id"]),
            category_id,
            tuple(round(float(value), 4) for value in (x, y, width, height)),
        )
        if duplicate_key in duplicate_keys:
            duplicate_annotation_count += 1
        else:
            duplicate_keys.add(duplicate_key)

    quality_report = {
        "summary": {
            "image_count": len(coco.get("images", [])),
            "annotation_count": len(coco.get("annotations", [])),
            "category_count": len(coco.get("categories", [])),
            "clean_annotation_count": len(coco.get("annotations", []))
            - invalid_bbox_format_count
            - invalid_bbox_dimension_count
            - out_of_bounds_annotation_count
            - missing_image_reference_count,
        },
        "issues": {
            "missing_images": missing_images,
            "missing_image_count": len(missing_images),
            "missing_image_reference_count": missing_image_reference_count,
            "unknown_category_annotation_count": unknown_category_annotation_count,
            "invalid_bbox_format_count": invalid_bbox_format_count,
            "invalid_bbox_dimension_count": invalid_bbox_dimension_count,
            "empty_annotation_count": empty_annotation_count,
            "out_of_bounds_annotation_count": out_of_bounds_annotation_count,
            "duplicate_annotation_count": duplicate_annotation_count,
        },
        "distribution": {
            "per_category_annotation_count": dict(sorted(per_category_annotation_count.items())),
        },
        "recommended_actions": [
            "Review out_of_bounds annotations before formal training runs."
            if out_of_bounds_annotation_count
            else "No out_of_bounds annotations detected.",
            "Deduplicate repeated annotations in the source dataset."
            if duplicate_annotation_count
            else "No duplicate annotations detected.",
            "Fix bbox format or dimension issues before export."
            if invalid_bbox_format_count or invalid_bbox_dimension_count
            else "Bounding box format and dimensions look valid.",
        ],
    }

    return DatasetValidationResult(
        image_count=len(coco.get("images", [])),
        annotation_count=len(coco.get("annotations", [])),
        category_count=len(coco.get("categories", [])),
        missing_images=missing_images,
        quality_report=quality_report,
    )
