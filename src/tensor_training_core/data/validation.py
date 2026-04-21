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
    for image in coco.get("images", []):
        if not (images_path / image["file_name"]).exists():
            missing_images.append(image["file_name"])

    return DatasetValidationResult(
        image_count=len(coco.get("images", [])),
        annotation_count=len(coco.get("annotations", [])),
        category_count=len(coco.get("categories", [])),
        missing_images=missing_images,
    )
