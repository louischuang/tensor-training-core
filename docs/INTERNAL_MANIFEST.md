# Internal Manifest Format

## Purpose

The internal manifest is the normalized project-owned representation used by training, evaluation, export, and inference verification.

## Current Storage

- manifest file: `data/manifests/*.jsonl`
- label map: `data/manifests/*_label_map.json`
- metadata: `data/manifests/*_metadata.json`

## Record Shape

Each JSONL row represents one annotation entry tied to one image and includes:

- `image_path`
- `width`
- `height`
- `category_id`
- `category_name`
- `bbox_xywh`

The manifest is intentionally simple so it can be reused by multiple pipeline stages.
