# COCO Dataset Import Flow

## Inputs

A COCO dataset in this repository is treated as:

- image files under an image directory
- one COCO annotations JSON file

## Flow

1. `dataset import-coco`
   - validates image references and counts annotations/categories
2. `dataset prepare`
   - converts the dataset into the internal manifest
   - writes label map, metadata, quality report, and split manifests

## Example

```bash
python -m tensor_training_core.cli dataset import-coco --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```
