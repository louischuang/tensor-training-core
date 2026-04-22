# iOS Minimal Integration

This document describes the smallest supported iOS integration path for a model exported by this repository.

## Example Project

See:

- `examples/ios/minimal-tflite-app/`

The example is intentionally small:

- one SwiftUI app entry point
- one content view
- one detector helper
- resources copied from an exported iOS bundle

## Bundle Source

Start from a generated bundle such as:

`artifacts/experiments/<experiment_id>/<run_id>/mobile/ios/float32/`

Copy the bundle files into:

`examples/ios/minimal-tflite-app/App/Resources/`

## Required Resources

- `model.tflite`
- `label.txt`
- `label_map.json`
- `export_metadata.json`
- `benchmark_report.json`
- `MODEL_CARD.md`
- `license_metadata.json`

Recommended for the demo app:

- `sample_input.jpg`

## Runtime Contract

The example reads these values from `export_metadata.json`:

- `image_size`
- `anchors`
- `postprocessing.nms.score_threshold`
- `postprocessing.nms.iou_threshold`

The detector:

1. resizes an input image to the exported image size
2. converts RGB values into the model input tensor
3. runs the TensorFlow Lite interpreter
4. identifies class logits and bbox offsets from the two raw output heads
5. decodes boxes relative to anchors
6. applies score filtering and NMS

## What This Example Is Good For

- validating that the exported iOS bundle is self-consistent
- showing the minimum files required by an iOS client
- giving iOS developers a code starting point for asset loading and raw output decoding

## What You Still Need For Production

- camera capture pipeline
- image orientation handling
- background queue management
- overlay rendering
- device-specific performance tuning
- stronger error handling around bundle mismatches
