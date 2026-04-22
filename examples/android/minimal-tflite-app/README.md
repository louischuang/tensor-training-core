# Android Minimal TFLite Integration Example

This example shows the smallest practical Android app structure for a model bundle exported by this repository.

## Purpose

Use this example when you want to:

- copy a generated Android bundle into an app
- load `model.tflite`, `label.txt`, and `export_metadata.json`
- run one image through the detector
- decode anchor-based boxes and apply score filtering and NMS

## Expected Bundle Files

Copy the contents of one exported bundle such as:

`artifacts/experiments/<experiment_id>/<run_id>/mobile/android/float32/`

into:

`app/src/main/assets/`

The example expects these files:

- `model.tflite`
- `label.txt`
- `label_map.json`
- `export_metadata.json`
- `benchmark_report.json`
- `MODEL_CARD.md`
- `license_metadata.json`
- `sample_input.jpg`

## What The App Does

- starts a minimal activity
- loads the exported model from assets
- reads anchors and thresholds from `export_metadata.json`
- runs inference on `sample_input.jpg`
- renders the top detections as text

## Notes

- This is a minimal integration example, not a production app
- The sample uses direct TensorFlow Lite `Interpreter` calls because the exported model uses custom raw heads
- For camera preview, image rotation handling, and threaded frame pipelines, build on top of this example rather than replacing it
