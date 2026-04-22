# iOS Minimal TFLite Integration Example

This example shows the smallest practical iOS app structure for a model bundle exported by this repository.

## Purpose

Use this example when you want to:

- copy a generated iOS bundle into an app target
- load `model.tflite`, `label.txt`, and `export_metadata.json`
- run one image through the detector
- decode anchor-based boxes and apply score filtering and NMS

## Expected Bundle Files

Copy the contents of one exported bundle such as:

`artifacts/experiments/<experiment_id>/<run_id>/mobile/ios/float32/`

into:

`App/Resources/`

The example expects these files:

- `model.tflite`
- `label.txt`
- `label_map.json`
- `export_metadata.json`
- `benchmark_report.json`
- `MODEL_CARD.md`
- `license_metadata.json`
- `sample_input.jpg`

## Project Notes

- This directory contains SwiftUI source files and resource layout for a minimal sample
- Add the TensorFlow Lite iOS package to your Xcode project before building
- The detector uses direct interpreter calls because the exported model exposes raw class and bbox heads

## What The App Does

- starts a minimal SwiftUI screen
- loads the exported model from the app bundle
- reads anchors and thresholds from `export_metadata.json`
- runs inference on `sample_input.jpg`
- renders the top detections as text

## Suggested Xcode Setup

- Create a new iOS App project in Xcode
- Copy the files from `App/` into the app target
- Copy the files from `App/Resources/` into the main bundle
- Add the TensorFlow Lite Swift package and import `TensorFlowLite`

## Notes

- This is a minimal integration example, not a production app
- For camera capture, image orientation handling, live overlays, and frame throttling, build on top of this example rather than replacing it
