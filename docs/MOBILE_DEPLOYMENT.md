# iOS and Android Deployment Workflow

## Output Bundles

The repository packages deployment outputs under:

- `mobile/android/<quantization>/`
- `mobile/ios/<quantization>/`

## Included Files

- `model.tflite`
- `label.txt`
- `label_map.json`
- `export_metadata.json`
- `benchmark_report.json`
- `MODEL_CARD.md`
- `license_metadata.json`
- `INTEGRATION.md`
- `bundle_verification.json`

## Workflow

1. Train a model
2. Export TFLite
3. Package mobile bundles
4. Hand the bundle directory to the mobile application team

## Minimal Integration Examples

- Android example: `examples/android/minimal-tflite-app/`
- Android guide: `docs/ANDROID_MINIMAL_INTEGRATION.md`
- iOS example: `examples/ios/minimal-tflite-app/`
- iOS guide: `docs/IOS_MINIMAL_INTEGRATION.md`

## Notes

- `label.txt` is intended for mobile runtime label lookup
- `export_metadata.json` contains thresholds, model name, and deployment assumptions
- `benchmark_report.json` provides model size, tensor memory estimates, and latency snapshots
- `MODEL_CARD.md` and `license_metadata.json` should travel with the bundle for downstream handoff
- `bundle_verification.json` is the quick machine-readable readiness check
