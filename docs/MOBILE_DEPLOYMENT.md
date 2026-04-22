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
- `INTEGRATION.md`
- `bundle_verification.json`

## Workflow

1. Train a model
2. Export TFLite
3. Package mobile bundles
4. Hand the bundle directory to the mobile application team

## Notes

- `label.txt` is intended for mobile runtime label lookup
- `export_metadata.json` contains thresholds, model name, and deployment assumptions
- `bundle_verification.json` is the quick machine-readable readiness check
