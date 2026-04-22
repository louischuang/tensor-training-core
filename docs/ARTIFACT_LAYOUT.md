# Experiment Artifact Layout

## Main Locations

- `artifacts/experiments/<experiment_id>/<run_id>/`
- `artifacts/reports/<experiment_id>/<run_id>/`
- `artifacts/logs/<run_id>/`
- `artifacts/jobs/<job_id>.json`

## Typical Training Run

- `checkpoints/latest.keras` or `latest.ckpt`
- `training_metrics.jsonl`
- `training_summary.json`
- `application.jsonl`

## Typical Export Run

- `export/saved_model/`
- `export/*.tflite`
- `export/export_manifest.json`
- `export/export_metadata_*.json`
- `export/label.txt`

## Typical Mobile Run

- `mobile/android/<quantization>/`
- `mobile/ios/<quantization>/`
- `INTEGRATION.md`
- `bundle_verification.json`
