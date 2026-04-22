# Experiment Artifact Layout

## Main Locations

- `artifacts/experiments/<experiment_id>/<run_id>/`
- `artifacts/reports/<experiment_id>/<run_id>/`
- `artifacts/logs/<run_id>/`
- `artifacts/jobs/<job_id>.json`
- `artifacts/models/index.json`
- `artifacts/models/<experiment_id>/<model_name>/index.json`
- `artifacts/models/<experiment_id>/<model_name>/<run_id>/model_version.json`

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
- `export/MODEL_CARD.md`
- `export/license_metadata.json`
- `export/benchmark_report.json`

## Model Registry Contract

- `artifacts/models/index.json`
  - global summary of registered models
- `artifacts/models/<experiment_id>/<model_name>/latest.json`
  - latest promoted descriptor for that model key
- `artifacts/models/<experiment_id>/<model_name>/index.json`
  - version history for that model key
- `artifacts/models/<experiment_id>/<model_name>/<run_id>/model_version.json`
  - immutable descriptor for one exported model version

## Typical Mobile Run

- `mobile/android/<quantization>/`
- `mobile/ios/<quantization>/`
- `INTEGRATION.md`
- `bundle_verification.json`
