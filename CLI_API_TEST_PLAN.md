# CLI And API Test Plan

## Purpose

This document tracks the current end-to-end test scope for the implemented CLI and API features.

The goal is to validate the shared phase-2 interface layer against the real `esp32-cam` dataset rather than only fixture data.

## Test Target

- Repository: `tensor-training-core`
- Shared dataset: `data/raw/esp32-cam`
- Primary experiment config: `configs/experiments/train_tensorflow_esp32_cam_dev.yaml`
- Execution style:
  - CLI tests through `tensor_training_core.cli`
  - API tests through FastAPI `create_app()` with real service calls
  - TensorFlow-heavy operations run in the Docker image `tensor-training-core-tf:latest`

## Test Dataset

- Dataset root: `data/raw/esp32-cam`
- Dataset format: COCO
- Config path: `configs/datasets/esp32_cam_train.yaml`
- Runtime experiment: `configs/experiments/train_tensorflow_esp32_cam_dev.yaml`

## CLI Test Scope

| ID | Command | Purpose | Expected Result | Status |
| --- | --- | --- | --- | --- |
| CLI-01 | `dataset import-coco --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Validate COCO import with real dataset | Job completes and returns image / annotation counts | Passed |
| CLI-02 | `dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Build manifests and quality report | Job completes and returns manifest / metadata / quality report paths | Passed |
| CLI-03 | `train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Run training on the real dataset dev config | Job completes and returns checkpoint / metrics / summary paths | Passed |
| CLI-04 | `train status --job-id <job_id>` | Read a created training job record | Existing job record is returned in JSON | Passed |
| CLI-05 | `evaluate run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Run evaluation and reports | Job completes and returns metrics / summary / report paths | Passed |
| CLI-06 | `export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Export TFLite and SavedModel artifacts | Job completes and returns SavedModel / TFLite / metadata paths | Passed |
| CLI-07 | `export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml` | Generate iOS and Android bundles | Job completes and returns bundle directories | Passed |
| CLI-08 | `artifact list --limit 5` | List recent jobs | JSON list of recent jobs is returned | Passed |
| CLI-09 | `artifact describe --artifact <path>` | Inspect generated artifact metadata | File metadata JSON is returned | Passed |
| CLI-10 | `serve api --host 127.0.0.1 --port 8010` | Start API server through CLI | Server starts successfully and `/health` returns `ok` | Passed |

## API Test Scope

| ID | Route | Purpose | Expected Result | Status |
| --- | --- | --- | --- | --- |
| API-01 | `GET /health` | Health check | `200` with `{\"status\":\"ok\"}` | Passed |
| API-02 | `POST /datasets/import/coco` | Import real COCO dataset | `200` with completed job payload | Passed |
| API-03 | `POST /datasets/prepare` | Prepare manifests from real dataset | `200` with completed job payload | Passed |
| API-04 | `POST /training/jobs` | Run training from real dataset config | `200` with completed job payload | Passed |
| API-05 | `GET /training/jobs/{job_id}` | Read existing training job | `200` with tracked job payload | Passed |
| API-06 | `POST /exports/tflite` | Export TFLite artifacts | `200` with completed export job payload | Passed |
| API-07 | `POST /exports/mobile-bundle` | Export mobile bundles | `200` with completed mobile job payload | Passed |
| API-08 | `GET /artifacts/{job_id}` | Read job-linked artifact metadata | `200` with stored job payload | Passed |

## Validation Targets

The test run should confirm the following outputs are actually generated:

- dataset quality report
- train / val / test manifests
- training checkpoint
- training summary
- evaluation summary
- evaluation report under `artifacts/reports/`
- SavedModel export
- TFLite `float32`, `float16`, and `int8`
- mobile bundles with `INTEGRATION.md` and `bundle_verification.json`
- job records under `artifacts/jobs/`
- API request log under `artifacts/logs/api/requests.jsonl`

## Result Summary

Execution completed on the real `esp32-cam` dataset.

- CLI overall: Passed
- API overall: Passed
- Dataset used: `esp32-cam`
- CLI result directory: `artifacts/tests/cli_api_real/cli`
- API result directory: `artifacts/tests/cli_api_real/api`
- Notes:
  - `CLI-06` was re-tested after suppressing TensorFlow stdout during export. The fixed result file is machine-readable JSON at `artifacts/tests/cli_api_real/cli/cli_06_export_tflite_fixed.json`.
  - API routes completed successfully through real HTTP requests, and request lifecycle events were appended to `artifacts/logs/api/requests.jsonl`.
  - Verified generated outputs include dataset quality report, split manifests, training checkpoint, evaluation report, SavedModel export, `float32` / `float16` / `int8` TFLite files, `label.txt`, and iOS / Android mobile bundle files including `INTEGRATION.md` and `bundle_verification.json`.
