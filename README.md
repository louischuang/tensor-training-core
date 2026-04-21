# Tensor Training Core

Language versions: [English](./README.md) | [繁體中文](./README.zh-TW.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

Tensor Training Core is a TensorFlow Lite object detection training repository focused on one practical baseline:

- COCO-format dataset import
- MobileNet-based object detection training
- evaluation and report generation
- TensorFlow Lite export
- iOS and Android mobile asset bundles
- shared Python service layer, CLI, API, and agent-facing skill contract

## What This Repository Does

The repository is designed for one shared workflow that can run on:

- Apple Silicon macOS for development and validation
- x64 Linux Docker environments for heavier training

The current implemented flow is:

1. Import and validate a COCO-format dataset
2. Normalize it into an internal manifest
3. Split it into train / val / test manifests
4. Train a MobileNet-based detector
5. Evaluate the checkpoint and generate reports
6. Export TensorFlow Lite models
7. Package iOS and Android integration bundles
8. Verify inference on exported TFLite models

## Repository Structure

```text
configs/                 YAML configs for dataset, model, training, and experiments
data/                    Raw datasets, manifests, and dataset metadata
docker/                  Docker build and runtime files
scripts/                 Helper scripts for setup and execution
src/tensor_training_core/
  api/                   FastAPI application and routes
  config/                Config loader and schema
  data/                  Dataset adapters, validation, conversion, and split logic
  evaluation/            Evaluation metrics and reports
  export/                TFLite, SavedModel, labels, and mobile export helpers
  inference/             TFLite verification and preview rendering
  interfaces/            Shared service layer, DTOs, and job records
  mobile/                Android and iOS bundle writers
  training/              Training runner, callbacks, and augmentation
artifacts/               Generated experiments, reports, logs, jobs, and bundles
tests/                   Unit tests
```

## Core Concepts

- `dataset`: images plus labeling files that can be imported into training
- `internal manifest`: normalized project-owned metadata generated from an imported dataset
- `job`: a tracked execution record stored under `artifacts/jobs/`
- `artifact`: generated files stored under `artifacts/experiments/`, `artifacts/reports/`, and `artifacts/logs/`

## Current Implemented Features

### Dataset Pipeline

- COCO dataset validation
- annotation quality checks and cleaning report output
- manifest generation
- dataset metadata output
- train / val / test split generation

### Training

- MobileNet-based baseline detector
- TensorFlow training backend
- training logs and per-run artifacts
- pretrained checkpoint loading
- resume-training support
- checkpoint and TensorBoard output

### Evaluation

- precision / recall / mAP50 metrics
- evaluation preview images
- evaluation reports under `artifacts/reports/`

### Export and Mobile Packaging

- SavedModel export
- TFLite export for `float32`, `float16`, and `int8`
- `label.txt` generation
- mobile bundles for iOS and Android
- integration assumptions and bundle verification files

### Interfaces

- shared service layer for all operations
- package CLI
- FastAPI application skeleton with working routes
- `SKILL.md` for agent and third-party integration

## Quick Start

### 1. Install Dependencies

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

For Docker-based training:

```bash
docker build -t tensor-training-core-tf:latest -f docker/Dockerfile.cuda .
```

### 2. Prepare a Dataset

Example:

```bash
python -m tensor_training_core.cli dataset prepare --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 3. Run Training

```bash
python -m tensor_training_core.cli train run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 4. Run Evaluation

```bash
python -m tensor_training_core.cli evaluate run --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 5. Export TFLite

```bash
python -m tensor_training_core.cli export tflite --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

### 6. Package Mobile Assets

```bash
python -m tensor_training_core.cli export mobile --config configs/experiments/train_tensorflow_esp32_cam_dev.yaml
```

## CLI Commands

```text
tensor-training-core dataset import-coco --config <path>
tensor-training-core dataset prepare --config <path>
tensor-training-core train run --config <path>
tensor-training-core train status --job-id <id>
tensor-training-core evaluate run --config <path>
tensor-training-core export tflite --config <path>
tensor-training-core export mobile --config <path>
tensor-training-core artifact list --limit <n>
tensor-training-core artifact describe --artifact <path>
tensor-training-core serve api
```

Wrapper scripts are also available:

- `scripts/run_cli.sh`
- `scripts/serve_api.sh`
- `scripts/prepare_coco.sh`
- `scripts/train.sh`
- `scripts/evaluate.sh`
- `scripts/export_tflite.sh`
- `scripts/package_mobile_assets.sh`

## HTTP API

Current routes:

```text
GET  /health
POST /datasets/import/coco
POST /datasets/prepare
POST /training/jobs
POST /training/jobs/async
GET  /training/jobs/{job_id}
GET  /training/jobs/{job_id}/logs
GET  /training/jobs/{job_id}/logs/stream
POST /exports/tflite
POST /exports/mobile-bundle
GET  /artifacts/{job_id}
```

Example request:

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

Start the API:

```bash
python -m tensor_training_core.cli serve api
```

or:

```bash
scripts/serve_api.sh
```

Interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Repository guide: [API_DOCUMENTATION.md](/Users/louischuang/CompanyStorage/嵐奕科技有限公司/第四處 技術處/正在進行中的專案/20260421-MyOpenSource/tensor-training-core/API_DOCUMENTATION.md)

For live training progress over HTTP, prefer:

- `POST /training/jobs/async`
- `GET /training/jobs/{job_id}`
- `GET /training/jobs/{job_id}/logs`
- `GET /training/jobs/{job_id}/logs/stream`

## Important Output Locations

- Jobs: `artifacts/jobs/`
- Experiment runs: `artifacts/experiments/<experiment_id>/<run_id>/`
- Reports: `artifacts/reports/<experiment_id>/<run_id>/`
- Logs: `artifacts/logs/<run_id>/`
- API request log: `artifacts/logs/api/requests.jsonl`

Typical generated outputs include:

- checkpoints
- `training_summary.json`
- `evaluation_summary.json`
- evaluation preview images
- SavedModel export
- `.tflite` files
- `label.txt`
- mobile bundle files
- structured logs and failure summaries

## Key Documents

- [Architecture Plan](./ARCHITECTURE.md)
- [TODO List](./TODO.md)
- [Agent Skill Contract](./SKILL.md)

## Dataset and License Notes

This repository supports COCO-format import, but dataset usage still needs review:

- the COCO API and code tooling are permissively licensed
- the COCO dataset itself has separate terms of use
- image sources may have attribution or redistribution implications

Keep raw datasets separate from project-owned manifests and generated artifacts.

## Upstream Projects and License Review

The following upstream repositories are relevant to this project. Based on their published repository metadata and license files, these repositories use permissive licenses that are generally suitable for internal or commercial projects, subject to normal compliance requirements.

| Package / Project | Purpose in this repo | Repository | Observed license | Usage note |
| --- | --- | --- | --- | --- |
| TensorFlow | Core training and export framework | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | Permissive; commonly suitable for internal and commercial projects |
| TensorFlow Models | Object detection reference implementations | [tensorflow/models](https://github.com/tensorflow/models) | Apache-2.0 | Useful reference for future model alignment |
| KerasCV | Optional future CV utilities | [keras-team/keras-cv](https://github.com/keras-team/keras-cv) | Apache-2.0 | Optional future expansion |
| TensorFlow Lite Support | Mobile metadata and TFLite helper tooling | [tensorflow/tflite-support](https://github.com/tensorflow/tflite-support) | Apache-2.0 | Useful for mobile metadata workflows |
| COCO API / pycocotools | COCO parsing and validation | [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | Simplified BSD | Suitable for dataset conversion tooling |

## Status

This repository is no longer documentation-only. The baseline Python workflow, CLI, API skeleton, and agent-facing contract are implemented. The next steps are mostly around CI, documentation expansion, and deeper production hardening.

## License Reminder

This README is an engineering summary, not legal advice. Before shipping a product, do a final review of exact dependency versions, model assets, and dataset licenses actually used.
