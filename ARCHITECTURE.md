# Tensor Training Core Architecture Plan

## Goal

Build a TensorFlow Lite object detection training project that can:

- run locally on Apple Silicon MacBook Pro M4 Pro for development, data preparation, lightweight training, evaluation, and export
- run inside Docker on x64 Linux with NVIDIA CUDA GPU for larger-scale training
- use a MobileNet-based object detection model as the first baseline
- export deployable TensorFlow Lite models for iOS and Android apps
- import COCO-format datasets and convert them into a training-ready internal manifest format
- expose a stable CLI, API, and agent-friendly skill contract only after the main Python module workflow is complete

## Definitions

- `dataset`: a set of images plus labeling files that can be imported into training
- `internal manifest`: the project-owned normalized metadata format derived from the imported dataset for training, validation, export, and reproducibility

This project should use `dataset` as the primary user-facing term. `internal manifest` is only an internal system format and should not replace the meaning of dataset.

## First Baseline Scope

The first deliverable should focus on one narrow and reliable path:

- model family: `ssd_mobilenet_v2_fpnlite_320x320`
- training framework: TensorFlow 2.x with TensorFlow Models Object Detection API as the main baseline
- export target: TensorFlow Lite `.tflite`
- mobile deployment targets:
  - iOS
  - Android
- dataset input format: COCO annotations with images
- internal prepared output:
  - normalized training manifest
  - split metadata for train / val / test
  - model-ready training records such as TFRecord

This path is chosen because MobileNet-based detectors are lighter, more practical for mobile inference, and easier to optimize for on-device use than heavier object detection backbones.

## Delivery Strategy

### Phase 1

Build the full workflow as Python modules first:

- dataset import
- dataset validation
- internal manifest generation
- training
- evaluation
- TensorFlow Lite export
- mobile bundle generation
- inference verification

### Phase 2

After the Python module workflow is validated end to end, add external interfaces:

- CLI
- HTTP API
- `SKILL.md`

This ordering is intentional. External interfaces should wrap a stable Python module core rather than defining behavior before the core workflow is proven.

## Target Use Cases

1. Local development on macOS Apple Silicon
2. Reproducible training in Docker on x64 CUDA environments
3. COCO dataset import and conversion
4. Export from trained checkpoint to TensorFlow Lite
5. Deployment-ready packaging for iOS and Android apps
6. Basic post-training validation for object detection inference quality
7. External integration through CLI, API, and AI-agent skill interface after the core Python modules are stable

## Design Principles

- single codebase, multiple runtimes
- MobileNet-first architecture for faster mobile deployment
- Python module workflow first, external interfaces second
- training pipeline separated from environment-specific setup
- configuration-driven execution
- reproducible experiments and outputs
- clear separation between raw dataset, prepared internal manifest, training artifacts, and mobile export artifacts

## Recommended Technical Direction

### Training Stack

- Primary framework: TensorFlow 2.x
- Baseline model family: `ssd_mobilenet_v2_fpnlite_320x320`
- Baseline training path: TensorFlow Models Object Detection API
- Export target: TensorFlow Lite
- Configuration format: YAML
- Core implementation style: reusable Python modules

### Runtime Strategy

- macOS Apple Silicon:
  - use native Python virtual environment
  - support `tensorflow-macos` and `tensorflow-metal` when applicable
  - focus on development, data checking, small experiments, export, and evaluation
- x64 CUDA:
  - use Docker as the standard training environment
  - include pinned CUDA, cuDNN, Python, and pip dependencies
  - use this environment for heavier training workloads

## Proposed Repository Structure

```text
tensor-training-core/
├── ARCHITECTURE.md
├── TODO.md
├── README.md
├── SKILL.md
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── macos.txt
│   └── cuda.txt
├── configs/
│   ├── datasets/
│   │   └── coco_detection.yaml
│   ├── models/
│   │   └── ssd_mobilenet_v2_fpnlite_320.yaml
│   ├── training/
│   │   └── baseline.yaml
│   └── experiments/
│       ├── dev_macos.yaml
│       └── train_cuda.yaml
├── docker/
│   ├── Dockerfile.cuda
│   ├── docker-compose.cuda.yml
│   └── entrypoint.sh
├── scripts/
│   ├── bootstrap_macos.sh
│   ├── bootstrap_cuda.sh
│   ├── prepare_coco.sh
│   ├── train.sh
│   ├── evaluate.sh
│   ├── export_tflite.sh
│   └── package_mobile_assets.sh
├── src/
│   └── tensor_training_core/
│       ├── __init__.py
│       ├── api/
│       │   ├── app.py
│       │   ├── schemas.py
│       │   └── routes/
│       │       ├── datasets.py
│       │       ├── training.py
│       │       ├── exports.py
│       │       └── health.py
│       ├── cli.py
│       ├── config/
│       │   ├── loader.py
│       │   └── schema.py
│       ├── data/
│       │   ├── adapters/
│       │   │   ├── coco.py
│       │   │   ├── pascal_voc.py
│       │   │   └── yolo.py
│       │   ├── converters/
│       │   │   ├── coco_to_manifest.py
│       │   │   └── manifest_to_tfrecord.py
│       │   ├── manifest/
│       │   │   ├── writer.py
│       │   │   └── schema.py
│       │   ├── validation.py
│       │   └── split.py
│       ├── interfaces/
│       │   ├── service.py
│       │   ├── jobs.py
│       │   └── dto.py
│       ├── models/
│       │   ├── factory.py
│       │   └── mobilenet/
│       ├── training/
│       │   ├── runner.py
│       │   ├── callbacks.py
│       │   ├── losses.py
│       │   └── metrics.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── reports.py
│       ├── export/
│       │   ├── saved_model.py
│       │   ├── tflite.py
│       │   ├── metadata.py
│       │   └── mobile_bundle.py
│       ├── inference/
│       │   ├── tflite_runner.py
│       │   └── visualize.py
│       ├── mobile/
│       │   ├── android/
│       │   │   └── bundle_writer.py
│       │   └── ios/
│       │       └── bundle_writer.py
│       └── utils/
│           ├── paths.py
│           ├── logging.py
│           └── seed.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── artifacts/
│   ├── checkpoints/
│   ├── exported/
│   ├── mobile/
│   ├── reports/
│   └── logs/
└── notebooks/
    └── exploration.ipynb
```

## Core Architecture

### 1. Config Layer

Responsibility:

- centralize dataset, model, training, export, and mobile packaging settings
- allow the same pipeline to run in macOS and Docker with environment overrides

Suggested config groups:

- dataset path and COCO annotation paths
- class mapping and label map
- image size and augmentation
- MobileNet detector checkpoint and pipeline settings
- optimizer, learning rate, batch size, epochs
- export settings such as float16 or int8 quantization
- mobile bundle settings for iOS and Android
- runtime selection such as `macos`, `cuda`, `cpu`

### 2. Data Layer

Responsibility:

- ingest COCO source datasets
- validate image and annotation consistency
- normalize labels and schema
- convert to a unified internal manifest
- generate training-ready records

Initial supported input:

- COCO JSON annotations with image folder structure

Initial prepared output:

- normalized internal manifest in JSONL
- label map
- train / val / test split files
- TFRecord or model-specific training input format

Pipeline:

1. Import raw COCO dataset
2. Validate paths, categories, image ids, and box coordinates
3. Normalize into internal manifest format
4. Split into train / val / test
5. Convert to TFRecord or model-ready input format

The internal manifest should be treated as a stable project-owned schema so the training pipeline is not tightly coupled to raw COCO JSON forever.

### 3. Model Layer

Responsibility:

- abstract the chosen MobileNet detection baseline
- keep future expansion possible without rewriting the whole pipeline

Baseline recommendation:

- start with `ssd_mobilenet_v2_fpnlite_320x320`
- keep the model factory open for future additions such as larger MobileNet variants or EfficientDet-Lite

This layer should expose:

- model selection
- pretrained checkpoint loading
- model-specific preprocessing and postprocessing
- TensorFlow Lite export compatibility constraints

### 4. Training Layer

Responsibility:

- run training jobs from configs
- support callbacks, checkpoints, logs, and resume behavior

Key features:

- deterministic seed handling where possible
- TensorBoard metrics logging
- checkpoint save and best-checkpoint tracking
- mixed precision on supported hardware
- experiment-based artifact directories

### 5. Evaluation Layer

Responsibility:

- compute validation metrics after training
- generate reports for comparison across experiments

Suggested outputs:

- mAP
- precision / recall
- per-class metrics
- sample prediction visualizations
- model size
- latency benchmark summary
- memory usage summary when benchmark data is available

### 6. Export Layer

Responsibility:

- convert trained model to SavedModel
- export to TensorFlow Lite
- optionally apply float16 or int8 quantization
- generate label metadata and deployment manifest

Suggested outputs:

- `.tflite`
- label map file
- export metadata JSON
- mobile deployment manifest
- validation report for exported model

### 7. Mobile Deployment Layer

Responsibility:

- package exported assets in a format easy to consume by iOS and Android teams
- document model input size, normalization, labels, score threshold, and NMS expectations

Suggested outputs:

- `artifacts/mobile/ios/`
  - model file
  - label file
  - metadata JSON
  - integration notes
- `artifacts/mobile/android/`
  - model file
  - label file
  - metadata JSON
  - integration notes

The generated mobile bundle should make the handoff to app development straightforward rather than only producing a raw `.tflite` file.

### 8. Inference Verification Layer

Responsibility:

- run smoke tests on exported TFLite models
- verify input/output tensor compatibility
- visualize sample predictions
- confirm exported model behavior matches mobile deployment assumptions

This is important because training success does not always mean TFLite deployment success.

### 9. External Interface Layer

Responsibility:

- expose the completed Python module workflow to other systems without forcing them to call internal modules directly
- provide one consistent service layer for future CLI, API, and AI-agent integrations
- make third-party orchestration predictable for jobs such as dataset import, training, export, and artifact retrieval

Design rule:

- business logic should first exist as Python modules
- CLI, API, and agent integration should call the same application service layer
- business logic should not live directly inside HTTP routes or CLI command handlers

Suggested internal contract:

- `interfaces/service.py` as the orchestration layer
- `interfaces/dto.py` for typed request and response payloads
- `interfaces/jobs.py` for long-running training or export job state

### 10. Experiment Tracking And Data Versioning Layer

Responsibility:

- record which dataset version, config snapshot, code version, and runtime environment produced each training run
- make every exported model traceable back to its source inputs

Suggested tracked metadata:

- dataset version id
- experiment id
- model family and checkpoint source
- training hyperparameters
- export settings such as quantization mode
- output artifact paths
- runtime environment such as `macos` or `cuda`

This layer is critical for reproducibility and should be treated as part of the core system, not as an optional reporting feature.

### 11. Data Quality And Cleaning Layer

Responsibility:

- catch common annotation and asset problems before training starts
- produce actionable reports for bad data rather than failing deep inside the training stack

Suggested checks:

- missing image files
- unreadable or corrupt images
- empty annotations
- invalid category ids
- negative or out-of-bounds bounding boxes
- duplicate or suspicious overlapping boxes
- class imbalance summaries

### 12. Inference Spec And Deployment Contract Layer

Responsibility:

- standardize preprocessing and postprocessing behavior across training, export, iOS, and Android
- prevent mobile-side integration drift

Deployment spec should define:

- input tensor size
- input normalization rules
- label ordering
- score threshold defaults
- non-max suppression assumptions
- output tensor format
- bounding box coordinate conventions

This spec should be emitted as part of the mobile deployment bundle.

### 13. Observability Layer

Responsibility:

- make training, export, and future API jobs debuggable in local and containerized environments

Suggested capabilities:

- structured logging
- job event timeline
- run status summary
- error summaries with root-cause hints
- optional metrics export for job duration and failures

### 14. Logging Strategy

Responsibility:

- provide one consistent log format across Python module runs first, then extend that same format to CLI, export jobs, and HTTP API requests
- make it easy to debug failures in local macOS runs and Docker CUDA runs

Recommended rollout:

- start with module-level application logs and error summaries
- expand logging fields and event types as each new stage becomes real
- add CLI and API-specific correlation fields only when those interfaces are introduced

Recommended log outputs:

- console logs for interactive development
- JSONL log files for machine-readable job history
- TensorBoard logs for training metrics
- per-run error summary files for failed jobs

Recommended log locations:

- `artifacts/logs/<run_id>/application.jsonl`
- `artifacts/logs/<run_id>/errors.log`
- `artifacts/logs/<run_id>/training.tensorboard/`

Recommended log fields:

- timestamp
- level
- run id
- experiment id
- dataset version
- job id when applicable
- component such as `dataset`, `training`, `export`, `api`
- message
- error type
- traceback location when available

Recommended logging events:

- dataset import started / completed / failed
- dataset validation warnings
- training started / checkpoint saved / completed / failed
- export started / completed / failed
- mobile bundle generated
- later phases: CLI request received / API request received / job status changed

Design rules:

- use structured logs by default and allow optional human-readable console formatting
- every long-running job must have a stable `run_id` or `job_id`
- expand logging coverage together with each completed feature stage
- error logs should link back to the generated artifact or failed stage when possible

### 15. Quality Automation Layer

Responsibility:

- protect the codebase from regressions as the project grows

Suggested capabilities:

- linting
- unit tests
- tiny end-to-end integration tests
- Docker smoke tests
- CLI contract checks in phase 2
- API smoke tests in phase 2

### 16. Optional Near-Term Extensions

These are not required for the first training milestone, but they are important enough to keep visible in the architecture:

- representative dataset pipeline for int8 quantization
- asynchronous job queue for training and export requests
- model registry for multiple model families or released checkpoints
- iOS and Android sample app integration
- export-time model card and license notice generation

## Python Module Planning

The Python modules are the first operator surface and should define the authoritative behavior of the system.

Core module goals:

- mirror the main workflow step by step
- expose reusable functions and service objects
- keep later CLI and API layers thin

Recommended core module groups:

- dataset import and validation
- manifest generation
- training run orchestration
- evaluation
- export and mobile packaging
- artifact inspection

## API Planning

The API should be added only after the first end-to-end Python module pipeline is stable.

Recommended stack:

- FastAPI for HTTP service
- Pydantic models for request and response schemas

Initial API scope:

- `GET /health`
- `POST /datasets/import/coco`
- `POST /datasets/prepare`
- `POST /training/jobs`
- `GET /training/jobs/{job_id}`
- `POST /exports/tflite`
- `POST /exports/mobile-bundle`
- `GET /artifacts/{artifact_id}`

Expected API behavior:

- call the shared service layer instead of embedding business logic in routes
- accept config file paths or structured payloads
- return job ids for long-running operations
- expose artifact locations and execution status
- remain usable in local development and containerized deployment
- leave room for authentication and authorization if the API later serves shared environments

## CLI Planning

The CLI should be added in phase 2 after the Python module workflow is proven.

CLI design goals:

- mirror the stable Python module workflow
- support both config-driven and argument-driven execution
- produce machine-readable outputs when needed for automation

Recommended CLI groups:

- `dataset import-coco`
- `dataset prepare`
- `train run`
- `train status`
- `evaluate run`
- `export tflite`
- `export mobile`
- `artifact list`
- `artifact describe`
- `serve api`

Recommended automation options:

- `--config`
- `--output-json`
- `--job-name`
- `--runtime`
- `--artifact-dir`

## Skill Planning For Third-Party Platforms And AI Agents

Add a repository-level `SKILL.md` only after the main Python module flow is stable and the external interfaces are defined.

Purpose of `SKILL.md`:

- describe what this project can do
- define the supported inputs and outputs
- document safe invocation patterns
- help AI agents and orchestration platforms choose the correct CLI or API entrypoint

The skill should cover:

- supported tasks such as COCO import, training, export, and mobile packaging
- required inputs such as dataset path, config path, runtime target, and output path
- expected artifacts such as internal manifest, checkpoints, `.tflite`, and mobile bundles
- constraints such as hardware requirements, long-running job expectations, and license reminders for datasets
- example invocations for both CLI and API

Recommended skill style:

- keep it operational rather than marketing-oriented
- make every task map to a stable command or endpoint
- document failure modes and preconditions clearly

## Additional Important Capabilities

The following items were identified as important and should remain visible in planning even if they are not all part of the first implementation wave.

### Must-Have Additions

- experiment tracking and dataset versioning
- annotation quality checks and data cleaning reports
- standardized inference preprocessing and postprocessing spec
- representative dataset planning for future int8 quantization
- observability for jobs, logs, and failures
- basic CI/CD automation for linting and smoke tests

### Strongly Recommended Additions

- iOS and Android minimal integration examples
- latency, size, and memory benchmark reporting
- asynchronous job execution model for long-running API requests
- model registry and model version tracking
- retry and resume strategy for failed runs
- export-time model card and license metadata

### Good Later Additions

- advanced augmentation policy library
- API authentication and access control
- multiple detector family expansion
- richer artifact browser or dashboard

## Cross-Platform Environment Plan

### macOS Apple Silicon

Use cases:

- code development
- COCO data conversion
- config tuning
- small training runs
- export and validation

Recommended approach:

- Python virtual environment
- separate dependency lock for macOS
- optional Metal acceleration

### x64 Linux + CUDA Docker

Use cases:

- main training environment
- long-running experiments
- larger batch sizes

Recommended approach:

- one Dockerfile dedicated to CUDA training
- volume mount `data/` and `artifacts/`
- env vars for GPU visibility and experiment names

## Execution Flow

### Standard Python Module Flow

1. Put COCO dataset under `data/raw`
2. Run COCO validation and normalization
3. Generate internal manifest and training records
4. Create or select MobileNet experiment config
5. Train model
6. Evaluate checkpoint
7. Export best model to TFLite
8. Generate iOS / Android mobile bundles
9. Run TFLite inference smoke test
10. Save reports and artifacts

### Example Python Module Usage

```python
from tensor_training_core.interfaces.service import TrainingService

service = TrainingService()
service.import_coco_dataset("configs/datasets/coco_detection.yaml")
service.prepare_dataset("configs/experiments/dev_macos.yaml")
service.train("configs/experiments/dev_macos.yaml")
service.evaluate("configs/experiments/dev_macos.yaml")
service.export_tflite("configs/experiments/dev_macos.yaml")
service.package_mobile_bundle("configs/experiments/dev_macos.yaml")
```

### Example Future CLI Commands

```bash
python -m tensor_training_core.cli dataset import-coco --config configs/datasets/coco_detection.yaml
python -m tensor_training_core.cli dataset prepare --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli train run --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli evaluate run --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli export tflite --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli export mobile --config configs/experiments/dev_macos.yaml
python -m tensor_training_core.cli artifact describe --artifact artifacts/exported/model.tflite
python -m tensor_training_core.cli serve api
```

## Artifact and Experiment Management

Each experiment should have a unique output folder, for example:

```text
artifacts/
└── experiments/
    └── 2026-04-21_ssd_mobilenet_v2_fpnlite_baseline/
        ├── config.snapshot.yaml
        ├── checkpoints/
        ├── logs/
        ├── eval/
        ├── export/
        └── mobile/
```

Recommended rules:

- snapshot the exact config used for each run
- separate dataset artifacts from model artifacts
- never overwrite exported models without versioning

## Testing Strategy

### Unit Tests

- config parsing
- COCO annotation conversion
- internal manifest generation
- bounding box validation
- model factory behavior
- export utility behavior

### Integration Tests

- small COCO-format fixture dataset end-to-end
- train for 1 epoch on toy data
- export a tiny TFLite detection model
- create mobile deployment bundle
- run inference smoke test
- phase 2: API smoke test
- phase 2: CLI contract test for machine-readable output
- CI smoke test across local and Docker-supported workflows

## Phased Delivery Plan

### Phase 1. Python Module Foundation

- repo scaffolding
- config system
- core Python modules
- dependency split for macOS and CUDA
- Docker baseline

### Phase 2. COCO Data Pipeline

- COCO importer
- internal manifest schema
- validators
- split generation
- TFRecord generation

### Phase 3. MobileNet Training Baseline

- MobileNet detection model integration
- training runner
- logging and checkpointing
- baseline configs for macOS and CUDA

### Phase 4. TFLite Export and Mobile Packaging

- SavedModel export
- TFLite conversion
- iOS / Android artifact bundle generation
- inference smoke tests
- metadata generation

### Phase 5. Quality and DX

- tests
- sample dataset
- documentation
- reproducible scripts

### Phase 6. External Interfaces

- shared service layer for wrapping the proven core modules
- CLI surface
- API endpoints
- `SKILL.md` for third-party platforms and AI agents

### Phase 7. Operational Maturity

- experiment tracking and dataset versioning
- data quality reports
- inference deployment spec generation
- observability and CI automation

## Risks and Early Decisions

### Decision 1: Exact MobileNet detector variant

Locked for phase 1:

- `ssd_mobilenet_v2_fpnlite_320x320`

Later alternatives:

- other MobileNet variants
- EfficientDet-Lite

### Decision 2: Training API ownership

Initial plan:

- use TensorFlow Models Object Detection API for the first baseline

Possible later evolution:

- wrap the baseline behind our own runner so we can swap internal implementation later without breaking project structure

### Decision 3: Quantization strategy

Need to decide whether initial release targets:

- float32 only
- float16 export
- full int8 quantization with calibration dataset

For the first milestone, float16 export is a practical starting point for mobile deployment, with int8 evaluated afterward if latency or size demands it.

## Recommended First Milestone

Build the smallest end-to-end Python module slice:

1. one COCO dataset import path
2. one internal manifest format
3. one MobileNet detection model
4. one training module
5. one TFLite export module
6. one mobile bundle generator for iOS and Android
7. one inference verification module

If this slice works on both macOS and CUDA Docker, the rest of the project can be expanded safely.

## Initial Recommendation Summary

- use one Python package with config-driven module design
- lock the first baseline to `ssd_mobilenet_v2_fpnlite_320x320`
- treat macOS as development and lightweight execution environment
- treat Docker CUDA as the main heavy-training environment
- make COCO import and internal manifest generation part of the first milestone
- prioritize iOS / Android deployment bundle output from the start
- add CLI, API, and `SKILL.md` only after the Python module workflow is stable
- expand logging incrementally as each stage becomes real
