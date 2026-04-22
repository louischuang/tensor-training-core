# TODO

## Phase 1. Lock the Baseline

- [x] Confirm first baseline model as `ssd_mobilenet_v2_fpnlite_320x320`
- [x] Confirm first dataset input format as COCO
- [x] Confirm first deployment targets as iOS and Android with TensorFlow Lite
- [x] Confirm phase-1 export strategy as float32 or float16

## Phase 2. Project Foundation

- [x] Create `README.md` with project overview, quick start, repo links, license notes, and phase separation
- [x] Create `pyproject.toml` and initialize Python package metadata
- [x] Create base directory structure under `src/`, `configs/`, `docker/`, `scripts/`, `tests/`, `data/`, and `artifacts/`
- [x] Add `.gitignore` for Python, TensorFlow artifacts, datasets, logs, and notebooks
- [x] Define initial structured logging format and shared run identifiers for Python module flows
- [x] Define standard log directory layout under `artifacts/logs/`
- [x] Define required initial log fields for dataset, training, and export flows

## Phase 3. Environment Support

- [x] Create `requirements/base.txt`
- [x] Create `requirements/macos.txt` for Apple Silicon development
- [x] Create `requirements/cuda.txt` for x64 CUDA Docker training
- [x] Add `scripts/bootstrap_macos.sh`
- [x] Add `scripts/bootstrap_cuda.sh`
- [x] Create `docker/Dockerfile.cuda`
- [x] Create `docker/docker-compose.cuda.yml`
- [x] Add Docker entrypoint script

## Phase 4. Configuration System

- [x] Define YAML config structure for dataset, model, training, export, and mobile package settings
- [x] Implement config loader
- [x] Add config schema validation
- [x] Create sample configs under `configs/datasets/`, `configs/models/`, `configs/training/`, and `configs/experiments/`
- [x] Add config fields for dataset version, experiment id, and runtime metadata

## Phase 5. COCO Import and Internal Manifest Pipeline

- [x] Define unified internal manifest schema
- [x] Implement COCO dataset validator
- [x] Implement COCO importer
- [x] Normalize COCO categories and annotations into internal manifest format
- [x] Implement train / val / test split utility
- [x] Implement conversion from internal manifest to TFRecord or model-ready input format
- [x] Prepare small COCO-format fixture dataset for tests
- [x] Add dataset versioning strategy and dataset metadata manifest
- [x] Add annotation quality checks and data cleaning reports

## Phase 6. MobileNet Training Baseline

- [x] Add baseline model config for `ssd_mobilenet_v2_fpnlite_320x320`
- [x] Implement model factory
- [x] Support pretrained checkpoint loading
- [x] Implement training runner
- [x] Add checkpointing and TensorBoard metrics logging
- [x] Add resume-training support
- [x] Add baseline experiment config for macOS
- [x] Add baseline experiment config for CUDA Docker
- [x] Track experiment metadata for every training run
- [x] Write per-run training logs and failure summaries to `artifacts/logs/`

## Phase 7. Evaluation

- [x] Implement validation pipeline
- [x] Add mAP / precision / recall metrics output
- [x] Generate sample prediction visualizations
- [x] Save evaluation reports to `artifacts/reports/`

## Phase 8. TensorFlow Lite Export

- [x] Implement SavedModel export
- [x] Implement `.tflite` conversion
- [x] Support float16 quantization
- [x] Evaluate whether int8 quantization is needed after the first working prototype
- [x] Generate label map and export metadata manifest
- [x] Add representative dataset planning for future int8 quantization
- [x] Track export metadata for every exported artifact
- [x] Write export logs and error summaries to `artifacts/logs/`

## Phase 9. Mobile Deployment Packaging

- [x] Create mobile asset bundle layout for iOS
- [x] Create mobile asset bundle layout for Android
- [x] Generate model metadata, labels, thresholds, and input spec documents for mobile apps
- [x] Add packaging functions for mobile deployment artifacts
- [x] Document integration assumptions for iOS and Android apps
- [x] Emit standardized inference preprocessing / postprocessing deployment spec

## Phase 10. Inference Verification

- [x] Implement TFLite inference runner
- [x] Add sample-image smoke test for exported model
- [x] Validate model input / output tensor formats
- [x] Verify exported bundle assumptions against iOS / Android integration expectations
- [x] Add simple visualization utility for detection output

## Phase 11. Shared Service Layer For Phase 2

- [x] Create a shared application service layer for dataset, training, export, and artifact operations
- [x] Define typed request / response DTOs for future external interfaces
- [x] Add job-state abstraction for long-running training and export tasks
- [x] Ensure future CLI, API, and agent integrations call the same orchestration layer
- [x] Add shared job status and failure summary model for observability
- [x] Add shared logging helpers so later interfaces extend the same structured log schema

## Phase 12. CLI and Scripts

- [x] Create package CLI entrypoint
- [x] Add `dataset import-coco` command group
- [x] Add `dataset prepare` command
- [x] Add `train run` command
- [x] Add `train status` command
- [x] Add `evaluate run` command
- [x] Add `export tflite` command
- [x] Add `export mobile` command
- [x] Add `artifact list` command
- [x] Add `artifact describe` command
- [x] Add `serve api` command
- [ ] Extend logging with CLI-specific correlation fields
- [x] Add shell wrappers in `scripts/`

## Phase 13. HTTP API

- [x] Create FastAPI app skeleton
- [x] Add health endpoint
- [x] Add COCO import endpoint
- [x] Add dataset prepare endpoint
- [x] Add training job submission endpoint
- [x] Add training job status endpoint
- [x] Add TFLite export endpoint
- [x] Add mobile bundle export endpoint
- [x] Add artifact retrieval or artifact metadata endpoint
- [x] Add structured API logging and request-to-job correlation ids
- [x] Persist API request and job lifecycle logs to machine-readable files

## Phase 14. Skill For Third-Party Platforms And AI Agents

- [x] Finalize `SKILL.md` after CLI and API are defined
- [x] Document supported tasks and safe usage boundaries
- [x] Document required inputs, outputs, and artifact locations
- [x] Add CLI examples for AI-agent usage
- [x] Add API examples for third-party platform usage
- [x] Document hardware/runtime expectations and long-running job behavior
- [x] Document dataset and license caveats for agent users

## Phase 15. Testing

- [x] Add unit tests for config parsing
- [x] Add unit tests for COCO annotation conversion
- [x] Add unit tests for internal manifest generation
- [x] Add unit tests for bounding box validation
- [x] Add integration test for tiny COCO end-to-end pipeline
- [x] Add export smoke test
- [x] Add mobile bundle smoke test
- [x] Add phase-2 API smoke tests
- [x] Add phase-2 CLI contract tests for machine-readable output
- [x] Add baseline CI checks for lint, unit tests, integration smoke tests, and Docker smoke tests

## Phase 16. Documentation

- [x] Document local macOS setup flow
- [x] Document CUDA Docker training flow
- [x] Document COCO dataset import flow
- [x] Document internal manifest format
- [x] Document experiment artifact layout
- [x] Document iOS / Android deployment workflow
- [x] Document phase-2 API usage workflow
- [x] Document phase-2 CLI automation workflow
- [x] Document phase-2 `SKILL.md` usage for third-party platforms and agents

## First End-to-End Milestone

- [x] Import one COCO-format dataset
- [x] Produce one normalized internal manifest
- [x] Train one MobileNet-based detector
- [x] Export one working `.tflite` model
- [x] Generate iOS and Android deployment bundles
- [x] Pass one inference smoke test on the exported model
- [x] Confirm the Python module workflow is complete before adding external interfaces

## Second Integration Milestone

- [x] Expose the stable workflow through shared service interfaces
- [x] Provide a usable CLI for operators and automation
- [x] Provide a usable HTTP API for third-party platforms
- [x] Provide finalized `SKILL.md` for AI-agent and platform integration

## Priority Backlog

### Must Add

- [x] Add dataset versioning and experiment tracking for every training and export run
- [x] Add annotation quality checks and data cleaning reports
- [x] Add standardized inference preprocessing / postprocessing deployment spec
- [x] Add representative dataset planning for int8 quantization
- [x] Add structured logging, job status, and failure observability
- [x] Add baseline CI checks for lint, unit tests, integration smoke tests, and Docker smoke tests

These items are already mapped into Phases 2, 4, 5, 6, 8, 9, 11, and 15 above and should be treated as committed scope candidates.

### Strongly Recommended

- [ ] Add iOS sample integration project or minimal usage example
- [ ] Add Android sample integration project or minimal usage example
- [ ] Add latency, model size, and memory benchmark reporting
- [ ] Add asynchronous job execution design for API-triggered long-running tasks
- [ ] Add model registry / model version management
- [ ] Add retry and auto-resume strategy for failed training or export jobs
- [ ] Add model card and license metadata generation in exported artifacts

### Nice To Have Later

- [ ] Add advanced augmentation policy presets
- [ ] Add API authentication and access control planning
- [ ] Add support for additional detector families beyond MobileNet baseline
- [ ] Add artifact browsing UI or dashboard
