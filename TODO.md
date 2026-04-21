# TODO

## Phase 1. Lock the Baseline

- [ ] Confirm first baseline model as `ssd_mobilenet_v2_fpnlite_320x320`
- [ ] Confirm first dataset input format as COCO
- [ ] Confirm first deployment targets as iOS and Android with TensorFlow Lite
- [ ] Confirm phase-1 export strategy as float32 or float16

## Phase 2. Project Foundation

- [ ] Create `README.md` with project overview, quick start, repo links, license notes, and phase separation
- [ ] Create `pyproject.toml` and initialize Python package metadata
- [ ] Create base directory structure under `src/`, `configs/`, `docker/`, `scripts/`, `tests/`, `data/`, and `artifacts/`
- [ ] Add `.gitignore` for Python, TensorFlow artifacts, datasets, logs, and notebooks
- [ ] Define initial structured logging format and shared run identifiers for Python module flows
- [ ] Define standard log directory layout under `artifacts/logs/`
- [ ] Define required initial log fields for dataset, training, and export flows

## Phase 3. Environment Support

- [ ] Create `requirements/base.txt`
- [ ] Create `requirements/macos.txt` for Apple Silicon development
- [ ] Create `requirements/cuda.txt` for x64 CUDA Docker training
- [ ] Add `scripts/bootstrap_macos.sh`
- [ ] Add `scripts/bootstrap_cuda.sh`
- [ ] Create `docker/Dockerfile.cuda`
- [ ] Create `docker/docker-compose.cuda.yml`
- [ ] Add Docker entrypoint script

## Phase 4. Configuration System

- [ ] Define YAML config structure for dataset, model, training, export, and mobile package settings
- [ ] Implement config loader
- [ ] Add config schema validation
- [ ] Create sample configs under `configs/datasets/`, `configs/models/`, `configs/training/`, and `configs/experiments/`
- [ ] Add config fields for dataset version, experiment id, and runtime metadata

## Phase 5. COCO Import and Internal Manifest Pipeline

- [ ] Define unified internal manifest schema
- [ ] Implement COCO dataset validator
- [ ] Implement COCO importer
- [ ] Normalize COCO categories and annotations into internal manifest format
- [ ] Implement train / val / test split utility
- [ ] Implement conversion from internal manifest to TFRecord or model-ready input format
- [ ] Prepare small COCO-format fixture dataset for tests
- [ ] Add dataset versioning strategy and dataset metadata manifest
- [ ] Add annotation quality checks and data cleaning reports

## Phase 6. MobileNet Training Baseline

- [ ] Add baseline model config for `ssd_mobilenet_v2_fpnlite_320x320`
- [ ] Implement model factory
- [ ] Support pretrained checkpoint loading
- [ ] Implement training runner
- [ ] Add checkpointing and TensorBoard metrics logging
- [ ] Add resume-training support
- [ ] Add baseline experiment config for macOS
- [ ] Add baseline experiment config for CUDA Docker
- [ ] Track experiment metadata for every training run
- [ ] Write per-run training logs and failure summaries to `artifacts/logs/`

## Phase 7. Evaluation

- [ ] Implement validation pipeline
- [ ] Add mAP / precision / recall metrics output
- [ ] Generate sample prediction visualizations
- [ ] Save evaluation reports to `artifacts/reports/`

## Phase 8. TensorFlow Lite Export

- [ ] Implement SavedModel export
- [ ] Implement `.tflite` conversion
- [ ] Support float16 quantization
- [ ] Evaluate whether int8 quantization is needed after the first working prototype
- [ ] Generate label map and export metadata manifest
- [ ] Add representative dataset planning for future int8 quantization
- [ ] Track export metadata for every exported artifact
- [ ] Write export logs and error summaries to `artifacts/logs/`

## Phase 9. Mobile Deployment Packaging

- [ ] Create mobile asset bundle layout for iOS
- [ ] Create mobile asset bundle layout for Android
- [ ] Generate model metadata, labels, thresholds, and input spec documents for mobile apps
- [ ] Add packaging functions for mobile deployment artifacts
- [ ] Document integration assumptions for iOS and Android apps
- [ ] Emit standardized inference preprocessing / postprocessing deployment spec

## Phase 10. Inference Verification

- [ ] Implement TFLite inference runner
- [ ] Add sample-image smoke test for exported model
- [ ] Validate model input / output tensor formats
- [ ] Verify exported bundle assumptions against iOS / Android integration expectations
- [ ] Add simple visualization utility for detection output

## Phase 11. Shared Service Layer For Phase 2

- [ ] Create a shared application service layer for dataset, training, export, and artifact operations
- [ ] Define typed request / response DTOs for future external interfaces
- [ ] Add job-state abstraction for long-running training and export tasks
- [ ] Ensure future CLI, API, and agent integrations call the same orchestration layer
- [ ] Add shared job status and failure summary model for observability
- [ ] Add shared logging helpers so later interfaces extend the same structured log schema

## Phase 12. CLI and Scripts

- [ ] Create package CLI entrypoint
- [ ] Add `dataset import-coco` command group
- [ ] Add `dataset prepare` command
- [ ] Add `train run` command
- [ ] Add `train status` command
- [ ] Add `evaluate run` command
- [ ] Add `export tflite` command
- [ ] Add `export mobile` command
- [ ] Add `artifact list` command
- [ ] Add `artifact describe` command
- [ ] Add `serve api` command
- [ ] Extend logging with CLI-specific correlation fields
- [ ] Add shell wrappers in `scripts/`

## Phase 13. HTTP API

- [ ] Create FastAPI app skeleton
- [ ] Add health endpoint
- [ ] Add COCO import endpoint
- [ ] Add dataset prepare endpoint
- [ ] Add training job submission endpoint
- [ ] Add training job status endpoint
- [ ] Add TFLite export endpoint
- [ ] Add mobile bundle export endpoint
- [ ] Add artifact retrieval or artifact metadata endpoint
- [ ] Add structured API logging and request-to-job correlation ids
- [ ] Persist API request and job lifecycle logs to machine-readable files

## Phase 14. Skill For Third-Party Platforms And AI Agents

- [ ] Finalize `SKILL.md` after CLI and API are defined
- [ ] Document supported tasks and safe usage boundaries
- [ ] Document required inputs, outputs, and artifact locations
- [ ] Add CLI examples for AI-agent usage
- [ ] Add API examples for third-party platform usage
- [ ] Document hardware/runtime expectations and long-running job behavior
- [ ] Document dataset and license caveats for agent users

## Phase 15. Testing

- [ ] Add unit tests for config parsing
- [ ] Add unit tests for COCO annotation conversion
- [ ] Add unit tests for internal manifest generation
- [ ] Add unit tests for bounding box validation
- [ ] Add integration test for tiny COCO end-to-end pipeline
- [ ] Add export smoke test
- [ ] Add mobile bundle smoke test
- [ ] Add phase-2 API smoke tests
- [ ] Add phase-2 CLI contract tests for machine-readable output
- [ ] Add baseline CI checks for lint, unit tests, integration smoke tests, and Docker smoke tests

## Phase 16. Documentation

- [ ] Document local macOS setup flow
- [ ] Document CUDA Docker training flow
- [ ] Document COCO dataset import flow
- [ ] Document internal manifest format
- [ ] Document experiment artifact layout
- [ ] Document iOS / Android deployment workflow
- [ ] Document phase-2 API usage workflow
- [ ] Document phase-2 CLI automation workflow
- [ ] Document phase-2 `SKILL.md` usage for third-party platforms and agents

## First End-to-End Milestone

- [ ] Import one COCO-format dataset
- [ ] Produce one normalized internal manifest
- [ ] Train one MobileNet-based detector
- [ ] Export one working `.tflite` model
- [ ] Generate iOS and Android deployment bundles
- [ ] Pass one inference smoke test on the exported model
- [ ] Confirm the Python module workflow is complete before adding external interfaces

## Second Integration Milestone

- [ ] Expose the stable workflow through shared service interfaces
- [ ] Provide a usable CLI for operators and automation
- [ ] Provide a usable HTTP API for third-party platforms
- [ ] Provide finalized `SKILL.md` for AI-agent and platform integration

## Priority Backlog

### Must Add

- [ ] Add dataset versioning and experiment tracking for every training and export run
- [ ] Add annotation quality checks and data cleaning reports
- [ ] Add standardized inference preprocessing / postprocessing deployment spec
- [ ] Add representative dataset planning for int8 quantization
- [ ] Add structured logging, job status, and failure observability
- [ ] Add baseline CI checks for lint, unit tests, integration smoke tests, and Docker smoke tests

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
