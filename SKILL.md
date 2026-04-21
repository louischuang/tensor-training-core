# Tensor Training Core Skill

## Purpose

This skill describes how third-party platforms and AI agents should interact with Tensor Training Core after phase 2 is implemented.

The project is intended to support:

- COCO dataset import
- dataset preparation into an internal manifest
- MobileNet-based object detection training
- TensorFlow Lite export
- iOS and Android mobile bundle generation

## Status

Current status:

- planning stage
- phase 1 focuses on Python modules
- this skill is a draft for the later CLI and API integration phase

This file should be finalized only after the CLI and API become real.

## Definitions

- `dataset`: images plus labeling files that can be imported into training
- `internal manifest`: the normalized project-owned metadata generated from an imported dataset

## Supported Task Intents

- import a COCO dataset
- prepare a dataset for training
- start a training job
- check training status
- evaluate a trained model
- export a TensorFlow Lite model
- package iOS and Android deployment assets
- inspect generated artifacts

## Preferred Invocation Order

1. Import or register a COCO dataset
2. Prepare the dataset into internal manifest format
3. Start training with a selected experiment config
4. Evaluate the produced checkpoint
5. Export a TensorFlow Lite model
6. Generate mobile deployment bundles
7. Verify exported model behavior

## Required Inputs

Typical inputs include:

- dataset root path
- COCO annotation file path
- experiment config path
- runtime target such as `macos` or `cuda`
- output artifact directory

## Expected Outputs

- normalized internal manifest
- train / val / test split metadata
- model checkpoints
- evaluation reports
- `.tflite` model
- label map
- metadata JSON
- iOS bundle
- Android bundle

## Integration Contract

Phase 1:

- do not rely on this skill yet
- core behavior should be validated through Python modules first

Phase 2:

- prefer stable CLI commands
- then stable HTTP API
- avoid importing private modules directly

Agents should avoid:

- importing private modules directly
- assuming file names that are not documented
- skipping dataset validation
- calling GPU-specific flows on unsupported hardware

## Planned CLI Surface

```text
tensor-training-core dataset import-coco --config <path>
tensor-training-core dataset prepare --config <path>
tensor-training-core train run --config <path>
tensor-training-core train status --job-id <id>
tensor-training-core evaluate run --config <path>
tensor-training-core export tflite --config <path>
tensor-training-core export mobile --config <path>
tensor-training-core artifact list --artifact-dir <path>
tensor-training-core artifact describe --artifact <path>
tensor-training-core serve api
```

## Planned API Surface

```text
GET  /health
POST /datasets/import/coco
POST /datasets/prepare
POST /training/jobs
GET  /training/jobs/{job_id}
POST /exports/tflite
POST /exports/mobile-bundle
GET  /artifacts/{artifact_id}
```

## Runtime Notes

- macOS Apple Silicon is intended for development, validation, lightweight training, and export
- x64 Linux with CUDA Docker is intended for heavier training
- long-running training and export operations should be treated as asynchronous jobs in the later API phase

## Logging Notes

Logging should grow together with the implemented features:

- phase 1: module-level logs for dataset, training, export, and failures
- phase 2: CLI and API correlation fields plus request and job lifecycle logs

## Safety And License Notes

- COCO-format input is supported, but dataset licensing and redistribution obligations still need review
- agents should preserve license notices and project metadata when generating bundles or manifests
- agents should not claim commercial clearance without a final dependency and dataset review

## Update Rule

Whenever phase-2 CLI commands, API routes, input schemas, or artifact layouts change, this `SKILL.md` should be updated in the same change set.
