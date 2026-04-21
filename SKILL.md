# Tensor Training Core Skill

## Purpose

This skill describes how third-party platforms and AI agents should interact with Tensor Training Core through the shared phase-2 orchestration layer.

The supported workflow is:

- COCO dataset import
- dataset preparation into an internal manifest
- MobileNet-based object detection training
- evaluation and report generation
- TensorFlow Lite export
- iOS and Android mobile bundle generation
- artifact and job inspection

## Definitions

- `dataset`: images plus labeling files that can be imported into training
- `internal manifest`: the normalized project-owned metadata generated from an imported dataset
- `job`: a tracked execution record under `artifacts/jobs/`
- `artifact`: generated files under `artifacts/experiments/`, `artifacts/reports/`, or `artifacts/logs/`

## Supported Task Intents

- import a COCO dataset
- prepare a dataset for training
- start a training job
- check training status
- evaluate a trained model
- export a TensorFlow Lite model
- package iOS and Android deployment assets
- inspect generated artifacts
- inspect generated job records

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
- job id when reading status or artifact metadata

## Expected Outputs

- normalized internal manifest
- train / val / test split metadata
- model checkpoints
- evaluation reports
- evaluation preview images
- SavedModel export
- `.tflite` model
- label map
- `label.txt`
- metadata JSON
- iOS bundle
- Android bundle
- job records
- structured logs and failure summaries

## Integration Contract

- Prefer stable CLI commands first.
- Use the HTTP API when an external platform needs request/response integration.
- Avoid importing private Python modules directly.
- Treat `TrainingService.execute_operation(...)` as the shared orchestration layer behind CLI and API.

Agents should avoid:

- importing private modules directly
- assuming file names that are not documented
- skipping dataset validation
- calling GPU-specific flows on unsupported hardware

## CLI Surface

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

CLI responses are JSON so agents can parse `job_id`, `state`, and `outputs`.

## API Surface

```text
GET  /health
POST /datasets/import/coco
POST /datasets/prepare
POST /training/jobs
GET  /training/jobs/{job_id}
POST /exports/tflite
POST /exports/mobile-bundle
GET  /artifacts/{job_id}
```

The API currently accepts:

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

and returns:

```json
{
  "job": {
    "job_id": "job_xxx",
    "operation": "train",
    "state": "completed",
    "message": "tensorflow training completed.",
    "outputs": {
      "checkpoint_path": "...",
      "summary_path": "..."
    }
  }
}
```

## Artifact Locations

- Job records: `artifacts/jobs/<job_id>.json`
- Run artifacts: `artifacts/experiments/<experiment_id>/<run_id>/`
- Evaluation reports: `artifacts/reports/<experiment_id>/<run_id>/`
- Structured logs: `artifacts/logs/<run_id>/application.jsonl`
- API request log: `artifacts/logs/api/requests.jsonl`

## Runtime Notes

- macOS Apple Silicon is intended for development, validation, lightweight training, and export
- x64 Linux with CUDA Docker is intended for heavier training
- job records already exist, but execution is currently synchronous from the caller perspective
- long-running training and export operations should still be treated as potentially slow operations

## Logging Notes

Logging currently includes:

- per-run structured logs under `artifacts/logs/<run_id>/application.jsonl`
- failure summaries for training, export, and mobile packaging
- API request lifecycle logs under `artifacts/logs/api/requests.jsonl`
- job status records under `artifacts/jobs/`

## Safety And License Notes

- COCO-format input is supported, but dataset licensing and redistribution obligations still need review
- agents should preserve license notices and project metadata when generating bundles or manifests
- agents should not claim commercial clearance without a final dependency and dataset review
- agents should not assume exported bundles are production-ready app packages; they are integration assets

## Update Rule

Whenever CLI commands, API routes, input schemas, or artifact layouts change, this `SKILL.md` should be updated in the same change set.
