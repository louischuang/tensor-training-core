# API Documentation

## Overview

Tensor Training Core exposes a FastAPI-based HTTP API for dataset preparation, model training, export, and artifact discovery.

Available documentation endpoints after the API server starts:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

Start the API locally:

```bash
python -m tensor_training_core.cli serve api --host 127.0.0.1 --port 8000
```

Then open:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

## Request Model

Most write operations use the same request body:

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```

`config_path` must point to a repository-relative experiment config file.

## Endpoints

### Health

- `GET /health`
- Purpose: service readiness check

Response:

```json
{
  "status": "ok"
}
```

### Datasets

- `POST /datasets/import/coco`
- Purpose: validate the configured COCO dataset and return dataset counts

- `POST /datasets/prepare`
- Purpose: generate manifests, split manifests, label map, metadata, and quality report

### Training

- `POST /training/jobs`
- Purpose: run a training job using the provided config

- `GET /training/jobs/{job_id}`
- Purpose: read a stored training job record

### Exports

- `POST /exports/tflite`
- Purpose: export SavedModel plus `float32`, `float16`, and `int8` TFLite artifacts

- `POST /exports/mobile-bundle`
- Purpose: generate iOS and Android integration bundles

- `GET /artifacts/{job_id}`
- Purpose: read a completed job record and its artifact paths

## Response Shape

Most operation routes return:

```json
{
  "job": {
    "job_id": "job_xxx",
    "operation": "train",
    "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml",
    "state": "completed",
    "message": "tensorflow training completed.",
    "outputs": {
      "summary_path": "/workspace/artifacts/experiments/...",
      "artifact_dir": "/workspace/artifacts/experiments/...",
      "log_dir": "/workspace/artifacts/logs/..."
    },
    "failure_summary_path": ""
  }
}
```

## Common Error Codes

### `404 Not Found`

Returned by:

- `GET /training/jobs/{job_id}`
- `GET /artifacts/{job_id}`

Example:

```json
{
  "detail": "Job not found: job_missing"
}
```

### `422 Unprocessable Entity`

Returned when the request body does not match the required schema, most commonly when `config_path` is missing or malformed.

Example:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "config_path"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

## Notes For Third-Party Integrators

- Treat `job_id` as the stable lookup key for follow-up reads.
- Treat `outputs` as the authoritative source of generated file paths.
- For API request tracing, provide an `x-request-id` header.
- Request lifecycle events are stored in `artifacts/logs/api/requests.jsonl`.
- The API is currently synchronous at the HTTP layer: long-running operations do not return until the job has completed.
