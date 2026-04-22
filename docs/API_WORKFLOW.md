# Phase-2 API Workflow

## Main Usage Patterns

### Synchronous workflow

```text
POST /datasets/prepare
POST /training/jobs
POST /exports/tflite
POST /exports/mobile-bundle
```

### Asynchronous training with logs

```text
POST /training/jobs/async
GET  /training/jobs/{job_id}
GET  /training/jobs/{job_id}/logs
GET  /training/jobs/{job_id}/logs/stream
```

## Documentation Endpoints

- `/docs`
- `/redoc`
- `/openapi.json`

## Request Body

```json
{
  "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"
}
```
