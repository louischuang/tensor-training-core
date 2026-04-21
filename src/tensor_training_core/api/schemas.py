from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigRequest(BaseModel):
    config_path: str = Field(
        ...,
        description="Repository-relative experiment config path, for example configs/experiments/train_tensorflow_esp32_cam_dev.yaml.",
        examples=["configs/experiments/train_tensorflow_esp32_cam_dev.yaml"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml"},
            ]
        }
    }


class JobPayload(BaseModel):
    job_id: str = Field(..., examples=["job_8bb14a044cc8"])
    operation: str = Field(..., examples=["train"])
    config_path: str = Field(..., examples=["configs/experiments/train_tensorflow_esp32_cam_dev.yaml"])
    state: str = Field(..., examples=["completed"])
    message: str = Field(..., examples=["tensorflow training completed."])
    outputs: dict[str, str] = Field(
        ...,
        examples=[
            {
                "checkpoint_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/checkpoints/latest.keras",
                "summary_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/training_summary.json",
                "artifact_dir": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53",
                "log_dir": "/workspace/artifacts/logs/20260421T233523Z_9991da53",
            }
        ],
    )
    failure_summary_path: str = Field(..., examples=[""])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job_8bb14a044cc8",
                    "operation": "train",
                    "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml",
                    "state": "completed",
                    "message": "tensorflow training completed.",
                    "outputs": {
                        "checkpoint_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/checkpoints/latest.keras",
                        "summary_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/training_summary.json",
                        "artifact_dir": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53",
                        "log_dir": "/workspace/artifacts/logs/20260421T233523Z_9991da53",
                    },
                    "failure_summary_path": "",
                }
            ]
        }
    }


class JobEnvelope(BaseModel):
    job: JobPayload

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job": {
                        "job_id": "job_8bb14a044cc8",
                        "operation": "train",
                        "config_path": "configs/experiments/train_tensorflow_esp32_cam_dev.yaml",
                        "state": "completed",
                        "message": "tensorflow training completed.",
                        "outputs": {
                            "checkpoint_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/checkpoints/latest.keras",
                            "summary_path": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53/training_summary.json",
                            "artifact_dir": "/workspace/artifacts/experiments/train_tensorflow_esp32_cam_dev/20260421T233523Z_9991da53",
                            "log_dir": "/workspace/artifacts/logs/20260421T233523Z_9991da53",
                        },
                        "failure_summary_path": "",
                    }
                }
            ]
        }
    }


class JobListResponse(BaseModel):
    jobs: list[JobPayload]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "ok"},
            ]
        }
    }


class ArtifactResponse(BaseModel):
    path: str
    is_dir: bool
    size_bytes: int | None


class ArtifactJobResponse(BaseModel):
    job: JobPayload


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Job not found: job_missing"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"detail": "Job not found: job_missing"},
            ]
        }
    }


class LogEntry(BaseModel):
    level: str | None = None
    logger: str | None = None
    message: str | None = None
    raw: str | None = None


class JobLogsResponse(BaseModel):
    job_id: str
    state: str
    log_path: str
    available: bool
    line_count: int
    lines: list[dict[str, object]]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job_8bb14a044cc8",
                    "state": "running",
                    "log_path": "/workspace/artifacts/logs/20260421T233523Z_9991da53/application.jsonl",
                    "available": True,
                    "line_count": 3,
                    "lines": [
                        {
                            "level": "INFO",
                            "logger": "training",
                            "message": "training_started total_epochs=4 batch_size=8",
                        },
                        {
                            "level": "INFO",
                            "logger": "training",
                            "message": "training_epoch_completed epoch=1/4 loss=1.92 class_output_accuracy=0.33",
                        },
                    ],
                }
            ]
        }
    }
