from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigRequest(BaseModel):
    config_path: str = Field(
        ...,
        description="Repository-relative experiment config path, for example configs/experiments/train_tensorflow_esp32_cam_dev.yaml.",
        examples=["configs/experiments/train_tensorflow_esp32_cam_dev.yaml"],
    )


class JobPayload(BaseModel):
    job_id: str
    operation: str
    config_path: str
    state: str
    message: str
    outputs: dict[str, str]
    failure_summary_path: str


class JobEnvelope(BaseModel):
    job: JobPayload


class JobListResponse(BaseModel):
    jobs: list[JobPayload]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class ArtifactResponse(BaseModel):
    path: str
    is_dir: bool
    size_bytes: int | None


class ArtifactJobResponse(BaseModel):
    job: JobPayload
