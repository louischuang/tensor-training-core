from __future__ import annotations

from pydantic import BaseModel


class ConfigRequest(BaseModel):
    config_path: str


class JobResponse(BaseModel):
    job_id: str
    operation: str
    state: str
    message: str
    outputs: dict[str, str]


class ArtifactResponse(BaseModel):
    path: str
    is_dir: bool
    size_bytes: int | None
