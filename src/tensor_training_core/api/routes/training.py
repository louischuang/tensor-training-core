from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, HTTPException, Request

from tensor_training_core.api.schemas import ConfigRequest, JobEnvelope

router = APIRouter(prefix="/training", tags=["training"])


def _serialize_job(job: object) -> dict[str, object]:
    if is_dataclass(job):
        return asdict(job)
    return dict(job.__dict__)


@router.post(
    "/jobs",
    response_model=JobEnvelope,
    summary="Start a training job",
    description="Run the configured training pipeline and return the completed job record with checkpoint, metrics, summary, and TensorBoard outputs.",
)
def submit_training_job(request_body: ConfigRequest, request: Request) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("train", request_body.config_path)
    request.app.state.api_logger.info("api_training_job_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.get(
    "/jobs/{job_id}",
    response_model=JobEnvelope,
    summary="Read a training job",
    description="Look up a previously stored job record by job ID.",
)
def get_training_job_status(job_id: str, request: Request) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_training_status_read job_id=%s state=%s", job.job_id, job.state)
    return {"job": _serialize_job(job)}
