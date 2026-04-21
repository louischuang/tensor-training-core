from __future__ import annotations

from typing import Annotated
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Body, HTTPException, Path, Request

from tensor_training_core.api.schemas import ConfigRequest, ErrorResponse, JobEnvelope

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
    responses={
        200: {"description": "Training completed and returned as a job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def submit_training_job(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to run model training."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("train", request_body.config_path)
    request.app.state.api_logger.info("api_training_job_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.get(
    "/jobs/{job_id}",
    response_model=JobEnvelope,
    summary="Read a training job",
    description="Look up a previously stored job record by job ID.",
    responses={
        200: {"description": "Stored job record returned successfully."},
        404: {
            "model": ErrorResponse,
            "description": "The requested job ID was not found in artifacts/jobs/.",
        },
    },
)
def get_training_job_status(
    job_id: Annotated[str, Path(description="Job identifier returned by a previous training request.", examples=["job_8bb14a044cc8"])],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_training_status_read job_id=%s state=%s", job.job_id, job.state)
    return {"job": _serialize_job(job)}
