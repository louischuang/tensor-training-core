from __future__ import annotations

from typing import Annotated
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Body, Request

from tensor_training_core.api.schemas import ConfigRequest, JobEnvelope

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _serialize_job(job: object) -> dict[str, object]:
    if is_dataclass(job):
        return asdict(job)
    return dict(job.__dict__)


@router.post(
    "/import/coco",
    response_model=JobEnvelope,
    summary="Import and validate a COCO dataset",
    description="Validate the configured COCO dataset and create a job record with image, annotation, and category counts.",
    responses={
        200: {"description": "Dataset import completed and returned as a job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def import_coco(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to resolve the target dataset import settings."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("import_coco_dataset", request_body.config_path)
    request.app.state.api_logger.info("api_dataset_import_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.post(
    "/prepare",
    response_model=JobEnvelope,
    summary="Prepare dataset manifests",
    description="Generate the internal manifest, label map, metadata, quality report, and split manifests for the configured dataset.",
    responses={
        200: {"description": "Dataset preparation completed and returned as a job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def prepare_dataset(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to resolve manifest output, label map, and split generation settings."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("prepare_dataset", request_body.config_path)
    request.app.state.api_logger.info("api_dataset_prepare_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}
