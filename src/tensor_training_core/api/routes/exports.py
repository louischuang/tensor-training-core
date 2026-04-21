from __future__ import annotations

from typing import Annotated
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Body, HTTPException, Path, Request

from tensor_training_core.api.schemas import ArtifactJobResponse, ConfigRequest, JobEnvelope
from tensor_training_core.api.schemas import ErrorResponse

router = APIRouter(tags=["exports"])


def _serialize_job(job: object) -> dict[str, object]:
    if is_dataclass(job):
        return asdict(job)
    return dict(job.__dict__)


@router.post(
    "/exports/tflite",
    response_model=JobEnvelope,
    summary="Export SavedModel and TFLite artifacts",
    description="Export the latest checkpoint for the configured experiment to SavedModel plus float32, float16, and int8 TFLite files.",
    responses={
        200: {"description": "Export completed and returned as a job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def export_tflite(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to resolve the latest checkpoint and export settings."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("export_tflite", request_body.config_path)
    request.app.state.api_logger.info("api_export_tflite_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.post(
    "/exports/mobile-bundle",
    response_model=JobEnvelope,
    summary="Package iOS and Android bundles",
    description="Create mobile integration bundles from the latest export, including model files, label.txt, integration notes, and verification metadata.",
    responses={
        200: {"description": "Mobile bundle packaging completed and returned as a job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def export_mobile_bundle(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to resolve the latest export manifest for mobile packaging."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("package_mobile_bundle", request_body.config_path)
    request.app.state.api_logger.info("api_export_mobile_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.get(
    "/artifacts/{job_id}",
    response_model=ArtifactJobResponse,
    summary="Read artifact metadata by job",
    description="Return the stored job record for a completed operation so downstream systems can discover linked artifact paths.",
    responses={
        200: {"description": "Stored artifact-linked job record returned successfully."},
        404: {
            "model": ErrorResponse,
            "description": "The requested job ID was not found in artifacts/jobs/.",
        },
    },
)
def get_artifact_metadata(
    job_id: Annotated[str, Path(description="Job identifier returned by a previous operation.", examples=["job_8bb14a044cc8"])],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_artifact_metadata_read job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}
