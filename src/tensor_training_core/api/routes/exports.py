from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, HTTPException, Request

from tensor_training_core.api.schemas import ArtifactJobResponse, ConfigRequest, JobEnvelope

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
)
def export_tflite(request_body: ConfigRequest, request: Request) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("export_tflite", request_body.config_path)
    request.app.state.api_logger.info("api_export_tflite_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.post(
    "/exports/mobile-bundle",
    response_model=JobEnvelope,
    summary="Package iOS and Android bundles",
    description="Create mobile integration bundles from the latest export, including model files, label.txt, integration notes, and verification metadata.",
)
def export_mobile_bundle(request_body: ConfigRequest, request: Request) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("package_mobile_bundle", request_body.config_path)
    request.app.state.api_logger.info("api_export_mobile_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.get(
    "/artifacts/{job_id}",
    response_model=ArtifactJobResponse,
    summary="Read artifact metadata by job",
    description="Return the stored job record for a completed operation so downstream systems can discover linked artifact paths.",
)
def get_artifact_metadata(job_id: str, request: Request) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_artifact_metadata_read job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}
