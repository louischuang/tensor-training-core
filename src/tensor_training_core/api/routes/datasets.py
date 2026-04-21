from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Request

from tensor_training_core.api.schemas import ConfigRequest

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _serialize_job(job: object) -> dict[str, object]:
    if is_dataclass(job):
        return asdict(job)
    return dict(job.__dict__)


@router.post("/import/coco")
def import_coco(request_body: ConfigRequest, request: Request) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("import_coco_dataset", request_body.config_path)
    request.app.state.api_logger.info("api_dataset_import_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}


@router.post("/prepare")
def prepare_dataset(request_body: ConfigRequest, request: Request) -> dict[str, object]:
    service = request.app.state.service
    job = service.execute_operation("prepare_dataset", request_body.config_path)
    request.app.state.api_logger.info("api_dataset_prepare_completed job_id=%s", job.job_id)
    return {"job": _serialize_job(job)}
