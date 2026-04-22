from __future__ import annotations

import asyncio
import json
from typing import Annotated
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Body, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse

from tensor_training_core.api.schemas import ConfigRequest, ErrorResponse, JobEnvelope, JobLogsResponse

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


@router.post(
    "/jobs/async",
    response_model=JobEnvelope,
    summary="Start an asynchronous training job",
    description="Queue a training job in a background thread and return immediately with the job ID, run ID, artifact directory, and log directory so clients can poll status or stream logs.",
    responses={
        200: {"description": "Training started asynchronously and returned as a running job record."},
        422: {"description": "Request validation failed. The config_path body field is missing or invalid."},
    },
)
def submit_training_job_async(
    request_body: Annotated[
        ConfigRequest,
        Body(description="Experiment config used to run model training asynchronously."),
    ],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.start_training_job_async(request_body.config_path)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_training_job_async_started job_id=%s", job.job_id)
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


@router.post(
    "/jobs/{job_id}/retry",
    response_model=JobEnvelope,
    summary="Retry a stored job",
    description="Create a new job attempt from a previously completed or failed job. This is the primary recovery path for failed exports, because retrying an export job reuses the same config and resolves the latest available checkpoint again.",
    responses={
        200: {"description": "Retry completed and returned as a new job record."},
        400: {
            "model": ErrorResponse,
            "description": "The requested job cannot be retried in its current state or operation type.",
        },
        404: {
            "model": ErrorResponse,
            "description": "The requested job ID was not found in artifacts/jobs/.",
        },
    },
)
def retry_training_job(
    job_id: Annotated[str, Path(description="Job identifier returned by a previous operation.", examples=["job_8bb14a044cc8"])],
    request: Request,
) -> dict[str, object]:
    service = request.app.state.service
    try:
        job = service.retry_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_job_retry_completed source_job_id=%s job_id=%s", job_id, job.job_id)
    return {"job": _serialize_job(job)}


@router.get(
    "/jobs/{job_id}/logs",
    response_model=JobLogsResponse,
    summary="Read training logs",
    description="Return the latest application log lines for a training job. This is useful for polling training progress from external clients.",
    responses={
        200: {"description": "Training log snapshot returned successfully."},
        404: {
            "model": ErrorResponse,
            "description": "The requested job ID was not found in artifacts/jobs/.",
        },
    },
)
def get_training_job_logs(
    job_id: Annotated[str, Path(description="Job identifier returned by a previous training request.", examples=["job_8bb14a044cc8"])],
    request: Request,
    limit: Annotated[int, Query(description="Maximum number of most recent log lines to return.", ge=1, le=1000)] = 200,
) -> dict[str, object]:
    service = request.app.state.service
    try:
        payload = service.get_job_logs(job_id, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.api_logger.info("api_training_logs_read job_id=%s line_count=%s", job_id, payload["line_count"])
    return payload


@router.get(
    "/jobs/{job_id}/logs/stream",
    summary="Stream training logs with SSE",
    description="Open a Server-Sent Events stream for training log updates. Existing recent lines are sent first, followed by new lines until the job completes or the client disconnects.",
    responses={
        200: {"description": "SSE stream opened successfully."},
        404: {
            "model": ErrorResponse,
            "description": "The requested job ID was not found in artifacts/jobs/.",
        },
    },
)
async def stream_training_job_logs(
    job_id: Annotated[str, Path(description="Job identifier returned by a previous training request.", examples=["job_8bb14a044cc8"])],
    request: Request,
    tail: Annotated[int, Query(description="Number of most recent log lines to send before live streaming begins.", ge=0, le=500)] = 20,
) -> StreamingResponse:
    service = request.app.state.service
    try:
        snapshot = service.get_job_logs(job_id, limit=tail if tail > 0 else 1)
        job = service.get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_path = service.get_job_log_path(job_id)
    initial_position = log_path.stat().st_size if log_path is not None and log_path.exists() else 0

    async def event_generator():
        if tail > 0:
            for line in snapshot["lines"]:
                yield f"event: log\ndata: {json.dumps(line, ensure_ascii=True)}\n\n"

        position = initial_position
        while True:
            if await request.is_disconnected():
                break

            current_job = service.get_job_status(job_id)
            current_log_path = service.get_job_log_path(job_id)
            if current_log_path is not None and current_log_path.exists():
                with current_log_path.open("r", encoding="utf-8") as handle:
                    handle.seek(position)
                    new_lines = handle.readlines()
                    position = handle.tell()
                for raw_line in new_lines:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        payload = {"raw": raw_line}
                    yield f"event: log\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"

            if current_job.state in {"completed", "failed"}:
                yield (
                    "event: status\n"
                    f"data: {json.dumps({'job_id': current_job.job_id, 'state': current_job.state}, ensure_ascii=True)}\n\n"
                )
                break

            yield "event: heartbeat\ndata: {}\n\n"
            await asyncio.sleep(1)

    request.app.state.api_logger.info("api_training_logs_stream_opened job_id=%s", job.job_id)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
