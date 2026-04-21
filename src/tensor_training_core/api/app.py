from __future__ import annotations

import json
from fastapi import FastAPI, Request

from tensor_training_core.api.routes.datasets import router as dataset_router
from tensor_training_core.api.routes.exports import router as export_router
from tensor_training_core.api.routes.health import router as health_router
from tensor_training_core.api.routes.training import router as training_router
from tensor_training_core.interfaces.service import TrainingService
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import ARTIFACTS_DIR, ensure_directory


def create_app() -> FastAPI:
    app = FastAPI(title="Tensor Training Core API", version="0.1.0")
    app.state.service = TrainingService()
    app.state.api_logger = get_logger("api")
    api_log_dir = ensure_directory(ARTIFACTS_DIR / "logs" / "api")
    app.state.api_request_log_path = api_log_dir / "requests.jsonl"

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", "")
        start_payload = {
            "event": "api_request_started",
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
        }
        app.state.api_logger.info(
            "api_request_started method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        with app.state.api_request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(start_payload, ensure_ascii=True) + "\n")
        response = await call_next(request)
        complete_payload = {
            "event": "api_request_completed",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "request_id": request_id,
        }
        app.state.api_logger.info(
            "api_request_completed method=%s path=%s status_code=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
        )
        with app.state.api_request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(complete_payload, ensure_ascii=True) + "\n")
        return response

    app.include_router(health_router)
    app.include_router(dataset_router)
    app.include_router(training_router)
    app.include_router(export_router)
    return app
