from __future__ import annotations

from fastapi import APIRouter
from tensor_training_core.api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Return a simple readiness response for service monitoring and smoke tests.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}
