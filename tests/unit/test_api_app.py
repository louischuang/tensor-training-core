from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tensor_training_core.api.app import create_app


@dataclass
class FakeJob:
    job_id: str
    operation: str
    state: str
    message: str
    outputs: dict[str, str] = field(default_factory=dict)


class FakeService:
    def execute_operation(self, operation: str, config_path: str) -> FakeJob:
        return FakeJob(
            job_id="job_test_123",
            operation=operation,
            state="completed",
            message="ok",
            outputs={"config_path": config_path},
        )

    def get_job_status(self, job_id: str) -> FakeJob:
        return FakeJob(
            job_id=job_id,
            operation="train",
            state="completed",
            message="done",
            outputs={"checkpoint_path": "artifacts/example.keras"},
        )


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_training_job_status_endpoint_uses_service_layer() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    response = client.get("/training/jobs/job_test_123")

    assert response.status_code == 200
    assert response.json()["job"]["job_id"] == "job_test_123"


def test_dataset_prepare_endpoint_uses_service_layer() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    response = client.post("/datasets/prepare", json={"config_path": "configs/example.yaml"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job"]["operation"] == "prepare_dataset"
    assert payload["job"]["outputs"]["config_path"] == "configs/example.yaml"
