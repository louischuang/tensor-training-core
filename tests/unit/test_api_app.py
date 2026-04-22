from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from tensor_training_core.api.app import create_app


@dataclass
class FakeJob:
    job_id: str
    operation: str
    state: str
    message: str
    config_path: str = "configs/example.yaml"
    attempt: int = 1
    retry_of: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    failure_summary_path: str = ""


class FakeService:
    def __init__(self) -> None:
        self.log_path = Path("artifacts/tests/fake_api_logs/application.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            '{"level":"INFO","logger":"training","message":"training_started total_epochs=4"}\n'
            '{"level":"INFO","logger":"training","message":"training_epoch_completed epoch=1/4"}\n',
            encoding="utf-8",
        )

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
            outputs={"checkpoint_path": "artifacts/example.keras", "log_dir": str(self.log_path.parent)},
        )

    def start_training_job_async(self, config_path: str) -> FakeJob:
        return FakeJob(
            job_id="job_async_123",
            operation="train",
            state="running",
            message="Training job is running asynchronously.",
            outputs={"run_id": "run_async_123", "log_dir": str(self.log_path.parent)},
        )

    def get_job_logs(self, job_id: str, limit: int = 200) -> dict[str, object]:
        return {
            "job_id": job_id,
            "state": "running",
            "log_path": str(self.log_path),
            "available": True,
            "line_count": 2,
            "lines": [
                {"level": "INFO", "logger": "training", "message": "training_started total_epochs=4"},
                {"level": "INFO", "logger": "training", "message": "training_epoch_completed epoch=1/4"},
            ][:limit],
        }

    def get_job_log_path(self, job_id: str) -> Path:
        return self.log_path

    def retry_job(self, job_id: str) -> FakeJob:
        return FakeJob(
            job_id="job_retry_123",
            operation="prepare_dataset",
            state="completed",
            message="Dataset preparation completed.",
            attempt=2,
            retry_of=job_id,
            outputs={"manifest_path": "artifacts/retried_manifest.jsonl"},
        )


class FakeAsyncConflictService(FakeService):
    def start_training_job_async(self, config_path: str) -> FakeJob:
        raise ValueError("An asynchronous training job is already running for config configs/example.yaml: job_async_123")


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


def test_training_job_async_endpoint_returns_running_job() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    response = client.post("/training/jobs/async", json={"config_path": "configs/example.yaml"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["job"]["job_id"] == "job_async_123"
    assert payload["job"]["state"] == "running"


def test_training_job_async_endpoint_returns_conflict_for_duplicate_config() -> None:
    app = create_app()
    app.state.service = FakeAsyncConflictService()
    client = TestClient(app)

    response = client.post("/training/jobs/async", json={"config_path": "configs/example.yaml"})

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_training_logs_endpoint_returns_log_lines() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    response = client.get("/training/jobs/job_test_123/logs?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job_test_123"
    assert payload["available"] is True
    assert payload["line_count"] == 2
    assert payload["lines"][0]["logger"] == "training"


def test_job_retry_endpoint_returns_new_attempt() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    response = client.post("/training/jobs/job_test_123/retry")

    assert response.status_code == 200
    payload = response.json()
    assert payload["job"]["job_id"] == "job_retry_123"
    assert payload["job"]["attempt"] == 2
    assert payload["job"]["retry_of"] == "job_test_123"


def test_training_logs_stream_endpoint_returns_sse_events() -> None:
    app = create_app()
    app.state.service = FakeService()
    client = TestClient(app)

    with client.stream("GET", "/training/jobs/job_test_123/logs/stream?tail=1") as response:
        body = b"".join(response.iter_bytes())

    text = body.decode("utf-8")
    assert response.status_code == 200
    assert "event: log" in text
    assert "training_started total_epochs=4" in text or "training_epoch_completed epoch=1/4" in text
    assert "event: status" in text
