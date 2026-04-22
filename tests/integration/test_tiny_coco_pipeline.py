from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tensor_training_core.api.app import create_app
from tensor_training_core.interfaces.service import TrainingService


SMOKE_CONFIG = "configs/experiments/dev_macos.yaml"


def test_tiny_coco_end_to_end_pipeline() -> None:
    service = TrainingService()

    imported = service.import_coco_dataset(SMOKE_CONFIG)
    prepared = service.prepare_dataset(SMOKE_CONFIG)
    trained = service.train(SMOKE_CONFIG)

    assert imported.status == "completed"
    assert prepared.status == "completed"
    assert trained.status == "completed"
    assert Path(prepared.outputs["manifest_path"]).exists()
    assert Path(prepared.outputs["quality_report_path"]).exists()
    assert Path(trained.outputs["checkpoint_path"]).exists()
    assert Path(trained.outputs["summary_path"]).exists()


def test_phase2_api_smoke_with_real_service() -> None:
    client = TestClient(create_app())

    prepare = client.post("/datasets/prepare", json={"config_path": SMOKE_CONFIG})
    assert prepare.status_code == 200
    prepare_payload = prepare.json()
    assert Path(prepare_payload["job"]["outputs"]["manifest_path"]).exists()

    train = client.post("/training/jobs", json={"config_path": SMOKE_CONFIG})
    assert train.status_code == 200
    train_payload = train.json()
    job_id = train_payload["job"]["job_id"]
    assert Path(train_payload["job"]["outputs"]["checkpoint_path"]).exists()

    status = client.get(f"/training/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["job"]["job_id"] == job_id

    artifacts = client.get(f"/artifacts/{job_id}")
    assert artifacts.status_code == 200
    assert artifacts.json()["job"]["outputs"]["checkpoint_path"].endswith(".ckpt")


def test_phase2_cli_contracts_machine_readable_output(tmp_path: Path) -> None:
    import subprocess
    import sys
    import os

    env = {**os.environ, "PYTHONPATH": "src"}

    prepare = subprocess.run(
        [sys.executable, "-m", "tensor_training_core.cli", "dataset", "prepare", "--config", SMOKE_CONFIG],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    prepare_payload = json.loads(prepare.stdout)
    assert prepare_payload["job"]["operation"] == "prepare_dataset"
    assert Path(prepare_payload["job"]["outputs"]["manifest_path"]).exists()

    train = subprocess.run(
        [sys.executable, "-m", "tensor_training_core.cli", "train", "run", "--config", SMOKE_CONFIG],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    train_payload = json.loads(train.stdout)
    job_id = train_payload["job"]["job_id"]
    assert train_payload["job"]["operation"] == "train"

    status = subprocess.run(
        [sys.executable, "-m", "tensor_training_core.cli", "train", "status", "--job-id", job_id],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    status_payload = json.loads(status.stdout)
    assert status_payload["job"]["job_id"] == job_id

    artifact = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensor_training_core.cli",
            "artifact",
            "describe",
            "--artifact",
            train_payload["job"]["outputs"]["summary_path"],
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    artifact_payload = json.loads(artifact.stdout)
    assert artifact_payload["is_dir"] is False
