from __future__ import annotations

from pathlib import Path

from tensor_training_core.interfaces.service import TrainingService


def test_prepare_and_train_smoke() -> None:
    service = TrainingService()
    prepare_result = service.prepare_dataset("configs/experiments/dev_macos.yaml")
    train_result = service.train("configs/experiments/dev_macos.yaml")

    assert prepare_result.status == "completed"
    assert "quality_report_path" in prepare_result.outputs
    assert Path(prepare_result.outputs["quality_report_path"]).exists()
    assert train_result.status == "completed"
    assert "checkpoint_path" in train_result.outputs
