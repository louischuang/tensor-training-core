from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunContext:
    run_id: str
    experiment_id: str
    dataset_version: str
    experiment_dir: Path
    artifact_dir: Path
    log_dir: Path


@dataclass(slots=True)
class OperationResult:
    name: str
    status: str
    message: str
    outputs: dict[str, str] = field(default_factory=dict)
