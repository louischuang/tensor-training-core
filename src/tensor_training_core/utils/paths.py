from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tensor_training_core.interfaces.dto import RunContext


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def ensure_run_context(experiment_id: str, dataset_version: str) -> RunContext:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    experiment_dir = ensure_directory(EXPERIMENTS_DIR / experiment_id)
    artifact_dir = ensure_directory(experiment_dir / run_id)
    log_dir = ensure_directory(LOGS_DIR / run_id)
    return RunContext(
        run_id=run_id,
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        experiment_dir=experiment_dir,
        artifact_dir=artifact_dir,
        log_dir=log_dir,
    )


def get_latest_run_dir(
    experiment_id: str,
    required_relative_path: str | None = None,
    exclude_run_id: str | None = None,
) -> Path:
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    run_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir())
    if exclude_run_id is not None:
        run_dirs = [path for path in run_dirs if path.name != exclude_run_id]
    if required_relative_path is not None:
        run_dirs = [path for path in run_dirs if (path / required_relative_path).exists()]
    if not run_dirs:
        raise FileNotFoundError(f"No run artifacts found under: {experiment_dir}")
    return run_dirs[-1]
