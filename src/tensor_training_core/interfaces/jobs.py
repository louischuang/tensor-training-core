from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tensor_training_core.utils.paths import JOBS_DIR, ensure_directory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    job_id: str
    operation: str
    config_path: str
    state: str
    attempt: int = 1
    retry_of: str = ""
    message: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    outputs: dict[str, str] = field(default_factory=dict)
    failure_summary_path: str = ""


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_directory(root or JOBS_DIR)

    def create(
        self,
        operation: str,
        config_path: str,
        *,
        attempt: int = 1,
        retry_of: str = "",
    ) -> JobRecord:
        job = JobRecord(
            job_id=f"job_{uuid4().hex[:12]}",
            operation=operation,
            config_path=config_path,
            state="queued",
            attempt=attempt,
            retry_of=retry_of,
            message="Job queued.",
        )
        self.write(job)
        return job

    def write(self, job: JobRecord) -> JobRecord:
        job.updated_at = _utc_now()
        path = self.root / f"{job.job_id}.json"
        path.write_text(json.dumps(asdict(job), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return job

    def read(self, job_id: str) -> JobRecord:
        path = self.root / f"{job_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Job record does not exist: {job_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return JobRecord(**payload)

    def list(self) -> list[JobRecord]:
        jobs = []
        for path in sorted(self.root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            jobs.append(JobRecord(**payload))
        return jobs
