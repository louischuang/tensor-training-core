from __future__ import annotations

import json
import logging
from pathlib import Path


_CURRENT_LOG_DIR: Path | None = None
_LOG_CONTEXT: dict[str, str] = {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_LOG_CONTEXT)
        return json.dumps(payload, ensure_ascii=True)


def initialize_run_logging(log_dir: Path) -> None:
    global _CURRENT_LOG_DIR
    if _CURRENT_LOG_DIR == log_dir:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_dir / "application.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    _CURRENT_LOG_DIR = log_dir


def set_logging_context(**fields: str) -> None:
    global _LOG_CONTEXT
    _LOG_CONTEXT = {key: value for key, value in fields.items() if value}


def get_logging_context() -> dict[str, str]:
    return dict(_LOG_CONTEXT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
