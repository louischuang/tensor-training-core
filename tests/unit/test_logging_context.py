from __future__ import annotations

import json
from pathlib import Path

from tensor_training_core.utils.logging import get_logger, initialize_run_logging, set_logging_context


def test_json_logs_include_cli_correlation_fields(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    initialize_run_logging(log_dir)
    set_logging_context(cli_invocation_id="cli_test_123", cli_command="train run")

    logger = get_logger("cli")
    logger.info("cli_command_started command=train run")

    payload = json.loads((log_dir / "application.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert payload["logger"] == "cli"
    assert payload["cli_invocation_id"] == "cli_test_123"
    assert payload["cli_command"] == "train run"
