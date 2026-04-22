from __future__ import annotations

from tensor_training_core.cli import build_parser


def test_cli_supports_train_status_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["train", "status", "--job-id", "job_test_123"])

    assert args.group == "train"
    assert args.command == "status"
    assert args.job_id == "job_test_123"


def test_cli_supports_dataset_prepare_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["dataset", "prepare", "--config", "configs/example.yaml"])

    assert args.group == "dataset"
    assert args.command == "prepare"
    assert args.config == "configs/example.yaml"


def test_cli_supports_job_retry_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["job", "retry", "--job-id", "job_test_123"])

    assert args.group == "job"
    assert args.command == "retry"
    assert args.job_id == "job_test_123"
