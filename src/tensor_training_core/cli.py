from __future__ import annotations

import argparse
import json
from uuid import uuid4
from dataclasses import asdict, is_dataclass

from tensor_training_core.interfaces.service import TrainingService
from tensor_training_core.utils.logging import get_logger, initialize_run_logging, set_logging_context
from tensor_training_core.utils.paths import ARTIFACTS_DIR, ensure_directory


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _serialize_job(job: object) -> dict[str, object]:
    if is_dataclass(job):
        return asdict(job)
    return dict(job.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensor-training-core", description="Tensor Training Core CLI")
    subparsers = parser.add_subparsers(dest="group", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="command", required=True)
    dataset_import = dataset_sub.add_parser("import-coco")
    dataset_import.add_argument("--config", required=True)
    dataset_prepare = dataset_sub.add_parser("prepare")
    dataset_prepare.add_argument("--config", required=True)

    train = subparsers.add_parser("train")
    train_sub = train.add_subparsers(dest="command", required=True)
    train_run = train_sub.add_parser("run")
    train_run.add_argument("--config", required=True)
    train_status = train_sub.add_parser("status")
    train_status.add_argument("--job-id", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate_sub = evaluate.add_subparsers(dest="command", required=True)
    evaluate_run = evaluate_sub.add_parser("run")
    evaluate_run.add_argument("--config", required=True)

    export = subparsers.add_parser("export")
    export_sub = export.add_subparsers(dest="command", required=True)
    export_tflite = export_sub.add_parser("tflite")
    export_tflite.add_argument("--config", required=True)
    export_mobile = export_sub.add_parser("mobile")
    export_mobile.add_argument("--config", required=True)

    artifact = subparsers.add_parser("artifact")
    artifact_sub = artifact.add_subparsers(dest="command", required=True)
    artifact_list = artifact_sub.add_parser("list")
    artifact_list.add_argument("--limit", type=int, default=20)
    artifact_describe = artifact_sub.add_parser("describe")
    artifact_describe.add_argument("--artifact", required=True)

    serve = subparsers.add_parser("serve")
    serve_sub = serve.add_subparsers(dest="command", required=True)
    serve_api = serve_sub.add_parser("api")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = TrainingService()
    cli_invocation_id = f"cli_{uuid4().hex[:12]}"
    cli_command = " ".join(part for part in [getattr(args, "group", ""), getattr(args, "command", "")] if part)
    cli_log_dir = ensure_directory(ARTIFACTS_DIR / "logs" / "cli" / cli_invocation_id)
    initialize_run_logging(cli_log_dir)
    set_logging_context(
        cli_invocation_id=cli_invocation_id,
        cli_command=cli_command,
    )
    logger = get_logger("cli")
    logger.info("cli_command_started command=%s", cli_command)

    if args.group == "dataset" and args.command == "import-coco":
        job = service.execute_operation("import_coco_dataset", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "dataset" and args.command == "prepare":
        job = service.execute_operation("prepare_dataset", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "train" and args.command == "run":
        job = service.execute_operation("train", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "train" and args.command == "status":
        job = service.get_job_status(args.job_id)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "evaluate" and args.command == "run":
        job = service.execute_operation("evaluate", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "export" and args.command == "tflite":
        job = service.execute_operation("export_tflite", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "export" and args.command == "mobile":
        job = service.execute_operation("package_mobile_bundle", args.config)
        logger.info("cli_command_completed command=%s job_id=%s state=%s", cli_command, job.job_id, job.state)
        _emit({"job": _serialize_job(job)})
        return
    if args.group == "artifact" and args.command == "list":
        jobs = [_serialize_job(job) for job in service.list_jobs()[-args.limit :]]
        logger.info("cli_command_completed command=%s result_count=%s", cli_command, len(jobs))
        _emit({"jobs": jobs})
        return
    if args.group == "artifact" and args.command == "describe":
        artifact = service.describe_artifact(args.artifact)
        logger.info("cli_command_completed command=%s artifact_path=%s", cli_command, artifact["path"])
        _emit(artifact)
        return
    if args.group == "serve" and args.command == "api":
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise SystemExit("uvicorn is required to serve the API.") from exc
        logger.info("cli_command_completed command=%s host=%s port=%s", cli_command, args.host, args.port)
        uvicorn.run("tensor_training_core.api.app:create_app", host=args.host, port=args.port, factory=True)
        return

    raise SystemExit("Unsupported command.")


if __name__ == "__main__":
    main()
