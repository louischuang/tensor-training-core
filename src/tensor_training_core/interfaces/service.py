from __future__ import annotations

import json
import threading
from pathlib import Path

from tensor_training_core.config.loader import (
    load_dataset_config,
    load_experiment_config,
    load_model_config,
    load_training_config,
)
from tensor_training_core.data.adapters.coco import load_coco_annotations
from tensor_training_core.data.converters.coco_to_manifest import convert_coco_dict_to_manifest_records
from tensor_training_core.data.split import split_rows
from tensor_training_core.data.manifest.writer import write_manifest
from tensor_training_core.data.validation import validate_coco_dataset
from tensor_training_core.evaluation.evaluator import evaluate_model
from tensor_training_core.export.mobile_bundle import package_mobile_bundles
from tensor_training_core.export.tflite import export_tflite_model
from tensor_training_core.inference.tflite_runner import verify_tflite_inference
from tensor_training_core.interfaces.dto import OperationResult, RunContext
from tensor_training_core.interfaces.jobs import JobRecord, JobStore
from tensor_training_core.training.runner import run_smoke_training, run_tensorflow_training
from tensor_training_core.utils.logging import get_logger, initialize_run_logging
from tensor_training_core.utils.paths import ensure_run_context, resolve_repo_path


class TrainingService:
    """Phase-1 orchestration surface for the core Python modules."""

    def __init__(self) -> None:
        self.job_store = JobStore()
        self._async_lock = threading.Lock()
        self._active_training_jobs: dict[str, str] = {}

    @staticmethod
    def _resolve_quality_report_path(dataset_config) -> Path:
        metadata_path = resolve_repo_path(dataset_config.dataset.metadata_output)
        stem = metadata_path.stem
        if stem.endswith("_metadata"):
            stem = stem[: -len("_metadata")]
        return metadata_path.with_name(f"{stem}_quality_report.json")

    @staticmethod
    def _resolve_train_manifest_path(dataset_config) -> Path:
        split_config = dataset_config.dataset.split
        if split_config is not None:
            return resolve_repo_path(split_config.train_manifest_output)
        return resolve_repo_path(dataset_config.dataset.manifest_output)

    @staticmethod
    def _resolve_eval_manifest_path(dataset_config) -> Path:
        split_config = dataset_config.dataset.split
        if split_config is not None:
            return resolve_repo_path(split_config.val_manifest_output)
        return resolve_repo_path(dataset_config.dataset.manifest_output)

    def _prepare_context(self, config_path: str | Path) -> RunContext:
        config = load_experiment_config(resolve_repo_path(config_path))
        return ensure_run_context(
            experiment_id=config.runtime.experiment_id,
            dataset_version=config.runtime.dataset_version,
        )

    def _load_phase1_configs(self, config_path: str | Path):
        experiment_config = load_experiment_config(resolve_repo_path(config_path))
        context = ensure_run_context(
            experiment_id=experiment_config.runtime.experiment_id,
            dataset_version=experiment_config.runtime.dataset_version,
        )
        dataset_config = load_dataset_config(experiment_config.dataset.config_path)
        model_config = load_model_config(experiment_config.model.config_path)
        training_config = load_training_config(experiment_config.training.config_path)
        return context, dataset_config, model_config, training_config

    def _load_phase1_configs_with_context(self, config_path: str | Path, context: RunContext):
        experiment_config = load_experiment_config(resolve_repo_path(config_path))
        dataset_config = load_dataset_config(experiment_config.dataset.config_path)
        model_config = load_model_config(experiment_config.model.config_path)
        training_config = load_training_config(experiment_config.training.config_path)
        return context, dataset_config, model_config, training_config

    def import_coco_dataset(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("dataset")
        logger.info("dataset_import_started config_path=%s dataset_root=%s", config_path, dataset_config.dataset.dataset_root)
        validation = validate_coco_dataset(
            dataset_root=dataset_config.dataset.dataset_root,
            annotations=dataset_config.dataset.annotations,
            images_dir=dataset_config.dataset.images_dir,
        )
        if validation.missing_images:
            raise FileNotFoundError(f"Missing images referenced by annotations: {validation.missing_images}")
        logger.info("Imported COCO dataset config: %s", config_path)
        return OperationResult(
            "import_coco_dataset",
            "completed",
            "COCO dataset import validation completed.",
            outputs={
                "image_count": str(validation.image_count),
                "annotation_count": str(validation.annotation_count),
                "category_count": str(validation.category_count),
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def prepare_dataset(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("dataset")
        logger.info("dataset_prepare_started config_path=%s dataset_root=%s", config_path, dataset_config.dataset.dataset_root)
        validation = validate_coco_dataset(
            dataset_root=dataset_config.dataset.dataset_root,
            annotations=dataset_config.dataset.annotations,
            images_dir=dataset_config.dataset.images_dir,
        )
        if validation.missing_images:
            raise FileNotFoundError(f"Missing images referenced by annotations: {validation.missing_images}")

        dataset_root = resolve_repo_path(dataset_config.dataset.dataset_root)
        coco = load_coco_annotations(dataset_root / dataset_config.dataset.annotations)
        records = convert_coco_dict_to_manifest_records(
            coco,
            images_root=dataset_root / dataset_config.dataset.images_dir,
        )
        manifest_path = resolve_repo_path(dataset_config.dataset.manifest_output)
        label_map_path = resolve_repo_path(dataset_config.dataset.label_map_output)
        metadata_path = resolve_repo_path(dataset_config.dataset.metadata_output)
        quality_report_path = self._resolve_quality_report_path(dataset_config)

        write_manifest(records, manifest_path)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        label_map = {str(item["id"]): item["name"] for item in coco.get("categories", [])}
        label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(
            json.dumps(validation.quality_report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "dataset_name": dataset_config.name,
            "dataset_root": str(dataset_root),
            "image_count": validation.image_count,
            "annotation_count": validation.annotation_count,
            "category_count": validation.category_count,
            "manifest_path": str(manifest_path),
            "label_map_path": str(label_map_path),
            "quality_report_path": str(quality_report_path),
        }

        split_config = dataset_config.dataset.split
        if split_config is not None:
            grouped_rows = {}
            for record in records:
                grouped_rows.setdefault(record.image_path, []).append(record)

            train_rows, val_rows, test_rows = split_rows(
                [{"image_path": image_path} for image_path in grouped_rows.keys()],
                train_ratio=split_config.train_ratio,
                val_ratio=split_config.val_ratio,
                seed=split_config.seed,
            )
            split_image_paths = {
                "train": {str(row["image_path"]) for row in train_rows},
                "val": {str(row["image_path"]) for row in val_rows},
                "test": {str(row["image_path"]) for row in test_rows},
            }
            split_output_paths = {
                "train": resolve_repo_path(split_config.train_manifest_output),
                "val": resolve_repo_path(split_config.val_manifest_output),
                "test": resolve_repo_path(split_config.test_manifest_output),
            }

            for split_name, output_path in split_output_paths.items():
                split_records = [record for record in records if record.image_path in split_image_paths[split_name]]
                write_manifest(split_records, output_path)

            metadata["split"] = {
                "train_manifest_path": str(split_output_paths["train"]),
                "val_manifest_path": str(split_output_paths["val"]),
                "test_manifest_path": str(split_output_paths["test"]),
                "train_image_count": len(split_image_paths["train"]),
                "val_image_count": len(split_image_paths["val"]),
                "test_image_count": len(split_image_paths["test"]),
                "train_record_count": sum(1 for record in records if record.image_path in split_image_paths["train"]),
                "val_record_count": sum(1 for record in records if record.image_path in split_image_paths["val"]),
                "test_record_count": sum(1 for record in records if record.image_path in split_image_paths["test"]),
                "seed": split_config.seed,
            }
            logger.info(
                "dataset_split_completed train_images=%s val_images=%s test_images=%s",
                len(split_image_paths["train"]),
                len(split_image_paths["val"]),
                len(split_image_paths["test"]),
            )

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        logger.info(
            "dataset_quality_report_completed quality_report_path=%s out_of_bounds=%s duplicates=%s invalid_bbox=%s",
            quality_report_path,
            validation.quality_report["issues"]["out_of_bounds_annotation_count"],
            validation.quality_report["issues"]["duplicate_annotation_count"],
            validation.quality_report["issues"]["invalid_bbox_format_count"]
            + validation.quality_report["issues"]["invalid_bbox_dimension_count"],
        )
        logger.info("Prepared dataset manifest from config: %s", config_path)
        return OperationResult(
            "prepare_dataset",
            "completed",
            "Dataset preparation completed.",
            outputs={
                "manifest_path": str(manifest_path),
                "label_map_path": str(label_map_path),
                "metadata_path": str(metadata_path),
                "quality_report_path": str(quality_report_path),
                "record_count": str(len(records)),
                "train_manifest_path": str(self._resolve_train_manifest_path(dataset_config)),
                "eval_manifest_path": str(self._resolve_eval_manifest_path(dataset_config)),
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def train(self, config_path: str | Path, context: RunContext | None = None) -> OperationResult:
        if context is None:
            context, dataset_config, model_config, training_config = self._load_phase1_configs(config_path)
        else:
            context, dataset_config, model_config, training_config = self._load_phase1_configs_with_context(config_path, context)
        initialize_run_logging(context.log_dir)
        logger = get_logger("training")
        manifest_path = self._resolve_train_manifest_path(dataset_config)
        logger.info("training_service_started config_path=%s manifest_path=%s", config_path, manifest_path)
        if not manifest_path.exists():
            self.prepare_dataset(config_path)
        if training_config.training.backend == "smoke":
            outputs = run_smoke_training(
                context=context,
                training_config=training_config,
                model_config=model_config,
                manifest_path=manifest_path,
            )
        elif training_config.training.backend == "tensorflow":
            outputs = run_tensorflow_training(
                context=context,
                training_config=training_config,
                model_config=model_config,
                manifest_path=manifest_path,
            )
        else:
            raise ValueError(f"Unsupported training backend: {training_config.training.backend}")
        logger.info(
            "Completed %s training run from config: %s",
            training_config.training.backend,
            config_path,
        )
        return OperationResult(
            "train",
            "completed",
            f"{training_config.training.backend} training completed.",
            outputs={
                **outputs,
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def evaluate(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, training_config = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("evaluation")
        logger.info("evaluation_service_started config_path=%s manifest_path=%s", config_path, self._resolve_eval_manifest_path(dataset_config))
        outputs = evaluate_model(
            context=context,
            experiment_id=context.experiment_id,
            manifest_path=self._resolve_eval_manifest_path(dataset_config),
            model_config=model_config,
            training_config=training_config,
        )
        logger.info("Completed evaluation run from config: %s", config_path)
        return OperationResult(
            "evaluate",
            "completed",
            "Evaluation completed.",
            outputs={
                **outputs,
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def export_tflite(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("export")
        logger.info("export_service_started config_path=%s", config_path)
        outputs = export_tflite_model(
            context=context,
            experiment_id=context.experiment_id,
            model_config=model_config,
            dataset_config=dataset_config,
        )
        logger.info("Completed TFLite export from config: %s", config_path)
        return OperationResult(
            "export_tflite",
            "completed",
            "TFLite export completed.",
            outputs={
                **outputs,
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def package_mobile_bundle(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("mobile")
        logger.info("mobile_service_started config_path=%s", config_path)
        outputs = package_mobile_bundles(
            context=context,
            experiment_id=context.experiment_id,
            label_map_path=str(resolve_repo_path(dataset_config.dataset.label_map_output)),
        )
        logger.info("Completed mobile bundle packaging from config: %s", config_path)
        return OperationResult(
            "package_mobile_bundle",
            "completed",
            "Mobile bundle packaging completed.",
            outputs={
                **outputs,
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def verify_inference(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("inference")
        logger.info("inference_service_started config_path=%s", config_path)
        outputs = verify_tflite_inference(
            context=context,
            experiment_id=context.experiment_id,
            manifest_path=resolve_repo_path(dataset_config.dataset.manifest_output),
            image_size=tuple(model_config.model.image_size),
            label_map_path=dataset_config.dataset.label_map_output,
            model_config=model_config,
        )
        logger.info("Completed TFLite inference verification from config: %s", config_path)
        return OperationResult(
            "verify_inference",
            "completed",
            "TFLite inference verification completed.",
            outputs={
                **outputs,
                "run_id": context.run_id,
                "artifact_dir": str(context.artifact_dir),
                "log_dir": str(context.log_dir),
            },
        )

    def execute_operation(self, operation: str, config_path: str | Path) -> JobRecord:
        handlers = {
            "import_coco_dataset": self.import_coco_dataset,
            "prepare_dataset": self.prepare_dataset,
            "train": self.train,
            "evaluate": self.evaluate,
            "export_tflite": self.export_tflite,
            "package_mobile_bundle": self.package_mobile_bundle,
            "verify_inference": self.verify_inference,
        }
        if operation not in handlers:
            raise ValueError(f"Unsupported operation: {operation}")

        job = self.job_store.create(operation=operation, config_path=str(config_path))
        return self._run_job(job, handlers[operation], config_path)

    def _run_job(self, job: JobRecord, handler, config_path: str | Path) -> JobRecord:
        job.state = "running"
        job.message = "Job is running."
        self.job_store.write(job)
        try:
            result = handler(config_path)
            job.state = "completed"
            job.message = result.message
            job.outputs = result.outputs
            self.job_store.write(job)
            return job
        except Exception as exc:
            job.state = "failed"
            job.message = str(exc)
            self.job_store.write(job)
            raise

    def retry_job(self, job_id: str) -> JobRecord:
        original = self.job_store.read(job_id)
        if original.state not in {"completed", "failed"}:
            raise ValueError(f"Only completed or failed jobs can be retried: {job_id}")

        handlers = {
            "import_coco_dataset": self.import_coco_dataset,
            "prepare_dataset": self.prepare_dataset,
            "train": self.train,
            "evaluate": self.evaluate,
            "export_tflite": self.export_tflite,
            "package_mobile_bundle": self.package_mobile_bundle,
            "verify_inference": self.verify_inference,
        }
        if original.operation not in handlers:
            raise ValueError(f"Unsupported retry operation: {original.operation}")

        retry_job = self.job_store.create(
            operation=original.operation,
            config_path=original.config_path,
            attempt=original.attempt + 1,
            retry_of=original.job_id,
        )
        retry_job.message = f"Retrying job {original.job_id}."
        self.job_store.write(retry_job)
        return self._run_job(retry_job, handlers[original.operation], original.config_path)

    def start_training_job_async(self, config_path: str | Path) -> JobRecord:
        config_key = str(resolve_repo_path(config_path))
        with self._async_lock:
            active_job_id = self._active_training_jobs.get(config_key)
            if active_job_id is not None:
                active_job = self.job_store.read(active_job_id)
                if active_job.state == "running":
                    raise ValueError(
                        f"An asynchronous training job is already running for config {config_path}: {active_job_id}"
                    )
                self._active_training_jobs.pop(config_key, None)

        context = self._prepare_context(config_path)
        job = self.job_store.create(operation="train", config_path=str(config_path))
        job.state = "running"
        job.message = "Training job is running asynchronously."
        job.outputs = {
            "run_id": context.run_id,
            "artifact_dir": str(context.artifact_dir),
            "log_dir": str(context.log_dir),
        }
        self.job_store.write(job)
        with self._async_lock:
            self._active_training_jobs[config_key] = job.job_id

        def worker() -> None:
            try:
                result = self.train(config_path, context=context)
                job.state = "completed"
                job.message = result.message
                job.outputs = {**job.outputs, **result.outputs}
                self.job_store.write(job)
            except Exception as exc:
                job.state = "failed"
                job.message = str(exc)
                failure_summary_path = Path(job.outputs.get("log_dir", "")) / "failure_summary.json"
                if failure_summary_path.exists():
                    job.failure_summary_path = str(failure_summary_path)
                self.job_store.write(job)
            finally:
                with self._async_lock:
                    if self._active_training_jobs.get(config_key) == job.job_id:
                        self._active_training_jobs.pop(config_key, None)

        threading.Thread(target=worker, daemon=True, name=f"training-job-{job.job_id}").start()
        return job

    def get_job_status(self, job_id: str) -> JobRecord:
        return self.job_store.read(job_id)

    def list_jobs(self) -> list[JobRecord]:
        return self.job_store.list()

    def describe_artifact(self, path: str | Path) -> dict[str, object]:
        artifact_path = resolve_repo_path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
        return {
            "path": str(artifact_path),
            "is_dir": artifact_path.is_dir(),
            "size_bytes": artifact_path.stat().st_size if artifact_path.is_file() else None,
        }

    def get_job_log_path(self, job_id: str) -> Path | None:
        job = self.get_job_status(job_id)
        log_dir = job.outputs.get("log_dir", "")
        if not log_dir:
            return None
        return resolve_repo_path(Path(log_dir) / "application.jsonl")

    def get_job_logs(self, job_id: str, limit: int = 200) -> dict[str, object]:
        job = self.get_job_status(job_id)
        log_path = self.get_job_log_path(job_id)
        if log_path is None:
            return {
                "job_id": job.job_id,
                "state": job.state,
                "log_path": "",
                "available": False,
                "line_count": 0,
                "lines": [],
            }
        if not log_path.exists():
            return {
                "job_id": job.job_id,
                "state": job.state,
                "log_path": str(log_path),
                "available": False,
                "line_count": 0,
                "lines": [],
            }

        raw_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected_lines = raw_lines[-limit:] if limit > 0 else raw_lines
        parsed_lines = []
        for line in selected_lines:
            try:
                parsed_lines.append(json.loads(line))
            except json.JSONDecodeError:
                parsed_lines.append({"raw": line})
        return {
            "job_id": job.job_id,
            "state": job.state,
            "log_path": str(log_path),
            "available": True,
            "line_count": len(raw_lines),
            "lines": parsed_lines,
        }
