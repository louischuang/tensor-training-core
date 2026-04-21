from __future__ import annotations

import json
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
from tensor_training_core.training.runner import run_smoke_training, run_tensorflow_training
from tensor_training_core.utils.logging import get_logger, initialize_run_logging
from tensor_training_core.utils.paths import ensure_run_context, resolve_repo_path


class TrainingService:
    """Phase-1 orchestration surface for the core Python modules."""

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

    def import_coco_dataset(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("dataset")
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
            },
        )

    def prepare_dataset(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("dataset")
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

        write_manifest(records, manifest_path)
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        label_map = {str(item["id"]): item["name"] for item in coco.get("categories", [])}
        label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        metadata = {
            "dataset_name": dataset_config.name,
            "dataset_root": str(dataset_root),
            "image_count": validation.image_count,
            "annotation_count": validation.annotation_count,
            "category_count": validation.category_count,
            "manifest_path": str(manifest_path),
            "label_map_path": str(label_map_path),
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

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        logger.info("Prepared dataset manifest from config: %s", config_path)
        return OperationResult(
            "prepare_dataset",
            "completed",
            "Dataset preparation completed.",
            outputs={
                "manifest_path": str(manifest_path),
                "label_map_path": str(label_map_path),
                "metadata_path": str(metadata_path),
                "record_count": str(len(records)),
                "train_manifest_path": str(self._resolve_train_manifest_path(dataset_config)),
                "eval_manifest_path": str(self._resolve_eval_manifest_path(dataset_config)),
            },
        )

    def train(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, training_config = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("training")
        manifest_path = self._resolve_train_manifest_path(dataset_config)
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
            outputs=outputs,
        )

    def evaluate(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, training_config = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("evaluation")
        outputs = evaluate_model(
            context=context,
            experiment_id=context.experiment_id,
            manifest_path=self._resolve_eval_manifest_path(dataset_config),
            model_config=model_config,
            training_config=training_config,
        )
        logger.info("Completed evaluation run from config: %s", config_path)
        return OperationResult("evaluate", "completed", "Evaluation completed.", outputs=outputs)

    def export_tflite(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("export")
        outputs = export_tflite_model(
            context=context,
            experiment_id=context.experiment_id,
            model_config=model_config,
            dataset_config=dataset_config,
        )
        logger.info("Completed TFLite export from config: %s", config_path)
        return OperationResult("export_tflite", "completed", "TFLite export completed.", outputs=outputs)

    def package_mobile_bundle(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, _, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("mobile")
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
            outputs=outputs,
        )

    def verify_inference(self, config_path: str | Path) -> OperationResult:
        context, dataset_config, model_config, _ = self._load_phase1_configs(config_path)
        initialize_run_logging(context.log_dir)
        logger = get_logger("inference")
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
            outputs=outputs,
        )
