from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.config.schema import DatasetConfig, ModelConfig
from tensor_training_core.data.manifest.reader import read_manifest
from tensor_training_core.export.labels import write_label_txt
from tensor_training_core.export.metadata import build_export_metadata, write_json_file
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import get_latest_run_dir, resolve_repo_path


def _representative_dataset(
    manifest_path: str | Path,
    image_size: tuple[int, int],
    max_samples: int = 32,
):
    def generator():
        for index, record in enumerate(read_manifest(manifest_path)):
            if index >= max_samples:
                break
            image = Image.open(record.image_path).convert("RGB")
            image = image.resize(image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            yield [np.expand_dims(image_array, axis=0)]

    return generator


def _convert_quantized_model(tf, model, quantization: str, manifest_path: str | Path, image_size: tuple[int, int]) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(manifest_path, image_size)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    return converter.convert()


def export_tflite_model(
    context: RunContext,
    experiment_id: str,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
) -> dict[str, str]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required to export a TFLite model.") from exc

    logger = get_logger("export")
    latest_run_dir = get_latest_run_dir(
        experiment_id,
        required_relative_path="checkpoints/latest.keras",
        exclude_run_id=context.run_id,
    )
    checkpoint_path = latest_run_dir / "checkpoints" / "latest.keras"
    manifest_path = dataset_config.dataset.manifest_output
    image_size = tuple(model_config.model.image_size)

    model = tf.keras.models.load_model(checkpoint_path)
    export_dir = context.artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    label_txt_path = write_label_txt(export_dir / "label.txt", resolve_repo_path(dataset_config.dataset.label_map_output))
    logger.info(
        "export_started checkpoint_path=%s export_dir=%s manifest_path=%s label_txt_path=%s",
        checkpoint_path,
        export_dir,
        manifest_path,
        label_txt_path,
    )

    quantized_outputs: dict[str, str] = {}
    export_index: dict[str, object] = {
        "source_run_id": latest_run_dir.name,
        "checkpoint_path": str(checkpoint_path),
        "exports": {},
    }

    for quantization in ("float32", "float16", "int8"):
        tflite_bytes = _convert_quantized_model(tf, model, quantization, manifest_path, image_size)
        tflite_path = export_dir / f"{model_config.model.name}_{quantization}.tflite"
        metadata_path = export_dir / f"export_metadata_{quantization}.json"
        tflite_path.write_bytes(tflite_bytes)
        metadata = build_export_metadata(
            model_config=model_config,
            dataset_config=dataset_config,
            tflite_path=tflite_path,
            source_checkpoint=checkpoint_path,
            quantization=quantization,
        )
        write_json_file(metadata_path, metadata)
        export_index["exports"][quantization] = {
            "tflite_path": str(tflite_path),
            "metadata_path": str(metadata_path),
        }
        quantized_outputs[f"tflite_path_{quantization}"] = str(tflite_path)
        quantized_outputs[f"metadata_path_{quantization}"] = str(metadata_path)
        logger.info(
            "export_quantization_completed quantization=%s tflite_path=%s metadata_path=%s",
            quantization,
            tflite_path,
            metadata_path,
        )

    export_manifest_path = export_dir / "export_manifest.json"
    write_json_file(export_manifest_path, export_index)
    logger.info("export_manifest_completed export_manifest_path=%s", export_manifest_path)
    quantized_outputs["checkpoint_path"] = str(checkpoint_path)
    quantized_outputs["export_manifest_path"] = str(export_manifest_path)
    quantized_outputs["label_txt_path"] = str(label_txt_path)
    return quantized_outputs
