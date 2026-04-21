from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tensor_training_core.data.manifest.reader import read_manifest
from tensor_training_core.inference.visualize import draw_detection_overlays
from tensor_training_core.interfaces.dto import RunContext
from tensor_training_core.models.anchors import decode_box_from_anchor, load_anchor_array
from tensor_training_core.utils.logging import get_logger
from tensor_training_core.utils.paths import get_latest_run_dir, resolve_repo_path


def _prepare_input_image(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(resolve_repo_path(image_path)).convert("RGB")
    image = image.resize(image_size)
    return np.asarray(image, dtype=np.float32) / 255.0


def _quantize_input_if_needed(input_details, image_array: np.ndarray) -> np.ndarray:
    input_tensor = np.expand_dims(image_array, axis=0)
    if input_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = input_details["quantization"]
        if scale > 0:
            input_tensor = np.clip(np.round(input_tensor / scale + zero_point), 0, 255).astype(input_details["dtype"])
        else:
            input_tensor = input_tensor.astype(input_details["dtype"])
    return input_tensor.astype(input_details["dtype"])


def _dequantize_output_if_needed(output_details, output_tensor: np.ndarray) -> np.ndarray:
    if output_details["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            return (output_tensor.astype(np.float32) - zero_point) * scale
    return output_tensor.astype(np.float32)


def _load_label_map(label_map_path: str | Path) -> dict[int, str]:
    payload = json.loads(resolve_repo_path(label_map_path).read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload.items()}


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _extract_head_outputs(output_tensors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = list(output_tensors.values())
    if len(values) != 2:
        raise ValueError("Expected exactly two model outputs for class logits and bbox.")

    class_scores = next(value for value in values if value.shape[-1] > 4)[0]
    boxes = next(value for value in values if value.shape[-1] == 4)[0]
    return class_scores, boxes


def _run_nms(
    class_scores: np.ndarray,
    box_offsets: np.ndarray,
    anchors: np.ndarray,
    label_map: dict[int, str],
    score_threshold: float,
    iou_threshold: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for anchor_index, anchor_scores in enumerate(class_scores):
        foreground_scores = anchor_scores[1:]
        class_index = int(np.argmax(foreground_scores))
        score = float(foreground_scores[class_index])
        if score < score_threshold:
            continue
        label_id = class_index + 1
        decoded_box = decode_box_from_anchor(box_offsets[anchor_index], anchors[anchor_index])
        candidates.append(
            {
                "anchor_index": anchor_index,
                "label_id": label_id,
                "label": label_map.get(label_id, f"class_{label_id}"),
                "score": score,
                "bbox_xywh_norm": [float(value) for value in decoded_box.tolist()],
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, object]] = []
    for candidate in candidates:
        if all(_box_iou(candidate["bbox_xywh_norm"], current["bbox_xywh_norm"]) < iou_threshold for current in selected):
            selected.append(candidate)
    return selected


def verify_tflite_inference(
    context: RunContext,
    experiment_id: str,
    manifest_path: str | Path,
    image_size: tuple[int, int],
    label_map_path: str | Path,
    model_config,
) -> dict[str, str]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorFlow is required to run TFLite inference verification.") from exc

    logger = get_logger("inference")
    latest_export_run = get_latest_run_dir(
        experiment_id,
        required_relative_path="export/export_manifest.json",
        exclude_run_id=context.run_id,
    )
    export_manifest_path = latest_export_run / "export" / "export_manifest.json"
    export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    sample_record = next(iter(read_manifest(manifest_path)))
    label_map = _load_label_map(label_map_path)
    anchors = load_anchor_array(model_config)
    image_array = _prepare_input_image(sample_record.image_path, image_size)

    verification_summary: dict[str, object] = {
        "source_run_id": latest_export_run.name,
        "sample_image_path": sample_record.image_path,
        "results": {},
        "bundle_checks": {},
    }
    logger.info(
        "inference_verification_started source_run_id=%s sample_image_path=%s",
        latest_export_run.name,
        sample_record.image_path,
    )
    preview_dir = context.artifact_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for quantization, export_info in export_manifest["exports"].items():
        interpreter = tf.lite.Interpreter(model_path=export_info["tflite_path"])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        input_tensor = _quantize_input_if_needed(input_details, image_array)
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()

        raw_outputs = {
            detail["name"]: _dequantize_output_if_needed(detail, interpreter.get_tensor(detail["index"]))
            for detail in output_details
        }
        class_scores, box_offsets = _extract_head_outputs(raw_outputs)
        detections = _run_nms(
            class_scores,
            box_offsets,
            anchors,
            label_map,
            score_threshold=model_config.model.score_threshold,
            iou_threshold=model_config.model.nms_iou_threshold,
        )
        if not detections:
            detections = _run_nms(
                class_scores,
                box_offsets,
                anchors,
                label_map,
                score_threshold=0.0,
                iou_threshold=model_config.model.nms_iou_threshold,
            )
            detections = detections[:1]

        preview_path = draw_detection_overlays(
            image_path=sample_record.image_path,
            output_path=preview_dir / f"{quantization}_prediction.jpg",
            detections=detections[:3],
        )

        verification_summary["results"][quantization] = {
            "input_dtype": str(input_details["dtype"]),
            "detections": detections,
            "preview_path": str(preview_path),
            "raw_outputs": {key: value.tolist() for key, value in raw_outputs.items()},
        }
        logger.info(
            "inference_quantization_completed quantization=%s detection_count=%s preview_path=%s",
            quantization,
            len(detections),
            preview_path,
        )

    try:
        latest_mobile_run = get_latest_run_dir(
            experiment_id,
            required_relative_path="mobile/android/float32/model.tflite",
            exclude_run_id=context.run_id,
        )
        for platform in ("android", "ios"):
            platform_checks: dict[str, object] = {}
            for quantization in export_manifest["exports"].keys():
                bundle_dir = latest_mobile_run / "mobile" / platform / quantization
                required_files = [
                    bundle_dir / "model.tflite",
                    bundle_dir / "label.txt",
                    bundle_dir / "label_map.json",
                    bundle_dir / "export_metadata.json",
                    bundle_dir / "INTEGRATION.md",
                    bundle_dir / "bundle_verification.json",
                ]
                missing_files = [str(path) for path in required_files if not path.exists()]
                platform_checks[quantization] = {
                    "bundle_dir": str(bundle_dir),
                    "missing_files": missing_files,
                    "status": "ready" if not missing_files else "incomplete",
                }
            verification_summary["bundle_checks"][platform] = platform_checks
        logger.info("inference_bundle_validation_completed source_mobile_run_id=%s", latest_mobile_run.name)
    except FileNotFoundError:
        verification_summary["bundle_checks"]["status"] = "mobile_bundle_not_found"
        logger.warning("inference_bundle_validation_skipped mobile_bundle_not_found")

    summary_path = context.artifact_dir / "tflite_inference_summary.json"
    summary_path.write_text(json.dumps(verification_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logger.info("inference_verification_completed summary_path=%s", summary_path)
    return {
        "export_manifest_path": str(export_manifest_path),
        "summary_path": str(summary_path),
        "sample_image_path": sample_record.image_path,
        "preview_dir": str(preview_dir),
    }
