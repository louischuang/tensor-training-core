from __future__ import annotations

from collections import defaultdict

import numpy as np

from tensor_training_core.models.anchors import compute_iou


def run_classwise_nms(
    detections: list[dict[str, object]],
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for detection in detections:
        grouped[int(detection["label_id"])].append(detection)

    selected: list[dict[str, object]] = []
    for label_id, items in grouped.items():
        del label_id
        ordered = sorted(items, key=lambda item: float(item["score"]), reverse=True)
        kept: list[dict[str, object]] = []
        for candidate in ordered:
            if all(
                compute_iou(
                    np.asarray(candidate["bbox_xywh_norm"], dtype=np.float32),
                    np.asarray(current["bbox_xywh_norm"], dtype=np.float32),
                )
                < iou_threshold
                for current in kept
            ):
                kept.append(candidate)
        selected.extend(kept)
    return sorted(selected, key=lambda item: float(item["score"]), reverse=True)


def compute_detection_metrics(
    predictions_by_image: dict[str, list[dict[str, object]]],
    ground_truth_by_image: dict[str, list[dict[str, object]]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    per_class: dict[str, dict[str, float]] = {}
    ap_values: list[float] = []

    for class_id in class_ids:
        class_key = str(class_id)
        gt_total = 0
        gt_used: dict[str, list[bool]] = {}
        class_predictions: list[dict[str, object]] = []

        for image_path, ground_truths in ground_truth_by_image.items():
            matches = [gt for gt in ground_truths if int(gt["label_id"]) == class_id]
            gt_total += len(matches)
            gt_used[image_path] = [False] * len(matches)

        for image_path, predictions in predictions_by_image.items():
            for prediction in predictions:
                if int(prediction["label_id"]) == class_id:
                    item = dict(prediction)
                    item["image_path"] = image_path
                    class_predictions.append(item)

        class_predictions.sort(key=lambda item: float(item["score"]), reverse=True)
        tp = np.zeros((len(class_predictions),), dtype=np.float32)
        fp = np.zeros((len(class_predictions),), dtype=np.float32)

        for index, prediction in enumerate(class_predictions):
            image_path = str(prediction["image_path"])
            gt_candidates = [gt for gt in ground_truth_by_image.get(image_path, []) if int(gt["label_id"]) == class_id]
            if not gt_candidates:
                fp[index] = 1.0
                continue

            prediction_box = np.asarray(prediction["bbox_xywh_norm"], dtype=np.float32)
            ious = [compute_iou(prediction_box, np.asarray(gt["bbox_xywh_norm"], dtype=np.float32)) for gt in gt_candidates]
            best_index = int(np.argmax(ious))
            best_iou = ious[best_index]
            if best_iou >= iou_threshold and not gt_used[image_path][best_index]:
                tp[index] = 1.0
                gt_used[image_path][best_index] = True
            else:
                fp[index] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        recall = tp_cum / max(float(gt_total), 1e-6)

        ap = 0.0
        if gt_total > 0 and len(class_predictions) > 0:
            recall_points = np.linspace(0.0, 1.0, 11)
            precision_envelope = []
            for point in recall_points:
                valid = precision[recall >= point]
                precision_envelope.append(float(np.max(valid)) if valid.size else 0.0)
            ap = float(np.mean(precision_envelope))
            ap_values.append(ap)

        per_class[class_key] = {
            "ground_truth_count": float(gt_total),
            "prediction_count": float(len(class_predictions)),
            "precision": float(precision[-1]) if precision.size else 0.0,
            "recall": float(recall[-1]) if recall.size else 0.0,
            "ap50": ap,
        }

    macro_precision = float(np.mean([item["precision"] for item in per_class.values()])) if per_class else 0.0
    macro_recall = float(np.mean([item["recall"] for item in per_class.values()])) if per_class else 0.0
    map50 = float(np.mean(ap_values)) if ap_values else 0.0

    return {
        "iou_threshold": iou_threshold,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "map50": map50,
        "per_class": per_class,
    }
