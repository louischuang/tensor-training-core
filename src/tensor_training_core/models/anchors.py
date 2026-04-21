from __future__ import annotations

import math

import numpy as np

from tensor_training_core.config.schema import ModelConfig


def load_anchor_array(model_config: ModelConfig) -> np.ndarray:
    anchors = [
        [anchor.cx, anchor.cy, anchor.w, anchor.h]
        for anchor in model_config.model.anchors
    ]
    anchor_array = np.asarray(anchors, dtype=np.float32)
    if anchor_array.shape[0] != model_config.model.max_detections:
        raise ValueError(
            "Anchor count must match max_detections: "
            f"{anchor_array.shape[0]} != {model_config.model.max_detections}"
        )
    return anchor_array


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.asarray([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def compute_iou(box_a_xywh: np.ndarray, box_b_xywh: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = xywh_to_xyxy(box_a_xywh)
    bx0, by0, bx1, by1 = xywh_to_xyxy(box_b_xywh)
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    union = max(0.0, (ax1 - ax0) * (ay1 - ay0)) + max(0.0, (bx1 - bx0) * (by1 - by0)) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def encode_box_to_anchor(box_xywh: np.ndarray, anchor_xywh: np.ndarray) -> np.ndarray:
    box_cx = box_xywh[0] + (box_xywh[2] / 2.0)
    box_cy = box_xywh[1] + (box_xywh[3] / 2.0)
    anchor_cx, anchor_cy, anchor_w, anchor_h = anchor_xywh
    eps = 1e-6
    tx = (box_cx - anchor_cx) / max(anchor_w, eps)
    ty = (box_cy - anchor_cy) / max(anchor_h, eps)
    tw = math.log(max(box_xywh[2], eps) / max(anchor_w, eps))
    th = math.log(max(box_xywh[3], eps) / max(anchor_h, eps))
    return np.asarray([tx, ty, tw, th], dtype=np.float32)


def decode_box_from_anchor(offsets: np.ndarray, anchor_xywh: np.ndarray) -> np.ndarray:
    anchor_cx, anchor_cy, anchor_w, anchor_h = anchor_xywh
    tx, ty, tw, th = offsets
    box_cx = (tx * anchor_w) + anchor_cx
    box_cy = (ty * anchor_h) + anchor_cy
    box_w = math.exp(float(tw)) * anchor_w
    box_h = math.exp(float(th)) * anchor_h
    x = box_cx - (box_w / 2.0)
    y = box_cy - (box_h / 2.0)
    return np.asarray(
        [
            float(max(0.0, min(1.0, x))),
            float(max(0.0, min(1.0, y))),
            float(max(0.0, min(1.0, box_w))),
            float(max(0.0, min(1.0, box_h))),
        ],
        dtype=np.float32,
    )
