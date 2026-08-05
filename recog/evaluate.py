"""Evaluation metrics for the recognition module.

All metrics here are pure-numpy so the evaluation pipeline remains
importable in environments without torch installed (for example, the
CI container). The supported metrics are:

* VOC-style 11-point interpolated Average Precision (per class and
  mean, at a configurable IoU threshold).
* Centroid error (Euclidean distance between the GT and predicted
  bounding-box centres, in pixels).
* Edge error (L-infinity over the four box edges, in pixels).

The 11-point protocol follows Everingham *et al.* (2010) rather than
the later all-points form so the numbers are directly comparable
with the published Faster R-CNN baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.types import BBox

# Four floats: (xmin, ymin, xmax, ymax).
Box = Tuple[float, float, float, float]


# ------------------------------------------------------ helper geometry --
    
def _iou(a: Box, b: Box) -> float:
    """Intersection-over-union for two (xmin, ymin, xmax, ymax) tuples."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0)
    ih = max(0.0, y1 - y0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# -------------------------------------------------------- public types ---

@dataclass
class ClassAPResult:
    """Per-class AP summary returned by :func:`per_class_ap`."""

    ap: float
    precision: np.ndarray
    recall: np.ndarray
    num_gt: int


# ----------------------------------------------------- AP computation ----

def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute the 11-point VOC interpolated Average Precision."""
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 11):
        mask = recall >= threshold
        p_at = precision[mask].max() if np.any(mask) else 0.0
        ap += p_at / 11.0
    return float(ap)


def per_class_ap(
    gts_by_image: Dict[int, List[Tuple[Box, int]]],
    preds_by_image: Dict[int, List[Tuple[Box, int, float]]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> ClassAPResult:
    """VOC AP for a single class.

    ``gts_by_image`` maps image id → list of ``(bbox, class_id)``.
    ``preds_by_image`` maps image id → list of ``(bbox, class_id,
    score)``. Both are dense dicts — images with no GTs or no
    predictions are simply absent from the mapping.
    """
    # Detections for this class, sorted by descending score.
    dets: List[Tuple[int, Box, float]] = []
    for img_id, preds in preds_by_image.items():
        for bbox, cls, score in preds:
            if cls == class_id:
                dets.append((img_id, bbox, float(score)))
    dets.sort(key=lambda d: -d[2])

    # Per-image availability of each GT box (True = still available).
    gt_available: Dict[int, List[bool]] = {}
    total_gt = 0
    for img_id, gts in gts_by_image.items():
        mask = [cls == class_id for (_, cls) in gts]
        gt_available[img_id] = list(mask)
        total_gt += sum(mask)

    if total_gt == 0:
        return ClassAPResult(
            ap=0.0,
            precision=np.zeros(0),
            recall=np.zeros(0),
            num_gt=0,
        )

    n = len(dets)
    tp = np.zeros(n, dtype=np.float64)
    fp = np.zeros(n, dtype=np.float64)

    for i, (img_id, pred_box, _score) in enumerate(dets):
        gts = gts_by_image.get(img_id, [])
        available = gt_available.setdefault(img_id, [False] * len(gts))

        best_iou = 0.0
        best_j = -1
        for j, (gt_box, cls) in enumerate(gts):
            if cls != class_id or not available[j]:
                continue
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            tp[i] = 1.0
            available[best_j] = False
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    return ClassAPResult(
        ap=voc_ap(recall, precision),
        precision=precision,
        recall=recall,
        num_gt=total_gt,
    )


def mean_ap(
    gts_by_image: Dict[int, List[Tuple[Box, int]]],
    preds_by_image: Dict[int, List[Tuple[Box, int, float]]],
    class_ids: Sequence[int],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Return a dict of per-class APs plus the mean AP across classes."""
    aps: Dict[str, float] = {}
    for c in class_ids:
        aps[f"AP_{c}"] = per_class_ap(
            gts_by_image, preds_by_image, c, iou_threshold,
        ).ap
    per_class = [aps[f"AP_{c}"] for c in class_ids]
    aps[f"mAP@{iou_threshold:.2f}"] = float(np.mean(per_class))
    return aps


# ------------------------------------------------------- pose metrics ---

def centroid_error_px(gt: BBox, pred: BBox) -> float:
    """Euclidean distance between the two boxes' centres, in pixels."""
    return float(np.hypot(gt.cx - pred.cx, gt.cy - pred.cy))


def edge_error_px(gt: BBox, pred: BBox) -> float:
    """Maximum per-edge absolute difference, in pixels (L∞ metric)."""
    return float(max(
        abs(gt.xmin - pred.xmin),
        abs(gt.ymin - pred.ymin),
        abs(gt.xmax - pred.xmax),
        abs(gt.ymax - pred.ymax),
    ))


__all__ = [
    "ClassAPResult",
    "centroid_error_px",
    "edge_error_px",
    "mean_ap",
    "per_class_ap",
    "voc_ap",
]
