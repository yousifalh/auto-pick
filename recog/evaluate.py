"""Evaluation metrics for the recognition module.

All metrics here are pure-numpy so the evaluation pipeline remains
importable in environments without torch installed (for example, the
CI container). The supported metrics are:

* VOC-style 11-point interpolated Average Precision (per class and
  mean, at a configurable IoU threshold).
* Centroid error (Euclidean distance between the GT and predicted
  bounding-box centres, in pixels).
* Edge error (L-infinity over the four box edges, in pixels).

The 11-point protocol follows the VOC2007 form described by Everingham
*et al.* (2010) rather than the later all-points (VOC2010/COCO) form.
That makes the numbers comparable with results scored the same way; it
does **not** make them directly comparable with any published Faster
R-CNN baseline, most of which report all-point AP and would read
slightly higher on the identical detections (measured on this
repository's real photos: 0.8647 eleven-point vs 0.8695 all-point).
Boxes here are 0-based with exclusive max edges throughout, which is
the modern convention and *not* the VOC devkit's 1-based inclusive one
— the IoU therefore differs from the devkit's by a sub-percent amount
that grows as boxes get smaller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.logging import get_logger
from common.types import BBox

log = get_logger("recog.evaluate")

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

# The eleven recall levels of the VOC2007 protocol, as EXACT tenths.
#
# `np.linspace(0.0, 1.0, 11)` is NOT this. It yields 0.30000000000000004,
# 0.6000000000000001 and 0.7000000000000001, each one ULP above the tenth
# it is meant to be, so a curve whose recall lands exactly on 0.3, 0.6 or
# 0.7 fails its own grid point and forfeits a whole 1/11 = 0.0909 bin.
# Measured (audit G, finding 4): 6 of 10 objects found at precision 1.0
# scored 0.5455 where the protocol gives 7/11 = 0.6364.
#
# `k / 10.0` is correctly rounded, so it returns the same double as the
# recall ratio does whenever the denominator divides cleanly (3/10, 6/20
# and 30/100 are all the same double as 0.3). The tolerance below covers
# the cases where it does not - a recall accumulated through cumsum, or a
# curve that plateaus a few ULP under a level - so that ">= 0.3" means
# 0.3 rather than "0.3 plus whatever float noise got here first".
_RECALL_LEVELS = tuple(k / 10.0 for k in range(11))
_RECALL_TOL = 1e-9


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """11-point interpolated AP — VOC2007 / Everingham *et al.* (2010).

    AP is the mean, over the eleven recall levels {0, 0.1, ..., 1.0}, of
    the maximum precision observed at any recall at or above that level.
    The levels are exact tenths (see :data:`_RECALL_LEVELS`), compared
    with a tolerance, because the whole protocol turns on whether a
    recall of exactly 0.3 clears the 0.3 grid point.
    """
    ap = 0.0
    for threshold in _RECALL_LEVELS:
        mask = recall >= threshold - _RECALL_TOL
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

    # No GT for this class: there is no PR curve, so there is no AP. The
    # 0.0 below is a placeholder, and `num_gt=0` is the field that says
    # so — :func:`mean_ap` reads `num_gt`, not `ap`, and excludes the
    # class from the mean. Do not "fix" this to 1.0 or nan: callers that
    # want the guarded policy get it from `mean_ap`, and a caller reading
    # `.ap` directly needs a float.
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

        # VOC's rule, in this order: find the GT this detection overlaps
        # MOST, and only then ask whether that GT is still free. Skipping
        # taken GT inside the search instead (`or not available[j]`) lets a
        # duplicate detection fall through to its SECOND-best GT and score
        # a true positive there — so a detector that found one object is
        # credited with two. Measured (audit G, finding 5) on two GT boxes
        # overlapping at IoU 0.82 with two detections both on GT-A: 1.0000
        # under the skip, 6/11 = 0.5455 under the rule below.
        best_iou = 0.0
        best_j = -1
        for j, (gt_box, cls) in enumerate(gts):
            if cls != class_id:
                continue
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold and available[best_j]:
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
    """Per-class APs plus the mean AP over the classes that HAVE ground truth.

    A class with no ground truth in the scored set is not a class the
    detector failed at — it is a class the question was never asked
    about. :func:`per_class_ap` reports ``ap=0.0`` for it (there is no
    curve, so there is no other number it could return), and averaging
    that 0.0 in caps a perfect detector on a single-class corpus at
    0.5000. Measured, audit G finding 2. ``eval_real.per_image_ap`` has
    guarded against exactly this since it was written, with a docstring
    naming the failure; this is the same policy applied to the two paths
    that actually publish numbers — ``eval_real.summarise`` and
    ``training.evaluate_model``, both of which pass a hardcoded class
    list rather than the classes present.

    So: absent classes are reported as ``AP_<c> = nan`` — "not asked",
    which is neither 0.0 nor a silent omission that would ``KeyError``
    the report table — and excluded from the mean. The returned dict
    also carries ``classes@<iou>``, the number of classes the mean was
    actually taken over, so a headline mAP can never again be quoted
    without it being visible how many classes produced it.

    A scored set in which *no* class has ground truth raises. There is
    no mean to take, and returning ``nan`` (or 0.0) would let a training
    run log a metric, select a checkpoint on it and write it into
    ``best.pt`` while nothing was ever evaluated.
    """
    aps: Dict[str, float] = {}
    scored: List[float] = []
    absent: List[int] = []
    for c in class_ids:
        result = per_class_ap(gts_by_image, preds_by_image, c, iou_threshold)
        if result.num_gt == 0:
            aps[f"AP_{c}"] = float("nan")
            absent.append(c)
        else:
            aps[f"AP_{c}"] = result.ap
            scored.append(result.ap)

    if not scored:
        raise ValueError(
            f"mean_ap was asked for classes {list(class_ids)} and NONE of "
            f"them has any ground truth in the scored set "
            f"({len(gts_by_image)} image(s)). There is no mean to take. "
            "Refusing to return a number that would be logged, selected on "
            "and written into a checkpoint as though something had been "
            "evaluated.")
    if absent:
        log.warning(
            "mAP@%.2f is the mean over %d of the %d requested class(es): "
            "%s ha%s no ground truth in this set and %s excluded rather than "
            "averaged in as 0.0 (which would cap the mean at %.4f).",
            iou_threshold, len(scored), len(class_ids), absent,
            "ve" if len(absent) > 1 else "s",
            "were" if len(absent) > 1 else "was",
            len(scored) / len(class_ids))

    aps[f"mAP@{iou_threshold:.2f}"] = float(np.mean(scored))
    aps[f"classes@{iou_threshold:.2f}"] = float(len(scored))
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
