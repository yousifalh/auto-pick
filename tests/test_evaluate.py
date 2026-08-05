"""Tests for VOC-style mAP and pose-error metrics (PPR §2 O1, O2)."""
from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox
from recog.evaluate import (
    centroid_error_px, edge_error_px, mean_ap, per_class_ap, voc_ap,
)


def test_voc_ap_perfect():
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 1.0, 1.0])
    assert voc_ap(rec, prec) == pytest.approx(1.0)


def test_voc_ap_all_miss():
    rec = np.array([0.0, 0.0, 0.0])
    prec = np.array([0.0, 0.0, 0.0])
    assert voc_ap(rec, prec) == 0.0


def test_per_class_ap_perfect_detection():
    gt = {0: [((10.0, 10.0, 50.0, 50.0), 1)]}
    pred = {0: [((10.0, 10.0, 50.0, 50.0), 1, 0.9)]}
    r = per_class_ap(gt, pred, class_id=1)
    assert r.ap == pytest.approx(1.0)
    assert r.num_gt == 1


def test_per_class_ap_wrong_class():
    gt = {0: [((10.0, 10.0, 50.0, 50.0), 1)]}
    pred = {0: [((10.0, 10.0, 50.0, 50.0), 2, 0.9)]}
    r = per_class_ap(gt, pred, class_id=1)
    assert r.ap == 0.0


def test_per_class_ap_low_iou():
    gt = {0: [((10.0, 10.0, 50.0, 50.0), 1)]}
    # Detection barely overlaps
    pred = {0: [((45.0, 45.0, 60.0, 60.0), 1, 0.9)]}
    r = per_class_ap(gt, pred, class_id=1, iou_threshold=0.5)
    assert r.ap == 0.0


def test_per_class_ap_no_gt():
    gt = {0: []}
    pred = {0: [((10.0, 10.0, 50.0, 50.0), 1, 0.9)]}
    r = per_class_ap(gt, pred, class_id=1)
    assert r.num_gt == 0
    assert r.ap == 0.0


def test_mean_ap_shape():
    gt = {0: [((0, 0, 10, 10), 1), ((20, 20, 30, 30), 2)]}
    pred = {0: [((0, 0, 10, 10), 1, 0.9),
                ((20, 20, 30, 30), 2, 0.9)]}
    aps = mean_ap(gt, pred, class_ids=[1, 2])
    assert aps["AP_1"] == pytest.approx(1.0)
    assert aps["AP_2"] == pytest.approx(1.0)
    assert aps["mAP@0.50"] == pytest.approx(1.0)


def test_centroid_error_zero():
    b = BBox(10, 10, 20, 20)
    assert centroid_error_px(b, b) == 0.0


def test_centroid_error_magnitude():
    a = BBox(0, 0, 10, 10)
    b = BBox(3, 4, 13, 14)  # centres differ by (3, 4) → distance 5
    assert centroid_error_px(a, b) == pytest.approx(5.0)


def test_edge_error_max_component():
    a = BBox(0, 0, 10, 10)
    b = BBox(1, 2, 13, 11)
    # edge diffs: 1, 2, 3, 1 → max 3
    assert edge_error_px(a, b) == pytest.approx(3.0)


def test_mean_ap_below_threshold_reduces_score():
    gt = {0: [((0, 0, 10, 10), 1), ((20, 20, 30, 30), 1)]}
    pred = {0: [((0, 0, 10, 10), 1, 0.9)]}  # miss the second
    aps = mean_ap(gt, pred, class_ids=[1])
    assert aps["AP_1"] < 1.0
