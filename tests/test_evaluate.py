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


def test_per_class_ap_reports_no_gt_through_num_gt_not_through_ap():
    """A class with no ground truth: ``num_gt`` carries the meaning.

    This test used to assert only ``ap == 0.0``, which read as "0.0 is the
    right AP for an absent class" — and ``mean_ap`` duly averaged that 0.0
    in, capping a perfect detector on a single-class corpus at 0.5000. Its
    sibling in tests/test_dataset.py,
    ``test_per_image_ap_uses_only_the_classes_present``, asserted the
    opposite policy for the per-image path, so the suite asserted both at
    once and never tested the path in between.

    Reconciled: ``per_class_ap`` still returns ``ap = 0.0``, because it
    returns a float and there is no curve to compute — but that 0.0 is a
    PLACEHOLDER, and ``num_gt = 0`` is the field that says so. The policy
    now lives in one place, ``mean_ap``, which reads ``num_gt`` and
    excludes the class (see the tests below). Both paths agree.
    """
    gt = {0: []}
    pred = {0: [((10.0, 10.0, 50.0, 50.0), 1, 0.9)]}
    r = per_class_ap(gt, pred, class_id=1)
    assert r.num_gt == 0, "num_gt is what marks the class as never asked about"
    assert r.ap == 0.0
    assert len(r.recall) == 0 and len(r.precision) == 0, (
        "there is no PR curve for a class with no ground truth; if one ever "
        "appears here the 0.0 above has become a real measurement")


def test_mean_ap_excludes_a_class_that_has_no_ground_truth():
    """A perfect detector on a single-class corpus must score 1.0, not 0.5.

    122 of the 1000 scenes in recog/dataset3d contain no cartridge, and the
    repo ships single-SKU configs (synth3d_18650, the four
    segmentation_cad_control_holdout_*) where this is certain rather than
    likely. Both publishing paths — eval_real.summarise and
    training.evaluate_model — pass a hardcoded two-class list, so the guard
    has to live in mean_ap.
    """
    box = (10.0, 10.0, 50.0, 50.0)
    gt = {0: [(box, 1)]}
    pred = {0: [(box, 1, 0.9)]}

    aps = mean_ap(gt, pred, class_ids=[1, 2])
    assert aps["AP_1"] == pytest.approx(1.0)
    assert np.isnan(aps["AP_2"]), (
        "an absent class must be reported as 'not asked' rather than as a "
        "score of 0.0")
    assert aps["mAP@0.50"] == pytest.approx(1.0)
    assert aps["classes@0.50"] == 1.0, (
        "the mAP must say how many classes it is the mean of, or an absent "
        "class is invisible in the published number")
    # And it agrees with asking for the present class only.
    assert aps["mAP@0.50"] == pytest.approx(
        mean_ap(gt, pred, class_ids=[1])["mAP@0.50"])


def test_mean_ap_refuses_a_set_with_no_ground_truth_at_all():
    """Nothing was evaluated, so there is no number to log or select on."""
    pred = {0: [((10.0, 10.0, 50.0, 50.0), 1, 0.9)]}
    with pytest.raises(ValueError, match="NONE of them has any ground truth"):
        mean_ap({0: []}, pred, class_ids=[1, 2])


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


# ------------------------------------------- the eleven recall levels ----
#
# The protocol is eleven EXACT tenths. `np.linspace(0, 1, 11)` is not that:
# it returns 0.30000000000000004, 0.6000000000000001 and 0.7000000000000001,
# so a curve whose recall lands exactly on 0.3, 0.6 or 0.7 fails its own
# grid point and forfeits a whole 1/11 = 0.0909 bin. No test in this file
# used to touch those three values — test_voc_ap_perfect uses {0, 0.5, 1.0},
# all exactly representable — which is why the defect survived.


@pytest.mark.parametrize("found, expected_bins", [
    (3, 4),    # recall 0.3 - lost a bin under linspace
    (4, 5),    # recall 0.4 - exactly representable, never affected
    (5, 6),
    (6, 7),    # recall 0.6 - lost a bin
    (7, 8),    # recall 0.7 - lost a bin
    (10, 11),
])
def test_voc_ap_awards_every_tenth_the_recall_actually_reaches(
        found, expected_bins):
    """``found`` of 10 objects at precision 1.0 is ``found + 1`` bins of 1/11.

    Hand-computed: recall reaches ``found/10`` at precision 1.0, so every
    level from 0.0 up to and including ``found/10`` scores 1.0 and the rest
    score 0. The three cases that used to be wrong: 3/10 gave 0.2727 for
    0.3636, 6/10 gave 0.5455 for 0.6364, 7/10 gave 0.6364 for 0.7273.
    """
    recall = np.array([(i + 1) / 10.0 for i in range(found)])
    precision = np.ones(found)
    assert voc_ap(recall, precision) == pytest.approx(expected_bins / 11.0)


def test_the_recall_levels_are_exact_tenths_not_linspace():
    """Guards the premise: linspace really does miss three of the eleven."""
    from recog.evaluate import _RECALL_LEVELS

    assert list(_RECALL_LEVELS) == [k / 10.0 for k in range(11)]
    linspace = np.linspace(0.0, 1.0, 11)
    off_grid = [k for k in range(11) if linspace[k] != k / 10.0]
    assert off_grid == [3, 6, 7], (
        f"linspace no longer misses the tenths it used to ({off_grid}); the "
        "comment explaining why _RECALL_LEVELS exists needs revisiting")


def test_voc_ap_end_to_end_through_per_class_ap_at_six_of_ten():
    """The same 6-of-10 case, driven through the real matcher."""
    gts = {i: [((0.0, 0.0, 10.0, 10.0), 1)] for i in range(10)}
    preds = {i: [((0.0, 0.0, 10.0, 10.0), 1, 0.9 - 0.01 * i)]
             for i in range(6)}
    assert per_class_ap(gts, preds, class_id=1).ap == pytest.approx(7 / 11.0)


# ------------------------------------------------ duplicate detections ----

def test_a_second_detection_on_the_same_object_is_a_false_positive():
    """VOC's rule: best-overlapping GT first, THEN ask if it is free.

    Two ground-truth boxes overlapping at IoU 0.8182 — a plausible pair of
    stacked or abutting cells, and the hard validation subset is selected
    specifically for scenes where parts occlude each other. Two detections
    are both on GT-A:

        pred0  IoU vs A = 1.0000   vs B = 0.8182   best = A
        pred1  IoU vs A = 0.9608   vs B = 0.8519   best = A (taken)

    Matching against only the *available* GT lets pred1 fall through to its
    second-best box and score a true positive there, so a detector that
    found one object is credited with two: AP 1.0000 instead of 6/11.
    """
    gt_a = (0.0, 0.0, 100.0, 100.0)
    gt_b = (0.0, 10.0, 100.0, 110.0)
    pred0 = gt_a                       # exact hit on A
    pred1 = (0.0, 2.0, 100.0, 102.0)   # also closest to A

    from recog.evaluate import _iou
    assert _iou(pred0, gt_a) == pytest.approx(1.0)
    assert _iou(pred0, gt_b) == pytest.approx(9000 / 11000)
    assert _iou(pred1, gt_a) == pytest.approx(9800 / 10200)
    assert _iou(pred1, gt_b) == pytest.approx(9200 / 10800)
    assert _iou(pred1, gt_a) > _iou(pred1, gt_b), "premise: A is pred1's best"

    gts = {0: [(gt_a, 1), (gt_b, 1)]}
    preds = {0: [(pred0, 1, 0.95), (pred1, 1, 0.90)]}
    r = per_class_ap(gts, preds, class_id=1, iou_threshold=0.5)

    # tp = [1, 0]; recall = [0.5, 0.5]; precision = [1.0, 0.5]
    assert r.recall.tolist() == pytest.approx([0.5, 0.5])
    assert r.precision.tolist() == pytest.approx([1.0, 0.5])
    assert r.ap == pytest.approx(6 / 11.0)


def test_two_detections_on_two_distinct_objects_both_still_score():
    """The guard on the guard: the fix must not turn real hits into FPs."""
    gt_a = (0.0, 0.0, 100.0, 100.0)
    gt_b = (0.0, 200.0, 100.0, 300.0)
    gts = {0: [(gt_a, 1), (gt_b, 1)]}
    preds = {0: [(gt_a, 1, 0.95), (gt_b, 1, 0.90)]}
    r = per_class_ap(gts, preds, class_id=1, iou_threshold=0.5)
    assert r.ap == pytest.approx(1.0)
    assert r.recall.tolist() == pytest.approx([0.5, 1.0])
