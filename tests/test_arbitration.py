"""Placement-area arbitration: two estimates, one conservative answer."""
from __future__ import annotations

import numpy as np
import pytest

from plan.arbitration import (CH_BACKGROUND, CH_BATTERY, CH_BAY,
                              CH_CARTRIDGE, CH_ELECTRONICS, CH_OBSTRUCTION,
                              arbitrate, centre_component, derived_placement,
                              direct_placement)


def _cartridge(h=60, w=60, wall=4):
    """A cartridge occupying the middle of the crop: wall ring of
    `cartridge`, bay inside."""
    m = np.zeros((h, w), np.int8)
    m[10:h - 10, 10:w - 10] = CH_CARTRIDGE
    m[10 + wall:h - 10 - wall, 10 + wall:w - 10 - wall] = CH_BAY
    return m


def test_centre_component_keeps_only_the_centre_cartridge():
    """An over-large or jittered crop catches a neighbour's edge. The
    neighbour must not contribute to the derived estimate."""
    m = np.zeros((60, 60), np.int8)
    m[20:40, 20:40] = CH_BAY            # the one we want, at the centre
    m[0:6, 0:6] = CH_CARTRIDGE          # a neighbour's corner
    keep = centre_component(m)
    assert keep[30, 30]
    assert not keep[2, 2]


def test_centre_component_spans_battery_so_a_cell_cannot_split_it():
    """A cell lying right across a bay would sever the region into two
    if battery were excluded, and the centre component would then cover
    half the cartridge."""
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:50, 28:32] = CH_BATTERY        # a full-height bar across it
    keep = centre_component(m)
    assert keep[15, 15] and keep[15, 45], "region was severed by the cell"


def test_centre_component_returns_empty_when_the_centre_is_background():
    """A badly-placed detector box: the crop centre lands on background,
    but a large blob sits elsewhere in the frame - most likely a
    neighbour's cartridge. The old behaviour fell back to the largest
    component and silently handed back that neighbour's blob; this
    must instead come back empty so the caller skips the cartridge
    rather than seating a cell in the wrong physical bay. Fails against
    the "fall back to the largest component" behaviour this replaced."""
    m = np.zeros((60, 60), np.int8)
    m[0:20, 0:20] = CH_BAY   # large, but nowhere near the (30, 30) centre
    keep = centre_component(m)
    assert not keep.any(), "fell back to a neighbour's blob instead of empty"


def test_direct_placement_is_exactly_the_bay_channel():
    m = _cartridge()
    assert np.array_equal(direct_placement(m), m == CH_BAY)


def test_derived_placement_subtracts_electronics_obstruction_and_battery():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:20, 10:50] = CH_ELECTRONICS
    m[30:34, 30:34] = CH_OBSTRUCTION
    m[40:46, 12:18] = CH_BATTERY

    d = derived_placement(m, wall_inset_px=0)
    assert not d[15, 30], "electronics not subtracted"
    assert not d[32, 32], "obstruction not subtracted"
    assert not d[43, 15], "battery not subtracted"
    assert d[25, 25], "clear floor was removed"


def test_derived_placement_erodes_by_the_wall_inset():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    d0 = derived_placement(m, wall_inset_px=0)
    d5 = derived_placement(m, wall_inset_px=5)
    assert d5.sum() < d0.sum()
    assert not d5[10, 30], "the wall band survived the inset"
    assert d5[30, 30]


def test_p_safe_is_the_intersection_not_the_union():
    """Conservative on purpose: siting a cell on a PCB is a damage
    event, skipping a cartridge costs one cycle.

    The cartridge wall ring is load-bearing here, not decoration.
    `derived_placement` never subtracts CH_CARTRIDGE, so wall pixels
    land in `derived` but never in `direct` (which is exactly
    `label_map == CH_BAY`). Without the ring, `derived` degenerates to
    exactly `direct` (see test_iou_falls_when_the_estimates_disagree),
    `direct & derived` equals `direct | derived` on every pixel, and
    this test would pass even if `arbitrate` used `|` by mistake -
    which is exactly the fixture defect this project already shipped
    once. Keep the ring if you ever "simplify" this mask.
    """
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_CARTRIDGE
    m[14:46, 14:46] = CH_BAY
    m[14:30, 14:46] = CH_ELECTRONICS    # direct says bay, derived says no

    safe, iou = arbitrate(m, wall_inset_px=0)
    assert not safe[15, 30], "P_safe took the union (electronics)"
    assert not safe[12, 30], "P_safe took the union (wall ring is derived-only)"
    assert safe[35, 30]
    assert 0.0 <= iou <= 1.0


def test_iou_is_one_when_the_estimates_agree():
    """No cartridge ring here on purpose: this fixture is an entirely
    open floor (no wall, no occluder anywhere), so `direct` and
    `derived` describe the literal same region - the agreement is
    genuine geometric identity, not the collapse-to-erode(bay) bug the
    other two tests guard against (that bug needs an *absent* ring
    plus electronics/obstruction/battery to subtract; this fixture has
    neither the ring nor anything to subtract, so there is nothing for
    the two computations to disagree about in the first place)."""
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    _, iou = arbitrate(m, wall_inset_px=0)
    assert iou == pytest.approx(1.0)


def test_iou_falls_when_the_estimates_disagree():
    """Half the bay is really PCB.

    The cartridge wall ring matters here too: `derived_placement`
    never subtracts CH_CARTRIDGE, so the ring is what makes `derived`
    genuinely larger than `direct`. Without it, `derived` collapses
    onto `direct` for *any* bay/electronics split - this is
    character-for-character the geometry bug that made P_direct and
    P_derived the same quantity computed twice on real predictions
    (IoU >= 0.95 on every crop), before the bay proxy and electronics
    module were inset by the measured wall thickness so the shell rim
    survives as `cartridge`. Drop the ring and this test goes back to
    asserting `1.0 < 0.6`.
    """
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_CARTRIDGE
    m[14:46, 14:46] = CH_BAY
    m[14:30, 14:46] = CH_ELECTRONICS    # half the bay is really PCB
    _, iou = arbitrate(m, wall_inset_px=0)
    assert iou < 0.6


def test_empty_estimates_give_zero_iou_not_a_crash():
    m = np.zeros((20, 20), np.int8)
    safe, iou = arbitrate(m, wall_inset_px=0)
    assert not safe.any()
    assert iou == 0.0


def test_arbitrate_yields_empty_p_safe_when_the_box_centres_on_a_neighbour():
    """`direct_placement` is not scoped by the crop centre, so if
    `centre_component` guessed a neighbour's blob (the old fallback),
    that guess could line up with `direct` well enough to produce a
    non-empty, plausible P_safe for the wrong physical cartridge - not
    detectably wrong downstream, since both estimates would then agree
    on the same wrong blob. Confirms the empty-centre-component fix
    closes that hole all the way through `arbitrate`, not just in
    `centre_component` in isolation."""
    m = np.zeros((60, 60), np.int8)
    m[0:20, 0:20] = CH_BAY        # a neighbour's bay, in frame but off-centre
    safe, iou = arbitrate(m, wall_inset_px=0)
    assert not safe.any(), "P_safe leaked a neighbour's placement area"
    assert iou == 0.0


def test_channel_constants_match_the_segmenter_contract():
    """plan.arbitration and recog.seg_dataset must agree, or the
    arbitration subtracts the wrong masks and nothing raises."""
    pytest.importorskip("torch")
    from recog.seg_dataset import SEG_CHANNELS

    assert SEG_CHANNELS == {
        "background": CH_BACKGROUND, "cartridge": CH_CARTRIDGE,
        "bay": CH_BAY, "electronics": CH_ELECTRONICS,
        "obstruction": CH_OBSTRUCTION, "battery": CH_BATTERY,
    }


def test_a_cell_sized_blob_admits_a_cell():
    from plan.arbitration import admits_a_cell

    m = np.zeros((80, 80), bool)
    m[10:40, 10:24] = True              # 30 x 14, a cell is 29 x 13
    assert admits_a_cell(m, cell_w_px=13, cell_h_px=29)


def test_a_boundary_rim_of_equal_area_does_not():
    """The whole point of a morphological criterion: same area, one is
    a misplacement and the other is noise.

    The rim needs a canvas wide enough to hold `blob.sum()` pixels in a
    single row - the brief's original 80-wide canvas silently clips the
    rim slice to 75 pixels (numpy does not raise on an out-of-range
    stop index), so `blob.sum() == rim.sum()` would compare 420 to 75
    and fail before the interesting assertions ever ran. Widening only
    the rim's canvas keeps the "identical area, one pixel tall" fixture
    honest without changing the blob.
    """
    from plan.arbitration import admits_a_cell

    blob = np.zeros((80, 80), bool)
    blob[10:40, 10:24] = True
    area = int(blob.sum())
    rim = np.zeros((80, area + 10), bool)
    rim[5, 5:5 + area] = True

    assert blob.sum() == rim.sum()
    assert admits_a_cell(blob, 13, 29)
    assert not admits_a_cell(rim, 13, 29)


def test_admits_a_cell_tries_both_orientations():
    from plan.arbitration import admits_a_cell

    m = np.zeros((80, 80), bool)
    m[10:24, 10:40] = True              # 14 tall, 30 wide - rotated
    assert admits_a_cell(m, cell_w_px=13, cell_h_px=29)


def test_empty_region_admits_nothing():
    from plan.arbitration import admits_a_cell

    assert not admits_a_cell(np.zeros((40, 40), bool), 13, 29)


def test_calibrate_picks_the_smallest_tau_meeting_the_budget():
    # `calibrate` is not imported here: this test reaches it through the
    # `calibrate_from_pairs` helper below (:262), which does its own import.
    # (iou, admits_a_cell_in_the_optimistic_error)
    records = [(0.99, False)] * 90 + [(0.50, True)] * 10
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert 0.50 < out["tau"] <= 0.99
    assert out["fail_rate"] <= 0.05
    assert out["rejected_fraction"] == pytest.approx(0.10, abs=0.02)


def test_calibrate_reports_failure_when_no_tau_works():
    """A negative result about the design, reported as one rather than
    tuned around."""
    records = [(0.99, True)] * 100
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert out["tau"] is None
    assert "no threshold" in out["note"].lower()


def test_calibrate_maximises_throughput_subject_to_safety():
    """Two thresholds both meet the budget; the smaller must win
    because it accepts more cartridges."""
    records = [(0.70, False)] * 50 + [(0.95, False)] * 50
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert out["tau"] <= 0.70
    assert out["rejected_fraction"] == pytest.approx(0.0)


def calibrate_from_pairs(pairs, fail_budget):
    """Adapter so the tests read as data rather than as fixtures."""
    from recog.calibrate_tau import calibrate
    return calibrate([{"iou": i, "admits_cell": a} for i, a in pairs],
                     fail_budget=fail_budget)


def test_delta_cells_is_zero_for_a_perfect_prediction():
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from recog.seg_ablation import delta_cells

    m = np.zeros((288, 131), np.int8)
    m[5:283, 5:126] = CH_CARTRIDGE
    m[12:276, 12:119] = CH_BAY
    assert delta_cells(m, m.copy(), mm_per_px=0.625) == 0


def test_delta_cells_is_positive_when_the_prediction_loses_room():
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_OBSTRUCTION
    from recog.seg_ablation import delta_cells

    gt = np.zeros((288, 131), np.int8)
    gt[5:283, 5:126] = CH_CARTRIDGE
    gt[12:276, 12:119] = CH_BAY

    pred = gt.copy()
    pred[12:150, 12:119] = CH_OBSTRUCTION      # half the bay hallucinated

    assert delta_cells(gt, pred, mm_per_px=0.625) > 0


def test_delta_cells_is_negative_when_the_prediction_claims_too_much():
    """An optimistic prediction packs MORE cells than truth allows. A
    negative delta is the damage case, not a good score."""
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS
    from recog.seg_ablation import delta_cells

    gt = np.zeros((288, 131), np.int8)
    gt[5:283, 5:126] = CH_CARTRIDGE
    gt[12:276, 12:119] = CH_BAY
    gt[12:100, 12:119] = CH_ELECTRONICS

    pred = gt.copy()
    pred[12:100, 12:119] = CH_BAY              # PCB predicted as placeable

    assert delta_cells(gt, pred, mm_per_px=0.625) < 0
