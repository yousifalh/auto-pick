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
