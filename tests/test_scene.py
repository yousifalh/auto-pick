"""Digital-twin tests.

Covers the tracking semantics of ``EnvironmentModel`` (match, retain,
re-acquire, expire), the occupancy state machine on ``Cartridge``
(``reserve`` / ``confirm`` / ``release_planned``), and the bookkeeping
helpers on ``OccupancyGrid``.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox, ClassLabel, Detection, Snapshot
from plan.scene import (
    Battery, Cartridge, CellState, EnvironmentModel, OccupancyGrid,
    PlacementCollision, Reservation, TrackingConfig, WorkspaceBounds,
)


# --------------------------- OccupancyGrid --------------------------------

class TestOccupancyGrid:
    def test_initial_state_all_free(self):
        g = OccupancyGrid(rows=4, cols=5, resolution_mm=1.5)
        assert g.free_count() == 20
        assert g.placed_count() == 0
        assert g.planned_count() == 0

    def test_set_and_get(self):
        g = OccupancyGrid(rows=2, cols=2, resolution_mm=1.0)
        g.set(0, 1, CellState.PLANNED)
        assert g.get(0, 1) == CellState.PLANNED
        assert g.get(0, 0) == CellState.FREE

    def test_transitions(self):
        g = OccupancyGrid(rows=2, cols=2, resolution_mm=1.0)
        g.set(0, 0, CellState.PLANNED)
        assert g.planned_count() == 1
        g.set(0, 0, CellState.PLACED)
        assert g.placed_count() == 1
        assert g.planned_count() == 0

    def test_mask_of_multiple(self):
        g = OccupancyGrid(rows=2, cols=2, resolution_mm=1.0)
        g.set(0, 0, CellState.FORBIDDEN)
        g.set(1, 1, CellState.PLANNED)
        m = g.mask_of(CellState.FORBIDDEN, CellState.PLANNED)
        assert m[0, 0] and m[1, 1]
        assert not m[0, 1] and not m[1, 0]


# --------------------------- Cartridge ------------------------------------

def test_cartridge_mark_cell_raises_without_grid():
    c = Cartridge(id=0, bbox=BBox(0, 0, 10, 10), confidence=1.0)
    with pytest.raises(RuntimeError):
        c.mark_cell(0, 0, CellState.PLANNED)


def test_cartridge_mark_cell_propagates_to_grid():
    c = Cartridge(id=0, bbox=BBox(0, 0, 10, 10), confidence=1.0,
                  occupancy=OccupancyGrid(rows=2, cols=2, resolution_mm=1.0))
    c.mark_cell(1, 1, CellState.PLACED)
    assert c.occupancy.placed_count() == 1


# ------------------- reserve / confirm / release_planned ------------------
#
# `tests/test_scene.py` had NO coverage of any of these (audit H): the
# reservation tests lived in test_planner.py, went through the packer,
# and were all single-frame. The state machine they drive is what decides
# whether a slot the robot has already filled can be planned into again,
# so it is tested here at its own altitude, in cells, with no packer in
# the way.

def _res(row0, col0, row1, col1, resolution_mm=1.0,
         state=CellState.PLANNED, wx=None, wy=None):
    """A reservation covering the half-open cell block, in millimetres.

    The workspace millimetres default to the strip millimetres, which is
    the single-cartridge case (origin at 0). Tests that need two
    cartridges in one workspace pass them explicitly.
    """
    x_mm, y_mm = col0 * resolution_mm, row0 * resolution_mm
    return Reservation(
        row=row0, col=col0, row0=row0, row1=row1, col0=col0, col1=col1,
        x_mm=x_mm, y_mm=y_mm,
        w_mm=(col1 - col0) * resolution_mm,
        h_mm=(row1 - row0) * resolution_mm,
        wx_mm=x_mm if wx is None else wx,
        wy_mm=y_mm if wy is None else wy,
        state=state,
    )


def _cartridge(rows=8, cols=8):
    return Cartridge(
        id=0, bbox=BBox(0, 0, 10, 10), confidence=1.0,
        occupancy=OccupancyGrid(rows=rows, cols=cols, resolution_mm=1.0))


def test_reserve_marks_the_whole_block_and_records_the_footprint():
    c = _cartridge()
    c.reserve(_res(0, 0, 4, 3))
    assert c.occupancy.planned_count() == 12
    assert (0, 0) in c.reservations
    assert c.reservations[(0, 0)].cell_count == 12


def test_reserve_refuses_a_footprint_that_quantises_to_nothing():
    """A reservation covering no cells reserves nothing while reading like
    a success."""
    c = _cartridge()
    with pytest.raises(ValueError, match="degenerate"):
        c.reserve(_res(2, 2, 2, 5))


def test_reserve_raises_without_a_grid():
    c = Cartridge(id=0, bbox=BBox(0, 0, 10, 10), confidence=1.0)
    with pytest.raises(RuntimeError, match="occupancy"):
        c.reserve(_res(0, 0, 2, 2))


def test_reserve_refuses_forbidden_cells_and_overlapping_millimetres():
    c = _cartridge()
    c.occupancy.set(1, 1, CellState.FORBIDDEN)
    with pytest.raises(PlacementCollision, match="FORBIDDEN"):
        c.reserve(_res(0, 0, 3, 3))

    c = _cartridge()
    c.reserve(_res(0, 0, 4, 4))
    with pytest.raises(PlacementCollision, match="PLANNED"):
        c.reserve(_res(2, 2, 6, 6))
    # Abutting is not overlapping: neighbours in one pack share a
    # boundary cell without sharing any physical space.
    c.reserve(_res(0, 4, 4, 8))
    assert len(c.reservations) == 2


def test_a_reservation_abutting_a_placed_battery_cannot_unplace_it():
    """PLACED -> PLANNED -> FREE was reachable (audit H, finding 5).

    `reserve` passed no `only_from`, so a block abutting a placed battery
    took the shared boundary cells PLANNED, and `release_planned` (or a
    failed `confirm`) then took those same cells FREE — 44 cells of a
    battery that is physically in the tray reading FREE, and the packer
    free to seat another one into them. Latent, because today's packer
    masks PLACED and rounds outward; asserted anyway, because it is the
    invariant `reserve`'s whole docstring is about.
    """
    c = _cartridge()
    a = _res(0, 0, 4, 4)
    c.reserve(a)
    c.confirm(a.row, a.col, True)
    assert c.occupancy.placed_count() == 16

    # Abutting in millimetres, sharing the boundary column in cells.
    b = _res(0, 3, 4, 7)
    b.x_mm, b.w_mm = 4.0, 3.0          # 4..7 mm: no physical overlap
    b.wx_mm = 4.0
    c.reserve(b)
    assert c.occupancy.placed_count() == 16, (
        "a PLACED cell was taken PLANNED by a neighbour's reservation")

    assert c.release_planned() == 1
    assert c.occupancy.placed_count() == 16, (
        "releasing the neighbour freed cells of a battery that is "
        "physically in the tray")


def test_confirm_success_then_failure_on_one_anchor_is_refused():
    """The second call used to delete the PLACED `Reservation` while
    leaving its cells PLACED — the millimetre interlock silently loses
    that battery's footprint while the grid still shows it."""
    c = _cartridge()
    r = _res(0, 0, 4, 4)
    c.reserve(r)
    c.confirm(0, 0, True)
    with pytest.raises(PlacementCollision, match="already PLACED"):
        c.confirm(0, 0, False)
    assert c.reservations[(0, 0)].state is CellState.PLACED
    assert c.occupancy.placed_count() == 16


def test_confirm_of_an_unplanned_anchor_raises():
    c = _cartridge()
    with pytest.raises(KeyError, match="no reservation"):
        c.confirm(1, 1, True)


def test_confirm_failure_frees_the_whole_block_and_drops_the_reservation():
    c = _cartridge()
    c.reserve(_res(0, 0, 4, 4))
    c.confirm(0, 0, False)
    assert c.occupancy.free_count() == 64
    assert c.reservations == {}


def test_release_planned_drops_planned_and_keeps_placed():
    c = _cartridge()
    a, b = _res(0, 0, 4, 4), _res(4, 0, 8, 4)
    c.reserve(a)
    c.confirm(a.row, a.col, True)
    c.reserve(b)
    assert c.occupancy.planned_count() == 16

    assert c.release_planned() == 1
    assert c.occupancy.planned_count() == 0
    assert c.occupancy.placed_count() == 16
    assert list(c.reservations) == [(0, 0)]
    assert c.release_planned() == 0


def test_invalidate_geometry_keeps_the_batteries():
    c = _cartridge()
    c.placeable_rectangle = BBox(0, 0, 10, 10)
    c.mm_per_px = 0.38
    r = _res(0, 0, 4, 4)
    c.reserve(r)
    c.confirm(0, 0, True)

    c.invalidate_geometry()

    assert c.occupancy is None and c.placeable_rectangle is None
    assert c.mm_per_px is None
    assert len(c.placed_reservations()) == 1, (
        "a measurement was thrown away, not a battery")


# --------------------------- Environment ----------------------------------

def _ws():
    return WorkspaceBounds(-100, 100, -100, 100)


def _snap(dets):
    return Snapshot(detections=dets)


def test_update_inserts_new_cartridge():
    env = EnvironmentModel(workspace=_ws())
    d = Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)
    env.update_from_snapshot(_snap([d]))
    assert len(env.cartridges) == 1
    cid = next(iter(env.cartridges))
    assert env.cartridges[cid].confidence == 0.9


def test_update_preserves_cartridge_across_frames():
    env = EnvironmentModel(workspace=_ws())
    d1 = Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)
    env.update_from_snapshot(_snap([d1]))
    cid = next(iter(env.cartridges))
    # Attach occupancy data
    env.cartridges[cid].occupancy = OccupancyGrid(
        rows=4, cols=4, resolution_mm=1.0)

    d2 = Detection(BBox(2, 2, 102, 102), ClassLabel.CARTRIDGE, 0.95)
    env.update_from_snapshot(_snap([d2]))
    assert cid in env.cartridges  # same ID kept
    assert env.cartridges[cid].occupancy is not None
    assert env.cartridges[cid].bbox.xmin == 2  # bbox updated


def test_update_keeps_an_undetected_cartridge_as_not_visible():
    """CORRECTED from `test_update_drops_disappeared_cartridges`, which
    asserted `env.cartridges == {}` after one missing frame and was
    therefore the defect's own regression test.

    The delete it asserted took the whole `Cartridge`: the occupancy
    grid with every PLACED cell, the `reservations` dict (the only input
    to the millimetre collision interlock), the placement rectangle, the
    scale and the ID. Measured (audit H finding 1): 572 placed cells ->
    one undetected frame -> a new ID holding 0, whose first place target
    was byte-identical to one already executed, with nothing left to
    compare it against. A single low-confidence frame is the ORDINARY
    case, not the adversarial one.

    What it asserts now: the cartridge is retained, marked not-visible,
    and keeps everything physical. The dropout is counted rather than
    silent.
    """
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.CARTRIDGE, 1.0),
    ]))
    cid = next(iter(env.cartridges))
    env.cartridges[cid].occupancy = OccupancyGrid(
        rows=4, cols=4, resolution_mm=1.0)
    env.cartridges[cid].occupancy.set(0, 0, CellState.PLACED)

    env.update_from_snapshot(_snap([]))  # nothing this frame

    assert cid in env.cartridges, "one missing frame must not erase the twin"
    ctg = env.cartridges[cid]
    assert not ctg.visible
    assert ctg.frames_since_seen == 1
    assert ctg.occupancy.placed_count() == 1, "the physical record survives"
    # It is not in this frame, so it indexes nothing in this frame's
    # cartridge_masks.
    assert ctg.detection_index == -1
    assert env.track_dropout_count == 1
    assert env.visible_cartridges() == []
    assert env.summary()["cartridges"] == 1
    assert env.summary()["visible_cartridges"] == 0


def test_a_returning_cartridge_keeps_its_id_and_its_placed_cells():
    """Present / absent / present, the same box each time.

    The audit's scenario 1 at the twin's own altitude: the returning
    cartridge must be the SAME entity, not a fresh empty one, or the
    next queue plans into slots that are already full.
    """
    env = EnvironmentModel(workspace=_ws())
    box = Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)
    env.update_from_snapshot(_snap([box]))
    cid = next(iter(env.cartridges))
    env.cartridges[cid].occupancy = OccupancyGrid(
        rows=4, cols=4, resolution_mm=1.0)
    env.cartridges[cid].occupancy.set(1, 1, CellState.PLACED)

    env.update_from_snapshot(_snap([]))
    env.update_from_snapshot(_snap([box]))

    assert list(env.cartridges) == [cid], "a new ID means a forgotten tray"
    assert env.cartridges[cid].visible
    assert env.cartridges[cid].frames_since_seen == 0
    assert env.cartridges[cid].occupancy.placed_count() == 1
    assert env.track_reacquired_count == 1
    assert env.track_expired_count == 0


def test_a_track_expires_after_its_bound_and_says_what_it_forgot():
    """Memory is bounded, and the bound is the moment physical state
    genuinely leaves the model — so it is counted in the twin's own
    currency (cells, batteries) and not merely in tracks."""
    env = EnvironmentModel(
        workspace=_ws(), tracking=TrackingConfig(max_missing_frames=2))
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)]))
    cid = next(iter(env.cartridges))
    ctg = env.cartridges[cid]
    ctg.occupancy = OccupancyGrid(rows=4, cols=4, resolution_mm=1.0)
    ctg.occupancy.set_block(0, 2, 0, 2, CellState.PLACED)
    ctg.reservations[(0, 0)] = _res(0, 0, 2, 2, state=CellState.PLACED)

    for expected in (1, 2):
        env.update_from_snapshot(_snap([]))
        assert cid in env.cartridges
        assert env.cartridges[cid].frames_since_seen == expected

    env.update_from_snapshot(_snap([]))  # one past the bound

    assert cid not in env.cartridges
    assert env.track_expired_count == 1
    assert env.expired_placed_cell_count == 4
    assert env.expired_placed_battery_count == 1
    assert env.track_dropout_count == 1, "one dropout event, not one a frame"


def test_matching_is_globally_best_not_first_come():
    """A detection at IoU 1.000 must never lose its track to one at 0.667.

    Audit H finding 4, executed: the old loop walked detections in
    snapshot order and handed each the best still-unmatched track, so a
    detection overlapping BOTH cartridges at 0.667 took id 0 on dict
    insertion order, inherited its PLACED cells while wearing the other
    cartridge's box, and the true id-0 detection — IoU 1.000 — arrived to
    find it taken and was inserted as a new empty id 2. One frame, and
    the twin's memory of which cartridge holds what is transposed.
    """
    env = EnvironmentModel(workspace=_ws())
    a = BBox(0, 0, 100, 100)
    b = BBox(60, 0, 160, 100)
    env.update_from_snapshot(_snap([
        Detection(a, ClassLabel.CARTRIDGE, 0.9),
        Detection(b, ClassLabel.CARTRIDGE, 0.9),
    ]))
    ids = sorted(env.cartridges)
    assert len(ids) == 2
    assert env.cartridges[ids[0]].bbox == a

    # The ambiguous detection comes FIRST, which is what used to decide
    # the outcome: it ties against both tracks and the tie broke on dict
    # insertion order. The second detection is id 0's box exactly.
    ambiguous = BBox(30, 0, 130, 100)
    assert ambiguous.iou(a) == pytest.approx(ambiguous.iou(b))
    assert ambiguous.iou(a) >= env.tracking.iou_match_threshold
    env.update_from_snapshot(_snap([
        Detection(ambiguous, ClassLabel.CARTRIDGE, 0.9),
        Detection(a, ClassLabel.CARTRIDGE, 0.9),
    ]))

    assert sorted(env.cartridges) == ids, "no new ID may be minted here"
    assert env.cartridges[ids[0]].bbox == a, (
        "the IoU-1.000 detection lost id 0 to a weaker one — the twin's "
        "record of which cartridge holds what is now transposed")
    assert env.cartridges[ids[1]].bbox == ambiguous


def test_the_match_threshold_comes_from_the_config():
    """It used to be a function default that no caller and no config
    could reach, governing whether the robot's memory survives a
    frame."""
    strict = EnvironmentModel(
        workspace=_ws(), tracking=TrackingConfig(iou_match_threshold=0.95,
                                                 duplicate_iou_threshold=0.95))
    loose = EnvironmentModel(
        workspace=_ws(), tracking=TrackingConfig(iou_match_threshold=0.2,
                                                 duplicate_iou_threshold=0.2))
    first = Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)
    # IoU 0.68 - below the strict threshold, above the loose one.
    second = Detection(BBox(0, 0, 100, 68), ClassLabel.CARTRIDGE, 0.9)
    for env in (strict, loose):
        env.update_from_snapshot(_snap([first]))
        env.update_from_snapshot(_snap([second]))

    assert len(strict.cartridges) == 2, "the strict threshold must refuse"
    assert len(loose.cartridges) == 1, "the loose threshold must accept"


def test_a_second_box_over_one_cartridge_does_not_become_a_second_entry():
    """Audit H finding 3: a split or duplicated cartridge box produced two
    twin entries over one physical object, each with its own grid and its
    own reservations, and the millimetre interlock is per-entry — so they
    double-booked the same space with nothing able to see it."""
    env = EnvironmentModel(workspace=_ws())
    box = BBox(0, 0, 100, 100)
    dup = BBox(20, 20, 120, 120)   # IoU 0.47: matches nothing, overlaps a lot
    assert dup.iou(box) < env.tracking.iou_match_threshold
    assert dup.iou(box) >= env.tracking.duplicate_iou_threshold

    env.update_from_snapshot(_snap([
        Detection(box, ClassLabel.CARTRIDGE, 0.9),
        Detection(dup, ClassLabel.CARTRIDGE, 0.9),
    ]))

    assert len(env.cartridges) == 1
    assert env.duplicate_detection_count == 1


def test_a_moved_cartridge_keeps_its_id_and_loses_its_stale_geometry():
    """Audit H finding 2. The rectangle and the grid are PIXELS of the
    frame that measured them; a matched cartridge kept them for ever, so
    a box that moved 38 mm kept commanding the millimetres it had before
    it moved. The identity and the batteries survive the move (they moved
    WITH the box); the measurement does not."""
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 100, 100), ClassLabel.CARTRIDGE, 0.9)]))
    cid = next(iter(env.cartridges))
    ctg = env.cartridges[cid]
    ctg.placeable_rectangle = BBox(5, 5, 95, 95)
    ctg.mm_per_px = 1.0
    ctg.occupancy = OccupancyGrid(rows=4, cols=4, resolution_mm=1.0)
    ctg.reservations[(0, 0)] = _res(0, 0, 2, 2, state=CellState.PLACED)

    # 1 px at 1.0 mm/px is under the 1.5 mm refresh threshold: detector
    # jitter must not throw the measurement away every frame.
    env.update_from_snapshot(_snap([
        Detection(BBox(1, 1, 101, 101), ClassLabel.CARTRIDGE, 0.9)]))
    assert env.cartridges[cid].placeable_rectangle is not None
    assert env.geometry_refresh_count == 0

    # 10 px = 10 mm: the grid could have told the truth about that.
    env.update_from_snapshot(_snap([
        Detection(BBox(11, 11, 111, 111), ClassLabel.CARTRIDGE, 0.9)]))
    ctg = env.cartridges[cid]
    assert list(env.cartridges) == [cid], "a moved cartridge is not a new one"
    assert env.geometry_refresh_count == 1
    assert ctg.placeable_rectangle is None and ctg.occupancy is None
    assert ctg.mm_per_px is None and ctg.pcb_mask is None
    assert len(ctg.placed_reservations()) == 1, (
        "the batteries moved WITH the box; only the measurement is stale")


def test_tracking_config_refuses_numbers_that_disable_it():
    with pytest.raises(ValueError, match="iou_match_threshold"):
        TrackingConfig(iou_match_threshold=0.0)
    with pytest.raises(ValueError, match="duplicate_iou_threshold"):
        TrackingConfig(iou_match_threshold=0.5, duplicate_iou_threshold=0.6)
    with pytest.raises(ValueError, match="max_missing_frames"):
        TrackingConfig(max_missing_frames=-1)
    with pytest.raises(ValueError, match="geometry_refresh_mm"):
        TrackingConfig(geometry_refresh_mm=0.0)
    with pytest.raises(ValueError, match="unknown keys"):
        TrackingConfig.from_dict({"max_missing_frame": 3})
    assert TrackingConfig.from_dict(
        {"max_missing_frames": 3}).max_missing_frames == 3
    assert TrackingConfig.from_dict(None).max_missing_frames == 5


def test_update_battery_replaces_wholesale():
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.8),
        Detection(BBox(20, 20, 30, 30), ClassLabel.BATTERY, 0.8),
    ]))
    assert len(env.batteries) == 2
    env.update_from_snapshot(_snap([
        Detection(BBox(40, 40, 50, 50), ClassLabel.BATTERY, 0.9),
    ]))
    assert len(env.batteries) == 1
    assert env.batteries[0].bbox.xmin == 40


def test_two_cartridges_get_separate_ids():
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 50, 50), ClassLabel.CARTRIDGE, 0.9),
        Detection(BBox(200, 200, 250, 250), ClassLabel.CARTRIDGE, 0.9),
    ]))
    assert len(env.cartridges) == 2
    ids = sorted(env.cartridges.keys())
    assert ids[0] != ids[1]


def test_available_batteries_filters_assigned():
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.8),
        Detection(BBox(20, 20, 30, 30), ClassLabel.BATTERY, 0.8),
    ]))
    env.batteries[0].assigned_to_pose = True
    avail = env.available_batteries()
    assert len(avail) == 1
    assert avail[0].id != env.batteries[0].id


def test_cartridge_detection_index_matches_full_detections_list_position():
    """Snapshot.cartridge_masks (Plan D Task 5) is keyed by position in
    snapshot.detections, not by position within the cartridge-only
    subset - so Cartridge.detection_index must track the former, or
    every mask lookup in the planner misaligns the moment a battery
    detection precedes a cartridge in the list."""
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.9),
        Detection(BBox(100, 100, 150, 150), ClassLabel.CARTRIDGE, 0.9),
    ]))
    cid = next(iter(env.cartridges))
    assert env.cartridges[cid].detection_index == 1


def test_summary_shape():
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.8),
        Detection(BBox(100, 100, 150, 150), ClassLabel.CARTRIDGE, 0.9),
    ]))
    s = env.summary()
    assert s["batteries"] == 1
    assert s["cartridges"] == 1
    assert s["placed_cells"] == 0
