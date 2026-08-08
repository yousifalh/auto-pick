"""Digital-twin tests.

Covers the match-or-insert semantics of ``EnvironmentModel`` and the
bookkeeping helpers on ``OccupancyGrid``.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox, ClassLabel, Detection, Snapshot
from plan.scene import (
    Battery, Cartridge, CellState, EnvironmentModel, OccupancyGrid,
    WorkspaceBounds,
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


def test_update_drops_disappeared_cartridges():
    env = EnvironmentModel(workspace=_ws())
    env.update_from_snapshot(_snap([
        Detection(BBox(0, 0, 10, 10), ClassLabel.CARTRIDGE, 1.0),
    ]))
    env.update_from_snapshot(_snap([]))  # nothing this frame
    assert env.cartridges == {}


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
