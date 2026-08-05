"""End-to-end planner tests.

Builds a deterministic snapshot and verifies that the planner produces a
well-formed queue with sensible row-major ordering, nearest-battery
assignment, and that cells get marked PLANNED.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox, ClassLabel, Detection, Snapshot
from plan.placement_area import PlacementAreaExtractor
from plan.planner import Planner, PlannerConfig
from plan.scene import CellState, WorkspaceBounds


def _synth_image(H=600, W=800) -> np.ndarray:
    """A large green cartridge with a central PCB — big enough that the
    FFDH step has room to pack several 18.5×65 mm batteries.

    At mm_per_px=0.38 the cartridge interior (roughly 700×500 px) maps
    to ~266×190 mm, which fits ~20 18650 cells."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[50:550, 50:750, 0] = 40
    img[50:550, 50:750, 1] = 200
    img[50:550, 50:750, 2] = 40
    # Small central PCB
    img[270:330, 370:430] = 20
    return img


def _make_planner():
    cfg = PlannerConfig(
        battery_width_mm=18.5, battery_length_mm=65.0,
        mm_per_px=0.38,
    )
    ext = PlacementAreaExtractor(safety_margin_px=3,
                                 mm_per_cell=1.5, mm_per_px=0.38)
    ws = WorkspaceBounds(-350, 350, -350, 350)
    return Planner(cfg, ext, ws)


def _snapshot_with_cart_and_batteries(batt_centres):
    dets = [Detection(BBox(45, 45, 755, 555), ClassLabel.CARTRIDGE, 0.95)]
    for i, (cx, cy) in enumerate(batt_centres):
        dets.append(Detection(
            BBox(cx - 5, cy - 10, cx + 5, cy + 10),
            ClassLabel.BATTERY, 0.9,
        ))
    return Snapshot(detections=dets, image_shape=(600, 800))


def test_cycle_produces_poses():
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries(
        [(5, 5), (10, 5), (15, 5), (5, 30)])
    queue = planner.cycle(snap, _synth_image())
    assert len(queue) > 0
    for p in queue:
        assert p.cartridge_id >= 0
        assert 0 <= p.grid_row < 10_000
        assert p.pick.z_mm == planner.cfg.pick_approach_height_mm
        assert p.place.z_mm == planner.cfg.place_insert_height_mm


def test_cycle_marks_cells_planned():
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5), (10, 10)])
    queue = planner.cycle(snap, _synth_image())
    cid = next(iter(planner.env.cartridges))
    ctg = planner.env.cartridge(cid)
    planned = ctg.occupancy.planned_count()
    assert planned == len(queue)


def test_queue_never_exceeds_batteries():
    planner = _make_planner()
    # Only 2 batteries; queue must never grow past that
    snap = _snapshot_with_cart_and_batteries([(5, 5), (15, 5)])
    queue = planner.cycle(snap, _synth_image())
    assert len(queue) <= 2


def test_confirm_placement_success_marks_placed():
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5)])
    queue = planner.cycle(snap, _synth_image())
    assert len(queue) >= 1
    pose = queue[0]
    planner.confirm_placement(pose.cartridge_id,
                              pose.grid_row, pose.grid_col, True)
    ctg = planner.env.cartridge(pose.cartridge_id)
    assert ctg.occupancy.get(pose.grid_row,
                             pose.grid_col) == CellState.PLACED


def test_confirm_placement_failure_reverts_to_free():
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5)])
    queue = planner.cycle(snap, _synth_image())
    pose = queue[0]
    planner.confirm_placement(pose.cartridge_id,
                              pose.grid_row, pose.grid_col, False)
    ctg = planner.env.cartridge(pose.cartridge_id)
    assert ctg.occupancy.get(pose.grid_row,
                             pose.grid_col) == CellState.FREE


def test_row_major_ordering():
    """Placements inside a cartridge must come out row-major."""
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries(
        [(5, 5), (10, 5), (5, 30), (10, 30)])
    queue = planner.cycle(snap, _synth_image())
    assert len(queue) >= 2
    # Inside one cartridge, poses by (grid_row, grid_col) must be
    # non-decreasing.
    by_cart = {}
    for p in queue:
        by_cart.setdefault(p.cartridge_id, []).append(p)
    for poses in by_cart.values():
        keys = [(p.grid_row, p.grid_col) for p in poses]
        assert keys == sorted(keys)


def test_planner_config_from_dict():
    cfg = PlannerConfig.from_dict({
        "battery": {"diameter_mm": 21.0, "length_mm": 70.0},
        "camera": {"mm_per_px_x": 0.5, "origin_offset_x_mm": 10.0,
                   "origin_offset_y_mm": -5.0},
        "motion": {"approach_height_mm": 50.0, "insert_height_mm": 3.0},
    })
    assert cfg.battery_width_mm == 21.0
    assert cfg.battery_length_mm == 70.0
    assert cfg.mm_per_px == 0.5
    assert cfg.origin_offset_x_mm == 10.0
    assert cfg.pick_approach_height_mm == 50.0
    assert cfg.place_insert_height_mm == 3.0


def test_empty_snapshot_produces_empty_queue():
    planner = _make_planner()
    empty = Snapshot(detections=[])
    assert planner.cycle(empty, _synth_image()) == []
