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

# _make_planner() below constructs the heuristic (green-channel)
# extractor deliberately, to exercise the planner's cycle end-to-end
# without a torch dependency; its scope-limit warning (spec 1.1) is
# expected here, not a regression to chase.
pytestmark = pytest.mark.filterwarnings(
    "ignore:HeuristicPlacementAreaExtractor assumes:RuntimeWarning")


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


def test_bad_detector_box_and_disagreement_are_counted_separately():
    """_ensure_placement_areas must not let a perception failure
    (BadDetectorBox) collapse into the same observable signal as an
    ordinary low-confidence disagreement - or vice versa."""
    from plan.placement_area import BadDetectorBox, PlacementDisagreement

    class _FlakyExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, image_rgb, cartridge_bbox, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise BadDetectorBox("bad box")
            raise PlacementDisagreement("disagree")

    ws = WorkspaceBounds(-100, 100, -100, 100)
    planner = Planner(PlannerConfig(), _FlakyExtractor(), ws)

    snap = Snapshot(detections=[
        Detection(BBox(0, 0, 50, 50), ClassLabel.CARTRIDGE, 0.9)])
    img = np.zeros((100, 100, 3), np.uint8)

    planner.cycle(snap, img)
    assert planner.bad_detector_box_count == 1
    assert planner.placement_disagreement_count == 0

    # Same cartridge (same bbox -> IoU-matched, same twin entity), still
    # unplanned, so this retries and hits the extractor's second branch.
    planner.cycle(snap, img)
    assert planner.bad_detector_box_count == 1
    assert planner.placement_disagreement_count == 1


def test_label_map_is_passed_by_detection_index_not_snapshot_order():
    """A battery detection ahead of the cartridge in snapshot.detections
    must not shift which mask the extractor receives - cartridge_masks
    is keyed by position in the FULL detections list (Plan D Task 5),
    and Cartridge.detection_index has to agree."""
    class _SpyExtractor:
        def __init__(self):
            self.received = "not called"

        def extract(self, image_rgb, cartridge_bbox, **kwargs):
            self.received = kwargs.get("label_map")
            raise RuntimeError("stop after capturing kwargs")

    ws = WorkspaceBounds(-100, 100, -100, 100)
    spy = _SpyExtractor()
    planner = Planner(PlannerConfig(), spy, ws)

    the_mask = np.full((5, 5), 7, np.int8)
    snap = Snapshot(detections=[
        Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.9),
        Detection(BBox(20, 20, 60, 60), ClassLabel.CARTRIDGE, 0.9),
    ])
    snap.cartridge_masks[1] = the_mask

    planner.cycle(snap, np.zeros((100, 100, 3), np.uint8))
    assert spy.received is the_mask


def test_heuristic_extractor_ignores_cartridge_masks_without_crashing():
    """Defect from the final whole-branch review: if cartridge_masks is
    ever populated while the HEURISTIC extractor is selected, an
    unconditional label_map kwarg raised TypeError (the heuristic's
    extract() declares no **kwargs), which _ensure_placement_areas'
    blanket except swallowed silently - every cartridge went permanently
    unplanned, uncounted, with no signal anywhere. Guarded now by
    Planner._accepts_label_map, checked once against the extractor's own
    signature. This proves the heuristic path still produces a queue
    when cartridge_masks is non-empty, instead of silently planning
    nothing forever."""
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5), (10, 5)])
    # A populated cartridge_masks dict, as a segmenter would leave behind
    # even though this planner's extractor is the heuristic one.
    snap.cartridge_masks[0] = np.zeros((10, 10), np.int8)

    queue = planner.cycle(snap, _synth_image())
    assert len(queue) > 0, (
        "heuristic extractor produced no queue - the label_map kwarg "
        "probably leaked through and silently broke extraction")


def test_segmentation_extract_arithmetic_stays_under_the_o3_budget():
    """FDR O3 budgets queue rebuild at <= 8 ms per cartridge, and queue
    rebuild is extract() (arbitration + rasterisation) PLUS FFDH packing
    against the resulting - denser, mask-derived - grid. This test times
    only extract(): the mask-arithmetic half Planning does once the
    segmenter (which runs in Recognition) has already produced a label
    map. It does not exercise FFDH, so it cannot by itself certify the
    full queue-rebuild budget - see README.md's "Two placement-area
    extractors" section, which states this scope plainly (arithmetic-
    only), and FDR 10.4 for the full-cycle latency evidence. Named for
    what it measures after the final whole-branch review found
    the old name ("...stays_under_the_o3_budget...") overclaimed FFDH
    coverage it never exercised."""
    import time

    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from plan.placement_area import SegmentationPlacementAreaExtractor

    label = np.zeros((288, 131), np.int8)
    label[5:283, 5:126] = CH_CARTRIDGE
    label[12:276, 12:119] = CH_BAY

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0, tau=0.0)
    img = np.zeros((720, 1280, 3), np.uint8)
    from common.types import BBox
    box = BBox(100, 100, 231, 388)

    ex.extract(img, box, label_map=label)          # warm caches
    t0 = time.perf_counter()
    for _ in range(20):
        ex.extract(img, box, label_map=label)
    per_call_ms = (time.perf_counter() - t0) / 20 * 1000

    assert per_call_ms < 8.0, (
        f"{per_call_ms:.1f} ms per cartridge breaks the O3 budget")
