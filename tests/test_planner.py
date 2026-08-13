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
from plan.scene import (
    CellState,
    OutOfWorkspace,
    PlacementCollision,
    WorkspaceBounds,
)

# _make_planner() below constructs the heuristic (green-channel)
# extractor deliberately, to exercise the planner's cycle end-to-end
# without a torch dependency; its scope-limit warning (spec 1.1) is
# expected here, not a regression to chase.
pytestmark = pytest.mark.filterwarnings(
    "ignore:HeuristicPlacementAreaExtractor assumes:RuntimeWarning")


def _synth_image(H=600, W=800, dx=0) -> np.ndarray:
    """A large green cartridge with a central PCB — big enough that the
    FFDH step has room to pack several 18.5×65 mm batteries.

    At mm_per_px=0.38 the cartridge interior (roughly 700×500 px) maps
    to ~266×190 mm, which fits ~20 18650 cells.

    ``dx`` translates the whole cartridge, PCB included, by that many
    pixels in x. Nothing about its shape changes, so a twin that
    re-measures a moved cartridge correctly must produce place targets
    displaced by exactly ``dx * mm_per_px``."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[50:550, 50 + dx:750 + dx, 0] = 40
    img[50:550, 50 + dx:750 + dx, 1] = 200
    img[50:550, 50 + dx:750 + dx, 2] = 40
    # Small central PCB
    img[270:330, 370 + dx:430 + dx] = 20
    return img


def _make_planner(workspace=None):
    cfg = PlannerConfig(
        battery_width_mm=18.5, battery_length_mm=65.0,
        mm_per_px=0.38,
    )
    ext = PlacementAreaExtractor(safety_margin_px=3,
                                 mm_per_cell=1.5, mm_per_px=0.38)
    # +/-350 mm covers the whole 800x600 px fixture at 0.38 mm/px
    # (304 x 228 mm), so the envelope is not what any of these tests is
    # measuring unless it says so. The envelope itself is exercised by
    # the workspace tests at the bottom of this file.
    ws = workspace or WorkspaceBounds(-350, 350, -350, 350)
    return Planner(cfg, ext, ws)


def _overlap_area_mm2(a, b) -> float:
    """Intersection area of two footprints, in mm^2.

    Deliberately re-derived here from the raw floats rather than calling
    ``Reservation.overlaps_mm``: a collision test that asks the code
    under test whether it collided proves nothing.
    """
    dx = min(a.x_mm + a.w_mm, b.x_mm + b.w_mm) - max(a.x_mm, b.x_mm)
    dy = min(a.y_mm + a.h_mm, b.y_mm + b.h_mm) - max(a.y_mm, b.y_mm)
    return max(0.0, dx) * max(0.0, dy)


def _bare_cartridge(cid=0, rows=80, cols=80):
    """A cartridge with a grid AND the rectangle that grid describes.

    The rectangle is not decoration: ``Reservation`` carries its
    footprint in WORKSPACE millimetres as well as strip millimetres, and
    the workspace pair is what the cross-cartridge interlock compares
    (one physical cartridge detected twice used to produce eight pairs of
    overlapping place targets with the per-cartridge interlock unable to
    see across them — audit H, finding 3). A cartridge holding an
    occupancy grid but no rectangle is not a state the planner ever
    produces, so ``_reservation_for`` refuses it rather than falling back
    to strip coordinates, which would silently disable that check.
    """
    from plan.scene import Cartridge, OccupancyGrid

    return Cartridge(
        id=cid, bbox=BBox(0, 0, 100, 100), confidence=0.9,
        placeable_rectangle=BBox(0, 0, 100, 100), mm_per_px=0.38,
        occupancy=OccupancyGrid(rows=rows, cols=cols, resolution_mm=1.5),
    )


def _snapshot_with_cart_and_batteries(batt_centres, dx=0, cartridges=1,
                                      extra_boxes=()):
    """The fixture snapshot. ``dx`` moves the cartridge box with
    ``_synth_image(dx=...)``; ``cartridges=0`` omits it entirely, which
    is the one-frame dropout; ``extra_boxes`` adds further cartridge
    detections over the same object."""
    dets = []
    if cartridges:
        dets.append(Detection(BBox(45 + dx, 45, 755 + dx, 555),
                              ClassLabel.CARTRIDGE, 0.95))
    for box in extra_boxes:
        dets.append(Detection(box, ClassLabel.CARTRIDGE, 0.9))
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
        assert p.pick.z_mm == planner.cfg.pick_grasp_height_mm
        assert p.place.z_mm == planner.cfg.place_insert_height_mm


def test_pick_z_is_a_grasp_height_the_controller_can_actually_grasp_at():
    """CORRECTED contract. `pick.z_mm` used to carry
    `pick_approach_height_mm` (60 mm), and the assertion above passed
    just as happily because it compared the pose against the same field.

    `krl_prog/routines.src` takes the commanded pick pose as the GRASP
    point and derives its own approach from it (`approach_pos.Z =
    pick_pos.Z + 60`, then `LIN pick_pos`). An approach height on the
    wire therefore made the controller descend to 60 mm - clear of an
    18.5 mm cell lying on the table - close the gripper on air, read
    `$IN[10]` false and return PICK_FAILED on every cycle.

    Asserted against `mock_kuka_server`'s own band constant rather than
    a number copied into this file, so the two cannot drift apart: the
    simulator warns above that band precisely because it models no parts
    and cannot fail the grasp itself.
    """
    from execution.mock_kuka_server import _GRASP_BAND_MM, Z_MIN_MM

    planner = _make_planner()
    queue = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5), (10, 5), (15, 5)]),
        _synth_image())
    assert queue, "fixture must produce poses for this to assert anything"
    for p in queue:
        assert Z_MIN_MM < p.pick.z_mm <= _GRASP_BAND_MM, (
            f"pick.z_mm = {p.pick.z_mm} is outside the grasp band "
            f"(0, {_GRASP_BAND_MM}] the controller grasps in; "
            "mock_kuka_server would warn and a real KRC would return "
            "PICK_FAILED")


def test_planner_config_refuses_the_retired_approach_height_key():
    """A `motion.approach_height_mm` that silently did nothing would be
    the same trap `configs/execution.yaml`'s five deleted motion keys
    were: parsed, plausible, and read by nothing."""
    with pytest.raises(ValueError, match="approach_height_mm"):
        PlannerConfig.from_dict({
            "camera": {"mm_per_px_x": 0.5},
            "motion": {"approach_height_mm": 60.0},
        })


def test_cycle_marks_the_whole_footprint_of_every_queued_cell():
    """CORRECTED from `test_cycle_marks_cells_planned`, which asserted
    `planned_count() == len(queue)` — one 1.5 mm grid cell per 18.5 x 65 mm
    battery. That is the defect (audit E finding 1), not the contract:
    `_pack_cartridge` re-packs from this grid every frame, so 571 of the
    572 cells a battery covers read FREE and the next cycle could legally
    seat a cell 1.5 mm from one already in the tray.

    What it asserts instead: one reservation per queued pose, the marked
    region is exactly the union of their cell blocks, and each block is
    the battery's real footprint rounded outward.
    """
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5), (10, 10)])
    queue = planner.cycle(snap, _synth_image())
    ctg = planner.env.cartridge(next(iter(planner.env.cartridges)))

    assert len(ctg.reservations) == len(queue) >= 2

    # 18.5 x 65 mm on a 1.5 mm grid is 13 x 44 cells whichever way round
    # it is placed. The old assertion allowed 1.
    for res in ctg.reservations.values():
        assert res.cell_count == 13 * 44, (
            f"{res.cell_count} cells for a {res.w_mm} x {res.h_mm} mm "
            "footprint at 1.5 mm — the block does not cover the battery")

    # Exactly the union of the blocks: no cell marked that no battery
    # covers, and none left FREE that one does. Adjacent batteries share
    # a boundary column (cells round outward), so this is a union, not a
    # sum - a sum would be the wrong assertion and would drift.
    expected = np.zeros((ctg.occupancy.rows, ctg.occupancy.cols), bool)
    for res in ctg.reservations.values():
        expected[res.row0:res.row1, res.col0:res.col1] = True
    assert (ctg.occupancy.mask_of(CellState.PLANNED) == expected).all()
    assert ctg.occupancy.planned_count() == int(expected.sum())
    assert ctg.occupancy.planned_count() >= len(queue) * 12 * 43


def test_a_later_cycle_never_plans_a_cell_into_one_already_placed():
    """THE regression test for audit E finding 1.

    Two cycles, one cell placed in the first, and no cell in the second
    overlapping it. Measured in millimetres against the real footprint,
    because the grid is the thing under suspicion: asking the occupancy
    grid whether the cells collide would just re-ask the code that got
    it wrong.

    Before the fix the second cycle's first pose sat 1.5 mm from the
    placed cell in x and 0.0 mm in y — about 17 mm of an 18.5 mm-wide
    battery driven into one already in the tray, with the queue, the
    `placed` counter and the robot's SUCCESS all reading normal.
    """
    planner = _make_planner()
    img = _synth_image()

    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    assert len(q1) == 1
    first = q1[0]
    planner.confirm_placement(
        first.cartridge_id, first.grid_row, first.grid_col, True)

    q2 = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5), (400, 300), (700, 500)]),
        img)
    assert q2, "the second cycle found no room in a mostly empty cartridge"

    # (a) From the poses alone, in workspace millimetres. An 18.5 mm
    # square centred on a place target lies inside that battery's real
    # footprint at EITHER rotation, so if two such squares intersect the
    # two batteries intersect. This is the assertion that fails on the
    # old planner (dx = 1.5, dy = 0.0).
    for pose in q2:
        dx = abs(pose.place.x_mm - first.place.x_mm)
        dy = abs(pose.place.y_mm - first.place.y_mm)
        assert max(dx, dy) >= 18.5, (
            f"pose at ({pose.place.x_mm:.1f}, {pose.place.y_mm:.1f}) mm is "
            f"{dx:.1f} mm / {dy:.1f} mm from the cell placed last cycle at "
            f"({first.place.x_mm:.1f}, {first.place.y_mm:.1f}) — two 18650s "
            "cannot be that close without occupying the same space")

    # (b) Exactly, against the footprints the planner reserved, using an
    # overlap area computed here from the raw millimetres.
    ctg = planner.env.cartridge(first.cartridge_id)
    placed = [r for r in ctg.reservations.values()
              if r.state is CellState.PLACED]
    assert len(placed) == 1, "the confirmed cell should be the only PLACED one"
    assert placed[0].w_mm * placed[0].h_mm == pytest.approx(18.5 * 65.0)
    for res in ctg.reservations.values():
        if res is placed[0]:
            continue
        assert _overlap_area_mm2(res, placed[0]) == 0.0, (
            f"{_overlap_area_mm2(res, placed[0]):.1f} mm^2 of overlap with "
            "the battery already in the tray")


def test_a_rotated_placement_reserves_the_rotated_block():
    """The block follows the orientation the packer CHOSE.

    `pack_best_effort` rotates freely (both orientations appear in this
    fixture's 30-item pack), and `PackedItem.width`/`.height` already
    report the placed orientation. Reserving the nominal 13 x 44 for a
    90-degree placement would leave 31 cells of its long axis unmarked -
    the same defect, one costume down.
    """
    from common.packing import Item, PackedItem
    from plan.scene import Cartridge, OccupancyGrid

    planner = _make_planner()
    ctg = _bare_cartridge()
    battery = Item(id=0, width=18.5, height=65.0)

    upright = planner._reservation_for(
        ctg, PackedItem(item=battery, x=0.0, y=0.0, rotated=False))
    assert (upright.row1 - upright.row0,
            upright.col1 - upright.col0) == (44, 13)

    turned = planner._reservation_for(
        ctg, PackedItem(item=battery, x=0.0, y=0.0, rotated=True))
    assert (turned.row1 - turned.row0, turned.col1 - turned.col0) == (13, 44)
    assert (turned.w_mm, turned.h_mm) == (65.0, 18.5)


def test_unexecuted_reservations_are_released_when_the_queue_is_rebuilt():
    """A queue is rebuilt from scratch every cycle, so the reservations
    of the queue it replaces have to go with it.

    `main` executes one pose per cycle. If the other N-1 stayed PLANNED
    the cartridge would fill with batteries nobody picked within a few
    frames, the packer would place nothing, and the run would report
    "queue empty, job done" over a nearly empty tray. The count is
    exposed rather than absorbed.
    """
    planner = _make_planner()
    img = _synth_image()

    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5), (10, 10),
                                                          (15, 15)]), img)
    assert len(q1) == 3
    assert planner.released_reservation_count == 0
    planner.confirm_placement(q1[0].cartridge_id, q1[0].grid_row,
                              q1[0].grid_col, True)

    ctg = planner.env.cartridge(q1[0].cartridge_id)
    placed_cells = ctg.occupancy.placed_count()
    assert placed_cells == 13 * 44

    planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    assert planner.released_reservation_count == 2, (
        "the two queued-but-never-executed reservations must be released")
    # The executed one is physical and survives untouched.
    assert ctg.occupancy.placed_count() == placed_cells
    assert [r.state for r in ctg.reservations.values()].count(
        CellState.PLACED) == 1


def test_queue_never_exceeds_batteries():
    planner = _make_planner()
    # Only 2 batteries; queue must never grow past that
    snap = _snapshot_with_cart_and_batteries([(5, 5), (15, 5)])
    queue = planner.cycle(snap, _synth_image())
    assert len(queue) <= 2


def test_confirm_placement_success_marks_placed():
    """The WHOLE reserved block flips, not the anchor cell.

    Strengthened alongside the footprint fix: the anchor-cell-only
    assertion this replaced passed just as happily when the other 571
    cells of a placed battery were left reading PLANNED for ever.
    """
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
    res = ctg.reservations[(pose.grid_row, pose.grid_col)]
    block = ctg.occupancy.grid[res.row0:res.row1, res.col0:res.col1]
    assert (block == CellState.PLACED.value).all()
    assert ctg.occupancy.planned_count() == 0


def test_confirm_placement_failure_reverts_to_free():
    """A failed place frees the whole footprint, so the next cycle can
    retry it - not one cell of it."""
    planner = _make_planner()
    snap = _snapshot_with_cart_and_batteries([(5, 5)])
    queue = planner.cycle(snap, _synth_image())
    pose = queue[0]
    ctg = planner.env.cartridge(pose.cartridge_id)
    res = ctg.reservations[(pose.grid_row, pose.grid_col)]
    planner.confirm_placement(pose.cartridge_id,
                              pose.grid_row, pose.grid_col, False)
    assert ctg.occupancy.get(pose.grid_row,
                             pose.grid_col) == CellState.FREE
    block = ctg.occupancy.grid[res.row0:res.row1, res.col0:res.col1]
    assert (block == CellState.FREE.value).all()
    assert (pose.grid_row, pose.grid_col) not in ctg.reservations


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


def test_two_fresh_planners_on_one_snapshot_produce_the_same_queue():
    """O5's other half: fixed input -> fixed output, observed.

    ``test_row_major_ordering`` above asserts the *ordering* half of O5.
    This asserts the half §3 actually states, by running the whole cycle
    twice and comparing. ``PickPlacePose`` and ``WorkspacePoint`` are
    frozen dataclasses, so ``==`` on the two lists is a field-by-field
    comparison; the ``float.hex()`` pass after it is stricter still, and
    is there because ``==`` on floats that agree to the last ulp is the
    claim being made, not agreement to a printed precision.

    TWO FRESH PLANNERS, not one planner cycled twice. That is not
    fussiness. ``Planner`` owns a persistent ``Scene`` tracker, so a
    second ``cycle`` on a live instance is a different input by design,
    and the one field that then differs is ``battery_detection_id`` - a
    monotonic per-instance counter that returns 0..7 on the first cycle
    and 8..15 on the second while every geometric field stays
    bit-identical. Writing this test the obvious way would have failed
    and would have looked like a determinism bug.
    """
    snap = _snapshot_with_cart_and_batteries(
        [(5, 5), (10, 5), (15, 5), (5, 30), (10, 30)])

    q1 = _make_planner().cycle(snap, _synth_image())
    q2 = _make_planner().cycle(snap, _synth_image())

    assert len(q1) > 1, "a degenerate empty queue would pass vacuously"
    assert q1 == q2

    def _hexed(queue):
        return [
            (p.cartridge_id, p.grid_row, p.grid_col, p.battery_detection_id,
             p.pick.x_mm.hex(), p.pick.y_mm.hex(), p.pick.z_mm.hex(),
             p.place.x_mm.hex(), p.place.y_mm.hex(), p.place.z_mm.hex())
            for p in queue
        ]

    assert _hexed(q1) == _hexed(q2)


def test_planner_config_from_dict():
    cfg = PlannerConfig.from_dict({
        "battery": {"diameter_mm": 21.0, "length_mm": 70.0},
        "camera": {"mm_per_px_x": 0.5, "origin_offset_x_mm": 10.0,
                   "origin_offset_y_mm": -5.0},
        "motion": {"grasp_height_mm": 6.0, "insert_height_mm": 3.0},
    })
    assert cfg.battery_width_mm == 21.0
    assert cfg.battery_length_mm == 70.0
    assert cfg.mm_per_px == 0.5
    assert cfg.origin_offset_x_mm == 10.0
    assert cfg.pick_grasp_height_mm == 6.0
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
    # An explicit fallback scale: PlannerConfig no longer defaults
    # mm_per_px, so a cycle with an uncalibrated snapshot and no
    # configured fallback raises UnknownScale before it reaches the
    # extractor at all.
    planner = Planner(PlannerConfig(mm_per_px=0.38), _FlakyExtractor(), ws)

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
    planner = Planner(PlannerConfig(mm_per_px=0.38), spy, ws)

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
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0)
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


# ------------------------------------------------- per-frame mm_per_px ----
#
# `mm_per_px` was a config constant while the renderer varied the framing
# per frame. Over recog/dataset3d_seg the true ground sample distance runs
# 0.490-1.045 mm/px against the configured 0.625: the planner under-read
# 24 of 30 cartridges by 27 % at the median (losing 9 cells) and over-read
# the other 6, reserving a footprint SMALLER than the cell it was about to
# place. Three of 17 placements landed on ground-truth non-floor material,
# worst 21.2 %. See docs/superpowers/specs/2026-08-11-scale-calibration.md.
#
# These tests exist so the constant cannot come back.


def _scaled_planner(config_fallback):
    """A planner whose ONLY possible scale is the one the frame supplies.

    `config_fallback=None` is load-bearing: if any code path reached for
    a configured constant instead of the snapshot's own calibration, it
    would find nothing and raise, so these tests cannot pass by accident
    on a planner that ignores the frame.
    """
    cfg = PlannerConfig(battery_width_mm=18.5, battery_length_mm=65.0,
                        mm_per_px=config_fallback)
    ext = PlacementAreaExtractor(safety_margin_px=3, mm_per_cell=1.5)
    # +/-500 mm, not the +/-350 this used to carry. The workspace
    # envelope is enforced now (it was inert before), and these tests
    # deliberately plan the same 800x600 px fixture at up to 0.50 mm/px,
    # which puts the far edge of the cartridge at ~370 mm. That is a
    # property of the fixture's framing, not of the scale handling these
    # tests measure, so the envelope is widened to stay out of the way -
    # the envelope's own behaviour is pinned by the workspace tests
    # below.
    return Planner(cfg, ext, WorkspaceBounds(-500, 500, -500, 500))


def test_planner_measures_each_frame_at_that_frames_own_scale():
    """THE regression test for the calibration defect.

    The same pixels, planned twice at two different frame calibrations,
    must give two different answers in millimetres. A planner that read a
    constant would give the same answer both times - which is exactly what
    it did, on a corpus where no two frames share a scale.
    """
    img = _synth_image()
    # Enough loose batteries that the CARTRIDGE is what limits the queue.
    # The queue is capped by available batteries (PPR 5.3), so too few
    # would make both frames report the same number for a reason that has
    # nothing to do with scale.
    coarse = _snapshot_with_cart_and_batteries([(80, 80)] * 60)
    coarse.mm_per_px = 0.50
    fine = _snapshot_with_cart_and_batteries([(80, 80)] * 60)
    fine.mm_per_px = 0.25

    q_coarse = _scaled_planner(None).cycle(coarse, img)
    q_fine = _scaled_planner(None).cycle(fine, img)

    assert q_coarse, "the coarse frame should plan something"
    # Halving mm_per_px halves how many millimetres those same pixels
    # span, so the SAME cartridge holds fewer 18.5 x 65 mm cells.
    assert len(q_fine) < len(q_coarse), (
        f"{len(q_fine)} at 0.25 mm/px vs {len(q_coarse)} at 0.50 - the "
        "planner is not using the frame's scale")
    # And the workspace millimetres differ, not merely the count.
    assert q_fine[0].place.x_mm != q_coarse[0].place.x_mm


def test_the_frames_scale_beats_a_configured_fallback():
    """A fallback exists for frames that carry no calibration. It must not
    override one that does - that ordering is the whole fix."""
    img = _synth_image()
    snap = _snapshot_with_cart_and_batteries([(80, 80)] * 12)
    snap.mm_per_px = 0.25

    from_frame = _scaled_planner(0.50).cycle(snap, img)
    from_fallback = _scaled_planner(0.25).cycle(
        _snapshot_with_cart_and_batteries([(80, 80)] * 12), img)

    assert len(from_frame) == len(from_fallback)
    assert from_frame[0].place.x_mm == pytest.approx(
        from_fallback[0].place.x_mm)


def test_an_uncalibrated_frame_with_no_fallback_raises():
    """Silent degradation is this project's characteristic failure. An
    unknown scale is a configuration error and reads as one, rather than
    quietly producing millimetres nobody can trace."""
    from plan.placement_area import UnknownScale

    snap = _snapshot_with_cart_and_batteries([(80, 80)])
    assert snap.mm_per_px is None
    with pytest.raises(UnknownScale, match="mm_per_px"):
        _scaled_planner(None).cycle(snap, _synth_image())


def test_a_configured_fallback_covers_a_frame_that_carries_no_scale():
    """The real-camera case: one lens, one working distance, one GSD,
    measured once and configured. It must still work."""
    snap = _snapshot_with_cart_and_batteries([(80, 80)] * 4)
    planner = _scaled_planner(0.38)
    assert planner.cycle(snap, _synth_image())
    ctg = next(iter(planner.env.cartridges.values()))
    assert ctg.mm_per_px == 0.38


def test_the_cartridge_records_the_scale_its_rectangle_was_measured_at():
    """`placeable_rectangle` is in pixels and the twin outlives the frame,
    so the scale has to travel with it."""
    snap = _snapshot_with_cart_and_batteries([(80, 80)] * 4)
    snap.mm_per_px = 0.42
    planner = _scaled_planner(None)
    planner.cycle(snap, _synth_image())

    ctg = next(iter(planner.env.cartridges.values()))
    assert ctg.placeable_rectangle is not None
    assert ctg.mm_per_px == 0.42


def test_a_cached_area_is_dropped_when_the_frames_scale_changes():
    """Cartridges persist across frames by IoU and carry their occupancy
    with them. If the scale moves, those pixels no longer mean the
    millimetres they were measured to mean - packing them against the new
    number would re-introduce the same defect by a different route."""
    img = _synth_image()
    planner = _scaled_planner(None)

    first = _snapshot_with_cart_and_batteries([(80, 80)] * 12)
    first.mm_per_px = 0.50
    planner.cycle(first, img)
    ctg_id = next(iter(planner.env.cartridges))
    assert planner.env.cartridges[ctg_id].mm_per_px == 0.50
    assert planner.rescaled_area_drop_count == 0

    # Same cartridge box -> IoU-matched to the same twin entity, but the
    # camera has been re-calibrated under it.
    second = _snapshot_with_cart_and_batteries([(80, 80)] * 12)
    second.mm_per_px = 0.25
    planner.cycle(second, img)

    assert planner.rescaled_area_drop_count == 1
    assert planner.env.cartridges[ctg_id].mm_per_px == 0.25


def test_an_unchanged_scale_does_not_drop_anything():
    """The fixed-mount case must not thrash: same scale, same area,
    re-extraction never happens and the counter stays 0."""
    img = _synth_image()
    planner = _scaled_planner(None)
    for _ in range(3):
        snap = _snapshot_with_cart_and_batteries([(80, 80)] * 12)
        snap.mm_per_px = 0.38
        planner.cycle(snap, img)
    assert planner.rescaled_area_drop_count == 0


# --------------------------------------------- the workspace envelope ----
#
# `WorkspaceBounds` was parsed from `planning.yaml`, stored on the
# EnvironmentModel and compared against nothing (audit E finding 5): a
# declared robot-safety envelope that enforced nothing, which is worse
# than none because it reads like an interlock. The proof it was inert
# was that a deliberately tiny bound changed no test's outcome.
#
# It never clamps. A place target is where a cartridge slot physically
# is and a pick point is where a battery physically lies; moving either
# onto the envelope's edge does not make it reachable, it makes it wrong
# - and a slightly-wrong motion that reports SUCCESS is exactly the
# silent class this audit is closing.
#
# CORRECTED 2026-08-12. The two tests below used to assert that an
# out-of-envelope CANDIDATE raises, and they were wrong about which
# condition they were describing. A camera legitimately images more
# table than the arm can reach: `configs/demo_seg.yaml`'s frames span up
# to 1338 x 752 mm against a 700 mm envelope, so no origin offset makes
# them fit and an ordinary frame was aborting the whole run
# (docs/superpowers/specs/2026-08-12-fix-demo-workspace.md). A candidate
# out of reach is a SCENE CONDITION and is now skipped and counted; a
# POSE out of reach is a planning bug and still raises, which
# `test_the_envelope_guard_fires_when_the_reachability_filter_is_bypassed`
# is what proves.


def test_a_place_target_outside_the_workspace_is_skipped_and_counted():
    """Not raised - and not silently dropped either.

    A five-millimetre envelope reaches no slot in this cartridge, so the
    queue is empty. The counter is what makes that distinguishable from
    "the cartridge is full", which is the whole point: they are the same
    empty queue and they want opposite responses.
    """
    planner = _make_planner(workspace=WorkspaceBounds(-5, 5, -5, 5))
    snap = _snapshot_with_cart_and_batteries([(5, 5)])

    queue = planner.cycle(snap, _synth_image())

    assert queue == []
    assert planner.unreachable_place_target_count > 0
    # The packer found room and not one slot of it was reachable.
    assert planner.unreachable_cartridge_count == 1


def test_a_pick_point_outside_the_workspace_is_skipped_and_counted():
    """A battery the camera can see but the arm cannot reach is not a
    pose to attempt at the nearest reachable point, and not a reason to
    abort a frame either. It is one cell the robot declines to serve."""
    planner = _make_planner()
    # 2000 px * 0.38 mm/px = 760 mm, well past the +/-350 envelope, while
    # the place targets stay inside it.
    snap = _snapshot_with_cart_and_batteries([(2000, 2000)])

    queue = planner.cycle(snap, _synth_image())

    assert queue == []
    assert planner.unreachable_battery_count == 1
    # The CARTRIDGE was reachable; only the battery was not. Collapsing
    # the two would be the same conflation this change is undoing.
    assert planner.unreachable_cartridge_count == 0


def test_a_camera_that_sees_further_than_the_arm_reaches_still_plans():
    """The demo's actual condition, and the regression test for it.

    Part of the field of view is served, the rest is declined, the run
    continues, and every pose that comes out is inside the envelope.
    """
    planner = _make_planner(workspace=WorkspaceBounds(-150, 150, -150, 150))
    # 20 cells well inside the reachable region, 4 well outside it.
    reachable = [(20 + 10 * i, 20) for i in range(20)]
    unreachable = [(2000, 2000)] * 4
    snap = _snapshot_with_cart_and_batteries(reachable + unreachable)

    queue = planner.cycle(snap, _synth_image())

    assert queue, "the reachable part of the cartridge must still be planned"
    ws = planner.env.workspace
    for p in queue:
        assert ws.contains(p.place.x_mm, p.place.y_mm), p.place
        assert ws.contains(p.pick.x_mm, p.pick.y_mm), p.pick
    # Both kinds of skip are visible, and the cartridge is NOT declared
    # unreachable - part of it was served.
    assert planner.unreachable_battery_count == 4
    assert planner.unreachable_place_target_count > 0
    assert planner.unreachable_cartridge_count == 0


def test_the_envelope_guard_fires_when_the_reachability_filter_is_bypassed():
    """THE proof that the invariant is still load-bearing.

    With the filter skipping unreachable candidates, `_build_pose`'s two
    `WorkspaceBounds.require` calls can no longer fire in normal
    operation - which is exactly how a guard rots into decoration. So the
    filter is disabled here (`Planner._reaches` exists as a named seam
    for precisely this) and the guard must still refuse to emit the pose.

    If this test goes green after someone deletes those `require` calls
    because "nothing reaches them", the deletion has been caught.
    """
    planner = _make_planner(workspace=WorkspaceBounds(-150, 150, -150, 150))
    # Everything now looks reachable to the filter. Nothing has changed
    # about the envelope itself.
    planner._reaches = lambda x_mm, y_mm: True

    snap = _snapshot_with_cart_and_batteries([(20 + 10 * i, 20)
                                              for i in range(20)])
    with pytest.raises(OutOfWorkspace, match="place target"):
        planner.cycle(snap, _synth_image())


def test_an_unreachable_slot_does_not_consume_a_battery():
    """Skipping must happen BEFORE `_nearest_battery` takes one.

    Row-major fill visits the low-x slots first, and this envelope makes
    exactly those unreachable. With one battery on the table, a filter
    that ran after the assignment would spend it on the first skipped
    slot and return an empty queue; the battery has to survive to the
    first slot the arm can actually serve.
    """
    planner = _make_planner(workspace=WorkspaceBounds(100, 350, -350, 350))
    # 400 px * 0.38 = 152 mm, inside x [100, 350].
    snap = _snapshot_with_cart_and_batteries([(400, 100)])

    queue = planner.cycle(snap, _synth_image())

    assert planner.unreachable_place_target_count > 0, (
        "fixture must actually skip some slots for this to assert anything")
    assert planner.unreachable_battery_count == 0
    assert len(queue) == 1
    assert queue[0].place.x_mm >= 100.0


def test_poses_inside_the_envelope_are_emitted_unchanged():
    """The guard must not be a clamp wearing a raise's clothes: an
    in-envelope pose comes out at the millimetres that were computed."""
    tight = _make_planner(workspace=WorkspaceBounds(-350, 350, -350, 350))
    loose = _make_planner(workspace=WorkspaceBounds(-9999, 9999,
                                                    -9999, 9999))
    img = _synth_image()
    a = tight.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    b = loose.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    assert a and len(a) == len(b)
    assert a[0].place.x_mm == b[0].place.x_mm
    assert a[0].place.y_mm == b[0].place.y_mm
    assert a[0].pick.x_mm == b[0].pick.x_mm


def test_an_empty_workspace_envelope_is_rejected_at_construction():
    """An inverted bound would reject every pose, which reads as a broken
    planner rather than as the typo it is."""
    with pytest.raises(ValueError, match="empty workspace envelope"):
        WorkspaceBounds(350, -350, -350, 350)
    with pytest.raises(ValueError, match="empty workspace envelope"):
        WorkspaceBounds(-350, 350, 0, 0)


def test_a_reservation_over_a_forbidden_cell_raises():
    """The cell-resolution half of the interlock. `_overlaps_forbidden`
    means a placement that reaches `reserve` cannot cover a FORBIDDEN
    cell; if one ever does, a battery is going onto the PCB and that has
    to be an exception rather than a marked grid."""
    from common.packing import Item, PackedItem

    planner = _make_planner()
    ctg = _bare_cartridge()
    ctg.occupancy.set(20, 5, CellState.FORBIDDEN)
    res = planner._reservation_for(
        ctg, PackedItem(item=Item(id=0, width=18.5, height=65.0),
                        x=0.0, y=0.0, rotated=False))
    with pytest.raises(PlacementCollision, match="FORBIDDEN"):
        ctg.reserve(res)


def test_reserving_over_an_existing_footprint_raises():
    """The millimetre-resolution half. Two batteries in the same space
    is a packer bug; continuing past it puts an 18650 on an 18650."""
    from common.packing import Item, PackedItem

    planner = _make_planner()
    ctg = _bare_cartridge()
    battery = Item(id=0, width=18.5, height=65.0)
    ctg.reserve(planner._reservation_for(
        ctg, PackedItem(item=battery, x=0.0, y=0.0)))

    # 17 mm in: exactly what the one-cell grid used to permit.
    with pytest.raises(PlacementCollision, match="PLANNED"):
        ctg.reserve(planner._reservation_for(
            ctg, PackedItem(item=battery, x=1.5, y=0.0)))

    # Abutting is not overlapping: sharing a boundary cell is normal
    # between neighbours in one pack and must stay legal.
    ctg.reserve(planner._reservation_for(
        ctg, PackedItem(item=battery, x=18.5, y=0.0)))
    assert len(ctg.reservations) == 2


def test_planner_config_does_not_invent_a_fallback_scale():
    """An absent `camera.mm_per_px_x` means the camera was not
    calibrated, not that it is 0.38. The old default silently supplied
    `planning.yaml`'s placeholder to callers who never chose it."""
    assert PlannerConfig.from_dict({"camera": {}}).mm_per_px is None
    assert PlannerConfig.from_dict(
        {"camera": {"mm_per_px_x": 0.5}}).mm_per_px == 0.5


# ------------------------------- cross-frame identity, end to end -------
#
# The twin is the only memory the system has of which cells in which
# cartridge are FREE / PLANNED / PLACED, and it is rebuilt from a fresh
# set of detections every frame. Audit H executed four ordinary
# perception events against it - a one-frame dropout, a cartridge that
# moves, one cartridge detected twice, and two detections competing for
# one track - and all four produced a full-length, in-envelope,
# clean-reporting queue aimed at space that is physically occupied.
#
# These tests are those four events, at the altitude the arm sees them:
# millimetres of place target, not internal state.


def _assert_no_two_queued_footprints_overlap(planner, queue) -> None:
    """No two queued place targets may share physical space, whichever
    twin entries they came from.

    Computed here from the raw workspace millimetres rather than by
    calling `Reservation.overlaps_workspace_mm`: a collision test that
    asks the code under test whether it collided proves nothing.
    """
    reserved = [
        planner.env.cartridge(p.cartridge_id).reservations[
            (p.grid_row, p.grid_col)]
        for p in queue
    ]
    for i, a in enumerate(reserved):
        for b in reserved[i + 1:]:
            dx = (min(a.wx_mm + a.w_mm, b.wx_mm + b.w_mm)
                  - max(a.wx_mm, b.wx_mm))
            dy = (min(a.wy_mm + a.h_mm, b.wy_mm + b.h_mm)
                  - max(a.wy_mm, b.wy_mm))
            assert max(0.0, dx) * max(0.0, dy) == 0.0, (
                f"queued targets at ({a.wx_mm:.1f}, {a.wy_mm:.1f}) and "
                f"({b.wx_mm:.1f}, {b.wy_mm:.1f}) mm overlap by "
                f"{dx:.1f} x {dy:.1f} mm")


def test_a_one_frame_dropout_does_not_re_plan_a_slot_already_filled():
    """The audit's scenario 1, executed: present / absent / present.

    Measured before the fix: 572 placed cells, one undetected frame, and
    the cartridge returned as a NEW id holding 0 whose first place target
    was byte-identical to the one already executed. The millimetre
    interlock could not fire, because the reservations it compares
    against were deleted with the cartridge. A collision that reports
    success.
    """
    planner = _make_planner()
    img = _synth_image()

    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    assert len(q1) == 1
    first = q1[0]
    planner.confirm_placement(
        first.cartridge_id, first.grid_row, first.grid_col, True)
    ctg = planner.env.cartridge(first.cartridge_id)
    placed_cells = ctg.occupancy.placed_count()
    assert placed_cells == 13 * 44

    # Frame 2: the cartridge is not detected at all. One low-confidence
    # frame, one gripper arm across the box.
    q2 = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5)], cartridges=0), img)
    assert q2 == [], "an unseen cartridge must not be planned into"
    assert first.cartridge_id in planner.env.cartridges, (
        "the twin forgot a tray because of one missing frame")
    assert planner.env.cartridge(
        first.cartridge_id).occupancy.placed_count() == placed_cells

    # Frame 3: the identical box comes back.
    q3 = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5), (400, 300), (700, 500)]),
        img)

    assert list(planner.env.cartridges) == [first.cartridge_id], (
        "the returning cartridge was inserted as a new, empty entity")
    ctg = planner.env.cartridge(first.cartridge_id)
    assert ctg.occupancy.placed_count() == placed_cells
    assert len([r for r in ctg.reservations.values()
                if r.state is CellState.PLACED]) == 1
    assert q3, "the rest of the cartridge must still be plannable"
    # The assertion that failed before the fix: an 18.5 mm square centred
    # on a place target lies inside that battery's real footprint at
    # either rotation, so two such squares intersecting means two
    # batteries intersecting.
    for pose in q3:
        dx = abs(pose.place.x_mm - first.place.x_mm)
        dy = abs(pose.place.y_mm - first.place.y_mm)
        assert max(dx, dy) >= 18.5, (
            f"pose at ({pose.place.x_mm:.1f}, {pose.place.y_mm:.1f}) mm is "
            f"{dx:.1f} / {dy:.1f} mm from the battery placed before the "
            "dropout - the twin came back empty")
    assert planner.env.track_dropout_count == 1
    assert planner.env.tracking_stats()["tracks_reacquired"] == 1
    assert planner.env.tracking_stats()["tracks_expired"] == 0


def test_a_track_that_never_comes_back_expires_and_says_what_it_cost():
    """Memory is bounded. The bound is also the moment the protection
    above stops applying, so it is loud: the cells are counted, not
    quietly absent."""
    planner = _make_planner()
    planner.env.tracking.max_missing_frames = 2
    img = _synth_image()
    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    planner.confirm_placement(q1[0].cartridge_id, q1[0].grid_row,
                              q1[0].grid_col, True)

    for _ in range(3):
        planner.cycle(
            _snapshot_with_cart_and_batteries([(5, 5)], cartridges=0), img)

    assert planner.env.cartridges == {}
    stats = planner.env.tracking_stats()
    assert stats["tracks_expired"] == 1
    assert stats["expired_placed_cells"] == 13 * 44
    assert stats["expired_placed_batteries"] == 1


def test_a_confirmation_for_an_expired_track_is_counted_not_swallowed():
    """`main` counts this as `placed` - a battery really was put down -
    while the twin has no record of where. That disagreement used to be a
    bare `return` (audit H, finding 1)."""
    planner = _make_planner()
    planner.env.tracking.max_missing_frames = 0
    img = _synth_image()
    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5)], cartridges=0), img)
    assert planner.env.cartridges == {}

    planner.confirm_placement(q1[0].cartridge_id, q1[0].grid_row,
                              q1[0].grid_col, True)

    assert planner.untracked_confirmation_count == 1


def test_the_place_target_follows_a_cartridge_that_moves():
    """Audit H finding 2, executed: the cartridge translated 100 px and
    the place target moved 0.00 mm, because `_ensure_placement_areas`
    skips any cartridge that already has a rectangle and the rectangle
    was extracted exactly once, ever. IoU 0.5 tolerates ~90 mm of drift
    on this fixture, so the twin kept commanding the old millimetres all
    the way through it - into the cartridge wall, onto the PCB, or onto
    the table.
    """
    planner = _make_planner()
    dx = 40
    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]),
                       _synth_image())
    q2 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)], dx=dx),
                       _synth_image(dx=dx))

    assert list(planner.env.cartridges) == [q1[0].cartridge_id], (
        "a cartridge that moved is the same cartridge")
    assert planner.env.geometry_refresh_count == 1
    moved_mm = q2[0].place.x_mm - q1[0].place.x_mm
    assert moved_mm == pytest.approx(dx * 0.38, abs=0.5), (
        f"the cartridge moved {dx * 0.38:.1f} mm and its place target "
        f"moved {moved_mm:.2f} mm")
    assert q2[0].place.y_mm == pytest.approx(q1[0].place.y_mm, abs=0.5)


def test_detector_jitter_does_not_re_measure_the_geometry_every_frame():
    """The threshold is a threshold in both directions: a box that
    wobbles by a pixel must not throw its measurement away, or the twin
    re-extracts every cartridge every frame for nothing."""
    planner = _make_planner()
    img = _synth_image()
    planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)
    for _ in range(3):
        planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)], dx=1), img)
    assert planner.env.geometry_refresh_count == 0


def test_a_battery_already_placed_survives_the_cartridge_moving():
    """The other half of the same event, and the reason the twin
    re-measures rather than rejecting the match: a cartridge that slides
    takes its batteries with it. Rejecting the match would hand it a
    fresh empty grid and plan straight into them."""
    planner = _make_planner()
    dx = 40
    q1 = planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]),
                       _synth_image())
    planner.confirm_placement(q1[0].cartridge_id, q1[0].grid_row,
                              q1[0].grid_col, True)
    cid = q1[0].cartridge_id
    placed_cells = planner.env.cartridge(cid).occupancy.placed_count()

    q2 = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5), (400, 300)], dx=dx),
        _synth_image(dx=dx))

    ctg = planner.env.cartridge(cid)
    assert ctg.occupancy.placed_count() == placed_cells, (
        "the placed battery was lost when its cartridge was re-measured")
    assert planner.placed_reservation_reprojected_count == 1
    assert planner.placed_reservation_lost_count == 0
    # And the next queue avoids it AT ITS NEW POSITION.
    placed = [r for r in ctg.reservations.values()
              if r.state is CellState.PLACED][0]
    assert q2, "the moved cartridge must still be plannable"
    for pose in q2:
        res = ctg.reservations[(pose.grid_row, pose.grid_col)]
        assert _overlap_area_mm2(res, placed) == 0.0


def test_one_cartridge_detected_twice_yields_one_entry_and_no_overlap():
    """Audit H finding 3, executed: a duplicate box at IoU 0.479 became a
    second twin entry with its own grid and its own reservations, and
    produced 8 pairs of physically overlapping place targets - roughly
    17.5 x 34.6 mm of overlap on an 18.5 x 65 mm cell - with no
    PlacementCollision raised, because the millimetre interlock is
    per-cartridge and cannot see across two of them."""
    planner = _make_planner()
    base = BBox(45, 45, 755, 555)
    duplicate = BBox(45 + 110, 45 + 110, 755 + 110, 555 + 110)
    assert base.iou(duplicate) < planner.env.tracking.iou_match_threshold
    assert base.iou(duplicate) >= planner.env.tracking.duplicate_iou_threshold
    snap = _snapshot_with_cart_and_batteries(
        [(5 + 10 * i, 5) for i in range(40)], extra_boxes=(duplicate,))

    queue = planner.cycle(snap, _synth_image())

    assert len(planner.env.cartridges) == 1
    assert planner.env.duplicate_detection_count == 1
    assert queue
    _assert_no_two_queued_footprints_overlap(planner, queue)


def test_two_entries_over_one_object_cannot_queue_overlapping_targets():
    """The interlock behind the suppression above, exercised where
    suppression does NOT apply: a box nested inside another scores below
    the duplicate threshold, so the twin legitimately holds two entries -
    and their placements still may not overlap in the workspace frame,
    which is the only frame in which two cartridges are comparable."""
    planner = _make_planner()
    nested = BBox(400, 300, 700, 540)
    assert BBox(45, 45, 755, 555).iou(nested) < \
        planner.env.tracking.duplicate_iou_threshold
    snap = _snapshot_with_cart_and_batteries(
        [(5 + 10 * i, 5) for i in range(40)], extra_boxes=(nested,))

    queue = planner.cycle(snap, _synth_image())

    assert len(planner.env.cartridges) == 2
    assert planner.cross_cartridge_conflict_count > 0, (
        "fixture must actually produce a conflict for this to assert")
    _assert_no_two_queued_footprints_overlap(planner, queue)


def test_the_cross_cartridge_guard_fires_when_its_filter_is_bypassed():
    """THE proof that the invariant is load-bearing, on the same pattern
    as the workspace envelope: disable the filter
    (`_conflicts_with_another_cartridge` exists as a named seam for
    exactly this) and `_build_pose` must still refuse to emit a pose into
    space another twin entry holds."""
    planner = _make_planner()
    planner._conflicts_with_another_cartridge = lambda ctg, res: None
    nested = BBox(400, 300, 700, 540)
    snap = _snapshot_with_cart_and_batteries(
        [(5 + 10 * i, 5) for i in range(40)], extra_boxes=(nested,))

    with pytest.raises(PlacementCollision, match="same physical space"):
        planner.cycle(snap, _synth_image())


def test_an_unseen_cartridge_is_remembered_but_never_planned_into():
    """A track is memory; a detection is evidence. Memory is enough to
    refuse a slot and not enough to command the arm into one - the box is
    not in this frame, and the pixels under it belong to whatever is
    there now."""
    planner = _make_planner()
    img = _synth_image()
    assert planner.cycle(_snapshot_with_cart_and_batteries([(5, 5)]), img)

    queue = planner.cycle(
        _snapshot_with_cart_and_batteries([(5, 5)] * 4, cartridges=0), img)

    assert queue == []
    assert len(planner.env.cartridges) == 1
    assert planner.env.visible_cartridges() == []
    # Its geometry is neither re-measured nor thrown away meanwhile: the
    # frame says nothing about a cartridge it does not contain.
    ctg = next(iter(planner.env.cartridges.values()))
    assert ctg.placeable_rectangle is not None
    assert planner.rescaled_area_drop_count == 0


def test_a_scale_change_while_a_cartridge_is_unseen_keeps_its_batteries():
    """`_drop_areas_measured_at_another_scale` iterated every cartridge,
    including ones this frame did not contain - so a re-calibration
    during a dropout cleared the reservations of a cartridge nobody was
    looking at. It is memory until it is seen again, and it is re-measured
    then."""
    planner = _scaled_planner(None)
    img = _synth_image()
    first = _snapshot_with_cart_and_batteries([(5, 5)])
    first.mm_per_px = 0.50
    q1 = planner.cycle(first, img)
    planner.confirm_placement(q1[0].cartridge_id, q1[0].grid_row,
                              q1[0].grid_col, True)
    cid = q1[0].cartridge_id

    missing = _snapshot_with_cart_and_batteries([(5, 5)], cartridges=0)
    missing.mm_per_px = 0.25
    planner.cycle(missing, img)

    assert planner.rescaled_area_drop_count == 0
    assert len(planner.env.cartridge(cid).placed_reservations()) == 1
    assert planner.rescaled_placed_reservation_drop_count == 0
