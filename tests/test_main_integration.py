"""End-to-end integration test for the full pipeline.

Spawns the mock robot, runs a short ``main.run`` with a synthetic
dataset, and asserts the PPR §5.4 sequential flow runs to completion
with no exceptions and sensible stats.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from main import run as main_run


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    """Materialise a minimal demo.yaml that points at a scratch dataset."""
    root = tmp_path / "auto-pick"
    (root / "configs").mkdir(parents=True)
    (root / "recog" / "dataset" / "images").mkdir(parents=True)
    (root / "recog" / "dataset" / "annotations").mkdir(parents=True)

    # Generate 3 synthetic images so the source yields enough frames
    from recog.synth_dataset import generate_dataset
    generate_dataset(str(root / "recog" / "dataset"), n=3, seed=9,
                     size=(480, 640))

    (root / "configs" / "recognition.yaml").write_text(
        "dataset:\n"
        f"  img_dir: {root}/recog/dataset/images\n"
        "training:\n  checkpoint_dir: /tmp/nothing\n"
    )
    (root / "configs" / "planning.yaml").write_text(
        "battery: {diameter_mm: 18.5, length_mm: 65.0}\n"
        "camera: {mm_per_px_x: 0.38, workspace_bounds_mm: "
        "{x_min: -350, x_max: 350, y_min: -350, y_max: 350}}\n"
        "cartridge: {safety_margin_px: 4}\n"
        "occupancy_grid: {resolution_mm_per_cell: 1.5}\n"
        "motion: {grasp_height_mm: 5, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54611, max_retries: 3}\n"
        "motion: {transport_height_mm: 80, vacuum_level_percent: 80}\n"
        "simulation: {listen_host: 127.0.0.1, listen_port: 54611, "
        "drop_probability: 0.0, simulated_move_time_ms_per_100mm: 5}\n"
    )
    demo = root / "configs" / "demo.yaml"
    demo.write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode:\n  source: synthetic\n  robot: mock\n  max_cycles: 3\n"
        "  log_level: WARNING\n"
    )
    return demo


@pytest.fixture
def one_scene_config(tmp_path: Path) -> Path:
    """`demo_config`, but ONE scene, cycled.

    `_synthetic_source` wraps around, so every cycle sees the same frame
    and the twin's IoU matcher keeps the same cartridge alive across
    them - the fixed-mount-camera case, where a reservation can actually
    outlive the queue that made it.
    """
    root = tmp_path / "auto-pick-1"
    (root / "configs").mkdir(parents=True)
    (root / "recog" / "dataset" / "images").mkdir(parents=True)
    (root / "recog" / "dataset" / "annotations").mkdir(parents=True)

    from recog.synth_dataset import generate_dataset
    generate_dataset(str(root / "recog" / "dataset"), n=1, seed=9,
                     size=(480, 640))

    (root / "configs" / "recognition.yaml").write_text(
        "dataset:\n"
        f"  img_dir: {root}/recog/dataset/images\n"
        "training:\n  checkpoint_dir: /tmp/nothing\n"
    )
    (root / "configs" / "planning.yaml").write_text(
        "battery: {diameter_mm: 18.5, length_mm: 65.0}\n"
        "camera: {mm_per_px_x: 0.38, workspace_bounds_mm: "
        "{x_min: -350, x_max: 350, y_min: -350, y_max: 350}}\n"
        "cartridge: {safety_margin_px: 4}\n"
        "occupancy_grid: {resolution_mm_per_cell: 1.5}\n"
        "motion: {grasp_height_mm: 5, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54617, max_retries: 3}\n"
        "motion: {transport_height_mm: 80, vacuum_level_percent: 80}\n"
        "simulation: {listen_host: 127.0.0.1, listen_port: 54617, "
        "drop_probability: 0.0, simulated_move_time_ms_per_100mm: 5}\n"
    )
    demo = root / "configs" / "demo_one.yaml"
    demo.write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode:\n  source: synthetic\n  robot: mock\n  max_cycles: 3\n"
        "  log_level: WARNING\n"
    )
    return demo


def test_main_run_short_cycle(demo_config: Path):
    stats = main_run(str(demo_config))
    assert stats["cycles"] >= 1
    # In a 0%-drop environment most cycles should place successfully
    assert stats["placed"] + stats["pick_failed"] + stats["place_failed"] == \
        stats["cycles"]


def test_run_surfaces_released_reservations(one_scene_config: Path,
                                            tmp_path: Path):
    """`Planner.released_reservation_count` reaches the run's output.

    The counter guards a failure that is invisible in every other number
    here: without the release, each re-plan leaves the twin's cartridge
    holding reservations for cells nobody picked, the packer runs out of
    room, and the run reports "queue empty, job done" over a tray that is
    still full - with `placed` reading exactly the same as a healthy run.

    Asserted NON-ZERO deliberately. `main` executes one pose per cycle
    and re-plans, so a re-planning run over a queue longer than one MUST
    release; a zero would mean either the release stopped happening or
    the statistic stopped being wired to it, and an
    `== stats["released_reservations"]` self-comparison would pass in
    both cases.

    It needs `one_scene_config`, not `demo_config`, and that is the point
    of the separate fixture: the twin matches cartridges across frames by
    IoU and DROPS any it does not see again
    (`EnvironmentModel.update_from_snapshot`), so on `demo_config`'s three
    unrelated scenes the previous frame's cartridge is discarded whole and
    its stale reservations go with it, uncounted - `released_reservations`
    reads 0 there while nothing is leaking. A repeated scene is the
    fixed-mount-camera case this counter is actually about.
    """
    receipt = tmp_path / "receipts" / "run.txt"
    stats = main_run(str(one_scene_config), receipt_path=str(receipt))

    assert stats["cycles"] >= 2, "fixture must re-plan for this to assert"
    assert stats["queue_poses"] > stats["cycles"], (
        "queues must be longer than one pose for any reservation to be "
        "left over")
    assert stats["released_reservations"] > 0, (
        "a re-planning run released no reservations: either the planner "
        "stopped releasing them or main stopped reporting the count")
    assert (f"reservations released  : {stats['released_reservations']} "
            in receipt.read_text(encoding="utf-8"))


# ------------------------------- the workspace envelope, end to end ----
#
# `planning.camera.workspace_bounds_mm` is not a preference: it is the
# only thing in this system that can refuse to command the arm somewhere
# it cannot go. The mock controller happily accepts MOVE_TO(5000, 5000,
# 5000) on a 706 mm-reach arm (audit F, finding 2) and the 16-byte frame
# carries no envelope, so nothing downstream of the planner will catch a
# pose that gets past it.
#
# A camera legitimately sees past the arm's reach, so the planner SKIPS
# unreachable candidates and counts them rather than aborting the frame
# (docs/superpowers/specs/2026-08-12-fix-demo-workspace.md). These two
# tests are the pair that has to hold together: the skipping must be
# visible, and it must not have opened a route for an out-of-envelope
# coordinate to reach the socket.

TIGHT_ENVELOPE = (80.0, 350.0, 80.0, 350.0)   # x_min, x_max, y_min, y_max


@pytest.fixture
def tight_envelope_config(tmp_path: Path) -> Path:
    """`demo_config` with an envelope narrower than the field of view.

    640 x 480 px at 0.38 mm/px is a 243 x 182 mm frame starting at the
    robot origin (`origin_offset_*` is 0), and the envelope covers only
    the far corner of it, from 80 mm out. So the arm can serve part of
    every frame and not the rest - the real cell's condition, and the one
    `configs/demo_seg.yaml` is in at 1338 mm of field of view against
    700 mm of envelope.

    The bound is a LOWER one, and that is load-bearing rather than
    arbitrary. `main` executes `queue[0]` and only `queue[0]`, and
    row-major fill makes that the pose with the SMALLEST coordinates - so
    an envelope that only cuts off the far edge could never put a
    violation on the wire no matter how broken the filter was, and the
    wire test below would pass while asserting nothing. Cutting off the
    NEAR edge puts the first pose of every unfiltered queue outside the
    envelope, which is what makes that test discriminating: with
    `Planner._reaches` forced True and `WorkspaceBounds.require` stubbed
    out, this fixture puts 4 out-of-envelope motions on the recorder -
    MOVE_TO(25, 48), PICK_AND_PLACE(42, 152), MOVE_TO(31, 112),
    PICK_AND_PLACE(156, 32) - and the test fails as it should.

    Measured on this seed: 1 pose queued, ~23 place targets, ~4 batteries
    and 1 whole cartridge declined.
    """
    root = tmp_path / "auto-pick-tight"
    (root / "configs").mkdir(parents=True)
    (root / "recog" / "dataset" / "images").mkdir(parents=True)
    (root / "recog" / "dataset" / "annotations").mkdir(parents=True)

    from recog.synth_dataset import generate_dataset
    generate_dataset(str(root / "recog" / "dataset"), n=3, seed=9,
                     size=(480, 640))

    (root / "configs" / "recognition.yaml").write_text(
        "dataset:\n"
        f"  img_dir: {root}/recog/dataset/images\n"
        "training:\n  checkpoint_dir: /tmp/nothing\n"
    )
    x0, x1, y0, y1 = TIGHT_ENVELOPE
    (root / "configs" / "planning.yaml").write_text(
        "battery: {diameter_mm: 18.5, length_mm: 65.0}\n"
        "camera: {mm_per_px_x: 0.38, workspace_bounds_mm: "
        f"{{x_min: {x0}, x_max: {x1}, y_min: {y0}, y_max: {y1}}}}}\n"
        "cartridge: {safety_margin_px: 4}\n"
        "occupancy_grid: {resolution_mm_per_cell: 1.5}\n"
        "motion: {grasp_height_mm: 5, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54623, max_retries: 3}\n"
        "motion: {transport_height_mm: 80, vacuum_level_percent: 80}\n"
        "simulation: {listen_host: 127.0.0.1, listen_port: 54623, "
        "drop_probability: 0.0, simulated_move_time_ms_per_100mm: 5}\n"
    )
    demo = root / "configs" / "demo_tight.yaml"
    demo.write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode:\n  source: synthetic\n  robot: mock\n  max_cycles: 3\n"
        "  log_level: WARNING\n  stop_on_empty_queue: false\n"
    )
    return demo


def test_no_coordinate_outside_the_envelope_reaches_the_wire(
        tight_envelope_config: Path, monkeypatch):
    """THE safety property, asserted on the BYTES, not on the queue.

    Every command is intercepted at `KukaClient._send` - the last call
    before `socket.sendall` - and parsed back with the protocol's own
    `unpack_command`, so what is checked is the int32 millimetres the
    controller would actually receive, not the float the planner
    computed. Asserting on `planner.cycle`'s return value instead would
    re-ask the module that does the filtering whether it filtered.

    This is the test that must survive any future change to how
    unreachable candidates are handled. Skipping them is a planning
    decision and can be revisited; a coordinate outside the envelope
    arriving at the socket cannot.
    """
    from execution.execution import KukaClient
    from execution.protocol import OpCode, unpack_command

    sent: list[bytes] = []
    real_send = KukaClient._send

    def _recording_send(self, packet):
        sent.append(bytes(packet))
        return real_send(self, packet)

    monkeypatch.setattr(KukaClient, "_send", _recording_send)

    stats = main_run(str(tight_envelope_config))

    x0, x1, y0, y1 = TIGHT_ENVELOPE
    motions = [
        c for c in (unpack_command(p) for p in sent)
        if c.op in (OpCode.MOVE_TO, OpCode.PICK_AND_PLACE)
    ]
    assert motions, "no motion command was sent; this test asserted nothing"
    # The envelope must actually have been in play, or a run that simply
    # fits inside it would pass this test while proving nothing.
    assert stats["unreachable_place_targets"] \
        + stats["unreachable_batteries"] > 0, (
        "fixture must put part of the frame out of reach")

    for cmd in motions:
        assert x0 <= cmd.x_mm <= x1 and y0 <= cmd.y_mm <= y1, (
            f"{cmd.op.name}({cmd.x_mm}, {cmd.y_mm}) mm went on the wire "
            f"outside the declared envelope x [{x0}, {x1}] y [{y0}, {y1}]")


def test_run_reports_what_the_arm_declined_to_serve(
        tight_envelope_config: Path, tmp_path: Path):
    """Skipped-as-unreachable has to be a NUMBER, not a shorter queue.

    "The arm cannot reach those cells" and "there are no cells" produce
    an identical `placed` count, an identical `queue_poses` deficit and
    an identical clean exit. A demo that serves a third of the scene
    while every other statistic reads like success is the silent
    degradation this project keeps finding; these counters are the only
    thing that distinguishes the two.
    """
    receipt = tmp_path / "receipts" / "tight.txt"
    stats = main_run(str(tight_envelope_config), receipt_path=str(receipt))

    assert stats["unreachable_place_targets"] > 0
    assert stats["queue_poses"] > 0, (
        "the reachable part of the frame must still be served")

    text = receipt.read_text(encoding="utf-8")
    assert (f"unreachable place targets: {stats['unreachable_place_targets']}"
            in text)
    assert (f"unreachable batteries    : {stats['unreachable_batteries']}"
            in text)
    assert (f"unreachable cartridges   : {stats['unreachable_cartridges']}"
            in text)


def test_a_run_that_can_reach_nothing_is_a_failed_run(tmp_path: Path):
    """Skipping the unreachable is deliberate. Skipping ALL of it and
    exiting 0 is not.

    An envelope disjoint from the field of view means the camera and the
    arm do not describe the same cell - `workspace_bounds_mm`,
    `origin_offset_*` or `mm_per_px` is wrong - and every other number
    the run prints is indistinguishable from an empty table.
    """
    root = tmp_path / "auto-pick-unreachable"
    (root / "configs").mkdir(parents=True)
    (root / "recog" / "dataset" / "images").mkdir(parents=True)
    (root / "recog" / "dataset" / "annotations").mkdir(parents=True)

    from recog.synth_dataset import generate_dataset
    generate_dataset(str(root / "recog" / "dataset"), n=2, seed=9,
                     size=(480, 640))

    (root / "configs" / "recognition.yaml").write_text(
        "dataset:\n"
        f"  img_dir: {root}/recog/dataset/images\n"
        "training:\n  checkpoint_dir: /tmp/nothing\n"
    )
    # Entirely in -x, while origin_offset_x_mm = 0 puts the whole frame
    # in +x. Nothing in view is reachable.
    (root / "configs" / "planning.yaml").write_text(
        "battery: {diameter_mm: 18.5, length_mm: 65.0}\n"
        "camera: {mm_per_px_x: 0.38, workspace_bounds_mm: "
        "{x_min: -700, x_max: -10, y_min: -350, y_max: 350}}\n"
        "cartridge: {safety_margin_px: 4}\n"
        "occupancy_grid: {resolution_mm_per_cell: 1.5}\n"
        "motion: {grasp_height_mm: 5, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54629, max_retries: 3}\n"
        "motion: {transport_height_mm: 80, vacuum_level_percent: 80}\n"
        "simulation: {listen_host: 127.0.0.1, listen_port: 54629, "
        "drop_probability: 0.0, simulated_move_time_ms_per_100mm: 5}\n"
    )
    demo = root / "configs" / "demo_nowhere.yaml"
    demo.write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode:\n  source: synthetic\n  robot: mock\n  max_cycles: 2\n"
        "  log_level: WARNING\n  stop_on_empty_queue: false\n"
    )

    with pytest.raises(RuntimeError, match="outside the robot workspace"):
        main_run(str(demo))


def test_full_cycle_with_a_stub_segmenter_attaches_masks_and_plans():
    """Recognition -> Planning boundary, end to end, with no torch and no
    trained checkpoint involved.

    A stub segmenter stands in for recog.bay_segmenter.BaySegmenter
    (same segment_batch(crops) -> List[np.ndarray] contract). Two
    cartridges are in frame: one whose direct/derived estimates agree
    (IoU ~0.97) and one where the bay is predicted over only the left
    half of the interior (IoU ~0.44).

    Asserts:
    * Snapshot.cartridge_masks is populated by attach_cartridge_masks
      (the Recognition-side batching from Task 5);
    * the planner produces a non-empty queue;
    * BOTH cartridges are planned. The second one used to be discarded
      by the tau gate; that gate is retired (FDR v3 section 13.2.1) and
      the low-IoU cartridge is a perfectly good one. Its placement area
      is still confined to where the bay was actually predicted, which
      is P_safe doing the work the gate was mistakenly credited with;
    * neither planner skip-counter fires, so a cartridge silently
      vanishing into the blanket `except Exception` still fails here.
    """
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from plan.placement_area import SegmentationPlacementAreaExtractor
    from plan.planner import Planner, PlannerConfig
    from plan.scene import WorkspaceBounds
    from recog.inference import attach_cartridge_masks

    # Cartridge A: direct (bay channel) and derived (eroded interior)
    # estimates agree - a normal, plannable cartridge.
    agrees = np.zeros((288, 131), np.int8)
    agrees[5:283, 5:126] = CH_CARTRIDGE
    agrees[12:276, 12:119] = CH_BAY

    # Cartridge B: the bay channel only covers the left half of the
    # interior, so direct and derived agree over only ~44% of their
    # union at wall_inset_mm=4.0, mm_per_px=0.625 - under every tau this
    # repo ever quoted.
    disagrees = np.zeros((288, 131), np.int8)
    disagrees[5:283, 5:126] = CH_CARTRIDGE
    disagrees[12:276, 12:60] = CH_BAY

    class _StubSegmenter:
        """Mimics recog.bay_segmenter.BaySegmenter's public contract."""

        def __init__(self):
            self.calls = 0

        def segment_batch(self, crops):
            self.calls += 1
            assert len(crops) == 2
            return [agrees, disagrees]

    img = np.zeros((720, 1280, 3), np.uint8)
    dets = [
        Detection(BBox(100, 100, 231, 388), ClassLabel.CARTRIDGE, 0.95),
        Detection(BBox(400, 100, 531, 388), ClassLabel.CARTRIDGE, 0.95),
        Detection(BBox(140, 150, 160, 190), ClassLabel.BATTERY, 0.9),
    ]
    snap = Snapshot(detections=dets, image_shape=(720, 1280))

    segmenter = _StubSegmenter()
    attach_cartridge_masks(snap, img, segmenter)
    assert segmenter.calls == 1, "one batched call, not one per cartridge"
    assert set(snap.cartridge_masks) == {0, 1}, \
        "Snapshot.cartridge_masks must be populated for both cartridges"

    cfg = PlannerConfig(
        battery_width_mm=18.5, battery_length_mm=65.0, mm_per_px=0.625)
    extractor = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0)
    workspace = WorkspaceBounds(-1000, 1000, -1000, 1000)
    planner = Planner(cfg, extractor, workspace)

    queue = planner.cycle(snap, img)   # must not raise

    assert len(queue) >= 1, "the planner must produce a queue"

    # Direct signal, not a battery-count coincidence: both cartridges
    # got a placement area. Checking the queue's cartridge_id
    # composition alone would not discriminate this from "cartridge 1
    # just ran out of batteries" - both cartridges share one pool and
    # cartridge 0 is processed first.
    assert planner.env.cartridge(0).placeable_rectangle is not None
    assert planner.env.cartridge(1).placeable_rectangle is not None, (
        "the low-agreement cartridge must be planned - the tau gate "
        "that used to discard it is retired")
    assert planner.placement_disagreement_count == 0
    assert planner.bad_detector_box_count == 0

    # ...but only over the half of the interior the segmenter actually
    # called `bay`. P_safe is what confines it, and it is still applied.
    wide = planner.env.cartridge(0).placeable_rectangle
    narrow = planner.env.cartridge(1).placeable_rectangle
    assert narrow.width < 0.6 * wide.width, (
        "P_safe must still confine the placement area to the predicted "
        "bay - removing the tau gate must not remove the intersection")


# ----------------------------------------- main.py's segmentation wiring ----
#
# The stub-segmenter test above builds the extractor and planner BY
# HAND, so it never touches main.py's own selection logic. These do. The
# failure they exist to catch is a segmentation config that loads, runs,
# reports a clean set of statistics and never once invokes the model -
# which is what main.py did before this wiring existed
# (`detector.segmenter is None`, `cartridge_masks` empty,
# `plan.arbitration` never imported).

class _StubSegmenter:
    """recog.bay_segmenter.BaySegmenter's public contract, no torch."""

    def __init__(self, label_map):
        self.label_map = label_map
        self.calls = 0
        self.crops_seen = 0

    def segment_batch(self, crops):
        import numpy as np
        self.calls += 1
        self.crops_seen += len(crops)
        out = []
        for c in crops:
            h, w = c.shape[:2]
            lm = np.zeros((h, w), np.int8)
            src = self.label_map[:h, :w]
            lm[:src.shape[0], :src.shape[1]] = src
            out.append(lm)
        return out


def _agreeing_label_map(h: int = 288, w: int = 131):
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    lm = np.zeros((h, w), np.int8)
    lm[5:h - 5, 5:w - 5] = CH_CARTRIDGE
    lm[12:h - 12, 12:w - 12] = CH_BAY
    return lm


def test_load_detector_forwards_the_segmenter_to_the_detector(
        tmp_path: Path, monkeypatch):
    """The seam this whole path was blocked on: load_detector took no
    segmenter, so FasterRCNNDetector's `segmenter=None` default could
    never be overridden from main.py, and the second-stage call inside
    detect() was unreachable in production."""
    import recog.inference as inference

    captured = {}

    class _FakeFRCNN:
        def __init__(self, checkpoint, cfg, segmenter=None):
            captured["checkpoint"] = checkpoint
            captured["segmenter"] = segmenter

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not really a checkpoint")

    monkeypatch.setattr(inference, "FasterRCNNDetector", _FakeFRCNN)
    monkeypatch.setattr(inference, "_TORCH_AVAILABLE", True)

    sentinel = object()
    inference.load_detector(str(ckpt), {}, segmenter=sentinel)
    assert captured["segmenter"] is sentinel

    # ...and the old two-argument call still works unchanged.
    inference.load_detector(str(ckpt), {})
    assert captured["segmenter"] is None


def test_load_detector_refuses_a_segmenter_it_cannot_run():
    """HeuristicDetector has no second stage. Returning it while holding
    a segmenter would produce empty cartridge_masks, a ValueError per
    cartridge inside the extractor, and a blanket `except Exception` in
    plan/planner.py absorbing every one of them - a run that reports
    zero placements and looks healthy."""
    from recog.inference import HeuristicDetector, load_detector

    with pytest.raises(RuntimeError, match="cannot run a second stage"):
        load_detector(None, {}, segmenter=object())

    # No segmenter: the documented fallback is unchanged.
    assert isinstance(load_detector(None, {}), HeuristicDetector)


def test_build_planner_selects_the_extractor_the_config_asks_for():
    from main import _build_planner
    from plan.placement_area import (HeuristicPlacementAreaExtractor,
                                     SegmentationPlacementAreaExtractor)

    plan_cfg = {
        "battery": {"diameter_mm": 18.5, "length_mm": 65.0},
        "camera": {"mm_per_px_x": 0.38},
        "cartridge": {"safety_margin_px": 4},
        "occupancy_grid": {"resolution_mm_per_cell": 1.5},
    }

    heuristic = _build_planner(plan_cfg)
    assert isinstance(heuristic.extractor, HeuristicPlacementAreaExtractor)
    # The heuristic must NOT be handed label maps - its extract() takes
    # no **kwargs, and the TypeError would vanish into the planner's
    # blanket except, unplanning every cartridge forever.
    assert heuristic._extractor_accepts_label_map is False

    seg = _build_planner(plan_cfg, segmentation=True, mm_per_px=0.625)
    assert isinstance(seg.extractor, SegmentationPlacementAreaExtractor)
    assert seg._extractor_accepts_label_map is True, (
        "selecting the segmentation extractor is pointless if the "
        "planner will not pass it a label_map")

    # mode.mm_per_px reaches BOTH consumers, or the wall-erosion radius
    # and the workspace mapping disagree about the same camera.
    assert seg.cfg.mm_per_px == 0.625
    assert seg.extractor.mm_per_px == 0.625
    assert heuristic.cfg.mm_per_px == 0.38


def test_build_segmenter_is_absent_or_loud_never_quietly_skipped(
        tmp_path: Path):
    from main import _build_segmenter

    # No block -> the torch-free default path, deliberately.
    assert _build_segmenter({}) is None

    # A block that names nothing is a typo, not a request for the
    # heuristic path.
    with pytest.raises(ValueError, match="checkpoint"):
        _build_segmenter({"config": "configs/segmentation.yaml"})

    with pytest.raises(FileNotFoundError):
        _build_segmenter({"checkpoint": str(tmp_path / "nope.pt")})


@pytest.fixture
def demo_seg_config(tmp_path: Path) -> Path:
    """A demo.yaml carrying a mode.segmentation block.

    The checkpoint is a real (empty) file so _build_segmenter's
    existence check passes; the tests monkeypatch the construction
    itself, because building a DeepLabv3 here would drag in torch and
    ~42 MB of COCO weights to assert something about main.py's wiring.
    """
    root = tmp_path / "auto-pick"
    (root / "configs").mkdir(parents=True)
    (root / "recog" / "dataset" / "images").mkdir(parents=True)
    (root / "recog" / "dataset" / "annotations").mkdir(parents=True)

    from recog.synth_dataset import generate_dataset
    generate_dataset(str(root / "recog" / "dataset"), n=3, seed=9,
                     size=(480, 640))

    ckpt = root / "seg.pt"
    ckpt.write_bytes(b"")

    (root / "configs" / "recognition.yaml").write_text(
        "dataset:\n"
        f"  img_dir: {root}/recog/dataset/images\n"
        "training:\n  checkpoint_dir: /tmp/nothing\n"
    )
    (root / "configs" / "planning.yaml").write_text(
        "battery: {diameter_mm: 18.5, length_mm: 65.0}\n"
        "camera: {mm_per_px_x: 0.38, workspace_bounds_mm: "
        "{x_min: -350, x_max: 350, y_min: -350, y_max: 350}}\n"
        "cartridge: {safety_margin_px: 4}\n"
        "occupancy_grid: {resolution_mm_per_cell: 1.5}\n"
        "motion: {grasp_height_mm: 5, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54613, max_retries: 3}\n"
        "motion: {transport_height_mm: 80, vacuum_level_percent: 80}\n"
        "simulation: {listen_host: 127.0.0.1, listen_port: 54613, "
        "drop_probability: 0.0, simulated_move_time_ms_per_100mm: 5}\n"
    )
    demo = root / "configs" / "demo_seg.yaml"
    demo.write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode:\n  source: synthetic\n  robot: mock\n  max_cycles: 2\n"
        "  log_level: WARNING\n  mm_per_px: 0.625\n"
        "  stop_on_empty_queue: false\n"
        "  segmentation:\n"
        f"    checkpoint: {ckpt.as_posix()}\n"
    )
    return demo


def _patch_detector(monkeypatch, holder, label_map):
    """Replace main.load_detector with a torch-free stand-in that runs
    the SAME second stage FasterRCNNDetector.detect does."""
    import main as main_mod
    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _StubDetector:
        def __init__(self, segmenter):
            self.segmenter = segmenter

        def __call__(self, image_rgb):
            snap = Snapshot(
                detections=[
                    Detection(BBox(100, 100, 231, 388),
                              ClassLabel.CARTRIDGE, 0.95),
                    Detection(BBox(20, 20, 40, 80),
                              ClassLabel.BATTERY, 0.9),
                ],
                image_shape=image_rgb.shape[:2],
            )
            if self.segmenter is not None:
                attach_cartridge_masks(snap, image_rgb, self.segmenter)
            return snap

    def _fake_load_detector(checkpoint, cfg, segmenter=None):
        assert segmenter is not None, (
            "main.run must hand the segmenter to load_detector")
        holder.append(segmenter)
        return _StubDetector(segmenter)

    monkeypatch.setattr(main_mod, "load_detector", _fake_load_detector)
    monkeypatch.setattr(
        main_mod, "_build_segmenter",
        lambda cfg: _StubSegmenter(label_map) if cfg else None)


def test_run_puts_the_segmenter_in_the_loop_and_receipts_it(
        demo_seg_config: Path, monkeypatch, tmp_path: Path):
    """The end-to-end claim, asserted rather than assumed: given a
    mode.segmentation block, main.run builds a segmenter, hands it to
    the detector, gets label maps back on the Snapshot, and plans from
    them through SegmentationPlacementAreaExtractor."""
    holder: list = []
    _patch_detector(monkeypatch, holder, _agreeing_label_map())

    receipt = tmp_path / "receipts" / "run.txt"
    stats = main_run(str(demo_seg_config), receipt_path=str(receipt))

    assert holder, "load_detector was never given a segmenter"
    assert holder[0].calls >= 1, "the segmenter was never invoked"
    assert stats["cartridge_masks"] >= 1, "no label map reached the planner"
    assert stats["placement_areas"] >= 1, (
        "the segmentation extractor produced no placement area, so the "
        "model ran for nothing")

    # The receipt is generated, not narrated.
    text = receipt.read_text(encoding="utf-8")
    assert "SegmentationPlacementAreaExtractor" in text
    assert f"placement areas        : {stats['placement_areas']}" in text
    # The receipt states WHERE the scale came from, not just its value.
    # `mode.mm_per_px: 0.625` is now a fallback for frames that carry no
    # calibration, and this fixture's frames (bare PNGs in a tmp dir with
    # no render sidecar) are exactly that case - so the fallback is what
    # applied, and the receipt has to be able to say so. A receipt that
    # printed one bare number could not distinguish this run from one
    # planned at the frames' own scales, which is the confusion that
    # made the constant survive as long as it did.
    assert "fallback 0.625" in text
    assert f"frames w/ own scale: {stats['frames_with_scale']} of " in text
    assert stats["frames_with_scale"] == 0, (
        "these fixture frames carry no render sidecar, so no frame "
        "should have reported a scale of its own")


def test_run_raises_when_a_configured_segmenter_plans_nothing(
        demo_seg_config: Path, monkeypatch):
    """The exact failure configs/demo_seg.yaml exists to avoid: the
    segmenter runs on frames it was not trained for, predicts no `bay`,
    and the pipeline completes with zero placements while looking
    healthy. An all-background label map stands in for that."""
    import numpy as np

    holder: list = []
    _patch_detector(monkeypatch, holder, np.zeros((288, 131), np.int8))

    with pytest.raises(RuntimeError, match="no placement area"):
        main_run(str(demo_seg_config))


def test_torch_free_demo_does_not_build_a_segmenter(
        demo_config: Path, monkeypatch):
    """configs/demo.yaml has no mode.segmentation block, and the FDR's
    reproducibility claim rests on that path staying torch-free. If this
    wiring ever starts constructing a segmenter unconditionally, this
    fails."""
    import main as main_mod

    def _explode(cfg):
        raise AssertionError(
            f"the torch-free path must not build a segmenter (got {cfg!r})")

    monkeypatch.setattr(
        main_mod, "_build_segmenter",
        lambda cfg: _explode(cfg) if cfg else None)

    stats = main_run(str(demo_config))
    assert stats["cycles"] >= 1
    assert stats["cartridge_masks"] == 0
    assert stats["placement_areas"] >= 1


# ------------------------------------------------ per-frame calibration ----

def test_synthetic_source_yields_each_frames_own_scale(tmp_path):
    """The image source is where the calibration enters, because a frame
    source stands in for a camera and a camera is the thing that knows how
    big a pixel is. `recog.generate3d` writes the render sidecar beside
    the images; `_synthetic_source` reads it per frame."""
    import json

    import cv2
    import numpy as np

    from main import _synthetic_source

    imgs = tmp_path / "images"
    meta = tmp_path / "meta"
    imgs.mkdir()
    meta.mkdir()
    for stem, ortho in (("scene_00000", 0.6291), ("scene_00001", 1.3377)):
        cv2.imwrite(str(imgs / f"{stem}.png"), np.zeros((72, 128, 3), np.uint8))
        (meta / f"{stem}.json").write_text(
            json.dumps({"camera": {"ortho_scale": ortho},
                        "width": 1280, "height": 720}), encoding="utf-8")

    src = _synthetic_source(str(imgs))
    _, first = next(src)
    _, second = next(src)

    assert first == pytest.approx(0.6291 * 1000 / 1280)
    assert second == pytest.approx(1.3377 * 1000 / 1280)
    # Two frames of one dataset, two scales. A single configured constant
    # cannot be right for both, which is the whole defect.
    assert first != second


def test_synthetic_source_yields_none_for_frames_with_no_sidecar(tmp_path):
    """`recog/synth_dataset.py` draws its rectangles with cv2 and records
    no camera. None means unknown, and the configured fallback - not a
    guess - is what covers it."""
    import cv2
    import numpy as np

    from main import _synthetic_source

    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((40, 40, 3), np.uint8))
    _, scale = next(_synthetic_source(str(tmp_path)))
    assert scale is None


def test_the_snapshot_carries_the_frame_scale_into_planning(
        demo_seg_config: Path, monkeypatch, tmp_path):
    """End to end: the scale on the frame reaches Snapshot.mm_per_px, and
    the planner records it on the cartridge it measured. The extractor and
    the planner must be looking at the SAME number - they used to hold
    separate copies that main._build_planner kept in step by hand."""
    holder: list = []
    _patch_detector(monkeypatch, holder, _agreeing_label_map())

    from plan.planner import Planner

    seen: list = []
    real_cycle = Planner.cycle

    def spy(self, snapshot, image_rgb):
        seen.append(snapshot.mm_per_px)
        return real_cycle(self, snapshot, image_rgb)

    monkeypatch.setattr(Planner, "cycle", spy)
    main_run(str(demo_seg_config))

    assert seen, "the planner never ran"
    # This fixture's frames carry no sidecar, so the frames report None
    # and the configured fallback applies - and that has to be visible as
    # None rather than pre-filled with the fallback, or the receipt could
    # not tell the two runs apart.
    assert all(s is None for s in seen)
