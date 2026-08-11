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
        "motion: {approach_height_mm: 60, insert_height_mm: 2}\n"
    )
    (root / "configs" / "execution.yaml").write_text(
        "kuka: {host: 127.0.0.1, port: 54611, max_retries: 3}\n"
        "motion: {approach_height_mm: 60, transport_height_mm: 80, "
        "insert_height_mm: 2, vacuum_level_percent: 80}\n"
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


def test_main_run_short_cycle(demo_config: Path):
    stats = main_run(str(demo_config))
    assert stats["cycles"] >= 1
    # In a 0%-drop environment most cycles should place successfully
    assert stats["placed"] + stats["pick_failed"] + stats["place_failed"] == \
        stats["cycles"]


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
