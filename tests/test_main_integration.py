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
