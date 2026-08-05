"""Heuristic inference tests.

These exercise the OpenCV fallback path (``HeuristicDetector``) on the
synthetic dataset, which is the route used in software-only smoke tests
when no trained Faster R-CNN checkpoint is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.types import ClassLabel
from recog.inference import HeuristicDetector, load_detector


def _synth(tmp_path: Path, n: int = 2):
    from recog.synth_dataset import generate_dataset
    generate_dataset(str(tmp_path), n=n, seed=7, size=(480, 640))
    img_dir = tmp_path / "images"
    files = sorted(img_dir.glob("*.png"))
    assert files, "synth dataset must produce PNGs"
    return files


def _read_rgb(p: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(p))
    return bgr[:, :, ::-1].copy()


def test_heuristic_detector_returns_snapshot(tmp_path: Path):
    f = _synth(tmp_path)[0]
    det = HeuristicDetector()
    snap = det(_read_rgb(f))
    assert snap is not None
    # Should find at least 1 cartridge and >=1 battery on a synth scene
    n_cart = len(snap.of(ClassLabel.CARTRIDGE))
    n_batt = len(snap.of(ClassLabel.BATTERY))
    assert n_cart >= 1
    assert n_batt >= 1


def test_heuristic_detector_confidence_in_range(tmp_path: Path):
    f = _synth(tmp_path)[0]
    snap = HeuristicDetector()(_read_rgb(f))
    for d in snap.detections:
        assert 0.0 <= d.confidence <= 1.0


def test_load_detector_falls_back_to_heuristic():
    # Non-existent checkpoint — must return HeuristicDetector, not raise.
    det = load_detector("/nonexistent/path.pt", {})
    assert isinstance(det, HeuristicDetector)
