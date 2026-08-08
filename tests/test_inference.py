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


def test_detector_without_a_segmenter_leaves_masks_empty():
    """The existing single-stage path must be completely unaffected."""
    import numpy as np

    from recog.inference import HeuristicDetector

    det = HeuristicDetector()
    snap = det.detect(np.zeros((240, 320, 3), np.uint8))
    assert snap.cartridge_masks == {}


def test_segmenter_is_called_once_per_frame_not_once_per_cartridge():
    """Batching is load-bearing: 8 cartridges cost 101 ms looped and
    18.5 ms batched, against a 50 ms end-to-end budget."""
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _SpySegmenter:
        def __init__(self):
            self.calls = 0
            self.batch_sizes = []

        def segment_batch(self, crops):
            self.calls += 1
            self.batch_sizes.append(len(crops))
            return [np.zeros(c.shape[:2], np.int8) for c in crops]

    spy = _SpySegmenter()
    snap = Snapshot(detections=[
        Detection(BBox(10, 10, 50, 60), ClassLabel.CARTRIDGE, 0.9),
        Detection(BBox(70, 10, 110, 60), ClassLabel.CARTRIDGE, 0.9),
        Detection(BBox(10, 80, 30, 100), ClassLabel.BATTERY, 0.9),
    ])
    attach_cartridge_masks(snap, np.zeros((200, 200, 3), np.uint8), spy)

    assert spy.calls == 1, f"segmenter called {spy.calls} times, not once"
    assert spy.batch_sizes == [2], "batteries must not be segmented"
    assert set(snap.cartridge_masks) == {0, 1}


def test_masks_are_keyed_by_detection_index():
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _Seg:
        def segment_batch(self, crops):
            return [np.full(c.shape[:2], 2, np.int8) for c in crops]

    snap = Snapshot(detections=[
        Detection(BBox(10, 80, 30, 100), ClassLabel.BATTERY, 0.9),
        Detection(BBox(10, 10, 50, 60), ClassLabel.CARTRIDGE, 0.9),
    ])
    attach_cartridge_masks(snap, np.zeros((200, 200, 3), np.uint8), _Seg())

    assert set(snap.cartridge_masks) == {1}, (
        "index 1 is the cartridge; keying by position within the "
        "cartridge subset would misalign every mask")
    assert snap.cartridge_masks[1].shape == (50, 40)


def test_no_cartridges_means_no_segmenter_call():
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _Seg:
        def __init__(self):
            self.calls = 0

        def segment_batch(self, crops):
            self.calls += 1
            return []

    seg = _Seg()
    snap = Snapshot(detections=[
        Detection(BBox(0, 0, 4, 4), ClassLabel.BATTERY, 0.9)])
    attach_cartridge_masks(snap, np.zeros((50, 50, 3), np.uint8), seg)
    assert seg.calls == 0
