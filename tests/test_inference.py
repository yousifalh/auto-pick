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
    """Batching is load-bearing: 8 cartridges cost 61.8 ms looped and
    18.1 ms batched at the deployed fp16/256 config, against a 50 ms
    end-to-end budget (docs/receipts/seg_eval.txt)."""
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


# ------------------------------------------------- FasterRCNN hot path --
#
# `FasterRCNNDetector.detect` is per-frame code inside a 50 ms budget and
# has no test that runs it, because constructing one needs a checkpoint.
# These build the object without one - `detect` reads only `self.model`,
# `self.device`, `self._u8_scale` and `self.segmenter` - so the frame
# preprocessing and the output decode are exercised directly. Both are
# EQUIVALENCE tests: they pin the values against the pre-optimisation
# expressions, which is the whole claim being made about the change.

# torch is imported defensively, NOT via a module-level importorskip:
# HeuristicDetector exists precisely so this pipeline stays exercisable
# without torch (see recog/inference.py's module docstring), and skipping
# the whole file would take those tests down with it.
try:                                    # pragma: no cover - import guard
    import torch
except Exception:                       # pragma: no cover - torch-free CI
    torch = None

_DEVICES = ([] if torch is None else
            ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])) \
    or [pytest.param("cpu", marks=pytest.mark.skip(reason="torch not installed"))]


def _recording_detector(out, device):
    """A `FasterRCNNDetector` whose model is a stub that records its input.

    Subclassed rather than mocked, and constructed without calling
    `__init__`: `detect` is inherited verbatim, so what these tests
    measure is the shipped method, and no 160 MB checkpoint is needed to
    reach it. `detect` reads only these four attributes.
    """
    from recog.inference import FasterRCNNDetector

    class _Recording(FasterRCNNDetector):
        def __init__(self):
            self.device = torch.device(device)
            self._u8_scale = torch.tensor(255.0, device=self.device)
            self.segmenter = None
            self.received = None
            self.model = self._record

        def _record(self, images):
            self.received = images[0]
            return [out]

    return _Recording()


def _stub_output(device):
    """Two real detections and one unknown class id, on `device`."""
    d = torch.device(device)
    return {
        "boxes": torch.tensor([[10.5, 20.25, 30.75, 40.125],
                               [1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]], device=d),
        "labels": torch.tensor([1, 3, 2], device=d),
        "scores": torch.tensor([0.9140625, 0.5, 0.703125], device=d),
    }


@pytest.mark.parametrize("device", _DEVICES)
def test_detect_hands_the_model_the_same_frame_the_float32_path_did(device):
    """Transferring uint8 and widening on the device must reproduce the
    old `astype(float32) / 255.0` + `np.transpose` frame BIT FOR BIT.

    Not a formality on CUDA. `div_(255)` with a Python scalar takes
    torch's Scalar overload, which multiplies by the reciprocal there and
    disagrees with a true divide by 1 ulp on 126 of the 256 possible byte
    values; pushed through a real Faster R-CNN that returned a different
    detection SET, not just different last bits. Dividing by a 0-dim
    tensor is the correctly-rounded path, and this asserts it.
    """
    rng = np.random.default_rng(0)
    image_rgb = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)

    det = _recording_detector(_stub_output(device), device)
    det.detect(image_rgb)

    before_the_change = torch.as_tensor(
        np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1)),
        dtype=torch.float32,
    ).to(torch.device(device))

    assert det.received.shape == before_the_change.shape
    assert det.received.dtype == torch.float32
    assert torch.equal(det.received, before_the_change), (
        "the frame handed to the model changed value - this optimisation "
        "is only allowed to change how the bytes travel, not what they are")


@pytest.mark.parametrize("device", _DEVICES)
def test_detect_decodes_exactly_what_the_per_element_loop_decoded(device):
    """The bulk-transfer decode must produce the same Detection list the
    `lbl.item()` / `score.item()` / `box.cpu()` loop produced, including
    dropping the unrecognised class id as background."""
    from common.types import BBox, ClassLabel, Detection
    from recog.inference import _ID_TO_LABEL

    out = _stub_output(device)
    det = _recording_detector(out, device)
    snap = det.detect(np.zeros((16, 16, 3), np.uint8))

    before_the_change = []
    for box, lbl, score in zip(out["boxes"], out["labels"], out["scores"]):
        label = _ID_TO_LABEL.get(int(lbl.item()), ClassLabel.BACKGROUND)
        if label == ClassLabel.BACKGROUND:
            continue
        x1, y1, x2, y2 = box.cpu().numpy().tolist()
        before_the_change.append(
            Detection(BBox(x1, y1, x2, y2), label, float(score.item())))

    assert len(snap.detections) == 2, "class id 3 is not ours; it must drop"
    for got, want in zip(snap.detections, before_the_change):
        assert got.label is want.label
        assert got.confidence == want.confidence      # exact, not approx
        assert (got.bbox.xmin, got.bbox.ymin, got.bbox.xmax, got.bbox.ymax) \
            == (want.bbox.xmin, want.bbox.ymin, want.bbox.xmax, want.bbox.ymax)


@pytest.mark.parametrize("device", _DEVICES)
def test_detect_reads_the_device_three_times_not_three_times_per_box(device):
    """The reason for the change, pinned so it cannot regress.

    Every `.item()` and every `.cpu()` on a CUDA tensor is a copy AND a
    full stream synchronisation. The old loop did three per detection -
    roughly 30-90 syncs per frame at the shipped 0.70 confidence
    threshold - to read numbers that arrive in one transfer per field.
    """
    calls = {"item": 0, "cpu": 0}
    item, cpu = torch.Tensor.item, torch.Tensor.cpu

    def counting_item(self):
        calls["item"] += 1
        return item(self)

    def counting_cpu(self, *a, **kw):
        calls["cpu"] += 1
        return cpu(self, *a, **kw)

    det = _recording_detector(_stub_output(device), device)
    torch.Tensor.item, torch.Tensor.cpu = counting_item, counting_cpu
    try:
        det.detect(np.zeros((16, 16, 3), np.uint8))
    finally:
        torch.Tensor.item, torch.Tensor.cpu = item, cpu

    assert calls["item"] == 0, (
        f"{calls['item']} per-element .item() calls; boxes, labels and "
        "scores must come back in bulk and be read from numpy")
    assert calls["cpu"] == 3, (
        f"{calls['cpu']} device->host transfers, not 3 (boxes, labels, "
        "scores)")


def test_detect_accepts_a_negative_stride_frame():
    """`bgr[:, :, ::-1]` without a copy is a plausible caller, and
    `torch.from_numpy` refuses a negative stride outright. The old
    `.astype(np.float32)` laundered that silently; losing it would be a
    behaviour change, not a speed-up."""
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (12, 20, 3), dtype=np.uint8)
    reversed_view = bgr[:, :, ::-1]
    assert reversed_view.strides[2] < 0, "fixture must be a negative-stride view"

    det = _recording_detector(_stub_output("cpu"), "cpu")
    det.detect(reversed_view)                     # must not raise

    expected = torch.as_tensor(
        np.transpose(reversed_view.astype(np.float32) / 255.0, (2, 0, 1)))
    assert torch.equal(det.received, expected)


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
