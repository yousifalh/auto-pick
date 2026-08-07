"""Per-ROI crop dataset built from the COCO-RLE sidecar."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.config import load_yaml
from recog.seg_dataset import (SEG_CHANNELS, _rng_for_worker, jitter_box,
                               rasterise_crop)
from recog.synth3d.annotate import rle_encode

ROOT = Path(__file__).resolve().parents[1]


def _ann(cls, mask, cat_id):
    ys, xs = np.nonzero(mask)
    return {"class": cls, "category_id": cat_id,
            "segmentation": rle_encode(mask),
            "bbox_xyxy": [int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1]}


def test_channel_order_is_the_contract():
    """Plan D's arbitration indexes these directly. Reordering them
    silently changes which mask is subtracted from which."""
    assert SEG_CHANNELS == {
        "background": 0, "cartridge": 1, "bay": 2,
        "electronics": 3, "obstruction": 4, "battery": 5,
    }


def test_jitter_stays_within_the_requested_fraction():
    rng = np.random.default_rng(0)
    box = (100, 200, 200, 400)          # 100 x 200
    for _ in range(500):
        x0, y0, x1, y1 = jitter_box(box, rng, 0.10)
        assert abs(x0 - 100) <= 10 and abs(x1 - 200) <= 10
        assert abs(y0 - 200) <= 20 and abs(y1 - 400) <= 20
        assert x1 > x0 and y1 > y0


def test_zero_jitter_is_the_identity():
    rng = np.random.default_rng(0)
    assert jitter_box((10, 20, 30, 40), rng, 0.0) == (10, 20, 30, 40)


def test_rasterise_paints_battery_over_bay():
    """A cell seated in a bay must win. That paint order IS the modal
    definition of placement_area: the free floor is what is left."""
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[5:35, 5:35] = 1
    cell = np.zeros((H, W), np.uint8); cell[10:20, 10:20] = 1
    anns = [_ann("placement_area", bay, 4), _ann("battery", cell, 1)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=40)
    assert lab[15, 15] == SEG_CHANNELS["battery"]
    assert lab[30, 30] == SEG_CHANNELS["bay"]
    assert lab[0, 0] == SEG_CHANNELS["background"]


def test_rasterise_paints_electronics_and_obstruction_over_bay():
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:40, 0:40] = 1
    pcb = np.zeros((H, W), np.uint8); pcb[0:10, :] = 1
    glue = np.zeros((H, W), np.uint8); glue[20:24, 20:24] = 1
    anns = [_ann("placement_area", bay, 4),
            _ann("electronics_module", pcb, 3),
            _ann("obstruction", glue, 5)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=40)
    assert lab[5, 5] == SEG_CHANNELS["electronics"]
    assert lab[22, 22] == SEG_CHANNELS["obstruction"]
    assert lab[35, 35] == SEG_CHANNELS["bay"]


def test_rasterise_resizes_with_nearest_neighbour():
    """Labels must never be interpolated: averaging class 2 and 4 gives
    class 3, which is a different object."""
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:40, 0:20] = 1
    pcb = np.zeros((H, W), np.uint8); pcb[0:40, 20:40] = 1
    anns = [_ann("placement_area", bay, 4),
            _ann("electronics_module", pcb, 3)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=8)
    assert set(np.unique(lab)) <= {SEG_CHANNELS["bay"],
                                   SEG_CHANNELS["electronics"]}


def test_crop_outside_the_annotation_is_all_background():
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:10, 0:10] = 1
    lab = rasterise_crop([_ann("placement_area", bay, 4)],
                         (20, 20, 40, 40), out_size=16)
    assert (lab == SEG_CHANNELS["background"]).all()


# --------------------------------------------------- native (out_size=None) --
#
# recog.seg_evaluate needs the un-resized crop (a jittered union box is not
# square, so a single scalar mm_per_px cannot describe a resized one) and
# gets it by calling these same functions with out_size=None, rather than
# keeping a hand-copied second version of either loop.

def test_rasterise_crop_out_size_none_stays_at_native_resolution():
    H, W = 40, 30
    bay = np.zeros((H, W), np.uint8); bay[5:15, 5:15] = 1
    lab = rasterise_crop([_ann("placement_area", bay, 4)], (0, 0, W, H),
                         out_size=None)
    assert lab.shape == (H, W)
    assert lab[10, 10] == SEG_CHANNELS["bay"]
    assert lab[0, 0] == SEG_CHANNELS["background"]


def test_extract_crop_out_size_none_stays_at_native_resolution():
    from recog.seg_dataset import extract_crop

    img = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(40, 30, 3)
    crop = extract_crop(img, (5, 5, 25, 35), out_size=None)
    assert crop.shape == (30, 20, 3)
    np.testing.assert_array_equal(crop, img[5:35, 5:25])


def test_extract_crop_pads_when_the_box_runs_off_the_image():
    from recog.seg_dataset import extract_crop

    img = np.full((10, 10, 3), 5, dtype=np.uint8)
    crop = extract_crop(img, (-2, -2, 8, 8), out_size=None)
    assert crop.shape == (10, 10, 3)
    assert (crop[0, 0] == 0).all(), "top-left corner should be zero-padded"
    assert (crop[9, 9] == 5).all(), "interior should be the real image"


# The training set itself, not recog/dev3d's 42-crop smoke corpus: dev3d
# guards the wrong thing (a corpus 8x smaller than what the model actually
# trained on) and is stale since Task 4 generated recog/dataset3d_seg.
# Reading the path from configs/segmentation.yaml rather than hardcoding
# it a second time means this test tracks whatever dataset training is
# actually pointed at.
_SEG_DS_CFG = load_yaml(ROOT / "configs" / "segmentation.yaml")["dataset"]
_COCO_PATH = ROOT / _SEG_DS_CFG["coco_path"]
_IMG_DIR = ROOT / _SEG_DS_CFG["img_dir"]


@pytest.mark.skipif(
    not _COCO_PATH.is_file(),
    reason="run recog.generate3d against configs/segmentation.yaml's "
          f"dataset.coco_path first ({_COCO_PATH} not found)")
def test_dataset_yields_crops_with_every_channel_present_somewhere():
    torch = pytest.importorskip("torch")
    from recog.seg_dataset import BaySegDataset

    ds = BaySegDataset(str(_COCO_PATH), str(_IMG_DIR), out_size=128)
    assert len(ds) > 0

    seen = set()
    for i in range(min(len(ds), 24)):
        img, lab = ds[i]
        assert img.shape == (3, 128, 128)
        assert lab.shape == (128, 128)
        assert img.dtype == torch.float32 and lab.dtype == torch.int64
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        assert int(lab.max()) < 6
        seen.update(int(v) for v in lab.unique())

    assert {0, 1, 2, 3} <= seen, f"missing channels; saw {sorted(seen)}"


# ------------------------------------------------------ per-worker rng --

def test_rng_for_worker_differs_per_worker_but_is_reproducible():
    """Without folding the worker id in, every DataLoader worker (a fork
    of the same dataset object, carrying the same self.rng state) would
    draw an IDENTICAL jitter stream the moment num_workers > 0."""
    a1 = _rng_for_worker(0, 0).uniform(size=5)
    a2 = _rng_for_worker(0, 0).uniform(size=5)
    b = _rng_for_worker(0, 1).uniform(size=5)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_rng_for_worker_none_matches_plain_default_rng():
    """worker_id=None (num_workers=0, the main process) must be byte-for-
    byte what the dataset drew before this fix - no behaviour change for
    the default config."""
    a = _rng_for_worker(5, None).uniform(size=3)
    b = np.random.default_rng(5).uniform(size=3)
    np.testing.assert_array_equal(a, b)
