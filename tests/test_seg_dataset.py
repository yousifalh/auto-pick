"""Per-ROI crop dataset built from the COCO-RLE sidecar."""
from __future__ import annotations

import os

import numpy as np
import pytest

from recog.seg_dataset import (SEG_CHANNELS, jitter_box, rasterise_crop)
from recog.synth3d.annotate import rle_encode


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


DEV = os.path.join(os.path.dirname(__file__), "..", "recog", "dev3d")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DEV, "instances_seg.json")),
    reason="run Plan B Task 7 first to generate recog/dev3d")
def test_dataset_yields_crops_with_every_channel_present_somewhere():
    torch = pytest.importorskip("torch")
    from recog.seg_dataset import BaySegDataset

    ds = BaySegDataset(os.path.join(DEV, "instances_seg.json"),
                       os.path.join(DEV, "images"), out_size=128)
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
