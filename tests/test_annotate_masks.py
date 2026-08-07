"""RLE encoding and the COCO sidecar. No bpy, no Blender."""
from __future__ import annotations

import json

import numpy as np
import pytest

from recog.synth3d.annotate import (masks_from_index, rle_decode, rle_encode,
                                    write_coco_json)


class _Cfg:
    min_px = 80
    min_side = 6
    min_visibility = 0.25
    drop_truncated = False


SEG_IDS = {"battery": 1, "cartridge": 2, "electronics_module": 3,
           "placement_area": 4, "obstruction": 5}


def test_rle_round_trips_an_empty_mask():
    m = np.zeros((7, 5), dtype=np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_round_trips_a_full_mask():
    m = np.ones((7, 5), dtype=np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_round_trips_a_patterned_mask():
    rng = np.random.default_rng(3)
    for _ in range(50):
        m = (rng.random((13, 17)) < 0.4).astype(np.uint8)
        assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_counts_start_with_a_zero_run_when_the_first_pixel_is_set():
    """COCO's convention: counts always begin with a background run."""
    m = np.ones((2, 2), dtype=np.uint8)
    assert rle_encode(m)["counts"][0] == 0


def test_rle_is_column_major():
    """COCO RLE runs down columns. A row-major encoder decodes
    transposed everywhere downstream and the error is silent."""
    m = np.zeros((2, 3), dtype=np.uint8)
    m[0, 0] = 1                       # single pixel, top-left
    counts = rle_encode(m)["counts"]
    assert counts[:2] == [0, 1], counts


def test_masks_carry_segmentation_alongside_boxes():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:35, 5:35] = 1
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1
    a = anns[0]
    assert a["bbox_xyxy"] == [5, 5, 35, 35]
    assert a["category_id"] == 4
    assert np.array_equal(rle_decode(a["segmentation"]), (ids == 1))


def test_placement_area_is_exempt_from_the_size_filters():
    """A nearly-full cartridge has a small, thin sliver of free floor.
    That is exactly what min_px / min_side / min_visibility discard, and
    it is exactly the cartridge where remaining room matters most."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[10:13, 10:12] = 1              # 6 px, 2 px on the short side
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1, "placement_area must survive the filters"
    assert dropped == []


def test_other_classes_still_obey_the_size_filters():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[10:13, 10:12] = 1
    meta = {1: {"class": "obstruction", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns == []
    assert dropped and dropped[0]["class"] == "obstruction"


def test_a_fully_occluded_bay_yields_no_instance():
    """Zero visible pixels means no free floor, which is correct."""
    ids = np.zeros((40, 40), dtype=np.int32)
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns == []


def test_coco_json_has_the_five_categories_and_parses(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "A", "variant": "v"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    out = tmp_path / "instances_seg.json"
    write_coco_json(
        str(out),
        images=[{"id": 0, "file_name": "x.png", "width": 20, "height": 20}],
        annotations=[dict(a, image_id=0, id=i + 1)
                     for i, a in enumerate(anns)],
        seg_class_ids=SEG_IDS,
    )
    doc = json.loads(out.read_text())
    assert [c["name"] for c in doc["categories"]] == list(SEG_IDS)
    assert doc["annotations"][0]["segmentation"]["size"] == [20, 20]
    assert doc["annotations"][0]["iscrowd"] == 0
    assert doc["annotations"][0]["bbox"] == [2, 2, 16, 16]   # xywh
