"""RLE encoding and the COCO sidecar. No bpy, no Blender."""
from __future__ import annotations

import json

import numpy as np
import pytest

from recog.synth3d.annotate import (boxes_from_mask, masks_from_index,
                                    rle_decode, rle_encode, write_coco_json)


class _Cfg:
    min_px = 80
    min_side = 6
    min_visibility = 0.25
    drop_truncated = False
    max_aspect = 4.0


SEG_IDS = {"battery": 1, "cartridge": 2, "electronics_module": 3,
           "placement_area": 4, "obstruction": 5}
VOC_IDS = {"battery": 1, "cartridge": 2}


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


# ------------------------------------------------------- per-class filter
# exemptions (fix round 1): min_px=500 in production is a CELL-sized
# threshold and silently discarded ~85% of real obstruction instances
# (adhesive/foam/tape/label are an order of magnitude smaller than a
# battery), while max_aspect=4.0 is tuned to battery cells and does not
# transfer to tape, which bay.sample_obstructions draws with an aspect
# ratio of ~4-19 BY CONSTRUCTION. Both are class-specific judgements, not a
# blanket exemption, so each is tested in isolation below. -------------- #

def test_obstruction_is_exempt_from_min_px_but_not_min_side():
    """A real adhesive/foam/tape/label instance is an order of magnitude
    smaller than min_px=500 (tuned for a battery cell), but must still be
    legible: min_side stays in force."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[10:16, 10:18] = 1              # 48 px, area < min_px=80, sides ok
    meta = {1: {"class": "obstruction", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1, "obstruction below min_px must still survive"
    assert dropped == []

    ids2 = np.zeros((40, 40), dtype=np.int32)
    ids2[10:13, 10:12] = 1             # 6 px, 2 px on the short side
    anns2, dropped2 = masks_from_index(ids2, meta, SEG_IDS, _Cfg())
    assert anns2 == [], "obstruction thinner than min_side must still drop"
    assert dropped2 and dropped2[0]["reason"].startswith("side<")


def test_obstruction_is_exempt_from_max_aspect():
    """bay.sample_obstructions draws a tape strip's width from 5-12% of the
    bay's short edge and its height from 50-95% of the same edge - an
    aspect ratio of ~4-19 by construction. max_aspect must not drop it."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[2:38, 10:16] = 1               # 36x6, aspect 6 > max_aspect=4.0
    meta = {1: {"class": "obstruction", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1, "a tape-shaped obstruction must survive max_aspect"
    assert dropped == []


def test_battery_and_cartridge_still_obey_max_aspect():
    """battery/cartridge get NO exemptions: they must filter identically to
    boxes_from_mask, since SEG_CLASSES ids 1/2 mean the same thing as
    CLASSES ids 1/2."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[2:32, 10:16] = 1               # 30x6, aspect 5 > max_aspect=4.0
    for cls in ("battery", "cartridge"):
        meta = {1: {"class": cls, "asset": "A", "variant": "v"}}
        anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
        assert anns == [], f"{cls} must still be dropped by max_aspect"
        assert dropped and dropped[0]["reason"].startswith("aspect>")


def test_voc_and_seg_paths_agree_on_shared_classes():
    """`_FILTER_EXEMPT` lives only in `masks_from_index`. A future entry
    added there for `battery` or `cartridge` - the two classes SEG_CLASSES
    shares with the VOC vocabulary - would silently desynchronise the VOC
    and sidecar outputs for those classes with nothing to catch it. This
    calls `boxes_from_mask` and `masks_from_index` on the SAME ids/id_meta
    and asserts the kept pass_index sets are identical.

    Three instances, chosen so every filter that could disagree actually
    exercises something: pid 1 (battery) passes min_px but fails
    max_aspect (30x6, aspect 5 > 4.0); pid 2 (cartridge) passes everything;
    pid 3 (battery) is a 3x3=9px sliver that fails min_px (80) outright.
    """
    ids = np.zeros((60, 60), dtype=np.int32)
    ids[2:32, 2:8] = 1      # 30x6 -> aspect 5, over max_aspect=4.0
    ids[10:50, 20:50] = 2   # 40x30 -> clean, must survive
    ids[0:3, 40:43] = 3     # 3x3=9px -> under min_px=80
    id_meta = {
        1: {"class": "battery", "asset": "A", "variant": "v"},
        2: {"class": "cartridge", "asset": "A", "variant": "v"},
        3: {"class": "battery", "asset": "A", "variant": "v"},
    }
    voc_anns, voc_dropped = boxes_from_mask(ids, id_meta, VOC_IDS, _Cfg())
    seg_anns, seg_dropped = masks_from_index(ids, id_meta, SEG_IDS, _Cfg())

    voc_kept = {a["pass_index"] for a in voc_anns}
    seg_kept = {a["pass_index"] for a in seg_anns
               if a["class"] in ("battery", "cartridge")}
    assert voc_kept == seg_kept == {2}, (voc_kept, seg_kept)

    voc_reasons = {d["pass_index"]: d["reason"] for d in voc_dropped}
    seg_reasons = {d["pass_index"]: d["reason"] for d in seg_dropped
                  if d["class"] in ("battery", "cartridge")}
    assert voc_reasons == seg_reasons == {
        1: "aspect>4.0", 3: "visible_px<80"}, (voc_reasons, seg_reasons)


def test_masks_from_index_carries_unit_id_through_from_id_meta():
    """scene.build stamps every id_meta entry with `unit_id`; masks_from_index
    must copy it into the annotation unchanged - it is what lets a consumer
    assemble a crop per physical unit instead of per instance."""
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "A", "variant": "v",
                "unit_id": "item0"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns[0]["unit_id"] == "item0"


def test_masks_from_index_defaults_unit_id_to_none_when_absent():
    """Every annotation must carry the key (even if a caller - like the
    other tests in this file - never set it), so a consumer can rely on
    a["unit_id"] existing rather than needing .get() itself."""
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "A", "variant": "v"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert "unit_id" in anns[0]
    assert anns[0]["unit_id"] is None


def test_write_coco_json_round_trips_unit_id(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "A", "variant": "v",
                "unit_id": "item3"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    out = tmp_path / "instances_seg.json"
    write_coco_json(
        str(out),
        images=[{"id": 0, "file_name": "x.png", "width": 20, "height": 20}],
        annotations=[dict(a, image_id=0, id=1) for a in anns],
        seg_class_ids=SEG_IDS,
    )
    doc = json.loads(out.read_text())
    assert doc["annotations"][0]["unit_id"] == "item3"


def test_annotations_sharing_a_unit_id_are_distinguishable_from_a_loose_one():
    """The contract Task 9 exists for: a cartridge, its module and its bay
    share a unit_id; a loose cell elsewhere in the same image does not."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[2:12, 2:12] = 1        # cartridge shell, unit "item0" (100 px)
    ids[2:12, 14:24] = 2       # its electronics_module, unit "item0" (100 px)
    ids[28:38, 28:38] = 3      # a loose battery cell, no unit (100 px)
    meta = {
        1: {"class": "cartridge", "asset": "A", "variant": "v",
            "unit_id": "item0"},
        2: {"class": "electronics_module", "asset": "A", "variant": "v",
            "unit_id": "item0"},
        3: {"class": "battery", "asset": "A", "variant": "v",
            "unit_id": "solo3"},
    }
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    by_class = {a["class"]: a["unit_id"] for a in anns}
    assert by_class["cartridge"] == by_class["electronics_module"] == "item0"
    assert by_class["battery"] == "solo3"
    assert by_class["battery"] != by_class["cartridge"]


def test_electronics_module_still_obeys_max_aspect():
    """electronics_module's true (untruncated) shape is fixed by the CAD -
    catalog.json's module_bay_mm never exceeds aspect 3.05 - so max_aspect
    only ever fires on a genuine frame-truncation sliver here."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[2:32, 10:16] = 1               # 30x6, aspect 5 > max_aspect=4.0
    meta = {1: {"class": "electronics_module", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns == []
    assert dropped and dropped[0]["reason"].startswith("aspect>")
