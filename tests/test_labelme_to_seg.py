"""LabelMe polygon export -> COCO-RLE segmentation sidecar. No torch."""
from __future__ import annotations

import json

import numpy as np
import pytest

from recog.labelme_to_seg import (LabelmeConversionError, SEG_CLASS_IDS,
                                  _unit_id, convert_labelme_dir,
                                  convert_labelme_file, polygon_to_mask,
                                  resolve_paint_order)
from recog.synth3d.annotate import rle_decode


def _rect(x0, y0, x1, y1):
    """Vertices for a rectangle whose rasterised mask spans exactly the
    half-open ``[x0, x1) x [y0, y1)`` numpy-slice convention used
    throughout this test file - NOT ``[x0, x1]`` inclusive.

    ``cv2.fillPoly`` (like any polygon rasteriser) fills the pixels AT
    and between its integer vertices inclusively on every edge - the
    correct, intuitive behaviour for a human's clicked polygon corners,
    verified directly: a square from (2,3) to (8,6) rasterises to 7x4
    pixels, not 6x3. Placing the far corner at ``x1 - 1, y1 - 1`` (only
    at test-construction time - production code never does this) keeps
    every expected-area arithmetic in this file matching plain
    ``mask[y0:y1, x0:x1]`` slicing.
    """
    return [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]]


def _shape(label, x0, y0, x1, y1, group_id=None):
    return {"label": label, "points": _rect(x0, y0, x1, y1),
           "group_id": group_id, "shape_type": "polygon", "flags": {}}


def _write_labelme(path, shapes, file_name, width, height):
    doc = {"version": "5.4.1", "flags": {}, "shapes": shapes,
          "imagePath": file_name, "imageData": None,
          "imageHeight": height, "imageWidth": width}
    path.write_text(json.dumps(doc), encoding="utf-8")


def _blank_image(path, width, height):
    from PIL import Image
    Image.new("RGB", (width, height), (0, 0, 0)).save(path)


# ---------------------------------------------------------- polygon_to_mask
def test_polygon_to_mask_rasterises_a_simple_rectangle():
    mask = polygon_to_mask(_rect(2, 3, 8, 6), height=10, width=10)
    assert mask.shape == (10, 10)
    assert mask[3:6, 2:8].all()
    assert not mask[0, 0]


def test_polygon_to_mask_returns_empty_for_fewer_than_three_points():
    mask = polygon_to_mask([[1, 1], [2, 2]], height=10, width=10)
    assert not mask.any()


def test_polygon_to_mask_rounds_rather_than_truncates_float_vertices():
    """int() truncation would bias every polygon half a pixel toward the
    origin; round() must not lose the far edge of a fractional box."""
    mask = polygon_to_mask([[0.6, 0.6], [4.6, 0.6], [4.6, 4.6], [0.6, 4.6]],
                           height=10, width=10)
    assert mask[1:5, 1:5].all()


# ------------------------------------------------------- resolve_paint_order
def _s(index, label, x0, y0, x1, y1, h, w):
    return {"shape_index": index, "label": label,
           "mask": polygon_to_mask(_rect(x0, y0, x1, y1), h, w),
           "group_id": None}


def test_resolve_paint_order_lets_a_generous_earlier_class_be_reclaimed():
    """cartridge (paint-order first) drawn generously over the whole
    frame must lose the pixels placement_area (later) legitimately
    owns - the mechanism that lets an annotator skip pixel-perfect
    boundary tracing between an earlier and a later class."""
    h = w = 20
    shapes = [
        _s(0, "cartridge", 0, 0, 20, 20, h, w),          # generous, whole frame
        _s(1, "placement_area", 5, 5, 15, 15, h, w),
    ]
    kept, dropped = resolve_paint_order(shapes, h, w)
    assert dropped == []
    by_class = {s["label"]: s["mask"] for s in kept}
    assert (by_class["cartridge"] & by_class["placement_area"]).sum() == 0
    assert by_class["placement_area"].sum() == 100          # untouched, 10x10
    assert by_class["cartridge"].sum() == 400 - 100          # frame minus the hole


def test_resolve_paint_order_does_not_rescue_a_later_class_drawn_too_generously():
    """The fix only runs one direction: a LATER class (e.g. battery)
    drawn over an earlier class's true territory wins outright - this is
    why the protocol asks for precise battery/obstruction tracing."""
    h = w = 20
    shapes = [
        _s(0, "placement_area", 0, 0, 20, 20, h, w),
        _s(1, "battery", 5, 5, 15, 15, h, w),   # generous into placement_area
    ]
    kept, _dropped = resolve_paint_order(shapes, h, w)
    by_class = {s["label"]: s["mask"] for s in kept}
    assert by_class["battery"].sum() == 100
    assert by_class["placement_area"].sum() == 400 - 100


def test_resolve_paint_order_drops_a_shape_fully_overlapped_by_a_later_class():
    h = w = 20
    shapes = [
        _s(0, "battery", 5, 5, 15, 15, h, w),
        _s(1, "placement_area", 6, 6, 10, 10, h, w),   # entirely inside battery
    ]
    kept, dropped = resolve_paint_order(shapes, h, w)
    assert [s["label"] for s in kept] == ["battery"]
    assert len(dropped) == 1
    assert dropped[0]["class"] == "placement_area"
    assert dropped[0]["reason"] == "fully_overlapped_by_paint_order"


def test_resolve_paint_order_breaks_same_class_ties_by_file_order():
    """Paint order has no opinion between two instances of ONE class;
    the shape listed first in the file keeps a contested pixel."""
    h = w = 20
    shapes = [
        _s(0, "obstruction", 2, 2, 10, 10, h, w),
        _s(1, "obstruction", 6, 6, 14, 14, h, w),   # overlaps shape 0's corner
    ]
    kept, dropped = resolve_paint_order(shapes, h, w)
    assert dropped == []
    first, second = (s for s in kept if s["shape_index"] == 0), \
        (s for s in kept if s["shape_index"] == 1)
    first_mask = next(first)["mask"]
    second_mask = next(second)["mask"]
    assert (first_mask & second_mask).sum() == 0
    # shape 0's full 8x8=64 square is untouched; shape 1 loses the overlap
    assert first_mask.sum() == 64
    assert second_mask.sum() == 64 - (10 - 6) * (10 - 6)


def test_resolve_paint_order_is_a_no_op_for_disjoint_shapes():
    h = w = 20
    shapes = [
        _s(0, "cartridge", 0, 0, 5, 5, h, w),
        _s(1, "battery", 10, 10, 15, 15, h, w),
    ]
    kept, dropped = resolve_paint_order(shapes, h, w)
    assert dropped == []
    assert {s["label"]: int(s["mask"].sum()) for s in kept} == {
        "cartridge": 25, "battery": 25}


# ------------------------------------------------------------------ _unit_id
def test_unit_id_groups_shapes_sharing_a_group_id():
    assert _unit_id("img1", 3, 0) == _unit_id("img1", 3, 7)


def test_unit_id_gives_each_ungrouped_shape_its_own_id():
    a = _unit_id("img1", None, 0)
    b = _unit_id("img1", None, 1)
    assert a != b


def test_unit_id_does_not_collide_across_images():
    assert _unit_id("img1", 1, 0) != _unit_id("img2", 1, 0)


# ------------------------------------------------------- convert_labelme_file
def test_convert_labelme_file_round_trips_rle_against_direct_rasterisation(
    tmp_path,
):
    """polygon -> RLE -> decode must equal directly rasterising the same
    polygon, for shapes paint order never touches - genuinely disjoint,
    not merely a different class: an overlapping pair would legitimately
    be clipped by resolve_paint_order, which is covered separately
    above and would make this comparison meaningless."""
    w, h = 40, 30
    _blank_image(tmp_path / "photo.jpg", w, h)
    shapes = [_shape("cartridge", 2, 2, 15, 15),
             _shape("battery", 20, 18, 30, 25)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", w, h)

    image_record, anns, dropped = convert_labelme_file(
        tmp_path / "photo.json", tmp_path)
    assert dropped == []
    assert image_record == {"file_name": "photo.jpg", "width": w, "height": h}

    by_class = {a["class"]: a for a in anns}
    direct_cartridge = polygon_to_mask(_rect(2, 2, 15, 15), h, w)
    direct_battery = polygon_to_mask(_rect(20, 18, 30, 25), h, w)
    np.testing.assert_array_equal(
        rle_decode(by_class["cartridge"]["segmentation"]), direct_cartridge)
    np.testing.assert_array_equal(
        rle_decode(by_class["battery"]["segmentation"]), direct_battery)
    assert by_class["cartridge"]["area"] == int(direct_cartridge.sum())
    assert by_class["battery"]["area"] == int(direct_battery.sum())


def test_convert_labelme_file_reads_actual_dimensions_from_images_dir(tmp_path):
    w, h = 50, 60
    _blank_image(tmp_path / "photo.jpg", w, h)
    shapes = [_shape("cartridge", 0, 0, 10, 10)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", w, h)

    image_record, _anns, _dropped = convert_labelme_file(
        tmp_path / "photo.json", tmp_path)
    assert (image_record["width"], image_record["height"]) == (w, h)


def test_convert_labelme_file_uses_the_file_when_json_omits_dimensions(
    tmp_path,
):
    w, h = 50, 60
    _blank_image(tmp_path / "photo.jpg", w, h)
    doc = {"version": "5.4.1", "flags": {},
          "shapes": [_shape("cartridge", 0, 0, 10, 10)],
          "imagePath": "photo.jpg", "imageData": None,
          "imageHeight": None, "imageWidth": None}
    (tmp_path / "photo.json").write_text(json.dumps(doc), encoding="utf-8")

    image_record, _anns, _dropped = convert_labelme_file(
        tmp_path / "photo.json", tmp_path)
    assert (image_record["width"], image_record["height"]) == (w, h)


def test_convert_labelme_file_errors_on_dimension_mismatch(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 50, 60)
    shapes = [_shape("cartridge", 0, 0, 10, 10)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 999, 999)
    with pytest.raises(LabelmeConversionError, match="does not match"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_errors_when_the_image_file_is_missing(tmp_path):
    shapes = [_shape("cartridge", 0, 0, 10, 10)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 50, 60)
    with pytest.raises(LabelmeConversionError, match="not found"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_errors_on_unknown_label(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [_shape("power_bank", 0, 0, 10, 10)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    with pytest.raises(LabelmeConversionError, match="power_bank"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_label_matching_is_case_insensitive(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [_shape("Battery", 0, 0, 10, 10)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    _image_record, anns, _dropped = convert_labelme_file(
        tmp_path / "photo.json", tmp_path)
    assert anns[0]["class"] == "battery"


def test_convert_labelme_file_errors_on_non_polygon_shape_type(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [{"label": "battery", "points": _rect(0, 0, 10, 10),
              "group_id": None, "shape_type": "rectangle", "flags": {}}]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    with pytest.raises(LabelmeConversionError, match="rectangle"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_errors_on_a_degenerate_polygon(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [{"label": "battery", "points": [[1, 1], [2, 2]],
              "group_id": None, "shape_type": "polygon", "flags": {}}]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    with pytest.raises(LabelmeConversionError, match="vertices"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_errors_on_a_zero_pixel_polygon(tmp_path):
    """A polygon entirely outside the image bounds rasterises to nothing."""
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [{"label": "battery",
              "points": [[100, 100], [105, 100], [105, 105], [100, 105]],
              "group_id": None, "shape_type": "polygon", "flags": {}}]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    with pytest.raises(LabelmeConversionError, match="zero"):
        convert_labelme_file(tmp_path / "photo.json", tmp_path)


def test_convert_labelme_file_reports_a_dropped_shape(tmp_path):
    _blank_image(tmp_path / "photo.jpg", 20, 20)
    shapes = [_shape("battery", 2, 2, 18, 18, group_id=1),
             _shape("placement_area", 4, 4, 8, 8, group_id=1)]
    _write_labelme(tmp_path / "photo.json", shapes, "photo.jpg", 20, 20)
    _image_record, anns, dropped = convert_labelme_file(
        tmp_path / "photo.json", tmp_path)
    assert [a["class"] for a in anns] == ["battery"]
    assert len(dropped) == 1 and dropped[0]["class"] == "placement_area"


# -------------------------------------------------------- convert_labelme_dir
def test_convert_labelme_dir_writes_a_coco_json_seg_dataset_can_read(tmp_path):
    from recog.seg_dataset import SEG_CHANNELS, rasterise_crop

    w, h = 30, 30
    _blank_image(tmp_path / "photo.jpg", w, h)
    shapes = [_shape("cartridge", 0, 0, 30, 30, group_id=1),
             _shape("placement_area", 5, 5, 25, 25, group_id=1)]
    lm_dir = tmp_path / "labelme"
    lm_dir.mkdir()
    _write_labelme(lm_dir / "photo.json", shapes, "photo.jpg", w, h)

    out = tmp_path / "annotations" / "instances_seg.json"
    n_images, n_anns, dropped = convert_labelme_dir(lm_dir, out, tmp_path)
    assert (n_images, n_anns, dropped) == (1, 2, [])

    doc = json.loads(out.read_text())
    assert [c["name"] for c in doc["categories"]] == list(SEG_CLASS_IDS)
    anns = [dict(a, class_=None) for a in doc["annotations"]]
    inv = {v: k for k, v in SEG_CLASS_IDS.items()}
    for a in anns:
        a["class"] = inv[a["category_id"]]

    label = rasterise_crop(
        [{"class": a["class"], "segmentation": a["segmentation"]}
        for a in anns], (0, 0, w, h))
    assert label[15, 15] == SEG_CHANNELS["bay"]      # inside placement_area
    assert label[1, 1] == SEG_CHANNELS["cartridge"]  # cartridge-only rim


def test_convert_labelme_dir_assigns_sequential_ids_across_files(tmp_path):
    lm_dir = tmp_path / "labelme"
    lm_dir.mkdir()
    for name in ("a", "b"):
        _blank_image(tmp_path / f"{name}.jpg", 10, 10)
        _write_labelme(lm_dir / f"{name}.json",
                       [_shape("battery", 1, 1, 5, 5)], f"{name}.jpg", 10, 10)

    out = tmp_path / "annotations" / "out.json"
    n_images, n_anns, _dropped = convert_labelme_dir(lm_dir, out, tmp_path)
    assert (n_images, n_anns) == (2, 2)

    doc = json.loads(out.read_text())
    assert [im["id"] for im in doc["images"]] == [1, 2]
    assert [a["id"] for a in doc["annotations"]] == [1, 2]
    assert [a["image_id"] for a in doc["annotations"]] == [1, 2]


def test_convert_labelme_dir_errors_when_the_directory_has_no_json(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LabelmeConversionError, match="no .json"):
        convert_labelme_dir(empty, tmp_path / "out.json", tmp_path)
