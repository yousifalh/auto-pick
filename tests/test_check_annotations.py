"""Validation of converted real-photo segmentation annotations. No torch."""
from __future__ import annotations

import json

import numpy as np

from recog.check_annotations import (CANONICAL_CATEGORY_IDS, Issue,
                                     check_category_schema, check_image_file,
                                     check_orphan_image_files,
                                     default_images_dir, format_report,
                                     main, pairwise_class_overlaps,
                                     same_class_instance_overlaps,
                                     validate_dir, validate_file)
from recog.synth3d.annotate import rle_encode
from recog.labelme_to_seg import convert_labelme_dir


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _ann(ann_id, image_id, cls, mask, unit_id=None):
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return {"id": ann_id, "image_id": image_id,
           "category_id": CANONICAL_CATEGORY_IDS[cls],
           "segmentation": rle_encode(mask),
           "bbox": [x0, y0, x1 - x0, y1 - y0],
           "area": int(mask.sum()), "iscrowd": 0, "unit_id": unit_id}


def _doc(images, annotations):
    return {
        "info": {}, "licenses": [],
        "categories": [{"id": i, "name": n, "supercategory": ""}
                       for n, i in sorted(CANONICAL_CATEGORY_IDS.items(),
                                          key=lambda kv: kv[1])],
        "images": images, "annotations": annotations,
    }


def _write_doc(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


# --------------------------------------------------- pairwise_class_overlaps
def test_pairwise_class_overlaps_reports_all_ten_pairs_even_when_zero():
    h = w = 20
    battery = _mask(h, w, 2, 6, 2, 6)
    cartridge = _mask(h, w, 10, 16, 10, 16)
    instances = [("battery", battery, {}), ("cartridge", cartridge, {})]
    overlaps = pairwise_class_overlaps(instances)
    assert len(overlaps) == 10
    assert overlaps[("battery", "placement_area")] == 0
    assert all(n == 0 for n in overlaps.values())


def test_pairwise_class_overlaps_detects_a_real_overlap():
    h = w = 20
    bay = _mask(h, w, 0, 20, 0, 20)
    battery = _mask(h, w, 5, 10, 5, 10)   # entirely inside bay: 25 px overlap
    overlaps = pairwise_class_overlaps(
        [("placement_area", bay, {}), ("battery", battery, {})])
    assert overlaps[("battery", "placement_area")] == 25


def test_pairwise_class_overlaps_unions_same_class_instances_first():
    h = w = 20
    bay1 = _mask(h, w, 0, 10, 0, 20)
    bay2 = _mask(h, w, 10, 20, 0, 20)
    battery = _mask(h, w, 8, 12, 0, 20)   # straddles both bay pieces
    overlaps = pairwise_class_overlaps([
        ("placement_area", bay1, {}), ("placement_area", bay2, {}),
        ("battery", battery, {}),
    ])
    assert overlaps[("battery", "placement_area")] == 4 * 20  # rows 8-12 x 20


# ----------------------------------------------- same_class_instance_overlaps
def test_same_class_instance_overlaps_detects_duplicate_polygons():
    h = w = 20
    a = _mask(h, w, 2, 10, 2, 10)
    b = _mask(h, w, 6, 14, 6, 14)   # overlaps a in [6:10, 6:10] = 16 px
    out = same_class_instance_overlaps(
        [("obstruction", a, {}), ("obstruction", b, {})])
    assert out == {"obstruction": 16}


def test_same_class_instance_overlaps_is_empty_for_disjoint_instances():
    h = w = 20
    a = _mask(h, w, 0, 5, 0, 5)
    b = _mask(h, w, 10, 15, 10, 15)
    out = same_class_instance_overlaps(
        [("obstruction", a, {}), ("obstruction", b, {})])
    assert out == {}


# --------------------------------------------------------- category schema
def test_check_category_schema_passes_the_canonical_vocabulary():
    doc = _doc([], [])
    assert check_category_schema(doc, "x.json") == []


def test_check_category_schema_flags_a_mismatched_vocabulary():
    doc = _doc([], [])
    doc["categories"] = [{"id": 1, "name": "battery", "supercategory": ""}]
    issues = check_category_schema(doc, "x.json")
    assert len(issues) == 1 and issues[0].severity == "ERROR"


# -------------------------------------------------------------- validate_file
def test_validate_file_flags_an_image_with_zero_annotations(tmp_path):
    doc = _doc([{"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}], [])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, totals = validate_file(p, None, max_overlap_px=0)
    assert any(it.check == "no_annotations" for it in issues)


def test_validate_file_flags_a_degenerate_rle(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    ann["segmentation"]["counts"] = [400, 0]   # all background: decodes empty
    doc = _doc([img], [ann])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, _totals = validate_file(p, None, max_overlap_px=0)
    assert any(it.check == "degenerate_polygon" for it in issues)


def test_validate_file_flags_an_area_mismatch(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    ann["area"] = 999
    doc = _doc([img], [ann])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, _totals = validate_file(p, None, max_overlap_px=0)
    assert any(it.check == "degenerate_polygon" and "area=999" in it.detail
              for it in issues)


def test_validate_file_flags_a_bbox_mismatch(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    ann["bbox"] = [0, 0, 1, 1]
    doc = _doc([img], [ann])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, _totals = validate_file(p, None, max_overlap_px=0)
    assert any(it.check == "degenerate_polygon" and "bbox=" in it.detail
              for it in issues)


def test_validate_file_flags_a_size_mismatch_and_excludes_it_from_overlaps(
    tmp_path,
):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    ann["segmentation"]["size"] = [40, 40]   # disagrees with the image record
    doc = _doc([img], [ann])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, totals = validate_file(p, None, max_overlap_px=0)
    assert any(it.check == "size_mismatch" for it in issues)
    assert totals["battery"] == 0   # excluded, not counted


def test_validate_file_reports_the_high_risk_pair_even_when_clean(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    doc = _doc([img], [ann])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, _totals = validate_file(p, None, max_overlap_px=0)
    hi = [it for it in issues if it.check == "placement_area_battery_overlap"]
    assert len(hi) == 1 and hi[0].severity == "OK"


def test_validate_file_flags_placement_area_battery_overlap(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    bay = _mask(20, 20, 0, 20, 0, 20)
    cell = _mask(20, 20, 5, 10, 5, 10)
    doc = _doc([img], [_ann(1, 1, "placement_area", bay),
                       _ann(2, 1, "battery", cell)])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    issues, _totals = validate_file(p, None, max_overlap_px=0)
    hi = [it for it in issues if it.check == "placement_area_battery_overlap"]
    assert hi[0].severity == "ERROR"
    cross = [it for it in issues if it.check == "cross_class_overlap"]
    assert any("placement_area" in it.detail and "battery" in it.detail
              for it in cross)


def test_validate_file_max_overlap_px_tolerance(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    bay = _mask(20, 20, 0, 20, 0, 20)
    cell = _mask(20, 20, 5, 8, 5, 6)   # 3 px overlap
    doc = _doc([img], [_ann(1, 1, "placement_area", bay),
                       _ann(2, 1, "battery", cell)])
    p = tmp_path / "annotations" / "instances_seg.json"
    _write_doc(p, doc)
    strict, _ = validate_file(p, None, max_overlap_px=0)
    tolerant, _ = validate_file(p, None, max_overlap_px=5)
    assert any(it.severity == "ERROR" and it.check == "cross_class_overlap"
              for it in strict)
    assert not any(it.severity == "ERROR" and it.check == "cross_class_overlap"
                  for it in tolerant)


# -------------------------------------------------------------- image checks
def test_check_image_file_flags_a_dimension_mismatch(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (10, 10)).save(images_dir / "a.jpg")
    record = {"file_name": "a.jpg", "width": 999, "height": 999}
    issues = check_image_file(record, images_dir, "src.json")
    assert issues and issues[0].check == "dimension_mismatch"


def test_check_image_file_flags_a_missing_file(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    record = {"file_name": "missing.jpg", "width": 10, "height": 10}
    issues = check_image_file(record, images_dir, "src.json")
    assert issues and issues[0].check == "missing_image_file"


def test_check_image_file_is_a_no_op_without_images_dir():
    record = {"file_name": "a.jpg", "width": 10, "height": 10}
    assert check_image_file(record, None, "src.json") == []


def test_check_orphan_image_files_flags_a_photo_with_no_record(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_bytes(b"\x00")
    (images_dir / "b.jpg").write_bytes(b"\x00")
    issues = check_orphan_image_files(["a.jpg"], images_dir, "src.json")
    assert len(issues) == 1
    assert issues[0].check == "unannotated_photo" and issues[0].image == "b.jpg"


# ---------------------------------------------------------------- validate_dir
def test_validate_dir_flags_a_class_with_zero_instances_as_a_warning(tmp_path):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    ann = _ann(1, 1, "battery", _mask(20, 20, 2, 6, 2, 6))
    _write_doc(tmp_path / "annotations" / "x.json", _doc([img], [ann]))
    issues, totals, files = validate_dir(tmp_path, images_dir=None,
                                         max_overlap_px=0)
    warn = [it for it in issues
           if it.check == "empty_class" and "obstruction" in it.detail]
    assert len(warn) == 1 and warn[0].severity == "WARNING"
    assert totals["obstruction"] == 0
    assert len(files) == 1


def test_validate_dir_errors_when_no_annotation_files_exist(tmp_path):
    issues, _totals, files = validate_dir(tmp_path, images_dir=None,
                                          max_overlap_px=0)
    assert files == []
    assert issues and issues[0].check == "no_annotation_files"


def test_default_images_dir_falls_back_to_none_when_absent(tmp_path):
    assert default_images_dir(tmp_path) is None
    (tmp_path / "images").mkdir()
    assert default_images_dir(tmp_path) == tmp_path / "images"


# ---------------------------------------------------------------- format/CLI
def test_format_report_says_pass_with_no_errors():
    text = format_report([], {c: 0 for c in CANONICAL_CATEGORY_IDS},
                         [], "d", 0)
    assert "RESULT: PASS" in text


def test_format_report_says_fail_with_an_error():
    issues = [Issue("ERROR", "x", "boom", source="s", image="a.jpg")]
    text = format_report(issues, {c: 0 for c in CANONICAL_CATEGORY_IDS},
                         [], "d", 0)
    assert "RESULT: FAIL" in text
    assert "boom" in text


def test_main_exits_nonzero_on_a_clean_directory_with_overlap(tmp_path, capsys):
    img = {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}
    bay = _mask(20, 20, 0, 20, 0, 20)
    cell = _mask(20, 20, 5, 10, 5, 10)
    _write_doc(tmp_path / "annotations" / "x.json",
              _doc([img], [_ann(1, 1, "placement_area", bay),
                          _ann(2, 1, "battery", cell)]))
    rc = main([str(tmp_path)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


# ------------------------------------------------- end-to-end with the converter
def test_a_cleanly_converted_labelme_export_passes_validation(tmp_path):
    """The converter's paint-order resolution (recog/labelme_to_seg.py)
    must produce output this validator accepts with zero errors even
    when the source polygons overlapped generously - that is the whole
    point of resolving paint order at conversion time."""
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    w, h = 30, 30
    Image.new("RGB", (w, h)).save(images_dir / "photo.jpg")

    def rect(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    shapes = [
        {"label": "cartridge", "points": rect(0, 0, 30, 30), "group_id": 1,
        "shape_type": "polygon", "flags": {}},
        {"label": "placement_area", "points": rect(3, 3, 27, 27),
        "group_id": 1, "shape_type": "polygon", "flags": {}},
        {"label": "obstruction", "points": rect(20, 20, 25, 25),
        "group_id": 1, "shape_type": "polygon", "flags": {}},
    ]
    lm_dir = tmp_path / "labelme"
    lm_dir.mkdir()
    doc = {"version": "5.4.1", "flags": {}, "shapes": shapes,
          "imagePath": "photo.jpg", "imageData": None,
          "imageHeight": h, "imageWidth": w}
    (lm_dir / "photo.json").write_text(json.dumps(doc), encoding="utf-8")

    out_dir = tmp_path / "annotations"
    convert_labelme_dir(lm_dir, out_dir / "instances_seg.json", images_dir)

    issues, _totals, _files = validate_dir(tmp_path, images_dir,
                                           max_overlap_px=0)
    errors = [it for it in issues if it.severity == "ERROR"]
    assert errors == [], errors
