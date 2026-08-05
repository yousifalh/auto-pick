"""Recognition dataset tests.

Uses the synthetic-dataset generator to create a tiny corpus and then
exercises ``parse_voc_xml`` and ``BatteryCartridgeDataset`` end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recog.dataset import (
    CLASS_MAP, BatteryCartridgeDataset, RealPhotoDataset, VocAnnotation,
    collate_fn, parse_coco_json, parse_voc_xml,
)


# ---------- Helpers --------------------------------------------------------

VOC_SAMPLE = """\
<annotation>
  <filename>img_0000.png</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>battery</name>
    <bndbox>
      <xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>50</ymax>
    </bndbox>
  </object>
  <object>
    <name>cartridge</name>
    <bndbox>
      <xmin>100</xmin><ymin>120</ymin><xmax>400</xmax><ymax>350</ymax>
    </bndbox>
  </object>
  <object>
    <name>unknown_label</name>
    <bndbox>
      <xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax>
    </bndbox>
  </object>
  <object>
    <name>battery</name>
    <bndbox>
      <xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>10</ymax>
    </bndbox>
  </object>
</annotation>
"""


def test_parse_voc_xml_basic(tmp_path: Path):
    f = tmp_path / "s.xml"
    f.write_text(VOC_SAMPLE)
    ann = parse_voc_xml(f)
    assert ann.filename == "img_0000.png"
    assert ann.width == 640
    assert ann.height == 480
    # 2 known + 1 unknown (skipped) + 1 degenerate (skipped)
    assert len(ann.boxes) == 2
    assert ann.labels == [CLASS_MAP["battery"], CLASS_MAP["cartridge"]]
    assert ann.boxes[0] == (10.0, 20.0, 30.0, 50.0)


def test_parse_voc_xml_no_objects(tmp_path: Path):
    f = tmp_path / "empty.xml"
    f.write_text(
        "<annotation><filename>e.png</filename>"
        "<size><width>10</width><height>10</height></size></annotation>")
    ann = parse_voc_xml(f)
    assert ann.boxes == []
    assert ann.labels == []


# ---------- Full dataset --------------------------------------------------

def test_battery_cartridge_dataset_roundtrip(tmp_path: Path):
    # Generate two images + annotations via the synth-dataset module.
    try:
        from recog.synth_dataset import generate_dataset
    except Exception as e:  # pragma: no cover
        pytest.skip(f"synth_dataset unavailable: {e}")

    # synth_dataset needs images big enough to accommodate its default
    # cartridge dimensions (up to 440 px wide, 320 px tall) plus margins.
    generate_dataset(str(tmp_path), n=2, seed=1, size=(480, 640))
    ds = BatteryCartridgeDataset(
        img_dir=str(tmp_path / "images"),
        ann_dir=str(tmp_path / "annotations"),
    )
    assert len(ds) == 2
    img, tgt = ds[0]
    # Without torch it returns numpy
    if isinstance(img, np.ndarray):
        assert img.ndim == 3
        assert "boxes" in tgt and "labels" in tgt
    else:  # torch tensor
        assert img.ndim == 3


def test_dataset_handles_missing_xml(tmp_path: Path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    # Write a single black PNG without any XML
    from PIL import Image
    Image.new("RGB", (40, 40), (0, 0, 0)).save(img_dir / "a.png")

    ds = BatteryCartridgeDataset(str(img_dir), str(ann_dir))
    assert len(ds) == 1
    _img, tgt = ds[0]
    # No annotation → empty boxes
    boxes = tgt["boxes"]
    assert len(boxes) == 0


def test_collate_fn_shape():
    samples = [("a", {"boxes": []}), ("b", {"boxes": []})]
    imgs, tgts = collate_fn(samples)
    assert imgs == ("a", "b")
    assert len(tgts) == 2


def test_class_map_invariants():
    assert CLASS_MAP["background"] == 0
    assert CLASS_MAP["battery"] == 1
    assert CLASS_MAP["cartridge"] == 2


# ---------- COCO reader ---------------------------------------------------

REALTEST_ANN = (
    Path(__file__).resolve().parents[1]
    / "recog" / "realtest" / "annotations" / "instances_default.json"
)


def _coco_doc(categories=None, annotations=None, images=None):
    """A minimal CVAT-shaped COCO document."""
    return {
        "licenses": [],
        "info": {},
        "categories": categories if categories is not None else [
            {"id": 1, "name": "Battery", "supercategory": ""},
            {"id": 2, "name": "Cartridge", "supercategory": ""},
        ],
        "images": images if images is not None else [
            {"id": 7, "width": 3024, "height": 4032,
             "file_name": "IMG_0001.jpg"},
        ],
        "annotations": annotations if annotations is not None else [],
    }


def _ann(ann_id, image_id, category_id, bbox):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [],
        "area": bbox[2] * bbox[3],
        "bbox": list(bbox),
        "iscrowd": 0,
        "attributes": {"occluded": False, "rotation": 0.0},
    }


def _write_coco(tmp_path: Path, doc) -> Path:
    import json
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_parse_coco_json_case_insensitive_names(tmp_path: Path):
    """CVAT exports 'Battery'/'Cartridge'; CLASS_MAP is lower case."""
    doc = _coco_doc(annotations=[
        _ann(1, 7, 1, (10.0, 20.0, 30.0, 40.0)),
        _ann(2, 7, 2, (100.0, 120.0, 300.0, 230.0)),
    ])
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    assert len(recs) == 1
    assert recs[0].labels == [CLASS_MAP["battery"], CLASS_MAP["cartridge"]]
    assert recs[0].file_name == "IMG_0001.jpg"
    assert (recs[0].width, recs[0].height) == (3024, 4032)
    assert recs[0].image_id == 7


def test_parse_coco_json_converts_xywh_to_xyxy(tmp_path: Path):
    doc = _coco_doc(annotations=[_ann(1, 7, 1, (10.0, 20.0, 30.0, 40.0))])
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    # [x, y, w, h] -> [x0, y0, x0 + w, y0 + h], max edges exclusive.
    assert recs[0].boxes == [(10.0, 20.0, 40.0, 60.0)]


def test_parse_coco_json_drops_degenerate_boxes(tmp_path: Path):
    doc = _coco_doc(annotations=[
        _ann(1, 7, 1, (10.0, 20.0, 30.0, 40.0)),   # keep
        _ann(2, 7, 1, (10.0, 20.0, 0.0, 40.0)),    # zero width
        _ann(3, 7, 2, (10.0, 20.0, 30.0, 0.0)),    # zero height
        _ann(4, 7, 2, (10.0, 20.0, -5.0, 40.0)),   # negative width
    ])
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    assert recs[0].boxes == [(10.0, 20.0, 40.0, 60.0)]
    assert recs[0].labels == [CLASS_MAP["battery"]]


def test_parse_coco_json_skips_unknown_classes(tmp_path: Path):
    doc = _coco_doc(
        categories=[
            {"id": 1, "name": "Battery"},
            {"id": 2, "name": "Cartridge"},
            {"id": 9, "name": "Screwdriver"},
        ],
        annotations=[
            _ann(1, 7, 9, (0.0, 0.0, 50.0, 50.0)),
            _ann(2, 7, 1, (10.0, 20.0, 30.0, 40.0)),
        ],
    )
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    assert recs[0].labels == [CLASS_MAP["battery"]]
    assert len(recs[0].boxes) == 1


def test_parse_coco_json_rejects_mismatched_category_ids(tmp_path: Path):
    """A re-export that renumbers categories must fail loudly."""
    doc = _coco_doc(
        categories=[
            {"id": 2, "name": "Battery"},
            {"id": 1, "name": "Cartridge"},
        ],
        annotations=[_ann(1, 7, 2, (10.0, 20.0, 30.0, 40.0))],
    )
    with pytest.raises(ValueError, match="CLASS_MAP"):
        parse_coco_json(_write_coco(tmp_path, doc))


def test_parse_coco_json_keeps_images_without_annotations(tmp_path: Path):
    doc = _coco_doc(images=[
        {"id": 1, "width": 640, "height": 480, "file_name": "a.jpg"},
        {"id": 2, "width": 640, "height": 480, "file_name": "b.jpg"},
    ], annotations=[_ann(1, 2, 1, (1.0, 2.0, 3.0, 4.0))])
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    assert [r.file_name for r in recs] == ["a.jpg", "b.jpg"]
    assert recs[0].boxes == [] and recs[0].labels == []
    assert len(recs[1].boxes) == 1


def test_realtest_annotations_are_intact():
    """Regression guard on the held-out real-photo set itself."""
    if not REALTEST_ANN.is_file():  # pragma: no cover - data is committed
        pytest.skip(f"real-photo test set missing: {REALTEST_ANN}")
    recs = parse_coco_json(REALTEST_ANN)
    assert len(recs) == 7
    labels = [lbl for r in recs for lbl in r.labels]
    assert len(labels) == 80
    assert labels.count(CLASS_MAP["battery"]) == 60
    assert labels.count(CLASS_MAP["cartridge"]) == 20
    # Every box must be well-formed and inside its image.
    for r in recs:
        assert (r.width, r.height) == (3024, 4032)
        for (x0, y0, x1, y1) in r.boxes:
            assert x1 > x0 and y1 > y0
            assert 0 <= x0 and 0 <= y0
            assert x1 <= r.width + 1 and y1 <= r.height + 1


# ---------- RealPhotoDataset ----------------------------------------------

def test_real_photo_dataset_roundtrip(tmp_path: Path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (64, 48), (30, 40, 50)).save(img_dir / "p0.jpg")
    # p1.jpg is annotated but absent on disk → dropped at construction.
    doc = _coco_doc(
        images=[
            {"id": 11, "width": 64, "height": 48, "file_name": "p0.jpg"},
            {"id": 12, "width": 64, "height": 48, "file_name": "p1.jpg"},
        ],
        annotations=[
            _ann(1, 11, 1, (4.0, 5.0, 10.0, 20.0)),
            _ann(2, 12, 2, (0.0, 0.0, 8.0, 8.0)),
        ],
    )
    ds = RealPhotoDataset(str(img_dir), str(_write_coco(tmp_path, doc)))
    assert len(ds) == 1

    img, tgt = ds[0]
    boxes, labels = tgt["boxes"], tgt["labels"]
    assert len(boxes) == 1 and len(labels) == 1
    assert list(np.asarray(boxes)[0]) == [4.0, 5.0, 14.0, 25.0]
    assert int(np.asarray(labels)[0]) == CLASS_MAP["battery"]
    # image_id carries the COCO id, not the dataset index.
    assert int(np.asarray(tgt["image_id"])[0]) == 11

    if isinstance(img, np.ndarray):  # torch-free fallback
        assert img.ndim == 3
    else:  # torch tensor: float32 CHW in [0, 1]
        assert img.ndim == 3 and img.shape[0] == 3
        assert str(img.dtype) == "torch.float32"
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


# ---------- eval_real: under-annotated images ------------------------------
#
# IMG_4428.jpg carries zero boxes in the CVAT export while the photograph
# itself is full of cells, shells and a PCB. With no ground truth to match,
# every correct detection on it scores as a false positive, so including it
# depresses precision - and therefore AP - for the whole set while looking
# exactly like a weak detector. eval_real excludes zero-GT images by default
# and says so in the report.

def _records_with_an_empty_image(tmp_path: Path):
    doc = _coco_doc(
        images=[
            {"id": 1, "width": 3024, "height": 4032, "file_name": "full.jpg"},
            {"id": 2, "width": 3024, "height": 4032, "file_name": "empty.jpg"},
        ],
        annotations=[
            _ann(1, 1, 1, (10.0, 20.0, 30.0, 40.0)),
            _ann(2, 1, 2, (100.0, 120.0, 300.0, 230.0)),
        ],
    )
    return parse_coco_json(_write_coco(tmp_path, doc))


def test_partition_records_excludes_zero_gt_images(tmp_path: Path):
    from recog.eval_real import partition_records

    recs = _records_with_an_empty_image(tmp_path)
    assert [len(r.boxes) for r in recs] == [2, 0]

    scored, excluded = partition_records(recs)
    assert [r.file_name for r in scored] == ["full.jpg"]
    assert [r.file_name for r in excluded] == ["empty.jpg"]

    # --include-empty puts it back, and nothing is reported as excluded.
    scored, excluded = partition_records(recs, include_empty=True)
    assert [r.file_name for r in scored] == ["full.jpg", "empty.jpg"]
    assert excluded == []


def test_zero_gt_image_is_named_in_the_report(tmp_path: Path):
    """The exclusion must be visible to a reader who only sees the summary."""
    from recog.eval_real import (
        build_image_rows, format_report, partition_records, summarise,
    )

    recs = _records_with_an_empty_image(tmp_path)
    scored, excluded = partition_records(recs)

    # The unlabelled image attracts predictions - that is the evidence it is
    # under-annotated rather than empty.
    preds = {
        1: [((10.0, 20.0, 40.0, 60.0), CLASS_MAP["battery"], 0.9)],
        2: [((0.0, 0.0, 50.0, 50.0), CLASS_MAP["battery"], 0.8),
            ((60.0, 60.0, 90.0, 90.0), CLASS_MAP["cartridge"], 0.75)],
    }
    rows = build_image_rows(scored, excluded, preds)
    by_name = {r.file_name: r for r in rows}
    assert by_name["empty.jpg"].scored is False
    assert by_name["empty.jpg"].n_gt == 0
    assert by_name["empty.jpg"].n_pred == 2
    assert by_name["empty.jpg"].ap is None
    assert by_name["full.jpg"].scored is True
    assert by_name["full.jpg"].ap is not None

    gts = {r.image_id: list(zip(r.boxes, r.labels)) for r in scored}
    scored_preds = {r.image_id: preds[r.image_id] for r in scored}
    text = format_report(
        summarise(gts, scored_preds),
        {CLASS_MAP["battery"]: 1, CLASS_MAP["cartridge"]: 1},
        {CLASS_MAP["battery"]: 1, CLASS_MAP["cartridge"]: 0},
        n_images=len(scored),
        confidence=0.7,
        detector_name="Stub",
        checkpoint=None,
        config_path=None,
        elapsed_s=1.0,
        rows=rows,
        n_found=len(recs),
    )
    assert "empty.jpg" in text
    assert "EXCLUDED" in text
    assert "--include-empty" in text
    # The scored-of-found count is stated, both up top and under the table.
    assert "1 of 2 scored" in text
    assert "over 1 of the 2 annotated image(s) found" in text


def test_report_flags_images_with_far_more_predictions_than_boxes(tmp_path: Path):
    """IMG_4435-shaped case: one box on a tray full of parts."""
    from recog.eval_real import build_image_rows, format_report, summarise

    doc = _coco_doc(
        images=[{"id": 5, "width": 3024, "height": 4032,
                 "file_name": "partial.jpg"}],
        annotations=[_ann(1, 5, 1, (10.0, 20.0, 30.0, 40.0))],
    )
    recs = parse_coco_json(_write_coco(tmp_path, doc))
    preds = {5: [((float(i), 0.0, float(i) + 20.0, 20.0),
                  CLASS_MAP["battery"], 0.9) for i in range(19)]}

    rows = build_image_rows(recs, [], preds)
    assert "under-annotated" in rows[0].note

    gts = {5: list(zip(recs[0].boxes, recs[0].labels))}
    text = format_report(
        summarise(gts, preds),
        {CLASS_MAP["battery"]: 1, CLASS_MAP["cartridge"]: 0},
        {CLASS_MAP["battery"]: 19, CLASS_MAP["cartridge"]: 0},
        n_images=1, confidence=0.7, detector_name="Stub", checkpoint=None,
        config_path=None, elapsed_s=1.0, rows=rows, n_found=1,
    )
    assert "POSSIBLY UNDER-ANNOTATED" in text
    assert "19.0x" in text


def test_per_image_ap_uses_only_the_classes_present(tmp_path: Path):
    """A photo of loose cells must not be capped at 0.5 for having no cartridge."""
    from recog.eval_real import per_image_ap

    box = (10.0, 20.0, 40.0, 60.0)
    gts = [(box, CLASS_MAP["battery"])]
    assert per_image_ap(gts, [(box, CLASS_MAP["battery"], 0.9)]) == pytest.approx(1.0)
    assert per_image_ap(gts, []) == 0.0
    # No ground truth at all: no score exists for that image.
    assert per_image_ap([], [(box, CLASS_MAP["battery"], 0.9)]) is None
