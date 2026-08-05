"""Recognition dataset tests.

Uses the synthetic-dataset generator to create a tiny corpus and then
exercises ``parse_voc_xml`` and ``BatteryCartridgeDataset`` end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recog.dataset import (
    CLASS_MAP, BatteryCartridgeDataset, VocAnnotation,
    collate_fn, parse_voc_xml,
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
