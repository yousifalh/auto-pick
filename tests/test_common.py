"""Tests for the shared type layer and YAML loader.

Every module in the project imports from ``common.types``; keeping these
tests green is a hard requirement.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from common.config import load_yaml, load_demo_config
from common.types import (
    BBox, ClassLabel, Detection, PickPlacePose, RobotStatus,
    RobotStatusCode, Snapshot, WorkspacePoint,
)


# ----------------------------- BBox ---------------------------------------

class TestBBox:
    def test_dimensions(self):
        b = BBox(10, 20, 30, 50)
        assert b.width == 20
        assert b.height == 30
        assert b.area == 600
        assert b.cx == 20.0
        assert b.cy == 35.0

    def test_iou_identical(self):
        a = BBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(100, 100, 110, 110)
        assert a.iou(b) == 0.0

    def test_iou_half_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 15, 10)
        # Intersection 50, union 150 → IoU = 1/3
        assert a.iou(b) == pytest.approx(50.0 / 150.0)

    def test_iou_touching(self):
        # zero-area intersection counts as no overlap
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 20, 10)
        assert a.iou(b) == 0.0

    def test_iou_degenerate(self):
        # zero-area boxes → undefined union, our impl returns 0.0
        a = BBox(5, 5, 5, 5)
        b = BBox(5, 5, 5, 5)
        assert a.iou(b) == 0.0

    def test_frozen(self):
        b = BBox(0, 0, 1, 1)
        with pytest.raises(Exception):
            b.xmin = 42  # type: ignore[misc]


# --------------------------- Detection ------------------------------------

def test_detection_to_dict_roundtrip():
    d = Detection(BBox(1, 2, 3, 4), ClassLabel.BATTERY, 0.9)
    out = d.to_dict()
    assert out["bbox"] == [1, 2, 3, 4]
    assert out["label"] == "battery"
    assert out["confidence"] == pytest.approx(0.9)


def test_snapshot_filter_by_label():
    dets = [
        Detection(BBox(0, 0, 1, 1), ClassLabel.BATTERY, 0.8),
        Detection(BBox(2, 2, 3, 3), ClassLabel.CARTRIDGE, 0.7),
        Detection(BBox(4, 4, 5, 5), ClassLabel.BATTERY, 0.6),
    ]
    s = Snapshot(detections=dets)
    assert len(s.of(ClassLabel.BATTERY)) == 2
    assert len(s.of(ClassLabel.CARTRIDGE)) == 1
    assert len(s.of(ClassLabel.BACKGROUND)) == 0


def test_snapshot_serialisation():
    s = Snapshot(
        detections=[Detection(BBox(0, 0, 10, 10), ClassLabel.BATTERY, 0.5)],
        image_shape=(480, 640),
        timestamp_ns=123,
    )
    d = s.to_dict()
    assert d["image_shape"] == [480, 640]
    assert d["timestamp_ns"] == 123
    assert len(d["detections"]) == 1


def test_snapshot_carries_optional_cartridge_masks():
    import numpy as np

    from common.types import Snapshot

    s = Snapshot()
    assert s.cartridge_masks == {}, "must default to empty, not None"

    s.cartridge_masks[0] = np.zeros((8, 8), np.int8)
    assert s.cartridge_masks[0].shape == (8, 8)


def test_snapshot_to_dict_summarises_masks_rather_than_embedding_them():
    """to_dict feeds logging and regression fixtures. Embedding a
    label map per cartridge would make every log line enormous."""
    import numpy as np

    from common.types import Snapshot

    s = Snapshot()
    s.cartridge_masks[3] = np.zeros((16, 32), np.int8)
    d = s.to_dict()
    assert d["cartridge_masks"] == {"3": [16, 32]}


def test_existing_snapshot_construction_is_unaffected():
    from common.types import BBox, ClassLabel, Detection, Snapshot

    s = Snapshot(detections=[
        Detection(BBox(0, 0, 4, 4), ClassLabel.BATTERY, 0.9)])
    assert len(s.of(ClassLabel.BATTERY)) == 1
    assert s.to_dict()["detections"][0]["label"] == "battery"


# -------------------------- Robot types -----------------------------------

def test_pick_place_pose_to_dict():
    p = PickPlacePose(
        pick=WorkspacePoint(1, 2, 3),
        place=WorkspacePoint(4, 5, 6),
        cartridge_id=7, grid_row=8, grid_col=9,
    )
    out = p.to_dict()
    assert out["pick"] == {"x_mm": 1, "y_mm": 2, "z_mm": 3}
    assert out["cartridge_id"] == 7


def test_robot_status_codes_are_unique():
    values = [c.value for c in RobotStatusCode]
    assert len(values) == len(set(values))


def test_robot_status_defaults():
    s = RobotStatus(code=RobotStatusCode.OK,
                    current_pose=WorkspacePoint(0, 0, 0))
    assert s.cycle_time_ms == 0.0
    assert s.message == ""


# ------------------------- YAML loader ------------------------------------

def test_load_yaml_roundtrip(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text("a: 1\nb:\n  c: 2\n")
    d = load_yaml(f)
    assert d == {"a": 1, "b": {"c": 2}}


def test_load_yaml_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_empty(tmp_path: Path):
    f = tmp_path / "empty.yaml"
    f.write_text("")
    assert load_yaml(f) == {}


def test_load_demo_config_resolves_subpaths(tmp_path: Path):
    base = tmp_path / "configs"
    base.mkdir()
    (base / "recognition.yaml").write_text("training: {lr: 0.01}\n")
    (base / "planning.yaml").write_text("battery: {diameter_mm: 18.5}\n")
    (base / "execution.yaml").write_text("kuka: {host: 127.0.0.1}\n")
    (base / "demo.yaml").write_text(
        "recognition: configs/recognition.yaml\n"
        "planning: configs/planning.yaml\n"
        "execution: configs/execution.yaml\n"
        "mode: {source: synthetic}\n"
    )
    cfg = load_demo_config(base / "demo.yaml")
    assert cfg["recognition"]["training"]["lr"] == 0.01
    assert cfg["planning"]["battery"]["diameter_mm"] == 18.5
    assert cfg["execution"]["kuka"]["host"] == "127.0.0.1"
    assert cfg["mode"]["source"] == "synthetic"
