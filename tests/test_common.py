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


# --------------------- contract enforcement, not just freezing ------------
#
# Frozen guarantees a value will not change. It guarantees nothing about
# the value being possible, and until 2026-08-12 possible was checked
# nowhere in common/types.py — the file whose own docstring calls itself
# the only thing that crosses Recognition -> Planning -> Execution. Each
# test below pins one value that used to construct cleanly.

class TestBBoxOrdering:
    """`xmin`/`ymin` inclusive, `xmax`/`ymax` exclusive — so
    `xmin <= xmax`. Stated in the docstring since the file was written,
    enforced by nothing."""

    def test_inverted_in_x_is_refused(self):
        with pytest.raises(ValueError, match="inverted"):
            BBox(100, 100, 0, 0)

    def test_inverted_in_x_only(self):
        with pytest.raises(ValueError):
            BBox(10, 0, 5, 10)

    def test_inverted_in_y_only(self):
        with pytest.raises(ValueError):
            BBox(0, 10, 10, 5)

    def test_an_inverted_box_used_to_launder_itself_into_zero_area(self):
        """Why this one matters more than it looks.

        `area` clamps with `max(0.0, ...)` and `iou` returns 0.0, so
        BBox(100, 100, 0, 0) — width -100 — produced a perfectly
        plausible zero-area value and propagated into Detection,
        Cartridge, Battery and PlacementArea.rectangle with nothing
        downstream able to tell it from a legitimately empty box.
        """
        with pytest.raises(ValueError):
            BBox(100, 100, 0, 0)
        # And the shape it used to take: the same numbers as a genuinely
        # empty box, which is why nothing downstream could tell them
        # apart.
        assert BBox(0, 0, 0, 0).area == 0.0

    def test_zero_area_is_still_legal(self):
        """The docstring calls a zero-area box a valid value and says it
        round-trips through iou as 0.0. Degenerate is not inverted."""
        b = BBox(5, 5, 5, 5)
        assert b.area == 0.0
        assert b.iou(b) == 0.0

    def test_ordinary_boxes_are_unaffected(self):
        b = BBox(10, 20, 30, 50)
        assert (b.width, b.height, b.area) == (20, 30, 600)

    def test_nan_is_refused(self):
        """A NaN coordinate is a detector that has diverged. It should
        stop at the boundary, not at the first arithmetic that
        propagates it silently."""
        with pytest.raises(ValueError):
            BBox(0, 0, float("nan"), 10)

    def test_replace_cannot_route_around_the_check(self):
        """`dataclasses.replace` re-runs __init__, so it re-runs
        __post_init__. Worth pinning: `replace` is the usual way
        validation gets skipped, and it was moot here only because there
        was no validation to skip."""
        import dataclasses
        with pytest.raises(ValueError):
            dataclasses.replace(BBox(0, 0, 10, 10), xmax=-5)


class TestDetectionConfidence:
    """Confidence feeds every score threshold and the NMS ordering in
    recog.inference. Out of [0, 1] it sorts first or last
    unconditionally, ahead of every real detection."""

    @pytest.mark.parametrize("c", [17.5, -3.0, 1.0000001, float("nan")])
    def test_out_of_range_is_refused(self, c):
        with pytest.raises(ValueError, match="probability"):
            Detection(BBox(0, 0, 1, 1), ClassLabel.BATTERY, c)

    @pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
    def test_the_closed_unit_interval_is_accepted(self, c):
        """Both ends closed: a detector may saturate at exactly 0 or 1."""
        assert Detection(BBox(0, 0, 1, 1), ClassLabel.BATTERY, c).confidence \
            == c

    def test_replace_cannot_route_around_the_check(self):
        import dataclasses
        d = Detection(BBox(0, 0, 1, 1), ClassLabel.BATTERY, 0.9)
        with pytest.raises(ValueError):
            dataclasses.replace(d, confidence=-1.0)


class TestPickPlacePoseIndices:
    """Negative grid indices are the classic numpy silent-wraparound
    hazard: grid[-5] addresses the fifth row from the END and marks a
    cell nobody asked for. OccupancyGrid.set_block already refuses one,
    but that defence is in the consumer, not on the contract."""

    def _pose(self, **kw):
        kw = {"pick": WorkspacePoint(0, 0, 0),
              "place": WorkspacePoint(0, 0, 0),
              "cartridge_id": 0, "grid_row": 0, "grid_col": 0, **kw}
        return PickPlacePose(**kw)

    @pytest.mark.parametrize("field", ["grid_row", "grid_col"])
    def test_negative_grid_index_is_refused(self, field):
        with pytest.raises(ValueError, match="non-negative"):
            self._pose(**{field: -5})

    def test_negative_cartridge_id_is_refused(self):
        with pytest.raises(ValueError, match="cartridge_id"):
            self._pose(cartridge_id=-99)

    def test_zero_indices_are_legal(self):
        assert self._pose().grid_row == 0

    def test_the_no_battery_sentinel_survives(self):
        """battery_detection_id defaults to -1 meaning "no battery is
        associated". Validating it would break the documented default."""
        assert self._pose().battery_detection_id == -1
        assert self._pose(battery_detection_id=-1).battery_detection_id == -1


class TestRobotStatusDuration:
    def test_negative_cycle_time_is_refused(self):
        """It flows straight into the latency statistics the demo prints
        and the FDR quotes; a negative duration makes a mean
        meaningless."""
        with pytest.raises(ValueError, match="duration"):
            RobotStatus(code=RobotStatusCode.OK,
                        current_pose=WorkspacePoint(0, 0, 0),
                        cycle_time_ms=-42.0)

    def test_zero_is_legal(self):
        s = RobotStatus(code=RobotStatusCode.OK,
                        current_pose=WorkspacePoint(0, 0, 0))
        assert s.cycle_time_ms == 0.0


def test_workspace_point_is_deliberately_unvalidated():
    """The line this module draws, asserted so it stays drawn.

    Any finite triple is a physically meaningful pose. What makes one
    unreachable is the deployment's envelope
    (`plan.scene.WorkspaceBounds`), which is configuration, not a
    property of the type — and `WorkspaceBounds.require` already raises
    rather than clamps. Range-checking here would hardcode one cell's
    geometry into the shared contract.
    """
    p = WorkspacePoint(-9999.0, 9999.0, -1.0)
    assert p.to_dict() == {"x_mm": -9999.0, "y_mm": 9999.0, "z_mm": -1.0}


def test_iter_labels_is_gone():
    """It had zero callers, not even a test, and its documented contract
    was to drop an unrecognised label silently — no exception, no log,
    one fewer detection than the caller passed in."""
    import common.types as types
    assert not hasattr(types, "iter_labels")
    with pytest.raises(ValueError):
        ClassLabel("not-a-real-label")


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
