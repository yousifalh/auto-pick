"""Tests for the shared type layer and YAML loader.

Every module in the project imports from ``common.types``; keeping these
tests green is a hard requirement.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.config import load_yaml, load_demo_config
from common.types import (
    BBox, ClassLabel, DEFAULT_WALL_INSET_MM, Detection, PickPlacePose,
    RobotStatus, RobotStatusCode, Snapshot, UnknownScale, WorkspacePoint,
)

# The shipped configs are asserted against by path below. Anchored on this
# file rather than on the working directory: `configs/planning.yaml` is a
# machine parameter file, and a test that silently skipped it when pytest
# ran from elsewhere would assert nothing about the file that ships.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


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


# ------------------- a wrong key is as loud as a wrong path ---------------
#
# Audit 2026-08-15 finding C1b. `load_demo_config` used to write
# `resolved[key] = {}` for any of its three sections that was absent or
# misspelled, and the run went on against `main._build_workspace`'s
# hardcoded +/-350 mm square and `PlannerConfig`'s dataclass defaults. A
# wrong PATH raised (`load_yaml`); a wrong KEY did not, which is the
# reverse of the two being equally fatal - the same "looks configured,
# but is not" class that `plan/planner.py` refuses
# `motion.approach_height_mm` by name for, and that
# `execution/execution.py` cross-checks the protocol constants for.
#
# Every test below writes a config that used to load clean.

def _demo_tree(tmp_path: Path, demo_body: str, planning: str = "",
               name: str = "demo.yaml") -> Path:
    """A three-file config tree; `demo_body` is the top-level YAML."""
    base = tmp_path / "configs"
    base.mkdir(exist_ok=True)
    (base / "recognition.yaml").write_text("training: {lr: 0.01}\n")
    (base / "planning.yaml").write_text(
        planning or "battery: {diameter_mm: 18.5}\n")
    (base / "execution.yaml").write_text("kuka: {host: 127.0.0.1}\n")
    (base / name).write_text(demo_body)
    return base / name


_GOOD_SECTIONS = (
    "recognition: configs/recognition.yaml\n"
    "planning: configs/planning.yaml\n"
    "execution: configs/execution.yaml\n"
)


class TestDemoConfigSectionsAreRequired:
    def test_a_misspelled_section_is_refused(self, tmp_path: Path):
        """`plannning:` used to load as `planning: {}`, and the run then
        planned against PlannerConfig's defaults - 18.5 x 65 mm cells, no
        camera scale, a +/-350 mm envelope - none of which the operator
        wrote down anywhere."""
        demo = _demo_tree(tmp_path, (
            "recognition: configs/recognition.yaml\n"
            "plannning: configs/planning.yaml\n"
            "execution: configs/execution.yaml\n"))
        with pytest.raises(ValueError, match="plannning"):
            load_demo_config(demo)

    def test_an_absent_section_is_refused(self, tmp_path: Path):
        demo = _demo_tree(tmp_path, (
            "recognition: configs/recognition.yaml\n"
            "execution: configs/execution.yaml\n"))
        with pytest.raises(ValueError, match="planning"):
            load_demo_config(demo)

    def test_a_section_that_loads_empty_is_refused(self, tmp_path: Path):
        """An empty file behind a correct path is the same silence as a
        missing key: nothing configured, defaults everywhere."""
        base = tmp_path / "configs"
        base.mkdir()
        (base / "recognition.yaml").write_text("training: {lr: 0.01}\n")
        (base / "planning.yaml").write_text("")
        (base / "execution.yaml").write_text("kuka: {host: 127.0.0.1}\n")
        (base / "demo.yaml").write_text(_GOOD_SECTIONS)
        with pytest.raises(ValueError, match="planning"):
            load_demo_config(base / "demo.yaml")

    def test_an_unknown_top_level_key_is_refused(self, tmp_path: Path):
        demo = _demo_tree(tmp_path, _GOOD_SECTIONS + "modes: {source: x}\n")
        with pytest.raises(ValueError, match="modes"):
            load_demo_config(demo)

    def test_mode_stays_optional(self, tmp_path: Path):
        """No `mode:` block is a legitimate config - every key in it has a
        documented default in main.run. Absent is not the same as
        misspelled."""
        cfg = load_demo_config(_demo_tree(tmp_path, _GOOD_SECTIONS))
        assert cfg["mode"] == {}

    def test_an_inline_section_still_works(self, tmp_path: Path):
        """`load_demo_config` has always accepted a nested mapping in
        place of a path. That is not the defect and must survive."""
        demo = _demo_tree(tmp_path, (
            "recognition: {training: {lr: 0.02}}\n"
            "planning: {battery: {diameter_mm: 21.0}}\n"
            "execution: {kuka: {host: 10.0.0.1}}\n"))
        cfg = load_demo_config(demo)
        assert cfg["planning"]["battery"]["diameter_mm"] == 21.0


class TestModeKeysAreChecked:
    def test_a_misspelled_mode_key_is_refused(self, tmp_path: Path):
        """`stop_on_empty_queu` reverts silently to the default True, and
        a segmentation run over a corpus of unrelated renders then ends
        after its first frame with no `bay` - one cycle, a clean exit and
        a config file that says otherwise."""
        demo = _demo_tree(
            tmp_path, _GOOD_SECTIONS + "mode: {stop_on_empty_queu: false}\n")
        with pytest.raises(ValueError, match="stop_on_empty_queu"):
            load_demo_config(demo)

    def test_a_misspelled_segmentation_key_is_refused(self, tmp_path: Path):
        demo = _demo_tree(
            tmp_path,
            _GOOD_SECTIONS + "mode: {segmentation: {checkpoints: a.pt}}\n")
        with pytest.raises(ValueError, match="checkpoints"):
            load_demo_config(demo)

    def test_the_shipped_mode_keys_are_accepted(self, tmp_path: Path):
        demo = _demo_tree(tmp_path, _GOOD_SECTIONS + (
            "mode:\n"
            "  source: synthetic\n"
            "  robot: mock\n"
            "  max_cycles: 10\n"
            "  log_level: INFO\n"
            "  camera_device: 0\n"
            "  img_dir: recog/dataset/images\n"
            "  mm_per_px: 0.625\n"
            "  stop_on_empty_queue: false\n"
            "  segmentation: {checkpoint: a.pt, config: b.yaml}\n"))
        assert load_demo_config(demo)["mode"]["mm_per_px"] == 0.625


class TestPlanningKeysAreChecked:
    """C1a's other half: refusing the keys back.

    Twelve keys were deleted from `configs/planning.yaml` because no
    Python read them. Deleting them fixes the file that exists; refusing
    them fixes the class, which is the part `configs/execution.yaml`'s
    five deleted `motion:` keys never got.
    """

    @pytest.mark.parametrize("dead", [
        "cartridge: {green_channel_thresh: otsu}",
        "cartridge: {pcb_exclusion_required: true}",
        "occupancy_grid: {dtype: uint8}",
        "camera: {mm_per_px_y: 0.38}",
    ])
    def test_a_key_no_code_reads_is_refused(self, tmp_path: Path, dead: str):
        demo = _demo_tree(tmp_path, _GOOD_SECTIONS,
                          planning=dead + "\n")
        with pytest.raises(ValueError, match="planning"):
            load_demo_config(demo)

    @pytest.mark.parametrize("dead", ["packing", "queue", "arbitration"])
    def test_a_whole_dead_section_is_refused(self, tmp_path: Path,
                                             dead: str):
        """`packing:` named FFDH while both planner call sites ran
        `common.packing.pack_best_effort`; `queue:` named a fill order
        and an assignment policy that are properties of the code. And
        `arbitration: {tau: ...}` is the one this repo already paid for -
        the README quoted a third value for a gate nothing consulted."""
        demo = _demo_tree(tmp_path, _GOOD_SECTIONS,
                          planning=f"{dead}: {{anything: 1}}\n")
        with pytest.raises(ValueError, match=dead):
            load_demo_config(demo)

    def test_a_misspelled_workspace_corner_is_refused(self, tmp_path: Path):
        """`main._build_workspace` reads the four corners with `.get`
        defaults of +/-350, so `x_mim` silently restored the placeholder
        envelope the operator was trying to replace."""
        demo = _demo_tree(tmp_path, _GOOD_SECTIONS, planning=(
            "camera: {workspace_bounds_mm: {x_mim: -100, x_max: 100, "
            "y_min: -100, y_max: 100}}\n"))
        with pytest.raises(ValueError, match="x_mim"):
            load_demo_config(demo)

    def test_approach_height_is_left_for_the_planner_to_refuse_by_name(
            self, tmp_path: Path):
        """`plan/planner.py` already raises on
        `motion.approach_height_mm` with the reason - that the wire Z is
        the GRASP pose and the controller derives its own approach. A
        generic "unknown key" here would replace a specific, correct
        explanation with a worse one, so the loader passes it through on
        purpose.
        """
        from plan.planner import PlannerConfig

        demo = _demo_tree(tmp_path, _GOOD_SECTIONS,
                          planning="motion: {approach_height_mm: 60}\n")
        cfg = load_demo_config(demo)          # must NOT raise here
        with pytest.raises(ValueError, match="approach_height_mm"):
            PlannerConfig.from_dict(cfg["planning"])

    def test_the_shipped_planning_keys_are_accepted(self):
        """The real file, through the real loader."""
        import common.config as config_mod

        cfg = load_yaml(CONFIGS / "planning.yaml")
        config_mod.validate_planning(cfg, "configs/planning.yaml")


class TestEveryShippedConfigStillLoads:
    """19 YAML files (plus 3 generated JSON sidecars), several of them
    frozen experiment records.

    Breaking one is a real regression, and a strict loader is exactly the
    change that would do it silently for the configs nothing in the test
    suite opens.
    """

    def test_every_configs_yaml_parses(self):
        files = sorted(CONFIGS.glob("*.yaml"))
        assert len(files) >= 19, "the corpus this test guards has shrunk"
        for f in files:
            assert isinstance(load_yaml(f), dict), f

    @pytest.mark.parametrize("name", ["demo.yaml", "demo_seg.yaml"])
    def test_the_two_demo_configs_pass_validation(self, name: str):
        cfg = load_demo_config(CONFIGS / name)
        assert cfg["planning"]["battery"]["diameter_mm"] == 18.5
        assert cfg["execution"]["kuka"]


def test_planning_yaml_declares_the_motion_heights_it_uses():
    """The mirror image of the dead keys, and it was in the same file.

    `PlannerConfig.from_dict` READS `motion.grasp_height_mm` and
    `motion.insert_height_mm`; `configs/planning.yaml` carried no
    `motion:` block at all, so the pick grasp height (5.0 mm) and the
    place insert height (2.0 mm) came from dataclass defaults that no
    configuration stated (FDR v3 Appendix B, corrected 2026-08-12). The
    values are unchanged - this asserts they are now written down.
    """
    from plan.planner import PlannerConfig

    cfg = load_yaml(CONFIGS / "planning.yaml")
    assert cfg["motion"]["grasp_height_mm"] == 5.0
    assert cfg["motion"]["insert_height_mm"] == 2.0
    planner_cfg = PlannerConfig.from_dict(cfg)
    assert planner_cfg.pick_grasp_height_mm == 5.0
    assert planner_cfg.place_insert_height_mm == 2.0


# --------------------------- logger levels --------------------------------

def test_get_logger_honours_a_level_on_a_logger_it_already_issued():
    """`level` used to be dropped on every call after the first.

    `get_logger` returns early when the logger already has handlers - the
    idempotence that stops duplicated lines - and the early return took
    the level with it. Every module in this pipeline builds its logger at
    IMPORT time with no level, so by the time any caller had a
    configured one to apply, the argument did nothing: a `level=` that
    looks like it sets the level and does not.
    """
    import logging as _logging

    from common.logging import get_logger

    name = "autopick.test.level-is-honoured"
    first = get_logger(name)
    assert first.level == _logging.INFO
    again = get_logger(name, "DEBUG")
    assert again is first
    assert again.level == _logging.DEBUG
    get_logger(name, "INFO")


def test_set_level_reaches_every_logger_get_logger_issued():
    """`mode.log_level` has to reach loggers that were built at import.

    `common.logging` sets `propagate = False` and attaches a handler per
    logger, so there is no single ancestor whose level governs the
    pipeline; the levels have to be set on the loggers themselves.
    """
    import logging as _logging

    from common.logging import get_logger, set_level

    a = get_logger("autopick.test.set-level-a")
    b = get_logger("autopick.test.set-level-b")
    try:
        set_level("WARNING")
        assert a.level == _logging.WARNING
        assert b.level == _logging.WARNING
    finally:
        set_level("INFO")


def test_set_level_refuses_something_that_is_not_a_level():
    """A typo'd `mode.log_level` must not leave the run at whatever
    level it happened to have."""
    from common.logging import set_level

    with pytest.raises(ValueError, match="not a log level"):
        set_level("INFOO")


def test_a_level_some_dependency_registered_is_not_a_level_here():
    """Why `common.logging` spells the five levels out.

    `logging.getLevelName` is the obvious validator and consults a
    PROCESS-GLOBAL registry any installed package may add to. Measured
    in this environment on 2026-08-15, `logging.getLevelName("VERBOSE")`
    returns 15 - a level contributed by a dependency and one that no
    logger in this repository emits at. Accepting it would turn a typo
    into silence that looks configured.
    """
    from common.logging import set_level

    with pytest.raises(ValueError, match="not a log level"):
        set_level("VERBOSE")


# ------------- the two contracts that are not dataclasses ------------------
#
# `UnknownScale` and `DEFAULT_WALL_INSET_MM` moved here from
# `plan.placement_area` on 2026-08-15. They are the vocabulary Recognition
# and Planning need in common to talk about a frame's scale, and holding
# them in `plan` meant `recog.seg_evaluate` and `recog.calibrate_tau`
# imported `plan` at module load - the back-edge `plan/bin_packing.py`'s
# docstring cites as the reason the packing algorithms live in
# `common.packing`. tests/test_seams.py pins the graph; these pin the
# contracts themselves.


def test_unknown_scale_is_a_value_error():
    """Callers written against the original contract keep working.

    `plan.planner` catches it to turn an uncalibrated frame into a
    counted skip, and `tests/test_placement_area.py` relies on
    `UnknownScale` being raised where a plain `ValueError` was expected
    before it existed.
    """
    assert issubclass(UnknownScale, ValueError)
    with pytest.raises(ValueError):
        raise UnknownScale("no calibration on this frame")


def test_unknown_scale_is_the_class_the_planning_layer_raises():
    """One class, not two that share a name.

    Two same-named classes would let `except UnknownScale` in the
    planner miss the one `recog.seg_evaluate` raises, which is exactly
    how "raise rather than default" becomes "raise in one of the two
    places".
    """
    from plan.placement_area import UnknownScale as from_plan
    from recog.seg_evaluate import UnknownScale as from_recog

    assert from_plan is UnknownScale
    assert from_recog is UnknownScale


def test_the_default_wall_inset_is_the_measured_worst_case():
    """4.25 mm is the MAX of the four CAD-measured `case_wall_mm`.

    recog/synth3d/assets/catalog.json records 4.0, 3.75, 3.7 and 4.25 mm
    and no SKU identifier crosses the Recognition -> Planning boundary,
    so one scalar has to stand in for all four. The MAX is deliberate:
    eroding too little reports wall material as safe to place a cell
    against. Pinned as a number because three packages default to it and
    a receipt quotes it.
    """
    import json

    assert DEFAULT_WALL_INSET_MM == 4.25

    catalog = REPO_ROOT / "recog" / "synth3d" / "assets" / "catalog.json"
    if not catalog.is_file():           # asset pack not present
        pytest.skip(f"{catalog} is not in this checkout")
    assets = json.loads(catalog.read_text(encoding="utf-8"))["assets"]
    walls = [float(a["case_wall_mm"]) for a in assets
             if "case_wall_mm" in a]
    assert walls, "catalog.json carries no case_wall_mm to check against"
    assert DEFAULT_WALL_INSET_MM == max(walls)


def test_common_types_stays_cheap_to_import():
    """Why this is a legal home for a shared contract.

    The constant was restated in `recog/seg_ablation.py` with a comment
    saying the copy existed to keep cv2 out of that module's import
    surface - true while it lived in `plan.placement_area`, which
    imports cv2 and `plan.scene` at module level. Importing it from here
    costs nothing, and this test is what keeps that true.
    """
    import subprocess
    import sys

    code = (
        "import sys; import common.types;"
        "heavy=[m for m in ('cv2','numpy','torch') if m in sys.modules];"
        "print(sorted(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout
