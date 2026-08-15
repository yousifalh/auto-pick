"""Thin YAML configuration loader.

The project deliberately avoids the pydantic / hydra / dynaconf stack:
configuration is a small number of shallow YAML files, and a function
that returns nested :class:`dict` is the simplest thing that keeps
this surface testable and dependency-light.

What that simplicity cost, and what is fixed here (audit 2026-08-15,
finding C1b). ``load_demo_config`` used to write ``resolved[key] = {}``
for any of its three sections that was absent or misspelled, so a
``plannning:`` typo produced a run planned against
``main._build_workspace``'s hardcoded +/-350 mm square and
``PlannerConfig``'s dataclass defaults, with every statistic reading
normal. A wrong PATH raised (:func:`load_yaml`); a wrong KEY did not.
The two are the same operator mistake and are now equally fatal.

The schema below is deliberately narrow. It covers the top-level demo
config, its ``mode:`` block and the ``planning:`` section — the three
surfaces ``main.py`` itself consumes — and states, key by key, what the
Python actually reads. ``recognition:`` and ``execution:`` are left to
their own modules (``ExecutionConfig.from_dict`` already cross-checks
the protocol constants it is handed and raises on a mismatch); an
enumeration of ``configs/recognition.yaml``'s surface belongs with the
code that reads it, not here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


_SUB_KEYS: tuple[str, ...] = ("recognition", "planning", "execution")
_TOP_KEYS: tuple[str, ...] = _SUB_KEYS + ("mode",)

# Every ``mode:`` key with a reader in ``main.py``, verified by grep:
# `source` (_image_source), `camera_device` (_camera_source), `img_dir`
# (_resolve_image_dir), `robot` (_start_robot), `segmentation`
# (_build_segmenter), `mm_per_px` (_build_planner), `max_cycles` and
# `stop_on_empty_queue` (run), `log_level` (run -> common.logging
# .set_level). A misspelling here is not cosmetic: `stop_on_empty_queu`
# reverts to the default True, and a segmentation run over a corpus of
# unrelated renders then ends after the first frame that predicts no
# `bay` — one cycle, a clean exit, and a config file that says
# otherwise.
_MODE_KEYS: tuple[str, ...] = (
    "source", "robot", "max_cycles", "log_level", "camera_device",
    "img_dir", "mm_per_px", "stop_on_empty_queue", "segmentation",
)
_SEGMENTATION_KEYS: tuple[str, ...] = ("checkpoint", "config")

# The ``planning:`` section, key by key, with the reader that justifies
# it. This is the list twelve keys failed on 2026-08-15:
# `cartridge.green_channel_thresh`, `.pcb_exclusion_required`,
# `occupancy_grid.dtype`, `packing.*`, `queue.*` and `camera.mm_per_px_y`
# had no consumer anywhere in the tree, and `cartridge.morph_*_ksize`
# were real constructor parameters of HeuristicPlacementAreaExtractor
# that `main.py` never passed. They are deleted from the file; this is
# what stops them coming back.
#
#   battery.*             PlannerConfig.from_dict; placement_feasibility
#   cartridge.safety_margin_px, .morph_close_ksize, .morph_open_ksize
#                         main._build_planner -> Heuristic...Extractor
#   cartridge.wall_inset_mm
#                         main._build_planner -> Segmentation...Extractor
#   occupancy_grid.resolution_mm_per_cell
#                         main._build_planner; scripts/placement_feasibility
#   camera.mm_per_px_x, .origin_offset_*
#                         PlannerConfig.from_dict
#   camera.workspace_bounds_mm
#                         main._build_workspace -> plan.scene.WorkspaceBounds
#   motion.*              PlannerConfig.from_dict
#   tracking.*            plan.scene.TrackingConfig.from_dict
_PLANNING_SCHEMA: dict[str, tuple[str, ...]] = {
    "battery": ("diameter_mm", "length_mm"),
    "cartridge": ("safety_margin_px", "morph_close_ksize",
                  "morph_open_ksize", "wall_inset_mm"),
    "occupancy_grid": ("resolution_mm_per_cell",),
    "camera": ("mm_per_px_x", "origin_offset_x_mm", "origin_offset_y_mm",
               "workspace_bounds_mm"),
    "motion": ("grasp_height_mm", "insert_height_mm"),
    "tracking": ("iou_match_threshold", "max_missing_frames",
                 "duplicate_iou_threshold", "geometry_refresh_mm"),
}

# `main._build_workspace` reads the four corners with `.get` defaults of
# +/-350, so `x_mim` silently restored the placeholder envelope the
# operator was editing the file to replace.
_WORKSPACE_BOUNDS_KEYS: tuple[str, ...] = ("x_min", "x_max", "y_min", "y_max")

# Keys this loader deliberately passes through so a downstream consumer
# can refuse them with a better message than "unknown key" could carry.
# `plan/planner.py` raises on `motion.approach_height_mm` and explains
# why — the wire Z is the GRASP pose and `krl_prog/routines.src` derives
# its own approach from it (+60 mm) — and that explanation is the whole
# value of the refusal.
_REFUSED_DOWNSTREAM: frozenset[tuple[str, str]] = frozenset({
    ("motion", "approach_height_mm"),
})


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Load a single YAML file and return its top-level mapping.

    An empty file is returned as an empty dict (YAML's canonical
    interpretation of *nothing*). A missing file raises
    :class:`FileNotFoundError`, since mis-pointed config paths are
    almost always a user error worth surfacing loudly.

    Deliberately schema-free: every training, evaluation and scene
    generation config in ``configs/`` comes through here, and each is
    owned by the module that reads it.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return dict(data) if isinstance(data, dict) else {}


def _reject_unknown(
    where: str,
    present: Iterable[str],
    allowed: Iterable[str],
    source: str,
) -> None:
    """Raise naming every key in ``present`` that ``allowed`` omits.

    All of them at once, not the first: an operator who mistyped two
    keys should not have to run twice to find out.
    """
    allowed = tuple(allowed)
    unknown = sorted(k for k in present if k not in allowed)
    if not unknown:
        return
    raise ValueError(
        f"{source}: {where} contains {len(unknown)} key(s) no code reads: "
        f"{', '.join(unknown)}. Accepted here: {', '.join(sorted(allowed))}. "
        "A key that looks live and is not is worse than a missing one - it "
        "is edited, it changes nothing, and the run reports success "
        "(configs/execution.yaml lost five motion keys to exactly this, "
        "and configs/planning.yaml twelve more). Delete it, or wire it up "
        "and add it here.")


def _as_mapping(value: Any, where: str, source: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{source}: {where} must be a mapping, got "
            f"{type(value).__name__}")
    return value


def validate_planning(cfg: Mapping[str, Any], source: str) -> None:
    """Refuse a ``planning:`` section carrying a key nothing reads.

    Public because ``configs/planning.yaml`` is a machine parameter file
    and is worth checking on its own, without a demo config wrapped
    around it.
    """
    _reject_unknown("planning", cfg, _PLANNING_SCHEMA, source)
    for section, keys in _PLANNING_SCHEMA.items():
        block = _as_mapping(cfg.get(section), f"planning.{section}", source)
        _reject_unknown(
            f"planning.{section}",
            (k for k in block
             if (section, k) not in _REFUSED_DOWNSTREAM),
            keys, source)
    bounds = _as_mapping(
        _as_mapping(cfg.get("camera"), "planning.camera", source)
        .get("workspace_bounds_mm"),
        "planning.camera.workspace_bounds_mm", source)
    if bounds:
        _reject_unknown("planning.camera.workspace_bounds_mm", bounds,
                        _WORKSPACE_BOUNDS_KEYS, source)


def validate_mode(cfg: Mapping[str, Any], source: str) -> None:
    """Refuse a ``mode:`` block carrying a key ``main.py`` never reads."""
    _reject_unknown("mode", cfg, _MODE_KEYS, source)
    seg = _as_mapping(cfg.get("segmentation"), "mode.segmentation", source)
    _reject_unknown("mode.segmentation", seg, _SEGMENTATION_KEYS, source)


def load_demo_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a top-level ``demo.yaml`` and inline the three sub-configs.

    ``demo.yaml`` names the three module configs by path (either
    absolute, or relative to the *project root* — i.e. the grandparent
    of ``demo.yaml``). Returns a single nested dict with the keys
    ``recognition``, ``planning``, ``execution`` and ``mode``.

    All three sections are REQUIRED and must resolve to a non-empty
    mapping. A section that is absent, misspelled, or points at an empty
    file used to become ``{}`` and let the run proceed on defaults
    nobody wrote down; it now raises, in the same refuse-loudly style as
    ``execution.ExecutionConfig.from_dict``'s protocol cross-checks and
    ``plan.planner``'s by-name refusal of ``motion.approach_height_mm``.
    ``mode:`` stays optional — every key in it has a documented default
    in ``main.run`` — but is key-checked when present.
    """
    source = str(path)
    top = load_yaml(path)
    project_root = Path(path).resolve().parent.parent

    _reject_unknown("the top level", top, _TOP_KEYS, source)

    resolved: dict[str, Any] = {}
    for key in _SUB_KEYS:
        sub = top.get(key)
        if isinstance(sub, str):
            sub_path = Path(sub)
            if not sub_path.is_absolute():
                sub_path = project_root / sub_path
            loaded = load_yaml(sub_path)
            origin = str(sub_path)
        elif isinstance(sub, Mapping):
            loaded = dict(sub)
            origin = f"{source} ({key}:, inline)"
        elif sub is None:
            raise ValueError(
                f"{source}: no `{key}:` section. All three of "
                f"{', '.join(_SUB_KEYS)} are required, as a path to a "
                "module config or as an inline mapping. This used to "
                "resolve to an empty dict and the run continued on "
                "hardcoded defaults - a wrong PATH raised while a wrong "
                "KEY did not.")
        else:
            raise ValueError(
                f"{source}: `{key}:` must be a path or a mapping, got "
                f"{type(sub).__name__}")

        if not loaded:
            raise ValueError(
                f"{source}: `{key}:` resolved to nothing from {origin}. An "
                "empty section is the same silence as a missing one: the "
                "run would plan against defaults no configuration states.")
        resolved[key] = loaded

    validate_planning(resolved["planning"], source)

    mode = _as_mapping(top.get("mode"), "mode", source)
    validate_mode(mode, source)
    resolved["mode"] = dict(mode)
    return resolved
