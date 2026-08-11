"""
recog.synth3d.config - every tunable in one place. No bpy import, so this
module can be read and tested outside Blender.

Units are METRES throughout. The CAD is millimetres; the converter writes
glTF in metres and records that in catalog.json.

Presets live in configs/synth3d.yaml. Blender's bundled Python has no
PyYAML, so a JSON sidecar (configs/synth3d.json, written by
`python -m recog.sync_config`) is read instead when yaml is unavailable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _HAVE_YAML = True
except ImportError:            # Blender's bundled Python
    _HAVE_YAML = False


# =========================================================================== #
#  CLASSES
#
#  FasterRCNN reserves label 0 for background, so ids start at 1. These must
#  stay identical to recog.dataset.CLASS_MAP - a test enforces it.
# =========================================================================== #

CLASSES: List[str] = ["battery", "cartridge"]

# The segmentation label set. Deliberately SEPARATE from CLASSES: the
# detector's classification head is sized by recog.dataset.CLASS_MAP, and
# growing that invalidates every committed checkpoint and every published
# number. The VOC output stays two-class; these five go only to the COCO
# sidecar that the segmenter reads.
#
# Order matters. SEG_CLASSES starts with CLASSES so that ids 1 and 2 mean
# the same thing in both files.
SEG_CLASSES: List[str] = [
    "battery", "cartridge", "electronics_module",
    "placement_area", "obstruction",
]


def class_ids() -> Dict[str, int]:
    return {c: i + 1 for i, c in enumerate(CLASSES)}


def seg_class_ids() -> Dict[str, int]:
    return {c: i + 1 for i, c in enumerate(SEG_CLASSES)}


# Sub-part name -> semantic ROLE (not class). Matched in order, first hit wins.
# Roles describe CAD geometry and match the real sub-part names, e.g.
# "004695_A;1-Cell_18651" and "004697_A;2-Case10000_top".
CLASS_RULES: List[Tuple[str, str]] = [
    # NX increments instance names, so cells appear as Cell_18650, Cell_18651,
    # Cell_18650_18652 ... - match "Cell_" + digits, not a literal.
    (r"Cell[_ ]?\d+", "cell"),
    # The lid gets its OWN role so `open_case` can drop it. Both halves shared
    # one role until now, which is why every "open" cartridge rendered closed
    # and the bay was painted on the outside of the lid. Order matters: the
    # `_top` rule must precede the general one.
    (r"Case.*_top", "case_lid"),
    (r"Case.*_btm", "case"),
]

ROLE_FALLBACK = "case"          # unmatched sub-parts are treated as shell


@dataclass
class Variant:
    """
    How one CAD assembly is presented in a scene.

    keep_roles   which sub-part roles are linked into the scene
    label        class for the whole assembly, or None to label each visible
                 sub-part by its own role
    label_roles  role -> class, used when label is None
    weight       relative sampling probability
    """
    name: str
    keep_roles: Tuple[str, ...] = ("cell", "case")
    label: Optional[str] = None
    label_roles: Dict[str, str] = field(default_factory=dict)
    weight: float = 1.0


VARIANTS: List[Variant] = [
    # Sealed unit: both shell halves AND the inner cell holder (its own role,
    # `case_liner`, since task 3b - see catalog.classify_case_parts), cells
    # inside contributing no visible pixels, so the mask pass drops them
    # automatically. Matches the closed black shells in the lower half of
    # IMG_4426. Must stay visually identical to before task 3b: nothing here
    # changed which roles this variant keeps overall, only that the liner now
    # has a name of its own instead of hiding under `case`.
    Variant("assembled", keep_roles=("cell", "case", "case_lid", "case_liner"),
            label="cartridge", weight=3.0),

    # Shell removed: loose 18650 cells, scattered individually. Matches the
    # top rows of cells in the real photos.
    Variant("cells_only", keep_roles=("cell",), label=None,
            label_roles={"cell": "battery"}, weight=2.0),

    # Opened unit: the TRAY only - lid AND inner liner both dropped, so the
    # cavity is a genuine open tray (9.15mm deep, half an 18650) rather than
    # filled by the liner the case shares its role with before task 3b. The
    # module and bay proxy sit on the cavity floor, under where the liner
    # used to sit. `case_liner` is simply absent from keep_roles - the same
    # "just don't keep it" mechanism that already dropped `case_lid`.
    Variant("open_case", keep_roles=("cell", "case"), label=None,
            label_roles={"cell": "battery", "case": "cartridge"},
            weight=1.0),
]


# =========================================================================== #
#  CELL DIMENSIONS
#
#  The 18650's own CAD-measured footprint - catalog.json: every Cell_*
#  subpart across all four Anker assemblies reports extents_mm
#  [18.3, 18.3, 65.0]. Millimetres, the one exception to this module's
#  otherwise-metres convention (see the module docstring): the consumers
#  outside this package that need it (recog/calibrate_tau.py,
#  recog/seg_ablation.py) both work in mm. world.py's
#  SEAT_CELL_FOOTPRINT_M and _gate_orientation.py's orientation checks
#  derive from this same pair rather than restating it a third and fourth
#  time. One number, so it cannot silently drift the way the two
#  independent mm copies in recog/ once could have (2026-08-10
#  consolidation - see docs/superpowers/specs/2026-08-10-generalisation-
#  groundwork.md Sec4.3 for the full pre-consolidation site list).
# =========================================================================== #

CELL_W_MM: float = 18.3
CELL_H_MM: float = 65.0

# Decision 3 (2026-08-10-generalisation-decisions.md): three cell formats
# in the training mix. "18650" is CELL_W_MM/CELL_H_MM itself, converted to
# metres rather than restated a third time - the same "one authoritative
# dimension" discipline that consolidation established. 21700/21x70mm and
# 26650/26x65mm have no CAD (groundwork.md Sec4.2: a parametric cylinder
# needs no CAD); they exist here as radii/lengths a Blender primitive_
# cylinder_add can be built from directly.
CELL_FORMATS: Dict[str, Tuple[float, float]] = {
    "18650": (CELL_W_MM / 1000.0, CELL_H_MM / 1000.0),
    "21700": (0.021, 0.070),
    "26650": (0.026, 0.065),
}


# =========================================================================== #
#  SCENE CONFIG
# =========================================================================== #

@dataclass
class RenderCfg:
    res: Tuple[int, int] = (1280, 720)
    samples: int = 192
    adaptive_threshold: float = 0.01
    denoise: bool = True
    device: str = "GPU"                  # "CPU" | "GPU"
    view_transform: str = "AgX"
    exposure: float = 0.0
    max_bounces: int = 12
    clamp_indirect: float = 10.0
    film_transparent: bool = False
    persistent_data: bool = True


@dataclass
class LayoutCfg:
    area: Tuple[float, float] = (0.80, 0.45)
    mode: str = "scatter"
    pad: float = 0.008
    max_tries: int = 500
    jitter_deg: float = 2.0
    allow_90s: bool = True
    # Largest padded-footprint IoU two scatter-placed parts may share. 0 is
    # exact non-overlap, i.e. the behaviour this solver has always had.
    max_overlap_iou: float = 0.0
    jig_clearance: float = 0.004
    jig_jitter_deg: float = 1.0
    jig_depth: Tuple[float, float] = (0.006, 0.012)
    jig_wall: float = 0.003     # metres of plate material left between pockets
    # Fraction of open cartridges that get cells seated in the bay, and how
    # full those bays are. The deployed camera sees partly-filled cartridges
    # for most of every run; a set of only-empty bays would not contain the
    # case the segmenter exists to handle.
    p_seated: float = 0.5
    seated_frac: Tuple[float, float] = (0.15, 0.85)


@dataclass
class CameraCfg:
    ortho: bool = True
    height: float = 0.90
    margin_range: Tuple[float, float] = (1.02, 1.10)
    shift_range: Tuple[float, float] = (-0.006, 0.006)
    focal: float = 50.0


@dataclass
class FilterCfg:
    min_px: int = 500
    min_side: int = 6
    min_visibility: float = 0.25
    drop_truncated: bool = False
    # Longest box side / shortest. 0 disables the check. See
    # configs/synth3d.yaml for the measurement that sets it.
    max_aspect: float = 0.0


@dataclass
class ObstructionCfg:
    """Foreign matter in a cartridge bay. See bay.sample_obstructions.

    IMG_4426 shows thermal adhesive, foam pads, tape crosses and printed
    labels in every opened real bay; none of it is in the CAD.
    """
    p_none: float = 0.40           # fraction of bays left clean
    n_adhesive: Tuple[int, int] = (0, 6)
    n_foam: Tuple[int, int] = (0, 1)
    n_tape: Tuple[int, int] = (0, 2)
    n_label: Tuple[int, int] = (0, 1)
    adhesive_frac: Tuple[float, float] = (0.04, 0.14)
    foam_frac: Tuple[float, float] = (0.15, 0.35)
    tape_frac: Tuple[float, float] = (0.05, 0.12)
    label_frac: Tuple[float, float] = (0.10, 0.22)


@dataclass
class TrayRangeCfg:
    """One procedural sampling range-set - either the anchored or the
    wide population (Decision 2). config.Config.tray_anchored and
    .tray_wide are two SEPARATE instances of this dataclass, never one
    config with a "how wide" knob, because the separation has to be
    structural or Decision 2's "kept separable" requirement has nothing
    to enforce it. See bay.sample_tray for how each field is used.

    Two distinct depth-ish quantities, deliberately not one field (see
    this plan's header note): `case_half_height_mm_range` is the Z-axis
    case half-height (measured 11.1mm, all four SKUs); `bay_margin_mm_
    range` is the XY-plane depth of the module bay strip (measured
    19.45-30.75mm).
    """
    cell_formats: Tuple[str, ...] = ("18650", "21700", "26650")
    n_cols_range: Tuple[int, int] = (3, 5)
    n_rows_range: Tuple[int, int] = (1, 2)
    pitch_mm_range: Tuple[float, float] = (0.5, 2.0)
    wall_mm_range: Tuple[float, float] = (3.3, 4.7)
    bay_margin_mm_range: Tuple[float, float] = (17.0, 34.0)
    case_half_height_mm_range: Tuple[float, float] = (10.5, 11.7)
    tray_floor_mm_range: Tuple[float, float] = (1.6, 2.3)
    # Design spec Sec9.1: anchored fixes the bay to a short edge (matching
    # 4/4 measured SKUs); wide samples freely among all four edges,
    # including physically implausible ones. False = anchored's rule.
    free_bay_edge: bool = False
    # Fillet radius, in mm, rolled onto the four TOP edges of the lid -
    # the only part of a procedural cartridge a SEALED unit shows the
    # camera. Default (0, 0) is degenerate on purpose: a config written
    # before this field existed samples 0.0 and world.build_procedural_
    # tray skips the bevel entirely, so the lid is the same planar cuboid
    # it always was.
    #
    # It exists because the four measured Anker lids are not planar. Off
    # the glTF, all four: long-edge fillet radius 11.10mm - the ENTIRE
    # lid height, 27-36% of the half-width - and 89% of each lid's
    # upward-facing polygons have a z-normal below 0.95 (median 0.69),
    # against the procedural cuboid's 0%. Rendered, a sealed procedural
    # crop's internal luminance spread (p95-p05 over the unit's own
    # cartridge mask) has median 0.0272 where a sealed CAD crop's has
    # 0.2719 - 10x, and the ONE appearance axis measured where the
    # procedural sealed population fails to cover the CAD one; backdrop,
    # lighting, exposure, zoom, shell preset and median shell brightness
    # are all already matched to sampling noise. See
    # docs/superpowers/specs/2026-08-11-sealed-unit-experiment.md.
    lid_crown_mm_range: Tuple[float, float] = (0.0, 0.0)


def _wide_tray_defaults() -> "TrayRangeCfg":
    """Decision 2's "well outside" population - unusual aspect ratios,
    thicker/thinner walls, denser packing, and (free_bay_edge) module
    positions no real SKU shows."""
    return TrayRangeCfg(
        n_cols_range=(2, 8), n_rows_range=(1, 4),
        pitch_mm_range=(0.3, 6.0), wall_mm_range=(1.5, 9.0),
        bay_margin_mm_range=(6.0, 60.0),
        case_half_height_mm_range=(4.0, 28.0),
        tray_floor_mm_range=(0.8, 4.5), free_bay_edge=True)


@dataclass
class Config:
    render: RenderCfg = field(default_factory=RenderCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    obstruction: ObstructionCfg = field(default_factory=ObstructionCfg)
    tray_anchored: TrayRangeCfg = field(default_factory=TrayRangeCfg)
    tray_wide: TrayRangeCfg = field(default_factory=_wide_tray_defaults)
    param_space: Dict[str, Any] = field(default_factory=dict)
    backdrops: Dict[str, dict] = field(default_factory=dict)
    lighting: Dict[str, dict] = field(default_factory=dict)
    materials: Dict[str, dict] = field(default_factory=dict)
    role_materials: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def config_to_dict(cfg: Config) -> dict:
    """Round-trippable plain-data form, used to write the JSON sidecar."""
    return cfg.to_dict()


# =========================================================================== #
#  LOADING
# =========================================================================== #

_SECTIONS = {"render": RenderCfg, "layout": LayoutCfg,
             "camera": CameraCfg, "filter": FilterCfg,
             "obstruction": ObstructionCfg,
             "tray_anchored": TrayRangeCfg, "tray_wide": TrayRangeCfg}
_PASSTHROUGH = ("param_space", "backdrops", "lighting",
                "materials", "role_materials")
_TUPLE_FIELDS = {"res", "area", "margin_range", "shift_range", "jig_depth",
                 "n_adhesive", "n_foam", "n_tape", "n_label",
                 "adhesive_frac", "foam_frac", "tape_frac", "label_frac",
                 "seated_frac", "cell_formats", "n_cols_range",
                 "n_rows_range", "pitch_mm_range", "wall_mm_range",
                 "bay_margin_mm_range", "case_half_height_mm_range",
                 "tray_floor_mm_range", "lid_crown_mm_range"}


def default_config_path() -> Path:
    """configs/synth3d.yaml, resolved relative to the project root."""
    return Path(__file__).resolve().parents[2] / "configs" / "synth3d.yaml"


def _read_raw(path: Path) -> dict:
    """YAML when available, else the JSON sidecar next to it."""
    if _HAVE_YAML and path.suffix in (".yaml", ".yml") and path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise RuntimeError(
            f"{sidecar} not found and PyYAML is unavailable in this "
            f"interpreter. Run:  python -m recog.sync_config"
        )
    if path.is_file() and sidecar.stat().st_mtime < path.stat().st_mtime:
        raise RuntimeError(
            f"{sidecar} is older than {path.name}; the config is stale. "
            f"Run:  python -m recog.sync_config"
        )
    with sidecar.open("r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def load_config(path: "str | os.PathLike | None" = None) -> Config:
    """Build a Config from configs/synth3d.yaml (or its JSON sidecar)."""
    p = Path(path) if path is not None else default_config_path()
    raw = _read_raw(p)

    unknown = set(raw) - set(_SECTIONS) - set(_PASSTHROUGH)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {p.name}: {sorted(unknown)}. "
            f"Valid keys: {sorted(set(_SECTIONS) | set(_PASSTHROUGH))}"
        )

    kwargs: Dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        valid = {f for f in cls.__dataclass_fields__}
        bad = set(section) - valid
        if bad:
            raise ValueError(f"unknown key(s) in {p.name}:{name}: {sorted(bad)}")
        coerced = {k: (tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v)
                   for k, v in section.items()}
        kwargs[name] = cls(**coerced)
    for name in _PASSTHROUGH:
        kwargs[name] = raw.get(name) or {}

    return Config(**kwargs)
