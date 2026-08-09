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
    # Sealed unit: both shell halves, cells inside contributing no visible
    # pixels, so the mask pass drops them automatically. Matches the closed
    # black shells in the lower half of IMG_4426.
    Variant("assembled", keep_roles=("cell", "case", "case_lid"),
            label="cartridge", weight=3.0),

    # Shell removed: loose 18650 cells, scattered individually. Matches the
    # top rows of cells in the real photos.
    Variant("cells_only", keep_roles=("cell",), label=None,
            label_roles={"cell": "battery"}, weight=2.0),

    # Opened unit: the TRAY only, lid dropped, so the cavity is visible and
    # the module and bay proxy sit inside it rather than on a closed lid.
    Variant("open_case", keep_roles=("cell", "case"), label=None,
            label_roles={"cell": "battery", "case": "cartridge"},
            weight=1.0),
]


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
class Config:
    render: RenderCfg = field(default_factory=RenderCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    obstruction: ObstructionCfg = field(default_factory=ObstructionCfg)
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
             "obstruction": ObstructionCfg}
_PASSTHROUGH = ("param_space", "backdrops", "lighting",
                "materials", "role_materials")
_TUPLE_FIELDS = {"res", "area", "margin_range", "shift_range", "jig_depth",
                 "n_adhesive", "n_foam", "n_tape", "n_label",
                 "adhesive_frac", "foam_frac", "tape_frac", "label_frac",
                 "seated_frac"}


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
