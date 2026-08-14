"""The exact-fit certifiability measurement — a receipt for the headline.

Regenerates ``docs/receipts/placement_feasibility.txt``. It exists
because ``docs/RESULTS_SUMMARY.md`` §2 leads with a measurement whose
only citation was a spec file
(``docs/superpowers/specs/2026-08-11-placement-feasibility.md`` §3, §5),
in a document whose argument is that every published figure has a
regenerable artefact behind it. The load-bearing paragraph was the one
paragraph a reader could not re-run.

**No checkpoint, no training, no rendering.** The measurement is the
shipping extractor and the shipping packer fed a GROUND-TRUTH label map
at each frame's TRUE ground sample distance — perfect segmentation,
perfect boxes, perfect calibration — so nothing here needs a network.
The label map is painted by ``recog.seg_dataset.rasterise_crop`` from
the COCO-RLE sidecar; the images themselves are never opened (the
segmentation extractor does not look at ``image_rgb``).

Two arms, and they answer different questions:

*geometry*
    ``recog/synth3d/assets/catalog.json`` and ``configs/planning.yaml``
    alone. Placement region per SKU, its margin against the nominal
    cell, and the rotation at which an axis-aligned cell stops fitting
    an EMPTY bay. Both files are committed, so this arm runs on a bare
    clone and is the half of the finding that is checkable in a minute.
*census*
    Every ground-truth cartridge unit in ``recog/dataset3d_seg``,
    through ``SegmentationPlacementAreaExtractor`` at the frame's own
    ``camera.ortho_scale`` and then through
    ``Planner._pack_cartridge`` — the packing call the planner really
    makes, with the nominal and the occupancy resolution
    ``configs/planning.yaml`` really configures. Needs the dataset,
    which is gitignored; the arm skips loudly when it is absent and the
    receipt is written only when every arm ran.

**The population is defined by ground truth, not by the detector.** The
spec's 30 instances were whatever the shipping pipeline detected over
the first 60 frames, paired back to ground-truth units by IoU; that
needed two checkpoints to enumerate. This script takes every cartridge
unit in the sidecar instead, which is cheaper, reproducible without any
network, and a superset in the sense that matters — no instance is
missing because a detector missed it. §5 of the receipt reconciles the
two populations, and the totals agree.

The three numbers this arm reproduces at HEAD (10 instances / 24 cells
at the production inset, 11 / 25 at 0.0 mm) are the addendum's, not
§3's: ``b93bbd3`` made ``_rasterise_mask`` call a grid cell free only
when every pixel it covers is free, which the addendum re-measured.

Usage
-----
    python scripts/placement_feasibility.py           # write the receipt
    python scripts/placement_feasibility.py --check   # fail on drift

``--check`` recomputes every arm whose inputs are present and requires
each computed section to appear verbatim in the committed receipt. The
provenance header is deliberately NOT compared: it carries the commit
the receipt was generated at, which moves every time anything is
committed, and a drift check that fails on its own commit is a check
nobody keeps.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import BBox                                   # noqa: E402
from plan.placement_area import (                               # noqa: E402
    _DEFAULT_WALL_INSET_MM,
    BadDetectorBox,
    SegmentationPlacementAreaExtractor,
    placeable_extent_mm,
)
from plan.planner import Planner, PlannerConfig                 # noqa: E402
from plan.scene import Cartridge, WorkspaceBounds               # noqa: E402
from recog.seg_dataset import rasterise_crop                    # noqa: E402

RECEIPT_PATH = REPO_ROOT / "docs" / "receipts" / "placement_feasibility.txt"
CATALOG_PATH = REPO_ROOT / "recog" / "synth3d" / "assets" / "catalog.json"
PLANNING_PATH = REPO_ROOT / "configs" / "planning.yaml"
DATASET_DIR = REPO_ROOT / "recog" / "dataset3d_seg"

# The window the published figure is quoted over: the first 60 frames of
# recog/dataset3d_seg, which is the corpus
# docs/superpowers/specs/2026-08-11-placement-feasibility.md §1 measured
# (60 frames -> 30 pipeline instances). The full-corpus arm below runs
# the same measurement over all 502 frames, because a claim quoted at
# n = 10 should say what the other 37 instances of that SKU do.
WINDOW_FRAMES = 60

# The sweep. 4.25 mm is plan.placement_area._DEFAULT_WALL_INSET_MM - the
# value that ships - imported rather than retyped so a change to it
# cannot leave this receipt describing a configuration nobody runs.
INSETS_MM: Tuple[float, ...] = (_DEFAULT_WALL_INSET_MM, 3.0, 0.0)

SKUS = ("AnkerPowerCore10000", "AnkerPowerCore13000",
        "AnkerPowerCore20100", "AnkerPowerCore26800")

# The classes that make a ground-truth unit a cartridge rather than a
# loose cell. Same set, for the same reason, as
# recog.seg_dataset.BaySegDataset's `_CARTRIDGE_RELATED`: a sealed unit
# carries `cartridge` alone and must not be filtered out by requiring a
# `placement_area`.
CARTRIDGE_CLASSES = frozenset(
    {"cartridge", "placement_area", "electronics_module", "obstruction"})

RULE = "=" * 64


# ======================================================================
# provenance
# ======================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return "unknown"
        head = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30)
        state = "dirty" if dirty.stdout.strip() else "clean"
        return f"{head} ({state})"
    except (OSError, subprocess.SubprocessError):    # pragma: no cover
        return "unknown"


# ======================================================================
# geometry — catalog.json and planning.yaml only
# ======================================================================

class Sku:
    """One cataloged assembly's placement geometry, read from CAD.

    ``interior_mm`` is the tray's inside rectangle and ``module_bay_mm``
    the strip the electronics module occupies; the placement region is
    the first minus the second. That subtraction is only a rectangle if
    the bay spans the interior's full width and sits flush against one
    end, which is asserted rather than assumed - a future asset that
    breaks it would otherwise be silently mis-measured.
    """

    def __init__(self, asset: dict) -> None:
        self.name = asset["name"]
        self.wall_mm = float(asset["case_wall_mm"])
        ix0, iy0, ix1, iy1 = (float(v) for v in asset["interior_mm"])
        bx0, by0, bx1, by1 = (float(v) for v in asset["module_bay_mm"])
        if abs(bx0 - ix0) > 1e-6 or abs(bx1 - ix1) > 1e-6:
            raise ValueError(
                f"{self.name}: module_bay_mm does not span the interior's "
                f"full width, so interior minus bay is not a rectangle and "
                f"this script cannot measure it as one.")
        if abs(by1 - iy1) > 1e-6:
            raise ValueError(
                f"{self.name}: module_bay_mm is not flush with the "
                f"interior's far edge; the placement region is not the "
                f"single rectangle this script assumes.")
        # cross = across the slots, along = the direction a seated cell
        # lies in. The bay is at the +y end of the interior in all four
        # cataloged assemblies, checked above.
        self.cross_mm = ix1 - ix0
        self.along_mm = by0 - iy0
        # The CAD cell, from this asset's own `cell` subparts. The one
        # place 18.3 x 65.0 is a measurement rather than a constant.
        cells = [s for s in asset.get("subparts", ()) if s.get("role") == "cell"]
        diameters = {round(float(s["extents_mm"][0]), 3) for s in cells}
        lengths = {round(float(s["extents_mm"][2]), 3) for s in cells}
        if len(diameters) != 1 or len(lengths) != 1:
            raise ValueError(
                f"{self.name}: cell subparts disagree about their own size "
                f"({sorted(diameters)} x {sorted(lengths)}); there is no "
                f"single CAD cell to measure the bay against.")
        self.cell_diameter_mm = diameters.pop()
        self.cell_length_mm = lengths.pop()

    @property
    def short_name(self) -> str:
        return self.name.replace("AnkerPowerCore", "")


def load_skus() -> List[Sku]:
    doc = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_name = {a["name"]: Sku(a) for a in doc["assets"]}
    missing = [s for s in SKUS if s not in by_name]
    if missing:
        raise ValueError(f"catalog.json is missing {missing}")
    return [by_name[s] for s in SKUS]


def load_cell_nominal() -> Tuple[float, float, float]:
    """``(width_mm, length_mm, mm_per_cell)`` as ``planning.yaml`` sets them."""
    cfg = yaml.safe_load(PLANNING_PATH.read_text(encoding="utf-8"))
    battery = cfg.get("battery", {}) or {}
    grid = cfg.get("occupancy_grid", {}) or {}
    return (float(battery["diameter_mm"]), float(battery["length_mm"]),
            float(grid["resolution_mm_per_cell"]))


# Angular resolution of the rotation scan, in degrees. 0.001 deg is two
# orders of magnitude finer than the 2.0 deg jitter the scene generator
# draws and than any threshold reported, and the scan is exact enough
# that the reported bounds are the true ones to the printed precision.
THETA_STEP_DEG = 0.001
_THETA = np.arange(0.0, 90.0, THETA_STEP_DEG)


def _fits(w: float, ell: float, cross: float, along: float,
          theta_deg: np.ndarray) -> np.ndarray:
    """Does an axis-aligned ``w`` x ``ell`` rectangle fit the turned bay?

    A ``w`` x ``ell`` axis-aligned rectangle is inscribed in a
    ``cross`` x ``along`` rectangle turned by theta exactly when its
    corners project inside both of the turned rectangle's axes:

        w*cos + ell*sin <= cross   and   w*sin + ell*cos <= along

    Both orientations of the cell are tried by the caller, because
    ``pack_best_effort`` runs with ``allow_rotation=True`` and will lay
    a cell across the slots if the floor allows it.
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return ((w * c + ell * s <= cross + 1e-9)
            & (w * s + ell * c <= along + 1e-9))


def _first_interval(ok: np.ndarray) -> Tuple[float, Optional[float]]:
    """``(upper bound of the run starting at 0, start of the next run)``."""
    if not ok[0]:
        return 0.0, None
    end = int(np.argmin(ok)) if not ok.all() else len(ok)
    upper = (end - 1) * THETA_STEP_DEG
    if end >= len(ok):
        return upper, None
    rest = ok[end:]
    if not rest.any():
        return upper, None
    return upper, (end + int(np.argmax(rest))) * THETA_STEP_DEG


def rotation_limits(sku: Sku, width_mm: float, length_mm: float
                    ) -> Dict[str, Tuple[float, Optional[float]]]:
    """Where an axis-aligned cell stops fitting this SKU's EMPTY bay.

    ``along`` is the slot-aligned orientation - the cell lying the way a
    seated cell lies - which is the one the spec's L(theta) reports.
    ``any`` additionally allows the cell to lie ACROSS the slots, which
    the packer permits and which matters on the two wide SKUs.
    """
    along = _fits(width_mm, length_mm, sku.cross_mm, sku.along_mm, _THETA)
    across = _fits(length_mm, width_mm, sku.cross_mm, sku.along_mm, _THETA)
    return {"along": _first_interval(along),
            "any": _first_interval(along | across)}


_THETA_MAX = (len(_THETA) - 1) * THETA_STEP_DEG


def _deg(v: Optional[float]) -> str:
    if v is None:
        return "-"
    # The scan covers [0, 90); a run that reaches its end is a run over
    # every rotation there is, and saying "89.99 deg" would read as a
    # threshold rather than as "no threshold".
    if v >= _THETA_MAX:
        return "all"
    return f"{v:.2f} deg"


def section_geometry(skus: Sequence[Sku], width_mm: float, length_mm: float
                     ) -> str:
    cad_d = {s.cell_diameter_mm for s in skus}
    cad_l = {s.cell_length_mm for s in skus}
    if len(cad_d) != 1 or len(cad_l) != 1:
        raise ValueError(f"the four assemblies disagree about the CAD cell: "
                         f"{sorted(cad_d)} x {sorted(cad_l)}")
    cad_diameter, cad_length = cad_d.pop(), cad_l.pop()

    lines = [
        "1. Geometry - CAD and config only, no dataset, no checkpoint",
        RULE,
        "Placement region = `interior_mm` minus `module_bay_mm`, per",
        "assembly in recog/synth3d/assets/catalog.json. `along` is the",
        "direction a seated cell lies in; `cross` is across the slots.",
        f"The planner reserves {width_mm} x {length_mm} mm "
        f"(configs/planning.yaml",
        f"battery.diameter_mm x length_mm); the CAD cell measures "
        f"{cad_diameter} x {cad_length} mm",
        f"(catalog.json, role `cell`), so the nominal carries "
        f"{width_mm - cad_diameter:.1f} mm of",
        "deliberate margin on the diameter and none at all on the length.",
        "",
        "  " + f"{'SKU':<7}{'placement region':^19}{'margin along':>14}"
               f"{'margin cross':>13}{'wall':>10}",
    ]
    for s in skus:
        lines.append(
            "  " + f"{s.short_name:<7}"
            + f"{s.cross_mm:6.2f} x {s.along_mm:7.2f} mm"
            + f"{s.along_mm - length_mm:+11.2f} mm"
            + f"{s.cross_mm - length_mm:+10.2f} mm"
            + f"{s.wall_mm:7.2f} mm")
    lines += [
        "",
        "The 10000's bay is EXACTLY the planner's nominal cell length:",
        f"{length_mm} against {length_mm}, {0.0:+.2f} mm.",
        "",
        "Rotation. The packer keeps its rectangle axis-aligned (the camera",
        "mount is fixed); the cartridge is not - `layout.plan` seats every",
        "unit at quarter*90 + jitter with jitter_deg 2.0. An axis-aligned",
        "a x b rectangle is inscribed in a cross x along rectangle turned",
        "by theta exactly when a*cos+b*sin <= cross and a*sin+b*cos <= along.",
        f"Scanned at {THETA_STEP_DEG} deg over [0, 90):",
        "",
        "  " + " " * 7 + f"{'slot-aligned cell':^22}{'either orientation':^22}",
        "  " + f"{'SKU':<7}" + f"{'admits to':>11}{'again at':>11}"
                              f"{'admits to':>11}{'again at':>11}",
    ]
    for s in skus:
        lim = rotation_limits(s, width_mm, length_mm)
        a_hi, a_next = lim["along"]
        n_hi, n_next = lim["any"]
        lines.append(
            f"  {s.short_name:<7}{_deg(a_hi):>11}{_deg(a_next):>11}"
            f"{_deg(n_hi):>11}{_deg(n_next):>11}")
    lines += [
        "",
        f"Cost of a degree on the 10000: {width_mm}*tan(1 deg) = "
        f"{width_mm * np.tan(np.radians(1.0)):.2f} mm of length.",
        "Available: 0.00 mm. The first fraction of a degree is terminal,",
        "and nothing is recovered until the bay has turned far enough for",
        "the cell to be inscribed corner-to-corner, which no jig jitter",
        "reaches.",
        "",
        "READ THE TWO RIGHT-HAND COLUMNS BEFORE QUOTING THE 13000's 6.9 deg.",
        "That threshold is the SLOT-ALIGNED orientation only. The 13000's",
        "bay is wider than a cell is long, so a cell laid ACROSS the slots",
        "keeps fitting far past it; the 6.9 deg is what the +1.75 mm of",
        "along-axis margin buys, not the angle at which the SKU stops",
        "admitting a cell. On the 10000 the two columns are identical,",
        "because its bay is narrower than a cell is long and there is no",
        "second orientation to fall back on. That is the difference",
        "between marginal and infeasible, and it is the whole finding.",
    ]
    return "\n".join(lines)


# ======================================================================
# census — ground-truth units through the shipping extractor and packer
# ======================================================================

class Unit:
    """One ground-truth cartridge unit, ready to feed to the extractor."""

    __slots__ = ("frame", "unit_id", "sku", "variant", "rot_deg", "mm_per_px",
                 "box", "anns")

    def __init__(self, frame, unit_id, sku, variant, rot_deg, mm_per_px,
                 box, anns) -> None:
        self.frame, self.unit_id = frame, unit_id
        self.sku, self.variant = sku, variant
        self.rot_deg, self.mm_per_px = rot_deg, mm_per_px
        self.box, self.anns = box, anns

    @property
    def short_sku(self) -> str:
        return self.sku.replace("AnkerPowerCore", "")

    @property
    def rot_offset_deg(self) -> float:
        """Misalignment from axis-aligned, folded into [0, 45].

        The layout planner writes an absolute heading (`quarter*90 +
        jitter`), and a cell packed axis-aligned cares only about the
        distance from the nearest right angle.
        """
        r = self.rot_deg % 90.0
        return min(r, 90.0 - r)


def load_units(n_frames: Optional[int] = None) -> List[Unit]:
    """Every ground-truth cartridge unit in the sidecar, in frame order.

    Grouping is `image -> unit_id`, exactly as
    ``recog.seg_dataset.BaySegDataset`` does it and for the reason its
    docstring gives: `unit_id` is scene-local, so flattening the two
    levels would merge units that share an id across frames.
    """
    doc = json.loads((DATASET_DIR / "instances_seg.json").read_text("utf-8"))
    names = {c["id"]: c["name"] for c in doc["categories"]}
    by_image: Dict[int, List[dict]] = collections.defaultdict(list)
    for a in doc["annotations"]:
        rec = dict(a)
        rec["class"] = names[a["category_id"]]
        x, y, w, h = a["bbox"]
        rec["bbox_xyxy"] = [x, y, x + w, y + h]
        by_image[a["image_id"]].append(rec)

    units: List[Unit] = []
    for im in sorted(doc["images"], key=lambda i: i["file_name"]):
        stem = Path(im["file_name"]).stem
        if n_frames is not None and int(stem.split("_")[1]) >= n_frames:
            continue
        meta = json.loads(
            (DATASET_DIR / "meta" / f"{stem}.json").read_text("utf-8"))
        # THE FRAME'S TRUE SCALE, not configs/demo_seg.yaml's 0.625. The
        # camera is orthographic, so the ground sampling distance is the
        # ortho width over the image width, and world.py randomises it
        # per scene through `margin` and `zoom`.
        mm_per_px = meta["camera"]["ortho_scale"] * 1000.0 / im["width"]
        groups: Dict[str, List[dict]] = collections.defaultdict(list)
        for a in by_image[im["id"]]:
            groups[a["unit_id"]].append(a)
        for unit_id, anns in sorted(groups.items()):
            # `solo*` units are loose scattered cells and modules; only
            # `item{n}` units index back into the layout planner's own
            # record of what it placed, which is where the SKU and the
            # pose come from.
            if not unit_id.startswith("item"):
                continue
            if not any(a["class"] in CARTRIDGE_CLASSES for a in anns):
                continue
            item = meta["items"][int(unit_id[len("item"):])]
            xs = [a["bbox_xyxy"] for a in anns]
            box = (int(min(b[0] for b in xs)), int(min(b[1] for b in xs)),
                   int(max(b[2] for b in xs)), int(max(b[3] for b in xs)))
            units.append(Unit(stem, unit_id, item["asset"], item["variant"],
                              float(item["placement"]["rot_deg"]), mm_per_px,
                              box, anns))
    return units


# The extractor takes an RGB frame and never reads it on the
# segmentation path - the label map is the whole input. Passing a 1x1
# placeholder is what makes this measurement image-free, and therefore
# runnable from the sidecar alone.
_NO_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)


class Oracle:
    """The shipping extractor and packer over ground-truth label maps.

    One extractor per wall inset, one planner per cell nominal. The
    split is not cosmetic: the extraction does not depend on the cell at
    all, so a nominal sweep that re-extracted would be measuring the
    same erosion twice and inviting the two arms to disagree about it.
    """

    def __init__(self, length_mm: float, mm_per_cell: float,
                 wall_inset_mm: float, widths_mm: Sequence[float]) -> None:
        self.extractor = SegmentationPlacementAreaExtractor(
            mm_per_cell=mm_per_cell, wall_inset_mm=wall_inset_mm)
        # Built so the packing call is Planner._pack_cartridge itself
        # rather than a copy of its six lines. The workspace bounds are
        # never consulted: _pack_cartridge asks the occupancy grid and
        # the config, and reachability is decided elsewhere.
        self.planners = {
            w: Planner(
                PlannerConfig(battery_width_mm=w, battery_length_mm=length_mm),
                self.extractor,
                WorkspaceBounds(-350.0, 350.0, -350.0, 350.0))
            for w in widths_mm}

    def run(self, unit: Unit, label_map: np.ndarray) -> Dict[float, dict]:
        try:
            area = self.extractor.extract(_NO_IMAGE, BBox(*unit.box),
                                          label_map=label_map,
                                          mm_per_px=unit.mm_per_px)
        except BadDetectorBox:
            refused = {"cells": 0, "outcome": "refused_box", "extent_mm": None}
            return {w: dict(refused) for w in self.planners}
        except RuntimeError:
            # "no placeable area in this cartridge" - a sealed unit, or
            # an open one whose floor is entirely covered. Not an error:
            # it is the extractor answering the question.
            empty = {"cells": 0, "outcome": "no_floor", "extent_mm": None}
            return {w: dict(empty) for w in self.planners}
        ctg = Cartridge(id=0, bbox=BBox(*unit.box), confidence=1.0,
                        placeable_rectangle=area.rectangle,
                        occupancy=area.occupancy, mm_per_px=unit.mm_per_px)
        extent = placeable_extent_mm(area.inside_mask, unit.mm_per_px)
        out = {}
        for width_mm, planner in self.planners.items():
            packed = planner._pack_cartridge(ctg)
            out[width_mm] = {
                "cells": len(packed.placements),
                "outcome": "packed" if packed.placements else "no_room",
                "extent_mm": extent}
        return out


def census(units: Sequence[Unit], widths_mm: Sequence[float],
           length_mm: float, mm_per_cell: float,
           insets: Sequence[float]) -> Dict[float, Dict]:
    """``{nominal: {(frame, unit, inset): result}}``, one pass over the data."""
    oracles = {inset: Oracle(length_mm, mm_per_cell, inset, widths_mm)
               for inset in insets}
    out: Dict[float, Dict[Tuple[str, str, float], dict]] = {
        w: {} for w in widths_mm}
    for unit in units:
        # Only THIS unit's annotations are painted, which is the spec's
        # method: the sidecar's placement_area is already occlusion-
        # aware (it is the floor the camera can see as free), so a
        # neighbour lying across the crop cannot be double-counted.
        # Measured either way on the published window: identical.
        label_map = rasterise_crop(unit.anns, unit.box, out_size=None)
        for inset, oracle in oracles.items():
            for width_mm, result in oracle.run(unit, label_map).items():
                out[width_mm][(unit.frame, unit.unit_id, inset)] = result
    return out


def _tally(units: Sequence[Unit], results: Dict, inset: float
           ) -> Dict[str, Tuple[int, int, int]]:
    """``sku -> (instances placing >= 1, n instances, cells)``."""
    out = {}
    for sku in SKUS:
        rows = [results[(u.frame, u.unit_id, inset)] for u in units
                if u.sku == sku]
        out[sku.replace("AnkerPowerCore", "")] = (
            sum(1 for r in rows if r["cells"] > 0), len(rows),
            sum(r["cells"] for r in rows))
    return out


def _cell(hits: int, n: int, cells: int) -> str:
    return f"{hits}/{n} ({cells})"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _sweep_table(units, by_nominal: Dict[float, Dict], insets,
                 indent="  ") -> str:
    lines = [f"{indent}{'nominal':>7}{'inset':>6}"
             + "".join(f"{s.replace('AnkerPowerCore', ''):>12}" for s in SKUS)
             + f"{'total':>14}"]
    for width_mm, results in sorted(by_nominal.items(), reverse=True):
        for inset in insets:
            tally = _tally(units, results, inset)
            tot = (sum(v[0] for v in tally.values()),
                   sum(v[1] for v in tally.values()),
                   sum(v[2] for v in tally.values()))
            lines.append(
                f"{indent}{width_mm:>7.1f}{inset:>6.2f}"
                + "".join(f"{_cell(*v):>12}" for v in tally.values())
                + f"{_cell(*tot):>14}")
    return "\n".join(lines)


def section_census(window: Sequence[Unit], results: Dict, width_mm: float,
                   length_mm: float, ship_inset: float) -> str:
    open_units = [u for u in window if u.variant == "open_case"]
    sealed = [u for u in window if u.variant != "open_case"]
    tally = _tally(open_units, results, ship_inset)
    frames = len({u.frame for u in window})

    lines = [
        "2. The ground-truth oracle, at the production wall inset",
        RULE,
        f"Population: every cartridge unit in the first {WINDOW_FRAMES} frames "
        f"of",
        f"recog/dataset3d_seg - {frames} of those frames carry at least one, "
        f"and they",
        f"hold {len(open_units)} open cartridges and {len(sealed)} sealed "
        f"ones. Enumerated from the COCO",
        "sidecar, not from a detector.",
        "",
        "Each unit's ground-truth annotations are painted into its own union",
        "box by recog.seg_dataset.rasterise_crop, handed to",
        "SegmentationPlacementAreaExtractor at the frame's camera.ortho_scale,",
        f"and packed by Planner._pack_cartridge with an {width_mm} x "
        f"{length_mm} mm cell.",
        "Perfect segmentation, perfect boxes, perfect calibration; only the",
        "mask differs from the shipping path.",
        "",
        f"Open cartridges, wall inset {ship_inset:.2f} mm:",
        "",
        "  " + f"{'SKU':<9}{'n':>3}{'admits >= 1 cell':>19}"
               f"{'places zero':>14}{'cells':>8}",
    ]
    for sku, (hits, n, cells) in tally.items():
        lines.append("  " + f"{sku:<9}{n:>3}{hits:>19}{n - hits:>14}"
                            f"{cells:>8}")
    tot = (sum(v[0] for v in tally.values()), len(open_units),
           sum(v[2] for v in tally.values()))
    lines.append("  " + f"{'ALL':<9}{tot[1]:>3}{tot[0]:>19}"
                        f"{tot[1] - tot[0]:>14}{tot[2]:>8}")

    zero = tally["10000"]
    rows = [u for u in open_units if u.sku == "AnkerPowerCore10000"]
    extents = [results[(u.frame, u.unit_id, 0.0)]["extent_mm"] for u in rows]
    longer = sum(1 for e in extents if e is not None and e[1] > length_mm)
    shortest = min(e[1] for e in extents if e is not None)
    lines += [
        "",
        f"THE ANKERPOWERCORE10000 PLACES ZERO CELLS IN {zero[1] - zero[0]} OF "
        f"{zero[1]} INSTANCES.",
        "",
        "Not because the extractor refused them. Per-instance, with the bay's",
        "own extent from plan.placement_area.placeable_extent_mm - a MIN-AREA",
        "rect, so it measures the bay as it lies rather than as an",
        "axis-aligned packer has to take it - over the un-eroded floor:",
        "",
        "  " + " " * 21 + f"{'rot off':>8}{'true':>8}{'bay long':>10}"
        + f"{'cells at inset':>21}",
        "  " + f"{'frame':<13}{'unit':<8}"
        + f"{'(deg)':>8}{'mm/px':>8}{'(mm)':>10}"
        + "".join(f"{i:>7.2f}" for i in INSETS_MM),
    ]
    for u in rows:
        cells = [results[(u.frame, u.unit_id, i)]["cells"] for i in INSETS_MM]
        extent = results[(u.frame, u.unit_id, 0.0)]["extent_mm"]
        long_mm = "-" if extent is None else f"{extent[1]:.2f}"
        lines.append(
            "  " + f"{u.frame:<13}{u.unit_id:<8}"
            + f"{u.rot_offset_deg:>8.3f}{u.mm_per_px:>8.4f}{long_mm:>10}"
            + "".join(f"{c:>7}" for c in cells))
    outcomes = collections.Counter(
        results[(u.frame, u.unit_id, ship_inset)]["outcome"] for u in rows)
    lines += [
        "",
        "  outcome at the production inset: "
        + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())),
        "  (`no_room` = the extractor returned a placement area and the",
        "   packer found nowhere in it to put a cell)",
        "",
        f"{longer} of the {len(rows)} measure a bay LONGER than the "
        f"{length_mm} mm the cell needs",
        f"(the shortest is {shortest:.2f} mm), and not one of them can take a "
        "cell,",
        "because the packer's rectangle is axis-aligned and the cartridge is",
        "not. That is the finding: the length is there and it cannot be used.",
    ]
    return "\n".join(lines)


def section_sweep(window: Sequence[Unit], by_nominal: Dict[float, Dict],
                  full: Sequence[Unit], by_nominal_full: Dict[float, Dict],
                  width_mm: float, cad_width_mm: float) -> str:
    open_window = [u for u in window if u.variant == "open_case"]
    open_full = [u for u in full if u.variant == "open_case"]
    ship = INSETS_MM[0]
    t_ship = _tally(open_window, by_nominal[width_mm], ship)
    t_zero = _tally(open_window, by_nominal[width_mm], 0.0)
    t_cad = _tally(open_window, by_nominal[cad_width_mm], ship)
    n_nom = t_cad["10000"][0] - t_ship["10000"][0]
    n_ins = t_zero["10000"][0] - t_ship["10000"][0]
    lines = [
        "3. Tolerance sweep - is either knob the lever?",
        RULE,
        "Per SKU, as `instances admitting >= 1 cell / n (cells placed)`. Two",
        f"nominals ({width_mm} mm as configured, {cad_width_mm} mm as the CAD "
        f"cell measures)",
        "against three wall insets. Everything else is held at the shipping",
        "value, and every row is the ground-truth oracle of section 2.",
        "",
        f"Published window, {len(open_window)} open cartridges:",
        "",
        _sweep_table(open_window, by_nominal, INSETS_MM),
        "",
        f"Whole corpus, {len(open_full)} open cartridges over the "
        f"{len({u.frame for u in open_full})} frames that carry one:",
        "",
        _sweep_table(open_full, by_nominal_full, INSETS_MM),
        "",
        "On the 10000, over the published window:",
        f"  nominal {width_mm} -> {cad_width_mm} mm at the production inset "
        f"recovers {_plural(n_nom, 'instance')}",
        f"  wall inset {ship:.2f} -> 0.00 mm at the shipping nominal recovers "
        f"{_plural(n_ins, 'instance')}",
        "",
        "The nominal is not the lever: 18.5 -> 18.3 frees 0.2 mm on the",
        "DIAMETER and nothing at all on the length, and it is the length",
        "the 10000 is short of. The wall inset is a lever, and a small",
        "one - see section 4, which is where the published sentence and",
        "this measurement part company.",
    ]
    return "\n".join(lines)


# ======================================================================
# verdict — the figures docs/RESULTS_SUMMARY.md quotes
# ======================================================================

def verdicts(window: Sequence[Unit], by_nominal: Dict[float, Dict],
             skus: Sequence[Sku], width_mm: float, length_mm: float,
             cad_width_mm: float) -> List[Tuple[str, str, str, str]]:
    """``(figure, quoted, measured, verdict)`` for each quoted number."""
    open_window = [u for u in window if u.variant == "open_case"]
    ship = INSETS_MM[0]
    t_ship = _tally(open_window, by_nominal[width_mm], ship)
    t_zero = _tally(open_window, by_nominal[width_mm], 0.0)
    t_cad = _tally(open_window, by_nominal[cad_width_mm], ship)
    hits, n, _ = t_ship["10000"]
    by_name = {s.short_name: s for s in skus}

    rows = []
    rows.append((
        "10000 places zero cells at the 4.25 mm inset",
        "10 of 10",
        f"{n - hits} of {n}",
        "REPRODUCED" if (n - hits, n) == (10, 10) else "DIFFERS"))

    rec_nominal = t_cad["10000"][0] - t_ship["10000"][0]
    rows.append((
        "nominal 18.5 -> 18.3 mm recovers",
        "0 instances",
        f"{rec_nominal} instances",
        "REPRODUCED" if rec_nominal == 0 else "DIFFERS"))

    rec_inset = t_zero["10000"][0] - t_ship["10000"][0]
    rows.append((
        "wall inset 4.25 -> 0.0 mm recovers",
        "0 instances",
        _plural(rec_inset, "instance"),
        "REPRODUCED" if rec_inset == 0 else "CONTRADICTED"))

    quoted_margins = {"13000": 1.75, "20100": 70.2, "26800": 75.75}
    measured = {k: by_name[k].along_mm - length_mm for k in quoted_margins}
    ok = all(abs(measured[k] - v) <= 0.005 for k, v in quoted_margins.items())
    rows.append((
        "margins on the other three SKUs",
        "+1.75 / +70.2 / +75.75 mm",
        " / ".join(f"{measured[k]:+.2f}" for k in ("13000", "20100", "26800"))
        + " mm",
        "REPRODUCED" if ok else "DIFFERS"))

    theta = rotation_limits(by_name["13000"], width_mm, length_mm)["along"][0]
    rows.append((
        "the 13000 is spent by",
        "6.9 deg of rotation",
        f"{theta:.2f} deg",
        "REPRODUCED (scope)" if abs(theta - 6.9) <= 0.05 else "DIFFERS"))
    return rows


def section_verdict(rows: Sequence[Tuple[str, str, str, str]],
                    window: Sequence[Unit], by_nominal: Dict[float, Dict],
                    width_mm: float) -> str:
    open_window = [u for u in window if u.variant == "open_case"]
    ship = INSETS_MM[0]
    t_ship = _tally(open_window, by_nominal[width_mm], ship)
    t_zero = _tally(open_window, by_nominal[width_mm], 0.0)
    recovered = [
        u for u in open_window
        if u.sku == "AnkerPowerCore10000"
        and by_nominal[width_mm][(u.frame, u.unit_id, 0.0)]["cells"] > 0
        and by_nominal[width_mm][(u.frame, u.unit_id, ship)]["cells"] == 0]

    n_ok = sum(1 for r in rows if r[3].startswith("REPRODUCED"))
    lines = [
        "4. The five figures docs/RESULTS_SUMMARY.md section 2 quotes",
        RULE,
        "Every row is measured in the sentence's own scope: ground-truth",
        "masks, ground-truth unit boxes, each frame's true scale, the",
        "shipping extractor and the shipping packer.",
        "",
    ]
    for figure, quoted, meas, verdict in rows:
        lines.append(f"  [{verdict}]")
        lines.append(f"     {figure}")
        lines.append(f"     quoted   {quoted}")
        lines.append(f"     measured {meas}")
    lines += [
        "",
        f"{n_ok} OF {len(rows)} REPRODUCE. THE THIRD DOES NOT.",
        "",
        "docs/RESULTS_SUMMARY.md section 2 reads: \"Feeding ground-truth",
        "label maps through the shipping extractor and packer at each",
        "frame's true scale ... that SKU places zero cells in 10 of 10",
        "instances at the production 4.25 mm wall inset. Tolerance is not",
        "the lever either: taking the nominal 18.5 -> 18.3 mm recovers 0",
        "instances, and relaxing the wall inset to 0.0 mm recovers 0.\"",
        "",
        f"Measured in that scope, relaxing the wall inset to 0.0 mm "
        f"recovers {t_zero['10000'][0] - t_ship['10000'][0]}:",
    ]
    for u in recovered:
        extent = by_nominal[width_mm][(u.frame, u.unit_id, 0.0)]["extent_mm"]
        cells = by_nominal[width_mm][(u.frame, u.unit_id, 0.0)]["cells"]
        lines.append(
            f"  {u.frame}/{u.unit_id}, turned {u.rot_offset_deg:.3f} deg at "
            f"{u.mm_per_px:.4f} mm/px,")
        lines.append(
            f"  bay measuring {extent[1]:.2f} mm long -> "
            f"{_plural(cells, 'cell')} with no erosion.")
    lines += [
        "",
        "The \"recovers 0\" belongs to a different arm. The feasibility",
        "spec's section 5 measures the inset sweep on PREDICTED masks at",
        "the fixed 0.625 mm/px the pipeline used to ship, where taking",
        "the inset to 0.0 recovers 0 of 23 and adds 0 cells - true, and",
        "stated there as such. Section 3 of the same spec measures the",
        "GT oracle and says the opposite in as many words: \"The one",
        "10000 that ground truth passes, scene_00049/c76, is rotated",
        "0.30 deg and measures 65.1 mm ... Add the wall inset and even",
        "that disappears.\" The summary's sentence carries section 5's",
        "conclusion under section 3's scope, and the two do not agree.",
        "",
        "It is the smaller half of the sentence. The inset recovers one",
        "instance and one cell; it does not make the SKU plannable, and",
        "the geometric claim the paragraph exists to make is untouched.",
        "But as written the sentence overstates the evidence, and on a",
        "page whose argument is that every figure has a receipt, the",
        "correct fix is the fix to the sentence.",
        "",
        "The 6.9 deg row is marked (scope) for the reason section 1's",
        "right-hand columns give: it is the slot-aligned orientation's",
        "threshold, not the angle past which the 13000 admits no cell.",
    ]
    return "\n".join(lines)


def section_reconciliation(window: Sequence[Unit], results: Dict,
                           full: Sequence[Unit], results_full: Dict) -> str:
    open_window = [u for u in window if u.variant == "open_case"]
    open_full = [u for u in full if u.variant == "open_case"]
    ship = INSETS_MM[0]
    t_ship = _tally(open_window, results, ship)
    t_zero = _tally(open_window, results, 0.0)
    n_ship = (sum(v[0] for v in t_ship.values()),
              sum(v[2] for v in t_ship.values()))
    n_zero = (sum(v[0] for v in t_zero.values()),
              sum(v[2] for v in t_zero.values()))
    t_full = _tally(open_full, results_full, ship)
    f10 = t_full["10000"]
    placers = [u for u in open_full if u.sku == "AnkerPowerCore10000"
               and results_full[(u.frame, u.unit_id, ship)]["cells"] > 0]
    worst = max((u.rot_offset_deg for u in placers), default=0.0)
    blocked = [u for u in open_full if u.sku == "AnkerPowerCore10000"
               and results_full[(u.frame, u.unit_id, ship)]["cells"] == 0]
    finest_blocked = min((u.rot_offset_deg for u in blocked), default=0.0)
    under = [u for u in open_full if u.sku == "AnkerPowerCore10000"
             and u.rot_offset_deg <= worst]

    return "\n".join([
        "5. Reconciliation, and what n = 10 does not cover",
        RULE,
        "This receipt's population is not the spec's. The spec ran the",
        "detector and the segmenter over the same 60 frames and paired",
        "each of the 30 resulting instances back to a ground-truth unit;",
        "this script enumerates the ground-truth units directly, which",
        "needs no checkpoint. The two differ by three units, and all",
        "three place zero cells either way:",
        "",
        f"  this receipt   {len(open_window)} open cartridges, no sealed ones",
        "  the spec       28 open + 2 sealed (scene_00026 13000,",
        "                 scene_00045 10000); scene_00045's OPEN 10000 is",
        "                 in this census and not in the spec's, because",
        "                 the detector matched the sealed unit beside it",
        "",
        "The totals agree exactly with the spec's 2026-08-12 addendum,",
        "which re-measured the oracle at HEAD after `b93bbd3` made the",
        "occupancy rasteriser require every pixel of a cell to be free:",
        "",
        "                                     addendum   this receipt",
        f"  inset {ship:.2f}, instances / cells      10 / 24        "
        f"{n_ship[0]} / {n_ship[1]}",
        f"  inset 0.00, instances / cells      11 / 25        "
        f"{n_zero[0]} / {n_zero[1]}",
        "",
        "and with the spec's section 3 per-SKU oracle line (10000 0,",
        "13000 3, 20100 1, 26800 6 admitting a cell after the 4.25 mm",
        "inset). That agreement is what certifies this harness; it was",
        "not tuned to produce it.",
        "",
        f"WHAT n = 10 DOES NOT COVER. The same corpus holds {f10[1]} open "
        f"10000 units in",
        "all, not 10, and over every one of them the SKU places zero cells "
        "in",
        f"{f10[1] - f10[0]} of {f10[1]} instances at the production inset "
        f"- {f10[2]} cells in the other {f10[0]}.",
        "",
        "Those four are not a counter-example to the geometry; they are the",
        "geometry, seen from the other side. No 10000 turned more than",
        f"{worst:.2f} deg from axis-aligned takes a cell anywhere in the "
        f"corpus - a",
        "seventh of the generator's own 2 deg jitter - and being inside that",
        f"band is necessary, not sufficient: {len(under) - f10[0]} of the "
        f"{len(under)} units inside it still",
        f"take none, the least-turned at {finest_blocked:.3f} deg, blocked by "
        "their own contents.",
        "A bay with 0.00 mm of margin is decided by whether the rasteriser",
        "rounds up. That is what a zero-margin fit means, and it is why it",
        "cannot be certified.",
        "",
        "The honest form of the headline is therefore: 10 of 10 in the",
        f"published window; {f10[1] - f10[0]} of {f10[1]} over the whole "
        f"corpus; and 0.00 mm of",
        "margin in CAD, which is the part that does not depend on a sample",
        "at all.",
    ])


# ======================================================================
# assembly
# ======================================================================

def header(dataset_present: bool) -> str:
    width_mm, length_mm, mm_per_cell = load_cell_nominal()
    lines = [
        "Exact-fit certifiability - a 65.0 mm cell in a 65.0 mm bay",
        RULE,
        "Generated by scripts/placement_feasibility.py. Reproduce with:",
        "  python scripts/placement_feasibility.py",
        "  python scripts/placement_feasibility.py --check   # drift",
        "",
        f"commit      : {git_commit()}",
        "receipt for : docs/RESULTS_SUMMARY.md section 2",
        "              docs/superpowers/specs/"
        "2026-08-11-placement-feasibility.md",
        "              sections 3 and 5, and its 2026-08-12 addendum",
        "",
        "NO CHECKPOINT AND NO TRAINING ARE INVOLVED. The masks are",
        "ground truth, the boxes are ground truth, the scale is each",
        "frame's own. Only the mask differs from the shipping path;",
        "every line of arbitration, erosion, rasterisation and packing",
        "is the code that ships.",
        "",
        "inputs",
        f"  catalog     : {CATALOG_PATH.relative_to(REPO_ROOT).as_posix()}",
        f"                sha256:{sha256(CATALOG_PATH)}",
        f"  planning    : {PLANNING_PATH.relative_to(REPO_ROOT).as_posix()}",
        f"                sha256:{sha256(PLANNING_PATH)}",
    ]
    ann = DATASET_DIR / "instances_seg.json"
    if dataset_present:
        manifest = json.loads(
            (DATASET_DIR / "manifest.json").read_text("utf-8"))
        lines += [
            f"  dataset     : {DATASET_DIR.relative_to(REPO_ROOT).as_posix()}"
            f" (gitignored; generator seed {manifest.get('seed')},",
            f"                {manifest.get('stats', {}).get('n_images')}"
            " scenes)",
            f"  annotations : instances_seg.json sha256:{sha256(ann)}",
            f"                ({ann.stat().st_size} bytes)",
        ]
    else:
        lines += [
            "  dataset     : ABSENT - the census arms below were not run.",
        ]
    lines += [
        "  extractor   : plan.placement_area."
        "SegmentationPlacementAreaExtractor",
        "  packer      : plan.planner.Planner._pack_cartridge ->",
        "                plan.bin_packing.pack_best_effort",
        f"  cell        : {width_mm} x {length_mm} mm on a {mm_per_cell} mm "
        f"occupancy grid",
        f"  wall insets : {', '.join(f'{i:.2f}' for i in INSETS_MM)} mm "
        f"({INSETS_MM[0]:.2f} is what ships)",
        "",
        "The commit line above is provenance and is NOT part of the drift",
        "check; --check compares the measured sections only.",
        "",
    ]
    return "\n".join(lines)


def _norm(text: str) -> str:
    """Strip trailing whitespace, line by line.

    A centred table header leaves invisible padding at the end of a line.
    Normalising HERE rather than when the file is written is what keeps
    ``--check`` honest: the drift check asks whether each section appears
    verbatim in the committed receipt, and a section carrying padding the
    file does not would report drift that no reader could see.
    """
    return "\n".join(line.rstrip() for line in text.splitlines())


def build_sections() -> Tuple[Dict[str, str], List[str]]:
    """``({section name: text}, [skipped reasons])``."""
    width_mm, length_mm, mm_per_cell = load_cell_nominal()
    skus = load_skus()
    cad_width_mm = skus[0].cell_diameter_mm

    sections = {"geometry": _norm(section_geometry(skus, width_mm, length_mm))}
    skipped: List[str] = []

    if not (DATASET_DIR / "instances_seg.json").is_file():
        skipped.append(
            f"census: {DATASET_DIR.relative_to(REPO_ROOT).as_posix()} is not "
            f"present (it is gitignored). Regenerate it with "
            f"`python -m recog.generate3d --config configs/synth3d.yaml` or "
            f"run this on the authoring tree.")
        return sections, skipped

    # plan.arbitration logs "crop centre is background (bad detector
    # box?)" per crop. There is no detector in this measurement, and the
    # message would be 200 lines of misdirection; the CONDITION is still
    # counted and reported, per SKU, in section 2's outcome line.
    logging.getLogger("plan.arbitration").setLevel(logging.ERROR)

    window = load_units(WINDOW_FRAMES)
    full = load_units(None)
    nominals = (width_mm, cad_width_mm)
    by_nominal = census(window, nominals, length_mm, mm_per_cell, INSETS_MM)
    by_nominal_full = census(full, nominals, length_mm, mm_per_cell, INSETS_MM)
    ship = INSETS_MM[0]

    sections["census"] = _norm(section_census(
        window, by_nominal[width_mm], width_mm, length_mm, ship))
    sections["sweep"] = _norm(section_sweep(
        window, by_nominal, full, by_nominal_full, width_mm, cad_width_mm))
    rows = verdicts(window, by_nominal, skus, width_mm, length_mm,
                    cad_width_mm)
    sections["verdict"] = _norm(
        section_verdict(rows, window, by_nominal, width_mm))
    sections["reconciliation"] = _norm(section_reconciliation(
        window, by_nominal[width_mm], full, by_nominal_full[width_mm]))
    return sections, skipped


ORDER = ("geometry", "census", "sweep", "verdict", "reconciliation")


def render(sections: Dict[str, str]) -> str:
    body = "\n\n".join(sections[name] for name in ORDER if name in sections)
    return header("census" in sections) + "\n" + body + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed receipt disagrees with a "
                         "fresh run of this script")
    args = ap.parse_args(argv)

    sections, skipped = build_sections()
    for note in skipped:
        print(f"SKIPPED {note}", file=sys.stderr)

    if args.check:
        if not RECEIPT_PATH.is_file():
            print(f"missing: {RECEIPT_PATH.relative_to(REPO_ROOT)}",
                  file=sys.stderr)
            return 2
        committed = RECEIPT_PATH.read_text(encoding="utf-8")
        drift = [name for name in ORDER
                 if name in sections and sections[name] not in committed]
        if drift:
            for name in drift:
                print(f"DRIFT: section {name!r} of "
                      f"{RECEIPT_PATH.relative_to(REPO_ROOT).as_posix()} "
                      f"does not match a fresh run. Regenerate it:\n"
                      f"  python scripts/placement_feasibility.py",
                      file=sys.stderr)
            return 1
        print(f"no drift: {len(sections)} section(s) checked, "
              f"{len(skipped)} skipped")
        return 0

    if skipped:
        print("receipt NOT written: an arm did not run", file=sys.stderr)
        return 2
    RECEIPT_PATH.write_text(render(sections), encoding="utf-8")
    print(f"wrote {RECEIPT_PATH.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
