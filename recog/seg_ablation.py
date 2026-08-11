"""
recog.seg_ablation - the end-to-end numbers.

Two independent measurements live here, and they are independent on
purpose: one needs ground truth that only synthetic data has, the other
runs on the real photographs that have no ground truth mask at all.

delta_cells is the metric stated in the unit the project cares about:
cells in cartridges. Mask IoU can improve while delta_cells gets worse,
if the error simply moves to where a cell would have gone.

Requires the packer fix (FDR 6.3.1, Plan A). On the unfixed shelf-cursor
logic the figure is dominated by the packer abandoning shelves - 23.0
cells at 0% forbidden coverage against 3.2 at 2.5% - and would say
nothing about the segmenter at all.

Sign convention: POSITIVE means the prediction lost cells the ground
truth would have placed (conservative, a throughput cost). NEGATIVE
means the prediction packed cells the ground truth forbids (optimistic,
a damage risk). Negative is the serious direction.

delta_cells needs a GROUND-TRUTH label map, which only the synthetic
validation split (recog/dataset3d_seg, via BaySegDataset) carries. It is
therefore computed there, not on recog/realtest.

heuristic_vs_segmenter is the real-photo half. recog/realtest carries
boxes only - 80 annotations, 2 categories (Battery, Cartridge), and
(verified at run time, not assumed) ZERO non-empty `segmentation`
polygons. Design spec Sec9's headline framing calls for scoring
placeable-area IoU against human polygons; spec Sec9.1 itself explains
why that cannot be done yet ("There is no existing real basis on which a
mask model can be validated") and proposes re-annotating the 7 photos
with 5-class polygons as FUTURE work. That re-annotation has not
happened as of this module. Scoring against a stand-in - e.g. treating
the cartridge detection box itself, or some erosion of it, as a
placement-area polygon - would produce a number shaped like an IoU
without being one, and the receipt would not tell a future reader that.
This module does not do that.

What CAN be measured honestly, without any ground-truth mask, is exactly
the quantity spec Sec1.1 already measured for the heuristic: placeable
area as a FRACTION of the cartridge ROI (pixels called placeable, over
total ROI pixels). It needs no polygon - only the extractor's own output
- and running the identical arithmetic for the segmenter is a fair,
like-for-like comparison against the measured baseline: mean 0.218,
7 of 20 cartridges at exactly zero.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.types import BBox
# The 18650's own CAD-measured footprint - see recog.synth3d.config's own
# comment for the figure's provenance. Imported rather than restated, the
# same reasoning as recog.calibrate_tau's copy (which independently
# duplicated this exact pair until this consolidation) and unlike
# _DEFAULT_WALL_INSET_MM below: recog.synth3d.config has no cv2/torch/bpy
# import of its own, so pulling it in here does not compromise this
# module's lazy-import discipline (2026-08-10 consolidation).
from recog.synth3d.config import CELL_H_MM, CELL_W_MM

# spec Sec1.1's measured heuristic baseline on recog/realtest, using the
# ground-truth cartridge boxes so detector error plays no part. This is
# the number every segmenter run is compared against - see the CLI's
# verdict line, and heuristic_vs_segmenter's "beats_baseline" field.
HEURISTIC_BASELINE_MEAN_FRACTION = 0.218
HEURISTIC_BASELINE_N_ZERO = 7
HEURISTIC_BASELINE_N = 20

# matches plan.placement_area.SegmentationPlacementAreaExtractor's own
# default (restated, not imported, so this module's import surface stays
# free of cv2 at module-load time - see the lazy imports throughout this
# file). Used as the default wall_inset_mm everywhere below; a caller
# that measures against a different inset should pass it explicitly, not
# rely on a hardcoded 4.0 that quietly diverges from what production
# actually erodes by (final review, D-integration-arbitration).
_DEFAULT_WALL_INSET_MM = 4.25


# =========================================================== delta_cells ===

def _pack_count(label_map: np.ndarray, mm_per_px: float,
                wall_inset_mm: float = _DEFAULT_WALL_INSET_MM) -> int:
    """Cells FFDH packs into ``label_map``'s arbitrated safe region."""
    from common.packing import Item, first_fit_decreasing
    from plan.arbitration import arbitrate
    from plan.placement_area import _rasterise_mask
    from plan.scene import CellState

    inset_px = max(0, int(round(wall_inset_mm / mm_per_px)))
    safe, _ = arbitrate(label_map, inset_px)
    if not safe.any():
        return 0

    ys, xs = np.nonzero(safe)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    strip_w = (x1 - x0) * mm_per_px
    strip_h = (y1 - y0) * mm_per_px

    # Reuse production's own rasteriser rather than re-deriving the
    # pixel/cell stride. The previous int(round(mm_per_cell/mm_per_px))
    # quantised the forbidden mask at 1.25 mm/cell (px_per_cell=2) while
    # telling the packer mm_per_cell=1.5 - a mismatched stride that
    # indexed only ~83% of the mask's columns and displaced obstacles by
    # up to ~10 mm. _rasterise_mask's float stride
    # (mm_per_cell / mm_per_px, unrounded) is what plan.placement_area
    # actually ships, so this ablation now measures the same quantisation
    # production does (final whole-branch review, D-integration-arbitration).
    mm_per_cell = 1.5
    grid = _rasterise_mask(safe, (x0, y0, x1, y1), mm_per_cell, mm_per_px)
    forbidden = grid.mask_of(CellState.FORBIDDEN).astype(np.uint8)

    n_max = max(4, int(strip_w * strip_h / (CELL_W_MM * CELL_H_MM)) * 2)
    items = [Item(id=i, width=CELL_W_MM, height=CELL_H_MM)
             for i in range(n_max)]
    return first_fit_decreasing(
        items, strip_w, strip_h, allow_rotation=True,
        forbidden_mask=forbidden, mm_per_cell=mm_per_cell).count


def delta_cells(gt_label_map: np.ndarray, pred_label_map: np.ndarray,
                mm_per_px: float,
                wall_inset_mm: float = _DEFAULT_WALL_INSET_MM) -> int:
    """Cells the packer places on truth, minus cells on the prediction.

    See the module docstring for the sign convention.
    """
    return (_pack_count(gt_label_map, mm_per_px, wall_inset_mm)
            - _pack_count(pred_label_map, mm_per_px, wall_inset_mm))


def evaluate_delta_cells(segmenter, full_dataset, val_indices: Sequence[int],
                         mm_per_px: float,
                         wall_inset_mm: float = _DEFAULT_WALL_INSET_MM
                         ) -> Dict[str, Any]:
    """delta_cells over every crop in the segmenter's OWN validation split.

    Native resolution (out_size=None), matching recog.seg_evaluate.evaluate,
    so mm_per_px stays isotropic - a jittered union box is not square, and
    a single scalar mm_per_px cannot describe a resized one. No jitter at
    eval, also matching seg_evaluate: the box is read straight from
    ``full_dataset.samples``, not passed through ``jitter_box``.
    """
    from PIL import Image

    from recog.seg_dataset import extract_crop, rasterise_crop

    rows: List[Dict[str, Any]] = []
    for idx in val_indices:
        img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)

        path = Path(full_dataset.img_dir) / img_meta["file_name"]
        image = np.asarray(Image.open(path).convert("RGB"))

        crop = extract_crop(image, box, out_size=None)
        gt = rasterise_crop(anns, box, out_size=None)
        pred = segmenter.segment_batch([crop])[0]

        d = delta_cells(gt, pred, mm_per_px, wall_inset_mm)
        rows.append({"file_name": img_meta["file_name"], "unit_box": box,
                     "delta_cells": d})

    arr = np.asarray([r["delta_cells"] for r in rows], dtype=float)
    n = len(arr)
    return {
        "n": n,
        "rows": rows,
        "wall_inset_mm": wall_inset_mm,
        "mean": float(arr.mean()) if n else float("nan"),
        "median": float(np.median(arr)) if n else float("nan"),
        "min": float(arr.min()) if n else float("nan"),
        "max": float(arr.max()) if n else float("nan"),
        "n_positive": int((arr > 0).sum()),
        "n_negative": int((arr < 0).sum()),
        "n_zero": int((arr == 0).sum()),
    }


# ================================================ heuristic_vs_segmenter ===

# recog/synth3d/assets/catalog.json extents_mm[0] for the four cataloged
# Anker assemblies - the only measured physical widths this project has.
# recog/realtest's photos carry no fixed-mount calibration (see
# plan.placement_area.SegmentationPlacementAreaExtractor.wall_inset_px's
# docstring: a fixed pixel inset is wrong across camera framings, and
# these are 3024x4032 handheld phone photos, not the deployed mount's
# frame), so mm_per_px cannot be read from a config the way the
# synthetic split's can. This mean width divided by each cartridge's own
# annotated bbox width in pixels stands in for it per cartridge - a
# geometric estimate grounded in measured CAD data, not a fabricated
# number, used ONLY to size the segmenter's wall-inset erosion (a few
# mm). It has no effect on either extractor's placeable-fraction
# numerator or denominator, which are both plain pixel-count ratios and
# so need no physical scale at all.
_CATALOG_ASSET_WIDTHS_MM: Tuple[float, ...] = (62.9, 80.7, 62.3, 81.7)
_CATALOG_MEAN_WIDTH_MM = sum(_CATALOG_ASSET_WIDTHS_MM) / len(_CATALOG_ASSET_WIDTHS_MM)

# _DEFAULT_WALL_INSET_MM now lives near CELL_W_MM at the top of the
# module - delta_cells needed it as a default too, and a duplicate
# definition down here would just be a second place for it to drift.


def _load_real_cartridges(real_dir: Path
                          ) -> Tuple[Path, List[Tuple[str, Tuple[float, float, float, float]]],
                                    bool, int]:
    """``(img_dir, [(file_name, bbox), ...], has_polygons, n_annotations)``.

    ``has_polygons`` is checked directly against the raw COCO JSON (not
    inferred) so a future re-annotation is picked up automatically rather
    than requiring this module to be edited.
    """
    from recog.dataset import CLASS_MAP, parse_coco_json

    ann_path = real_dir / "annotations" / "instances_default.json"
    img_dir = real_dir / "images"
    if not ann_path.is_file():
        raise FileNotFoundError(f"annotations not found: {ann_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {img_dir}")

    with open(ann_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    n_annotations = len(raw.get("annotations", []))
    has_polygons = any(a.get("segmentation") for a in raw.get("annotations", []))

    records = parse_coco_json(ann_path)
    cartridge_cls = CLASS_MAP["cartridge"]
    cartridges: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for rec in records:
        for box, label in zip(rec.boxes, rec.labels):
            if label == cartridge_cls:
                cartridges.append((rec.file_name, box))
    return img_dir, cartridges, has_polygons, n_annotations


def _heuristic_fraction(image: np.ndarray,
                        box: Tuple[float, float, float, float],
                        mm_per_px_estimate: float
                        ) -> Tuple[float, str, bool]:
    """``(placeable_fraction, note, warned)`` for one cartridge ROI.

    A fresh extractor per cartridge, matching how spec Sec1.1's own 20
    figures were produced. The constructor is what raises
    HeuristicPlacementAreaExtractor's scope-limit RuntimeWarning, so this
    fires (and is caught, not printed) once per cartridge - twenty times
    over the full set. That is noise here, but whether it fired is
    recorded, not silently dropped.

    ``mm_per_px_estimate`` is the SAME per-cartridge estimate the
    segmenter arm gets, rather than a constructor default. It cannot move
    this metric - the fraction is ``inside_mask.sum() / inside_mask.size``
    and ``inside_mask`` is built entirely from pixel quantities
    (``safety_margin_px``, the green contour, the dark-region PCB), while
    ``mm_per_px`` reaches only the occupancy grid, which is not read here.
    It is passed because the extractor no longer invents a scale when
    nobody supplies one, and because two arms of one comparison silently
    calibrated differently is how this file's neighbours went wrong.
    """
    from plan.placement_area import HeuristicPlacementAreaExtractor

    bbox = BBox(*box)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        extractor = HeuristicPlacementAreaExtractor(
            mm_per_px=mm_per_px_estimate)
        try:
            pa = extractor.extract(image, bbox)
        except (ValueError, RuntimeError) as exc:
            warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
            return 0.0, f"exception: {type(exc).__name__}: {exc}", warned

    warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
    area = int(pa.inside_mask.size)
    frac = float(pa.inside_mask.sum()) / area if area else 0.0
    return frac, "", warned


def _segmenter_fraction(segmenter, image: np.ndarray,
                        box: Tuple[float, float, float, float],
                        mm_per_px_estimate: float, wall_inset_mm: float
                        ) -> Tuple[float, str]:
    """``(placeable_fraction, note)`` for one cartridge ROI.

    Nothing is skipped on disagreement - every cartridge is scored,
    matching the brief: "the comparison covers every cartridge." That
    used to need an explicit tau=0; the tau gate is now retired from
    the extractor entirely (FDR v3 section 13.2.1), so the default
    behaviour is already the one this measurement wants. A cartridge
    can still fail with no placeable area at all (both estimates
    empty), which is scored as fraction 0.0 with a note, not silently
    dropped from the summary.
    """
    from plan.placement_area import SegmentationPlacementAreaExtractor

    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, "exception: empty cartridge bbox"

    crop = image[y0:y1, x0:x1, :]
    label_map = segmenter.segment(crop)

    extractor = SegmentationPlacementAreaExtractor(
        mm_per_px=mm_per_px_estimate, wall_inset_mm=wall_inset_mm)
    try:
        pa = extractor.extract(image, BBox(x0, y0, x1, y1), label_map=label_map)
    except (ValueError, RuntimeError) as exc:
        return 0.0, f"exception: {type(exc).__name__}: {exc}"

    area = int(pa.inside_mask.size)
    frac = float(pa.inside_mask.sum()) / area if area else 0.0
    return frac, ""


def heuristic_vs_segmenter(real_dir: str | Path, segmenter,
                           cfg: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """Heuristic vs segmenter, both scored as placeable fraction of ROI,
    over every cartridge in ``real_dir`` (spec Sec9.1 tier 1).

    See the module docstring for why this is NOT an IoU against human
    polygons: recog/realtest carries none. ``result["has_ground_truth_
    polygons"]`` records what was actually found in the annotation file
    at run time, and the CLI's report states it plainly.
    """
    from recog.eval_real import load_image_rgb

    cfg = cfg or {}
    wall_inset_mm = float((cfg.get("evaluation") or {}).get(
        "wall_inset_mm", _DEFAULT_WALL_INSET_MM))

    real_dir = Path(real_dir)
    img_dir, cartridges, has_polygons, n_annotations = \
        _load_real_cartridges(real_dir)

    rows: List[Dict[str, Any]] = []
    n_warned = 0
    for i, (file_name, box) in enumerate(cartridges, start=1):
        image = load_image_rgb(img_dir / file_name)

        bbox_w_px = max(1.0, box[2] - box[0])
        mm_per_px_estimate = _CATALOG_MEAN_WIDTH_MM / bbox_w_px

        h_frac, h_note, warned = _heuristic_fraction(
            image, box, mm_per_px_estimate)
        if warned:
            n_warned += 1

        s_frac, s_note = _segmenter_fraction(
            segmenter, image, box, mm_per_px_estimate, wall_inset_mm)

        rows.append({
            "index": i, "file_name": file_name, "bbox": tuple(box),
            "heuristic_fraction": h_frac, "heuristic_note": h_note,
            "segmenter_fraction": s_frac, "segmenter_note": s_note,
            "fraction_delta": s_frac - h_frac,
        })

    h_fracs = np.array([r["heuristic_fraction"] for r in rows], dtype=float)
    s_fracs = np.array([r["segmenter_fraction"] for r in rows], dtype=float)
    n = len(rows)

    return {
        "n_cartridges": n,
        "n_images": len(sorted(set(r["file_name"] for r in rows))),
        "n_annotations_total": n_annotations,
        "has_ground_truth_polygons": has_polygons,
        "wall_inset_mm": wall_inset_mm,
        "rows": rows,
        "heuristic": {
            "mean": float(h_fracs.mean()) if n else float("nan"),
            "median": float(np.median(h_fracs)) if n else float("nan"),
            "n_zero": int((h_fracs == 0.0).sum()),
            "n_warned": n_warned,
        },
        "segmenter": {
            "mean": float(s_fracs.mean()) if n else float("nan"),
            "median": float(np.median(s_fracs)) if n else float("nan"),
            "n_zero": int((s_fracs == 0.0).sum()),
        },
        "baseline_mean": HEURISTIC_BASELINE_MEAN_FRACTION,
        "baseline_n_zero": HEURISTIC_BASELINE_N_ZERO,
        "baseline_n": HEURISTIC_BASELINE_N,
        "beats_baseline_mean": bool(
            n and s_fracs.mean() > HEURISTIC_BASELINE_MEAN_FRACTION),
        "has_zero_area_cartridges": bool(n and (s_fracs == 0.0).any()),
    }


# ===================================================================== CLI ==

def format_report(delta_result: Dict[str, Any], real_result: Dict[str, Any], *,
                  checkpoint: Optional[str], config_path: str,
                  synth_source: str, mm_per_px_synth: float,
                  real_dir: str, device: str) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("Segmentation ablation: delta_cells and heuristic-vs-segmenter")
    lines.append("=" * 66)
    lines.append(f"  checkpoint    : {checkpoint or '(none)'}")
    lines.append(f"  config        : {config_path}")
    lines.append(f"  device        : {device}")
    lines.append("")

    # ---- delta_cells --------------------------------------------------
    lines.append("delta_cells - the end-to-end number, in cells. Positive = "
                 "cells the ground")
    lines.append("truth would have placed but the prediction lost "
                 "(throughput cost). Negative =")
    lines.append("cells the prediction packed where the ground truth "
                 "forbids them (damage risk,")
    lines.append("the serious direction). Measured on the segmenter's OWN "
                 "validation split")
    lines.append(f"({synth_source}, mm_per_px={mm_per_px_synth:.4f}) "
                 "because delta_cells needs a ground-")
    lines.append("truth label map, which only synthetic data carries - "
                 "recog/realtest has none.")
    lines.append("Requires the Plan A packer fix (FDR 6.3.1): on the "
                 "unfixed shelf-cursor logic")
    lines.append("this figure was dominated by the packer abandoning "
                 "shelves, not by mask error.")
    lines.append("")
    n = delta_result["n"]
    lines.append(f"  wall_inset_mm used: {delta_result['wall_inset_mm']:.2f}")
    lines.append(f"  validation crops (n={n}):")
    lines.append(f"    mean   : {delta_result['mean']:.3f}")
    lines.append(f"    median : {delta_result['median']:.3f}")
    lines.append(f"    range  : [{delta_result['min']:.0f}, "
                 f"{delta_result['max']:.0f}]")
    lines.append(f"    positive (cells lost)   : {delta_result['n_positive']} / {n}")
    lines.append(f"    negative (cells packed where forbidden) : "
                 f"{delta_result['n_negative']} / {n}")
    lines.append(f"    zero (no difference)    : {delta_result['n_zero']} / {n}")
    lines.append("")

    # ---- heuristic vs segmenter ----------------------------------------
    lines.append("Heuristic vs segmenter on the real photographs "
                 f"({real_dir})")
    lines.append("-" * 66)
    if not real_result["has_ground_truth_polygons"]:
        lines.append("NOTE: recog/realtest/annotations/instances_default.json "
                     "carries boxes only")
        lines.append(f"({real_result['n_annotations_total']} annotations, "
                     "2 categories) and ZERO segmentation polygons (checked")
        lines.append("at run time, not assumed). Design spec Sec9's headline "
                     "framing calls for")
        lines.append("scoring placeable-area IoU against human polygons; "
                     "spec Sec9.1 itself says why")
        lines.append("that is not yet possible and proposes re-annotation "
                     "as future work. That has")
        lines.append("not happened. The numbers below are NOT that IoU. "
                     "They are placeable area as")
        lines.append("a fraction of the cartridge ROI - the same quantity "
                     "spec Sec1.1 measured for")
        lines.append("the heuristic baseline (mean 0.218, 7/20 zero) - "
                     "which needs no ground-truth")
        lines.append("mask at all, so it is comparable without inventing "
                     "one.")
    else:
        lines.append("Segmentation polygons ARE present in the annotation "
                     "file; this run still")
        lines.append("reports placeable-fraction-of-ROI for continuity "
                     "with the measured 0.218")
        lines.append("baseline, not polygon IoU - a later revision could "
                     "add the polygon-IoU")
        lines.append("comparison now that ground truth exists.")
    lines.append("")
    lines.append(f"  cartridges: {real_result['n_cartridges']} "
                 f"(over {real_result['n_images']} image(s)) - a smoke test, "
                 "not a validation")
    lines.append("  (spec Sec9.1: with 20 cartridges this cannot support a "
                 "transfer claim).")
    lines.append("")

    head = f"  {'#':>3} {'image':<14}{'heuristic':>11}{'segmenter':>11}{'delta':>9}  note"
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for r in real_result["rows"]:
        note = r["segmenter_note"] or r["heuristic_note"]
        lines.append(
            f"  {r['index']:>3} {r['file_name']:<14}"
            f"{r['heuristic_fraction']:>11.3f}"
            f"{r['segmenter_fraction']:>11.3f}"
            f"{r['fraction_delta']:>+9.3f}  {note}".rstrip())
    lines.append("  " + "-" * (len(head) - 2))
    lines.append("")

    h, s = real_result["heuristic"], real_result["segmenter"]
    lines.append(f"  {'':<18}{'mean':>10}{'median':>10}{'n_zero':>9}")
    lines.append(f"  {'heuristic':<18}{h['mean']:>10.3f}{h['median']:>10.3f}"
                 f"{h['n_zero']:>9} / {real_result['n_cartridges']}")
    lines.append(f"  {'segmenter':<18}{s['mean']:>10.3f}{s['median']:>10.3f}"
                 f"{s['n_zero']:>9} / {real_result['n_cartridges']}")
    lines.append(f"  {'baseline (spec 1.1)':<18}{real_result['baseline_mean']:>10.3f}"
                 f"{'':>10}{real_result['baseline_n_zero']:>9} / "
                 f"{real_result['baseline_n']}")
    lines.append("")
    lines.append(f"  heuristic RuntimeWarning fired: {h['n_warned']} / "
                 f"{real_result['n_cartridges']} cartridges (suppressed "
                 "above; scope-limit warning,")
    lines.append("  see plan.placement_area.HeuristicPlacementAreaExtractor).")
    lines.append("")

    if real_result["beats_baseline_mean"] and not real_result["has_zero_area_cartridges"]:
        verdict = (f"the segmenter beats the measured baseline "
                  f"({s['mean']:.3f} > {real_result['baseline_mean']:.3f}) "
                  "and has no zero-area cartridges.")
    elif real_result["beats_baseline_mean"]:
        verdict = (f"the segmenter beats the baseline mean "
                  f"({s['mean']:.3f} > {real_result['baseline_mean']:.3f}) "
                  f"but still returns zero placeable area on "
                  f"{s['n_zero']} / {real_result['n_cartridges']} "
                  "cartridges.")
    else:
        verdict = (f"the segmenter does NOT beat the measured baseline "
                  f"({s['mean']:.3f} <= {real_result['baseline_mean']:.3f}). "
                  "This is the result, not a defect to adjust the "
                  "comparison until it disappears: a segmenter that "
                  "cannot beat a measured baseline is not worth "
                  "shipping.")
    lines.append(f"  verdict: {verdict}")
    lines.append("")
    lines.append(f"  wall_inset_mm used: {real_result['wall_inset_mm']:.2f}. "
                 "mm_per_px for these photos is NOT calibrated (no fixed-")
    lines.append("  mount framing exists for handheld phone photos); each "
                 "cartridge's own scale is")
    lines.append("  estimated from its annotated bbox width against the "
                 "mean of the four cataloged")
    lines.append(f"  Anker case widths ({_CATALOG_MEAN_WIDTH_MM:.1f} mm). "
                 "This affects only the segmenter's")
    lines.append("  wall-inset erosion depth, not either extractor's "
                 "placeable-fraction ratio.")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.seg_ablation",
        description="delta_cells (on the synthetic validation split) and "
                    "the heuristic-vs-segmenter placeable-fraction "
                    "comparison on recog/realtest.")
    ap.add_argument("--checkpoint", default="recog/checkpoints/seg/best.pt")
    ap.add_argument("--config", default="configs/segmentation.yaml")
    ap.add_argument("--synth-config", default=None,
                    help="render/layout config to compute mm_per_px from "
                         "for delta_cells. Defaults to the dataset's own "
                         "manifest.json, falling back to "
                         "configs/synth3d.yaml.")
    ap.add_argument("--real", default="recog/realtest")
    ap.add_argument("--wall-inset-mm", type=float, default=_DEFAULT_WALL_INSET_MM)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="docs/receipts/seg_ablation.txt")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    from common.config import load_yaml

    args = build_arg_parser().parse_args(argv)

    if args.checkpoint and not Path(args.checkpoint).exists():
        raise SystemExit(f"error: checkpoint not found: {args.checkpoint}")

    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model") or {}
    ds_cfg = cfg.get("dataset") or {}

    num_classes = int(model_cfg.get("num_classes", 6))
    crop_size = int(model_cfg.get("crop_size", 256))
    half = bool(model_cfg.get("half", True))

    from recog.bay_segmenter import BaySegmenter
    from recog.seg_dataset import BaySegDataset
    from recog.seg_evaluate import (check_split_matches_checkpoint,
                                    compute_val_instance_counts,
                                    load_synth_config, resolve_mm_per_px)
    from recog.seg_training import _split_dataset

    segmenter = BaySegmenter(checkpoint=args.checkpoint, device=args.device,
                             crop_size=crop_size, half=half,
                             num_classes=num_classes)

    # ---- delta_cells: the synthetic validation split -------------------
    synth_cfg, synth_source = load_synth_config(
        ds_cfg["coco_path"], args.synth_config)
    synth_source = Path(synth_source).as_posix()
    mm_per_px = resolve_mm_per_px(synth_cfg)

    full_dataset = BaySegDataset(
        coco_path=ds_cfg["coco_path"], img_dir=ds_cfg["img_dir"],
        out_size=crop_size, jitter_frac=float(ds_cfg.get("jitter_frac", 0.06)),
        train=True, transform=None)
    if len(full_dataset) == 0:
        raise SystemExit(f"error: no crops found for {ds_cfg['coco_path']}")

    split_seed = int(ds_cfg.get("split_seed", 0))
    _train_set, val_set = _split_dataset(
        full_dataset, float(ds_cfg.get("train_val_split", 0.85)),
        seed=split_seed)
    val_indices = list(val_set.indices)

    if args.checkpoint:
        val_counts_now = compute_val_instance_counts(
            full_dataset, val_indices, num_classes=num_classes)
        # `coco_path` is not optional and never was here: the guard only
        # fires when this eval is against the SAME dataset the checkpoint
        # trained on, and it needs the path to know. The argument was
        # added in 138105d (cross-dataset scoring) and this caller was
        # not updated, so `python -m recog.seg_ablation` has raised
        # TypeError before reaching a single measurement ever since -
        # which is why docs/receipts/seg_ablation.txt could not be
        # regenerated. Matches recog.seg_evaluate.main's own call.
        check_split_matches_checkpoint(args.checkpoint, ds_cfg["coco_path"],
                                       val_counts_now)

    delta_result = evaluate_delta_cells(
        segmenter, full_dataset, val_indices, mm_per_px, args.wall_inset_mm)

    # ---- heuristic vs segmenter: the real photographs -------------------
    real_result = heuristic_vs_segmenter(
        args.real, segmenter,
        {"evaluation": {"wall_inset_mm": args.wall_inset_mm}})

    report = format_report(
        delta_result, real_result,
        checkpoint=args.checkpoint, config_path=args.config,
        synth_source=synth_source, mm_per_px_synth=mm_per_px,
        real_dir=args.real, device=str(segmenter.device))
    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_W_MM",
    "CELL_H_MM",
    "HEURISTIC_BASELINE_MEAN_FRACTION",
    "HEURISTIC_BASELINE_N_ZERO",
    "HEURISTIC_BASELINE_N",
    "delta_cells",
    "evaluate_delta_cells",
    "heuristic_vs_segmenter",
    "format_report",
    "build_arg_parser",
    "main",
]
