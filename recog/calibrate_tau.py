"""
recog.calibrate_tau - choose the placement-disagreement threshold.

tau is calibrated, never asserted. A hardcoded 0.85 that was never
measured is a placeholder wearing a number's clothes.

For each validation cartridge, record the arbitration IoU and whether
the OPTIMISTIC error - area P_safe claims is placeable where the ground
truth says it is not - can contain a whole cell. The second is the
morphological test in plan.arbitration.admits_a_cell, not an area
threshold: error spread along a boundary cannot hold a cell, the same
area in one blob can.

Both of those quantities are measured in PIXELS and specified in
MILLIMETRES, so both depend on the frame's ground sample distance - and
until 2026-08-11 this module converted at the generator's nominal
0.625 mm/px, a framing no frame in the corpus was rendered at. It now
takes each crop's scale from that crop's own render sidecar
(`recog.calibration`, via `recog.seg_evaluate.resolve_frame_scales`).
Unlike the figures in `recog.seg_evaluate`, this is not only a reporting
correction: the wall inset and the cell footprint feed the measurement
itself, so the arbitration IoUs and the admits-a-cell verdicts this
module produces genuinely differ from the ones it produced before.

tau is then the SMALLEST threshold at which at most `fail_budget` of
accepted cartridges admit a cell. Smallest, because a larger tau rejects
more cartridges - so this maximises throughput subject to the safety
bound rather than simply being safe by refusing everything.

Population: only cartridges whose PREDICTED direct estimate (the `bay`
channel) is non-empty. That is not cherry-picking - it is the only
population for which the question is non-vacuous. `P_safe = P_direct &
P_derived`, so whenever P_direct is empty, P_safe is empty regardless of
tau, the optimistic error is empty, and admits_a_cell is False by
construction. SegmentationPlacementAreaExtractor.extract() confirms this
operationally: an empty P_direct always ends in an empty placement
region, either via the tau gate (PlacementDisagreement) or via the
"no placeable area" branch straight after it - a cell is never placed
either way. Folding those crops into the calibration population would
only dilute the accepted set with cases tau never actually decides,
making the reported rejected_fraction describe cartridges that were
never in question.

CLI::

    python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt \\
        --config configs/segmentation.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.config import load_yaml
from common.logging import get_logger

log = get_logger("recog.calibrate_tau")

# The 18650's own CAD-measured footprint, not planning.yaml's 18.5 mm
# nominal spec figure. admits_a_cell is a WORST-CASE test - a slightly
# smaller structuring element is easier to fit, so testing against the
# measured (smaller) footprint only ever flags MORE optimistic-error
# regions as unsafe, never fewer. Imported rather than restated - same
# reasoning as DEFAULT_WALL_INSET_MM just below - so this can never
# again independently drift from recog/seg_ablation.py's own copy the
# way it once did (2026-08-10 consolidation).
from recog.synth3d.config import CELL_H_MM, CELL_W_MM  # noqa: E402

# The wall inset plan.placement_area's SegmentationPlacementAreaExtractor
# defaults to - imported rather than restated so the inset this
# calibration scores against can never silently drift from the one
# production erodes by.
#
# From common.types since 2026-08-15, not plan.placement_area. Same
# object, and this was one of the two module-level back-edges from
# `recog` into `plan` (tests/test_seams.py pins that there are now
# none): a constant three packages read belongs in the package all three
# already depend on, and importing it from `plan` pulled cv2 and
# plan.scene into a module that is otherwise lazy about both.
from common.types import DEFAULT_WALL_INSET_MM  # noqa: E402


# ------------------------------------------------------------ calibrate --

def _sweep(records: Sequence[dict], fail_budget: float) -> List[Dict]:
    """Every candidate tau, ascending, with its accepted-set stats.

    `calibrate` picks its answer from this same table (rather than
    reimplementing the scan) so the receipt's full trade-off curve and
    the single number `calibrate` returns can never drift apart.
    """
    # Raw floats, NOT rounded: a candidate has to compare >= against the
    # very iou it came from. Rounding here (e.g. to 4dp) could round a
    # candidate UP past its own source record - 0.74919999... rounds to
    # 0.7492, but 0.74919999... >= 0.7492 is False - silently dropping
    # the sample-minimum record from its own row and making "the sample
    # minimum" wrong by an epsilon (final whole-branch review).
    candidates = sorted({float(r["iou"]) for r in records})
    n = len(records)
    rows: List[Dict] = []
    for tau in candidates:
        accepted = [r for r in records if float(r["iou"]) >= tau]
        n_accepted = len(accepted)
        fails = sum(1 for r in accepted if r["admits_cell"])
        rate = (fails / n_accepted) if n_accepted else float("nan")
        rows.append({
            "tau": tau,
            "n_accepted": n_accepted,
            "n_fail": fails,
            "fail_rate": rate,
            "rejected_fraction": (1.0 - n_accepted / n) if n else float("nan"),
            "meets_budget": n_accepted > 0 and rate <= fail_budget,
        })
    return rows


def calibrate(records: Sequence[dict], fail_budget: float = 0.05) -> Dict:
    """Pick tau from `records` of {"iou": float, "admits_cell": bool}.

    Returns the SMALLEST tau at which at most `fail_budget` of the
    accepted cartridges admit a cell inside their optimistic error, plus
    the full sweep table (`"table"`) so a caller can show the whole
    accuracy/throughput trade, not just the winning row. `tau: None`
    means no threshold satisfies the budget - a negative result about
    the segmenter, reported as one rather than tuned around.
    """
    if not records:
        return {"tau": None, "note": "no records", "n": 0, "table": []}

    n = len(records)
    rows = _sweep(records, fail_budget)

    for row in rows:
        if row["meets_budget"]:
            return {
                "tau": float(row["tau"]),
                "fail_rate": row["fail_rate"],
                "n": n,
                "n_accepted": row["n_accepted"],
                "rejected_fraction": row["rejected_fraction"],
                "fail_budget": fail_budget,
                "note": (f"smallest tau meeting the budget; "
                         f"{row['n_accepted']}/{n} cartridges accepted"),
                "table": rows,
            }

    return {
        "tau": None,
        "n": n,
        "fail_budget": fail_budget,
        "note": ("No threshold meets the budget: even the most confident "
                 "cartridges admit a cell inside their optimistic error. "
                 "This is a negative result about the segmenter, and is "
                 "reported rather than tuned around."),
        "table": rows,
    }


# --------------------------------------------------------- record collection --

def collect_records(segmenter, full_dataset, val_indices: Sequence[int],
                     scales: Dict[int, float], wall_inset_mm: float,
                     cell_w_mm: float, cell_h_mm: float
                     ) -> List[Dict[str, Any]]:
    """One record per validation crop: arbitration IoU, whether its
    optimistic error admits a cell, and whether its predicted direct
    estimate (`bay` channel) is non-empty at all.

    Runs the segmenter at NATIVE crop resolution (`out_size=None`),
    matching recog.seg_evaluate.evaluate() - a jittered union box is not
    square, so a single scalar mm_per_px could only describe the native
    frame anyway.

    ``scales`` maps crop index -> that crop's OWN mm_per_px
    (:func:`recog.seg_evaluate.resolve_frame_scales`), and the three
    millimetre quantities are converted to pixels PER CROP rather than
    once for the split. That conversion is not cosmetic here the way it
    is for a reported figure: a 4.25 mm tray wall is 7 px at the nominal
    0.625 mm/px but 4-9 px across this corpus's frames (5 px at their
    0.82 median), and an 18.3 x 65 mm cell is 29 x 104 px against
    22 x 79 px at that median. Both feed the measurement
    itself - the erosion radius `arbitrate` applies and the structuring
    element `admits_a_cell` tries to fit - so calibrating at one constant
    scale eroded too far and then tested with a cell too large, on a
    corpus where no frame had that scale.

    A scalar is refused rather than accepted, for the same reason
    ``evaluate`` refuses one: a signature that still takes one number can
    regress to one number without anything failing.
    """
    from PIL import Image

    from plan.arbitration import CH_BAY, admits_a_cell, arbitrate, direct_placement
    from recog.seg_dataset import extract_crop, rasterise_crop

    if isinstance(scales, (int, float)):
        raise TypeError(
            "collect_records() takes a per-crop {index: mm_per_px} map, "
            f"not one constant ({scales!r}) - see "
            "recog.seg_evaluate.resolve_frame_scales.")

    records: List[Dict[str, Any]] = []
    for idx in val_indices:
        img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)   # no jitter at eval
        mm_per_px = scales[idx]                 # THIS frame's own GSD

        wall_inset_px = max(0, int(round(wall_inset_mm / mm_per_px)))
        cell_w_px = cell_w_mm / mm_per_px
        cell_h_px = cell_h_mm / mm_per_px

        path = Path(full_dataset.img_dir) / img_meta["file_name"]
        image = np.asarray(Image.open(path).convert("RGB"))
        crop = extract_crop(image, box, out_size=None)
        pred = segmenter.segment_batch([crop])[0]

        safe, iou = arbitrate(pred, wall_inset_px)
        predicts_bay = bool(direct_placement(pred).any())

        gt = rasterise_crop(anns, box, out_size=None)
        gt_bay = gt == CH_BAY
        optimistic_error = safe & ~gt_bay
        admits = admits_a_cell(optimistic_error, cell_w_px, cell_h_px)

        records.append({
            "idx": int(idx),
            "file": img_meta["file_name"],
            "mm_per_px": float(mm_per_px),
            "wall_inset_px": int(wall_inset_px),
            "cell_px": (float(cell_w_px), float(cell_h_px)),
            "iou": float(iou),
            "admits_cell": bool(admits),
            "predicts_bay": predicts_bay,
            "optimistic_error_px": int(optimistic_error.sum()),
        })
    return records


# --------------------------------------------------------------- report --

def _table_lines(rows: Sequence[Dict], picked_tau: Optional[float]) -> List[str]:
    head = (f"  {'tau':>8}{'n_accepted':>12}{'n_fail':>8}"
            f"{'fail_rate':>11}{'rejected_frac':>15}{'meets_budget':>14}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for row in rows:
        marker = " <-- picked" if picked_tau is not None and \
            row["tau"] == picked_tau else ""
        rate_s = "nan" if np.isnan(row["fail_rate"]) else f"{row['fail_rate']:.4f}"
        lines.append(
            f"  {row['tau']:>8.4f}{row['n_accepted']:>12}{row['n_fail']:>8}"
            f"{rate_s:>11}{row['rejected_fraction']:>15.4f}"
            f"{str(row['meets_budget']):>14}{marker}")
    lines.append("  " + "-" * (len(head) - 2))
    return lines


def format_report(result: Dict, *, records: Sequence[dict],
                  all_records: Sequence[dict], all_result: Dict,
                  fail_budget: float,
                  checkpoint: str, config_path: str,
                  synth_config_source: str,
                  scale_stats: Dict[str, float],
                  scale_provenance: Dict[str, Any],
                  nominal_mm_per_px: Optional[float],
                  wall_inset_mm: float,
                  cell_w_mm: float, cell_h_mm: float,
                  n_val_total: int, split_seed: int) -> str:
    median_mm_per_px = scale_stats["median"]
    inset_px = [r["wall_inset_px"] for r in all_records] or [0]
    lines: List[str] = []
    lines.append("")
    lines.append("Tau calibration (placement-disagreement threshold)")
    lines.append("=" * 52)
    lines.append(f"  checkpoint        : {checkpoint}")
    lines.append(f"  config            : {config_path}")
    lines.append(f"  synth config      : {synth_config_source}")
    # Not one number: this receipt quoted mm_per_px = 0.6250 for a corpus
    # in which no frame is rendered at that framing, and every millimetre
    # -> pixel conversion below inherited it. See
    # docs/superpowers/specs/2026-08-11-scale-figures.md.
    lines.append("  mm_per_px         : per frame, from each frame's own "
                 "render sidecar (camera.ortho_scale*1000 / width)")
    lines.append(f"    measured        : {scale_provenance['n_measured']} of "
                 f"{scale_provenance['n_measured'] + scale_provenance['n_fallback']}"
                 f" crops over {scale_provenance['n_frames']} frames")
    lines.append(f"    distribution    : median {median_mm_per_px:.4f}, "
                 f"range {scale_stats['min']:.4f}-{scale_stats['max']:.4f} "
                 "mm/px")
    fb = scale_provenance.get("fallback")
    lines.append("    fallback        : "
                 + ("none configured - an uncalibrated frame raises "
                    "UnknownScale rather than reverting to a constant"
                    if fb is None else
                    f"{fb:.4f} mm/px, applied to "
                    f"{scale_provenance['n_fallback']} crops"))
    if nominal_mm_per_px:
        lines.append(
            f"    note            : the generator's NOMINAL framing "
            f"({nominal_mm_per_px:.4f}) is the framing at margin=1.0, "
            f"zoom=1.0 and describes NO frame here. This calibration used "
            f"it as a constant before 2026-08-11, which eroded the wall "
            f"inset too deep AND tested with a cell too large, on every "
            f"frame at once.")
    lines.append(f"  wall_inset        : {wall_inset_mm:.2f} mm - "
                 "common.types' DEFAULT_WALL_INSET_MM - eroded "
                 f"per frame at {min(inset_px)}-{max(inset_px)} px "
                 f"({int(round(wall_inset_mm / median_mm_per_px))} px at "
                 "the median GSD)")
    lines.append(f"  cell footprint    : {cell_w_mm:.1f} x {cell_h_mm:.1f} mm "
                 "(18650, CAD-measured), "
                 f"{cell_w_mm / median_mm_per_px:.0f} x "
                 f"{cell_h_mm / median_mm_per_px:.0f} px at the median GSD")
    lines.append(f"  fail budget       : {fail_budget:.2%} of accepted "
                 "cartridges may admit a cell")
    lines.append(f"  split             : seed={split_seed}, "
                 f"{n_val_total} val crops total")
    lines.append("")

    lines.append(f"Population: cartridges with a PREDICTED bay region "
                 f"(P_direct non-empty) - {len(records)} of {n_val_total} "
                 "val crops. See the module docstring for why this is the "
                 "only non-vacuous population: P_safe is empty whenever "
                 "P_direct is empty, regardless of tau, so those crops can "
                 "never admit a cell and tau never actually gates them.")
    if records:
        ious = np.array([r["iou"] for r in records], dtype=np.float64)
        lines.append(
            f"  IoU distribution: n={ious.size}, mean={ious.mean():.4f}, "
            f"std={ious.std():.4f}, min={ious.min():.4f}, "
            f"max={ious.max():.4f}, "
            f"fraction>0.95={float((ious > 0.95).mean()):.4f}")
    lines.append("")
    lines.append("Per-cartridge records (this population). mm_per_px is "
                 "THIS frame's own ground")
    lines.append("sample distance, and inset_px / the cell footprint "
                 "admits_a_cell tried to fit are")
    lines.append("derived from it per row, not from one split-wide "
                 "constant:")
    head = (f"  {'file':<28}{'mm/px':>8}{'inset_px':>10}{'iou':>8}"
            f"{'opt_err_px':>12}{'admits_cell':>13}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for r in sorted(records, key=lambda r: r["iou"]):
        lines.append(f"  {r['file']:<28}{r['mm_per_px']:>8.4f}"
                     f"{r['wall_inset_px']:>10}{r['iou']:>8.4f}"
                     f"{r['optimistic_error_px']:>12}"
                     f"{str(r['admits_cell']):>13}")
    lines.append("  " + "-" * (len(head) - 2))
    lines.append("")

    lines.append("Calibration sweep (every observed IoU tried as a "
                 "candidate tau, ascending):")
    lines.extend(_table_lines(result["table"], result["tau"]))
    lines.append("")

    if result["tau"] is None:
        lines.append("RESULT: tau = None.")
        lines.append(f"  {result['note']}")
        lines.append("  No confidence threshold makes this segmenter safe "
                     "at the stated budget. The correct response is to "
                     "improve the segmenter, not to lower the budget until "
                     "a number appears.")
    else:
        lines.append(f"RESULT: tau = {result['tau']:.4f}")
        lines.append(f"  fail_rate         : {result['fail_rate']:.4f} "
                     f"(budget {fail_budget:.4f})")
        lines.append(f"  accepted          : {result['n_accepted']}/{result['n']} "
                     "cartridges")
        lines.append(f"  rejected_fraction : {result['rejected_fraction']:.4f}")
        lines.append(f"  {result['note']}")
    lines.append("")

    lines.append(f"Sample size caveat: n={len(records)} cartridges. With "
                 f"fail_budget={fail_budget:.2%}, a single admitting "
                 f"cartridge in an accepted set this small already exceeds "
                 f"the budget for any accepted set under "
                 f"{1.0 / fail_budget:.0f} cartridges - so at this sample "
                 "size the calibration is, in practice, closer to a "
                 "zero-tolerance test than a genuine 5% tolerance test. "
                 "Treat tau as provisional until the validation split "
                 "grows.")
    lines.append("")

    n_fail_total = sum(1 for r in records if r["admits_cell"])
    if records and n_fail_total == 0:
        # Per record, because a crop's error in pixels only means
        # anything against a cell footprint in THAT crop's pixels.
        worst = max(records,
                    key=lambda r: r["optimistic_error_px"]
                    / (r["cell_px"][0] * r["cell_px"][1]))
        cell_area_px = worst["cell_px"][0] * worst["cell_px"][1]
        max_err_px = worst["optimistic_error_px"]
        lines.append(
            "IMPORTANT caveat, not just a sample-size one: NOT ONE of the "
            f"{len(records)} accepted cartridges admitted a cell, at any "
            "observed IoU down to the sample minimum - so fail_budget "
            "never actually bound, and the returned tau is simply the "
            "LOWEST IoU this split happened to contain, not a threshold "
            "discovered by trading safety against throughput. It is a "
            "lower bound on where an unsafe boundary might sit, not "
            "evidence that IoU below it is unsafe or that IoU at or above "
            "it is safe - this split never observed a failure to locate "
            "that boundary against.")
        lines.append(
            f"  Additionally: the largest optimistic error observed was "
            f"{max_err_px} px, against a {cell_w_mm:.1f}x{cell_h_mm:.1f} mm "
            f"cell footprint of {cell_area_px:.0f} px^2 at this framing - "
            f"{max_err_px / cell_area_px:.1%} of one cell's area. Every "
            "record in this split fails on AREA ALONE; the morphological "
            "vs. areal distinction admits_a_cell exists for (Task 2's "
            "blob-vs-rim demonstration) is never exercised by this "
            "particular validation split, because no crop's error gets "
            "close enough to a cell's footprint to make shape the "
            "deciding factor. A validation set with larger errors would "
            "be a stronger test of tau, not just a larger one.")
        lines.append("")

    lines.append("Cross-check only, NOT the calibration result - calibrate() "
                 f"run over ALL {len(all_records)} val crops (including "
                 "those with no predicted bay, which can never admit a "
                 "cell and should never be read as evidence of safety):")
    if all_result["tau"] is None:
        lines.append(f"  tau = None ({all_result['note']})")
    else:
        lines.append(f"  tau = {all_result['tau']:.4f}, "
                     f"rejected_fraction = {all_result['rejected_fraction']:.4f}, "
                     f"n={all_result['n']}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- CLI --

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.calibrate_tau",
        description="Calibrate the arbitration IoU threshold tau against "
                    "the trained segmenter's validation split.")
    ap.add_argument("--checkpoint", default="recog/checkpoints/seg/best.pt")
    ap.add_argument("--config", default="configs/segmentation.yaml")
    ap.add_argument("--synth-config", default=None,
                    help="render/layout config to read the generator's "
                         "NOMINAL framing from, for the receipt's note "
                         "only - it is no longer what millimetres are "
                         "converted at. Defaults to the dataset's own "
                         "manifest.json, falling back to "
                         "configs/synth3d.yaml.")
    ap.add_argument("--fallback-mm-per-px", type=float, default=None,
                    help="scale for frames that carry NO render sidecar. "
                         "Unset by default: an uncalibrated frame then "
                         "raises rather than silently reverting to a "
                         "constant.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fail-budget", type=float, default=0.05)
    ap.add_argument("--wall-inset-mm", type=float,
                    default=DEFAULT_WALL_INSET_MM,
                    help="Defaults to the extractor's own default, "
                         "so this scores against the same erosion "
                         "production actually applies.")
    ap.add_argument("--cell-w-mm", type=float, default=CELL_W_MM)
    ap.add_argument("--cell-h-mm", type=float, default=CELL_H_MM)
    ap.add_argument("--out", default="docs/receipts/tau_calibration.txt")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"error: checkpoint not found: {args.checkpoint}")

    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model") or {}
    ds_cfg = cfg.get("dataset") or {}

    num_classes = int(model_cfg.get("num_classes", 6))
    crop_size = int(model_cfg.get("crop_size", 256))
    half = bool(model_cfg.get("half", True))

    from recog.seg_evaluate import load_synth_config, resolve_mm_per_px

    # The NOMINAL framing, kept for the receipt's note only. Nothing
    # below converts millimetres with it.
    synth_cfg, synth_source = load_synth_config(
        ds_cfg["coco_path"], args.synth_config)
    synth_source = Path(synth_source).as_posix()
    nominal_mm_per_px = resolve_mm_per_px(synth_cfg)

    from recog.bay_segmenter import BaySegmenter
    from recog.seg_dataset import BaySegDataset
    from recog.seg_evaluate import (check_split_matches_checkpoint,
                                    compute_val_instance_counts,
                                    resolve_frame_scales, scale_stats)
    from recog.seg_training import _split_dataset

    full_dataset = BaySegDataset(
        coco_path=ds_cfg["coco_path"],
        img_dir=ds_cfg["img_dir"],
        out_size=crop_size,
        jitter_frac=float(ds_cfg.get("jitter_frac", 0.06)),
        train=True,
        transform=None,
    )
    if len(full_dataset) == 0:
        raise SystemExit(f"error: no crops found for {ds_cfg['coco_path']}")

    split_seed = int(ds_cfg.get("split_seed", 0))
    _train_set, val_set = _split_dataset(
        full_dataset, float(ds_cfg.get("train_val_split", 0.85)),
        seed=split_seed)
    val_indices = list(val_set.indices)
    log.info("val split: seed=%d, %d of %d crops", split_seed,
             len(val_indices), len(full_dataset))

    # Same split-integrity guard recog.seg_evaluate uses, and for the
    # same reason: the dataset is gitignored and resumable, so a moved
    # split would silently calibrate against crops this checkpoint
    # trained on.
    #
    # Two corrections against the version that shipped before
    # 2026-08-11, both of which made this CLI unrunnable or wrong:
    # `coco_path` was never passed, so this line raised TypeError before
    # reaching a single measurement ever since 75db46a added the argument
    # (the same defect 380e7d5 fixed in recog.seg_ablation - and the
    # reason docs/receipts/tau_calibration.txt could not be regenerated);
    # and `out_size` was left at native while seg_training's stored
    # counts come off a DataLoader that rasterises at crop_size, so the
    # guard would have fired on a dataset that never moved. Matches
    # recog.seg_evaluate.main's own call on both counts.
    val_counts_now = compute_val_instance_counts(
        full_dataset, val_indices, num_classes=num_classes,
        out_size=crop_size)
    check_split_matches_checkpoint(args.checkpoint, ds_cfg["coco_path"],
                                   val_counts_now)

    # Resolved BEFORE inference: an uncalibrated frame is a configuration
    # error and should cost nothing but the time to notice it.
    scales, scale_provenance = resolve_frame_scales(
        full_dataset, val_indices, fallback=args.fallback_mm_per_px)
    stats = scale_stats(scales)
    log.info("scale: %d of %d crops calibrated from their own frame; "
             "median %.4f, range %.4f-%.4f mm/px (nominal framing %.4f)",
             scale_provenance["n_measured"], len(val_indices),
             stats["median"], stats["min"], stats["max"], nominal_mm_per_px)

    segmenter = BaySegmenter(checkpoint=args.checkpoint, device=args.device,
                             crop_size=crop_size, half=half,
                             num_classes=num_classes)

    all_records = collect_records(
        segmenter, full_dataset, val_indices, scales, args.wall_inset_mm,
        args.cell_w_mm, args.cell_h_mm)
    records = [r for r in all_records if r["predicts_bay"]]

    result = calibrate(records, fail_budget=args.fail_budget)
    # Cross-check row only, NOT the calibration result - computed here
    # rather than inside format_report, which should format a result,
    # not derive one (final whole-branch review).
    all_result = calibrate(list(all_records), fail_budget=args.fail_budget)

    report = format_report(
        result, records=records, all_records=all_records,
        all_result=all_result,
        fail_budget=args.fail_budget, checkpoint=args.checkpoint,
        config_path=args.config, synth_config_source=synth_source,
        scale_stats=stats, scale_provenance=scale_provenance,
        nominal_mm_per_px=nominal_mm_per_px,
        wall_inset_mm=args.wall_inset_mm, cell_w_mm=args.cell_w_mm,
        cell_h_mm=args.cell_h_mm, n_val_total=len(val_indices),
        split_seed=split_seed)

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")

    if result["tau"] is None:
        log.error("no tau satisfies the fail budget - this is a finding "
                  "about the segmenter, not something to tune around")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_W_MM",
    "CELL_H_MM",
    "calibrate",
    "collect_records",
    "format_report",
    "main",
]
