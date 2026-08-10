"""
recog.seg_evaluate - the metrics that decide whether the segmenter ships.

Per-class IoU is table stakes. Three others carry the argument:

  boundary_displacement_mm  the quantity that chose this architecture.
                            Spec 3.1 rejected a Mask R-CNN head on a
                            2.9 x 6.4 mm quantisation figure; reporting
                            only IoU makes that argument unfalsifiable.

  signed_area_error_mm2     optimistic error sites a cell where one
                            cannot go, a damage event. Conservative
                            error refuses where one can, a lost cell.
                            One unsigned number hides the difference.

  latency_table             the 50 ms budget has no slack. See
                            bay_segmenter's module docstring.

CLI::

    python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt \\
        --config configs/segmentation.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import load_yaml
from common.logging import get_logger

# Derived from the dataset's contract rather than restated. Two
# hand-written copies of the same order drift, and the drift is silent:
# the metrics would simply be reported under the wrong class names.
# extract_crop / rasterise_crop are the SAME functions BaySegDataset
# trains against (out_size=None asks for native, un-resized output) -
# see their docstrings in seg_dataset.py. A hand-copied second version of
# either painting loop would drift the moment one side changed and
# nothing would notice; delegating means there is only one place to fix.
from recog.seg_dataset import SEG_CHANNELS, extract_crop, rasterise_crop

CHANNEL_NAMES = [name for name, _ in
                 sorted(SEG_CHANNELS.items(), key=lambda kv: kv[1])]

log = get_logger("recog.seg_evaluate")

# The three classes the placement mask is actually built from - matches
# configs/segmentation.yaml's training.select_on. IoU, boundary
# displacement and area error are all reported for these three; the
# other three (background, cartridge, battery) are context, not target.
SELECT_ON: Tuple[str, ...] = ("bay", "electronics", "obstruction")

# Spec 3.1's rejection figure for a 28x28 Mask R-CNN head at the
# generator's framing. Boundary displacement below this is the
# acceptance criterion for having chosen a per-ROI segmenter instead.
MASK_HEAD_QUANTISATION_MM: Tuple[float, float] = (2.9, 6.4)


# ------------------------------------------------------------- metrics --

def _class_confusion(pred: np.ndarray, target: np.ndarray,
                     c: int) -> Tuple[int, int]:
    """(intersection, union) pixel counts for class ``c``.

    The one place this per-class boolean comparison happens.
    ``per_class_iou`` calls it for a single (pred, target) pair;
    ``evaluate`` calls it once per crop and pools the results across the
    split - so the pooled IoU the receipt reports and the only IoU
    function with a test are the same code, not two copies that can
    drift apart.
    """
    p, t = pred == c, target == c
    return int((p & t).sum()), int((p | t).sum())


def per_class_iou(pred: np.ndarray, target: np.ndarray,
                  num_classes: int = 6) -> Dict[str, float]:
    """IoU per class. NaN where a class appears in neither array.

    NaN rather than 0.0: a class that was never present was never
    tested, and scoring it zero drags the mean down for a question that
    was not asked.
    """
    out: Dict[str, float] = {}
    for c in range(num_classes):
        inter, union = _class_confusion(pred, target, c)
        out[CHANNEL_NAMES[c]] = (float(np.nan) if union == 0
                                 else float(inter) / union)
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` with at least one 4-neighbour outside it."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    interior = (pad[:-2, 1:-1] & pad[2:, 1:-1] &
                pad[1:-1, :-2] & pad[1:-1, 2:])
    return m & ~interior


def boundary_displacement_mm(pred: np.ndarray, target: np.ndarray,
                             cls: int, mm_per_px: float) -> float:
    """Mean distance from each predicted boundary pixel to the nearest
    ground-truth boundary pixel, in millimetres."""
    pb = _boundary(pred == cls)
    tb = _boundary(target == cls)
    if not pb.any() or not tb.any():
        return float("nan")

    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~tb)
    except Exception:
        ty, tx = np.nonzero(tb)
        yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]]
        dist = np.min(
            np.sqrt((yy[..., None] - ty) ** 2 + (xx[..., None] - tx) ** 2),
            axis=-1)

    return float(dist[pb].mean() * mm_per_px)


def signed_area_error_mm2(pred: np.ndarray, target: np.ndarray,
                          cls: int, mm_per_px: float
                          ) -> Tuple[float, float]:
    """``(optimistic_mm2, conservative_mm2)`` for class ``cls``.

    Optimistic = predicted placeable where truth is not. That is the
    error that puts a cell on a PCB.
    """
    p, t = pred == cls, target == cls
    px_mm2 = mm_per_px * mm_per_px
    return (float((p & ~t).sum()) * px_mm2,
            float((~p & t).sum()) * px_mm2)


def latency_table(segmenter, counts: Sequence[int] = (1, 2, 4, 8),
                  crop_hw: Tuple[int, int] = (131, 288),
                  repeats: int = 20) -> List[dict]:
    """Wall-clock per batch size, against the 50 ms PPR budget.

    Each row also carries ``looped_ms``: the same ``n`` crops pushed
    through ``segmenter.segment_batch`` one at a time (batch size 1, `n`
    calls) rather than one batched call of size `n` - the path Task 5
    replaced. This is the number that justifies batching, and it used to
    live only as three different hand-measured figures scattered across
    the FDR, the README and two docstrings (final whole-branch review).
    Measuring it here, on the SAME segmenter/crops/warmup as the batched
    figure right next to it, gives one receipt both can be quoted from
    instead of drifting independently.
    """
    rows: List[dict] = []
    for n in counts:
        crops = [np.zeros((crop_hw[0], crop_hw[1], 3), np.uint8)
                 for _ in range(n)]

        for _ in range(3):
            segmenter.segment_batch(crops)
        t0 = time.perf_counter()
        for _ in range(repeats):
            segmenter.segment_batch(crops)
        total_ms = (time.perf_counter() - t0) / repeats * 1000.0

        for _ in range(3):
            for c in crops:
                segmenter.segment_batch([c])
        t0 = time.perf_counter()
        for _ in range(repeats):
            for c in crops:
                segmenter.segment_batch([c])
        looped_ms = (time.perf_counter() - t0) / repeats * 1000.0

        rows.append({"cartridges": n,
                     "total_ms": round(total_ms, 1),
                     "per_cartridge_ms": round(total_ms / n, 2),
                     "within_50ms_budget": total_ms <= 50.0,
                     "looped_ms": round(looped_ms, 1)})
    return rows


def latency_within_budget(latency: List[dict], budget_batch: int = 8) -> bool:
    """Whether the `budget_batch`-cartridge row is within the 50 ms
    budget. Pulled out of main() so the plan's acceptance criterion
    (latency vs the FDR 10.4 budget) is a function CI can call and gate
    on, not only a log line main() decided to emit."""
    return all(r["within_50ms_budget"] for r in latency
              if r["cartridges"] == budget_batch)


# --------------------------------------------------------------- config --

def resolve_mm_per_px(synth_cfg: Dict[str, Any]) -> float:
    """``layout.area[0] * 1000 / render.res[0]`` - the generator's
    framing. Read from config rather than hardcoded, or it silently goes
    wrong the first time anyone changes the framing."""
    layout = synth_cfg.get("layout") or {}
    render = synth_cfg.get("render") or {}
    if "area" not in layout or "res" not in render:
        raise SystemExit(
            "error: synth config is missing layout.area / render.res; "
            "cannot compute mm_per_px. Pass --synth-config explicitly.")
    area_w_m = float(layout["area"][0])
    res_w_px = float(render["res"][0])
    return area_w_m * 1000.0 / res_w_px


def load_synth_config(coco_path: str, override: Optional[str]
                      ) -> Tuple[Dict[str, Any], str]:
    """The render/layout config the dataset's images were actually
    generated under.

    Prefers the manifest.json sidecar next to ``coco_path`` - it is a
    frozen copy of the config at generation time, so it stays correct
    even if configs/synth3d.yaml is edited afterwards. Falls back to
    configs/synth3d.yaml (the authored default) only if no manifest is
    found, and to an explicit --synth-config if one is given.
    """
    if override:
        return load_yaml(override), str(override)

    manifest_path = Path(coco_path).parent / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        cfg = manifest.get("config")
        if cfg:
            return cfg, str(manifest_path)

    default = Path("configs/synth3d.yaml")
    if default.is_file():
        return load_yaml(default), str(default)

    raise SystemExit(
        "error: could not find a render/layout config to compute "
        f"mm_per_px from ({manifest_path} missing 'config', and "
        f"{default} not found). Pass --synth-config explicitly.")


# --------------------------------------------------------------- report --

def _mean_or_nan(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def evaluate(segmenter, full_dataset, val_indices: Sequence[int],
            mm_per_px: float, num_classes: int = 6
            ) -> Dict[str, Any]:
    """Run the segmenter over every validation crop at native resolution
    and accumulate every metric this module reports.

    "Native resolution" (extract_crop / rasterise_crop called with
    out_size=None) rather than the model's fixed square input: the
    render's mm_per_px (layout.area / render.res) is isotropic only in
    that native frame - a jittered union box is not square, so resizing
    it to out_size x out_size would rescale x and y by different factors,
    and a single scalar mm_per_px could not describe the result
    correctly. Boundary displacement and area error need that native
    frame; IoU is computed there too so all four numbers describe the
    same pixels.

    IoU is pooled pixel-wise over the whole split (sum of intersections
    / sum of unions across crops), matching recog.seg_training's
    evaluate_model so the two are directly comparable. Boundary
    displacement and area error are per-crop: a per-crop mean is the
    "typical cartridge" figure, and the per-crop values are also summed
    for a "total exposure over the split" figure.

    Also measures, per crop, whether `cartridge` and `bay` pixels
    co-occur (``cartridge_bay_crops`` in the return value) - this used
    to be assumed disjoint in prose; it no longer is (see
    seg_dataset.py's module docstring), so format_report reads the
    real count instead of restating an assumption.
    """
    from PIL import Image

    inter = [0] * num_classes
    union = [0] * num_classes
    counts = [0] * num_classes            # crops containing class c

    boundary_mm: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    boundary_defined: Dict[str, int] = {c: 0 for c in SELECT_ON}
    area_opt_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    area_cons_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}

    # cartridge/bay crop-population overlap, measured fresh every run
    # rather than asserted in prose. Before the tray-interior fix
    # (27cbd97..9fcf136) a closed shell's cartridge mask and an open
    # unit's bay mask genuinely never shared a crop; the fix gave open
    # units real tray walls, which paint `cartridge` pixels alongside
    # their own bay/electronics/obstruction pixels in the SAME crop
    # (see seg_dataset.py's module docstring). Recomputing this here
    # means the receipt's note can never silently outlive the geometry
    # it describes again.
    cart_c, bay_c = SEG_CHANNELS["cartridge"], SEG_CHANNELS["bay"]
    cb_cart_only = cb_bay_only = cb_both = cb_neither = 0

    for idx in val_indices:
        img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)     # no jitter at eval

        path = Path(full_dataset.img_dir) / img_meta["file_name"]
        image = np.asarray(Image.open(path).convert("RGB"))

        crop = extract_crop(image, box, out_size=None)
        target = rasterise_crop(anns, box, out_size=None)
        pred = segmenter.segment_batch([crop])[0]

        for c in range(num_classes):
            i, u = _class_confusion(pred, target, c)
            inter[c] += i
            union[c] += u
            if (target == c).any():
                counts[c] += 1

        has_cart = bool((target == cart_c).any())
        has_bay = bool((target == bay_c).any())
        if has_cart and has_bay:
            cb_both += 1
        elif has_cart:
            cb_cart_only += 1
        elif has_bay:
            cb_bay_only += 1
        else:
            cb_neither += 1

        for name in SELECT_ON:
            c = SEG_CHANNELS[name]
            bd = boundary_displacement_mm(pred, target, c, mm_per_px)
            if not np.isnan(bd):
                boundary_mm[name].append(bd)
                boundary_defined[name] += 1
            opt, cons = signed_area_error_mm2(pred, target, c, mm_per_px)
            area_opt_mm2[name].append(opt)
            area_cons_mm2[name].append(cons)

    ious = {CHANNEL_NAMES[c]: (float(inter[c]) / union[c]
                               if union[c] > 0 else float("nan"))
            for c in range(num_classes)}
    counts_by_name = dict(zip(CHANNEL_NAMES, counts))

    present = [ious[c] for c in SELECT_ON if not np.isnan(ious[c])]
    selected_mean_iou = (sum(present) / len(present)) if present else float("nan")

    return {
        "n_val_crops": len(val_indices),
        "ious": ious,
        "instance_counts": counts_by_name,
        "selected_mean_iou": selected_mean_iou,
        "boundary_mm": {c: _mean_or_nan(boundary_mm[c]) for c in SELECT_ON},
        "boundary_n": boundary_defined,
        "area_opt_mm2_mean": {c: _mean_or_nan(area_opt_mm2[c]) for c in SELECT_ON},
        "area_cons_mm2_mean": {c: _mean_or_nan(area_cons_mm2[c]) for c in SELECT_ON},
        "area_opt_mm2_total": {c: float(sum(area_opt_mm2[c])) for c in SELECT_ON},
        "area_cons_mm2_total": {c: float(sum(area_cons_mm2[c])) for c in SELECT_ON},
        "cartridge_bay_crops": {
            "cartridge_only": cb_cart_only,
            "bay_only": cb_bay_only,
            "both": cb_both,
            "neither": cb_neither,
        },
    }


# ----------------------------------------------------------- split guard --
#
# recog.seg_training writes split_seed and val_instance_counts into every
# checkpoint (seg_training.py:404-413) precisely so the split a checkpoint
# was selected/reported against can be checked later. recog/dataset3d_seg
# is gitignored and came from a resumable generation run stopped at 220 of
# 300 scenes; resuming it (or regenerating it, or pointing --config at a
# different coco_path) changes len(full_dataset), so random_split returns
# a DIFFERENT partition with the same seed, and best.pt would silently be
# scored on crops it trained on. There is no way to auto-correct a moved
# split - only to say so, loudly, naming both sets of numbers.

def compute_val_instance_counts(full_dataset, val_indices: Sequence[int],
                                num_classes: int = 6) -> Dict[str, int]:
    """Per-class crop counts for the val split (no jitter, matching
    evaluate()'s boxes), computed the same way seg_training.instance_counts
    does: a crop "contains" a class if it has at least one pixel of it."""
    counts = [0] * num_classes
    for idx in val_indices:
        _img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)
        target = rasterise_crop(anns, box, out_size=None)
        for c in range(num_classes):
            if (target == c).any():
                counts[c] += 1
    return dict(zip(CHANNEL_NAMES, counts))


def group_indices_by_asset(full_dataset, val_indices: Sequence[int]
                           ) -> Dict[Optional[str], List[int]]:
    """`val_indices` partitioned by which catalog asset (SKU) each crop's
    unit belongs to, via `BaySegDataset.sample_assets` (Task 12).

    Design spec Sec7/Sec10/Sec12: every comparison this measurement makes
    has to be reported per-SKU first, pooled figure alongside it, never
    in place of it - this is what makes that grouping possible by
    calling the EXISTING `evaluate()` once per group, rather than
    threading a new SKU-aware code path through it.
    """
    out: Dict[Optional[str], List[int]] = {}
    for idx in val_indices:
        out.setdefault(full_dataset.sample_assets[idx], []).append(idx)
    return out


def format_per_sku_table(per_sku_results: Dict[str, Dict[str, Any]]) -> str:
    """A compact per-SKU IoU table, one row per asset, for the classes in
    SELECT_ON plus 'battery' (design spec Sec12's regression floor names
    both explicitly). Appended to the same report `format_report`
    produces - not folded into it, to keep the well-tested pooled report
    untouched by this addition.
    """
    cols = SELECT_ON + ("battery",)
    lines = ["", "Per-SKU IoU (design spec Sec7/Sec10/Sec12):",
            "  " + "asset".ljust(24) + "n_crops".rjust(9)
            + "".join(c.rjust(14) for c in cols)]
    for asset, res in sorted(per_sku_results.items(),
                             key=lambda kv: (kv[0] is None, kv[0])):
        ious = res.get("ious", {})
        row = ("  " + str(asset).ljust(24)
              + str(res.get("n_val_crops", 0)).rjust(9)
              + "".join(f"{ious.get(c, float('nan')):.4f}".rjust(14)
                        for c in cols))
        lines.append(row)
    return "\n".join(lines)


def check_split_matches_checkpoint(checkpoint_path: str,
                                   recomputed: Dict[str, int]) -> None:
    """Fail loudly if today's recomputed split disagrees with what the
    checkpoint itself recorded at training time. Never silently re-split
    or auto-correct - that would hide exactly the failure this guards."""
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = state.get("val_instance_counts")
    if saved is None:
        log.warning("checkpoint %s has no val_instance_counts (older "
                    "checkpoint format) - cannot verify the validation "
                    "split matches what it was trained against",
                    checkpoint_path)
        return
    if dict(saved) != dict(recomputed):
        raise SystemExit(
            "error: the recomputed validation split does not match the "
            f"checkpoint's own record - scoring would silently include "
            f"crops this checkpoint trained on.\n"
            f"  checkpoint {checkpoint_path} recorded: {dict(saved)}\n"
            f"  this run recomputes:                   {dict(recomputed)}\n"
            "The dataset behind --config's dataset.coco_path most likely "
            "changed size or content since this checkpoint was written "
            "(e.g. a resumable recog.generate3d run was resumed or "
            "regenerated) - random_split now returns a different "
            "partition for the same split_seed. Point --config at the "
            "exact dataset this checkpoint was trained on, or retrain "
            "against the current dataset. Not auto-correcting: guessing "
            "the right split here would hide the same class of bug this "
            "check exists to catch.")


def _sibling_checkpoint_note(checkpoint_path: str) -> Optional[str]:
    """A caveat line comparing best.pt and last.pt's OWN recorded
    selection metric, read straight from the checkpoint files (no
    re-inference needed - seg_training already wrote it). Both are
    shipped (seg_training.py always writes last.pt unconditionally
    alongside a conditional best.pt); if the two are within noise of
    each other a reader needs to know checkpoint SELECTION was
    noise-limited, not that best.pt strictly dominates last.pt."""
    import torch

    ckpt_dir = Path(checkpoint_path).parent
    paths = {"best.pt": ckpt_dir / "best.pt", "last.pt": ckpt_dir / "last.pt"}
    if not all(p.is_file() for p in paths.values()):
        return None

    stats: Dict[str, Any] = {}
    for name, p in paths.items():
        state = torch.load(p, map_location="cpu", weights_only=True)
        stats[name] = (state.get("selected_mean_iou"),
                       state.get("val_instance_counts"))

    (b_iou, b_counts), (l_iou, l_counts) = stats["best.pt"], stats["last.pt"]
    if b_iou is None or l_iou is None:
        return None

    counts_str = "?"
    if b_counts:
        counts_str = "/".join(str(b_counts.get(c)) for c in SELECT_ON)
    return (f"  note: checkpoint selection is noise-limited - best.pt "
           f"{b_iou:.4f} and last.pt {l_iou:.4f} differ by "
           f"{abs(b_iou - l_iou):.4f} on {counts_str} "
           f"({'/'.join(SELECT_ON)}) val instances, and both ship.")


def format_report(results: Dict[str, Any], latency: List[dict], *,
                  checkpoint: Optional[str], config_path: str,
                  synth_config_source: str, mm_per_px: float,
                  device: str,
                  checkpoint_note: Optional[str] = None) -> str:
    """Plain-text receipt, in the style of recog/eval_real.py's report."""
    lines: List[str] = []
    lines.append("")
    lines.append("Bay segmenter evaluation")
    lines.append("=" * 40)
    lines.append(f"  checkpoint        : {checkpoint or '(none - random init)'}")
    lines.append(f"  config            : {config_path}")
    lines.append(f"  synth config      : {synth_config_source}")
    lines.append(f"  mm_per_px         : {mm_per_px:.4f} "
                 "(layout.area[0]*1000 / render.res[0])")
    lines.append(f"  device            : {device}")
    lines.append(f"  validation crops  : {results['n_val_crops']}")
    lines.append("")

    # ---- per-class IoU -----------------------------------------------
    lines.append("Per-class IoU (pooled over the validation split; NaN = "
                 "class absent from both pred and truth over the whole "
                 "split, not scored as 0):")
    head = f"  {'class':<12}{'IoU':>10}{'instances':>12}"
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for c in CHANNEL_NAMES:
        iou = results["ious"][c]
        iou_s = "NaN" if np.isnan(iou) else f"{iou:.4f}"
        n = results["instance_counts"][c]
        lines.append(f"  {c:<12}{iou_s:>10}{n:>12}")
    lines.append("  " + "-" * (len(head) - 2))
    sel_n = {c: results["instance_counts"][c] for c in SELECT_ON}
    lines.append(f"  selected mean IoU over {list(SELECT_ON)} "
                 f"(instances={sel_n}): {results['selected_mean_iou']:.4f}")
    bg_iou = results["ious"].get("background", float("nan"))
    bg_n = results["instance_counts"].get("background", 0)
    if not np.isnan(bg_iou):
        lines.append(
            f"  note: background IoU ({bg_iou:.4f} on {bg_n} crops) is "
            "STRUCTURAL, not a comparable failure - a crop is the union "
            "of the unit's OWN boxes, so background is only the thin "
            "leftover corners and gaps, a small region where IoU is "
            "naturally harsh. It is not evidence against the model.")
    cart_n = results["instance_counts"].get("cartridge", 0)
    bay_n = results["instance_counts"].get("bay", 0)
    if cart_n and bay_n:
        cart_iou = results["ious"].get("cartridge", float("nan"))
        bay_iou = results["ious"].get("bay", float("nan"))
        overlap = results.get("cartridge_bay_crops")
        if overlap is not None:
            verdict = "DISJOINT" if overlap["both"] == 0 else "OVERLAPPING"
            pop_note = (
                f"{verdict} on this validation split, measured fresh this "
                f"run (not assumed): {overlap['cartridge_only']} crops "
                f"carry cartridge only, {overlap['bay_only']} carry bay "
                f"only, {overlap['both']} carry BOTH in the SAME crop, "
                f"{overlap['neither']} carry neither. See seg_dataset.py's "
                "module docstring for why this can no longer be assumed "
                "disjoint (real tray-wall geometry, commits "
                "27cbd97..9fcf136).")
        else:                                    # pragma: no cover - guard
            pop_note = "of unmeasured overlap (cartridge_bay_crops missing)"
        lines.append(
            f"  note: cartridge ({cart_n} instances) and bay ({bay_n} "
            f"instances) crop populations are {pop_note} Either way, "
            f"{cart_iou:.4f} and {bay_iou:.4f} are each pooled "
            "independently over their own class's pixels across the "
            "whole split - a marginal per-class statistic, not a "
            "per-crop joint one - so this is still not evidence the "
            "model handles both together within the SAME crop, even on "
            "crops where both truly co-occur.")
    if checkpoint_note:
        lines.append(checkpoint_note)
    lines.append("")

    # ---- boundary displacement ----------------------------------------
    lo, hi = MASK_HEAD_QUANTISATION_MM
    lines.append("Boundary displacement, mm (mean distance from a "
                 "predicted boundary pixel to")
    lines.append("the nearest ground-truth boundary pixel; the figure "
                 "that chose a per-ROI")
    lines.append("segmenter over a 28x28 Mask R-CNN head, whose "
                 f"quantisation at this framing is {lo:.1f} x {hi:.1f} mm):")
    head = f"  {'class':<12}{'mm':>10}{'crops':>10}"
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    all_below = True
    for c in SELECT_ON:
        bd = results["boundary_mm"][c]
        n = results["boundary_n"][c]
        bd_s = "NaN" if np.isnan(bd) else f"{bd:.3f}"
        lines.append(f"  {c:<12}{bd_s:>10}{n:>10}")
        if not np.isnan(bd) and bd >= lo:
            all_below = False
    lines.append("  " + "-" * (len(head) - 2))
    verdict = ("BELOW the mask-head quantisation figure - supports the "
               "architecture choice." if all_below else
               "NOT below the mask-head quantisation figure for at "
               "least one class - the architecture choice in spec 3.1 "
               "is NOT supported by this measurement and should be "
               "revisited, not defended.")
    lines.append(f"  verdict: {verdict}")
    lines.append("")

    # ---- signed area error ---------------------------------------------
    lines.append("Signed placeable-area error, mm^2 (optimistic = "
                 "predicted placeable where truth")
    lines.append("is not, the error that can put a cell on a PCB; "
                 "conservative = the opposite,")
    lines.append("a lost cell, not a safety issue). Mean per crop and "
                 "total over the split:")
    head = (f"  {'class':<12}{'opt mean':>11}{'opt total':>12}"
           f"{'cons mean':>12}{'cons total':>13}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for c in SELECT_ON:
        lines.append(
            f"  {c:<12}"
            f"{results['area_opt_mm2_mean'][c]:>11.1f}"
            f"{results['area_opt_mm2_total'][c]:>12.1f}"
            f"{results['area_cons_mm2_mean'][c]:>12.1f}"
            f"{results['area_cons_mm2_total'][c]:>13.1f}")
    lines.append("  " + "-" * (len(head) - 2))
    lines.append("")

    # ---- latency ----------------------------------------------------
    lines.append("Latency vs the FDR 10.4 50 ms end-to-end budget "
                 "(fp16 batched inference):")
    lines.append("looped_ms is the SAME crops through segment_batch() one "
                 "at a time (batch size 1,")
    lines.append("n calls) - the path batching replaced. This is the "
                 "single measured source for")
    lines.append("the batching-latency figure; quote THIS receipt rather "
                 "than a hand-measured one.")
    head = (f"  {'cartridges':>11}{'total_ms':>11}{'per_cart_ms':>13}"
           f"{'within_50ms':>13}{'looped_ms':>12}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for row in latency:
        lines.append(f"  {row['cartridges']:>11}{row['total_ms']:>11.1f}"
                     f"{row['per_cartridge_ms']:>13.2f}"
                     f"{str(row['within_50ms_budget']):>13}"
                     f"{row['looped_ms']:>12.1f}")
    lines.append("  " + "-" * (len(head) - 2))
    batch8 = next((r for r in latency if r["cartridges"] == 8), None)
    if batch8 is not None and batch8["total_ms"] > 0:
        factor = batch8["looped_ms"] / batch8["total_ms"]
        lines.append(
            f"  at 8 cartridges: {batch8['total_ms']:.1f} ms batched vs "
            f"{batch8['looped_ms']:.1f} ms looped ({factor:.1f}x) - "
            "batching is load-bearing, not an optimisation: the looped "
            "figure breaches the 50 ms end-to-end budget on its own.")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- CLI --

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.seg_evaluate",
        description="Evaluate the bay segmenter: per-class IoU, boundary "
                    "displacement, signed area error, latency.")
    ap.add_argument("--checkpoint", default="recog/checkpoints/seg/best.pt")
    ap.add_argument("--config", default="configs/segmentation.yaml")
    ap.add_argument("--synth-config", default=None,
                    help="render/layout config to compute mm_per_px from. "
                         "Defaults to the dataset's own manifest.json, "
                         "falling back to configs/synth3d.yaml.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="docs/receipts/seg_eval.txt")
    ap.add_argument("--per-sku", action="store_true",
                    help="also report per-catalog-asset (SKU) IoU - "
                         "design spec Sec12. Only meaningful against a "
                         "dataset whose sidecar carries 'asset' per "
                         "annotation (Task 12).")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.checkpoint and not Path(args.checkpoint).exists():
        raise SystemExit(f"error: checkpoint not found: {args.checkpoint}")

    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model") or {}
    ds_cfg = cfg.get("dataset") or {}

    num_classes = int(model_cfg.get("num_classes", 6))
    crop_size = int(model_cfg.get("crop_size", 256))
    half = bool(model_cfg.get("half", True))

    synth_cfg, synth_source = load_synth_config(
        ds_cfg["coco_path"], args.synth_config)
    synth_source = Path(synth_source).as_posix()
    mm_per_px = resolve_mm_per_px(synth_cfg)

    from recog.bay_segmenter import BaySegmenter
    from recog.seg_dataset import BaySegDataset
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

    # Fail BEFORE spending inference time if the split this run recomputes
    # does not match what the checkpoint was actually selected against -
    # see check_split_matches_checkpoint's docstring.
    if args.checkpoint:
        val_counts_now = compute_val_instance_counts(
            full_dataset, val_indices, num_classes=num_classes)
        check_split_matches_checkpoint(args.checkpoint, val_counts_now)

    segmenter = BaySegmenter(checkpoint=args.checkpoint, device=args.device,
                             crop_size=crop_size, half=half,
                             num_classes=num_classes)

    results = evaluate(segmenter, full_dataset, val_indices, mm_per_px,
                       num_classes=num_classes)
    latency = latency_table(segmenter)

    checkpoint_note = (_sibling_checkpoint_note(args.checkpoint)
                       if args.checkpoint else None)
    report = format_report(
        results, latency,
        checkpoint=args.checkpoint, config_path=args.config,
        synth_config_source=synth_source, mm_per_px=mm_per_px,
        device=str(segmenter.device), checkpoint_note=checkpoint_note)

    if args.per_sku:
        by_asset = group_indices_by_asset(full_dataset, val_indices)
        per_sku_results = {
            asset: evaluate(segmenter, full_dataset, idxs, mm_per_px,
                            num_classes=num_classes)
            for asset, idxs in by_asset.items() if idxs
        }
        report += "\n" + format_per_sku_table(per_sku_results)

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")

    if not latency_within_budget(latency):
        # A non-zero exit, not just a log line: this is the plan's
        # acceptance criterion, and only a non-zero exit can gate CI.
        log.error("8-cartridge latency is OVER the 50 ms budget")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHANNEL_NAMES",
    "SELECT_ON",
    "MASK_HEAD_QUANTISATION_MM",
    "per_class_iou",
    "boundary_displacement_mm",
    "signed_area_error_mm2",
    "latency_table",
    "latency_within_budget",
    "resolve_mm_per_px",
    "load_synth_config",
    "evaluate",
    "compute_val_instance_counts",
    "check_split_matches_checkpoint",
    "format_report",
    "main",
]
