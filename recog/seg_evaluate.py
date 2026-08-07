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
from recog.seg_dataset import SEG_CHANNELS

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

def per_class_iou(pred: np.ndarray, target: np.ndarray,
                  num_classes: int = 6) -> Dict[str, float]:
    """IoU per class. NaN where a class appears in neither array.

    NaN rather than 0.0: a class that was never present was never
    tested, and scoring it zero drags the mean down for a question that
    was not asked.
    """
    out: Dict[str, float] = {}
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = int((p | t).sum())
        out[CHANNEL_NAMES[c]] = (float(np.nan) if union == 0
                                 else float((p & t).sum()) / union)
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
    """Wall-clock per batch size, against the 50 ms PPR budget."""
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
        rows.append({"cartridges": n,
                     "total_ms": round(total_ms, 1),
                     "per_cartridge_ms": round(total_ms / n, 2),
                     "within_50ms_budget": total_ms <= 50.0})
    return rows


# ------------------------------------------------------- native ground truth --
#
# recog.seg_dataset.rasterise_crop force-resizes its output to a SQUARE
# out_size, because that is what the model's fixed-size input needs.
# Boundary displacement and area error need the opposite: native,
# un-resized pixels. The render's mm_per_px (layout.area / render.res)
# is isotropic only in that native frame - a jittered union box is not
# square, so resizing it to out_size x out_size rescales x and y by
# different factors, and a single scalar mm_per_px could not describe
# the result correctly. This mirrors rasterise_crop's painting loop
# exactly, minus the final resize.

def _rasterise_native(anns: Sequence[dict], box: Tuple[int, int, int, int]
                      ) -> np.ndarray:
    """Ground-truth label map at the crop's own native resolution."""
    from recog.seg_dataset import _PAINT_ORDER
    from recog.synth3d.annotate import rle_decode

    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    label = np.zeros((h, w), dtype=np.int64)

    by_class: Dict[str, List[dict]] = {}
    for a in anns:
        by_class.setdefault(a["class"], []).append(a)

    for cls, channel in _PAINT_ORDER:
        for a in by_class.get(cls, ()):
            full = rle_decode(a["segmentation"])
            fh, fw = full.shape
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(fw, x1), min(fh, y1)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            sub = full[sy0:sy1, sx0:sx1]
            label[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][sub > 0] = \
                SEG_CHANNELS[channel]
    return label


def _extract_native_crop(image: np.ndarray,
                         box: Tuple[int, int, int, int]) -> np.ndarray:
    """The crop's own pixels, zero-padded where the box runs off the
    image. Mirrors BaySegDataset.__getitem__'s crop extraction, minus
    the final resize to out_size."""
    x0, y0, x1, y1 = box
    pad_t, pad_l = max(0, -y0), max(0, -x0)
    crop = image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if pad_t or pad_l or crop.shape[0] != y1 - y0 or crop.shape[1] != x1 - x0:
        padded = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        padded[pad_t:pad_t + crop.shape[0],
               pad_l:pad_l + crop.shape[1]] = crop
        crop = padded
    return crop


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

    IoU is pooled pixel-wise over the whole split (sum of intersections
    / sum of unions across crops), matching recog.seg_training's
    evaluate_model so the two are directly comparable. Boundary
    displacement and area error are per-crop: a per-crop mean is the
    "typical cartridge" figure, and the per-crop values are also summed
    for a "total exposure over the split" figure.
    """
    from PIL import Image

    inter = [0] * num_classes
    union = [0] * num_classes
    counts = [0] * num_classes            # crops containing class c

    boundary_mm: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    boundary_defined: Dict[str, int] = {c: 0 for c in SELECT_ON}
    area_opt_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    area_cons_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}

    for idx in val_indices:
        img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)     # no jitter at eval

        path = Path(full_dataset.img_dir) / img_meta["file_name"]
        image = np.asarray(Image.open(path).convert("RGB"))

        crop = _extract_native_crop(image, box)
        target = _rasterise_native(anns, box)
        pred = segmenter.segment_batch([crop])[0]

        for c in range(num_classes):
            p, t = pred == c, target == c
            inter[c] += int((p & t).sum())
            union[c] += int((p | t).sum())
            if t.any():
                counts[c] += 1

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
    }


def format_report(results: Dict[str, Any], latency: List[dict], *,
                  checkpoint: Optional[str], config_path: str,
                  synth_config_source: str, mm_per_px: float,
                  device: str) -> str:
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
    head = f"  {'cartridges':>11}{'total_ms':>11}{'per_cart_ms':>13}{'within_50ms':>13}"
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for row in latency:
        lines.append(f"  {row['cartridges']:>11}{row['total_ms']:>11.1f}"
                     f"{row['per_cartridge_ms']:>13.2f}"
                     f"{str(row['within_50ms_budget']):>13}")
    lines.append("  " + "-" * (len(head) - 2))
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

    segmenter = BaySegmenter(checkpoint=args.checkpoint, device=args.device,
                             crop_size=crop_size, half=half,
                             num_classes=num_classes)

    results = evaluate(segmenter, full_dataset, val_indices, mm_per_px,
                       num_classes=num_classes)
    latency = latency_table(segmenter)

    report = format_report(
        results, latency,
        checkpoint=args.checkpoint, config_path=args.config,
        synth_config_source=synth_source, mm_per_px=mm_per_px,
        device=str(segmenter.device))

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path}")

    if not all(r["within_50ms_budget"] for r in latency if r["cartridges"] == 8):
        log.warning("8-cartridge latency is OVER the 50 ms budget")

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
    "resolve_mm_per_px",
    "load_synth_config",
    "evaluate",
    "format_report",
    "main",
]
