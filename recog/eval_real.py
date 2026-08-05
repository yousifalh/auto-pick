"""Evaluate a detector on the held-out set of *real* photographs.

    python -m recog.eval_real --checkpoint recog/checkpoints/best.pt
    python -m recog.eval_real --limit 2 --save-overlays /tmp/overlays

This is the honest test of the synthetic-data pipeline: the detector
is trained only on Blender renders, and ``recog/realtest`` is seven
phone photographs of the real thing, annotated in CVAT and exported
as COCO. Nothing in this file is used during training.

Metrics come from :mod:`recog.evaluate` — the same pure-numpy VOC
11-point implementation the rest of the project reports, so the
numbers here are directly comparable with the synthetic validation
figures.

``--save-overlays`` is not decoration. A single mAP number cannot
tell you whether a bad score means "found nothing", "found everything
at the wrong scale", or "confused the two classes"; the overlays can,
at a glance.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import load_yaml
from common.types import ClassLabel
from recog.dataset import CLASS_MAP, parse_coco_json
from recog.evaluate import mean_ap

# Defaults point at the in-repo held-out set so the command is a
# one-liner; both are overridable for a re-export elsewhere.
DEFAULT_IMG_DIR = "recog/realtest/images"
DEFAULT_ANN_PATH = "recog/realtest/annotations/instances_default.json"
DEFAULT_CONFIG = "configs/recognition.yaml"

# Evaluated classes, in report order. Background is not a detection class.
EVAL_CLASSES: Tuple[Tuple[str, int], ...] = (
    ("battery", CLASS_MAP["battery"]),
    ("cartridge", CLASS_MAP["cartridge"]),
)

# RGB. Predictions are warm, ground truth cool — legible on both the
# metallic cells and the green cartridges.
PRED_COLOUR = (255, 92, 40)
GT_COLOUR = (40, 190, 255)

Box = Tuple[float, float, float, float]


# ------------------------------------------------------------ loading ----

def load_config(path: Optional[str]) -> dict:
    """Load the recognition YAML, tolerating a missing default file."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        if path == DEFAULT_CONFIG:
            return {}
        raise SystemExit(f"error: config not found: {p}")
    return load_yaml(p)


def build_detector(checkpoint: Optional[str], cfg: dict):
    """Load the detector, turning a checkpoint/config mismatch into prose.

    ``recog/checkpoints/*.pt`` are tied to the anchor geometry they
    were trained with. Loading one against a config whose
    ``anchor_scales`` have since changed fails inside
    ``load_state_dict`` with a wall of tensor shapes; that is a
    configuration problem, not a crash, so it is reported as one.
    """
    from recog.inference import load_detector

    # load_detector silently falls back to the heuristic when the path
    # is missing. That is right for the main pipeline and wrong here:
    # reporting heuristic numbers under a checkpoint's name would be a
    # lie, so a named-but-absent checkpoint is an error.
    if checkpoint and not Path(checkpoint).exists():
        raise SystemExit(f"error: checkpoint not found: {checkpoint}")

    try:
        return load_detector(checkpoint, cfg)
    except (RuntimeError, KeyError, ValueError, TypeError) as exc:
        scales = (cfg.get("model") or {}).get("anchor_scales")
        ratios = (cfg.get("model") or {}).get("anchor_ratios")
        raise SystemExit(
            "error: could not load the checkpoint into the model described "
            "by the config.\n"
            f"  checkpoint   : {checkpoint}\n"
            f"  anchor_scales: {scales}\n"
            f"  anchor_ratios: {ratios}\n"
            "  This normally means the checkpoint was trained with "
            "different anchor geometry or a different number of classes. "
            "Retrain against the current config, or evaluate with the "
            "config the checkpoint was trained under.\n"
            f"  underlying error: {_brief(exc)}"
        ) from exc


def _brief(exc: BaseException, limit: int = 400) -> str:
    """One short line for an exception.

    ``load_state_dict`` failures list every mismatched tensor, which
    is thousands of characters of noise around a one-line diagnosis.
    """
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[:limit].rstrip() + " ... (truncated)"
    return f"{type(exc).__name__}: {text}"


def load_image_rgb(path: Path) -> np.ndarray:
    """Read an image as an HWC uint8 RGB array (what ``Detector`` wants)."""
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover - PIL is a project dependency
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read {path}")
        return bgr[:, :, ::-1].copy()

    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


# --------------------------------------------------------- evaluation ----

def collect_predictions(
    detector,
    img_dir: Path,
    records: Sequence,
    confidence: float,
    log=None,
) -> Tuple[Dict[int, List[Tuple[Box, int, float]]], float]:
    """Run the detector over every record, returning preds + elapsed secs."""
    preds_by_image: Dict[int, List[Tuple[Box, int, float]]] = {}
    t0 = time.perf_counter()
    for rec in records:
        image = load_image_rgb(img_dir / rec.file_name)
        snap = detector(image)

        preds: List[Tuple[Box, int, float]] = []
        for det in snap.detections:
            if det.label is ClassLabel.BACKGROUND:
                continue
            score = float(det.confidence)
            if score < confidence:
                continue
            cls = CLASS_MAP.get(det.label.value)
            if cls is None:
                continue
            b = det.bbox
            preds.append(((b.xmin, b.ymin, b.xmax, b.ymax), cls, score))

        preds_by_image[rec.image_id] = preds
        if log is not None:
            log(f"  {rec.file_name}: {len(preds)} pred / "
                f"{len(rec.boxes)} gt")
    return preds_by_image, time.perf_counter() - t0


def summarise(
    gts_by_image: Dict[int, List[Tuple[Box, int]]],
    preds_by_image: Dict[int, List[Tuple[Box, int, float]]],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
) -> Dict[float, Dict[str, float]]:
    """``mean_ap`` at each requested IoU threshold, keyed by threshold."""
    class_ids = [cid for _name, cid in EVAL_CLASSES]
    return {
        float(t): mean_ap(gts_by_image, preds_by_image, class_ids, float(t))
        for t in iou_thresholds
    }


def _counts(items, index: int) -> Dict[int, int]:
    """Count entries per class id, where the class id sits at ``index``."""
    out = {cid: 0 for _n, cid in EVAL_CLASSES}
    for entries in items.values():
        for entry in entries:
            cid = entry[index]
            if cid in out:
                out[cid] += 1
    return out


def format_report(
    results: Dict[float, Dict[str, float]],
    gt_counts: Dict[int, int],
    pred_counts: Dict[int, int],
    *,
    n_images: int,
    confidence: float,
    detector_name: str,
    checkpoint: Optional[str],
    config_path: Optional[str],
    elapsed_s: float,
) -> str:
    """Render the whole run as a plain-text block."""
    thresholds = sorted(results)
    lines: List[str] = []
    lines.append("")
    lines.append("Real-photo held-out evaluation")
    lines.append(f"  detector    : {detector_name}")
    lines.append(f"  checkpoint  : {checkpoint or '(none)'}")
    lines.append(f"  config      : {config_path or '(none)'}")
    lines.append(f"  images      : {n_images}")
    lines.append(f"  confidence  : {confidence:.2f}")
    lines.append(f"  inference   : {elapsed_s:.1f} s total"
                 f" ({elapsed_s / max(1, n_images):.2f} s/image)")
    lines.append("")

    head = f"{'class':<12}{'GT':>6}{'pred':>7}"
    for t in thresholds:
        head += f"{'AP@' + format(t, '.2f'):>11}"
    lines.append(head)
    lines.append("-" * len(head))

    for name, cid in EVAL_CLASSES:
        row = f"{name:<12}{gt_counts.get(cid, 0):>6}{pred_counts.get(cid, 0):>7}"
        for t in thresholds:
            row += f"{results[t][f'AP_{cid}']:>11.4f}"
        lines.append(row)

    lines.append("-" * len(head))
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    row = f"{'mAP':<12}{total_gt:>6}{total_pred:>7}"
    for t in thresholds:
        row += f"{results[t][f'mAP@{t:.2f}']:>11.4f}"
    lines.append(row)
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------- overlays ----

def _scaled_font(size: int):
    """A bitmap font at ``size`` px, or ``None`` on older Pillow.

    ``ImageFont.load_default(size=...)`` needs Pillow >= 10.1; without
    it the labels are drawn at the default 11 px, which is unreadable
    on a 4000-px photo but is still better than failing the run.
    """
    try:
        from PIL import ImageFont  # type: ignore

        return ImageFont.load_default(size=size)
    except Exception:  # pragma: no cover - very old Pillow
        return None


def save_overlays(
    img_dir: Path,
    records: Sequence,
    preds_by_image: Dict[int, List[Tuple[Box, int, float]]],
    out_dir: Path,
    max_side: int = 1600,
) -> int:
    """Draw GT and predicted boxes onto each photo; return files written.

    Boxes are drawn at full resolution and the result downscaled, so
    line weight stays proportional and a 3024x4032 phone photo becomes
    something you can actually open.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover - PIL is a dependency
        raise SystemExit(f"error: --save-overlays needs Pillow ({exc})")

    inv = {cid: name for name, cid in EVAL_CLASSES}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for rec in records:
        path = img_dir / rec.file_name
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        long_side = max(im.width, im.height)
        thick = max(3, round(long_side / 400))
        # The image is downscaled on save, so the label has to be drawn
        # oversized to survive it.
        font = _scaled_font(max(12, round(long_side / 45)))
        # Anchors are unsupported for the default bitmap font, so they
        # are only used when a sized font was actually obtained.
        gt_kw = {"font": font, "anchor": "lb"} if font else {}
        pr_kw = {"font": font, "anchor": "lt"} if font else {}

        for box, cid in zip(rec.boxes, rec.labels):
            draw.rectangle(list(box), outline=GT_COLOUR, width=thick)
            draw.text((box[0] + thick, max(0.0, box[1] - 3.2 * thick)),
                      f"GT {inv.get(cid, cid)}", fill=GT_COLOUR, **gt_kw)

        for box, cid, score in preds_by_image.get(rec.image_id, []):
            draw.rectangle(list(box), outline=PRED_COLOUR, width=thick)
            draw.text((box[0] + thick, box[3] + thick),
                      f"{inv.get(cid, cid)} {score:.2f}",
                      fill=PRED_COLOUR, **pr_kw)

        if max(im.width, im.height) > max_side:
            im.thumbnail((max_side, max_side))
        im.save(out_dir / (Path(rec.file_name).stem + "_overlay.jpg"),
                quality=88)
        written += 1

    return written


# --------------------------------------------------------------- main ----

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.eval_real",
        description=(
            "Evaluate a detector on the held-out real photographs in "
            "recog/realtest (VOC 11-point AP, IoU 0.50 and 0.75)."
        ),
    )
    ap.add_argument("--checkpoint", default=None,
                    help="detector checkpoint (.pt). Omit to evaluate the "
                         "HeuristicDetector fallback.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"recognition YAML (default: {DEFAULT_CONFIG})")
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--annotations", default=DEFAULT_ANN_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images")
    ap.add_argument("--confidence", type=float, default=None,
                    help="score threshold (default: the config's "
                         "model.confidence_threshold, else 0.70)")
    ap.add_argument("--save-overlays", default=None, metavar="DIR",
                    help="write GT-vs-prediction overlay JPEGs to DIR")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-image progress lines")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    img_dir = Path(args.img_dir)
    ann_path = Path(args.annotations)
    if not ann_path.is_file():
        raise SystemExit(f"error: annotations not found: {ann_path}")
    if not img_dir.is_dir():
        raise SystemExit(f"error: image directory not found: {img_dir}")

    cfg = load_config(args.config)
    model_cfg = dict(cfg.get("model") or {})

    confidence = (
        float(args.confidence) if args.confidence is not None
        else float(model_cfg.get("confidence_threshold", 0.70))
    )
    # Keep the model's own score threshold in step with ours, otherwise a
    # low --confidence silently has no effect: the detector would have
    # dropped those boxes before we ever saw them.
    model_cfg["confidence_threshold"] = confidence
    cfg = dict(cfg)
    cfg["model"] = model_cfg

    records = parse_coco_json(ann_path)
    records = [r for r in records if (img_dir / r.file_name).is_file()]
    if args.limit is not None:
        records = records[:max(0, args.limit)]
    if not records:
        raise SystemExit(
            f"error: no annotated images found under {img_dir}"
        )

    detector = build_detector(args.checkpoint, cfg)

    log = None if args.quiet else (lambda msg: print(msg, flush=True))
    if log is not None:
        print(f"Scoring {len(records)} real photo(s) from {img_dir} ...",
              flush=True)

    preds_by_image, elapsed = collect_predictions(
        detector, img_dir, records, confidence, log,
    )
    gts_by_image: Dict[int, List[Tuple[Box, int]]] = {
        r.image_id: list(zip(r.boxes, r.labels)) for r in records
    }

    results = summarise(gts_by_image, preds_by_image)
    print(format_report(
        results,
        _counts(gts_by_image, 1),
        _counts(preds_by_image, 1),
        n_images=len(records),
        confidence=confidence,
        detector_name=type(detector).__name__,
        checkpoint=args.checkpoint,
        config_path=args.config,
        elapsed_s=elapsed,
    ))

    if args.save_overlays:
        out_dir = Path(args.save_overlays)
        n = save_overlays(img_dir, records, preds_by_image, out_dir)
        print(f"wrote {n} overlay(s) to {out_dir}")

    return 0


__all__ = [
    "build_arg_parser",
    "build_detector",
    "collect_predictions",
    "format_report",
    "main",
    "save_overlays",
    "summarise",
]


if __name__ == "__main__":
    sys.exit(main())
