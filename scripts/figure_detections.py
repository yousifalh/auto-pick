#!/usr/bin/env python3
"""Draw the two recognition stages onto held-out synthetic frames.

Writes ``docs/figures/fig12_detections.png``: a 2 x N grid where the top
row carries the Faster R-CNN's boxes and the bottom row carries the
per-ROI bay segmenter's label map composited back into each detected
cartridge.

Why this figure exists. ``fig10_segmenter.png`` shows the segmenter on
256 px crops and ``fig11_architecture.png`` shows the block diagram;
neither shows the detector at all, and nothing in the repository showed
the two stages on the same frame. A reader who wants to know what the
system actually sees had to run it.

Why THESE frames. The detector checkpoint trained on
``recog/dataset3d`` (configs/recognition.yaml) and the shipping
segmenter trained on ``recog/dataset3d_seg`` (configs/segmentation.yaml).
The default ``--images`` here is ``recog/dataset3d_seg_cad_test``, which
is neither, so both stages are drawn on frames neither model was fitted
to. That is a weaker claim than a metric and a stronger one than a
figure drawn on training data; the per-class numbers live in
``docs/receipts/seg_eval*.txt`` and this is an illustration.

Determinism: the frame list is sorted and sampled from a fixed seed, so
the same command draws the same frames. The chosen filenames are printed
so the figure can be tied to its inputs.

    python scripts/figure_detections.py

Requires torch, a detector checkpoint and a segmenter checkpoint. None
of the three is in this repository - see the README's note on
``recog/checkpoints/`` being gitignored.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_yaml  # noqa: E402
from common.types import ClassLabel  # noqa: E402

# Six classes, in the index order recog/seg_dataset.py paints them:
# background 0, cartridge 1, bay 2, electronics 3, obstruction 4,
# battery 5. The colours match fig10_segmenter.png so the two figures
# can be read side by side - green bay, orange electronics, blue
# cartridge wall.
SEG_COLOURS = {
    1: (0.20, 0.45, 0.95),   # cartridge  - blue
    2: (0.15, 0.80, 0.35),   # bay        - green
    3: (1.00, 0.60, 0.10),   # electronics- orange
    4: (0.90, 0.20, 0.20),   # obstruction- red
    5: (0.98, 0.90, 0.20),   # battery    - yellow
}
SEG_NAMES = {1: "cartridge", 2: "bay", 3: "electronics",
             4: "obstruction", 5: "battery"}

# Detector box colours. Deliberately NOT the segmenter palette: these
# are different predictions from a different model and a reader should
# not have to wonder whether a green box means the same as green fill.
BOX_COLOURS = {ClassLabel.CARTRIDGE: "#00e5ff", ClassLabel.BATTERY: "#ff2fd0"}


def _crop_window(snapshot, shape, pad: float):
    """The union of every detection box, padded, clamped to the frame.

    A 1280x720 render of a 700 mm table puts an 18 mm cell at about 25 px
    across. Drawn full-frame at README width that is three or four
    pixels, so the figure would be honest and unreadable. Cropping to
    where the detections actually are is a display choice and changes no
    prediction - both rows use the SAME window so they stay comparable.
    """
    h, w = shape[:2]
    if not snapshot.detections:
        return 0, 0, w, h
    x0 = min(d.bbox.xmin for d in snapshot.detections)
    y0 = min(d.bbox.ymin for d in snapshot.detections)
    x1 = max(d.bbox.xmax for d in snapshot.detections)
    y1 = max(d.bbox.ymax for d in snapshot.detections)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(w, int(x1 + mx)), min(h, int(y1 + my)))


def _select_frames(img_dir: Path, n: int, detector, segmenter,
                   min_bay_px: int, image_loader):
    """First ``n`` frames in sorted order that show an OPEN cartridge.

    The rule, stated because it is a selection: scan the directory in
    sorted filename order and keep a frame when the detector finds at
    least one cartridge AND the segmenter predicts at least
    ``min_bay_px`` pixels of ``bay`` inside it. Stop at ``n``.

    Random frames were tried first and made a poor figure for a reason
    worth recording: most cartridges in this corpus are sealed, so a
    random draw shows the segmenter painting one flat `cartridge` region
    and none of the classes the placement decision actually uses. This
    rule is mechanical and reported with the number of frames it had to
    scan, which is the honest version of "pick a good one" - the same
    convention fig10_segmenter.png uses when it says its five crops are
    the first five bay-carrying ones in split order, not hand-picked.
    """
    from recog.inference import attach_cartridge_masks

    frames = sorted(p for p in img_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"no .png under {img_dir}")

    chosen, scanned = [], 0
    for path in frames:
        scanned += 1
        rgb = image_loader(path)
        snapshot = detector.detect(rgb)
        if not any(d.label is ClassLabel.CARTRIDGE for d in snapshot.detections):
            continue
        attach_cartridge_masks(snapshot, rgb, segmenter)
        bay_px = sum(int((np.asarray(m) == 2).sum())
                     for m in snapshot.cartridge_masks.values())
        if bay_px >= min_bay_px:
            chosen.append((path, rgb, snapshot, bay_px))
        if len(chosen) == n:
            break

    if not chosen:
        raise SystemExit(
            f"scanned {scanned} frames, none had a cartridge with "
            f">= {min_bay_px} bay pixels")
    return chosen, scanned


def _composite(rgb: np.ndarray, snapshot, alpha: float) -> np.ndarray:
    """Paint each cartridge's label map back onto a copy of the frame."""
    out = rgb.astype(np.float32) / 255.0
    for det_index, mask in snapshot.cartridge_masks.items():
        box = snapshot.detections[det_index].bbox
        x0, y0 = max(0, int(box.xmin)), max(0, int(box.ymin))
        x1 = min(out.shape[1], int(box.xmax))
        y1 = min(out.shape[0], int(box.ymax))
        if x1 <= x0 or y1 <= y0:
            continue

        region = np.asarray(mask)
        if region.shape[:2] != (y1 - y0, x1 - x0):
            # segment_batch resizes to the crop it was given; if the
            # caller cropped differently, nearest-neighbour it back
            # rather than silently mis-registering the overlay.
            ys = (np.linspace(0, region.shape[0] - 1, y1 - y0)).astype(int)
            xs = (np.linspace(0, region.shape[1] - 1, x1 - x0)).astype(int)
            region = region[ys][:, xs]

        tile = out[y0:y1, x0:x1]
        for label, colour in SEG_COLOURS.items():
            sel = region == label
            if sel.any():
                tile[sel] = (1.0 - alpha) * tile[sel] + alpha * np.array(colour)
    return np.clip(out, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="recog/dataset3d_seg_cad_test/images")
    ap.add_argument("--recognition", default="configs/recognition.yaml")
    ap.add_argument("--segmentation", default="configs/segmentation.yaml")
    ap.add_argument("--detector", default="recog/checkpoints/best.pt")
    ap.add_argument("--segmenter", default="recog/checkpoints/seg/best.pt")
    ap.add_argument("--out", default="docs/figures/fig12_detections.png")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pad", type=float, default=0.10,
                    help="crop margin around the detections, as a fraction")
    ap.add_argument("--min-bay-px", type=int, default=2000,
                    help="a frame qualifies at this many predicted bay pixels")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    from recog.bay_segmenter import BaySegmenter
    from recog.inference import attach_cartridge_masks, load_detector

    for path in (args.detector, args.segmenter):
        if not Path(path).exists():
            raise SystemExit(
                f"missing checkpoint: {path}\n"
                "No .pt is committed to this repository; train one with "
                "`python -m recog.training` / `python -m recog.seg_training`."
            )

    recog_cfg = load_yaml(args.recognition)
    seg_model = (load_yaml(args.segmentation) or {}).get("model", {}) or {}

    segmenter = BaySegmenter(
        checkpoint=args.segmenter,
        crop_size=int(seg_model.get("crop_size", 256)),
        half=bool(seg_model.get("half", True)),
        num_classes=int(seg_model.get("num_classes", 6)),
    )
    detector = load_detector(checkpoint=args.detector, cfg=recog_cfg,
                            segmenter=None)

    def _load(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"))

    chosen, scanned = _select_frames(
        Path(args.images), args.n, detector, segmenter,
        args.min_bay_px, _load)

    cols = len(chosen)
    fig, axes = plt.subplots(2, cols, figsize=(5.2 * cols, 6.0),
                             squeeze=False)

    totals = {"cartridge": 0, "battery": 0, "segmented": 0}
    for col, (frame_path, rgb, snapshot, bay_px) in enumerate(chosen):
        x0, y0, x1, y1 = _crop_window(snapshot, rgb.shape, args.pad)
        view = rgb[y0:y1, x0:x1]
        painted = _composite(rgb, snapshot, args.alpha)[y0:y1, x0:x1]

        top = axes[0][col]
        top.imshow(view)
        for det in snapshot.detections:
            b = det.bbox
            colour = BOX_COLOURS.get(det.label, "#ffffff")
            top.add_patch(Rectangle(
                (b.xmin - x0, b.ymin - y0), b.xmax - b.xmin, b.ymax - b.ymin,
                fill=False, edgecolor=colour, linewidth=2.0))
            if det.label is ClassLabel.CARTRIDGE:
                # Only the cartridges get a label. Scoring every one of a
                # dozen loose cells buries the frame in overlapping text
                # and the per-class numbers are in the receipts anyway.
                top.text(b.xmin - x0, b.ymin - y0 - 5,
                         f"{det.label.value} {det.confidence:.2f}",
                         color=colour, fontsize=9, weight="bold")
            totals[det.label.value] = totals.get(det.label.value, 0) + 1

        axes[1][col].imshow(painted)
        totals["segmented"] += len(snapshot.cartridge_masks)

        for ax in (top, axes[1][col]):
            ax.set_xticks([])
            ax.set_yticks([])
        top.set_title(f"{frame_path.name}   ·   {bay_px:,} bay px",
                      fontsize=9, color="#444444")

    axes[0][0].set_ylabel("Faster R-CNN\nboxes", fontsize=10, weight="bold")
    axes[1][0].set_ylabel("per-ROI segmenter\nlabel map", fontsize=10,
                          weight="bold")

    handles = [Patch(facecolor=c, label=SEG_NAMES[k])
               for k, c in SEG_COLOURS.items()]
    handles += [Patch(facecolor=BOX_COLOURS[ClassLabel.CARTRIDGE],
                      label="box: cartridge"),
                Patch(facecolor=BOX_COLOURS[ClassLabel.BATTERY],
                      label="box: battery")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")

    print(f"wrote {args.out}")
    print(f"images   : {args.images}")
    print(f"detector : {args.detector}")
    print(f"segmenter: {args.segmenter}")
    print(f"selection: first {args.n} of {scanned} frames scanned in sorted "
          f"order having a detected cartridge with >= {args.min_bay_px} "
          f"predicted bay pixels")
    for f, _rgb, _snap, bay_px in chosen:
        print(f"  frame  : {f.name}  ({bay_px} bay px)")
    print(f"detections: {totals['cartridge']} cartridge, "
          f"{totals['battery']} battery; "
          f"{totals['segmented']} cartridge(s) segmented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
