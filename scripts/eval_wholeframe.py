#!/usr/bin/env python3
"""Score the ROI and whole-frame segmenters on the SAME validation crops.

The architecture comparison pre-registered in
`docs/superpowers/specs/2026-08-16-wholeframe-preregistration.md`. Read
that first: the prediction, the resolvable effect and the falsifier were
written down before either arm was trained, and this script does not
decide any of them - it only measures and prints whether the threshold
was crossed.

Both arms go through `recog.seg_evaluate.evaluate`, on the same crops,
the same ground truth and the same per-frame scales. They differ in
exactly one argument: `predict`. Everything that turns pixels into
millimetres is shared code, which is the only reason the two numbers can
be put in one table.

    python scripts/eval_wholeframe.py \\
        --roi-checkpoint recog/checkpoints/seg/best.pt \\
        --wholeframe-checkpoint recog/checkpoints/seg_wholeframe/best.pt \\
        --out docs/receipts/wholeframe_comparison.txt

Exit status is 0 whichever way the result lands. A null is a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_yaml  # noqa: E402

# Pre-registered in the spec above. Copied here as constants so the
# script cannot quietly be judged against a threshold invented after the
# fact - if these disagree with the spec, the spec wins and this is a bug.
ROI_BASELINE_MM = 1.226
PREDICTED_EXCESS_MM = 0.50      # "will be worse by at least this"
FALSIFIER_WITHIN_MM = 0.25      # "refuted if it comes this close"
FALSIFIER_MS = 40.0             # "...and runs within this per frame"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roi-config", default="configs/segmentation.yaml")
    ap.add_argument("--wholeframe-config",
                    default="configs/wholeframe_segmentation.yaml")
    ap.add_argument("--roi-checkpoint", default="recog/checkpoints/seg/best.pt")
    ap.add_argument("--wholeframe-checkpoint",
                    default="recog/checkpoints/seg_wholeframe/best.pt")
    ap.add_argument("--out", default="docs/receipts/wholeframe_comparison.txt")
    args = ap.parse_args()

    import numpy as np

    from recog.bay_segmenter import BaySegmenter
    from recog.seg_dataset import BaySegDataset
    from recog.seg_evaluate import evaluate, resolve_frame_scales
    from recog.seg_training import _split_dataset
    from recog.seg_wholeframe import (WholeFrameSegDataset,
                                      frame_split_from_crop_split,
                                      predict_val_crops_from_frames)

    for path in (args.roi_checkpoint, args.wholeframe_checkpoint):
        if not Path(path).exists():
            raise SystemExit(
                f"missing checkpoint: {path}\n"
                "Train the whole-frame arm first:\n"
                "  python -m recog.seg_training "
                "--config configs/wholeframe_segmentation.yaml")

    roi_cfg = load_yaml(args.roi_config)
    wf_cfg = load_yaml(args.wholeframe_config)
    ds_cfg = roi_cfg["dataset"]

    # ONE crop dataset, built from the ROI config, scored by both arms.
    # Building a second one from the whole-frame config would risk two
    # different splits and there would be no way to notice from the
    # output.
    crops = BaySegDataset(
        coco_path=ds_cfg["coco_path"], img_dir=ds_cfg["img_dir"],
        out_size=int(roi_cfg["model"].get("crop_size", 256)),
        jitter_frac=float(ds_cfg.get("jitter_frac", 0.06)),
        train=True, transform=None, seed=int(roi_cfg["training"]["seed"]))

    train_subset, val_subset = _split_dataset(
        crops, float(ds_cfg["train_val_split"]),
        seed=int(ds_cfg.get("split_seed", 0)))
    val_indices = sorted(int(i) for i in val_subset.indices)
    train_indices = sorted(int(i) for i in train_subset.indices)

    scales, provenance = resolve_frame_scales(crops, val_indices)

    # ---- the leak check, before any number is produced -----------------
    #
    # Section 3 of the pre-registration: if a whole-frame TRAINING frame
    # contributed any validation crop, the whole-frame model was trained
    # on the pixels it is about to be scored on and the comparison is
    # void. Checked here rather than trusted, and it aborts rather than
    # warns.
    frames = WholeFrameSegDataset(
        coco_path=ds_cfg["coco_path"], img_dir=ds_cfg["img_dir"],
        out_size=int(wf_cfg["model"].get("crop_size", 512)))
    wf_train_frames, wf_val_frames = frame_split_from_crop_split(
        frames, train_indices)

    val_crop_set = set(val_indices)
    leaked = [f for f in wf_train_frames
              if any(c in val_crop_set for c in frames.crops_of_frame[f])]
    if leaked:
        raise SystemExit(
            f"VOID: {len(leaked)} whole-frame training frames contribute "
            f"validation crops. frame_split_from_crop_split is not doing "
            f"what its docstring claims; fix that before quoting anything.")

    roi = BaySegmenter(checkpoint=args.roi_checkpoint,
                       crop_size=int(roi_cfg["model"].get("crop_size", 256)),
                       half=bool(roi_cfg["model"].get("half", True)),
                       num_classes=6)
    wf = BaySegmenter(checkpoint=args.wholeframe_checkpoint,
                      crop_size=int(wf_cfg["model"].get("crop_size", 512)),
                      half=bool(wf_cfg["model"].get("half", True)),
                      num_classes=6)

    roi_res = evaluate(roi, crops, val_indices, scales)
    wf_res = evaluate(wf, crops, val_indices, scales,
                      predict=predict_val_crops_from_frames)

    roi_bd = roi_res["boundary_mm"]["bay"]
    wf_bd = wf_res["boundary_mm"]["bay"]
    delta = wf_bd - roi_bd

    lines = []
    add = lines.append
    add("whole-frame vs per-ROI segmentation - architecture comparison")
    add("pre-registration: docs/superpowers/specs/"
        "2026-08-16-wholeframe-preregistration.md")
    add("")
    add(f"  val crops       : {roi_res['n_val_crops']}  (identical for both arms)")
    add(f"  whole-frame split: {len(wf_train_frames)} train frames, "
        f"{len(wf_val_frames)} held out")
    add(f"  leak check      : PASS - no training frame feeds a val crop")
    add(f"  mm/px           : {provenance}")
    add("")
    add(f"{'metric':<34}{'ROI':>12}{'whole-frame':>14}{'delta':>10}")
    add("-" * 70)
    add(f"{'bay boundary displacement (mm)':<34}{roi_bd:>12.3f}"
        f"{wf_bd:>14.3f}{delta:>+10.3f}")
    for cls in ("bay", "electronics", "obstruction"):
        r, w = roi_res["ious"][cls], wf_res["ious"][cls]
        add(f"{cls + ' IoU':<34}{r:>12.4f}{w:>14.4f}{w - r:>+10.4f}")
    add(f"{'selected mean IoU':<34}"
        f"{roi_res['selected_mean_iou']:>12.4f}"
        f"{wf_res['selected_mean_iou']:>14.4f}"
        f"{wf_res['selected_mean_iou'] - roi_res['selected_mean_iou']:>+10.4f}")
    add("")

    # The pre-registered verdict. Stated mechanically so it cannot be
    # narrated into whichever answer is more flattering.
    add(f"baseline (published ROI figure) : {ROI_BASELINE_MM:.3f} mm")
    add(f"prediction: whole-frame >= {ROI_BASELINE_MM + PREDICTED_EXCESS_MM:.3f} mm "
        f"(worse by >= {PREDICTED_EXCESS_MM:.2f} mm)")
    add(f"falsifier : whole-frame within {FALSIFIER_WITHIN_MM:.2f} mm "
        f"AND <= {FALSIFIER_MS:.0f} ms/frame")
    add("")
    if np.isnan(wf_bd):
        add("VERDICT: INCONCLUSIVE - no whole-frame bay boundary was defined.")
    elif delta >= PREDICTED_EXCESS_MM:
        add("VERDICT: PREDICTION HELD. Spec 3.1's conclusion is confirmed by "
            "measurement; its stated reason (connected components) should be "
            "replaced by the measured one (resolution per object at this scale).")
    elif abs(delta) <= FALSIFIER_WITHIN_MM:
        add("VERDICT: PREDICTION REFUTED on accuracy. Check the per-frame "
            "wall clock against 40 ms before amending spec 3.1 - the falsifier "
            "requires BOTH.")
    else:
        add("VERDICT: NEITHER. Whole-frame is worse, but by less than the "
            "0.50 mm called resolvable in advance. Report as an effect too "
            "small to resolve at this n - not as a win for either arm.")

    text = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
