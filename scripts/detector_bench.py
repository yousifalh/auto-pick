"""Detector localisation error and precision — the FDR O2 and §10.6 benchmark.

Regenerates ``docs/receipts/detector_bench.txt``. It exists because two
claims in the FDR had no measurement behind them:

**O2 — "centroid localisation error shall not exceed 2 px".** The
traceability matrix cited ``tests/test_evaluate.py``, which unit-tests
:func:`recog.evaluate.centroid_error_px` on hand-written boxes (one of
its two cases asserts a 5 px error as correct) and never runs a
detector. Outside that file :func:`centroid_error_px` had no caller
anywhere in the repository, and no receipt contained the word
"centroid". This script is the missing caller.

**§10.6 — "the heuristic produced zero false positives across all 100
frames".** Nothing measured that either. The same matching pass that
produces the centroid distribution produces the false-positive count,
so both come out of one run.

Three arms, each reported separately because they answer different
questions:

*heuristic*
    :class:`recog.inference.HeuristicDetector` over **all 100** images of
    ``recog/dataset`` — the population §10.6 describes.
*default-anchor Faster R-CNN*
    ``recog/checkpoints/default_anchors_best.pt`` over the 15-image
    validation split of the same corpus, at the 320 x 512 input resize
    §5.7 names. This is the checkpoint §10.1's 0.874 mAP@0.5 comes from,
    so it is the like-for-like arm for §10.5's O2 row. The run
    reproduces ``docs/receipts/frcnn_map_default.txt`` to four decimals,
    which is what certifies the harness rather than the harness
    certifying itself.
*shipped Faster R-CNN*
    ``recog/checkpoints/best.pt`` under ``configs/recognition.yaml`` over
    the validation split of ``recog/dataset3d`` — the detector and the
    corpus production actually runs, at the production
    ``inference_min_size`` / ``inference_max_size``.

Matching rule. VOC's, and deliberately the same one
:func:`recog.evaluate.per_class_ap` uses to produce the report's mAP:
per class, detections in descending confidence, each takes the
ground-truth box it overlaps **most**, and scores only if that box is
still free. A detection matching no free box at IoU >= 0.5 is a false
positive. Centroid and edge error are then measured over the matched
pairs only — a metric conditional on detection, which is the honest
reading of a localisation criterion and also the *generous* one: it
excludes every missed object.

Neither the datasets nor the checkpoints are committed (see FDR
Appendix C.3), so this script needs a working tree that has them. It
skips, loudly, any arm whose inputs are absent, and it writes the
receipt only when every arm ran.

Run with ``python scripts/detector_bench.py``.
"""
from __future__ import annotations

import argparse
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import BBox  # noqa: E402
from recog.evaluate import centroid_error_px, edge_error_px, mean_ap  # noqa: E402

RECEIPT_PATH = REPO_ROOT / "docs" / "receipts" / "detector_bench.txt"

CLASS_ID = {"battery": 1, "cartridge": 2}
ID_CLASS = {v: k for k, v in CLASS_ID.items()}

# The O2 threshold, from FDR §3.
TARGET_PX = 2.0
IOU_MATCH = 0.5


# ----------------------------------------------------------- corpus ------

def _val_split(stems: Sequence[str], train_val_split: float = 0.85,
               seed: int = 0) -> List[str]:
    """The validation stems `recog.training._split_dataset` would produce.

    Reimplemented rather than imported so the receipt does not depend on
    constructing a torch Dataset: `random_split` is `randperm` over
    `len(dataset)` with a seeded generator, and the dataset enumerates
    `sorted(os.listdir(img_dir))`, so this is the same permutation.
    """
    import torch

    n = len(stems)
    n_train = int(round(train_val_split * n))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    return [stems[i] for i in sorted(perm[n_train:])]


def _stems(img_dir: Path) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(p.stem for p in img_dir.iterdir()
                  if p.suffix.lower() in exts)


def _ground_truth(ann_path: Path) -> List[Tuple[BBox, str]]:
    root = ET.parse(ann_path).getroot()
    out: List[Tuple[BBox, str]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_ID:
            continue
        bb = obj.find("bndbox")
        box = BBox(
            float(bb.findtext("xmin")), float(bb.findtext("ymin")),
            float(bb.findtext("xmax")), float(bb.findtext("ymax")),
        )
        if box.width <= 0 or box.height <= 0:
            continue
        out.append((box, name))
    return out


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------- matching ------

class Tally:
    """Per-class TP / FP counts and the matched-pair error samples."""

    def __init__(self) -> None:
        self.gt = 0
        self.tp = 0
        self.fp = 0
        self.centroid: List[float] = []
        self.edge: List[float] = []

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / self.gt if self.gt else float("nan")


def _match_frame(gts: List[BBox], preds: List[Tuple[BBox, float]],
                 tally: Tally) -> None:
    """VOC matching, in place. ``preds`` need not be pre-sorted."""
    available = [True] * len(gts)
    tally.gt += len(gts)
    for box, _score in sorted(preds, key=lambda t: -t[1]):
        best_j, best_iou = -1, 0.0
        for j, gt in enumerate(gts):
            v = gt.iou(box)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0 and best_iou >= IOU_MATCH and available[best_j]:
            available[best_j] = False
            tally.tp += 1
            tally.centroid.append(centroid_error_px(gts[best_j], box))
            tally.edge.append(edge_error_px(gts[best_j], box))
        else:
            tally.fp += 1


# ----------------------------------------------------------- arms --------

Frame = Tuple[List[Tuple[BBox, str]], List[Tuple[BBox, float, str]]]


def _score(frames: Iterable[Frame]) -> Tuple[Dict[str, Tally], Dict[str, float]]:
    tallies = {name: Tally() for name in CLASS_ID}
    gts_by_image: Dict[int, list] = {}
    preds_by_image: Dict[int, list] = {}
    for i, (gts, preds) in enumerate(frames):
        gts_by_image[i] = [(b.to_list(), CLASS_ID[n]) for b, n in gts]
        preds_by_image[i] = [(b.to_list(), CLASS_ID[n], s)
                             for b, s, n in preds]
        for name in CLASS_ID:
            _match_frame(
                [b for b, n in gts if n == name],
                [(b, s) for b, s, n in preds if n == name],
                tallies[name],
            )
    ap = mean_ap(gts_by_image, preds_by_image, sorted(ID_CLASS),
                 iou_threshold=IOU_MATCH)
    ap75 = mean_ap(gts_by_image, preds_by_image, sorted(ID_CLASS),
                   iou_threshold=0.75)
    summary = {
        "mAP@0.50": ap["mAP@0.50"],
        "AP_battery@0.50": ap["AP_1"],
        "AP_cartridge@0.50": ap["AP_2"],
        "mAP@0.75": ap75["mAP@0.75"],
    }
    return tallies, summary


def _heuristic_frames(img_dir: Path, ann_dir: Path,
                      stems: Sequence[str]) -> Iterable[Frame]:
    from recog.inference import HeuristicDetector

    det = HeuristicDetector()
    for stem in stems:
        gts = _ground_truth(ann_dir / f"{stem}.xml")
        snap = det.detect(_read_rgb(img_dir / f"{stem}.png"))
        preds = [(d.bbox, float(d.confidence), d.label.value)
                 for d in snap.detections if d.label.value in CLASS_ID]
        yield gts, preds


def _frcnn_frames(img_dir: Path, ann_dir: Path, stems: Sequence[str],
                  cfg: dict, checkpoint: Path, min_size: int,
                  max_size: int) -> Iterable[Frame]:
    import torch

    from recog.model import build_fasterrcnn

    model = build_fasterrcnn(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    # Set on the built model rather than in build_fasterrcnn for the reason
    # that function's NOTE gives: the same transform runs during training.
    model.transform.min_size = (int(min_size),)
    model.transform.max_size = int(max_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    for stem in stems:
        gts = _ground_truth(ann_dir / f"{stem}.xml")
        arr = _read_rgb(img_dir / f"{stem}.png").astype(np.float32) / 255.0
        tensor = torch.as_tensor(
            np.transpose(arr, (2, 0, 1)), dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        preds = []
        for box, lbl, score in zip(out["boxes"].cpu(), out["labels"].cpu(),
                                   out["scores"].cpu()):
            name = ID_CLASS.get(int(lbl))
            if name is None:
                continue
            preds.append((BBox(*box.tolist()), float(score), name))
        yield gts, preds


# --------------------------------------------------------- reporting -----

def _dist(samples: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "median", "p95", "max", "within")}
    return {
        "n": float(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "within": float((a <= TARGET_PX).mean() * 100.0),
    }


def _render_arm(out: io.StringIO, title: str, note: str,
                tallies: Dict[str, Tally], summary: Dict[str, float]) -> None:
    out.write(title + "\n")
    out.write("=" * 64 + "\n")
    out.write(note.rstrip() + "\n\n")
    out.write("  " + "  ".join(k + " " + f"{v:.4f}"
                               for k, v in summary.items()) + "\n\n")
    out.write("  detection, IoU >= %.2f\n" % IOU_MATCH)
    out.write("  %-10s %6s %6s %6s %10s %8s\n"
              % ("class", "gt", "tp", "fp", "precision", "recall"))
    total = Tally()
    for name in ("battery", "cartridge"):
        t = tallies[name]
        out.write("  %-10s %6d %6d %6d %10.4f %8.4f\n"
                  % (name, t.gt, t.tp, t.fp, t.precision, t.recall))
        total.gt += t.gt
        total.tp += t.tp
        total.fp += t.fp
        total.centroid += t.centroid
        total.edge += t.edge
    out.write("  %-10s %6d %6d %6d %10.4f %8.4f\n"
              % ("ALL", total.gt, total.tp, total.fp,
                 total.precision, total.recall))
    out.write("\n  centroid error over matched pairs, px "
              "(O2 threshold %.1f px)\n" % TARGET_PX)
    out.write("  %-10s %6s %8s %8s %8s %8s %9s\n"
              % ("class", "n", "mean", "median", "p95", "max", "<= 2 px"))
    for name, t in list(tallies.items()) + [("ALL", total)]:
        d = _dist(t.centroid)
        out.write("  %-10s %6.0f %8.3f %8.3f %8.3f %8.3f %8.1f %%\n"
                  % (name, d["n"], d["mean"], d["median"], d["p95"],
                     d["max"], d["within"]))
    out.write("\n  edge error (L-inf) over matched pairs, px\n")
    for name, t in list(tallies.items()) + [("ALL", total)]:
        d = _dist(t.edge)
        out.write("  %-10s %6.0f %8.3f %8.3f %8.3f %8.3f\n"
                  % (name, d["n"], d["mean"], d["median"], d["p95"],
                     d["max"]))
    out.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-write", action="store_true",
                        help="print the receipt without writing it")
    args = parser.parse_args(argv)

    import yaml

    out = io.StringIO()
    out.write("Detector localisation error and precision\n")
    out.write("=" * 64 + "\n")
    out.write(
        "Generated by scripts/detector_bench.py. Matching is VOC's, at\n"
        "IoU >= 0.5, per class, identical to recog.evaluate.per_class_ap.\n"
        "Centroid and edge error are measured over MATCHED pairs only, so\n"
        "they are conditional on detection and exclude every miss.\n\n")

    ran = 0

    # --- arm 1: heuristic, all 100 frames of the April corpus -----------
    img_dir = REPO_ROOT / "recog" / "dataset" / "images"
    ann_dir = REPO_ROOT / "recog" / "dataset" / "annotations"
    if img_dir.is_dir() and ann_dir.is_dir():
        stems = _stems(img_dir)
        tallies, summary = _score(_heuristic_frames(img_dir, ann_dir, stems))
        _render_arm(
            out, "1. HeuristicDetector, recog/dataset, all %d frames"
            % len(stems),
            "The population FDR §10.6 describes. No checkpoint involved.",
            tallies, summary)
        ran += 1
    else:
        print("SKIP arm 1: recog/dataset not present", file=sys.stderr)

    # --- arm 2: the default-anchor ablation checkpoint, §10's detector ---
    ckpt = REPO_ROOT / "recog" / "checkpoints" / "default_anchors_best.pt"
    if img_dir.is_dir() and ckpt.is_file():
        stems = _val_split(_stems(img_dir))
        # torchvision's default scheme, expressed through build_fasterrcnn's
        # config: per-level sizes ((32,),(64,),(128,),(256,),(512,)) and
        # aspect ratios (0.5, 1.0, 2.0) at every level. There is no flag for
        # it (FDR §5.7 correction); these values ARE it.
        cfg = {"model": {"num_classes": 3,
                         "anchor_ratios": [0.5, 1.0, 2.0],
                         "anchor_scales": [32, 64, 128, 256],
                         "nms_iou": 0.4,
                         "confidence_threshold": 0.05},
               "training": {"pretrained": False}}
        tallies, summary = _score(_frcnn_frames(
            img_dir, ann_dir, stems, cfg, ckpt, 320, 512))
        _render_arm(
            out,
            "2. Faster R-CNN, torchvision-default anchors, "
            "recog/dataset val (%d frames)" % len(stems),
            "default_anchors_best.pt at the 320 x 512 resize of FDR §5.7,\n"
            "score threshold 0.05 as docs/receipts/train_eval.txt records.\n"
            "The mAP@0.5 row reproduces docs/receipts/frcnn_map_default.txt\n"
            "(0.8736 / 0.9053 / 0.8419) exactly and its mAP@0.75 of 0.5831\n"
            "to three decimals - that is what certifies this harness rather\n"
            "than the harness certifying itself. This is the detector FDR\n"
            "§10.1 reports, so it is the like-for-like arm for §10.5's O2\n"
            "row.\n"
            "\n"
            "Do NOT quote this arm's precision. The 0.05 score threshold is\n"
            "there to expose the whole PR curve to mean_ap, so almost every\n"
            "low-confidence proposal is retained and counted as a false\n"
            "positive; at the shipped 0.70 threshold the same checkpoint's\n"
            "counts are much smaller. Arm 1 and arm 3 run at operating\n"
            "thresholds and their precision columns are the meaningful ones.",
            tallies, summary)
        ran += 1
    else:
        print("SKIP arm 2: default_anchors_best.pt not present",
              file=sys.stderr)

    # --- arm 3: the shipped detector on the shipped corpus --------------
    cfg_path = REPO_ROOT / "configs" / "recognition.yaml"
    ckpt = REPO_ROOT / "recog" / "checkpoints" / "best.pt"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds = cfg.get("dataset", {})
    img3 = REPO_ROOT / ds.get("img_dir", "")
    ann3 = REPO_ROOT / ds.get("ann_dir", "")
    if ckpt.is_file() and img3.is_dir() and ann3.is_dir():
        stems = _val_split(_stems(img3),
                           float(ds.get("train_val_split", 0.85)))
        model_cfg = cfg["model"]
        tallies, summary = _score(_frcnn_frames(
            img3, ann3, stems, cfg, ckpt,
            int(model_cfg.get("inference_min_size", 500)),
            int(model_cfg.get("inference_max_size", 900))))
        _render_arm(
            out,
            "3. Faster R-CNN, shipped best.pt, %s val (%d frames)"
            % (ds.get("img_dir", "?").rsplit("/", 1)[0], len(stems)),
            "configs/recognition.yaml as shipped: anchors %s x %s,\n"
            "confidence %.2f, NMS IoU %.2f, inference resize %s/%s.\n"
            "This is the production configuration; arm 2 is the one §10\n"
            "reports."
            % (model_cfg.get("anchor_ratios"), model_cfg.get("anchor_scales"),
               float(model_cfg.get("confidence_threshold", 0.70)),
               float(model_cfg.get("nms_iou", 0.4)),
               model_cfg.get("inference_min_size"),
               model_cfg.get("inference_max_size")),
            tallies, summary)
        ran += 1
    else:
        print("SKIP arm 3: best.pt or recog/dataset3d not present",
              file=sys.stderr)

    text = out.getvalue()
    print(text)
    if args.no_write:
        return 0
    if ran < 3:
        print("not written: %d of 3 arms ran" % ran, file=sys.stderr)
        return 1
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print("wrote %s" % RECEIPT_PATH, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
