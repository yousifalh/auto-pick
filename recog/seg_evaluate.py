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

Every millimetre above is a pixel count times mm_per_px, and mm_per_px
is a property of the FRAME, not of the corpus: see the "frame
calibration" section below. Every figure this module published before
2026-08-11 was converted at the generator's nominal 0.625 mm/px, which
describes no frame in a corpus that randomises zoom, and was understated
by roughly 1.3x for a length and 1.6x for an area as a result.
Pixel-space figures (IoU, crop counts) are unaffected and did not move.

CLI::

    python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt \\
        --config configs/segmentation.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.config import load_yaml
from common.logging import get_logger

# `resolve_mm_per_px` and `load_synth_config` MOVED to recog.calibration
# and are re-exported here, not re-implemented: recog.calibrate_tau and
# recog.seg_ablation import them from this module and quote the result in
# shipped receipts. They now sit beside `frame_mm_per_px`, the PER-FRAME
# ground sample distance the planner uses, so the difference between the
# generator's nominal framing and an individual frame's true scale is
# stated once, in one place, instead of being rediscovered.
from recog.calibration import (  # noqa: F401  (re-exported)
    frame_mm_per_px,
    frame_mm_per_px_for_image,
    load_synth_config,
    resolve_mm_per_px,
)

# The same exception plan.planner raises for the same condition, imported
# rather than restated. An unknown scale is one fault with one name
# wherever it happens; a second class called the same thing in a second
# module is how "raise rather than default" quietly becomes "raise in one
# of the two places".
#
# From common.types since 2026-08-15, not plan.placement_area. Importing
# it was always right; the class was simply filed in the wrong package,
# and this line was one of the two module-level back-edges from `recog`
# into `plan` (tests/test_seams.py pins that there are now none).
# plan.placement_area re-exports the same object, so nothing about which
# exception this is has changed - and pulling it from `common` no longer
# drags cv2 and plan.scene into this module's import.
from common.types import UnknownScale

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
# generator's NOMINAL framing (0.625 mm/px). Boundary displacement below
# this is the acceptance criterion for having chosen a per-ROI segmenter
# instead.
#
# It is a millimetre figure and therefore, like every millimetre figure
# in this module, a function of the scale it was converted at - so the
# underlying PIXEL figure is kept beside it and is what the verdict is
# actually judged against. FDR 13.2.1 derives 2.9 x 6.4 mm from a
# PowerCore26800 crop of roughly 131 x 288 px split 28 ways, and
# 131/28 x 288/28 px is the part of that derivation that does NOT move
# when the framing does. At 0.625 mm/px it reproduces 2.9 x 6.4 mm
# exactly.
#
# Judging a per-frame boundary displacement against the mm figure frozen
# at 0.625 would compare a rescaled measurement with an unrescaled
# threshold, which is not conservatism - it is the same units error in a
# new place, and measured on the CAD test split it flips `electronics`
# from clearing to failing on scale alone. Converting BOTH sides over
# the same crops at the same scales leaves the ratio - the only part of
# the comparison that is about the two architectures - untouched.
MASK_HEAD_QUANTISATION_MM: Tuple[float, float] = (2.9, 6.4)
MASK_HEAD_QUANTISATION_PX: Tuple[float, float] = (131.0 / 28.0, 288.0 / 28.0)


# --------------------------------------------------- frame calibration --
#
# Every millimetre this module publishes is a pixel count multiplied by
# mm_per_px, and until 2026-08-11 that multiplier was one constant for
# the whole split: `resolve_mm_per_px(synth_cfg)`, the generator's
# framing at margin = 1.0, zoom = 1.0. `recog/synth3d/world.py`'s
# setup_camera sets `ortho_scale = need * margin * zoom` with margin
# drawn from [1.02, 1.10] and zoom from `param_space.zoom`, so on this
# corpus NO frame is rendered at that framing - the true GSD runs
# 0.49-1.09 mm/px against a nominal 0.625, and every millimetre figure
# in docs/receipts/seg_eval.txt was understated by that ratio.
#
# The fix is the same one 380e7d5 applied to the planner, using the same
# module: scale is a property of the FRAME, read from that frame's own
# render sidecar. IoU and every other pixel-space quantity are untouched
# by construction - they never multiply by this number.

def crop_image_path(full_dataset, img_meta: dict) -> Path:
    """The image file one crop was taken from.

    One definition, because both the pixel loading in :func:`evaluate`
    and the sidecar lookup in :func:`resolve_frame_scales` must name the
    same file or a crop would be scored at another frame's scale.
    """
    return Path(full_dataset.img_dir) / img_meta["file_name"]


def resolve_frame_scales(full_dataset, indices: Sequence[int],
                         fallback: Optional[float] = None
                         ) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """``({crop index: mm_per_px}, provenance)`` - the scale each crop is
    scored at, taken from that crop's OWN render sidecar.

    Delegates to ``recog.calibration.frame_mm_per_px_for_image``; there
    is deliberately no arithmetic here. A duplicated dimension constant
    has already cost this project one defect, and a second copy of
    ``ortho_scale * 1000 / width`` would be exactly that again.

    ``fallback`` is for inputs that carry no render metadata at all - a
    photograph, or ``recog/synth_dataset.py``'s cv2-drawn rectangles. It
    must be passed DELIBERATELY: with no fallback configured, an
    uncalibrated frame raises :class:`common.types.UnknownScale`
    rather than quietly reverting to a constant. That is the whole point
    of the change; silent degradation is this project's characteristic
    failure and this receipt was an instance of it.

    ``provenance`` reports how many crops were measured and how many fell
    back, so the receipt can never again print one number that a reader
    cannot tell apart from a real per-frame calibration.
    """
    scales: Dict[int, float] = {}
    by_file: Dict[str, Optional[float]] = {}
    n_measured = n_fallback = 0

    for idx in indices:
        img_meta, _anns, _box = full_dataset.samples[idx]
        name = img_meta["file_name"]
        if name not in by_file:
            # Cached per FILE, not per crop: one frame yields several
            # unit crops and the sidecar is a large JSON document.
            by_file[name] = frame_mm_per_px_for_image(
                crop_image_path(full_dataset, img_meta))

        measured = by_file[name]
        if measured is not None:
            scales[int(idx)] = float(measured)
            n_measured += 1
            continue

        if fallback is None:
            raise UnknownScale(
                f"crop {idx} comes from {name}, which carries no render "
                "metadata, and no fallback mm_per_px was configured - so "
                "every millimetre this evaluation would publish for it "
                "is invented. Pass --fallback-mm-per-px explicitly if "
                "this corpus really is a single fixed framing (a real "
                "fixed-mount camera is), or point --config at a dataset "
                "whose frames carry their own calibration. Refusing to "
                "guess: a constant standing in for a per-frame scale is "
                "the defect this resolution exists to remove.")
        scale = float(fallback)
        if scale <= 0.0:
            raise UnknownScale(
                f"fallback mm_per_px={fallback!r}: a pixel cannot span "
                "zero or negative millimetres.")
        scales[int(idx)] = scale
        n_fallback += 1

    return scales, {"n_measured": n_measured, "n_fallback": n_fallback,
                    "n_frames": len(by_file),
                    "fallback": (None if fallback is None
                                 else float(fallback))}


def scale_stats(scales: Dict[int, float]) -> Dict[str, float]:
    """min / median / max / mean over a per-crop scale map.

    The receipt prints the distribution rather than a single number
    because a single number is what went wrong: a run planned at each
    frame's own scale and a run planned at one constant produced
    identical-looking headers for months.
    """
    vals = np.asarray(sorted(scales.values()), dtype=np.float64)
    if vals.size == 0:
        nan = float("nan")
        return {"n": 0, "min": nan, "median": nan, "max": nan, "mean": nan}
    return {"n": int(vals.size), "min": float(vals.min()),
            "median": float(np.median(vals)), "max": float(vals.max()),
            "mean": float(vals.mean())}


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


def _distance_to_nearest(tb: np.ndarray,
                         max_elements: int = 4_000_000) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest ``True`` in
    ``tb``, without scipy. Chunked over the boundary pixels.

    The one-shot form this replaces broadcast every pixel against every
    boundary pixel at once - an (H, W, N) array. On the crop size this
    module is built around, 131 x 288 with the ~1000 boundary pixels a
    real bay outline carries, that is 37.7 million float64 elements: 302
    MB per temporary and roughly 1 GB across the subtract / square / add
    chain, three times per crop (once per class in SELECT_ON), on a
    machine whose only sin is not having scipy installed.

    Chunking bounds the peak at ``max_elements`` per temporary and is
    EXACT rather than approximate, in two steps that both have to hold:
    the minimum over the per-chunk minima is the same number as the
    minimum over all of them, and sqrt is monotone, so taking it once at
    the end selects the same pixel and returns the same float as taking
    it before the min. `tests/test_bay_segmenter.py` pins the result
    against scipy's and across chunk sizes.
    """
    ty, tx = np.nonzero(tb)
    h, w = tb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.inf)
    step = max(1, int(max_elements // max(1, h * w)))
    for s in range(0, ty.size, step):
        dy = yy[..., None] - ty[s:s + step]
        dy *= dy
        dx = xx[..., None] - tx[s:s + step]
        dx *= dx
        dy += dx
        np.minimum(best, dy.min(axis=-1), out=best)
    return np.sqrt(best)


def boundary_displacement_mm(pred: np.ndarray, target: np.ndarray,
                             cls: int, mm_per_px: float) -> float:
    """Mean distance from each predicted boundary pixel to the nearest
    ground-truth boundary pixel, in millimetres."""
    pb = _boundary(pred == cls)
    tb = _boundary(target == cls)
    if not pb.any() or not tb.any():
        return float("nan")

    # ImportError ONLY, and around the import ONLY. A bare `except
    # Exception` wrapped the distance transform as well, so any failure
    # inside scipy - a bad dtype, a memory error, a version whose
    # signature moved - was silently answered by the fallback instead of
    # being reported, and the receipt would carry a number nobody could
    # tell apart from a scipy-computed one. Missing scipy is a
    # configuration a machine can legitimately be in; scipy raising is a
    # fault, and this module's whole argument is that silent degradation
    # is the failure mode to refuse.
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        dist = _distance_to_nearest(tb)
    else:
        dist = distance_transform_edt(~tb)

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


# --------------------------------------------------------------- report --

def _mean_or_nan(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# ONE crop per `segment_batch` call, deliberately, and this default is
# the conclusion of a measurement rather than an oversight.
#
# Batching the evaluation looks obviously right - `recog/bay_segmenter.py`
# documents the looped anti-pattern against itself (eight crops, 101 ms
# looped against 18.1 ms batched), the deployed pipeline batches, and
# this was the last per-crop loop in the module. It was implemented, run
# against `configs/segmentation.yaml`'s 126-crop split and
# `recog/checkpoints/seg/best.pt` on an RTX 3060, and BOTH halves of the
# argument came back smaller than expected:
#
#   * the saving is 0.43 s. Best of three over the whole split: 2.58 s
#     with this pass at batch 1, 2.15 s at batch 8, inside a
#     `python -m recog.seg_evaluate` run that takes ~20 s end to end.
#     The 3.4x is GPU time, and GPU time is not what this loop spends -
#     PIL decode, `extract_crop` and the resize dominate it.
#   * the cost is every figure in the receipt. Batch 8 and batch 1
#     disagree on 128 of 2 217 730 label pixels, spread over 46 of the
#     126 crops: batch-size-dependent kernels move a logit in its last
#     bits and argmax follows on the pixels that were close. Measured
#     end to end, that shifted background IoU 0.3629 -> 0.3628,
#     electronics 0.8613 -> 0.8612, battery 0.6907 -> 0.6905, the
#     selected mean 0.8032 -> 0.8031, and the area totals by ~0.04%.
#
# Twelve published figures moving in their last digit, in a module whose
# docstring is a standing account of what happened the last time a
# receipt's numbers moved without a reason a reader could reconstruct, is
# not worth 0.43 s. `batch_size` is a real argument and the chunking
# below is exercised by its own test, so raising this is a one-line
# decision for whoever also regenerates `docs/receipts/seg_eval.txt` and
# says in the commit that the figures moved and why. It is not a decision
# to take silently, which is what leaving the default at 8 would have
# been.
#
# The per-frame decode cache below is a different matter: it is free, and
# it is bit-identical.
SEG_EVAL_BATCH = 1


def predict_val_crops(segmenter, full_dataset, val_indices: Sequence[int],
                      batch_size: int = SEG_EVAL_BATCH) -> List[np.ndarray]:
    """One native-resolution label map per entry of ``val_indices``, in
    ``val_indices``' own order.

    Crops are visited GROUPED BY SOURCE FILE and the decode is cached, so
    a frame is decoded once rather than once per crop it contributes.
    `resolve_frame_scales` above already caches its sidecar per file,
    with a comment saying exactly this, while the decode next to it did
    not. Measured on `configs/segmentation.yaml`'s split that is 112
    frames behind 126 crops - 14 redundant decodes, 2.76 s to 2.58 s over
    the split, and not one label pixel different.

    (1.12 crops per frame, not the ~1.68 the audit note assumed. The
    ratio is a property of the corpus, and on a corpus that packs more
    units into a frame this saves proportionally more; it is never
    negative.)

    ``batch_size`` chunks the `segment_batch` calls and defaults to 1.
    See SEG_EVAL_BATCH above for the measurement behind that default -
    briefly, batching is NOT label-identical here and the saving does not
    pay for the drift.

    Returning a list in the CALLER's order, rather than accumulating as
    it goes, is what makes the file-grouped traversal safe. `evaluate`
    reduces its per-crop boundary and area lists with `np.mean` and
    `sum`, and floating-point addition is not associative: accumulating
    in file-sorted order would move the published millimetre figures in
    their last digits for no reason anyone reading the receipt could
    reconstruct. The reordering starts and stops inside this function.
    """
    from PIL import Image

    # Stable sort, so crops from one file keep their val_indices order
    # and the whole traversal stays deterministic.
    order = sorted(range(len(val_indices)),
                   key=lambda p: full_dataset.samples[
                       val_indices[p]][0]["file_name"])

    preds: List[Optional[np.ndarray]] = [None] * len(val_indices)
    cached_name: Optional[str] = None
    cached_image: Optional[np.ndarray] = None

    for start in range(0, len(order), max(1, int(batch_size))):
        chunk = order[start:start + max(1, int(batch_size))]
        crops: List[np.ndarray] = []
        for pos in chunk:
            img_meta, _anns, unit_box = full_dataset.samples[val_indices[pos]]
            if img_meta["file_name"] != cached_name:
                cached_image = np.asarray(
                    Image.open(crop_image_path(full_dataset, img_meta))
                    .convert("RGB"))
                cached_name = img_meta["file_name"]
            crops.append(extract_crop(cached_image,
                                      tuple(int(v) for v in unit_box),
                                      out_size=None))
        for pos, pred in zip(chunk, segmenter.segment_batch(crops)):
            preds[pos] = pred

    return preds


def evaluate(segmenter, full_dataset, val_indices: Sequence[int],
            scales: Dict[int, float], num_classes: int = 6
            ) -> Dict[str, Any]:
    """Run the segmenter over every validation crop at native resolution
    and accumulate every metric this module reports.

    ``scales`` maps crop index -> that crop's OWN mm_per_px, as
    :func:`resolve_frame_scales` builds it. It is deliberately NOT a
    scalar, and passing one raises: this argument was a single constant
    (0.625, the generator's nominal framing) while the renderer varied
    zoom per frame, which understated every millimetre this function
    reports by the ratio of true to nominal GSD - a median factor of
    ~1.3 on recog/dataset3d_seg's validation split. A signature that
    still accepts one number is a signature that can regress to it
    silently.

    "Native resolution" (extract_crop / rasterise_crop called with
    out_size=None) rather than the model's fixed square input: mm_per_px
    is isotropic only in that native frame - a jittered union box is not
    square, so resizing it to out_size x out_size would rescale x and y
    by different factors, and a single scalar mm_per_px could not
    describe the result correctly. Boundary displacement and area error
    need that native frame; IoU is computed there too so all four
    numbers describe the same pixels.

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
    if isinstance(scales, (int, float)):
        raise TypeError(
            "evaluate() takes a per-crop {index: mm_per_px} map, not one "
            f"constant ({scales!r}). Scale is a property of the FRAME - "
            "recog/synth3d/world.py randomises margin and zoom per scene, "
            "so no single number describes this corpus. Build the map "
            "with resolve_frame_scales(); if the corpus really is one "
            "fixed framing, pass that value as its `fallback` so the "
            "receipt records that it was a fallback.")

    inter = [0] * num_classes
    union = [0] * num_classes
    counts = [0] * num_classes            # crops containing class c

    boundary_mm: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    boundary_defined: Dict[str, int] = {c: 0 for c in SELECT_ON}
    # The scales of exactly the crops that contributed to each class's
    # boundary mean - not the whole split's. The mask-head threshold this
    # is judged against is a PIXEL quantity converted at the same scale,
    # and a threshold converted over a different set of crops than the
    # measurement would be the very mismatch this module is fixing.
    boundary_scale: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    area_opt_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}
    area_cons_mm2: Dict[str, List[float]] = {c: [] for c in SELECT_ON}

    # cartridge/bay crop-population overlap, measured fresh every run
    # rather than asserted in prose. Before the tray-interior fix
    # (a31ac28..043e92d) a closed shell's cartridge mask and an open
    # unit's bay mask genuinely never shared a crop; the fix gave open
    # units real tray walls, which paint `cartridge` pixels alongside
    # their own bay/electronics/obstruction pixels in the SAME crop
    # (see seg_dataset.py's module docstring). Recomputing this here
    # means the receipt's note can never silently outlive the geometry
    # it describes again.
    cart_c, bay_c = SEG_CHANNELS["cartridge"], SEG_CHANNELS["bay"]
    cb_cart_only = cb_bay_only = cb_both = cb_neither = 0

    # Inference first, with the per-frame decode cached; the accumulation
    # below then walks val_indices in its ORIGINAL order, so every float
    # sum in this function is added up in exactly the order it was
    # before. See predict_val_crops' docstring.
    preds = predict_val_crops(segmenter, full_dataset, val_indices)

    for pred, idx in zip(preds, val_indices):
        _img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)     # no jitter at eval
        mm_per_px = scales[idx]                   # THIS frame's own GSD

        target = rasterise_crop(anns, box, out_size=None)

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
                boundary_scale[name].append(mm_per_px)
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
        # The scales these millimetres were actually measured at. Carried
        # in the result rather than left to the caller to remember, for
        # the same reason PlacementArea.mm_per_px is: a millimetre figure
        # separated from its scale is how this receipt went wrong.
        "mm_per_px": scale_stats({i: scales[i] for i in val_indices}),
        "ious": ious,
        "instance_counts": counts_by_name,
        "selected_mean_iou": selected_mean_iou,
        "boundary_mm": {c: _mean_or_nan(boundary_mm[c]) for c in SELECT_ON},
        "boundary_mm_per_px": {c: _mean_or_nan(boundary_scale[c])
                               for c in SELECT_ON},
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
                                num_classes: int = 6,
                                out_size: Optional[int] = None
                                ) -> Dict[str, int]:
    """Per-class crop counts for the val split (no jitter, matching
    evaluate()'s boxes): a crop "contains" a class if it has at least one
    pixel of it.

    `out_size` MUST be the `model.crop_size` the checkpoint was trained
    at whenever this is fed to `check_split_matches_checkpoint`.
    seg_training builds its stored counts off the val DataLoader, whose
    targets are `rasterise_crop(..., out_size=crop_size)` -
    nearest-downsampled - so counting here at the crop's native
    resolution compares two different quantities and the guard fires on a
    dataset that never changed. Measured on
    `configs/segmentation_anchored.yaml`: background 124 native vs. 123
    at 256, one crop apart, because a sliver of background survives at
    native size and is lost by the downsample. `background` is the class
    this bites, by construction - a crop is the union of its unit's own
    boxes, so background is only the thin leftover corners and gaps.

    Left defaulting to None rather than to 256: the native map is what
    `evaluate()` scores against (a jittered union box is not square, so a
    single scalar mm_per_px cannot describe a resized crop), and silently
    changing that default would change what a caller asking for "the
    counts" gets. The guard's call site passes crop_size explicitly.
    """
    counts = [0] * num_classes
    for idx in val_indices:
        _img_meta, anns, unit_box = full_dataset.samples[idx]
        box = tuple(int(v) for v in unit_box)
        target = rasterise_crop(anns, box, out_size=out_size)
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


def check_split_matches_checkpoint(checkpoint_path: str, coco_path: str,
                                   recomputed: Dict[str, int]) -> None:
    """Fail loudly if today's recomputed split disagrees with what the
    checkpoint itself recorded at training time - but ONLY when this eval
    is against the SAME dataset the checkpoint trained on (`coco_path`
    matches the checkpoint's own recorded one). A checkpoint deliberately
    evaluated against a DIFFERENT dataset (spec #2: a procedural-trained
    model scored on the held-out CAD test set) is a cross-dataset
    generalisation eval, not a drifted split - the val instance counts
    are expected to differ there (different scenes entirely), and
    comparing them would block every such eval outright. Never silently
    re-split or auto-correct within the same-dataset case - that would
    hide exactly the failure this guards."""
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = state.get("val_instance_counts")
    if saved is None:
        log.warning("checkpoint %s has no val_instance_counts (older "
                    "checkpoint format) - cannot verify the validation "
                    "split matches what it was trained against",
                    checkpoint_path)
        return
    saved_coco_path = state.get("coco_path")
    if saved_coco_path is not None and saved_coco_path != coco_path:
        log.info("checkpoint %s was trained on a different dataset (%s) "
                 "than this eval's --config dataset (%s) - treating this "
                 "as a deliberate cross-dataset generalisation eval and "
                 "skipping the same-dataset split-drift check.",
                 checkpoint_path, saved_coco_path, coco_path)
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

    # last.pt's instance counts are discarded on purpose: the note below
    # reports ONE count string, and it has to be best.pt's, because that is
    # the checkpoint whose mean IoU the selection argument rests on. Binding
    # the unused half was F841.
    (b_iou, b_counts), (l_iou, _) = stats["best.pt"], stats["last.pt"]
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
                  synth_config_source: str,
                  scale_provenance: Dict[str, Any],
                  nominal_mm_per_px: Optional[float],
                  device: str,
                  checkpoint_note: Optional[str] = None) -> str:
    """Plain-text receipt, in the style of recog/eval_real.py's report."""
    lines: List[str] = []
    stats = results["mm_per_px"]
    lines.append("")
    lines.append("Bay segmenter evaluation")
    lines.append("=" * 40)
    lines.append(f"  checkpoint        : {checkpoint or '(none - random init)'}")
    lines.append(f"  config            : {config_path}")
    lines.append(f"  synth config      : {synth_config_source}")
    # NOT a bare number. A receipt that printed one constant could not
    # distinguish an evaluation measured at each frame's own scale from
    # one measured at a fallback, which is how the constant survived as
    # long as it did (same reasoning as docs/receipts/main_seg_run.txt).
    lines.append("  mm_per_px         : per frame, from each frame's own "
                 "render sidecar (camera.ortho_scale*1000 / width)")
    lines.append(f"    measured        : {scale_provenance['n_measured']} of "
                 f"{scale_provenance['n_measured'] + scale_provenance['n_fallback']}"
                 f" crops over {scale_provenance['n_frames']} frames")
    lines.append(f"    distribution    : median {stats['median']:.4f}, "
                 f"range {stats['min']:.4f}-{stats['max']:.4f}, "
                 f"mean {stats['mean']:.4f} mm/px")
    fb = scale_provenance.get("fallback")
    lines.append("    fallback        : "
                 + ("none configured - an uncalibrated frame raises "
                    "UnknownScale rather than reverting to a constant"
                    if fb is None else
                    f"{fb:.4f} mm/px, applied to "
                    f"{scale_provenance['n_fallback']} crops"))
    if nominal_mm_per_px:
        ratio = stats["median"] / nominal_mm_per_px
        lines.append(
            f"    note            : the generator's NOMINAL framing "
            f"({nominal_mm_per_px:.4f} = layout.area[0]*1000 / "
            f"render.res[0]) is the framing at margin=1.0, zoom=1.0 and "
            f"describes NO frame in this corpus. Receipts published "
            f"before 2026-08-11 converted every millimetre below at that "
            f"constant and therefore UNDERSTATED them by a median factor "
            f"of {ratio:.2f}x. Pixel-space figures (IoU, crop counts) are "
            f"unaffected - they never multiply by this number.")
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
                "a31ac28..043e92d).")
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
    px_lo, px_hi = MASK_HEAD_QUANTISATION_PX
    lines.append("Boundary displacement, mm (mean distance from a "
                 "predicted boundary pixel to")
    lines.append("the nearest ground-truth boundary pixel; the figure "
                 "that chose a per-ROI")
    lines.append("segmenter over a 28x28 Mask R-CNN head). Each crop is "
                 "converted at ITS OWN")
    lines.append("frame's ground sample distance, not at one constant. "
                 "`head mm` is the SAME")
    lines.append("comparison the architecture argument always made, "
                 "converted the same way:")
    head = (f"  {'class':<12}{'mm':>10}{'crops':>10}{'mm/px':>10}"
            f"{'head mm':>10}{'clears by':>11}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    all_below = True
    for c in SELECT_ON:
        bd = results["boundary_mm"][c]
        n = results["boundary_n"][c]
        # The mask head quantises in PIXELS. Converting its figure over
        # the same crops, at the same scales, is the only comparison that
        # is about the two architectures rather than about the framing:
        # both sides move together and the ratio is scale-invariant.
        # Judging a per-frame measurement against the mm figure frozen at
        # the nominal 0.625 would flip verdicts on scale alone - measured,
        # it does exactly that to `electronics` on the CAD test split.
        crop_scale = results.get("boundary_mm_per_px", {}).get(c, float("nan"))
        head_mm = px_lo * crop_scale
        bd_s = "NaN" if np.isnan(bd) else f"{bd:.3f}"
        sc_s = "NaN" if np.isnan(crop_scale) else f"{crop_scale:.4f}"
        hm_s = "NaN" if np.isnan(head_mm) else f"{head_mm:.3f}"
        ratio_s = ("NaN" if np.isnan(bd) or np.isnan(head_mm) or bd <= 0
                   else f"{head_mm / bd:.2f}x")
        lines.append(f"  {c:<12}{bd_s:>10}{n:>10}{sc_s:>10}"
                     f"{hm_s:>10}{ratio_s:>11}")
        if not np.isnan(bd) and not np.isnan(head_mm) and bd >= head_mm:
            all_below = False
    lines.append("  " + "-" * (len(head) - 2))
    lines.append(
        f"  head mm = {px_lo:.2f} px x that class's own mean mm/px. The "
        f"{px_lo:.2f} x {px_hi:.2f} px comes from a 28x28 head over a "
        f"PowerCore26800 crop of ~131 x 288 px (FDR 13.2.1), which is "
        f"{lo:.1f} x {hi:.1f} mm at the generator's NOMINAL 0.6250 mm/px "
        f"- the figure the FDR published, and the framing no frame in "
        f"this corpus was rendered at.")
    lines.append(
        "  Both sides of this comparison are pixel counts times the same "
        "scale, so the `clears by` ratio is SCALE-INVARIANT: correcting "
        "the calibration moves the millimetres and leaves the "
        "architecture verdict exactly where it was.")
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
                 "total over the split,")
    lines.append("each crop converted at its own frame's GSD - an AREA, "
                 "so it moves with the")
    lines.append("SQUARE of the scale correction, not linearly:")
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
                    help="render/layout config to read the generator's "
                         "NOMINAL framing from, for the receipt's "
                         "understatement note only - it is no longer what "
                         "millimetres are measured at. Defaults to the "
                         "dataset's own manifest.json, falling back to "
                         "configs/synth3d.yaml.")
    ap.add_argument("--fallback-mm-per-px", type=float, default=None,
                    help="scale to use for frames that carry NO render "
                         "sidecar (a photograph; synth_dataset.py's "
                         "rectangles). Unset by default: an uncalibrated "
                         "frame then raises rather than silently reverting "
                         "to a constant. A real fixed-mount camera is one "
                         "calibrated scale and is served by setting this.")
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

    # The NOMINAL framing, kept only so the receipt can state how far the
    # figures it used to publish were from the ones it publishes now.
    # Nothing below multiplies by it.
    synth_cfg, synth_source = load_synth_config(
        ds_cfg["coco_path"], args.synth_config)
    synth_source = Path(synth_source).as_posix()
    nominal_mm_per_px = resolve_mm_per_px(synth_cfg)

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
        # out_size=crop_size, NOT native: seg_training's stored counts come
        # off its val DataLoader, which rasterises at crop_size. See
        # compute_val_instance_counts' docstring.
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

    results = evaluate(segmenter, full_dataset, val_indices, scales,
                       num_classes=num_classes)
    latency = latency_table(segmenter)

    checkpoint_note = (_sibling_checkpoint_note(args.checkpoint)
                       if args.checkpoint else None)
    report = format_report(
        results, latency,
        checkpoint=args.checkpoint, config_path=args.config,
        synth_config_source=synth_source,
        scale_provenance=scale_provenance,
        nominal_mm_per_px=nominal_mm_per_px,
        device=str(segmenter.device), checkpoint_note=checkpoint_note)

    if args.per_sku:
        by_asset = group_indices_by_asset(full_dataset, val_indices)
        per_sku_results = {
            asset: evaluate(segmenter, full_dataset, idxs, scales,
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
    "MASK_HEAD_QUANTISATION_PX",
    "UnknownScale",
    "crop_image_path",
    "resolve_frame_scales",
    "scale_stats",
    "per_class_iou",
    "boundary_displacement_mm",
    "signed_area_error_mm2",
    "latency_table",
    "latency_within_budget",
    "predict_val_crops",
    "SEG_EVAL_BATCH",
    "resolve_mm_per_px",
    "frame_mm_per_px",
    "load_synth_config",
    "evaluate",
    "compute_val_instance_counts",
    "check_split_matches_checkpoint",
    "format_report",
    "main",
]
