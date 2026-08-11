"""
plan.arbitration - reconciling two placement-area estimates.

The segmenter produces one estimate directly (its `bay` channel) and a
second derivable from the other channels. Both are modal - what the
camera can see - so they answer the same question two ways and can
disagree.

The planner consumes their INTERSECTION. The asymmetry is deliberate:
siting a cell on a PCB is a damage event, whereas skipping a cartridge
costs one cycle.

Their IoU is NOT a confidence signal and nothing gates on it. It was
measured to correlate with placement error in the wrong direction in
all four cataloged SKUs and retired - see `derived_placement`'s
docstring for the mechanism, FDR v3 section 13.2.1 for the numbers. It
is still computed and reported (`PlacementArea.consistency_iou`) as
observability; treat it as a diagnostic, never as a permission.

Pure NumPy and cv2. No torch: this runs inside the planning cycle, which
FDR O3 caps at 8 ms per cartridge.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception as exc:                # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc

_LOG = logging.getLogger(__name__)


# Fixed order, matching recog.seg_dataset.SEG_CHANNELS. A test in
# tests/test_arbitration.py pins the two together.
CH_BACKGROUND = 0
CH_CARTRIDGE = 1
CH_BAY = 2
CH_ELECTRONICS = 3
CH_OBSTRUCTION = 4
CH_BATTERY = 5


def centre_component(label_map: np.ndarray) -> np.ndarray:
    """The connected component of non-background containing the centre.

    Cartridges sit adjacent in the jig, so a jittered or over-large crop
    catches a neighbour's edge; without this the derived estimate would
    erode the union of two cartridges.

    Connectivity spans ALL non-background channels, battery included. A
    cell lying across a bay would otherwise sever the region and leave
    the centre component covering half the cartridge.

    If the crop centre itself lands on background - a badly-placed
    detector box - this returns an EMPTY mask. It does NOT fall back to
    the largest (or any other) foreground component. That fallback was
    tried and is wrong: this robot seats lithium cells, and guessing
    which blob is "ours" risks silently handing back a *neighbouring*
    cartridge's placement area under this cartridge's identity.
    ``direct_placement`` is not scoped by the crop centre either, so a
    guessed component can line up with it well enough to produce a
    non-empty, plausible-looking ``P_safe`` for the wrong physical
    object - and the arbitration IoU will not catch it, because both
    estimates would then be describing the same wrong blob consistently.
    An empty mask instead propagates to an empty ``P_derived`` and hence
    an empty ``P_safe``; the caller skips the cartridge and retries next
    frame. That is a lost cycle, not a misplaced cell - the asymmetry
    this whole module is built on. Do not restore the guess.
    """
    fg = (label_map != CH_BACKGROUND).astype(np.uint8)
    if not fg.any():
        return fg.astype(bool)

    n, comps = cv2.connectedComponents(fg, connectivity=8)
    h, w = label_map.shape
    centre = comps[h // 2, w // 2]
    if centre == 0:
        # The crop centre landed on background - a badly-placed box.
        # Logged (not raised: a new exception type here would just move
        # the guess-vs-skip decision to the caller without adding
        # information) so a monitoring layer can count how often the
        # detector hands us bad boxes, distinct from ordinary empty
        # frames where this branch never fires. A future task that
        # wants the caller to branch on this programmatically will need
        # a signature change here (e.g. a reason alongside the mask) -
        # this log line is the hook for that.
        _LOG.warning(
            "centre_component: crop centre (%d, %d) is background "
            "(bad detector box?) - returning an empty mask instead of "
            "guessing at a neighbouring blob", h // 2, w // 2)
        return np.zeros_like(fg, dtype=bool)
    return comps == centre


def direct_placement(label_map: np.ndarray) -> np.ndarray:
    """The network's own answer: the `bay` channel."""
    return label_map == CH_BAY


def derived_placement(label_map: np.ndarray,
                       wall_inset_px: int) -> np.ndarray:
    """Interior floor minus everything occupying it.

    Built from the channels the direct estimate does NOT use, so the two
    are genuinely independent and their disagreement carries
    information.

    NOT LOAD-BEARING, measured: the loop below (minus CH_ELECTRONICS /
    CH_OBSTRUCTION / CH_BATTERY) is a structural no-op on P_safe's
    content, and therefore on the confidence signal arbitrate() reports.
    recog/bay_segmenter.py:110 emits logits.argmax(dim=1) - one label
    per pixel, mutually exclusive by construction - so wherever
    direct_placement() is True, label_map == CH_BAY already implies the
    pixel is none of these three classes. The subtraction can only ever
    remove pixels where direct_placement() is already False, which
    P_safe = P_direct & P_derived discards regardless. P_direct and
    P_derived are consequently NOT independent estimates: they are the
    same argmax read twice, with an erosion band applied to one read.
    Per-SKU correlation between the resulting IoU and the optimistic
    error is POSITIVE in all four cataloged SKUs (the wrong sign for a
    confidence gate) - docs/receipts/tau_independence_correlation.txt,
    docs/FDR_v3.md section 13.2.1. tau is retired as a confidence gate
    for this reason; P_safe itself is kept, because the geometric
    intersection is a real constraint independent of what the IoU means.
    This function's BEHAVIOUR is unchanged by that finding - do not
    "simplify" this loop away on the strength of this comment alone.
    """
    keep = centre_component(label_map)
    interior = keep.astype(np.uint8)

    if wall_inset_px > 0:
        k = 2 * int(wall_inset_px) + 1
        interior = cv2.erode(
            interior, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    out = interior.astype(bool)
    for ch in (CH_ELECTRONICS, CH_OBSTRUCTION, CH_BATTERY):
        out &= label_map != ch
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def arbitrate(label_map: np.ndarray,
              wall_inset_px: int) -> Tuple[np.ndarray, float]:
    """``(P_safe, iou)``.

    ``P_safe`` is the intersection: a pixel is placeable only where both
    estimates agree, and it is applied unconditionally. ``iou`` is
    returned for reporting only - no caller gates on it (the threshold
    in docs/receipts/tau_calibration.txt is retired; see this module's
    docstring).
    """
    direct = direct_placement(label_map)
    derived = derived_placement(label_map, wall_inset_px)
    return direct & derived, mask_iou(direct, derived)


def admits_a_cell(region: np.ndarray, cell_w_px: int,
                  cell_h_px: int) -> bool:
    """Can ``region`` contain a cell footprint at any packer orientation?

    This is the criterion τ is calibrated against, and it is
    MORPHOLOGICAL rather than areal on purpose. An earlier draft of the
    design spec tested whether the optimistic-error AREA exceeded one
    cell footprint. That is the wrong question: 1190 mm2 spread along a
    boundary as a one-pixel rim cannot hold a cell, while the same area
    in one blob is a misplacement waiting to happen.

    Eroding by the cell's own footprint answers the right question
    directly - a non-empty result means some position exists where the
    whole cell fits inside the error.
    """
    if not region.any():
        return False

    src = region.astype(np.uint8)
    for w, h in ((int(cell_w_px), int(cell_h_px)),
                 (int(cell_h_px), int(cell_w_px))):
        if w < 1 or h < 1:
            continue
        if w > region.shape[1] or h > region.shape[0]:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
        if cv2.erode(src, kernel).any():
            return True
    return False
