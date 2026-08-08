"""
plan.arbitration - reconciling two placement-area estimates.

The segmenter produces one estimate directly (its `bay` channel) and a
second derivable from the other channels. Both are modal - what the
camera can see - so they answer the same question two ways and can
disagree.

The planner consumes their INTERSECTION. The asymmetry is deliberate:
siting a cell on a PCB is a damage event, whereas skipping a cartridge
costs one cycle. Their IoU is the confidence signal that says how much
to trust the frame at all.

Pure NumPy and cv2. No torch: this runs inside the planning cycle, which
FDR O3 caps at 8 ms per cartridge.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except Exception as exc:                # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc


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
    """
    fg = (label_map != CH_BACKGROUND).astype(np.uint8)
    if not fg.any():
        return fg.astype(bool)

    n, comps = cv2.connectedComponents(fg, connectivity=8)
    h, w = label_map.shape
    centre = comps[h // 2, w // 2]
    if centre == 0:
        # The crop centre landed on background - a badly-placed box.
        # Fall back to the largest component rather than returning
        # nothing, so one bad box does not silently void the cartridge.
        counts = np.bincount(comps.ravel())
        counts[0] = 0
        if counts.max() == 0:
            return np.zeros_like(fg, dtype=bool)
        centre = int(counts.argmax())
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
    estimates agree. ``iou`` is the per-cartridge confidence the caller
    gates on - see plan.placement_area and the calibrated threshold in
    docs/receipts/tau_calibration.txt.
    """
    direct = direct_placement(label_map)
    derived = derived_placement(label_map, wall_inset_px)
    return direct & derived, mask_iou(direct, derived)
