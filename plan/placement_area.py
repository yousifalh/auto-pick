"""Valid-placement-area extraction.

Two extractors share one contract — ``extract(image_rgb, cartridge_bbox,
...) -> PlacementArea`` — so ``plan/planner.py`` and ``main.py`` can select
either without caring which one they got:

* :class:`HeuristicPlacementAreaExtractor` (formerly ``PlacementAreaExtractor``,
  kept as an alias) matches PPR §5.3.2's green-channel pipeline. It was
  measured on this project's own held-out photographs returning **zero**
  placeable area on 7 of 20 real cartridges (spec §1.1), because it assumes
  a light tray with a dark interior module and the real cartridges are
  black. Its role is narrowed to the torch-free demo on
  ``recog/synth_dataset.py``'s flat green rectangles — the hardware its PPR
  assumption actually describes. It is **not** an equal alternative for real
  imagery, and warns about that scope limit at construction time.

1. Crop the cartridge ROI from the RGB image.
2. Isolate the green channel and apply Otsu thresholding.
3. Morphological close + open to consolidate the mask.
4. Largest external contour → axis-aligned bounding rectangle.
5. Inset by a safety margin (a few pixels, ≈2-3 mm) so the gripper
   fingers don't catch on the tray rim.
6. Subtract the central PCB mask (either supplied by a template, or
   inferred from the dark region inside the cartridge).
7. Rasterise the remaining polygon to an occupancy grid at
   ``mm_per_cell`` resolution, with PCB / margin cells marked
   :class:`CellState.FORBIDDEN`.

* :class:`SegmentationPlacementAreaExtractor` consumes a segmenter's label
  map (``Snapshot.cartridge_masks``) and arbitrates two independent
  placement estimates via :mod:`plan.arbitration` — see that module's
  docstring for why two estimates and not one. This is the path for real
  imagery.

The camera mount is fixed on the cell, so keeping the placement
rectangle axis-aligned costs nothing in accuracy and simplifies the
downstream grid bookkeeping.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    import cv2
except Exception as exc:  # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc

from common.types import BBox
from plan.scene import Cartridge, CellState, OccupancyGrid


# ------------------------------------------------------------- result ---

@dataclass
class PlacementArea:
    """Output of :meth:`PlacementAreaExtractor.extract`."""

    rectangle: BBox           # image-space axis-aligned inset bbox
    inside_mask: np.ndarray   # uint8 0/1 mask inside cartridge ROI
    pcb_mask: np.ndarray      # uint8 0/1 mask of excluded PCB region
    occupancy: OccupancyGrid  # rasterised placement grid
    mm_per_cell: float
    # Segmentation path only; the heuristic path leaves these empty.
    # pcb_mask above is retained and populated with electronics_mask, so
    # plan/planner.py's existing assignment keeps working unchanged.
    bay_mask: Optional[np.ndarray] = None
    electronics_mask: Optional[np.ndarray] = None
    obstruction_mask: Optional[np.ndarray] = None
    consistency_iou: float = 1.0


# --------------------------------------------------- shared rasteriser ----

def _rasterise_mask(
    inside_mask: np.ndarray,
    rect: Tuple[int, int, int, int],
    mm_per_cell: float,
    mm_per_px: float,
) -> OccupancyGrid:
    """Rasterise ``inside_mask`` over ``rect`` into an :class:`OccupancyGrid`.

    Shared by both extractors so their grids are identically shaped by
    construction rather than by coincidence.
    """
    ix1, iy1, ix2, iy2 = rect
    px_per_cell = max(1.0, mm_per_cell / mm_per_px)
    rows = max(1, int((iy2 - iy1) / px_per_cell))
    cols = max(1, int((ix2 - ix1) / px_per_cell))

    grid = OccupancyGrid(rows=rows, cols=cols, resolution_mm=mm_per_cell)
    h_mask, w_mask = inside_mask.shape

    for r in range(rows):
        for c in range(cols):
            ypx = int(iy1 + r * px_per_cell + px_per_cell / 2)
            xpx = int(ix1 + c * px_per_cell + px_per_cell / 2)
            inside_bounds = 0 <= ypx < h_mask and 0 <= xpx < w_mask
            if inside_bounds and inside_mask[ypx, xpx] == 0:
                grid.set(r, c, CellState.FORBIDDEN)
    return grid


# --------------------------------------------------------- extractor ----

class HeuristicPlacementAreaExtractor:
    """Green-channel extractor — tuneables come from ``planning.yaml``.

    SCOPE LIMIT: this assumes a light tray with a dark interior module
    (PPR §5.3.2). Measured on the black cartridges in ``recog/realtest/``,
    it returns zero placeable area for 7 of 20 (spec §1.1). It exists to
    keep the torch-free demo runnable on ``synth_dataset.py``'s flat green
    rectangles, not as an equal alternative to
    :class:`SegmentationPlacementAreaExtractor` for real imagery.
    """

    def __init__(
        self,
        safety_margin_px: int = 5,
        morph_close_ksize: int = 5,
        morph_open_ksize: int = 3,
        mm_per_cell: float = 1.5,
        mm_per_px: float = 0.38,
    ) -> None:
        warnings.warn(
            "HeuristicPlacementAreaExtractor assumes a LIGHT tray with a "
            "DARK interior module (PPR 5.3.2). On the black cartridges in "
            "recog/realtest/ it returns zero placeable area for 7 of 20 "
            "cartridges, because Otsu on the green channel selects "
            "whichever foreign matter is brightest and the gray<80 "
            "threshold classifies the black bay as PCB. Use "
            "SegmentationPlacementAreaExtractor for real imagery; this "
            "path exists for the torch-free demo on synth_dataset.py's "
            "green rectangles.",
            RuntimeWarning, stacklevel=2,
        )
        self.safety_margin_px = int(safety_margin_px)
        self.k_close = int(morph_close_ksize)
        self.k_open = int(morph_open_ksize)
        self.mm_per_cell = float(mm_per_cell)
        self.mm_per_px = float(mm_per_px)

    # ---- public entrypoint ---------------------------------------------

    def extract(
        self,
        image_rgb: np.ndarray,
        cartridge_bbox: BBox,
        pcb_template_mask: Optional[np.ndarray] = None,
    ) -> PlacementArea:
        height, width = image_rgb.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(cartridge_bbox.xmin)),
            max(0, int(cartridge_bbox.ymin)),
            min(width, int(cartridge_bbox.xmax)),
            min(height, int(cartridge_bbox.ymax)),
        )
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Empty cartridge bbox")

        roi = image_rgb[y1:y2, x1:x2, :]

        mask = self._green_mask(roi)
        rect = self._largest_contour_rect(mask)
        inset_rect = self._inset(rect)

        inside = self._rect_mask(mask.shape, inset_rect)
        pcb = (
            pcb_template_mask.astype(np.uint8)
            if pcb_template_mask is not None
            else self._infer_pcb_mask(roi, inset_rect)
        )
        inside = cv2.bitwise_and(
            inside, cv2.bitwise_not(pcb.astype(np.uint8)),
        )

        occupancy = self._rasterise(inside, inset_rect)

        ix1, iy1, ix2, iy2 = inset_rect
        return PlacementArea(
            rectangle=BBox(x1 + ix1, y1 + iy1, x1 + ix2, y1 + iy2),
            inside_mask=inside,
            pcb_mask=pcb,
            occupancy=occupancy,
            mm_per_cell=self.mm_per_cell,
        )

    # ---- pipeline stages -----------------------------------------------

    def _green_mask(self, roi: np.ndarray) -> np.ndarray:
        """Otsu threshold on the green channel, plus close→open cleanup."""
        green = roi[:, :, 1]
        _, mask = cv2.threshold(
            green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        close_k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.k_close, self.k_close),
        )
        open_k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.k_open, self.k_open),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        return mask

    def _largest_contour_rect(
        self, mask: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Bounding rect of the largest contour in ``mask``."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            raise RuntimeError("No green region found inside cartridge ROI")
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return x, y, x + w, y + h

    def _inset(
        self, rect: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        m = self.safety_margin_px
        ix1, iy1, ix2, iy2 = rect[0] + m, rect[1] + m, rect[2] - m, rect[3] - m
        if ix2 - ix1 <= 10 or iy2 - iy1 <= 10:
            raise RuntimeError("Placement region too small after inset")
        return ix1, iy1, ix2, iy2

    def _rect_mask(
        self,
        shape: Tuple[int, int],
        rect: Tuple[int, int, int, int],
    ) -> np.ndarray:
        ix1, iy1, ix2, iy2 = rect
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(mask, (ix1, iy1), (ix2, iy2), 1, thickness=-1)
        return mask

    # ---- PCB inference --------------------------------------------------

    def _infer_pcb_mask(
        self,
        roi: np.ndarray,
        rect: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Heuristic PCB mask: the dark region inside the inset rectangle.

        A fixed dark threshold is stable because the cartridge's green
        tray colour is locally normalised — the PCB is markedly darker
        in every scene the recogniser will see.
        """
        ix1, iy1, ix2, iy2 = rect
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        pcb = np.zeros_like(gray, dtype=np.uint8)
        centre = gray[iy1:iy2, ix1:ix2]
        dark = (centre < 80).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        )

        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(
                pcb, (ix1 + x, iy1 + y), (ix1 + x + w, iy1 + y + h),
                1, thickness=-1,
            )
        return pcb

    # ---- rasterisation --------------------------------------------------

    def _rasterise(
        self,
        inside_mask: np.ndarray,
        rect: Tuple[int, int, int, int],
    ) -> OccupancyGrid:
        return _rasterise_mask(
            inside_mask, rect, self.mm_per_cell, self.mm_per_px)


# Backward-compatible alias: main.py:124 and plan/planner.py import this
# name. Kept so neither needs to change to keep working.
PlacementAreaExtractor = HeuristicPlacementAreaExtractor


# ---------------------------------------------------- segmentation path ----

class PlacementDisagreement(RuntimeError):
    """The two placement estimates disagreed beyond tau.

    plan/planner.py's _ensure_placement_areas already catches every
    exception and leaves the cartridge unplanned for the cycle, so
    raising is exactly the behaviour wanted: the cartridge is skipped
    and retried next frame.
    """


class BadDetectorBox(PlacementDisagreement):
    """The detector's cartridge box looks misaligned, not merely low-IoU.

    Cartridge material IS visible somewhere in the crop, but not at the
    crop's centre — exactly the condition
    ``plan.arbitration.centre_component`` already detects and logs a
    warning for. Left to ``arbitrate()`` alone this is indistinguishable
    from a genuinely full cartridge: both reduce to an empty ``P_safe``
    and IoU 0.0, because ``mask_iou`` reports "both estimates empty" as
    maximally disagreeing (0.0) rather than as the trivial agreement it
    actually is. Checked explicitly, ahead of the tau gate, so the two
    do not silently collapse into the same signal.

    Subclasses :class:`PlacementDisagreement` on purpose: a plain
    ``except PlacementDisagreement`` still catches it (skip-and-retry is
    still correct here), but code that wants to know *why* — a
    perception failure, not a low-confidence-but-real disagreement —
    can catch this more specific type. A full cartridge is a normal,
    expected state; a detector that is systematically handing back bad
    boxes is not, and needs to be visible as itself rather than read as
    "every cartridge is full".
    """


# catalog.json (recog/synth3d/assets/catalog.json) carries a measured
# `case_wall_mm` per asset from CAD conversion: 4.0, 3.75, 3.7, 4.25 mm
# across the four cataloged cartridges. No asset/SKU identifier crosses
# the Recognition -> Planning boundary (Detection/BBox/Snapshot carry
# none), so there is no way to look up the *specific* cartridge's wall
# thickness at inference time - a single scalar has to stand in for all
# of them. The MAX of the four measured values is used, deliberately:
# eroding a bit further than necessary only costs a sliver of floor
# space at the cartridge's edge, while eroding too little would let
# actual wall material be reported as safe to place a cell against -
# the same "skip costs a cycle, misplacement costs a cell" asymmetry
# plan.arbitration is built on. A deployment that knows its cartridge
# SKU should pass its measured case_wall_mm explicitly instead of
# relying on this default.
_DEFAULT_WALL_INSET_MM = 4.25


class SegmentationPlacementAreaExtractor:
    """Placement area from a segmenter's label map.

    Does mask arithmetic only. The model runs in Recognition and its
    output arrives on Snapshot.cartridge_masks - FDR O3 caps planning at
    8 ms per cartridge and a DeepLabv3 forward is ~12.6 ms.
    """

    def __init__(self, mm_per_cell: float = 1.5, mm_per_px: float = 0.625,
                 wall_inset_mm: float = _DEFAULT_WALL_INSET_MM,
                 tau: float = 0.85) -> None:
        self.mm_per_cell = float(mm_per_cell)
        self.mm_per_px = float(mm_per_px)
        self.wall_inset_mm = float(wall_inset_mm)
        self.tau = float(tau)

    @property
    def wall_inset_px(self) -> int:
        """Wall thickness in pixels at the CURRENT framing.

        Not the old hardcoded safety_margin_px = 5: a fixed pixel inset
        is wrong at two different camera framings. mm_per_px must be the
        calibrated scale for the deployed mount, not a config default.
        """
        return max(0, int(round(self.wall_inset_mm / self.mm_per_px)))

    def extract(self, image_rgb: np.ndarray, cartridge_bbox: BBox,
                pcb_template_mask: Optional[np.ndarray] = None,
                label_map: Optional[np.ndarray] = None) -> PlacementArea:
        from plan.arbitration import (CH_BACKGROUND, CH_ELECTRONICS,
                                      CH_OBSTRUCTION, arbitrate,
                                      centre_component, derived_placement,
                                      direct_placement)

        if label_map is None:
            raise ValueError(
                "SegmentationPlacementAreaExtractor needs a label_map "
                "from Snapshot.cartridge_masks. Falling back to the "
                "heuristic silently would hide a perception failure "
                "behind a plausible-looking rectangle.")

        # Bad-box gate: cartridge material is visible SOMEWHERE in the
        # crop, but not at its centre. See BadDetectorBox's docstring
        # for why this has to be checked here, ahead of the tau gate,
        # rather than left for arbitrate() to fold into the same
        # (empty P_safe, iou 0.0) a full cartridge also produces.
        has_foreground = bool((label_map != CH_BACKGROUND).any())
        if has_foreground and not centre_component(label_map).any():
            raise BadDetectorBox(
                "crop centre lands on background, but the crop is not "
                "empty - the detector box for this cartridge is likely "
                "misaligned (see plan.arbitration.centre_component's "
                "warning log for coordinates). A PERCEPTION failure, "
                "not an empty cartridge.")

        safe, iou = arbitrate(label_map, self.wall_inset_px)

        if iou < self.tau:
            direct = direct_placement(label_map)
            derived = derived_placement(label_map, self.wall_inset_px)
            if not direct.any() and not derived.any():
                # Both independent estimates agree there is nothing
                # here. mask_iou's union == 0 rule reports that as 0.0,
                # same as a real disagreement, but two empty sets are
                # not a disagreement - this is a normal, expected "the
                # cartridge is full" state and must not be counted as
                # one.
                raise RuntimeError(
                    "no placeable area in this cartridge: both "
                    "placement estimates are empty (the cartridge may "
                    "simply be full)")
            raise PlacementDisagreement(
                f"placement estimates disagree: IoU {iou:.3f} < "
                f"tau {self.tau:.3f}")

        ys, xs = np.nonzero(safe)
        if xs.size == 0:
            raise RuntimeError("no placeable area in this cartridge")

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

        inside = safe.astype(np.uint8)
        electronics = (label_map == CH_ELECTRONICS).astype(np.uint8)
        obstruction = (label_map == CH_OBSTRUCTION).astype(np.uint8)

        occupancy = _rasterise_mask(
            inside, (x0, y0, x1, y1), self.mm_per_cell, self.mm_per_px)

        ox, oy = int(cartridge_bbox.xmin), int(cartridge_bbox.ymin)
        return PlacementArea(
            rectangle=BBox(ox + x0, oy + y0, ox + x1, oy + y1),
            inside_mask=inside,
            pcb_mask=electronics,          # backward compatibility
            occupancy=occupancy,
            mm_per_cell=self.mm_per_cell,
            bay_mask=direct_placement(label_map).astype(np.uint8),
            electronics_mask=electronics,
            obstruction_mask=obstruction,
            consistency_iou=iou,
        )


# ----------------------------------------------- convenience helper ----

def attach_placement_area(
    cartridge: Cartridge,
    extractor: PlacementAreaExtractor,
    image_rgb: np.ndarray,
    pcb_template_mask: Optional[np.ndarray] = None,
) -> Cartridge:
    """Extract the placement area for ``cartridge`` and attach it."""
    pa = extractor.extract(image_rgb, cartridge.bbox, pcb_template_mask)
    cartridge.placeable_rectangle = pa.rectangle
    cartridge.occupancy = pa.occupancy
    cartridge.pcb_mask = pa.pcb_mask
    return cartridge


__all__ = [
    "PlacementArea",
    "PlacementAreaExtractor",
    "HeuristicPlacementAreaExtractor",
    "SegmentationPlacementAreaExtractor",
    "PlacementDisagreement",
    "BadDetectorBox",
    "attach_placement_area",
]
