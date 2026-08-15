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

from common.types import BBox, DEFAULT_WALL_INSET_MM, UnknownScale
from plan.scene import CellState, OccupancyGrid

# `UnknownScale` and the wall inset MOVED to common.types on 2026-08-15
# and are re-exported here, not re-implemented. They are contracts two
# packages share - recog.seg_evaluate raises the exception and
# recog.calibrate_tau / recog.seg_ablation score against the inset - and
# holding them here meant `recog` imported `plan` at module load. See
# common/types.py for the full note. Every existing
# `from plan.placement_area import UnknownScale` keeps working and keeps
# naming the SAME class, which is the part that matters: the planner
# catches it to turn an uncalibrated frame into a counted skip.
#
# The private alias is kept because scripts/placement_feasibility.py - a
# frozen receipt-generating script - imports `_DEFAULT_WALL_INSET_MM`
# from this module by that name.
_DEFAULT_WALL_INSET_MM = DEFAULT_WALL_INSET_MM


# ------------------------------------------------------------- result ---

@dataclass
class PlacementArea:
    """Output of :meth:`PlacementAreaExtractor.extract`."""

    rectangle: BBox           # image-space axis-aligned inset bbox
    inside_mask: np.ndarray   # uint8 0/1 mask inside cartridge ROI
    pcb_mask: np.ndarray      # uint8 0/1 mask of excluded PCB region
    occupancy: OccupancyGrid  # rasterised placement grid
    mm_per_cell: float
    # The scale this area was MEASURED at, millimetres per pixel.
    # `rectangle` is in pixels and `occupancy` was rasterised against it,
    # so neither means anything without this number. Returned rather than
    # left to the caller to remember, because the caller (a persistent
    # digital twin) may hold the rectangle across frames whose scales
    # differ - and pairing one frame's rectangle with another frame's
    # scale is exactly the class of error this field exists to make
    # impossible to commit silently.
    mm_per_px: float
    # Segmentation path only; the heuristic path leaves these empty.
    # pcb_mask above is retained and populated with electronics_mask, so
    # plan/planner.py's existing assignment keeps working unchanged.
    bay_mask: Optional[np.ndarray] = None
    electronics_mask: Optional[np.ndarray] = None
    obstruction_mask: Optional[np.ndarray] = None
    consistency_iou: float = 1.0


# ------------------------------------------------------------- scale ----

def _resolve_scale(per_frame: Optional[float], fallback: Optional[float],
                   who: str) -> float:
    """The scale to measure at: the frame's own, else the fallback.

    The only place that precedence is decided, so the extractor and the
    planner cannot disagree about which number won.
    """
    scale = per_frame if per_frame is not None else fallback
    if scale is None:
        raise UnknownScale(
            f"{who} has no mm_per_px: this frame carries no calibration "
            f"and none was configured as a fallback. Set "
            f"planning.camera.mm_per_px_x (or mode.mm_per_px) for a "
            f"fixed-mount camera, or supply Snapshot.mm_per_px per "
            f"frame. Refusing to guess - every distance downstream is "
            f"in millimetres derived from this number.")
    scale = float(scale)
    if scale <= 0.0:
        raise UnknownScale(
            f"{who} got mm_per_px={scale!r}: a pixel cannot span zero or "
            f"negative millimetres.")
    return scale


# --------------------------------------------------- shared rasteriser ----

def _rasterise_mask(
    inside_mask: np.ndarray,
    rect: Tuple[int, int, int, int],
    mm_per_cell: float,
    mm_per_px: float,
) -> OccupancyGrid:
    """Rasterise ``inside_mask`` over ``rect`` into an :class:`OccupancyGrid`.

    Shared by both extractors so their grids are identically shaped by
    construction rather than by coincidence, and by
    ``recog.seg_ablation._pack_count`` so its Delta-cells metric
    quantises exactly the way production does.

    A cell is FREE only if EVERY pixel it covers is free. It used to be
    free if its CENTRE pixel was, which reported a cell as placeable when
    up to half of it was wall: at ``mm_per_cell = 1.5`` that dilated the
    free region by up to 0.75 mm per edge, in the unsafe direction, and
    docs/superpowers/specs/2026-08-11-placement-feasibility.md section 5
    finding 3 recorded that it was *producing* placements. Measured over
    the 30-cartridge corpus (2026-08-11-placement-safety.md): the same 26
    cells are still placed, in the same 13 cartridges, but two of them
    stop landing on ground-truth tray wall - the packer simply seats them
    a pixel or two further inside. Optimism here is not worth anything;
    it was only ever moving cells outward.

    A cell that runs off the edge of ``inside_mask`` is FORBIDDEN for the
    same reason: pixels nobody measured are not known to be free. In tree
    this cannot fire - both callers derive ``rect`` from the mask itself -
    but "unmeasured" defaulting to "safe" is how the class of defect this
    change repairs gets in.

    ``resolution_mm`` is the size of a cell AFTER the one-pixel clamp,
    not the requested ``mm_per_cell``. A cell can never be finer than a
    pixel, so on a frame coarser than ``mm_per_cell`` the clamp binds and
    a cell IS one pixel - and the grid used to keep reporting
    ``mm_per_cell`` regardless. Measured at 2.0 mm/px with 1.5 mm cells:
    a 40 px crop became 40 cells claiming 60 mm of an 80 mm crop, the
    forbidden mask compressed 25 % toward the origin with the far 20 mm
    never measured. The corpus tops out near 1.4 mm/px so nothing in tree
    reaches the clamp - but ``OccupancyGrid.resolution_mm`` is read as
    the true millimetres of a cell by ``_overlaps_forbidden``, by
    ``Planner._cell_block_for`` and (since 2026-08-15) by
    ``Planner._pack_cartridge`` to size the packing strip, so a grid that
    lies about its own extent would make all three wrong together.

    Written as a branch rather than ``px_per_cell * mm_per_px`` because
    the two differ in the last bit: at 1.4 mm/px that product is
    1.4999999999999998, and the resolution is compared for equality and
    multiplied by the row count to give the strip extent.
    """
    ix1, iy1, ix2, iy2 = rect
    if mm_per_cell >= mm_per_px:
        px_per_cell = mm_per_cell / mm_per_px
        resolution_mm = mm_per_cell
    else:
        # The clamp binds: one cell, one pixel, and a pixel spans
        # mm_per_px millimetres.
        px_per_cell = 1.0
        resolution_mm = mm_per_px
    rows = max(1, int((iy2 - iy1) / px_per_cell))
    cols = max(1, int((ix2 - ix1) / px_per_cell))

    grid = OccupancyGrid(rows=rows, cols=cols, resolution_mm=resolution_mm)
    h_mask, w_mask = inside_mask.shape

    # Cell (r, c) covers pixels [y0, y1) x [x0, x1). At least one pixel
    # per cell even where px_per_cell puts two cell boundaries on the
    # same pixel.
    r_i, c_i = np.arange(rows), np.arange(cols)
    y0 = (iy1 + r_i * px_per_cell).astype(np.int64)
    x0 = (ix1 + c_i * px_per_cell).astype(np.int64)
    y1 = np.maximum(y0 + 1, (iy1 + (r_i + 1) * px_per_cell).astype(np.int64))
    x1 = np.maximum(x0 + 1, (ix1 + (c_i + 1) * px_per_cell).astype(np.int64))

    # Summed-area table, so the whole-cell test costs four lookups per
    # cell rather than one per pixel. The per-pixel version of exactly
    # this arithmetic cost 15.1 ms on a 288x131 crop against FDR O3's
    # 8 ms per-cartridge planning budget (tests/test_planner.py
    # ::test_segmentation_extract_arithmetic_stays_under_the_o3_budget).
    integral = cv2.integral((inside_mask != 0).astype(np.uint8),
                            sdepth=cv2.CV_32S)
    y0c, y1c = np.clip(y0, 0, h_mask), np.clip(y1, 0, h_mask)
    x0c, x1c = np.clip(x0, 0, w_mask), np.clip(x1, 0, w_mask)
    free_px = (integral[np.ix_(y1c, x1c)] - integral[np.ix_(y0c, x1c)]
               - integral[np.ix_(y1c, x0c)] + integral[np.ix_(y0c, x0c)])
    covered = ((y0 >= 0) & (y1 <= h_mask))[:, None] \
        & ((x0 >= 0) & (x1 <= w_mask))[None, :]
    whole_cell = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    grid.grid[~(covered & (free_px == whole_cell))] = \
        CellState.FORBIDDEN.value
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
        mm_per_px: Optional[float] = None,
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
        # FALLBACK scale, used only for frames that carry none of their
        # own. None means there is no fallback and an uncalibrated frame
        # raises. See UnknownScale.
        self.mm_per_px = None if mm_per_px is None else float(mm_per_px)

    # ---- public entrypoint ---------------------------------------------

    def extract(
        self,
        image_rgb: np.ndarray,
        cartridge_bbox: BBox,
        pcb_template_mask: Optional[np.ndarray] = None,
        mm_per_px: Optional[float] = None,
    ) -> PlacementArea:
        """``mm_per_px`` is THIS frame's scale and wins over the fallback.

        Declared even though this extractor's own corpus
        (``recog/synth_dataset.py``'s flat green rectangles) carries no
        per-frame calibration: ``plan.planner`` passes the resolved scale
        to whichever extractor it holds, and a signature that silently
        refused the kwarg would put the two extractors on different
        contracts again - the condition ``_accepts_label_map`` exists to
        work around, which nobody wants a second instance of.
        """
        scale = _resolve_scale(mm_per_px, self.mm_per_px,
                               type(self).__name__)
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

        occupancy = _rasterise_mask(
            inside, inset_rect, self.mm_per_cell, scale)

        ix1, iy1, ix2, iy2 = inset_rect
        return PlacementArea(
            rectangle=BBox(x1 + ix1, y1 + iy1, x1 + ix2, y1 + iy2),
            inside_mask=inside,
            pcb_mask=pcb,
            occupancy=occupancy,
            mm_per_cell=self.mm_per_cell,
            mm_per_px=scale,
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


# Backward-compatible alias: main.py:124 and plan/planner.py import this
# name. Kept so neither needs to change to keep working.
PlacementAreaExtractor = HeuristicPlacementAreaExtractor


# ---------------------------------------------------- segmentation path ----

class PlacementDisagreement(RuntimeError):
    """The two placement estimates disagree about a REAL cartridge.

    plan/planner.py's _ensure_placement_areas already catches every
    exception and leaves the cartridge unplanned for the cycle, so
    raising is exactly the behaviour wanted: the cartridge is skipped
    and retried next frame.

    NOTHING IN THIS REPOSITORY RAISES THIS CLASS BARE any more. It used
    to mean "iou < tau"; that gate was measured to be inert and deleted
    (FDR v3 section 13.2.1), and an empty ``P_safe`` is deliberately NOT
    re-labelled as a disagreement - on 15 dataset3d_seg frames 18 of 26
    cartridges have no predicted `bay` at all, and calling that a
    disagreement would bury the real signal under the ordinary "this
    cartridge has no visible placement area" case.

    The class is kept, rather than deleted as vestigial API, because it
    is still READ in three places: :class:`BadDetectorBox` derives from
    it (so ``except PlacementDisagreement`` catches a bad box),
    ``plan/planner.py`` catches it and counts it, and it is exported.
    It is the family name for "perception says this cartridge is not
    safe to plan, skip and retry" - a subclass, or an out-of-tree
    extractor, raises it; the base class itself is now only a type.
    """


class BadDetectorBox(PlacementDisagreement):
    """The detector's cartridge box does not describe one cartridge.

    Two conditions raise it, and they are complementary — one is about
    WHERE the crop landed, the other about WHAT it contains:

    1. **The crop centre is on background** while the crop is not empty.
    2. **The placeable floor is bigger than the largest cartridge**, at
       the frame's own scale — see
       :meth:`SegmentationPlacementAreaExtractor
       .reject_if_not_one_cartridge_floor`. A box spanning a cartridge
       *and* the loose cells beside it passes (1) — its centre is on real
       foreground — and the planner then commands a cell onto bare
       backdrop. (1) alone was measured to miss exactly that, once in 30
       cartridges, at 100 % overlap with ground-truth non-floor.

    On (1): cartridge material IS visible somewhere in the crop, but not
    at the crop's centre — exactly the condition
    ``plan.arbitration.centre_component`` already detects and logs a
    warning for. Left to ``arbitrate()`` alone this is indistinguishable
    from a genuinely full cartridge: both reduce to an empty ``P_safe``
    and IoU 0.0, because ``mask_iou`` reports "both estimates empty" as
    maximally disagreeing (0.0) rather than as the trivial agreement it
    actually is. Checked explicitly, before ``arbitrate`` is called at
    all, so the two do not silently collapse into the same signal.

    Subclasses :class:`PlacementDisagreement` on purpose: a plain
    ``except PlacementDisagreement`` still catches it (skip-and-retry is
    still correct here), but code that wants to know *why* — a
    perception failure, not a low-confidence-but-real disagreement —
    can catch this more specific type. A full cartridge is a normal,
    expected state; a detector that is systematically handing back bad
    boxes is not, and needs to be visible as itself rather than read as
    "every cartridge is full".
    """


# DEFAULT_WALL_INSET_MM (4.25 mm, the MAX of the four CAD-measured
# `case_wall_mm` in recog/synth3d/assets/catalog.json) now lives in
# common.types, imported at the top of this file. It is read by
# recog.calibrate_tau and recog.seg_ablation as well as by the extractor
# below, and a constant three packages share belongs in the package all
# three already depend on. Its full provenance note travelled with it.
#
# _MAX_CARTRIDGE_EXTENT_MM below did NOT travel, and the asymmetry is
# deliberate: nothing outside this module reads it, and it is a bound
# this extractor enforces rather than a vocabulary anyone shares.


# The OUTER footprint of the largest cartridge in
# recog/synth3d/assets/catalog.json. The four cataloged `extents_mm` are
# 62.9x90.9, 80.7x97.0, 62.3x167.8 and 81.7x180.0 mm, and this is their
# per-axis MAX - the same rule, for the same reason, as
# common.types.DEFAULT_WALL_INSET_MM: no asset/SKU identifier crosses the
# Recognition -> Planning boundary, so one scalar pair has to stand in
# for every cartridge, and it has to be the one that cannot refuse a real
# one.
#
# The OUTER footprint, deliberately, and not the interior (73.2x171.5) or
# the placement region (73.2x140.8), even though P_safe is a subset of
# both. Those tighter bounds are within reach of ordinary segmentation
# slop: over the 30-cartridge corpus the predicted P_safe overran the
# 140.8 mm placement-region length on SEVEN instances (worst 147.4 mm),
# six of them productive and holding 19 of the 25 cells that ship, so a
# bound there would trade real placements for nothing. Against the outer
# footprint those 29 good instances clear it by 8 % or more on the short
# axis and 18 % on the long, and the one bad crop misses by 33 %. A
# bound that only fires on the physically
# impossible needs no tolerance term, and a tolerance term is a knob
# somebody would later tune to make a number move.
#
# THIS CONSTANT ALSO BOUNDS PACKING LATENCY, which is an accident and is
# recorded here because nothing else would say so. `common.packing.
# pack_best_effort` costs grow with the occupancy grid's cell count and
# with the item count, and `plan/planner.py` ties the two together
# (`n_est = 2*area/(18.5*65)`). Bisected against FDR requirement O3's
# 8 ms per-cartridge budget (audit K, 2026-08-12, worst over wall,
# 15 %-blob and checkerboard masks): a floor at this bound - 81.7 x 180.0
# mm, 6 480 cells, n_est 24 - packs in 2.04 ms; 140 x 278 mm takes
# 5.70 ms; 158 x 314 mm takes 8.16 ms and BREACHES. The packer never sees
# such an input only because `reject_if_not_one_cartridge_floor` refuses
# it here, BEFORE the occupancy grid is built. O3's 3.9x margin is
# therefore held up by a bound written for an unrelated correctness
# reason, and the breach point is only ~1.94x this bound's long axis and
# ~1.93x its short axis. Raising this for a larger SKU raises the packing
# cost with it: re-measure `pack_best_effort` against the 8 ms budget
# before doing so, and see that function's own docstring.
_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)


def placeable_extent_mm(mask: np.ndarray, mm_per_px: float
                        ) -> Tuple[float, float]:
    """``(short_side, long_side)`` of ``mask``'s min-area rect, in mm.

    ROTATED rather than axis-aligned: the cartridges sit at
    ``quarter * 90 + jitter`` and a bounding box would grow with the
    jitter angle, which has nothing to do with how big the cartridge is.

    Raises on an empty mask rather than returning ``(0, 0)``. A zero
    extent satisfies every upper bound, so a caller using this to decide
    whether something is too big would silently stop deciding.
    """
    pts = cv2.findNonZero(np.asarray(mask, dtype=np.uint8))
    if pts is None:
        raise ValueError(
            "placeable_extent_mm: the mask is empty, so it has no extent. "
            "An empty region is a different question ('is there anywhere "
            "to place?') and has a different answer; it must not be "
            "reported as a region of size zero.")
    (_centre, (w_px, h_px), _angle) = cv2.minAreaRect(pts)
    # minAreaRect measures between pixel CENTRES, so a run of n pixels
    # comes back as n-1 and a single pixel as 0. Add the pixel back.
    short_px, long_px = sorted((float(w_px) + 1.0, float(h_px) + 1.0))
    return short_px * float(mm_per_px), long_px * float(mm_per_px)


class SegmentationPlacementAreaExtractor:
    """Placement area from a segmenter's label map.

    Does mask arithmetic only. The model runs in Recognition and its
    output arrives on Snapshot.cartridge_masks - FDR O3 caps planning at
    8 ms per cartridge and a DeepLabv3 forward is ~12.6 ms.

    NO ``tau``. ``P_safe = P_direct & P_derived`` is applied
    unconditionally; the IoU between the two estimates is reported on
    ``PlacementArea.consistency_iou`` as observability and is not acted
    on. The gate that used to sit here was measured and retired (FDR v3
    section 13.2.1, docs/receipts/tau_independence_correlation.txt): the
    IoU correlates POSITIVELY with placement error in all four cataloged
    SKUs, raw and area-normalised, which is the wrong sign for a
    confidence gate. Measured cost of leaving it in, on 15
    recog/dataset3d_seg frames through the real detector and segmenter:
    26 cartridge crops, 8 with a predicted bay, of which the gate
    admitted 3 at tau=0.85/mm_per_px=0.625 and 0 at planning.yaml's own
    mm_per_px=0.38, against 8 with the gate gone at either scale
    (docs/superpowers/specs/2026-08-11-segmenter-integration.md). The
    ``tau`` constructor argument is DELETED rather than accepted and
    ignored, so a caller still passing it fails loudly.
    """

    def __init__(self, mm_per_cell: float = 1.5,
                 mm_per_px: Optional[float] = None,
                 wall_inset_mm: float = DEFAULT_WALL_INSET_MM,
                 max_cartridge_extent_mm: Tuple[float, float]
                 = _MAX_CARTRIDGE_EXTENT_MM) -> None:
        self.mm_per_cell = float(mm_per_cell)
        # FALLBACK scale only. It used to default to 0.625 - "this
        # dataset's actual framing" - and that default was the defect:
        # no frame in the corpus it named is rendered at that framing,
        # because recog/synth3d/world.py randomises margin and zoom per
        # scene. The default is now None, so a caller that forgets to
        # say how big a pixel is gets UnknownScale instead of a plausible
        # wrong answer.
        self.mm_per_px = None if mm_per_px is None else float(mm_per_px)
        self.wall_inset_mm = float(wall_inset_mm)
        # Validated here, not read on trust at the point of use. This
        # bound is a safety interlock, and the way an interlock dies in
        # this project is by being handed a value that can never fire
        # (None, inf, a swapped pair) while the code around it keeps
        # looking like it checks something.
        try:
            short_mm, long_mm = (float(v) for v in max_cartridge_extent_mm)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"max_cartridge_extent_mm must be a (short_mm, long_mm) "
                f"pair, got {max_cartridge_extent_mm!r}. Pass the outer "
                f"footprint of the largest cartridge this cell handles; "
                f"there is no 'unbounded' setting, because an unbounded "
                f"bound is a check that has quietly stopped checking."
            ) from exc
        if not (0.0 < short_mm <= long_mm) or long_mm == float("inf"):
            raise ValueError(
                f"max_cartridge_extent_mm must be finite and positive "
                f"with the short side first, got "
                f"({short_mm!r}, {long_mm!r}).")
        self.max_cartridge_extent_mm = (short_mm, long_mm)

    # ---- box validity ---------------------------------------------------

    def reject_if_not_one_cartridge_floor(
        self, safe: np.ndarray, mm_per_px: float,
    ) -> Tuple[float, float]:
        """Raise :class:`BadDetectorBox` if ``safe`` is too big to be one
        cartridge's floor. Returns the measured extent otherwise.

        The second half of the bad-box check, and the half the first one
        cannot see. ``centre_component``'s test is about WHERE the crop
        landed - is the crop centre on foreground - and it passes happily
        on a box that landed on a real cartridge *and* three loose cells
        beside it, because the centre is still foreground. This one is
        about WHAT the crop turned out to contain: a placeable floor
        larger than the largest cartridge that exists is not a cartridge
        floor, whatever the crop centre is standing on.

        Measured case, ``scene_00014/c25``: the detector returned one box
        spanning a cartridge and three loose cells (190 x 142 px at
        0.998 mm/px); the segmenter read the jig backdrop inside that box
        as a cartridge and its bare panel as `bay`; the planner commanded
        a cell 100 % onto ground-truth background. Its P_safe measures
        108.8 x 148.1 mm - it does not fit inside any cartridge in the
        catalog on its short axis, by 33 %. The 29 other cartridges in
        that corpus measure at most 75.4 mm across (and 147.4 mm long),
        and are unaffected.
        docs/superpowers/specs/2026-08-11-placement-safety.md.

        Needs the frame's scale, which is why it could not have been
        written before ``mm_per_px`` became a property of the frame
        (`380e7d5`): at the old fixed 0.625 this same P_safe measured
        67.5 mm on its short axis and would have passed.
        """
        short_mm, long_mm = placeable_extent_mm(safe, mm_per_px)
        max_short, max_long = self.max_cartridge_extent_mm
        if short_mm <= 0.0 or long_mm <= 0.0:
            raise RuntimeError(
                f"placeable region measured {short_mm} x {long_mm} mm from "
                f"{int(np.asarray(safe).sum())} non-zero pixels at "
                f"{mm_per_px} mm/px. A non-empty region cannot have zero "
                f"extent, and a zero extent passes every bound - this "
                f"check would be reporting 'fine' without measuring "
                f"anything.")
        if short_mm > max_short or long_mm > max_long:
            raise BadDetectorBox(
                f"the placeable floor in this crop measures "
                f"{short_mm:.1f} x {long_mm:.1f} mm at {mm_per_px:.4f} "
                f"mm/px, which does not fit inside the largest cartridge "
                f"this cell handles ({max_short:.1f} x {max_long:.1f} mm "
                f"outer footprint). One cartridge cannot have a floor "
                f"bigger than the whole cartridge, so this crop is not "
                f"one cartridge - most likely a detector box spanning a "
                f"cartridge and whatever is lying next to it. A "
                f"PERCEPTION failure, not an empty cartridge.")
        return short_mm, long_mm

    def wall_inset_px_at(self, mm_per_px: float) -> int:
        """Wall thickness in pixels at ``mm_per_px``.

        Not the old hardcoded safety_margin_px = 5: a fixed pixel inset
        is wrong at two different camera framings. It was a `wall_inset_px`
        PROPERTY reading a constructor constant until 2026-08-11, which
        was the same mistake one level down - at the corpus's true scales
        this eroded 6.98 mm of a 4.25 mm wall on the coarsest frames and
        3.34 mm on the finest. Takes the frame's scale as an argument so
        there is no stored answer to go stale.
        """
        scale = _resolve_scale(mm_per_px, self.mm_per_px,
                               type(self).__name__)
        return max(0, int(round(self.wall_inset_mm / scale)))

    def extract(self, image_rgb: np.ndarray, cartridge_bbox: BBox,
                pcb_template_mask: Optional[np.ndarray] = None,
                label_map: Optional[np.ndarray] = None,
                mm_per_px: Optional[float] = None) -> PlacementArea:
        """``mm_per_px`` is THIS frame's scale and wins over the fallback.

        It sets both the wall erosion radius and the occupancy grid's
        pixels-per-cell, so a frame measured at the wrong scale is wrong
        twice over and in the same direction.
        """
        from plan.arbitration import (CH_BACKGROUND, CH_ELECTRONICS,
                                      CH_OBSTRUCTION, arbitrate,
                                      centre_component, derived_placement,
                                      direct_placement)

        scale = _resolve_scale(mm_per_px, self.mm_per_px,
                               type(self).__name__)
        inset_px = self.wall_inset_px_at(scale)

        if label_map is None:
            raise ValueError(
                "SegmentationPlacementAreaExtractor needs a label_map "
                "from Snapshot.cartridge_masks. Falling back to the "
                "heuristic silently would hide a perception failure "
                "behind a plausible-looking rectangle.")

        # Bad-box gate: cartridge material is visible SOMEWHERE in the
        # crop, but not at its centre. See BadDetectorBox's docstring
        # for why this has to be checked here, ahead of arbitrate(),
        # rather than left for arbitrate() to fold into the same
        # (empty P_safe, iou 0.0) a full cartridge also produces. This
        # is the one perception check that survives; it is a statement
        # about WHERE the crop landed, not about how well two estimates
        # of the same argmax happen to agree.
        has_foreground = bool((label_map != CH_BACKGROUND).any())
        if has_foreground and not centre_component(label_map).any():
            raise BadDetectorBox(
                "crop centre lands on background, but the crop is not "
                "empty - the detector box for this cartridge is likely "
                "misaligned (see plan.arbitration.centre_component's "
                "warning log for coordinates). A PERCEPTION failure, "
                "not an empty cartridge.")

        safe, iou = arbitrate(label_map, inset_px)

        ys, xs = np.nonzero(safe)
        if xs.size == 0:
            # P_safe is empty, so there is nothing to plan on either
            # way - but the two sub-cases are still worth telling apart
            # in the message, which is the part of the old tau branch
            # that was never about tau. mask_iou's union == 0 rule
            # reports "both estimates empty" as 0.0, the same number a
            # real disagreement produces, so the IoU cannot make this
            # distinction and the masks have to be re-read.
            #
            # NEITHER sub-case raises PlacementDisagreement. "No bay
            # predicted here" is the ordinary majority case - 18 of 26
            # cartridges over 15 dataset3d_seg frames - and routing it
            # to the disagreement counter would drown the interlock it
            # is supposed to be.
            direct = direct_placement(label_map)
            derived = derived_placement(label_map, inset_px)
            if not direct.any() and not derived.any():
                raise RuntimeError(
                    "no placeable area in this cartridge: both "
                    "placement estimates are empty (the cartridge may "
                    "simply be full)")
            raise RuntimeError(
                f"no placeable area in this cartridge: P_direct "
                f"({int(direct.sum())} px) and P_derived "
                f"({int(derived.sum())} px) do not overlap "
                f"(IoU {iou:.3f})")

        # Second bad-box gate, on the CONTENTS of the crop rather than on
        # where its centre landed. Placed after arbitrate() because the
        # thing being size-checked is the region the planner would
        # actually pack into, not the raw prediction - and before the
        # rectangle and the grid are built, so nothing downstream ever
        # sees an area this rejected.
        self.reject_if_not_one_cartridge_floor(safe, scale)

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

        inside = safe.astype(np.uint8)
        electronics = (label_map == CH_ELECTRONICS).astype(np.uint8)
        obstruction = (label_map == CH_OBSTRUCTION).astype(np.uint8)

        occupancy = _rasterise_mask(
            inside, (x0, y0, x1, y1), self.mm_per_cell, scale)

        # The crop origin, clamped the way the crop itself was clamped.
        # `recog.inference.attach_cartridge_masks` cuts the label map with
        # `max(0, int(bbox.xmin))` / `max(0, int(bbox.ymin))`, so for a
        # box hanging off the left or top of the frame the mask starts at
        # the IMAGE edge and the raw corner is not where its (0, 0) is.
        # Adding the raw corner back shifted every place target in that
        # cartridge by the overhang, silently - the rectangle it produces
        # is internally consistent and nothing downstream re-derives it.
        # The heuristic extractor has clamped here since it was written.
        ox = max(0, int(cartridge_bbox.xmin))
        oy = max(0, int(cartridge_bbox.ymin))
        # The clamp is only recoverable if the mask really is the crop
        # this box produces. A mask LARGER than the box cannot be, and
        # then neither corner tells you where its (0, 0) sits.
        if (inside.shape[0] > int(cartridge_bbox.ymax) - oy
                or inside.shape[1] > int(cartridge_bbox.xmax) - ox):
            raise ValueError(
                f"label_map is {inside.shape[0]} x {inside.shape[1]} px "
                f"but the cartridge box {cartridge_bbox} can only crop "
                f"{int(cartridge_bbox.ymax) - oy} x "
                f"{int(cartridge_bbox.xmax) - ox} px from the image. The "
                f"mask and the box describe different crops, so the "
                f"placement rectangle cannot be put back into image "
                f"coordinates.")
        return PlacementArea(
            rectangle=BBox(ox + x0, oy + y0, ox + x1, oy + y1),
            inside_mask=inside,
            pcb_mask=electronics,          # backward compatibility
            occupancy=occupancy,
            mm_per_cell=self.mm_per_cell,
            mm_per_px=scale,
            bay_mask=direct_placement(label_map).astype(np.uint8),
            electronics_mask=electronics,
            obstruction_mask=obstruction,
            consistency_iou=iou,
        )


# There is deliberately no ``attach_placement_area(cartridge, extractor,
# ...)`` helper here any more. It existed until 2026-08-12, was called by
# nothing, and duplicated the four-field assignment inside
# ``Planner._ensure_placement_areas`` while omitting both of the things
# that make that assignment safe: the ``try/except`` that turns
# ``UnknownScale`` / ``BadDetectorBox`` / ``PlacementDisagreement`` into a
# counted skip instead of an exception the caller never expected, and
# participation in ``_drop_areas_measured_at_another_scale``, without
# which a cartridge carries a rectangle whose scale-consistency invariant
# nothing maintains. Being in ``__all__`` made the unsafe one look like
# the supported way to attach a placement area. Attach through the
# planner.

__all__ = [
    "DEFAULT_WALL_INSET_MM",
    "PlacementArea",
    "PlacementAreaExtractor",
    "HeuristicPlacementAreaExtractor",
    "SegmentationPlacementAreaExtractor",
    "PlacementDisagreement",
    "BadDetectorBox",
    "UnknownScale",
    "placeable_extent_mm",
]
