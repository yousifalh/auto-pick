"""Valid-placement-area extraction.

Given a camera image and a cartridge bounding box, this module
computes the rectangular region inside the cartridge where a battery
may legitimately be placed. The pipeline matches PPR §5.3.2:

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

The camera mount is fixed on the cell, so keeping the placement
rectangle axis-aligned costs nothing in accuracy and simplifies the
downstream grid bookkeeping.
"""
from __future__ import annotations

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


# --------------------------------------------------------- extractor ----

class PlacementAreaExtractor:
    """Reusable extractor — tuneables come from ``planning.yaml``."""

    def __init__(
        self,
        safety_margin_px: int = 5,
        morph_close_ksize: int = 5,
        morph_open_ksize: int = 3,
        mm_per_cell: float = 1.5,
        mm_per_px: float = 0.38,
    ) -> None:
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
        ix1, iy1, ix2, iy2 = rect
        px_per_cell = max(1.0, self.mm_per_cell / self.mm_per_px)
        rows = max(1, int((iy2 - iy1) / px_per_cell))
        cols = max(1, int((ix2 - ix1) / px_per_cell))

        grid = OccupancyGrid(
            rows=rows, cols=cols, resolution_mm=self.mm_per_cell,
        )
        h_mask, w_mask = inside_mask.shape

        for r in range(rows):
            for c in range(cols):
                ypx = int(iy1 + r * px_per_cell + px_per_cell / 2)
                xpx = int(ix1 + c * px_per_cell + px_per_cell / 2)
                inside_bounds = 0 <= ypx < h_mask and 0 <= xpx < w_mask
                if inside_bounds and inside_mask[ypx, xpx] == 0:
                    grid.set(r, c, CellState.FORBIDDEN)
        return grid


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


__all__ = ["PlacementArea", "PlacementAreaExtractor", "attach_placement_area"]
