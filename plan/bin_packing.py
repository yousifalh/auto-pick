"""Shelf-based First-Fit Decreasing Height (FFDH) bin-packing.

This implements the 2-D orthogonal strip-packing variant selected in
PPR §5.3.3. The strip is the cartridge's placement rectangle; items
are identical battery footprints (each cartridge type permits at most
two rotations, 0° or 90°). The algorithm sorts items by decreasing
height, then for each item:

1. Tries the leftmost "shelf" that can still accommodate it (first-fit).
2. If no shelf works, opens a new shelf above the last one.

Properties (Berkey & Wang 1987; Martello, Pisinger & Toth 2000):

* Deterministic and reproducible across runs.
* Worst-case packing ratio: 1.7 × OPT.
* Runs in O(n log n) — well under the 50 ms / 8 ms budgets in the PPR.

The module also exposes a convenience adapter,
:func:`pack_cartridge`, that pulls the strip geometry and forbidden
mask directly off a :class:`plan.scene.Cartridge` and invokes FFDH.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------- types ----

@dataclass(frozen=True)
class Item:
    """A rectangular item to pack (e.g., a battery footprint)."""

    id: int
    width: float
    height: float


@dataclass(frozen=True)
class PackedItem:
    """Placement output — ``(x, y)`` is the top-left corner in mm."""

    item: Item
    x: float
    y: float
    rotated: bool = False

    @property
    def width(self) -> float:
        return self.item.height if self.rotated else self.item.width

    @property
    def height(self) -> float:
        return self.item.width if self.rotated else self.item.height


@dataclass
class PackResult:
    """Bundle of everything the planner needs after FFDH runs."""

    placements: List[PackedItem]
    unplaced_ids: List[int]
    shelf_heights: List[float]

    @property
    def count(self) -> int:
        return len(self.placements)


# ------------------------------------------------------ geometry ----

def _overlaps_forbidden(
    mask: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    mm_per_cell: float,
) -> bool:
    """``True`` if ``[x, y, x+w, y+h]`` intersects any forbidden cell.

    ``mask`` is indexed ``[row, col]``, with each cell ``mm_per_cell`` mm
    square and aligned to the strip origin.
    """
    c1 = max(0, int(x / mm_per_cell))
    r1 = max(0, int(y / mm_per_cell))
    c2 = min(mask.shape[1], int(np.ceil((x + w) / mm_per_cell)))
    r2 = min(mask.shape[0], int(np.ceil((y + h) / mm_per_cell)))
    if c2 <= c1 or r2 <= r1:
        return False
    return bool(mask[r1:r2, c1:c2].any())


# --------------------------------------------------------- FFDH ----

# A shelf is (y_bottom, shelf_height, x_cursor).
_Shelf = Tuple[float, float, float]


def first_fit_decreasing(
    items: Sequence[Item],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool = True,
    forbidden_mask: Optional[np.ndarray] = None,
    mm_per_cell: float = 1.5,
    tol: float = 1e-6,
) -> PackResult:
    """Pack ``items`` into a strip using shelf-based FFDH.

    Parameters
    ----------
    items:
        Items to pack. Sorted internally by decreasing height.
    strip_width, strip_height:
        Strip dimensions in millimetres.
    allow_rotation:
        If ``True``, each item may be placed at 0° or 90°. The chosen
        orientation is the first one that fits on an existing shelf,
        otherwise the first one that opens a viable new shelf.
    forbidden_mask:
        Optional binary grid of forbidden cells at ``mm_per_cell`` mm
        resolution. Items overlapping any forbidden cell are skipped.
    """
    sorted_items = sorted(
        enumerate(items), key=lambda idx_item: -idx_item[1].height,
    )

    shelves: List[_Shelf] = []
    placements: List[PackedItem] = []
    unplaced: List[int] = []

    for _, it in sorted_items:
        placed = _try_place_item(
            it, shelves, strip_width, strip_height,
            allow_rotation, forbidden_mask, mm_per_cell, tol,
        )
        if placed is None:
            unplaced.append(it.id)
        else:
            placements.append(placed)

    return PackResult(
        placements=placements,
        unplaced_ids=unplaced,
        shelf_heights=[s[1] for s in shelves],
    )


def _try_place_item(
    item: Item,
    shelves: List[_Shelf],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool,
    forbidden_mask: Optional[np.ndarray],
    mm_per_cell: float,
    tol: float,
) -> Optional[PackedItem]:
    """Try to place ``item`` — first on each shelf, then on a new one."""
    orientations = [(item.width, item.height, False)]
    if allow_rotation and item.width != item.height:
        orientations.append((item.height, item.width, True))

    # (1) First-fit across existing shelves.
    for shelf_idx, (y0, sh_h, x_cursor) in enumerate(shelves):
        for (w, h, rot) in orientations:
            if w > strip_width - x_cursor + tol:
                continue
            if h > sh_h + tol:
                continue
            if forbidden_mask is not None and _overlaps_forbidden(
                forbidden_mask, x_cursor, y0, w, h, mm_per_cell,
            ):
                continue
            shelves[shelf_idx] = (y0, sh_h, x_cursor + w)
            return PackedItem(item=item, x=x_cursor, y=y0, rotated=rot)

    # (2) Open a new shelf above the last one.
    last_y = shelves[-1][0] + shelves[-1][1] if shelves else 0.0
    for (w, h, rot) in orientations:
        if w > strip_width + tol:
            continue
        if last_y + h > strip_height + tol:
            continue
        if forbidden_mask is not None and _overlaps_forbidden(
            forbidden_mask, 0.0, last_y, w, h, mm_per_cell,
        ):
            continue
        shelves.append((last_y, h, w))
        return PackedItem(item=item, x=0.0, y=last_y, rotated=rot)

    return None


# ---------------------------------- adapter for the digital twin ----

def pack_cartridge(
    cartridge,
    battery_width_mm: float,
    battery_length_mm: float,
    allow_rotation: bool = True,
    mm_per_px: float = 0.38,
) -> PackResult:
    """Build an FFDH instance for ``cartridge`` and run it.

    The cartridge's placement rectangle (in pixels) is converted to a
    strip in millimetres. The forbidden mask is derived from the
    cartridge's occupancy grid, unioning FORBIDDEN / PLACED / PLANNED
    cells so already-assigned positions aren't packed over.
    """
    pr = cartridge.placeable_rectangle
    if pr is None:
        raise ValueError(
            "cartridge.placeable_rectangle is None — "
            "extract placement area first"
        )

    strip_w_mm = pr.width * mm_per_px
    strip_h_mm = pr.height * mm_per_px

    # Build an upper-bound number of candidate items. We over-estimate
    # so FFDH has enough identical items to saturate the strip.
    n_max_est = max(
        4,
        int((strip_w_mm * strip_h_mm)
            / (battery_width_mm * battery_length_mm)) * 2,
    )
    items = [
        Item(id=i, width=battery_width_mm, height=battery_length_mm)
        for i in range(n_max_est)
    ]

    forbidden = None
    mm_per_cell = 1.5
    if cartridge.occupancy is not None:
        from plan.scene import CellState

        forbidden = cartridge.occupancy.mask_of(
            CellState.FORBIDDEN, CellState.PLACED, CellState.PLANNED,
        )
        mm_per_cell = cartridge.occupancy.resolution_mm

    return first_fit_decreasing(
        items, strip_w_mm, strip_h_mm,
        allow_rotation=allow_rotation,
        forbidden_mask=forbidden,
        mm_per_cell=mm_per_cell,
    )


__all__ = [
    "Item",
    "PackResult",
    "PackedItem",
    "first_fit_decreasing",
    "pack_cartridge",
]
