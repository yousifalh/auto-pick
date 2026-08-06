"""Shelf-based First-Fit Decreasing Height (FFDH) 2-D strip packing.

Lives in ``common`` rather than ``plan`` because two callers need it:
the planner packs batteries into a cartridge's placement rectangle, and
:mod:`recog.synth3d.layout` packs part footprints into jig pockets when
generating synthetic scenes. Neither should import the other, so the
shared algorithm sits below both.

Units are whatever the caller uses consistently; the planner works in
millimetres. Properties (Berkey & Wang 1987; Martello, Pisinger & Toth
2000): deterministic, worst-case 1.7 x OPT, O(n log n).
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


def _next_free_x(
    mask: np.ndarray,
    x_from: float,
    y: float,
    w: float,
    h: float,
    mm_per_cell: float,
    strip_width: float,
    tol: float,
) -> Optional[float]:
    """Leftmost ``x >= x_from`` where a ``w x h`` footprint clears ``mask``.

    Returns ``None`` when no such position fits inside ``strip_width``.

    This is what lets a shelf survive an obstacle. The previous
    implementation abandoned the whole shelf as soon as the cursor
    position overlapped a forbidden cell, which FDR §6.3.1 measured as
    23.0 -> 3.2 cells placed at 2.5% coverage — worse than not being
    obstacle-aware at all.

    The returned x is snapped to cell boundaries. Snapping can only move
    the item right, never left. Every returned position is guaranteed to
    satisfy ``not _overlaps_forbidden(mask, x, y, w, h, mm_per_cell)``.
    Sub-cell-precision positions may be skipped; the mask is quantised
    at mm_per_cell anyway, so sub-cell precision is not information it carries.
    """
    if x_from + w > strip_width + tol:
        # Even the leftmost candidate can't fit. x only increases from here
        # on, so every later candidate would fail too — bail before scanning
        # the whole mask to reach a foregone conclusion.
        return None

    r1 = max(0, int(y / mm_per_cell))
    r2 = min(mask.shape[0], int(np.ceil((y + h) / mm_per_cell)))
    if r2 <= r1:
        # Empty row band: position clears by definition.
        return x_from if x_from + w <= strip_width + tol else None

    # Collapse the row band: a column is blocked if any cell in it is.
    blocked = mask[r1:r2, :].any(axis=0)

    n_cols = max(1, int(np.ceil(w / mm_per_cell)))
    # Start scanning at the first cell boundary at or after x_from. tol is a
    # millimetre tolerance; convert it to cell space (tol / mm_per_cell)
    # before subtracting from a value already in cells — mixing the units
    # here let the effective slack scale with mm_per_cell, which breached
    # the x >= x_from - tol contract for mm_per_cell > 1. Clamp to 0: a
    # negative x_from (not reachable today, but Task 2 wires this to a
    # running shelf cursor) must not turn into a negative index below,
    # which Python would silently wrap instead of raising.
    c_start = max(0, int(np.ceil(x_from / mm_per_cell - tol / mm_per_cell)))

    # Scan for the first run of n_cols consecutive clear columns.
    run = 0
    for c in range(c_start, blocked.shape[0]):
        run = 0 if blocked[c] else run + 1
        if run >= n_cols:
            # Found n_cols clear columns ending at column c.
            x = (c - n_cols + 1) * mm_per_cell
            # Validate with _overlaps_forbidden: integer column arithmetic in
            # the scan can disagree with the float-based indexing the
            # predicate re-derives, due to IEEE-754 rounding. Concrete case:
            # cell=0.7, c=3 -> x=2.0999999999999996; the predicate computes
            # x/mm_per_cell = 2.9999999999999996, and int() truncates that to
            # 2, one column short of the run just cleared. Validating catches
            # it; on the rare truncation the loop just continues to the next
            # run, so this is at most a second validation call, not a
            # per-candidate probe loop.
            if x + w <= strip_width + tol and not _overlaps_forbidden(
                mask, x, y, w, h, mm_per_cell
            ):
                return x
            # Predicate check failed; continue scanning for the next run.
    return None


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
