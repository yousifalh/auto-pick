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
            if h > sh_h + tol:
                continue
            if w > strip_width - x_cursor + tol:
                continue

            x = x_cursor
            if forbidden_mask is not None and _overlaps_forbidden(
                forbidden_mask, x, y0, w, h, mm_per_cell,
            ):
                # Advance past the obstacle rather than abandoning the
                # shelf. FDR §6.3.1: abandoning cost 23.0 -> 3.2 cells
                # placed at 2.5% forbidden coverage.
                nxt = _next_free_x(
                    forbidden_mask, x, y0, w, h,
                    mm_per_cell, strip_width, tol,
                )
                if nxt is None:
                    continue
                x = nxt

            # The shelf's cursor advances to x + w, not x_cursor + w: the
            # region we skipped over to clear the obstacle is deliberately
            # given up. Reclaiming it would need a per-shelf free-list,
            # which this design explicitly avoids.
            shelves[shelf_idx] = (y0, sh_h, x + w)
            return PackedItem(item=item, x=x, y=y0, rotated=rot)

    # (2) Open a new shelf above the last one.
    last_y = shelves[-1][0] + shelves[-1][1] if shelves else 0.0
    for (w, h, rot) in orientations:
        if last_y + h > strip_height + tol:
            continue
        if w > strip_width + tol:
            continue

        x = 0.0
        if forbidden_mask is not None and _overlaps_forbidden(
            forbidden_mask, x, last_y, w, h, mm_per_cell,
        ):
            nxt = _next_free_x(
                forbidden_mask, x, last_y, w, h,
                mm_per_cell, strip_width, tol,
            )
            if nxt is None:
                continue
            x = nxt

        # The new shelf's cursor advances to x + w, not w: the region we
        # skipped over to clear the obstacle is deliberately given up.
        # Reclaiming it would need a per-shelf free-list, which this
        # design explicitly avoids. Consistent with branch (1) above.
        shelves.append((last_y, h, x + w))
        return PackedItem(item=item, x=x, y=last_y, rotated=rot)

    return None


# ------------------------------------------ obstacle-aware strategies ----
#
# `first_fit_decreasing` above is deliberately FROZEN. Its behaviour is
# pinned by docs/FDR_v3.md 6.3.1 (which reproduces it as pseudocode), by
# docs/receipts/forbidden_bench.*, and — load-bearing — by
# recog.synth3d.bay / recog.synth3d.layout, which use it to lay out
# synthetic scenes. Changing it would silently redraw a training corpus.
#
# The planner's ceiling is fixed by ADDING strategies and picking the best
# of them (`pack_best_effort`), so no existing caller moves.
#
# Diagnosis that motivated them (docs/superpowers/specs/
# 2026-08-11-packing-ceiling.md): FFDH pins its first shelf to y = 0 and
# every later shelf to the top of the previous one — it never scans in y.
# `_next_free_x` collapses the shelf's whole row band with `.any(axis=0)`,
# which is exact for the band but means ONE mostly-blocked row poisons
# every column in it. On the real frame scene_00005 the cartridge wall
# occupies grid row 0, so the 44-row band at y = 0 is blocked in all 62
# columns, no shelf ever opens, `last_y` never advances, and all 24
# identical items fail identically: zero placed on a 93%-free grid whose
# very next candidate offset (y = 1.5 mm) would have worked.


def _shelf_scan(
    items: Sequence[Item],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool = True,
    forbidden_mask: Optional[np.ndarray] = None,
    mm_per_cell: float = 1.5,
    tol: float = 1e-6,
) -> PackResult:
    """FFDH whose new-shelf origin scans downward past blocked bands.

    The only difference from :func:`first_fit_decreasing` is branch (2):
    where FFDH tries the single offset ``last_y`` and gives up, this tries
    ``last_y``, ``last_y + mm_per_cell``, ``last_y + 2 * mm_per_cell`` …
    and opens the shelf at the first offset that admits the item. Cell
    steps, because the mask carries no finer information than that.

    Shelf discipline is otherwise untouched, so x stays continuous — which
    is why this does not pay the cell-quantisation cost `_grid_greedy`
    does, and why it ties FFDH exactly on an unobstructed strip.
    """
    sorted_items = sorted(
        enumerate(items), key=lambda idx_item: -idx_item[1].height,
    )
    shelves: List[_Shelf] = []
    placements: List[PackedItem] = []
    unplaced: List[int] = []
    # (w, h, last_y) triples already proven to admit no shelf at any y.
    # Without this the packer rescans the whole strip for every one of the
    # identical items that follow the first failure — quadratic, and the
    # answer cannot have changed because `last_y` did not.
    exhausted: set = set()

    for _, it in sorted_items:
        placed = _try_place_item(
            it, shelves, strip_width, strip_height,
            allow_rotation, forbidden_mask, mm_per_cell, tol,
        )
        if placed is not None:
            placements.append(placed)
            continue

        opened = _open_scanned_shelf(
            it, shelves, strip_width, strip_height, allow_rotation,
            forbidden_mask, mm_per_cell, tol, exhausted,
        )
        if opened is None:
            unplaced.append(it.id)
        else:
            placements.append(opened)

    return PackResult(placements, unplaced, [s[1] for s in shelves])


def _open_scanned_shelf(
    item: Item,
    shelves: List[_Shelf],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool,
    forbidden_mask: Optional[np.ndarray],
    mm_per_cell: float,
    tol: float,
    exhausted: set,
) -> Optional[PackedItem]:
    """Open a new shelf at the lowest y >= ``last_y`` that admits ``item``."""
    orientations = [(item.width, item.height, False)]
    if allow_rotation and item.width != item.height:
        orientations.append((item.height, item.width, True))

    last_y = shelves[-1][0] + shelves[-1][1] if shelves else 0.0
    best: Optional[Tuple[float, float, float, float, bool]] = None

    for (w, h, rot) in orientations:
        if w > strip_width + tol:
            continue
        key = (w, h, last_y)
        if key in exhausted:
            continue
        found = None
        y = last_y
        while y + h <= strip_height + tol:
            if forbidden_mask is None:
                found = (y, 0.0)
                break
            if not _overlaps_forbidden(
                forbidden_mask, 0.0, y, w, h, mm_per_cell,
            ):
                found = (y, 0.0)
                break
            nxt = _next_free_x(
                forbidden_mask, 0.0, y, w, h, mm_per_cell, strip_width, tol,
            )
            if nxt is not None:
                found = (y, nxt)
                break
            y += mm_per_cell
        if found is None:
            exhausted.add(key)
            continue
        # Lowest y wins; ties keep the unrotated orientation, matching
        # FFDH's own orientation preference.
        if best is None or found[0] < best[0] - tol:
            best = (found[0], found[1], w, h, rot)

    if best is None:
        return None
    y, x, w, h, rot = best
    shelves.append((y, h, x + w))
    return PackedItem(item=item, x=x, y=y, rotated=rot)


def _grid_greedy(
    items: Sequence[Item],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool = True,
    forbidden_mask: Optional[np.ndarray] = None,
    mm_per_cell: float = 1.5,
    tol: float = 1e-6,
) -> PackResult:
    """Shelf-free greedy: place each item at the topmost-leftmost free cell.

    Abandons shelves entirely. The free region is tracked as a cell grid —
    the forbidden mask with every placement burned into it — and each item
    goes at the lexicographically smallest ``(row, col)`` whose
    ``ceil(h / cell) x ceil(w / cell)`` block is clear, found with a
    summed-area table rather than a scan.

    This is the arm that survives a fragmented mask, because nothing about
    it spans the strip width. It PAYS for that: positions are cell-snapped
    and footprints round UP to whole cells, so on an unobstructed 200 x 150
    strip it places 20 where FFDH places 23. That regression is real and is
    exactly why :func:`pack_best_effort` keeps FFDH as a competing arm
    rather than replacing it.
    """
    cell = float(mm_per_cell)
    if cell <= 0.0:
        raise ValueError("mm_per_cell must be positive")

    n_rows = int(strip_height / cell + tol)
    n_cols = int(strip_width / cell + tol)
    if forbidden_mask is not None:
        n_rows = min(n_rows, forbidden_mask.shape[0])
        n_cols = min(n_cols, forbidden_mask.shape[1])
    if n_rows <= 0 or n_cols <= 0:
        return PackResult([], [it.id for it in items], [])

    if forbidden_mask is None:
        occupied = np.zeros((n_rows, n_cols), dtype=bool)
    else:
        occupied = np.asarray(
            forbidden_mask[:n_rows, :n_cols], dtype=bool,
        ).copy()

    placements: List[PackedItem] = []
    unplaced: List[int] = []
    # Footprints already proven not to fit. `occupied` only ever gains
    # cells, so a repeat of a footprint that failed cannot succeed later.
    # Without this the planner's identical items each pay a full
    # summed-area sweep to re-derive the same no — which is most of the
    # cost on a heavily obstructed strip, where most items fail.
    failed: set = set()

    for it in sorted(items, key=lambda i: -i.height):
        if (it.width, it.height) in failed:
            unplaced.append(it.id)
            continue
        orientations = [(it.width, it.height, False)]
        if allow_rotation and it.width != it.height:
            orientations.append((it.height, it.width, True))

        # One summed-area table per item, shared by both orientations.
        free = (~occupied).astype(np.int32)
        sat = np.pad(free.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        best = None
        for (w, h, rot) in orientations:
            if w > strip_width + tol or h > strip_height + tol:
                continue
            # ceil, with no epsilon slack: rounding a footprint DOWN is the
            # unsafe direction. Matches `_next_free_x`'s own convention.
            bh = max(1, int(np.ceil(h / cell)))
            bw = max(1, int(np.ceil(w / cell)))
            if bh > n_rows or bw > n_cols:
                continue
            counts = (
                sat[bh:, bw:] - sat[:-bh, bw:] - sat[bh:, :-bw] + sat[:-bh, :-bw]
            )
            flat = np.flatnonzero((counts == bh * bw).ravel())
            if flat.size == 0:
                continue
            n_pos_cols = counts.shape[1]
            for idx in flat:
                r, c = int(idx) // n_pos_cols, int(idx) % n_pos_cols
                x, y = c * cell, r * cell
                if x + w > strip_width + tol or y + h > strip_height + tol:
                    continue
                # The cell block is clear by construction, but re-derive
                # the predicate the safety property is stated in: its
                # float indexing can disagree with this integer arithmetic
                # by a column (see `_next_free_x`'s note on cell=0.7). A
                # disagreement is always in the conservative direction, so
                # skipping to the next candidate is the whole fix.
                if forbidden_mask is not None and _overlaps_forbidden(
                    forbidden_mask, x, y, w, h, cell,
                ):
                    continue
                if best is None or (r, c) < (best[0], best[1]):
                    best = (r, c, bh, bw, w, h, rot)
                break

        if best is None:
            failed.add((it.width, it.height))
            unplaced.append(it.id)
            continue
        r, c, bh, bw, w, h, rot = best
        occupied[r:r + bh, c:c + bw] = True
        placements.append(
            PackedItem(item=it, x=c * cell, y=r * cell, rotated=rot),
        )

    return PackResult(placements, unplaced, [])


# The arms `pack_best_effort` competes, in preference order. FFDH is first
# so that a tie leaves today's placements byte-identical: a strategy that
# merely matches it must not be allowed to move the queue.
_STRATEGIES = (first_fit_decreasing, _shelf_scan, _grid_greedy)


def _drop_unsafe(
    result: PackResult,
    forbidden_mask: Optional[np.ndarray],
    mm_per_cell: float,
    tol: float,
) -> PackResult:
    """Discard any placement that hits the mask or an earlier placement.

    A safety net, not a load-bearing step: every arm is meant to be correct
    by construction and the tests assert that directly. It exists because
    the failure direction matters asymmetrically — dropping a placement
    costs one cell, keeping a bad one puts a battery on a PCB — so the net
    is worth its cost even when it never fires.

    **That cost is O(p²) in placements, not O(n).** Each kept placement is
    tested against every earlier kept placement, and `pack_best_effort`
    runs this THREE times per pack, once per arm. Measured (audit K §1.6),
    single call / ×3: 25 placements 0.042 / 0.126 ms, 100 → 0.558 / 1.675,
    200 → 2.266 / 6.799, 400 → 8.586 / 25.757, 800 → 37.968 / 113.904 —
    a clean 4×-per-doubling. Below ~50 placements it is noise; above ~200
    it is the entire 8 ms O3 budget on its own.

    Today p ≤ 24 (`plan/planner.py`'s `n_est` at the largest floor the
    detector interlock admits — see `pack_best_effort`), so this costs
    ~0.1 ms and no test can contradict a claim about its asymptotics,
    which is exactly how the docstring managed to say "O(n)" until
    2026-08-12. The algorithm is unchanged and deliberately so: it is not
    binding, and the packer's real-world margin is 3.9×. If it ever does
    bind, sort by x and sweep, or bucket into grid cells.
    """
    kept: List[PackedItem] = []
    dropped: List[int] = []
    for p in result.placements:
        if forbidden_mask is not None and _overlaps_forbidden(
            forbidden_mask, p.x, p.y, p.width, p.height, mm_per_cell,
        ):
            dropped.append(p.item.id)
            continue
        if any(
            not (p.x + p.width <= q.x + tol or q.x + q.width <= p.x + tol
                 or p.y + p.height <= q.y + tol or q.y + q.height <= p.y + tol)
            for q in kept
        ):
            dropped.append(p.item.id)
            continue
        kept.append(p)
    if not dropped:
        return result
    return PackResult(kept, list(result.unplaced_ids) + dropped,
                      result.shelf_heights)


def pack_best_effort(
    items: Sequence[Item],
    strip_width: float,
    strip_height: float,
    allow_rotation: bool = True,
    forbidden_mask: Optional[np.ndarray] = None,
    mm_per_cell: float = 1.5,
    tol: float = 1e-6,
) -> PackResult:
    """Run every packing strategy and return whichever placed the most.

    Same signature and same :class:`PackResult` as
    :func:`first_fit_decreasing`, so it is a drop-in for the planner.

    Best-of rather than replace-with is the point. The three arms win in
    different regimes and each loses somewhere — FFDH is best on a clean or
    lightly obstructed strip and collapses as coverage rises; the grid arm
    is the reverse. Taking the maximum makes the result **provably no worse
    than today's** on every instance, which is the property worth having
    here: the alternative is a change that lifts the average and quietly
    regresses a frame nobody re-measured.

    Ties go to the earliest arm in :data:`_STRATEGIES`, so an instance no
    arm improves on comes back with FFDH's exact placements.

    **Its O3 latency margin is held up by a guard in another module, and
    nothing here would tell you.** Cost grows with the mask's cell count
    (via `_grid_greedy`, which rebuilds a full summed-area table per item)
    and with the item count (via the two shelf arms), and
    `plan/planner.py` ties the two together — `n_est = 2·area/(18.5×65)`.
    Bisected against the 8 ms per-cartridge budget (audit K §1.5, worst
    over wall / 15 %-blob / checkerboard masks): 81.7 × 180.0 mm floor →
    2.04 ms, 140 × 278 → 5.70 ms, **158 × 314 → 8.16 ms, the first
    breach**, 279 × 555 → 63 ms.

    The packer never sees an input near that, and the reason is
    `SegmentationPlacementAreaExtractor.reject_if_not_one_cartridge_floor`
    (`plan/placement_area.py`), which refuses any placeable floor larger
    than `_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)` mm **before the
    occupancy grid is built**. That interlock was written to catch a
    detector box spanning a cartridge and three loose cells — a
    correctness bug with nothing to do with latency — and it happens to
    cap this function at 2.04 ms worst case, a 3.9× margin. Raising
    `max_cartridge_extent_mm` for a larger SKU raises the packing cost
    with it; the breach point is ~1.94× the interlock's long axis and
    ~1.93× its short axis. Re-measure this function against the 8 ms
    budget before widening that bound, and see the note beside
    `_MAX_CARTRIDGE_EXTENT_MM` itself.
    """
    best: Optional[PackResult] = None
    for strategy in _STRATEGIES:
        res = _drop_unsafe(
            strategy(
                items, strip_width, strip_height,
                allow_rotation=allow_rotation,
                forbidden_mask=forbidden_mask,
                mm_per_cell=mm_per_cell,
                tol=tol,
            ),
            forbidden_mask, mm_per_cell, tol,
        )
        if best is None or res.count > best.count:
            best = res
    assert best is not None  # _STRATEGIES is never empty
    return best
