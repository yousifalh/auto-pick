# Forbidden-mask FFDH shelf-advance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the forbidden-mask FFDH packer advance past an obstacle on a shelf instead of abandoning the shelf, so pixel-precise placement masks improve packing instead of destroying it.

**Architecture:** `common/packing.py`'s `_try_place_item` currently `continue`s to the next shelf when a candidate position overlaps a forbidden cell, discarding the rest of that shelf. This plan adds a `_next_free_x` scan that finds the leftmost position on the same shelf where the item's footprint clears the mask, and retries there. The shelf state tuple `(y_bottom, height, x_cursor)` is unchanged — the obstacle positions are read from the mask on demand, which is simpler than FDR §6.3.1's proposed per-shelf obstacle list and has the same effect.

**Tech Stack:** Python 3.10+, NumPy, pytest. No new dependencies.

## Global Constraints

- Units are millimetres throughout `common/packing.py`; the mask is indexed `[row, col]` with each cell `mm_per_cell` mm square, aligned to the strip origin.
- `mm_per_cell` default is `1.5`.
- Floating-point comparisons use the existing `tol` parameter, default `1e-6`.
- The algorithm must stay deterministic: same inputs, same output, every run.
- `first_fit_decreasing`'s signature must not change — `recog/synth3d/layout.py:208` and `plan/bin_packing.py:87` both call it.
- With `forbidden_mask=None` the behaviour must be byte-for-byte what it is today. `tests/test_bin_packing.py` and `tests/test_packing_move.py` must pass unchanged.
- Reference receipt to beat: `docs/receipts/forbidden_bench.csv`, currently 23.0 cells at 0 % coverage, **3.2 at 2.5 %**, 1.2 at 5 %, 0.1 at 10 %.

---

## File Structure

| File | Responsibility |
|---|---|
| `common/packing.py` | FFDH algorithm. Gains `_next_free_x`; `_try_place_item` gains the retry loop. |
| `tests/test_packing_forbidden.py` | **New.** Unit tests for `_next_free_x` and the shelf-advance behaviour. |
| `tests/test_bin_packing.py` | Existing. Must keep passing untouched — the no-mask path is the regression guard. |
| `docs/receipts/forbidden_bench.csv` | Regenerated after the fix; the before/after delta is the deliverable. |

---

### Task 1: `_next_free_x` — find the leftmost clear position on a shelf

**Files:**
- Modify: `common/packing.py` (add function after `_overlaps_forbidden`, around line 85)
- Test: `tests/test_packing_forbidden.py` (create)

**Interfaces:**
- Consumes: `_overlaps_forbidden(mask, x, y, w, h, mm_per_cell) -> bool` (existing, `common/packing.py:65`)
- Produces: `_next_free_x(mask, x_from, y, w, h, mm_per_cell, strip_width, tol) -> Optional[float]` — the smallest `x >= x_from` at which a `w × h` footprint at row-band `y` clears every forbidden cell and still fits inside `strip_width`, or `None` if no such `x` exists.

Why a scan rather than the FDR's per-shelf obstacle list: the mask is already the authoritative record of where obstacles are, and reading it on demand needs no change to the shelf tuple, so `first_fit_decreasing`'s signature and every existing caller stay untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_packing_forbidden.py`:

```python
"""Forbidden-mask shelf-advance behaviour (FDR §6.3.1 fix)."""
from __future__ import annotations

import numpy as np
import pytest

from common.packing import Item, _next_free_x, first_fit_decreasing

CELL = 1.0  # 1 mm cells keep the arithmetic readable in these tests


def _mask(rows: int, cols: int, blocks=()):
    """Zero mask with `blocks` of (r0, r1, c0, c1) set to 1."""
    m = np.zeros((rows, cols), dtype=np.uint8)
    for r0, r1, c0, c1 in blocks:
        m[r0:r1, c0:c1] = 1
    return m


def test_next_free_x_returns_x_from_when_already_clear():
    m = _mask(10, 20)
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) == 0.0


def test_next_free_x_advances_past_a_blocking_column_run():
    # Columns 2..5 blocked across the whole row band.
    m = _mask(10, 20, [(0, 10, 2, 6)])
    # A 4 mm wide item at x=0 would span columns 0..3 and hit the block.
    # The first clear run of 4 columns starts at column 6.
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) == 6.0


def test_next_free_x_ignores_obstacles_outside_the_row_band():
    # Block sits in rows 5..9; an item occupying rows 0..2 is unaffected.
    m = _mask(10, 20, [(5, 10, 2, 6)])
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) == 0.0


def test_next_free_x_returns_none_when_no_run_is_wide_enough():
    # Blocks leave only 2-column gaps; a 4 mm item never fits.
    m = _mask(10, 20, [(0, 10, 2, 4), (0, 10, 6, 8), (0, 10, 10, 20)])
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) is None


def test_next_free_x_respects_the_strip_width():
    # Clear from column 16 on, but a 4 mm item there would end at 20 mm,
    # which exactly fits a 20 mm strip and must be accepted...
    m = _mask(10, 20, [(0, 10, 0, 16)])
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) == 16.0
    # ...but not an 18 mm strip.
    assert _next_free_x(m, 0.0, 0.0, 4.0, 3.0, CELL, 18.0, 1e-6) is None


def test_next_free_x_never_moves_left_of_x_from():
    m = _mask(10, 20)
    assert _next_free_x(m, 7.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6) == 7.0


def test_next_free_x_result_actually_clears_the_mask():
    """Whatever it returns must satisfy _overlaps_forbidden == False and x >= x_from."""
    from common.packing import _overlaps_forbidden

    rng = np.random.default_rng(0)
    for _ in range(200):
        m = (rng.random((12, 30)) < 0.15).astype(np.uint8)
        # Test with both cell-aligned and fractional x_from values.
        x_from = rng.random() * 25.0
        x = _next_free_x(m, x_from, 0.0, 5.0, 4.0, CELL, 30.0, 1e-6)
        if x is not None:
            # Must clear the mask and satisfy x >= x_from and fit in strip.
            assert not _overlaps_forbidden(m, x, 0.0, 5.0, 4.0, CELL)
            assert x >= x_from - 1e-6
            assert x + 5.0 <= 30.0 + 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_packing_forbidden.py -v`
Expected: FAIL — `ImportError: cannot import name '_next_free_x' from 'common.packing'`

- [ ] **Step 3: Implement `_next_free_x`**

Add to `common/packing.py`, immediately after `_overlaps_forbidden` (after line 85):

```python
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
    r1 = max(0, int(y / mm_per_cell))
    r2 = min(mask.shape[0], int(np.ceil((y + h) / mm_per_cell)))
    if r2 <= r1:
        # Empty row band: position clears by definition.
        return x_from if x_from + w <= strip_width + tol else None

    # Collapse the row band: a column is blocked if any cell in it is.
    blocked = mask[r1:r2, :].any(axis=0)

    n_cols = max(1, int(np.ceil(w / mm_per_cell)))
    # Start scanning at the first cell boundary at or after x_from.
    c_start = int(np.ceil(x_from / mm_per_cell - tol))

    # Scan for the first run of n_cols consecutive clear columns.
    run = 0
    for c in range(c_start, blocked.shape[0]):
        run = 0 if blocked[c] else run + 1
        if run >= n_cols:
            # Found n_cols clear columns ending at column c.
            # The item occupies columns [c - n_cols + 1, c + 1).
            # Return the position directly; it is correct by construction:
            # _overlaps_forbidden will compute c1=c-n_cols+1, c2=c+1,
            # checking exactly the columns we just cleared.
            x = (c - n_cols + 1) * mm_per_cell
            return x if x + w <= strip_width + tol else None
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_packing_forbidden.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add common/packing.py tests/test_packing_forbidden.py
git commit -m "feat(packing): add _next_free_x shelf obstacle scan

Finds the leftmost position on a shelf's row band where a footprint
clears the forbidden mask. Snapped to cell boundaries, which can only
move an item right, so the result never overlaps.

Not yet wired into _try_place_item."
```

---

### Task 2: Advance the cursor instead of abandoning the shelf

**Files:**
- Modify: `common/packing.py:158-170` (the first-fit-across-shelves loop in `_try_place_item`)
- Test: `tests/test_packing_forbidden.py` (append)

**Interfaces:**
- Consumes: `_next_free_x(...)` from Task 1; `_Shelf = Tuple[float, float, float]` being `(y_bottom, shelf_height, x_cursor)`
- Produces: no new names. `_try_place_item`'s signature is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_packing_forbidden.py`:

```python
def test_shelf_survives_an_obstacle_instead_of_being_abandoned():
    """The FDR §6.3.1 defect, reduced to its smallest reproduction.

    A 100 x 10 mm strip with a 10 mm obstacle at x=0..10. Three 20 mm
    items should pack at x=10, 30, 50 on one shelf. The old code placed
    at most one, on a new shelf, because the first overlap at x=0 killed
    the shelf for every item.
    """
    mask = _mask(10, 100, [(0, 10, 0, 10)])
    items = [Item(id=i, width=20.0, height=10.0) for i in range(3)]
    res = first_fit_decreasing(
        items, 100.0, 10.0, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count == 3
    assert sorted(p.x for p in res.placements) == [10.0, 30.0, 50.0]


def test_no_placement_overlaps_the_forbidden_mask():
    """The safety invariant: whatever gets placed must clear the mask."""
    from common.packing import _overlaps_forbidden

    rng = np.random.default_rng(7)
    for seed in range(50):
        mask = (rng.random((40, 60)) < 0.08).astype(np.uint8)
        items = [Item(id=i, width=6.0, height=5.0) for i in range(30)]
        res = first_fit_decreasing(
            items, 60.0, 40.0, allow_rotation=True,
            forbidden_mask=mask, mm_per_cell=CELL,
        )
        for p in res.placements:
            assert not _overlaps_forbidden(
                mask, p.x, p.y, p.width, p.height, CELL,
            ), f"seed {seed}: placement at {p.x},{p.y} overlaps"


def test_placements_do_not_overlap_each_other():
    """Advancing the cursor must not let two items share space."""
    mask = _mask(40, 60, [(0, 40, 10, 14), (0, 40, 30, 33)])
    items = [Item(id=i, width=6.0, height=5.0) for i in range(20)]
    res = first_fit_decreasing(
        items, 60.0, 40.0, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    boxes = [(p.x, p.y, p.x + p.width, p.y + p.height)
             for p in res.placements]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= 1e-6 or overlap_y <= 1e-6, f"{a} overlaps {b}"


def test_unmasked_behaviour_is_unchanged():
    """With no mask the fix must be inert."""
    items = [Item(id=i, width=18.5, height=65.0) for i in range(12)]
    a = first_fit_decreasing(items, 200.0, 150.0, allow_rotation=True)
    b = first_fit_decreasing(
        items, 200.0, 150.0, allow_rotation=True, forbidden_mask=None,
    )
    assert [(p.item.id, p.x, p.y, p.rotated) for p in a.placements] == \
           [(p.item.id, p.x, p.y, p.rotated) for p in b.placements]
    assert a.count == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_packing_forbidden.py -v -k "shelf_survives or do_not_overlap"`
Expected: `test_shelf_survives_an_obstacle_instead_of_being_abandoned` FAILS with `assert 1 == 3` (or similar low count). The others may pass already — they are invariants the fix must not break.

- [ ] **Step 3: Rewrite the first-fit loop**

Replace `common/packing.py:158-170` — the block beginning `# (1) First-fit across existing shelves.` and ending with the `return PackedItem(...)` — with:

```python
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

            shelves[shelf_idx] = (y0, sh_h, x + w)
            return PackedItem(item=item, x=x, y=y0, rotated=rot)
```

Two things to note while editing. The `h > sh_h` test moves **above** the width test — it is the cheaper rejection and its result cannot change when `x` moves. And the shelf's new cursor is `x + w`, not `x_cursor + w`: the skipped-over region is given up deliberately, because reclaiming it would need the per-shelf free-list that this design avoids.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_packing_forbidden.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Run the existing packer suites for regressions**

Run: `pytest tests/test_bin_packing.py tests/test_packing_move.py tests/test_planner.py -v`
Expected: PASS, no failures. If anything fails here, the fix has changed unmasked behaviour and Step 3 is wrong.

- [ ] **Step 6: Commit**

```bash
git add common/packing.py tests/test_packing_forbidden.py
git commit -m "fix(packing): advance the shelf cursor past a forbidden region

_try_place_item abandoned an entire shelf when the candidate position
overlapped a forbidden cell. FDR 6.3.1 measured the cost: 23.0 cells
placed at 0% coverage, 3.2 at 2.5% - worse than the obstacle-unaware
rejection-sampling baseline it was meant to beat.

It now scans for the leftmost clear position on the same shelf and
places there. The skipped region is given up rather than tracked in a
per-shelf free list; that is the simpler half of FDR 6.3.1's proposed
fix and recovers the same shelf.

Unmasked behaviour is unchanged and covered by a test."
```

---

### Task 3: Apply the same rule when opening a new shelf

**Files:**
- Modify: `common/packing.py:172-184` (the new-shelf branch of `_try_place_item`)
- Test: `tests/test_packing_forbidden.py` (append)

**Interfaces:**
- Consumes: `_next_free_x(...)` from Task 1
- Produces: no new names

The new-shelf branch has the identical defect: it tries `x = 0.0` and gives up on the whole shelf if that overlaps. A mask blocking the left edge of a fresh shelf currently costs every remaining item.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_packing_forbidden.py`:

```python
def test_new_shelf_opens_to_the_right_of_a_left_edge_obstacle():
    """A block at the left edge must not veto the whole new shelf."""
    # 100 x 20 strip. Rows 0..9 (the first shelf band) blocked at x=0..30.
    mask = _mask(20, 100, [(0, 10, 0, 30)])
    items = [Item(id=0, width=20.0, height=10.0)]
    res = first_fit_decreasing(
        items, 100.0, 20.0, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count == 1
    assert res.placements[0].x == 30.0
    assert res.placements[0].y == 0.0


def test_new_shelf_falls_through_when_the_whole_band_is_blocked():
    """If no x on the band is clear, the item is genuinely unplaceable."""
    mask = _mask(20, 100, [(0, 20, 0, 100)])
    items = [Item(id=0, width=20.0, height=10.0)]
    res = first_fit_decreasing(
        items, 100.0, 20.0, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count == 0
    assert res.unplaced_ids == [0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_packing_forbidden.py -v -k new_shelf`
Expected: `test_new_shelf_opens_to_the_right_of_a_left_edge_obstacle` FAILS with `assert 0 == 1`.

- [ ] **Step 3: Rewrite the new-shelf branch**

Replace `common/packing.py:172-184` — the block beginning `# (2) Open a new shelf above the last one.` through its `return PackedItem(...)` — with:

```python
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

        shelves.append((last_y, h, x + w))
        return PackedItem(item=item, x=x, y=last_y, rotated=rot)

    return None
```

The new shelf's cursor is `x + w`, so the region left of the obstacle is not re-offered — consistent with Task 2.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_packing_forbidden.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: PASS. Record the count; it is the baseline for Plan B.

- [ ] **Step 6: Commit**

```bash
git add common/packing.py tests/test_packing_forbidden.py
git commit -m "fix(packing): open a new shelf to the right of a left-edge obstacle

The new-shelf branch had the same defect as the first-fit branch: it
tried x=0 and vetoed the entire shelf on overlap, so a mask covering
the left edge of a fresh band cost every remaining item."
```

---

### Task 4: Regenerate the benchmark and update the FDR

**Files:**
- Modify: `docs/receipts/forbidden_bench.csv`, `docs/receipts/forbidden_bench.txt`
- Modify: `docs/FDR_v3.md` §6.3.1 and §13.2(2)
- Test: none — this task reports, it does not implement

**Interfaces:**
- Consumes: the fixed `first_fit_decreasing` from Tasks 2–3
- Produces: an updated receipt whose 2.5 % row is the acceptance criterion for this whole plan

- [ ] **Step 1: Locate the benchmark generator**

Run: `grep -rn "forbidden_bench" --include=*.py .`

The benchmark parameters are recorded in FDR §6.3.1: a 200 × 150 mm strip, 40 candidate 18.5 × 65 mm items, 40 random masks per coverage level, masks drawn as 2–6 cell rectangular blobs, coverage levels 0.0 / 2.5 / 5.0 / 10.0 / 15.0 / 25.0 %. If no generator script exists, write one at `scripts/forbidden_bench.py` reproducing exactly those parameters with a fixed seed, and commit it — a receipt that cannot be regenerated is not a receipt.

- [ ] **Step 2: Run the benchmark and capture the output**

Run the generator, writing to `docs/receipts/forbidden_bench.csv` and `.txt`.

Expected shape of the result: the 0 % row stays at 23.0 (the fix is inert without a mask, which Task 2 tests). Every non-zero row should rise. The acceptance bar is the 2.5 % row exceeding the **rejection-sampling baseline of 10.6**, since beating that baseline is the entire justification for the forbidden-mask variant existing.

If the 2.5 % row lands between 3.2 and 10.6, the fix works but does not yet justify the variant, and that is a finding to report rather than paper over — record it and stop, rather than tuning until the number looks right.

- [ ] **Step 3: Update FDR §6.3.1**

Replace the results table with the new numbers, keep the old table alongside it as the before-state, and rewrite the "Proposed fix and operational impact" paragraph in the past tense with the measured delta. The prose currently proposes "changing the shelf-state representation to track a list of obstacles per shelf"; record that the implemented fix scans the mask on demand instead, needs no representation change, and recovers the same shelf.

- [ ] **Step 4: Update FDR §13.2(2)**

The item currently reads as priority-2 future work. It is done. Rewrite it as completed work with a pointer to §6.3.1's new table, and remove it from the future-work list. Per the segmentation spec §8, this was a blocking prerequisite for §13.2(5) — note that the block is now lifted.

- [ ] **Step 5: Commit**

```bash
git add docs/receipts/forbidden_bench.csv docs/receipts/forbidden_bench.txt docs/FDR_v3.md scripts/forbidden_bench.py
git commit -m "docs: regenerate the forbidden-mask FFDH benchmark after the shelf fix

Records the measured before/after and rewrites FDR 6.3.1's proposed-fix
paragraph in the past tense. 13.2(2) moves from future work to done,
which lifts the blocking prerequisite on 13.2(5)."
```

---

## Acceptance

- [ ] `pytest -q` passes with no regressions against the pre-plan baseline.
- [ ] `docs/receipts/forbidden_bench.csv`'s 2.5 % row exceeds 10.6 (the rejection-sampling baseline), or the shortfall is recorded as a finding.
- [ ] The 0 % row is still 23.0 — the fix is inert without a mask.
- [ ] No placement in any test overlaps the forbidden mask or another placement.
- [ ] FDR §6.3.1 and §13.2(2) reflect the implemented fix, not the proposed one.
