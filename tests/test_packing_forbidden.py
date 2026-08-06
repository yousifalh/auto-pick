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


def test_next_free_x_negative_x_from_never_returns_negative_x():
    """A negative x_from must not wrap `blocked` via negative indexing.

    `c_start = int(x_from / mm_per_cell)` without a `max(0, ...)` clamp goes
    negative for a negative x_from, and Python silently wraps `blocked[c]`
    from the end of the array instead of raising — which can return a
    negative x. Not reachable today (shelf cursors are non-negative), but
    Task 2 wires this to a running shelf cursor, so pin the contract now.
    """
    m = _mask(10, 20)
    x = _next_free_x(m, -5.0, 0.0, 4.0, 3.0, CELL, 20.0, 1e-6)
    assert x is not None
    assert x >= 0.0


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


def test_next_free_x_ieee754_truncation_regression():
    """Regression test for IEEE-754 rounding bug in float column conversion.

    When integer column arithmetic is converted to float position and back,
    rounding can cause the predicate to check a different column than the scan
    intended. This test pins the exact case from the fuzz.
    """
    from common.packing import _overlaps_forbidden

    cell = 0.7
    m = _mask(3, 10)
    m[:, 2] = 1  # Block column 2
    w = 3 * cell
    x_from = 3 * cell

    x = _next_free_x(m, x_from, 0.0, w, 3.0, cell, 20.0, 1e-6)
    # Should find a clear position: the function returns 2.8, i.e. column 4
    # (spanning 4..6), not columns 3/4/5 as the width might suggest.
    assert x is not None, "Should find a valid position"
    # Must actually clear the mask.
    assert not _overlaps_forbidden(m, x, 0.0, w, 3.0, cell), \
        f"Position {x} must clear the mask"
    assert x >= x_from - 1e-6


@pytest.mark.parametrize("mm_per_cell", [0.7, 0.3, 1/3])
def test_next_free_x_non_binary_cell_sizes_property(mm_per_cell):
    """Property test over non-binary cell sizes that trigger IEEE-754 rounding.

    Binary-exact cell sizes (like 1.0, 1.5) hide rounding bugs in the
    float-to-column conversion. Non-binary sizes (0.7, 0.3, 1/3) expose them.
    Keep fractional x_from and all invariant assertions.

    Dimensions are scaled by mm_per_cell (not absolute) so the geometry
    stays coherent as the cell size varies: a 30-column, 30 mm-wide mask
    with an absolute w=5.0 needs 17 columns at mm_per_cell=0.3, and
    x_from up to 25.0 lands past column 83 of a 30-column mask, so the
    scan finds nothing and the body below never runs. Scaling keeps the
    item/mask/x_from relationship — and thus the hit rate — constant
    across cell sizes.
    """
    from common.packing import _overlaps_forbidden

    rng = np.random.default_rng(42)
    strip_width = 40 * mm_per_cell
    w = 2.5 * mm_per_cell
    h = 2.0 * mm_per_cell
    hits = 0

    for _ in range(200):
        m = (rng.random((12, 40)) < 0.15).astype(np.uint8)
        x_from = rng.random() * 10 * mm_per_cell
        x = _next_free_x(m, x_from, 0.0, w, h, mm_per_cell, strip_width, 1e-6)
        if x is not None:
            hits += 1
            # All three properties must hold.
            assert not _overlaps_forbidden(m, x, 0.0, w, h, mm_per_cell), \
                f"mm_per_cell={mm_per_cell}: x={x} must not overlap"
            assert x >= x_from - 1e-6, \
                f"mm_per_cell={mm_per_cell}: x={x} must be >= x_from={x_from}"
            assert x + w <= strip_width + 1e-6, \
                f"mm_per_cell={mm_per_cell}: x={x} must fit in strip"

    # Anti-vacuity guard: without this, a future dimension tweak could
    # silently re-hollow the test (drive hits to 0) and nothing would
    # complain, since `if x is not None:` bodies that never run still
    # make the test pass.
    assert hits > 20, f"test is vacuous: only {hits}/200 iterations exercised the assertions"


def test_shelf_survives_an_obstacle_instead_of_being_abandoned():
    """The FDR §6.3.1 defect, reduced to its smallest reproduction.

    A 100 x 10 mm strip with a 10 mm obstacle at x=30..40. Item 0 opens
    the shelf at x=0 unobstructed. Item 1's cursor lands at x=20, whose
    footprint spans 20..40 and hits the obstacle: the old code abandoned
    the whole shelf at that point, so items 1 and 2 were lost. The fix
    advances the cursor past the obstacle and keeps packing.

    The obstacle is deliberately NOT at x=0. An obstacle at the strip's
    left edge is hit while OPENING a shelf, which is the new-shelf
    branch's failure mode and is Task 3's test to write.
    """
    mask = _mask(10, 100, [(0, 10, 30, 40)])
    items = [Item(id=i, width=20.0, height=10.0) for i in range(3)]
    res = first_fit_decreasing(
        items, 100.0, 10.0, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count == 3
    assert sorted(p.x for p in res.placements) == [0.0, 40.0, 60.0]
    assert len({p.y for p in res.placements}) == 1, \
        "all three must land on ONE shelf; a second shelf means the first " \
        "was abandoned rather than advanced past"


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
