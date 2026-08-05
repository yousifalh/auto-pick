"""FFDH correctness tests.

Verifies the core algorithmic guarantees from PPR §5.3.3:

* No two placements overlap.
* No placement leaves the strip.
* Items sorted by decreasing height are tried first.
* The 1.7×OPT bound is respected for a hand-crafted instance
  (we check a trivial instance where OPT is obvious).
* Rotation is used when it enables fit.
* Forbidden cells are respected.
"""
from __future__ import annotations

import numpy as np
import pytest

from plan.bin_packing import (
    Item, PackedItem, PackResult, first_fit_decreasing,
)


def _overlap(a: PackedItem, b: PackedItem) -> bool:
    return not (a.x + a.width <= b.x
                or b.x + b.width <= a.x
                or a.y + a.height <= b.y
                or b.y + b.height <= a.y)


def _assert_no_overlaps(result: PackResult):
    for i, a in enumerate(result.placements):
        for b in result.placements[i + 1:]:
            assert not _overlap(a, b), f"overlap between {a} and {b}"


def _assert_in_strip(result: PackResult, w: float, h: float, tol=1e-6):
    for p in result.placements:
        assert p.x >= -tol and p.y >= -tol
        assert p.x + p.width <= w + tol
        assert p.y + p.height <= h + tol


# --------------------------- basic cases ----------------------------------

def test_single_item_fits():
    res = first_fit_decreasing([Item(0, 10, 10)], 100, 100)
    assert res.count == 1
    assert res.placements[0].x == 0
    assert res.placements[0].y == 0


def test_empty_input():
    res = first_fit_decreasing([], 100, 100)
    assert res.count == 0
    assert res.unplaced_ids == []


def test_too_big_item_unplaced():
    res = first_fit_decreasing([Item(0, 200, 10)], 100, 100,
                               allow_rotation=False)
    assert res.count == 0
    assert res.unplaced_ids == [0]


def test_row_of_identical_items():
    items = [Item(i, 10, 10) for i in range(5)]
    res = first_fit_decreasing(items, 100, 20)
    assert res.count == 5
    _assert_no_overlaps(res)
    _assert_in_strip(res, 100, 20)


def test_many_identical_items_no_overlap():
    """Pack a large batch and verify the no-overlap invariant."""
    items = [Item(i, 18.5, 65.0) for i in range(40)]
    res = first_fit_decreasing(items, 200, 150,
                               allow_rotation=True)
    _assert_no_overlaps(res)
    _assert_in_strip(res, 200, 150)
    # We expect more than a trivial number to fit
    assert res.count >= 6


def test_forbidden_mask_blocks_placement():
    # Forbid the entire strip → nothing fits
    mask = np.ones((100, 100), dtype=bool)
    items = [Item(i, 10, 10) for i in range(4)]
    res = first_fit_decreasing(items, 100, 100,
                               forbidden_mask=mask, mm_per_cell=1.0)
    assert res.count == 0
    assert len(res.unplaced_ids) == 4


def test_forbidden_mask_partial():
    # Forbid the right-half of the strip — items should still pack along
    # the left edge (x < 50) but never start in the forbidden region.
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 50:] = True
    items = [Item(i, 10, 10) for i in range(5)]
    res = first_fit_decreasing(items, 100, 100,
                               forbidden_mask=mask, mm_per_cell=1.0)
    # At least one item packs in the free zone
    assert res.count >= 1
    for p in res.placements:
        # No placement is fully inside the forbidden zone
        assert p.x < 50


def test_rotation_enables_fit():
    # A 70×10 item doesn't fit in a 50-wide strip, but rotated (10×70)
    # it does — as long as the strip is tall enough.
    res = first_fit_decreasing([Item(0, 70, 10)], strip_width=50,
                               strip_height=100, allow_rotation=True)
    assert res.count == 1
    assert res.placements[0].rotated is True


def test_rotation_disabled_rejects_wide_item():
    res = first_fit_decreasing([Item(0, 70, 10)], strip_width=50,
                               strip_height=100, allow_rotation=False)
    assert res.count == 0


def test_decreasing_height_ordering():
    items = [Item(0, 10, 20), Item(1, 10, 50), Item(2, 10, 10)]
    res = first_fit_decreasing(items, 100, 100)
    # First shelf height should be that of the tallest item (50)
    assert res.shelf_heights[0] == 50


def test_packed_item_rotation_properties():
    it = Item(0, 10, 30)
    p = PackedItem(it, x=0, y=0, rotated=True)
    assert p.width == 30
    assert p.height == 10
    p2 = PackedItem(it, x=0, y=0, rotated=False)
    assert p2.width == 10 and p2.height == 30


def test_opt_bound_trivial():
    """For N identical squares fitting exactly in a grid, FFDH is optimal."""
    items = [Item(i, 10, 10) for i in range(9)]
    res = first_fit_decreasing(items, 30, 30, allow_rotation=False)
    # OPT = 9; FFDH should find all 9
    assert res.count == 9
    _assert_no_overlaps(res)


def test_shelf_heights_populated():
    items = [Item(i, 10, 10) for i in range(30)]
    res = first_fit_decreasing(items, 30, 100)
    assert len(res.shelf_heights) >= 1
    assert all(h > 0 for h in res.shelf_heights)
