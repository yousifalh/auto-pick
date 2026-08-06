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
    """Whatever it returns must satisfy _overlaps_forbidden == False."""
    from common.packing import _overlaps_forbidden

    rng = np.random.default_rng(0)
    for _ in range(200):
        m = (rng.random((12, 30)) < 0.15).astype(np.uint8)
        x = _next_free_x(m, 0.0, 0.0, 5.0, 4.0, CELL, 30.0, 1e-6)
        if x is not None:
            assert not _overlaps_forbidden(m, x, 0.0, 5.0, 4.0, CELL)
