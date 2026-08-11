"""The packing ceiling: FFDH's shelf origin never scans in y.

Companion to ``docs/superpowers/specs/2026-08-11-packing-ceiling.md``.

FFDH pins its first shelf to y = 0 and every later shelf to the top of
the previous one. ``_next_free_x`` collapses a shelf's whole row band
with ``.any(axis=0)`` — exact for the band, but it means one mostly
blocked row poisons every column in it. A cartridge wall on grid row 0
is therefore enough to stop any shelf ever opening, so every identical
item fails identically and the packer returns zero on a nearly empty
grid.

These tests run on bare grids and masks — no renderer, no checkpoint.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from common.packing import (
    Item,
    PackedItem,
    PackResult,
    _drop_unsafe,
    _grid_greedy,
    _overlaps_forbidden,
    _shelf_scan,
    first_fit_decreasing,
    pack_best_effort,
)

CELL = 1.5
BATTERY_W = 18.5
BATTERY_H = 65.0


# ------------------------------------------------------------ helpers ----

def _cells(n_items: int = 24, w: float = BATTERY_W, h: float = BATTERY_H):
    return [Item(id=i, width=w, height=h) for i in range(n_items)]


def _scene_00005_mask() -> np.ndarray:
    """Structural reproduction of the measured ``scene_00005`` cartridge.

    The real instance is a 62 x 123 grid of 1.5 mm cells over a
    93.1 x 185.0 mm strip, 93.4 % free, whose forbidden cells are the
    cartridge wall along row 0 and column 0 plus two interior
    obstructions. Reproduced rather than embedded as a 7626-cell blob:
    the wall is the whole mechanism, and a literal nobody can read is a
    fixture nobody can check.
    """
    mask = np.zeros((123, 62), dtype=bool)
    mask[0, :] = True           # top wall — the row that poisons the band
    mask[:, 0] = True           # left wall
    mask[40:46, 42:54] = True   # interior obstruction
    mask[76:84, 22:26] = True   # interior obstruction
    return mask


SCENE_W, SCENE_H = 93.125, 185.0


def _assert_safe(res: PackResult, mask, cell, strip_w, strip_h, tol=1e-6):
    """The safety property, asserted in full: mask, overlap, and bounds."""
    for i, p in enumerate(res.placements):
        if mask is not None:
            assert not _overlaps_forbidden(
                mask, p.x, p.y, p.width, p.height, cell,
            ), f"placement at ({p.x}, {p.y}) overlaps a forbidden cell"
        assert p.x >= -tol and p.y >= -tol, f"negative corner {p.x},{p.y}"
        assert p.x + p.width <= strip_w + tol, "placement runs off the right"
        assert p.y + p.height <= strip_h + tol, "placement runs off the bottom"
        for q in res.placements[i + 1:]:
            apart = (
                p.x + p.width <= q.x + tol or q.x + q.width <= p.x + tol
                or p.y + p.height <= q.y + tol or q.y + q.height <= p.y + tol
            )
            assert apart, f"({p.x},{p.y}) overlaps ({q.x},{q.y})"


# ------------------------------------------------- the defect, pinned ----

def test_ffdh_places_nothing_on_the_scene_00005_grid():
    """The defect itself. A wall on row 0 costs the entire pack."""
    mask = _scene_00005_mask()
    assert mask.mean() < 0.05, "the grid really is almost entirely free"
    res = first_fit_decreasing(
        _cells(), SCENE_W, SCENE_H,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count == 0


def test_the_same_grid_admits_cells_one_cell_lower():
    """Not a full strip: shifting the shelf down by ONE cell clears it."""
    mask = _scene_00005_mask()
    # y = 0 is blocked in every column of the 44-row band...
    assert _overlaps_forbidden(mask, 0.0, 0.0, BATTERY_W, BATTERY_H, CELL)
    # ...and y = 1.5 mm, one cell lower, is not.
    assert not _overlaps_forbidden(mask, CELL, 0.0, BATTERY_W, BATTERY_H, CELL) \
        or not _overlaps_forbidden(mask, CELL, CELL, BATTERY_W, BATTERY_H, CELL)


def test_pack_best_effort_recovers_the_scene_00005_grid():
    """The fix. Measured on the real frame: 0 -> 7."""
    mask = _scene_00005_mask()
    res = pack_best_effort(
        _cells(), SCENE_W, SCENE_H,
        forbidden_mask=mask, mm_per_cell=CELL,
    )
    assert res.count >= 6, f"only recovered {res.count} cells"
    _assert_safe(res, mask, CELL, SCENE_W, SCENE_H)


def test_one_blocked_row_is_the_whole_mechanism():
    """Minimal reproduction: a single forbidden row at the top."""
    mask = np.zeros((60, 40), dtype=bool)
    mask[0, :] = True
    items = _cells(4, w=10.0, h=20.0)
    strip_w, strip_h = 40.0, 60.0
    assert first_fit_decreasing(
        items, strip_w, strip_h, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=1.0,
    ).count == 0
    fixed = pack_best_effort(
        items, strip_w, strip_h, allow_rotation=False,
        forbidden_mask=mask, mm_per_cell=1.0,
    )
    assert fixed.count == 4
    _assert_safe(fixed, mask, 1.0, strip_w, strip_h)


# ------------------------------------------------ no-regression by construction --

def test_pack_best_effort_never_places_fewer_than_ffdh():
    """The property that makes this safe to ship. Fuzzed over masks."""
    rng = np.random.default_rng(20260811)
    for seed in range(120):
        rows, cols = int(rng.integers(20, 90)), int(rng.integers(20, 90))
        mask = np.zeros((rows, cols), dtype=bool)
        for _ in range(int(rng.integers(0, 25))):
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            r, c = int(rng.integers(0, rows - h)), int(rng.integers(0, cols - w))
            mask[r:r + h, c:c + w] = True
        cell = float(rng.choice([0.7, 1.0, 1.5, 2.3]))
        sw, sh = cols * cell, rows * cell
        iw = float(rng.choice([6.0, 8.3, 18.3, 18.5]))
        ih = float(rng.choice([9.0, 12.7, 20.0, 65.0]))
        items = [Item(id=i, width=iw, height=ih) for i in range(12)]
        a = first_fit_decreasing(items, sw, sh, forbidden_mask=mask,
                                 mm_per_cell=cell)
        b = pack_best_effort(items, sw, sh, forbidden_mask=mask,
                             mm_per_cell=cell)
        assert b.count >= a.count, (
            f"seed {seed}: best-effort {b.count} < ffdh {a.count} "
            f"(cell {cell}, item {iw}x{ih}, grid {rows}x{cols})"
        )


def test_an_unimprovable_instance_returns_ffdhs_own_placements():
    """Ties go to FFDH, so a queue nothing improves does not move."""
    items = _cells(12)
    a = first_fit_decreasing(items, 200.0, 150.0)
    b = pack_best_effort(items, 200.0, 150.0)
    assert [(p.item.id, p.x, p.y, p.rotated) for p in a.placements] == \
           [(p.item.id, p.x, p.y, p.rotated) for p in b.placements]


def test_ffdh_itself_is_untouched_on_a_clean_strip():
    """recog.synth3d lays out synthetic scenes with this exact output.

    Pinned here, not only in test_bin_packing.py, because the whole
    argument for adding strategies rather than replacing FFDH is that
    changing it would silently redraw a training corpus.
    """
    items = _cells(40)
    assert first_fit_decreasing(items, 200.0, 150.0).count == 23


# ------------------------------------------------------ safety, adversarial --

def test_a_fully_forbidden_grid_places_nothing():
    mask = np.ones((60, 40), dtype=bool)
    res = pack_best_effort(_cells(8, 10.0, 20.0), 40.0, 60.0,
                           forbidden_mask=mask, mm_per_cell=1.0)
    assert res.count == 0
    assert len(res.unplaced_ids) == 8


def test_a_rotation_wider_than_the_strip_is_rejected():
    """The strip bound is checked in BOTH orientations.

    An earlier prototype of the y-scanning arm dropped this check and
    happily placed a 65 mm-wide rotated cell into a 50 mm strip — the
    mask said clear because the grid simply ended before the item did.
    """
    mask = np.zeros((29, 33), dtype=bool)
    mask[0, :] = True
    res = pack_best_effort(_cells(4), 50.0, 44.375,
                           forbidden_mask=mask, mm_per_cell=CELL)
    for p in res.placements:
        assert p.x + p.width <= 50.0 + 1e-6
        assert p.y + p.height <= 44.375 + 1e-6


def test_a_mask_smaller_than_the_strip_does_not_grant_free_space():
    """Cells beyond the mask's extent must not be treated as placeable."""
    # 40 x 40 mm strip but only a 10 x 10 cell (15 x 15 mm) mask, fully set.
    mask = np.ones((10, 10), dtype=bool)
    res = pack_best_effort(_cells(4, 6.0, 6.0), 40.0, 40.0,
                           forbidden_mask=mask, mm_per_cell=CELL)
    for p in res.placements:
        assert not _overlaps_forbidden(mask, p.x, p.y, p.width, p.height, CELL)


@pytest.mark.parametrize("cell", [0.7, 1.0, 1.5, 2.3])
@pytest.mark.parametrize("item_w", [18.3, 18.5])
def test_awkward_sizes_never_breach_the_mask(cell, item_w):
    """Non-power-of-two cells and item widths, which have bitten before.

    A previous fix's correctness argument recovered a column index via
    ``int(x / mm_per_cell)`` and was false for exactly these sizes; the
    fuzz test that missed it used only binary-exact ones.
    """
    rng = np.random.default_rng(hash((cell, item_w)) % (2 ** 32))
    for _ in range(25):
        rows, cols = 55, 70
        mask = (rng.random((rows, cols)) < 0.06)
        sw, sh = cols * cell, rows * cell
        items = [Item(id=i, width=item_w, height=41.7) for i in range(10)]
        res = pack_best_effort(items, sw, sh, forbidden_mask=mask,
                               mm_per_cell=cell)
        _assert_safe(res, mask, cell, sw, sh)


@pytest.mark.parametrize("arm", [first_fit_decreasing, _shelf_scan,
                                 _grid_greedy, pack_best_effort])
def test_every_arm_is_safe_on_its_own(arm):
    """Each strategy must hold the invariant unaided, not via the net."""
    rng = np.random.default_rng(4242)
    for _ in range(40):
        mask = (rng.random((80, 60)) < 0.10)
        sw, sh = 60 * CELL, 80 * CELL
        items = _cells(10, 12.4, 27.9)
        res = arm(items, sw, sh, forbidden_mask=mask, mm_per_cell=CELL)
        _assert_safe(res, mask, CELL, sw, sh)


def test_drop_unsafe_removes_a_placement_that_breaches_the_mask():
    """The safety net fires when handed a bad placement."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    bad = PackResult(
        placements=[PackedItem(Item(0, 3.0, 3.0), x=7.5, y=7.5)],
        unplaced_ids=[], shelf_heights=[],
    )
    out = _drop_unsafe(bad, mask, CELL, 1e-6)
    assert out.count == 0
    assert out.unplaced_ids == [0]


def test_drop_unsafe_removes_a_placement_that_overlaps_another():
    a = PackedItem(Item(0, 10.0, 10.0), x=0.0, y=0.0)
    b = PackedItem(Item(1, 10.0, 10.0), x=5.0, y=5.0)
    out = _drop_unsafe(PackResult([a, b], [], []), None, CELL, 1e-6)
    assert out.count == 1
    assert out.unplaced_ids == [1]


def test_drop_unsafe_is_inert_on_a_clean_result():
    res = first_fit_decreasing(_cells(12), 200.0, 150.0)
    assert _drop_unsafe(res, None, CELL, 1e-6) is res


# ------------------------------------------------------------ contracts --

def test_allow_rotation_false_is_honoured_by_every_arm():
    mask = np.zeros((60, 40), dtype=bool)
    mask[0, :] = True
    for arm in (_shelf_scan, _grid_greedy, pack_best_effort):
        res = arm([Item(0, 70.0, 10.0)], 50.0, 60.0, allow_rotation=False,
                  forbidden_mask=mask, mm_per_cell=1.0)
        assert res.count == 0, f"{arm.__name__} rotated despite the flag"


def test_result_accounts_for_every_item():
    mask = _scene_00005_mask()
    items = _cells(24)
    res = pack_best_effort(items, SCENE_W, SCENE_H,
                           forbidden_mask=mask, mm_per_cell=CELL)
    assert res.count + len(res.unplaced_ids) == len(items)
    assert set(res.unplaced_ids) | {p.item.id for p in res.placements} == \
           {i.id for i in items}


def test_no_mask_at_all_is_accepted_by_every_arm():
    for arm in (_shelf_scan, _grid_greedy, pack_best_effort):
        res = arm(_cells(4, 10.0, 10.0), 30.0, 30.0)
        assert res.count == 4, arm.__name__


def test_degenerate_strip_places_nothing_rather_than_raising():
    for arm in (_shelf_scan, _grid_greedy, pack_best_effort):
        assert arm(_cells(4), 0.0, 0.0).count == 0, arm.__name__


def test_stays_inside_the_o3_latency_budget():
    """O3: queue rebuild <= 8 ms per cartridge (FDR v3 §10.4).

    Sized to the largest cartridge observed in the corpus (93 x 185 mm,
    62 x 123 cells) rather than to a toy grid.
    """
    mask = _scene_00005_mask()
    items = _cells(24)
    pack_best_effort(items, SCENE_W, SCENE_H,
                     forbidden_mask=mask, mm_per_cell=CELL)  # warm
    t0 = time.perf_counter()
    for _ in range(10):
        pack_best_effort(items, SCENE_W, SCENE_H,
                         forbidden_mask=mask, mm_per_cell=CELL)
    ms = (time.perf_counter() - t0) / 10 * 1e3
    assert ms < 8.0, f"{ms:.2f} ms exceeds the 8 ms O3 budget"
