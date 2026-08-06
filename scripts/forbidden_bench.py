"""Forbidden-mask FFDH vs rejection-sampling FFDH — the FDR §6.3.1 benchmark.

Regenerates ``docs/receipts/forbidden_bench.csv`` and
``docs/receipts/forbidden_bench.txt``.

Two arms are compared on identical masks:

*aware*
    :func:`common.packing.first_fit_decreasing` called with
    ``forbidden_mask=mask`` — the obstacle-aware variant that is the
    project's algorithmic contribution.
*naive*
    The same function called with ``forbidden_mask=None`` — i.e. stock
    FFDH with no obstacle awareness — followed by discarding every
    placement that overlaps the mask. This is *rejection sampling*: it
    is the trivial baseline the aware arm has to beat to justify
    existing.

Parameters are those recorded in FDR §6.3.1: a 200 x 150 mm strip, 40
candidate 18.5 x 65 mm items, 1.5 mm cells, 40 random masks per
coverage level, masks drawn as small rectangular blobs.

Provenance note. The original generator behind the pre-existing receipt
was never committed; only its output was. This script is a
reconstruction, and its mask-generation parameters were recovered from
that output rather than assumed:

* the mask grid is 100 x 134 cells — ``round(actual_cov * 13400)`` is
  integral for all 240 recorded rows, and 13400 = ceil(150 / 1.5) *
  ceil(200 / 1.5);
* :data:`_BLOB_HI` and :data:`_ASSUMED_BLOB_AREA` were fixed by matching
  both the mean and the standard deviation of the recorded forbidden-cell
  counts at all five non-zero coverage levels (every level lands within
  1.2 sigma; see the task-4 report). Note that ``_BLOB_HI`` is the
  exclusive upper bound of :meth:`numpy.random.Generator.integers`, so
  blob sides are 2..5 cells — the "2-6 cell blobs" of FDR §6.3.1 names
  the call's arguments, not its support.

Run with ``python scripts/forbidden_bench.py``.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.packing import (  # noqa: E402
    Item,
    _overlaps_forbidden,
    first_fit_decreasing,
)

# ------------------------------------------------------------ config ----

STRIP_W = 200.0
STRIP_H = 150.0
MM_PER_CELL = 1.5

ITEM_W = 18.5
ITEM_H = 65.0
N_ITEMS = 40

N_SEEDS = 40
COVERAGES: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15, 0.25)

# Blob side lengths are drawn from ``rng.integers(_BLOB_LO, _BLOB_HI)``,
# i.e. uniform on {2, 3, 4, 5}.
_BLOB_LO = 2
_BLOB_HI = 6

# Blob count is sized off the *largest* blob a draw can produce
# (5 x 5 = 25 cells), not its mean, so realised coverage runs below the
# nominal target. The ``actual_cov`` column records what was actually
# masked; ``target_cov`` is only the knob.
_ASSUMED_BLOB_AREA = 25

MASTER_SEED = 20260806

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "docs" / "receipts" / "forbidden_bench.csv"
TXT_PATH = REPO / "docs" / "receipts" / "forbidden_bench.txt"

N_ROWS = int(np.ceil(STRIP_H / MM_PER_CELL))
N_COLS = int(np.ceil(STRIP_W / MM_PER_CELL))


# ------------------------------------------------------------- masks ----

def make_mask(rng: np.random.Generator, coverage: float) -> np.ndarray:
    """Random forbidden mask of small rectangular blobs.

    Blobs are placed independently and may overlap, so the realised
    coverage is at or below ``n_blobs * mean_blob_area``.
    """
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    n_blobs = int(coverage * N_ROWS * N_COLS / _ASSUMED_BLOB_AREA)
    for _ in range(n_blobs):
        h = int(rng.integers(_BLOB_LO, _BLOB_HI))
        w = int(rng.integers(_BLOB_LO, _BLOB_HI))
        r = int(rng.integers(0, N_ROWS - h))
        c = int(rng.integers(0, N_COLS - w))
        mask[r:r + h, c:c + w] = True
    return mask


# -------------------------------------------------------------- arms ----

def build_items() -> List[Item]:
    return [Item(id=i, width=ITEM_W, height=ITEM_H) for i in range(N_ITEMS)]


def run_aware(items: List[Item], mask: np.ndarray) -> Tuple[int, float]:
    """Obstacle-aware FFDH. Returns (placed, microseconds)."""
    t0 = time.perf_counter()
    res = first_fit_decreasing(
        items, STRIP_W, STRIP_H,
        forbidden_mask=mask, mm_per_cell=MM_PER_CELL,
    )
    us = (time.perf_counter() - t0) * 1e6
    return res.count, us


def run_naive(items: List[Item], mask: np.ndarray) -> Tuple[int, float]:
    """Stock FFDH then discard mask-overlapping placements."""
    t0 = time.perf_counter()
    res = first_fit_decreasing(items, STRIP_W, STRIP_H)
    survivors = sum(
        1 for p in res.placements
        if not _overlaps_forbidden(
            mask, p.x, p.y, p.width, p.height, MM_PER_CELL,
        )
    )
    us = (time.perf_counter() - t0) * 1e6
    return survivors, us


def check_valid(mask: np.ndarray, placements) -> None:
    """Assert no placement overlaps the mask or another placement."""
    for i, p in enumerate(placements):
        assert not _overlaps_forbidden(
            mask, p.x, p.y, p.width, p.height, MM_PER_CELL,
        ), "aware arm placed an item on a forbidden cell"
        for q in placements[i + 1:]:
            sep = (
                p.x + p.width <= q.x or q.x + q.width <= p.x
                or p.y + p.height <= q.y or q.y + q.height <= p.y
            )
            assert sep, "aware arm produced overlapping placements"


# --------------------------------------------------------------- run ----

def main() -> None:
    items = build_items()
    rows = []

    # Warm the interpreter so the first timed call is not an outlier.
    for _ in range(5):
        first_fit_decreasing(items, STRIP_W, STRIP_H)

    for level, coverage in enumerate(COVERAGES):
        for seed in range(N_SEEDS):
            rng = np.random.default_rng([MASTER_SEED, level, seed])
            mask = make_mask(rng, coverage)
            actual = float(mask.sum()) / mask.size

            n_aware, us_aware = run_aware(items, mask)
            n_naive, us_naive = run_naive(items, mask)

            aware_res = first_fit_decreasing(
                items, STRIP_W, STRIP_H,
                forbidden_mask=mask, mm_per_cell=MM_PER_CELL,
            )
            check_valid(mask, aware_res.placements)

            rows.append({
                "target_cov": coverage,
                "actual_cov": actual,
                "seed": seed,
                "n_aware": n_aware,
                "n_naive": n_naive,
                "us_aware": us_aware,
                "us_naive": us_naive,
            })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "target_cov", "actual_cov", "seed",
            "n_aware", "n_naive", "us_aware", "us_naive",
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Forbidden-mask FFDH vs rejection-sampling FFDH",
        "=" * 64,
        "   cov   aware mean   naive mean     gain   aware us   naive us",
    ]
    for coverage in COVERAGES:
        sub = [r for r in rows if r["target_cov"] == coverage]
        a = sum(r["n_aware"] for r in sub) / len(sub)
        n = sum(r["n_naive"] for r in sub) / len(sub)
        ua = sum(r["us_aware"] for r in sub) / len(sub)
        un = sum(r["us_naive"] for r in sub) / len(sub)
        lines.append(
            "%5.1f%%%13.2f%13.2f%+9.2f%11.1f%11.1f"
            % (coverage * 100, a, n, a - n, ua, un)
        )
    with TXT_PATH.open("w", newline="\n", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
