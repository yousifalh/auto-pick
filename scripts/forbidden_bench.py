"""Forbidden-mask FFDH vs rejection-sampling FFDH — the FDR §6.3.1 benchmark.

Regenerates ``docs/receipts/forbidden_bench.csv`` and
``docs/receipts/forbidden_bench.txt``. Also writes
``docs/receipts/forbidden_bench_timings.csv``, which is gitignored: the
counts in the main CSV are fully deterministic (fixed seeds throughout),
so that file verifies byte-identical across regenerations, but wall-clock
timings are not deterministic and were previously interleaved into the
same CSV — producing a spurious diff on every regeneration with no way to
tell a real change from noise. The timings live in the separate,
uncommitted file instead; the human-readable summary in
``forbidden_bench.txt`` still reports the per-level mean timings.

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

Run with ``python scripts/forbidden_bench.py`` to regenerate the
committed receipt. ``--seed N`` re-runs under a different master seed
for sensitivity checking; any non-default seed prints without writing,
so a sweep cannot overwrite the committed receipt.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
# Wall-clock timings, not committed (see module docstring): they are
# non-deterministic, so keeping them out of CSV_PATH is what lets CSV_PATH
# verify byte-identical across regenerations.
TIMINGS_CSV_PATH = REPO / "docs" / "receipts" / "forbidden_bench_timings.csv"

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

def collect(master_seed: int) -> List[dict]:
    """Run both arms over every (coverage, seed) pair."""
    items = build_items()
    rows: List[dict] = []

    # Warm the interpreter so the first timed call is not an outlier.
    for _ in range(5):
        first_fit_decreasing(items, STRIP_W, STRIP_H)

    for level, coverage in enumerate(COVERAGES):
        for seed in range(N_SEEDS):
            rng = np.random.default_rng([master_seed, level, seed])
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

    return rows


def paired_t(diffs: List[int]) -> float:
    """Paired t against H0: mean difference = 0. NaN when all diffs equal."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0.0:
        return float("nan")
    return mean / ((var / n) ** 0.5)


def summarise(rows: List[dict]) -> List[str]:
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

    # Paired within-run statistics. Both arms see identical masks, so the
    # per-seed difference is the primary evidence: it is unaffected by how
    # faithfully the mask generator reconstructs the original one.
    lines += [
        "",
        "Paired within-run comparison (same mask for both arms, 40 seeds)",
        "=" * 64,
        "   cov     mean diff    paired t   win/tie/loss",
    ]
    for coverage in COVERAGES:
        sub = [r for r in rows if r["target_cov"] == coverage]
        diffs = [r["n_aware"] - r["n_naive"] for r in sub]
        t = paired_t(diffs)
        wins = sum(1 for d in diffs if d > 0)
        ties = sum(1 for d in diffs if d == 0)
        loss = sum(1 for d in diffs if d < 0)
        t_txt = "     n/a" if t != t else "%8.2f" % t
        lines.append(
            "%5.1f%%%14.3f    %s   %d/%d/%d"
            % (coverage * 100, sum(diffs) / len(diffs), t_txt,
               wins, ties, loss)
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed", type=int, default=MASTER_SEED,
        help="master seed (default %(default)s — the committed receipt)",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="print the summary without touching the receipt files; "
             "implied whenever --seed differs from the default, so a "
             "sensitivity sweep cannot clobber the committed receipt",
    )
    args = parser.parse_args(argv)

    rows = collect(args.seed)
    lines = summarise(rows)

    write = not args.no_write and args.seed == MASTER_SEED
    if write:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic columns only, so this file verifies byte-identical
        # across regenerations. Timings go to TIMINGS_CSV_PATH instead.
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "target_cov", "actual_cov", "seed", "n_aware", "n_naive",
            ], extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with TIMINGS_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "target_cov", "seed", "us_aware", "us_naive",
            ], extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with TXT_PATH.open("w", newline="\n", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        lines.insert(1, "(seed %d - not written to the receipt)" % args.seed)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
