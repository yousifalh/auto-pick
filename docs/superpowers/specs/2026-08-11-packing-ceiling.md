# The packing ceiling: FFDH's shelf origin never scans in y

Date: 2026-08-11
Baseline: `12134c2`, 678 tests passing.
Area: `common/packing.py`, `plan/bin_packing.py`, `plan/planner.py`,
`scripts/forbidden_bench.py`, `tests/test_packing_ceiling.py`.

Perception improved to the point that the packer became the binding
constraint. On frame `scene_00005` the packer was handed a grid that is
**93.4 % free**, containing a clear **48 × 112 mm** rectangle, and placed
**zero** 18.5 × 65 mm cells.

---

## 1. Diagnosis

The brief's hypothesis was that shelf packing loses a usable rectangle to
*fragmentation* — shelves span the full width, so a forbidden region
splits the space. **The measurement says something narrower and more
specific, and the difference matters for the fix.**

### The mechanism

`first_fit_decreasing` opens its first shelf at `y = 0` and every later
shelf at `last_y` = the top of the previous shelf. **It never scans in
y.** There is exactly one candidate offset per new shelf.

`_next_free_x` then collapses the shelf's whole row band:

```python
blocked = mask[r1:r2, :].any(axis=0)
```

That is *exact* for a shelf — an item of height `h` really does span all
those rows — but it means **one mostly-blocked row poisons every column
in the band**.

Put the two together and the cartridge wall is fatal. Measured on the
real `scene_00005` instance (62 × 123 cells at 1.5 mm, 93.1 × 185.0 mm
strip, 505 forbidden cells of 7626):

| orientation | band rows | columns blocked | longest clear run | needed |
| --- | --- | --- | --- | --- |
| upright 18.5 × 65 | `[0, 44)` | **62 / 62** | 0 | 13 |
| rotated 65 × 18.5 | `[0, 13)` | **61 / 62** | 1 | 44 |

`_next_free_x` returns `None` for both orientations. No shelf opens, so
`shelves` stays empty, so `last_y` stays `0.0` — and since the planner
hands FFDH 24 *identical* items, all 24 fail identically. Zero placed.

The grid row responsible is row 0: the cartridge wall the segmentation
extractor rasterises. **The very next candidate offset would have
worked**: 79 of the 80 admissible shelf origins on that grid accept an
upright cell, the first at `y = 1.5 mm`. The packer was one cell away
from working and had no way to take the step.

So the defect is not fragmentation. It is that **the shelf origin is
pinned**, and a single boundary row is enough to trigger it. Fragmentation
is real and costs space too (§2), but it is the second-order term.

### How often it bites

Two independent axes, both showing the same signature.

**(a) Real frames.** 60 frames of `recog/dataset3d_seg` through the real
detector + segmenter + `SegmentationPlacementAreaExtractor`, dumping the
exact `(strip_w, strip_h, mask, mm_per_cell)` each cartridge hands the
packer: 30 packing instances.

Reference = best over a family of *achievable* greedy packings (four
deterministic rules plus 30 randomised restarts), so every reference
count is realisable — a lower bound on OPT, not an area estimate.

| | instances |
| --- | --- |
| total | 30 |
| can hold ≥ 1 cell at all | **7** |
| FFDH places fewer than the reference | **3 of those 7** |
| total cells: FFDH vs reference | **8 vs 18** (44 %) |

The other 23 instances are cartridges too small for an 18.5 × 65 cell in
any orientation. That is a perception/geometry fact, not a packer fact,
and it is stated here so the 8-vs-18 is not read as 8-of-30.

**(b) The project's own published benchmark.** `docs/receipts/
forbidden_bench.txt` already contained the evidence; it had simply never
been read as a defect, because the comparison on offer was against an
even worse baseline.

| coverage | FFDH | achievable | FFDH as % |
| --- | --- | --- | --- |
| 0 % | 23.00 | 21.00 | 110 % |
| 2.5 % | 14.28 | 15.12 | 94 % |
| 5 % | 9.05 | 11.60 | 78 % |
| 10 % | 2.60 | 6.08 | **43 %** |
| 15 % | 0.57 | 3.15 | **18 %** |
| 25 % | 0.03 | 0.35 | **7 %** |

On a 200 × 150 mm strip that is still 75 % free, FFDH places 0.03 cells.
FFDH *beats* the greedy reference on a clean strip (23 vs 21) — that
number is what stopped this becoming a naive replace-FFDH change.

One dramatic frame would not justify replacing an algorithm. A monotone
collapse across two unrelated corpora does.

---

## 2. The fix

Three findings constrained it:

1. FFDH is **best on a clean or lightly obstructed strip** and must not
   be lost (23 vs 21 at 0 % coverage; it wins outright on 4 of the 7
   capable real instances).
2. A cell-quantised grid packer — the obvious "maximal rectangles"
   answer — **regresses** on clean strips: 23 → 20 at 0 % coverage and
   14.28 → 13.60 at 2.5 %. It snaps positions to cell boundaries and
   rounds footprints up to whole cells, losing 1.0 mm per column and per
   row, which costs the third shelf outright. Measured, not assumed.
3. `first_fit_decreasing` is **load-bearing elsewhere**:
   `recog.synth3d.bay` and `recog.synth3d.layout` lay out synthetic
   scenes with it, and `docs/FDR_v3.md` §6.3.1 reproduces it as
   pseudocode. Changing it in place would silently redraw a training
   corpus — which the brief forbids ("do not change any dataset").

So: **add strategies, keep FFDH frozen, take the maximum.**

`common.packing.pack_best_effort(...)` — same signature and same
`PackResult` as `first_fit_decreasing` — runs three arms and returns
whichever placed most:

| arm | what it is |
| --- | --- |
| `first_fit_decreasing` | unchanged, bit-for-bit |
| `_shelf_scan` | FFDH whose **new-shelf origin scans downward** in `mm_per_cell` steps until an offset admits the item. Shelf discipline otherwise untouched, so x stays continuous and it ties FFDH exactly on a clean strip. This is the direct answer to §1. |
| `_grid_greedy` | shelf-free. Places each item at the topmost-leftmost free cell block, found with a summed-area table. Nothing about it spans the strip width, so it survives fragmentation — at the cell-quantisation cost in finding 2. |

Ties go to the earliest arm, so **an instance no arm improves comes back
with FFDH's exact placements**. `best ≥ aware` therefore holds *by
construction* on every instance, which is the property worth buying here:
the failure mode this project keeps hitting is a change that lifts an
average and quietly regresses a case nobody re-measured. That is not
possible with a maximum over arms that includes the incumbent.

### Safety

**Cells must never overlap a forbidden region.** Each arm is correct by
construction and `tests/test_packing_ceiling.py` asserts that for each
arm *unaided*. On top, `_drop_unsafe` re-checks every returned placement
against the mask, against every other placement, and against the strip
bounds, discarding violators. It is a net, not a step — the asymmetry
justifies its O(n) cost: dropping a placement costs one cell, keeping a
bad one puts a battery on a PCB.

Two hazards were tested explicitly rather than argued:

- **Awkward sizes.** An earlier `_next_free_x` fix took four rounds, and
  a plausible correctness argument recovering a column index via
  `int(x / mm_per_cell)` turned out false for non-power-of-two cells; the
  fuzz test that missed it used only binary-exact sizes. The new fuzz
  parametrises cells over `{0.7, 1.0, 1.5, 2.3}` and item widths over
  `{18.3, 18.5}`. `_grid_greedy` re-validates each candidate against
  `_overlaps_forbidden` and skips to the next on disagreement, exactly
  as `_next_free_x` already does.
- **The strip bound in both orientations.** A prototype of the scanning
  arm omitted it and cheerfully placed a 65 mm-wide rotated cell into a
  50 mm strip — the mask reported "clear" because the *grid ended before
  the item did*. It inflated the prototype's real-frame total from 17 to
  19 before it was caught. Pinned by
  `test_a_rotation_wider_than_the_strip_is_rejected` and
  `test_a_mask_smaller_than_the_strip_does_not_grant_free_space`.

### Interface

Unchanged. `first_fit_decreasing` keeps its signature, its behaviour and
its export. `pack_best_effort` is additive and takes the same arguments.
The two planner call sites (`plan.planner.Planner._pack_cartridge` and
`plan.bin_packing.pack_cartridge`) switch to it; **`recog.synth3d` is
deliberately not switched**, so no dataset moves.

---

## 3. Results

### Real frames — before → after

Per cartridge instance, cells placed. All 30 instances were run; the 23
that place zero under both packers (cartridges too small for a cell) are
omitted.

| instance | forbidden | FFDH | after | Δ | winning arm |
| --- | --- | --- | --- | --- | --- |
| `scene_00005/c7` | 6.6 % | **0** | **7** | +7 | grid |
| `scene_00008/c14` | 19.3 % | 1 | 1 | +0 | ffdh |
| `scene_00015/c28` | 11.3 % | 2 | 2 | +0 | ffdh |
| `scene_00031/c53` | 11.8 % | **0** | **1** | +1 | scan |
| `scene_00033/c57` | 9.7 % | 3 | 3 | +0 | ffdh |
| `scene_00040/c64` | 3.1 % | 2 | 2 | +0 | ffdh |
| `scene_00046/c70` | 5.9 % | **0** | **1** | +1 | scan |
| **total** | | **8** | **17** | **+9** | |

**Regressions: none.** Not "none observed" — none possible, by the
maximum-over-arms construction (§2), and fuzz-asserted over 120 random
instances with awkward cell sizes.

Both new arms earn their place, and neither would do on its own — each
scores 16 to the combination's 17. `_grid_greedy` alone loses
`scene_00033` (3 → 2, cell quantisation again); `_shelf_scan` alone loses
`scene_00005` (7 → 6). FFDH alone wins outright on four instances.

### The Plan A benchmark

The published headline is **3.17 → 14.28 cells at 2.5 % coverage with
40/40 paired seed wins**. That figure measures `first_fit_decreasing`,
which this change freezes, so **it has not moved**: the `n_aware` and
`n_naive` columns of `docs/receipts/forbidden_bench.csv` regenerate
byte-identical (verified by diff), and the paired block still reads
`3.325 / t = 13.35 / 40-0-0` at 2.5 %.

It was nonetheless **stale in the sense that matters** — it described a
packer the planner no longer calls. `scripts/forbidden_bench.py` gained a
third arm so the receipt reports what ships, regenerated by its own
tooling:

| coverage | aware (FFDH) | best (ships) | gain | win/tie |
| --- | --- | --- | --- | --- |
| 0.0 % | 23.00 | 23.00 | +0.00 | 0/40 |
| 2.5 % | 14.28 | **14.55** | +0.28 | 6/34 |
| 5.0 % | 9.05 | **10.90** | +1.85 | 32/8 |
| 10.0 % | 2.60 | **5.53** | +2.93 | 40/0 |
| 15.0 % | 0.57 | **2.85** | +2.28 | 38/2 |
| 25.0 % | 0.03 | **0.33** | +0.30 | 12/28 |

The headline number for the shipping packer at 2.5 % coverage is
therefore **14.55**, and the interesting movement is at 10–15 % coverage,
where the ceiling actually was.

### Latency

O3 budget is 8 ms per cartridge (FDR v3 §10.4). Worst real cartridge
(93 × 185 mm, 62 × 123 cells, 24 items): **1.9 ms**. Bench masks
(200 × 150 mm, 100 × 134 cells, 40 items — larger than any cartridge in
the corpus): **3.4 ms mean, 4.6 ms worst single mask**, against 0.9 ms
for FFDH alone. Pinned by `test_stays_inside_the_o3_latency_budget`.

Both new arms memoise failed footprints — occupancy only ever grows and
`last_y` only moves when a shelf opens, so a repeat of a footprint that
already failed cannot succeed. Without it the identical items the planner
supplies each re-derive the same "no" and the bench worst case is 7.8 ms,
inside budget but not comfortably.

---

## 4. Concerns, stated rather than buried

- **`docs/FDR_v3.md` §6.3.1 is now partial.** Its pseudocode still
  matches `first_fit_decreasing` exactly, and §6.3.1's own measurements
  are untouched. But the FDR describes FFDH as *the* planner packer, and
  as of this change the planner runs `pack_best_effort`. The FDR is a
  submitted report and was not rewritten here; this file is the record of
  what superseded it.
- **`_grid_greedy` is cell-quantised and that is a real cost**, not a
  rounding detail — it is why it cannot replace FFDH. If mask resolution
  ever drops below 1.5 mm the arm gets better; if it rises, worse.
- **The reference packer is a lower bound on OPT, not OPT.** The "44 % of
  achievable" figure is therefore, if anything, generous to FFDH. No
  claim of optimality is made for `pack_best_effort` either — at 10 %
  coverage it reaches 5.53 against a demonstrated-achievable 6.08, so
  roughly 9 % of headroom remains unclaimed.
- **23 of 30 real instances cannot hold a single cell.** The packer is no
  longer the ceiling on those; the placement rectangles are. That is the
  next constraint, and it is a perception question, not a packing one.
