# Audit K — algorithmic complexity and scaling

**Date** 2026-08-12 · **HEAD** `39429a4` · **Scope** read-only. Nothing in the
repo was modified, staged or committed. No training, no dataset regeneration,
no Blender. All measurement scripts live outside the tree, in the session
scratchpad (`bench_pack.py`, `bench_corpus.py`, `bench_planner.py`,
`bench_edges.py`).

**Question.** The FDR publishes two hard budgets — O3, queue rebuild ≤ 8 ms per
cartridge, and the PPR's 50 ms end-to-end. Every number behind them was taken
on small scenes. Nobody had established the complexity of the hot paths, or the
scene size at which either budget first breaks.

Everything below is labelled **DERIVED** (read from the code) or **MEASURED**
(timed on this machine). Where the two disagree the measurement wins and the
disagreement is stated.

### Measurement conditions — read this before quoting a number

**The machine was NOT quiet.** Five other audits were running concurrently in
this session. Sampled during the run: CPU 19–20 % busy across 16 threads, GPU
33 % utilised with 12 foreign compute contexts attached. This is precisely the
condition that produced this project's contaminated 40.9 ms segmentation figure
(FDR §13.2.1), so it is handled rather than ignored:

* **Every CPU figure quoted is `min` over 5–7 repetitions**, not mean and not
  median. Under additive contention the minimum is the best available estimator
  of the uncontended time; the mean is not.
* Where the median is far from the minimum the gap is reported, because that
  gap *is* the contention. Worst case observed: segmenter at 8 crops, 14.3 ms
  min against 34.5 ms median — a 2.4× inflation, the same magnitude as the
  figure the FDR retracted.
* **The GPU numbers in §6 should be re-taken on an idle machine before
  publication.** They are directionally sound (the *shape* of the curve is
  contention-invariant) but their absolute values are upper bounds.
* CPU numbers (§§1–5) are the load-bearing ones and are much less exposed:
  min-of-N on a 16-thread box at 20 % load is close to clean.

Hardware: AMD 16-thread, RTX 3060 12 GB, Python 3.14.3, numpy 2.4.3, cv2 5.0.0,
torch 2.13.0+cu126.

---

## Summary

| Hot path | Complexity (derived) | Confirmed | Budget breaks at |
|---|---|---|---|
| `first_fit_decreasing` | O(n·S·R_band·C) | yes | — (never dominant) |
| `_shelf_scan` | O(n·R·R_band·C), memoised to ~O(R²·C) for identical items | yes | — |
| `_grid_greedy` | **O(k·R·C)**, k = placements+1, full SAT rebuilt per item | yes | dominant term |
| `pack_best_effort` | sum of the three arms + 3× `_drop_unsafe` | yes | **158 × 314 mm floor** |
| `_drop_unsafe` | **O(p²)** in placements, run 3× per pack | yes | ~200 placements |
| `OccupancyGrid.set_block` | O(block), **not** O(grid) | yes | — |
| `mask_of` / `free_count` | O(R·C), one full-grid allocation each | yes | — |
| `Reservation.overlaps_mm` | O(1); but `reserve()` is O(k) → **O(k²)** per fill | yes | ~400 cells/cartridge |
| `update_from_snapshot` | **O(D·C)** — yes, quadratic in cartridges | yes | ~500 cartridges/frame |
| `_nearest_battery` | **O(P·B)** (+ O(B) `list.remove`) | yes | ~250 slots × 480 cells |
| `SegmentationPlacementAreaExtractor.extract` | **O(crop px) = O(1/mm_per_px²)** | yes | **0.282 mm/px** |
| `segment_batch` | O(crops), size-independent (all resized to 256²) | yes | **~22 crops/frame** |

**Headline: every budget holds at every scene size this system can physically
present, and the margin at the worst real input is 1.75× (O3) and 2.3×
(end-to-end).** That was unproven in either direction before this audit. The
fine end of the `mm_per_px` range — flagged in the brief as the likeliest hiding
place for a breach — **is safe, but it is the thinnest margin in the system**,
and it is thin in the extractor, not in the packer.

---

## 1 · The packer — `pack_best_effort` and its three arms

### 1.1 Derived complexity

Notation: `n` items, `R × C` mask cells, `S` shelves, `R_band` = rows spanned by
one item (44 for an upright 18650 at 1.5 mm), `p` placements returned.

**`first_fit_decreasing`** — sort O(n log n), then per item a first-fit sweep
across existing shelves. Each shelf trial may call `_next_free_x`, which
collapses the row band with `mask[r1:r2, :].any(axis=0)` (O(R_band·C) in numpy)
and then runs a **Python-level** column scan, O(C). So O(n·S·R_band·C), with the
numpy term cheap per cell and the Python scan expensive per cell.

**`_shelf_scan`** — identical, plus branch (2) walks `y` downward in
`mm_per_cell` steps: up to R iterations, each doing an `_overlaps_forbidden`
(O(R_band · item_cols), cheap) and possibly a full `_next_free_x`
(O(R_band·C)). Naively O(n·R·R_band·C). The `exhausted` set collapses the `n`
factor for *identical* items — which is exactly what the planner feeds it — so
in practice it is O(R·R_band·C) once, plus O(n·S·…) for the first-fit half.

**`_grid_greedy`** — this is the expensive arm and the reason `pack_best_effort`
costs what it does. `common/packing.py:490-491` rebuilds a **full summed-area
table per item**: `(~occupied).astype(int32)` then `.cumsum(0).cumsum(1)` then
`np.pad`. That is ≥ 4 full R×C passes and ~3 R×C int32 allocations *per item*.
The `failed` set memoises footprints that failed, so the cost is borne by
successes: **O(k·R·C)** with k = p + 1.

**`pack_best_effort`** — runs all three arms unconditionally and takes the
maximum, so it costs the sum, plus **three** `_drop_unsafe` calls.

**`_drop_unsafe`** (`common/packing.py:570-574`) is **O(p²)**: for each kept
placement it tests against every earlier kept placement in a Python generator
expression. Its docstring calls this "its O(n) cost". That is wrong — it is
quadratic — and it is run three times per pack.

### 1.2 MEASURED — cell count, item count held at 24

93.1 × 185.0 mm is the `scene_00005` fixture (`tests/test_packing_ceiling.py`),
the largest instance the corpus is known to produce. Wall mask, min ms:

| strip mm | cells | ffdh | shelf_scan | grid_greedy | **best** |
|---|---:|---:|---:|---:|---:|
| 47 × 92 | 1 891 | 0.15 | 0.27 | 0.04 | 0.47 |
| 93 × 185 | 7 626 | 0.34 | 0.34 | 0.62 | **1.48** |
| 140 × 278 | 17 205 | 0.41 | 0.15 | 3.17 | 3.75 |
| 279 × 555 | 68 820 | 0.62 | 0.14 | 11.71 | 12.60 |
| 559 × 1110 | 275 280 | 1.03 | 0.17 | 114.09 | 122.90 |

`grid_greedy` is linear in cells (68 820/7 626 = 9.0× cells → 11.71/0.62 =
18.9× time, super-linear only because more items place). The other two arms are
almost flat in cell count. **The cell-count term is `_grid_greedy`'s, and
nobody else's.**

### 1.3 MEASURED — item count, cells held at 62 × 123

| n items | ffdh | shelf_scan | grid_greedy | **best** |
|---:|---:|---:|---:|---:|
| 4 | 0.06 | 0.07 | 0.28 | 0.44 |
| 24 | 0.56 | 0.58 | 0.94 | **2.18** |
| 96 | 1.80 | 2.04 | 1.05 | 5.09 |
| 384 | 8.19 | 5.09 | 0.67 | 11.76 |

The two shelf arms are the item-count term (linear-to-super-linear because each
failing item re-sweeps the shelves); `grid_greedy` is flat because `failed`
memoises the identical footprint after the first rejection.

**The two terms are on different arms**, which is why `pack_best_effort` is
robust in each variable alone and fragile in both together — and both grow
together in reality, because `n_est = 2·area/(18.5 × 65)`
(`plan/planner.py:527-531`) ties item count directly to cell count.

### 1.4 MEASURED — adversarial masks at real scale (93 × 185 mm, n = 24)

| mask | free | ffdh | shelf | grid | **best** | placed |
|---|---:|---:|---:|---:|---:|---:|
| clear | 100 % | 0.05 | 0.05 | 0.77 | 0.99 | 12 |
| wall (`scene_00005`) | 96.4 % | 0.35 | 0.35 | 0.62 | 1.41 | 8 |
| blobs 15 % | 92.5 % | 0.37 | 2.15 | 0.13 | 2.69 | 1 |
| **checker p=7** | 97.9 % | 0.59 | 2.84 | 0.10 | **3.68** | 0 |
| full | 0 % | 0.34 | 1.69 | 0.07 | 2.13 | 0 |

The adversarial mask is **not** the densest one — it is the *most fragmented*
one: isolated single forbidden cells on a lattice, 97.9 % free, nothing fits
anywhere. That makes `_shelf_scan` walk the entire y range for every
orientation before `exhausted` can memoise, and it is 6× the cost of the real
`scene_00005` wall. **3.68 ms is the true worst case at real cartridge size,
against 8 ms.** The committed test (`test_stays_inside_the_o3_latency_budget`)
exercises the wall mask at 1.5 ms, i.e. 41 % of the real worst case.

### 1.5 MEASURED — where the 8 ms budget actually breaks

Scaling the placement rectangle with `n_est` computed by the planner's own
formula, worst over {wall, blobs 15 %, checker p=13}, bisected:

| floor mm | cells | n_est | worst ms | |
|---|---:|---:|---:|---|
| 73.8 × 142.7 (**largest real**) | 4 655 | 16 | **1.51** | ok |
| 81.7 × 180.0 (**interlock ceiling**) | 6 480 | 24 | **2.04** | ok |
| 93.1 × 185.0 (test fixture) | 7 626 | 28 | 2.80 | ok |
| 140 × 278 | 17 205 | 64 | 5.70 | ok |
| 151 × 301 | 20 000 | 74 | 7.31 | ok |
| **158 × 314** | **21 945** | **82** | **8.16** | **first breach** |
| 279 × 555 | 68 820 | 256 | 63.08 | 7.9× over |

**The packer cannot reach that input.** `SegmentationPlacementAreaExtractor`
already refuses any placeable floor larger than
`_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)` mm
(`plan/placement_area.py:496`, `reject_if_not_one_cartridge_floor`) **before the
occupancy grid is built** — I confirmed this by construction: a synthetic
147 × 147 mm floor raises `BadDetectorBox` rather than reaching the packer. So
the O3 budget is bounded by a *safety interlock written for an unrelated
reason*, at **2.04 ms worst case against 8 ms — a 3.9× margin**.

That is a genuinely load-bearing accident and should be recorded as such: the
bound exists because a detector box once spanned a cartridge and three loose
cells, not because anyone was thinking about latency. **Anyone who raises
`max_cartridge_extent_mm` for a larger SKU also raises the packing cost, and
nothing in the code or the tests says so.** The breach point is at 1.94× the
interlock's long axis and 1.93× its short axis — a deployment handling
cartridges roughly twice today's linear size in *both* axes breaches O3.

### 1.6 MEASURED — `_drop_unsafe` is quadratic

| placements | `_drop_unsafe` | ×3 in `pack_best_effort` |
|---:|---:|---:|
| 25 | 0.042 ms | 0.126 ms |
| 100 | 0.558 ms | 1.675 ms |
| 200 | 2.266 ms | 6.799 ms |
| 400 | 8.586 ms | 25.757 ms |
| 800 | 37.968 ms | 113.904 ms |

Clean 4×-per-doubling. Below ~50 placements it is noise; above ~200 it is the
*whole* budget. Today p ≤ 24, so it costs ~0.1 ms and is invisible — which is
also why its docstring's "O(n) cost" has never been contradicted by a test.

---

## 2 · The occupancy grid

MEASURED, min ms:

| grid | cells | `mask_of(3)` | `free_count` | `set_block` 13×44 | `set_block only_from` | grid bytes |
|---|---:|---:|---:|---:|---:|---:|
| 95 × 49 (real max) | 4 655 | 0.0100 | 0.0068 | 0.0018 | 0.0062 | 4 655 |
| 123 × 62 (fixture) | 7 626 | 0.0100 | 0.0085 | 0.0017 | 0.0060 | 7 626 |
| 400 × 400 | 160 000 | 0.0316 | 0.1050 | 0.0015 | 0.0057 | 160 000 |
| 800 × 800 | 640 000 | 0.0989 | 0.4019 | 0.0018 | 0.0067 | 640 000 |

**`set_block` is O(block), independent of grid size** — the whole-block
reservation change (13 × 44 = 572 cells) costs 1.7 µs, i.e. the change from
marking one cell to marking 572 is free. This was the specific worry in the
brief; it is a non-issue. `only_from` costs 3.5× more (a boolean compare plus a
masked write) and is still 6 µs.

`mask_of` and `free_count` are O(R·C) and each allocate a full grid. At real
sizes both are ~10 µs against an 8 ms budget: 0.13 %. Neither is worth touching.

**`Cartridge.reserve` is O(k) in live reservations** (`plan/scene.py:227-234`
loops `overlaps_mm` over every existing reservation), so filling a cartridge is
**O(k²)**. MEASURED, min per call: #1 = 39 µs (cold), #100 = 366 µs, #800 =
121 µs, #1600 = 197 µs; 2 158 reservations cost 316 ms cumulative. At the real
ceiling (k ≤ 24 per cartridge) this is under 0.5 ms total and irrelevant. It
becomes the dominant per-cartridge cost above ~400 live reservations — reachable
only with a cell an order of magnitude smaller than an 18650.

---

## 3 · The digital twin — `update_from_snapshot`

**Yes, it is O(n²) in cartridges.** DERIVED from `plan/scene.py:481-489`: for
each cartridge detection, a linear scan over every tracked cartridge computing
`bbox.iou`. No spatial index, no early exit, and `matched` only shrinks the scan
after a match. Batteries are O(B), rebuilt wholesale.

MEASURED (n cartridges *and* n batteries, min ms):

| n | update ms | per-cartridge µs | ratio vs previous |
|---:|---:|---:|---:|
| 8 | 0.018 | 2.3 | — |
| 32 | 0.172 | 5.4 | 3.2× |
| 128 | 2.814 | 22.0 | 4.5× |
| 256 | 10.387 | 40.6 | 3.7× |

Clean ~4×-per-doubling: quadratic, confirmed. This is a **whole-frame** cost,
not per-cartridge, so it is charged against the 50 ms budget: it crosses 50 ms
at roughly **500–560 cartridges per frame**. The real corpus maximum is **4**
(§5). Quadratic and harmless — but it should be *known* to be quadratic, since
the module docstring advertises cartridge persistence as a feature and says
nothing about its cost.

`_nearest_battery` is **O(P·B)** — a `min()` over `available` per placement,
plus an O(B) `list.remove` (`plan/planner.py:596-597, 663-666`). MEASURED:
8 slots × 15 cells = 0.044 ms; 64 × 120 = 2.25 ms; 256 × 480 = 30.19 ms. At the
real maximum (16 slots × 28 cells) it is ~0.05 ms.

---

## 4 · `mm_per_px` — the brief's prime suspect

**The premise in the brief is wrong, and the correct answer is more
interesting.** The brief states that grid cell count scales with the inverse
square of resolution, so a 0.49 mm/px frame has ~4× the cells of a 1.05 mm/px
one. It does not.

DERIVED from `plan/placement_area.py:163-165`:

```
px_per_cell = max(1.0, mm_per_cell / mm_per_px)
rows        = int((iy2 - iy1) / px_per_cell)
```

The pixel extent grows as 1/mm_per_px and `px_per_cell` grows as 1/mm_per_px,
so they cancel exactly: `rows = physical_mm / mm_per_cell`. **The occupancy grid
is scale-invariant.** The cell count is set by `mm_per_cell` (1.5 mm) and the
cartridge's physical size, and by nothing else.

MEASURED, over all 211 ground-truth placement areas in `recog/dataset3d_seg`
with a per-frame scale from the render sidecar: `corr(mm_per_px, cells) =
+0.143` — and the per-band means (2 403 / 2 582 / 2 863 / 2 904 cells across
0.45–0.6 / 0.6–0.8 / 0.8–1.0 / 1.0–1.2 mm/px) move by 21 % across a 2.2× scale
range, not by 4.8×. The residual correlation is *which cartridges happen to sit
at which scale*, not scale itself.

MEASURED, fixed 73 × 143 mm cartridge, scale swept:

| mm/px | crop px | grid cells | `extract` ms | `pack` ms | total |
|---:|---|---:|---:|---:|---:|
| 1.09 | 66 × 131 | 3 870 | 0.78 | 0.57 | 1.35 |
| 0.78 | 93 × 183 | 3 696 | 1.49 | 0.57 | 2.06 |
| **0.49 (corpus min)** | 148 × 291 | 3 480 | **2.62** | 0.26 | **2.88** |
| 0.40 | 182 × 357 | 3 480 | 3.96 | 0.38 | 4.34 |
| **0.282** | 259 × 507 | 3 440 | 8.08 | 0.23 | **8.31 BREACH** |
| 0.10 | 730 × 1430 | 3 440 | 95.37 | 0.22 | 95.59 |

Cell count is flat at ~3 500 across a 10.9× scale range, and **`pack` time is
flat at 0.22–0.57 ms**. The packer does not care about `mm_per_px` at all.

**What does scale is `extract()`**, quadratically: it is per-pixel mask
arithmetic over the crop — `cv2.connectedComponents`, `cv2.erode` with a
`2·wall_inset_px+1` kernel that *also* grows as 1/mm_per_px, `cv2.integral`,
`cv2.minAreaRect`. Measured ratio 0.78 → 95.37 ms across a 119× pixel-count
change: 122×. Textbook O(crop px) = O(1/mm_per_px²).

**Verdict on the fine end: safe, with the thinnest margin in the system.**
Bisected first breach at **0.282 mm/px**, against a corpus minimum of
**0.490 mm/px** — **1.74× margin in linear resolution, 3.0× in pixel count.**
The worst *actual* cartridge crop in the corpus (362 × 172 px at 0.503 mm/px,
62 264 px², the largest of 840) measures **extract 3.92 + pack 0.65 = 4.57 ms**,
i.e. **57 % of the O3 budget consumed by the single worst real cartridge.**

Two consequences worth stating plainly:

1. **The FDR's "2.0–2.2 ms per cartridge" (§13.2.1) is roughly the median, not
   the worst case.** The worst real cartridge is 4.57 ms — inside the budget,
   but 2.1× the published figure, and the published figure carries no
   qualifier.
2. **A camera upgrade is the realistic breach path.** Halving the ground sample
   distance from 0.49 to 0.25 mm/px — an ordinary sensor change, and one that
   would be *sold* as improving perception — quadruples `extract` and breaks O3
   on every cartridge. Nothing in the code, the config or the tests would
   report it as a resolution problem.

**Unrelated correctness note found while testing this axis.** Above
`mm_per_px = 1.5`, `max(1.0, …)` clamps and the grid silently coarsens while
`OccupancyGrid.resolution_mm` keeps reporting 1.5. MEASURED: at 2.0 mm/px the
true cell is 2.00 mm; at 3.0 mm/px it is 3.00 mm. Every reservation would then
be under-marked by (true/1.5 − 1) per axis — 33 % at 2.0 mm/px. The corpus
maximum is 1.091 mm/px so this cannot fire today, and it is a silent-failure
finding rather than a complexity one, but it is the same family as audit E
finding 1 and there is no guard on it.

---

## 5 · What scene size do the published numbers assume?

MEASURED from `recog/dataset3d_seg` (502 frames, 5 179 annotations) and
`docs/receipts/main_seg_run.txt`:

| quantity | median | p95 | **max** |
|---|---:|---:|---:|
| cartridges / frame | 1 | 3 | **4** |
| batteries / frame | 6 | 17 | **28** |
| occupancy cells / cartridge | 2 070 | 4 560 | **4 655** |
| planner `n_est` / cartridge | 8 | 16 | **16** |
| cartridge crop | 15 990 px² | 39 848 px² | **62 264 px²** |
| mm/px | 0.781 | 1.032 | 0.490 – 1.091 |
| total cells / frame | 2 998 | 6 725 | **10 895** |

The demo receipt is 15 frames, 26 cartridge detections, 78 loose batteries —
**1.7 cartridges and 5.2 batteries per frame**, not the "7 cartridges / 15
crops" the brief quotes. (The 7 is `placement areas`, a run total; the 26 is
crops summed over 15 frames.) **The real per-frame scene is smaller than
anybody assumed, in both directions of the estimate.**

**Does the FDR state its scene size? Partially — and the gaps are exactly where
the claims are weakest.**

* **Stated.** §10.2 gives "10, 20, 40 and 80 identical 18.5 × 65 mm footprints
  in a 200 × 150 mm strip, 40 seeds". §1 gives "200 × 150 mm strip, 80
  candidate items". §13.2.1 gives "16.6 ms for **8 crops** batched". These are
  proper claims.
* **Not stated.** §10.4's end-to-end table — the row that carries the O3
  verdict — says only "100 consecutive perception+planning cycles on the
  synthetic dataset". No cartridges per frame, no batteries per frame, no
  cell count, and **no `mm_per_px`**, which §4 shows is the variable that
  actually moves the per-cartridge number. The §10.6 and §14 conformance
  tables give "≤ 8 ms per cartridge / Pass — 3 ms median" with no scene size
  at all.
* **Worse: the cited evidence does not exist.** Both conformance tables
  attribute O3 to `bench_cycles.py`. There is no such file anywhere in the
  tree (`scripts/` contains only `forbidden_bench.py` and `seed_check.py`).
  The O3 median is not currently reproducible from the repository.
* **The 200 × 150 mm bench strip is not a cartridge.** It is 1.6× the area of
  the interlock ceiling and 2.3× the largest real placement area. The bench is
  *conservative*, which is the right direction — but it means the published
  packing latency describes an instance the pipeline cannot produce, while the
  instance it *does* produce (§1.5) has never been published.

---

## 6 · The segmenter — batching

DERIVED (`recog/bay_segmenter.py:92-116`): every crop is resized to 256 × 256
before batching, so cost is **O(crops) and independent of crop size**. One
`torch.no_grad()` forward over the whole batch;
`recog/inference.py:295` collects every cartridge crop in the frame into a
single call. No cap on batch size anywhere.

MEASURED (RTX 3060, fp16, 256 px, `recog/checkpoints/seg/best.pt`) — **see the
contention warning at the top; these are upper bounds**:

| crops | batched min | batched median | looped min | per-crop µs | GPU MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.83 | 7.25 | 7.04 | 6 831 | 28.9 |
| 4 | 9.42 | 9.92 | 28.88 | 2 356 | 41.5 |
| **8 (FDR's figure)** | **14.30** | 34.51 | 58.08 | 1 788 | 61.6 |
| 15 | 32.55 | 37.87 | 112.12 | 2 170 | 96.2 |
| **24** | **54.77** | 57.00 | 245.93 | 2 282 | 139.6 |
| 64 | 87.39 | 96.05 | — | 1 365 | 334.0 |
| 96 | 138.83 | 147.42 | — | 1 446 | 490.0 |

* **The 8-crop figure reproduces.** 14.3 ms min here against the FDR's
  16.6–21.2 ms range across five clean regenerations. The batched/looped ratio
  reproduces too (4.1× here, 3.4× published). The architecture claim is sound.
* **The 50 ms budget for segmentation alone first breaks at ~21–22 crops**
  (interpolating 15 → 24). That is **5.5× the corpus maximum of 4 cartridges
  per frame** and 2.7× the batch the FDR sized for.
* **Linear, with a fixed ~6 ms floor.** Per-crop marginal cost settles at
  ~1.4–1.5 ms; the first crop costs 6.8 ms of launch overhead. This is why
  batching is worth 4×, and it means the curve is predictable — there is no
  cliff, just a slope.
* **The median column is the contention.** At 8 crops the median is 2.4× the
  minimum. Do not publish these medians.

---

## 7 · Memory

MEASURED. **Verdict: no growth on the planning side. Nothing resembling the
render loop's multi-GB creep.**

* **One occupancy grid is small.** `uint8`, one byte per cell. Largest real
  cartridge 95 × 49 = **4.5 KiB**; test fixture 123 × 62 = **7.4 KiB**. A frame
  with 4 cartridges holds **~18 KiB** of grid. Even a hypothetical 400 × 400
  grid is 156 KiB.
* **How many at once:** one per *tracked* cartridge, and
  `_match_or_insert_cartridges` deletes any cartridge absent from the current
  snapshot (`plan/scene.py:507-509`), so the count is bounded by cartridges in
  frame, not by frames processed. There is no accumulation path.
* **Transients dominate, and they are `_grid_greedy`'s.** Each item allocates
  `free` (int32, R·C·4), two `cumsum` results and a padded copy — ~4 int32
  grids per item, ~61 KiB per item at fixture size, ~1.2 MiB per item at
  400 × 400. Sequential and immediately collectable.
* **200 × `pack_best_effort` on a 62 × 123 grid: retained delta 22.8 KiB,
  Python peak 0.28 MiB.** Flat.
* **1 000 × `update_from_snapshot` on a persistent twin (8 cartridges,
  30 batteries): retained +4.5 KiB at frame 100 and still +4.5 KiB at frame
  1 000.** Cartridge and battery counts stayed at 8 and 30. Flat.
* One unbounded-but-harmless counter: `_next_battery_id` reached 30 030 after
  1 000 frames (batteries are ephemeral and re-IDed every frame). It is a
  Python int, so it costs nothing and overflows never — but any log or metric
  that treats battery IDs as a population count will read 30 030 for 30
  batteries.
* GPU: `segment_batch` peaks at 61.6 MiB at 8 crops and 490 MiB at 96 —
  linear, and 25× under the card at the largest batch measured.

---

## 8 · Worst case versus typical — the summary the budgets need

| path | typical (median real) | **worst real** | worst constructible | budget | margin at worst real |
|---|---:|---:|---:|---:|---:|
| `pack_best_effort` | 0.5 ms | **2.04 ms** (interlock ceiling, checker mask) | 3.68 ms at 93 × 185 | 8 ms / cartridge | **3.9×** |
| `extract()` + pack | ~1.4 ms | **4.57 ms** (362 × 172 px @ 0.503 mm/px) | 8.31 ms @ 0.282 mm/px | 8 ms / cartridge | **1.75×** |
| `Planner.cycle` whole frame | 2.9 ms @ 1 cartridge | **~5.7 ms @ 4 cartridges** | 48.0 ms @ 60 cartridges | 50 ms / frame | 8.8× |
| `segment_batch` | 9.4 ms @ 4 crops | **9.4 ms @ 4 crops** | 54.8 ms @ 24 crops | 50 ms / frame | 5.3× |
| `update_from_snapshot` | 0.01 ms | **0.01 ms** | 10.4 ms @ 256 cartridges | 50 ms / frame | ~500× |

`Planner.cycle` measured on the heuristic-extractor path (green rectangles), warm
twin: 1 cartridge 2.92 ms, 7 cartridges 5.62 ms, 15 cartridges 11.71 ms, 30
cartridges 22.06 ms, 60 cartridges 48.01 ms. Per-cartridge warm cost settles at
**0.74–0.80 ms** — comfortably inside O3, and the whole-frame 50 ms crosses at
**~62 cartridges**.

**Every budget holds at every plausible scene size.** The binding constraint is
not any of the things the brief suspected — not `set_block`, not the whole-block
reservation, not the twin's re-identification, not cell count versus
`mm_per_px`. It is **`extract()`'s quadratic dependence on camera resolution**,
at 1.75× margin.

---

## 9 · If headroom is needed: the one change

**`_grid_greedy` should hoist its summed-area table out of the item loop.**

`common/packing.py:490-491` rebuilds the full SAT for every item, but `occupied`
changes only where the previous item landed. This is an **implementation
detail, not inherent**: the arm's semantics (topmost-leftmost free block) are
unchanged if the SAT is built once and incrementally repaired over the
`bh × bw` block just written, or simply rebuilt only when a placement succeeded.
It removes the `k` factor from **O(k·R·C)**, taking `_grid_greedy` from linear
in placements to effectively constant, which at the 8 ms breach point (§1.5) is
the difference between 8.16 ms and roughly 3 ms. It buys the most headroom per
line changed, and it is confined to one function with an existing test file
(`tests/test_packing_ceiling.py`) that pins the arm's output.

Second, if that is not enough: **`_drop_unsafe`'s O(p²)** (§1.6) should sort by
x and sweep, or bucket into grid cells. Also an implementation detail. It does
not bind today (p ≤ 24) but it is the next wall, and its docstring currently
tells the reader it is linear.

**Neither is needed at present scene sizes.** The one change I would make first
is not a performance change at all: **state the scene size next to every latency
claim, and add a note at `_MAX_CARTRIDGE_EXTENT_MM` that it bounds the packing
cost as well as the detector box.** Right now the O3 budget is protected by a
constant that nobody knows is protecting it, cited to a benchmark script that
does not exist.

**Nothing was implemented. No repo file other than this report was touched.**
