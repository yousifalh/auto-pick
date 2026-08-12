# A box that is not one cartridge, and a grid cell that is not free

Date: 2026-08-11
Baseline: `380e7d5`, 737 tests passing. After: +7 tests, none deleted.
Acting on: `docs/superpowers/specs/2026-08-11-scale-calibration.md`
section 4 — the perception-attributable residual the scale fix left
behind.

**Result: commanded placements overlapping ground-truth non-floor
material go 5 of 26 to 2 of 25, worst case 100 % to 8.3 %. Exactly one
cell is lost, and it is the 100 % one.** The remaining two are one
cartridge's left tray wall in one frame, and nothing inside the planner
reaches them — §2.4 measures the family of fixes that would try, and it
makes the number worse.

All figures below use the **centre-anchored physical footprint**: an
18.5 x 65 mm cell at the frame's true GSD, centred on the point
`Planner._build_pose` commands. That is what the robot executes. It is
also, since `380e7d5`, identical to the reserved footprint and to the
top-left-anchored one — the three collapsed when the scale stopped being
a constant, and they are still collapsed here.

---

## 1. The measurement

Reproduced exactly as the scale-calibration spec section 3 specifies: 60
frames of `recog/dataset3d_seg` through `recog/checkpoints/best.pt` +
`recog/checkpoints/seg/best.pt` +
`SegmentationPlacementAreaExtractor(mm_per_cell=1.5,
wall_inset_mm=4.25)`, per-frame calibration from the render sidecar,
cartridge identity from `EnvironmentModel`'s running counter, then
`Planner._pack_cartridge` and `Planner._cell_to_workspace`. Every step is
a shipping method; the harness is a scratch script outside the tree and
reimplements nothing.

The reproduction is exact — 30 instances, 13 productive, 26 cells, the
same per-instance counts, and the same five offenders at the same
percentages as the published table. Everything below is therefore about
the same objects.

---

## 2. What the causes actually are

The scale-calibration spec's account is **right about `scene_00014/c25`
and right about the four slivers being wall**, and wrong about what would
catch either. Both proposed directions were measured, and one of them is
inert.

### 2.1 `scene_00014/c25`: not "a box spanning a cartridge and three
loose cells" so much as a **jig panel read as a cartridge**

The box (330,130–520,272) does span a cartridge and three loose cells,
as reported. But the mechanism downstream is worth stating precisely,
because it is what makes the case hard to catch: the segmenter, handed
that crop, labels **the dark blue jig backdrop panel** as one cartridge —
its rim as `cartridge`, the three loose cells merged into a single
`battery` blob, and the bare panel below them as `bay`. The prediction is
locally coherent in every way a mask-internal check can test:

| candidate check | c25 | the 29 good instances | verdict |
| --- | --- | --- | --- |
| crop centre on foreground (`BadDetectorBox` as it was) | passes | pass | **inert** — this is why it missed |
| exactly one foreground connected component | 1 | 1 (all 30) | **inert** |
| `P_safe` inside the shell's filled body | 1.00 | 1.00 on 23, **0.00 on 6** — including `c82`, which places 3 safe cells | **worse than inert** |
| detector box smaller than the largest cartridge | 189.9 x 141.4 mm | up to 201.6 x 91.8 mm (`c82`, 3 cells, all safe) | **inert** |
| `P_safe` short axis vs the catalog | **108.8 mm** | **≤ 75.4 mm** | **discriminates** |

The last row is the whole finding. `P_safe`'s rotated extent for `c25` is
108.8 x 148.1 mm. The largest cartridge in
`recog/synth3d/assets/catalog.json` has an **outer** footprint of
81.7 x 180.0 mm, so a placeable floor 108.8 mm across cannot be the floor
of any cartridge that exists — it is 33 % wider than the whole object.
The worst other instance in the corpus sits 8 % under the bound on the
short axis and 18 % under it on the long:

| | short axis | long axis |
| --- | ---: | ---: |
| bound (largest cataloged outer footprint) | 81.7 | 180.0 |
| **`scene_00014/c25`** | **108.8** | 148.1 |
| worst other instance | 75.4 (`c61`) | 147.4 (`c36`) |

This could not have been written before `380e7d5`. At the old fixed
0.625 mm/px the same `P_safe` measures 68 mm across and passes.

**Why the outer footprint and not the interior (73.2 x 171.5) or the
placement region (73.2 x 140.8),** both of which `P_safe` is a subset of
by construction: those are inside the range of ordinary segmentation
slop. **Seven** instances overrun the 140.8 mm placement-region length on
the long axis (worst 147.4), six of them productive and holding 19 of the
25 cells that ship; two overrun 73.2 mm on the short axis. A bound at
either would cost real placements and catch nothing extra. Against
the outer footprint, the bound only ever fires on the physically
impossible — so it needs no tolerance term, and a tolerance term is a
knob somebody would later tune to move a number.

### 2.2 The four slivers are wall, and the planner is optimistic about
its own grid

Confirmed as reported: `c36` 59 px, `c57` 155+41 px and 135 px, `c80`
101 px of ground-truth tray wall (and, on one of `c57`'s, 24 px of
background). The predicted bay boundary overruns the true one by 1–2 px
at these frames' GSD, i.e. roughly 1–1.5 mm.

What is *also* true, and is a planner defect rather than a segmenter one:
`_rasterise_mask` marked an occupancy cell FREE if its **centre pixel**
was free. At `mm_per_cell = 1.5` that reports a cell as placeable when up
to half of it is wall — 0.75 mm per edge, in the unsafe direction. The
placement-feasibility spec section 5 finding 3 flagged it and predicted
it was *producing* placements. Measured here, at the true scale, on the
30-cartridge corpus: making a cell free only when **all** of it is free
places **exactly the same 26 cells in the same 13 cartridges** and takes
two of them off the tray wall. The packer simply seats them a pixel or
two further in. The optimism was buying nothing; it was only moving cells
outward.

### 2.3 The proposed geometric guard is inert — measured

> *"a final geometric guard before a pose is committed — reject a
> placement whose footprint is not fully inside the predicted free floor
> with some margin."*

Every commanded footprint, for all 26 placements:

| | fraction of the footprint inside the predicted free floor |
| --- | ---: |
| `scene_00014/c25` (100 % on GT background) | **1.000** |
| `scene_00033/c57` x 3 | 1.000, 1.000, 0.999 |
| `scene_00052/c80` | 0.999 |
| `scene_00019/c36` | 0.975 |
| the safest placement in the corpus | 0.972 (`c14`, 0 % overlap) |

Four of the five offenders are **100.0 %** inside the predicted free
floor, and the one that is not (0.975) sits *between* two placements that
are perfectly safe. There is no threshold that separates them, in either
direction, and at 100 % coverage the guard rejects nothing at all. This
is expected in hindsight and worth stating plainly: the packer already
only places on cells the grid calls free, so re-checking the placement
against the same prediction asks the prediction whether it is right. The
residual is the prediction *being wrong*, and it is invisible from
inside.

### 2.4 So is a clearance margin — and it makes the number worse

The only remaining planner-side lever for the wall slivers is to demand
N mm of predicted free floor all round the footprint. Measured (not
shipped), on top of both changes below:

| clearance | instances with ≥ 1 cell | cells | overlaps > 5 % | worst |
| ---: | ---: | ---: | ---: | ---: |
| **0.0 mm (ships)** | **12** | **25** | **2** | **8.3 %** |
| 1.0 mm | 11 | 23 | 2 | 8.3 % |
| 1.5 mm | 10 | 21 | **3** | 7.8 % |
| 2.0 mm | 10 | 21 | **3** | 7.8 % |
| 3.0 mm | 5 | 13 | 2 | 8.5 % |

It never reaches 0, it costs up to 12 cells, and at 1.5–2.0 mm it
*creates* an overlap: eroding a bay whose predicted boundary is already
wrong just moves the packer somewhere else that is equally wrong. This
kills the whole margin family on this corpus. It is also, in effect,
`wall_inset_mm` under another name, which the brief forbids for
independently good reasons.

---

## 3. What changed

Two changes, both in `plan/placement_area.py`.

### 3.1 `BadDetectorBox` gets a second condition, on the crop's CONTENTS

`SegmentationPlacementAreaExtractor.reject_if_not_one_cartridge_floor`
runs after `arbitrate()` and before the rectangle or the grid exist, so
nothing downstream ever sees an area it rejected. The existing centre
check and this one are complementary and the docstrings now say so: one
is about **where the crop landed**, the other about **what it turned out
to contain**.

The bound is `_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)`, the per-axis max
of the four cataloged `extents_mm` — the same rule, for the same reason,
as the `_DEFAULT_WALL_INSET_MM = 4.25` beside it: no SKU crosses the
Recognition → Planning boundary, so one scalar pair stands in for all of
them and it has to be the one that cannot refuse a real cartridge. A
deployment that knows its cartridge passes
`max_cartridge_extent_mm=` explicitly.

It raises `BadDetectorBox`, so `Planner._ensure_placement_areas` already
counts it (`bad_detector_box_count`, `bad detector boxes` in the run
receipt) and already skips-and-retries. No new exception type, no new
counter, no planner change at all.

**Where it could silently stop checking, it raises instead:**

* the constructor rejects a bound that can never fire — `None`, a
  non-pair, a non-positive side, a swapped pair, `inf`. There is
  deliberately no "unbounded" setting.
* `placeable_extent_mm` raises on an empty mask rather than returning
  `(0, 0)`. Zero satisfies every upper bound, so a zero-extent answer is
  the guard reporting "fine" without having measured anything.
* the guard itself raises `RuntimeError` if a non-empty region measures
  zero, for the same reason.
* `minAreaRect` measures between pixel centres, so a run of n pixels
  returns n-1 and a single pixel returns 0. The pixel is added back.
* `test_the_largest_real_cartridge_is_not_rejected` pins the other
  failure direction — a bound that rejects everything also rejects the
  bad box, and its only symptom is placements quietly not happening.

### 3.2 An occupancy cell is FREE only if all of it is free

`_rasterise_mask` now tests the cell's whole pixel footprint instead of
its centre pixel, and treats a cell that runs off the edge of the
measured mask as FORBIDDEN (in tree it cannot, since both callers derive
the rect from the mask — but "unmeasured" defaulting to "safe" is how
this class of defect arrives).

Vectorised over a summed-area table rather than looped per pixel: the
straightforward per-pixel version cost **15.1 ms** on the 288 x 131 crop
in `test_segmentation_extract_arithmetic_stays_under_the_o3_budget`,
against FDR O3's 8 ms per-cartridge planning budget — the test caught it.
`extract()` on that crop now costs **2.97 ms**, against **4.25 ms** for
the centre-sampling version it replaces, so the honest test is also the
faster one.

---

## 4. Results

30 cartridge instances, 60 frames, centre-anchored physical footprint,
threshold > 5 %.

| | instances with an area | with ≥ 1 cell | **cells** | **overlaps (non-floor)** | overlaps (material) | worst material |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `380e7d5` (before) | 30 | 13 | **26** | **5** | 4 | 9.1 % |
| + bad-box extent guard only | 29 | 12 | 25 | 4 | 4 | 9.1 % |
| + whole-cell occupancy only | 30 | 13 | 26 | 3 | 2 | 7.6 % |
| **both (ships)** | **29** | **12** | **25** | **2** | **2** | **7.6 %** |

Per-instance, every productive cartridge keeps exactly the cell count it
had — `c7` 4, `c14` 2, `c36` 1, `c51` 1, `c53` 1, `c56` 1, `c57` 3,
`c61` 1, `c64` 6, `c70` 1, `c80` 1, `c82` 3 — and `scene_00014/c25`'s
single cell is the only one lost. **One cell traded, and it is the cell
that was being driven into bare backdrop.**

The five, individually:

| instance | before | after | what happened |
| --- | ---: | ---: | --- |
| `scene_00014/c25` | 100 % (1170 px of GT background) | **not placed** | the crop is rejected as not one cartridge |
| `scene_00019/c36` | 5.2 % (59 px wall) | **0** | whole-cell occupancy moved it inboard |
| `scene_00052/c80` | 9.1 % (101 px wall) | **0** | as above |
| `scene_00033/c57` | 8.6 % (155 px wall, 41 obstruction, 24 bg) | 8.3 % (214 px) | unchanged in substance |
| `scene_00033/c57` | 5.3 % (135 px wall) | 5.2 % (133 px) | unchanged in substance |

### The criterion is still not 0, and this is the honest trade

Two placements, both on `scene_00033/c57`'s left tray wall in one frame,
overlap by 8.3 % and 5.2 %. They are segmenter boundary error — the
predicted bay reaches 2 px past the true wall — and the segmenter is out
of scope by the brief. §2.3 and §2.4 measure the two planner-side levers
that would be aimed at them: one rejects nothing, and the other costs up
to 12 cells while going from 2 overlaps to 3. **I would rather report
25 placements with 2 marginal wall contacts than suppress four more cells
to no measured benefit, but that is a judgement stated, not made
silently.** If the owner prefers fewer placements, the 3.0 mm clearance
row is the honest price: 13 cells, and still 2 overlaps.

---

## 5. Verification

* `pytest tests/` — **752 passed, 0 failed**. 737 at `380e7d5`. **+7
  here** (`tests/test_placement_area.py` 19 → 26); the rest of the delta
  is concurrent `recog/seg_evaluate.py` work in the same tree
  (`tests/test_calibration.py` 12 → 20), which is why the total is a
  moving number this session and the per-file count is the one to check.
  No test deleted.
* `python main.py --config configs/demo.yaml` — runs, **10 cycles,
  10/10 placed, 33 placement areas, 62 queue poses**. Byte-for-byte the
  same summary as with the old rasteriser (verified by running the demo
  against a monkeypatched centre-sampling `_rasterise_mask`): the
  heuristic path's flat green rectangles are large and clean enough that
  whole-cell and centre agree everywhere.
* `python main.py --config configs/demo_seg.yaml` — runs, 15 of 15
  frames calibrated, and now reports **1 bad detector box**, which is
  `c25` being caught on the shipping path.
* `python -m recog.seg_ablation` — runs.

Four tests fail if this reverts:
`test_a_box_spanning_a_cartridge_and_loose_cells_is_rejected` (the c25
geometry at c25's own scale),
`test_the_guard_needs_the_frames_own_scale_to_fire` (the same pixels pass
at 0.625 and are rejected at 0.998 — it fails if a constant comes back),
`test_the_extent_bound_cannot_be_set_to_never_fire`, and
`test_a_grid_cell_is_free_only_if_all_of_it_is_free`.

---

## 6. What this invalidates

Both are in `docs/receipts/`, which another agent owns in this tree, so
neither is regenerated here. The numbers below are measured, into scratch
paths, with the shipping commands.

* **`docs/receipts/main_seg_run.txt` — needs regeneration.**
  `python main.py --config configs/demo_seg.yaml --receipt ...` now gives
  **2 cycles / 2 placed / 7 placement areas / 6 poses queued / 13 empty
  queue / 1 bad detector box**, against the recorded 3 / 3 / 8 / 7 / 12 /
  0. The whole difference is `c25`: one bad box rejected, so one fewer
  placement area, one fewer pose, one fewer pick. `frames_with_scale`
  stays 15 of 15.
* **`docs/receipts/seg_ablation.txt` — needs regeneration**, because
  `recog.seg_ablation._pack_count` deliberately calls
  `plan.placement_area._rasterise_mask` "so this ablation measures the
  same quantisation production does" — so it follows production here, by
  design. Three lines move, all in the direction of *less* disagreement:
  mean Δcells **0.032 → 0.008**, positive (cells lost) **4 → 2**, zero
  **120 → 122**. Median 0.000, range [-2, 2] and negative 2/126 are
  unchanged. No conclusion in that receipt or in FDR v3 turns on the
  difference; it strengthens the existing one.

**Both were regenerated on 2026-08-11, after this section was written,
by the agent that owns `docs/receipts/`.** Every figure predicted above
reproduced exactly — all six of `main_seg_run`'s (2 / 2 / 7 / 6 / 13 /
1, `frames_with_scale` 15 of 15) and all three of `seg_ablation`'s,
with median, range and the negative count unmoved as stated. The
predictions above are left as written; they were right. `README.md`,
`docs/FDR_v3.md` §8, §10.6 and §13.2.1, and `docs/NEXT_STEPS.md` items
2 and 3 were updated to the new figures in the same pass, each naming
the superseded value and this commit.

Four existing tests were edited, all in the same way and all for one
reason: `tests/test_placement_area.py::_bay_label` was a 140 x 200 px
crop, which at the 1.0 mm/px those tests use is a 110 x 170 mm cartridge
floor — wider than any cartridge that exists, and now correctly rejected.
It is 100 x 180 px now, a possible cartridge at every scale those tests
use. They are tests about the scale reaching the arithmetic; a fixture
that could not exist is not a good witness for anything.

---

## 7. Left open

1. **The last two overlaps are the segmenter's bay boundary**, the same
   quantity `docs/receipts/seg_eval.txt` reports as 0.949 mm of boundary
   displacement *computed at 0.625* (≈1.4 mm at `scene_00033`'s true
   0.686). Nothing in `plan/` can see it. This is now the only remaining
   term, and it is a segmenter task.
2. **The extent bound is a catalog constant on the Planning side of a
   boundary that carries no SKU.** `recog/synth3d/annotate.py` writes
   `asset` into the COCO sidecar as of `bb880cb`, so a per-SKU bound is
   at least *labellable* now; the same argument applies to
   `wall_inset_mm`, and the two should be decided together if either is.
3. **One rejection in 30 is a thin basis for the 8 % headroom** between
   the worst good instance (75.4 mm) and the bound (81.7 mm). It is a
   physical bound rather than a fitted one, which is why it is defensible
   at n=1, but a corpus with genuinely adjacent cartridges in one crop —
   `plan.arbitration.centre_component`'s docstring says the jig has them
   — would be the test that matters. Two touching cartridges would
   produce exactly this signature, and would be rejected, which is the
   correct answer for the wrong-ish reason.
