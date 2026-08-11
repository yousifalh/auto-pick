# Why 23 of 30 real cartridges yield no placeable area

Date: 2026-08-11
Baseline: `9bfc25f`, 708 tests passing (re-run at the end of this work:
708 passed, exit 0 — **no code, config, dataset or model was changed**).
Area: diagnosis only. `plan/`, `recog/`, `configs/` are untouched; every
computation below was run from a scratch script outside the repository,
calling the shipping code unmodified.

`docs/superpowers/specs/2026-08-11-packing-ceiling.md` §3 records that of
30 real cartridge packing instances, **23 place zero cells under either
packer**, and calls them "cartridges too small for an 18.5 x 65 cell in
any orientation … a perception/geometry fact, not a packer fact". This
file tests that sentence against ground truth.

**It does not survive.** Not one of the 23 is a full cartridge. The most
heavily filled cartridge anywhere in the 30-instance sample is half full
(2 of 4 slots, and 3 of 6); **20 of the 30 — and 14 of the 23 — have no
cell seated at all.** The correct split is:

| | of the 23 | |
| --- | ---: | --- |
| **(a) correct behaviour** | **10** | 2 sealed units with no bay at all; 8 open cartridges whose actual contents (1–3 seated cells and/or up to 8 pieces of obstruction debris) genuinely leave no 18.5 x 65 mm strip |
| **(b) infeasible by construction** | **13** | 7 `AnkerPowerCore10000` instances whose *empty* bay cannot hold the planner's nominal cell at the pose the cartridge is actually in; 6 where ground truth does admit a cell and the pipeline lost it downstream |

The dominant mechanism for the second (b) group is **not** the wall inset
and **not** the 18.5 vs 18.3 nominal. It is `mm_per_px`.

---

## 1. Method, and how to reproduce it

The 30 instances are reproduced exactly: 60 frames of
`recog/dataset3d_seg` through `recog/checkpoints/best.pt` +
`recog/checkpoints/seg/best.pt` +
`SegmentationPlacementAreaExtractor(mm_per_cell=1.5, mm_per_px=0.625,
wall_inset_mm=4.25)`, cartridge identity assigned by
`plan.scene.EnvironmentModel`'s running counter exactly as `main.py`
does. This yields 30 `PlacementArea`s and, through
`common.packing.pack_best_effort` with an 18.5 x 65.0 mm cell, the same
7 productive instances and the same per-instance counts the packing
ceiling spec published (`scene_00005/c7` 7, `scene_00008/c14` 1,
`scene_00015/c28` 2, `scene_00031/c53` 1, `scene_00033/c57` 3,
`scene_00040/c64` 2, `scene_00046/c70` 1 — total 17). The reproduction is
exact, so everything below is about the same 30 objects.

Each instance is then paired with its ground-truth **unit**: the COCO
sidecar's `unit_id` groups a cartridge shell, its `electronics_module`,
its `placement_area` bay proxy, its `obstruction`s and any cells seated
in it; `recog/dataset3d_seg/meta/<frame>.json` maps `item{n}` to the
asset (SKU) and to the pose the layout planner drew. Matching is by IoU
between the detector box and the unit's union box; the 30 matches run
0.22–0.95 IoU with 27 of 30 above 0.7.

**The feasibility oracle** is the shipping extractor and the shipping
packer, fed a ground-truth label map instead of a predicted one:
`recog.seg_dataset.rasterise_crop` paints the unit's GT annotations into
the same crop window, `SegmentationPlacementAreaExtractor.extract` is
called on that, and `pack_best_effort` is asked for an 18.5 x 65.0 mm
cell. Only the mask differs from the shipping path; every line of
arbitration, rasterisation and packing code is the same.

### 1.1 A calibration error that has to be handled first

`configs/demo_seg.yaml` sets `mode.mm_per_px: 0.625`, documented as
"this dataset's actual framing", from
`layout.area[0] * 1000 / render.res[0]`. That formula is the framing at
`margin = 1.0, zoom = 1.0`. **No frame in this dataset is rendered at
that framing.** `recog/synth3d/world.py:setup_camera` sets
`ortho_scale = need * margin * zoom` with `margin` drawn from
`[1.02, 1.10]` and `zoom` from `param_space.zoom = [0.75, 1.6]` — the
per-scene zoom that `configs/synth3d.yaml` introduces deliberately, to
give the detector scale variety.

The true ground sampling distance is recorded per scene in
`meta/<frame>.json` as `camera.ortho_scale`, and
`ortho_scale * 1000 / 1280` is verified here against an independent
measurement — the median long side of that frame's unoccluded flat 18650
annotations, which are 65.0 mm by CAD:

| frame | from `ortho_scale` | from cell length | ratio |
| --- | ---: | ---: | ---: |
| `scene_00004` | 0.9510 | 0.9420 | 0.991 |
| `scene_00005` | 0.4915 | 0.4887 | 0.994 |
| `scene_00033` | 0.6856 | 0.6842 | 0.998 |
| `scene_00052` | 1.0451 | 1.0484 | 1.003 |
| `scene_00058` | 0.9038 | 0.9028 | 0.999 |

Across the 30 instances the true GSD runs **0.490 – 1.045 mm/px, median
0.858**. The pipeline uses 0.625 for all of them. The scale ratio
`k = 0.625 / true` runs 0.60 – 1.27 with median 0.73: on **24 of 30
instances the planner under-reads every distance in the scene, by 27 %
at the median and by 40 % at the worst.**

So there are two different questions, and they must not be conflated:

* *Does this cartridge have room?* — answered at the frame's TRUE scale.
* *What did the pipeline see?* — answered at the configured 0.625.

Every "GT says" figure below is at the true scale. Every "shipping"
figure is at 0.625, exactly as deployed.

---

## 2. The 30 instances

`L_free` is the longest all-free axis-aligned rectangle of width >= 18.5 mm
inside the GT free floor, in true mm, in either orientation.
`L_free >= 65.0` is **necessary** for a cell to fit and not quite
sufficient — the packer works on the 1.5 mm occupancy grid, so a strip
that clears 65.0 mm continuously can still fail to survive quantisation
(`scene_00019/c36` at stage S2 in §4 is exactly that case).
`L_axis(empty)` is the same quantity computed from CAD alone for a
**completely empty** bay of that SKU at that cartridge's actual rotation
(see §3). `oracle` and `shipping` are cells actually placed by
`pack_best_effort`, which is the binding statement in both columns.

| frame | unit | SKU | seated / slots | debris | free floor (GT) | % of nominal | L_free | L_axis(empty) | oracle | shipping | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `scene_00004` | `c5` | 10000 | 0/3 | 5 | 2589 mm2 | 73% | 49.5 | 64.7 | 0 | 0 | **(b) SKU** |
| `scene_00005` | `c7` | 26800 | 0/8 | 4 | 9527 mm2 | 92% | 83.6 | 140.7 | 3 | 7 | placed |
| `scene_00006` | `c9` | 10000 | 0/3 | 4 | 2801 mm2 | 79% | 24.7 | 64.4 | 0 | 0 | **(b) SKU** |
| `scene_00007` | `c13` | 13000 | 1/4 | 3 | 2542 mm2 | 52% | 42.5 | 66.3 | 0 | 0 | (a) contents |
| `scene_00008` | `c14` | 26800 | 0/8 | 4 | 8933 mm2 | 87% | 140.7 | 140.3 | 2 | 1 | placed |
| `scene_00009` | `c17` | 13000 | 0/4 | 4 | 3622 mm2 | 74% | 46.0 | 66.5 | 0 | 0 | (a) contents |
| `scene_00011` | `c21` | 13000 | 0/4 | 6 | 3891 mm2 | 80% | 39.4 | 66.7 | 0 | 0 | (a) contents |
| `scene_00014` | `c25` | 10000 | 1/3 | 0 | 2246 mm2 | 63% | 64.8 | 64.7 | 0 | 0 | **(b) SKU** |
| `scene_00015` | `c28` | 13000 | 2/4 | 0 | 2343 mm2 | 48% | 66.6 | 66.6 | 1 | 2 | placed |
| `scene_00017` | `c32` | 13000 | 1/4 | 5 | 2955 mm2 | 60% | 48.1 | 66.6 | 0 | 0 | (a) contents |
| `scene_00019` | `c36` | 26800 | 0/8 | 4 | 8907 mm2 | 86% | 79.9 | 140.7 | 2 | 0 | **(b) lost** |
| `scene_00026` | `c44` | 13000 | 0/4 | 0 | 0 mm2 | 0% | 0.0 | 66.5 | 0 | 0 | (a) sealed |
| `scene_00029` | `c51` | 13000 | 1/4 | 1 | 3313 mm2 | 68% | 66.4 | 66.7 | 2 | 0 | **(b) lost** |
| `scene_00031` | `c53` | 20100 | 0/6 | 6 | 6183 mm2 | 83% | 55.2 | 134.8 | 0 | 1 | placed |
| `scene_00032` | `c55` | 13000 | 1/4 | 2 | 3120 mm2 | 64% | 27.7 | 66.7 | 0 | 0 | (a) contents |
| `scene_00032` | `c56` | 20100 | 0/6 | 4 | 6203 mm2 | 84% | 72.8 | 134.7 | 1 | 0 | **(b) lost** |
| `scene_00033` | `c57` | 26800 | 0/8 | 6 | 9164 mm2 | 89% | 140.5 | 140.3 | 3 | 3 | placed |
| `scene_00038` | `c61` | 13000 | 0/4 | 0 | 3349 mm2 | 69% | 68.1 | 66.5 | 1 | 0 | **(b) lost** |
| `scene_00040` | `c64` | 26800 | 0/8 | 0 | 9744 mm2 | 95% | 140.8 | 140.6 | 7 | 2 | placed |
| `scene_00044` | `c66` | 10000 | 0/3 | 6 | 2680 mm2 | 75% | 27.5 | 64.5 | 0 | 0 | **(b) SKU** |
| `scene_00044` | `c67` | 10000 | 0/3 | 8 | 2739 mm2 | 77% | 29.4 | 64.5 | 0 | 0 | **(b) SKU** |
| `scene_00045` | `c69` | 10000 | 0/3 | 0 | 0 mm2 | 0% | 0.0 | 64.9 | 0 | 0 | (a) sealed |
| `scene_00046` | `c70` | 10000 | 0/3 | 0 | 3280 mm2 | 92% | 64.7 | 64.6 | 1 | 1 | placed |
| `scene_00048` | `c74` | 13000 | 1/4 | 0 | 3290 mm2 | 67% | 61.7 | 66.7 | 0 | 0 | (a) contents |
| `scene_00049` | `c76` | 10000 | 0/3 | 1 | 3162 mm2 | 89% | 65.1 | 64.9 | 1 | 0 | **(b) lost** |
| `scene_00052` | `c80` | 10000 | 0/3 | 3 | 3153 mm2 | 88% | 64.8 | 64.9 | 0 | 0 | **(b) SKU** |
| `scene_00053` | `c82` | 26800 | 3/8 | 0 | 6333 mm2 | 61% | 75.6 | 140.6 | 3 | 0 | **(b) lost** |
| `scene_00057` | `c85` | 20100 | 3/6 | 3 | 3028 mm2 | 41% | 23.1 | 135.2 | 0 | 0 | (a) contents |
| `scene_00058` | `c87` | 10000 | 0/3 | 3 | 3001 mm2 | 84% | 35.2 | 64.8 | 0 | 0 | **(b) SKU** |
| `scene_00058` | `c88` | 20100 | 1/6 | 4 | 4942 mm2 | 67% | 54.2 | 135.2 | 0 | 0 | (a) contents |

Fill distribution over all 30: `0/3` x 9, `0/8` x 5, `0/4` x 4, `0/6` x 2,
`1/4` x 5, `1/3` x 1, `1/6` x 1, `2/4` x 1, `3/8` x 1, `3/6` x 1.
**Maximum fill anywhere: 50 %.** The generator draws `p_seated = 0.5` and
`seated_frac` from `[0.15, 0.85]` (`manifest.json`), so a full cartridge
is not something this corpus contains. "Genuinely full" is therefore not
available as an explanation for any of the 23, and the packing-ceiling
spec's sentence about them should be read as retracted.

---

## 3. The `AnkerPowerCore10000` cannot accept a cell. Ever.

Seven of the 23 are `AnkerPowerCore10000` instances that ground truth
also refuses — and reading that as "(a), correct behaviour" would be the
single biggest mistake available here. They fail with **nothing in the
bay at all**.

From `catalog.json`, the placement region is `interior_mm` minus
`module_bay_mm`:

| SKU | placement region | margin on the long axis | wall (`case_wall_mm`) | rotations that still admit a cell |
| --- | --- | ---: | ---: | --- |
| 10000 | 54.9 x **65.0** | **+0.00 mm** | 4.00 | **exactly 0.000°**, then nothing until 31.8° |
| 13000 | 73.2 x 66.75 | +1.75 mm | 3.75 | 0–6.89°, then 24.9°+ |
| 20100 | 54.9 x 135.2 | +70.20 mm | 3.70 | 0–38.44° |
| 26800 | 73.2 x 140.8 | +75.80 mm | 4.25 | 0–45° (all) |

The 10000's bay is **exactly** the planner's nominal cell length. Not
approximately: 65.0 against 65.0.

The planner keeps its placement rectangle axis-aligned by construction —
`plan/placement_area.py`'s module docstring says "The camera mount is
fixed on the cell, so keeping the placement rectangle axis-aligned costs
nothing in accuracy". That is true of the *camera*. It is not true of the
*cartridge*: `layout.plan` places every unit at `quarter*90 + jitter`
with `jitter_deg: 2.0`, and a real jig has clearance too. An axis-aligned
`w x h` rectangle fits inside a `W x H` rectangle rotated by θ only if
`w·sinθ + h·cosθ <= H`, so the longest axis-aligned strip of width 18.5
available in an empty bay is

    L(θ) = (H − 18.5·sinθ) / cosθ

For H = 65.0, W = 54.9 this is **below 65.0 for every θ in (0°, 31.8°)**
— it recovers only past 31.8°, where the bay has turned far enough for a
strip to be inscribed the other way, which no jig jitter will ever reach.
The measured rotations of the ten 10000 instances run 0.30°–1.81°, giving
`L_axis` **64.45–64.90 mm** against 65.0 needed — the `L_axis(empty)`
column above, and it agrees with the measured GT free floor to within
0.2 mm on every empty, debris-free instance (`c70`: predicted 64.56,
measured 64.7; `c25`: 64.73 vs 64.8; `c80`: 64.89 vs 64.8).

**Cost of a degree: 18.5·tanθ ≈ 0.32 mm. Available: 0.00 mm.**

The one 10000 that ground truth passes, `scene_00049/c76`, is rotated
0.30° and measures 65.1 mm — it clears 65.0 by 0.1 mm, and only because
the GT mask rasterises at 0.892 mm/px and rounds up. That is not a
system that works; that is a coin landing on its edge.

Add the wall inset and even that disappears. Per-SKU, oracle versus the
same GT masks through the extractor's own 4.25 mm erosion:

| SKU | n | GT admits >= 1 cell | ... and still does after the 4.25 mm inset | shipping |
| --- | ---: | ---: | ---: | ---: |
| 10000 | 10 | 2 | **0** | 1 |
| 13000 | 10 | 3 | 3 | 1 |
| 20100 | 4 | 1 | 1 | 1 |
| 26800 | 6 | 6 | 6 | 4 |

**With perfect segmentation, a perfect detector box and perfect
calibration, the `AnkerPowerCore10000` is unplannable in 10 of 10
instances.** Half the SKU mix by instance count (10000 + 13000 = 20 of
30) sits on 0.00 mm and 1.75 mm of longitudinal margin; the task as
specified — an 18.5 x 65.0 mm nominal footprint, packed axis-aligned,
into a 65.0 mm bay — is infeasible on the 10000 and marginal on the
13000. That is a specification finding, and it is the honest answer.

The short axis says the same thing in a quieter voice. The real 10000
holds three 18.3 mm cells in 54.9 mm exactly; `configs/planning.yaml`'s
`diameter_mm: 18.5` needs 55.5 mm for three. A cartridge with exactly one
free column offers 18.3 mm and the planner will refuse it. No instance in
this sample has exactly one free column, so **18.5 -> 18.3 recovers
nothing here (measured, §5)** — but the trap is real and latent.

---

## 4. Where the other six were lost: a per-stage budget

Six instances — `c36`, `c51`, `c56`, `c61`, `c76`, `c82` — have ground
truth room and produced nothing. Five stages, each isolated by changing
one thing and holding the rest at the shipping setting. All lengths are
`L`, the longest all-free strip of width >= 18.5 mm, in true mm; 65.0 is
required.

| stage | what changes | `c82` (26800, 3/8 seated) | `c76` (10000, empty) | `c36` (26800, empty, 4 debris) |
| --- | --- | ---: | ---: | ---: |
| S0 | GT mask, GT box, true scale, **no inset** — the oracle | 75.6 -> **3 cells** | 65.1 -> **1 cell** | 79.9 -> **2 cells** |
| S1 | + wall inset 4.25 mm | 75.6 -> 3 | **40.1 -> 0** | 79.9 -> 2 |
| S2 | predicted segmentation instead of GT | 72.7 -> 3 | 64.2 -> 1 | **66.4 -> 0** |
| S3 | detector box instead of GT box | 74.6 -> 3 | 62.4 -> 0 | 70.6 -> 1 |
| S4 | `mm_per_px` 0.625 instead of the true GSD — **ships** | **45.6 -> 0** | **30.6 -> 0** | **36.2 -> 0** |

`scene_00053/c82` is the clean case: a 26800 with 3 of 8 slots filled and
no debris, 75.6 mm of free strip against 65.0 needed — **+10.6 mm of
genuine margin**. The wall inset costs it nothing (its wall *is*
4.25 mm). Segmentation costs 2.9 mm, the detector box gives 1.9 mm back.
Then `k = 0.625/0.969 = 0.645` turns 74.6 mm of real floor into 45.6 mm
of apparent floor — a 29.0 mm haircut on a 10.6 mm margin. Terminal, and
nothing else in the chain matters.

`scene_00049/c76` is the double-terminal case: a 10000 with 0.1 mm of
margin, killed independently by the wall inset (`round(4.25/0.892) = 5 px
= 4.46 mm` of erosion against a 2.45 mm bottom wall, so 2.01 mm intrudes
into a bay that had 0.1 mm to give) and again by the scale.

`scene_00019/c36` is the only one of the six where perception is the
first cutter: the predicted bay is 13.5 mm shorter than the true one on
its binding axis. Worth noting that `docs/receipts/seg_eval.txt`'s
0.949 mm bay boundary displacement is itself computed at 0.625 mm/px; at
this corpus's median true GSD it is **1.30 mm**, and every other
millimetre figure in that receipt is understated by the same factor.

---

## 5. Headroom — what actually recovers instances

Each row changes exactly one thing from the shipping configuration
(`mm_per_px 0.625`, `wall_inset_mm 4.25`, cell 18.5 x 65.0,
`mm_per_cell 1.5`), on the real predicted masks and real detector boxes,
through `pack_best_effort`.

| change | instances with >= 1 cell (of 30) | recovered of the 23 | total cells |
| --- | ---: | ---: | ---: |
| *(baseline — ships today)* | 7 | — | 17 |
| cell nominal 18.5 -> 18.3 | 7 | **0** | 17 |
| wall inset 4.25 -> 3.0 | 7 | **0** | 17 |
| wall inset 4.25 -> 2.45 | 7 | **0** | 17 |
| wall inset 4.25 -> 2.0 | 7 | **0** | 17 |
| wall inset 4.25 -> 1.0 | 7 | **0** | 17 |
| wall inset 4.25 -> 0.0 | 7 | **0** | 17 |
| occupancy 1.5 -> 0.5 mm cells | 5 | 0 | 12 |
| **`mm_per_px` 0.625 -> the frame's true GSD** | **13** | **7** | **26** |
| true GSD + inset 3.0 | 13 | 7 | 26 |
| true GSD + inset 0.0 | 15 | 8 | 28 |
| true GSD + inset 0.0 + 18.3 nominal | 15 | 8 | 28 |

And with perception removed from the question entirely (GT masks, GT unit
box, true scale):

| | instances (of 30) | of the 23 | cells |
| --- | ---: | ---: | ---: |
| GT, inset 4.25 | 10 | 5 | 24 |
| GT, inset 3.0 | 11 | 6 | 26 |
| GT, inset 0.0 — the oracle | 12 | 6 | 27 |
| GT, inset 0.0, 18.3 nominal | 12 | 6 | 27 |
| GT, inset 4.25, at 0.625 | 6 | 1 | 13 |

Three things fall out, and two of them contradict the hypothesis this
investigation started from.

1. **At the shipping calibration, the wall inset is not the constraint.**
   Taking it to 0.0 mm — no erosion at all — recovers **zero** of the 23
   and adds **zero** cells. The inset is a real cost (it costs 3 cells
   and 2 instances once the scale is right, and it is the sole cause of
   `c76`), but at 0.625 mm/px it is masked entirely by a larger error.
2. **18.5 -> 18.3 recovers nothing on this sample**, at any calibration.
   It is a latent trap on the short axis, not an active one here.
3. **A finer occupancy grid makes things worse**, 7 -> 5 instances and
   17 -> 12 cells. `_rasterise_mask` samples the mask at cell *centres*,
   so a 1.5 mm grid dilates the free region by up to 0.75 mm per edge.
   Some of today's 17 placements exist only because of that optimism.
   That is worth knowing before anyone "improves" the resolution, and
   worth worrying about on its own terms.

---

## 6. The placements that do happen are not all safe

Of the 17 cells the shipping pipeline places, each was tested against the
GT label map twice: once over the footprint the planner *reserved*
(`w_mm / 0.625` px) and once over the footprint a real 18.5 x 65 mm cell
*occupies* at the frame's true GSD.

| | placements overlapping GT non-floor by > 5 % |
| --- | ---: |
| reserved footprint | 1 of 17 |
| physical footprint | **3 of 17** |

The worst is on `scene_00005/c7`, where `k = 1.27`: the planner reserves
30 x 104 px while a real cell needs 38 x 132 px, and one of its seven
cells lands **21.2 %** on top of ground-truth non-floor material.
`scene_00031/c53` is a different failure of the same family — the oracle
says no cell fits (`L_free` 55.2 mm, six pieces of debris) and the
pipeline places one anyway, because the predicted bay over-reports free
floor across the debris. `docs/receipts/seg_eval.txt` already reports the
optimistic placeable-area error as 51.5 mm2 per crop; this is what that
number does downstream.

So the end-to-end result is not merely conservative. Its conservatism and
its errors have the same root, and the root points both ways.

---

## 7. What this means for "1 pick from 15 frames"

`docs/receipts/main_seg_run.txt` reports 26 cartridges -> 26 masks ->
8 placement areas -> 1 pose -> 1 pick over 15 frames. Two corrections,
both measured:

* **The receipt is stale.** It was committed at `12134c2`; the packing
  ceiling fix landed at `d6c46ac`, after it. Re-running the identical
  command at `9bfc25f` (into a scratch path — the tracked receipt was
  left alone) gives `cycles 2, placed 2, queue_poses 8, empty_queue 13`.
  (Re-run on this working tree; the commits after `9bfc25f` touch only
  `README.md` and `docs/`, so the code is `9bfc25f`'s.) The figure to
  quote for the shipping code is **2 picks from 15 frames, 8 poses
  queued**, not 1.
* **Neither number measures what it appears to.** 24 of the 30 instances
  are planned at a `mm_per_px` that under-states the scene by 27 % at the
  median. Fixing only that — same masks, same boxes, same packer — takes
  the corpus from 7 productive instances and 17 cells to 13 and 26.

**Verdict: a tolerance artefact, with a genuine specification failure
underneath it.** Neither of the brief's two options is right on its own.

* The scenes are not "hard" in the sense of being full. Nothing in this
  corpus is more than half full, and 14 of 30 cartridges are empty.
* The pipeline is not simply correct. On 13 of the 23 the cartridge had
  room or would have had room empty, and it was consumed before the
  packer saw it.
* But the biggest single consumer — `mm_per_px` — is a property of *this
  measurement harness*, not necessarily of a deployed cell. A real fixed
  camera has one true GSD, and `planning.yaml` says as much ("Replace
  with real intrinsics when available"). A fixed-scale planner evaluated
  over a deliberately scale-randomised corpus will under-place, and that
  is what 23-of-30 mostly is. **The 23-of-30 figure should not be quoted
  as a property of the system.**
* Underneath that, and *not* an artefact: the `AnkerPowerCore10000` has
  0.00 mm of longitudinal margin against the planner's own nominal cell,
  and 0.32 mm is consumed per degree of cartridge rotation by the
  axis-aligned packer alone. It is unplannable in 10 of 10 instances even
  with perfect perception and perfect calibration. Half the SKU mix by
  instance count is at or near this boundary.

---

## 8. Decisions this leaves open (owner's call, not implemented here)

Nothing in this file was implemented. These are safety-relevant margins;
the numbers are here so the trade can be made on evidence.

1. **`mm_per_px` must stop being a single constant, or the corpus must
   stop being multi-scale.** The evaluation harness has a per-frame
   `camera.ortho_scale` sitting unused in every sidecar. Worth 7 of the
   23 and +9 cells, and it is a correctness fix, not a margin trade — at
   `k = 1.27` the current constant is *unsafe*, not conservative (§6).
2. **`wall_inset_mm` 4.25 -> 3.0** buys 1 instance and 2 cells once (1)
   is fixed, and 0 before it. 4.25 is the max over four measured
   `case_wall_mm` values; the 10000's *bottom* wall is 2.45 mm, so the
   erosion intrudes 1.80 mm into that SKU's bay. Note the extractor has
   no SKU to look up (`plan/placement_area.py` says so, and that is still
   true across the Recognition -> Planning boundary) — though
   `recog/synth3d/annotate.py` now writes `asset` into the COCO sidecar
   (`19f64be`), so a per-SKU inset is at least *labellable*. The
   `dataset3d_seg` sidecar used here predates that commit and carries no
   `asset` field; SKU here comes from `meta/<frame>.json` instead.
3. **18.5 -> 18.3 buys nothing measurable here.** Recommend leaving it;
   it is a deliberate safety margin and this evidence does not argue
   against it. Its cost shows up only on a cartridge with exactly one
   free 18.3 mm column, which this sample does not contain.
4. **Axis-aligned packing on a rotated cartridge is the unmodelled
   constraint.** No inset or nominal change reaches it. Either the
   placement rectangle follows the cartridge's pose, or the 10000 is
   declared out of scope for this cell. Those are the two honest options.
5. **`_rasterise_mask` samples cell centres**, which dilates the free
   region by up to 0.75 mm per edge and is currently *producing*
   placements (§5, finding 3). This is the one item here that argues for
   *less* placement, and it should be decided together with (1).
