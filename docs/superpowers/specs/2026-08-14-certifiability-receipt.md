# A receipt for the certifiability finding — and one figure that does not survive it

Date: 2026-08-14
Baseline: `0f4292e`, working tree otherwise clean.
Area: `scripts/`, `docs/receipts/`, `tests/`, and one row of
`docs/RESULTS_SUMMARY.md`'s receipt index. **No extractor, packer,
`catalog.json`, `planning.yaml` or metric definition was touched; nothing
was retrained and no dataset was regenerated.**

## Why

`docs/RESULTS_SUMMARY.md` §2 leads with the project's strongest claim and
quotes five numbers for it. Every other headline in that document cites a
`docs/receipts/` artefact. That paragraph cited a spec file
(`docs/superpowers/specs/2026-08-11-placement-feasibility.md` §3, §5) — in
a document whose whole argument is that every figure has a receipt, the
load-bearing paragraph was the least auditable one.

It is now `docs/receipts/placement_feasibility.txt`, written by
`scripts/placement_feasibility.py`, which runs in **3.9 s** and needs **no
checkpoint, no training and no rendering**.

## What the generator measures

Two arms, deliberately separated because they have different input
requirements and different standing.

**Geometry** — `recog/synth3d/assets/catalog.json` and
`configs/planning.yaml` alone, both committed, so this arm runs on a bare
clone. Placement region per SKU (`interior_mm` minus `module_bay_mm`, with
the rectangle assumption asserted rather than assumed), its margin against
the nominal cell on both axes, and the rotation at which an axis-aligned
cell stops fitting an empty bay. The rotation model is the full
inscribed-rectangle condition, `a·cosθ + b·sinθ ≤ cross` and
`a·sinθ + b·cosθ ≤ along`, scanned at 0.001°, evaluated for **both**
orientations of the cell because `pack_best_effort` runs with
`allow_rotation=True`.

**Census** — every ground-truth cartridge unit in `recog/dataset3d_seg`,
its own annotations painted into its own union box by
`recog.seg_dataset.rasterise_crop`, through
`SegmentationPlacementAreaExtractor` at the frame's own
`camera.ortho_scale`, then through **`Planner._pack_cartridge`** — the
packing call the planner really makes, not a copy of its six lines. The
images are never opened: the segmentation extractor does not read
`image_rgb`, so the sidecar and the meta files are the whole input.

The population is defined by ground truth rather than by the detector.
The spec's 30 instances were whatever the shipping pipeline detected over
the first 60 frames, paired back to GT units by IoU — that needed two
checkpoints to enumerate, which is what made the figure expensive to
re-run and therefore un-receipted. This script enumerates the units
directly.

## The five quoted figures

| quoted in §2 | measured | verdict |
|---|---|---|
| 10000 places zero cells in **10 of 10** instances at the 4.25 mm inset | 10 of 10 | reproduced |
| nominal 18.5 → 18.3 mm recovers **0** | 0 instances | reproduced |
| wall inset 4.25 → 0.0 mm recovers **0** | **1 instance** | **contradicted** |
| margins **+1.75 / +70.2 / +75.75** mm | +1.75 / +70.20 / +75.75 mm | reproduced |
| the 13000 is spent by **6.9°** | 6.89° | reproduced, with a scope caveat |

### The one that fails, and why it is worth knowing

> "…and relaxing the wall inset to 0.0 mm recovers **0**."

Measured in the scope the sentence itself sets — ground-truth masks,
ground-truth boxes, each frame's true scale — taking the inset to 0.0 mm
recovers **one** instance and one cell: `scene_00049/item5`, turned 0.297°
at 0.8917 mm/px, bay measuring 65.10 mm. It also recovers at 3.0 mm, so
the production 4.25 mm is what costs it.

The "recovers 0" is a true figure imported from the wrong arm. The
feasibility spec's §5 measures the inset sweep on **predicted** masks at
the fixed 0.625 mm/px the pipeline used to ship, where 4.25 → 0.0 recovers
0 of 23 and adds 0 cells. §3 of the same spec measures the GT oracle and
says the opposite in as many words: *"The one 10000 that ground truth
passes, `scene_00049/c76`, is rotated 0.30° and measures 65.1 mm … Add the
wall inset and even that disappears."* The summary carries §5's conclusion
under §3's scope.

It is the smaller half of the sentence — one instance, one cell; the SKU
is not made plannable and the geometric claim is untouched. But the
document is about to go to a professor who directs a manufacturing
research centre, on the strength of "every figure has a receipt", and
this is the figure a careful reader would push on. **The correct fix is a
fix to the sentence, not to the receipt.** I have not made it: §2 is not
mine to edit under this task's constraints, and quietly correcting it
would defeat the point. Suggested minimal edit, entirely within §2:

> …taking the nominal 18.5 → 18.3 mm recovers **0** instances, and
> relaxing the wall inset to 0.0 mm recovers **1 of the 10** — the one
> instance that clears 65.0 mm by 0.1 mm on rounding.

### Two scope caveats the receipt now records

**The 13000's 6.9° is the slot-aligned orientation only.** Its bay is
73.2 mm across and a cell is 65.0 mm long, so a cell laid *across* the
slots keeps fitting to 65°; 6.9° is what the +1.75 mm of along-axis margin
buys, not the angle past which the SKU admits nothing. On the 10000 the
two are identical — its bay is narrower than a cell is long and there is
no second orientation. That difference *is* the finding (marginal versus
infeasible), so the receipt prints both columns and says so above the
table.

**n = 10 is a window, not the corpus.** The same 502-scene dataset holds
**47** open 10000 units, and the SKU places zero cells in **43 of 47** at
the production inset. The four exceptions are the geometry seen from the
other side: no 10000 turned more than **0.28°** from axis-aligned takes a
cell anywhere in the corpus — a seventh of the generator's own 2° jitter —
and being inside that band is necessary, not sufficient (7 of the 11 units
inside it still take none). A bay with 0.00 mm of margin is decided by
whether the rasteriser rounds up, which is precisely why it cannot be
certified. §2 already discloses n = 10 honestly; the corpus-wide figure
makes the claim stronger, not weaker, and is now on the record.

Also noted: the spec's §3 table gives the 26800's margin as **+75.80 mm**
from a rounded 140.8 mm bay. `catalog.json` gives 140.75, so the margin is
**+75.75** — which is what `RESULTS_SUMMARY.md` quotes. The summary is
right and the spec's table is the rounded one.

## What certifies the harness

The census population is not the spec's, and the totals agree anyway:

| | spec addendum (HEAD) | this receipt |
|---|---|---|
| inset 4.25, instances / cells | 10 / 24 | 10 / 24 |
| inset 0.00, instances / cells | 11 / 25 | 11 / 25 |

Per-SKU at the production inset, the spec's §3 line — 10000 0, 13000 3,
20100 1, 26800 6 admitting a cell — reproduces exactly. The two
populations differ by three units and all three place zero either way:
this census has 29 open cartridges; the spec's 30 were 28 open plus 2
sealed (`scene_00026`'s 13000 and `scene_00045`'s 10000), and
`scene_00045`'s *open* 10000 is in this census and not in the spec's,
because the detector matched the sealed unit beside it. The agreement was
not tuned for.

The receipt reproduces the addendum's HEAD figures rather than §3's
`ce1d9cd` ones (12 / 27 at inset 0.0), which is correct: `b93bbd3` made
`_rasterise_mask` require every pixel of a grid cell to be free, and the
addendum re-measured for exactly that reason.

Robustness check, run and discarded: painting *all* of a frame's
annotations into the crop window (what `BaySegDataset` does) instead of
only the unit's (what the spec's method says) gives identical per-SKU cell
counts on the published window. The choice does not carry the result.

## Files

* `scripts/placement_feasibility.py` — the generator. `--check` recomputes
  every arm whose inputs are present and requires each section to appear
  verbatim in the committed receipt; it skips the census arm loudly when
  the gitignored dataset is absent, and refuses to write a partial
  receipt. The provenance header (which carries the commit) is
  deliberately outside the comparison, so the check does not fail on its
  own commit.
* `docs/receipts/placement_feasibility.txt` — the receipt. Command,
  commit, SHA-256 of `catalog.json`, of `planning.yaml` and of the
  gitignored `instances_seg.json`, the cell and grid it ran with, the
  per-SKU counts, the per-instance table for all ten 10000s, the 2 × 3
  sweep over both windows, the graded verdict block and the
  reconciliation.
* `tests/test_placement_feasibility_receipt.py` — six tests. The geometry
  section is recomputed from the committed files and must appear in the
  receipt (runs anywhere); `--check` must return 0; the receipt must name
  its inputs and its own rebuild command; the summary's receipt index must
  cite the artefact and the artefact must exist; the receipt's
  zero-placement count and the summary's quoted "10 of 10" are coupled, so
  neither can move without the other; and every one of the five figures
  must carry both a quoted and a measured value.
* `docs/RESULTS_SUMMARY.md` — one row of the receipt index, and nothing
  else.

## Left open

1. **§2's inset sentence is still wrong.** Owner's call; the wording above
   is a suggestion, not an edit.
2. `docs/RESULTS_SUMMARY.md` §3 quotes "1,210 tests pass, 2 skip" from
   `pytest-cov.txt`; the tree ran 1,276 before this work and 1,282 after
   it. That figure was already stale and is outside this task's scope.
3. The spec's §3 rotation column for the 13000 (`0–6.89°, then 24.9°+`)
   describes the slot-aligned orientation only. It is not wrong for what
   it measures, but it reads as a statement about the SKU. The receipt
   states both; the spec is left as written, per its own convention that
   published measurements are not retrofitted.
