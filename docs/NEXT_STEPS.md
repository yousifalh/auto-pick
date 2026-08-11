# Where this stands, and what to do next

Written 2026-08-08, after the segmentation extension landed on
`feat/blender-synth-dataset`; figures refreshed 2026-08-09 after the
tray-interior fix regenerated the dataset and checkpoint (see the note
below the "What exists" table); revised again 2026-08-09 because the
project owner has confirmed real photographs cannot be obtained for this
project — read "The constraint this plan works around", immediately
below, before anything else. This is the pick-up-here document: what
exists, what is honestly unfinished, and what to do about it in what
order.

The goal is a **fully working pipeline** — CAD to a robot placing cells into
real cartridges. Most of it exists. The part that does not is named plainly
below, because it is the only thing standing between "built" and "works".

---

## The constraint this plan works around

**Real photographs are not obtainable for this project.** The owner has
confirmed this directly — not a scheduling gap, a fact this plan now has
to work around permanently. Nothing below waits for photographs to
arrive, and nothing in the reordered plan is written on the assumption
that they will.

The direct consequence: **sim-to-real transfer cannot be measured on
this project.** Not "not yet measured" — cannot be measured, for as long
as this constraint holds. The three real-photo figures already in this
document (item 1 below: checkpoints scoring 0.211, 0.232 and 0.318
against the heuristic's fixed 0.217, on the same 20 cartridges) were
always a smoke test at n = 20, never a transfer claim on their own
terms — but with no route to more real data, that measurement cannot
mature into one either. Any statement of real-world performance
anywhere in this project's documentation — this file, the FDR, a
receipt — must be read as unvalidated and labelled as such.

~~`docs/FDR_v3.md` does not yet carry a dedicated, explicit limitation
statement to this effect (§13.2(5)'s prose gets close but stops short of
one). It needs one. That is a separate, deliberately larger edit and is
intentionally **not** done as part of this pass — noted here so it is
not forgotten.~~ **Done, 2026-08-11: `docs/FDR_v3.md` §13.2.2,
"Limitation: sim-to-real transfer is unvalidated, and cannot be
validated under this project's constraints."** It states the constraint,
names every class of figure in the report as synthetic-to-synthetic,
labels the three real-photo numbers as the smoke test they are, records
what stands in for a transfer measurement (cross-distribution
generalisation and domain randomisation) and how much weaker each is,
and bounds what the limitation does *not* touch. §13.2's future-work
list now signposts it, so items (1) and (4) can no longer be read as
work this project deferred.

Two corrections to how this document previously reasoned about the
constraint, both worked through in full below:

1. **τ calibration was implied to be blocked on real data, then on
   error size in the synthetic split. Neither is the live diagnosis
   any more.** τ is now RETIRED as a confidence gate outright (item 4
   below), not merely waiting on harder scenes: the two masks it
   compares are not independent (an argmax mechanism, `plan/arbitration.py`)
   and, measured directly per SKU, their IoU and the optimistic error
   correlate in the WRONG direction for a gate, in all four SKUs. No
   scene-difficulty work fixes a mis-signed correlation. `P_safe`'s
   geometric intersection is unaffected and stays in place. What *does*
   still lead "What to do, in order" is spec #2's generalisation
   measurement (item 4 no longer motivates the clutter/occlusion work
   the way it used to — see Step 2 below for why that work is retained
   on different grounds).
2. **Everywhere else, the strategy changes from measuring a gap to
   widening a distribution nothing can confirm is wide enough.** Two
   proxies replace "measure transfer": spec #2's cross-distribution
   generalisation test (train on one synthetic distribution, test on a
   disjoint one — an answerable question with no photograph in it), and
   spec #3 plus the remainder of spec #4 as domain randomisation
   (broaden training coverage instead of measuring the distance to the
   target domain). Neither substitutes for a transfer measurement, and
   neither should ever be reported as one. Full reasoning in "What to
   do, in order" below.

---

## What exists

Four plans were executed end to end. Every number below has a receipt in
`docs/receipts/`.

| Plan | What it built | Headline |
|---|---|---|
| A | Forbidden-mask FFDH shelf advance | 3.17 → **14.28 cells** at 2.5 % coverage, 40/40 paired seed wins (that measures `first_fit_decreasing`, which is frozen; the planner now runs `common.packing.pack_best_effort` at **14.55** — see "The packing ceiling" below) |
| B | Five-class segmentation ground truth from CAD | `placement_area` = currently-free floor, **0 overlapping pixels** across 3280 mask pairs (full 502-scene tray-interior dataset; was 139 pairs on a 32-scene spot check) |
| C | Per-ROI bay segmenter | IoU 0.8126; boundary displacement **0.949 mm** (bay) vs the 2.9 mm a mask head would quantise to |
| D | Integration and arbitration | Planning **2.0 ms/cartridge** vs an 8 ms budget; segmentation 20.2 ms for 8 crops vs 50 ms (was 16.7 ms; an intermediate 40.9 ms reading was GPU-contention noise, since superseded by a clean re-measurement — see the tray-interior retrain note below) |

New modules: `recog/synth3d/bay.py`, `recog/seg_dataset.py`,
`recog/seg_training.py`, `recog/seg_evaluate.py`, `recog/bay_segmenter.py`,
`recog/seg_ablation.py`, `recog/calibrate_tau.py`, `plan/arbitration.py`,
`scripts/forbidden_bench.py`.

708 tests (was 621 when this table was written; the baseline moved
through 666 / 678 / 704 as the three changes recorded in "Landed
2026-08-11" below and one concurrent agent's work landed). The
torch-free demo (`python main.py --config configs/demo.yaml`) still
runs and is **unchanged**, which is what the FDR's reproducibility claim
rests on. Since `12134c2` there is also a second, torch-requiring path —
`python main.py --config configs/demo_seg.yaml` — that puts the trained
segmenter in the same loop; the reproducibility claim does *not* rest on
it and must not be moved onto it.

The segmentation checkpoint referenced throughout this document (Plan C's
row above, and items 1, 4 and 5 below) is `recog/checkpoints/seg/best.pt`
trained for the **full 40-epoch schedule** on the 502-scene / 841-crop
dataset — not the epoch-24, cut-off checkpoint an earlier pass of this
document described. `recog/seg_training.py` now has a `--resume` flag
(model, optimiser, scheduler, epoch and best-so-far are all checkpointed
every epoch to `train_state.pt`) specifically so a run can be continued
across as many invocations as a time-limited environment requires,
instead of losing every epoch after the last one saved.

**The dataset and checkpoint were regenerated from scratch on
2026-08-09 for a rendering defect, not a scale-up: `open_case`
cartridges had been rendering closed and upside down** (Blender's glTF
importer inverts this CAD's up-axis, and `lay_flat` had no notion of
which end of the vertical axis was up), so the electronics module and
`placement_area` plane were painted on the outside of a closed lid
instead of seated inside the tray's cavity. Every label geometry moved
once this was fixed (four commits, `27cbd97`..`9fcf136`); the dataset
was deleted and fully re-rendered at the same 502-scene / 841-crop
scale (not resumed — resuming would have silently mixed the old and
new label conventions in one dataset), and the checkpoint was retrained
from a fresh initialisation, not fine-tuned from the pre-fix weights.
See FDR §13.2.1 for the full before/after and `docs/receipts/`.

---

## Landed 2026-08-11 — three things that were true in the docs and not in the code

Recorded here because each one changes what a figure elsewhere in this
document means. All three are measured, not asserted; none changed a
metric definition.

**1. τ is deleted from `plan/placement_area.py` (`5a619fc`).** It had
been documented as retired since `dee9854` while the branch went on
running, and at `configs/planning.yaml`'s own `mm_per_px: 0.38` it
rejected **every** plannable cartridge it was offered — 0 before, 8
after. Detail and the three-way value inconsistency it resolved: item 4
below.

**2. The segmenter is in the end-to-end loop (`12134c2`).** Before this,
`main.py` ran only the heuristic extractor and the segmenter was
unreachable under *any* config — `load_detector` took no `segmenter`
argument and `_build_planner` hardcoded the heuristic. **Any earlier
statement that the pipeline demonstrated the segmenter end to end was
overstated; it is true now.** `configs/demo_seg.yaml` plus
`docs/receipts/main_seg_run.txt`: 26 cartridges detected → 26 segmented
→ 8 placement areas → 1 pick-and-place. Every way the new path could
no-op raises instead, including a completed run that produced zero
placement areas. Those frames are the segmenter's own training corpus,
so the receipt is evidence about the **wiring**, not a generalisation
measurement — that is what "The generalisation measurement" section
below is for. `configs/demo.yaml` is untouched and still torch-free.

**3. The packing ceiling is lifted (`d6c46ac`).** `first_fit_decreasing`
never scans its shelf origin in y, and `_next_free_x` collapses a
shelf's whole row band, so one forbidden region in the first shelf's band
kills the entire pack: `scene_00005` handed the packer a 93 %-free grid
containing a clear 112 × 48 mm rectangle and got **zero** cells. The
planner now calls `common.packing.pack_best_effort` — FFDH plus a
shelf-origin-scanning arm plus a shelf-free grid-greedy arm, maximum
taken, ties to FFDH — so `best ≥ FFDH` holds **by construction** and no
instance can regress. Real frames **8 → 17 cells** over the 7 capable
instances of 30. `first_fit_decreasing` is frozen and `recog/synth3d`
still calls it, so no dataset moves; FDR §6.3.1's pseudocode remains
accurate *for FFDH*. Two consequences for figures in this document: Plan
A's 14.28 is still the FFDH number (the shipping packer is 14.55 at the
same coverage), and item 3's Δcells figures were measured under the
FFDH-only planner and have not been re-run.

Specs: `docs/superpowers/specs/2026-08-11-segmenter-integration.md`
(items 1 and 2), `2026-08-11-packing-ceiling.md` (item 3).

---

## The generalisation measurement (spec #2, 2026-08-11)

**Every number in this section is synthetic-to-synthetic. None of it is a
sim-to-real measurement, and none of it should be quoted as one.** Real
photographs with segmentation ground truth do not exist for this project
and will not (see item 2 below), so sim-to-real transfer cannot be
measured at all. What *can* be measured is whether a segmenter trained on
**procedurally generated cartridge trays it has never seen a real example
of** transfers to the four **real measured Anker CAD assemblies** — and
that is what this is. Design spec §0 and §12.

Six models, each trained from a fresh initialisation on the identical
40-epoch schedule, differing only in dataset: `anchored` and `wide`
(procedural trays, 502 scenes each), and four leave-one-SKU-out CAD
controls (502 scenes each, one Anker SKU excluded from training). All six
are scored against the **same** 836 held-out CAD test crops
(`recog/dataset3d_seg_cad_test`, 500 scenes, disjoint from every training
set). Receipts: `docs/receipts/seg_eval_{anchored,wide}_on_cad_test.txt`,
`seg_eval_cad_control_<SKU>_on_cad_test.txt` (×4),
`seg_eval_{anchored,wide}_on_{anchored,wide}_val.txt`.

### Headline: procedural training reaches the CAD-trained ceiling on some classes and not others

Pooled over all 836 CAD test crops (selected mean is over
`bay`/`electronics`/`obstruction`, as `select_on` defines it):

| trained on | bay | electronics | obstruction | battery | cartridge | selected mean |
|---|---:|---:|---:|---:|---:|---:|
| procedural, anchored | 0.6555 | 0.7541 | 0.6306 | 0.5593 | 0.8088 | **0.6801** |
| procedural, wide | 0.6536 | 0.7565 | 0.6280 | 0.5502 | 0.7833 | **0.6794** |
| CAD control, hold out 10000 | 0.9131 | 0.8634 | 0.6507 | 0.7833 | 0.9424 | 0.8091 |
| CAD control, hold out 13000 | 0.9045 | 0.8611 | 0.6320 | 0.7728 | 0.9412 | 0.7992 |
| CAD control, hold out 20100 | 0.9032 | 0.8530 | 0.6412 | 0.7444 | 0.9387 | 0.7991 |
| CAD control, hold out 26800 | 0.9044 | 0.8600 | 0.6322 | 0.7439 | 0.9437 | 0.7989 |
| **procedural, anchored + crowned lid** (2026-08-11, see below) | **0.8755** | 0.7819 | 0.6360 | 0.6906 | 0.9120 | **0.7645** |

**Do not quote the `bay` column of that table on its own — it conflates
two unrelated quantities and understates the procedural result.** A
pooled per-class IoU accumulates one union over all 836 crops, while the
instance count printed beside it counts only the crops that *contain*
the class, so painting `bay` on a closed cartridge is charged against
the same number as segmenting a real bay badly. Split apart:

| model | pooled `bay` (836 crops) | **present-only `bay`** (the 213 crops with a GT bay) | **sealed crops given a hallucinated bay** |
|---|---:|---:|---:|
| procedural, anchored | 0.6555 | **0.8801** | **136 / 623 = 21.8 %**, 675 460 px |
| procedural, anchored + crowned lid | 0.8755 | **0.8856** | **16 / 623 = 2.6 %**, 22 559 px |
| CAD control (each SKU scored by the fold that never saw it) | 0.9009 | **0.9013** | **2 / 623 = 0.3 %**, 722 px |

On the crops that actually contain a bay, procedural training was
already within **0.021** of the CAD ceiling before any fix — not the
0.246 the pooled row shows. **91.4 % of the published gap was
false-positive `bay` on sealed cartridges.** `battery` is the same
mechanism (0.5593 pooled → 0.6924 present-only, control 0.7500);
`electronics` (0.7541 → 0.7652) and `obstruction` (0.6306 → 0.6316) are
barely affected and their pooled figures read as published. Report both
halves — present-only IoU **and** the sealed false-positive rate —
wherever this headline appears.

The CAD-trained control is what makes this readable. Without it, a
procedural selected mean of 0.68 would be ambiguous between "the model
fails to generalise" and "the procedural trays are unrealistic". **The
numbers support neither as a blanket answer: the shortfall is
class-by-class, and it tracks almost exactly how much of each class's
geometry the procedural tray builder actually generates.**

- `bay` (`placement_area`) is the free tray floor — geometry the
  procedural builder invents wholesale. Largest gap: 0.655 vs 0.904,
  **−0.25**.
- `cartridge` is the tray's own outer silhouette — also invented.
  0.81/0.78 vs 0.94, **−0.14**.
- `battery` is cells seated at the packer's pitch inside that invented
  cavity, and the procedural sets deliberately mix three cell formats
  (18650/21700/26650) where all four CAD SKUs are 18650-only. **−0.20**.
- `obstruction` is foreign matter dropped onto the bay floor by
  `world.build_obstructions` — **one call site, byte-identical code for
  CAD and procedural scenes**. It is the one class where procedural
  training matches the CAD-trained control: 0.6306 vs 0.6320–0.6507
  pooled.

That last row reads like a success and is not one. It was checked rather
than celebrated: obstruction geometry is *shared source code*, not
something the procedural pipeline had to generalise to, so parity there
is the expected result and is **not** evidence of transfer. Read the
`bay`/`cartridge`/`battery` gaps as the actual answer.

> **Superseded in part, 2026-08-11.** The class-by-class reading above —
> "the shortfall tracks how much of each class's geometry the procedural
> tray builder invents" — was diagnosed directly and **is wrong as stated
> for `bay` and `battery`**, the two largest gaps. Both are dominated by
> false positives on *sealed* cartridges, not by segmentation quality on
> real bays: restricted to the 213 crops whose ground truth contains a bay,
> the procedural model pools to **0.8801** against the CAD control's
> **0.9013** (0.021 apart, not 0.246), and 91 % of the published gap is the
> 675 460 px of `bay` it paints on closed shells, where the control paints
> 722 px. `battery` is the same mechanism (0.5593 → 0.6924 present-only).
> The cell-format explanation offered for `battery` two bullets up was
> tested with a purpose-built 18650-only procedural model and **came out
> null** (+0.017 of the 0.224 available). `cartridge` and `electronics`
> keep the original reading — they have no hallucination component. Full
> measurement, correlates and the ruled-out confounds:
> `docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md`; receipt
> `docs/receipts/seg_eval_anchored_18650_on_cad_test.txt`.
>
> **Acted on, 2026-08-11 — the sealed-unit false positives are 92 % closed.**
> The cause was measured and it is **not** appearance randomisation (the
> procedural and CAD pipelines draw from one shared pool, verified identical
> to sampling noise on backdrop, lighting, exposure, zoom and shell preset).
> It is that `world.build_procedural_tray` built the lid as a planar cuboid
> while all four Anker lids are barrel-crowned — long-edge fillet radius
> 11.10 mm, 89 % of upward-facing polygons non-planar against the procedural
> lid's 0 % — so a closed cartridge with any top-face shading structure was
> absent from training and the model had learned "featureless flat top ⇒
> closed". One procedural set was re-rendered with a sampled lid crown as the
> single change (labels, unit boxes and 99.8 % of sealed `cartridge` masks
> pixel-identical to `anchored`) and one model trained on the same 40-epoch
> schedule. On the same 836 crops: **sealed false-positive rate 21.8 % →
> 2.6 %** (control 0.3 %), pooled `bay` **0.6555 → 0.8755**, `cartridge`
> 0.8088 → 0.9120, `battery` 0.5593 → 0.6906, selected mean 0.6801 → 0.7645;
> present-only `bay` rose 0.8801 → 0.8856 and open-crop recall rose, so this
> is not a threshold shift. **This is domain randomisation informed by a
> measured coverage gap, not a transfer claim** — the crown range was chosen
> after measuring the CAD, and it remains synthetic-to-synthetic throughout.
> `electronics` (0.7819 vs 0.8530) and `cartridge` (0.9120 vs 0.9382) are now
> the largest honest gaps. Full measurement, the pre-registered thresholds and
> six suspicion checks on a large favourable result:
> `docs/superpowers/specs/2026-08-11-sealed-unit-experiment.md`; receipt
> `docs/receipts/seg_eval_anchored_crown_on_cad_test.txt`.

### Leave-one-SKU-out: each control scored on the SKU it never saw

For SKU *X*, `control_X` was trained on the other three SKUs only, so on
*X* it is itself generalising to unseen geometry — the fair ceiling for
"trained on real measured trays, tested on a new one". Compared against
the anchored procedural model on the same SKU:

| SKU (crops) | model | bay | electronics | obstruction | battery |
|---|---|---:|---:|---:|---:|
| 10000 (202) | anchored | 0.6376 | 0.7623 | 0.6612 | 0.3399 ⚠ |
| | crowned | 0.8430 | 0.8051 | 0.6665 | 0.5963 ⚠ |
| | control (held out) | 0.9005 | 0.9023 | 0.6837 | 0.8173 ⚠ |
| 13000 (218) | anchored | 0.6750 | 0.7856 | 0.7010 | 0.7267 |
| | crowned | 0.8783 | 0.8030 | 0.7093 | 0.7856 |
| | control (held out) | 0.8884 | 0.9067 | 0.6967 | 0.8019 |
| 20100 (214) | anchored | 0.6344 | 0.6997 | 0.5043 | 0.5365 |
| | crowned | 0.8706 | 0.7454 | 0.4897 | 0.7101 |
| | control (held out) | 0.8988 | 0.7787 | 0.5000 | 0.7447 |
| 26800 (202) | anchored | 0.6665 | 0.7573 | 0.6197 | 0.5331 |
| | crowned | 0.8890 | 0.7748 | 0.6316 | 0.6101 |
| | control (held out) | 0.9098 | 0.8310 | 0.6180 | 0.6763 |

(The `crowned` rows were added 2026-08-11 from
`docs/receipts/seg_eval_anchored_crown_on_cad_test.txt`; the `anchored`
and `control` rows are unchanged. The crowned model closes most of the
per-SKU `bay` gap on every SKU, and the ⚠ 14-crop caveat below applies
to its `battery` figure exactly as to the other two rows.)

⚠ AnkerPowerCore10000's `battery` figures rest on **14 crops**, below the
~24–36-instance density this project treats as reportable. Flagged before
the numbers were seen, not after: they are small-sample estimates in both
rows and must not be read at the same confidence as the other three SKUs.

`obstruction` is at or slightly above the control on 3 of 4 SKUs — the
same shared-code artefact as above, not a win. `20100` is the hardest SKU
for `obstruction` for *every* model measured (0.488–0.514), procedural
and CAD alike.

### Wide vs anchored: extra variation neither helped nor hurt

Decision 2 asked whether widening the procedural sampling band beyond
what the real SKUs span helps transfer. **It did not, in either
direction**: 0.6801 vs 0.6794 selected mean, and no per-SKU per-class
difference larger than the noise these instance counts support. The
honest conclusion is that this comparison came out null. Wide is
meaningfully worse on its *own* validation split (0.6489 vs anchored's
0.7161) — it is a harder distribution to fit — without buying anything on
the CAD test set.

### In-distribution vs out-of-distribution

| model | own val split | CAD test | Δ |
|---|---:|---:|---:|
| anchored | 0.7161 (127 crops) | 0.6801 (836 crops) | −0.036 |
| wide | 0.6489 (124 crops) | 0.6794 (836 crops) | +0.031 |

The wide model scoring *higher* out-of-distribution is not a
generalisation success. It is driven almost entirely by `electronics`
(0.4547 in-distribution → 0.7565 on CAD), and wide's in-distribution
`electronics` figure rests on **18 crops** — below the reportable floor.
The procedural module bay varies far more than any real one does, so the
procedural val split is simply harder for that class than the CAD test
set is. Both procedural models lose ground where it matters (`bay`
−0.21/−0.23, `battery` −0.20/−0.30 going out of distribution).

### What got worse

Reported rather than tuned away:

- **`obstruction` is below the previously published 0.6579 floor
  (`docs/receipts/seg_eval.txt`) for all six models** — procedural
  (0.6306/0.6280) *and* all four CAD-trained controls (0.6320–0.6507).
  Because even the CAD-trained model misses it, this is a property of
  the new, genuinely disjoint CAD test set being harder than the old
  same-distribution validation split, not evidence against procedural
  training. The old 0.6579 was measured on 24 instances of a split drawn
  from the same 220-scene render the model trained on.
- **`battery` is below the published 0.6907 floor for the two procedural
  models** (0.5593/0.5502) while all four CAD controls clear it
  (0.7439–0.7833). Here the control *does* separate the two
  explanations: this one is the procedural trays, specifically their
  three-cell-format mix against an 18650-only CAD test set. **The
  cell-format half of that sentence was tested on 2026-08-11 and is
  false** — an 18650-only procedural model, one variable changed, scored
  `battery` 0.5763, +0.017 of the 0.224 available. The real mechanism is
  false-positive `battery` on sealed cartridges; present-only, the
  procedural figure is 0.6924 against the control's 0.7500. See
  `docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md`.

### Regression checks

Five-class disjointness held at **0 overlapping pixels** on every dataset
generated for this work — anchored 5426 pairs, wide 6374, CAD test 11450,
and the four leave-one-SKU-out controls at 15669 / 14270 / 13328 / 11040
pairs — and on the two datasets rendered since (18650-only 13 689 pairs,
crowned lid 13 589 pairs, both 0 overlapping). Suite green at **708
passing** (was 666 when this line was written). `python main.py --config
configs/demo.yaml` still runs torch-free (10 cycles, 10 placed). The
`assembled` variant seals for procedural trays exactly as for CAD,
verified numerically at full scale: 614 anchored / 593 wide / 627 CAD
assembled units emit only `cartridge` and zero interior-class
annotations.

---

## What is honestly unfinished

### 1. The segmenter-vs-heuristic real-photo comparison has now moved three times

**This is the gap that matters, and a third measurement made it harder to
read, not easier.** On the 20 annotated cartridges in `recog/realtest/`,
three checkpoints across two training runs have now been scored against
the same 0.218 design-spec threshold: an epoch-24 checkpoint at
**0.211** (below), a completed 40-epoch checkpoint from the same run at
**0.232** (above), and this task's checkpoint — retrained from scratch
on the tray-interior-corrected dataset — at **0.318** (above, by the
widest margin yet). The heuristic itself scores 0.217 throughout
(unchanged; it is a fixed baseline, re-measured each time for parity).

That is not evidence the domain gap has closed, and this third point
makes it *weaker* evidence, not stronger. The first two checkpoints
shared a training run and dataset, differing only by epoch; this one
differs in every respect at once — fresh initialisation, a fully
re-rendered dataset, corrected label geometry — so no single factor can
be credited for the jump. The synthetic-domain IoU actually moved the
*other* way this time (pooled mean IoU fell slightly, 0.8045 → 0.8032),
while the real-photo number rose the most it ever has, which argues
against reading the real-photo number as tracking synthetic model
quality at all. The raw `bay` channel is still genuinely tiny on real
images before erosion — verified as a true domain gap across all three
runs, not a measurement artefact.

The segmenter still learned to segment *renders*. Whether it also
transfers to photographs is now three-for-three unresolved, having
produced 0.211, 0.232 and 0.318 against the same fixed comparison set —
which is worse for planning purposes than a stable answer in either
direction would be: a stable negative would at least have ruled out
shipping it as-is, and a stable positive would have supported shipping
it. Neither exists. See FDR §13.2.1 for the full before/after and
receipts.

### 2. Real-photo ground truth does not exist, and now will not

`recog/realtest/` has 7 photographs and 20 cartridges annotated with **boxes
only — no segmentation polygons**. That is why the comparison above is a
placeable-fraction proxy rather than an IoU against human masks. This used to
be framed as a gap to close — the original Step 1 asked for 50–100 more
images, polygon-annotated. It is now permanent: no more real photographs are
obtainable for this project (see "The constraint this plan works around"
above), so no mask-level real-world claim will be possible here.

**What the existing ~20-cartridge set is still good for: a qualitative
sanity check**, and unlabelled diagnostics — does the model fire at all on a
photograph, do predictions look plausible, is the arbitration IoU
distribution in a sane range. **What it is not good for is a quantitative
claim.** The evidence for that is item 1's own number series: three
same-recipe checkpoints scored **0.211, 0.232 and 0.318** against the
heuristic's fixed **0.217**, on the identical 20 cartridges. Changes that
should be irrelevant to real-photo transfer — a different training epoch, a
from-scratch retrain on corrected label geometry — moved the score by more
than the effect the comparison exists to detect. Run-to-run noise exceeds
the signal. That series stays in this document because it is the evidence,
not despite being awkward.

**The annotation tooling built to consume such ground truth is retained, not
deleted, and is currently dormant.** `docs/ANNOTATION_PROTOCOL.md`,
`recog/labelme_to_seg.py` and `recog/check_annotations.py` (commit
`09326f3`) convert LabelMe polygon exports into this project's COCO-RLE
sidecar format and validate them for pixel overlap, degenerate RLEs and
missing annotations — 51 tests, all passing, no photographs touched. It
exists for the counterfactual: if real photographs ever become obtainable,
this is the first thing to run. It is not in use now, and nothing in this
plan depends on it being used.

### 3. Damage-direction crops — got WORSE this retrain, and still not re-investigated on the current split

Δcells is mean **+0.032** over the 126-crop validation split — numerically
identical to the pre-tray-fix figure — but the negative-direction count
**rose to 2 of 126** (was 1 of 126 pre-fix; 2 of 54 further back, before
the dataset was first scaled to 502/841). This is a regression on the
metric that matters most, reported as one rather than smoothed over by
the unchanged mean: the tray fix corrected the geometry but did not
improve, and by this one measure slightly worsened, the fraction of
crops where the prediction packs a cell the ground truth forbids.

**The investigation below describes an EARLIER split's two negative
crops under an EARLIER checkpoint; it has not been re-run against
either the previous single negative crop or this task's current two.**
Both the dataset and the model have changed twice since that
investigation. Read what follows as the documented explanation for the
original finding, generalisable lessons included, not as a claim
already verified about the current two negative crops — one of which
may plausibly involve the tray wall geometry that did not exist when
this investigation was written, and that possibility has not been
checked.

Investigated in full on the prior split (see
`.superpowers/sdd/2026-08-06-D-integration-arbitration/damage-case-investigation.md`).
The conclusion reverses the initial reading:

- **The "region too thin to admit a cell" explanation is wrong.**
  `admits_a_cell` is `True` for the ground truth at every pipeline stage in
  both crops.
- **The two crops fail for two unrelated reasons.** `scene_00106` (−3) is a
  *packer* artefact: FFDH's shelf algorithm is blocked by the union of six
  small scattered ground-truth obstructions, which the segmenter mostly
  recall-misses, leaving the prediction a cleaner mask the packer can fill.
  `scene_00117` (−1) is a ~0.6 mm, one-pixel boundary-quantisation
  coincidence at the electronics/bay edge.
- **Realised severity is negligible.** Mapping the placed-cell rectangles back
  to pixel space: **0 of 6 predicted placements across both crops overlap
  ground-truth electronics, obstruction or battery.** `scene_00106`'s three
  cells are entirely safe by the ground truth's own reckoning;
  `scene_00117` has a ~0.5 mm wall-rim graze on two of three.
- **Neither is an arbitration bug.** The arithmetic preserves admissibility
  faithfully in every case checked.

**The generalisable lesson: a negative Δcells is not the same as a damage
event.** The sign convention treats the ground truth as a safety oracle, but
here the ground truth is *pessimistic* — the packer under-fills it because of
its own shelf behaviour, not because the space is unsafe.

**The τ gate cannot catch this class of failure, structurally.** Both crops
pass at IoU 0.91–0.94, comfortably above both τ = 0.7492 and τ = 0.85 — because
the gate measures a single prediction's *self-consistency*, not its
*correctness against truth*. Worth remembering before relying on it for
anything it was not designed to do. (Historical: that gate no longer
exists in the code as of `5a619fc` — see item 4 below. The reasoning
stands as a reason not to rebuild it.)

**`scene_00106`'s packer artefact was a real defect and has since been
fixed** — FFDH never scanned its shelf origin in y, so a forbidden region
in the first shelf's row band voided the whole pack. The planner now runs
`common.packing.pack_best_effort` (`d6c46ac`), which competes FFDH
against two obstacle-tolerant arms and takes the maximum. Δcells has
**not** been re-measured under the new packer, so the +0.032 mean and the
2/126 negative-direction count above are both figures from the FFDH-only
planner. Re-measuring them is now part of resolving this item.

**Mitigations were tested and rejected on cost-benefit**: a larger wall inset
(to 7.5 mm), requiring `P_safe` itself to admit a cell, and extra `P_safe`
erosion (to 2 mm). None removes both negatives — `scene_00117` is untouched
across the entire sweep — while costing up to 30 % of the ground truth's
placeable cells. No code changed.

### 4. τ is retired as a confidence gate — measured, not merely uncalibrated

**This item used to read "blocked on error size, fixable with harder
synthetic scenes." That diagnosis has been superseded by a stronger,
measured one: the gate cannot work at all, and no amount of scene
difficulty changes that.** Two structural reasons, both confirmed
against the tray-fix checkpoint rather than argued from first
principles. First, `recog/bay_segmenter.py:110` emits
`logits.argmax(dim=1)`, a single label per pixel; `P_direct` and
`P_derived` are therefore not independent — the electronics/
obstruction/battery subtraction inside `P_derived` is a structural
no-op on `P_safe`'s content, since any pixel `P_direct` claims already
excludes those three classes by construction. Second, measured directly
per SKU on the 35-crop population below
(`docs/receipts/tau_independence_correlation.txt`): IoU and the
optimistic error correlate POSITIVELY in all four SKUs — the opposite
sign a confidence gate needs — and normalising by area does not rescue
it. Full derivation and the correlation table: FDR §13.2.1. `P_safe`'s
intersection remains a real, retained safety property (it still keeps
placements inside the visible cartridge cavity); only the IoU threshold
on top of it is retired. The paragraphs below are kept as the
historical record of how that error-size diagnosis was reached — they
are still accurate as a description of the 35-crop split's error sizes —
but the conclusion they point toward ("harder scenes will fix τ") no
longer holds and should not be acted on.

Retrained to completion again — from a fresh initialisation, on a fully
re-rendered 502-scene / 841-crop dataset (the tray-interior fix; see the
note near the top of this document) — with 35 `bay` / 35 `electronics`
/ 24 `obstruction` validation instances (was 37/37/18 pre-fix; 19/19/11
before that). τ came out at **0.5715**, accepting 35 of 35 cartridges —
sharply *up* from 0.3180. (Both remain lower bounds set by the sample's
own minimum rather than a calibrated threshold, so which one is quoted
does not change the conclusion below.)

**It is still uninformative, but scaling is no longer the story — geometric
correctness is.** Not one of the 35 accepted cartridges admitted a cell at
any observed IoU, so the safety budget never bound and τ remains the
sample's lowest observed value rather than a threshold found by trading
safety against throughput. That part is unchanged.

**The diagnosis moved in the OPPOSITE direction from every previous
scale-up, which is itself the finding worth recording.** The largest
optimistic error observed is now **1278 px against a cell footprint of
3045 px² — 42.0 % of one cell's area** — roughly HALF the pre-fix
figure of 79.4 %, and further from (not closer to) the 100%+ needed to
make the test bite. Every previous scale-up (19→37 cartridges) grew this
number; a geometry correction shrank it instead. The likely mechanism:
`P_direct` and `P_derived` are now both computed against a real cavity
floor rather than one of them inferring a floor from a flat top face, so
the two independent estimates agree more often — which is a genuine
improvement in estimate quality, and simultaneously a harder validation
target for τ, because the whole test needs the estimates to *disagree*
enough to approach a cell's footprint. Every record in the split still
fails the admission test *on area alone*; the morphological-versus-areal
distinction that `admits_a_cell` exists for (Task 2's blob-versus-rim
demonstration) is still never exercised.

**Everything in the two paragraphs above was the state of the diagnosis
before the mechanism and correlation measurement described at the top
of this item.** Growing the validation split's errors — via cluttered
bays and occlusion, spec #4's τ-targeted subset — was the conclusion
that diagnosis pointed to, and it is **no longer the right conclusion**:
a larger or harder-error validation split cannot fix a gate whose two
inputs are not independent and whose measured correlation has the wrong
sign. Building cluttered-bay content is still worth doing (Step 2
below), but on its own merits — `obstruction` and `battery` are
measurably the weakest two segmentation classes, IoU 0.6579 and 0.6907
respectively (`docs/receipts/seg_eval.txt`) — not because it will ever
make τ calibratable. (Both figures are from the *same-distribution*
validation split of the 220-scene render the model trained on. On the
disjoint CAD test set built for spec #2 they are lower for every model
measured, CAD-trained controls included — see "The generalisation
measurement" above before treating 0.6579/0.6907 as a floor.)

~~`plan/placement_area.py` still defaults to 0.85 and nothing reads the
calibrated value. That remains the right call, now for a stronger
reason than before: the calibration is not merely uninformative on this
split, it is retired as a mechanism. Wiring in any calibrated number
would misrepresent an inert gate as a working safety threshold.~~

**Superseded 2026-08-11 — τ is now retired in the CODE, not only in the
prose, and the delay had a measured cost.** Everything above described a
gate that was still running. `dee9854` changed the documentation and the
comments; `plan/placement_area.py` went on evaluating
`if iou < self.tau: raise PlacementDisagreement`. Three inconsistent
values were live at once and none agreed — constructor default **0.85**
(what every in-tree caller got), `configs/planning.yaml`'s
`arbitration.tau: 0.7492` (read by **nothing**, grep-verified), and the
README's **0.5715** (which described the YAML value as live). Measured on
15 `recog/dataset3d_seg` frames through the real detector and segmenter
(26 crops, 8 with a predicted `bay`): the gate admitted **3 of 8** at
0.85, 6 at 0.7492, 7 at 0.5715 — and at `configs/planning.yaml`'s own
`mm_per_px: 0.38`, **0 of 8**. That calibration widens the wall erosion
7 px → 11 px, shrinking `P_derived` until the whole observed IoU range
(0.639–0.848) sits below 0.85. **In the project's own configured
calibration the gate rejected every cartridge it was ever offered,
silently.** `5a619fc` deleted the branch, `self.tau`, the constructor
argument (deleted rather than accepted-and-ignored, so a stale caller
gets a `TypeError`) and the dead `arbitration.tau` key; both rows now
read 8 of 8. `P_safe = P_direct ∩ P_derived` is unchanged, applied
unconditionally, and separately pinned. `PlacementArea.consistency_iou`
is still computed and reported; nothing acts on it. **There is no live
description of τ anywhere in the docs any more — check before
reintroducing one.** Record:
`docs/superpowers/specs/2026-08-11-segmenter-integration.md`.

### 5. The validation split is small — still modest, and its per-class composition keeps shifting with the generator

Now 35 `bay`, 35 `electronics`, 24 `obstruction` instances over 126
validation crops (was 37/37/18 pre-fix; 19/19/11 before that) — the same
126 crops, but which units land in each class shifts each time the
generator or the split's underlying scenes change. The per-class numbers
moved with the tray fix, but **not uniformly for the better**: boundary
displacement improved on all three classes this time (bay 1.299→0.949 mm,
electronics 1.085→0.987 mm, obstruction 1.633→1.184 mm — see FDR §13.2.1
for the full table), but the pooled selected mean IoU dipped slightly
(0.8045 → 0.8032) even as the checkpoint's own per-epoch selection metric
rose (0.8096 → 0.8126), and Δcells' negative-direction fraction got worse
(1/126 → 2/126). `obstruction` in particular is still a 24-instance number
and should be read as such. And per item 4 (superseded above): for τ
specifically, neither sample size nor error size is the binding
constraint any more — τ is retired as a confidence gate outright,
independent of what a larger or harder split would show.

---

## What to do, in order

**Reordered 2026-08-09 around the constraint above.** The original plan put
real-photo collection first because it unlocked everything else; that step
is now impossible, and the three remaining specs
(`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8) are
reordered around what is actually measurable without photographs. The
reasoning for the new order is given inline at each step, not just the
order itself.

### Step 1 — Spec #2: procedural trays and 21700/26650 cell formats

**Now the single highest-value action**, promoted ahead of the
clutter/occlusion work that used to lead this list. That work was
ordered first on the theory that it would make τ calibratable; per
item 4 above, τ is now retired as a confidence gate for structural
reasons no amount of scene difficulty can fix (the two masks it
compares are not independent, and their measured correlation has the
wrong sign in all four SKUs), so that justification no longer applies
and nothing else took its place at the top until now. Spec #2 does not
depend on that outcome at all — it answers a different, still-open
question — which is why it leads instead.

`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8: 21700 and
26650 cell formats (the `battery` class definition already names 21700; no
CAD exists for it yet) plus a procedural cartridge-tray family — sampled
footprint, wall thickness, bay depth, cell count and pitch.

**This paragraph originally said the four Anker assemblies stay "in the
mix as real-CAD anchors" for training. That is superseded, not merely
imprecise.** `docs/superpowers/specs/2026-08-10-generalisation-decisions.md`
Decision 1 (dated 2026-08-10, settled after this paragraph was first
written) instead splits training and test entirely: the model trains
**only** on procedurally generated trays, and all **four** Anker CAD
assemblies are held out as a pure test set — "the model never sees real
measured geometry during training." A reader who remembers the earlier
"anchors" plan should read this as a deliberate change, not a
contradiction to reconcile: the CAD assemblies still matter — they are
now the entire test set, and a separate CAD-trained control model exists
specifically to give that test set a ceiling to compare against (see
`docs/superpowers/specs/2026-08-10-generalisation-design.md` §10) — but
they are no longer part of what any procedural model trains on. See that
design spec in full for the resulting 2×2 (anchored/wide procedural
training × held-out CAD test) and the control.

**This spec now carries a second purpose beyond variety.** Training on one
synthetic distribution (today: the four hand-modelled Anker assemblies) and
testing on a **disjoint** one (procedurally generated trays and cell
formats the training set never saw) is a legitimate generalisation
measurement — the model either holds up on shapes it was never shown or it
does not, and that question is answerable without a single photograph. It
is the best available proxy for robustness this project has left.

**State this plainly so it cannot be misquoted later: a synthetic-to-synthetic
generalisation result is NOT a sim-to-real measurement, and must never be
reported as one.** It answers a different question — does the
model generalise across procedural variation it was not trained on — not
whether it transfers to photographs, which item 1 above and "The
constraint this plan works around" have already established cannot be
measured on this project.

### Step 2 — The clutter/occlusion subset of spec #4: cluttered bays and occlusion

**Retained, but on different grounds than before.** This used to be
Step 1, justified as the way to make τ calibratable. Per item 4 above,
that justification is gone — no scene difficulty fixes a gate whose two
inputs are provably not independent and whose measured correlation has
the wrong sign. It is kept, demoted to second, because it is separately
worth doing: `obstruction` and `battery` are the weakest two
segmentation classes measured, IoU 0.6579 (24 instances) and 0.6907 (24
instances) respectively (`docs/receipts/seg_eval.txt`), and cluttered
bays / occlusion place exactly the kind of content — foreign objects
inside the visible bay floor — that these two classes need to get
better at distinguishing from clear floor. The success criterion is
therefore **segmenter IoU/boundary-displacement improvement on
`obstruction` and `battery`**, not a τ that starts rejecting cartridges.

- Build the cluttered-bay and occlusion halves of spec #4
  (`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8: "4 —
  Difficulty. Occlusion and clutter, lighting extremes, truncation and
  framing, and cluttered bays.") ahead of its lighting, truncation and
  framing halves, which stay in Step 3 below.
- Re-run `python -m recog.seg_evaluate` against the harder split once it
  exists and compare `obstruction`/`battery` IoU and boundary
  displacement to the current 0.6579/0.6907 and 1.184 mm baselines
  (`docs/receipts/seg_eval.txt`). **Compare against the disjoint CAD test
  set's figures too, not only those** — spec #2 measured
  `obstruction` at 0.6320–0.6507 there for CAD-trained models, i.e.
  below the 0.6579 quoted here, which means part of what this step is
  chasing is test-set difficulty rather than model weakness. See "The
  generalisation measurement" above. Re-running
  `python -m recog.calibrate_tau` is no longer a meaningful success
  criterion for this step — τ is retired regardless of what that split
  shows (item 4) — though the receipt can still be regenerated for the
  record.

### Step 3 — Spec #3 realism and the remainder of spec #4, as domain randomisation

`docs/superpowers/specs/2026-08-09-spec3-realism-decisions.md` plus the
parts of spec #4 not pulled into Step 2: lighting extremes, truncation and
framing.

**The logic changes here and should be stated, not assumed.** When the
target domain cannot be validated against — because there are no
photographs to validate against — the response is to widen the training
distribution to *cover* the target domain rather than measure the distance
to it. That is domain randomisation, and it is a strictly weaker claim than
a measured transfer result: it says "the training set now spans more of
what might be encountered," never "this was tested against what will be
encountered."

- The camera decision is already made, not still open: a fixed overhead
  machine-vision rig, ~400 mm working distance, near top-down, 0–10°
  tilt, no roll — not the handheld phone geometry of the existing photos
  in `recog/realtest/` (`2026-08-09-spec3-realism-decisions.md`, "Target
  camera: rig-realistic only"). Scene clutter — the blue jig plate with a
  real material, loose cells on the bench outside the cartridge, tools and
  cables, other cartridges partly in frame — is still to build.
- The material palette in `configs/synth3d.yaml` can still be informed
  qualitatively by the existing ~20-cartridge real set (colour, lighting
  character) even though, per item 2 above, that set cannot validate the
  result quantitatively.
- This step stays last among the three: it is the widest and least
  targeted of them. τ is no longer "the one thing currently measured and
  blocked" this step could address — it is retired outright, not scene-
  difficulty-dependent (item 4) — and unlike Step 1's generalisation
  measurement, this step does not produce a measurement at all, only
  broader coverage.

### Step 4 — Scale the synthetic set — DONE

~~1 hour of GPU time, and it firms up three things at once: τ calibration,
the small-class IoUs, and Δcells.~~ **Completed.** The dataset was grown
to 502 scenes / 841 crops; `recog/seg_training.py` gained a `--resume`
flag (model, optimiser, scheduler, epoch and best-so-far checkpointed
every epoch), needed because the retrain did not fit inside this
environment's per-command time cap in one invocation; the full 40-epoch
schedule was then run to completion across several `--resume`
invocations. Results: τ calibration is still uninformative, but the
reason is now precise rather than just "small sample" (item 4 above);
the small-class IoUs and boundary displacements moved, not uniformly for
the better (item 5 above); Δcells improved (item 3 above). The commands
below are kept for reference against a future further scale-up, not as a
pending action:

```
blender -b --python recog/generate3d.py -- --n 1000 --out recog/dataset3d_seg --device GPU --resume
python -m recog.seg_training --config configs/segmentation.yaml [--resume]
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
```

Note the Task 10 trade when re-reading results: insetting the bay proxy by
the wall thickness cost validation IoU 0.8158 → 0.7633 **on the original
220-scene / 361-crop dataset**, concentrated in `electronics` and
`obstruction`. That was accepted because it made the arbitration
informative at all. The dataset has since been scaled and retrained
(items 4/5 above carry the current numbers); whether the wall-inset trade
still looks right at the new scale has not been separately re-measured —
that would need the vacuous, uninset arbitration re-run on the current
dataset, which this pass did not do.

**A second, unrelated regeneration happened after this — not a scale-up,
a bug fix.** The tray had been rendering upside down and closed since the
generator existed; see the note near the top of this document and FDR
§13.2.1. The dataset was deleted and re-rendered at the *same* 502/841
scale (not scaled further), and the checkpoint retrained from scratch.
Every figure in items 1, 3, 4 and 5 above is from that regeneration, not
the scale-up described in this step.

### Step 5 — Close the loop on the open items

- ~~Delete the `tau` config key, or repurpose it as a fixed, explicitly
  un-calibrated conservative default (currently the key exists and
  nothing reads it; 0.85 is what `plan/placement_area.py` actually
  applies).~~ **DONE, `5a619fc`** — deleted, along with the `iou < tau`
  branch, `self.tau` and the constructor argument, after the gate was
  measured to reject 8 of 8 plannable cartridges at the project's own
  `mm_per_px`. See item 4 above.
- Resolve the 2/126 damage cases (was 1/126 pre-fix, 2/54 before that) per
  item 3 above — not yet individually re-investigated on the current split.
- Consider hardening `tests/test_synth3d.py`'s bpy-boundary check: it is a
  substring grep for `import bpy`, which `from bpy import context` walks
  straight past. It is the only enforcement of the architecture constraint
  that keeps `bay.py` testable.

### Step 6 — Real-robot validation

FDR §13.2(3). Out of scope until the lab KR 6 returns, and gated on everything
above.

---

## Things worth knowing before you touch this

**The generator now has real interior geometry (2026-08-09), and the two
paragraphs that used to stand here are obsolete.** `open_case` cartridges
were closed shells with a fake PCB and bay plane laid on the *outer top
face* until the tray-interior fix (four commits, `27cbd97`..`9fcf136`;
see the note near the top of this document and FDR §13.2.1). The tray is
now dropped from its lid (`case_lid` is a separate role), the cavity is
measured from the CAD (`tray_outer_mm`, `tray_floor_mm`, `interior_mm`,
`case_wall_mm` in `catalog.json` — `case_interior_mm` no longer exists,
having been the outer AABB despite its name), and the module, bay proxy,
obstructions and seated cells are all seated on the measured cavity
floor. An open cartridge now genuinely has depth and visible walls even
under the near-orthographic bird's-eye camera; this was NOT true before.

**"Two disjoint crop populations" is now WRONG and should not be
repeated as fact.** It described the pre-fix generator, where the module
and bay proxy covered a closed shell's entire top face so no `cartridge`
pixels survived on an open unit. That is no longer true: the tray walls
are now real standing geometry, mostly not covered by the floor-level
module/proxy, so an open unit's crop typically carries **both**
`cartridge` (the visible walls) and the bay classes together — measured
at 176 of 502 scenes carrying both in the same image. `recog/seg_dataset.py`'s
module docstring and a hardcoded note in `recog/seg_evaluate.py`'s
receipt output both still assert the old, now-false claim; this task's
brief scoped out source changes, so they were left as-is and are flagged
here as a known documentation-drift item for a follow-up, not fixed.

**The VOC data distribution moved**, though the schema did not. Seated cells
add `battery` instances, every open case now carries a PCB, bay plane and
glue, and cartridge boxes span whole units. The detector cannot break on
schema, but its numbers will differ from the FDR's published figures if
retrained.

**Execution ledgers** for all four plans are under `.superpowers/sdd/`, one
directory per plan. They record every review round, every ruling, and every
deferred minor with its reasoning. If something below looks arbitrary, the
reason is probably there.
