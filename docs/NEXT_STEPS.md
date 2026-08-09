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

`docs/FDR_v3.md` does not yet carry a dedicated, explicit limitation
statement to this effect (§13.2(5)'s prose gets close but stops short of
one). It needs one. That is a separate, deliberately larger edit and is
intentionally **not** done as part of this pass — noted here so it is
not forgotten.

Two corrections to how this document previously reasoned about the
constraint, both worked through in full below:

1. **τ calibration was implied to be blocked on real data. It is not**
   (item 4 below). The measured diagnosis is that the *error size* in
   the current synthetic validation split is too small, not that the
   data is synthetic. The fix is *harder synthetic scenes* — still
   synthetic, still reachable without a single photograph — which is
   why it now leads "What to do, in order."
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
| A | Forbidden-mask FFDH shelf advance | 3.17 → **14.28 cells** at 2.5 % coverage, 40/40 paired seed wins |
| B | Five-class segmentation ground truth from CAD | `placement_area` = currently-free floor, **0 overlapping pixels** across 3280 mask pairs (full 502-scene tray-interior dataset; was 139 pairs on a 32-scene spot check) |
| C | Per-ROI bay segmenter | IoU 0.8126; boundary displacement **0.949 mm** (bay) vs the 2.9 mm a mask head would quantise to |
| D | Integration and arbitration | Planning **2.0 ms/cartridge** vs an 8 ms budget; segmentation 20.2 ms for 8 crops vs 50 ms (was 16.7 ms; an intermediate 40.9 ms reading was GPU-contention noise, since superseded by a clean re-measurement — see the tray-interior retrain note below) |

New modules: `recog/synth3d/bay.py`, `recog/seg_dataset.py`,
`recog/seg_training.py`, `recog/seg_evaluate.py`, `recog/bay_segmenter.py`,
`recog/seg_ablation.py`, `recog/calibrate_tau.py`, `plan/arbitration.py`,
`scripts/forbidden_bench.py`.

621 tests. The torch-free demo (`python main.py --config configs/demo.yaml`)
still runs, which is what the FDR's reproducibility claim rests on.

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
anything it was not designed to do.

**Mitigations were tested and rejected on cost-benefit**: a larger wall inset
(to 7.5 mm), requiring `P_safe` itself to admit a cell, and extra `P_safe`
erosion (to 2 mm). None removes both negatives — `scene_00117` is untouched
across the entire sweep — while costing up to 30 % of the ground truth's
placeable cells. No code changed.

### 4. τ is still uncalibrated — not because the data is synthetic, but because the errors in it are too small

**A precision correction, stated explicitly because an earlier reading of
this section could be taken as "blocked until real data arrives," and
that reading is wrong.** τ calibration is not blocked on the data being
synthetic. It is blocked on the *errors* in the current synthetic split
being too small — measured below, not assumed. The fix is harder
synthetic scenes: still synthetic, and reachable without a single
photograph. That is why the τ-targeted half of spec #4 now leads "What
to do, in order," ahead of everything else remaining on this project.

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

So the blocker is **still not sample size** — and now demonstrably not
simply "scale the dataset further," since a geometry fix moved the
number backward relative to that trend. A validation set with *larger
errors* would be a stronger test of τ than merely a larger one — and
with real photographs not obtainable for this project (see "The
constraint this plan works around" above), the only route left is
**deliberately harder synthetic scenes**: cluttered bays and occlusion,
which directly enlarge the optimistic error this section measures. That
is spec #4's τ-targeted subset, promoted ahead of the rest of that spec
in "What to do, in order" below for exactly this reason — this is a
reachable next step, not a dead end. More of the same renders, at any
scale, will not get there, and a more geometrically accurate generator
moves further away, not closer — but a *harder* one should move it
forward.

`plan/placement_area.py` still defaults to 0.85 and nothing reads the
calibrated value. That remains the right call: wiring in a number this split
cannot justify would make it look calibrated without being so.

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
and should be read as such. And per item 4: for τ specifically, sample
size is still not the binding constraint — error size is, and it moved in
the harder direction this time.

---

## What to do, in order

**Reordered 2026-08-09 around the constraint above.** The original plan put
real-photo collection first because it unlocked everything else; that step
is now impossible, and the three remaining specs
(`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8) are
reordered around what is actually measurable without photographs. The
reasoning for the new order is given inline at each step, not just the
order itself.

### Step 1 — The τ-targeted subset of spec #4: cluttered bays and occlusion

**Now the single highest-value action**, replacing the old Step 1 ("collect
real photographs"), which is no longer possible. Per item 4's precision
correction: τ needs *larger* errors in the validation split, not more of
the same renders and not real data. Cluttered bays and occlusion are the
scene elements most likely to enlarge the optimistic error between
`P_direct` and `P_derived` past a cell's footprint — currently 42.0% of one
cell's area at the largest, down from 79.4% pre-tray-fix
(`docs/receipts/tau_calibration.txt`).

- Build the cluttered-bay and occlusion halves of spec #4
  (`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8: "4 —
  Difficulty. Occlusion and clutter, lighting extremes, truncation and
  framing, and cluttered bays.") ahead of its lighting, truncation and
  framing halves, which move to Step 3 below.
- **This is a deliberate deviation from the ordering that spec recorded**,
  which scheduled #4 last "so it stresses the generalised and realistic
  generator rather than the current one" — an ordering that assumed
  photographs would eventually be available to validate against. With none
  coming, τ becomes the primary measurable deliverable left on this
  project, and the τ-relevant half of #4 does not need #2 or #3 to exist
  first — it only needs harder bays, which sit in scope for the current
  generator already. The rest of spec #4 (lighting extremes, truncation,
  framing) stays where it was: last, folded into Step 3.
- **What this unblocks:** `plan/arbitration.py`'s two-estimate confidence
  gate runs on every cartridge today but has never yet rejected one in any
  calibration run measured (currently 0 of 35 — see item 4 and
  `docs/receipts/tau_calibration.txt`) — not because the gate is broken,
  but because no validation split so far has contained an error close
  enough to a cell's footprint to trip it. This step is what would make
  that gate functional rather than structurally inert.
- Re-run `python -m recog.calibrate_tau` against the harder split once it
  exists. Success looks like a τ the sweep *locates* by trading safety
  against throughput, rather than one it reports because it is the
  sample's minimum (contrast with the current sweep result in
  `docs/receipts/tau_calibration.txt`, where the fail budget never binds).

### Step 2 — Spec #2: procedural trays and 21700/26650 cell formats

`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8: 21700 and
26650 cell formats (the `battery` class definition already names 21700; no
CAD exists for it yet) plus a procedural cartridge-tray family — sampled
footprint, wall thickness, bay depth, cell count and pitch — with the four
Anker assemblies kept in the mix as real-CAD anchors.

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

### Step 3 — Spec #3 realism and the remainder of spec #4, as domain randomisation

`docs/superpowers/specs/2026-08-09-spec3-realism-decisions.md` plus the
parts of spec #4 not pulled into Step 1: lighting extremes, truncation and
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
  targeted of them. It does not address the one thing currently measured
  and blocked (τ, Step 1), and unlike Step 2 it does not produce a
  measurement at all — only broader coverage.

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

- Wire the calibrated τ, or delete the config key and state that 0.85 is a
  fixed conservative default (currently the key exists and nothing reads it).
  Depends on Step 1 producing a τ worth wiring — the current value is not.
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
