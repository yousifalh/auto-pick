# Where this stands, and what to do next

Written 2026-08-08, after the segmentation extension landed on
`feat/blender-synth-dataset`; figures refreshed 2026-08-09 after the
tray-interior fix regenerated the dataset and checkpoint (see the note
below the "What exists" table). This is the pick-up-here document: what
exists, what is honestly unfinished, and what to do about it in what
order.

The goal is a **fully working pipeline** — CAD to a robot placing cells into
real cartridges. Most of it exists. The part that does not is named plainly
below, because it is the only thing standing between "built" and "works".

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

570 tests. The torch-free demo (`python main.py --config configs/demo.yaml`)
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

### 2. Real-photo ground truth does not exist

`recog/realtest/` has 7 photographs and 20 cartridges annotated with **boxes
only — no segmentation polygons**. That is why the comparison above is a
placeable-fraction proxy rather than an IoU against human masks. No mask-level
real-world claim can be made until this exists.

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

### 4. τ cannot be calibrated on synthetic data — and the tray fix moved the diagnosis in a new direction

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
errors* would still be a stronger test of τ than merely a larger one —
which means either real photographs (where the errors are demonstrably
bigger) or deliberately harder synthetic scenes. More of the same
renders, at any scale, will not get there — and a more geometrically
accurate generator moves further away, not closer.

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

### Step 1 — Collect and polygon-annotate real photographs

**The single highest-value action.** Nothing else closes the gap in §1.

- Target **50–100 images**, which is what the design spec
  (`docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md` §9.1)
  names as the prerequisite for any synth-to-real claim. It folds into FDR
  §13.2(4)'s existing 200–500 image programme.
- Cover what the synthetic set cannot: the black cartridges, bench lighting
  variation, cells at frame edges, partly-filled bays, and the adhesive/foam/
  tape/label clutter visible in `IMG_4426.jpg`.
- Annotate in CVAT with **polygons, not boxes**, across all five classes.
  Ruling 5 in the design spec applies: label only what the camera can see —
  trace the visible free floor, stopping at the edge of any cell, module or
  obstruction resting on it.
- Keep the existing 7 photos as a held-out set; do not train on them.

This is camera-and-CVAT work, not code.

### Step 2 — Fine-tune on real data, or bridge the domain gap

Once real masks exist, the options in rough order of cost:

1. **Fine-tune** the existing checkpoint on a real/synthetic mix. Cheapest,
   and the most likely to work given the model already segments the geometry
   correctly on renders.
2. **Widen the synthetic domain randomisation** toward the real photographs —
   the CAD carries no colour, and the hardware is black on a blue jig. The
   material palette in `configs/synth3d.yaml` was tuned before the real
   photos were closely compared against.
3. ~~Re-run the ablation and see whether 0.211 now clears 0.217~~ **Done
   twice, and it has not settled the question either time.** The
   completed 40-epoch retrain reran this exact comparison at 0.232 vs
   0.217 (clears it), the mid-schedule checkpoint from the *same* run
   scored 0.211 (did not), and this task's tray-interior retrain scored
   **0.318** (clears it by the widest margin yet). See item 1 above and
   FDR §13.2.1: at n = 20 no single number decides whether the segmenter
   ships, and three training runs producing three different verdicts is
   the reason why. What would decide it is real polygon-annotated ground
   truth (Step 1) or a larger real-photo set, not another retrain of the
   same recipe.

### Step 3 — Scale the synthetic set — DONE

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

### Step 4 — Close the loop on the open items

- Wire the calibrated τ, or delete the config key and state that 0.85 is a
  fixed conservative default (currently the key exists and nothing reads it).
- Resolve the 2/126 damage cases (was 1/126 pre-fix, 2/54 before that) per
  item 3 above — not yet individually re-investigated on the current split.
- Consider hardening `tests/test_synth3d.py`'s bpy-boundary check: it is a
  substring grep for `import bpy`, which `from bpy import context` walks
  straight past. It is the only enforcement of the architecture constraint
  that keeps `bay.py` testable.

### Step 5 — Real-robot validation

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
