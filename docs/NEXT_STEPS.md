# Where this stands, and what to do next

Written 2026-08-08, after the segmentation extension landed on
`feat/blender-synth-dataset`. This is the pick-up-here document: what exists,
what is honestly unfinished, and what to do about it in what order.

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
| B | Five-class segmentation ground truth from CAD | `placement_area` = currently-free floor, **0 overlapping pixels** across 139 mask pairs |
| C | Per-ROI bay segmenter | IoU 0.8096; boundary displacement **1.299 mm** (bay) vs the 2.9 mm a mask head would quantise to |
| D | Integration and arbitration | Planning **2.0 ms/cartridge** vs an 8 ms budget; segmentation 16.7 ms for 8 crops vs 50 ms |

New modules: `recog/synth3d/bay.py`, `recog/seg_dataset.py`,
`recog/seg_training.py`, `recog/seg_evaluate.py`, `recog/bay_segmenter.py`,
`recog/seg_ablation.py`, `recog/calibrate_tau.py`, `plan/arbitration.py`,
`scripts/forbidden_bench.py`.

533 tests. The torch-free demo (`python main.py --config configs/demo.yaml`)
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

---

## What is honestly unfinished

### 1. The segmenter-vs-heuristic real-photo comparison has flipped sign once already

**This is the gap that matters, and it just got harder to read, not easier.**
On the 20 annotated cartridges in `recog/realtest/`, the completed
40-epoch checkpoint scores a placeable fraction of **0.232** against the
heuristic's **0.217** — beating it, where the mid-schedule checkpoint
(epoch 24, which had the *better* synthetic IoU) scored **0.211** on the
identical set and did not. Two checkpoints of the same training recipe,
same dataset, same config gave opposite verdicts against the same 0.218
design-spec threshold.

That is not evidence the domain gap has closed. It is evidence that at
n = 20, this comparison sits inside the noise of a single training run, in
either direction. The raw `bay` channel is still genuinely tiny on real
images before erosion — verified as a true domain gap in both runs, not a
measurement artefact — and nothing about the retrain targeted or fixed
that gap; scaling the synthetic dataset was not a real-photo intervention.

The segmenter still learned to segment *renders*. Whether it also
transfers to photographs is now genuinely unresolved rather than answered
negatively — which is worse for planning purposes, not better: a stable
negative would at least have ruled out shipping it as-is. See FDR
§13.2.1 for the full before/after and receipts.

### 2. Real-photo ground truth does not exist

`recog/realtest/` has 7 photographs and 20 cartridges annotated with **boxes
only — no segmentation polygons**. That is why the comparison above is a
placeable-fraction proxy rather than an IoU against human masks. No mask-level
real-world claim can be made until this exists.

### 3. Damage-direction crops — investigated on the prior split, not re-investigated on this one

Δcells is now mean **+0.032** over the grown 126-crop validation split,
with **1 of 126 negative** (down from +0.037 over 54 crops, 2 of 54
negative, on the pre-retrain split). Negative nominally means the
prediction packed cells the ground truth forbids.

**The investigation below describes the prior split's two negative crops
under the prior checkpoint; it has not been re-run against the current
single negative crop.** The retrain changed both the dataset and the
model, so which crop is negative now — and whether the same mechanism
explains it — was not re-identified. Read what follows as the documented
explanation for the earlier finding, generalisable lessons included, not
as a claim already verified about the current one.

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

### 4. τ cannot be calibrated on synthetic data — and now we know why, more sharply

Retrained to completion — the full 40-epoch schedule, not the epoch-24
checkpoint an earlier pass of this document reported against — on 502
scenes / 841 crops (2.3x the original), with the validation split's key
classes nearly doubled to 37 `bay` / 37 `electronics` / 18 `obstruction`.
τ came out at **0.3180**, accepting 37 of 37 cartridges. (The epoch-24
checkpoint had given τ = 0.6260, 36/36; both are lower bounds set by the
sample's own minimum rather than a calibrated threshold, so which one is
quoted does not change the conclusion below — if anything the completed
schedule's lower τ makes the point harder, not softer.)

**It is still uninformative, and scaling did not fix it.** Not one of the 37
accepted cartridges admitted a cell at any observed IoU, so the safety budget
never bound and τ remains the sample's lowest observed value rather than a
threshold found by trading safety against throughput.

**The diagnosis has moved on, and scaling made it sharper rather than
resolving it.** The largest optimistic error observed is now **2418 px
against a cell footprint of 3045 px² — 79.4 % of one cell's area** (was
51.9 % at the epoch-24 checkpoint, and 27 % on the original 19-cartridge
split before any of this retrain — the error grew both times the split
grew). Every record in the split fails the admission test *on area
alone*. The morphological-versus-areal distinction that `admits_a_cell`
exists for (Task 2's blob-versus-rim demonstration) is never exercised,
because no crop's error comes close enough to a cell's footprint for
shape to be the deciding factor.

So the blocker is **not sample size** — the sample has now grown twice,
across two different checkpoints, and the same finding held both times.
A validation set with *larger errors* would be a stronger test of τ than
merely a larger one — which means either real photographs (where the
errors are demonstrably bigger) or deliberately harder synthetic scenes.
More of the same renders will not get there.

`plan/placement_area.py` still defaults to 0.85 and nothing reads the
calibrated value. That remains the right call: wiring in a number this split
cannot justify would make it look calibrated without being so.

### 5. The validation split is small — improved, still modest, and no longer τ's binding constraint

Now 37 `bay`, 37 `electronics`, 18 `obstruction` instances over 126 validation
crops (was 19/19/11 over 54). The per-class numbers moved with it, but **not
uniformly for the better**: `obstruction`'s boundary displacement improved
(2.241 → 1.633 mm) and Δcells' negative-direction fraction shrank (2/54 →
1/126), but `bay`'s boundary displacement worsened (0.817 → 1.299 mm at the
completed schedule vs. the epoch-24 checkpoint) and the pooled/checkpoint
mean IoU is slightly lower at the completed schedule (0.8045 / 0.8096) than
it was at epoch 24 (0.8116 / 0.8209) — see FDR §13.2.1 for the full table.
`obstruction` in particular is still an 18-instance number and should be
read as such. And per item 4: for τ specifically, sample size is no longer
even the more binding of the two constraints.

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
3. ~~Re-run the ablation and see whether 0.211 now clears 0.217~~ **Done,
   and it did not settle the question.** The completed 40-epoch retrain
   reruns this exact comparison at 0.232 vs 0.217 — clears it — but the
   mid-schedule checkpoint from the *same* run scored 0.211 on the
   identical set and did not. See item 1 above and FDR §13.2.1: at n = 20
   this single number does not decide whether the segmenter ships, and a
   second training run flipping it is the reason why. What would decide
   it is real polygon-annotated ground truth (Step 1) or a larger
   real-photo set, not another retrain of the same recipe.

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

### Step 4 — Close the loop on the open items

- Wire the calibrated τ, or delete the config key and state that 0.85 is a
  fixed conservative default (currently the key exists and nothing reads it).
- Resolve the 1/126 damage case (was 2/54 on the prior split) per item 3
  above — not yet individually re-investigated on the new split.
- Consider hardening `tests/test_synth3d.py`'s bpy-boundary check: it is a
  substring grep for `import bpy`, which `from bpy import context` walks
  straight past. It is the only enforcement of the architecture constraint
  that keeps `bay.py` testable.

### Step 5 — Real-robot validation

FDR §13.2(3). Out of scope until the lab KR 6 returns, and gated on everything
above.

---

## Things worth knowing before you touch this

**The generator has no interior geometry.** Cartridges are closed shells with
a fake PCB and bay plane laid on the *top face* — valid only under the
near-orthographic bird's-eye camera (`camera.ortho: true`). A perspective or
oblique viewpoint would expose it.

**`case_interior_mm` is the outer AABB**, not a true interior. `case_wall_mm`
was added so the module and proxy could be inset off it. This caused one
subtle bug already; do not assume the name.

**Two disjoint crop populations.** Sealed cartridges carry `cartridge` pixels
and no bay; open ones carry the bay classes. No single crop teaches the
relationship, which is why `P_derived` needed the wall inset to be independent
of `P_direct` at all.

**The VOC data distribution moved**, though the schema did not. Seated cells
add `battery` instances, every open case now carries a PCB, bay plane and
glue, and cartridge boxes span whole units. The detector cannot break on
schema, but its numbers will differ from the FDR's published figures if
retrained.

**Execution ledgers** for all four plans are under `.superpowers/sdd/`, one
directory per plan. They record every review round, every ruling, and every
deferred minor with its reasoning. If something below looks arbitrary, the
reason is probably there.
