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
| C | Per-ROI bay segmenter | IoU 0.763; boundary displacement **0.963 mm** vs the 2.9 mm a mask head would quantise to |
| D | Integration and arbitration | Planning **2.0 ms/cartridge** vs an 8 ms budget; segmentation 16.8 ms for 8 crops vs 50 ms |

New modules: `recog/synth3d/bay.py`, `recog/seg_dataset.py`,
`recog/seg_training.py`, `recog/seg_evaluate.py`, `recog/bay_segmenter.py`,
`recog/seg_ablation.py`, `recog/calibrate_tau.py`, `plan/arbitration.py`,
`scripts/forbidden_bench.py`.

521 tests. The torch-free demo (`python main.py --config configs/demo.yaml`)
still runs, which is what the FDR's reproducibility claim rests on.

---

## What is honestly unfinished

### 1. The segmenter does not beat the heuristic on real photographs

**This is the gap that matters.** On the 20 annotated cartridges in
`recog/realtest/`, the segmenter scores a placeable fraction of **0.211**
against the heuristic's **0.217**. It was built to replace that heuristic.

On synthetic data it wins decisively. On real photographs it does not. The
raw `bay` channel is genuinely tiny on real images before erosion — verified
as a true domain gap, not a measurement artefact.

The segmenter learned to segment *renders*.

### 2. Real-photo ground truth does not exist

`recog/realtest/` has 7 photographs and 20 cartridges annotated with **boxes
only — no segmentation polygons**. That is why the comparison above is a
placeable-fraction proxy rather than an IoU against human masks. No mask-level
real-world claim can be made until this exists.

### 3. Two crops in the damage direction

Δcells is mean **+0.037** over 54 crops, with **2 of 54 negative** — the
predicted placement mask would let the packer site a cell where the ground
truth says it cannot fit. Under investigation as of this writing; see
`.superpowers/sdd/2026-08-06-D-integration-arbitration/damage-case-investigation.md`.

### 4. τ is not meaningfully calibrated

τ came out at 0.7492 with a rejected fraction of 0.0 — but **no cartridge ever
admitted a cell at any threshold**, so the safety budget never bound and τ is
simply the sample's lowest observed IoU. `plan/placement_area.py` still
defaults to 0.85 and nothing reads the calibrated value. That disconnect is
disclosed in FDR §13.2.1 and is deliberate: 0.85 is the conservative
direction, and calibrating properly needs more data.

### 5. The validation split is small

19 `bay`, 19 `electronics`, 11 `obstruction` instances. Every per-class number
for the three classes checkpoint selection depends on rests on double-digit
counts at best.

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
3. **Re-run the ablation** (`python -m recog.seg_ablation`) and see whether
   0.211 now clears 0.217. That is the number that decides whether the
   segmenter ships.

### Step 3 — Scale the synthetic set

~1 hour of GPU time, and it firms up three things at once: τ calibration, the
small-class IoUs, and Δcells.

```
blender -b --python recog/generate3d.py -- --n 1000 --out recog/dataset3d_seg --device GPU --resume
python -m recog.seg_training --config configs/segmentation.yaml
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt
python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt
```

Note the Task 10 trade when re-reading results: insetting the bay proxy by the
wall thickness cost validation IoU 0.8158 → 0.7633, concentrated in
`electronics` and `obstruction`. That was accepted because it made the
arbitration informative at all. With a larger set it is worth re-checking
whether the trade still looks right.

### Step 4 — Close the loop on the open items

- Wire the calibrated τ, or delete the config key and state that 0.85 is a
  fixed conservative default (currently the key exists and nothing reads it).
- Resolve the 2/54 damage cases per the investigation.
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
