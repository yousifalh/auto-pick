# Transfer-gap diagnosis — 2026-08-11

Diagnosis of the `bay` and `battery` shortfalls reported in
`docs/NEXT_STEPS.md` § "The generalisation measurement (spec #2)". Two jobs,
both measurement only: nothing about the metric definitions, the model
architecture, the training schedule or `TrayRangeCfg` was changed, no
existing dataset was regenerated, no existing checkpoint retrained. One new
model was trained.

**Every number in this document is synthetic-to-synthetic.** Real
photographs are unobtainable for this project, so sim-to-real transfer
cannot be measured at all and nothing here is evidence about it. The
question answered is narrower: does a segmenter trained on procedurally
generated cartridge trays transfer to the four real measured Anker CAD
assemblies it never saw?

Base commit `4fe2c85`, 666 tests passing, tree clean.

---

## Headline

**The `bay` transfer gap is not a gap in segmenting bays. It is a gap in
deciding whether a cartridge is open at all.**

On the 213 CAD test crops whose ground truth actually contains a bay, the
procedural model pools to **0.8801** against the CAD control's **0.9013** —
a difference of 0.021, not the 0.246 the headline figures show. The other
**91.4 %** of the published gap is produced on the **623 sealed crops**,
where the procedural model paints `bay` on the closed shell of 136 of them
(675 460 px of pure false positive) and the CAD control does so on 2 (722 px).

The cell-format hypothesis, tested with a purpose-built control model, is
**dead**: restricting the procedural trays to 18650 moved `battery` by
+0.017 where +0.15 or more was pre-registered as the material effect, and
moved `bay` slightly the wrong way.

---

## Job 1 — the cell-format mismatch hypothesis: NULL

### The prediction, stated before the render was started

Recorded in `.superpowers/sdd/2026-08-10-generalisation/progress.md` ahead
of any number, verbatim:

> If the three-format mix is what depresses `battery`, then on the same 836
> CAD test crops the 18650-only model should show `battery` rising
> materially from anchored's 0.5593 toward the CAD control's ~0.78, while
> `bay` moves little (< ~0.03) from anchored's 0.6555 — tray geometry is
> unchanged by dropping two cell formats, so there is no mechanism for a
> large `bay` move.

with the resolution limit stated in the same breath: n = 1 model per
condition, no seed-variance estimate exists for a from-scratch 40-epoch run
on this project, so only a move of roughly +0.15 (two thirds of the
available 0.22) was ever going to be resolvable, and **a null result is the
result**.

### What was built

`configs/synth3d_18650.yaml` is `configs/synth3d.yaml` with exactly one
parsed field changed — `tray_anchored.cell_formats`, `["18650", "21700",
"26650"]` → `["18650"]` — asserted field-by-field against the original
before the render, not eyeballed. `configs/segmentation_anchored_18650.yaml`
differs from `configs/segmentation_anchored.yaml` in exactly three keys
(`dataset.coco_path`, `dataset.img_dir`, `training.checkpoint_dir`), also
asserted.

`recog/dataset3d_seg_anchored_18650`: 502 scenes rendered at the same seed,
resolution, samples and `--tray-set anchored` as the anchored set. 502/502/502
images/annotations/meta, min image 763 745 bytes (no truncation), 2820 seg
annotations, **13 689 mask pairs checked, 0 overlapping**, manifest confirms
`cell_formats: ["18650"]`. The procedural pool was rebuilt in-process from
the same seed and is **502/502 18650**, against the anchored pool's
182/169/151 (18650/21700/26650) split. Crop count came out at **848 —
identical to the anchored set's 848**, with the same 721/127 train/val split
and the same `721 % 8 == 1` singleton final batch that `10867df` already
handles.

Trained from a fresh initialisation (empty `checkpoint_dir`, no `--resume`,
no fine-tuning) on the identical 40-epoch schedule. In-distribution best
selected mean IoU **0.7333** on 44/36/29 bay/electronics/obstruction val
instances, against anchored's 0.7322 on 43/35/29 — the two runs are
indistinguishable on their own splits.

### The result, on the same 836 CAD test crops

Receipt: `docs/receipts/seg_eval_anchored_18650_on_cad_test.txt`, generated
by `python -m recog.seg_evaluate --per-sku`, never hand-edited.

| class | procedural, anchored (3 formats) | procedural, **18650-only** | Δ | CAD control (hold out 10000) |
|---|---:|---:|---:|---:|
| **bay** | 0.6555 | **0.6191** | −0.0364 | 0.9131 |
| **battery** | 0.5593 | **0.5763** | **+0.0170** | 0.7833 |
| electronics | 0.7541 | 0.7534 | −0.0007 | 0.8634 |
| cartridge | 0.8088 | 0.7914 | −0.0174 | 0.9424 |
| obstruction | 0.6306 | 0.6306 | 0.0000 | 0.6507 |
| selected mean | 0.6801 | **0.6677** | −0.0124 | 0.8091 |

**`battery` moved +0.017 of the 0.224 available — 7.6 % of the gap.** The
prediction called for a material move toward 0.78 and did not get one. For
scale: `anchored` vs `wide`, two procedural sets that differ across *every*
sampled tray parameter, already differ by 0.009 on `battery` on these same
crops, so +0.017 is the same order as the project's own measured
dataset-to-dataset spread. **The cell-format mismatch does not explain the
`battery` gap.** Reported as the null it is.

`bay` moved −0.036, slightly the wrong way and slightly more than the
"< 0.03" the prediction allowed. Job 2's decomposition says exactly where it
went and it is not a bay-shape regression — see below.

Two things checked rather than assumed:

- **`obstruction` is identical to four decimal places (0.6306) between the
  two models.** That looks like a copied number and is not one: the per-SKU
  obstruction figures differ throughout (0.6612/0.7010/0.5043/0.6197 vs
  0.6757/0.6980/0.4919/0.6214). The pooled coincidence is a coincidence.
  `obstruction` parity with the CAD control remains a shared-code artefact
  (`world.build_obstructions`, one call site) and is still not transfer
  evidence.
- The receipt names checkpoint `seg_anchored_18650/best.pt` and reports
  best/last 0.7333/0.7333, matching this run's training log and not
  anchored's 0.7323/0.7309 — it is the new model, not a mis-pointed eval.

Per-SKU (18650-only model): `bay` 0.6668 / 0.6476 / 0.5763 / 0.6213 and
`battery` 0.3469 / 0.7365 / 0.5512 / 0.5574 for 10000 / 13000 / 20100 /
26800. AnkerPowerCore10000's `battery` still rests on 14 crops and is still
a small-sample estimate in every row it appears in.

---

## Job 2 — where `bay` transfer actually fails

Measured with the **existing** `seg_anchored` checkpoint on the **same** CAD
test crops, via a read-only diagnostic that reuses `recog.seg_evaluate`'s own
pixel path (`extract_crop` / `rasterise_crop` at native resolution, no
jitter, `BaySegmenter.segment_batch`). Its pooled `bay` IoU reproduces the
receipt's 0.6555 exactly, which is what licenses everything below.

### The `--per-sku` sanity check (its first real exercise, per the brief)

Independently recomputed and **matched to four decimal places on all eight
published per-SKU `bay` figures** — anchored 0.6376 / 0.6750 / 0.6344 /
0.6665 and the four leave-one-out controls 0.9005 / 0.8884 / 0.8988 /
0.9098. Crop counts 202 + 218 + 214 + 202 = 836, no `None` group. In the
sidecar, all 4139 unit groups carry a single consistent `asset` and none is
`None`, so `sample_assets` cannot be mis-attributing a unit.

One quantified caveat, found by checking rather than assuming: a crop is a
window, and `rasterise_crop` paints every annotation that falls in it, so a
crop can contain a *neighbouring* unit's pixels. **72 of 836 crops (8.6 %)
overlap an annotation belonging to a different SKU, but only 3 overlap a
foreign `placement_area` box** — so the per-SKU `bay` numbers are not
materially contaminated. `--per-sku` is trustworthy for what it claims.

### What kind of failure it is: two populations, not one distribution

The receipt's `bay` row reads `IoU 0.6555, instances 213`. The instance
count is the number of crops that *contain* a bay; the IoU is pooled over
the union accumulated across **all 836** crops. Those are different
populations and the single number hides it. Decomposing the 2 646 919 px of
union:

| | px | share of union |
|---|---:|---:|
| intersection | 1 735 064 | 65.6 % |
| **false positive on SEALED crops (no GT bay at all)** | **675 460** | **25.5 %** |
| false positive on crops that do have a bay | 152 343 | 5.8 % |
| false negative (missed GT bay) | 84 052 | 3.2 % |

| model | pooled `bay`, all 836 crops | pooled `bay`, only the 213 crops with a GT bay | sealed crops with a hallucinated bay |
|---|---:|---:|---:|
| procedural, anchored | 0.6555 | **0.8801** | **136 / 623 (21.8 %)**, 675 460 px |
| procedural, 18650-only | 0.6191 | **0.8839** | 154 / 623 (24.7 %), 838 185 px |
| CAD control (each SKU scored by the model that never saw it) | 0.9009 | **0.9013** | **2 / 623 (0.3 %)**, 722 px |

**On crops that contain a bay, procedural training is within 0.021 IoU of
the CAD-trained ceiling.** 91.4 % of the published gap is the sealed-crop
false positives.

It is not a leak of one class either — the model is deciding the *unit* is
open. Of the 136 sealed crops with a hallucinated bay, **92 (68 %) also
predict `electronics` or `battery`** on the same closed shell; the CAD
control produces that combination **zero** times. 30.7 % of every
`bay` pixel the procedural model predicts lands on ground-truth `cartridge`.

**On the open crops, the failure is mild, unimodal and boundary-shaped:**

- per-crop IoU: mean 0.8648, p05 0.753, p25 0.846, **p50 0.883**, p75 0.926,
  p95 0.952. 91 % of open crops are above 0.8.
- **Right place**: centroid offset p50 1.9 px (1.19 mm), p90 4.4 px, p99
  12.4 px; exactly **1 of 213** crops has its predicted bay more than half a
  GT-bay-diameter away.
- **Slightly too large, not too small**: predicted/GT area p50 1.029, pooled
  1.0375; 154 of 213 crops over-predict. Recall 0.934 mean vs precision
  0.907 — the model finds the bay and then spills over its edge.
- Boundary displacement 1.714 mm (pred→GT) / 1.354 mm (GT→pred), both well
  under the 2.9 mm mask-head quantisation figure.
- **The 3 crops below IoU 0.1 are not catastrophes of shape**: their ground
  truth bays are 2 px, 39 px and 322 px — slivers of a neighbouring unit
  caught at a crop edge. Two of the three are the sealed-unit failure
  showing up inside an open-crop statistic.

So the mean of 0.66 was neither uniform mediocrity nor a tail of
catastrophes *within* the bay-bearing crops. It is a good, slightly-fat
segmentation on every crop that has a bay, averaged with a large volume of
invented bay on crops that have none — and the catastrophic population is
structurally invisible to a per-crop IoU distribution, because those crops
have no ground-truth bay to score against.

### What correlates, and what does not

Spearman ρ of per-crop `bay` IoU on the 213 bay-bearing crops:

| correlate | ρ | reading |
|---|---:|---|
| bay area as a fraction of the crop | **+0.653** | strong |
| GT bay solidity (px / bbox area) | **+0.545** | strong |
| n obstructions in the unit | **−0.390** | moderate |
| obstruction px | −0.377 | moderate |
| GT bay area (px) | +0.375 | moderate |
| fill = battery px / (battery + bay) px | −0.344 | moderate |
| battery px in crop | −0.300 | moderate |
| n cells seated | −0.288 | moderate |
| electronics px | +0.132 | weak |
| crop area / crop scale | +0.085 | **nothing** |
| **GT bay aspect ratio** | **+0.017** | **nothing** |

The coherent story is *fragmentation*, not size or shape class: a bay broken
into thin strips between seated cells and scattered obstructions is the hard
case (solidity +0.545, obstructions −0.390, fill −0.344), and the loss is to
`battery` and `obstruction`, which together absorb 2.5 % of all GT bay
pixels. Looking at the worst open crops directly confirms it — `scene_00057`,
`scene_00298`, `scene_00413`, `scene_00394` are all densely-filled bays where
the ground truth keeps narrow yellow strips of free floor between the red
cells and the prediction merges them into cells. **Aspect ratio showed
nothing (ρ = +0.017) and apparent scale showed nothing (ρ = +0.085), both
checked because they were plausible and both reported as null.** The CAD
control degrades on the same axis (0.879 at 0 cells → 0.766 at 4), so this is
task difficulty, not a procedural-training weakness.

For the sealed-crop hallucinations the correlates are *appearance*, not
geometry:

| correlate | result |
|---|---|
| backdrop | **paper 54.3 %, concrete 17.9 %, belt 14.6 %, metal 12.1 %, fabric 10.0 %** |
| lighting | **mixed_daylight 45.8 %, high_bay_led 23.1 %, dim_workshop 22.8 %, warm_indoor 20.3 %, fluorescent 18.9 %, harsh_inspection 14.3 %, overcast_softbox 7.5 %** |
| layout mode | jig 28.1 %, scatter 19.5 % |
| apparent crop size (quartile) | 18.1 % → 19.2 % → 25.0 % → 25.0 % — weak |
| camera zoom (quartile) | 25.8 / 18.6 / 25.6 / 17.3 % — **nothing** |
| shell brightness (luma decile over the unit's own pixels) | 34 / 42 / 18 / 22 / 11 / 19 / 14 / 21 / 16 / 21 % — **no monotone trend, nothing** |
| exposure, cartridge px, n_assemblies (ρ on hallucinated area) | −0.020, +0.086, +0.091 — **nothing** |

Looking at the six largest hallucinations directly: they are tightly-framed,
top-down views of a **closed** cartridge — a plain flat slab filling the
crop — whose entire top face is painted `bay`, often with `battery` at one
end and `electronics` at the other. The model is reconstructing a whole open
cartridge on a closed one. Where it does this, it does it big: the
hallucinated bay covers a median 15 % of the closed shell's own pixels, and
**57 of the 136 cover more than 30 %**.

### The confound that had to be ruled out, and was

"The procedural sets simply contain fewer sealed examples" would explain
this trivially. It is false. Open-unit share of crops: anchored **27.6 %**,
18650-only **27.6 %**, wide 28.2 %, CAD control folds 27.8 % / 27.7 %, CAD
test set 25.0 %. Every training set carries the same open/sealed balance, so
the difference is not class prior.

What remains is the plain reading: the four Anker SKUs are one visual family
with one shell, so a leave-one-SKU-out control has already seen the test
SKU's *appearance* even though it never saw its geometry — and it makes this
error 2 times in 623. The procedural model has never seen an Anker shell at
all, and the discrimination it fails is appearance-level and binary: **is
this cartridge open or closed?** Neither cell format nor tray geometry has
any purchase on that question, which is exactly why Job 1 came out null.

---

## What this changes

Stated as findings, not as work items; nothing below was acted on.

1. **The published `bay` −0.25 and `battery` −0.20 gaps are not
   segmentation-quality gaps.** They are dominated by false positives on
   sealed cartridges. `docs/NEXT_STEPS.md`'s reading — "the shortfall tracks
   how much of each class's geometry the procedural tray builder invents" —
   survives for `cartridge` and `electronics` but is wrong as stated for
   `bay` and `battery`.
2. **Pooled per-class IoU with an instance count next to it invites this
   misreading.** `bay 0.6555, instances 213` pools a union accumulated over
   836 crops. The metric was not changed (out of scope here), but any future
   reading of it should carry the split. `electronics` (0.7541 → 0.7652
   present-only) and `obstruction` (0.6306 → 0.6316) are barely affected;
   `bay` (0.6555 → 0.8801) and `battery` (0.5593 → 0.6924) are transformed.
3. **`cartridge` is the honest remaining gap** — 0.8088 against the
   leave-one-out control composite's 0.9382, present in 835 of 836 crops,
   with no hallucination component available to explain it. It and
   `electronics` (0.7652 vs the composite's 0.8530, both present-only) are
   where procedural trays genuinely fail to look like the CAD. (The
   composite scores each SKU with the control that never saw it; the
   per-control receipts' 0.9387–0.9437 `cartridge` figures include the three
   SKUs each control did train on and are not the same statistic.)
4. The 18650-only model's `bay` drop is entirely in the hallucination channel
   (675 460 → 838 185 px); on crops with a real bay it is marginally *better*
   than anchored (0.8839 vs 0.8801). Dropping two cell formats did not make
   the model worse at bays; it made it slightly more willing to invent them.

---

## Receipts and reproduction

- `docs/receipts/seg_eval_anchored_18650_on_cad_test.txt` — the new model on
  the 836 CAD test crops, `--per-sku`, generated by `recog.seg_evaluate`.
- `configs/synth3d_18650.yaml` (+ its JSON sidecar), `configs/segmentation_anchored_18650.yaml`.
- Render: `blender -b --python recog/generate3d.py -- --n 502 --out
  recog/dataset3d_seg_anchored_18650 --device GPU --tray-set anchored
  --config configs/synth3d_18650.yaml --resume`
- Train: `python -m recog.seg_training --config configs/segmentation_anchored_18650.yaml`
- Score: `python -m recog.seg_evaluate --checkpoint
  recog/checkpoints/seg_anchored_18650/best.pt --config
  configs/segmentation_cad_test.yaml --per-sku --out
  docs/receipts/seg_eval_anchored_18650_on_cad_test.txt`
- Job 2's per-crop decomposition was produced by a scratch diagnostic that
  imports `recog.seg_evaluate`/`recog.seg_dataset` unmodified; it is not
  committed because it adds no capability the repo lacks — every figure it
  produced is reproducible from the checkpoints and the test set, and its
  headline number (pooled `bay` = 0.6555) is the receipt's own.

Datasets and checkpoints are gitignored, as every previous render in this
plan has been. Suite green at 666 throughout.
