# Model card

**Two trained models, nine checkpoints, one held-out set.** This page is the
index into them: what they are, what they were trained on, how well they do,
where they fail, and what they may not be used for.

Every table below is **generated from committed artefacts** by
[`scripts/model_card_tables.py`](../scripts/model_card_tables.py) — the
`docs/receipts/seg_eval*.txt` receipts, the `configs/segmentation*.yaml`
training configs and the dataset manifests in [`datasets/`](datasets/).
`python scripts/model_card_tables.py --check` fails if any figure here has
drifted from its receipt. Numbers that appear in prose rather than in a table
carry their source inline, and the four that trace to **no** committed receipt
are named as such in [§8](#8-what-is-not-receipt-backed) rather than left to
look like the rest.

> **Read this first.** Every accuracy figure on this page is
> **synthetic-to-synthetic**: measured on Blender/Cycles renders, against
> ground truth derived from those same renders. Sim-to-real transfer is
> **unvalidated, and cannot be validated within this project** — the
> photographs that would settle it will not be collected (FDR §13.2.2). No
> figure here is evidence about a photograph. Nothing is deployed, no
> checkpoint is published, and the checkpoints that exist **cannot be
> reproduced** ([§7](#7-reproducibility-and-provenance)).

---

## 1. What these models are

Two networks, in series, doing different jobs.

| | detector | segmenter |
|---|---|---|
| **architecture** | Faster R-CNN, ResNet-34 + FPN backbone | DeepLabv3 + MobileNetV3-Large |
| **task** | 2-class box detection (`battery`, `cartridge`) over the whole frame | 6-class per-pixel labelling of one cartridge crop |
| **classes** | background, battery, cartridge | background, cartridge, `bay`, electronics, obstruction, battery |
| **input** | frame resized to min 500 / max 900 px | 256 × 256 crop, fp16 |
| **runs** | once per frame | once per frame, **batched over every cartridge crop** |
| **code** | [`recog/model.py`](../recog/model.py), [`recog/inference.py`](../recog/inference.py) | [`recog/bay_segmenter.py`](../recog/bay_segmenter.py) |
| **config** | [`configs/recognition.yaml`](../configs/recognition.yaml) | [`configs/segmentation*.yaml`](../configs/) |

The segmenter is batched rather than looped for a measured reason, not a
stylistic one: at 8 cartridges it runs **17.0 ms batched against 52.7 ms
looped**, and the looped path breaches the 50 ms end-to-end budget on its own
(`docs/receipts/seg_eval_anchored_on_cad_test.txt`, latency table). Crop size
256 is at or above native crop resolution, so it is not a resample upward.

**Why the README gives different milliseconds for this same claim.** It quotes
**16.2 ms batched against 58.6 ms looped**, from `docs/receipts/seg_eval.txt`.
Neither figure is wrong and neither supersedes the other: they are two
different models on two different splits, and that latency row is wall-clock,
re-taken every time either receipt is regenerated — the spread across six
clean runs on this hardware is 16.2–21.2 ms batched (FDR §13.2.1). What both
receipts establish is the load-bearing part, identically: batched fits the
50 ms end-to-end budget and looped breaches it on its own. Cite the receipt
you are quoting and do not mix the two. *Note added 2026-08-14: the two
documents had carried two numbers for one architectural claim with nothing
saying why.*

There are **nine segmenter checkpoints**, not one. Eight are experiments
answering a single question — *does procedurally generated tray geometry
substitute for CAD?* — and one is what `configs/demo_seg.yaml` loads. They
share an architecture, a schedule and an augmentation pipeline; they differ
only in the dataset they saw.

## 2. Intended use, and explicit non-use

**Intended use.** Reproducing and inspecting the measurements in this
repository: the generalisation study in FDR §13.1.1, the sealed-unit
experiment, and the end-to-end wiring demonstration in `main.py`. These models
exist to support an argument about measurement, not to be run on anything.

**Out of scope, explicitly.**

* **Any real image.** See the banner above. There are now **two** measurements
  in this repository taken on photographs, and neither supports an accuracy
  claim made here. The first is a detector *input-resolution* ablation
  (`configs/recognition.yaml`: `inference_min_size` 500 → mAP@0.50 0.614 /
  @0.75 0.404, against 800 → 0.457 / 0.023, on the real-photo set), and it
  measures a preprocessing choice, not accuracy — **and it has no receipt**:
  those figures are an undated comment in the config that names no checkpoint,
  and they do not reconcile with the receipted run below taken at that same
  `min_size: 500`. Read them as a direction, not as numbers. The second is a
  held-out detector evaluation on the same phone photographs
  (`docs/receipts/real_photo_eval.txt`, 2026-08-13, generator
  `recog/eval_real.py`): shipped `best.pt` at confidence 0.70, **6 of the 7
  images scored** — one excluded for carrying no ground-truth boxes at all —
  **mAP@0.50 0.8484 / @0.75 0.8044** over 80 ground-truth boxes. Six
  photographs is not a sim-to-real validation and is not offered as one
  (FDR §13.2.2). *Corrected 2026-08-14: this bullet said "the one measurement
  in this repository taken on photographs" and named only the ablation; the
  second landed at `9b38de9`.*
* **Any safety-relevant decision about a lithium-ion cell.** The segmenter's
  characteristic error is *optimistic* — it labels tray wall as placeable
  floor — which is the direction that puts a cell on a PCB rather than the
  direction that loses one ([§6](#6-known-failure-modes)).
* **Any cartridge geometry outside the sampled band.** The four CAD SKUs lie
  strictly *inside* the `anchored` sampling band on every scalar axis, so the
  strong control results below are **interpolation, not extrapolation**. The
  extrapolation arm is `wide`, and `wide` came out null.
* **Deployment of any kind.** There is no export, no serving path and no
  target device — deliberately, and the reasoning is in FDR §10.4 and audit O
  §4: one in-process consumer, no second runtime, the latency budget already
  met, and no published checkpoint, so an export would be an artefact nobody
  could verify.

## 3. Training data

Every dataset is generated by [`recog/generate3d.py`](../recog/generate3d.py)
in Blender/Cycles from **measured Anker PowerCore CAD**, and each writes a
`manifest.json` carrying its full generator config, its seed, its class list
and its per-class instance counts. Those manifests are gitignored with the
datasets; **copies are committed under [`datasets/`](datasets/)** with
SHA-256 checksums, so a cloner who cannot regenerate ~8 GPU-hours of renders
can still read exactly what they contained. See
[`datasets/README.md`](datasets/README.md).

<!-- BEGIN GENERATED: training-data -->
| model | config | dataset | scenes | crops | generator seed | `split_seed` | `seed` |
|---|---|---|---:|---:|---:|---:|---:|
| `seg (shipping)` | [`segmentation.yaml`](../configs/segmentation.yaml) | [`dataset3d_seg`](datasets/dataset3d_seg.manifest.json) | 502 | 840 | 0 | 0 | 20260812 |
| `anchored` | [`segmentation_anchored.yaml`](../configs/segmentation_anchored.yaml) | [`dataset3d_seg_anchored`](datasets/dataset3d_seg_anchored.manifest.json) | 502 | 848 | 0 | 0 | 20260812 |
| `wide` | [`segmentation_wide.yaml`](../configs/segmentation_wide.yaml) | [`dataset3d_seg_wide`](datasets/dataset3d_seg_wide.manifest.json) | 502 | 814 | 0 | 0 | 20260812 |
| `anchored_18650` | [`segmentation_anchored_18650.yaml`](../configs/segmentation_anchored_18650.yaml) | [`dataset3d_seg_anchored_18650`](datasets/dataset3d_seg_anchored_18650.manifest.json) | 502 | 848 | 0 | 0 | 20260812 |
| `anchored_crown` | [`segmentation_anchored_crown.yaml`](../configs/segmentation_anchored_crown.yaml) | [`dataset3d_seg_anchored_crown`](datasets/dataset3d_seg_anchored_crown.manifest.json) | 502 | 848 | 0 | 0 | 20260812 |
| `cad_control_10000` | [`segmentation_cad_control_holdout_AnkerPowerCore10000.yaml`](../configs/segmentation_cad_control_holdout_AnkerPowerCore10000.yaml) | [`dataset3d_seg_cad_control_holdout_AnkerPowerCore10000`](datasets/dataset3d_seg_cad_control_holdout_AnkerPowerCore10000.manifest.json) | 502 | 852 | 0 | 0 | 20260812 |
| `cad_control_13000` | [`segmentation_cad_control_holdout_AnkerPowerCore13000.yaml`](../configs/segmentation_cad_control_holdout_AnkerPowerCore13000.yaml) | [`dataset3d_seg_cad_control_holdout_AnkerPowerCore13000`](datasets/dataset3d_seg_cad_control_holdout_AnkerPowerCore13000.manifest.json) | 502 | 852 | 0 | 0 | 20260812 |
| `cad_control_20100` | [`segmentation_cad_control_holdout_AnkerPowerCore20100.yaml`](../configs/segmentation_cad_control_holdout_AnkerPowerCore20100.yaml) | [`dataset3d_seg_cad_control_holdout_AnkerPowerCore20100`](datasets/dataset3d_seg_cad_control_holdout_AnkerPowerCore20100.manifest.json) | 502 | 852 | 0 | 0 | 20260812 |
| `cad_control_26800` | [`segmentation_cad_control_holdout_AnkerPowerCore26800.yaml`](../configs/segmentation_cad_control_holdout_AnkerPowerCore26800.yaml) | [`dataset3d_seg_cad_control_holdout_AnkerPowerCore26800`](datasets/dataset3d_seg_cad_control_holdout_AnkerPowerCore26800.manifest.json) | 502 | 852 | 0 | 0 | 20260812 |
<!-- END GENERATED: training-data -->

`crops` is the per-dataset `cartridge` instance count — one crop per physical
unit, which is what the segmenter actually trains on. `scenes` is rendered
frames. The 0.85 / 0.15 train–val split is applied over the flat crop list at
`split_seed: 0`.

**The hyperparameters are identical across all nine models.** That is the
premise the whole comparison rests on, so it is checked mechanically rather
than asserted — the generator reports any field that differs between models as
*differing* instead of collapsing it:

<!-- BEGIN GENERATED: shared-hyperparameters -->
| setting | value | key |
|---|---|---|
| crop_size | `256` | `model.crop_size` |
| half | `True` | `model.half` |
| train_val_split | `0.85` | `dataset.train_val_split` |
| jitter_frac | `0.06` | `dataset.jitter_frac` |
| epochs | `40` | `training.epochs` |
| batch_size | `8` | `training.batch_size` |
| learning_rate | `0.01` | `training.learning_rate` |
| lr_scheduler | `cosine` | `training.lr_scheduler` |
| dice_weight | `0.5` | `training.dice_weight` |
| deterministic | `warn` | `training.deterministic` |
| select_on | `bay, electronics, obstruction` | `training.select_on` |
<!-- END GENERATED: shared-hyperparameters -->

Checkpoint selection is on `bay`, `electronics` and `obstruction` **only**.
Including `background` and `cartridge` would let a model that gets the large,
easy regions right mask a failure on the three classes the placement mask is
actually built from.

## 4. Evaluation

### The held-out set

All eight comparable models are scored on **one common set**:
`recog/dataset3d_seg_cad_test` — a separate 500-scene render at
`train_val_split: 0.0`, yielding **836 crops over 434 frames**. It is built
from the four measured Anker CAD assemblies, which is what makes it a
*held-out asset pool* for the procedural and leave-one-SKU-out models scored
against it (FDR §13.1.1's phrase). The per-class instance counts behind every
row are constant:
**`bay` n = 213, `electronics` n = 213, `obstruction` n = 128**. Disjointness
from the training corpora is not assumed: FDR §13.1.1 records an MD5 check
over 4,536 renders across nine datasets finding **zero shared images in 36
pairings**.

Millimetre figures in the receipts are converted **per frame**, from each
frame's own render sidecar (median 0.7815, range 0.4903–1.0915 mm/px). An
uncalibrated frame raises `UnknownScale` rather than reverting to a constant.

### Consolidated comparison

Selected mean IoU over `[bay, electronics, obstruction]`, on the 836-crop CAD
test set. **This table is the one place all nine models appear together**;
before it existed, `anchored_18650`'s result appeared in no table in the
repository.

<!-- BEGIN GENERATED: segmenter-comparison -->
| model | trained on | scenes | `bay` | `electronics` | `obstruction` | **selected mean IoU** |
|---|---|---:|---:|---:|---:|---:|
| `seg (shipping)` | `dataset3d_seg` | 502 | — | — | — | **—** |
| `anchored` | `dataset3d_seg_anchored` | 502 | 0.6555 | 0.7541 | 0.6306 | **0.6801** |
| `wide` | `dataset3d_seg_wide` | 502 | 0.6536 | 0.7565 | 0.6280 | **0.6794** |
| `anchored_18650` | `dataset3d_seg_anchored_18650` | 502 | 0.6191 | 0.7534 | 0.6306 | **0.6677** |
| `anchored_crown` | `dataset3d_seg_anchored_crown` | 502 | 0.8755 | 0.7819 | 0.6360 | **0.7645** |
| `cad_control_10000` | `dataset3d_seg_cad_control_holdout_AnkerPowerCore10000` | 502 | 0.9131 | 0.8634 | 0.6507 | **0.8091** |
| `cad_control_13000` | `dataset3d_seg_cad_control_holdout_AnkerPowerCore13000` | 502 | 0.9045 | 0.8611 | 0.6320 | **0.7992** |
| `cad_control_20100` | `dataset3d_seg_cad_control_holdout_AnkerPowerCore20100` | 502 | 0.9032 | 0.8530 | 0.6412 | **0.7991** |
| `cad_control_26800` | `dataset3d_seg_cad_control_holdout_AnkerPowerCore26800` | 502 | 0.9044 | 0.8600 | 0.6322 | **0.7989** |
<!-- END GENERATED: segmenter-comparison -->

**How to read it, in one paragraph.** The four CAD controls (0.7989–0.8091)
are the ceiling: models that trained on renders of the same four measured CAD
assemblies the test set is built from. The two procedural arms — `anchored`
(0.6801) and `wide` (0.6794) — are the question, and they sit **0.12–0.13
below** that ceiling; widening the sampling band from `anchored` to `wide`
moved the result by 0.0007, which is a null result and is reported as one.
`anchored_18650` (0.6677) restricted the procedural cell format to the 18650
the CAD assemblies all use, and came out **below** `anchored`, closing none of
the gap — also null. `anchored_crown` (0.7645) is the one procedural change
that moved the number, and [§5](#5-the-crown-result-and-what-it-does-not-mean)
says what it does and does not license. The shipping `seg` checkpoint has **no
row here at all** — see the scope note below.

`obstruction` sits at ~0.63 for every procedural *and* CAD model. That is a
property of the harder test set, not a procedural weakness, and it is the one
class where the CAD controls buy almost nothing.

**Scope note — the checkpoint that ships is the one with no held-out row.**
`recog/checkpoints/seg`, which `configs/demo_seg.yaml` loads, was trained on
`dataset3d_seg` and has been scored **only on its own validation split** — 126
crops, `bay` n = 36, selected mean IoU 0.8032 (`docs/receipts/seg_eval.txt`).
That figure is not comparable to any row above: different set, one fifth the
crops, and subject to the frame-sharing bias in §"Checkpoint selection" below.
No receipt scores it on `cad_test`, and no document in this repository records
why not. Note that such a score would also mean something different from the
eight rows above: `seg` trained on renders of **the same four CAD assemblies
`cad_test` is built from**, so it would be held out on *images* but not on
*assets*, whereas the eight rows are the entire point of holding assets out.
**The eight comparable models are experiments; the shipping model has an
own-val number and nothing else.** Closing that gap needs a run this card does
not perform.

### Per-SKU, and the leave-one-SKU-out diagonal

`bay` IoU per CAD asset on the same 836-crop set. The four controls are
leave-one-SKU-out folds: each trained on a render set with one SKU removed and
is scored here on all four.

<!-- BEGIN GENERATED: segmenter-per-sku -->
| model | `PC10000` | `PC13000` | `PC20100` | `PC26800` |
|---|---:|---:|---:|---:|
| `seg (shipping)` | — | — | — | — |
| `anchored` | 0.6376 | 0.6750 | 0.6344 | 0.6665 |
| `wide` | 0.5930 | 0.6204 | 0.6618 | 0.6916 |
| `anchored_18650` | 0.6668 | 0.6476 | 0.5763 | 0.6213 |
| `anchored_crown` | 0.8430 | 0.8783 | 0.8706 | 0.8890 |
| `cad_control_10000` | **0.9005** † | 0.9037 | 0.9141 | 0.9221 |
| `cad_control_13000` | 0.8997 | **0.8884** † | 0.9097 | 0.9117 |
| `cad_control_20100` | 0.8961 | 0.8960 | **0.8988** † | 0.9129 |
| `cad_control_26800` | 0.9092 | 0.8907 | 0.9055 | **0.9098** † |

† the SKU that fold never trained on. The four held-out values are 0.9005, 0.8884, 0.8988, 0.9098 — mean **0.8994** — and the held-out SKU is the worst cell in its own row in **2 of 4** folds.
<!-- END GENERATED: segmenter-per-sku -->

The diagonal is the informative cell, and it is close to the rest of its row.
Holding a SKU out costs very little — which is a statement about **how similar
these four SKUs are to each other**, not evidence of general transfer. Four
folds over four assets from one product family is a small n by any standard,
and every one of them is a render.

### Checkpoint selection, and its noise floor

`best.pt` and `last.pt` are both shipped for every model because the
difference between them is **at or below the noise floor** — the largest gap
across all nine is 0.0036. Selection therefore is not meaningfully choosing.

<!-- BEGIN GENERATED: checkpoint-selection -->
| model | own-val n (bay/elec/obs) | `best.pt` | `last.pt` | Δ | re-scored | gap |
|---|---:|---:|---:|---:|---:|---:|
| `seg (shipping)` | 36/36/24 | 0.8126 | 0.8091 | 0.0036 | 0.8032 | +0.0094 |
| `anchored` | 43/35/29 | 0.7323 | 0.7309 | 0.0013 | 0.7161 | +0.0162 |
| `wide` | 36/18/19 | 0.6708 | 0.6677 | 0.0031 | 0.6489 | +0.0219 |
| `anchored_18650` | 44/36/29 | 0.7333 | 0.7333 | 0.0000 | — | — |
| `anchored_crown` | 43/35/29 | 0.7273 | 0.7268 | 0.0005 | — | — |
| `cad_control_10000` | 34/32/21 | 0.7864 | 0.7853 | 0.0011 | — | — |
| `cad_control_13000` | 38/36/20 | 0.7960 | 0.7960 | 0.0000 | — | — |
| `cad_control_20100` | 34/33/21 | 0.8382 | 0.8377 | 0.0005 | — | — |
| `cad_control_26800` | 39/39/21 | 0.8207 | 0.8195 | 0.0012 | — | — |
<!-- END GENERATED: checkpoint-selection -->

Three scope statements belong with this table.

1. **Own-val n is small** — 34–44 `bay` instances against the held-out set's
   213. These columns are not comparable to §4's table and are not a second
   opinion on it.
2. **Own-val is optimistically biased.** `_split_dataset` splits the flat crop
   list, and crops from one rendered frame land on both sides: **93 of
   `anchored`'s 127 validation crops (73.2 %) come from a frame that also
   contributed to training** (FDR §13.1.1). The held-out 836-crop figures are
   unaffected — `cad_test` is a separate render.
3. **The training-time metric reads high.** Where a `seg_evaluate` re-score of
   the same checkpoint on the same split exists, it lands **0.0094–0.0219
   below** the figure training recorded (the `gap` column, three of nine
   models). The card quotes the re-score where it has one. The systematic
   direction is unexplained here; fp16 evaluation against fp32 training is the
   obvious candidate and has not been tested.

### The detector — and why it has no comparable number

<!-- BEGIN GENERATED: detector -->
| metric | k-means anchors (`best.pt`) | torchvision-default anchors |
|---|---:|---:|
| mAP@0.50 | 0.7643 | 0.8736 |
| AP_battery@0.50 | 0.7896 | 0.9053 |
| AP_cartridge@0.50 | 0.7390 | 0.8419 |
| mAP@0.75 | 0.3051 | 0.5831 |

Both columns are the **same 15-frame validation split**, ground truth {2: 29, 1: 105} (`2` = cartridge, `1` = battery). Fifteen frames is a very small n and these are the widest confidence intervals in this card.
<!-- END GENERATED: detector -->

**This table is an anchor-set ablation, not the shipping detector's accuracy,
and the distinction is load-bearing.** Both columns come from the
**2026-04-20** run recorded in `docs/receipts/train_eval.txt`: 100 images from
`recog/synth_dataset.py` — flat cv2-drawn rectangles, not renders — split
85/15, ResNet-34 + FPN trained **from scratch with no COCO pretraining** for
15 epochs. As a controlled comparison of two anchor sets under one schedule it
is valid, and its finding stands: the Project Plan Report's k-means anchors
*hurt* mAP@0.50 by 0.11 relative to torchvision's defaults.

What ships is neither column. `configs/recognition.yaml` carries a **third**
anchor set (`anchor_ratios: [0.28, 0.5, 1.0, 2.0, 3.5]`,
`anchor_scales: [40, 64, 96, 144]`) re-tuned against the Blender render corpus
that replaced the April one, trained on `recog/dataset3d` with COCO
pretraining for 35 epochs (FDR §5.7 and ADR-005, both corrected 2026-08-12).
Its *in-training* metric is uninformative by the config's own admission: the
full-set synthetic metric "saturates at mAP 1.0 within 0–5 epochs", which is
why `hard_val_fraction` exists at all. But it **is** scored on a held-out
split, by a committed receipt with a committed generator.

**`docs/receipts/detector_bench.txt`, arm 3** — the shipped
`recog/checkpoints/best.pt` at the production configuration (confidence 0.70,
NMS IoU 0.40, inference resize 500/900) over the **`recog/dataset3d`
validation split, 150 frames / 1 205 ground-truth boxes** — reports
**mAP@0.50 = 0.9053** (AP_battery 0.9046, AP_cartridge 0.9061), mAP@0.75
0.7607, precision 0.9488, recall 0.9544. Generator: `scripts/detector_bench.py`.
FDR §10.5 and Appendix G cite this same row. **Scope: in-domain** — a held-out
split of the same Blender corpus, same generator, same catalogue. Out of
domain, the same checkpoint scores mAP@0.50 0.8484 on six phone photographs
(`docs/receipts/real_photo_eval.txt`), which does not clear the 0.90 bar and
does not sample enough to settle it.

**Corrected 2026-08-14: this paragraph said "No committed receipt scores that
model" and that "the detector that ships has no published held-out mAP".**
Both were contradicted by `detector_bench.txt` arm 3, which is committed,
regenerable, and cited for exactly this purpose elsewhere in the FDR. What
remains true is the narrower point the passage was reaching for: the two
anchor-ablation columns above are April-corpus numbers on a different
checkpoint and still must not be quoted as the shipping detector's score.

**Quote the 0.9053 with its arm attached.** The same receipt's **arm 2** — the
torchvision-default-anchor checkpoint `default_anchors_best.pt`, on the
15-frame `recog/dataset` val split at a 0.05 score threshold — reports
`AP_battery@0.50` = **0.9053** as well. That collision is a coincidence of
rounding, not one measurement quoted twice: different arm, different quantity
(a single-class AP against a mAP), different checkpoint, different split, and
arm 2's precision column is explicitly not quotable at that threshold. Arm 3's
**mAP@0.50** is the shipping detector's held-out figure; arm 2's `AP_battery`
is not. (Nor should either be paired with the 0.8647 in earlier audit prose —
that is `last.pt`, a third network.)

The anchor set that *does* ship is the best-documented hyperparameter in the
repository: chosen on 2,434 boxes from 300 freshly generated scenes at two
seeds, with an exhaustive search over 84,156 four-tuples recorded **and
rejected**, reasoning in `configs/recognition.yaml`'s own comments.

Latency, CPU, 2 threads, 320 × 512 input, 100 frames: mean **446.0 ms**,
p95 484.2 ms (`docs/receipts/frcnn_latency.txt`). The `HeuristicDetector`
fallback runs at median 3.3 ms and is what a torch-free clone gets.

## 5. The crown result, and what it does not mean

`anchored_crown` is the only procedural intervention that moved the held-out
number (0.6801 → **0.7645**), and the mechanism is understood: a closed
cartridge whose top face is *not flat* was absent from the procedural training
distribution, so the model had learned "featureless flat top ⇒ open, sealed
unit ⇒ hallucinate a bay". Adding a lid crown to the generator put that case
into the distribution. `bay` IoU went 0.6555 → 0.8755, the largest single-class
movement in this card.

**It is not evidence of transfer, and the reason is in the method.** The crown
range `[0.0, 12.0]` mm was chosen *after* measuring the Anker lids, and the
real value 11.10 mm lies inside it. The honest claim is the narrow one: **the
measured coverage gap was the mechanism.** It is domain randomisation informed
by a measurement, which says nothing about a shell family nobody measured. The
experiment is `docs/superpowers/specs/2026-08-11-sealed-unit-experiment.md`;
its own §"What this does and does not license" says the same thing.

Two things stayed put and are reported as such: `obstruction` did not move,
and the `paper` backdrop's separate ~50–55 % effect is unexplained by the
crown and was not chased.

## 6. Known failure modes

This project knows how it fails better than it knows how well it works. The
table is the consolidated view; each row names where the evidence lives.

| # | failure mode | how it presents | how you would detect it | what it costs | evidence |
|---|---|---|---|---|---|
| 1 | **Optimistic bay boundary.** The predicted `bay` runs wide into the tray wall. | Placements land on wall, not floor. | Signed placeable-area error against ground truth — it is *signed* precisely so the direction is visible. | `bay` boundary displacement **1.226 mm**; placeable area optimistic by **+79.2 mm²/crop** on 126 val crops. | `docs/receipts/seg_eval.txt` |
| 2 | **Residual unsafe placements.** | 2 of 25 placed cells overlap a tray wall, by 8.3 % and 5.2 % of footprint. | Overlap test against **ground-truth** masks; the predicted masks cannot see it (see #3). | 2 cells mis-seated on one cartridge's left wall. Down from 5 of 26, worst case 100 %. | `docs/superpowers/specs/2026-08-11-placement-safety.md` §2.3 |
| 3 | **Self-checking does not work.** Every planner guard — `P_safe`, clearance inset, overlap test — is computed from the same label map that is wrong. | Guards agree with the error. Four of five original offenders sit **100.0 %** inside the predicted free floor. | You cannot, from the prediction. Only reduced boundary error or force-sensed contact at placement time. | A clearance margin was proposed, measured, and **rejected**: 1.5 mm gives up 4 cells and *creates* a third overlap. | README "The structural insight"; `docs/superpowers/specs/2026-08-11-placement-safety.md` §2.4 |
| 4 | **Sealed-unit hallucination.** A closed cartridge gets a confident `bay` where there is no cavity. | The pipeline plans into a sealed unit. | Per-crop false-positive rate on sealed crops. | 21.8 % of sealed crops → 2.6 % after the crown fix. **Not receipt-backed** — see [§8](#8-what-is-not-receipt-backed). | `docs/superpowers/specs/2026-08-11-sealed-unit-experiment.md` |
| 5 | **One cartridge cannot be certified at any accuracy.** `AnkerPowerCore10000`'s bay is 54.9 × **65.0 mm**; the cell is 18.5 × **65.0 mm**. Exactly. | Zero cells placed, always. | Geometry, not perception — the packer alone consumes ~0.32 mm per degree of the ±2° seating jitter, against 0.00 mm of margin. | **10 of 10** instances place zero cells *on ground-truth masks*, and **47 of 47** over the whole 502-scene corpus. Neither knob is the lever: 18.5 → 18.3 mm recovers 0, and the 4.25 mm wall inset taken to 0.00 mm recovers 0 (it recovered 1 until the 2026-08-15 packing-strip fix). | `docs/receipts/placement_feasibility.txt` §2–§3; README "A cartridge that cannot be certified"; FDR §3 |
| 6 | **Domain cliff on the demo corpus.** | The segmenter predicts **no `bay` at all** on `synth_dataset.py`'s flat green rectangles. | A run producing zero placement areas — which `main.py` treats as a failed run and raises on, rather than completing quietly. | `configs/demo_seg.yaml` must read renders, not the demo corpus. | README "The same loop with the trained segmenter" |
| 7 | **The heuristic extractor does not work on real cartridges.** Otsu on green assumes a light tray with a dark module. | Zero placeable area. | Run it on the annotated real photographs. | **Zero placeable area on 7 of the 20** real annotated cartridges. Demo-only path; warns at construction. | README "Two placement-area extractors" |
| 8 | **Checkpoint selection is noise-limited.** | `best.pt` and `last.pt` are indistinguishable. | The Δ column in §4. | ≤ 0.0036 across nine models — selection is not choosing. | the receipts' own notes |
| 9 | **Detector input resolution is a cliff, not a slope.** | Boxes collapse at the wrong `min_size`. | Sweep it. | 500 → mAP@0.75 **0.404**; 800 → **0.023**. Measured on real photographs. | `configs/recognition.yaml` |
| 10 | **Own-val figures are optimistic.** | In-distribution scores read high. | Check whether val crops share a frame with train crops. | 73.2 % of `anchored`'s val crops do. Fixing it means retraining all eight models. | FDR §13.1.1 |

Row 1 and row 2 are the same defect seen from opposite ends, and row 3 is why
neither can be patched downstream.

**The operating envelope, stated plainly.** A bay packed to exact tolerance
cannot be certified by *any* vision system with non-zero measurement error,
because certification needs margin to absorb that error and an exact fit
offers none. This is a specification problem, and the available responses are
specification changes — not accuracy work. It is the most transferable finding
in this repository and it is not about these models at all.

## 7. Reproducibility and provenance

**The published checkpoints cannot be reproduced, and this is stated rather
than worked around.** Two reasons, both dated:

1. **They predate seeding.** Until 2026-08-12 training here was genuinely
   unseeded — no `torch.manual_seed`, no `use_deterministic_algorithms`, and
   `DataLoader(shuffle=True)` with no `generator=`. Two one-epoch runs of the
   same command on the same data gave selected mean IoU **0.4111 and 0.3957**.
2. **They predate a BatchNorm fix.** Runs after that commit differ for a
   second, independent reason.

A from-scratch reproduction therefore returns *a sample from the same
distribution*, and a figure that comes back 0.01 off is not a discrepancy. The
mechanical confirmation is in the checkpoints themselves:
`recog/seg_training.py` writes `seed` and `seeding` into every checkpoint it
saves, and **no checkpoint in the working tree carries either key** —
[§8](#8-what-is-not-receipt-backed) records how that was checked.

**Training is seeded now, and the seeding has its own receipt.**
`recog/seeding.py` fixes Python's `random`, NumPy, torch on CPU and CUDA, the
DataLoader `generator` and `worker_init_fn`, the albumentations pipeline and
the crop jitter. Seeding the RNGs was **not enough**: with kernels
unconstrained, two same-seed runs still diverged by up to 0.0409 selected IoU,
so `training.deterministic` defaults to `warn`, under which two runs at one
seed produce **bit-identical weights**. `strict` refuses to train this model at
all — `nll_loss2d_forward_out_cuda_template` has no deterministic CUDA
implementation. The claim is therefore **reproducible on the same machine and
toolchain**, not reproducible unqualified.
Receipt: `docs/receipts/seed_reproducibility.txt`.

**What *is* exact** is the evaluation: given the checkpoints and datasets, the
eleven `seg_eval*.txt` receipts regenerate byte-identically on every metric.

**Dataset identity.** Each checkpoint records `coco_path`, `split_seed`,
`ious` and `val_instance_counts`. But `coco_path` is a **path, not an
identity** — two different renders can occupy the same path, and nothing in
the checkpoint pins the *content* it trained on. [`datasets/`](datasets/)
records the SHA-256 of every annotation file to convert one into the other;
what that does and does not buy is stated there. It is not idle: the
`anchored` and `anchored_crown` annotation files are **the same size to the
byte** (3,408,089) and differ only in their hash.

## 8. What is not receipt-backed

Four things on this page, named so they are not mistaken for the rest.

1. **The 21.8 % → 2.6 % sealed-crop rates** (136/623 → 16/623, failure mode
   #4) came from a **scratch diagnostic that was never committed and emitted
   no receipt**. Its anchor — pooled `bay` 0.6555 — *is* a receipt figure, and
   the 0.6555 → 0.8755 movement in §4's table is fully receipt-backed. The
   per-crop rates are not. README "Where to look" says the same.
2. **The 2-of-25 placement figures** (failure mode #2) and the *shipping* side
   of the oracle comparison trace to a results table in
   `docs/superpowers/specs/2026-08-11-placement-safety.md`, a committed document, but not to a
   receipt with a committed generator. (**Retracted in part, 2026-08-15: this
   item used to put the whole "oracle comparison" here.** The ground-truth
   oracle acquired both at `b69bcb3` — `docs/receipts/placement_feasibility.txt`,
   generated by the committed `scripts/placement_feasibility.py` and guarded by
   `tests/test_placement_feasibility_receipt.py`, with a `--check` mode that
   fails on drift. It regenerates from the committed CAD catalogue, the
   committed planning config and the ground-truth annotations, needs no
   checkpoint, and at HEAD reports 9 of 29 open cartridges and 23 cells at
   every wall inset from 4.25 mm down to 0.00 mm. Only the shipping-pipeline
   half, which needs the detector-and-segmenter pairing pass, is still
   un-receipted; it has not been re-measured since the 2026-08-15 packing-strip
   fix and should not be quoted until it is.)
3. **"No checkpoint carries a `seed`"** (§7) was measured by loading all nine
   `best.pt` files in the author's working tree and listing their non-weight
   keys. The checkpoints are gitignored, so a cloner cannot re-run it. Command:
   `python -c "import torch,glob,os; [print(d, sorted(k for k in torch.load(os.path.join(d,'best.pt'), map_location='cpu', weights_only=True) if k!='model')) for d in sorted(glob.glob('recog/checkpoints/seg*'))]"`
   Result: every checkpoint carries `epoch`, `ious`, `selected_mean_iou`,
   `select_on`, `val_instance_counts` and `split_seed`; all but
   `recog/checkpoints/seg` also carry `coco_path`; **none carries `seed` or
   `seeding`.**
4. **The dataset SHA-256 checksums** in [`datasets/`](datasets/) were computed
   from the gitignored datasets in the author's working tree. A cloner cannot
   verify them against anything. What they buy is stated in
   [`datasets/README.md`](datasets/README.md) and it is narrower than it looks.

Sixteen of the repository's **forty** committed receipts are inherited from
before this repository's history and have **no surviving generator**; FDR
Appendix C enumerates them. (**Corrected 2026-08-15: the denominator read
thirty-nine**, which was the count before `placement_feasibility.txt` landed at
`b69bcb3`. `git ls-files docs/receipts | wc -l` is 40. The numerator is
unchanged — every receipt added since the appendix was written has a committed
generator.)

## 9. Regenerating this page

```bash
python scripts/model_card_tables.py            # rewrite every table
python scripts/model_card_tables.py --check    # fail if a figure has drifted
```

`--check` reads only committed artefacts and runs on a bare clone without
torch. Refreshing the dataset manifests needs the datasets present and is the
author's command:

```bash
python scripts/model_card_tables.py --sync-datasets
```

---

**Trademarks.** Anker and PowerCore are trademarks of their respective owner;
this project is unaffiliated with and not endorsed by them. The power banks
named throughout are retail units used as measurement subjects for academic
research. Full notice: [`README.md`](../README.md#trademarks).
