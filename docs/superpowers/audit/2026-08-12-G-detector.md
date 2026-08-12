# Audit G — the detector half

**Date** 2026-08-12 · **HEAD** `39429a4` · **Scope** read-only. Nothing modified,
nothing staged, nothing retrained, nothing regenerated. The only compute spent
was two forward passes of `recog/checkpoints/last.pt` over the seven real
photographs, and a few thousand augmentation draws.

**Scope in files** `recog/model.py` (127), `recog/training.py` (526),
`recog/dataset.py` (433), `recog/augmentation.py` (364), `recog/evaluate.py`
(189), `recog/inference.py` (384) — plus `recog/eval_real.py` (631), which is
what actually publishes the detector's headline numbers and could not be left
out, and `recog/seeding.py:369-402`, which turned out to be the site of the
worst finding.

Everything below is labelled **MEASURED** (I constructed a case, computed the
answer by hand or with an independent implementation, and ran the code) or
**READ** (inferred from the source without exercising it). The previous audits'
lesson is that reading agreement with a docstring is worth very little here, so
the ranked findings are all MEASURED.

---

## Summary — ranked by realistic consequence

| # | Site | What it does silently | Status |
|---|------|----------------------|--------|
| 1 | `recog/seeding.py:383-391` × `recog/augmentation.py:105-109` | Seeding hands all 17 child transforms **the same seed**, so `HorizontalFlip` and `VerticalFlip` draw identical streams and fire in lockstep. The dihedral group collapses from **8 symmetries to 4**; `p_geometric` silently becomes 0.55 instead of 0.25 | **MEASURED**, fires on every seeded run — i.e. every run since 2026-08-12. Same defect in the segmenter pipeline |
| 2 | `recog/evaluate.py:104-110` + `:160` | A class with **zero GT** contributes `AP = 0.0` to the mean, halving mAP. `eval_real.per_image_ap` guards this and has a test saying so; the two headline paths do not | **MEASURED**; currently inert on the shipped corpora, one corpus change from biting |
| 3 | `recog/model.py:117-124` | `freeze_batchnorm` is one-way. BN affine params are frozen at epoch 0 and **never thawed**, so `frozen_bn_epochs: 8` actually means all 35 | **MEASURED** |
| 4 | `recog/evaluate.py:68` | `np.linspace(0, 1, 11)` is not eleven exact tenths. A recall of exactly 0.3, 0.6 or 0.7 fails its own grid point and loses a whole 1/11 bin | **MEASURED**; worth 0.0016 on today's published number, up to 0.27/class in the worst case |
| 5 | `recog/evaluate.py:122-132` | The matcher skips already-matched GT instead of taking the best-overlapping GT and *then* testing whether it is free, so a **duplicate detection can score TP** | **MEASURED**; inflates AP, zero effect on today's numbers |
| 6 | `recog/evaluate.py:94` | Ties in confidence are broken by insertion order, and the order changes the AP | **MEASURED**; low |
| 7 | `plan/placement_area.py:579` | The `BadDetectorBox` bound rejects `inf` but accepts `1e300`; "there is no 'unbounded' setting" is not enforced | **MEASURED**; not reachable from config |

**Four areas came back clean**, three of them verified by execution: the box
conventions end to end (the item the brief flagged highest-yield), the
augmentation *box geometry*, the VOC/COCO class-index handling, and
`load_detector`'s segmenter interlock. A fifth thing I expected to be a defect —
the score threshold baked into `box_score_thresh` truncating the mAP ranking —
is a real protocol deviation that **measures zero** on the current data. See
"Regions that are clean". Do not spend time there.

---

# 1. Seeding collapses the augmentation group from 8 symmetries to 4

`recog/augmentation.py:104-109` and `recog/seeding.py:383-391`.
**MEASURED.** This is the worst finding and it is one day old.

The module docstring for the dihedral block is unusually explicit about why it
exists:

> a flip or a 90° rotation therefore maps a valid scene to another equally
> valid scene … it is exact, free, and **multiplies the effective dataset by
> eight**.

It multiplies it by four.

`seed_transform` (`seeding.py:383`) calls `Compose.set_random_seed(seed)`. In
albumentations 2.0.8 that propagates **one** seed to every child. Measured on
the as-configured pipeline:

```
ROOT CAUSE - each child's RNG after set_random_seed(4242):
  1 distinct RNG state(s) across 17 child transforms:
    ['Compose', 'RandomBrightnessContrast', 'RandomGamma', 'HueSaturationValue',
     'RandomShadow', 'Compose', 'HorizontalFlip', 'VerticalFlip',
     'RandomRotate90', 'Compose', 'Affine', 'Compose', 'OneOf', 'MotionBlur',
     'Defocus', 'ISONoise', 'GaussNoise']

  first 6 draws from each transform's own RNG:
    HorizontalFlip: [0.8625, 0.4157, 0.0285, 0.3903, 0.321, 0.1849]
    VerticalFlip  : [0.8625, 0.4157, 0.0285, 0.3903, 0.321, 0.1849]
```

`HorizontalFlip` and `VerticalFlip` are structurally identical — one `p` draw
per call — so their identical streams never desynchronise. They fire together or
not at all, forever.

Measured end to end: an asymmetric marker painted inside the box, 3000 draws
through `build_train_transform(configs/recognition.yaml::augmentation)`,
counting distinct output orientations.

```
UNSEEDED (each child entropy-seeded)     : 8 distinct outcomes
   613 / 574 / 554 / 527  (in-plane)  +  185 / 183 / 183 / 181 (transposed)

AFTER seed_transform(t, 20260812)        : 4 distinct outcomes
   1145 / 1105           (in-plane)  +  397 / 353              (transposed)
```

The two missing in-plane outcomes are the horizontal-only and vertical-only
mirrors. What survives is `{identity, 180°}` × `{transpose, no transpose}` — the
cyclic subgroup, not the dihedral group. Half the claimed augmentation is gone,
and the half that is gone is the half that breaks left–right symmetry, which is
the one that matters for a scene of cylinders lying at arbitrary angles.

There is a second-order consequence on the affine block. With the flips'
identical streams driving the pipeline, the `Compose(p=p_geometric)` gate and
the inner `Affine(p=0.5)` gate fall into phase:

```
Affine fire rate among non-transposed samples; documented = p_geometric * 0.5 = 0.250
   flips OFF, seeded       : 0.251     <- as designed
   flips ON,  seeded       : 0.546     <- 2.2x, as trained
```

`configs/recognition.yaml` spends fourteen lines reasoning about the
`scale_limit: 0.20` widening on the premise that "the block fires on
`p_geometric * Affine.p` = 25% of samples". It fires on 55%.

**What would have to be true for it to bite.** A seeded run. `training.seed:
20260812` is committed, `seed_transform` is called unconditionally at
`training.py:339`, and `seg_training.py:372` does the same — the segmenter
pipeline measures **4 mask orientations over 2000 draws** where 8 are possible.
Every detector and segmenter run since the seeding landed has this.

**Would any existing test catch it?** No, and the reason is instructive.
`tests/test_augmentation.py` is a good file — it has a painted-box content check,
a "did RandomRotate90 actually draw k≠0" guard, and a measured photometric-range
assertion. Every one of them is blind to this:

* `test_horizontal_flip_maps_a_known_box_to_the_mirrored_column` and its vertical
  twin force `p=1.0` on **one** op. A correlation between two ops is invisible.
* `test_rotate90_box_still_bounds_the_pixels_it_labelled` and
  `test_rotate90_actually_transposes_a_landscape_frame` build a fresh
  `A.Compose(..., seed=seed)` per draw. Reseeding per sample re-imposes the
  correlation identically each time, so it cannot show up as a difference.
* `test_photometric_range_actually_spans_dark_to_bright` calls
  `t.set_random_seed(seed)` inside the loop, same problem — and photometric is a
  *single* gate, whose marginal is unaffected (measured 0.849 against the
  configured 0.85).
* `test_train_pipeline_carries_all_three_dihedral_ops` asserts only that the
  three ops are *present in the pipeline object*.

The one shape that would catch it — seed once, draw a few thousand samples,
assert the joint distribution of the three dihedral ops — is exactly the shape
nobody wrote, because every other test needed per-draw determinism.

**Consequence.** Silently wrong at runtime, not a wrong published number. The
model is quietly weaker than the config describes, `seed_reproducibility.txt`
still holds (the run *is* reproducible — that is the trap), and a re-run
reproduces the defect perfectly. Note the direction of the irony: `seeding.py`'s
own docstring says "Nothing here is allowed to no-op quietly … a seed that is
set but not used … this project's characteristic defect in its purest form."
The seed *was* used. It was used seventeen times with the same value.

---

# 2. A class with zero ground truth drags mAP to half

`recog/evaluate.py:104-110` returns `ap=0.0` when `total_gt == 0`, and
`mean_ap:160-161` averages that 0.0 in with the real classes.
**MEASURED.**

```
Perfect detector, one battery in the corpus, no cartridge anywhere:
  mean_ap(..., [1, 2]) -> {'AP_1': 1.0, 'AP_2': 0.0, 'mAP@0.50': 0.5000}
  mean_ap(..., [1])    -> {'AP_1': 1.0, 'mAP@0.50': 1.0000}
HAND: the detector found every object that exists. mAP is 1.0.
```

The project already knows this. `eval_real.per_image_ap:255` filters to
`present` classes, with a docstring that names the failure exactly — "folding in
a 0.0 for a class that cannot be found would cap every such image at 0.5 for
reasons that have nothing to do with the detector" — and
`tests/test_dataset.py:423`, `test_per_image_ap_uses_only_the_classes_present`,
asserts it.

The two paths that publish numbers do not do this:

* `eval_real.summarise:209` — `class_ids = [cid for _name, cid in EVAL_CLASSES]`,
  unconditionally both.
* `training.evaluate_model:291` — `class_ids = [1, 2]`, hardcoded. This is the
  number that goes into the epoch log, into `best.pt["metrics"]`, and into the
  hard-subset selection metric.

**What would have to be true for it to bite.** A scored set containing only one
of the two classes. Measured on today's corpora, it does not:

```
recog/realtest          : 60 battery + 20 cartridge GT      -> safe
recog/dataset3d         : 1000 scenes; 122 have no cartridge, 237 no battery
hard-val subset (n=38)  : 310 battery + 71 cartridge boxes  -> safe today
                          (5 of the 38 scenes carry no cartridge)
```

So it is inert — but the margin is thin. `select_hard_subset`'s `min_scenes=8`
floor means a smaller validation split can select as few as 8 scenes, and 12% of
the corpus has no cartridge. A single-SKU corpus makes it certain, and the repo
already ships `configs/synth3d_18650.yaml` and four
`segmentation_cad_control_holdout_*` configs built on exactly that premise. The
symptom would be a headline mAP of "0.44" that is really 0.88, with nothing in
the report saying which.

**Would any existing test catch it?** No — and one test cements it.
`tests/test_evaluate.py:48`, `test_per_class_ap_no_gt`, asserts `r.ap == 0.0`
for a class with no GT. At the `per_class_ap` level that is arguably the right
contract; the defect is that `mean_ap` then averages it in, and there is no test
of `mean_ap` with an absent class. So the suite simultaneously asserts the
behaviour is correct (per-class) and that the opposite policy is correct
(per-image), and never looks at the path in between.

**Consequence.** Wrong number published, conditional on corpus composition.

---

# 3. `freeze_batchnorm` is one-way

`recog/model.py:117-124`, called from `recog/training.py:431-437`.
**MEASURED.**

```
after model.train(), epoch 0        training=True   weight.requires_grad=True
after freeze_batchnorm()            training=False  weight.requires_grad=False
epoch 8: model.train(), no freeze   training=True   weight.requires_grad=False
```

`train_one_epoch` calls `freeze_batchnorm` only while
`epoch < frozen_bn_epochs`. From epoch 8 on it stops calling it, and
`model.train()` does restore `bn.training = True` — so the **running statistics**
resume updating exactly as intended. But `requires_grad` is set to `False` and
nothing anywhere sets it back. Grepping `recog/` for `requires_grad` returns
exactly two lines: the optimiser's filter at `training.py:420` and the
assignment at `model.py:124`.

So the BN affine parameters (γ, β) are pinned at their ImageNet-pretrained
values for all 35 epochs, not the first 8. The module docstring
("BatchNorm frozen for `training.frozen_bn_epochs` (default 20)") and the PPR
claim it mirrors are false for half of what "frozen" means.

One thing that is *not* wrong, and I checked because it would have been worse:
the optimiser is built at `training.py:419` **before** the first freeze, so the
BN params sit in its param list. They simply receive no gradient, and torch's
SGD skips params with `p.grad is None` — so there is no silent weight-decay
drift on frozen parameters. The failure is confined to "the config knob is
half-inert".

**Would any existing test catch it?** No. `tests/test_training.py` covers the
hard-subset selection logic and the structural `last.pt` rule, and nothing else
in `recog/training.py` or `recog/model.py`. There is no test of
`freeze_batchnorm` at all.

**Consequence.** Silently wrong at runtime. Not catastrophic — freezing BN for
a whole fine-tune is a defensible choice someone might make deliberately — but
nobody made it, and the config says otherwise.

---

# 4. `voc_ap`'s 11-point grid is not eleven tenths

`recog/evaluate.py:68`: `for threshold in np.linspace(0.0, 1.0, 11)`.
**MEASURED.**

```
linspace(0,1,11) = [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
                    0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0]

recall 3/10 = 0.3  >=  grid[3] = 0.30000000000000004  ->  False
recall 6/10 = 0.6  >=  grid[6] = 0.6000000000000001   ->  False
recall 7/10 = 0.7  >=  grid[7] = 0.7000000000000001   ->  False
```

Recall values landing exactly on 0.3, 0.6 or 0.7 fail their own grid point and
forfeit a whole 1/11 = 0.0909 bin. Hand-checked cases against an exact-grid
implementation:

```
found  shipped voc_ap   exact grid    hand
 3/10     0.2727          0.3636       4/11 = 0.3636
 4/10     0.4545          0.4545       5/11 = 0.4545   (0.4 is exact, no loss)
 6/10     0.5455          0.6364       7/11 = 0.6364
 7/10     0.6364          0.7273       8/11 = 0.7273
```

**Does it reach the published numbers?** Yes, but barely. I re-ran `last.pt`
over the six scorable real photos, captured the full ranked detection list, and
scored the same curves four ways:

```
                              AP_battery  AP_cartridge      mAP@0.50
shipped (published)             0.9675       0.7587          0.8631
exact 11-point grid             0.9675       0.7619          0.8647
all-point (VOC2010/COCO)        0.9720       0.7669          0.8695

                                                             mAP@0.75
shipped (published)             0.8776       0.6947          0.7862
exact 11-point grid             0.8776       0.6979          0.7878
```

Recall *does* land exactly on 0.3, 0.6 and 0.7 for both classes (GT counts 60
and 20), but the loss is only 0.0016 because precision is flat enough around
those points that `precision[mask].max()` recovers the value from a later
detection. The worst case is a curve that plateaus at exactly one of the three
bad grid points, which forfeits the full 1/11 — and three of them at once
forfeits 0.27.

Worth noting where the margin matters: FDR objective O1 is `mAP@0.5 ≥ 0.90`,
reported as **0.8736** (`docs/receipts/frcnn_map_default.txt`, 15-epoch
default-anchor run). That is 0.026 short of the objective, and one lost bin is
0.091. I could not re-derive that run's PR curve without retraining, so whether
the grid bug is implicated in a "Partial" objective is **READ, not measured** —
but it is the one number in the FDR where 0.0909 changes a verdict, and it is
cheap to check by re-scoring a stored prediction dump.

**Would any existing test catch it?** No. `test_voc_ap_perfect` uses recall
`{0.0, 0.5, 1.0}` — all three exactly representable. `test_voc_ap_all_miss`
returns 0. `test_mean_ap_below_threshold_reduces_score` asserts only
`AP_1 < 1.0`, which is satisfied by any wrong answer below 1. No test in the
file touches recall 0.3, 0.6 or 0.7.

**Consequence.** Wrong number published; small today, unbounded in principle.

---

# 5. Duplicate detections can be scored as true positives

`recog/evaluate.py:122-132`. **MEASURED.**

```python
for j, (gt_box, cls) in enumerate(gts):
    if cls != class_id or not available[j]:   # <-- skips already-matched GT
        continue
    iou = _iou(pred_box, gt_box)
    if iou > best_iou:
        best_iou, best_j = iou, j
```

VOC's rule is: find the GT with the **highest** overlap, then check whether it is
still free; if it is taken, the detection is a duplicate and scores FP. This
code removes taken GT from the search, so a duplicate detection is free to fall
through to its *second*-best GT and score TP there.

Constructed case — two GT boxes overlapping at IoU 0.82 (a plausible pair of
stacked or abutting cells), two detections both essentially on GT-A:

```
  pred0  IoU vs GT_A = 1.0000   vs GT_B = 0.8182    best GT = A
  pred1  IoU vs GT_A = 0.9608   vs GT_B = 0.8519    best GT = A

  code   : AP = 1.0000   (recall [0.5, 1.0], precision [1.0, 1.0])
  HAND   : det0 -> TP(A); det1's best GT is A, which is taken -> FP
           tp = [1, 0], recall = [0.5, 0.5], precision = [1.0, 0.5]
           11-point AP = 6/11 = 0.5455
```

The detector found one object and was credited with two.

**Does it reach the published numbers?** No. I re-scored the real photos with a
VOC-correct matcher on the same detections: `AP_battery` and `AP_cartridge` are
identical to four decimals at both IoU 0.50 and 0.75. Real-photo GT boxes do not
overlap enough for one detection to clear 0.5 IoU against two of them.

**Where it could bite.** The hard validation subset is selected *specifically*
for scenes "where parts occlude each other" (`training.py:59-72`), which is the
one place in this project where overlapping GT boxes are guaranteed. Whether the
overlaps are severe enough is measurable but I did not measure it — that would
need a val-set inference pass.

**Would any existing test catch it?** No. `tests/test_evaluate.py` has no
duplicate-detection case at all — no test where two predictions compete for one
GT, and none where two GT boxes overlap.

**Consequence.** Wrong number published, in the inflating direction. Ranked
below the grid bug only because it measures zero today while the grid bug
measures 0.0016.

---

# 6. Ties in confidence change the answer

`recog/evaluate.py:94`, `dets.sort(key=lambda d: -d[2])` — a stable sort, so
equal scores keep insertion order, which is dict-iteration order over
`preds_by_image` and then list order within an image. **MEASURED:**

```
One GT. Two detections, both score 0.90: one exact, one at IoU 0.33.
  order [exact, poor] -> AP 1.0000
  order [poor, exact] -> AP 0.5000
```

Deterministic in practice — torchvision returns detections already sorted by
score, and dicts preserve insertion order — and exact float ties between two
sigmoid outputs are rare. But `HeuristicDetector` assigns **constant**
confidences (`0.91` for every cartridge, `0.82` for every battery,
`inference.py:227` and `:280`), so every heuristic evaluation is entirely
tie-broken by contour-discovery order. `docs/FDR_v3.md:1574` publishes heuristic
baseline numbers (mAP@0.5 = 0.397) computed through this path.

**Would any existing test catch it?** No; there is no tie case in the suite.

**Consequence.** Low for the learned detector, structural for the heuristic
baseline's published number.

---

# 7. The `BadDetectorBox` bound rejects `inf` but accepts `1e300`

`plan/placement_area.py:569-584`. The brief asked me to confirm the bound cannot
be set so it never fires. **MEASURED:**

```
in-tree default                       : (81.7, 180.0)
(inf, inf)                            : rejected (ValueError)
(0.0, 1.0)      non-positive short    : rejected (ValueError)
(100.0, 10.0)   swapped pair          : rejected (ValueError)
(1e12, 1e12)                          : ACCEPTED
(1e300, 1e300)                        : ACCEPTED
```

The validation catches every *shape* of bad value the docstring names — `None`,
`inf`, a swapped pair — and its own prose says "there is no 'unbounded' setting,
because an unbounded bound is a check that has quietly stopped checking". A
finite absurd value is an unbounded setting. There is no upper sanity bound
(nothing like "no cartridge is wider than 500 mm").

**Strong mitigant, which is why this is ranked last:** `max_cartridge_extent_mm`
is a constructor keyword with **no config surface at all**. Grepping the whole
tree finds it only in `placement_area.py` itself and in one negative test. No
YAML key sets it; no production call site passes it. It is reachable only by
editing Python, which is a different threat model from a mistuned config.

`NaN` is handled correctly by accident — `0.0 < nan` is `False`, so
`not (0.0 < short <= long)` is `True` and it raises.

**Consequence.** Theoretical. Reported for completeness because the brief asked.

---

# Minor observations, not ranked

* **`recog/inference.py:314-318`** — the second-stage crop bounds use `int()`,
  which truncates toward zero rather than flooring the min and ceiling the max.
  MEASURED: `BBox(10.9, 10.9, 49.1, 59.1)` (38.2 × 48.2) yields a 39 × 49 crop
  shifted ~0.9 px up and left. Sub-pixel; it changes the segmenter's framing
  slightly and nothing else. `max(0, ...)` / `min(w, ...)` correctly clamp
  negative and over-wide boxes.
* **`recog/dataset.py:102-103`** — `int(size.findtext("width") or 0)` raises
  `ValueError` on `"1280.0"`, and `parse_voc_xml` propagates `ET.ParseError` from
  a truncated file. One corrupt XML takes a training run down mid-epoch, from
  inside a DataLoader worker. Contrast `training._scene_facts:51`, which is
  explicitly defensive about exactly this for the sidecars. MEASURED, including
  that a box far outside the image (`-50, -50, 500, 500`) is kept **verbatim** —
  neither clamped nor rejected; only non-positive width/height is dropped.
* **`recog/dataset.py:273-282`** — an image with no matching XML is returned with
  empty boxes, i.e. trained as pure background, silently and by design per the
  docstring. MEASURED as inert: 1000 images, 1000 XML, 1000 sidecars, zero
  orphans. It stays a live trap for any hand-assembled corpus.
* **`configs/recognition.yaml::dataset.class_map`** is dead config.
  `training.train:342` constructs `BatteryCartridgeDataset` without
  `class_map=`, so the module-level `CLASS_MAP` is always used. It happens to
  match, so nothing is wrong today.
* **`recog/training.py:277`** — `img_id = batch_idx * 1000 + i` collides for
  batch sizes above 1000. Both validation loaders are `batch_size=1`. Harmless,
  worth a comment.
* **`recog/training.py:353`** — `_split_dataset` is called without a seed, so the
  train/val split is always `seed=0` regardless of `--seed`. This is probably
  deliberate and correct (a fixed split makes seeds comparable), but the seed
  record written into the checkpoint does not distinguish "split seed" from
  "run seed", so nobody reading `best.pt["seeding"]` can tell.
* **Docstring overclaims.** `common/types.BBox:29` says "The convention matches
  Pascal VOC"; `recog/evaluate.py:13-15` says the 11-point protocol makes the
  numbers "directly comparable with the published Faster R-CNN baselines". Real
  Pascal VOC is 1-based with *inclusive* max edges and a `+1` in its IoU
  (`iw = xmax - xmin + 1`). This codebase is 0-based exclusive throughout, which
  is the modern convention and is self-consistent — but it is not VOC's, and the
  IoU therefore differs from the devkit's by a sub-percent amount that grows as
  boxes get smaller. Two docstrings, not two defects.

---

# Regions that are clean

## Box conventions — clean, end to end. MEASURED.

This was the brief's highest-yield item and it is the one place the codebase is
unambiguously right. Everything is **0-based, exclusive-max `xyxy`**, and there
is exactly one conversion in the whole detector half, which is explicit and
tested.

| Site | Convention | Note |
|------|-----------|------|
| `recog/synth3d/annotate.py:82-83` | `x0 = int(xs.min())`, `x1 = int(xs.max()) + 1` | exclusive by construction from the index pass |
| `recog/synth3d/annotate.py:327-330` | written into `<bndbox>` verbatim | no `+1`, no re-basing |
| `dataset.parse_voc_xml:114-120` | read verbatim | correct **because** the generator is not writing true VOC; a VOC-conformant `-1` here would be the bug |
| `dataset.parse_coco_json:201` | `(x, y, x + w, y + h)` | the one conversion; `test_parse_coco_json_converts_xywh_to_xyxy` covers it |
| `common/types.BBox:43-48` | `width = xmax - xmin` | exclusive |
| `evaluate._iou:38-39` | `iw = max(0, x1 - x0)` | exclusive, no `+1` |
| albumentations `format="pascal_voc"` | `[x_min, y_min, x_max, y_max]` px | same |
| torchvision `FasterRCNN` | `xyxy`, 0-based exclusive | same |

Nothing is normalised anywhere in the detector path — no `/W`, no `/H`, no
`cxcywh`. The origin is top-left throughout. The class round-trip
`model id → ClassLabel → CLASS_MAP` was checked and closes: `1 → BATTERY →
CLASS_MAP['battery'] = 1`, `2 → CARTRIDGE → 2`.

I looked specifically for the failure shape the brief described — a convention
*assumed* rather than converted — in `dataset.py`, `augmentation.py`,
`model.py`, `inference.py`, `eval_real.py` and `evaluate.py`, and did not find
one.

## Augmentation box geometry — clean. MEASURED.

Separate from finding #1, which is about the *rates* at which the blocks fire.
The box math itself is right. 600 seeds of the as-configured pipeline with the
photometric and camera blocks suppressed so the probe measures geometry only, on
a 40 × 120 px box (the corpus's actual cell size) in a 1280 × 720 frame:

```
boxes dropped entirely                              : 0
degenerate (width <= 0 or height <= 0)              : 0
outside the image bounds                            : 0
worst px of painted content falling OUTSIDE the box : 0.5
worst px of box slack beyond the painted content    : 0.6
```

Sub-pixel on both sides, which is resampling, not error.
`A.BboxParams` runs with `check_each_transform=True` (the 2.x default), so boxes
are clipped and filtered after **every** op rather than only at the end.
`min_visibility=0.25` behaves correctly at the boundary: a box shifted so that
20 of its 40 px remain is kept and clipped to `[0, 20]`; shifted so that 5 remain,
it is dropped. Widening `shift_limit` to 0.45 — an order of magnitude past the
configured 0.04 — still produced **zero** boxes outside the frame; at 0.60 boxes
begin to be dropped, correctly, rather than surviving in the wrong place.

The validation transform is a true identity: 50 seeds, box out ≡ box in
`(600.0, 300.0, 640.0, 420.0)`, shape preserved.

On the numpy fallback: it applies **no geometric ops at all**, only brightness,
contrast and noise, and returns the boxes unchanged. So the two paths cannot
disagree about box handling — they differ in *coverage*, not in correctness, and
the docstring says so honestly ("it exists to keep the suite runnable, not to be
a second training pipeline"). It does hold a hardcoded `default_rng(0)` that
`seed_transform` overwrites at `seeding.py:395`, which is correct.

One caveat I could not close: my first probe reported a "689 px miss" and "boxes
kept while zero painted pixels remain". Both were artifacts of running the probe
with the photometric block enabled — brightness of +0.55 lifts the black
background above the detection threshold, so the measured content extent became
the whole frame. Re-measured with photometric off, both vanish. Recording this
because it is the kind of false positive that could waste someone's afternoon.

## VOC/COCO class-index handling — clean, and better guarded than most of the tree.

`parse_coco_json:180-185` **raises** `ValueError` when a COCO category id
disagrees with `CLASS_MAP`, rather than remapping — precisely the "a re-export
that renumbers the categories would otherwise silently mislabel the whole set"
failure this project is prone to, caught and made loud. Background (id 0) is
skipped in both readers. Names match case-insensitively so CVAT's `"Battery"`
lines up. Non-positive width/height is dropped in both readers, identically.
Images with no annotations survive with empty lists rather than disappearing.
There is no VOC/COCO off-by-one: both produce `1 = battery`, `2 = cartridge`,
`0` reserved for the detector's background head, which is torchvision's
convention. `tests/test_dataset.py` covers all of this — nine tests on
`parse_coco_json` alone, including the id-mismatch rejection.

## `load_detector`'s segmenter interlock — clean.

`inference.py:361-368` **raises** rather than warns when a segmenter is supplied
but the checkpoint is missing or torch is absent. Given that `plan/planner.py`
carries a blanket `except Exception`, a warning here would have produced exactly
the "clean run of zero placements while the model under test never ran" outcome
that audits D and E found elsewhere. The docstring reasons through that chain
explicitly. This is the shape the rest of the tree should look like.

## The score-threshold truncation — a real deviation that measures zero.

`model.py:113` bakes `model.confidence_threshold` into torchvision's
`box_score_thresh`, so the detector *never emits* a box below 0.70. Both
`training.evaluate_model` and `eval_real` therefore compute AP over a ranked
list truncated at 0.70. That is not the VOC protocol — the reference
implementations run detection at ~0.05 precisely so the PR curve can reach full
recall — and `eval_real.py:545-548` makes it deliberate ("keep the model's own
score threshold in step with ours").

I predicted from reading that this understates the published real-photo mAP, and
constructed a case where it does badly: a detector that finds all 10 objects but
scores 4 of them below 0.70 scores AP 1.0 on the full list and **0.5455**
truncated.

Measured on the actual data, it costs nothing:

```
python -m recog.eval_real --checkpoint recog/checkpoints/last.pt
  battery    60 GT,  73 pred   AP@0.50 0.9675   AP@0.75 0.8776
  cartridge  20 GT,  40 pred   AP@0.50 0.7587   AP@0.75 0.6947
  mAP                          0.8631           0.7862

  ... --confidence 0.05
  battery    60 GT, 111 pred   AP@0.50 0.9675   AP@0.75 0.8776
  cartridge  20 GT,  59 pred   AP@0.50 0.7587   AP@0.75 0.6947
  mAP                          0.8631           0.7862
```

Identical to four decimals despite the detection count rising from 113 to 170.
Every one of the 57 extra detections is a false positive, so recall never
extends and the interpolated precision never changes. The deviation is real,
it should be documented as a deviation, and it is not worth changing on the
evidence available.

(Incidental: `last.pt` scores mAP@0.50 = 0.8631 on the real photos, well above
the 0.6136 in `inference.py`'s min_size comment table. That table is stale
relative to the current checkpoint. Not a defect, but anyone quoting it would
be understating the result by 0.25.)

---

# What I verified by executing, and what disagreed with the code

Five hand-computed answers were checked against the code's answer. Four
disagreed:

| Case | Hand | Code |
|------|------|------|
| Perfect detector, one class absent from the corpus | mAP 1.0 | **0.5000** |
| 6 of 10 objects found at precision 1.0, 11-point AP | 7/11 = 0.6364 | **0.5455** |
| Two detections on one object, two overlapping GT, VOC rule | 6/11 = 0.5455 | **1.0000** |
| Dihedral group after seeding, distinct orientations | 8 | **4** |
| `min_visibility=0.25` at 20/40 px and 5/40 px visible | keep, drop | keep, drop ✓ |

Plus one structural claim checked by execution rather than reading: BN affine
params are `requires_grad=False` at epoch 8 and after, against a docstring
saying frozen for the first 8 only.

And one prediction of mine that the data refuted: the score-threshold truncation
measures exactly zero on the published real-photo numbers.

---

# Suggested order of work

1. **Finding #1.** Seed each child transform with a *distinct* derived seed
   (e.g. `seed + i` over `_flatten(pipeline)`), or drop `HorizontalFlip` +
   `VerticalFlip` + `RandomRotate90` in favour of a single `A.D4()`, which draws
   one symmetry from all eight and cannot desynchronise. Then add the test that
   does not exist: seed once, draw ≥2000 samples, assert all eight orientations
   appear and that the three ops' joint distribution factorises. Re-check
   `seg_training.py` at the same time — it has the identical defect.
2. **Finding #2.** Have `mean_ap` skip classes with `num_gt == 0` and report the
   skip, matching what `per_image_ap` already does; or make the caller pass
   present classes. Either way, report the class count alongside the mAP so an
   absent class is visible in the output rather than folded into the average.
3. **Finding #3.** One line: restore `requires_grad = True` when
   `epoch == frozen_bn_epochs`, or state in the config that BN is frozen for the
   whole run and delete the knob.
4. **Finding #4.** Replace `np.linspace(0, 1, 11)` with
   `[k / 10.0 for k in range(11)]` and subtract a small epsilon in the
   comparison. Then re-score the stored predictions behind
   `docs/receipts/frcnn_map_default.txt` before deciding whether FDR objective
   O1 is still "Partial".
5. **Finding #5.** Restore the VOC rule: take the best-overlapping GT
   unconditionally, then test `available[best_j]`.
