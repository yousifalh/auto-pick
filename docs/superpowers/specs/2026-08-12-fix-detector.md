# Fixing audit G — the detector half

**Date** 2026-08-12 · **Against** `docs/superpowers/audit/2026-08-12-G-detector.md`
· **Scope** findings 1–5, plus the two docstring overclaims. Nothing
retrained, no dataset regenerated. The only compute spent was one forward
pass of `best.pt` over the 150-image synthetic validation split, one of
`last.pt` over the seven real photographs, one heuristic pass over the
100-frame April corpus, and a few tens of thousands of augmentation draws.

Every before/after number below was measured on this machine, not carried
over from the audit. Where my measurement disagrees with the audit's
framing, I say so.

---

## Summary

| # | Site | Decision | Published metric moved? |
|---|------|----------|------------------------|
| 1 | `recog/seeding.py::seed_transform` | Derive a distinct seed per child transform from the run seed, and **prove** the streams are distinct | No — runtime defect only |
| 2 | `recog/evaluate.py::mean_ap` | Absent classes are `nan` and excluded from the mean; an all-absent set raises | No (inert on today's corpora) |
| 3 | `recog/model.py::freeze_batchnorm` | **Deleted** `frozen_bn_epochs`; BN frozen, both halves, for the whole run | No |
| 4 | `recog/evaluate.py::voc_ap` | Exact tenths + tolerance (VOC2007 11-point) | **Yes** — real-photo mAP@0.50 0.8631 → **0.8647** |
| 5 | `recog/evaluate.py::per_class_ap` | VOC's rule: best GT first, *then* test availability | No (measures zero everywhere I could score it) |

Suite: **1103 → 1128 passing**, 25 added, 0 failing.

---

# 1. The seeding regression — 4 orientations back to 8

`recog/seeding.py:383` called albumentations' `Compose.set_random_seed`
once. In 2.0.8 that propagates **one** seed to all 17 child transforms, so
`HorizontalFlip` and `VerticalFlip` — structurally identical, one `p` draw
per call — drew identical streams and never desynchronised.

**Fix.** After the root call (which is kept, so the existing "did the seed
take" check still means something), every descendant is re-seeded from
`np.random.SeedSequence(run_seed).generate_state(n)`. That is reproducible
— a pure function of the run seed — and independent, which is precisely
what `SeedSequence` exists to provide. `seg_training.py:372` calls the same
function, so the segmenter was fixed by the same change.

**Measured, one seeding call then 2000–3000 draws of the as-configured
pipeline (`configs/recognition.yaml::augmentation`, seed 20260812):**

| | before | after | unseeded reference |
|---|---|---|---|
| distinct dihedral orientations (detector) | **4** — 1145/1105/397/353 | **8** — 596/568/562/545/186/184/182/177 | 8 |
| distinct mask orientations (segmenter) | **4** | **8** | 8 |
| `P(hflip, vflip)` joint | `{(0,0): 0.494, (0,1): 0, (1,0): 0, (1,1): 0.506}` | `{0.251, 0.246, 0.259, 0.244}` | ~0.25 each |
| `P(affine)` marginal | 0.2505 | 0.2510 | 0.2497 |
| `P(affine \| flipped)` | **0.4951** | 0.2510 | 0.2541 |
| `P(affine \| not flipped)` | **0.0000** | 0.2510 | 0.2456 |
| same seed twice → identical draws | yes | **yes** | n/a |
| different seed → different draws | yes | **yes** | n/a |

The four orientations that came back are the ones that were missing:
horizontal-only and vertical-only mirrors, in both parities. The rate split
0.1875 / 0.0625 across the eight is expected, not residual coupling —
`RandomRotate90`'s own gate fires at 0.5 and then picks *k* uniformly.

**One correction to the brief.** The affine *marginal* was never wrong. It
measured 0.2505 under the defect, against the documented 0.25, so the
config's fourteen lines of `scale_limit` reasoning looked satisfied. What
was wrong was the **conditional**: affine fired on 0.495 of flipped samples
and **0.000** of unflipped ones. No unflipped image in any training run
since 2026-08-12 ever received a camera-pose perturbation. That is a
sharper statement than "the rate is 0.546", and it is the one the new test
asserts. (I could not reproduce 0.546 under any conditioning I tried; the
audit does not define its "non-transposed" subset precisely enough to
reconstruct.)

**Assert loudly.** `seed_transform` now reads every transform's RNG state
back — via `random.Random.getstate()` and `BitGenerator.state`, both pure
reads that do not advance the stream — and raises `SeedingError` unless all
of them are distinct. If a future albumentations changes the per-child
seeding contract again, the run stops instead of quietly halving the
augmentation group. A nested transform with no `set_random_seed` also
raises rather than being skipped.

**The test that did not exist.** Four in `tests/test_augmentation.py`, all
of the shape the audit identified as missing — seed once, draw thousands,
look at the ops *jointly*:

* `test_one_seeding_call_leaves_all_eight_dihedral_orientations_reachable`
* `test_one_seeding_call_leaves_the_two_flips_independent`
* `test_one_seeding_call_leaves_the_affine_gate_free_of_the_flip_gate`
* `test_the_independent_child_seeds_are_still_perfectly_reproducible`

Verified as regression tests by running them against the old
`seed_transform`: the first three **fail** (4 of 8 orientations; joint
`{(0,0): 0.4925, (1,1): 0.5075}`; `P(affine|flipped) = 0.4818` vs
`P(affine|unflipped) = 0.0000`). The fourth **passes** under both — which
is exactly the trap: the defect was perfectly reproducible.

Plus four in `tests/test_seeding.py` covering the mechanism: distinct child
streams, same-seed reproducibility of those streams, and two negative cases
(`_CollidingCompose`, a nested unseedable transform) that pin the loud
raise.

---

# 2. A class with no ground truth

**Policy adopted:** `mean_ap` reads `ClassAPResult.num_gt`, reports an
absent class as `AP_<c> = nan` — "not asked", which is neither a score of
0.0 nor a silent omission that would `KeyError` `eval_real`'s report table
at line 401 — and excludes it from the mean. The returned dict also carries
`classes@<iou>`, so a headline mAP can no longer be quoted without it being
visible how many classes produced it. A set in which *no* requested class
has ground truth **raises `ValueError`**: nothing was evaluated, and
returning a number there would let a training run log it, select a
checkpoint on it and write it into `best.pt`.

Putting the policy in `mean_ap` fixes both publishing paths
(`eval_real.summarise:209`, `training.evaluate_model:291`) without either
of them changing, since both pass a hardcoded class list.

**The two contradictory tests, reconciled.** They were never about the same
function — the gap was that nothing tested the path between them.

* `tests/test_evaluate.py::test_per_class_ap_no_gt` → renamed
  **`test_per_class_ap_reports_no_gt_through_num_gt_not_through_ap`**. It
  still asserts `ap == 0.0`, but now says why that is a *placeholder*
  (`per_class_ap` returns a float and there is no curve), asserts
  `num_gt == 0` as the field that carries the meaning, and additionally
  asserts the PR arrays are empty — so if a real curve ever appears there,
  the 0.0 has stopped being a placeholder and the test fails.
* `tests/test_dataset.py::test_per_image_ap_uses_only_the_classes_present`
  — **unchanged, and now consistent.** It asserted the guarded policy for
  the per-image path; that policy now lives in `mean_ap`, which the
  per-image path calls. It passes unmodified.
* **New:** `test_mean_ap_excludes_a_class_that_has_no_ground_truth` (the
  path in between: perfect detector, one class absent → mAP 1.0 not 0.5,
  `AP_2` is `nan`, `classes@0.50 == 1.0`, and the result equals asking for
  the present class alone) and
  `test_mean_ap_refuses_a_set_with_no_ground_truth_at_all`.

Inert on today's corpora, as the audit measured. Nothing published moves.

---

# 3. `freeze_batchnorm` — I deleted the key

**Decision: delete `training.frozen_bn_epochs`. BatchNorm is now frozen —
running statistics *and* affine parameters — for the whole run.**

The audit's framing is that the key does nothing. That is half right, and
the half it gets wrong is what decided this. The key *was* live for the
running statistics: `model.train()` restored `bn.training = True` at epoch
8 exactly as intended. Only `requires_grad` was one-way. So the actual
trained behaviour was **incoherent rather than merely stale**: from epoch 8
the running statistics migrated toward the render domain while γ and β,
calibrated against ImageNet's statistics, were pinned and could not follow.

That means there was no zero-behaviour-change option. Both repairs alter
training:

* *Honour the key* — thaw γ/β at epoch 8. Moves **away** from the standard
  detection recipe, and I cannot evaluate it without retraining.
* *Delete the key* — freeze both halves throughout. This is torchvision's
  own detection recipe (its reference backbones ship as
  `FrozenBatchNorm2d`), and it is the right one at `batch_size: 2`, where
  batch statistics estimated from two images are noise. It is also the
  coherent version of what the code was already 90% doing.

I chose delete, per the brief's stated preference and because it is the
smaller and better-supported change. `freeze_batchnorm` now returns the
**count** of modules frozen — not `None` — and `train_one_epoch` raises if
that count is zero, because a backbone swapped for GroupNorm would
otherwise make the call a silent no-op. It is called unconditionally after
every `model.train()`, since `train()` un-freezes BN.

**Behaviour delta to report:** runs after this commit differ from the
published checkpoints. `best.pt` / `last.pt` were trained with running
statistics updating from epoch 8; a re-run now keeps them frozen for all 35.
No published metric changes — the existing checkpoints are untouched — but
the recipe is no longer bit-identical to the one that produced them.

Removed from `configs/recognition.yaml` (with a comment recording why) and
from `recog/training.py`. `docs/FDR_v3.md:3422` still documents
`.frozen_bn_epochs (int, 20)` — **not corrected here; that file is owned by
another agent mid-pass.** `tests/test_training.py` gained four tests
including `test_no_recognition_config_still_carries_the_removed_bn_knob`,
which fails if the key reappears in any config or is read anywhere in
`training.py`.

---

# 4. The recall grid — VOC2007 eleven exact tenths

**Convention implemented: the VOC2007 11-point interpolated AP of
Everingham *et al.* (2010)** — AP is the mean, over the eleven recall
levels {0.0, 0.1, …, 1.0}, of the maximum precision at any recall at or
above that level. That is the convention `recog/evaluate.py`'s module
docstring already claimed. It is *not* the all-point (VOC2010/COCO) form,
and the fix does not change which convention is used, only whether the
implementation matches it.

`np.linspace(0.0, 1.0, 11)` returns 0.30000000000000004,
0.6000000000000001 and 0.7000000000000001. Replaced with
`tuple(k / 10.0 for k in range(11))` — correctly rounded, so it returns the
same double as the recall ratio whenever the denominator divides cleanly —
plus a `1e-9` tolerance in the comparison for recalls that arrive through
`cumsum` or plateau a few ULP short.

**Published metric that moved.** `python -m recog.eval_real --checkpoint
recog/checkpoints/last.pt`, six scorable real photographs:

| | before | after |
|---|---|---|
| `AP_battery@0.50` | 0.9675 | 0.9675 |
| `AP_cartridge@0.50` | 0.7587 | **0.7619** |
| **`mAP@0.50`** | **0.8631** | **0.8647** |
| `AP_cartridge@0.75` | 0.6947 | **0.6979** |
| **`mAP@0.75`** | **0.7862** | **0.7878** |

+0.0016 on both, matching the audit's predicted exact-grid values to four
decimals. These figures appear in no shipped document — I grepped
`docs/`, `README.md` and `docs/receipts/` and found them only in audit G
itself — so nothing needs rewriting. Any future `eval_real` run reports
0.8647.

**FDR objective O1 — measured, and the grid bug is not implicated.** The
brief flagged this as possibly Pass/Fail-deciding, O1 sitting 0.026 short
of `mAP@0.5 ≥ 0.90` at a published 0.8736 while one bin is worth 0.091. I
re-scored one stored detection list over the 150-image synthetic validation
split under all three conventions:

```
val split: 150 images of 1000    GT: battery=963  cartridge=242
IoU 0.50   shipped / grid fix / both fixes :  mAP = 0.9998 in all three
IoU 0.75   shipped / grid fix / both fixes :  mAP = 0.9081 in all three
```

Identical to four decimals. The mechanism explains it and generalises past
the checkpoint I happened to score: the grid bug can only bite when recall
lands *exactly* on 0.3, 0.6 or 0.7, which needs a GT count divisible by 10.
The real photos have 60 and 20 (so it bit); this validation split has 963
and 242 (so it cannot). The O1 receipt was produced on this same split, so
its 0.8736 does not move and **O1's verdict is unaffected by the grid bug**.

*Caveat, stated because it is a real one:* I scored `best.pt` (custom
anchors), not the `default_anchors_best.pt` run behind
`docs/receipts/frcnn_map_default.txt`. The argument above rests on the GT
counts, which are a property of the split rather than of the model, so it
holds for either checkpoint — but I did not re-run that specific ablation.

---

# 5. Duplicate detections

`per_class_ap` matched against only the *available* ground truth, so a
duplicate detection could fall through to its second-best box and score a
true positive. Restored to VOC's order: take the best-overlapping GT
unconditionally, *then* test `available[best_j]`.

Pinned with the audit's constructed case,
`test_a_second_detection_on_the_same_object_is_a_false_positive`: two GT
boxes at IoU 0.8182, two detections both closest to GT-A (IoU 1.0000/0.8182
and 0.9608/0.8519). The test asserts the four IoUs, the premise that A is
pred1's best, and then `tp = [1, 0]`, `precision = [1.0, 0.5]`,
`AP = 6/11 = 0.5455` — where the old matcher gave 1.0000. A companion test,
`test_two_detections_on_two_distinct_objects_both_still_score`, guards
against over-correcting real hits into false positives.

**Measures zero on every corpus I could score:** real photos (identical to
four decimals), the 150-image synthetic validation split (above), and the
100-frame heuristic baseline (below). Fixed anyway.

Incidental: `scripts/detector_bench.py::_match_frame` already implemented
the VOC rule correctly, and its receipt claims to be "identical to
`recog.evaluate.per_class_ap`". That claim was false and is now true.

---

# The docstring overclaims

Wording only; no behaviour touched. Box conventions came back clean end to
end in the audit and were left alone.

* `common/types.BBox` — "The convention matches Pascal VOC" now says the
  order and units match VOC's `<bndbox>` but the *indexing* does not: real
  VOC is 1-based with inclusive max edges and a `+1` in its IoU, whereas
  this codebase is 0-based exclusive throughout (torchvision,
  albumentations' `pascal_voc`, COCO after conversion), so the IoU differs
  from the devkit's by a sub-percent amount that grows as boxes shrink.
* `recog/evaluate.py` module docstring — "directly comparable with the
  published Faster R-CNN baselines" was wrong twice over (most publish
  all-point AP, and the box convention is not the devkit's). It now names
  the VOC2007 form explicitly, states the measured eleven-point vs
  all-point gap on this repository's real photos (0.8647 vs 0.8695), and
  states the box convention.

---

# What I did not touch

* `plan/scene.py`, `plan/planner.py`, `main.py` — owned by another agent.
* `docs/FDR_v3.md`, `docs/NEXT_STEPS.md`, `README.md`, `docs/PORTFOLIO.md`,
  `docs/CV_BULLETS.md` — owned by another agent, mid-pass. **`FDR_v3.md:3422`
  documents the now-removed `frozen_bn_epochs`, and no shipped doc carries
  a real-photo mAP, so the only doc edit outstanding is that one line.**
* Audit findings 6 (confidence ties) and 7 (`BadDetectorBox` bound) — out
  of scope for this brief.
* Box conventions, per the brief. Not improved, only re-described.
* `docs/receipts/*` — not regenerated. `frcnn_map_default.txt`,
  `detector_bench.txt` and `frcnn_map.txt` were all verified unchanged or
  re-derived above rather than rewritten.

# Files changed

```
recog/seeding.py          per-child derived seeds + independence proof
recog/evaluate.py         recall grid, VOC matcher, zero-GT policy, docstring
recog/model.py            freeze_batchnorm: whole-run, returns a count
recog/training.py         frozen_bn_epochs removed; unconditional refreeze
common/types.py           BBox docstring wording
configs/recognition.yaml  frozen_bn_epochs removed, with the reason
tests/test_augmentation.py  +4  joint-distribution tests
tests/test_seeding.py       +4  per-child seeding mechanism
tests/test_evaluate.py     +12  recall grid, duplicates, zero-GT; 1 reconciled
tests/test_training.py      +4  frozen BN, and that the knob stays gone
```
