# Audit C — methodology of the cross-distribution generalisation measurement

Read-only adversarial review, 2026-08-12, at HEAD `fa7a4f0`. Nothing was
modified, staged, committed, retrained or regenerated. Every figure below
marked **(measured)** was read off data on disk during this audit; figures
marked **(read)** were read out of code or documents without independent
re-measurement.

**Brief:** try to invalidate the project's headline scientific claim — a
segmenter trained only on procedurally generated cartridge trays, scored on
held-out real measured CAD assemblies. Find the leak.

**Verdict up front: the measurement stands.** The CAD test set is genuinely
disjoint, the leave-one-SKU-out controls genuinely hold out their SKU, and
the eight compared models differ in nothing but their dataset. One real
methodological defect was found — the train/val split partitions crops, not
scenes — and it does not touch the headline number. Three reporting gaps are
listed at the end; each is a caveat to add, not a number to withdraw.

---

## 1. Is the CAD test set genuinely disjoint from every training set?

**Yes, on every check that could have shown otherwise.**

**Asset pools are disjoint by content (measured).** Every annotation in the
COCO sidecars carries an `asset` field. Grouped by dataset:

| dataset | annotations | distinct assets | asset families present |
|---|---:|---:|---|
| `dataset3d_seg_cad_test` | 5156 | 4 | `AnkerPowerCore{10000,13000,20100,26800}` only |
| `dataset3d_seg_anchored` | 2802 | 455 | `anchored_*` only |
| `dataset3d_seg_wide` | 3098 | 453 | `wide_*` only |
| `dataset3d_seg_anchored_18650` | 2820 | 455 | `anchored_*` only |
| `dataset3d_seg_anchored_crown` | 2802 | 455 | `anchored_*` only |
| `dataset3d_seg_cad_control_holdout_*` (×4) | 4738–5957 | 3 each | 3 of the 4 Anker SKUs |

Zero CAD assets appear in any procedural training set; zero procedural
assets appear in `cad_test`. This is checked against the annotations, not
against directory names or config comments.

**No rendered image is shared (measured).** MD5 over all 4536 renders across
the nine datasets: **0 identical images** in any of the 36 pairings,
including `cad_test` against each of the four CAD controls.

**No model trained on any `cad_test` crop (read).**
`configs/segmentation_cad_test.yaml` sets `train_val_split: 0.0`, so
`_split_dataset`'s `n_train = round(0.0 * n) = 0` and all 836 crops are
validation. It is an eval-only config; no `checkpoint_dir` under
`recog/checkpoints/` corresponds to it.

### 1a. What *is* shared, and why it is not a leak — but is a scoping limit

All nine datasets were generated with `seed = 0` (measured, every
`manifest.json`). `scene_generator` at `recog/synth3d/scene.py:117` builds
the per-scene RNG as `random.Random((seed * 1_000_003) ^ (i + 1))`, and
`sample_params` consumes that stream **before** any asset is drawn. The
consequence, measured directly off the render sidecars rather than inferred:

> For **all 500** scene indices, the scene-level parameter dict
> (`n_assemblies`, `backdrop`, `lighting`, `layout_mode`, `exposure`,
> `zoom`, `allow_overlap`) is **byte-identical** between `cad_test` scene *i*
> and the same index in `anchored`, `wide`, `anchored_crown` and
> `cad_control_holdout_AnkerPowerCore10000`. 500/500 in all four cases.

This is **not** a leak. No information about a test label reaches training
through a matched nuisance draw, and pairing the nuisance parameters is if
anything a variance-reduction benefit for the comparison. But it does bound
what the word "cross-distribution" is doing: the model is **not** tested on
unseen lighting rigs, unseen backdrops, unseen exposure, unseen framing, or
unseen obstruction geometry. The shift is confined to the tray.

The project knows this and says so, though for a different purpose —
`NEXT_STEPS.md` and `2026-08-11-sealed-unit-experiment.md` both record that
backdrop, lighting, exposure, zoom and shell preset are "matched to sampling
noise" between the two pipelines, and use it to rule *out* an appearance
explanation for the transfer gap. The same fact is also the honest scope of
the claim, and is not restated that way anywhere.

A second, related scoping fact (read, from `recog/synth3d/config.py:252-263`
and `configs/synth3d.yaml:199-208` against the four measured SKUs in
`2026-08-10-generalisation-design.md` §1.1): **all four CAD SKUs lie strictly
inside the anchored sampling band on every scalar axis** — wall 3.70–4.25 mm
inside 3.3–4.7; case half-height 11.1 mm inside 10.5–11.7; tray floor 1.95 mm
inside 1.6–2.3; bay margin 19.45–30.75 mm inside 17.0–34.0. That is what
"anchored" was *defined* to mean (Decision 2, "within and slightly beyond the
range the four Anker assemblies span"), so it is disclosed by construction in
the design docs. It is worth restating at the point of claim: the
anchored→CAD result is interpolation within the parameter band, not
extrapolation. The extrapolation test is `wide`, and it came out null.

Neither of these invalidates anything. Both are sentences that should sit
next to the headline so an interviewer finds them there rather than finding
them themselves.

---

## 2. Do the leave-one-SKU-out controls actually hold out their SKU?

**Yes. Verified against the annotations, not the config (measured).**

| control set | SKU instances present | instances of the excluded SKU |
|---|---|---:|
| `holdout_AnkerPowerCore10000` | 13000: 1442, 20100: 1981, 26800: 2534 | **0** |
| `holdout_AnkerPowerCore13000` | 10000: 1212, 20100: 1942, 26800: 2521 | **0** |
| `holdout_AnkerPowerCore20100` | 10000: 1179, 13000: 1518, 26800: 2525 | **0** |
| `holdout_AnkerPowerCore26800` | 10000: 1202, 13000: 1505, 20100: 2031 | **0** |

Each set contains exactly three distinct assets and no trace of the fourth.

### 2a. The control receipts publish contaminated rows alongside the honest ones

Each `control_X` model is scored on the **whole** 836-crop `cad_test` set,
and its receipt's per-SKU table prints rows for the three SKUs it *trained
on*. Only the diagonal row is a held-out number; 634 of the 836 crops behind
each control's pooled figure are trained-on SKUs.

The project handles this correctly wherever it matters. `NEXT_STEPS.md`'s
leave-one-SKU-out table uses the diagonal explicitly ("control (held out)"),
and the corrected ceiling is the composite that "scores each SKU with the
control that never saw it" (0.9009 `bay`, 0.9382 `cartridge`).
`2026-08-11-transfer-gap-diagnosis.md:304-305` states the distinction in as
many words: the per-control 0.9387–0.9437 `cartridge` figures "include the
three SKUs each control did train on and are not the same statistic."

Magnitude of the contamination, so it can be dismissed on numbers rather
than on trust: the crop-weighted held-out diagonal for `bay` is ≈0.899
against the contaminated pooled 0.9032–0.9131, i.e. **the contamination
inflates that ceiling by roughly 0.005–0.014 IoU** — negligible against the
0.25 gap the discussion is about. The residual issue is presentational: the
contaminated rows (selected mean 0.7989–0.8091) sit in `NEXT_STEPS.md`'s
headline table above the corrected ones with no inline marker.

---

## 3. Is the train/val split stable and honest?

**The guard is now correct. The split it guards is not scene-disjoint — this
is the one genuine defect this audit found, and it is undocumented.**

### 3a. What the guard now guarantees (read)

`check_split_matches_checkpoint` (`recog/seg_evaluate.py:604-650`) loads the
checkpoint's recorded `val_instance_counts` and compares them to counts
recomputed this run, raising `SystemExit` on any mismatch and never
auto-correcting. It skips the check when the checkpoint's recorded
`coco_path` differs from the eval config's — correct, and necessary: every
cross-dataset generalisation eval in this project would otherwise be blocked
outright.

The "comparing two different quantities" bug is fixed. The call site
(`recog/seg_evaluate.py:995-997`) now passes `out_size=crop_size`, matching
the resolution `seg_training` counted at; `compute_val_instance_counts`'s
docstring documents the one-crop `background` discrepancy that motivated it.
(The fix commit is `488927b`, "fix(recog): compare the split guard's counts
at the training resolution" — the SHA `58dd21d` named in the audit brief does
not exist in this repository.)

**What the guard does not guarantee:** it verifies the partition has not
*drifted* since training. It says nothing about whether the partition is
scene-disjoint, and it would not fire on §3b.

### 3b. The defect: `random_split` partitions crops, not scenes

`_split_dataset` (`recog/seg_training.py:172-180`) calls
`torch.utils.data.random_split` over the flat crop list. `BaySegDataset`
emits one crop per physical *unit*, and a scene contains one to several
units, so crops from the same rendered frame land on both sides of the
split. Measured at `split_seed: 0`, `train_val_split: 0.85`:

| dataset | val crops | from a frame also in train | using a tray asset also in train | val crops with genuinely unseen tray geometry |
|---|---:|---:|---:|---:|
| `anchored` | 127 | **93 (73.2 %)** | **97 (76.4 %)** | 30 |
| `wide` | 124 | **90 (72.6 %)** | **95 (76.6 %)** | 23 |
| `cad_control_holdout_10000` | 128 | **106 (82.8 %)** | 128 (100 %) | 0 |
| `cad_control_holdout_26800` | 128 | **96 (75.0 %)** | 128 (100 %) | 0 |

(The controls' 100 % asset reuse is by design — only three assets exist.)

Severity, measured rather than assumed: the crop *boxes* barely overlap in
pixels. Across `anchored` only **7** val×train crop-box pairs from a shared
frame overlap at all (3 above box-IoU 0.05, max 0.176); `wide` has 25 (17
above 0.05, max 0.188). So this is **not** pixel duplication. What is shared
is the rendered frame's backdrop, lighting realisation, exposure and camera
framing, and — the part that actually matters — the procedural tray sample
itself in 76 % of val crops.

**Which published claims this touches, and how much.**

`docs/NEXT_STEPS.md:372-373`:

| model | own val split | CAD test | Δ |
|---|---:|---:|---:|
| anchored | 0.7161 (127 crops) | 0.6801 (836 crops) | −0.036 |
| wide | 0.6489 (124 crops) | 0.6794 (836 crops) | +0.031 |

The **in-distribution column is optimistically biased** — it is neither
scene-disjoint nor tray-asset-disjoint. Only 30 of 127 (anchored) and 23 of
124 (wide) val crops carry a tray the model never saw. The published Δ is
therefore an **upper bound** on the in-distribution→out-of-distribution drop,
not the drop. Same for the "shape of a genuine out-of-distribution result"
argument at `docs/PORTFOLIO.md:19` ("the crowned model is very slightly worse
on its own validation split"), which rests on the same split.

The same split also selects every `best.pt`. Both procedural receipts already
carry a "checkpoint selection is noise-limited" note (0.0013 and 0.0031
between `best.pt` and `last.pt`), so the practical exposure there is small.

**What it does not touch: the headline.** `cad_test` is a separate render
with a disjoint asset pool at `train_val_split: 0.0`. Every 836-crop figure —
0.6801, 0.6794, 0.7645, 0.6677, the four control rows, and the per-SKU
tables — is unaffected by this defect.

The fix, if it is ever worth making, is to split on `image_id` rather than on
crop index. It would require retraining all eight models, which is out of
proportion to what it would change; the cheaper and honest option is to state
the limitation next to the in-distribution column.

---

## 4. Are the compared models actually comparable?

**Yes. No confound found.**

All eight training configs — `segmentation.yaml`, `_anchored`, `_wide`,
`_anchored_18650`, `_anchored_crown`, and the four
`_cad_control_holdout_<SKU>` — are **identical apart from
`dataset.coco_path`, `dataset.img_dir`, `training.checkpoint_dir` and comment
text** (measured by diff, comments and those three keys stripped). Shared
across all eight: `num_classes 6`, `pretrained: true`, `crop_size 256`,
`half: true`, `epochs 40`, `batch_size 8`, `learning_rate 0.01`,
`momentum 0.9`, `weight_decay 1e-4`, `lr_scheduler cosine`,
`dice_weight 0.5`, `select_on [bay, electronics, obstruction]`,
`train_val_split 0.85`, `split_seed 0`, `jitter_frac 0.06`, and a
byte-identical augmentation block.

The checkpoints agree (measured): `split_seed 0` on all nine,
`coco_path` on each pointing at its own dataset. `best.pt` lands at epoch 30,
31, 33, 33, 34, 37, 37, 39, 39 across the models — that is *selection*
variance inside one common 40-epoch schedule, not a schedule difference.
Dataset scale is matched: 502 scenes for every training set, 500 for the
eval-only `cad_test`.

One near-confound, named so it is not mistaken for one later: the datasets
differ in class composition (`electronics_module` kept 184 in `anchored` vs
119 in `wide`; `battery` 896 vs 1300). That is a *consequence* of the
distribution under test, not an independent variable, so it does not confound
the CAD-test comparison — which scores all models on the same 836 crops. It
does make the two models' own-val numbers non-comparable; see §5.

`anchored_crown` deserves a note in its favour: it shares `seed = 0` and
produces byte-identical asset names and annotation counts to `anchored`
(2802 annotations, 455 distinct assets, both sets), i.e. it is the anchored
render redone with the lid crown as the single changed variable. That is a
well-controlled experiment, and the corresponding claim in
`docs/PORTFOLIO.md:21` correctly refuses to read it as a transfer result
("I picked the crown range *after* measuring the real lids"). That
test-set-informed caveat is present in `PORTFOLIO.md` and `NEXT_STEPS.md`; it
is **absent** from `docs/CV_BULLETS.md:27`, where the 0.6555 → 0.8755 figure
is quoted standalone.

---

## 5. Per-SKU reporting: every figure whose n cannot carry its claim

`format_per_sku_table` (`recog/seg_evaluate.py:582-601`) prints one
`n_crops` column per row — the *union* crop count for that SKU. It is not the
n behind any individual cell in the row, and a reader will take it as such.
Measured per-SKU per-class crop counts on `cad_test`, which the published
tables do not carry:

| SKU | published `n_crops` | bay | electronics | obstruction | battery |
|---|---:|---:|---:|---:|---:|
| AnkerPowerCore10000 | 202 | 47 | 46 | 31 | **14** |
| AnkerPowerCore13000 | 218 | 61 | 61 | 36 | 42 |
| AnkerPowerCore20100 | 214 | 53 | 53 | **27** | 45 |
| AnkerPowerCore26800 | 202 | 52 | 53 | 34 | 44 |
| **pooled** | 836 | 213 | 213 | 128 | 145 |

Against the project's own reportable floor of ~24–36 instances
(`2026-08-10-generalisation-design.md` §9.2):

**Already flagged, correctly and prominently:**
- **`battery` @ AnkerPowerCore10000, n = 14.** Three published cells rest on
  it: anchored 0.3399, crowned 0.5963, control 0.8173. `NEXT_STEPS.md:347`
  and `FDR_v3.md:1827` both carry the ⚠ and state the caveat was registered
  before the numbers were seen. This is handled as well as it can be.
- **wide's in-distribution `electronics`, n = 18.** `NEXT_STEPS.md` names it
  "below the reportable floor" at the exact point it explains the anomalous
  `+0.031`.

**Not flagged, and should be:**
- **Every per-SKU `obstruction` figure**: n = 27 (20100), 31 (10000), 34
  (26800), 36 (13000). Two published claims sit on these. "`20100` is the
  hardest SKU for `obstruction` for *every* model measured (0.488–0.514)"
  rests on **27 crops**. "`obstruction` is at or slightly above the control
  on 3 of 4 SKUs" rests on 31/36/34. Both are reinforced by holding across
  six independently trained models, which is the right defence — but neither
  carries an n, and the row above them prints "202" and "214".
- **wide's own-val split as a whole**: bay 36 / electronics 18 /
  obstruction 19, against anchored's 43 / 35 / 29. The published sentence
  "Wide is meaningfully worse on its *own* validation split (0.6489 vs
  anchored's 0.7161)" compares **two different validation sets**, drawn from
  two different datasets, with different class composition and roughly half
  the `electronics` instances on one side. "Meaningfully worse" is not
  supported by that comparison. The *conclusion* it supports — that wide
  bought nothing — is safe, because it rests on the CAD-test comparison,
  which is like-for-like on 836 identical crops.

The cheapest fix: `evaluate()` already returns `instance_counts` in the
per-SKU result dict. `format_per_sku_table` could print it per class instead
of the single `n_crops`, and every claim above would carry its own n.

---

## 6. Class balance and the pooling method

**Sound, and accurately described in the receipts.**

`evaluate()` accumulates per-class intersection and union over **pixels**
across the whole split (`recog/seg_evaluate.py:480-482`), giving a
micro-averaged per-class IoU. `selected_mean_iou` is then the unweighted
**arithmetic mean of the three `SELECT_ON` class IoUs**, skipping NaN
(lines 485-486). So: macro over classes, micro over pixels.

The receipts describe it exactly that way — "pooled over the validation
split" — and print the per-class instance counts inline
(`instances={'bay': 213, 'electronics': 213, 'obstruction': 128}`). The
class imbalance is real (128 vs 213 of 836 crops) but the macro-mean is
insensitive to it by construction, which is the safer choice; a
crop-weighted mean would have let `bay` and `electronics` swamp
`obstruction`.

The one place imbalance genuinely did mislead has already been caught and
corrected by the project itself, without prompting from this audit: a pooled
per-class IoU accumulates one union over **all 836** crops, while the
instance count printed beside it counts only the crops that *contain* the
class — so false positives on the 623 sealed crops are charged into a
number labelled "instances 213". Both `NEXT_STEPS.md` and `FDR_v3.md` now
carry an explicit "**Do not quote the `bay` column of that table on its
own**" warning plus the present-only decomposition. That is the right fix,
and it is the single most reviewer-proof thing in this measurement.

---

## 7. The oracle comparison

Checked in parallel; summarised here because it bears on the same
"like-with-like" question.

**Like-for-like on code state and inputs — confirmed.** Both arms were
re-run in one harness at commit `83348fa` (752 tests, clean tree, the same
60 frames of `recog/dataset3d_seg`, the same 30 instance IDs at the same
frames). The shipping arm reproduced `2026-08-11-placement-safety.md` §4
per-instance exactly; the validation arm reproduced the old 12/27 and 10/24
under a centre-pixel monkeypatch. The `b9960e4` fix moved only the oracle
number (27→25, 12→11), but the shipping side was genuinely re-measured, so
this is a both-arms run. The mixed-code-state bug is diagnosed at
`2026-08-12-portfolio-verification.md:259-275` and resolved at `:283-290`.

Three residual defects, none of which is a confound:
- **Neither arm has a receipt.** The oracle came from a scratch script
  outside the repository (`2026-08-11-placement-feasibility.md:462`, and
  `:536` states "`docs/receipts/` was not touched"); the shipping 25/12
  traces only to a results table in `2026-08-11-placement-safety.md:238`.
- **The wall-inset asymmetry** (oracle at 0.0 mm, shipping at 4.25 mm) is
  disclosed at `README.md:24,26` and `PORTFOLIO.md:41,45` but **not** at
  `CV_BULLETS.md:30` or `:37` — the two lines most likely to be lifted
  standalone.
- **Stale pre-fix figures survive** at `FDR_v3.md:400-401` and
  `2026-08-11-scale-figures.md:336` ("12 of 30", should be 11), and
  `2026-08-12-portfolio-verification.md:67` still says the two sets share
  ten cartridges where the published docs now say nine.

---

## 8. Provenance gap worth naming

The **corrected** headline figures — present-only `bay` 0.8801 (anchored),
0.8856 (crowned), 0.9013 (control), the composite ceiling 0.9009, "91.4 % of
the gap", and the 136/623 and 16/623 sealed-crop hallucination rates — have
**no receipt in `docs/receipts/` and no committed script**.
`2026-08-11-transfer-gap-diagnosis.md:326-330` discloses this honestly ("a
scratch diagnostic ... not committed because it adds no capability the repo
lacks") and correctly notes that its own anchor, pooled `bay` = 0.6555, *is*
the receipt's.

These are nevertheless the numbers `CV_BULLETS.md:25,27` and
`PORTFOLIO.md:12,19` lead with. `CV_BULLETS.md:53` claims a receipts
discipline where "every published number [is] regenerated by committed
tooling, never hand-edited". That claim is not true of these figures, nor of
the oracle figures in §7 — and they are, between them, the two most quotable
results in the portfolio. Either commit the diagnostic and emit a receipt, or
soften the discipline claim to match what is actually true (which is still
strong: everything scored by `recog.seg_evaluate` does have one).

---

## 9. Verdict

**The generalisation result stands.** Stated plainly, because it is a real
result and manufacturing doubt about it would be worse than useless:

- **No leak.** The CAD test set shares no asset, no rendered image and no
  training exposure with any of the eight trained models. Checked against
  annotation content and image hashes, not against directory names.
- **The holdouts hold out.** Zero instances of the excluded SKU in all four
  control sets, verified against annotations.
- **No confound between compared models.** Eight configs identical apart
  from dataset and checkpoint paths; identical schedule, initialisation
  policy, augmentation, crop pipeline, split seed and dataset scale.
- **The pooling is honest and correctly described**, and the one place class
  imbalance did mislead was found and corrected by the project before this
  audit.

**One real defect, undocumented, not touching the headline:** the train/val
split partitions crops rather than scenes, so 73 % of validation crops come
from a frame that also contributed to training and 76 % reuse a tray the
model saw. This makes the *in-distribution* column of the in-dist/OOD table
optimistic and its published Δ an upper bound. The 836-crop CAD-test figures
are unaffected.

**Reporting gaps to close, in order of how likely an interviewer is to find
them:**
1. Per-SKU `obstruction` claims rest on n = 27–36 with no n printed; the
   `n_crops` column beside them reads 202–218. (§5)
2. The in-distribution val numbers are not scene- or asset-disjoint, and
   "wide is meaningfully worse on its own validation split" compares two
   different val sets with 18 vs 35 `electronics` instances. (§3b, §5)
3. The corrected headline figures and both oracle figures have no committed
   receipt, against an explicit claim that every published number does. (§8)

None of these requires withdrawing a number. All three are sentences to add.

---

## Verification

Read-only throughout. No file in the working tree was modified, created
outside `docs/superpowers/audit/`, staged, or committed by this audit. No
dataset was regenerated and no model was retrained; the only computation run
was rasterisation of existing annotations and MD5 hashing of existing
renders, both against data already on disk.
