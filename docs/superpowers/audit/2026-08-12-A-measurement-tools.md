# Audit A — the measurement tools behind `docs/receipts/`

**Date:** 2026-08-12 · **Tree:** `d:\dev\auto-pick` @ `fa7a4f0` (`feat/blender-synth-dataset`)
**Scope:** `recog/seg_evaluate.py`, `seg_ablation.py`, `eval_real.py`, `seg_training.py`,
`calibrate_tau.py`, `model.py`, `generate3d.py`, `convert_cad.py`, `verify3d.py`
**Mode:** read-only. Nothing was modified, staged, committed, retrained or regenerated.
Every command below wrote only into the session scratchpad.

Throughout, **[EXEC]** marks a claim established by running code and reading its output;
**[READ]** marks a claim inferred from reading the source. The distinction is load-bearing
and is not blurred anywhere in this document.

---

## 0. One correction to the brief before anything else

The brief lists nine tools as having "no test file at all". That is true of the *filenames*
and misleading about the *coverage*. Tests for these modules exist, filed under the name of
the thing being measured rather than the tool doing the measuring: **[EXEC]**

| tool | tested where | what is covered |
|---|---|---|
| `seg_evaluate.py` | `tests/test_bay_segmenter.py`, `tests/test_calibration.py` | `per_class_iou`, `boundary_displacement_mm`, `signed_area_error_mm2`, `compute_val_instance_counts`, `check_split_matches_checkpoint` (×5), `latency_within_budget`, `_sibling_checkpoint_note`, `group_indices_by_asset`, `format_per_sku_table`, `resolve_frame_scales`, `evaluate` (scale behaviour only) |
| `seg_ablation.py` | `tests/test_arbitration.py` | `delta_cells` sign convention (×3) — and **nothing else** |
| `eval_real.py` | `tests/test_dataset.py` | `partition_records`, `per_image_ap`, `build_image_rows`, `format_report`, `summarise` |
| `calibrate_tau.py` | `tests/test_arbitration.py`, `tests/test_calibration.py` | `calibrate` (×2), `collect_records` (×2) |
| `seg_training.py` | `tests/test_bay_segmenter.py` | `dice_loss`, `checkpoint_state_dict`, `drop_last_batch` |
| `convert_cad.py` | `tests/test_synth3d.py` (`from recog import convert_cad as CC`) | STEP unit parsing and catalog merge |
| `model.py` | indirectly via `test_training.py` / `test_inference.py` | factory construction |
| `generate3d.py` | indirectly (`test_synth3d.py` resume/PNG validity) | — |
| `verify3d.py` | **nothing** | — |

The real, uniform hole is not "these modules are untested". It is that **every `main()` in
this set is untested, and so is every function that only `main()` calls.** That is precisely
the shape of the two `TypeError` regressions the brief cites, and — as §2 below shows — of a
third defect of the same kind that is still live at `fa7a4f0`.

---

## 1. FINDING (critical) — `delta_cells` is measured at a scale that describes no frame, and the safety-direction count is understated by 2.5×

### 1.1 What the code does

`recog/seg_ablation.py:599`:

```python
mm_per_px = resolve_mm_per_px(synth_cfg)      # the NOMINAL framing, 0.6250
```

That single constant is handed to `evaluate_delta_cells` → `delta_cells` → `_pack_count`,
where it is not a reporting unit but an *input to the measurement itself*
(`recog/seg_ablation.py:89-126`):

```python
inset_px  = max(0, int(round(wall_inset_mm / mm_per_px)))   # erosion radius
strip_w   = (x1 - x0) * mm_per_px                           # strip size in mm
strip_h   = (y1 - y0) * mm_per_px
grid      = _rasterise_mask(safe, (x0, y0, x1, y1), mm_per_cell, mm_per_px)
```

The packer then fits fixed-millimetre 18650 cells (`CELL_W_MM × CELL_H_MM`) into a strip
whose millimetre size was computed from `mm_per_px`. Commit `502ef00` removed exactly this
constant from `seg_evaluate` and `calibrate_tau` — its own message says "`380e7d5` fixed the
planner and left these two" — and `seg_ablation` was in neither pass. **[READ]**

The corpus does not have one scale. `docs/receipts/seg_eval.txt`, generated from the *same*
dataset and the *same* 126-crop split, reports the true per-frame GSD as **median 0.8211,
range 0.4903–1.0739 mm/px**, and prints a paragraph saying the nominal 0.6250 "describes NO
frame in this corpus". `docs/receipts/seg_ablation.txt`, sitting next to it, prints
`mm_per_px=0.6250` with no such note. **[READ]**

### 1.2 The distortion, measured on the real validation split

Ground-truth label maps only — no segmenter involved, so this isolates the scale: **[EXEC]**

```
Cells the GROUND TRUTH label map admits, over all 126 val crops:
  at the NOMINAL 0.6250 mm/px (what seg_ablation uses):
      total =  4   mean = 0.032   max = 2   crops packing 0 cells = 124/126
  at each frame's OWN GSD (what seg_evaluate/calibrate_tau use):
      total = 17   mean = 0.135   max = 4   crops packing 0 cells = 118/126
  ratio of total capacity measured: 4.25x
```

At the scale `seg_ablation` uses, the entire validation split has a total packing capacity of
**four cells**. The metric has almost no dynamic range: 124 of 126 crops cannot fit a single
cell in the ground truth, so "the prediction and the truth agree" is nearly forced by
construction rather than earned by the segmenter.

### 1.3 The shipped figure, recomputed at both scales

Same split, same checkpoint (`recog/checkpoints/seg/best.pt`), the actual segmenter in the
loop, `delta_cells` computed twice per crop — once at 0.6250, once at that crop's own GSD:
**[EXEC]**

```
NOMINAL 0.6250 (what the shipped receipt reports):
  mean 0.000  median 0.000  range [-2, 2]
  positive (cells lost) 2/126   negative (packed where forbidden) 2/126   zero 122/126

per-frame true GSD:
  mean 0.056  median 0.000  range [-2, 4]
  positive (cells lost) 6/126   negative (packed where forbidden) 5/126   zero 115/126

crops where the two scales disagree: 9/126
sign flips (one says safe/zero, the other says damage-risk): 3
```

The nominal column reproduces `docs/receipts/seg_ablation.txt` exactly on the counts
(2 / 2 / 122, range [−2, 2]); the mean differs in the third decimal (0.000 vs the receipt's
0.008) because this run was CPU/fp32 against the receipt's CUDA/fp16. The comparison that
matters — nominal vs true, within one run — is like-for-like.

### 1.4 Why this is not the caveat already on record

`docs/superpowers/specs/2026-08-11-scale-calibration.md` §5 already flags the nominal scale,
and says:

> *Both sides are at the same wrong scale so the sign is trustworthy, but the magnitude is
> compressed.*

**That claim is false, and the measurement above is what falsifies it.** Cell packing is a
discrete, non-monotone function of scale — it is not a multiplicative rescaling that cancels
in a difference. Three crops flip sign, and the count in the *damage* direction goes from
2/126 to **5/126**. A constructed minimal case shows the same thing in isolation: **[EXEC]**

```
 mm_per_px  pack(gt)  pack(pred)   delta
    0.6250         2           2       0   <- NOMINAL
    0.4903         2           2       0   <- true range end
    0.8211         9          10      -1   <- true median GSD
    1.0739        19          19       0   <- true range end
```

So the caveat on record understates the problem in exactly the direction the project cares
about most. It is filed as a magnitude caveat; it is a sign caveat.

### 1.5 Published figures affected

Every one of these quotes the negative-direction count or the mean, and every one inherits
the wrong scale:

- `docs/receipts/seg_ablation.txt` — the whole Δcells block.
- `docs/FDR_v3.md:2200` §13.2.1 — "**+0.008 cells** over the same **126** validation crops:
  **122 of 126 exact**, 2 losing a cell to conservatism, and — *the figure that matters for
  safety* — **2 of 126 in the negative direction** (range [−2, +2])". Measured at the frames'
  own scales this is **5 of 126, range [−2, +4]**.
- `docs/FDR_v3.md:2636-2639` — the "Δcells got *worse* on the metric that matters most"
  regression argument, which turns on the size of that negative count.
- `docs/NEXT_STEPS.md:495`, `:504`, `:564` — item 3, including "the +0.008 mean and the
  2/126 negative-direction count above are both figures from the FFDH-only planner".

The direction of the error is the unsafe one: the receipt reports **fewer** damage-direction
events than a correctly-scaled measurement of the same checkpoint on the same crops.

---

## 2. FINDING (high) — `seg_ablation`'s split guard is a false alarm on every non-default config

`recog/seg_evaluate.py:527` `compute_val_instance_counts` is explicit in its own docstring:

> *`out_size` MUST be the `model.crop_size` the checkpoint was trained at whenever this is fed
> to `check_split_matches_checkpoint`.* … *counting here at the crop's native resolution
> compares two different quantities and the guard fires on a dataset that never changed.*

Three callers. Two obey it: **[READ]**

- `recog/seg_evaluate.py:995` — `out_size=crop_size` ✅
- `recog/calibrate_tau.py:525` — `out_size=crop_size` ✅ (and its comment at `:516-524`
  names this exact defect as one of the two it was fixing)
- `recog/seg_ablation.py:615` — **no `out_size` at all**, so it defaults to `None` = native ❌

`seg_ablation` received only half of the fix `calibrate_tau` received. Executed against the
anchored config: **[EXEC]**

```
$ python -m recog.seg_ablation --config configs/segmentation_anchored.yaml \
      --checkpoint recog/checkpoints/seg_anchored/best.pt --device cpu --out <scratch>
error: the recomputed validation split does not match the checkpoint's own record -
scoring would silently include crops this checkpoint trained on.
  checkpoint ... recorded: {'background': 123, 'cartridge': 127, 'bay': 43, ...}
  this run recomputes:     {'background': 124, 'cartridge': 127, 'bay': 43, ...}
rc=1
```

The dataset did not move. Counting the same crops both ways confirms it: **[EXEC]**

```
configs/segmentation_anchored.yaml | recog/checkpoints/seg_anchored/best.pt
  checkpoint recorded  : {'background': 123, ...}
  seg_evaluate  (256)  : {'background': 123, ...}   -> guard passes: True
  seg_ablation (native): {'background': 124, ...}   -> guard passes: False
```

One crop's sliver of background survives at native resolution and is lost to the nearest-
neighbour downsample at 256 — the exact single-crop discrepancy `compute_val_instance_counts`'
docstring predicts. The failure is loud, not silent, which is the good direction; but it means
`docs/receipts/seg_ablation.txt` is **regenerable only for `configs/segmentation.yaml`**,
where native and 256 happen to agree (126/126/36/36/24/24 both ways **[EXEC]**). No ablation
receipt can be produced for the anchored, wide, CAD-test or the four CAD-control-holdout
configs without this line being fixed.

Related, one line lower in severity: `check_split_matches_checkpoint` skips the guard entirely
when `saved_coco_path != coco_path`, comparing paths by **string equality**. A config that
spells the same dataset differently (separator, `./` prefix, absolute vs relative) silently
disables the drift check rather than failing. Also note `recog/checkpoints/seg/best.pt` records
`coco_path: None` **[EXEC]**, which falls through to the count comparison — the safe direction,
by luck of the `is not None` test rather than by design.

---

## 3. Limitation (medium) — boundary displacement is one-way, and the verdict line does not say so

`boundary_displacement_mm` computes the mean distance from each **predicted** boundary pixel to
the nearest ground-truth boundary pixel. It is *not* symmetric — not a Hausdorff, not a mean
symmetric surface distance. The docstring says exactly this, and so does the receipt header
("mean distance from a predicted boundary pixel to the nearest ground-truth boundary pixel"),
so **the text matches the implementation**; this is not a mislabelled metric. **[READ]**

The consequence is still worth stating, because it is invisible in the receipt's verdict.
Constructed case: an 8×8 truth block (32 boundary px) against a prediction consisting of a
**single pixel** sitting on that block's corner: **[EXEC]**

```
[BD] 1-px pred on an 8x8 truth block          -> 0.0000 mm
[BD] arguments swapped                        -> 5.7715 mm   (asymmetric: 0.0000 vs 5.7715)
```

A prediction that finds 1/32 of the boundary scores a **perfect** displacement. The receipt
turns this number into an architecture verdict —

> `verdict: BELOW the mask-head quantisation figure - supports the architecture choice.`

— and that verdict is, on its own, satisfiable by an under-segmenting model. The
`signed_area_error_mm2` conservative column is what would catch such a model, and it is
printed in the same receipt, so the report as a whole is not misleading. But the boundary
block carries no caveat that its metric is one-way, and it is the block the architecture
argument rests on.

Two smaller notes on the same block, both **[READ]**:

- `head_mm = px_lo * mean(scale over the crops that contributed)` while
  `bd = mean(bd_px_i × scale_i)`. The ratio is invariant to a *global* rescaling of the corpus
  (both sides scale by the same factor), which is what the receipt's scale-invariance claim
  needs, and it is correct. It is **not** invariant to correlation between `bd_px` and `scale`
  within the split. On the shipped receipt the three per-class mean scales are 0.8217 / 0.8217
  / 0.8220 against a split range of 0.4903–1.0739, so the effect is small here, but it is an
  assumption, not an identity.
- `px_lo = 131/28 = 4.678` is the **short** axis of the mask-head figure. Using the short axis
  is the conservative choice (a tighter threshold); `px_hi = 10.29` is computed and printed in
  the prose but never compared against. Deliberate and stated — noted only so it is not
  rediscovered as a bug.

---

## 4. What checks out — verified by hand-computation, not by reading

Every metric below was checked by constructing a case with a known answer and comparing the
tool's output to a number computed by hand. **[EXEC]** throughout.

**Per-class IoU — correct, union denominator.** 4×4 map, truth class 2 = rows 0–1 (8 px),
prediction class 2 = rows 1–2 (8 px). Intersection = row 1 = 4 px; union = rows 0–2 = 12 px.
Expected 4/12 = 0.333333; got **0.333333**. A *sum* denominator would have given 4/16 = 0.25.
`_class_confusion` uses `(p | t).sum()` and is the single implementation shared by
`per_class_iou` and the `evaluate` accumulator, so the tested path and the published path are
the same code.

**Empty-prediction and empty-truth handling — correct and asymmetry-free.**
- prediction empty, truth non-empty → IoU **0.0** (union > 0, correctly penalised)
- prediction non-empty, truth empty → IoU **0.0**
- class absent from both → **NaN**, and NaN classes are dropped from the mean rather than
  scored 0. This matches the docstring and the receipt's own header text
  ("NaN = class absent from both pred and truth over the whole split, not scored as 0").

**Pooled IoU — correct, and correctly described.** Two crops: crop A contributes
intersection 50 / union 150, crop B contributes 10 / 10. `evaluate` returned
**0.375000** = (50+10)/(150+10), i.e. pooled pixel-wise across the split. A per-crop mean of
per-crop IoUs would have been 0.666667. The receipt says "pooled over the validation split"
and the docstring says "sum of intersections / sum of unions across crops" — both accurate.
`recog/seg_training.py:208 _update_confusion` / `:218 _per_class_iou` accumulate identically
(`None` where `seg_evaluate` uses `NaN`), so the docstring's claim that the two are directly
comparable holds. **[READ]**

**`selected_mean_iou` — an unweighted mean over present classes, and the receipt is honest
about it.** `sum(present)/len(present)` over `SELECT_ON` = (`bay`, `electronics`,
`obstruction`), each term itself pixel-pooled within its class. So it is pixel-weighted
*within* a class and equal-weighted *across* classes. The receipt does not use the word
"unweighted", but it prints the per-class instance counts on the same line
(`instances={'bay': 36, 'electronics': 36, 'obstruction': 24}`), which is enough for a reader
to reconstruct it. No misdescription found.

**Boundary displacement magnitude — correct.** Truth = a 1-px band at row 2, prediction = a
1-px band at row 5, both full width. Every predicted boundary pixel is 3 px from the nearest
truth boundary pixel. At `mm_per_px=2.0`, expected 6.0 mm; got **6.000000**. The scipy EDT
path and the pure-numpy fallback both compute distance to `~tb`, which is 0 on truth-boundary
pixels — correct.

**Signed area error — correct, and the sign convention matches the docstring.** Same 4×4 case
at `mm_per_px=2.0` (4 mm²/px): optimistic = `pred & ~truth` = 4 px = 16.0 mm²; conservative =
`~pred & truth` = 4 px = 16.0 mm². Both matched exactly. "Optimistic = predicted placeable
where truth is not" is what the code computes.

**Division by zero — none found unguarded.** Systematically checked across all five reporting
paths **[READ]**: `per_class_iou` (`union == 0` → NaN), `evaluate`'s pooled IoU (`union[c] > 0`),
`selected_mean_iou` (`if present`), `_mean_or_nan` (`if xs`), `scale_stats` (`vals.size == 0`),
`format_report`'s `clears by` (`bd <= 0` and NaN guarded), the nominal-ratio note
(`if nominal_mm_per_px`), the batching factor (`total_ms > 0`), `latency_table`'s
`total_ms / n` (`n ≥ 1` by construction), `seg_ablation`'s two `frac = … if area else 0.0`,
its `bbox_w_px = max(1.0, …)`, `calibrate_tau`'s `fail_rate` (`if n_accepted`) and
`rejected_fraction` (`if n`), and `eval_real`'s `max(1, …)` denominators. `resolve_frame_scales`
additionally rejects a non-positive fallback with `UnknownScale` rather than dividing by it.

**The scale-provenance machinery does what it says.** On the real split,
`resolve_frame_scales` returned `{'n_measured': 126, 'n_fallback': 0, 'n_frames': 112,
'fallback': None}` **[EXEC]** — 126 crops calibrated from 112 distinct frames' own sidecars,
nothing falling back. `evaluate()` and `collect_records()` both reject a scalar `scales` with
`TypeError`, so the old single-constant regression cannot return silently. **[READ]**

---

## 5. Do the CLIs run?

`--help` on all nine: **[EXEC]**

| tool | `--help` | note |
|---|---|---|
| `seg_evaluate` | OK | |
| `seg_ablation` | OK | but see §2 — dies at the guard on non-default configs |
| `eval_real` | OK | |
| `seg_training` | OK | |
| `calibrate_tau` | OK | the `138105d` `TypeError` is genuinely fixed at `502ef00` |
| `model` | OK | not a CLI; imports cleanly |
| `generate3d` | **FAIL** | `ModuleNotFoundError: No module named 'bpy'` at `generate3d.py:27`, module-scope. By design (it runs inside Blender), but it means the module cannot be imported — let alone unit-tested — outside Blender. |
| `convert_cad` | OK | |
| `verify3d` | OK | |

Real runs, non-destructive, all output redirected to scratch: **[EXEC]**

- `seg_ablation --config configs/segmentation_anchored.yaml` → **rc=1**, false split-drift
  error (§2). Reached the guard, never reached a measurement.
- `seg_evaluate`'s and `seg_ablation`'s measurement internals were driven directly against the
  real dataset and the shipped checkpoint (§1.2, §1.3, §2) rather than through `main()`, to
  avoid any chance of touching `docs/receipts/`. No receipt was regenerated or written.

Both CLIs the brief names as broken since `138105d` — `seg_ablation` and `calibrate_tau` —
now parse and reach their measurements on the default config. `seg_ablation` carries the
residual half-fix in §2.

---

## 6. Risk ranking (LOC × how load-bearing the output is)

| # | tool | LOC | why it ranks here |
|---|---|---|---|
| 1 | **`seg_ablation.py`** | 667 | Produces the *end-to-end* number — the one the project says matters more than IoU — and the only real-photo comparison against the measured 0.218 baseline. Two live defects (§1, §2). Its entire real-photo arm (`heuristic_vs_segmenter`), its packer bridge (`_pack_count`), and `evaluate_delta_cells` have **zero** tests; only `delta_cells`' three sign tests exist. Highest LOC-to-coverage ratio in the set. |
| 2 | **`seg_evaluate.py`** | 1080 | Largest, and every segmentation claim in FDR §13.2.1 and the eleven `seg_eval_*.txt` receipts flows through it. Ranked second only because its metrics are the best-covered in the set and **all of them verified correct by hand** (§4). The untested surface is `main()`'s wiring and `format_report`. |
| 3 | **`calibrate_tau.py`** | 593 | Emits τ, which gated production placement. Recently repaired and now the *reference* implementation for the split guard. `calibrate`/`collect_records` tested; `format_report` and `main` are not. |
| 4 | **`eval_real.py`** | 631 | The honest sim-to-real test (`frcnn_map.txt`). Better covered than its LOC suggests — `partition_records`, `per_image_ap`, `build_image_rows`, `format_report`, `summarise` all tested in `test_dataset.py`, and `mean_ap` itself in `test_evaluate.py`. The exclusion accounting (`n_images` vs `n_found`) is where a silent error would hide. |
| 5 | **`seg_training.py`** | 587 | Produces every checkpoint and the `val_instance_counts` the guard trusts. Its `evaluate_model` IoU was read against `seg_evaluate`'s and matches. `evaluate_model` and `_split_dataset` are untested — and `_split_dataset` determines what "validation" *means* for four other tools. |
| 6 | **`convert_cad.py`** | 570 | Sets the physical dimensions every millimetre downstream depends on. Genuinely tested in `test_synth3d.py` (unit parsing, catalog merge), which is why it is not higher. |
| 7 | **`generate3d.py`** | 549 | Makes the corpus, but cannot be imported outside Blender, so nothing here is unit-testable as written. Risk is real but structural, not addressable by a test file. |
| 8 | **`verify3d.py`** | 187 | Zero tests, but emits **no numbers** — it draws contact sheets. A defect produces a wrong-looking picture, not a wrong claim. |
| 9 | **`model.py`** | 127 | A thin `FasterRCNN` factory, exercised indirectly by training and inference tests. |

---

## 7. The single highest-value missing test, per tool

Ranked; the top three are the ones worth writing first.

1. **`seg_ablation` — `_pack_count` at two different `mm_per_px` on one label map must not
   change the *conclusion*.** This is the test that would have caught §1 the day the constant
   was left behind. It needs no checkpoint and no dataset: build one label map, pack it at
   0.625 and at 0.821, assert the cell counts agree (they do not — 2 vs 9 **[EXEC]**). The
   assertion fails today, which is the point.
2. **`seg_ablation` — `main()` must reach a measurement on a config whose `crop_size`
   rasterisation differs from native.** A CLI smoke test over a two-crop fixture with the
   anchored config's shape. Catches §2, and would have caught both prior `TypeError`s. The
   general form — *every `main()` in this set gets one smoke test* — is the single change with
   the widest coverage return, since all three historical breakages were `main()`-only.
3. **`seg_evaluate` — `evaluate()`'s pooling must be pooled, not averaged.** Two fake crops
   with hand-chosen intersection/union (50/150 and 10/10) and an assertion that the result is
   0.375 and *not* 0.667. Today this required a fake dataset, a fake segmenter and two
   monkeypatches to check by hand (§4); the function that produces every published IoU has no
   test of the arithmetic that produces it — only of the scale behaviour around it.
4. **`calibrate_tau` — `_sweep` must not skip a candidate τ equal to its own source record's
   IoU.** The code comment at `:88-92` describes a float-rounding trap it consciously avoided;
   nothing pins that behaviour down, so a later "tidy the table formatting" change reintroduces
   it silently.
5. **`eval_real` — the scored/found accounting must survive an exclusion.** Assert that with
   one zero-GT image, `n_images` is the scored count, `n_found` the found count, the excluded
   image still appears with its prediction count, and `mAP` is computed over the scored set
   only. This is the number a reader takes at face value.
6. **`seg_training` — `_split_dataset` must return the same partition for the same
   (len, seed).** Four tools call it and the split guard exists solely because it can move.
7. **`seg_evaluate` — `format_report`'s `verdict:` line must flip when a class fails.** The
   architecture claim is one boolean derived from a loop; nothing tests that it can say "NOT
   below".
8. **`convert_cad` — an implausible-extents STEP must be refused, not rescaled silently.**
9. **`model` — anchor geometry must round-trip config → `AnchorGenerator` sizes.**
10. **`verify3d`** — lowest value in the set; it publishes no figure.

---

## 8. Bottom line

The metrics in `seg_evaluate.py` — the largest tool and the one carrying the most published
claims — **are correct.** IoU uses a union denominator, empty predictions are penalised and
absent classes are excluded rather than zeroed, pooling is pooling, area error's signs match
its docstring, boundary displacement's magnitude is right and its text honestly describes the
one-way distance it computes, and there is no unguarded division by zero anywhere in the five
reporting paths. That was checked by hand, against constructed cases, not by reading the code
and agreeing with it.

`seg_ablation.py` is a different matter. Its Δcells figure is measured at a scale the project
has already established describes no frame in the corpus, and the consequence is not the
"compressed magnitude" the existing caveat claims: **the damage-direction count is 2/126 as
published and 5/126 when the same checkpoint is scored on the same crops at the frames' own
scales, with three crops changing sign.** The figure that FDR §13.2.1 explicitly labels "the
figure that matters for safety" is understated by a factor of 2.5, in the unsafe direction.
And the same file's split guard compares two different quantities, so the receipt cannot be
regenerated for any config but the default.

Both are defects of the same species as the two `TypeError`s: a caller left behind when a
shared function moved. Both live in the one tool in this set whose `main()`-only code path has
no test at all.
