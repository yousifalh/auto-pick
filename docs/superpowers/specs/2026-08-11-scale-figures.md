# The published millimetres were understated. Corrected, not retracted.

Date: 2026-08-11
Baseline: `58dd21d`, 737 tests passing. Concurrent work landed `0d7d204`
(+7 tests) while this was in progress, so the branch point is 744.
After: **752** — 8 added here, none deleted, none edited.
Acting on: `docs/superpowers/specs/2026-08-11-scale-calibration.md` §5
and `2026-08-11-placement-feasibility.md` §1.1 and §4.

`58dd21d` made the **planner** take each frame's scale from that
frame's render metadata. It left the two tools that publish millimetres
into `docs/receipts/` — `recog.seg_evaluate` and `recog.calibrate_tau` —
converting at `resolve_mm_per_px`, the generator's nominal framing at
`margin = 1.0, zoom = 1.0`. `recog/synth3d/world.py` randomises both per
scene, so that framing describes **no frame in any of these corpora**;
over `recog/dataset3d_seg`'s 126 validation crops the true ground sample
distance runs 0.490–1.074 mm/px with a median of 0.821, against a
nominal 0.625.

Every millimetre those receipts published was therefore understated, by
**1.31× for a length and 1.72× for an area** on the primary split. Every
pixel-space figure was, and remains, correct.

**The headline claim survives.** `bay` boundary displacement is
**1.226 mm**, not 0.949 mm. It still clears what a 28 × 28 mask head
would quantise to, and by the same margin as before — see §3, which is
the part of this correction most easily got wrong.

---

## 1. What changed in the code

Both tools now resolve scale **per frame**, through
`recog.calibration` — the module `58dd21d` created for exactly this and
which already owns both `resolve_mm_per_px` (nominal) and
`frame_mm_per_px_for_image` (per frame). **No arithmetic was added.** A
second copy of `ortho_scale * 1000 / width` is the defect this project
already paid for once, and `recog/calibration.py`'s own docstring says
so.

The glue is one function, `recog.seg_evaluate.resolve_frame_scales`,
which maps crop index → that crop's mm/px and is imported by
`recog.calibrate_tau` — the same direction `calibrate_tau` already
imports `load_synth_config`, `resolve_mm_per_px`,
`check_split_matches_checkpoint` and `compute_val_instance_counts`.
`recog.seg_ablation` is deliberately untouched: its Δcells metric is
defined at the nominal scale, both arms are at the same scale, and
changing it would change a metric definition rather than correct a
conversion (`2026-08-11-scale-calibration.md` §5 records the same
judgement and the same caveat).

**An unknown scale raises.** `resolve_frame_scales` takes a `fallback`
that defaults to `None`; a frame with no render sidecar and no
configured fallback raises `plan.placement_area.UnknownScale` — the
same class the planner raises for the same condition, imported rather
than restated. A real fixed-mount camera *is* one calibrated scale and
is served by the new `--fallback-mm-per-px` on both CLIs, which records
in the receipt that it was a fallback and on how many crops. **Neither
tool can now be handed one constant at all:** `evaluate()` and
`collect_records()` raise `TypeError` on a scalar. A signature that
still accepts one number is a signature that can regress to one number
silently, which is what happened.

**Both receipts stopped printing a bare number.** A header reading
`mm_per_px : 0.6250` cannot distinguish a run measured at each frame's
own scale from one measured at a constant, which is how the constant
survived. They now print the distribution, the measured/fallback split,
and a note naming the nominal framing and the factor it understated by.

### 1.1 A second unrunnable CLI, found the way the brief predicted

`recog.calibrate_tau`'s `main` called
`check_split_matches_checkpoint(checkpoint, counts)` — two arguments to
a function that has taken three since `138105d`. **`python -m
recog.calibrate_tau` has raised `TypeError` before reaching a single
measurement ever since, so `docs/receipts/tau_calibration.txt` was as
unregenerable as `seg_ablation.txt` was, for the same reason and since
the same commit.** `58dd21d` fixed the `seg_ablation` instance and did
not know about this one. It also passed `out_size=None` where
`seg_training`'s stored counts come off a DataLoader rasterising at
`crop_size`, so the guard would have compared two different quantities
once it ran at all. Both are fixed here, matching
`recog.seg_evaluate.main`'s own call. Every other receipt-producing CLI
in scope was run end to end; the list is in §5.

---

## 2. The figures that moved

`docs/receipts/seg_eval.txt`, the 126-crop `recog/dataset3d_seg`
validation split, at a measured median 0.8211 mm/px:

| figure | published | corrected | factor |
| --- | ---: | ---: | ---: |
| boundary displacement, `bay` | 0.949 mm | **1.226 mm** | 1.29× |
| boundary displacement, `electronics` | 0.987 mm | **1.273 mm** | 1.29× |
| boundary displacement, `obstruction` | 1.184 mm | **1.582 mm** | 1.34× |
| optimistic area error, `bay` (mean/crop) | 51.5 mm² | **79.2 mm²** | 1.54× |
| optimistic area error, `electronics` | 25.3 mm² | **40.4 mm²** | 1.60× |
| optimistic area error, `obstruction` | 18.3 mm² | **28.8 mm²** | 1.57× |
| conservative area error, `bay` | 50.3 mm² | **81.7 mm²** | 1.62× |

Areas move with the square of the scale correction; lengths move
linearly. That is why the area column moves further, and it is a check
on the change rather than a surprise.

**Every pixel-space figure is bit-identical.** Per-class IoU
(0.3629 / 0.9398 / 0.8903 / 0.8613 / 0.6579 / 0.6907), the selected mean
(0.8032), instance counts, crop counts and the cartridge/bay
co-occurrence contingency are unchanged in all eleven regenerated
`seg_eval*` receipts. They never multiply by mm/px, and if they had
moved the change would have done something other than fix the scale.

The other ten receipts are the same correction on other splits; the
per-split factor tracks each split's own median GSD. The full
before/after is §4.

### 2.1 τ, where it is not only a reporting correction

`docs/receipts/tau_calibration.txt` is different in kind. It converts
the 4.25 mm wall inset and the 18650's footprint into pixels and then
**feeds them to the measurement** — the inset is `arbitrate`'s erosion
radius and the footprint is `admits_a_cell`'s structuring element — so
correcting the scale changes the arbitration IoUs themselves. At the
nominal 0.625 every frame was eroded by 7 px and tested against a
30 × 104 px cell; at each frame's own scale the erosion is 4–9 px and
the cell 22 × 79 px at the median.

| figure | published | corrected |
| --- | ---: | ---: |
| τ | 0.5715 | **0.5695** |
| IoU distribution (mean / min / max) | 0.7807 / 0.5715 / 0.9607 | 0.7826 / 0.5695 / 0.9522 |
| largest optimistic error | 1278 px of a 3045 px² cell, 42.0 % | **851 px of a 1375 px², 61.9 %** |
| population, cartridges admitting a cell | 35 of 126, **0** | 35 of 126, **0** |

The null result is untouched: not one cartridge admits a cell at any
threshold, so the fail budget still never binds and τ is still the
sample minimum rather than a calibrated boundary. The FDR's "the
largest optimistic error SHRANK, 79.4 % → 42.0 %" becomes 79.4 % →
61.9 % — the direction survives, the margin is narrower than reported.
The gate is retired in code (`5a619fc`), so no behaviour depends on any
of this.

---

## 3. Does any claim change sign? One nearly did, and it is a units error

**No claim changes sign. One would have, if the correction had been
applied to one side of a comparison only — and it is worth recording
because the error was made and caught here, not avoided.**

The architecture argument (FDR §13.2.1) is that the per-ROI segmenter's
boundary displacement beats what a 28 × 28 Mask R-CNN head would
quantise to. That threshold was a constant in the code,
`MASK_HEAD_QUANTISATION_MM = (2.9, 6.4)`. Rescaling the measurement
while leaving the threshold at 2.9 flipped `electronics` on the CAD
test split from clearing (2.678 mm) to failing (3.177 mm) — a verdict
change produced entirely by comparing a per-frame number against a
constant-framing number.

**The mask head quantises in PIXELS.** FDR §13.2.1 derives 2.9 × 6.4 mm
from a PowerCore26800 crop of ~131 × 288 px split 28 ways — that is
**4.68 × 10.29 px**, and 2.9 × 6.4 mm is only its value at 0.625.
Converting it over the same crops at the same scales gives 3.844 mm on
the primary split. Both sides are pixel counts times the same number,
so the ratio is scale-invariant:

| class | clears by, published | clears by, corrected |
| --- | ---: | ---: |
| `bay` | 3.1× | **3.14×** |
| `electronics` | 2.9× | **3.02×** |
| `obstruction` | 2.4× | **2.43×** |

The module now stores `MASK_HEAD_QUANTISATION_PX` beside the published
mm figure and judges the verdict per class against `4.68 px × that
class's own mean mm/px`. `tests/test_calibration.py::
test_the_mask_head_comparison_is_scale_invariant` fails if that
comparison is ever re-mixed. **Every verdict in all eleven receipts is
what it was before the correction** — including
`seg_eval_wide_on_wide_val.txt`'s pre-existing *NOT below*, which was
already failing on `electronics` and still is.

So: the millimetres moved, the ratios did not, and the sentence to
quote is the one the brief supplied — 1.226 mm still beats what a mask
head would quantise to, and this is a correction, not a retraction.

---

## 4. Full before/after, all eleven receipts

Boundary displacement, mm, `bay` / `electronics` / `obstruction`, and
the mean optimistic area error for `bay`. `verdict` is the receipt's own
architecture verdict.

| receipt | bd before | bd after | opt `bay` before | after | verdict |
| --- | --- | --- | ---: | ---: | --- |
| `seg_eval` | 0.949 / 0.987 / 1.184 | **1.226 / 1.273 / 1.582** | 51.5 | **79.2** | BELOW → BELOW |
| `seg_eval_anchored_on_anchored_val` | 1.511 / 0.956 / 1.690 | **1.847 / 1.262 / 2.164** | 130.4 | **163.3** | BELOW → BELOW |
| `seg_eval_wide_on_wide_val` | 2.984 / 6.012 / 2.411 | **3.534 / 6.379 / 2.922** | 330.5 | **435.2** | NOT below → NOT below |
| `seg_eval_anchored_on_cad_test` | 1.714 / 2.678 / 1.877 | **2.052 / 3.177 / 2.430** | 386.8 | **522.9** | BELOW → BELOW |
| `seg_eval_anchored_18650_on_cad_test` | 1.495 / 2.263 / 1.810 | **1.799 / 2.764 / 2.394** | 457.2 | **597.7** | BELOW → BELOW |
| `seg_eval_anchored_crown_on_cad_test` | 1.265 / 2.193 / 1.749 | **1.561 / 2.699 / 2.286** | 77.8 | **110.9** | BELOW → BELOW |
| `seg_eval_wide_on_cad_test` | 1.251 / 2.461 / 1.752 | **1.524 / 3.025 / 2.319** | 387.9 | **560.3** | BELOW → BELOW |
| `seg_eval_cad_control_…10000_on_cad_test` | 0.831 / 1.138 / 1.534 | **1.028 / 1.411 / 2.003** | 36.4 | **53.9** | BELOW → BELOW |
| `seg_eval_cad_control_…13000_on_cad_test` | 0.892 / 1.203 / 1.369 | **1.105 / 1.501 / 1.795** | 44.8 | **65.1** | BELOW → BELOW |
| `seg_eval_cad_control_…20100_on_cad_test` | 0.909 / 1.242 / 1.481 | **1.127 / 1.553 / 1.955** | 42.5 | **62.9** | BELOW → BELOW |
| `seg_eval_cad_control_…26800_on_cad_test` | 0.858 / 1.221 / 1.539 | **1.062 / 1.519 / 2.027** | 35.6 | **52.2** | BELOW → BELOW |

Lengths move by 1.06–1.34×, areas by 1.25–1.62×, and the spread across
receipts is each split's own median GSD showing through. **Eleven of
eleven verdicts are unchanged, and the per-class IoU block and the
per-SKU IoU table are byte-identical in every one** — verified by
extracting those blocks from `git show HEAD:<receipt>` and from the
regenerated file and comparing them directly, not by eye.

`seg_eval_wide_on_wide_val`'s pre-existing *NOT below* is unrelated to
this correction: `electronics` fails on that split at either
calibration (6.012 against 2.9 before; 6.379 against 3.569 now). Its
`bay` row is the only place where a per-class ratio moved
materially — 0.97× to 1.03×, from failing to just clearing — because a
mean over crops at differing scales is not the mean at the mean scale.
It does not change that receipt's verdict, which `electronics` decides.

---

## 5. Verification

* `pytest tests/` — **752 passed**, exit 0. Eight added, none deleted,
  none edited: `tests/test_calibration.py` goes 12 → 20 test functions.
  The other seven of the 737 → 752 move are `0d7d204`'s, which landed
  concurrently.
* `python main.py --config configs/demo.yaml` — the torch-free demo
  runs, **10/10 placed**, `frames_with_scale: 0` (the cv2 generator
  writes no sidecar, so `demo.yaml`'s configured fallback applies —
  which is exactly the case the fallback exists for).
* `python -m recog.seg_evaluate` — runs; all eleven receipts
  regenerated from the CLI, never hand-edited.
* `python -m recog.calibrate_tau` — runs **for the first time since
  `138105d`** (§1.1); receipt regenerated.
* `python -m recog.seg_ablation` — not re-run and not changed by this
  work. Its Δcells metric is defined at the nominal scale on both arms;
  see §1. **Note for whoever picks this up next:** `0d7d204` states
  that its `_rasterise_mask` change moves `docs/receipts/seg_ablation.txt`
  and `docs/receipts/main_seg_run.txt` and that it did not regenerate
  them. Neither is a receipt this work touches or is affected by — no
  code path here reaches `_rasterise_mask` — so they are left for that
  change's owner rather than regenerated blind from here.

The tests that fail if either tool reverts to a constant scale:

* `test_seg_evaluate_millimetres_move_with_the_scale_but_iou_does_not`
  — the same pixels and the same predictions scored twice, at each
  frame's own GSD and at 0.625. Every millimetre must differ; every
  pixel-space figure must be identical.
* `test_seg_evaluate_refuses_one_constant_for_the_whole_split` and
  `test_calibrate_tau_refuses_one_constant_for_the_whole_split` — the
  signatures themselves reject the thing that went wrong.
* `test_calibrate_tau_erodes_the_wall_inset_per_frame` — 4.25 mm is
  9 px at 0.4915 and 4 px at 1.0915; one constant cannot be both.
* `test_an_uncalibrated_frame_raises_rather_than_reverting_to_a_constant`
  and `test_a_deliberate_fallback_is_honoured_and_recorded_as_one`.
* `test_the_mask_head_comparison_is_scale_invariant` — §3.

---

## 6. Documents corrected

Each names the superseded figure rather than replacing it silently, so
a reader who remembers 0.949 mm finds out what happened to it.

* **`docs/FDR_v3.md` §13.2.1** — the boundary-displacement table
  (0.949 / 0.987 / 1.184 → 1.226 / 1.273 / 1.582 mm), with the
  mask-head column and the scale-invariant *clears by* ratios beside
  it; the τ paragraphs (0.5715 → 0.5695, 42.0 % → 61.9 %); and the
  latency figures, which the same regeneration re-took (§7).
* **`docs/FDR_v3.md` §8** — `main_seg_run.txt`'s figures had been stale
  since `d6c46ac` and `58dd21d` (1 → 3 pick-and-places, 7 poses), and
  the sentence *"at `mm_per_px: 0.625` (this dataset's true framing)"*
  asserted the exact falsehood this work corrects.
* **`docs/FDR_v3.md` §3** — the operating envelope, a separate finding;
  see `2026-08-11-placement-feasibility.md` and §8 below.
* **`docs/FDR_v3.md` §13.2.1's CAD geometric-ceiling paragraph** — its
  "4.25 mm / 7 px" is now "4–9 px, 5 px at the median", and its
  standing request to render-verify the ceiling before relying on it is
  answered.
* **`docs/FDR_v3.md` §13.3** — the status paragraph's latency and
  `main_seg_run` figures.
* **`docs/NEXT_STEPS.md`** — Plan C's headline row, Plan D's latency
  row, the tray-fix boundary comparison (item 5), the τ item, and Step
  2's `obstruction` baseline.
* **`README.md`** — the segmentation latency sentence, which cites the
  receipt directly.

Two documents were deliberately **not** rewritten.
`docs/superpowers/specs/2026-08-10-tau-difficulty-design.md` is a
historical record of a measurement taken at the time; it is superseded
by this file rather than edited. `docs/receipts/seg_ablation.txt` is
unaffected, per §1.

---

## 7. One thing moved that is not a scale figure

`recog.seg_evaluate`'s latency table is wall-clock and is re-taken on
every regeneration; it cannot be carried forward. Regenerating
`seg_eval.txt` moved the 8-crop figures from **20.2 / 76.5 ms (3.8×)**
to **21.2 / 88.0 ms (4.2×)**. A second run in the same session gave
18.1 / 53.6 ms, so the spread is wide and this is the run-to-run
variation FDR §13.2.1 already documents at length, not an effect of
this change — which touches no code on the inference path. The
conclusion is unchanged on every pair ever measured: the looped figure
alone breaches the 50 ms budget. The citations in FDR §13.2.1, §13.3,
`NEXT_STEPS.md` and `README.md` were moved to the current receipt and
the spread stated, rather than left pointing at a receipt that no
longer says that.

---
