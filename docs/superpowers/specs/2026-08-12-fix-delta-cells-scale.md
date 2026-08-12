# Δcells was packed at a scale no frame has — the safety figure was 2.5× understated

Date: 2026-08-12. Baseline `ea83635` (752 tests). Other work landed in the tree
while this ran; the packing path it measures through — `common/packing.py`,
`plan/arbitration.py`, `plan/placement_area.py` — is byte-identical between that
baseline and the commit this lands on, and `plan/scene.py`'s changes over the
same span are purely additive, so every receipt below is valid at HEAD. Acting on
[`audit/2026-08-12-A-measurement-tools.md`](../audit/2026-08-12-A-measurement-tools.md)
findings §1 (critical) and §2 (high), and on the receipt-SHA item recorded as
unrepairable-here in [`2026-08-12-sha-remap.md`](2026-08-12-sha-remap.md) §4a.

**The headline, first, because it is the part that changes what the project may
claim: the figure FDR §13.2.1 calls "the figure that matters for safety" moves
from 2 of 126 to 5 of 126 crops in the damage direction, and the error was in
the unsafe direction.** Nothing was retrained, no dataset was regenerated, no
model changed, and `delta_cells` still measures exactly what it measured before.
Only the scale it measures at changed.

---

## 1. What was wrong

`recog/seg_ablation.py` converted at `resolve_mm_per_px(synth_cfg)` — the
generator's **nominal** framing, 0.6250 mm/px, which is the framing at
`margin = 1.0, zoom = 1.0`. `recog/synth3d/world.py`'s `setup_camera` draws
`margin` from `[1.02, 1.10]` and `zoom` from `param_space.zoom`, so **no frame
in the corpus is rendered at that framing**: the 126 validation crops' true
ground sample distance runs 0.4903–1.0739 mm/px, median 0.8211. `380e7d5` took
that constant out of the planner and `502ef00` took it out of `seg_evaluate` and
`calibrate_tau` — its own message says "`380e7d5` fixed the planner and left
these two". `seg_ablation` was in neither pass.

**Scale here is not a reporting unit, and that is the whole finding.** Everywhere
else the project corrected, mm_per_px was the multiplier on a pixel count: get it
wrong and a published millimetre is understated, but the comparison it feeds
survives. In `_pack_count` it is an *input to the measurement*:

```python
inset_px = max(0, int(round(wall_inset_mm / mm_per_px)))   # erosion RADIUS
strip_w  = (x1 - x0) * mm_per_px                           # strip size in mm
grid     = _rasterise_mask(safe, bbox, mm_per_cell, mm_per_px)   # grid STRIDE
```

The packer then fits fixed-millimetre 18650s into a strip whose millimetre size
came from that number. It decides **what the packer does**, not how the answer
is labelled.

### 1.1 Why "both sides are at the same wrong scale" does not save it

`2026-08-11-scale-calibration.md` §5 filed this as a magnitude caveat:

> *Both sides are at the same wrong scale so the sign is trustworthy, but the
> magnitude is compressed.*

That is false, and the correction is now written into that file rather than left
standing. A shared multiplier cancels in a difference **only where the metric is
linear in it**. Packing is discrete and non-monotone in scale, so `pack(gt)` and
`pack(pred)` do not move together. One label map, one prediction, three scales
(`tests/test_calibration.py::test_the_pack_count_conclusion_is_not_invariant_to_mm_per_px`
is exactly this case):

| mm_per_px | pack(gt) | pack(pred) | Δ | reads as |
| ---: | ---: | ---: | ---: | --- |
| 0.4903 (split low) | 2 | 2 | 0 | agree |
| **0.6250 (nominal)** | **7** | **7** | **0** | **agree** |
| 0.8211 (split median) | 12 | 13 | −1 | **damage risk** |
| 1.0739 (split high) | 24 | 25 | −1 | damage risk |

It was filed as a magnitude caveat; it was a sign caveat.

### 1.2 The metric also had almost no dynamic range at the nominal

Ground-truth label maps only, no segmenter involved, over all 126 val crops:

| | total cells the GT admits | crops packing 0 |
| --- | ---: | ---: |
| at the nominal 0.6250 | **4** | 124 / 126 |
| at each frame's own GSD | **17** | 118 / 126 |

At the scale the receipt used, "the prediction and the truth agree" on 122 of
126 crops was very largely forced by the ground truth admitting nothing to
disagree about, rather than earned by the segmenter.

---

## 2. The corrected figures

Same 126-crop split, same `recog/checkpoints/seg/best.pt`, same wall inset, same
packer, `delta_cells` computed twice per crop — once at the nominal, once at that
crop's own GSD. Produced by the shipped functions only.

| | published (nominal 0.6250) | corrected (per-frame GSD) |
| --- | ---: | ---: |
| mean | +0.008 | **+0.056** |
| median | 0.000 | 0.000 |
| range | [−2, +2] | **[−2, +4]** |
| positive (cells lost, throughput) | 2 / 126 | **6 / 126** |
| **negative (packed where forbidden, damage)** | **2 / 126** | **5 / 126** |
| zero (exact) | 122 / 126 | 115 / 126 |

The nominal column reproduces the previously shipped
`docs/receipts/seg_ablation.txt` **exactly** on every one of those six numbers.
That is the control: it establishes that nothing but the scale moved.

Per-crop, the two scales **disagree on 8 of 126** crops and **7 change sign**.
Three go from zero into the damage direction:

| crop | true GSD | at 0.6250 | at its own GSD |
| --- | ---: | ---: | ---: |
| `scene_00330.png` | 0.8566 | 0 | **−1** |
| `scene_00324.png` | 0.8865 | 0 | **−1** |
| `scene_00086.png` | 0.6994 | 0 | **−1** |

All three pack **zero cells on both sides** at 0.6250 — they were invisible to
the metric, not judged safe by it. The two damage-direction crops the project
already knew about are still negative; the count is 2 + 3 = 5.

---

## 3. Does a conclusion change?

**Yes — one, and it is the one that matters.**

The project could previously say that on its own validation split the segmenter
packs a cell where the ground truth forbids it on **2 of 126** crops, ~1.6 %.
It must now say **5 of 126, ~4.0 %** — 2.5× worse, on the metric FDR §13.2.1
itself nominates as the safety metric, in the unsafe direction, on the same
checkpoint and the same crops. Three of those five crops have never been looked
at by any measurement this project has run, because at the scale it ran them the
crops packed nothing at all.

Three things that did **not** change, stated so the correction is not read as
wider than it is:

- **The sign of the overall result.** The mean is still positive (+0.056): the
  segmenter is still net *conservative* across the split, losing more cells than
  it wrongly gains. The damage direction is a minority behaviour, as before.
- **Every other published figure.** IoU, boundary displacement, signed area
  error, latency, per-SKU tables, τ and the real-photograph comparison are all
  untouched — `_pack_count` is not on any of their paths, and the regenerated
  receipts show every one of those numbers byte-identical (§5).
- **The direction of travel of §13.2.1's regression argument.** It argued the
  damage-direction fraction got *worse* after the tray fix. It is more worse
  than it said.

What follows from it, and is now flagged in `NEXT_STEPS.md` item 3 rather than
done here: **five crops need individual investigation, not two**, and the three
new ones cannot be explained by the mechanism documented for the earlier pair
(`805adb0`), because that investigation was carried out at the nominal scale on
a different split.

One thing this correction does *not* license: reading 5/126 as a real-world
damage rate. It is synthetic-to-synthetic, on the segmenter's own validation
split, under an FFDH-only packer that the shipping planner no longer uses
(`562ca75`). All three caveats predate this work and are unchanged by it.

---

## 4. The second defect: the split guard compared two different quantities

`compute_val_instance_counts`' docstring is explicit — `out_size` MUST be the
checkpoint's `model.crop_size` whenever the result is fed to
`check_split_matches_checkpoint`, because `seg_training` builds its stored counts
off a val DataLoader that rasterises at `crop_size`. Of three callers,
`seg_evaluate.py:995` and `calibrate_tau.py:525` passed it; `seg_ablation.py:615`
did not, and defaulted to native resolution. `calibrate_tau`'s comment at `:516`
names this exact defect as one of the two it was fixing — `seg_ablation` got half
of that fix.

The consequence was a **false** split-drift error on any config where a single
sliver of background survives at native size and is lost to the 256 downsample:
anchored recomputed `background: 124` against the checkpoint's `123`, and the run
died before reaching a measurement. `docs/receipts/seg_ablation.txt` was
regenerable for `configs/segmentation.yaml` only, where the two happen to agree.

Fixed by passing `out_size=crop_size`.

**One correction to the audit while confirming this.** The audit says no ablation
receipt can be produced "for the anchored, wide, CAD-test or the four
CAD-control-holdout configs". That is broader than the measurement supports. The
guard was recomputed both ways against each checkpoint's own recorded counts —
no inference needed — and it fires on **four** of the ten config/checkpoint
pairs, not eight; the rest passed by luck of native and downsampled counts
happening to agree, exactly as the default config does.

| config | old (native) guard | new (crop_size) guard | CLI now |
| --- | --- | --- | --- |
| `segmentation.yaml` | pass | pass | rc=0, 126 crops |
| **`segmentation_anchored.yaml`** | **FIRES** (bg 124 vs 123) | pass | **rc=0**, 127 crops |
| **`segmentation_wide.yaml`** | **FIRES** (bg 119/118, battery 24/23) | pass | **rc=0**, 124 crops |
| `segmentation_anchored_18650.yaml` | pass | pass | rc=0, 127 crops |
| **`segmentation_anchored_crown.yaml`** | **FIRES** (bg 124 vs 123) | pass | **rc=0**, 127 crops |
| `..._holdout_AnkerPowerCore10000.yaml` | pass | pass | — |
| `..._holdout_AnkerPowerCore13000.yaml` | pass | pass | — |
| **`..._holdout_AnkerPowerCore20100.yaml`** | **FIRES** (battery 19 vs 18) | pass | **rc=0**, 128 crops |
| `..._holdout_AnkerPowerCore26800.yaml` | pass | pass | rc=0, 128 crops |
| `segmentation_cad_test.yaml` | skipped (cross-dataset) | skipped | — |

So the four genuinely blocked configs — anchored, wide, anchored_crown and the
20100 control — now reach a measurement, and each was run end to end to confirm
it (rc=0, every crop calibrated from its own sidecar). The failure mode is also
not only `background`: `wide` and `20100` disagree on `battery`, which the
docstring's worked example does not mention.

No receipt is committed for those runs — none ever existed, and this work does
not add published figures it was not asked for. What it establishes is that the
tool can now produce one.

---

## 5. The receipts

**`docs/receipts/seg_ablation.txt`** — regenerated by
`python -m recog.seg_ablation`. The Δcells block moves as §2; the whole
real-photograph half is **byte-identical**, which is the isolation check. Its
header no longer prints `mm_per_px=0.6250`; it prints the same per-frame
provenance block `seg_evaluate` prints (measured 126 of 126 over 112 frames,
median 0.8211, range 0.4903–1.0739), plus a note stating what the nominal was
and why packing does not cancel it.

**The eleven `seg_eval*.txt` receipts** — regenerated by `recog.seg_evaluate`.
The reason is `seg_evaluate.py:778`, which emits `commits 27cbd97..9fcf136` into
every one of them; both SHAs died in the history rewrite. `2026-08-12-sha-remap.md`
§4a declined to hand-edit them, correctly: *"a receipt that no longer matches its
generator is worse than one with a stale SHA, because the first defect is
invisible and the second is not."* The source string is now `a31ac28..043e92d`
and the receipts were regenerated rather than edited. Two sibling comments in the
same file (`:429` same range, `:124` `58dd21d` → `380e7d5`) and one in
`seg_ablation.py` (`138105d` → `75db46a`) were corrected in the same pass.

Across all eleven, the diff is **only** that line plus the four-row wall-clock
latency table — every IoU, boundary displacement, area error, instance count and
per-SKU figure is byte-identical, which independently confirms the reproducibility
audit's determinism finding. The latency table cannot be carried across a
regeneration; it re-measured at 16.6 ms batched / 57.1 ms looped at 8 crops
(the pair was 21.2 / 88.0 before). The verdict is unchanged — inside the 50 ms
budget, batching still load-bearing — and the four documents quoting the old pair
(`FDR_v3.md`, `NEXT_STEPS.md`, `README.md`, `CV_BULLETS.md`) now quote the new
one and the 16.6–21.2 ms spread across five clean runs.

---

## 6. Tests

Four added to `tests/test_calibration.py`, beside the tests that pin the same
property for `seg_evaluate` and `calibrate_tau`.

1. **`test_the_pack_count_conclusion_is_not_invariant_to_mm_per_px`** — the test
   the audit named (§7 item 1), with one correction it needs in order to be true.
   The audit asks that packing "must not change the conclusion" across two
   scales and proposes asserting the two counts agree. **They cannot agree, and
   should not:** the same pixel rectangle at a coarser GSD is a larger *physical*
   rectangle, and 18650s are packed in millimetres, so the count must rise. That
   assertion is physically false and would fail before *and* after any fix. What
   the caveat on record actually claimed — that the *conclusion* survives the
   wrong scale — is testable, and this test is it: one map, one prediction,
   Δ = 0 at 0.4903 and 0.6250, Δ < 0 at 0.8211. It documents the defect rather
   than guarding against its return, which is why it is paired with 2 and 3.
2. **`test_seg_ablation_refuses_one_constant_for_the_whole_split`** — the
   signature must not accept the thing that went wrong. Failed before the fix
   (no `TypeError` was raised); passes after.
3. **`test_seg_ablation_measures_each_crop_at_its_own_frames_scale`** — the
   regression at the level the receipt publishes. Two frames differing only in
   the GSD their own sidecars record: at the nominal the split reports **0**
   damage-direction crops, at the frames' own scales **1**. Failed before.
4. **`test_seg_ablation_counts_val_instances_at_the_checkpoints_crop_size`** — a
   `main()` smoke test over a fixture whose one background pixel survives natively
   and is lost at `crop_size`, with the fixture asserting its own discriminating
   power first. Before the fix it reproduced the anchored config's false
   split-drift `SystemExit` verbatim. It also covers what all three of this
   module's historical breakages had in common: they were `main()`-only, which is
   the audit's §0 conclusion.

Suite: **814 passed** (4 added by this work; the baseline was 752 at `ea83635`
and the rest of the growth is other work in the tree at the same time —
`execution/`, `plan/`, `main.py`, none of it touched here). Excluding the two
socket-based `execution` test files, **769 passed, 0 failed**, and those two
files pass 45/45 in isolation on three consecutive runs. They flake
intermittently when the whole suite runs — two different tests on two different
full runs — which is port/timing contention in another agent's in-flight work,
not this change: nothing here imports `execution` and nothing in `execution`
imports `recog.seg_ablation`.

---

## 7. What was deliberately not done

- **No retrain, no dataset regeneration, no model change.** None was needed: the
  defect is in the measuring instrument, and the control in §2 proves it.
- **`delta_cells`' definition is unchanged.** Same function, same sign
  convention, same inputs, same packer. Only the scale each crop is packed at.
- **The five damage-direction crops were not investigated.** That is
  `NEXT_STEPS.md` item 3's work and it is now larger, not smaller.
- **`recog/calibrate_tau.py:518-519` still cites two dead SHAs** (`138105d`,
  `58dd21d`). Comments only, no behaviour depends on them, and that file is
  outside this work's scope; recorded here so it is not rediscovered as new.
- **The one-way boundary-displacement caveat (audit §3)** is untouched. It is a
  real limitation and correctly described by the code and the receipt; it is not
  a wrong figure.
