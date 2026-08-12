# mm_per_px becomes a property of the frame

Date: 2026-08-11
Baseline: `1de9b02`, 708 tests passing. After: 737 passing.
Acting on: `docs/superpowers/specs/2026-08-11-placement-feasibility.md`
sections 1.1, 4, 5 and 6.

**Read section 4 before quoting anything from this file. The hard
acceptance criterion — placements overlapping ground-truth non-floor
material must reach 0 — is NOT met, and the reason is not the scale.**

---

## 1. What was wrong

`mm_per_px` was a configuration constant while the renderer varied the
framing per frame. `configs/demo_seg.yaml` set `0.625`, derived as
`layout.area[0] * 1000 / render.res[0]` — the framing at `margin = 1.0,
zoom = 1.0`. `recog/synth3d/world.py:setup_camera` sets `ortho_scale =
need * margin * zoom` with `margin` drawn from `[1.02, 1.10]` and `zoom`
from `param_space.zoom`, so **no frame in the corpus is rendered at that
framing.** Measured over the 30 cartridge instances the true ground
sample distance runs 0.490–1.045 mm/px, median 0.858.

The consequence ran both ways, and the second way is the one that
matters:

* On 24 of 30 the planner **under-read** the scene, by 27 % at the
  median — 9 cells not placed.
* On the other 6 it **over-read**: the planner reserved a footprint
  *smaller than the cell it was about to place*. A cell reserved as
  30 × 104 px is 38 × 132 px of real 18650 at `scene_00005`'s true
  0.4915 mm/px. The robot is commanded to a point the planner checked
  at the wrong size.

Every distance downstream is derived from this one number: the
extractor turns it into the cartridge wall's erosion radius in pixels
*and* into the occupancy grid's pixels-per-cell, and the planner turns
the resulting rectangle into workspace millimetres. A frame measured at
the wrong scale is wrong three times, in the same direction.

---

## 2. Where the scale now enters, and why there

**The `Snapshot`.** `common.types.Snapshot` gains `mm_per_px: float |
None`, filled by the image source and read once per cycle by
`Planner.cycle`, which hands the resolved value to everything that needs
it.

Three candidates were considered:

| candidate | rejected because |
| --- | --- |
| the extractor | It is one of *two* consumers. The planner also converts pixels to millimetres, and `main._build_planner`'s docstring already recorded that the two "must not drift" — a constraint kept by hand. Putting the scale only in the extractor leaves the hand-maintained coupling in place. |
| the planner config | That is what it already was. A config cannot know that the renderer randomises framing per scene. |
| **the snapshot** | **Scale is a property of the frame**, and `Snapshot` is the only object that crosses Recognition → Planning per frame. One resolution point (`Planner.frame_scale`) means the extractor and the planner agree *by construction* rather than by convention. |

The chain, end to end:

1. `recog.calibration.frame_mm_per_px(meta)` — `camera.ortho_scale *
   1000 / width`, one arithmetic definition shared with the nominal
   `resolve_mm_per_px`, which **moved here from `recog.seg_evaluate` and
   is re-exported**, not copied. `tests/test_calibration.py` asserts the
   two names are the same object, because `recog.calibrate_tau` and
   `recog.seg_ablation` import it from the old location and quote the
   result in shipped receipts.
2. `main._synthetic_source` yields `(rgb, mm_per_px)`, reading the
   render sidecar per frame. The image source is where calibration
   belongs: a frame source stands in for a camera, and a camera is the
   thing that knows how big a pixel is. `_camera_source` yields `None` —
   a webcam frame carries no calibration.
3. `_run_one_cycle` sets `snapshot.mm_per_px`. Not the detector: the
   detector is handed an array and has no idea where it came from.
4. `Planner.cycle` resolves frame-scale-then-fallback once and passes it
   to `_ensure_placement_areas`, which passes it to
   `extractor.extract(..., mm_per_px=...)`.
5. `PlacementArea.mm_per_px` reports the scale actually measured at, and
   `Cartridge.mm_per_px` stores it. Packing and pose construction read
   the **cartridge's** scale, not the current frame's, because the
   rectangle is in pixels and the twin outlives the frame.

**The fallback is explicit.** `PlannerConfig.mm_per_px` and both
extractors' constructor argument default to `None`, not to `0.38` /
`0.625`. An absent `camera.mm_per_px_x` means the camera was not
calibrated, not that it is 0.38. `configs/demo_seg.yaml` keeps `0.625`
and now says in the file that it is a fallback for sidecar-less frames
only. A real fixed-mount camera *is* a single calibrated scale and is
served by exactly this.

**An unknown scale raises.** `plan.placement_area.UnknownScale` fires
when neither the frame nor the config supplies one, and
`_ensure_placement_areas` re-raises it explicitly rather than letting the
blanket `except Exception` absorb it — that handler means "skip this
cartridge, retry next frame", which for a configuration error means
retrying forever while reporting a clean zero.

**One consequence worth stating.** Cartridges persist across frames by
IoU and carry their pixel rectangle with them. If the scale moves under a
cached area, those pixels no longer mean the millimetres they were
measured to mean. `Planner._drop_areas_measured_at_another_scale`
discards such areas and counts it. On a fixed-mount camera it never
fires; it measured 0 on all three runs below.

---

## 3. Measurements

Reproduced exactly as the feasibility spec section 1 specifies: 60 frames
of `recog/dataset3d_seg` through `recog/checkpoints/best.pt` +
`recog/checkpoints/seg/best.pt` + `SegmentationPlacementAreaExtractor
(mm_per_cell=1.5, wall_inset_mm=4.25)`, cartridge identity from
`EnvironmentModel`'s running counter, then `pack_best_effort` with an
18.5 × 65.0 mm cell. Every step is a shipping method — `Planner
.frame_scale`, `EnvironmentModel.update_from_snapshot`, `Planner
._drop_areas_measured_at_another_scale`, `Planner
._ensure_placement_areas`, `Planner._pack_cartridge`. Nothing is
reimplemented.

The pre-fix column is produced by withholding each frame's calibration so
the 0.625 fallback applies — the shipping configuration before this
change — not by checking out the old code. It reproduces the published
baseline exactly, including the per-instance counts (`c7` 7, `c14` 1,
`c28` 2, `c53` 1, `c57` 3, `c64` 2, `c70` 1).

| | before (0.625 for all) | after (per-frame true GSD) | predicted |
| --- | ---: | ---: | ---: |
| instances | 30 | 30 | — |
| instances with ≥ 1 cell | **7** | **13** | 13 |
| cells placed | **17** | **26** | 26 |
| areas dropped (rescale) | 0 | 0 | — |

**The prediction is met exactly**: 13 instances and 26 cells, the
feasibility spec section 5's `mm_per_px 0.625 -> the frame's true GSD`
row. Seven of the 23 previously-unplaceable cartridges recover
(`c25`, `c36`, `c51`, `c56`, `c61`, `c80`, `c82`).

### 3.1 Safety

Each placed cell scored against the frame's ground-truth label map over
two footprints: the one the planner **reserved**, and the one a real
18.5 × 65 mm cell **physically occupies** at the frame's true GSD,
centred on the point the planner commands. Threshold > 5 %.

| footprint | before | after |
| --- | ---: | ---: |
| reserved | 1 / 17 | 5 / 26 |
| physical, top-left anchored | **3 / 17, worst 21.2 %** | 5 / 26 |
| physical, centre anchored | 8 / 17, worst 16.7 % | 5 / 26 |
| physical, "material" only (excludes background) | 3 / 17, worst 21.2 % | **4 / 26, worst 9.1 %** |

Two notes on the conventions, because they matter for reading the
before-column:

* The feasibility spec's **3 of 17 at 21.2 %** is reproduced exactly by
  anchoring the physical footprint at the reserved rectangle's
  **top-left corner**. `Planner._build_pose` commands the **centre**, and
  under that anchoring the same baseline is **8 of 17**. The spec's
  figure understates the pre-fix exposure.
* **After the fix all three conventions give the same number**, because
  reserved and physical footprints are now identical. That collapse *is*
  the scale symptom being gone: the planner is no longer reserving space
  of a size the cell does not have.

---

## 4. The acceptance criterion is not met, and why

**5 of 26 placements (4 by the "material" reading) still overlap
ground-truth non-floor by more than 5 %.** The criterion was 0. It is not
reached, and it cannot be reached by anything in this change's scope.

The residual is not scale. Feeding **ground-truth label maps** through
the identical path at the identical true scale — same detector boxes,
same wall inset, same packer:

| | instances | cells | overlaps > 5 % | worst |
| --- | ---: | ---: | ---: | ---: |
| predicted masks, true scale (ships) | 13 | 26 | **5** | 100 % |
| **GT masks, true scale (oracle)** | 10 | 24 | **0** | **1.1 %** |

The oracle row also reproduces the feasibility spec's `GT, inset 4.25`
row (10 instances, 24 cells) independently. **With perfect segmentation
the overlap count is 0.** Every residual overlap is perception error that
the scale defect was previously masking.

The five, individually:

| instance | true GSD | overlap | what it lands on |
| --- | ---: | ---: | --- |
| `scene_00014/c25` | 0.998 | **100 %** | 1170 px of pure GT **background** — 0 px of material |
| `scene_00019/c36` | 1.038 | 5.2 % | 59 px of tray wall |
| `scene_00033/c57` | 0.686 | 8.6 % | 155 px wall, 41 px obstruction |
| `scene_00033/c57` | 0.686 | 5.3 % | 135 px wall |
| `scene_00052/c80` | 1.045 | 9.1 % | 101 px wall |

Four of the five are the predicted bay boundary overrunning the tray wall
by roughly 3 px. `docs/receipts/seg_eval.txt` reports 0.949 mm boundary
displacement *computed at 0.625*; at these frames' true GSD that is
≈1.5 mm, and its optimistic placeable-area error of 51.5 mm² per crop is
the same quantity. These are that receipt, downstream.

`scene_00014/c25` is different and worse. The detector returned **one box
spanning the cartridge and three loose cells beside it** (box
330,130–520,272 against a GT unit at 362,156–426,248); the segmenter then
called the backdrop between those loose cells `bay`, and the planner
placed a cell into open air. `BadDetectorBox` does not catch it — the
crop centre lands on real foreground, which is the only condition that
gate tests. At 0.625 this instance placed nothing, so the bad box was
invisible; at the true scale it places one cell entirely off the
cartridge.

**This is the honest verdict: the scale fix removes the scale-attributable
unsafe placements completely (reserved ≡ physical, and the oracle scores
0 of 24), and leaves a perception-attributable residual of 5 of 26 that
was previously hidden underneath it.** By the raw count the safety metric
moves 3 → 5 (or 8 → 5 under the anchoring the planner actually uses);
worst-case severity on material falls 21.2 % → 9.1 %.

Nothing was adjusted to chase 0. The two levers that would move it —
the segmenter and the detector — are out of scope by the brief, and the
two that are in reach (`wall_inset_mm`, the 18.5 mm nominal) are
explicitly forbidden and were measured in the feasibility spec to recover
nothing anyway.

### Recommended next, not done here

1. **The bad-box family.** `scene_00014/c25` is a detector failure that
   produces a *confident, plausible* placement. It is the only 100 %
   miss in the corpus and the only one that is not a boundary sliver.
   Widening `BadDetectorBox` to test whether the predicted bay is
   consistent with a single cartridge — rather than only whether the
   crop centre is foreground — would catch it.
2. **The four wall slivers** need the segmenter's optimistic boundary
   error, not a planner change. Note the feasibility spec section 5
   finding 3: `_rasterise_mask` samples cell *centres*, dilating the free
   region by up to 0.75 mm per edge, and is currently *producing*
   placements. That is the same error in the same direction and should be
   decided together with this.

---

## 5. What else this invalidates

* **`docs/receipts/main_seg_run.txt` — regenerated** from
  `python main.py --config configs/demo_seg.yaml --receipt ...`. It was
  doubly stale: it predated the packer fix, and it predated this. The
  recorded 1 pick / 8 poses is now **3 picks / 7 poses over 15 frames**,
  8 placement areas, and a new line records that **15 of 15 frames
  carried their own scale**. The `mm_per_px` line no longer prints a bare
  number — a receipt that printed one constant could not distinguish a
  run planned at the frames' own scales from one planned at a fallback,
  which is how the constant survived as long as it did.
* **Δcells — re-measured, unchanged.** `docs/NEXT_STEPS.md` flags it as
  measured under the FFDH-only planner. Re-running
  `python -m recog.seg_ablation` regenerates
  `docs/receipts/seg_ablation.txt` **byte-identical** (mean 0.032, median
  0.000, range [-2, 2], 4 lost / 2 forbidden / 120 zero of 126 crops).
  The reason it did not move: `recog.seg_ablation._pack_count` calls
  `first_fit_decreasing` directly and never used the planner's packer, so
  `pack_best_effort` shipping cannot change it. **The outstanding flag
  can be cleared, with that correction.**
  *Caveat, not fixed here:* Δcells is still computed at the nominal
  `resolve_mm_per_px` = 0.625, now known to describe no frame. Changing
  it would change a metric definition, which the brief forbids.

  > **CORRECTION, 2026-08-12 — the rest of this caveat, as originally
  > written, was false.** It read: *"Both sides are at the same wrong
  > scale so the sign is trustworthy, but the magnitude is compressed —
  > a 65 mm cell is 104 px there against a true median of 76 px."* The
  > second half is right and the first half is not, and the first half
  > is the load-bearing one. A shared multiplier cancels in a
  > difference; **packing does not**, because it is a discrete,
  > non-monotone function of scale. `_pack_count` answers "how many
  > fixed-millimetre 18650s fit", so mm_per_px is not a unit on the
  > answer — it sets the wall-inset erosion radius, the strip's
  > millimetre size and the occupancy grid's stride, and therefore
  > changes what the packer does. Measured on the same 126 crops with
  > the same `best.pt`: 8 crops disagree between the two scales, **7
  > change sign**, and **3 go from zero into the damage direction**. The
  > published damage-direction count moves **2 of 126 → 5 of 126** and
  > the range **[−2, +2] → [−2, +4]** — in the unsafe direction. Nor was
  > "the magnitude is compressed" the whole of the second half's story:
  > at 0.625 the split's ground truth admits 4 cells in total and 124 of
  > 126 crops pack none, against 17 cells at the frames' own scales, so
  > the metric had almost no dynamic range to compress. This was filed
  > as a magnitude caveat; it was a sign caveat. Fixed in
  > `2026-08-12-fix-delta-cells-scale.md` and pinned by
  > `tests/test_calibration.py::test_the_pack_count_conclusion_is_not_invariant_to_mm_per_px`.
  > The reasoning error worth keeping: "both sides are at the same wrong
  > scale" licenses cancellation only where the metric is *linear* in
  > the scale. Neither side of this one is.
* **Two pre-existing defects surfaced and fixed, both blockers.**
  `recog.seg_ablation`'s CLI has raised `TypeError` before reaching a
  single measurement since `75db46a` gave
  `check_split_matches_checkpoint` a third argument without updating this
  caller — `seg_ablation.txt` has been unregenerable that whole time. And
  its heuristic arm constructed `HeuristicPlacementAreaExtractor()` with
  no scale, relying on the deleted 0.38 default; it now receives the same
  per-cartridge estimate the segmenter arm gets. That is provably neutral
  to the metric (the fraction is `inside_mask.sum() / inside_mask.size`
  and `inside_mask` is built from pixel quantities only), and the
  regenerated receipt is byte-identical, which demonstrates it.
* **`docs/receipts/seg_eval.txt` millimetre figures are understated** by
  the ratio of true GSD to 0.625 — a median factor of 1.37. The
  feasibility spec already says this (0.949 mm → 1.30 mm). Not
  regenerated here: it is a segmenter metric and its `mm_per_px` is part
  of its definition.
* **`docs/receipts/tau_calibration.txt`** quotes `mm_per_px = 0.6250` for
  the same corpus and inherits the same understatement. The gate it
  calibrates is retired, so no behaviour depends on it.
* **The 23-of-30 figure is now historical.** It is 17 of 30 at the true
  scale. The feasibility spec already asks that it not be quoted as a
  property of the system.

---

## 6. Verification run

* `pytest tests/` — **737 passed** (708 before; 29 added). No test was
  deleted. Four existing tests were edited, each to supply an explicit
  scale where they had been relying on a constructor default that no
  longer exists; one receipt assertion was updated to the new
  provenance-carrying format.
* `python main.py --config configs/demo.yaml` — runs, **10/10 placed**,
  `frames_with_scale: 0` (the cv2 generator writes no sidecar, so the
  0.38 fallback applies, which is correct).
* `python main.py --config configs/demo_seg.yaml --receipt ...` — runs,
  15 of 15 frames calibrated, receipt regenerated.
* `python -m recog.seg_ablation` — runs (after the `TypeError` fix),
  receipt regenerated byte-identical.

The test that fails if the scale reverts to a constant is
`tests/test_planner.py::test_planner_measures_each_frame_at_that_frames_own_scale`:
the same pixels planned at two different frame calibrations must give
different answers in millimetres, with the config fallback set to `None`
so no code path can satisfy it by reaching for a constant.
`test_a_zoomed_frame_does_not_have_the_nominal_scale` pins the underlying
arithmetic — that the generator's nominal framing and a zoomed frame's
true GSD are different numbers.
