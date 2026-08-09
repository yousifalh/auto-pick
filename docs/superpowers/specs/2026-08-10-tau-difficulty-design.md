# Tau-difficulty design: what would actually make τ calibratable

Design spec — 2026-08-10. Diagnosis and design only; no generator change, no
retrain, no dataset regeneration in this document or the work that produced
it.

This is the fourth of the four specs named in
`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8 ("4 —
Difficulty"), scoped down to the half of it that `docs/NEXT_STEPS.md`
promoted to the top of the project's queue: the τ-targeted subset. It answers
one question precisely — **what scene content would enlarge the disagreement
between `P_direct` and `P_derived` past a cell's footprint, and is that even
possible** — rather than building all four difficulty modes the owner named
and hoping one of them works.

**Labelling convention used throughout:** a claim tagged **[measured]** comes
from a receipt in `docs/receipts/` or a number already published in
`docs/FDR_v3.md` / `docs/NEXT_STEPS.md`. A claim tagged **[code]** is read
directly off a source file, cited by path and line. A claim tagged
**[derived]** is arithmetic this document performs from measured/code inputs
— shown so it can be checked, not asserted. A claim tagged **[judgement]** is
this document's opinion, not a fact any of the above establishes. This
project has been bitten by unlabelled figures before (`docs/NEXT_STEPS.md`,
`docs/FDR_v3.md` §13.3); the labels are there so this document does not add
a fifth instance.

---

## 1. The mechanism

### 1.1 What quantity τ thresholds

`plan/arbitration.py:128-139` (`arbitrate`) computes two masks per cartridge
crop from one segmenter label map:

- `P_direct = label_map == CH_BAY` — the segmenter's own `bay` channel
  (`plan/arbitration.py:94-96`).
- `P_derived = erode(centre_component(label_map), wall_inset_px)` minus
  every pixel the SAME label map classifies as electronics, obstruction or
  battery (`plan/arbitration.py:99-118`).
- `iou = mask_iou(P_direct, P_derived)` — this is **τ's thresholded
  quantity**: intersection over union of the two masks
  (`plan/arbitration.py:121-125,139`).
- `P_safe = P_direct & P_derived` — the placement region actually handed to
  the packer when `iou >= τ`.

`plan/placement_area.py:373-447`
(`SegmentationPlacementAreaExtractor.extract`) is the consumer: if
`iou < self.tau`, it raises `PlacementDisagreement` and the planner skips
the cartridge for one cycle (`plan/placement_area.py:296-303,404-420`); if
both estimates are empty it raises a plain `RuntimeError` ("cartridge may
simply be full") instead, so an empty/empty pair is never miscounted as a
disagreement (`plan/placement_area.py:406-417`). Above τ, `P_safe` becomes
the occupancy grid the FFDH packer places cells into.

### 1.2 What "optimistic error" means operationally

`recog/calibrate_tau.py:178-179` (`collect_records`):

```
optimistic_error = safe & ~gt_bay
admits = admits_a_cell(optimistic_error, cell_w_px, cell_h_px)
```

`safe` is `arbitrate`'s `P_safe` run against the trained checkpoint's
*prediction*; `gt_bay` is the *ground-truth* bay mask for the same crop. The
optimistic error is therefore: **pixels the pipeline is about to hand the
packer as safe, that the ground truth says are not bay** — electronics,
obstruction, battery, cartridge wall, or background. This is the one error
direction that can put a cell where it shouldn't go; the opposite direction
(conservative error, safe-but-marked-unsafe) only costs throughput and is
not what τ is calibrated against (`recog/calibrate_tau.py:1-17`).

### 1.3 What the decision is used for

`recog/calibrate_tau.calibrate` (`recog/calibrate_tau.py:100-139`) sweeps
every observed IoU as a candidate τ, ascending, and returns the **smallest**
τ at which at most `fail_budget` (5%, `recog/calibrate_tau.py:352`) of
*accepted* cartridges (`iou >= τ`) have `admits_a_cell(optimistic_error)`
True. Smallest, because a larger τ only ever rejects more cartridges
(more skipped cycles) — the search is for the most permissive threshold
that still keeps the damage-direction failure rate under budget. The
population is restricted to crops with a non-empty `P_direct`
(`recog/calibrate_tau.py:19-31`, `144-189`): whenever `P_direct` is empty,
`P_safe` is empty regardless of τ, so those crops can never admit a cell and
τ never actually gates them — folding them in would dilute the accepted set
with cases τ never decides.

**Current state [measured], `docs/receipts/tau_calibration.txt`:** τ =
0.5715, 35/35 accepted cartridges, fail_rate 0.0000. Not one of the 35
admitted a cell at any observed IoU down to the sample minimum, so the 5%
budget never bound — τ is simply the lowest IoU the split happened to
contain, not a threshold located by trading safety against throughput
(`tau_calibration.txt:96-104`). The largest optimistic error observed is
1278 px against a 3045 px² cell footprint — **42.0%**, down from **79.4%**
before the tray-wall fix (`tau_calibration.txt:105`; `docs/FDR_v3.md:1833-
1841`; `docs/NEXT_STEPS.md:267-271`).

---

## 2. What actually drives `P_direct` / `P_derived` disagreement

### 2.1 The class-exclusion term is a no-op wherever it matters

`recog/bay_segmenter.py:110` — `pred = logits.argmax(dim=1)...` — confirms
the label map is a **single class per pixel**: one argmax over the six
`SEG_CHANNELS`, mutually exclusive by construction **[code]**. Given that,
look again at `derived_placement`:

```python
out = interior.astype(bool)          # eroded centre-component
for ch in (CH_ELECTRONICS, CH_OBSTRUCTION, CH_BATTERY):
    out &= label_map != ch
```

`P_safe = P_direct & P_derived`. For any pixel where `P_direct` is True,
`label_map[p] == CH_BAY`. Since the label map is single-valued per pixel,
`label_map[p] == CH_BAY` **implies** `label_map[p] != CH_ELECTRONICS` and
`!= CH_OBSTRUCTION` and `!= CH_BATTERY`, automatically, for every such
pixel. The three-class subtraction loop therefore contributes **nothing**
to the content of `P_safe` — it can only ever remove pixels from
`P_derived` where `P_direct` is already False, which the intersection
discards regardless. Algebraically:

```
P_safe = P_direct ∩ P_derived
       = {p : label(p) == BAY} ∩ erode(centre_component(label_map), wall_inset_px)
```

**[derived]**, but not a novel claim — the test suite already states this
exact fact. `tests/test_arbitration.py:87-100`
(`test_p_safe_is_the_intersection_not_the_union`) says outright: *"`derived_
placement` never subtracts `CH_CARTRIDGE`, so wall pixels land in `derived`
but never in `direct`... Without the ring, `derived` degenerates to exactly
`direct`."* That is precisely the collapse shown above, restated from the
CARTRIDGE-ring side of the same fact. `test_iou_falls_when_the_estimates_
disagree` (`tests/test_arbitration.py:128-147`) makes the same point from
the IoU side: without the wall ring, "derived collapses onto direct for
*any* bay/electronics split" and the fixture regresses from `IoU < 0.6` to
`IoU == 1.0`.

**Consequence:** the ONLY thing that can make `P_safe` diverge from
`P_direct` at all is the geometric erosion — stripping a `wall_inset_px`
band (7 px / 4.375 mm at the current 0.625 mm/px framing,
`tau_calibration.txt:8`) off the outer edge of the centre-connected
foreground blob — plus `centre_component`'s connectivity/background
exclusion. The electronics/obstruction/battery channels do real work inside
`derived_placement` on its own (they shrink it on pixels `P_direct` never
claimed), which feeds the **union** term of `mask_iou` and therefore the
**τ gate** — but they cannot add or remove anything from `P_safe`'s
*content*, and therefore cannot add or remove anything from the optimistic
error `admits_a_cell` tests.

### 2.2 What this means for "harder scenes"

Restating §2.1 operationally: the optimistic error is exactly

```
{p : label(p) == BAY, gt(p) != BAY, p survives the wall_inset erosion}
```

i.e. **compact, deep-interior `bay`-class false positives** — the segmenter
predicting "clear floor" over something that truth says is wall, PCB,
obstruction, battery, or background, located more than 7 px inside the
crop's own foreground/background boundary (closer than that, the erosion
removes it before it ever reaches `P_safe`, whether the ground truth agrees
or not). No amount of scene content that does not change the segmenter's
own bay-class false-positive rate, at that location, changes this quantity.
Scene difficulty is therefore a **lever on segmenter accuracy in a
particular failure mode**, not a direct lever on the arbitration arithmetic
— a mode must make the model *itself* misclassify bay-vs-something-else,
in a spatially compact way, away from the rim.

### 2.3 Did the tray-wall fix make `P_derived` independent?

The tray-interior fix (`docs/superpowers/specs/2026-08-08-tray-interior-
design.md`) gave `open_case` cartridges real 3-D tray walls: `cartridge`-
labelled pixels on open units went from documented-zero to **mean 3403 px,
median 3087 px, range 1133–11416 px across 210 open-unit instances**
**[measured]**, `docs/FDR_v3.md:1606-1610`. Per that spec, real geometry was
meant to make the wall ring "the primary mechanism" rather than "a fallback"
for an artificial label-level inset (`2026-08-08-tray-interior-design.md`
§3, consequence 2).

**Measured effect [measured], both from `docs/FDR_v3.md:1815-1866` and
`docs/NEXT_STEPS.md:241-297`:** IoU rose from τ = 0.3180 (pre-fix) to
τ = 0.5715 (post-fix) on the 35-cartridge accepted population, and the
largest optimistic error **shrank** from 79.4% to 42.0% of a cell's area.
Both moves are in the direction of *more* agreement between `P_direct` and
`P_derived`, not less.

**Reading this against §2.1 [derived, this document's own conclusion]:** the
fix did not and structurally could not increase the *information*
independence of `P_derived` from `P_direct`, because that independence was
already capped by the argmax redundancy shown above — real walls changed
*where* the erosion band sits and how realistically it renders (self-
shadowing etc.), not the algebraic relationship between the two masks. What
plausibly *did* change is the segmenter's own geometric accuracy near the
true wall boundary — a more physically correct scene is a better-posed
learning problem, and the model apparently learned it well, producing
*fewer* deep-interior bay-vs-not-bay mistakes, which is exactly what §2.2
says shrinks the optimistic error. So: **no, the fix did not make the two
estimates more independent in the sense the arbitration docstring claims —
it made the segmenter more accurate, which is a real improvement, but one
that moves τ calibration further from resolvable, not closer**, consistent
with `docs/NEXT_STEPS.md`'s own conclusion ("a more geometrically accurate
generator moves further away, not closer"). This document's contribution is
the *mechanism* for why: the redundancy in §2.1 means there was never a
second, independent source of positive information for `P_safe` to gain
from more realistic geometry in the first place — only a more accurate
single estimate, arbitrated against a version of itself with a rim shaved
off.

---

## 3. Ranking the four difficulty modes against the mechanism

The owner's four modes, assessed by whether they can plausibly produce
**compact, deep-interior bay-class false positives** (§2.2) — the only
thing that moves τ.

### 1st — Cluttered bays (highest leverage)

Places foreign content **inside** the bay, i.e. exactly the region where a
misclassification survives the wall-inset erosion by construction (it's
already deep inside the interior, nowhere near the outer rim). If the
segmenter fails to recognise an obstruction/battery/electronics object and
predicts `bay` over it instead, that is *precisely* the optimistic-error
mechanism in §2.2, and it produces a **compact blob** (the object's own
footprint) rather than a scattered or rim-shaped error — the shape
`admits_a_cell` is built to distinguish from noise
(`plan/arbitration.py:142-170`; `tests/test_arbitration.py:194-217`).

This axis already exists (`ObstructionCfg`, `configs/synth3d.yaml:179-191`,
`recog/synth3d/bay.py:410-465`) and is already active in the 502-scene
dataset the 42.0% figure was measured on — `p_none: 0.40`, i.e. 60% of bays
already carry adhesive/foam/tape/label objects. That it has not yet moved
τ past 42.0% is itself informative: the currently-modelled obstruction
kinds are, on average, well enough separated visually from the bay floor
that the segmenter classifies most of them correctly. **[measured]**
supporting evidence: `docs/receipts/seg_eval.txt:14-19` — `obstruction` IoU
0.6579 (24 instances) and `battery` IoU 0.6907 (24 instances) are already
the *weakest* two of the three occupied-content classes (`electronics`
0.8613, `bay` 0.8903), i.e. these are the classes most prone to being
confused with `bay` today, and they are already the ones this axis
stresses. `docs/receipts/git-log`-adjacent finding, `git show a1b2a38`
(the damage-direction investigation): one of the two negative-`delta_cells`
crops was explicitly caused by "six small ground-truth obstructions the
segmenter recall-misses" — direct, already-observed evidence that the
model *does* sometimes fail to recognise obstruction content, just not (yet)
compactly or largely enough to flip `admits_a_cell` in the crops sampled so
far.

**[judgement]**: making this axis actually move τ needs objects chosen for
classification *ambiguity* against the bay floor — texture/colour
similarity, objects near the model's spatial resolution limit, denser
crowding so edges blur together — not simply more of the same distinct
shapes. "More clutter" alone is not obviously sufficient; "harder-to-
classify clutter" is the actual target.

### 2nd — Occlusion and clutter (conditional leverage)

Per `docs/superpowers/specs/2026-08-09-spec3-realism-decisions.md`, this
mode is scoped as bench-level clutter: tools, cables, loose cells *outside*
the cartridge, other cartridges partly in frame. Neither this content nor a
generator hook for it exists yet **[code]** — `grep` across
`recog/synth3d/*.py` and `configs/synth3d.yaml` finds no bench-clutter,
tool, cable, or perspective-camera implementation; spec #3 records these as
still-open decisions, not shipped code.

Mechanistically, this content sits **outside a given cartridge's own crop**
in the common case. `calibrate_tau.collect_records` evaluates per-cartridge
crops (`extract_crop(image, box, ...)`, `recog/calibrate_tau.py:170`), and
`centre_component`'s own docstring (`plan/arbitration.py:42-67`) describes
exactly this scenario — a neighbour's material catching the edge of a
jittered crop — and its resolution: excluded from the centre-connected
blob entirely, contributing nothing to either estimate. So generic bench
clutter has close to **zero** mechanistic path into a specific cartridge's
own `P_direct`/`P_derived` disagreement, unless it is deliberately placed
so it overlaps the **interior of the bay itself** (a cable draped across
the floor, a tool resting partly inside an open tray) — at which point it
is functionally the same lever as "cluttered bays" above, just entering via
a different asset category. As scoped in spec #3 today (bench-level,
outside the cartridge), this mode would do little for τ; redesigned to
place occluders over the visible bay floor, it converges with mode 1.

### 3rd — Lighting extremes (low, diminishing returns)

Already substantially built: 7 lighting presets spanning `harsh_inspection`
through `dim_workshop`, sampled per scene
(`configs/synth3d.yaml:198-199,457-595`), plus a −5.2 to −3.2 stop exposure
range (`configs/synth3d.yaml:219`) — and already active in the 502-scene /
42.0%-error dataset. "More lighting variety" is therefore marginal, not new
territory, on top of an axis already fully in play for the current
measurement.

Mechanistically, lighting stresses the segmenter **diffusely** — it
degrades general perceptual accuracy across the whole frame rather than
targeting the specific deep-interior bay-vs-occupied-content confusion
§2.2 needs. Errors it induces are as likely to land at the rim (eroded
away regardless of ground truth) or to be scattered/thin (failing
`admits_a_cell`'s morphological test even at large total area — exactly the
blob-vs-rim distinction `docs/superpowers/plans/2026-08-06-D-integration-
arbitration.md:348` was written to guard against) as they are to be the
compact interior blob needed. It has a real secondary channel — reducing
contrast between an obstruction/battery object and the bay floor under
harsh or dim illumination could push mode 1's ambiguity higher — but that
is better reached by *pairing* hard lighting with cluttered-bay content
than by varying lighting on its own, which is already saturated.

### 4th — Truncation and framing (lowest; may be actively unhelpful)

Truncation's main effects are (a) reducing or removing the visible bay
region for a partially-framed cartridge, not creating a false-positive bay
region inside a fully visible one, and (b) any confusion at a cut frame
edge occurring in the same near-rim band the wall-inset erosion already
strips regardless of ground truth. `configs/synth3d.yaml:101-125` already
notes 2.71% of boxes are truncated by the existing zoom range, so this axis
is also already partially active without moving τ.

**[judgement]**: pushing this axis harder is more likely to **shrink**
the τ-eligible population (crops with a non-empty predicted `P_direct`,
`recog/calibrate_tau.py:19-31`) than to enlarge the error within it — a
badly truncated cartridge more often predicts no bay at all than a
confidently-wrong one. Not a lever on the mechanism, and a plausible net
negative for calibration sample size.

**Summary ranking:** cluttered bays > occlusion-of-the-bay (a redesigned
subset of "occlusion and clutter") > lighting extremes > truncation and
framing. The gap between rank 1 and rank 4 is not small — ranks 3 and 4 are
axes already present in the dataset that produced 42.0%, so building more
of them without also engineering rank 1/2 content specifically for
classification ambiguity is unlikely to move the number at all.

---

## 4. Quantified target

### 4.1 The morphological bar, precisely

From `recog/calibrate_tau.py:57-58,422-424` and the current framing
(`tau_calibration.txt:5-9`): `mm_per_px = 0.6250`, `CELL_W_MM = 18.3`,
`CELL_H_MM = 65.0`. `cell_w_px = 18.3 / 0.625 = 29.28`,
`cell_h_px = 65.0 / 0.625 = 104.0`. `format_report` uses the continuous
product for its percentage figure: `29.28 × 104.0 = 3045.1 px²`
(`recog/calibrate_tau.py:297`, matches `tau_calibration.txt`'s "3045 px²").

**[derived] — a detail the receipt's percentage does not reflect:**
`admits_a_cell` itself truncates to `int()` before building the structuring
element (`plan/arbitration.py:161-162`): `int(29.28) = 29`,
`int(104.0) = 104`. The actual kernel tried is **29 × 104 = 3016 px²**, not
3045 px² — about 1% smaller, i.e. the real bar is marginally *easier* to
clear than the receipt's percentage implies (worth noting; not worth
re-deriving every percentage in this document over a 1% gap).

**The bar is not an area threshold.** `admits_a_cell` returns True only if
`cv2.erode` by a 29×104 (or 104×29) rectangular structuring element leaves
a non-empty result — i.e., some position exists where a **solid, unbroken
29×104 px rectangle fits entirely inside the optimistic-error mask**. Total
area reaching 3016 px² is **necessary but not sufficient**: the same 3016
px scattered as a one-pixel-wide rim, or as several small disconnected
blobs, still fails (`tests/test_arbitration.py:194-217` demonstrates
exactly this: identical area, rim fails, blob passes).

### 4.2 The distance from today

Current largest optimistic error **[measured]**: 1278 px
(`tau_calibration.txt:105`). Reaching the *necessary* area floor (3016–3045
px²) is roughly a **2.4× growth in total optimistic-error pixels** over
today's observed maximum — and that number is silent on whether the growth
is shaped as a compact blob, which §4.1 says is the actual constraint. A
mode that grows total error 3× while keeping it rim-shaped moves the
percentage figure in a report without ever making `admits_a_cell` fire.
**The generation work's acceptance criterion should be phrased as "produces
a compact miss at least 29×104 px," not "produces more error area."**

### 4.3 A geometric ceiling this document found, per cartridge SKU

`catalog.json` carries `tray_outer_mm` (the whole cartridge footprint,
matching what a segmenter's non-background prediction spans) and
`module_bay_mm` (where the electronics module sits) per asset. Eroding
`tray_outer_mm` by the wall_inset actually applied at inference —
`wall_inset_px = round(4.25 / 0.625) = 7` px = **4.375 mm per side**
(`plan/placement_area.py:355-371`, matching `tau_calibration.txt:8`'s
stated 7 px), then subtracting `module_bay_mm`, gives the largest region
`P_safe` (hence the optimistic error, which is a subset of it) can *ever*
occupy for that SKU — independent of segmenter accuracy, independent of
scene difficulty, a pure geometry bound. **[derived]**, cross-checked
against `cell_union_mm` (which independently confirms cells are placed
spanning exactly `y ∈ [-43.0, 22.0]`, 65.0 mm, on every asset):

| Asset | `tray_outer_mm` (W×H) | eroded, minus module (W × remaining H) | vs. cell 18.3×65 mm | margin |
|---|---|---|---|---|
| AnkerPowerCore10000 (3-cell) | 62.9 × 90.9 | 54.15 × 63.075 | **fails, both orientations** | short by 1.9 mm |
| AnkerPowerCore13000 (4-cell) | 80.7 × 97.0 | 71.95 × 66.125 | fits (tall orientation) | **+1.1 mm — razor-thin** |
| AnkerPowerCore20100 (6-cell) | 62.3 × 167.8 | 53.55 × 134.525 | fits comfortably | +69.5 mm |
| AnkerPowerCore26800 (8-cell) | 81.7 × 180.0 | 72.95 × 140.625 | fits comfortably | +75.6 mm |

**Reading:** on the smallest cartridge (`AnkerPowerCore10000`), the eroded,
module-subtracted placement region is **smaller than one cell's own
footprint in every orientation** — `admits_a_cell` is geometrically
**unsatisfiable** for this SKU under the current default wall_inset,
regardless of how bad the segmenter's predictions get, because `P_safe`
(and everything derived from it, including the optimistic error) can never
exceed this eroded region. The root cause is visible in `cell_union_mm`
itself: this asset's cells span `y ∈ [-43.0, 22.0]`, exactly 65.0 mm,
against an interior that (per `interior_mm`) also starts at `y = -43.0` —
the CAD design gives this cartridge **zero slack** in that axis even before
any erosion; the wall_inset (needed because no SKU identifier crosses the
Recognition→Planning boundary, `plan/placement_area.py:330-344`) then makes
it strictly negative. `AnkerPowerCore13000` is technically satisfiable but
by a margin (1.1 mm) thin enough that rendering/annotation discretization
could plausibly erase it in either direction. Only `AnkerPowerCore20100`
and `AnkerPowerCore26800` have real headroom (≥69 mm in the constrained
axis).

**[judgement]**: this table is arithmetic from CAD-measured bounding boxes,
not a rendered-mask measurement — it assumes erosion of an axis-aligned
rectangle by a square structuring element shrinks each side by exactly the
inset (true for an ideal rectangle, but the rendered/predicted foreground
blob is not a perfect rectangle at pixel resolution). **Recommend verifying
by rendering `AnkerPowerCore10000` and `AnkerPowerCore13000` open-case
scenes and directly measuring the eroded ground-truth mask before relying
on this table**, but the arithmetic is corroborated two independent ways
(`interior_mm`-vs-`tray_outer_mm` cross-check landing within ~1.5 mm of each
other, and `cell_union_mm` independently confirming the zero-slack claim
for the 10000 asset), so it should be treated as a strong hypothesis, not
noise. **Practical implication for §5: difficulty scenes aimed at moving τ
should be weighted toward `AnkerPowerCore20100` and `AnkerPowerCore26800`
crops** — the two SKUs where an `admits_a_cell=True` record is geometrically
possible at all — rather than spread evenly across all four assets by the
existing weight distribution (`recog/synth3d/catalog.py`/asset sampling
currently has no per-SKU weighting for this purpose). The calibration
report should also be extended to break its per-cartridge table down by
asset/SKU (derivable from scene metadata already recorded per crop), so
this hypothesis can be checked directly against real records instead of
staying an inference from CAD numbers.

---

## 5. What to generate

In priority order, following §3's ranking and §4.3's SKU finding:

1. **Cluttered bays, engineered for classification ambiguity, biased toward
   `AnkerPowerCore20100`/`26800` crops.** Extend `ObstructionCfg`
   (`recog/synth3d/config.py:185-201`) with harder variants of the existing
   adhesive/foam/tape/label kinds: material/colour draws closer to the
   sampled bay-floor material range (rather than the current independent
   draw), smaller feature sizes near the segmenter's effective resolution
   (boundary displacement is already measured at 0.95–1.18 mm per class,
   `docs/receipts/seg_eval.txt:26-35` — objects near that scale are where
   confusion should concentrate), and higher object density per bay so
   individual footprints crowd and their boundaries blur. Do **not** simply
   raise `n_adhesive`/`n_foam`/etc. ranges without also addressing
   ambiguity — §3 already found the existing (more-of-the-same) obstruction
   population insufficient at n=502 scenes.
2. **Occluders placed over the interior bay floor**, not generic bench
   clutter — a redesigned, narrow slice of spec #3's occlusion content
   (e.g. a cable or tool resting partly inside an open tray) scoped
   specifically to overlap the visible placement region. This is
   mechanistically indistinguishable from mode 1 once scoped this way; it
   is listed separately only because it uses different asset geometry
   (elongated/irregular shapes vs. the current adhesive/foam/tape/label
   primitives) and may reach different segmenter failure modes.
3. **Lighting extremes paired with modes 1/2**, not as an independent axis —
   e.g. render a subset of cluttered-bay scenes under `dim_workshop` or
   `harsh_inspection` specifically to test whether reduced contrast pushes
   mode 1's near-miss classifications over the line. Do not invest in new
   lighting presets on their own; the existing 7 are not the bottleneck
   (§3).
4. **Truncation/framing: no new generation work recommended for the
   τ-calibration goal specifically.** It remains in scope for spec #4's
   other stated purpose (detector/framing robustness generally), but should
   not be justified by, or measured against, the τ target in this document.

None of this requires spec #3's still-open perspective-camera or full
bench-clutter decisions (`2026-08-09-spec3-realism-decisions.md` "Still
open") — modes 1 and 2 as scoped above are buildable on the current
orthographic camera and current `Config` schema (`recog/synth3d/config.py`),
extending `ObstructionCfg` and the existing obstruction-placement code path
in `recog/synth3d/bay.py`.

---

## 6. What must not regress

Restated from `2026-08-08-tray-interior-design.md` §6, because this work
touches the same generator and the same invariants apply:

- **Five-class pixel disjointness at 0 overlapping pixels** —
  `placement_area` against `battery`/`obstruction`/`electronics_module`.
  Currently verified at 0 across 3280 mask pairs on the full regenerated
  dataset (`docs/FDR_v3.md:1601-1604`). Any new obstruction/occluder content
  must be seated using the same occlusion-ordering discipline the existing
  `sample_obstructions`/`obstruction_world_poses`/`obstruction_forbidden_
  mask` pipeline already uses (`recog/synth3d/bay.py:410-568`), not a new
  ad-hoc placement path.
- **The `assembled` variant** — geometry, labels, and VOC boxes unchanged.
  New difficulty content is scoped to `open_case` (and `cells_only` where
  relevant), never to sealed units.
- **`unit_id` grouping and unit-scoped VOC boxes** (Plan B Task 9) — any new
  in-bay object must carry correct grouping metadata so per-unit box merging
  stays correct.
- **The torch-free demo**, `python main.py --config configs/demo.yaml` —
  runs on `HeuristicPlacementAreaExtractor`/`recog/synth_dataset.py`'s flat
  green rectangles, not the 3-D generator this document concerns. No change
  proposed here touches that path; call it out explicitly in the
  implementation plan so it stays untouched.
- **`plan/arbitration.py`'s channel-order contract** — `CH_BACKGROUND`
  through `CH_BATTERY` must keep matching `recog.seg_dataset.SEG_CHANNELS`,
  pinned by `tests/test_arbitration.py:173-183`. No difficulty content
  should add a new segmentation class without updating both sides and that
  test.
- **The 621-test suite green at the commit this work builds on
  (`0fb9d7e`).** This document changes no code, so nothing here should move
  that number; the implementation that follows it must keep it green,
  particularly `tests/test_arbitration.py` and `tests/test_synth3d.py`.

---

## 7. How success is measured

1. Regenerate the dataset with the new difficulty content mixed in (weight
   TBD in the implementation plan; §4.3 argues for biasing toward
   `AnkerPowerCore20100`/`26800`), retrain, and re-run
   `python -m recog.calibrate_tau` exactly as today.
2. **Primary metric:** the largest optimistic error as a percentage of one
   cell's footprint (`tau_calibration.txt`'s own headline number). Success
   is `admits_a_cell` firing at least once in the accepted population —
   the point at which the fail-budget sweep has something to actually
   trade against, for the first time. A percentage increase alone (e.g.
   42.0% → 70%) without a single `admits_a_cell=True` record is **not**
   success against this document's target — §4.1 explains why.
3. **Break the per-cartridge table down by asset/SKU** (not currently done,
   `tau_calibration.txt:16-54` lists scenes without SKU attribution) — this
   is needed both to check §4.3's hypothesis and to confirm any newly-large
   errors are concentrated on the two SKUs where they are geometrically
   possible, rather than being an artefact spread evenly (which would be a
   sign the change affected general segmenter accuracy rather than the
   targeted failure mode).
4. Re-run the §6 regression checks (disjointness sweep, `assembled`
   contact-sheet check by eye, `unit_id` grouping check, demo smoke test,
   full pytest suite) before treating the run as valid — same discipline as
   `2026-08-09-tray-interior.md`'s acceptance checklist.
5. If `admits_a_cell` fires: re-run the fail-budget sweep for real. At that
   point the sample-size caveat (`tau_calibration.txt:284-292`) becomes the
   live concern instead of the vacuous-test problem — 35–40 cartridges is
   still small enough that a single admitting cartridge is close to the 5%
   budget on its own, so growing the *validation* population (more scenes,
   not just harder ones) becomes relevant again, but only once error size
   is no longer the binding constraint.

---

## 8. Honest negative findings

**τ may not be calibratable even after this work, and that would be a
finding, not a failure of this document.** Two structural reasons, both
argued above rather than asserted:

- §4.3: two of the four cataloged cartridge SKUs (`10000`, and marginally
  `13000`) cannot ever produce an `admits_a_cell=True` record under the
  current default wall_inset, no matter how hard the scene is, because the
  geometric ceiling on `P_safe` for those assets sits at or below one
  cell's own footprint. If the real deployed fleet skews toward those
  SKUs, harder scenes on them specifically are wasted effort by
  construction — the fix there is not a harder scene but a smaller (or
  per-SKU) wall_inset, which is a production-code change out of this
  document's scope.
- §3: two of the owner's four named modes (lighting extremes, truncation
  and framing) are already present in the dataset that produced the
  current 42.0% figure and are diffuse rather than targeted at the
  mechanism in §2.2. Building more of them is not obviously going to move
  the number — this document recommends against investing further there
  for the τ goal specifically (§5.3, §5.4), which is a smaller-scope
  recommendation than the owner's original four-mode plan.

**What would make this a dead end rather than a hard problem:** if,
after building the cluttered-bay content in §5.1–§5.2 targeted at
`20100`/`26800` with genuinely ambiguous obstruction material, the
optimistic error *still* does not approach a cell's footprint, that is
evidence the segmenter is simply accurate enough — at the resolution this
architecture operates at (0.95–1.18 mm boundary displacement,
`seg_eval.txt:26-35`) — that no synthetic scene within the generator's
current asset/material vocabulary reproduces the failure mode τ needs to
observe. At that point the honest conclusion (matching `docs/NEXT_STEPS.md`
§4's own framing) would be that τ is not calibratable from synthetic data
at all, and the arbitration mechanism should either ship at a fixed,
un-calibrated τ (as it already does — `plan/placement_area.py:357` defaults
to 0.85, and the calibrated value has never been wired in,
`docs/NEXT_STEPS.md:299-301`) or be redesigned so its threshold is not the
thing standing between the current geometry-only wall_inset margin and
safety. This document does not attempt that redesign; it is out of scope
for "docs and analysis only."

---

## 9. Summary table of confidence

| Claim | Label | Where |
|---|---|---|
| τ = 0.5715, 35/35 accepted, largest error 42.0% (was 79.4%) | measured | `tau_calibration.txt`, `FDR_v3.md` |
| `label_map` is single-channel argmax | code | `recog/bay_segmenter.py:110` |
| Class-exclusion in `derived_placement` is redundant on the intersection | derived (corroborated by existing test docstrings) | §2.1 |
| Tray-wall fix did not create real independence, only better accuracy | derived (this document's interpretation of measured numbers) | §2.3 |
| Ranking of the four difficulty modes | judgement, argued from §2.2's mechanism | §3 |
| 2.4× area growth needed as a floor; compact-blob requirement | derived | §4.1–4.2 |
| Per-SKU geometric ceiling (10000/13000 unsatisfiable) | derived, not yet rendered-and-verified | §4.3 |
| τ may be uncalibratable from synthetic data even after this work | judgement | §8 |
