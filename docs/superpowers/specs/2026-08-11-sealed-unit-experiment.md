# The sealed-unit experiment — 2026-08-11

Acting on `docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md`,
whose finding was that the procedural segmenter's `bay` transfer gap is
**91.4 % false positives on sealed cartridges** — 136 of 623 sealed CAD
crops, 675 460 px, against the CAD control's 2 of 623.

Base commit `1f4a63d`, 666 tests passing, tree clean. **Every number here is
synthetic-to-synthetic**; real photographs are unobtainable for this project
(`docs/NEXT_STEPS.md`, "The constraint this plan works around"), so nothing
below is evidence about sim-to-real transfer.

Target metric, fixed in advance: **sealed-crop false-positive rate**,
currently 136/623 = 21.8 % against the control's 2/623 = 0.3 %.

---

## Step 1 — investigation, before anything was built

The brief's own leading candidate was "add appearance randomisation". It was
checked first, and it is **dead on arrival**. Findings in the order asked for.

### 1. The procedural and CAD pipelines draw from ONE appearance pool, not two

There is no per-tray-set appearance code path anywhere. `scene.sample_params`
draws `backdrop`, `lighting`, `exposure`, `zoom`, `layout_mode` and
`n_assemblies` from `cfg.param_space` with no knowledge of whether the assets
are CAD or procedural; `scene_generator` is the single entry point for both.
`materials.for_role` indexes `cfg.role_materials` by role name, and
`world.build_procedural_tray` returns exactly the roles CAD returns
(`case`, `case_lid`, `cell`), so a procedural tray draws from the same four
`shell_*` presets a CAD shell does. `configs/synth3d.yaml` carries one
`param_space` / `backdrops` / `lighting` / `materials` / `role_materials`
block and no override.

Measured from the two datasets' own 502 and 500 scene meta files rather than
asserted from the code:

| | procedural anchored | CAD test |
|---|---|---|
| backdrop | metal .199 / concrete .179 / belt .191 / fabric .233 / paper .197 | .198 / .180 / .192 / .232 / .198 |
| lighting (7 rigs) | .149 / .147 / .149 / .153 / .135 / .157 / .108 | .146 / .148 / .150 / .154 / .136 / .158 / .108 |
| exposure p05/p50/p95 | −5.10 / −4.23 / −3.31 | −5.10 / −4.23 / −3.31 |
| zoom p05/p50/p95 | 0.79 / 1.18 / 1.55 | 0.79 / 1.18 / 1.55 |
| `case_lid` preset mix | alu .262 / black .272 / navy .288 / white .178 | .289 / .227 / .287 / .197 |

Identical to sampling noise on every axis. On top of that,
`configs/segmentation_anchored.yaml` already applies heavy photometric
augmentation at train time (`brightness_limit` 0.55, `contrast_limit` 0.50,
`gamma_limit` 45–190, hue/sat/val shifts, motion blur, Gaussian noise,
`p_photometric` 0.85). **There is no appearance-pool difference to close, and
no mechanism by which more of the same would help.** Reported as the null it
is, before spending a render on it.

### 2. What a sealed procedural unit is, versus a sealed CAD shell

`world.build_procedural_tray` builds the lid as
`primitive_cube_add` scaled to the case footprint — **a perfectly planar,
sharp-edged cuboid slab**. The `assembled` variant seals the unit by keeping
that lid, so under this near-orthographic top-down camera a closed procedural
cartridge is a **flat rectangle of one uniform shell colour, with no internal
luminance structure whatsoever**.

The four Anker lids are not that. Measured directly off the glTF
(`trimesh`, all four SKUs):

| | lid height | footprint | top plateau | crown roll-in |
|---|---:|---|---|---|
| 10000 | 11.10 mm | 62.9 × 90.9 | 40.7 × 85.2 | **11.10 mm** on the long edges (35 % of the half-width), 2.84 mm on the short |
| 13000 | 11.10 | 80.7 × 97.0 | 58.5 × 91.3 | 11.10 (28 %), 2.84 |
| 20100 | 11.10 | 62.3 × 167.8 | 40.1 × 162.1 | 11.10 (36 %), 2.84 |
| 26800 | 11.10 | 81.7 × 180.0 | 59.5 × 174.3 | 11.10 (27 %), 2.84 |

The long-edge fillet radius equals the **entire lid height** — the top is a
barrel, not a flat face with a chamfer. **89 % of each lid's upward-facing
polygons have a z-normal below 0.95** (median 0.69, i.e. ~46° from vertical);
the procedural lid's figure is 0 %.

The rendered consequence, measured over each dataset's own sealed crops
(luminance p95−p05 over the unit's own `cartridge` mask, at three erosion
depths so the silhouette edge cannot be what is being measured):

| erosion | procedural sealed | CAD sealed | ratio |
|---|---:|---:|---:|
| 2 px | 0.0350 | 0.2972 | 8.5× |
| 6 px | 0.0323 | 0.2570 | 8.0× |
| 10 px | 0.0318 | 0.1846 | 5.8× |

Median over all sealed crops: **0.0272 (procedural) vs 0.2719 (CAD), 10.0×**.
64.5 % of CAD sealed crops sit above the procedural population's 95th
percentile and 26.8 % above its 99th. Median shell luminance, by contrast, is
essentially the same in both (p05/p50/p95 0.100/0.526/0.875 procedural,
0.130/0.555/0.894 CAD) — **brightness is covered, shading structure is not.**

Visually: a CAD sealed shell reads as a dark falloff band along each long
edge framing a bright central crown — which is exactly what an open tray's
wall framing its bay floor looks like. Open units, by contrast, look the
same in both sets (shared `build_bay_proxy` / `build_pcb` /
`build_obstructions` code), which is why present-only `bay` is already within
0.021 of the CAD ceiling.

**Named candidate cues, in the order the evidence supports them:**
1. **Non-planar top face** (the CAD lid's crown) — the only appearance axis
   measured where the procedural sealed population does not cover the CAD one,
   and it is out by an order of magnitude.
2. Rounded plan-view corners and a visible case/lid seam — same root cause,
   smaller signal.
3. Shell darkness — present but non-monotone (see below), and the diagnosis
   already reported it as null.

### 3. Sealed share, and how diverse the sealed population is

Sealed share is not the confound — the diagnosis already ruled that out and
it reproduces exactly. Grouping every annotation by `unit_id`, the crops
`recog/seg_dataset.py` actually builds are:

| | sealed cartridge | open cartridge | open share |
|---|---:|---:|---:|
| procedural anchored | 614 | 234 | 27.6 % |
| CAD test | 627 | 208 | 25.0 % |
| CAD control (holdout 10000) | 615 | 237 | 27.8 % |

**The diversity is the problem.** Within the procedural sealed population the
lid is the same primitive every time: the only things that vary are its
footprint, its shell preset, and the scene's lighting/backdrop/exposure. Its
own internal-shading spread has p50 0.0272 and p95 0.2011 — i.e. a single
stereotyped appearance, *flat*. This is exactly the case the brief flagged as
"a different fix from randomising backdrops": there is nothing to memorise on
a real shell that the model can generalise from a flat slab.

### The mechanism, measured on the model rather than argued

Re-running `seg_anchored` over the CAD test crops through
`seg_evaluate`'s own pixel path reproduces the diagnosis exactly — **623
sealed crops, 136 hallucinating (21.8 %)** — which is what licenses the rest.
Hallucination rate by quintile of the sealed shell's own internal luminance
gradient:

| quintile of shell `grad_mean` | halluc. rate |
|---|---:|
| q1 [0.0023, 0.0066) | 6.4 % |
| q2 [0.0066, 0.0094) | 14.5 % |
| q3 [0.0094, 0.0132) | 22.4 % |
| q4 [0.0132, 0.0183) | 26.6 % |
| q5 [0.0183, 0.0588) | **39.2 %** |

Monotone, a 6× spread, and it survives controlling for the two correlates the
diagnosis found. Within **every one of the seven lighting rigs** the trend is
monotone across gradient terciles (`overcast_softbox` 0.0/0.0/30.8 %,
`mixed_daylight` 15.2/68.2/78.6 %, `dim_workshop` 13.0/30.8/50.0 %,
`fluorescent_factory` 8.0/15.2/31.2 %, `high_bay_led` 8.0/19.4/37.1 %,
`warm_indoor` 0.0/21.4/25.9 %, `harsh_inspection` 7.1/13.6/16.7 %), and
within 4 of 5 backdrops. Controlling for shell brightness as well
(tercile × tercile, n = 623):

| | grad lo | grad mid | grad hi |
|---|---:|---:|---:|
| luma lo | 16.2 % (99) | 43.1 % (72) | 54.1 % (37) |
| luma mid | 1.8 % (57) | 9.0 % (89) | 32.8 % (61) |
| luma hi | 3.8 % (52) | 8.7 % (46) | 30.9 % (110) |

Gradient is monotone in every luma row. Brightness on its own is **not**
monotone (39.2/20.2/13.6/13.7/22.4 % by quintile), reproducing the
diagnosis's own "no monotone trend, nothing" for shell brightness. The
`paper` backdrop carries a separate effect of its own (flat at ~50–55 % across
all gradient terciles) that this experiment does not explain and does not
claim to.

One more check that could have gone the other way and did not: of the
**606** procedural sealed crops the model trained on, it hallucinates a bay on
exactly **1**, and that one crop's internal spread is 0.3988 — a 15× outlier
for its own population, sitting squarely inside the CAD range.

### Redirection

**The "add appearance randomisation" plan is wrong and was not run.** The
pools are already shared and already identical; the augmentation is already
aggressive. What the procedural set is missing is not appearance variation,
it is a **closed cartridge whose top face is not flat**. The model learned
"featureless flat top ⇒ closed" because in 614 of 614 training examples that
was true, and a real moulded shell violates it.

---

## Step 2 — the experiment

### The one change

**Roll a sampled fillet onto the four TOP edges of the procedural lid.**
`TrayRangeCfg.lid_crown_mm_range` (new, default `(0.0, 0.0)`) →
`bay.sample_tray` draws `lid_crown_mm` → `catalog.build_tray_entry` carries it
→ `world._crown_lid` bevels the lid with a 12-segment, profile-0.5 roll and
shade-smooths only the bevel faces.

Three properties make this a one-variable experiment rather than a
two-variable one, and each was asserted rather than assumed:

1. **It touches the sealed population and nothing else.**
   `config.VARIANTS` keeps `case_lid` in `assembled` and drops it from
   `open_case`, so an open procedural unit's geometry is untouched by
   construction. Bevelling a cube's top edges also leaves the AABB alone
   (face centres do not move), so `_load_template`'s shared re-centre step —
   which measures `group_bbox` over every returned mesh, lid included — sees
   identical numbers.
2. **The crown is drawn LAST in `sample_tray`.** `build_procedural_pool`
   gives each tray its own seeded `Random`, so taking this draw after every
   other leaves the whole preceding stream bit-identical between a `(0, 0)`
   config and a `(0, 12)` one. Verified by rebuilding both pools at the
   render's own seed and comparing entry field by entry field: **0 non-crown
   mismatches across all 502 trays**, and the cell-format split is
   182/169/151, identical to anchored's. Pinned by
   `test_lid_crown_is_drawn_LAST_so_every_other_tray_field_is_identical`.
3. **A zero crown reproduces the old lid exactly** — `_assert_procedural_tray_
   geometry` check 7 requires a plain 6-face cuboid with 0 rolled faces when
   `lid_crown_mm == 0`, so every config written before this field existed
   builds precisely what it always built.

Range `[0.0, 12.0]` mm, clamped in `sample_tray` to
`min(case_half_height_mm, 0.45 × shortest outer side)`. **Uniform from zero
on purpose**: the goal is a sealed population that is *diverse* in top-face
shading, not one that reproduces the Anker value. Realised over the 502-tray
pool: p05 0.66, p25 3.14, p50 6.19, p75 9.22, p95 11.13, max 11.87 mm; 17
trays clamped at their own lid height. The Anker figure (11.10 mm) lies
inside the range — **stated as coverage, not luck**: a favourable result
shows the measured coverage gap was the mechanism, *not* that procedural
training generalises to an arbitrary unseen shell.

### `world.py` is bpy-space, so the assertions are the evidence

`_crown_lid` cannot be reached from pytest. `_assert_procedural_tray_geometry`
gained three checks that run on **every** tray built, not in a test:

- **6.** the crown did not move the lid's AABB (`lid_hi.z == 2 ×
  case_half_height_mm`, alongside the footprint check that already existed);
- **7.** `lid_crown_mm == 0` ⇒ exactly 6 polygons, 0 rolled;
- **8.** `lid_crown_mm > 0` ⇒ the top plateau is inset by exactly
  `lid_crown_mm` on all four sides, and the majority of upward-facing
  polygons are non-planar (`n_z < 0.95`) — the same 89 %-of-faces property
  measured on the four Anker lids.

**Check 8 earned its keep on the smoke render.** Its first version demanded
`4 × segments = 48` rolled faces and fired at 40 of 49: the one or two
segments nearest the plateau are within 0.95 of vertical by construction (a
12-segment quarter-round steps 7.5° at a time), and a face *count* is
independent of the radius anyway, so only a fraction says anything about
whether the roll is real. Restated as a fraction. This is the failure class
the project's standing caution #2 exists for — it would have rendered
plausibly and told nobody.

### What was built and verified

- `configs/synth3d_crown.yaml` (+ JSON sidecar). Loaded through
  `config.load_config` beside `configs/synth3d.yaml` and diffed field by
  field: **exactly one parsed difference**, `tray_anchored.lid_crown_mm_range`
  `(0.0, 0.0)` → `(0.0, 12.0)`. Same check on the JSON sidecar Blender
  actually reads: also exactly one. `tray_wide` untouched.
- `configs/segmentation_anchored_crown.yaml` diffed key by key against
  `configs/segmentation_anchored.yaml`: **exactly three differences**, all
  paths (`coco_path`, `img_dir`, `checkpoint_dir`).
- Smoke render (6 scenes, 1280×720, `--variant assembled`) before committing
  the GPU hour.

### The dataset

`recog/dataset3d_seg_anchored_crown`, 502 scenes at the same seed,
resolution, samples and `--tray-set anchored` as `recog/dataset3d_seg_anchored`.

- **502 / 502 / 502** images / annotations / meta; min image 771 936 bytes
  (max 1 345 968) — no truncation.
- **2802 seg annotations**, and the per-class counts are **identical to
  anchored's**: cartridge 848, battery 896, obstruction 640,
  placement_area 234, electronics_module 184.
- **Disjointness sweep: 13 589 mask pairs checked, 0 overlapping**, worst
  0 px. The identical sweep on `recog/dataset3d_seg_anchored` gives the same
  13 589 pairs and the same 0 — the pairing scheme is all-pairs-within-image
  and the two sets are directly comparable on it.
- Anchor check clean and identical to anchored's Task 14 figures:
  p05 39.9 / p50 79.0 / p95 229.9 px against the matchable 28–407 px band.
- **1489 units, and the unit keys, kinds and boxes are identical to
  anchored's** — 0 kind mismatches, 0 unit-box mismatches. 614 sealed / 234
  open / 641 loose, the same split, so the 848 crops and the 721/127 train/val
  partition are unchanged too.

The change landed where it was supposed to and nowhere else, measured rather
than asserted:

| unit population | n | mean per-pixel difference vs anchored, p50 | p95 |
|---|---:|---:|---:|
| **sealed** | 614 | **5.4428** | 22.2754 |
| open | 234 | 0.0025 | 0.4491 |
| loose cells | 641 | 0.0063 | 0.8735 |

Open and loose-cell crops are unchanged to within Cycles sampling noise and
indirect bounce off neighbouring sealed units in the same frame. And the
sealed **silhouette** is unchanged as well — comparing each sealed unit's
`cartridge` mask pixel for pixel, **99.8 % are exactly identical**, min −1 px,
max 0 px. The crown altered appearance *inside* the mask and nothing about
the shape, which rules out "the model is simply seeing a different outline".

The target statistic moved as intended. Sealed shell luminance p95−p05:

| | p50 | p75 | p95 | fraction > 0.20 |
|---|---:|---:|---:|---:|
| anchored (flat lid) | 0.0275 | 0.0665 | 0.1861 | **4.4 %** |
| **crowned lid** | **0.1794** | 0.3049 | 0.5495 | **45.8 %** |
| CAD test (the target) | 0.2719 | — | — | — |

Note the crowned median (0.179) is *below* the CAD median (0.272): the
uniform-from-zero draw widens the population to **cover** the CAD range
rather than to sit on it, which is what was intended.

### Training

One model, fresh initialisation (`recog/checkpoints/seg_anchored_crown`
created empty, no `--resume`, no fine-tuning), the identical 40-epoch
schedule, `configs/segmentation_anchored_crown.yaml`. Best at epoch 37,
last at 39. Val instance counts **43 / 35 / 29 bay/electronics/obstruction —
identical to anchored's**, confirming the same split partition.

In-distribution best selected mean IoU **0.7273**, against anchored's 0.7322
and 18650-only's 0.7333. **The crowned model is very slightly WORSE on its
own validation split.** That is worth stating first, because it is the shape
a real out-of-distribution result has: nothing here made a better segmenter,
it made one that transfers.

---

## Results, on the same 836 CAD test crops

Receipt: `docs/receipts/seg_eval_anchored_crown_on_cad_test.txt`, generated by
`python -m recog.seg_evaluate --per-sku`, never hand-edited.

| model | bay | battery | electronics | obstruction | cartridge | selected mean | **sealed FP rate** |
|---|---:|---:|---:|---:|---:|---:|---:|
| procedural, anchored (flat lid) | 0.6555 | 0.5593 | 0.7541 | 0.6306 | 0.8088 | 0.6801 | **136/623 = 21.8 %** |
| procedural, 18650-only | 0.6191 | 0.5763 | 0.7534 | 0.6306 | 0.7914 | 0.6677 | 154/623 = 24.7 % |
| **procedural, CROWNED lid** | **0.8755** | **0.6906** | **0.7819** | 0.6360 | **0.9120** | **0.7645** | **16/623 = 2.6 %** |
| CAD control (leave-one-out composite) | 0.9009 | 0.7419 | 0.8530 | 0.6341 | 0.9382 | 0.7960 | 2/623 = 0.3 % |

And the decomposition the pre-registration named as the falsifier:

| model | `bay`, present-only | `battery`, present-only | open-crop recall | open-crop precision | hallucinated px |
|---|---:|---:|---:|---:|---:|
| procedural, anchored | 0.8801 | 0.6924 | 0.9538 | 0.9193 | 675 460 |
| procedural, 18650-only | 0.8839 | 0.6869 | 0.9521 | 0.9251 | 838 185 |
| **procedural, CROWNED** | **0.8856** | **0.7153** | **0.9557** | 0.9235 | **22 559** |
| CAD control composite | 0.9013 | 0.7500 | 0.9447 | 0.9514 | 722 |

Per-SKU `bay` for the crowned model: 0.8430 / 0.8783 / 0.8706 / 0.8890 for
10000 / 13000 / 20100 / 26800, against anchored's 0.6376 / 0.6750 / 0.6344 /
0.6665 and the controls' 0.9005 / 0.8884 / 0.8988 / 0.9098. AnkerPowerCore
10000's `battery` still rests on 14 crops and is still a small-sample estimate.

### Against the pre-registered thresholds, stated as thresholds

- **Primary — sealed FP rate: 21.8 % → 2.6 % (136 → 16 of 623).** Pre-registered
  "the explanation is right" was **< 10 % (≤ 62/623)**; "null" was **19–25 %**.
  **The prediction is met**, by a margin of nearly four times the threshold.
  The remaining 16 sits between the CAD control's 2 and the pre-registered
  target.
- **Falsifier — present-only `bay` must not fall more than 0.02 below 0.8801.**
  It **rose**, 0.8801 → **0.8856**. Open-crop recall rose too (0.9538 →
  0.9557) and open-crop precision rose (0.9193 → 0.9235). **The threshold-shift
  falsifier is not triggered**: the model did not become reluctant to predict
  `bay`, it became able to tell where a bay is not.
- **Pooled `bay`: 0.6555 → 0.8755.** The pre-registration computed 0.880 as
  the arithmetic ceiling for a *complete* close of the hallucination channel
  (1 735 064 / (2 646 919 − 675 460)). The measured 0.8755 sits just under it,
  which is exactly what a 92 %-but-not-100 % close predicts, and far outside
  the 0.0007–0.017 noise band the anchored-vs-wide and format-mix comparisons
  established.

### The mechanism, re-measured on the new model

The correlate that drove the failure has collapsed, and collapsed hardest
exactly where the explanation says it should — in the high-gradient tail:

| quintile of the sealed shell's own luminance gradient | anchored (flat lid) | crowned lid |
|---|---:|---:|
| q1 | 6.4 % | **0.0 %** |
| q2 | 14.5 % | 1.6 % |
| q3 | 22.4 % | 3.2 % |
| q4 | 26.6 % | 4.0 % |
| q5 | **39.2 %** | **4.0 %** |
| all 623 | 21.8 % | 2.6 % |

A 6× monotone gradient dependence is now essentially flat. That is the
signature of a covered distribution, not of a global threshold move — a
threshold move would have lowered all five quintiles proportionally and taken
present-only `bay` down with them.

### Treating the favourable result as suspect

Six checks, run because the result is large and this project has twice been
burned by encouraging-looking ones:

1. **Is it the right checkpoint?** The receipt names
   `seg_anchored_crown/best.pt` and reports best/last 0.7273/0.7268, matching
   this run's own checkpoint metadata (epoch 37 / 39) and matching neither
   anchored's 0.7323/0.7309 nor 18650-only's 0.7333/0.7333.
2. **Did the measurement path drift?** The same read-only harness reproduces
   every published anchored figure to 4 dp — pooled `bay` 0.6555, present-only
   0.8801, 136/623, 675 460 px — and every published CAD-control figure —
   0.9009, 0.9013, 2/623, 722 px. It reproduced them *after* a defect in it
   was found and fixed (see 6).
3. **Is the crowned model just better?** No — it is **worse** in-distribution
   (0.7273 vs 0.7322). The gain is out-of-distribution only.
4. **Did the labels or the shape change?** No. Identical per-class annotation
   counts, identical unit keys/kinds/boxes, and 99.8 % of sealed `cartridge`
   masks pixel-identical.
5. **Why did `cartridge` (+0.103) and `battery` (+0.131) move, when the crown
   touches neither?** Because both are the same hallucination channel. The
   diagnosis measured that 30.7 % of every `bay` pixel the anchored model
   predicts lands on ground-truth `cartridge`; removing 653 000 px of invented
   bay returns those pixels to `cartridge` by construction. `battery`'s
   present-only figure moved far less (0.6924 → 0.7153) than its pooled one
   (0.5593 → 0.6906), which is the same signature. **The genuine internal
   control is `obstruction`: it lives inside open bays, the crown cannot touch
   it, and it did not move (0.6306 → 0.6360, inside the noise band).** A
   result that had lifted `obstruction` too would have been evidence of
   something non-specific and would have needed a different explanation.
6. **A defect in this experiment's own tooling, found and fixed.** The first
   version of the leave-one-out composite read each crop's SKU off *every
   annotation in the image* rather than off the unit, so in multi-unit scenes
   it occasionally selected the wrong control checkpoint. It produced
   `bay` 0.9038 / 4 of 623 — plausible, slightly wrong, and it would have gone
   unnoticed had it not disagreed with the published 0.9009 / 2 of 623. Fixed
   to use `BaySegDataset.sample_assets`, the canonical value
   `seg_evaluate.group_indices_by_asset` itself uses; it then reproduced the
   published figures exactly. **Only the control row was ever affected** — the
   three single-checkpoint rows do not consult the asset at all.

### What this does and does not license

**It does not license "procedural training now transfers".** The crown range
`[0, 12]` mm was chosen *after* measuring the Anker lids, and 11.10 mm lies
inside it. The honest claim is the narrow one: **the measured coverage gap
was the mechanism.** A closed cartridge whose top face is not flat was absent
from the procedural training distribution, the model had therefore learned
"featureless flat top ⇒ closed", and putting that case into the distribution
removed 92 % of the false positives it caused. It is domain randomisation
informed by a measured gap, which is the strictly weaker claim
`docs/NEXT_STEPS.md` Step 3 already describes — never a statement that this
model would hold up on a shell family nobody measured.

**Every number here is synthetic-to-synthetic.** None of it is evidence about
photographs.

Three things also stayed put and are reported as such:

- `obstruction` is still ~0.63 for every model measured, procedural and CAD
  alike — still the harder-test-set property, still not a procedural weakness.
- The `paper` backdrop's separate ~50–55 % effect (flat across gradient
  terciles) was identified in Step 1, is not explained by the crown, and was
  not chased.
- `electronics` (0.7541 → 0.7819 pooled, against the control composite's
  0.8530) and the residual `cartridge` gap (0.9120 vs 0.9382) remain genuine
  appearance gaps with no hallucination component left to explain them. They
  are now the largest honest shortfalls.

### Regression checks

Five-class disjointness **0 overlapping pixels** on the new dataset (13 589
pairs). Suite green at **708 passing** — the baseline moved from 666 to 678
and then to 704 under this work as other agents landed changes; 4 of the 708
are this experiment's own, in `tests/test_bay.py`. `python main.py --config
configs/demo.yaml` runs (10 cycles, 10 placed; it is non-deterministic, so
"still runs" is the criterion). No existing dataset, checkpoint, metric
definition, model architecture or training schedule was changed.

### Reproduction

```
blender -b --python recog/generate3d.py -- --n 502 \
    --out recog/dataset3d_seg_anchored_crown --device GPU \
    --tray-set anchored --config configs/synth3d_crown.yaml --resume
python -m recog.seg_training --config configs/segmentation_anchored_crown.yaml
python -m recog.seg_evaluate \
    --checkpoint recog/checkpoints/seg_anchored_crown/best.pt \
    --config configs/segmentation_cad_test.yaml --per-sku \
    --out docs/receipts/seg_eval_anchored_crown_on_cad_test.txt
```

Datasets and checkpoints are gitignored, as every render in this plan has
been. The sealed-FP / present-only decomposition came from a read-only
harness that imports `recog.seg_evaluate` / `recog.seg_dataset` unmodified;
it is not committed because it adds no capability the repo lacks, and its
anchor figures are the published receipts' own.
