# Audit N — Objective closure: what it would take to reach a defensible Pass

**Date:** 2026-08-12 · **HEAD:** `f1989e9` · **Suite:** 1 211 collected
**Base:** `2026-08-12-J-claim-verification.md` and
`docs/superpowers/specs/2026-08-12-fdr-claim-corrections.md`.
**Read-only.** Nothing was staged, committed or edited. Four figures below were
produced by executing code at HEAD; every one is labelled with how it was
obtained. Scratch scripts live in the session scratchpad, not in the tree.

---

## Headline

**Two of the five open objectives can be closed today for almost nothing, one
cannot be closed at all, and one is worth closing only if the project intends to
keep quoting the number it currently cannot support.**

- **O5** is a 15-line test. I wrote and ran it. **It passes, bit-identically.**
  There is exactly one trap, and it is not what anyone would guess.
- **O1** needs no GPU time. A committed receipt at HEAD already contains a
  measurement that clears the 0.90 bar on the shipped detector. The report cites
  a superseded April checkpoint instead.
- **O6** should carry a third figure — **93 %** over the scope that is actually
  defensible — and drop the middle one it currently leads with.
- **O3**'s published distribution is not reproducible *and does not reproduce*.
  I measured the planner cycle at HEAD: **7.96 ms median cold, 5.73 ms warm**,
  against a published 3.0 ms median and a claimed "under 2 ms" steady state.
- **O2 has no defensible Pass under any reading the document supports.** Not the
  absolute reading, not the median reading, not the "cartridge corners" reading.
  Detail in §1, because the reason is not the one the corrections doc gives.

---

## Ranking

| Objective | Honest verdict today | What would close it | Effort | Recommendation |
|---|---|---|---|---|
| **O5** determinism | Half-supported — ordering asserted, fixed-input/fixed-output unobserved | One test, two fresh `Planner` instances, `assert q1 == q2`. **Written and run below: passes, bit-identical, stable across `PYTHONHASHSEED`** | **~15 lines, <1 h** | **Close it.** Best effort-to-value item in the project |
| **O1** mAP ≥ 0.90 | Partial at 0.874 — but that is the *April* checkpoint on the *April* corpus | Re-cite to `docs/receipts/detector_bench.txt` arm 3: **mAP@0.50 = 0.9053**, shipped `best.pt`, 150-image val, operating thresholds. Already committed and reproducible | **~0 h compute; a citation and a scope sentence** | **Close it, in-domain only.** Do **not** retrain |
| **O6** coverage | Pass at 89 % over a scope that is a stale April snapshot | Re-scope to `main.py`'s transitive import closure. Measured at HEAD: **93 %** (19 modules, 1 845 stmts). Full config scope is **67 %** | **One `coverage report` + a scope sentence, ~1 h** | **Close it, and retire the 18-module figure** |
| **O3** ≤ 8 ms | Threshold Pass on two committed tests; distribution unreproducible | Commit a cycle benchmark (~200 lines, fixtures already exist). **It will not reproduce 3.0/5.0/13.0 — measured 7.96 ms median cold today** | **~0.5 day, and the number gets worse** | **Withdraw the distribution, keep the threshold.** Build the bench only if the distribution is to be quoted |
| **O2** ≤ 2 px | **Fail** — and fails under every reading, including in mm | Nothing reaches a Pass. A mm-space re-measurement (~20 lines) makes the Fail *precise* rather than closing it | **~2 h to state it exactly; unbounded to fix** | **Document as unmet.** Adding the mm columns is worth it; chasing a Pass is not |

---

## 1. O2 — the interpretation question

### 1.1 What the document actually said, across three drafts

| Source | Wording |
|---|---|
| `docs/FDR.md` §1.2 (v1 prose) | "centroid localisation error **on cartridge corners** shall not exceed **5 px** (≈ 2 mm at the specified camera calibration)" |
| `docs/FDR.md` §3 table (v1, same document) | "Centroid error **≤ 2 px** (≈ 0.8 mm)" |
| `docs/FDR_v2.md` §1.2 | "centroid localisation error **on cartridge corners** shall not exceed **5 px** (≈ 2 mm)" |
| `docs/FDR_v2.md` §3 table | "Centroid error **≤ 2 px**" |
| `docs/FDR_v2.md` Appendix matrix | "Centroid error ≤ 2 px **on detected cells**" |
| `docs/FDR_v3.md` §1.2 and §3 | "≤ 2 px (≈ 0.8 mm at 0.38 mm/px)" — the 5 px prose silently dropped |

So the requirement was **never consistent within a single draft**: v1 and v2 each
carry 5 px in prose and 2 px in their own criteria table, over two different
measurands (*corners*, an L∞/edge quantity, versus *centroid*). v3 harmonised
downward to 2 px without recording that it had done so.

The derivation is identical in all three and yields a **third** number:

> an 18650 has a 9 mm radius, the vacuum gripper has a 6 mm grasp radius, and at
> a 0.38 mm/px calibration the recogniser is allowed one third of the 3 mm
> end-to-end budget.

One third of 3 mm is 1 mm; at 0.38 mm/px that is **2.63 px**. Neither 2 px
(0.76 mm) nor 5 px (1.90 mm) is what the stated derivation produces.

### 1.2 What it was intended to mean

**Per-instance, absolute — not a quantile.** The evidence for this is the
derivation, not the wording, and it is one-directional:

The 3 mm is a *tolerance stack-up*. A 6 mm-radius cup on a 9 mm-radius cell has
3 mm of lateral slop before the cup overhangs the cell edge and the grasp
degrades. That is a **geometric limit on each individual pick** — a given grasp
either lands inside it or it does not; there is no such thing as a grasp that
succeeds at the median. Allocating "one third of the budget" to the recogniser is
worst-case allocation, which is the default in tolerance analysis unless the
document declares a statistical (RSS) allocation at a stated confidence. It does
not. Nothing in any of the three drafts, in the derivation, in
`configs/recognition.yaml`'s `evaluation.centroid_error_target_px: 2.0`, or in
`recog/evaluate.py::centroid_error_px` mentions a median, a p95, or a yield.

**So the reading the document meant is the absolute one, and FDR v3's current
verdict — "Fail as an absolute bound" — is the reading the document meant.**

I want to be explicit that I did not pick this because it fails. I checked
whether the median reading could be justified from anything in the corpus and it
cannot: there is no text anywhere that attaches a statistic to the number, and
the only derivation present points the other way.

### 1.3 The median reading fails too — in the units the requirement was derived in

This is the part the corrections doc does not have, and it is decisive.

O2's 2 px is only meaningful at the 0.38 mm/px calibration it was derived at —
it is a *physical* allowance of **0.76 mm** wearing pixel clothing. The 1.13 px
median was measured on `recog/dataset3d`, whose Blender camera samples
`camera.ortho_scale` **per scene**. Measured over all 1 000 scene sidecars
(`recog/dataset3d/meta/scene_*.json`, `ortho_scale × 1000 / width`):

```
mm/px   min 0.488   p05 0.519   median 0.771   p95 1.026   max 1.086
```

The corpus has no single calibration, and its median scale is **2.0× coarser**
than the 0.38 mm/px O2 was written against. Consequences:

- O2's 0.76 mm allowance is worth **0.99 px** at the median `dataset3d` scene —
  not 2 px.
- The shipped detector's 1.13 px median is therefore ≈ **0.87 mm** at the median
  scene, **outside** the 0.76 mm the requirement allocates. *(First-order:
  median error × median scale. The exact figure needs a per-pair join of error
  to frame scale, which `scripts/detector_bench.py` does not currently do — see
  §1.5.)*
- Arm 2's 4.88 px on `recog/dataset` at the FDR's declared 0.38 mm/px is
  **1.85 mm**, 2.4× the allowance.

**The reading that appears to pass in pixels does not survive being restated in
millimetres, which is the only form in which the requirement's own derivation
makes sense.** That removes the last candidate for a defensible Pass.

### 1.4 Honest wording for each reading

If the project wants to state O2 under a named interpretation, these are the
three wordings that are true. All three are Fails; they differ in how much they
concede.

1. **Absolute (what the document meant).** *"O2 is not met. The criterion states
   an absolute bound of 2 px and no configuration satisfies it: the shipped
   detector exceeds 2 px on 24 % of matched detections (p95 4.37 px), and the
   detector §10 reports exceeds it on 81 % (median 4.88 px). The figures are
   conditional on detection, so every missed instance is excluded and the result
   is the generous one."*

2. **Physical, at the median quantile (the most favourable defensible reading).**
   *"O2 derives a 0.76 mm perception allowance from a 3 mm gripper stack-up. On
   the production corpus, whose per-scene scale spans 0.49–1.09 mm/px (median
   0.771), the shipped detector's 1.13 px median is ≈ 0.87 mm — outside the
   allowance at the median and roughly 2.7 mm at p95, which consumes the whole
   end-to-end budget. O2 is not met at any quantile once stated in millimetres."*

3. **Corners (v1/v2's prose measurand).** *"Read on cartridge corners as §1.2 of
   FDR v1 and v2 state it, the governing metric is edge error, not centroid: the
   shipped detector gives 2.64 px median / 7.94 px p95 (L∞), and O2 fails at the
   median as well as the tail."*

**Recommendation: state reading 1 as the verdict and reading 2 as the
engineering consequence, and add one sentence recording that the requirement was
inconsistent across drafts (5 px prose / 2 px table / 2.63 px derivation) and
carried no quantile — a requirements finding worth more marks than a Pass would
have been.** This is what the FDR already half-says; §1.1–1.3 above give it
evidence instead of an assertion.

### 1.5 If any work is done on O2, do this and only this

Add a millimetre column to `scripts/detector_bench.py`. `recog/calibration.py`
already exposes `frame_mm_per_px_for_image()` (line 158) which reads the
per-scene sidecar; joining it to each matched pair and reporting the error
distribution in mm alongside the px one is roughly **20 lines**. It converts an
imprecise Fail into an exact one and kills the "but the median passes" objection
permanently. It does not produce a Pass. **~2 h; worth doing; do not expect it
to change the verdict.**

---

## 2. O2's second question — why §10 reports a detector that scores 4.88 px

**Short answer: because §10 is the April chapter. Arm 2 is not the ablation
baseline — it is §10's *winner*. The ablation's loser no longer exists on disk.**

`docs/receipts/train_eval.txt` (dated 2026-04-20) describes the April run: the
100-image cv2-composited `recog/dataset`, 85/15 split, ResNet-34+FPN **from
scratch** (no COCO pretrain — no network in the sandbox), **15 epochs, batch 1,
CPU-only, ~31 min**. Two anchor configurations were trained for §10.7's ablation:
custom k-means (mAP 0.764, `frcnn_map.txt`) and torchvision defaults (mAP 0.874,
`frcnn_map_default.txt`). **The default-anchor arm is the better of the two and
is what §10.1 headlines.** So the 4.88 px is not a deliberately weak baseline —
it is the best detector §10 knows about.

Since then the project trained a different detector on a different corpus, and
§10 was never rewritten; those results live in §13. The gap is real, and it
decomposes into four compounding causes, only some of which are model quality:

| | §10's detector (arm 2) | Shipped (arm 3) |
|---|---|---|
| corpus | `recog/dataset`, 100 cv2 composites, 15 val images / 134 GT | `recog/dataset3d`, 1 000 Blender scenes, 150 val images / 1 205 GT |
| training | 15 epochs, bs 1, CPU, from scratch | 35 epochs, bs 2, GPU |
| **inference resize** | **min 320 / max 512** | **min 500 / max 900** |
| score threshold | 0.05 (full PR curve) | 0.70 (operating point) |
| mAP@0.50 | 0.8736 | 0.9053 |
| centroid median | 4.88 px | 1.13 px |

**The resize row is the mechanically important one for a localisation metric.**
Both corpora are natively 1280×720. torchvision's transform scales by
`min(min_size/720, max_size/1280)`:

- arm 2: `min(320/720, 512/1280) = 0.400` → the network sees **512×288**
- arm 3: `min(500/720, 900/1280) = 0.694` → the network sees **889×500**

Boxes are mapped back to 1280×720, so **one network-space pixel of regression
error becomes 2.50 px in arm 2 and 1.44 px in arm 3**. A factor of **1.74** in
the reported centroid error is pure evaluation configuration, present before any
question of model quality. The remaining ~2.5× is corpus, schedule and
initialisation.

**Does the project have a better detector than the one it reports? Yes,
substantially** — 0.9053 vs 0.8736 mAP@0.50 at a *stricter* score threshold,
0.9488 precision / 0.9544 recall (arm 3) against a detector whose precision the
receipt explicitly says must not be quoted, and 4.3× better centroid
localisation. That is worth knowing and worth saying in §10.

Two caveats, both load-bearing:

1. **Arm 3's val split is in-domain synthetic** — a held-out 150 of the same
   1 000-scene Blender corpus, same generator, same asset catalogue. It
   certifies in-domain performance and nothing else.
2. **The April custom-anchor arm is no longer reproducible.**
   `docs/receipts/train_eval.txt` and `frcnn_map.txt` both name
   `recog/checkpoints/best.pt` as the epoch-11 custom-anchor checkpoint scoring
   0.7643. The file at that path today has an mtime of **2026-08-06** and scores
   0.9053 on Blender data — it is the shipped model, written over the April one.
   `default_anchors_best.pt` (mtime 2026-04-29) survives, so §10.7's *winning*
   arm is reproducible and its losing arm is not. (Checkpoints are gitignored —
   this is inferred from mtime plus score, not from git.)

---

## 3. O1 — the cheapest honest route is a citation, not a rerun

**Current verdict:** Partial, 0.874 vs 0.90 — 0.026 short.

**A committed, receipted Pass already exists at HEAD and the report does not cite
it.** `docs/receipts/detector_bench.txt` arm 3, generated by the committed
`scripts/detector_bench.py`:

```
mAP@0.50 0.9053   AP_battery@0.50 0.9046   AP_cartridge@0.50 0.9061
shipped best.pt, configs/recognition.yaml, recog/dataset3d val (150 frames),
confidence 0.70, NMS IoU 0.40, inference resize 500/900
```

That is **0.9053 ≥ 0.90**, on the production configuration, at the *operating*
confidence threshold rather than the 0.05 threshold used to inflate a PR curve —
i.e. the conservative way to measure it. Cost to adopt: **one row of §10.5, one
citation, and a scope sentence. Zero GPU-hours.**

Three things must be said with it or the Pass is not honest:

- **It is in-domain.** Held-out split of the same Blender corpus. The original
  O1 wording ("under varying lighting conditions", FDR v1 §1.2) is arguably
  satisfied — `synth3d.yaml` samples lighting preset, colour temperature,
  intensity and exposure per scene — but "varying lighting" and "real imagery"
  are not the same claim.
- **The out-of-domain figure is 0.8647**, from `recog/eval_real.py` (committed)
  over **six** scorable photographs. Six photographs certify nothing in either
  direction and the FDR should say so rather than treat 0.8647 as a Fail with
  the authority of a measurement.
- **Do not quote 0.9998.** `docs/superpowers/specs/2026-08-12-fix-detector.md`
  reports mAP@0.50 = 0.9998 / mAP@0.75 = 0.9081 for the shipped checkpoint on the
  same 150-image split, from a stored detection list scored ad hoc. It has no
  committed generator — quoting it would repeat the exact defect (`bench_cycles.py`,
  `pr_curves.py`) the audit campaign is correcting. The 0.9053 arm has one.

**Do not retrain.** The cost is not only GPU-hours. Training is now seeded and
BatchNorm behaviour changed at `dd36329`, so a rerun produces a **different
network**, which invalidates `detector_bench.txt` (all three arms), the real-photo
0.8647, and every §10/§13 figure keyed to the current checkpoint — to chase
0.026 on a metric the shipped detector already clears on its own corpus. The
honest framing is a scope statement, not a better model.

**Recommendation: restate O1 as Pass in-domain (0.9053) with the real-photo
0.8647 (n = 6) reported alongside as not meeting the bar and not adequately
sampled.** That is defensible, costs nothing, and is more informative than either
"Partial — 0.87" or a bare Pass.

---

## 4. O3 — rebuildable, but the published distribution will not survive it

**What it would take.** `scripts/forbidden_bench.py` (370 lines) and
`scripts/detector_bench.py` (439 lines) are the template: argparse, direct import
of the target, `time.perf_counter()` sampling, receipt to `docs/receipts/*.txt`,
refusal to write a partial receipt. The planner entry point is
`plan/planner.py:337  def cycle(self, snapshot: Snapshot, image_rgb) -> List[PickPlacePose]`,
and **every fixture it needs already exists** in `tests/test_planner.py`:
`_make_planner()` (line 50), `_synth_image()` (line 30),
`_snapshot_with_cart_and_batteries()` (line 99). No dataset, no checkpoint, no
torch for the heuristic arm. **~200 lines, roughly half a day.**

**Can the published 3.0 ms median be reproduced? No — and it does not reproduce.**
Measured at HEAD on this machine, the repo's own fixture (800×600, one cartridge,
eight batteries — so per-cartridge equals per-cycle), 100 cycles each:

```
cold  (new cartridge, extractor runs) : mean 8.29  median 7.96  p95  9.88  max 12.52 ms
warm  (twin cached)                   : mean 5.84  median 5.73  p95  6.50  max  8.53 ms
min-of-5 cold 7.91 ms · min-of-50 warm 5.63 ms   (so this is not machine noise)
```

Against §10.4's published **mean 5.0 / median 3.0 / p95 13.0** and its claim that
"in steady state … the planner's FFDH-only path runs in **under 2 ms**":

- the cold median is **2.7× the published median** and sits essentially *on* the
  8 ms budget;
- the warm path is **~3× the claimed "under 2 ms"**;
- the published p95 of 13.0 ms is, ironically, *pessimistic* relative to today's
  9.88 ms — the distribution moved in both directions, which is exactly why it
  cannot be repaired by adjustment and must be re-measured or withdrawn.

Caveat stated plainly: this exercises `HeuristicPlacementAreaExtractor`, the
torch-free demo path, which is what §10.4's arm was. The **shipping** path uses
`SegmentationPlacementAreaExtractor`, receipted at 2.0–2.2 ms per cartridge in
§13.2.1. A rebuilt benchmark should carry both arms and say which is which; the
current text conflates them.

**On audit K's 2.04 ms.** That figure is the *packer alone* under the
`_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)` guard in
`plan/placement_area.py::reject_if_not_one_cartridge_floor`, which raises
`BadDetectorBox` before the occupancy grid is built. Audit K's own header records
that its measurement scripts (`bench_pack.py`, `bench_corpus.py`,
`bench_planner.py`, `bench_edges.py`) **live outside the tree in a session
scratchpad** — so 2.04 ms has no committed generator either. Three of O3's four
supporting figures (13.0 ms p95, "under 2 ms", 2.04 ms) are currently
unreproducible; only the two passing tests are not.

**Recommendation: withdraw the distribution, keep the threshold.** O3's
*threshold* is genuinely supported by
`tests/test_planner.py::test_segmentation_extract_arithmetic_stays_under_the_o3_budget`
(2.83 ms measured) and
`tests/test_packing_ceiling.py::test_stays_inside_the_o3_latency_budget`
(1.56 ms measured), both asserting `< 8.0` and both passing, plus §13.2.1.
Build the benchmark **only if** the report intends to keep quoting a
distribution — and if it does, budget for the verdict getting worse, because the
cold-start path now runs at the budget rather than at 38 % of it. That is a
finding worth having; it is not a finding that makes the report look better.

---

## 5. O5 — cheapest item in the audit, and I have already run it

**Confirmed cheap.** Executed at HEAD:

```
A  two FRESH planners, same snapshot+image : len 8/8   equal=True   bit-identical=True
B  ONE planner instance, cycled twice      : len 8/8   equal=False
C  five fresh repeats                      : all equal=True
PYTHONHASHSEED=1 vs 9999, separate processes: identical output
```

"Bit-identical" is `float.hex()` on every `pick.x_mm`, `pick.y_mm`, `place.x_mm`,
`place.y_mm`, not `==` on rounded values. `PickPlacePose` and `WorkspacePoint`
are `@dataclass(frozen=True)` (`common/types.py:283`, `:298`), so
`assert q1 == q2` on the returned lists works directly — no serialisation needed.

**The one trap, and it is not any of the three you named.** Construction B — the
obvious way to write the test — **fails**, and it would look like a determinism
bug. It is not. The only field that differs is **`battery_detection_id`**: a
monotonic per-instance counter that returns `0…7` on the first cycle and `8…15`
on the second. *Every geometric field is bit-identical across the two calls* —
same `grid_row`/`grid_col`, same pick mm, same place mm. The test must therefore
construct **two fresh `Planner` instances**, which is also the semantically
correct thing: `Planner` owns a persistent `Scene` tracker, so a second `cycle`
on a live instance is a different input by design.

**Nothing else on that path is nondeterministic.** Verified:

- **No RNG anywhere on the planner path.** Zero hits for `random.` / `np.random` /
  `shuffle` / `sample(` / `id(` / `hash(` across `plan/`, `common/` and `recog/`
  excluding `synth3d/` and `tests/`.
- **The mock server's unseeded drop is not reachable.**
  `execution/mock_kuka_server.py:223` (`random.random() < self.drop_prob`) and
  `:237` are in `_RobotState.pick_and_place`, downstream of `Planner.cycle` and
  only reached through `main.py`'s execution loop. A planner-only test never
  touches it. (It *is* genuinely unseeded and should stay logged as an
  end-to-end reproducibility gap — it just is not O5's problem.)
- **Dict/set ordering does not leak.** `plan/scene.py:268` and `:271` build sets
  for error messages only; `:852` uses one for membership. Sorts at
  `plan/planner.py:361` (unique `cartridge.id` key) and `:367` (stable sort over
  deterministic packer output) cannot tie ambiguously. Confirmed empirically by
  the `PYTHONHASHSEED` result above.
- **No timestamps enter the output.** `Snapshot.timestamp_ns` has zero uses in
  `plan/`.
- One nuisance: `HeuristicPlacementAreaExtractor` emits a `RuntimeWarning`.
  `tests/test_planner.py` already carries the `pytestmark` filter for it.

**Effort: ~15 lines in `tests/test_planner.py`, under an hour including the
docstring that explains why two instances rather than one.** It converts a
headline objective from half-supported to observed, and the comment about the
`battery_detection_id` counter is worth as much as the assertion.

**Recommendation: do it first.**

---

## 6. O6 — both figures should stand, but the wrong one is in the middle

**Measured at HEAD (`f1989e9`, 1 211 tests), three scopes:**

| Scope | Modules | Stmts | Branch cover |
|---|---:|---:|---:|
| Everything `[tool.coverage.run] source` resolves to | 49+ | 7 208 | **67 %** |
| O6's cited 18-module list (the 2026-04-20 receipt's) | 18 | 1 784 | **91 %** |
| **`main.py`'s transitive import closure** | **19** | **1 845** | **93 %** |

(The 89 %/65 % in the FDR were measured at `82cff22` / 1 032 tests; both have
risen. The runtime scope excludes `recog/model.py`, which the coverage config
omits because it needs torch.)

**The 18-module list is not a scope statement — it is a stale snapshot**, and it
is wrong in both directions. Computed from the AST import graph of `main.py`:

- **On the runtime path but outside the 89/91 % denominator:**
  `common/packing.py` (the packer O3 is certified on), `plan/arbitration.py`,
  `recog/bay_segmenter.py` (the shipping segmentation path §13 certifies),
  `recog/calibration.py`, `recog/model.py`.
- **In the 18 but never loaded by `main.py`:** `recog/augmentation.py`,
  `recog/dataset.py`, `recog/evaluate.py` — training and evaluation tooling.

So the headline figure currently excludes the packer and the segmenter — the two
components §6.3.1 and §13 spend the most pages certifying — while including three
modules that never run in production. **O6 says "over the production code"; the
18-module list is demonstrably not that set.**

**Recommendation: publish 93 % (runtime closure) as O6's figure and 67 % (all
modules the coverage config resolves to) as the disclosure, with the scope of
each named in one sentence. Retire the 18-module figure** to a dated
"as measured 2026-04-20" note. This is *stronger* than the current position, not
weaker: it raises the headline from 89 % to 93 %, replaces an unprincipled
denominator with a mechanically derivable one, and the 67 % disclosure — below
the 70 % bar — stays, honestly, because Blender-only renderers and CLI entry
points cannot be unit-tested outside Blender. Both figures should stand; it is
the middle one that should go. **~1 h.**

---

## 7. What I would actually do

In order, by effort-to-value:

1. **O5 — write the test.** 15 lines, <1 h, already verified to pass. Closes a
   headline objective properly.
2. **O1 — re-cite to `detector_bench.txt` arm 3.** No compute. Restate as Pass
   in-domain at 0.9053 with the six-photograph 0.8647 alongside. Do not retrain.
3. **O6 — re-scope and re-measure.** ~1 h. Headline improves and becomes
   defensible at the same time.
4. **O2 — add the mm columns to `detector_bench.py`, then document as unmet.**
   ~2 h. No reading passes; §1.3 shows even the flattering one fails once the
   requirement is stated in the units its own derivation uses. The requirements
   finding (5 px / 2 px / 2.63 px across three drafts, no quantile anywhere)
   is the deliverable here, not a Pass.
5. **O3 — withdraw the distribution; build the bench only if it will be
   quoted.** ~0.5 day, and the honest result is worse than the published one.
   The threshold survives on committed tests either way.

After 1–3 the report reads: **O1 Pass (in-domain, stated), O3 Pass on threshold
with the distribution withdrawn, O5 Pass observed, O6 Pass at 93 %, O2 Fail with
a precise reason, O4 Not tested.** That is four of six met, two unmet and
explained — reached without a single GPU-hour and without softening one number.

The brief's instinct is right and worth stating as a finding in its own right:
**every one of these five objectives was weakened by a stale citation rather than
by the system underperforming.** O1 and O2 both cite an April checkpoint the
project has superseded; O3 cites a script that never existed; O6 cites a module
list that predates the two subsystems it most needs to cover; O5 cites a test
that asserts half of what the criterion says. The engineering is in better shape
than the traceability matrix claims — in four of five cases the fix is to point
the citation at what the project already built and measured.
