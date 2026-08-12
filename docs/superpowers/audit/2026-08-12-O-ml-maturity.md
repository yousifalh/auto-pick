# Audit O — ML-engineering maturity, read as a hiring manager

**Date** 2026-08-12 · **HEAD** `f1989e9` · **Scope** read-only. Nothing was
modified, staged or committed. No training, no dataset regeneration, no
Blender, no GPU work. Three other audits ran concurrently; this one measured
nothing timing-sensitive, so contention does not affect any figure below.

**Question.** The repository is about to be published as a portfolio piece
aimed at machine-learning engineer roles. What would an experienced ML
engineer or hiring manager look for here and **not find**? Not "what would
make it better research" — what would make it read as the work of someone
ready to own models in production.

Every claim below is labelled **DERIVED** (established by reading the tree)
or **MEASURED** (established by running a command). No claim rests on
inference from prose alone.

---

## Verdict

The engineering is not the problem. The **packaging** is.

This repository is assembled as a *research record* — one document per
investigation, one receipt per number, a 4,000-line report, twelve audits.
Judged as that, it is better than most published work. But an ML-engineering
reader does not read a research record; they look for four artefacts in
conventional places, and then they leave. Those artefacts are a model card, a
results table, a data card and a failure-mode summary. **All four are absent
as artefacts and present as material.** Every ingredient exists, committed and
verified. Nothing assembles them.

That is the whole finding, and it is a good problem to have: the gap is
roughly six hours of assembly, not six weeks of work. The risk is that a
reader who spends fifteen minutes here sees a very long README about a robot
and never learns that eight models were trained under controlled conditions
and scored against a common held-out set.

The second finding is narrower and sharper. **This project's central thesis
is that systems fail silently and that instrumenting the silence is the
job** — and its own run-summary instrumentation cannot see the model. All 23
counters in `main.py` are system events. Not one is a model-quality
distribution, and two suitable signals are already computed and discarded.
That is the one inconsistency in the repository that an ML reader will
notice and be able to name.

---

## 1. Model card — absent (highest signal per hour)

**MEASURED.** `grep -rniE "model card|data card|datasheet|intended use"` over
every `.md` and `.py` in the tree returns **zero hits**.

**DERIVED.** Eight-plus trained models are discussed as results:

| model | its config | its held-out receipt | its checkpoint dir |
|---|---|---|---|
| detector (Faster R-CNN R34-FPN) | `configs/recognition.yaml` | `docs/receipts/frcnn_map.txt`, `frcnn_latency.txt`, `detector_bench.txt` | `recog/checkpoints/` |
| segmenter, `anchored` | `configs/segmentation_anchored.yaml` | `seg_eval_anchored_on_cad_test.txt` | `recog/checkpoints/seg_anchored/` |
| segmenter, `wide` | `configs/segmentation_wide.yaml` | `seg_eval_wide_on_cad_test.txt` | `recog/checkpoints/seg_wide/` |
| segmenter, `anchored_18650` | `configs/segmentation_anchored_18650.yaml` | `seg_eval_anchored_18650_on_cad_test.txt` | `recog/checkpoints/seg_anchored_18650/` |
| segmenter, `anchored_crown` | `configs/segmentation_anchored_crown.yaml` | `seg_eval_anchored_crown_on_cad_test.txt` | `recog/checkpoints/seg_anchored_crown/` |
| 4 × leave-one-SKU-out CAD control | `configs/segmentation_cad_control_holdout_*.yaml` | `seg_eval_cad_control_*_on_cad_test.txt` | `recog/checkpoints/seg_cad_control_*/` |
| shipping segmenter | `configs/segmentation.yaml` | `seg_eval.txt`, `seg_eval_anchored_on_anchored_val.txt` | `recog/checkpoints/seg/` |

**MEASURED.** All ten `segmentation*.yaml` and `configs/recognition.yaml` are
tracked at HEAD (`git ls-files configs` → 22 files). So the hyperparameters
of every model are committed, in a file that also argues for each value.

**MEASURED.** `recog/seg_training.py` writes into every `best.pt`/`last.pt`:
`epoch`, `ious`, `selected_mean_iou`, `select_on`, `val_instance_counts`,
`coco_path`, `split_seed`, `seed`, `seeding`. The comment beside it — "a
checkpoint that does not record its seed is a checkpoint nobody can
re-derive" — is the right instinct and it is already implemented.

**So the model card is not missing information; it is missing a page.** A
stranger who wants "what was this trained on, with what hyperparameters, how
well does it do, on what, and where does it fail" must currently cross-join a
4,000-line FDR, ten YAML files and eleven receipts. Nobody does that before
deciding whether to keep reading.

*Recommendation, ~2–3 h.* One `docs/MODEL_CARD.md`: one row per model
(architecture, training data with its scene/crop count, hyperparameters by
reference to the committed config, selected-mean-IoU on the 836-crop CAD test
set **with the per-class n beside it**, checkpoint-selection metric and its
noise floor), then a shared section for the things that are true of all of
them — intended use, the synthetic-to-synthetic scope clause, the "these
checkpoints predate seeding and cannot be recovered" clause, and the failure
modes from §5 below. Generate the table from the receipts rather than typing
it, so it is a receipt and not a paragraph; that is this project's own
standard and it should apply to its most-read page.

## 2. Consolidated results table — partial, and split across the report

**DERIVED.** FDR §13.1.1 does carry comparison tables, and they are good.
They are also incomplete as a cross-model view:

* The main table (FDR_v3.md ~L2588) has three rows: `anchored` 0.6801,
  `wide` 0.6794, and **the four CAD controls collapsed into a range**
  ("0.7989–0.8091"). Per-fold results exist in four receipts and appear
  nowhere as four rows.
* `anchored_crown` (0.7645) is in a **different table ~140 lines later**.
* **MEASURED.** `anchored_18650`'s selected mean IoU of **0.6677** appears in
  `docs/receipts/seg_eval_anchored_18650_on_cad_test.txt` and in no table in
  the FDR or the README. It is quoted only as "+0.017 of the 0.224
  available" in the null-result prose. A reader cannot see where that model
  sits against the other seven.
* Nothing anywhere puts the detector, the eight segmenters, the τ
  independence study, the FFDH ablation, the clearance-margin sweep and the
  packing benchmark on one page. They are six separate receipts and four
  separate specs.

*Recommendation, ~1 h, and it folds into #1.* A generator script — read the
eleven `seg_eval_*.txt`, emit one markdown table — costs an hour and makes
the table regenerable. This is the cheapest item on the list because the
parsing target is a format this project already controls.

## 3. Data provenance — strong in code, invisible to a reader

This is the item where the repository is furthest ahead of a typical
portfolio project and gets the least credit for it.

**DERIVED, what exists.** `recog/generate3d.py` writes a `manifest.json` per
dataset carrying the **full generator config**, the seed, the class list and
the class→id map. `recog/calibration.py` *prefers* that manifest over the
authored default when resolving scale, and raises rather than falling back
silently. Checkpoints record `coco_path` and `split_seed`. `seg_evaluate` has
a split guard that checks the checkpoint's recorded split against the config
it is being scored under. FDR §13.1.1 records an MD5 disjointness check over
**4,536 renders across nine datasets, zero shared images in 36 pairings**.
That is real data-lineage discipline.

**MEASURED, the gap.** `.gitignore` excludes `recog/dataset3d_seg*/`
wholesale, and `git ls-files | grep manifest` returns **nothing**. Not one
manifest is tracked. A reader who clones this repository can see **no
dataset description of any kind** — no scene count, no crop count, no class
balance, no generator parameters — except as prose inside a config comment.
The nine datasets cost ~8 GPU-hours to regenerate and are, to a cloner,
entirely hypothetical.

**DERIVED, second gap.** `coco_path` is a **path, not an identity**. Two
different renders can occupy the same path. Nothing in the checkpoint pins
the *content* of the dataset it was trained on.

*Recommendation, ~1 h.* Copy the nine `manifest.json` files (~32 KB each,
**MEASURED**) to `docs/datasets/<name>.manifest.json` and write a one-page
data card over them: per dataset, scenes / crops / per-class instance counts
/ seed / asset pool / what it is for, plus the disjointness result and a
pointer to `docs/superpowers/blender-dataset-known-issues.md`. Separately,
write a hash of the annotation file into the checkpoint alongside `coco_path`
— one line, and it converts a path into an identity.

## 4. Deployment — under-sold, and the right answer is mostly "document it"

**DERIVED, what exists and is genuinely inference engineering.** The
`recog/bay_segmenter.py` docstring records a precision × crop-size ablation
against the 50 ms budget (one crop 12.6 ms; eight looped 101 ms; eight
batched fp32 59.6 ms; eight batched fp16 at 256² 18.5 ms) and states that
**only the last configuration fits**. Crop size 256 is justified as at-or-
above native crop resolution rather than as a round number. There is a CPU
fallback when CUDA is absent, `torch.no_grad`, and `weights_only=True` on
load. And the module boundary itself is a latency decision: segmentation runs
**once per frame, batched, in Recognition** because a per-cartridge forward
pass cannot fit inside Planning's 8 ms budget. That is a shipping engineer's
argument, and it is currently buried in a docstring and one README paragraph.

**MEASURED, what is absent.** `grep -rniE "onnx|torchscript|torch\.jit|
quantiz|tensorrt|triton"` over the shipping tree returns **zero** hits. The
only occurrence anywhere is `docs/FDR.md` L399 — the **superseded** first
revision — which rejects TensorFlow Serving in favour of "a single
`torch.jit.script`-compiled checkpoint". That intent was never carried into
FDR_v3 and was never implemented. A reader who greps for it finds a dangling
promise in a document marked superseded.

**DERIVED, two real inference defects, both cheap.**

1. **No warmup on the deployed path.** `latency_table` in
   `recog/seg_evaluate.py` discards **3 warmup iterations** before timing.
   `BaySegmenter.segment_batch` has none, and `main.py` calls it cold. So the
   published 16.6 ms figure is a steady-state number and the first frame of a
   real run pays CUDA context + cuDNN autotune on top of it. This is exactly
   the class of gap this project catches elsewhere, and it is ~3 lines.
2. **Unbounded batch.** `segment_batch` allocates one
   `(len(crops), 3, 256, 256)` tensor for *every* crop in the frame with no
   cap. At 8 cartridges this is correct and measured; the bound is a property
   of the corpus, not of the code.

*Recommendation, ~1 h, and mostly prose.* **Do not build an ONNX export.**
Write the paragraph that says why there is none: nothing is deployed, there
is one in-process consumer, there is no second runtime or target device, the
latency budget is already met on the only hardware available, and no
checkpoint is published — so an export would be an artefact nobody could
verify. A project that explains its absent export reads better than one with
a broken one, and this project has earned the right to that argument. Fix the
warmup and cap the batch, because those are correctness, not ceremony.

## 5. Failure modes — exhaustively known, consolidated nowhere

**DERIVED.** The project knows more about how it fails than most production
systems do. It is spread across at least seven documents:

| failure mode | where it currently lives |
|---|---|
| optimistic bay boundary: 0.949 mm displacement, +51.5 mm²/crop, puts cells on wall | README headline §, `seg_eval.txt`, `2026-08-11-placement-safety.md` |
| 2 residual unsafe placements, worst 8.3 % / 5.2 % of footprint | README, `2026-08-11-placement-safety.md` §2.3 |
| sealed-unit hallucination: 21.8 % → 2.6 %, mechanism + dose-response | README, `2026-08-11-sealed-unit-experiment.md`, PORTFOLIO |
| `AnkerPowerCore10000` unplannable at any accuracy (65.0 vs 65.0 mm) | README, FDR §3, §13.1 table |
| heuristic extractor: zero placeable area on 7 of 20 real cartridges | README "Two placement-area extractors" |
| segmenter predicts no `bay` on `synth_dataset.py` frames | README, and a `RuntimeError` in `main.py` |
| own-val figures optimistic (73.2 % of val crops share a frame with train) | FDR §13.1.1 third scope statement |
| checkpoint selection noise-limited (0.0013 between `best` and `last`) | inside the receipts, as a note |
| detector `inference_min_size` 500 vs 800: mAP@0.75 0.404 → 0.023 | a comment in `configs/recognition.yaml` |

*Recommendation, ~1.5 h, as a section of the model card.* Five to nine rows:
what fails, how it presents, how you would detect it, what it costs. The
material is written; it needs one table and a link. The last two rows above
are the ones a reader is most likely to miss and most likely to be impressed
by, because both are self-criticism found by the project's own tooling.

## 6. Monitoring — the signals exist and are thrown away

**MEASURED.** `main.py`'s run summary carries 23 counters:
`bad_detector_boxes`, `batteries_detected`, `cartridge_masks`,
`cartridges_detected`, `cross_cartridge_conflicts`, `cycles`, `empty_queue`,
`frames_with_scale`, `pick_failed`, `place_failed`, `placed`,
`placement_areas`, `placement_disagreements`, `queue_poses`,
`released_reservations`, `reprojected_placed_batteries`,
`rescale_dropped_placed_batteries`, `rescaled_area_drops`,
`unreachable_batteries`, `unreachable_cartridges`,
`unreachable_place_targets`, `unrepresentable_placed_batteries`,
`untracked_confirmations`.

They are excellent, and the comments beside them are better than the counters
— several exist specifically because the failure they guard is invisible from
every other number. **But every one of them is a system event.** A model that
quietly got worse would move none of them until it moved them all at once.

**MEASURED, two signals already computed and discarded.**

* `PlacementArea.consistency_iou` is computed per cartridge and
  `plan/placement_area.py` L551 says in terms that it is kept "as
  observability and is not acted on" — and then it is never aggregated,
  never logged in the summary, never written to the receipt.
* The detector's per-detection `score` (`recog/inference.py` L143) is used to
  threshold and then dropped. No mean, no minimum, no distribution.

*Recommendation, ~1 h.* Add four fields to the run summary: mean/min
`consistency_iou`, mean/p05 detection confidence, detections per frame, and
predicted-`bay` pixel fraction per cartridge. Then state the expected range
of each from the held-out corpus, so an operator has something to compare
against. That is the difference between "the model is degrading" and "the
system is erroring", and this repository is one twenty-line diff away from
being able to tell them apart. Given the project's thesis, this is the item
whose absence is most *quotable*, even though #1 is worth more hours.

---

## What is genuinely strong, and a reader may miss it

Listed in the order I would foreground them.

1. **`configs/recognition.yaml` is an outstanding artefact and is invisible
   from the README.** The anchor-scale choice is an 8-row ablation over
   **2,434 boxes from 300 freshly generated scenes at two seeds**, scored on
   best-centred IoU *after* torchvision's own input transform; an exhaustive
   search over **84,156 four-tuples** found a higher minimum and is
   **recorded and rejected**, with the reasoning (it trades 0.07 at the worst
   box for 3.7 % more of the corpus under 0.6, and the RPN's
   `allow_low_quality_matches` makes the bulk the thing that trains). The
   0.504 floor is then explained as *structural* — a factor-2 anchor gap has
   a trough at √2·s — rather than as a tuning failure. `batch_size: 2`
   explains a fragmentation OOM. `inference_min_size: 500` is measured on
   real photographs. A half-inert BN-freeze knob was **deleted** rather than
   repaired. Most candidates cannot explain one hyperparameter from memory;
   this file explains ten, in writing, with numbers. It deserves a line in
   the README that says so.
2. **The receipts are not logs — they carry their own caveats inline.**
   `seg_eval_*` states the n behind every figure, warns that background IoU
   is structural rather than a comparable failure, re-measures whether the
   `cartridge`/`bay` crop populations overlap *this run* rather than
   assuming, notes that checkpoint selection is noise-limited at 0.0013, and
   records the mm/px distribution the millimetres were converted at. This is
   model-card-grade honesty one level below where a reader looks for it.
3. **Seeding taken to the honest end.** `recog/seeding.py` plus
   `docs/receipts/seed_reproducibility.txt`: seeding the RNGs was **not
   enough** (two same-seed runs diverged by up to 0.04 selected IoU),
   kernels had to be pinned, `strict` refuses to train this model at all
   because `nll_loss2d_forward` has no deterministic CUDA implementation, and
   the resulting claim is scoped to one machine and toolchain. This is the
   single most interview-ready thing in the repository and it is currently a
   paragraph in the middle of a long README.
4. **Checkpoint selection on `[bay, electronics, obstruction]` only**, with
   the reason stated in the config and the code: including the big easy
   classes lets a model mask a failure on the three the placement mask is
   built from. Val instance counts are saved into the checkpoint so the
   metric's n travels with the weights.
5. **The oracle / ceiling framing** — building a ground-truth upper bound,
   discovering further perception work was worth zero net cells, stopping,
   and then catching that the two sides had been measured at different
   commits and revising the headline *downward*. Knowing when to stop, and
   correcting yourself in public, are both rarer than accuracy.
6. **The `--per-sku` evaluator and its split guard**, which verifies that the
   checkpoint's recorded split matches the config being scored against —
   this is the mechanism that would catch the most embarrassing possible
   error in this study, and it exists.

## What I would explicitly not add, and why

* **ONNX / TorchScript / TensorRT export, quantisation, a serving container,
  Docker/K8s manifests.** Nothing is deployed, there is one in-process
  consumer, and there is no target hardware. Any of these would be an
  artefact nobody ran against anything. Document the absence instead (§4).
* **A model registry (MLflow / W&B), a feature store, DVC or LakeFS.** Eight
  models with committed configs, a manifest per dataset and metadata inside
  each checkpoint already provide what a registry would at this scale. Adding
  a tool with one user is the opposite of the judgement this project
  demonstrates everywhere else.
* **CI-gated retraining or a drift-detection service.** Training costs ~8
  GPU-hours on hardware the CI does not have, and the README already explains
  why a from-scratch reproduction returns a sample from a distribution.
* **A "reproduce the published checkpoints" script.** It cannot work — the
  published checkpoints predate seeding. The existing explanation of why is
  worth more than a script that would mislead.
* **Rewriting the FDR as a model card.** Keep both. The report is the
  evidence; the card is the index into it.
* **More tests.** The project has 1,210 and has independently established
  that its recurring defect class is tests written from the same
  understanding that produced the bug. Test count is not the missing signal
  for this audience.

## The single change I would make first

**Write `docs/MODEL_CARD.md`, generated from the eleven `seg_eval_*` receipts
plus the ten committed configs plus the checkpoint metadata, and link it from
the README above the architecture diagram.**

It closes gaps 1, 2 and 5 in one artefact; it is the item whose absence is
most conventionally noticed by this specific audience; every input to it is
already committed and verified; and it is the only recommendation here that
costs hours rather than a paragraph. Everything else on this list is either a
paragraph (§4), a file copy (§3) or a twenty-line diff (§6).
