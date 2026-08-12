# FDR claim corrections — 2026-08-12

**Base:** `docs/superpowers/audit/2026-08-12-J-claim-verification.md` (16 claims:
5 false, 5 stale, 6 unsubstantiated), corroborated on two items by
`2026-08-12-K-complexity.md`.
**HEAD when this started:** `39429a4` (1 074 tests collected).
**HEAD when it finished:** `dd36329` — two concurrent commits landed mid-task
(`47c86d9` cartridge tracking, `dd36329` audit-G detector fixes, 1 128 tests).
`docs/receipts/detector_bench.txt` was regenerated at `dd36329` and came back
**byte-identical**, so every F1/O2 figure below is valid at both commits.
`dd36329`'s deletion of `training.frozen_bn_epochs` is folded in — see the
`frozen_bn_epochs` entry under "The rest".
**Files changed:** `docs/FDR_v3.md`, `docs/FDR_v2.md`, `docs/NEXT_STEPS.md`,
`README.md`, `docs/CV_BULLETS.md`, `docs/PORTFOLIO.md`, plus two new files —
`scripts/detector_bench.py` and `docs/receipts/detector_bench.txt`.

Every figure below was re-measured or recomputed here, not copied from the audit.
Where my number disagreed with the brief I say so and give mine.

---

## New tooling, and why it was necessary

`scripts/detector_bench.py` → `docs/receipts/detector_bench.txt`.

Two of the three headline corrections (F1 and O2) needed a measurement that did
not exist. Writing them as prose corrections citing an uncommitted scratch
script would have repeated the exact defect being corrected — O3's
`bench_cycles.py` and O1.a's `pr_curves.py` are in the report *because* their
generators were never committed. So the generator is committed.

It runs three arms and reports, per class, TP/FP/precision/recall at IoU ≥ 0.5
and the centroid and edge-error distributions over matched pairs:

1. `HeuristicDetector` over all 100 frames of `recog/dataset` — §10.6's population.
2. `default_anchors_best.pt` over that corpus's 15-image val split at the
   320 × 512 resize — the detector §10.1 reports.
3. Shipped `best.pt` under `configs/recognition.yaml` over `recog/dataset3d`'s
   150-image val split — production.

Matching is VOC's rule, deliberately the same one `recog.evaluate.per_class_ap`
uses for the mAP the report publishes: highest-IoU ground truth first, then
availability. **Arm 2 reproduces `docs/receipts/frcnn_map_default.txt` exactly
at IoU 0.5 (0.8736 / 0.9053 / 0.8419) and to three decimals at IoU 0.75
(0.5833 vs 0.5831, CPU-vs-GPU float drift), and arm 1 reproduces §10.1's
0.3971 / 0.4466 / 0.3475 and `train_eval.txt`'s heuristic mAP@0.75 of 0.387.**
That is what certifies the harness rather than the harness certifying itself.

One non-obvious thing cost most of the time and is recorded so nobody repeats
it: `build_fasterrcnn` no longer sets `min_size`/`max_size`, so a naive reload
of the April checkpoint runs at torchvision's 800/1333 default and scores
**mAP 0.072**, not 0.874. `docs/receipts/train_eval.txt`'s "input resize: min
320 x max 512" is the missing piece; set it and the receipt reproduces to four
decimals. Datasets and checkpoints are not tracked by git, so the script skips
loudly and refuses to write a partial receipt.

---

## F1 — "100 % precision / zero false positives" (§1, §10.6, §13.3)

**Measured, all 100 frames, IoU ≥ 0.5, VOC matching:**

| Class | GT | TP | FP | Precision | Recall |
|-------|---:|---:|---:|----------:|-------:|
| battery | 630 | 291 | 9 | 0.9700 | 0.4619 |
| cartridge | 192 | 90 | 41 | 0.6870 | 0.4688 |
| **All** | **822** | **381** | **50** | **0.8840** | **0.4635** |

431 detections, 50 of them false positives. Agrees with the brief to four
decimal places.

**Corrected at all three sites**, each with the original text struck rather than
replaced, so a reader who remembers "100 % precision" sees what superseded it:

* **§1 finding (2)** — withdrawn in place, replaced with the measured figures
  and the note that the measurement had never been taken.
* **§10.6** — the claim struck inside the sentence; the `(LOW_IOU) | 50` row
  removed from the taxonomy table and **reconciled explicitly**: the row *was*
  the false-positive count, relabelled and moved out of the denominator. A
  precision table replaces it. The prose that described LOW_IOU as "fired
  correctly" is rewritten to say the mechanism is blob-merging and that a
  mega-box overlapping real foreground is still noise to a planner. §10.1's
  "~95–100 % on batteries and ~75–90 % on cartridges" is named as the statement
  that already contradicted the headline from inside the document.
* **§13.3** — the "what surprised" paragraph struck and replaced. The
  replacement is a better reflection than the original: the surprise is that a
  *derived* quantity was read as a measured one.

Propagated: nothing in `README.md`, `docs/PORTFOLIO.md` or `docs/CV_BULLETS.md`
repeated the precision claim, so a **prohibition** was added to `CV_BULLETS.md`'s
"Notes for adapting these" instead — the brief is right that this is the sort
of thing that reaches a CV, and the CV file is where it would have entered.
FDR v2's two copies are struck in place.

---

## F3 — "the executor bounds vacuum dwell to 5 s by construction" (§11.2)

Verified myself: `grep -ri dwell` over `*.py`, `*.yaml`, `*.src`, `*.xml`
returns **nothing**. `PickAndPlace` in `krl_prog/routines.src` turns the vacuum
on at step 2 and off at step 5 with `WAIT SEC 0.05` / `WAIT SEC 0.03` settle
waits either side and no timer between. `_emergency_stop`'s docstring says the
opposite case. The nearest 5 s is `kuka.command_timeout_ms`.

Withdrawn in place, and one thing the audit did not note is added: the
obligation the paragraph opens with — "never retain a cell under vacuum beyond
the specified dwell" — **has no specified dwell either.** No config key, no
constant, no comment names one. The claim fails twice.

**Assessment of implementability (asked for, not implemented).** Dwell fails
*differently* from the deleted `safety_max_velocity_mm_s`, and the difference is
the substantive part of the correction:

* The velocity cap was impossible because the 16-byte frame carries **no
  velocity field** — the host could not express the quantity at all.
* Dwell is not blocked there. The host owns `VACUUM_OFF` (0x03) and `ESTOP`
  (0x06), so it has an actuator.
* It is blocked one level up, by **who runs the grasp**. A pick is one
  `PICK_AND_PLACE` (0x04); the whole on–transport–off sequence executes
  controller-side while the client blocks on a single status packet. The host is
  never told when the vacuum came on, so it cannot start a timer at the right
  instant, and being synchronous it cannot interleave a `VACUUM_OFF`.
* What the host *can* do — time out and E-stop — bounds host waiting, not
  gripper holding, and if the link is what failed the E-stop cannot be sent.
* Decomposing the pick into host-driven steps would make a host timer
  expressible (the opcodes exist) but would put a grasp's real-time discipline
  on a TCP link with a 3-attempt retry and no live heartbeat — same hazard,
  worse route.

**Conclusion recorded in §11.2: the bound belongs controller-side, in
`PickAndPlace`, where it needs no protocol change and no frame field. It does
not exist, and neither does a specified dwell.** Both are logged in
`NEXT_STEPS.md` item 7(a), with the note that `mock_kuka_server` tracks
`vacuum_on` but has no notion of elapsed dwell, so a controller-side bound would
need a test that measures time rather than state.

---

## O2 — "centroid error ≤ 2 px · Pass" (§3, §10.5, Appendix E)

**Measured, not marked "Not measured".** The detector and data existed; the
harness now exists and is committed.

| Detector · corpus | n | mean | median | p95 | max | ≤ 2 px |
|---|---:|---:|---:|---:|---:|---:|
| default anchors, `recog/dataset` val (the detector §10.1 reports) | 129 | 7.75 | 4.88 | 22.65 | 57.24 | **18.6 %** |
| shipped `best.pt`, `recog/dataset3d` val (production) | 1 150 | 1.62 | 1.13 | 4.37 | 19.77 | **75.6 %** |
| `HeuristicDetector`, `recog/dataset`, 100 frames | 381 | 2.37 | 0.71 | 4.53 | 120.00 | **94.2 %** |

Per class on the shipped detector: battery 0.99 px median / 83.3 % inside;
cartridge 2.21 px median / **42.7 %** inside.

**Verdict set to Fail against the criterion as written**, not to a vague
softening. O2 says error "shall not exceed 2 px" — an absolute bound — and no
configuration meets that. The shipped detector meets it typically (1.13 px
median) and misses it in the tail (4.37 px p95, 24 % of matched detections
outside); the detector §10 reports fails outright. §10.5 also records the
observation the threshold itself invites: **a 2 px bound with no stated quantile
is not a testable criterion**, and §3's derivation (a third of a 3 mm gripper
budget at 0.38 mm/px) puts the shipped p95 at 1.7 mm — inside the 3 mm total,
outside the third O2 allocated to perception.

Two caveats stated at the measurement: errors are over IoU-matched pairs only,
so misses are excluded and the figure is the **generous** one; and the cited
`tests/test_evaluate.py` is named as measuring nothing, with its 5 px assertion
quoted. §1's "four of six objectives fully met" becomes three of six.

---

## The rest

| Item | What I did | My figure vs the brief |
|---|---|---|
| **F2** anchors (§5.7, §10.7, ADR-005) | Disposition claim struck at all three sites; ablation left standing. Recorded that `build_fasterrcnn` builds a custom generator unconditionally, no flag exists, and the shipped config carries a third set. | Agrees. Added: torchvision's scheme **is** expressible as `[0.5,1,2] × [32,64,128,256]` — `detector_bench.py` uses exactly that to reload the checkpoint. "Not selectable" is too strong; "not selected, and not named by any flag" is right. |
| **F4** thresholds (§5.1) | Struck; 0.70 / 0.4 stated, with the 0.05 eval threshold explained so the contradiction cannot recur. | Agrees. |
| **F5** Appendix E closing | Replaced with a per-cell statement. Added two caveats the audit did not: no dataset or checkpoint is tracked by git, and `forbidden_bench_timings.csv` is deliberately gitignored. | Agrees. |
| **S1** coverage (§9.3) | Table kept and marked "as measured 2026-04-20 … superseded"; receipt's real figures (1 032 tests, 89 % / 65 %) given; §1, abstract and Appendix C updated; C.2's "left as written" decision reversed and marked. | Agrees. |
| **S2** §5.3 / §5.4 / Appendix B | Comparison tables added; all eight stale values and the three omitted transforms (`VerticalFlip`, `RandomRotate90`, `MotionBlur`/`Defocus` + `ISONoise`) named, plus train `min_visibility=0.25`. | Agrees, **except `frozen_bn_epochs` — see below.** |
| **`frozen_bn_epochs`** (§5.4, Appendix B) | The brief said "frozen-BN 20→8". By the time I wrote it, `dd36329` had **deleted the key**, so "8" would have been a fresh staleness. §5.4 now withdraws the sentence rather than re-numbering it, and gives the mechanism: the knob was *half-inert* — `requires_grad` was never restored so γ/β stayed pinned for all 35 epochs, while `model.train()` let the **running statistics thaw at epoch 8**, leaving the two halves disagreeing for 27 epochs. No setting of it was coherent, which is why it was removed rather than repaired. BN is now frozen both halves for the whole run (torchvision's detection recipe, right at `batch_size: 2`), and `train_one_epoch` raises if `freeze_batchnorm` returns 0. **Consequence recorded: runs after `dd36329` differ from the published checkpoints** — a second, independent reason (alongside Appendix C.3's unseeded training) that reproduction lands *near* rather than *on*. Appendix B strikes the key. | Brief's "20→8" superseded by a commit that landed mid-task. |
| **S3** §6.3.1 latency | Recomputed from `forbidden_bench_timings.csv` (240 rows). **mean 2.831, p50 2.687, p90 3.650, p95 3.929, p99 5.346, max 7.426 ms**; per-level maxima 5.42 / 3.21 / 4.15 / 4.91 / 3.99 / 7.43. Corrected in §1, §6.3.1 ×2, §13.1(3), README, CV_BULLETS. Both directions reported. Also fixed the aware arm: **0.32 / 1.05 ms**, not 0.33 / 1.14. | Agrees. |
| **S4** line count (§11.3) | Withdrawn; scope table given. | Agrees **only at HEAD**. See below. |
| **S5** `planning.yaml` (Appendix B) | Nine keys struck with a grep-verified list. | **Nine, not eight** — see below. |
| **O3 / O1.a / O5** traceability | Appendix E rows rewritten; §10.4 gets a provenance paragraph; §10.6's `bench_cycles` `empty_queue` rate flagged; §10.5's O3 row re-cited to two committed tests. | Agrees. |
| **Audit K interlock** | Recorded in §10.4 (see below). | Agrees. |
| **U5** perception latency (§10.4) | Folded into the same provenance paragraph: three artefacts, three values (3.0/4.1 table, 3.3/5.5 receipt, 5.78 ablation), the table's has no receipt. | Not in the brief; corrected because it sits in the paragraph being fixed. |
| **U6** "prevents any place pose over an exposed busbar" (§11.2) | Qualified to the extracted mask, with §13.2.1's 2-of-25 residual. | Not in the brief; corrected because it is the adjacent clause of the sentence F3 lives in and is an absolute safety claim. |

### Audit K's interlock note

Recorded in §10.4, next to the O3 budget: `_MAX_CARTRIDGE_EXTENT_MM = (81.7,
180.0)` in `plan/placement_area.py::reject_if_not_one_cartridge_floor` raises
`BadDetectorBox` **before the occupancy grid is built**, capping
`pack_best_effort` at 2.04 ms against 8 ms (3.9× margin) and putting the first
breach at a 158 × 314 mm floor the extractor cannot deliver. The guard was
written to reject a detector box spanning a cartridge and three loose cells —
an unrelated reason. **Raising `max_cartridge_extent_mm` for a larger SKU
raises the packing cost and no test would report it.** Also logged in
`NEXT_STEPS.md` 7(b), where the cheapest fix (a comment at the constant) is
named.

---

## Where I disagree with the brief

Three checks came out differently. All three are in the direction of *more*
findings, not fewer.

1. **`planning.yaml` dead keys: nine, not eight.** The eight the brief lists are
   confirmed. **`camera.mm_per_px_y` is a ninth** — listed in Appendix B, present
   in the file, read by no Python (only `mm_per_px_x` is). A tenth,
   `cartridge.green_channel_thresh`, is dead in the file but was never listed in
   Appendix B, so it is mentioned parenthetically rather than struck.

2. **The line counts reproduce only at HEAD.** 3,852 / 10,692 / 15,974 are
   exact — but only when counted from `git show 39429a4:` blobs. In the working
   tree they read 4,395 / 11,371 / 16,693, because another agent's in-flight
   edits to `plan/planner.py` and `plan/scene.py` add ~540 lines to the smallest
   scope. The FDR now names the commit alongside the numbers. If the same count
   is repeated later against a dirty tree it will not match, and that is why.

3. **Appendix B has a fourth, opposite defect in the paragraph S5 touches.** It
   states "The planner's own approach and insert heights are separate keys in
   `configs/planning.yaml`'s `motion:` block." **There is no `motion:` block in
   that file.** `PlannerConfig.from_dict` genuinely reads
   `motion.grasp_height_mm` and `motion.insert_height_mm` (and correctly refuses
   `motion.approach_height_mm` by name), but the shipped config supplies none of
   them, so the pick grasp height (5.0 mm) and place insert height (2.0 mm) are
   dataclass defaults no configuration states. That is the mirror image of the
   dead-key defect: a live key no config declares. Corrected in Appendix B.

Minor: arm 2's mAP@0.75 reproduces as 0.5833 against the receipt's 0.5831 —
CPU/GPU float drift, noted in the receipt rather than smoothed over.

Fourth, from the coordinator's mid-task note: the brief's "frozen-BN 20→8" was
overtaken by `dd36329`, which deleted the key. Writing "8" would have introduced
a new staleness into a correction pass. Checked before writing.

The real-photo figures that moved with the VOC-bin fix (mAP@0.50 0.8631 →
0.8647, mAP@0.75 0.7862 → 0.7878) were grepped for across `FDR_v3`, `FDR_v2`,
`README`, `PORTFOLIO`, `CV_BULLETS` and `NEXT_STEPS`: **they appear in none of
them**, so nothing needed correcting and nothing new was cited.

---

## What could not be made truthful without code changes

Stated in the report as gaps, not softened:

1. **The vacuum-dwell bound (F3).** Needs a KRL timer in `PickAndPlace` and a
   *specified* dwell. Not implementable host-side under the shipped opcode set,
   for the structural reason given above. §11.2 says what exists, what does not,
   and what would settle it.
2. **O3's published distribution.** `bench_cycles.py` was never committed, so
   mean 5.0 / median 3.0 / p95 13.0 ms cannot be checked in either direction. The
   *threshold* survives on two committed tests and §13.2.1; the distribution does
   not. Committing a cycle benchmark is the fix.
3. **O5's "fixed input → fixed output".** No test runs the planner twice on one
   snapshot and compares. The closest,
   `test_the_frames_scale_beats_a_configured_fallback`, asserts one field of one
   element across two planner instances and does so to check scale precedence.
   Marked half-supported rather than Pass.
4. **O1.a's regenerability.** The number checks out (0.4463 + 0.5121 → 0.4792);
   `pr_curves.py` was never committed. Marked "present, not regenerable".
5. **The nine dead `planning.yaml` keys.** Appendix B now strikes them from the
   schema list, but the *file* still carries them. Deleting them is a change to
   a config another agent may be editing; logged in `NEXT_STEPS.md` 7(c).
6. **`evaluation.centroid_error_target_px` / `.edge_error_target_px`** in
   `recognition.yaml` are read by no code — the O2 threshold itself, declared and
   never enforced, which is part of why O2 went unmeasured. Noted in Appendix B;
   not wired up.

---

## Sections touched

**`docs/FDR_v3.md`** — Executive summary (findings 1, 2 and 3, status-against-
criteria, coverage, reading-order claim); Abstract; §1.2 (unchanged, verified);
§3 (O2 row); §5.1; §5.3; §5.4; §5.7; §6.3.1 (scope note + the aware-arm
paragraph); §9.1; §9.3; §9.5; §10.4; §10.5; §10.6; §10.7; §11.2; §11.3;
§13.1(3); §13.3; Appendix B (both config blocks); Appendix C intro and C.2;
Appendix E (O1.a, O2, O3, O5 rows and the closing paragraph); ADR-005.

**`docs/FDR_v2.md`** — §10.5 verdict table (O2, O3), §10.6 zero-FP claim, §11.2
dwell, §13.3 "what surprised", Appendix traceability matrix (O2, O3).

**`docs/NEXT_STEPS.md`** — new item 7 under "What is honestly unfinished".

**`README.md`** — packer latency; suite size.

**`docs/CV_BULLETS.md`** — blurb; packer latency; suite size ×2; three new
prohibitions (precision, O2, anchors).

**`docs/PORTFOLIO.md`** — suite size.

**New:** `scripts/detector_bench.py`, `docs/receipts/detector_bench.txt`.

## Not touched

`plan/scene.py`, `plan/planner.py`, `main.py` and their tests — another agent
owns those. No config file was edited. No source module was edited; the only
code added is a new benchmark script.
