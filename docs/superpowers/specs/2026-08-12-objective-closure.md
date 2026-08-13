# Objective closure — what was closed, what was restated, and what was not touched

**Date:** 2026-08-13 · **Base HEAD:** `885a044` (tree moved to `9b38de9`
mid-pass; see §6) · **Evidence base:**
`docs/superpowers/audit/2026-08-12-N-objective-closure.md` and
`2026-08-12-J-claim-verification.md`.

Every figure below was re-derived by executing something at HEAD, not
carried over from the audit. Where my measurement and the audit's differ,
mine is the one reported and the difference is stated.

---

## Headline

**The FDR now reports four of six objectives met, up from three, without
a single GPU-hour and without softening a number.** O1, O5 and O6 are
closed; O2 stays Fail and is made *worse* by being stated exactly; O3
keeps its threshold and loses its distribution. §10's own detector figure
is corrected against the one the project actually ships.

| Objective | Was | Now | How |
|---|---|---|---|
| **O1** mAP@0.5 ≥ 0.90 | Partial — 0.874 | **Pass, in domain — mAP@0.50 = 0.9053** | Re-citation only. No retrain. |
| **O2** centroid ≤ 2 px | Fail as an absolute bound | **Fail, and now exact** — fails at the median too, once stated in mm | Restated in the derivation's own units |
| **O3** rebuild ≤ 8 ms | Pass on threshold; distribution "not reproducible" | **Pass on threshold; distribution withdrawn** | Re-measured; it does not reproduce and the new number is worse |
| **O4** pick-failure recovery | Not tested | **Not tested** | Untouched — needs hardware |
| **O5** determinism | Half-supported | **Pass, both halves observed** | One 15-line test, landed |
| **O6** coverage ≥ 70 % | Pass — 89 % on an April module list | **Pass — 93 %** over `main.py`'s import closure | Re-scoped, receipt regenerated |

---

## 1. O5 — closed, and the test is the deliverable

`tests/test_planner.py::test_two_fresh_planners_on_one_snapshot_produce_the_same_queue`.
It cycles **two fresh `Planner` instances** over one snapshot and asserts
the queues are equal field-by-field, then again on `float.hex()` of every
millimetre coordinate — so the claim is bit-identity, not agreement at a
printed precision. It also asserts `len(q1) > 1` so an empty queue cannot
pass it vacuously.

**Two instances, not one cycled twice, and the docstring says why.**
`Planner` owns a persistent `Scene`, so a second `cycle` on a live
instance is a different input by design. The only field that then differs
is `battery_detection_id` — a monotonic per-instance counter returning
0–7 on the first cycle and 8–15 on the second — while every geometric
field stays bit-identical. Written the obvious way the test fails and
looks like a determinism bug. That comment is worth as much as the
assertion.

**Verified, by execution:**

- passes in the working tree and in a clean worktree of `885a044`;
- passes under `PYTHONHASHSEED` 0, 1 and 9999 as a pytest run;
- across four *separate processes* at `PYTHONHASHSEED` 0, 1, 4242 and
  9999, the SHA-256 of the queue's `float.hex()` tuple is
  `c7b87b86e7ffd65a…` in all four — identical, 5 poses each;
- runs and passes in a **torch-free** environment (confirmed in the
  JUnit XML of a torch-blocked run, not inferred).

The known nondeterminism sources were checked rather than assumed. The
mock server's unseeded drop (`execution/mock_kuka_server.py:223`, `:237`
— `random.random()`, and grep confirms nothing seeds it) is downstream of
`Planner.cycle` and unreachable from a planner-only test; it is a real
end-to-end reproducibility gap and is now recorded as such in §6.5 rather
than left implicit. Dict ordering is excluded empirically by the
cross-process digest above.

## 2. O1 — a citation, not a model

**Verified against the receipt myself** (`docs/receipts/detector_bench.txt`,
arm 3): `mAP@0.50 0.9053  AP_battery@0.50 0.9046  AP_cartridge@0.50
0.9061  mAP@0.75 0.7607`; precision 0.9488, recall 0.9544 over
1 205 GT / 1 150 matched; shipped `best.pt`, shipped anchors, 150-image
`recog/dataset3d` validation split, confidence 0.70, NMS IoU 0.40,
inference resize 500/900.

**One trap, flagged mid-pass and confirmed by reading the file: 0.9053
appears twice in that receipt with two different meanings.** Arm 2's
`AP_battery@0.50` is 0.9053 and arm 3's `mAP@0.50` is 0.9053 — same four
digits, different quantity, different checkpoint, different corpus.
Every citation I wrote names the arm, the quantity, the checkpoint and
the split, and §10.1 now carries an explicit paragraph warning a verifier
about the collision.

**The out-of-domain figure is the shipped checkpoint's, not another
one.** `docs/receipts/real_photo_eval.txt` (read and verified): `best.pt`,
sha `77647a3e…`, **0.8484** mAP@0.50 / 0.8044 mAP@0.75 over **6 of 7**
images (80 GT), the seventh excluded because it carries no ground-truth
boxes. The 0.8647 that appears in earlier audit prose is **`last.pt`**, a
different network; the FDR now says so in two places so the pairing
cannot be made by accident.

No retraining was done and the FDR now records why none should be:
training was seeded and BatchNorm behaviour changed at `dd36329`
(commit verified to exist), so a rerun produces a different network and
invalidates all three arms of `detector_bench.txt`, the real-photo
figures, and every §10/§13 number keyed to this checkpoint.

## 3. O6 — re-scoped, regenerated, and the old figure retired on the record

`docs/receipts/pytest-cov.txt` is regenerated from one `pytest --cov` run
on a clean git worktree of `885a044` (1 212 collected, **1 210 passed, 2
skipped**, exit 0). Three scopes, all read off that single run:

| Scope | Modules | Stmts | Branch |
|---|---:|---:|---:|
| **A — `main.py`'s transitive import closure (O6's figure)** | **19** | **1 845** | **93 %** |
| B — everything `[tool.coverage.run] source` resolves to (disclosure) | 49 | 7 208 | 67 % |
| C — the 2026-04-20 18-module list, **retired** | 18 | 1 784 | 91 % |

All three reproduce audit N §6 exactly, which is a useful independent
check on the audit and on me.

**The receipt contains no hand-written result lines.** The previous one
had nine — a smoke-test summary `pytest --cov` never emitted. This one is
the pytest output, the two `coverage report --include=` commands and
their output, and the AST closure script **with the output of actually
running it** (21 files: the 19 modules, plus `main.py`, which is not
under `source`, plus `recog/model.py`, which `omit` excludes for torch).
A reader can check the membership rather than take it.

**Why the re-scope is stronger and not just larger.** The 18-module list
was a snapshot of what one run printed, and it is wrong in both
directions: it omits `common/packing.py` (the packer O3 is certified on),
`plan/arbitration.py`, `recog/bay_segmenter.py` (the shipping
segmentation path §13 spends most of its pages certifying) and
`recog/calibration.py`, and it includes `recog/augmentation.py`,
`recog/dataset.py` and `recog/evaluate.py`, which `main.py` never loads.
A headline that excludes the packer and the segmenter is not "the
production code", which is what O6's own wording claims.

## 4. O2 — the Fail restated exactly, and not softened

**Measured, not relayed.** Over all 1 000 `recog/dataset3d/meta/scene_*.json`
sidecars, `ortho_scale × 1000 / width`:

```
mm/px   min 0.4879   p05 0.5197   median 0.7713   p95 1.0260   max 1.0856
```

`recog/synth3d/world.py:499` sets `ortho_scale` per scene, and the file's
own comment at `:518` documents the same `× 1000 / width` conversion, so
this is the corpus's calibration by its generator's definition.

O2's 2 px is a **0.76 mm** physical allowance at the 0.38 mm/px §3
declares. At the median `dataset3d` scene that allowance is worth
**0.99 px**, and the shipped detector's 1.13 px median is **≈ 0.87 mm —
outside it**. So the Fail reaches the median, not only the tail. The FDR
now carries a millimetre table beside the pixel one:

| | scale | median | p95 | max | allowance |
|---|---|---:|---:|---:|---|
| shipped `best.pt`, all | 0.771 mm/px | **0.87 mm** | 3.37 mm | 15.25 mm | 0.76 mm |
| — battery only | 0.771 | 0.76 mm | 3.02 mm | — | 0.76 mm |
| — cartridge only | 0.771 | 1.71 mm | 5.58 mm | — | 0.76 mm |
| default anchors (§10's detector) | 0.38 | 1.86 mm | 8.61 mm | 21.75 mm | 0.76 mm |
| `HeuristicDetector` | 0.38 | 0.27 mm | 1.72 mm | 45.60 mm | 0.76 mm |

**One correction to the audit I am obliged to make.** Audit N §1.4 gives
the shipped detector's p95 as "roughly 2.7 mm". Recomputed: 4.370 px ×
0.7713 mm/px = **3.37 mm**. That is worse than the audit stated, and it
matters, because 3.37 mm exceeds the **entire 3 mm** end-to-end
stack-up — in the tail, perception alone consumes the whole gripper
budget and leaves nothing for calibration or the arm.

**And one place where I did not follow the brief, because the arithmetic
does not support it.** The brief asked me to record that no quantile
reading rescues O2. That is true of the criterion **as written** (2 px →
0.76 mm), and I have written it that way. But the *derivation* in §3
allocates one third of 3 mm = **1.00 mm = 2.63 px**, and the 0.87 mm
median **is** inside that. Stating "no quantile reading passes" without
that caveat would itself have been a figure that does not survive
checking. The FDR therefore enumerates four readings and says exactly
what each costs:

- **(a) absolute** — what the stack-up means, since a tolerance stack-up
  bounds each individual grasp and there is no grasp that succeeds at the
  median. 24 % of matched detections exceed 2 px, p95 3.37 mm, worst
  matched pair 15.2 mm. **Fail.**
- **(b) median, in mm, against the table's 2 px** — 0.87 mm vs 0.76 mm.
  **Fail**, and the battery class alone lands *on* the allowance rather
  than inside it.
- **(c) median, in mm, against the derivation's own 2.63 px** — 0.87 mm
  vs 1.00 mm. **This is the single arrangement in which any part of O2
  passes**, and it needs a quantile no draft states *and* the derivation's
  number in place of the table's. Even then the p95 exceeds the whole
  3 mm stack-up.
- **(d) on cartridge corners**, as v1/v2 §1.2 word it — the metric becomes
  edge error: 2.64 px median / 7.94 px p95 (L∞). **Fail at the median in
  pixels**, before any conversion.

The `HeuristicDetector`'s 0.27 mm median is inside the allowance and is
addressed explicitly rather than left for a reader to notice: it is
conditional on the 46 % of objects it detects at all, 5.8 % of even those
exceed 2 px, its worst matched pair is 60× the bound, and §10.1 already
says it is not the answer to O1.

The requirements finding is recorded in **§3**, where the derivation
lives: three numbers across three drafts (5 px prose on corners, 2 px in
their own tables on centroids, 2.63 px from the shared derivation), two
measurands, and no quantile anywhere — including
`configs/recognition.yaml`'s `evaluation.centroid_error_target_px: 2.0`.

## 5. O3 — threshold kept, distribution withdrawn

**Re-measured myself**, on `tests/test_planner.py`'s own fixture
(800 × 600, one cartridge, eight batteries, so per-cartridge equals
per-cycle), 100 cycles per arm after a 5-cycle warm-up, two independent
runs:

| arm | mean | median | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| cold (fresh `Planner`) | 7.89 / 7.95 | **7.82 / 7.91** | 8.27 / 8.48 | 7.55 | 8.62 / 9.06 |
| warm (twin cached) | 5.80 / 6.18 | **5.71 / 5.98** | 6.18 / 7.09 | 5.53 | 7.79 / 9.47 |

Audit N reported 7.96 / 5.73; both of my runs bracket that, so the audit
reproduces. Against §10.4's published **mean 5.0 / median 3.0 / p95
13.0 ms** and its "under 2 ms" steady state: the cold median is **~2.6×**
the published median and sits *on* the 8 ms budget rather than at 38 % of
it; the warm path is **~3×** the claimed 2 ms; and the published p95 is
conversely *pessimistic* against today's 8.3–8.5 ms. **The distribution
moved in both directions, so it cannot be repaired by adjustment — it is
withdrawn.** The threshold survives on two committed tests and §13.2.1,
neither of which depends on the table.

I did **not** commit a cycle benchmark. The brief did not ask for one and
the honest reason to build one is only if the report intends to quote a
distribution again — in which case the verdict gets worse, not better.
The measurement above is labelled in the FDR as scratch-derived and
explicitly not a replacement figure to be quoted.

**The interlock dependency is now stated as a dependency.** O3's margin
is held by `plan/placement_area.py::reject_if_not_one_cartridge_floor`
(`_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)`, verified at line 513), a
*perception* guard written because a detector box once spanned a
cartridge and three loose cells. It is the only thing bounding the item
count the packer is handed. Removing or relaxing it leaves every O3 test
passing while the budget silently stops holding, because those tests
exercise fixed fixtures rather than the ceiling. With the cold path now
running *at* the budget, the guard is not spare margin — it is the
margin. The FDR also records that the 2.04 ms ceiling is itself
un-receipted (audit K's scripts were never committed), so the ceiling is
argued, not certified.

## 6. §10's own detector — corrected

§10 is the April chapter. Verified from `docs/receipts/train_eval.txt`:
100 synthetic images, 85/15 seed-0 split, ResNet-34+FPN **from scratch**
(no COCO pretrain), 15 epochs, batch 1, ~31 min on 2 CPU threads, eval
score threshold 0.05. The 0.874 default-anchor arm is **§10's winner**,
not an ablation baseline set up to lose, and the section was never
rewritten when the Blender detector landed in §13.

**The localisation gap is partly an evaluation artefact, and the FDR now
nets it out.** Both corpora are natively 1280 × 720 (verified: the
sidecars report `1280 × 720`, and `PIL` reports the same for
`recog/dataset`). `scripts/detector_bench.py` sets
`model.transform.min_size/max_size` (lines 236–237) to 320/512 for arm 2
and 500/900 for arm 3, and torchvision scales by
`min(min_size / 720, max_size / 1280)`:

- arm 2 → `min(0.4444, 0.4000) = 0.400`, network sees 512 × 288;
- arm 3 → `min(0.6944, 0.7031) = 0.694`, network sees 889 × 500.

Boxes are mapped back to 1280 × 720 before scoring, so one network-space
pixel of regression error is reported as **2.50 px in arm 2 and 1.44 px
in arm 3 — a factor of 1.74 present before any question of model
quality**. The remaining ~2.5× of the 4.3× centroid improvement is
corpus, schedule and initialisation.

Also recorded: §10.7's *losing* arm is no longer reproducible.
`train_eval.txt:13` names `recog/checkpoints/best.pt` as the epoch-11
custom-anchor checkpoint at 0.7643; the file at that path today is dated
2026-08-06 and is the shipped Blender model (sha matches
`real_photo_eval.txt`). `default_anchors_best.pt` survives at 2026-04-29,
so the *winning* arm can be re-run and the losing one cannot.

## 7. Sections touched

`docs/FDR_v3.md`, corrected in place, in the document's own dated-strike
voice. No restructuring, no renumbering, no section added or removed.

- **Executive summary** — objective tally three → four of six; coverage
  89 % → 93 % / 65 % → 67 %; test count; O1, O2, O5, O6 status.
- **Abstract** — coverage and test count; O3's distribution marked
  withdrawn with the re-measured figures; a clause noting the shipped
  detector clears 0.90 in domain.
- **§3** — O1, O5, O6 "Verified in" citations; a dated note recording
  the 5 px / 2 px / 2.63 px inconsistency, the two measurands, the
  missing quantile, and why a stack-up cannot be read at a median.
- **§6.5** — the determinism paragraph's "Verified by
  `test_row_major_ordering`" struck as covering only half the claim;
  the new test cited; the mock server's unseeded RNG recorded as an
  end-to-end gap outside O5's scope.
- **§9.1** — coverage figure and scope.
- **§9.3** — re-scoped in full: the 18-module list retired with its
  four runtime omissions and three non-runtime inclusions named, the
  three-scope table, the current suite count.
- **§9.4** — repeatability added as a fourth property-based invariant.
- **§10.1** — the April-chapter correction, the arm-2/arm-3 comparison
  table, the resize arithmetic, the 0.9053-appears-twice warning, the
  "shipped detector has no published held-out mAP" correction.
- **§10.4** — O3's distribution withdrawn with the re-measurement; the
  interlock recorded as an O3 dependency.
- **§10.5** — verdict table (all six rows); O2's millimetre restatement
  and the four readings; O1's promotion with its in-domain scope and the
  `best.pt`/`last.pt` caution; O5's closure and the two-instance trap;
  the tally.
- **§11.3** — the LOC table's 18-module row marked as a retired scope
  (the line count itself is unaffected).
- **§13.2** — priority-1 item re-aimed: its in-domain half is done, the
  transfer half is what remains.
- **§13.3** — coverage figure; a reflection on O5 arriving four months
  late and what the pattern actually is.
- **Appendix C, C.2** — receipt description and the coverage narrative.
- **Appendix E** — O1, O2, O3, O5, O6 rows and the summary paragraph.

Also: `tests/test_planner.py` (+40 lines, one test),
`docs/receipts/pytest-cov.txt` (regenerated), `docs/NEXT_STEPS.md`
(item 7(b) updated with the re-measurement and the "build it only if you
will quote it" guidance; item 7(d) added for O2's residual mm join).

## 8. What I could not close, and what I did not touch

- **O4** needs the robot. Untouched.
- **O2** cannot be closed. Nothing in the corpus, the config or any
  draft supports a Pass, and the work that remains (§7(d) of
  `NEXT_STEPS.md`) makes the Fail exact rather than removing it.
- **O2's millimetre figures are first-order** — the pixel statistic times
  the corpus's median scale, not a per-pair join, because
  `scripts/detector_bench.py` does not attach each matched pair to its
  own frame's scale. This is labelled as such at both sites. The exact
  version is ~20 lines against
  `recog/calibration.py::frame_mm_per_px_for_image`; `scripts/` was not
  mine to edit in this pass.
- **The coverage receipt is measured at `885a044`, not at final HEAD.**
  The tree moved to `9b38de9` mid-pass (a concurrent agent's
  `recog/eval_real.py` work). That module is **not** in `main.py`'s
  import closure, so scope A — the figure O6 is verdicted on — is
  unaffected; scope B's 7 208-statement denominator would shift slightly.
  The receipt names the commit it was measured at, which is the
  discipline the previous receipt already used.
- **A pre-existing flake, seen once and not caused by me.**
  `tests/test_main_integration.py::test_torch_free_demo_does_not_build_a_segmenter`
  failed once in a clean worktree with
  `ConnectionResetError: [WinError 10054]` on the mock server's
  handshake, and passed on the immediate re-run and in every other run.
  It is a local socket race in the test harness, worth a look, and it is
  not in my scope.
- **Not touched, by instruction:** `docs/MODEL_CARD.md`,
  `README.md`, `docs/README.md`, `recog/realtest/`, `recog/eval_real.py`
  and its receipts. The real-photo figures cited above were **read** from
  `docs/receipts/real_photo_eval.txt`, not written by me.

## 9. Verification

- `pytest --cov` on a clean git worktree of `885a044` with the new test:
  **1 212 collected, 1 210 passed, 2 skipped, exit 0**.
- Same worktree with `torch` and `torchvision` blocked by a meta-path
  finder: **0 failures, 0 errors**, exit 0; 1 164 passed, 48 skipped (30
  of them `tests/test_bay_segmenter.py`, which `importorskip`s torch at
  module level and leaves collection entirely). The new O5 test is
  present and **passed** in that run — checked in the JUnit XML, not
  assumed.
- `tests/test_planner.py` green in the working tree (46 tests).
- Every figure quoted in the FDR edits was re-derived at HEAD: the
  detector receipt read directly, the mm/px distribution recomputed from
  1 000 sidecars, the planner cycle timed twice, the coverage measured on
  one clean run, the resize arithmetic recomputed from the values in
  `scripts/detector_bench.py`, `dd36329` and `train_eval.txt:13`
  confirmed to exist and say what is claimed.
