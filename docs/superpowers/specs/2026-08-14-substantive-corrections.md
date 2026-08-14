# Substantive corrections — 2026-08-14

**Base HEAD:** `83a1383` · **Scope:** the twelve result-figure errors that
`2026-08-14-pre-publication-cleanup.md` §5 reported rather than fixed.
No code, config, metric definition, dataset, checkpoint or receipt was
changed. **Suite before and after: 1 276 passing, 1 skipped**;
`python scripts/model_card_tables.py --check` reports no drift.

The cleanup pass declined to correct these because "correcting a result means
choosing which measurement is authoritative, and that is the author's call".
That call is now made, and the rule used to make it was: **the receipt wins.**
Every replacement figure below was read out of its receipt during this pass,
not carried over from the report that flagged it — a discipline that turned up
one error *in the flagging report itself* (#5) and one the report missed (#13).

Each correction is written to be legible rather than silent: a reader who
remembers the old figure sees it struck or named, with the reason. That is this
repository's existing convention and it is why the audit trail is worth
anything.

---

## The twelve

| # | Where | Now reads | Verified against |
|---|---|---|---|
Line numbers are **post-edit**; the cleanup report's §5 gives the pre-edit ones.

| # | Where | Now reads | Verified against |
|---|---|---|---|
| 1 | `README.md`:36, `PORTFOLIO.md`:55 | Placeable-area error **79.2 mm²**/crop (was 51.5) | `docs/receipts/seg_eval.txt`:52 — `bay` opt mean 79.2, over 126 val crops |
| 2 | `README.md`:36 | Boundary displacement **1.226 mm** (was 0.949) | `docs/receipts/seg_eval.txt`:37 — `bay` 1.226 mm, 35 crops |
| 3 | `FDR_v3.md`:487–497 | Oracle reaches **11 of 30** and 25 cells at 0.0 mm inset (was 12) | `2026-08-11-placement-feasibility.md` §5 Result table — "inset 0.0, whole-cell occupancy (HEAD): 11 / 25" |
| 4 | `README.md`:182, `MODEL_CARD.md`:492, `FDR_v3.md`:151, 4574, 4828 | **thirty-nine** receipts (was thirty-seven), in all five places | `git ls-files docs/receipts` → 39 |
| 5 | `README.md`:67–69 | See **"The config claim"** below — the report's own replacement figure was wrong | `md5sum` over comment-stripped, path-normalised configs |
| 6 | `MODEL_CARD.md`:328–360 | The shipping detector **does** have a published held-out mAP: **arm 3**, `mAP@0.50 = 0.9053` | `docs/receipts/detector_bench.txt`:68–81 |
| 7 | `MODEL_CARD.md`:75 | **Two** photograph measurements, not one | `docs/receipts/real_photo_eval.txt`:45–52 |
| 8 | `FDR_v3.md`:3328, and 3279 and 3915 for consistency | `dataset3d_seg` is **840** crops (was 841) | `docs/datasets/dataset3d_seg.manifest.json` — `per_seg_class_kept.cartridge` = 840; 0.15 × 840 = the receipt's 126 |
| 9 | `FDR_v3.md`:3858–3872 | 1.0 and 2.0 mm **move the count without moving the decision** (was "move nothing") | `2026-08-11-placement-safety.md`:150–156 |
| 10 | `README.md`:238 | Aware-FFDH arm **0.32 ms at 2.5 % coverage, peaking at 1.05 ms at 15 %** (was "0.9 ms for FFDH alone") | `docs/receipts/forbidden_bench.txt`:3–9 |
| 11 | `MODEL_CARD.md`:48 | New paragraph reconciling **17.0 / 52.7** with the README's **16.6 / 57.1** | `seg_eval_anchored_on_cad_test.txt`:66 vs `seg_eval.txt`:66 |
| 12 | `README.md`:85 | Boundary displacement over **35** of the 36 bay-carrying crops | `seg_eval.txt`:20 (IoU, 36) vs :37 (boundary, 35) |

**None of the twelve turned out to be a non-error.** Eleven were exactly as
reported. #5 was an error, but not the one reported — see below.

## The config claim (#5), which is load-bearing

`README.md`:67 said *"the **ten** `segmentation*.yaml` configs are
byte-identical apart from dataset and checkpoint paths"*. The cleanup report
called this wrong three ways — 11 files, 9 training configs, family of 8 — and
was itself wrong on the third. Measured here by stripping comments and blank
lines, normalising the three path values, and comparing MD5:

* **11** files match `configs/segmentation*.yaml`.
* **9** are training configs. The two that are not say so themselves:
  `segmentation_cad_test.yaml` is an evaluation config at
  `train_val_split: 0.0`, and `segmentation_seedcheck.yaml` is a one-epoch
  seeding probe whose header disclaims membership.
* **9** — not 8 — are byte-identical apart from `dataset.coco_path`,
  `dataset.img_dir` and `training.checkpoint_dir`. All nine hash to
  `1b3a58cb…`. The report inherited "8" from `FDR_v3.md` Appendix C and the
  seedcheck header without measuring it.

**The number the claim needs is nine.** The claim exists to say that no pair of
compared models differs in anything but its data — so the right scope is the
set of models actually compared, which is nine: one per row of
`MODEL_CARD.md`'s generated training-data table and one per committed segmenter
checkpoint. Eleven over-counts by including an eval config and a probe that
were never trained into a compared model; eight under-counts by one.

**The "eight" in `FDR_v3.md` Appendix C was left standing, because it is
correct as scoped.** It names the eight *generalisation* runs, which exclude
`segmentation_anchored_crown.yaml` — the sealed-unit experiment, a different
investigation that happens to share the family's byte-identity. That reading is
confirmed by history: the eight training configs that existed at `4e8828f` are
exactly the eight, and `ce1d9cd` added the crown config immediately after. The
README now states all three numbers and which one the claim rests on.

## Naming arms, not just numbers (#6)

`detector_bench.txt` contains a genuine collision: **arm 2**'s
`AP_battery@0.50` and **arm 3**'s `mAP@0.50` are both **0.9053**. They are not
the same measurement — different arm, quantity, checkpoint and split — so the
model card now states all four for the one it means: *arm 3, `mAP@0.50`,
shipped `recog/checkpoints/best.pt`, `recog/dataset3d` validation split, 150
frames / 1 205 GT boxes, confidence 0.70*. The collision is called out in place
so the next reader does not merge them, along with `FDR_v3.md` §10.5's existing
warning not to pair either with `last.pt`'s 0.8647.

## A thirteenth, found in passing

`README.md`:180 said `docs/superpowers/audit/` holds **"six adversarial reviews
… run on 2026-08-12"**. There are **nineteen**: sixteen on 2026-08-12 (A–P) and
three on 2026-08-14 (T, U, V). The cleanup pass corrected this same count in
`docs/README.md` and `specs/README.md` (§2 of its report) and missed the root
README. Corrected, since it is the same class of error in a file already in
scope. `2026-08-14-W-omnifactory-context.md` is *not* counted — it is a research
brief rather than an adversarial review, and it remains untracked.

## What could not be sourced, and was therefore not repaired

**The `inference_min_size` ablation** (`MODEL_CARD.md`:75 — 500 → mAP@0.50
0.614 / @0.75 0.404, against 800 → 0.457 / 0.023) **has no receipt.** Its only
source is an undated comment in `configs/recognition.yaml`:60–63 that names no
checkpoint, and it does not reconcile with `real_photo_eval.txt`'s 0.8484 /
0.8044 taken at that same `min_size: 500`. The same config's line 73 quotes a
third pair (best.pt 0.6858 against last.pt 0.7296 "on the six real photos"), so
the figures plainly predate the shipped checkpoint. **No number was substituted
— an unsourced number is what created several of these.** The bullet now says
the figures have no receipt and should be read as a direction rather than a
measurement. Settling them needs a re-run of `recog.eval_real` at both
resolutions, which is a measurement task, not a documentation one.

**Carried forward unchanged** from the cleanup report's own "checked and left
alone": the *4 536 renders* leak-hunt figure (committed manifests sum to 4 516;
the count was taken on gitignored image directories and cannot be settled by
reading the tree).

## Not touched

`docs/RESULTS_SUMMARY.md` — being drafted concurrently by another agent.
Dated records under `../audit/` and the 2026-08-1x specs were left unedited per
the convention that a correction is recorded in a successor rather than by
editing the original; the superseded figures they quote (0.949 / 0.987 /
1.184 mm at `NEXT_STEPS.md`:843, `FDR_v3.md`:3384) are historical narrations of
the correction itself and are correct as history.
