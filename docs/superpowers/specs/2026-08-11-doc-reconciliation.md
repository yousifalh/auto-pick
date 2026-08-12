# Documentation reconciliation — 2026-08-11

Six things had become true in the code and the receipts while remaining
wrong, missing or misleading in `docs/NEXT_STEPS.md`, `docs/FDR_v3.md`
and `README.md`, plus one limitation that had been flagged as missing
for three sessions running and never written. This pass closed all
seven.

Base commit `ce1d9cd`, 708 tests passing. **Documentation only** — no
source file, config, metric definition, dataset, checkpoint or receipt
was touched, and no measurement was re-run. Every figure below was read
out of a receipt or a source spec and verified against it before being
written; none was retyped from the brief.

`docs/superpowers/specs/2026-08-11-placement-feasibility.md` belongs to a
concurrent agent and was not touched.

---

## What was wrong, and what each document now says

### 1. The `bay` generalisation headline conflated two quantities

Published as `bay 0.6555` for a procedural-trained segmenter against a
CAD-trained control's `0.9009`, with an instance count of 213 printed
beside it. Those are different populations: the IoU pools one union over
all **836** crops while the count reports the **213** crops that contain
a bay, so painting `bay` on a closed cartridge is charged against the
same number as segmenting a real bay badly.

Separated (`2026-08-11-transfer-gap-diagnosis.md`):

- **present-only `bay`** — procedural **0.8801** against the control's
  **0.9013**, a difference of 0.021, not 0.246;
- **sealed false-positive rate** — **136 / 623 (21.8 %)**, 675 460 px,
  against the control's **2 / 623 (0.3 %)**, 722 px. **91.4 % of the
  published gap.**

Both halves are now reported together wherever the headline appears:
`docs/FDR_v3.md` §13.1.1 (a decomposition table immediately under the
pooled table, with a bolded instruction not to quote the `bay` column
alone), `docs/NEXT_STEPS.md` "The generalisation measurement" (same, in
the position a reader would copy from), and `README.md`, which carried
no transfer figure at all before this pass and now carries the split one.
`battery` is the same mechanism (0.5593 → 0.6924 present-only) and is
noted alongside; `electronics` and `obstruction` are barely affected by
the split and their pooled figures read as published.

### 2. The crown result, and the caveat it must never be separated from

Crowning the procedural lid moved the sealed false-positive rate
**21.8 % → 2.6 %** (16 / 623; control 0.3 %), pooled `bay` **0.6555 →
0.8755**, and the selected mean **0.6801 → 0.7645** (control composite
0.7960). The pre-registered threshold-shift falsifier did **not** fire:
present-only `bay` *rose*, 0.8801 → **0.8856**, with open-crop recall
and precision both up and the 6× monotone gradient dependence collapsing
to flat rather than shifting down proportionally.

The caveat is carried in substance in all three documents, in bold, and
placed so it cannot be read past: **the `[0, 12]` mm crown range was
chosen after measuring the real Anker lids, so this does not license
"procedural training transfers."** The claim the measurement supports is
the narrow one — the missing shading-structure coverage was the
*mechanism* behind the transfer gap, and closing it recovers most of the
gap. That is domain randomisation informed by a measured gap, the
strictly weaker thing the FDR's own future-work section already
describes. The looser sentence is exactly what a reader would take away
by default, so the narrow one is stated in the same breath as the
number, not in a footnote.

### 3. `obstruction` parity is an artefact, not transfer evidence

`world.build_obstructions` has a single call site and procedural scenes
execute the same bytes CAD scenes do, so procedural-versus-CAD parity on
`obstruction` would hold under *any* hypothesis, including a model that
generalises not at all. Both places the per-class table appears
(`FDR_v3.md` §13.1.1, `NEXT_STEPS.md`) now say so directly beneath the
table, in the terms a would-be quoter needs: the row is not a transfer
result and must not be cited as one; the load-bearing rows are `bay`,
`cartridge` and `battery`. `README.md`'s new transfer section carries
the same warning.

### 4. τ was retired in prose but not in code

`plan/placement_area.py` kept evaluating `if iou < self.tau: raise
PlacementDisagreement` after commit `8744947` retired it in the
documentation. Three inconsistent values were simultaneously live —
constructor default 0.85 (what every in-tree caller got),
`configs/planning.yaml`'s `arbitration.tau: 0.7492` (read by nothing),
and `README.md`'s 0.5715 (which described the YAML value as live) — and
at the project's own `mm_per_px: 0.38` the gate admitted **0 of 8**
plannable cartridges, silently. Deleted in `cdd97fc` along with the dead
config key; both rows now read 8 of 8.

Every live-τ description was removed or explicitly marked historical:

- `FDR_v3.md` §10.6 described `placement_disagreement` as firing on "a
  calibrated IoU threshold τ" and said no rate was quoted because τ was
  "calibrated but not yet informative" — both corrected; the counter now
  reads zero by construction, not by configuration.
- `FDR_v3.md` §13.2.1's "the extractor still defaults to the
  pre-calibration τ = 0.85" sentence is struck through with a pointer to
  a new paragraph recording the deletion and its measured cost.
- `FDR_v3.md` §13.2.1's τ-calibration paragraph, written in the present
  tense while the gate was live, is marked as such.
- `NEXT_STEPS.md` item 4's closing paragraph is struck through and
  replaced; item 3's τ = 0.7492 / 0.85 reference is marked historical;
  Step 5's "delete the `tau` config key" bullet is marked DONE.
- `README.md` already said "There is no `tau`" correctly and needed no
  change on this point.

A sweep for live-τ phrasing across all three documents returns only
struck-through text, explicit historical markers, and the record of the
retirement itself.

### 5. The segmenter is now in the end-to-end loop

Before `f40cc1b` `main.py` ran only the heuristic extractor and the
segmenter was unreachable under *any* configuration —
`recog.inference.load_detector` took no `segmenter` argument and
`_build_planner` hardcoded the heuristic. Any statement that the
pipeline demonstrated the segmenter end to end was, before that commit,
**overstated**; it is true now, of `configs/demo_seg.yaml`, receipted at
`docs/receipts/main_seg_run.txt` (26 detected → 26 segmented → 8
placement areas → 1 pick-and-place).

`FDR_v3.md` §8 now says which extractor the loop runs and states the
before/after explicitly; §8.1 gains the `demo_seg.yaml` invocation
beside the existing smoke test; §13.2.1's Status paragraph records it.
`NEXT_STEPS.md` and `README.md` carry the same. In all three the
torch-free `configs/demo.yaml` path is stated as **unchanged** and as
what the reproducibility claim rests on, with an explicit instruction
not to move that claim onto the segmentation path — and the receipt is
labelled as evidence about the *wiring*, since those frames are the
segmenter's own training corpus.

### 6. FDR §6.3.1 was partial on packing

§6.3.1 presents FFDH as *the* planner packer. FFDH never scans its shelf
origin in y, and `_next_free_x` collapses a shelf's whole row band, so a
forbidden region in the first shelf's band kills the entire pack —
`scene_00005` handed the packer a 93 %-free grid containing a clear
112 × 48 mm rectangle and got zero cells. The planner now calls
`common.packing.pack_best_effort`, which competes FFDH against two
obstacle-tolerant arms and takes the maximum, so `best ≥ FFDH` holds by
construction. Real frames **8 → 17 cells**.

§6.3.1's pseudocode is accurate *for FFDH* and the section's own
measurements are unmoved, so the section was not rewritten. A scope note
was added opening it, saying precisely that: accurate about FFDH,
incomplete about the planner. Pointers were added at §6.3, the executive
summary and §13.1(3) — all three quoted FFDH's headroom as the packer's
— and the Appendix E O3.c row now names the arm set that ships.
`README.md` gains a "How the packer picks" section and loses the stale
claim that "FFDH declines most of these bays".

### 7. The sim-to-real limitation, written at last

New **`FDR_v3.md` §13.2.2** — "Limitation: sim-to-real transfer is
unvalidated, and cannot be validated under this project's constraints".
It states that real photographs are unobtainable (owner-confirmed, not a
scheduling gap); that transfer is therefore not measured and *cannot* be
measured here, which is a different claim from "not yet"; that every
segmentation, placement-area and packing-on-real-masks figure in the
report is synthetic-to-synthetic, listing which ones; that the three
real-photo figures (0.211 / 0.232 / 0.318 against the heuristic's 0.217,
n = 20 cartridges over 6 images, boxes only, no polygons) are a smoke
test whose run-to-run variation exceeds the effect it exists to detect,
and are not a transfer claim in either direction; what stands in for the
measurement (cross-distribution generalisation, domain randomisation)
and how much weaker each is; and what the limitation does **not** touch
— the box-level recognition results, §6.3.1's packing contribution, and
§10.3's separately-stated hardware non-validation.

§13.2's future-work list is signposted to it so items (1) and (4) can no
longer be read as work this project deferred, and the three places in
§13.1.1 and §13.2.1 that pointed at item (4) as pending now say it is
unreachable. `NEXT_STEPS.md`'s standing note that the FDR "does not yet
carry" such a statement is struck through and resolved. `README.md`
states the constraint before any segmenter figure it quotes.

---

## Verification

- Every figure written was checked against its source before use.
  `docs/receipts/seg_eval_anchored_on_cad_test.txt` and
  `seg_eval_anchored_crown_on_cad_test.txt` were read directly to confirm
  the pooled per-class rows, the selected means, the 836/213/128/145
  instance counts and the per-SKU blocks;
  `docs/receipts/forbidden_bench.txt` to confirm 14.28 / 14.55 and the
  10–15 % coverage rows; `docs/receipts/main_seg_run.txt` to confirm
  26 / 26 / 8 / 1.
- `plan/`, `common/`, `configs/` and `scripts/` were read (never
  written) to confirm that τ is gone from `plan/placement_area.py` and
  `configs/planning.yaml`, that both planner call sites use
  `pack_best_effort`, and that `recog/synth3d` still calls
  `first_fit_decreasing`.
- **No figure was found that contradicted its own receipt.** Two
  supporting claims were found stale rather than wrong and are corrected
  in the text, not silently: §6.3.1's "the generator does not render PCB
  components inside cartridge interiors", which described the
  pre-tray-interior generator (the 7 capable real instances carry
  3.1–19.3 % forbidden coverage), and `NEXT_STEPS.md`'s test count of
  621, now 708.
- Every changed claim is marked as superseded rather than silently
  replaced — strikethrough plus a dated correction, or a "corrected /
  superseded on measurement" block — so a reader who remembers the old
  number sees that it moved and why.

## Judged out of scope, and why

- **Re-measuring anything.** Δcells (+0.032 mean, 2 of 126 negative) was
  measured under the FFDH-only planner and has not been re-run against
  `pack_best_effort`. That is flagged in `NEXT_STEPS.md` item 3 as work,
  not quietly left to read as current. *(Both figures are at the nominal
  0.6250 mm/px and were corrected on 2026-08-12 to +0.056 and 5 of 126 —
  see `2026-08-12-fix-delta-cells-scale.md`. The packer half of this note
  still stands: it is still FFDH under there.)*
- **Restructuring `FDR_v3.md`.** It is a submitted report. §6.3.1's
  pseudocode, §13.2.1's τ measurements and §6.3.1's benchmark tables are
  all still accurate about what they measure and were left in place;
  corrections were added around them.
- **Source changes.** Two known documentation-drift items in code
  comments remain: `recog/seg_dataset.py`'s module docstring and
  `recog/seg_evaluate.py`'s receipt output still assert that the
  `cartridge` and bay crop populations are disjoint, which the tray-
  interior fix made false. Both were already recorded as follow-ups in
  `FDR_v3.md` §13.2.1 and `NEXT_STEPS.md`; this pass is documentation
  only and did not touch them.
- **`docs/FDR.md` and `docs/FDR_v2.md`.** Superseded earlier revisions,
  left as history; `README.md` now says so and points at `FDR_v3.md`,
  which is what all its `§` references mean.
