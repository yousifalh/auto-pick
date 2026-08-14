# Results summary for Professor Svetan Ratchev FREng — build report

**Date:** 2026-08-14 · **HEAD:** `83a1383` · **Artefact:** `docs/RESULTS_SUMMARY.md`
**Constraint honoured:** only `docs/RESULTS_SUMMARY.md` and this file were
written. `README.md`, `docs/FDR_v3.md` and `docs/MODEL_CARD.md` were read but
never edited (a concurrent agent owns those). Staging used explicit paths only.

---

## 1. Framing inputs

`docs/superpowers/audit/2026-08-14-W-omnifactory-context.md` was read in full
first and determined the framing. Specifically honoured:

- Addressed **Omnifactory Mini** by its declared envelope (≤ 4 kg, ≤ 500 mm
  x/y/z) and its named use case (aerospace battery pack assembly), claiming the
  *class* of problem and explicitly not the identity.
- Opened the substance with the **certifiability geometry**, which is immune to
  every stated limitation, and closed the offer section on **frugal industrial
  AI** (Martínez-Arellano & Ratchev) as the problem the pipeline answers.
- **"Digital twin" is never used unqualified** — a dedicated short paragraph in
  §5 says the project's is a cell-level occupancy model in `plan/scene.py` and
  is not a plant-level twin.
- No mention of MBSE, knowledge graphs, aerostructures or large-part assembly.
- Title given as Professor Svetan Ratchev FREng throughout.

## 2. Section structure used

1. **What it set out to do, and where it landed** — brief; the six-criteria
   tally (3 pass, 1 pass-with-withdrawal, 1 fail, 1 untested) carries it.
2. **The finding: an exact fit cannot be certified by a camera** — the opener of
   substance, as instructed. Geometry first, the n = 10 measurement second and
   labelled as such, "verify contact by force, not by camera" as the
   most-transferable sentence.
3. **Results** — perception in-domain, perception cross-distribution, boundary
   accuracy/latency, planning, reproducibility, and a final paragraph on the two
   figures that moved *downward* on evidence (the mm/px scale correction and the
   retirement of τ).
4. **What this project cannot claim** — no robot, no five-class real-photograph
   ground truth, everything synthetic-to-synthetic, and the constraint declared
   permanent rather than deferred. Ends by measuring the project against the
   facility's own de-risking mission and conceding it sits below that rung.
5. **What I think is worth offering Omnifactory Mini** — the CAD-to-labelled-data
   pipeline, with the non-novelty of the idea stated in bold.
6. **The ask — the one experiment I could not run** — the protocol, converter and
   validator, all tested, never run on a photograph.
7. **Receipt index** — a 15-row table mapping every claim to its artefact.

Length: ~2,300 words / roughly three pages of markdown.

## 3. Headline figures, each verified against its receipt in this session

Every figure was read out of `docs/receipts/` directly. **No figure was copied
from `README.md`, `docs/FDR_v3.md`, `docs/MODEL_CARD.md`, `PORTFOLIO.md` or any
spec.** The four load-bearing ones:

| Figure | Scope as stated in the document | Receipt |
|---|---|---|
| **bay IoU 0.8755** | procedural-only training set, no CAD; scored on the 836-crop held-out CAD test split (434 frames, 4 SKUs); synthetic-to-synthetic | `seg_eval_anchored_crown_on_cad_test.txt` |
| **bay IoU 0.6555** (same recipe, one generator field removed) and **0.9032–0.9131** (four CAD-trained leave-one-SKU-out controls) | identical 836-crop split | `seg_eval_anchored_on_cad_test.txt`, `seg_eval_cad_control_*_on_cad_test.txt` |
| **mAP@0.50 0.9053** | shipped detector at its 0.70 operating threshold, 150 held-out **synthetic** frames, 1 205 GT boxes | `detector_bench.txt` arm 3 |
| **mAP@0.50 0.8484** | 6 of 7 handheld phone photographs, 80 boxes, 2 classes, 1 image excluded for zero GT | `real_photo_eval.txt` |

Supporting figures, all re-read from receipts: 1.226 mm bay boundary
displacement at 3.14× margin and 16.6 vs 57.1 ms at 8 cartridges
(`seg_eval.txt`); 437.4 ms median CPU detector latency (`frcnn_latency.txt`);
+3.25 cells at 5 % forbidden coverage, paired t 15.18, 40/40
(`forbidden_bench.txt`); +24.2 % / +57.1 % / 0.0 % rotation ablation
(`ffdh_ablation.txt`); 1 210 passed / 2 skipped and 93 % / 67 % branch coverage
(`pytest-cov.txt`); same-seed bit-identical weights
(`seed_reproducibility.txt`); τ's fail budget never binding
(`tau_calibration.txt`); 0.318 vs 0.217 placeable fraction at n = 20
(`seg_ablation.txt`).

**The certifiability geometry was re-derived independently rather than
relayed.** `recog/synth3d/assets/catalog.json` gives
`AnkerPowerCore10000` `interior_mm` [-27.45, -43.0, 27.45, 41.45] and
`module_bay_mm` [-27.45, 22.0, 27.45, 41.45] → a placement region of exactly
**54.9 × 65.0 mm**; `configs/planning.yaml` gives `diameter_mm: 18.5`,
`length_mm: 65.0`. All four SKU margins were recomputed (+0.00 / +1.75 / +70.20
/ +75.75 mm) and `L(θ) = (65.0 − 18.5·sinθ)/cosθ` evaluated (64.687 mm at 1°;
18.5·tan 1° = 0.3229 mm). The FDR's rounded "+75.80" for the 26800 is 75.75 on
the committed catalogue; the document uses 75.75.

## 4. Framing of the offer and the ask

**Offer** — the CAD-to-labelled-synthetic-data pipeline, addressed to Mini as a
high-mix demonstrator where every new product is a new annotation campaign, and
positioned against the facility's own changeover-cost thesis ("the
reconfigurable floor already lets the cell change product; the perception stack
still cannot"). The non-novelty is conceded in bold — *"Synthetic training data
for manufacturing vision is a crowded field and I am not claiming the idea"* —
with the artefact, the rigour and the measured limits offered instead. Three
concrete hooks: the scale guard in `convert_cad.py` and why it exists, the
object-index pass giving zero-overlap masks by construction, and the
`manifest.json` that *is* the resolved generator config. It notes the input is
STEP CAD an MBD shop already holds.

**Ask** — framed as the exact inverse of §4: *"you can photograph your parts,
and I could not."* Names the three existing, tested artefacts
(`docs/ANNOTATION_PROTOCOL.md`, `recog/labelme_to_seg.py`,
`recog/check_annotations.py`, the latter two at 82 % and 93 % branch coverage)
and states that the experiment's result is publishable in either direction —
explicitly arguing that a negative result is worth more to a de-risking testbed
than another positive-only report.

The "revised its headline downward on evidence" quality is carried structurally
rather than asserted: the O2 failure and the withdrawn O3 distribution appear in
§1's tally, the mm/px correction and τ's retirement close §3, and §4 concedes
the ceiling before §5 makes any offer. The closing line of §6 is the only place
it is even gestured at ("including the figures that got worse when they were
measured properly").

## 5. Cut for being unsourceable

- **"21.8 % → 2.6 % hallucinated `bay` on sealed cartridges"** and the
  **"0.9009 / 0.880 arithmetic ceiling"**. Both appear in `README.md`,
  `PORTFOLIO.md`, `MODEL_CARD.md`, `NEXT_STEPS.md` and the sealed-unit spec, but
  **grep finds neither in any file under `docs/receipts/`**. The receipted half
  of the same experiment — bay IoU 0.6555 → 0.8755 on the identical held-out
  split — carries the point, so the document uses only that.
- **A render-time-per-dataset figure.** No manifest carries an elapsed time. The
  "~8 GPU-hours" is attributed to `docs/datasets/README.md` (the data card) in
  the receipt index rather than presented as a receipt, and "hours, not an
  annotation campaign" is stated as a property of the workflow, not as a timing
  measurement.
- **Any end-to-end throughput claim.** `main_seg_run.txt` records 26 cartridges
  detected, 7 placement areas and **1 cell placed** in one cycle; the mechanism
  (one pose per cycle, plus reachability) is documented, but the figure would be
  read as a system-performance number and it is not one. Omitted rather than
  explained at length.
- **The 3 ms queue-rebuild median.** Withdrawn in the FDR (its benchmark
  artefact was never committed); §1 records the withdrawal and quotes no number.

## 6. Things the author should reconsider before sending

1. **The 10-of-10 measurement has no `docs/receipts/` artefact.** It lives in
   `docs/superpowers/specs/2026-08-11-placement-feasibility.md`. §2 says so
   plainly, but a receipt generated by a committed script — the same treatment
   every other claim in the project gets — would make the strongest section in
   the document as auditable as the weakest. This is the single highest-value
   hour of work before sending.
2. **`recog/realtest/` contains seven photographs of real hardware.** §4 and §6
   both reference them. If any collaboration follows, decide in advance whether
   those images can be shared and under what terms.
3. **The document does not name a specific grant or programme**, per the
   research context's warning that Gateway to Research shows no currently-active
   award. If the author has better information about a live programme, that is a
   deliberate omission worth revisiting — but the safe default was taken.
4. **Consider whether §3's cross-distribution result wants one sentence on
   *why*.** The mechanism (flat synthetic lids lack the shading structure a real
   sealed lid has, so the model learns "no shading ⇒ open bay") is the most
   quotable insight in the whole result and is currently compressed into a
   parenthesis. It was kept short for length; one more clause would earn its
   space with this reader.
5. **Check the covering email does not re-inflate anything the document
   deflates.** Every hedge in §4 is load-bearing, and a cover note that opens
   with "a working autonomous kitting cell" would undo all of it in one line.
