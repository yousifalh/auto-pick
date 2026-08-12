# Audit J — Claim verification against `docs/FDR_v3.md`

**Date:** 2026-08-12 · **HEAD:** `39429a4` · **Suite:** 1074 collected, `pytest` exit 0 (verified by execution)
**Scope:** every numbered requirement, every traceability row, and every substantive
runtime/number/safety claim in `docs/FDR_v3.md` (3 952 lines). Read-only; nothing changed.

Each verdict below records whether it was reached **by executing** (running a test, running
the shipped code, recomputing from a committed receipt) or **by reading** (source
inspection, grep). Where a claim is false or stale, the honest version is stated.

---

## Headline

**16 claims could not be substantiated as written: 5 false, 5 stale, 6 unsubstantiated.**

Set against roughly 20 substantiated claim clusters, including every figure in §6.3.1,
§13.1.1, §13.2.1, §8 and §3 — those reproduce from committed receipts exactly, several to
four decimal places. The report's *dated corrections* (2026-08-11 / 2026-08-12) are
accurate: I checked the heartbeat, the E-stop escalation, the mock-server latch, the
planner reservations and the CRC/framing verification, and all of them say what the code
does. **The unreliable material is the older, uncorrected prose — §5, §10.6, §11.2 and
Appendix B — and one traceability row (O2) that has the same defect as the O4.b row
already found.**

The single most consequential finding is F1: a headline empirical claim, stated three
times including in the executive summary, that I falsified by running the shipped code.
The report's own §10.6 table already contains the contradicting number.

---

## Verdicts by class

| # | Claim (location) | Evidence it rests on | Verdict |
|---|---|---|---|
| **F1** | "The heuristic detector achieves **100 % precision**… its failure mode is exclusively recall, not noise" (§1 finding 2); "produced **zero false positives** across all 100 frames — when it fires it is right 100 % of the time" (§10.6); "What surprised: the heuristic detector's *zero false positives*" (§13.3) | none — no receipt reports a precision or FP count | **FALSE** (executed) |
| **F2** | "production now defaults to torchvision's anchors with the custom set kept behind a flag" (§10.7); "the recommended default is the torchvision scheme" (§5.7); "Adopt torchvision defaults as production; retain custom anchors behind a config flag" (ADR-005) | `recog/model.py`, `configs/recognition.yaml` | **FALSE** (read) |
| **F3** | "The executor bounds vacuum dwell to **5 s** by construction" (§11.2) | none | **FALSE** (executed grep + read) |
| **F4** | "Confidence threshold defaults to **0.5**; NMS threshold **0.3**, per class" (§5.1) | `recog/model.py`, `configs/recognition.yaml`, and Appendix B — which gives 0.70 / 0.4 | **FALSE** (read) |
| **F5** | "any cell can be re-verified by re-running the named script against the committed source tree" (Appendix E, closing) | contradicted by Appendix C.3 in the same document | **FALSE** (executed) |
| **S1** | §9.3 coverage table (`common/ 149·91 %`, `recog/ 303·78 %`, `plan/ 395·92 %`, `execution/ 295·84 %`, `Total 1142·86 %`), "Receipt at `docs/receipts/pytest-cov.txt`"; "102 passing tests" (§1, Appendix C) | `docs/receipts/pytest-cov.txt` — contains none of these figures | **STALE** (executed) |
| **S2** | §5.3 augmentation list, §5.4 training hyperparameters and anchor values, Appendix B's `recognition.yaml` defaults | `configs/recognition.yaml`, `recog/augmentation.py` | **STALE** (read) |
| **S3** | "Latency rises from ~0.9 ms to **3.4 ms mean / 4.6 ms worst** on bench masks" (§6.3.1); "peaks at 1.14 ms at 15 %" | `docs/receipts/forbidden_bench_timings.csv` | **STALE** (executed) |
| **S4** | "**~2,750 lines** of production Python (non-blank, non-comment) across four modules" (§11.3) | none | **STALE** (executed) |
| **S5** | Appendix B's `configs/planning.yaml` field list, incl. `packing.algorithm ("ffdh")` | grep over the tree | **STALE** (executed) |
| **U1** | **O2 — "Centroid error ≤ 2 px · Pass"** (§3, §10.5, Appendix E), cited to `tests/test_evaluate.py` | that file | **UNSUBSTANTIATED** (read + executed grep) |
| **U2** | "FFDH runs in **under 0.2 ms p95** (200×150 mm, 80 items)… two orders of magnitude under the 8 ms budget" (§1, §10.2, §13.1(3), ADR-002) | Figure 4 left panel; no receipt | **UNSUBSTANTIATED** (executed) |
| **U3** | **O3** planning distribution "mean 5.0 / median 3.0 / **p95 13.0 ms**" (§10.4), "Pass — 3 ms median" (§10.5), cited to `bench_cycles.py` | `bench_cycles.py` | **UNSUBSTANTIATED** (executed) |
| **U4** | "(LOW_IOU) **50**" row (§10.6) | `heuristic_failure_taxonomy.json` | **UNSUBSTANTIATED** (executed) |
| **U5** | Heuristic perception "mean 3.2 / median 3.0 / p95 4.1 ms" (§10.4) | three committed receipts give three different values | **UNSUBSTANTIATED** (executed) |
| **U6** | "The cartridge PCB subtraction **prevents any** place pose over an exposed busbar" (§11.2) | contradicted in degree by §13.2.1 | **UNSUBSTANTIATED** (read) |

---

## The false claims, with the honest version

### F1 — the heuristic's "zero false positives" (§1, §10.6, §13.3)

I ran the shipped `recog.inference.HeuristicDetector` over all 100 images of
`recog/dataset/`, matched greedily by confidence at the **IoU ≥ 0.5 criterion §10.6 itself
declares**, and counted:

```
GT total 822  battery 630  cartridge 192   |  detections 431
battery:   gt=630  tp=291  fp= 9   precision=0.9700  recall=0.4619
cartridge: gt=192  tp= 90  fp=41   precision=0.6870  recall=0.4688
ALL:       tp=381  fp=50           precision=0.8840  recall=0.4635
mAP: {'AP_1': 0.4466, 'AP_2': 0.3475, 'mAP@0.50': 0.3971}
```

The GT counts (822 = 630 + 192), the ~47 % match rate and the mAP figures reproduce
§10.1 and §10.6 exactly — the section is right about everything except this.

There are **50 false positives**, not zero. Overall precision is **0.884**, and on
cartridges it is **0.687**. The report already holds this number: §10.6's own table
carries a row reading `(LOW_IOU) | 50 | separate — see below`, described as "detections
that fired correctly but produced a bounding box too loose to count at IoU = 0.5." Under
the criterion the section declares, a detection that matches no ground truth at IoU 0.5
*is* a false positive. The 50 were counted, relabelled, moved to a "separate" row, and
then denied in the next paragraph. §10.1's own description of Figure 8 —
"~95–100 % on batteries and ~75–90 % on cartridges" — is the accurate statement and
contradicts the headline within the same document.

**Honest version:** *"The heuristic's precision is 0.97 on batteries (9 false positives)
and 0.69 on cartridges (41 false positives) across the 100-frame set — 0.88 overall.
Its dominant failure is recall (46 %), but it is not noise-free: its cartridge errors are
merged mega-boxes that fire on real foreground and miss the IoU threshold. The
'high-precision safety net' framing holds for batteries and does not hold for
cartridges."* §1's finding (2) and §13.3's "what surprised" paragraph both need rewriting;
the LOW_IOU row should be renamed and folded into a precision figure.

### F2 — "production defaults to torchvision anchors" (§5.7, §10.7, ADR-005)

`recog/model.py::build_fasterrcnn` **unconditionally** constructs a custom
`AnchorGenerator`:

```python
ratios = tuple(float(r) for r in model_cfg.get("anchor_ratios", [0.33, 0.5, 1.0, 1.5, 2.0]))
base_scales = tuple(int(s) for s in model_cfg.get("anchor_scales", [4, 8, 16, 32]))
...
anchor_generator = AnchorGenerator(sizes=per_level_scales, aspect_ratios=(ratios,) * len(per_level_scales))
```

There is no flag, sentinel or code path that selects torchvision's default generator —
omitting the config keys falls back to the *k-means* set, not the default one. And the
shipped `configs/recognition.yaml` specifies a **third** set, retuned for the Blender
corpus: `anchor_ratios: [0.28, 0.5, 1.0, 2.0, 3.5]`, `anchor_scales: [40, 64, 96, 144]` —
neither the PPR k-means set nor torchvision's `[32,64,128,256,512] × [0.5,1,2]`.

The **ablation itself is sound** — `docs/receipts/frcnn_map_default.txt` is headed
"ablation: torchvision DEFAULT anchors (no custom k-means)" and gives 0.8736 / 0.9053 /
0.8419 / 0.5831, matching §10.1 and §5.7 exactly. Only the disposition claim is false.

**Honest version:** *"The ablation showed torchvision's defaults beat the PPR's k-means set
on the April cv2 corpus (0.874 vs 0.764). Production did not adopt torchvision's defaults;
it re-tuned the custom anchor set against the current render corpus
(`configs/recognition.yaml`, with the sweep recorded in that file's comments). The
torchvision-default configuration is not selectable in the shipped code."*

### F3 — "The executor bounds vacuum dwell to 5 s by construction" (§11.2)

`grep -ri dwell` over `*.py`, `*.yaml`, `*.src` and `*.xml` returns **nothing**.
`execution/krl_prog/routines.src` turns the vacuum on at step 2 and off at step 5 with no
timer between them (`WAIT SEC 0.05` and `WAIT SEC 0.03` are settle waits, not bounds).
`execution/execution.py` has no dwell timer. Its own `_emergency_stop` docstring states
the opposite case explicitly:

> "the arm may be holding a cell with the vacuum on and nothing can tell it to let go."

This is a cell-safety claim, in the section that lists "never retain a cell under vacuum
beyond the specified dwell" as an ethical obligation. The nearest 5 s in the system is
`command_timeout_ms: 5000`, which bounds one status wait, not a grasp.

**Honest version:** *"Nothing bounds vacuum dwell at either end. A dwell limit would have
to be enforced controller-side in `routines.src`; it is not implemented, and the
host-initiated E-stop of §7.2 is the only mechanism that would drop the vacuum — and only
while the host is alive (§7.5)."*

### F4 — detector thresholds (§5.1)

§5.1: "Confidence threshold defaults to 0.5; NMS threshold 0.3, per class."
`recog/model.py`: `box_nms_thresh=float(model_cfg.get("nms_iou", 0.4))`,
`box_score_thresh=float(model_cfg.get("confidence_threshold", 0.70))`.
`configs/recognition.yaml`: `nms_iou: 0.4`, `confidence_threshold: 0.70`.
Appendix B lists 0.4 and 0.70 — so the report contradicts itself.
**Honest version:** confidence 0.70, NMS IoU 0.4.

### F5 — Appendix E's re-verifiability claim

Appendix E closes: "The matrix is generated from the same receipts referenced inline
throughout §10 and Appendix C so **any cell can be re-verified by re-running the named
script** against the committed source tree." Appendix C.3, in the same document, states
that sixteen of thirty-four receipts have no surviving tool. Two matrix cells name
artefacts that cannot be re-run at all: **O3 cites `bench_cycles.py`**, which is not in
the tree and, by `git log --all -- bench_cycles.py`, was never committed; **O1.a cites
`pr_summary.txt`**, whose generator `pr_curves.py` was likewise never committed.
**Honest version:** *"Nine of the eleven cells can be re-verified from committed tooling.
O3's artefact (`bench_cycles.py`) is not in the repository and never was; O1.a's receipt is
committed but its generator is not."* (Overlaps U3 — count once.)

---

## The stale claims, with the honest version

### S1 — §9.3's coverage table no longer matches its own receipt

§9.3 presents `common/ 149 · 91 %`, `recog/ 303 · 78 %`, `plan/ 395 · 92 %`,
`execution/ 295 · 84 %`, `Total 1142 · 86 %`, and says "Receipt at
`docs/receipts/pytest-cov.txt`." The receipt contains **none** of those numbers: it reports
1032 tests, `65 %` over 6 897 statements (49 modules) and `89 %` over 1 604 statements (the
18-module scope). Appendix C.2 discloses the currency and is correct that the **86 %
conclusion survives** (89 % on the scope O6 was written against). But §1, §9.3 and §10.5
carry no marker, and §9.3's per-module table is a snapshot the cited receipt no longer
holds. §1's "102 passing tests" likewise describes a suite that now stands at 1074
(verified: `pytest --collect-only` → 1074, run exit 0).
**Honest version:** annotate §9.3 "as measured 2026-04-20 (102 tests, 1 142 statements);
superseded — see Appendix C.2", the same treatment Appendix E's O6 row already gives.

### S2 — §5 and Appendix B describe a configuration the repository no longer ships

Measured against `configs/recognition.yaml` and `recog/augmentation.py`:

| FDR says | Repository ships |
|---|---|
| `training.epochs` 60 | **35** |
| `training.batch_size` 4 ("imposed by GPU memory") | **2** (fragmentation OOM, per the config comment) |
| `frozen_bn_epochs` 20 | **8** |
| anchors `[0.33,0.5,1.0,1.5,2.0] × [4,8,16,32]` (§5.4) | **`[0.28,0.5,1.0,2.0,3.5] × [40,64,96,144]`** |
| brightness/contrast ±0.40 | **0.55 / 0.50** |
| gamma `[60,140]` | **`[45,190]`** |
| saturation ±25 | **±30** |
| "additive Gaussian noise (variance ∈ [10,50])" | `GaussNoise(std_range=(0.02, 0.14))` |
| "composes **seven** operations … and horizontal flip at p=0.5" | also `VerticalFlip`, `RandomRotate90`, `Affine` (translate ±4 %, scale ±20 %, rotate ±4°), `MotionBlur`/`Defocus`, `ISONoise`; train `min_visibility=0.25` |

The val transform's `min_visibility=0.01` (§9.6) **is** correct and still in the code.
Internally, §5.7's "the custom set's scales `[4, 8, 16, 32, 64]` px" also disagrees with
§5.4's four-element list in the same document.

### S3 — §6.3.1's `pack_best_effort` latency

"Latency rises from ~0.9 ms to **3.4 ms mean / 4.6 ms worst** on bench masks — still inside
the 8 ms O3 budget." Recomputed from the committed
`docs/receipts/forbidden_bench_timings.csv` (240 rows, `us_best`):

```
mean 2.83 ms   p50 2.69   p90 3.65   p95 3.93   p99 5.35   max 7.43 ms
per-level max: 0.0%→5.42  2.5%→3.21  5%→4.15  10%→4.91  15%→3.99  25%→7.43 ms
```

No summary statistic of that column is 3.4 or 4.6. The **worst observed pack is 7.43 ms**,
leaving **0.57 ms** of the 8 ms O3 budget rather than the 3.4 ms the text implies.
(`forbidden_bench.txt`'s per-level means are 2.29–3.51 ms.) Similarly, "the aware arm …
peaks at **1.14 ms** at 15 %" against a receipted 1.05 ms.
**Honest version:** *"mean 2.8 ms, p95 3.9 ms, worst observed 7.4 ms on bench masks — inside
the 8 ms O3 budget, but with 0.6 ms of margin at the worst mask measured, not two orders of
magnitude."* This matters more than the arithmetic: it is the only figure that says how
close the shipping packer runs to the budget it is certified against.

### S4 — "~2,750 lines of production Python" (§11.3)

Counted today (non-blank, non-comment):

```
18 modules the coverage receipt scopes : 3 852
common/ plan/ recog/ execution/ excl. synth3d + main.py : 10 692
including recog/synth3d/ : 15 974 (52 files)
```

Unlike the coverage figure, this one carries no currency note anywhere. The sustainability
argument ("the software is sustainable by design: ~2,750 lines…") is built on it.

### S5 — Appendix B lists `configs/planning.yaml` keys read by nothing

`grep` over the whole tree finds **no Python consumer** for `packing.algorithm`,
`packing.rotation_allowed`, `packing.worst_case_bound`, `queue.fill_order`,
`queue.assignment`, `cartridge.morph_close_ksize`, `cartridge.morph_open_ksize`,
`cartridge.pcb_exclusion_required`, or `cartridge.green_channel_thresh`. Five of these are
listed in Appendix B as live configuration fields. `packing.algorithm: ffdh` is the worst
of them: both planner call sites run `common.packing.pack_best_effort` (verified in
`plan/planner.py:550` and `plan/bin_packing.py:100`), so the key declares the wrong
algorithm and changing it changes nothing.

This is the **same defect class** the report itself corrected for `configs/execution.yaml`'s
five `motion:` keys on 2026-08-12 — "worse than a missing key because it looks live."
`planning.yaml` was not given the same pass. (`safety_margin_px`,
`occupancy_grid.resolution_mm_per_cell`, `camera.mm_per_px_x` and
`camera.workspace_bounds_mm` **are** read; those four are fine.)

---

## The unsubstantiated claims, and what would settle each

### U1 — **O2 is the O4.b defect repeated** (§3, §10.5, Appendix E)

O2's threshold is "Centroid localisation error shall not exceed 2 px", verdict
**Pass (in-domain)**, evidence `tests/test_evaluate.py` in all three places it appears.
That file contains no measurement of any detector's centroid error. Its two centroid tests
are unit tests of the metric function:

```python
def test_centroid_error_zero():
    b = BBox(10, 10, 20, 20); assert centroid_error_px(b, b) == 0.0

def test_centroid_error_magnitude():
    a = BBox(0, 0, 10, 10); b = BBox(3, 4, 13, 14)   # centres differ by (3, 4)
    assert centroid_error_px(a, b) == pytest.approx(5.0)
```

The second **asserts a 5 px error as correct** — the metric is being exercised at 2.5× the
threshold the row certifies. Beyond this file, `centroid_error_px` has **no caller anywhere
in the repository** (grep-verified: only its definition in `recog/evaluate.py`, its
`__all__` entry, and these two tests), and **no receipt in `docs/receipts/` contains the
word "centroid"** (grep-verified). `evaluation.centroid_error_target_px: 2.0` in
`configs/recognition.yaml` is read by nothing.

This is exactly the pattern already found at O4.b: the cited test would pass unchanged if
the criterion were violated, and would pass unchanged if the criterion were deleted. It is
arguably worse than O4.b, because O2 is a *headline* objective reported as fully met in
§1's "Four of six numbered project objectives are fully met (centroid error ≤ 2 px, …)".

**What would settle it:** run `recog.evaluate.centroid_error_px` over the IoU-matched
detections of the default-anchor checkpoint on the 15-image val split, receipt the
distribution, and cite that. Until then O2's honest verdict is **Not measured**, not Pass.

### U2 — the FFDH latency headline (§1, §10.2, §13.1(3), ADR-002)

"under 0.2 ms p95 (200×150 mm strip, 80 candidate items)"; "median 12 µs at n=10 rising to
75 µs at n=80, with the p95 never exceeding 0.12 ms"; "two orders of magnitude of headroom".
No receipt in `docs/receipts/` contains an n ∈ {10, 20, 40, 80} sweep. `ffdh_ablation.csv`
is a rotation × strip-size sweep at n = 40, and its own p95 column reaches 121.4 µs — above
the quoted 0.12 ms ceiling. The figures rest on Figure 4's left panel, and per Appendix C.3
no plotting script for any of the nine figures exists in the tree
(`grep -rn "matplotlib\|savefig" --include=*.py` finds nothing outside `tests/`).
**What would settle it:** a committed benchmark emitting the n-sweep, as
`scripts/forbidden_bench.py` already does for §6.3.1.

### U3 — O3's planning distribution (§10.4, §10.5, §8, abstract, ADR-004)

"mean 5.0 / median 3.0 / **p95 13.0 ms**", cited to `bench_cycles.py`. That file is absent
and `git log --all -- bench_cycles.py` returns nothing — it was never committed. The
"3.0 ms median for planning" in the abstract and §8, and ADR-004(b)'s "planner cycle time
dropped to a 3 ms median", inherit the same gap; so does §10.4's "the planner's FFDH-only
path runs in under 2 ms" and ADR-004's "4.9 ms median" extractor cost.

**O3 is not unsupported** — the 8 ms budget is independently exercised by
`tests/test_planner.py::test_segmentation_extract_arithmetic_stays_under_the_o3_budget`
(passes) and by §13.2.1's receipted 2.0–2.2 ms per cartridge — but the specific
distribution, including the p95 of 13 ms that the section uses to concede the budget is
exceeded on cold start, has no artefact.
**What would settle it:** commit the cycle benchmark and receipt it.

### U4 — the LOW_IOU count (§10.6)

`docs/receipts/heuristic_failure_taxonomy.json` carries exactly five keys — `EDGE_CLIP`,
`AREA_FLOOR_CART`, `AREA_FLOOR_BAT`, `OCCLUSION` (95), `RULE_FAIL` (344). There is no
`LOW_IOU` key and no 50 anywhere in the file. My own re-measurement gives 50 unmatched
detections, so the number is right — but it has no receipt, and per F1 it is the
false-positive count under a different name.

### U5 — the heuristic's perception latency (§10.4)

Three committed artefacts give three values for one measurement:

| source | median | p95 |
|---|---:|---:|
| §10.4's table | 3.0 ms | 4.1 ms |
| `docs/receipts/frcnn_latency.txt` (which cites "§10.4" by name) | **3.3 ms** | **5.5 ms** |
| `docs/receipts/heuristic_ablation.txt` (morph on) | **5.78 ms** | — |

The table's is the only one with no receipt. The derived "146 × latency penalty" and the
"6 ms median per cycle" headline both use it. The discrepancy is small in absolute terms
but the report claims every measurement is receipted.

### U6 — "prevents **any** place pose over an exposed busbar" (§11.2)

Stated absolutely, in the ethics section. §13.2.1 measures, on the shipping segmentation
path, **2 of 25 commanded placements overlapping ground-truth non-floor material** (8.3 %
and 5.2 %), and states that no planner-side guard can catch it "because every guard the
planner has … is derived from the same prediction that is wrong". The heuristic path's own
PCB subtraction has never been measured against a busbar at all.
**Honest version:** *"the packer rejects any placement overlapping the extracted exclusion
mask; the residual is the extractor's own boundary error, measured at 2 of 25 placements
grazing tray wall in §13.2.1, and no planner-side guard removes it."*

---

## Traceability rows whose cited evidence does not support them

| Row | Cited evidence | Finding |
|---|---|---|
| **O2** — "Centroid error ≤ 2 px · Pass" | `tests/test_evaluate.py` | **Does not support it.** No centroid measurement exists in that file, anywhere else in `tests/`, or in any receipt. The cited test asserts a 5 px error as correct. The same defect as O4.b. |
| **O3** — "Queue rebuild ≤ 8 ms median · Pass — 3 ms median" | `bench_cycles.py`, Figure 4 | **Artefact does not exist and never did** (`git log --all` empty). The threshold is defensible on other evidence; the cited artefact is not retrievable. |
| **O5** — "Deterministic queue, row-major · Pass" | `tests/test_planner.py` | **Half-supported.** `test_row_major_ordering` genuinely asserts placements sort by `(grid_row, grid_col)`. But §3's stated threshold is "Fixed input → fixed output", and **no test in the repository runs the planner twice on one snapshot and compares** (grep-verified over all 31 test files). §6.5's determinism argument is sound by construction; it is not observed. |
| **O1.a** — "Heuristic baseline measurable · Pass — 0.479 val" | `pr_summary.txt`, Figure 8 | **Number checks out** (0.4463 and 0.5121 → 0.4792), but its generator `pr_curves.py` was never committed and the raw arrays live at `/tmp/meas/pr_curves.npz`, outside the repository. Present, not regenerable. |

Rows **O1, O1.b, O3.a, O3.b, O3.c, O4, O6** were checked and their cited evidence does
support them. O3.a's `_assert_no_overlaps` genuinely asserts pairwise non-overlap across a
40-item batch; O3.c's receipt matches the §6.3.1 tables cell for cell; O6's dual-scoped
89 %/65 % is honest and matches the regenerated receipt; O4's "Not tested" is correct.

---

## What I verified as sound (so the count is not read as a verdict on the whole report)

Verified **by executing**:

- **CRC-16/MODBUS and the 16-byte frame (§7.1).** `crc16_modbus(b"123456789") == 0x4B37`.
  A packed `MOVE_TO(-12345, 6789, -321, 0xBEEF)` unpacks field-for-field as §7.1
  describes: version u8, opcode u8, x/y int32 **big-endian**, z int16 BE, aux u16 BE,
  CRC u16 **little-endian** over the first 14 bytes. This is the strongest claim in §7 and
  it holds.
- **§6.3.1's entire empirical section.** Both packing tables, the paired-*t* table
  (13.35 / 15.18 / 6.60 / 1.31 / 1.00), the `pack_best_effort` block (14.55 vs 14.28;
  2.60 → 5.53; 0.57 → 2.85) reproduce from `forbidden_bench.txt` exactly.
- **§10.7's rotation ablation.** +0.0 % / +24.2 % / +57.1 % / +5.0 % exact.
- **§10.1 and §10.6's detection numbers.** 0.397 / 0.447 / 0.348 reproduced by running the
  detector; 0.479 / 0.446 / 0.512 and 0.874 / 0.905 / 0.842 / 0.583 and 0.764 / 0.305 all
  match their receipts; 822 = 630 + 192 and ~47 % recall confirmed by execution.
- **§13.1.1 in full.** All eight `seg_eval_*_on_cad_test.txt` receipts match the pooled
  table, the control ranges, the crown table and the 18650 control to four decimals.
- **§13.2.1 in full.** Boundary displacement 1.226 / 1.273 / 1.582 at 0.822 mm/px, head
  3.844 mm, clears-by 3.14× / 3.02× / 2.43×, IoU 0.890 / 0.861 / 0.658, latency
  16.6 / 57.1 ms, Δcells +0.056 with 115 exact / 6 lost / **5 in the damage direction**,
  real-photo 0.318 vs 0.217 with 0/20 vs 7/20 — every figure matches its receipt.
- **§8's `main_seg_run.txt` figures.** 26 → 26 → 7 → 4 → 1, 1 bad detector box, 57/2/1
  outside the envelope, 15 of 15 frames with their own scale.
- **§3's operating envelope.** Recomputed from `recog/synth3d/assets/catalog.json`:
  54.9 × **65.00**, 73.2 × 66.75, 54.9 × 135.2, 73.2 × 140.75 → margins +0.00 / +1.75 /
  +70.20 / +75.75. The `10000`'s exact-tolerance bay is real, and 18.5·tan(1°) = 0.323 mm.
- **The suite.** 1074 tests collected, exit 0.

Verified **by reading**:

- **§7.5's heartbeat correction.** `OpCode.HEARTBEAT` appears in exactly two functional
  places — its definition in `protocol.py:108` and the dispatch arm at
  `mock_kuka_server.py:326`. Nothing sends one. `heartbeat_interval_ms` is used only as the
  constant inter-retry pause (three `time.sleep` sites). The correction is exact.
- **§7.2's escalation.** `test_retry_exhaustion_sends_the_estop` asserts
  `[HANDSHAKE, MOVE_TO, MOVE_TO, MOVE_TO, ESTOP]` on the wire and passes.
  `test_corrupt_status_frames_retry_then_escalate` and `test_controller_estop_stops_the_client`
  assert the other two routes. The transient/fatal split is real (`_TRANSIENT = (socket.timeout,
  ValueError)`, with `struct.error`/`ConnectionError` deliberately falling to the fatal path).
- **§7.4's mock corrections.** `test_estop_latches_across_reconnects` asserts the vacuum
  drops, a fresh `HANDSHAKE` answers `ESTOP` and the pose does not move;
  `test_estop_has_no_reset_path`, the half-frame timeout test, and the distinct
  fault-code tests all exist and assert what §7.4 says. `REACH_MM = 706.0` is real.
- **§6.5's corrections.** `test_a_reservation_over_a_forbidden_cell_raises`,
  `test_reserving_over_an_existing_footprint_raises` (including the abutting-is-legal case)
  and `test_an_empty_workspace_envelope_is_rejected_at_construction` all assert the
  described behaviour rather than merely exercising it.
- **§7.3's KRL claims.** `$VEL.CP = 0.150`, `WAIT SEC 0.05` then `IF $IN[10] == FALSE →
  RETURN 2`. (One nit: §7.3 says the routine returns "SUCCESS, PICK_FAILED, or
  PLACE_FAILED"; `routines.src` returns only 1 and 2 — there is no PLACE_FAILED return.)

---

## Minor discrepancies, listed but not counted

- §3's O6 row says "≥ 70 % **line** coverage"; §9.3, §10.5 and Appendix E say **branch**.
- §9.5 says "six derived sub-requirements"; Appendix E's matrix carries **five** (it says
  so itself — two rows were dropped from FDR_v2).
- §4.3 says "four YAML files in `configs/`" and Appendix B says "Three YAML files";
  `configs/` holds 22 today.
- §3 and §13.2.1 cite `recog/synth3d/catalog.json`; the file is at
  `recog/synth3d/assets/catalog.json`.
- Abstract: "median 3.0 ms for perception and 3.0 ms for planning (both **well inside** the
  8 ms O3 budget)" — §10.4 reports the planning p95 at 13.0 ms, above the budget. The
  abstract's qualifier is true of the median only and reads as if it were true of the
  distribution.

---

## Overall verdict

**The report cannot be defended exactly as written, but it is close, and the gap is
concentrated in identifiable places.**

What is defensible: everything the project has corrected under a date since 2026-08-11.
Those corrections are unusually good — I attacked the heartbeat, the E-stop escalation,
the mock-server latch, the reservation interlocks, the CRC and the frame layout, and every
one of them says what the code does, with tests that genuinely assert rather than exercise.
The heavy empirical sections (§6.3.1, §13.1.1, §13.2.1, §8, §3) reproduce from committed
receipts to four decimal places, and the report's disclosure discipline around them —
sample sizes, provenance, the crown result's narrow licence, §13.2.2's flat statement that
sim-to-real is unmeasurable — is stronger than the norm for this kind of document.

What is not defensible: **three items must be fixed before submission would survive a
determined examiner.**

1. **F1**, because it is a headline finding in the executive summary, is restated twice,
   is falsified by running the project's own code, and is contradicted by a row in its own
   §10.6 table. An examiner who runs the detector finds 50 false positives in one command.
2. **U1 (O2)**, because it is the O4.b defect repeated on a *headline* objective reported
   as fully met, and because the report has already been publicly embarrassed by exactly
   this pattern. O2's honest verdict today is **Not measured**.
3. **F3**, because a fabricated safety bound in the ethics section is the most damaging
   single sentence in the document, and the code's own docstring says the opposite.

Beyond those, §5 and Appendix B should be marked as describing the April configuration (or
updated), §6.3.1's latency sentence should be corrected in the unsafe direction (worst case
7.4 ms, not 4.6 ms), and Appendix E's closing re-verifiability sentence should be brought
into line with Appendix C.3, which already contradicts it.

The pattern across all sixteen findings is consistent and worth stating: **the report is
reliable exactly where it has been re-measured, and unreliable exactly where it has not.**
Every false or stale claim above sits in prose that predates the audit campaign and carries
no dated correction. None of them is a case of the project overstating something it
measured; all of them are cases of a description outliving the thing it described — or, in
F1's and F3's case, of a claim that was never measured at all being written in the voice of
one that was.
