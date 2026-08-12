# Figures audit — which of the nine may be published, and what the README now shows

**Date:** 2026-08-12
**Baseline:** `f680d88`, 752 tests passing, tree clean
**Scope:** `README.md`, `docs/figures/`. No code, config, metric definition or
receipt was changed. Nothing was retrained.

The repository README carried **no images at all** while `docs/figures/` held
nine tracked figures. The obstacle to simply using them is that all nine were
produced 2026-04-29..2026-05-05 for `docs/FDR_v3.md`, and the 2026-08-11 work
moved latency, packing behaviour, scale handling and the shape of the pipeline
itself. This document is the audit that had to happen before any of them could
go on the front page.

## 1. Method

Three sources, in this order of authority:

1. **The figure itself** — read the image, list every quantity and every file
   path it asserts.
2. **The receipt that governs those numbers**, if one exists in
   `docs/receipts/`. A figure whose numbers match its receipt digit-for-digit
   is current *with respect to that receipt*.
3. **The 2026-08-11 / 2026-08-12 specs**, to establish whether the receipt
   itself was superseded, or whether the *subject* of the figure changed even
   though nobody re-measured it.

The third check is the one that matters. A figure can match its receipt exactly
and still be stale, because the thing it depicts no longer exists. That is
precisely what happened to `fig4`.

Two claims were verified by execution rather than by reading:

* **fig3 reproduces at HEAD.** `first_fit_decreasing` on 10 × 21700 and 9 ×
  18650 footprints into a 200 × 150 mm strip returns 19 placements, shelf
  origins at y = 0.0 and y = 70.0, 85.1 % strip area used, 4478 mm² unused —
  every number the figure prints. FFDH is genuinely frozen.
* **The new segmenter figure's crops are genuinely held out.**
  `recog.seg_evaluate.check_split_matches_checkpoint` was run against
  `recog/checkpoints/seg/best.pt` before rendering, and passed, so the 126-crop
  validation split drawn from is the one that checkpoint was selected against.

## 2. Classification of all nine

| # | Figure | Verdict | Basis |
|---|---|---|---|
| 1 | `fig1_architecture` | **stale (structural)** | Regenerated as `fig11` — see §3 |
| 2 | `fig2_extractor` | **current, demo-only scope** | §2.2 |
| 3 | `fig3_packing` | **current, superseded caption** | Re-executed at HEAD |
| 4 | `fig4_latency` | **stale, not regenerable** | §2.4 |
| 5 | `fig5_training` | **current, superseded scope** | `train_curve.csv` epoch 14 |
| 6 | `fig6_gantt` | **timeless** | A record of Oct 2025 – May 2026 |
| 7 | `fig7_ablations` | **current, superseded scope** | Three receipts, all match |
| 8 | `fig8_pr_curves` | **current, superseded scope** | `pr_summary.txt` |
| 9 | `fig9_failures` | **current, demo-only scope** | `heuristic_failure_taxonomy.json` |

### 2.1 fig1_architecture — stale (structural)

Carries no numbers, so nothing about it could drift numerically. Its *shape* —
three sequential modules, a typed contract on every arrow, `RobotStatus`
feedback, a withdrawn real robot beside a mock — is still exactly right.

What is wrong is the contents of two of its three boxes, and both were made
wrong by 2026-08-11 commits:

* The **Recognition** box lists `recog/inference.py`, `FasterRCNNDetector`,
  `HeuristicDetector`, `recog/evaluate.py`, `recog/augmentation.py`. There is
  **no segmenter in it.** `f40cc1b` put the trained bay segmenter into
  Recognition as a second stage. The README's own ASCII sketch, four lines
  above where this figure would sit, already says "Faster R-CNN + per-cartridge
  bay segmenter" — so publishing fig1 there would have had the page contradict
  itself within one screen.
* The **Planning** box labels the packer `plan/bin_packing.py (FFDH)`.
  `562ca75` moved both planner call sites onto `common.packing.pack_best_effort`.
  `docs/FDR_v3.md` §6.3 already carries a "Superseded in part, 2026-08-11"
  note about exactly this wording; the figure never got the same treatment.

Neither omission is a wrong number, which is why no receipt and no spec caught
it — the nine specs of 2026-08-11/12 do not mention `docs/figures/` at all.

### 2.2 fig2_extractor — current, demo-only scope

The seven stages are still literally what `plan/placement_area.py` does: green
channel, `THRESH_OTSU`, `MORPH_CLOSE` k=5 then `MORPH_OPEN` k=3,
largest-contour `boundingRect`, inset, subtract PCB. The inset is still
`safety_margin_px: int = 5`.

The one annotation worth flagging is "safety inset (5 px ≈ 2 mm)". The *5 px*
is exact. The *≈ 2 mm* was 5 × 0.38 mm/px, and after `380e7d5` the 0.38 is no
longer a constant belonging to the extractor — it survives only as
`configs/planning.yaml`'s camera fallback, which is what the torch-free demo
path still runs at. So the annotation is still numerically right for the path
this figure describes, and would be wrong for any other.

Not featured for a different reason: it depicts the **heuristic, demo-only**
extractor, which is measured at zero placeable area on 7 of the 20 cartridges
annotated in the real photographs. It is the wrong thing to put on a front
page that is otherwise about the segmentation path.

### 2.3 fig3_packing — current, superseded caption

Re-executed at HEAD; every printed quantity reproduces (see §1). FFDH is frozen
by deliberate decision, and `recog/synth3d` lays out training scenes with it,
so it cannot move without silently redrawing a corpus.

The figure is fine; the *caption it came with* is not. It illustrates FFDH, and
FFDH is no longer what the planner runs on its own. The shipping packer ties it
exactly at 0 % forbidden coverage (23.00 vs 23.00), which is why this picture
is still a fair illustration of the algorithm and not of the planner.

Not featured: it is a synthetic illustration on an empty strip, and the
interesting packing result — 8 → 17 cells on the 30 real instances — is a table
in the README already, with no figure behind it.

### 2.4 fig4_latency — stale, not regenerable

The most important verdict in this audit, and the one that justifies the whole
exercise. Panel by panel:

* **Left — bare FFDH runtime vs item count, 10–75 µs.** Numerically
  untouchable: FFDH is frozen. But as a statement about *the planner's packing
  cost* it is now wrong by more than an order of magnitude. The planner runs
  `pack_best_effort`, measured at **3.4 ms mean / 4.6 ms worst on bench masks**
  and **1.9 ms on the worst real cartridge**, against the 0.9 ms this panel's
  subject cost. The panel plots microseconds; the shipping figure is
  milliseconds.
* **Middle — perception latency CDF, HeuristicDetector, median 3.0 ms.** The
  heuristic detector is untouched, so the measurement stands. But it describes
  the demo-only path. Shipping perception is Faster R-CNN plus a segmenter
  whose batched forward pass alone is **21.2 ms for 8 crops**
  (`docs/receipts/seg_eval.txt`) — seven times this panel's whole budget.
* **Right — planning latency CDF, median 3.0 ms, p95 13.0 ms.** Measured on a
  planner that no longer exists. Its packer was replaced (`562ca75`) and its
  occupancy rasteriser was replaced (`b93bbd3`, whole-cell instead of
  centre-pixel; `extract()` moved 4.25 ms → 2.97 ms as a result).

**Nobody re-measured it**, so the honest statement is not "it moved by X" but
"its subject changed and the new distribution has never been taken". There is
no `bench_cycles.py` in the tree and no plotting script for any of the nine, so
this cannot be regenerated without new measurement work — which is out of scope
here. **Left out of the README, and recorded as a gap** rather than published
with a hedging caption.

### 2.5 fig5_training, fig7_ablations, fig8_pr_curves — current, superseded scope

All three match their receipts exactly:

| figure asserts | receipt | value |
|---|---|---|
| custom anchors final mAP@0.5 = 0.713 | `train_curve.csv` epoch 14 | 0.712854 |
| custom anchors final mAP@0.75 = 0.375 | `train_curve.csv` epoch 14 | 0.374685 |
| defaults final mAP@0.5 = 0.874 | `frcnn_map_default.txt` | 0.8736 |
| heuristic AP battery / cartridge = 0.446 / 0.512 | `pr_summary.txt` | 0.4463 / 0.5121 |
| Faster R-CNN AP battery / cartridge = 0.905 / 0.842 | `pr_summary.txt` | 0.9053 / 0.8419 |
| FFDH rotation gain 0 / +24 / +57 / +5 % | `ffdh_ablation.txt` | +0.0 / +24.2 / +57.1 / +5.0 |
| morphology Δ median ms = +0.495 | `heuristic_ablation.txt` | +0.4951 |

The detector was not retrained, not re-evaluated, and no detector receipt was
regenerated on 2026-08-11/12 — every receipt touched that day was a segmenter
or arbitration receipt. So these three are **current**.

The scope caveat, which is why none of them is featured: they record a
**15-epoch, CPU, from-scratch, no-COCO-pretrain** run on the flat `cv2`
synthetic split (`frcnn_map.txt`: val size 15). The detector that actually
ships trains 35 epochs on the Blender corpus (`configs/recognition.yaml`:
`recog/dataset3d`). These figures are accurate records of an April experiment,
not descriptions of the shipping detector, and putting them on the front page
under a "results" heading would imply otherwise.

### 2.6 fig6_gantt — timeless

A project timeline, Oct 2025 to May 2026, with the mid-March hardware pivot
marked. No measurement can invalidate it. It stops at FDR submission and the
repository has three further months of work in it, so it is a complete record
of a period rather than of the project — accurate for what it records. Not
featured: a Gantt chart is not what a reader comes to a perception repository
for.

### 2.7 fig9_failures — current, demo-only scope

Matches `heuristic_failure_taxonomy.json`; nothing on 2026-08-11/12 touched the
`cv2` generator's frames or the heuristic detector. It is the most visually
interesting of the nine and was the strongest candidate for the "failures"
slot.

It was still left out. It is a taxonomy of the **HeuristicDetector** — the
demo-only baseline that exists so the loop can run without torch — drawn on
`recog/synth_dataset.py`'s flat coloured rectangles. As the only failure figure
on a README about a learned perception stack, it would read as this project's
failure modes. They are not; they are the failure modes of the straw man the
learned detector was measured against.

## 3. What was added

Two new figures. Neither replaces a tracked file: `fig1_architecture.png` and
the other eight are untouched, because `docs/FDR_v3.md` is a submitted report
that references them.

### `docs/figures/fig11_architecture.png` — 124,855 bytes

A current redraw of fig1, fixing the two structural errors in §2.1. Recognition
now carries `recog/bay_segmenter.py` and `recog/calibration.py`; Planning
carries `plan/arbitration.py` and `common/packing.py`'s `pack_best_effort`.

Carries **no measured quantity**. The only number is the 8 ms per-cartridge O3
budget, which is a requirement and does not drift — a deliberate choice, since
a diagram with a latency figure printed on it is a diagram that goes stale the
next time anyone runs a benchmark.

Placed near the top of the README, directly under the existing ASCII pipeline
sketch it expands.

### `docs/figures/fig10_segmenter.png` — 212,252 bytes

The figure this repository did not have: **what the segmenter actually
produces.** Three rows of five — camera crop, predicted label map, ground-truth
label map — over the six classes `background`, `cartridge`, `bay`,
`electronics`, `obstruction`, `battery`.

Provenance, all of it current:

* **Checkpoint** `recog/checkpoints/seg/best.pt` — the one `configs/demo_seg.yaml`
  loads and the one `docs/receipts/seg_eval.txt` was measured on.
* **Code path** `BaySegmenter.segment_batch()` at HEAD, fp16 — the shipping
  batched call, not a bespoke inference routine.
* **Data** the 126-crop validation split of `recog/dataset3d_seg`, reconstructed
  with the same `BaySegDataset` + `_split_dataset(0.85, seed=0)` that
  `recog.seg_evaluate` uses, and confirmed by that module's own split guard to
  be the partition the checkpoint was selected against.
* **Selection rule** the first five bay-carrying crops in split order. Not
  hand-picked, and the figure says so on its face, along with the pooled
  `bay` IoU of 0.8903 over all 36 such crops — so that five good panels cannot
  be mistaken for the summary statistic.
* **Synthetic**, stated on the figure itself and again in the README caption.

It shows the boundary defect the README's headline section is about: the
predicted bay runs slightly wide into the tray wall, which is the optimistic
direction, and is the picture of the 1.226 mm mean boundary displacement.

**Reproduction.** Both figures were rendered by throwaway scripts, not committed
tooling — consistent with the existing nine, none of which has a generator in
the tree. `fig10` is the one worth reproducing, and the recipe is exact: build
`BaySegDataset(coco_path, img_dir, out_size=256, jitter_frac=0.06, train=True)`,
split it with `recog.seg_training._split_dataset(ds, 0.85, seed=0)`, take
`val.indices` in order, keep the first five whose label map contains
`SEG_CHANNELS["bay"]` (indices **703, 486, 139, 500, 231** of 841), and pass
their crops through `BaySegmenter(checkpoint="recog/checkpoints/seg/best.pt",
crop_size=256, half=True).segment_batch()`.

## 4. Where the figures went, and what was deliberately left out

Two images, both earning their place:

| Figure | Section | Why there |
|---|---|---|
| `fig11_architecture` | under the opening pipeline sketch | It is the detailed form of the ASCII block directly above it |
| `fig10_segmenter` | end of "The same loop with the trained segmenter in it" | That section names the checkpoint; this is what it produces |

Seven of the nine were left out, and one new figure was not attempted:

* **fig4** — stale and not regenerable without new measurement (§2.4). This is
  the honest gap. If it is ever redrawn, the distinction to preserve is the one
  `2026-08-12-portfolio-verification.md` §1.9 already had to correct once: the
  3.4 ms / 4.6 ms figures are **bench-mask** numbers, and the worst *real*
  cartridge is 1.9 ms.
* **fig1** — superseded by fig11 rather than published.
* **fig2, fig9** — accurate, but both depict the demo-only heuristic path.
* **fig5, fig7, fig8** — accurate, but they record the April CPU experiment
  rather than the shipping detector.
* **fig3** — accurate, but an illustration of an algorithm the planner no
  longer runs alone.
* **fig6** — accurate and irrelevant to a code reader.

Total bytes added to the repository: **337,107** across two PNGs. Both are flat
enough to compress well; the README's images together are lighter than any one
of the seven tracked real photographs.

## 5. Two things left for the author

Neither is in this task's scope and neither was touched.

1. **`docs/README.md` line 41** describes `figures/` as "The figures referenced
   by `FDR_v3.md`". That is now two figures short of true — `fig10` and `fig11`
   are referenced by the repository README instead. A one-line fix, left alone
   because `docs/README.md` was outside this task's file boundary.
2. **`docs/FDR_v3.md` §6.4** still says the planner will "invoke FFDH per
   cartridge" — the same stale wording as fig1's Planning box, which the
   2026-08-11 reconciliation pass caught in §6.3 and §6.3.1 but not here.
   Reported by the spec sweep during this audit; not fixed, because the FDR is
   a submitted document and out of scope.
