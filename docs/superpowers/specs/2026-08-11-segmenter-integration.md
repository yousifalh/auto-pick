# Retiring the τ gate in code, and putting the segmenter in the loop

Date: 2026-08-11
Commits: `5a619fc` (job 1 — the τ gate); job 2 (the segmenter wiring) is
the commit that adds this file.
Baseline: `1f4a63d`, 666 tests passing.

Two defects with one shape: a decision that had been made and written
down, but never reached the running code.

---

## Job 1 — the τ gate outlived its own retirement

### What was wrong

`docs/FDR_v3.md` §13.2.1 retired τ on measurement. The two placement
estimates `plan/arbitration.py` reconciles are not independent — they
are the same `argmax` label map read twice, once with an erosion band —
and their IoU correlates with placement error **positively** in all four
cataloged SKUs, raw and area-normalised, which is the wrong sign for a
confidence gate. The FDR says plainly that the extractor "should keep
applying `P_safe` unconditionally rather than gating on `iou >= tau`;
the gate is inert data".

Commit `dee9854` changed docs and comments. `plan/placement_area.py:404`
still read `if iou < self.tau: ... raise PlacementDisagreement`.

Three different values were live at once, none of which agreed:

| Source | Value | Read by |
| --- | --- | --- |
| `SegmentationPlacementAreaExtractor.__init__` default | 0.85 | everything that constructed it without an override |
| `configs/planning.yaml` `arbitration.tau` | 0.7492 | **nothing** — grep-verified, no Python read the key |
| `README.md` | 0.5715 | nothing; it described the YAML value as live |

### Measured cost

Real detector (`recog/checkpoints/best.pt`) plus real segmenter
(`recog/checkpoints/seg/best.pt`) over the first 15 frames of
`recog/dataset3d_seg`. 26 cartridge crops, 8 of which carry a predicted
`bay` pixel. Plannable = `extract()` returned a `PlacementArea`.

| `mm_per_px` | τ=0.85 | τ=0.7492 | τ=0.5715 | no gate |
| --- | --- | --- | --- | --- |
| 0.625 (the dataset's true framing) | **3** | 6 | 7 | **8** |
| 0.38 (`configs/planning.yaml`'s own value) | **0** | — | — | **8** |

The second row is the finding that matters. `mm_per_px` sets the wall
erosion radius — 7 px at 0.625, 11 px at 0.38 — and the wider erosion
shrinks `P_derived`, dragging every IoU below the threshold. The
observed IoU range at 0.38 is 0.639–0.848, entirely under the code's
0.85 default. **In the project's own configured calibration the gate
rejected every plannable cartridge it was ever offered**, and did so
silently, one `except PlacementDisagreement: continue` at a time.

After removal both rows read 8. The IoU is unchanged and still reported
on `PlacementArea.consistency_iou`; nothing acts on it.

### What changed, and what deliberately did not

- **Deleted**: the `iou < tau` branch, the `tau` constructor argument,
  `self.tau`, and `configs/planning.yaml`'s `arbitration` block. The
  argument is *deleted* rather than accepted-and-ignored — a constructor
  that swallows `tau=0.85` and does nothing with it is the same defect
  as the one being repaired. Callers now get a `TypeError`; the two in
  the tree (`recog/seg_ablation.py`, the tests) were updated.
- **Kept**: `P_safe = P_direct ∩ P_derived`, applied unconditionally.
  It is a geometric constraint, not a threshold. A test now pins it
  separately from the gate's removal, so "the gate is gone" cannot
  quietly become "the exclusion is gone".
- **Kept**: the nested "both estimates empty" case. It used to live
  inside the τ branch but was never about τ: `mask_iou` reports two
  empty sets as 0.0, the same number a real disagreement gives, so the
  IoU cannot tell a full cartridge from a broken one and the masks have
  to be re-read. It now sits under the empty-`P_safe` check with the
  same message, and the sibling case (estimates that do not overlap)
  gained a message of its own.
- **Kept**: the `BadDetectorBox` check, ahead of `arbitrate()`. It is a
  statement about *where the crop landed*, not about how well two reads
  of one `argmax` agree, and it is unaffected by the finding.

### `PlacementDisagreement` — kept, and why

Nothing raises it bare any more. It is still **read** in three places,
so removing it would not be tidying:

- `BadDetectorBox` derives from it, so `except PlacementDisagreement`
  still catches a misaligned detector box;
- `plan/planner.py:205` catches and counts it;
- it is exported and covered by a planner test that pins the counter
  contract for any extractor that raises it.

Consequence, stated rather than hidden: `Planner
.placement_disagreement_count` now reads 0 forever, and its `except`
branch is unreachable in-tree (`BadDetectorBox` is caught above it). A
comment in `planner.py` says so. Deleting the branch would push a
refusing extractor into the blanket `except Exception`, where it would
stop being counted at all — a worse outcome than an honest zero.

**An empty `P_safe` is deliberately NOT re-labelled a disagreement.**
18 of those 26 crops have no predicted `bay`; routing the ordinary
"nothing to place here" state into the safety counter would bury the
interlock it is supposed to be.

### Not done, deliberately

The brief asked for `python -m recog.sync_config` after editing
`configs/planning.yaml`. It was **not** run, because it does not apply:
`sync_config` transcribes `configs/synth3d.yaml` to the JSON sidecar
Blender reads and never touches `planning.yaml` (verified — the sidecar
is already byte-identical to its YAML). Running it would have rewritten
a tracked file belonging to another agent's concurrent work for no
effect.

---

## Job 2 — the segmenter was not in the pipeline at all

### What was wrong

`main.py` is the loop the "fully working pipeline" claim rests on.
Running it: image source → Faster R-CNN → **heuristic** extractor →
FFDH → mock KUKA. `plan/arbitration.py` never imported,
`detector.segmenter is None`, `Snapshot.cartridge_masks` empty. Two
hardcoded seams:

- `recog/inference.py:324` — `load_detector(checkpoint, cfg)` took no
  `segmenter`, so `FasterRCNNDetector`'s `segmenter=None` default could
  never be overridden and the second-stage call inside `detect()` was
  unreachable in production;
- `main.py:124` — `_build_planner` hardcoded the heuristic extractor.

### The wiring

`mode.segmentation` selects the path, and selects it in **both** places
from one key. That is not convenience: the two halves are silent when
separated. A segmenter with no consumer runs for nothing; the
segmentation extractor with no segmenter raises `ValueError` per
cartridge into `plan/planner.py`'s blanket `except Exception`, which
reports a clean run of zero placements.

Everywhere the new path could no-op, it raises instead:

| Failure | Behaviour |
| --- | --- |
| `mode.segmentation` present, no `checkpoint` key | `ValueError` |
| checkpoint path missing | `FileNotFoundError` |
| segmenter supplied but detector falls back to `HeuristicDetector` (no checkpoint / no torch) | `RuntimeError` in `load_detector` — the heuristic has no second stage |
| cartridges detected in a frame but no masks attached | `RuntimeError` mid-run — the second stage is not running |
| whole run completes with 0 placement areas | `RuntimeError` — a segmentation run that planned nothing is a failed run |

The last one fired for real during development: pointed at frames whose
first render has no plannable cartridge, the run ended after one cycle
and would have reported a tidy zero.

### One behaviour change to `main.py`, gated behind a default

`mode.stop_on_empty_queue`, defaulting to `true` — today's behaviour,
so `configs/demo.yaml` is untouched. PPR §5.4's stop-when-empty is right
for a real cell working one physical scene. It is wrong for
`source: synthetic`, which cycles through *unrelated* renders: an empty
queue on one frame says nothing about the next. The heuristic demo never
noticed, because it finds a placement area in every green rectangle;
the segmenter predicts no `bay` on most crops, so the first such frame
ended the entire run. `demo_seg.yaml` sets it `false`.

### `configs/demo_seg.yaml`

Points at `recog/dataset3d_seg/images`, not `recog/dataset/images`. On
`synth_dataset.py`'s flat green rectangles the segmenter emits no `bay`
whatsoever and the demo produces zero — which is now a hard error rather
than a clean exit, but the fix is the corpus.

It also sets `mode.mm_per_px: 0.625`, overriding `planning.yaml`'s 0.38
placeholder ("Replace with real intrinsics when available"). 0.625 is
this dataset's actual framing —
`layout.area[0] * 1000 / render.res[0]` from
`recog/dataset3d_seg/manifest.json`, the frozen generator config these
images were rendered under, the same figure
`docs/receipts/seg_ablation.txt` quotes. It is applied to the planner
and the extractor together; they must not drift, since one turns it into
an erosion radius and the other into workspace millimetres.

**Caveat on what this demonstrates.** These renders are the segmenter's
own training corpus (its 0.85/0.15 split is internal to
`recog.seg_dataset` and not reproduced by the demo's frame cycler). The
receipt is evidence that the *wiring* works end to end. It is not a
generalisation measurement; `recog.seg_evaluate` against a held-out
dataset is, and `docs/receipts/seg_eval*.txt` is where those numbers
live.

### The receipt

Generated by tooling — `main.py --receipt PATH` — never hand-written.
`docs/receipts/main_seg_run.txt`, from
`python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt`:

```
  detector      : FasterRCNNDetector
  segmenter     : BaySegmenter
  extractor     : SegmentationPlacementAreaExtractor
  mm_per_px     : 0.625

  cycles executed        : 1
  cartridges detected    : 26
  cartridges segmented   : 26
  placement areas        : 8
  placement disagreements: 0
  bad detector boxes     : 0

  loose batteries detected: 78
  poses queued           : 1
  placed                 : 1
```

26 / 26 / 8 reproduces the hand-wired reference exactly, and the 8 is
the post-τ number from Job 1 (it would have been 3, or 0 at
`planning.yaml`'s calibration).

### Finding: FFDH, not perception, is now the binding constraint

8 placement areas produced 1 queued pose. That is **not** a mask
problem. Measuring the largest all-free axis-aligned rectangle in each
resulting occupancy grid, against an 18.5 × 65 mm cell:

| frame | grid | free | largest free rect | FFDH placements |
| --- | --- | --- | --- | --- |
| `scene_00005` | 123 × 62 | 93 % | 112 × 48 mm | **0** |
| `scene_00007` | 43 × 45 | 73 % | 46 × 38 mm | 0 |
| `scene_00008` | 68 × 37 | 81 % | 102 × 22 mm | 1 |

`scene_00005` has a 112 × 48 mm clear rectangle and a 93 % free grid,
and FFDH places nothing in it — while the same packer places 12 cells in
an all-free grid of identical dimensions. Shelf-aligned packing under a
sparse forbidden mask misses free rectangles that demonstrably exist.
`scene_00007`'s bay genuinely cannot hold a cell (46 × 38 mm), so the
constraint is not uniform.

This is pre-existing (`plan/bin_packing.py`, FDR §6.3.1) and was left
alone — the brief scoped this work to wiring and to the τ gate, and the
packer is a separate, benchmarked subsystem. But it should be recorded
that the throughput ceiling on real masks now sits in **planning**, not
in perception: perception delivered 8 usable placement areas and the
packer converted one.

Two further limits on `placed: 1`, both by design and not defects: the
loop executes at most one pose per cycle before re-planning (PPR §5.4),
and a pose needs a plannable cartridge and a free battery in the *same*
frame.

---

## Verification

- Full suite: **678 passed** (baseline 666; +4 from a concurrent agent's
  `tests/test_bay.py`, +8 mine — 7 new wiring/gate tests and one test
  replaced by two).
- `python main.py --config configs/demo.yaml` — run, not assumed:
  10 cycles, 10/10 placed, `cartridge_masks: 0`, heuristic extractor.
  Unchanged, and non-deterministic at 9–10/10 as `demo.yaml` documents.
- The three new `main.py` tests were mutation-checked: forcing
  `_build_planner` to always pick the heuristic fails all three,
  including the zero-placement guard.
- τ counts before and after measured on identical frames with identical
  checkpoints.
