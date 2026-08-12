# The demo that contradicted its own README, and the red suite behind it

**Date** 2026-08-12 · **Baseline** HEAD `f1989e9`, suite 1,211 collected
with torch (1,181 on the torch-free install) · **Fixes** audit P findings
P1, P2, P3, P5, P6 (`docs/superpowers/audit/2026-08-12-P-stranger-experience.md`).
**Files** `README.md`, `configs/demo.yaml`, `tests/test_calibration.py`,
`.github/workflows/ci.yml`, `docs/receipts/main_run.txt` (new),
`docs/PORTFOLIO.md`, `docs/CV_BULLETS.md`, `docs/FDR_v3.md`.

**Not allowed to change, and did not:** any metric definition, model,
dataset, config value or algorithm. Every change here is documentation,
one test guard, and one generated receipt. `configs/demo.yaml`'s
`max_cycles` and `stop_on_empty_queue` were deliberately left alone —
making the demo run its declared ten cycles is a behaviour change, and
the honest minimum is to publish the number the command actually gives.

---

## 1. The defect, reproduced

Audit P cloned the repository into a scratch directory, made a fresh
`venv`, and followed the README's three commands literally. This work
repeated that, and got the same thing:

```
$ python -m recog.synth_dataset --out recog/dataset --n 50
$ python main.py --config configs/demo.yaml
[WARNING] recog.inference: No checkpoint at recog/checkpoints/best.pt
(or torch unavailable). Using HeuristicDetector fallback
...
Run summary: {'cycles': 1, 'placed': 1, 'cartridges_detected': 3,
'batteries_detected': 5, 'placement_areas': 3, 'queue_poses': 3,
'released_reservations': 2, ...}
```

`README.md` §"Running it" and `configs/demo.yaml`'s header comment both
claimed **37 cartridges, 77 batteries, 33 placement areas, 41 queue
poses over 10 cycles**, "identical in all ten runs". `synth_dataset.py`
is seeded (`--seed`, default `0`), so this is not variance: every reader
gets the same 50 frames and the same contradiction. Five consecutive
runs on the clean clone gave byte-identical counts.

## 2. Diagnosis: the numbers were real, on a path no clone can take

Three candidate explanations were available — a different corpus, a
different `--n`, or a behaviour change since the measurement. Each was
tested rather than assumed.

**The generator has not changed.** Regenerating at `--n 100` into a
scratch directory reproduced the working tree's `recog/dataset/images`
byte-for-byte (`md5sum` on frames 0, 1, 49 and 99). The corpus in the
developer tree, dated April, is exactly what today's tool emits. So the
figures were not produced by an older generator.

**Frame count is not the variable.** The demo's frame source
(`main.py:_synthetic_source`) walks `sorted(glob("*.png"))` from index 0,
so frame *i* is frame *i* regardless of how many frames exist. Confirmed
directly: with the trained detector loaded and a corpus of exactly the
50 frames the README's step 2 makes, the run returns the same
`cycles 10, cartridges_detected 37, batteries_detected 77,
queue_poses 41`. Audit P reached the same conclusion from the other
direction, at `--n 200`.

**The variable is the detector.** `recog/inference.py:load_detector`
returns `FasterRCNNDetector` when `checkpoint and Path(checkpoint).exists()
and _TORCH_AVAILABLE`, and `HeuristicDetector` otherwise. The working
tree in which 37/77/41 was measured holds `recog/checkpoints/best.pt`
(147 MB) with torch installed, so it took the first branch and ran the
**trained Faster R-CNN**. `recog/checkpoints/` is gitignored and no `.pt`
is tracked anywhere in the repository, so every clone takes the second
branch and runs the pure-OpenCV heuristic on `synth_dataset.py`'s green
rectangles.

| | detector | cycles | cartridges | batteries | queue poses |
|---|---|---|---|---|---|
| Clean clone, 50 frames (documented path) | `HeuristicDetector` | 1 | 3 | 5 | 3 |
| Tree with checkpoint, 50 frames | `FasterRCNNDetector` | 10 | 37 | 77 | 41 |
| Tree with checkpoint, 100 frames | `FasterRCNNDetector` | 10 | 37 | 77 | 41 |

The cycle count follows from the same cause. `main.py:385` defaults
`stop_on_empty_queue` to `True`; the heuristic finds one cartridge and no
executable pose on frame 1, so the loop breaks at cycle 1 of `max_cycles: 10`.
That is the demo's designed exit, not a fault.

**The irony is exact.** The README introduces these numbers with *"That
path is torch-free by design and is what the reproducibility claim rests
on"* — and then quotes figures obtainable only with torch and a
checkpoint. The sentence and the numbers beside it describe two
different programs.

**A second, smaller drift was found and is recorded rather than
published.** Even in a tree that does have the checkpoint, HEAD no longer
returns the documented `placement_areas: 33` and `released_reservations: 2`;
it returns 29 and 29, and `unreachable_place_targets` 17 rather than 14.
Those figures moved under commits `dd36329`..`f1989e9`. They are not
restated in the README, because a number that needs a 147 MB gitignored
file to reproduce does not belong in a section about the torch-free path.

## 3. What was changed

**The documentation was corrected to the command, not the command to the
documentation.** The demo genuinely works; only the prose about it was
wrong, and a README that overstates is worse than one that reports a
modest true number. `README.md` and `configs/demo.yaml` now give
1 cycle / 3 / 5 / 3 / 3, state that the run is a wiring demonstration on
three green rectangles rather than a result, explain the cycle-1 exit,
and record the withdrawn figures and why they were unreachable. The
placed/pick-failed caveat is kept and its arithmetic corrected: at one
cycle, `P(failure) = 0.0298` means roughly one run in thirty-four
reports `pick_failed: 1`, not the "one run in four of ten cycles" that
applied at ten.

**A receipt now pins it.** `docs/receipts/main_run.txt` was generated by
committed tooling — `python main.py --config configs/demo.yaml --receipt
docs/receipts/main_run.txt` — from the clean clone, not from the
developer tree. Its eighth line reads `detector      : HeuristicDetector`,
which is precisely the discriminator whose absence let this survive:
`main_seg_run.txt` existed for the segmenter path, but the torch-free
demo, the one path the README stakes its reproducibility claim on, had
no receipt at all.

**CI now diffs that receipt** instead of only asserting exit 0. The job
regenerates the receipt and compares it against `git show HEAD:` with
the `generated:` timestamp filtered out. Exit 0 was never going to catch
this: the command ran perfectly, and reported numbers nobody compared to
anything.

**A "Start here" block** was added immediately after the opening diagram,
pointing at `docs/PORTFOLIO.md` (previously first mentioned at line 327
of 335), the run command with its receipt, `docs/superpowers/specs/README.md`
(line 168), and `docs/README.md` — which the root README had **never
linked**, in a repository whose best navigation aid it is. It ships in
this commit rather than earlier because directing more attention at a
contradicted command would have made the artefact worse, not better.

## 4. The red suite (P2)

```
FAILED tests/test_calibration.py::test_seg_ablation_counts_val_instances_at_the_checkpoints_crop_size
E   ModuleNotFoundError: No module named 'torch'
```

`tests/test_calibration.py:539` did a bare `import torch` inside the test
body. Its sibling at `tests/test_seeding.py:150` does it correctly, and
so do the other ten torch-using modules. One line, following the
established pattern:

```python
torch = pytest.importorskip("torch")
```

`.github/workflows/ci.yml` asserted in a comment that *"Each guards
itself with an importorskip/skipif, so this run is green rather than
quietly thinner"*, which meant **CI should have been red at HEAD on all
three matrix Pythons**. The fix makes the comment true; the comment's own
figures were stale and were re-measured rather than left.

**Verified by execution, not by inspection.** The repository was cloned
to a scratch directory at `f1989e9`, a fresh `venv` created, and
`pip install -e ".[dev]"` run with torch absent (`importlib.util.find_spec("torch")`
→ `None`).

| | result |
|---|---|
| Clean clone at `f1989e9` | `1 failed, 1162 passed, 19 skipped` |
| Same clone, one-line fix applied | `1162 passed, 20 skipped`, **exit 0** |
| Working tree, torch present | `1210 passed, 1 skipped`, exit 0 |

## 5. Test and receipt counts (P5, P6)

The suite was quoted as 1,074 in three places and 814 in a fourth. Both
are stale, and the two figures that matter are different from each
other, so the documents now say which is which.

* **1,211 collected** with torch installed.
* **1,181 collected** on the documented torch-free install, because
  `tests/test_bay_segmenter.py` calls `pytest.importorskip("torch")` at
  module level and its 30 tests are never collected. `pytest -q` reports
  `1162 passed, 20 skipped` — 17 of those skips are torch (16 individual
  tests plus that whole module), 1 is `bpy`, 1 is a Blender-generated
  COCO dataset, 1 is the `.[cad]` STEP converter.

Corrected at `README.md` (prose and the layout block), `docs/PORTFOLIO.md`
and `docs/CV_BULLETS.md`. Dated audit and spec documents that record a
count *at their own baseline commit* were deliberately left alone: those
are measurements, not claims about today, and this directory's convention
is retraction-by-successor rather than silent editing.

"Sixteen of the **thirty-four** committed receipts" was 36 at HEAD and is
**37** with `main_run.txt` added. Corrected in `README.md` and at the
three sites in `docs/FDR_v3.md`. The numerator is unchanged: all three
receipts added since Appendix C was written — `detector_bench.txt`,
`seed_reproducibility.txt`, `main_run.txt` — have committed generators.

## 6. What is still true and unfixed

* **`configs/demo_seg.yaml` remains unrunnable by any reader**, and no
  reader-derivable *number* exists anywhere in the repository. Audit P's
  §5 recommendation (ship the 43 MB segmenter checkpoint plus a 15-scene
  sample, ≈58 MB) is not actioned here. The demo receipt narrows the
  credibility gap; it does not close the engagement one.
* **The demo runs one cycle, not ten.** `max_cycles: 10` overstates what
  the torch-free path does. Setting `stop_on_empty_queue: false` would
  exercise all ten, and is a behaviour change that has deliberately not
  been made.
* **The torch-free demo produces no result, only a wiring
  demonstration.** Three cartridges of green rectangles is not evidence
  about perception, and the README now says so in the same paragraph as
  the numbers rather than leaving the reader to infer it.
