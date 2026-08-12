# Audit D — reproducibility: can the receipts be re-derived?

Read-only audit, 2026-08-12, at HEAD `fa7a4f0`. Nothing in the repository was
modified, staged, committed, retrained or regenerated; every command run below
wrote its output to a scratch directory outside the tree, and `git status` was
clean before and after.

Findings are tagged **(measured)** when I established them by executing a
command during this audit, and **(read)** when I established them by reading
code or documents without independent execution.

**Brief:** the project's discipline is *"every number has a receipt,
regenerated from its own tooling"*. A receipt is only as good as the ability to
re-derive it. How much of the published evidence can someone who clones this
repo actually reproduce, and what would it cost them?

**Verdict up front.** The *current* evidence is in good shape and better than
the claim: I re-derived two committed segmentation receipts and one ablation
receipt **exactly**, and reproduced the packing benchmark's committed counts
**exactly**, without touching the repo. The problem is the *inherited* evidence
and the *chain*: **16 of 34 committed receipts have no surviving tool**, all 16
predate the repository's git history, and **no receipt records the commit it
was produced at**. Two things also turned out not to be what the documentation
says they are: training is genuinely unseeded (not merely
hardware-nondeterministic), and the demo's non-determinism is not GPU jitter —
it is a deliberate 2 % simulated vacuum-drop in the mock robot.

---

## 1. Which receipts are regenerable at all?

34 receipts are committed; a 35th (`forbidden_bench_timings.csv`) exists on
disk but is deliberately `.gitignore`d, so a cloner never sees it.

| class | count |
|---|---:|
| **A — reproducible from a clean clone** (repo + pip-installable deps only) | **4** |
| **B — needs artefacts not in the repo** (checkpoint and/or Blender dataset) | **14** |
| **C — not reproducible** (no surviving script) | **16** |
| *(D — regenerable but excluded from the clone by `.gitignore`)* | *(1, untracked)* |

### Class A — reproducible from a clean clone (4)

| receipt | command | status |
|---|---|---|
| `forbidden_bench.csv` | `python scripts/forbidden_bench.py` | **(measured)** re-ran twice; every count column byte-identical run-to-run *and* identical to the committed file |
| `forbidden_bench.txt` | same | **(measured)** all 12 count rows match the committed receipt exactly; only the µs timing columns move |
| `forbidden_bench_seeds.txt` | `for s in 20260806 0 1 7 42 123 999 2024; do python scripts/forbidden_bench.py --seed $s --no-write; done` | **(read)** the only receipt in the corpus that records its own regeneration command *inside the file*; the 8-row table is then assembled by hand from 8 runs |
| `pytest-cov.txt` | `pytest -q --cov` | **(read)** the tool exists, but the content is stale: it records **102 tests** and a `platform linux, python 3.10.12` coverage run, against 752 tests on Windows/py3.14 today |

### Class B — needs artefacts not in the repo (14)

All 14 need a trained checkpoint (`recog/checkpoints/**`, gitignored, 2.4 GB
on this machine) **and** a Blender-rendered dataset (`recog/dataset3d_seg*`,
gitignored, ~500 MB each). None of those artefacts is obtainable from the
clone; both must be regenerated (§4) or transferred.

| receipt(s) | command | artefacts required |
|---|---|---|
| `seg_eval.txt` | `python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml` | `seg/best.pt` + `dataset3d_seg` |
| `seg_eval_anchored_on_anchored_val.txt`, `seg_eval_wide_on_wide_val.txt` | same, `--config configs/segmentation_{anchored,wide}.yaml` | 2 checkpoints + 2 datasets |
| `seg_eval_{anchored,wide,anchored_18650,anchored_crown}_on_cad_test.txt` | same, `--config configs/segmentation_cad_test.yaml --per-sku` | 4 checkpoints + `dataset3d_seg_cad_test` |
| `seg_eval_cad_control_AnkerPowerCore{10000,13000,20100,26800}_on_cad_test.txt` | same | 4 checkpoints + `dataset3d_seg_cad_test` |
| `seg_ablation.txt` | `python -m recog.seg_ablation` | `seg/best.pt` + `dataset3d_seg` |
| `tau_calibration.txt` | `python -m recog.calibrate_tau --checkpoint … --config …` | `seg/best.pt` + `dataset3d_seg` |
| `main_seg_run.txt` | `python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt` | `seg/best.pt` + `dataset3d_seg/images` |

Two of these were re-derived during this audit and **matched the committed
file exactly** — see §2. That is the strongest single piece of evidence for the
project's central claim, and it is worth stating plainly: given the artefacts,
the current receipts are real, and they re-derive.

### Class C — not reproducible: no surviving script (16)

| receipt | why not |
|---|---|
| `ffdh_ablation.csv`, `ffdh_ablation.txt` | no committed generator; the string `ffdh_ablation` appears nowhere in the tree outside `docs/` |
| `frcnn_map.txt`, `frcnn_map_default.txt` | `recog/evaluate.py` provides `mean_ap` but has **no CLI and no `__main__`**; nothing writes these files. `_default` additionally needs the gitignored `default_anchors_best.pt` |
| `frcnn_latency.txt` | no producing script. The file is also **damaged**: its title line is repeated 66 times, each prefixed `=` — a separator line written into a loop. It has evidently never been regenerated |
| `heuristic_ablation.txt` | no producing script |
| `heuristic_failure_picks.json`, `heuristic_failure_taxonomy.json` | no producing script |
| `pr_summary.txt` | no producing script |
| `train.log`, `train_curve.csv`, `train_default.log`, `train_curve_default.csv` | `recog/training.py` writes none of these paths; the strings appear nowhere in the code |
| `train_eval.txt` | a hand-written narrative of an April run on "100 synthetic images" from the retired cv2 generator; that dataset and that training configuration no longer exist |
| `tau_independence_correlation.txt` | **self-declared**: "method: ad hoc analysis script, **not part of the committed CLI surface**" |
| `git-log.txt` | not a receipt: a placeholder stating "No git history available — project was developed outside a versioned workspace". Superseded by the actual history since 2026-08-05, and still cited once in the docs |

**`tau_independence_correlation.txt` is the one that matters.** It is the
evidence that retired the `tau` confidence gate — cited twice in the docs and
twice in shipping code (`plan/arbitration.py:126`, `plan/placement_area.py:536`)
as the justification for a design decision. It is the only *load-bearing,
current* number in the corpus whose tool was never committed. Everything else
in class C is inherited April-2026 material.

**Why class C is uniformly unreachable (measured).** `git log --diff-filter=D
-- '*.py'` returns **nothing**: no producing script was ever committed and then
deleted. These tools were never in version control at all. The first commit
(`69fad79`, 2026-08-05) is "baseline commit of existing auto-pick project", and
all 16 class-C receipts carry April 2026 mtimes — they predate the repository.

### Figures

Confirmed independently: `grep -rn "matplotlib\|savefig" --include=*.py`
returns **no hits outside `tests/`** (measured). None of the nine FDR figures
has a plotting script, so `fig4_latency.png` is not merely stale but
unregenerable, and the same is structurally true of the other eight. The
figures are a class-C corpus of their own.

---

## 2. Is the pipeline deterministic? (tested by re-running, per stage)

| stage | verdict | evidence |
|---|---|---|
| Scene generation — cv2 (`recog.synth_dataset`) | **byte-identical** | **(measured)** generated 5 scenes twice at `--seed 0`; `diff -r` clean across images, annotations and metadata |
| Scene generation — Blender (`recog/generate3d.py`) | **labels byte-identical; pixels not** | **(measured)** rendered 2 scenes twice at `--seed 0` with Blender 5.0: every `annotations/*.xml` and `meta/*.json` hash-identical; both PNGs differ. Magnitude: **max \|Δ\| = 1/255, on 0.01 % of pixels** |
| Training (`recog.seg_training`) | **non-deterministic — genuinely unseeded** | **(measured)** two 1-epoch runs, same config, same data: loss `1.7839` vs `1.7608`, selected mean IoU `0.4111` vs `0.3957`. The *split* is identical (715/126, seeded) |
| Evaluation (`recog.seg_evaluate`) | **all metrics byte-identical; embedded latency block is not** | **(measured)** two runs on the anchored val split differ in nothing but the 4-row wall-clock latency table |
| Planning / packing (`scripts/forbidden_bench.py`, `recog.seg_ablation`) | **byte-identical** | **(measured)** bench counts identical across runs and to the committed receipt; `recog.seg_ablation` re-derived its committed receipt **with zero diff** |
| Demo entry point (`main.py --config configs/demo.yaml`) | **non-deterministic outcome, deterministic perception** | **(measured)** 10 runs — see §5 |

### Where determinism breaks, and of which kind

**Genuinely unseeded — training.** There is no `torch.manual_seed`, no
`torch.use_deterministic_algorithms`, and no `cudnn.deterministic` anywhere in
the repository (measured, grep). Exactly three RNGs are seeded: the train/val
split generator in both trainers (`torch.Generator().manual_seed(seed)`), the
numpy fallback augmenter (`default_rng(0)`), and the procedural catalog
(`random.Random((seed * 1_000_003) ^ (i + 1))`). Everything else runs off
process-global RNGs seeded from OS entropy:

- `torch.initial_seed()` differs per process — **708248098508900** vs
  **708250640669900** across two runs (measured);
- a freshly constructed `Conv2d`'s weights hash differently per process
  (measured) — so **model initialisation is not reproducible**;
- `DataLoader(..., shuffle=True)` is constructed **without a `generator=`**
  in both `recog/training.py:340` and `recog/seg_training.py:389`, so the epoch
  order differs per process (measured: two different 20-element orders);
- `A.Compose(...)` in `recog/augmentation.py:145` is constructed without
  albumentations 2.x's `seed=` argument (read).

This is the important correction to make: the project's ~222 `seed` mentions
create the impression that training is seeded-but-unspecified. It is not. Two
runs of the same command produce different weights on the same hardware, and
the 1-epoch probe shows the effect is not small — a 1.5-point swing in the
selection metric after a single epoch. The repository already contains
independent corroboration: the README records three same-recipe segmenter
checkpoints scoring 0.211, 0.232 and 0.318 placeable fraction on the real
photos, and correctly reports the *series* rather than a point.

**Hardware/renderer-dependent — Blender pixels.** Same seed, same machine,
same Blender build, different pixels — but only just: 1 grey level on 1 pixel
in 10,000, which is Cycles' adaptive sampling (`adaptive_threshold: 0.01`) plus
GPU denoising, not a scene-description difference. The *labels* are exact.
Practical consequence: a regenerated dataset is semantically the same dataset
(same objects, same boxes, same masks), so metrics computed on it should track;
but no pixel-level or hash-level comparison will ever succeed, and a different
Blender version or a CPU render would move it further (read).

**Deliberate injected randomness — the demo.** See §5.

**Benign — wall-clock columns.** Every receipt that embeds a measured latency
(`seg_eval*.txt`, `forbidden_bench.txt`, `seg_eval`'s batching table) cannot be
byte-reproduced, because those numbers *are* the measurement. The project
already recognised this for `forbidden_bench_timings.csv` and moved timings to
a gitignored file precisely so regeneration produces a clean diff — but did not
apply the same treatment to the `seg_eval*` latency block, which is why those
11 receipts still churn on every regeneration. That is the mechanism behind the
README's own note that the batching figure "moves a little each time".

---

## 3. Is the provenance chain intact?

**No. It is the weakest link in the whole scheme.** Of 34 committed receipts:

| provenance element recorded | receipts |
|---|---:|
| git commit / code state | **0 of 34** |
| the command that produced it | **2 of 34** (`main_seg_run.txt`, `forbidden_bench_seeds.txt`) |
| a generation timestamp | **1 of 34** (`main_seg_run.txt`) |
| checkpoint + config + dataset *paths* | 12 of 34 (the `seg_eval*` / `seg_ablation` / `tau_calibration` family) |
| a content hash of any input | **0 of 34** |

Consequences worth naming:

1. **No receipt can be dated from the clone.** git does not preserve mtimes, so
   a fresh clone stamps all 34 files with checkout time. The April/August
   ordering I used above came from *this working tree's* mtimes, which a reader
   of the repository does not have. The only durable ordering signal is
   `git log -- <receipt>`, and for the 16 class-C receipts that log begins at
   the baseline commit that imported them — it dates the import, not the
   measurement.

2. **Paths are not identities.** The 12 receipts that do record their inputs
   record `recog/checkpoints/seg/best.pt` — a *mutable* path that every
   training run overwrites, and one that is not in the repository. Two receipts
   naming the same path may have been produced from different weights, and
   nothing in either file would show it.

3. **The 16 class-C receipts have no code state at all**, because their tools
   were never committed. This is exactly the failure mode the brief describes —
   a figure measured at one code state compared against another — and it is
   structural, not accidental, for that third of the corpus.

4. **`docs/README.md` overstates the position**: it describes `receipts/` as
   "Regenerated by the commands recorded alongside each claim". For the seg
   family that is defensible — the commands are in the repository README's
   entry-point table — but 16 receipts have no such command anywhere.

**The bright spot: datasets have better provenance than receipts do.** Each
`manifest.json` records the seed, the *fully resolved* config (render, layout,
camera, filter, obstruction, tray bands), the asset catalog with measured
extents, and corpus statistics; each per-scene `meta/*.json` records every
drawn parameter — exposure, zoom, camera shift, per-material colours,
lighting kelvin/strength. And the generating command is recorded in the
consuming config's comments (verified in `configs/segmentation_cad_test.yaml`
and `configs/segmentation_anchored.yaml`). What the manifest does **not**
record is the command line, the Blender version, or the generator's commit.
Adding those three fields, plus a commit line in every receipt header, would
close most of this section at negligible cost.

---

## 4. What would it actually cost to reproduce the headline result?

Measured on this machine (RTX 3060 12 GB, torch 2.13+cu126, Blender 5.0).

**Per-unit costs (measured unless noted):**

| operation | cost |
|---|---|
| Blender render, 1280×720, 192 samples, GPU | **~4.2 s/scene** (2-scene run: 15.8 s cold incl. startup, 10.8 s warm). The README's "~3.5 s per frame on an RTX 3060" is honest |
| Segmenter training, 40 epochs | **~20 s/epoch** measured on 841 crops → **~19 min/run** wall, corroborated by the spacing of the four CAD-control `best.pt` mtimes (19, 21, 15, 20 min) |
| Segmenter evaluation, 836-crop CAD test set | **32 s** |
| Detector (Faster R-CNN) training, 35 epochs × 1000 images | **~4–4.5 h** *(inferred from checkpoint mtime spacing on 2026-08-06, not re-run)* |
| Disk, one 500-scene seg dataset | **~500 MB** (measured: 497–514 MB across seven of them) |

**To reproduce the headline generalisation result** — eight models scored on
836 held-out CAD test crops:

| item | quantity | cost |
|---|---|---|
| Blender renders | 8 training sets + 1 CAD test set ≈ 4,500 scenes | **~5.3 GPU-hours** |
| Segmenter training | 8 runs × 19 min | **~2.5 GPU-hours** |
| Evaluation | 8–10 runs × 32 s | **~6 minutes** |
| Disk | 9 datasets + 8 checkpoints | **~4.9 GB** |
| **Total** | | **≈ 8 GPU-hours, ~5 GB, one CUDA GPU + a Blender 5.0 install** |

**To reproduce everything in `docs/receipts/` that is reproducible at all**,
add `dataset3d` (1000 scenes, ~1.2 h render, 985 MB), `dataset3d_seg` (502
scenes, ~0.6 h, 509 MB), the detector training run (~4.5 h) and the original
segmenter (~0.3 h): **≈ 15 GPU-hours and 8.3 GB** (the measured size of
`recog/` on this machine, of which 2.4 GB is checkpoints).

**Two caveats a reader deciding whether to trust this should be told.**

1. **The cost is all upstream, and the expensive step is the unseeded one.**
   The evaluations that produce the published tables are cheap (32 s) and
   exact. The 8 GPU-hours buy the datasets and the checkpoints — and because
   training is unseeded (§2), a from-scratch reproduction will *not* return the
   published IoUs. It will return a sample from the same distribution. The
   comparison the headline rests on is *between* eight models trained under
   identical conditions, which is the right design and is robust to this; but a
   reader who reproduces one number and finds it 0.01 off has not found a
   discrepancy, and nothing currently tells them that.

2. **Nothing here needs CAD conversion or the source STEP files.** The four
   converted `.glb` assemblies and `catalog.json` **are committed** (~860 KB
   total); `cad/*.stp` is gitignored and is not required. That is a good
   decision and removes what would otherwise be the hardest dependency to
   satisfy. Blender itself is the remaining non-pip dependency.

---

## 5. The demo path — why it is non-deterministic

`python main.py --config configs/demo.yaml` is the foundation of the
reproducibility claim, and `configs/demo.yaml` carries a 20-line comment
attributing its variable outcome to CUDA:

> "its per-cycle pick/place outcome is then run-to-run non-deterministic
> (observed: 10/0, 9/1, 8/2 placed/pick_failed splits) — almost certainly
> CUDA/cuDNN algorithm-selection jitter on box scores that sit near main.py's
> decision thresholds … the mock KUKA server and HeuristicDetector are
> themselves fully deterministic"

**That explanation is wrong, and I can rule it out by measurement.** I ran the
demo **10 times** (measured). Nine runs gave `placed: 10, pick_failed: 0`; one
gave `placed: 9, pick_failed: 1`. In **all ten runs, every perception and
planning number was identical**:

```
cartridges_detected: 37   batteries_detected: 77   placement_areas: 33
queue_poses: 62           placement_disagreements: 0  bad_detector_boxes: 0
```

A full normalised log diff across three of those runs was empty except for the
mock server's ephemeral TCP port and wall-clock timings. If the cause were box
scores jittering across a decision threshold, the detection and queue counts
would move. They do not — not once in ten runs.

**The actual cause is deliberate simulated hardware failure** (read,
`execution/mock_kuka_server.py:96,110`):

```python
if random.random() < self.drop_prob:          # vacuum failed to grip
    self.vacuum_on = False
    return total, _PICK_FAILED
...
if random.random() < self.drop_prob / 2:      # dropped on insertion
    return total, _PLACE_FAILED
```

`drop_prob` comes from `configs/execution.yaml`'s `simulation.drop_probability:
0.02`, wired in at `main.py:281`. The module-level `random` is **never seeded**
anywhere in the repository. Per cycle, P(failure) = 0.02 + 0.98 × 0.01 =
0.0298, so P(a 10-cycle run is not 10/10) = 1 − 0.9702¹⁰ = **26 %**. Observed
1 of 10 runs — well inside binomial noise for p = 0.26.

So the answer to "seeded randomness, a race, or genuine sampling?" is:
**genuine sampling, injected on purpose, from an unseeded global RNG.** Three
things follow:

1. **It is not GPU-related at all.** The injection sits behind the robot
   protocol, downstream of everything torch touches. A clean clone with no
   torch — the torch-free path `demo.yaml` is designed around — will show the
   *same* variance, contradicting that comment's claim that the heuristic path
   is "fully deterministic". The comment's own observation (9/1, 8/2) is
   consistent with a 2 % drop rate and inconsistent with detector jitter, since
   detector jitter cannot produce a `pick_failed`.
2. **It is defensible behaviour**, correctly documented at its source (the mock
   server's docstring says failures are "configurable via `drop_prob` so tests
   can" exercise them) — it exists so the retry and re-plan logic is exercised.
   The defect is purely in the explanation offered 400 lines away.
3. **It is trivially fixable** if determinism is wanted: seed `random` in the
   mock server, or set `simulation.drop_probability: 0.0` for the demo config
   and keep the injection for tests. Either would let the demo promise its
   placed count, not just "cycle count, latency profile and no crash".

---

## Summary of gaps, in priority order

1. **`tau_independence_correlation.txt` has no committed tool** — a current,
   load-bearing receipt cited in two source files as the justification for
   removing a design element. Commit the ad hoc script.
2. **No receipt records a commit.** One `git rev-parse HEAD` line per receipt
   header would close §3 items 1–3 for every future regeneration.
3. **`configs/demo.yaml`'s non-determinism comment misattributes the cause**
   to CUDA jitter; it is the mock robot's unseeded 2 % drop injection (§5).
4. **Training is unseeded, and the docs read as though it is seeded.** Either
   seed it (`torch.manual_seed`, a `generator=` on the shuffling DataLoader, a
   `seed=` on `A.Compose`) or state plainly that checkpoints are samples.
5. **16 receipts cannot be regenerated**, all inherited from before the
   repository's history. They should be marked as historical *in the receipt
   files themselves*, not merely be undated.
6. **`frcnn_latency.txt` is damaged** (title line repeated 66×) and
   `pytest-cov.txt` is stale (102 tests recorded; 752 today) — the latter is in
   class A and costs one command to refresh.
7. **`seg_eval*` receipts embed a wall-clock latency block**, which is why they
   never reproduce byte-exactly. The fix the project already applied to
   `forbidden_bench_timings.csv` applies here too.

## What was executed during this audit

All output went to a scratch directory outside the repository.

```
python scripts/forbidden_bench.py --no-write                       x2
python main.py --config configs/demo.yaml                          x10
python -m recog.synth_dataset --out <scratch> --n 5 --seed 0        x2
blender -b --python recog/generate3d.py -- --n 2 --out <scratch> --seed 0 --tray-set cad   x2
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_anchored/best.pt \
    --config configs/segmentation_anchored.yaml --out <scratch>      x2
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_wide/best.pt \
    --config configs/segmentation_cad_test.yaml --out <scratch>      x1
python -m recog.seg_ablation --out <scratch>                         x1
python -m recog.seg_training --config <scratch copy, epochs:1>       x2
```

No repository file was written except this report.
