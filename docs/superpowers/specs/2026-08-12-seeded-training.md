# Seeding training, and the receipt that could not be regenerated

**Date** 2026-08-12 · **At** `51f3590`
**Job 1** regenerate `docs/receipts/main_seg_run.txt` — **blocked, see §1**.
**Job 2** make training reproducible — done, §2 onward.
**Files** `recog/seeding.py` (new), `recog/seg_training.py`,
`recog/training.py`, `configs/segmentation*.yaml`, `configs/recognition.yaml`,
`scripts/seed_check.py` (new), `tests/test_seeding.py` (new),
`docs/receipts/seed_reproducibility.txt` (new), README, FDR v3 Appendix C,
NEXT_STEPS, PORTFOLIO, CV_BULLETS.

---

## 1. Job 1: `main_seg_run.txt` cannot currently be regenerated at all

The receipt was to be refreshed because two behaviour-changing fixes landed
after it was last written (`a649f43`, whole-footprint reservation;
`d48dcbc`, latching E-stop and the removal of `place.z_mm`). It was not
refreshed, and it was not hand-edited. **Its own command aborts.**

```
$ python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt
...
2026-08-12 16:14:58 [INFO] autopick.main: cycle=7 perc=1651.5ms plan=4.5ms cartridges=4 masks=4 queue=0
plan.scene.OutOfWorkspace: place target for cartridge 14 at (618.9, 399.2) mm
is outside the robot workspace x [-350.0, 350.0] y [-350.0, 350.0] mm
```

`configs/demo.yaml`, the torch-free demo the README's reproducibility claim
rests on, aborts the same way (`place target for cartridge 0 at (358.9,
145.0) mm`). Both were run at `51f3590` with a clean tree, and `demo.yaml`
was re-checked at `ad1796c` (after another agent's `01001b7`, which changed
the pick Z) and aborts identically.

**The cause is understood and is one of the two named fixes**, so this is not
a surprise in the "some third thing moved" sense. `a649f43`'s second half
(audit E finding 5) made `WorkspaceBounds` load-bearing for the first time:
every pose is now checked and an out-of-envelope pose **raises rather than
clamps**, deliberately, because clamping a place target inserts a cell into a
wall and records it PLACED. The check is right. What it has found is that the
demo configuration was never internally consistent:

| quantity | value at `51f3590` |
|---|---|
| `planning.camera.origin_offset_{x,y}_mm` | `0.0`, so pixel (0, 0) maps to the robot origin and the whole field of view lies in +x/+y |
| `planning.camera.workspace_bounds_mm` | ±350 mm about that origin |
| `demo.yaml` frame extent | 1280 × 720 px at 0.38 mm/px = **486 × 274 mm** |
| `demo_seg.yaml` frame extent | 1280 × 720 px at the sidecar's own 0.490–1.045 mm/px = **627 × 353 to 1338 × 752 mm** |

No offset makes the second row fit the fourth: a 0.7 m envelope cannot contain
a 1.34 m field of view, centred or not. So the envelope, the origin offset and
the frames disagree, and until `a649f43` nothing compared them —
`configs/planning.yaml` even labels that block "Replace with real intrinsics
when available".

**Not fixed here, and deliberately.** Choosing new numbers decides what the
demo depicts (how big the table is, where the robot base sits), which is a
design decision about the cell and not a receipt regeneration; and quietly
widening an envelope until the run completes is the exact move this project
refuses everywhere else. The commit's own message shows the author widening
`_scaled_planner`'s test fixture from ±350 to ±500 mm for the same reason —
"a property of the framing rather than of the scale handling those tests
measure" — so the mismatch was seen in the fixture and not carried through to
the shipped configs.

Also relevant to whoever picks this up: the working tree was being edited
concurrently by another agent (`plan/planner.py`, `execution/protocol.py`,
`main.py`, three test files) while this was measured. A receipt regenerated
against a tree in that state would correspond to no commit, which is a second
reason to leave it.

**Consequences, unrecorded elsewhere as of this writing:**

1. `docs/receipts/main_seg_run.txt` (2 picks / 7 areas / 6 poses) describes a
   run that no longer completes. Nothing in the docs quoting it — README, FDR
   §13.1.1 area, NEXT_STEPS — has been changed, because the numbers *were*
   true when measured and the code that would produce new ones is being
   edited right now.
2. The README's 10-run demo table (37 cartridges, 77 batteries, 33 placement
   areas, 62 queue poses) is in the same position: measured 2026-08-12, on a
   command that now aborts partway.

---

## 2. Job 2: what was unseeded, and what is seeded now

Audit D measured training as *genuinely* unseeded — not
hardware-nondeterministic, unseeded: no `torch.manual_seed`, no
`use_deterministic_algorithms`, `DataLoader(shuffle=True)` with no
`generator=`, `A.Compose` with no `seed=`. Two one-epoch runs gave loss
1.7839 vs 1.7608 and selected mean IoU 0.4111 vs 0.3957.

`recog/seeding.py` is now the single entry point, called by both trainers
**before** the transforms, the loaders or the model are built (weight
initialisation draws from torch's global CPU generator the moment
`build_segmenter` runs, so seeding after it would seed everything except the
initialisation).

| component | how it is seeded now |
|---|---|
| Python `random` | `random.seed(seed)` |
| NumPy legacy global | `np.random.seed(seed)` |
| torch CPU + every CUDA device | `torch.manual_seed` + `cuda.manual_seed_all` |
| shuffle order | `DataLoader(generator=…)` from `dataloader_kwargs(seed)` |
| dataloader workers | `worker_init_fn=seed_worker`, which re-seeds `random` and `numpy` per worker from the worker's own torch seed |
| augmentation | `Compose.set_random_seed(seed)` (albumentations 2.x keeps a per-instance RNG the global seed does not reach); the numpy fallback augmenter's `.rng` is replaced |
| crop jitter | `BaySegDataset(seed=run_seed)` — follows the run seed so two seeds are independent samples in every stochastic component |
| train/val split | **deliberately not** the run seed. `dataset.split_seed` stays its own knob: the split defines what the metric is measured *on*, so changing the seed must not change the yardstick |

The seed comes from `training.seed` (default `DEFAULT_SEED = 20260812`),
`--seed` overrides it, the resolved value is logged with a three-RNG
fingerprint, and it is written into `best.pt`, `last.pt` and
`train_state.pt`. `--resume` restores the saved RNG states (so a resumed run
continues the stream rather than replaying epoch 0's shuffle) and **refuses**
a seed different from the one the checkpoint recorded.

### 2.1 Everything that could silently no-op raises

The brief's own framing: a seed that is set but not used would be this
project's characteristic defect in its purest form. So —

* `seed_everything` re-reads `torch.initial_seed()` and raises if it differs.
* `assert_loader_seeded(loader, …, expected_seed=…)` raises unless the loader
  actually carries this module's generator *and* `seed_worker`, and unless
  the generator's `initial_seed()` is the run's seed. A generator built and
  then not passed through is the failure it exists for.
* `seed_transform` raises on any pipeline it cannot seed, and raises again if
  `set_random_seed` returned without taking (`tests/test_seeding.py`'s
  `_LyingTransform`).
* `normalise_deterministic` raises on an unrecognised value rather than
  falling back to "off", which would silently drop a determinism request.
* Structural tests parse both trainers' `train()` and fail if any seeding
  step disappears, if a shuffling `DataLoader` is built without
  `**dataloader_kwargs(...)`, or if a checkpoint payload stops recording the
  seed.

### 2.2 A regression caught on the way

`seed_everything`'s record first carried `torch.__version__` itself. That is
a `TorchVersion` object, and a checkpoint containing one **cannot be loaded
with `weights_only=True`** — which is how `BaySegmenter` and the detector
loader have read `best.pt` since the 2026-08-12 security pass. Every new
checkpoint would have been unloadable by the only inference path in the
project. It is a `str` now, and
`test_the_record_survives_a_weights_only_checkpoint_round_trip` pins it.

---

## 3. `use_deterministic_algorithms`: the decision, measured

Seeding every RNG was **not sufficient**, and finding that out is the most
useful thing in this work. A six-step, two-process probe (RTX 3060, torch
2.13+cu126, cuDNN 9.10.2) with everything seeded and no kernel constraints:

* initial weights **bitwise identical** (`32e0740aca680998` both runs);
* every input batch's image and label hashes **identical** at all six steps —
  so shuffle, jitter and augmentation are fully fixed;
* step 0's loss already differs: `3.622172117233` vs `3.622171878815`
  (~7e-8, one float32 ULP), and by step 5 it is `2.510210037231` vs
  `2.510541439056`.

The divergence is arithmetic, not sampling: cuDNN chooses convolution
algorithms by heuristic and several backward kernels accumulate with atomics.
One epoch of SGD amplifies it into the reported metrics — two same-seed pairs
measured 0.0197 and 0.0076 apart in mean loss and 0.0101 and 0.0409 apart in
selected mean IoU, which is the size of the *unseeded* spread. A default of
"seeded only" would therefore have shipped a reproducibility claim that the
project's own probe refutes.

Three modes were measured; `training.deterministic` selects one.

| mode | what it sets | measured |
|---|---|---|
| `off` | nothing beyond the seeds (`cudnn.benchmark=False` in all modes) | same-seed runs diverge as above. 26.3 s mean per 1-epoch run |
| **`warn`** (default) | `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)` | **loss, selected IoU and the weight SHA-256 identical across two runs** at one seed; forward is bit-stable from step 0. 28.0 s mean per run |
| `strict` | the same with `warn_only=False` | **does not train this model**: `RuntimeError: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation` on the first batch |

`warn` and not `strict` because strict does not run; `warn` and not `off`
because off does not reproduce. The cost is ~1.7 s on a 26 s run whose
compute is ~11 s — call it 15 % of the compute, roughly +2 minutes on the
40-epoch schedule, against a 19-minute run.

The one op that stays nondeterministic under `warn` is named rather than
glossed: the NLL loss's forward *reduction*. Its gradient does not depend on
the summation order, which is why the weights still come back bit-identical
while a printed loss can move by an ULP — and that is exactly the shape of
claim this project should be making about the ops it cannot pin.

Nothing in any mode changes the metric definitions, the architecture or the
schedule. Seeding changes only whether the run repeats.

---

## 4. The proof, and how to re-run it

`python scripts/seed_check.py` (≈90 s) runs `recog.seg_training` three times
on `configs/segmentation_seedcheck.yaml` — one epoch each, twice at one seed
and once at another, into a scratch checkpoint directory it refuses to point
at `recog/checkpoints/seg` — and writes
`docs/receipts/seed_reproducibility.txt`.

```
  run   seed        loss      selected IoU   wall s   weights sha256
  A     20260812    1.8164    0.3590           28.0   2b587394efd2eeb6
  B     20260812    1.8164    0.3590           27.9   2b587394efd2eeb6
  C     20260813    1.7972    0.3426           28.0   d4a4a86e3305daa6
```

Same seed: loss delta 0.000000, IoU delta 0.000000, weights bit-identical.
Different seed: loss delta 0.0192, IoU delta 0.0164. Exit status is 0 only if
both of those hold.

At `--deterministic off`, for contrast, the same three runs gave A 1.8126 /
0.3415, B 1.8050 / 0.3824, C 1.7921 / 0.3432 — the same-seed pair further
apart on IoU (0.0409) than the different-seed pair (0.0017).

---

## 5. The residual, stated precisely

* **"Reproducible on the same machine and toolchain"** is the claim. A
  different GPU, driver, cuDNN or torch build can change the arithmetic and
  nothing here prevents that. Unqualified "reproducible" would be false.
* **The NLL forward reduction is still nondeterministic** under the default
  mode. Weights repeat; a reported loss can move by a float32 ULP.
* **`strict` is unusable on this model** until that op gains a deterministic
  implementation upstream. It is left available because it is the honest way
  to ask the question on some future model or torch version.
* **The published checkpoints cannot be recovered.** Every checkpoint in
  `recog/checkpoints/` and every figure derived from one comes from the
  unseeded era. Seeding is not retroactive. This is now said in the README,
  FDR Appendix C item 4, NEXT_STEPS, PORTFOLIO and CV_BULLETS rather than
  implied.
* **Only the trainers are covered.** The mock KUKA server's 2 % simulated
  vacuum drop is still drawn from an unseeded module-global `random` — on
  purpose, documented in `configs/demo.yaml`, and out of scope here. Blender
  renders remain label-exact and pixel-inexact.
* **Detector seeding is untested end to end.** `recog/training.py` gets the
  same treatment and the same structural tests, but the 4.5-hour detector run
  was not executed; the measured evidence in §3 and §4 is the segmenter's.

---

## 6. Suite

1026 passed at the end of this work (the 60 new `tests/test_seeding.py` cases
included), with `tests/test_synth3d_world.py`'s 149 belonging to another
agent's concurrent work in the same tree.
