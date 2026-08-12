# CI, and two tone cuts — 2026-08-12

Baseline: `f680d88`, 752 tests passing, tree clean. Two independent jobs,
committed separately. No code, config, metric definition or receipt was
changed; the suite is unchanged at **752 passed** (re-run at the end).

Files: `.github/workflows/ci.yml` (new), `docs/PORTFOLIO.md` (two
sentences removed), this spec. `README.md` and `docs/figures/` were owned
by another agent during this work and were not touched — including the
badge, which is left for the author to paste (§1.5).

---

## 1. Continuous integration

### 1.1 What it runs

`.github/workflows/ci.yml`, on every push and every pull request, one job
on `ubuntu-latest`, Python **3.14**:

1. `sudo apt-get install -y libgl1` — `opencv-python` (not `-headless`)
   is a hard dependency and its manylinux wheel links `libGL`, which a
   headless runner does not ship.
2. `pip install -e .` — verbatim the README's "Running it" step 1, on a
   clean checkout.
3. `pip install -e ".[dev]"` — the README's dev install; adds pytest and
   pytest-cov, nothing else.
4. `pytest -q`.
5. `python -m recog.synth_dataset --out recog/dataset --n 50` then
   `python main.py --config configs/demo.yaml` — the README's steps 2 and
   3, i.e. the torch-free demo loop.

Step 2 is the point of the workflow as much as step 4 is. `pip install
-e .` failed outright on a clean checkout until `3fedcb6` (setuptools'
flat-layout discovery refused to guess between the top-level packages),
which is the first thing a reader hits and the last thing the local dev
environment can notice, because an already-installed editable package
does not re-run discovery.

### 1.2 What it covers, measured rather than assumed

Method: `git clone` of `f680d88` into a scratch directory (so no
gitignored artefact from the dev tree could leak in), fresh venv, then
exactly the steps above.

| environment | result |
| --- | --- |
| dev machine, all extras present | **752 passed**, 0 skipped |
| clean clone, base + `[dev]` only | **719 passed, 4 skipped** |

So CI exercises **719 of the 752 tests**, ~95 %. The 33 it does not:

| tests | needs | guard |
| --- | --- | --- |
| 30 (`tests/test_bay_segmenter.py`) | torch | module-level `importorskip("torch")` |
| 1 (`tests/test_arbitration.py`) | torch | `importorskip("torch")` |
| 1 (`tests/test_seg_dataset.py`) | a Blender-generated COCO dataset (`recog/dataset3d_seg`), and torch | `skipif` |
| 1 (`tests/test_synth3d.py`) | `cascadio` (the `[cad]` extra, STEP → glTF) | `skipif` |

Each of those guards itself, so the CI run is *green*, not green-because-
filtered: nothing is deselected by a `-k` or a marker expression in the
workflow. `pytest -q` in CI is the same command the README documents, and
the skip lines appear in the log.

### 1.3 What CI deliberately does not cover

- **Model training and the learned segmenter.** `[train]` (torch,
  torchvision, tensorboard, scipy) is a ~GB install and the tests behind
  it are the ones that most want a GPU. Not installed.
- **Blender.** `recog/generate3d.py` imports `bpy`; the 3-D dataset,
  every render-derived receipt and the `demo_seg.yaml` path are outside
  CI entirely. This is a pre-existing property of the project (those
  modules are documented as not unit-testable), not a new gap.
- **STEP → glTF conversion** (`[cad]`: trimesh, cascadio).
- **Python versions other than 3.14.** `pyproject.toml` declares
  `>=3.10` and the README says "Python 3.10+"; only 3.14 has actually
  been verified from a clean clone, so only 3.14 is pinned. A matrix over
  3.10–3.14 would either widen that claim honestly or turn the badge red;
  it is a deliberate follow-up, not an oversight.
- **Anything on real hardware.** There is none — see FDR §10.3.

The badge therefore means: *a fresh clone installs with the documented
command and the torch-free 95 % of the suite plus the demo loop pass on a
GPU-less Linux runner.* It does not mean the training path is exercised.

### 1.4 One thing that could not be verified locally

The workflow has never been executed — this machine is Windows and the
repository has no `origin` remote yet, so the first real run will be the
one after publication. The three failure modes worth knowing about, in
order of likelihood:

1. `libgl1` insufficient — if `import cv2` also wants
   `libgthread-2.0.so.0`, add `libglib2.0-0t64` to the apt line (the
   name changed in the 64-bit `time_t` transition; it is `libglib2.0-0`
   before Ubuntu 24.04). GitHub's runner images preinstall glib via the
   bundled browsers, which is why it is not in the line already.
2. `actions/setup-python@v6` not offering 3.14 on the runner image.
3. Nothing else: the install, the 719 tests and the demo loop were all
   run end-to-end in a clean 3.14 venv from a clean clone of this commit,
   with `main.py --config configs/demo.yaml` exiting 0 in ~17 s.

### 1.5 The badge, for the author to paste into `README.md`

```markdown
[![CI](https://github.com/yousifalh/auto-pick/actions/workflows/ci.yml/badge.svg)](https://github.com/yousifalh/auto-pick/actions/workflows/ci.yml)
```

The owner is a guess from `git config user.name` — there is no remote to
read it from. Correct `yousifalh` to whatever the published path is.

---

## 2. Two tone cuts in `docs/PORTFOLIO.md`

Two of the five sentences flagged as performed rather than plain in the
2026-08-12 verification pass; the author kept the other three, which are
untouched. Both cuts are removals only — no rewriting, no new claims.

**"Where the gap actually was", final paragraph.** An unverifiable
self-audit sitting inside a claim about honesty; the surrounding
documents are the evidence for it, and asserting it invites the reader to
go hunting for the exception.

> ~~The honest claim is narrower and I have stated it that way
> everywhere:~~ → The honest claim is narrower:

**"Knowing when to stop", second paragraph.** The paragraph already does
this; naming it repeats a move that was cut elsewhere in the same pass
for the same reason.

> ~~That is not the number I first published, and the correction is the
> point.~~ → That is not the number I first published.

---

## 3. Verification

- `pytest -q` on the dev machine after both edits: **752 passed**.
- `git status --porcelain` before each commit: only the intended paths.
- `README.md` and `docs/figures/`: not opened for writing at any point.
