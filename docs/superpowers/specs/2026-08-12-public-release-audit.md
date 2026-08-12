# Public-release audit

Written 2026-08-12 against `2a6e96c`, before the repository is made public on
GitHub. Scope: find what would embarrass or block a public release. Everything
below was checked by running it, not assumed. Items marked **[author]** were
deliberately left alone because they are the author's call, not the audit's.

---

## 1. Personal and sensitive data

### The student ID — REDACTED

Present in three files at the time of audit:

| File | Line | Action |
| --- | --- | --- |
| `docs/FDR.md` | 5 | **Removed** (title block now reads `Yousif Al-Haidary`). |
| `docs/FDR_v3.md` | 6 | **Removed** (`Student ID:` line deleted). |
| `README.md` | 5 | **[author]** Not touched — another agent owns this file during this session. **This still needs removing before the repository goes public.** |

A student ID is a university identifier tied to a named individual. It buys a
public reader nothing and it is exactly the sort of field that gets scraped.

### It is in the git history, and deleting it now does not remove it

`REDACTED` appears in the tree of **all 168 commits** — it entered at the
baseline commit `69fad79` and has been in `docs/FDR.md` ever since. The edits
above remove it from the *working tree only*. Anyone can still recover it with
`git log -S REDACTED` or by checking out any earlier commit.

Removing it from history genuinely requires rewriting history —
`git-filter-repo` (or BFG), then a force-push — which **changes the SHA of
every commit in the repository**. That is not a cleanup; it is a different
repository. **[author]** This was not done, per the standing instruction not to
alter history. Three options, in the order I would consider them:

1. **Publish a fresh repository with a squashed or truncated history.** Cleanest
   outcome, loses the sprint-by-sprint commit progression that FDR §12.1 refers
   to as evidence of process.
2. **`git-filter-repo --replace-text` before the first push.** Keeps the shape
   of the history; every SHA changes. Harmless *if done before publishing*,
   because no one has cloned it yet. This is the cheapest moment to do it — the
   cost rises sharply the day after the repository goes public.
3. **Accept it.** The ID on its own is low-severity; it is not a credential and
   grants no access. Reasonable if the author judges the process evidence worth
   more.

Option 2 is only free *right now*. That is the decision this audit most wants
in front of the author.

### Everything else came back clean

Checked across all 198 tracked files:

| Checked for | Result |
| --- | --- |
| Email addresses | None. |
| API keys, tokens, passwords, private keys (`ghp_`, `sk-`, `AKIA`, PEM headers) | None. The one regex hit was the word "token" in a `test_synth3d.py` assertion message. |
| Absolute paths containing a username (`C:\Users\...`, `/home/...`, `/Users/...`) | None. |
| Phone numbers, postcodes, postal addresses | None. The one hit was COCO bounding-box data in `recog/realtest/annotations/instances_default.json`. |
| **EXIF / GPS in the 7 tracked photographs** | **None — already stripped.** `recog/realtest/images/IMG_44*.jpg` carry no EXIF block at all, so no GPS coordinates and no device identifiers. |

The supervisor's name (Dr Svetan Ratchev) appears in `README.md` and the FDRs.
**[author]** Left as-is — naming an academic supervisor is normal attribution
and he is a public figure at the university, but it does name a third party, so
the author may want to confirm he is content to be named.

---

## 2. A fresh clone — tested, not assumed

Cloned to a scratch directory, installed into a **clean Python 3.14 venv with
no GPU, no camera, no controller and no checkpoint**, and ran the README's
three-step sequence.

### The blocker: `pip install -e .` failed outright

README step 1 did not work on a clean checkout:

```
error: Multiple top-level packages discovered in a flat-layout:
       ['cad', 'plan', 'recog', 'common', 'configs', 'execution'].
```

setuptools refuses to guess between top-level directories in a flat layout, and
it was also picking up the non-package data directories `cad/` and `configs/`.
Both `pip install .` and `pip install -e .` failed. A reader would have hit this
inside the first thirty seconds, on the very first command the README gives.

**Fixed** in `pyproject.toml` by declaring the real packages explicitly:

```toml
[tool.setuptools]
packages = ["common", "execution", "plan", "recog", "recog.synth3d"]
```

This is packaging metadata only. No code, no metric, no receipt is affected.

### With that fixed, the demo runs

All three README steps then succeeded end to end:

```
pip install -e .                                            # OK
python -m recog.synth_dataset --out recog/dataset --n 50    # 50 scenes
python main.py --config configs/demo.yaml                   # OK
```

Final line of the run:

```
Run summary: {'cycles': 9, 'placed': 9, 'pick_failed': 0, 'place_failed': 0,
'empty_queue': 1, 'cartridges_detected': 12, 'batteries_detected': 27, ...}
```

torch was never installed. `recog.inference.load_detector` logged the expected
fall-back to `HeuristicDetector`, the mock KUKA server handshook on
`127.0.0.1`, and the loop completed. **The README's central claim — that the
whole loop runs with no GPU, no camera and no controller — holds.**

### Step 2 is load-bearing, not optional

`configs/demo.yaml` reads `mode.img_dir: recog/dataset/images`, and
`recog/dataset/` is correctly gitignored as regenerable. So the frames are
genuinely absent from a fresh clone and step 2 must be run. The README already
gives the three steps in order, so this is not a defect — but it does mean
skipping to step 3 fails, and the config's own comment claiming the demo frames
are "small and always present" is misleading about a directory that is not in
the repository.

**[author]** Not edited — it is a comment inside a config the audit was told not
to change behaviourally, and the README sequence is already correct.

### Dependencies now match what is imported

Three third-party imports were undeclared. All are lazily imported and none is
needed for the demo or the test suite, but they were invisible to a reader:

| Package | Imported by | Now declared as |
| --- | --- | --- |
| `scipy` | `recog/seg_evaluate.py` (boundary-F1 distance transform) | added to the `train` extra |
| `trimesh`, `cascadio` | `recog/convert_cad.py`, `recog/synth3d/catalog.py` | new `cad` extra |

`bpy` / `bmesh` / `mathutils` are Blender's bundled interpreter and correctly
absent from `pyproject.toml`.

Also fixed: the README's `pip install -e '.[dev]'` referenced a `dev` extra that
**did not exist** — `pyproject.toml` defined only `train` and `test`, so that
command silently installed nothing. A `dev` extra was added as the umbrella
alias so the documented command resolves.

### Size and what is tracked

- **Fresh clone: 53 MB.** `.git` is 29 MB. Nothing large is wrongly tracked.
- The working directory is 8.5 GB, but that is entirely gitignored generated
  output (datasets, renders, checkpoints, `.blend` files) — none of it is in the
  repository.
- Largest tracked files are the 7 real photographs (~2–2.8 MB each, ~17 MB
  total) and the two FDR PDFs (~2.2 MB). All are legitimately part of the work.
- Checkpoints (`recog/checkpoints/`), datasets and source CAD (`cad/*`) are
  correctly excluded, and the demo works without any of them.

### `.claude/` was untracked but not ignored

A 24 MB `.claude/` directory — Claude Code local state, including agent
worktrees holding **whole copies of the repository** — was sitting untracked and
unignored, one `git add -A` away from being committed. **Added to
`.gitignore`.**

---

## 3. Licensing — the real gap

**There is no `LICENSE` file, and no copyright header anywhere in the tree.**

Without one, default copyright applies: the code is "all rights reserved" by
law. Nobody may legally copy, modify or reuse it. For a portfolio repository
this actively works against the goal — a reviewer who wants to borrow an idea,
or an employer's legal team assessing it, finds no permission granted.

**Recommendation: MIT.** It is the conventional choice for a portfolio
repository, is short, permissive, and is what a reader will expect.

**[author] Not added, because it is not purely a technical call:**

1. **The university may have a claim.** This is an MEng individual project
   carried out at the University of Nottingham under supervision. IP in student
   project work is governed by the university's IP regulations, and those vary
   by institution and by whether the project was industrially sponsored. The
   author should confirm he holds the rights before granting anyone a licence.
   Licensing work you do not own is worse than shipping no licence at all.
2. **Code and non-code may want different terms.** MIT is a software licence.
   The CAD, the renders, the photographs, the figures and the FDR text are not
   software. A common split is MIT for the code and CC BY 4.0 for the documents
   and media. Worth a sentence in the README either way.

Third-party attribution was checked and is clean: **no vendored code**, no
copied source files, no `Copyright (c)` / `SPDX-License-Identifier` /
"Adapted from" markers anywhere in the tree.

One thing to note: the CAD assets and the derived `.glb` files are named for
commercial products (`AnkerPowerCore10000`, and so on), and the tracked
photographs are of real retail battery packs. The geometry was measured and
modelled by the author, so the models are his own work — but "Anker" and
"PowerCore" are third-party trademarks. **[author]** Nominative use to identify
what was modelled is ordinarily fine; a one-line disclaimer that the project is
unaffiliated with and unendorsed by Anker would remove any doubt.

---

## 4. Document hierarchy

`docs/` held eleven files with no signal about which mattered: three FDR
revisions (`FDR.md`, `FDR_v2.md`, `FDR_v3.md`, only v3 current), two PDFs, an
internal-facing `NEXT_STEPS.md`, and the portfolio material.

**Added [`docs/README.md`](../../README.md)** — an index stating what each document is
and, explicitly, which is current. GitHub renders it automatically when a reader
clicks into `docs/`, which is where this confusion actually happens. It states
that `FDR_v3.md` is authoritative, that `FDR.md` and `FDR_v2.md` are superseded
and must not be quoted, and that `NEXT_STEPS.md` is a dated working note
superseded by the FDR wherever the two differ.

All twelve links in it were verified to resolve.

### The archive subdirectory was deliberately not created

The obvious companion move — `docs/archive/FDR.md`, `docs/archive/FDR_v2.md` —
**was not done, on purpose.**

Those two files are referenced by path from `README.md` at **lines 67 and 294**,
and `README.md` is owned by another agent this session and off-limits to this
audit. Moving the files would have left two broken links in the front-page
README — precisely the class of defect this audit exists to prevent. The
standing instruction was to fix every reference or not move the file; the
references could not be fixed, so the files did not move.

**[author] Ready to execute once the README is settled.** The full reference set
is small — I checked every tracked file:

- `README.md:67`, `README.md:294` — the only true path references.
- `docs/receipts/train_eval.txt:35` — prose ("priority 1 in FDR_v2 §13.2"), not
  a path. Do not edit; it is a receipt.
- `docs/superpowers/specs/2026-08-11-doc-reconciliation.md:228` — prose mention.

So the move costs exactly two line edits in `README.md`. No `§` cross-reference
is affected: `FDR_v3.md` does not cite its predecessors at all, and every
unprefixed `§` in the README already points at v3.

Note also that `FDR_v2.md`'s title block still carries unfilled
`Author: *[fill in: full name]*` / `Student ID: *[fill in]*` placeholders. That
reads as unfinished to anyone who opens it, which is a second, independent
argument for moving it out of the top level of `docs/`.

---

## 5. Everything else

Checked and clean:

- **Broken links.** Every relative Markdown link in every tracked `.md` file was
  resolved against its own directory. **Zero broken.**
- **TODO / FIXME / XXX / HACK** in user-facing docs, code and configs: **none**.
  (The `superpowers/plans/` and `specs/` working documents contain checklist
  boxes, which is what they are for.)
- **Stale commands.** Every command in the README's entry-point table was
  checked against the module it names; all resolve. The only stale one was
  `pip install -e '.[dev]'`, fixed above.
- **`.gitattributes`** is present and sane.
- **Test suite: 752 passed**, re-run after every change in this audit. The three
  warnings are the deliberate `HeuristicPlacementAreaExtractor` RuntimeWarning,
  which is the documented behaviour of the torch-free demo path.

**[author]** Left alone by choice:

- **The `RuntimeWarning` printed by the demo.** It is loud and it is the first
  thing a reader sees. It is also correct, deliberate and well-argued, and
  suppressing it would hide a real limitation. Keep it.
- **`docs/NEXT_STEPS.md` stays where it is.** It is internal-facing and reads
  that way, but it is referenced from many working documents by name and by line
  number. The new `docs/README.md` labels it clearly instead, which solves the
  reader's problem without breaking anything.
- **`configs/` contains seven untracked experiment configs** (`segmentation_*`,
  from the working branch). They are not part of this audit's scope and were not
  staged.

---

## What was changed

| File | Change |
| --- | --- |
| `pyproject.toml` | Explicit `[tool.setuptools] packages` (fixes the install blocker); `scipy` added to `train`; new `cad` extra; new `dev` alias extra. |
| `docs/FDR.md` | Student ID removed from title block. |
| `docs/FDR_v3.md` | Student ID line removed from title block. |
| `.gitignore` | `.claude/` ignored. |
| `docs/README.md` | New — the document index. |

No code behaviour, metric definition or receipt was altered.

## What the author must decide

1. **Remove the student ID from `README.md:5`.** Still present.
2. **Whether to scrub the ID from all 168 commits before the first push.** Free
   today, expensive after publication.
3. **Which licence, and whether the university's IP regulations permit granting
   it.**
4. **Whether to move the two superseded FDR drafts to `docs/archive/`** — two
   line edits in `README.md`.
5. **Whether to add an Anker trademark disclaimer**, and whether the supervisor
   is content to be named.
