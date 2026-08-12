# Audit P — the stranger's first fifteen minutes

**Date:** 2026-08-12
**Method:** `git clone` into a scratch directory outside the repo, fresh `venv`, Python 3.14.3
on Windows 11. Every command below was run from that clone, in README order, with no
prior knowledge applied. HEAD `f1989e9`.
**Posture:** hard marker on the first fifteen minutes, generous on depth.

---

## 1. The fifteen-minute timeline, as it actually happened

| # | Step | Source | Wall clock | Outcome |
|---|---|---|---|---|
| 0 | `git clone` | — | ~10 s | **OK.** 67 MB checkout, 30 MB of it `.git`. |
| 1 | Read README opening | `README.md` §1–36 | ~2 min | **Strong.** See §5. |
| 2 | `pip install -e .` | README "Running it" 1 | 26 s | **OK.** ~100 MB of wheels. Resolves cleanly on Python 3.14 / numpy 2.5.2 / albumentations 2.0.8 / opencv-python-headless 5.0.0.93. |
| 3 | `python -m recog.synth_dataset --out recog/dataset --n 50` | README 2 | 1 s | **OK.** 50 PNGs. |
| 4 | `python main.py --config configs/demo.yaml` | README 3 | 2 s | **Runs, exits 0 — but the numbers are ~10× below what the README says they are.** See §2. |
| 5 | `pip install -e ".[dev]"` + `pytest -q` | README "Other entry points" | 38 s | **RED. 1 failed, 1162 passed, 19 skipped.** See §3. |
| 6 | `python main.py --config configs/demo_seg.yaml` | README "The same loop with the trained segmenter in it" | instant | **Hard stop — `FileNotFoundError`, no checkpoint.** Fails *well* (loud, actionable), but the reader cannot run this path at all. |

Mechanical work totals **~5 minutes**. That leaves a genuine ten minutes for reading, which
is the good news: nothing in the install path is slow, and nothing needs a GPU, Blender or
CAD. `pip install -e .` is fixed and stays fixed — CI now runs it verbatim.

The bad news is that **two of the three things a reader can independently verify disagree
with the repository's own claims about them.**

---

## 2. The documented demo does not produce the documented numbers

`README.md` §"Running it" and `configs/demo.yaml`'s own header comment both state, twice
and emphatically:

> Re-measured over 10 runs on 2026-08-12 … every perception and planning number was
> identical in all ten (37 cartridges, 77 batteries, 33 placement areas, 41 queue poses,
> 2 released reservations, 0 disagreements, 0 bad boxes)

Following the README's three commands verbatim produces, deterministically:

```
cycles 1  cartridges 3  batteries 5  placement_areas 3  queue_poses 3  placed 1
```

Three consecutive runs gave byte-identical counts. This is **not** run-to-run variance —
`recog/synth_dataset.py` is seeded (`--seed`, default `0`), so *every reader on earth gets
these same 50 frames and these same numbers*.

The run also ends at **cycle 1 of `max_cycles: 10`**. `main.py:385` sets
`stop_on_empty_queue` to `True` by default, and frame index 1 yields an empty queue, so the
loop breaks. Regenerating at `--n 200` (the count used in the README's "Other entry points"
table) changes nothing — the frames are drawn sequentially from one seeded RNG, so frame 1
is frame 1 either way, and the result is identical: `cycles 1  cart 3  batt 5  queue 3`.

**There is no route from the current tree to 37/77/33/41.** Those figures were measured
against a `recog/dataset` produced by a different code state and are now stale. There is no
`main_run.txt` receipt committed (only `main_seg_run.txt`), so nothing in the repository
would have caught the drift, and CI asserts only that the command exits 0 — never that the
counts match.

This is the most damaging single finding, for three reasons:

1. It is in the **first command the reader runs**.
2. The README stakes its central methodological claim on precisely this path — *"That path
   is torch-free by design and is what the reproducibility claim rests on."*
3. The surrounding prose is a model of care — it explains *why* the placed/pick-failed split
   is non-deterministic and *why* the perception counts are not. A reader who checks that
   careful argument against reality finds the careful part wrong. That damages the document's
   credibility far more than a plain error would, because the document taught the reader to
   check.

**Cost to fix: minutes.** Re-run the three README commands, paste the true summary, and commit
it as `docs/receipts/main_run.txt`. Then have CI diff the run summary against that receipt so
it cannot drift again. Optionally set `stop_on_empty_queue: false` in `demo.yaml` so the demo
actually exercises its declared 10 cycles — but that is a behaviour change and the honest
minimum is simply to publish the true numbers.

---

## 3. `pytest -q` is red on a clean clone

```
FAILED tests/test_calibration.py::test_seg_ablation_counts_val_instances_at_the_checkpoints_crop_size
E   ModuleNotFoundError: No module named 'torch'
1 failed, 1162 passed, 19 skipped in 33.66s
```

`tests/test_calibration.py:539` does a bare `import torch`. Its sibling at
`tests/test_seeding.py:150` does it correctly — `torch = pytest.importorskip("torch")`.
Ten test modules use `importorskip`; this is the one that does not.

`.github/workflows/ci.yml` states in a comment:

> 31 need torch (`.[train]`, a ~GB download whose useful tests want a GPU) … Each guards
> itself with an importorskip/skipif, so this run is green rather than quietly thinner.

That claim is false for exactly one test, which means **CI should be red at HEAD on all
three matrix Pythons**. The workflow is otherwise excellent and is the right instinct —
it runs README steps 1–3 verbatim precisely to catch the "broken from clean" class of
failure. It is one line away from doing its job.

**Cost to fix: one line.**

Note also that `pytest -q` is offered in the README's entry-point table with no indication
that the torch-free install is expected to leave 19 tests skipped. A reader who has just
been told the suite has 1,074 tests, sees 1,182 collected and 19 skipped, has no way to
know whether that is by design.

### Test-count claims disagree with each other and with reality

| Source | Claim |
|---|---|
| `README.md:99` | "The suite is at **1,074** tests now" |
| `README.md:214` (layout block) | "`tests/` Pytest suite (**814 tests**)" |
| Measured at HEAD | **1,182 collected** (1,162 passed + 19 skipped + 1 failed) |

The prose at line 99 is self-aware about this ("it was 814 when this paragraph was
written"), which is honest, but the layout block at line 214 is simply stale and reads as
carelessness. Three numbers in one document is worse than one number that is slightly old.
Derive it in CI or state it once.

### Other stale figures found by spot-check

* `README.md:170` — "sixteen of the **thirty-four** committed receipts". `git ls-files
  docs/receipts` returns **36**.
* `README.md:169` — "the eleven `seg_eval*.txt` receipts" — **correct**, 11. Verified.

---

## 4. What the reader can and cannot verify

### Can verify, unaided, in fifteen minutes

* The package installs from clean on a current Python with no GPU, no Blender, no CAD.
* 1,162 tests pass, and the test *names* are unusually informative — `test_no_coordinate_outside_the_envelope_reaches_the_wire`,
  `test_a_run_that_can_reach_nothing_is_a_failed_run`. Skimming `pytest --collect-only` is
  genuinely one of the best advertisements in the repository, and nothing points the reader at it.
* The end-to-end loop runs: detection → planning → packing → CRC-framed TCP to a mock
  controller, with per-cycle latencies.
* That failures are loud by design — `demo_seg.yaml` refusing to start without a checkpoint
  is the README's stated policy, honoured exactly.
* The code, the specs, the receipts and the FDR as *text*.

### Cannot verify at any price without ~8 GPU-hours and a Blender install

* Every quantitative claim in the project. All of it: the 0.8903 bay IoU, the 0.6555 → 0.8755
  crowned-lid result, the 12-of-30 headline, the oracle comparison, the boundary displacement,
  the batched-vs-looped latency table.
* `recog/checkpoints/` is gitignored — **no `.pt` exists anywhere in the tree.** Every
  `--checkpoint` command in the README is therefore unrunnable as written, which the README
  does acknowledge (§Requirements, in the `weights_only=True` note).
* `recog/dataset3d` (985 MB), `recog/dataset3d_seg` (509 MB) and the CAD test set are all
  gitignored.

So the reader's position is: **the engineering is verifiable, the science is not.** They must
take all 40-odd headline numbers on trust from receipts they cannot regenerate.

### How much does that matter?

Less than it first appears, and the repository has already done most of the work of making
it not matter. `docs/receipts/` commits 36 tool-generated artefacts; `FDR_v3.md` Appendix C
enumerates which figures lack a receipt and why; the README names its own three unreceipted
figure groups in §"Where to look" rather than glossing them. That is a *stronger* posture
than most portfolio repositories, which simply assert numbers.

The residual gap is not credibility, it is **engagement**. A reader who cannot run anything
that produces a number has nothing to be curious about after minute five. The fix is not to
make the science reproducible — that is genuinely infeasible — but to give the reader one
thing that produces a real number on their own machine.

---

## 5. The cheapest meaningful mitigation, with real numbers

Measured sizes from the working tree:

| Artefact | Size | Notes |
|---|---|---|
| `recog/checkpoints/seg/best.pt` | **43 MB** | DeepLabv3 + MobileNetV3-Large. Under GitHub's 100 MB hard per-file limit — **no LFS needed**. |
| `recog/checkpoints/best.pt` (detector) | **147 MB** | Faster R-CNN ResNet-34 FPN. **Exceeds the 100 MB limit; would require LFS. Do not commit.** |
| `recog/dataset3d_seg` | 509 MB / 502 scenes | ≈ **1.01 MB per scene** (image ~960 KB + meta ~47 KB + annotation ~3.6 KB). |
| `recog/dataset3d` | 985 MB / ~2000 scenes | Detector training corpus. |
| `recog/realtest` (already committed) | 20 MB | Precedent: the repo already ships 20 MB of binary evidence. |
| Current clone | **67 MB** | |

### Recommendation

Commit **the segmenter checkpoint plus the 15 scenes `demo_seg.yaml` actually consumes**
(`max_cycles: 15`, `stop_on_empty_queue: false`):

```
15 scenes × 1.01 MB  =  ~15 MB
recog/checkpoints/seg/best.pt =  43 MB
                        total ≈  58 MB added
```

Clone goes **67 MB → ~125 MB**. That is unremarkable for a repository that already carries
20 MB of photographs, needs no LFS, and stays far below any GitHub warning threshold.

**What 58 MB unlocks:** `python main.py --config configs/demo_seg.yaml` — the trained
segmenter running inside the real pipeline, on real path-traced renders, producing real
label maps and real placement areas. This is the project's *actual* headline architecture,
and right now it is **100 % unrunnable by any reader**. It turns the repository's central
claim from an assertion into a demonstration.

**What it does not unlock, and must be labelled as not unlocking:**

* **Any receipt.** `seg_evaluate`'s split guard checks the partition recorded in the
  checkpoint against the dataset it is pointed at; a 15-scene subsample is not the 502-scene
  corpus and the guard will correctly refuse. Ship the sample as its own directory
  (`recog/sample3d_seg/`) with a README line saying "wiring demonstration, not a metric" —
  do not let it look like a reproduction of `seg_eval.txt`.
* Training, the oracle comparison, the crowned-lid experiment, or anything in the FDR.

**One honest caveat:** this path needs torch, which the torch-free demo deliberately avoids.
A CPU-only torch install is a several-hundred-MB download and will push a cold reader past
fifteen minutes. So the sample is the *right* second impression, not the first one — which
is another reason the zero-byte fixes in §2 and §3 outrank it.

**Ranking by value per byte:** fixing the demo numbers (§2) and the red test (§3) costs
**0 MB** and buys more credibility than the 58 MB does. Do those first. The sample is the
best *next* investment.

---

## 6. Does the documentation navigate?

**`docs/README.md` is excellent and almost nobody will read it.** It does precisely the job
a stranger needs — a table of what is current, what is superseded, what is internal working
notes, plus a genuinely impressive paragraph explaining that `FDR_v3.pdf` and `FDR_v3.md`
have diverged and which one wins. It even explains the `git-filter-repo` SHA rewrite.

**The root `README.md` never links to it.** `grep -c "docs/README" README.md` → **0**. The
single best navigation aid in the repository is unreachable from the front door.

The volume itself: **61 tracked Markdown files under `docs/`** (39 specs, 6 audits, 7 plans,
4 FDR revisions, plus the rest), 760 KB of specs and 460 KB of plans. That volume reads as
*thorough* rather than unfinished — but **only because `docs/README.md` and
`docs/superpowers/specs/README.md` exist and frame it**. Without those two index pages it
would read as a dump. With them, it reads as a lab notebook. The framing is already written;
it is only mis-filed.

Two documents are genuinely mis-placed for a public artefact:

* **`docs/CV_BULLETS.md`** — "CV and LinkedIn phrasing". A hiring manager finding the
  candidate's own suggested CV lines inside the repository they are evaluating is an odd
  moment. It costs nothing to move this out of the published tree.
* **`docs/superpowers/plans/`** (460 KB, 7 files) — these open with
  `> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
  … Steps use checkbox syntax for tracking.` These are AI-agent execution plans. Whether to
  publish them is the owner's call and there is a defensible case either way, but it should
  be a *decision* rather than an oversight, because a stranger browsing `docs/superpowers/`
  will hit them and they are the least self-explanatory thing in the repository. The
  directory name `superpowers/` compounds this — it is the name of the tooling framework,
  means nothing to a reader, and sits over the project's best material.

---

## 7. How the audit trail reads to a stranger

**As the strongest thing here — and it is already framed correctly, one level too deep.**

`docs/superpowers/specs/README.md` is very good. It states the shape of every document
(*diagnosis → hypothesis with a threshold stated before the run → experiment → result,
including when the result is null*), then indexes the six core investigations in a
Question/Result table where the results are allowed to be failures: *"Its own acceptance
criterion is not met, and §4 says so before any result is quoted."* *"the claim they were
'too full' does not survive contact with ground truth."*

That table is the single most persuasive artefact in the repository. It demonstrates, in
about ninety seconds of reading, pre-registration, negative results kept at full prominence,
retraction-by-successor-document rather than silent editing, and per-document baseline
commits and test counts. Very few portfolio repositories contain anything comparable.

The six audit files (`2026-08-12-A-measurement-tools.md` … `F-execution-and-config.md`,
180 KB) read as deliberate adversarial review, and `docs/README.md`'s description of them —
*"each briefed to invalidate rather than confirm … Every claim is labelled as
measured-by-execution or established-by-reading"* — is exactly the right framing.

**The problem is purely one of placement.** The reader is pointed at this material in
`README.md` §"Where to look", at **line 168 of 335** — after 160 lines of the densest prose
in the document. Anyone skimming has stopped by then. So the differentiator reads as
internal clutter *only* to the reader who never reaches its index, and that is most readers.

**Framing needed:** none. It is already written. It needs **promotion**, not authorship.

---

## 8. First impressions, in order

**What the reader sees, in order:** title and one-sentence framing → the pipeline ASCII
diagram → the "no GPU, no camera, no controller" promise → `fig11_architecture.png` →
"The headline result, and what it is not".

**Does the opening earn the next thirty seconds? Yes, comfortably.** The one-line summary is
concrete, the ASCII diagram orients immediately, and *"runs the whole loop with no GPU, no
camera and no controller"* is exactly the right promise to make a reader who has fifteen
minutes. Both figures are valid PNGs (1550×645 and 1182×814) with genuinely excellent alt
text, and will render on GitHub.

**Is the headline stated with its limits? Yes — better than almost anything I have read.**
The section is literally titled "The headline result, and what it is not"; it states 12/30
and 25 cells, immediately gives the oracle comparison, then explains why a shipping system
at or above its own oracle *needs explaining and the explanation is not skill*. Four limits
follow under "stated here rather than buried", including the unusually honest *"Sim-to-real
transfer is unvalidated and cannot be validated in this project… This is not 'not yet
measured'; the data that would settle it will not be collected."* That is a candidate
disclosing the ceiling of their own evidence, unprompted. It is the most creditable thing in
the document.

**The failure is density, not honesty.** The README is **53 KB / 335 lines**, and the prose
is relentless: §"The headline result" is a single 400-word paragraph containing eleven
distinct numbers, three commit SHAs and four document cross-references. Every sentence has
been revised toward precision and away from readability. The corrections are inlined into
the argument — *"The 27-cell figure this README carried until 2026-08-12 was measured at
`ce1d9cd`…"*, *"The queue-pose figure was 62 until 2026-08-12"*, *"this line said 'retries,
heartbeat and E-stop' until 2026-08-12"* — so the reader is repeatedly asked to hold a
superseded value and its replacement simultaneously, in a document they have been reading
for ninety seconds. Individually each correction is admirable. Cumulatively, in the first
screen, they read as a document arguing with itself.

Meanwhile **`docs/PORTFOLIO.md` is the document this reader actually wants** and it is
mentioned at **line 327 of 335**, in the second-to-last section. It is 65 lines. It opens
*"Five things were broken. None of them was the model."*, puts the no-robot/no-photograph
limitation in its third paragraph, and tells the crowned-lid story as a narrative with the
dose-response evidence intact. It is the best-written thing in the repository and it is
buried below the trademark notice's neighbour.

---

## 9. The single highest-value change to the first impression

**Put a four-line "Start here" block immediately after the opening diagram, above the
headline section, and make the two zero-cost corrections it will expose.**

```markdown
**Start here** — 15 minutes, no GPU:
1. The story, in 5 minutes: [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md)
2. Run it, in 2: `pip install -e . && python -m recog.synth_dataset --out recog/dataset --n 50 && python main.py --config configs/demo.yaml`
3. How every number was arrived at: [`docs/superpowers/specs/README.md`](docs/superpowers/specs/README.md)
4. What is authoritative vs. history: [`docs/README.md`](docs/README.md)
```

This costs four lines and fixes the four largest navigation failures at once: `PORTFOLIO.md`
promoted from line 327 to line 20, the specs index promoted from line 168, `docs/README.md`
linked from the front door for the first time, and the runnable path stated before the
reader has to survive 400 words of oracle comparison to reach it.

It also front-loads the differentiator. A hiring manager who reads `PORTFOLIO.md` and the
specs index — 5 minutes, ~150 lines — comes away with the correct impression of this project
(measurement discipline, pre-registered nulls, mechanisms diagnosed to root cause). A hiring
manager who starts at line 24 comes away thinking the author cannot stop revising.

**Do it together with §2 and §3.** A "Start here" block that hands the reader a command
whose output contradicts the README, and a `pytest -q` that goes red, is worse than no
block at all — it directs more attention to the two places the artefact currently fails.
The three changes are minutes of work between them and they are, jointly, the difference
between a reader forgiving the density and a reader closing the tab.

---

## Summary of defects found

| # | Severity | Defect | Fix cost |
|---|---|---|---|
| P1 | **High** | `demo.yaml` run produces 3 cartridges / 5 batteries / 3 queue poses over 2 cycles; README and `demo.yaml` both claim 37 / 77 / 41 over 10, "identical in all ten runs". Deterministic, so every reader sees the contradiction. No `main_run.txt` receipt exists to catch drift. | Minutes + a CI diff |
| P2 | **High** | `pytest -q` red from a clean documented install: bare `import torch` at `tests/test_calibration.py:539` should be `pytest.importorskip`. CI comment explicitly asserts every torch test is guarded; it is not, so CI should be red at HEAD. | One line |
| P3 | **Medium** | `docs/PORTFOLIO.md` (best front door, 65 lines) at README line 327; specs index at line 168; `docs/README.md` (best navigation) never linked from root at all. | Four lines |
| P4 | **Medium** | No checkpoint or dataset anywhere → `demo_seg.yaml` and every `--checkpoint` command unrunnable; no reader-derivable number exists. | 58 MB (see §5) |
| P5 | Low | Test count stated as 1,074 (line 99) and 814 (line 214); actual 1,182. | One line |
| P6 | Low | "thirty-four committed receipts" (line 170); actual 36. | One word |
| P7 | Low | `docs/CV_BULLETS.md` and `docs/superpowers/plans/` (agent execution plans, "REQUIRED SUB-SKILL: …") are internal-facing artefacts in a public portfolio tree. Publish-or-not should be a decision, not an oversight. | Judgement call |

**Not defects, and worth saying explicitly:** `pip install -e .` works and CI now guards it;
failures are loud and actionable exactly as documented; both figures render with strong alt
text; the limitations are stated more honestly than the norm; and the specs/audit trail is a
genuine differentiator that needs promotion rather than rewriting.
