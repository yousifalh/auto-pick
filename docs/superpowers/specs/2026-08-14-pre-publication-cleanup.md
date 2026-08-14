# Pre-publication cleanup — 2026-08-14

**Base HEAD:** `fc032c2` · **Scope:** documentation, navigation and hygiene only.
No code, config value, metric definition, dataset, checkpoint or receipt was
changed. One CI *comment* was corrected; no YAML key or value was touched.
**Suite before and after: 1 276 passing, 1 skipped.**

The brief was to make the repository read as a finished, navigable artefact
without damaging the audit trail — which a prior review judged to be its
strongest differentiator. The audit trail was therefore treated as read-only:
nothing under `../audit/` or in this directory was deleted, and no dated record
was re-worded. Where a figure turned out to be wrong, it is **reported in §5
rather than corrected**, because correcting a result means choosing which
measurement is authoritative, and that is the author's call, not a cleanup's.

---

## 1. The two internal-facing documents, decided

Audit P finding **P7** flagged `docs/CV_BULLETS.md` and `docs/superpowers/plans/`
as internal-facing artefacts in a public tree, and said the disposition should be
a decision rather than an oversight. Both are now decided.

### `docs/CV_BULLETS.md` — **removed from the public tree**

Copied to `D:\dev\auto-pick-private\CV_BULLETS.md` and `git rm`-ed. It remains in
git history; this was a publication decision, not a redaction.

The reason is not that it is self-promotional. It is that roughly half the
document is *coaching*, addressed to the author and written in the second person
about a reader who has not yet arrived: *"Have the explanation ready before you
are asked"*, *"Present it against the oracle's 11 / 25, or it reads as a
failure"*, *"Volunteer this — an interviewer who works it out first will read the
original phrasing as spin"*. Those notes are honest, and several are the sharpest
writing in the repository. But published alongside the results they are about,
they change how the results read: a repository that states its own limits looks
like discipline, and the same repository shipping the stage directions for
stating them looks like technique. The document undercuts the thing it exists to
present.

Nothing in it is load-bearing. Every figure it quotes is stated with its caveat
in `FDR_v3.md`, `MODEL_CARD.md`, `PORTFOLIO.md` or the repository `README.md`,
and every prohibition it carries (the withdrawn precision claim, the un-met 2 px
bound, "named before the run" rather than "pre-registered", the not-receipt-backed
decomposition figures) is already recorded in the FDR and in the fix specs.

Twelve documents still cite it by path — this directory, `../audit/`, `FDR_v3.md`
and `NEXT_STEPS.md`. Those citations were **left standing**, per this project's
convention that a correction is recorded in a successor rather than by editing
the original. `docs/README.md` now carries the paragraph that answers them.

### `docs/superpowers/plans/` — **kept, with framing**

The seven plans stay. They are the clearest evidence in the repository that the
work was specified before it was executed — each names its spec, its
architecture and its constraints before its first task — and removing them would
leave the specs referring to an execution record that is not there.

What was odd was not their presence but their first line. Every one of them
opened with

> `> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development …`

with nothing above it to say what the document was. A stranger's first
encounter with `docs/superpowers/` is therefore an unexplained instruction to a
tool they have never heard of. Each plan now carries one line above that note
saying what the document is, that the note below is tooling direction for the
agent that executed it, and where to go for context (`../specs/README.md`). The
note itself is unchanged — it is part of the record of how the work was done.

`docs/README.md`'s row for the directory now says the same thing, and links it.

## 2. Navigation

`docs/README.md` and `specs/README.md` are the two index pages the stranger-
experience audit found were doing all the framing work while buried. Both were
also **stale**, in a way that made the newest and best material unreachable.

| Fixed | Was | Now |
|---|---|---|
| `docs/README.md` audit row | "Six adversarial reviews run on 2026-08-12", naming A–F only | Nineteen, all named: sixteen on 2026-08-12 (A–P) and three on 2026-08-14 (T, U, V) |
| `specs/README.md` audit paragraph | Same "six" | Sixteen, all named |
| `specs/README.md` fix table | Ten of the twenty-four 2026-08-12 specs indexed | Twelve rows added — audits G, H, I, J, L, M, N, O, K and the test-harness / follow-up passes |
| `specs/README.md` | **No mention of the 2026-08-14 work at all** | New section covering audits T/U/V, the KUKA conformance fixes, the driver abstraction, and this document |
| `specs/README.md` | "Two supporting records from the same week", then lists three | "Three" |
| `docs/README.md` plans row | Unlinked prose | Linked, and says what the plans are |

The root `README.md`'s **"Start here"** block reached `PORTFOLIO.md`,
`MODEL_CARD.md`, `datasets/`, `main_run.txt`, `specs/README.md` and
`docs/README.md` — but **not the FDR**. The project's principal deliverable was
mentioned twice in the README as backticked text, which GitHub does not
linkify, so `docs/FDR_v3.md` was not clickable from the front page at all. It is
now item 5 of "Start here", with the §13.2.2 pointer attached, and the two prose
mentions are links. `docs/superpowers/audit/` was added to item 4 alongside the
specs index, so the audit trail is one click from the front door rather than two.

Everything a reader needs is now reachable within two clicks. Verified by
following every link in `README.md` and `docs/README.md` mechanically.

## 3. Counts re-measured, not assumed

The suite figures were quoted from a stale base in four places. Every number
below was **measured at `fc032c2`**, the torch-free half on a freshly created
virtualenv with `pip install -e ".[dev]"` and nothing else — the README's own
documented install.

| | measured 2026-08-14 |
|---|---|
| with torch | **1 277 collected · 1 276 passed · 1 skipped** (the skip needs `bpy`) |
| torch-free (`.[dev]`) | **1 248 collected · 1 228 passed · 20 skipped** |
| of the 20 skips | 18 are torch — 17 individual, plus `tests/test_bay_segmenter.py`'s module-level `importorskip`, which takes its 30 tests out of collection entirely. The other two need `bpy` and `.[cad]`'s `cascadio`. |

Corrected in `README.md` (the "five more times" paragraph and the repository
tree), `docs/PORTFOLIO.md`, and the explanatory comment in
`.github/workflows/ci.yml`. They read 1,211 / 1,181 / `1162 passed, 20 skipped`
before, which was the 2026-08-12 state.

**Two corrections to the record, in the other direction:**

* **The "1 236 torch-free" figure in `2026-08-14-robot-driver-abstraction.md` §0
  is not reproducible, and appears to be arithmetic rather than a measurement**
  — 1 181 (the then-current torch-free *collected* count) + 55. The measured
  value is **1 228 passing**. That spec is a dated record and was left unedited;
  this paragraph supersedes the number.
* The "1 needs a Blender-generated COCO dataset" skip named in the CI comment
  **no longer exists**. There are two non-torch skips, not three.

Counts verified against the filesystem and found **correct**: eleven
`seg_eval*.txt` receipts; eleven generated datasets and eleven committed
manifests; nine segmenter checkpoints and nine table rows; eleven figures
(`fig1`–`fig11`); seven plans; seven lighting rigs; four SKUs; 188 remapped
commit citations; the `621 → 752` and `533–570` suite-growth anchors.

## 4. Hygiene

* **Broken internal links: none.** Every relative markdown link in every tracked
  `.md` resolves to a tracked file. (Three apparent hits in
  `../audit/2026-08-12-P-stranger-experience.md` are inside a fenced
  ` ```markdown ` block — a *proposed* snippet, not live links.)
* **`docs/MODEL_CARD.md` cited four spec paths as `specs/…`**, which resolves to
  the non-existent `docs/specs/` — while §5 of the same file used the full
  `docs/superpowers/specs/…`. Normalised to the full path. Path strings only; no
  claim changed.
* **The README's own commands were run, not assumed.** In the torch-free venv:
  `pip install -e .`, `python -m recog.synth_dataset --out recog/dataset --n 50`
  and `python main.py --config configs/demo.yaml --receipt …` all succeed, and
  the regenerated receipt is **byte-identical to `docs/receipts/main_run.txt`**
  apart from the `generated:` timestamp and the `--receipt` path. Every count
  the README promises for that run — 1 cycle, 3 cartridges, 5 loose cells, 3
  placement areas, 3 queued poses, 2 released reservations, 0 disagreements, 0
  bad boxes, and 4 / 2 / 1 declined as unreachable — is what it produces.
  `python scripts/model_card_tables.py --check` reports no drift.
* **TODO / FIXME markers: none**, in any tracked `.py`, `.yaml` or user-facing
  `.md`. The `*[fill in]*` placeholders in `FDR_v2.md`'s title block are
  deliberate history and are already labelled as such in `docs/README.md`.
* **`.gitignore`: nothing to add.** No cache, coverage, checkpoint, dataset or
  agent-scratch artefact is tracked. `.coverage`, `__pycache__/`,
  `.pytest_cache/`, `.ruff_cache/`, `pytest-cache-files-*/`, `.superpowers/` and
  `.claude/` are all covered and all clean.
* **Superseded drafts.** Nothing anywhere cites `docs/FDR.md` or `docs/FDR_v2.md`
  as authoritative. Every reference to either calls it superseded, quotes it as
  a *former* wording, or is a dated audit reading it deliberately.
* **Orphans: none worth removing.** The only `docs/` files nothing references by
  name are the four `seg_eval_cad_control_*_on_cad_test.txt` receipts (cited by
  glob in `README.md` and `MODEL_CARD.md`), audit M, and four of the plans — all
  now reachable through the index rows repaired in §2.

**One name, corrected everywhere.** `README.md` credited *"Dr Svetan Ratchev"*.
He is **Professor Svetan Ratchev FREng**, Cripps Professor of Production
Engineering, elected to the Royal Academy of Engineering in 2025 — established
with sources in `../audit/2026-08-14-W-omnifactory-context.md`. Corrected in
`README.md` (title block and Authors) and, as a marked editorial correction, in
the superseded `docs/FDR.md` (title block and §12.2), where the same error stood.
`FDR_v2.md` carries no supervisor name; `FDR_v3.md` and `PORTFOLIO.md` never
credited one. Those are all the occurrences in the repository.

## 5. Found and **not** fixed — for the author

These are result figures and claims, not counts. Each is reported with the
evidence rather than corrected, because correcting one means deciding which
measurement is authoritative.

| # | Where | What | Evidence |
|---|---|---|---|
| 1 | `README.md`:36, `PORTFOLIO.md`:55 | Placeable-area error given as **51.5 mm²/crop**. The receipt says **79.2 mm²** over 126 val crops. 51.5 is the pre-2026-08-11 value converted at the nominal 0.625 mm/px. | `docs/receipts/seg_eval.txt`; the correction is already recorded in `2026-08-11-scale-figures.md`:109 ("51.5 → 79.2, 1.54×") and was flagged again in `2026-08-12-model-card.md`:148 |
| 2 | `README.md`:36 | Boundary displacement given as **0.949 mm**. Everywhere else — including `README.md`:83 thirty lines later — it is **1.226 mm**. Same sentence, same nominal-scale defect as #1. | `docs/receipts/seg_eval.txt`:37; `FDR_v3.md`:3331 explicitly retires 0.949 |
| 3 | `FDR_v3.md`:487–488 | The oracle is stated as reaching **"at most 12 of 30"**. The corrected, same-code-state figure is **11 of 30 / 25 cells**, and it is 11 in `README.md`, `PORTFOLIO.md` and the source spec. This is the headline comparison. | `2026-08-11-placement-feasibility.md`:475–479; the 12/27 row is the superseded `ce1d9cd` centre-pixel measurement |
| 4 | `README.md`:180, `MODEL_CARD.md`:438, `FDR_v3.md`:151, 4547, 4797 | Receipt count given as **thirty-seven**. There are **39** tracked files in `docs/receipts/`. `real_photo_eval.txt` and `real_photo_eval_include_empty.txt` landed at `9b38de9`, after the "34 + 3" arithmetic was written. Both have a committed generator, so the *"sixteen without a generator"* numerator is unaffected — only the denominator. | `git ls-files docs/receipts` |
| 5 | `README.md`:67 | *"the **ten** `segmentation*.yaml` configs are byte-identical apart from dataset and checkpoint paths"*. The glob matches **11** files; only **9** are training configs; and the byte-identical family is **8**. `configs/segmentation_seedcheck.yaml` says so about itself in its own header, and `segmentation_cad_test.yaml` differs at `train_val_split: 0.0`. `FDR_v3.md`:4642 says eight. | The claim is load-bearing — it is what makes the cross-model comparisons robust to the unseeded era — so it should be right |
| 6 | `MODEL_CARD.md`:306–308 | *"No committed receipt scores that model … the detector that ships has no published held-out mAP"* — contradicted by a committed receipt with a committed generator: shipped `best.pt`, mAP@0.50 **0.9053**. | `docs/receipts/detector_bench.txt`:69–81, `scripts/detector_bench.py`; also quoted in `FDR_v3.md`:2222, 4704 |
| 7 | `MODEL_CARD.md`:64–66 | *"The one measurement in this repository taken on photographs …"* — there are now two. `real_photo_eval.txt` (2026-08-13) is a second, receipted photograph measurement. | `docs/receipts/real_photo_eval.txt`:45–51 |
| 8 | `FDR_v3.md`:3319 | `dataset3d_seg` given as **841** crops; the manifest and both cards say **840** (and 0.15 × 840 = the 126 val crops the receipt reports). | `docs/datasets/dataset3d_seg.manifest.json` |
| 9 | `FDR_v3.md`:3843 | *"1.0 and 2.0 mm were measured too and move nothing."* The source table says 1.0 mm costs one instance and two cells (11 / 23). The FDR's own table omits that row; `README.md`:156 has it right. | `2026-08-11-placement-safety.md`:153 |
| 10 | `README.md`:236 | *"against 0.9 ms for FFDH alone"* reproduces no statistic in the receipt (level mean ≈ 0.66 ms, max 1.05 ms). It survives from a passage struck at `FDR_v3.md`:910. The corrected aware-arm figures, 0.32 / 1.05 ms, are at `FDR_v3.md`:1195. | `docs/receipts/forbidden_bench.txt` |
| 11 | `README.md`:91, 280 vs `MODEL_CARD.md`:43 | Batched-vs-looped inference at 8 crops: **16.6 / 57.1 ms** against **17.0 / 52.7 ms**. Neither is wrong — each matches its own cited receipt, from a different run — but two reader-facing documents give two numbers for one architectural claim, with no note saying why. | `seg_eval.txt`:66,68 vs `seg_eval_anchored_on_cad_test.txt` |
| 12 | `README.md`:83 | Attributes the 1.226 mm boundary figure to *"all 36 bay-carrying validation crops"*; the receipt's boundary row is over **35**. 36 is the pooled-IoU instance count. Minor scope slip. | `seg_eval.txt`:20 vs :37 |

**Two things checked and left alone, deliberately.** The *"4 536 renders"* leak-hunt
figure is internally consistent (36 = C(9,2) pairings) but the committed manifests
sum to 4 516 for any nine datasets; the count was taken on the gitignored image
directories, so neither a cloner nor this pass can settle it. And the
`inference_min_size` ablation in `MODEL_CARD.md`:66, 356 is sourced to an undated
comment in `configs/recognition.yaml` that names no checkpoint, so it cannot be
reconciled against `real_photo_eval.txt`'s 0.8044 at the same `min_size: 500`.

## 6. One thing left for the next pass

`../audit/2026-08-14-W-omnifactory-context.md` was being written concurrently by
another agent and is not part of this commit. It is a research brief on the
supervisor's lab, not an adversarial review of this repository, so the "nineteen
adversarial reviews" count in `docs/README.md` stays correct when it lands — but
it is not yet indexed anywhere, and it should be.
