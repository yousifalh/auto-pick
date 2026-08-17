# What is in `docs/`, and which document is current

Start with the repository [`README.md`](../README.md). This page exists so that
anyone browsing `docs/` directly can tell at a glance which document is
authoritative and which is history.

## The report

| Document | Status |
| --- | --- |
| [`FDR_v3.md`](FDR_v3.md) | **Current, and the only corrected text.** The Final Design Report — requirements, literature review, detailed design, test strategy, risk assessment, AHEP-4 mapping. Every unprefixed `§` reference in the repository README points here. |

**`FDR_v3.pdf` was removed from the published tree on 2026-08-17, and the reason is not editorial.**
Page 2 of it carries the author's student ID in the title block. This repository's history was
rewritten with `git-filter-repo` before first publication *specifically* to strip that ID from every
commit tree — publishing the PDF would have put it straight back, in a tracked file, and defeated
the rewrite entirely. It is untracked and gitignored rather than deleted, and it remains in the git
history like everything else removed here.

Two corrections go with it, because the row that described it was wrong in both halves. The PDF was
dated "exported at the baseline commit (`69fad79`, 2026-08-05)"; its own title block reads
**"Submission date: 5 May 2026"**, and 2026-08-05 is a repository baseline, not a submission. And
the report it froze describes a **materially different project** from the one `FDR_v3.md` now
describes: 100 procedurally-drawn OpenCV images against today's Blender/Cycles renders of four
measured CAD assemblies, no bay segmenter at all, 102 tests against ~1,400, and O2 recorded as a
**Pass** where it is now measured as a **Fail**. Treating the two as one document with a stale
figure or two understated the gap by a wide margin.

`FDR.md` and `FDR_v2.md`, the two superseded revisions, and `FDR_v2.pdf`, were **removed from the
published tree on 2026-08-16** along with `NEXT_STEPS.md` and `superpowers/plans/`. Nothing in them
was load-bearing: the superseded revisions were kept only for history and every figure they carried
that still stands is stated with its caveat in `FDR_v3.md`; `NEXT_STEPS.md` was an internal working
note describing where the project stood on 2026-08-09; and the plans were addressed to the coding
agent that executed them rather than to a reader. Documents written before that date still cite all
five — this paragraph is the answer to those citations, and they are left standing as the record of
what was done. They remain in the git history.

**The PDF and the Markdown are no longer the same document, and the Markdown
is the one to read.** Until 2026-08-12 the `FDR_v3.md` row above ended
"`FDR_v3.pdf` is the same document as submitted", and that sentence was
doing quiet work: the PDF *is* the
artefact that was submitted, and precisely because it is frozen it carries
none of the corrections `FDR_v3.md` has accumulated since — the Δcells safety
figure (2 of 126 → 5 of 126 in the damage direction), the withdrawal of the
§7.5 heartbeat, the three E-stop bypass routes, the five deleted
`execution.yaml` motion keys, the scope bounds on §13.1.1's generalisation
claim, and every commit-SHA citation added after the baseline. It is retained
deliberately, as the record of what was assessed. **Where the two disagree,
`FDR_v3.md` is correct and the PDF is not**; do not quote a figure from the
PDF without checking it against the Markdown first.

If a figure in `FDR.md` or `FDR_v2.md` disagrees with one in `FDR_v3.md`, the
v3 figure is the one that was regenerated against a receipt. Do not quote the
superseded revisions.

## Written for a reader

| Document | What it is |
| --- | --- |
| [`MODEL_CARD.md`](MODEL_CARD.md) | **The models, in one page.** Architecture, training data, the consolidated held-out comparison of all nine segmenter checkpoints, the detector's scope, ten known failure modes, intended use and explicit non-use. Every table is generated from the receipts and configs by `scripts/model_card_tables.py`; `--check` fails if any figure has drifted. Start here if you came to read about the machine learning. |
| [`datasets/`](datasets/README.md) | **Data card.** The eleven generated datasets — per-class instance counts, disjointness, and a committed byte-for-byte copy of each dataset's `manifest.json` (its full generator config and seed), since the datasets themselves are gitignored. |
| [`PORTFOLIO.md`](PORTFOLIO.md) | Narrative account of the measurement work, for a general engineering reader. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | **Every figure this repository has published and then withdrawn, in one place** — what it read, what it reads now, when it changed and why. The repository `README.md` used to carry this inline; the errata are consolidated here without loss and the README keeps a short clause and a link where the history is load-bearing. |

`CV_BULLETS.md` — CV and LinkedIn phrasing, plus notes on how to present each
result in an interview — **was withdrawn from the public tree on 2026-08-14**
and kept privately by the author. It was addressed to the author rather than to
a reader, and nothing in it was load-bearing: every figure it quoted is stated
with its caveat in `FDR_v3.md`, `MODEL_CARD.md`, `PORTFOLIO.md` or the
repository `README.md`. Documents written before that date still cite it — the
receipt-coverage notes in `FDR_v3.md` and `NEXT_STEPS.md`, and several files
under `superpowers/`. Those citations are left standing as the record of what
was done, and this paragraph is the answer to them.

## Working documents

| Document | What it is |
| --- | --- |
| [`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md) | Operational procedure for producing repeatable real-photo polygon ground truth. Written for whoever holds the camera and mouse. |
| [`superpowers/specs/`](superpowers/specs/README.md) | The working record: one document per investigation, written at the time and kept as evidence. This is what the report's numbers were derived from. |
| [`superpowers/audit/`](superpowers/audit/) | **Nineteen adversarial reviews**, each briefed to invalidate rather than confirm. Sixteen were run on 2026-08-12 — A measurement tools, B security, C methodology, D reproducibility, E silent failures, F execution and configuration, G detector, H digital twin, I data pipeline, J claim verification, K complexity, L reachability, M real-photo unlock, N objective closure, O ML maturity, P stranger experience — and three on 2026-08-14: T KUKA conformance, U robot-interface survey, V execution seam. Every claim in them is labelled as measured-by-execution or established-by-reading. The `specs/2026-08-12-fix-*.md` and `specs/2026-08-14-*.md` documents record what was done about them, and [`superpowers/specs/README.md`](superpowers/specs/README.md) indexes both. |
| [`superpowers/blender-dataset-known-issues.md`](superpowers/blender-dataset-known-issues.md) | Measured known issues in the Blender dataset generator. |

## Generated evidence

| Directory | What it is |
| --- | --- |
| `receipts/` | Committed tool output backing the figures quoted in the README and the FDR. Regenerated by the commands recorded alongside each claim, not edited by hand. |
| `figures/` | Figures. `fig1`–`fig9` belong to `FDR_v3.md`; of those, `fig1_architecture` and `fig4_latency` are superseded and must not be reused (see `superpowers/specs/2026-08-12-figures-audit.md`). `fig10_segmenter` and `fig11_architecture` are current and are the two the README shows. |

## A note on the commit SHAs cited throughout

Before this repository was first published, its history was rewritten with
`git-filter-repo` to remove a student ID that was present in the tree of every
commit. That changed the SHA of every commit, so any short SHA taken from an
older clone, an archived copy or a quotation of these documents predating the
rewrite will not resolve here. Every commit citation in the documentation has
been remapped to its post-rewrite equivalent using the rewrite's own commit map
(see [`superpowers/specs/2026-08-12-sha-remap.md`](superpowers/specs/2026-08-12-sha-remap.md)),
and each one has been checked to resolve against the current history.

---

Anker and PowerCore are trademarks of their respective owner; this project is
unaffiliated with and not endorsed by them. The power banks named throughout
these documents are retail units used as measurement subjects for academic
research. Full notice: [`README.md`](../README.md#trademarks).
