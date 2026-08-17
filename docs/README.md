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

**`superpowers/` was untracked on 2026-08-17 and is not in this repository.** Seventy files: the
working record, one spec per investigation written at the time and naming its own baseline commit
and test count, plus nineteen adversarial reviews each briefed to invalidate rather than confirm,
plus the Blender generator's known-issues note.

It is cited about **144 times from 50 files** that remain — this README, the repository README,
`FDR_v3.md`, and a number of source comments that explain why a particular line is the way it is.
Those citations are left standing rather than stripped, for the same reason the `CV_BULLETS.md` ones
were: each records what was investigated and when, and rewriting them would erase that. This
paragraph is the answer to all of them, and the files remain in the git history.

The consequence worth stating plainly: **claims of the form "measured in spec X" can no longer be
followed to spec X here.** What can still be checked is `receipts/`, which holds the tool-generated
output every published figure comes from and is regenerated by the command recorded beside the
claim, and [`CORRECTIONS.md`](CORRECTIONS.md), which records every figure that moved and why. Where
a sentence cites a spec you cannot open, the receipt named alongside it is the thing to check.

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
