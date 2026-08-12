# The last recorded cleanups

Written 2026-08-12 against `82cff22`. Four items that earlier audits found and
recorded rather than fixed. Three are now fixed; the fourth is assessed and
deliberately left alone. Suite green at **1032** throughout.

---

## 1. The last two dangling commit SHAs

The history rewrite that redacted a student ID changed every SHA.
[`2026-08-12-sha-remap.md`](2026-08-12-sha-remap.md) remapped 141 documentation
citations and listed 17 in code it was forbidden to touch;
[`2026-08-12-followups.md`](2026-08-12-followups.md) §3b cleared five more, and
`a1f87a2` cleared `.github/workflows/ci.yml`. Two were left.

| File | Line | Was | Now | Target subject |
| --- | --- | --- | --- | --- |
| `recog/seg_training.py` | 593 | `dedf700` | `0ac6c5b` | `fix(training): always save last.pt, not only the best epoch` |
| `tests/test_synth3d.py` | 1979 | `ac54743` | `f4596e8` | `refactor(recog,synth3d): one authoritative 18650 dimension, not four copies` |

Both are comments — one line of a comment block and one word of a docstring.
The diff is two words. Both replacements were checked with
`git cat-file -e <sha>^{commit}` and, more usefully, **corroborated
semantically**: the prose at each site describes the commit its new SHA points
at ("fixed exactly this bug (best-only saving silently discarding every later
epoch)" → *always save last.pt, not only the best epoch*; "the one authoritative
18650 figure" → *one authoritative 18650 dimension*). A mapping error would show
up as a subject that does not match the sentence around it. Neither does.

### The sweep

The brief warned not to assume two was the true count, and it was right to. The
sweep extracted every `[0-9a-f]{7,40}` token from all 206 tracked text files
(697 distinct), batch-resolved them with `git cat-file --batch-check`, and
cross-referenced the non-resolving ones against the 172 old→new pairs in
`.git/filter-repo/commit-map`. Anything that fails to resolve *and* prefixes a
known pre-rewrite commit is a dangling citation; a 7-hex string matching one of
172 known commits by chance is a ~1-in-10⁶ event.

After the two fixes, **no dangling citation remains anywhere in code, config,
tests, receipts or the FDR.** The receipts are clean too — the 22
`seg_eval*.txt` citations that `sha-remap.md` §4a recorded have since cleared,
as predicted, on regeneration.

What the sweep *does* still find is 60-odd hits confined to
`docs/superpowers/specs/` and `docs/superpowers/audit/`. Nearly all are
deliberate and must not be touched:

- `2026-08-12-sha-remap.md` and `2026-08-12-followups.md` contain the old→new
  **mapping tables themselves**. Rewriting the left-hand column would destroy
  the record. `sha-remap.md` §5 already excludes itself from the resolve check
  by design.
- `2026-08-12-C-methodology.md:156` reads *"the SHA `58dd21d` named in the audit
  brief does not exist in this repository"* — a sentence whose whole point is
  that the SHA is dead. Remapping it makes it false.
- `2026-08-12-A-measurement-tools.md:331,345` quote `138105d` from an audit
  brief written before the rewrite, alongside live SHAs. Quotation, not citation.

**Two are genuine dangling provenance citations, and both are outside this
brief's file scope.** They are recorded here so the fix is mechanical:

| File | Line | Cited | Should be |
| --- | --- | --- | --- |
| `docs/superpowers/audit/2026-08-12-C-methodology.md` | 352 | `83348fa` | `2a2da37` |
| `docs/superpowers/audit/2026-08-12-E-silent-failures.md` | 55 | `d6c46ac` | `562ca75` |

Both are prose citing where work happened ("re-run in one harness at commit
`83348fa`", "fixed for the planner at `d6c46ac`"), not quoting a dead SHA on
purpose. Recommended, not done: the brief scoped this work to
`recog/seg_training.py`, `tests/test_synth3d.py`, `docs/receipts/` and
`docs/FDR_v3.md`, and these are audit records owned elsewhere.

---

## 2. `docs/receipts/frcnn_latency.txt` — header repaired, no number touched

The file's title line repeated 64 times. The damage was **exactly**
`(title + "\n" + "=") * 64` — verified byte-for-byte, not inferred:

```python
d.startswith((title + "\n" + "=") * 64)   # True
```

That is the signature of a generator that applied `* 64` to the whole
concatenation instead of to the underline alone. The intended header is
therefore `title + "\n" + "=" * 64`, and the repair is to collapse the
repetition back to one title and a 64-character rule. Two things corroborate 64
as the right width: it is the multiplier the bug itself used, and
`forbidden_bench.txt` and `forbidden_bench_seeds.txt` both carry 64-character
underlines already (underline widths in this corpus are per-tool constants, not
derived from title length).

The repair is provably lossless. Everything after the damaged prefix was copied
through untouched, and the check was run both ways:

- byte-level — `title + "\n" + "="*64 + old[len(damage):] == new` is `True`;
- number-level — the numeric tokens of the old file are `['2','320','512'] * 64`
  followed by 13 body numbers; the new file's are the same 13 body numbers after
  a single title. **`old[192:] == new[3:]` is `True`.**

4294 bytes → 325 bytes, all of the loss being duplicated title text.

The file now reads:

```
frames:  100
mean:    446.0 ms   median: 437.4 ms
p95:     484.2 ms   p99:    638.3 ms
min/max: 406.4 / 689.5 ms
```

**The measurement is the original one.** It was not re-taken: no producing tool
survives in the tree (Appendix C item 3 lists `frcnn_latency.txt` among the
sixteen receipts with no surviving generator), and re-measuring on today's
hardware would silently invalidate the heuristic comparison the file holds in
its last two lines — the ×10.4 ratio against `HeuristicDetector` is only
meaningful because both arms were timed on the same machine in the same run.
Only the header was repaired. Nothing was appended to the file to say so,
because that would make it something other than tool output; it is recorded
here and in the commit message instead.

---

## 3. `docs/receipts/pytest-cov.txt` — regenerated, at two scopes

The old receipt recorded 102 tests, an Ubuntu / Python 3.10 run, 18 modules and
a 1 142-statement total, and FDR Appendix E's O6 row rests on its 86 %. It also
carried nine hand-appended lines — an end-to-end smoke-test summary that
`pytest --cov` never emitted.

### The measurement

Both figures were measured on **a clean checkout of `82cff22`**, not in the
working tree. This was not fastidiousness: a concurrent refactor of
`recog/synth3d/bay.py` and `world.py` was in flight, and because `coverage`
re-parses source files when it reports, consecutive runs in the working tree
disagreed (65 % then 62 %, with statement totals moving 6 897 → 6 953). Pinning
the measurement to a commit makes it reproducible by anyone from a clone. The
project's generated dataset (`recog/dataset3d_seg/`, gitignored, 509 MB) was
junctioned in, because without it `tests/test_seg_dataset.py:148` skips and
`recog/seg_dataset.py` reads 13 points lower. Three independent environments
then agreed on identical numbers.

### The two figures

| Scope | Modules | Statements | Branch coverage |
| --- | ---: | ---: | ---: |
| The 18 modules the 2026-04-20 receipt listed | 18 | 1 604 | **89 %** |
| Everything `[tool.coverage.run] source` resolves to today | 49 | 6 897 | **65 %** |

Reporting one of these alone would mislead in one direction or the other. 89 %
says the code O6 was written about is better covered than the 86 % on record —
true, and it means **86 % still holds and was slightly conservative** — but it
quietly measures a 2026-04 subset of a project that has since roughly sextupled
in statement count. 65 % is the honest size of the codebase today, but read
without its scope it implies a regression, and there was none: **no module lost
coverage.** The denominator grew. What landed since April is largely
Blender-only — `render.py`, `scene.py`, `generate3d.py`, `verify3d.py`,
`_gate_orientation.py` all sit at 0 % because they cannot be imported outside
Blender, let alone unit-tested — plus CLI entry points that are exercised
end-to-end rather than by unit test.

So the receipt reports both, and names each scope by **showing the command that
produced it**. The file is now tool output plus a title and those two `$` lines;
nothing is hand-written, and nothing was appended. The interpretation above
lives here and in the FDR, which is where it belongs.

### Two figures in the brief did not reproduce

The brief anticipated 90 % and 55 %. The measured values are **89 % and 65 %**,
and both were re-derived three times.

- 89 % vs 90 % is a rounding-scale difference and immaterial; the conclusion
  ("86 % still holds and is slightly conservative") is unchanged.
- 65 % vs 55 % is a real 10-point gap, and it matters, because it is the
  difference between two ways of failing the 70 % O6 threshold. It was not
  chased to ground. One plausible cause was tested and eliminated: measuring
  without `pyproject.toml`'s `omit` list (which would pull in `recog/model.py`,
  `recog/training.py` and `synth_dataset.py`) gives 64 %, not 55 %. Whatever
  produced 55 % is not reproducible at `82cff22` under the project's own
  coverage configuration, so 65 % is what is recorded. **65 % is still below
  70 %**, so the conclusion the brief drew from 55 % — that full-scope coverage
  would flip O6 on scope alone — survives the correction intact.

### FDR changes

Two edits, both in the appendices:

- **Appendix E, O6 row** now reads Pass at 89 % *with the scope named* — "the 18
  production modules O6 was scoped to" — states the 65 % full-scope figure and
  that it is below threshold, and says explicitly that the two differ by scope
  rather than regression, with the 1 142 → 6 897 statement growth given as the
  reason.
- **Appendix C item 2** said the figure "has not been re-measured". That is now
  false, so it was rewritten to record the regeneration, both figures, both
  scopes, the removal of the nine appended lines, and the corrected suite size
  (it still said 814; it is 1032).

The 86 % figures in §1, §9.3 and §10.5 were **left as written**, and Appendix C
now says so and why. They form a self-consistent snapshot of the April run —
102 tests, 1 142 statements, a four-row module table that adds to that total —
and rewriting the figure without rewriting the snapshot around it would produce
a paragraph that no longer adds up. Appendix C is the document's designated
place for recording where a headline figure stands today, and it does.

---

## 4. `docs/receipts/git-log.txt` — assessed, deliberately not generated

**Recommendation: leave it. Not implemented, on purpose.**

The file has never contained a commit line; it holds a bracketed placeholder
saying no git history was available. FDR §12.1 leans on commit-by-commit
progression as evidence of process, and Appendix C item 1 already states
plainly that the receipt which is supposed to *be* that evidence is empty, that
this predates the history rewrite, and that the real history is truthful in a
clone even though the receipt is not.

`git log --oneline > docs/receipts/git-log.txt` would produce a real artefact of
real work. It would also be the wrong artefact, for two reasons that compound:

1. **It is not the record the claim needs.** Appendix C's claim is about a
   contemporaneous submission-time log. A log taken today is a record of the
   repository as it stands months later — and, after the redaction rewrite, of a
   history whose every SHA was regenerated. Dropping it into a slot labelled
   "`git log --oneline` at submission time" would answer a question about *then*
   with evidence from *now*, which is precisely the substitution the receipt's
   emptiness currently makes visible.
2. **The gap is already honestly disclosed, and the disclosure is worth more
   than the file.** A reader today learns that a receipt is missing and why. If
   the file is filled, that reader instead sees a plausible-looking log and has
   no way to know it was generated after the fact. Appendix C's own stated
   reason — that overwriting a placeholder "would misrepresent *when* it was
   taken" — is correct, and nothing found in this pass weakens it.

The honest fix is not a receipt. It is what Appendix C already does: point the
reader at `git log --oneline` in a clone and say the contemporaneous record does
not exist. Generating the file would trade a visible, documented gap for an
invisible, undocumented one — which is the definition of manufacturing evidence,
however truthful each individual line in it would be.

No change made to `git-log.txt` or to Appendix C, which needs none.

---

## Verification

- `git cat-file -e <sha>^{commit}` on both replacements — both resolve; both old
  SHAs confirmed dead; commit subjects match the surrounding prose.
- Full-tree SHA sweep re-run after the fixes — **0 dangling citations** outside
  the historical records described in §1.
- `frcnn_latency.txt` — body proven byte-identical and numeric tokens proven
  sequence-identical to the pre-repair file.
- `pytest-cov.txt` — both figures reproduced in three environments (working
  tree, and two clean checkouts of `82cff22`); receipt regenerated from captured
  tool output, never transcribed.
- Suite: **1032 passed** at `82cff22`. The two source edits are one word of a
  comment and one word of a docstring; `recog/seg_training.py` parses and
  `tests/test_synth3d.py::test_cell_formats_18650_matches_the_authoritative_constant`
  passes.
- No code behaviour, metric definition, dataset or model was changed. Nothing
  was retrained or re-rendered.
