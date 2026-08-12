# Portfolio verification pass — 2026-08-12

Baseline: `917e0a0`, 752 tests passing, tree clean. Scope: `README.md`,
`docs/PORTFOLIO.md`, `docs/CV_BULLETS.md` only. No code, config, metric
definition or receipt was changed; the suite is unchanged at **752
passed** (re-run at the end of this work).

An independent read of the three published documents, claim by claim,
against `docs/receipts/` and `docs/superpowers/specs/`, done adversarially
— the question asked of every sentence was "what does an interviewer say
when they open the receipt".

About **140 distinct assertions** were checked: roughly 115 quantitative
(most of them appearing in more than one document) and the fifteen
qualitative claims the documents actually rest on. **Nine defects were
found and corrected. Six items are flagged for the author** because they
are judgement calls about tone or emphasis rather than accuracy.

---

## 1. What was corrected

### 1.1 "The suite grew from 621 to 752 tests and never failed on any of them"

**The most serious finding in this pass.** True as stated and misleading
in exactly the way the section is trying not to be.

Checked commit by commit over `09326f3..917e0a0`. For **0 of the 5**
defects did a regression test exist in the suite before the fix. In every
case the tests arrived in the same commit as the repair:

| # | fix | tests at that commit | pre-existing regression test? |
|---|---|---|---|
| 1 | `27cbd97`..`9fcf136` | +4, +3 (`needs_flip`, created in the same commit) | no — **and it updated five pre-existing tests that asserted the pre-fix behaviour** |
| 2 | `5a619fc` | +2, **−1** | no — the deleted test *required the gate to fire* |
| 3 | `12134c2` | +7 (+290 lines to `test_main_integration.py`) | no — nothing pre-existing exercised the wiring |
| 4 | `d6c46ac` | +20, new file `test_packing_ceiling.py` | no — FFDH was deliberately left bit-identical, so the old packing tests *could not* go red |
| 5 | `58dd21d` | +29, new file `test_calibration.py` | no — and a deleted test asserted `"mm_per_px     : 0.625" in text` |

Two of the five are worse than neutral: the green suite was green
**because it asserted the defect.** `5a619fc`'s replacement test is
docstringed "The inverse of the test this replaces."

Separately, the window is wrong. Both of defect 1's fix commits **precede
the 621 anchor** (`git merge-base --is-ancestor 9fcf136 09326f3` → true).
The suite was at 533–570 tests for the whole of defect 1. So "621 → 752
across the work" describes defects 2–5, not five. Defects 2–5 all landed
on one afternoon, 2026-08-11 16:57–19:16.

The 621 and 752 endpoints themselves are sound: 621 first recorded at
`09326f3`, 752 at `0d7d204`, confirmed here by `pytest -q` (752 progress
characters, no F or E). Monotone non-decreasing throughout, though
individual test functions were deleted and replaced. Note that
`docs/receipts/pytest-cov.txt` is stale at 102 tests and corroborates
neither figure — the only evidence is commit and spec prose.

**Corrected in all three documents** to the claim the evidence supports,
which is also the stronger one: *no test caught any of them*, the suite
never went red because the tests did not exist yet, and twice it was green
because it pinned the bug.

### 1.2 "The shipping system places 25 across **the same 12**" (PORTFOLIO)

False. The two sets of twelve share **ten** cartridges.

* shipping (`placement-safety` §4): c7 c14 c36 c51 **c53** c56 c57 c61 c64 c70 **c80** c82
* oracle (`placement-feasibility` §2): c7 c14 **c28** c36 c51 c56 c57 c61 c64 c70 **c76** c82

The coincidence of the count is a coincidence. `c53` is in fact the
instance `placement-feasibility` §6 singles out as one the oracle refuses
and the pipeline places into anyway. Corrected to "12 — the same count,
and ten of the same cartridges".

### 1.3 The oracle is also given a planner relaxation, not only perfect perception

The oracle runs at `wall_inset 0.0`; the shipping extractor runs at 4.25 mm.
`placement-feasibility` §5 records **GT at inset 4.25 = 10 instances, 24
cells** — *below* the shipping pipeline's 12 / 25. So "perfect perception"
through the shipping planner does **worse** than what ships, because the
segmenter over-reports free floor and some of the 25 exist only on that
optimism (§6: 3 of 17 physical footprints overlapped GT non-floor at the
time of that measurement).

README disclosed "with no wall inset" but framed it as "perception removed
from the question entirely"; PORTFOLIO omitted the inset altogether. Both
corrected, and the 24 / 10 figure added to the caveat line in each so the
comparison cannot be read as a like-for-like one.

### 1.4 "This SKU places zero cells in 10 of 10 instances … and the wall inset recovers 0 all the way down to 0.0 mm"

The two halves of that sentence come from different configurations and
contradict each other.

`placement-feasibility` §3's per-SKU table: for the 10000, **GT admits ≥ 1
cell in 2 of 10**; after the extractor's 4.25 mm inset, 0 of 10. The
"recovers 0 down to 0.0 mm" figure is from §5's **predicted-mask** sweep,
not the ground-truth one. Under ground truth, relaxing the inset to zero
recovers exactly those two — `c70` (64.7 mm) and `c76` (65.1 mm) against
the 65.0 mm needed, both surviving only because `_rasterise_mask`
quantises optimistically. Both are inside the oracle's own 27.

Corrected in all three documents: the "10 of 10" is now explicitly
*through the shipping extractor*, the two recovered instances are named
with their margins, and the inset sweep is attributed to predicted masks.

This also matters for §1.3: the two marginal 10000 placements are the ones
most likely to move under the whole-cell occupancy rule the oracle was
never re-run against.

### 1.5 "Three comparisons were run with the criterion fixed before the result existed"

Only one of the three carried a numeric threshold fixed in advance.

* **cell format** — pre-registered at ≥ 0.15, verbatim, ahead of the render
  (`transfer-gap-diagnosis` "The prediction, stated before the render was
  started"). Genuinely pre-registered.
* **anchored vs wide** — named as comparison 1 of 4 in
  `2026-08-10-generalisation-design.md` §12 "before any run exists", but
  with **no effect size attached**. Only §12's regression checks (item 4)
  carry numbers, and they are floors, not this comparison's criterion.
* **`tau`** — directional. The sign was fixed in advance (a gate needs a
  negative correlation); no magnitude was.

Corrected in README and `CV_BULLETS.md` to "named before the run", with
the one genuine pre-registration identified as such. PORTFOLIO made no
pre-registration claim and needed no change.

### 1.6 "zero placeable area on 7 of 20 held-out real photographs" (README)

There are **seven** photographs in `recog/realtest/`, carrying 20 annotated
cartridges. `main.py`'s own construction-time warning is the source and
says "zero placeable area for 7 of 20 **cartridges**"; FDR §13.2.1 agrees
("7 of the 20 hand-annotated", "the same 20 cartridges in 6 images").
Corrected.

### 1.7 "Six models, all scored on the same 836 held-out CAD test crops"

The FDR's "all six" is anchored + wide + four leave-one-SKU-out controls,
written before the 18650-only and crowned-lid models existed. The glob the
same sentence cites, `docs/receipts/seg_eval_*_on_cad_test.txt`, now holds
**eight** receipts. Corrected to eight, with the models enumerated and the
FDR's figure noted as predating the last two.

### 1.8 The `tau` correlation coefficients are quoted without their n

`tau_independence_correlation.txt` closes with: "n = 8–10 crops per SKU …
No single coefficient above should be read as a precise estimate at this
sample size — the load-bearing evidence is the SIGN pattern." README
quoted 0.76 / 0.34 / 0.65 / 0.53 without either. The sign claim is
correct and is the one the argument needs; the sample size is now stated
beside it.

### 1.9 "8 → 17 cells on real frames, 3.4 ms mean" (CV_BULLETS)

Two different populations in one parenthesis. 3.4 ms mean / 4.6 ms worst is
`packing-ceiling` §3's **bench-mask** figure on 200 × 150 mm masks
explicitly "larger than any cartridge in the corpus"; the worst real
cartridge is **1.9 ms**. Corrected to name both.

---

## 2. Claims checked and found sound

Listed because a verification pass that only reports failures is not
evidence of anything.

**The crown result.** Every figure reproduces: 0.6555 / 0.8801 / 136 of
623 (flat), 0.8755 / 0.8856 / 16 of 623 (crowned), 0.9009 / 0.9013 / 2 of
623 (control); the 6.4 / 14.5 / 22.4 / 26.6 / 39.2 % dose-response and its
0.0 / 1.6 / 3.2 / 4.0 / 4.0 % collapse; `obstruction` 0.6306 → 0.6360;
in-distribution 0.7273 vs 0.7322; 11.10 mm fillet; 89 % of upward-facing
polygons below a 0.95 z-normal (PORTFOLIO's "more than 18° off vertical"
is the correct translation, `acos(0.95) = 18.2°`); 10.0× luminance ratio;
614 of 614. **The `[0, 12]` mm caveat is present and correctly scoped in
all three documents** — README's "The narrow claim only" paragraph,
PORTFOLIO's "I will not claim this shows procedural training transfers",
CV_BULLETS' adapting note. This was the single most over-claimable result
in the project and it is not over-claimed anywhere.

**The operating envelope.** Nothing says "half the SKU mix" — README says
"a third of the corpus by instance count" (10 of 30 ✓). The
`placement-feasibility` spec *does* say "half the SKU mix" at §3 and §7;
the portfolio documents correctly do not inherit it. The 13000's +1.75 mm
is right and the documents do not claim it fails.

**The five defects.** All five are genuinely defects and genuinely not the
model: glTF up-axis (render pipeline), the live `tau` gate (planner), the
unreachable segmenter (wiring), FFDH's pinned shelf origin (packer),
`mm_per_px` (calibration). Each sub-figure verified: `z ∈ [11.1, 22.2]`
against `[0, 11.1]` (FDR:2022); IoU 0.639–0.848 under the 0.85 default and
0 of 8 admitted at `mm_per_px` 0.38 (`segmenter-integration`); 93.4 %-free
grid, 48 × 112 mm rectangle, 79 of 80 shelf origins (`packing-ceiling` §1);
GSD 0.490–1.045 against 0.625, 24 of 30 under-reading by 27 % at the
median, 6 over-reading (`placement-feasibility` §1.1, §6).

**Latency and packing.** 21.2 ms batched / 88.0 ms looped at 8 crops
(`seg_eval.txt`); 2.0–2.2 ms planning (FDR:2143); 2.97 vs 4.25 ms and the
15.1 ms naive version (`placement-safety` §3.2); 81.7 mm extent bound;
14.55 vs 14.28 at 2.5 %, 2.60 → 5.53 and 0.57 → 2.85 at 10–15 %
(`forbidden_bench.txt`); 8 → 17 cells and 16-vs-17 per arm
(`packing-ceiling` §3).

**The clearance sweep.** All four quoted rows match `placement-safety`
§2.4 exactly, including the 1.5 mm row *creating* a third overlap and 3.0 mm
costing twelve cells. "Four of the five offenders at 100.0 % inside the
predicted free floor" matches §2.3 (0.999 rounds to 100.0 % there too, and
the spec says so itself).

**The nulls' arithmetic.** 0.6801 vs 0.6794 = 0.0007 on 836 crops
(receipts, directly); +0.017 of 0.224 = 7.6 %, with 0.78 correctly taken
from the *hold-out-10000* control (0.7833) because that is the figure the
pre-registration named, not the leave-one-out composite's 0.7419.

**The limitations.** Sim-to-real unvalidated and unvalidatable, no physical
robot, mock KUKA: unmissable on the first screen of README (bullets 1–2 of
four) and in line 3 of `CV_BULLETS.md`. In PORTFOLIO they sat only in the
second-to-last section — a compact statement has been added after the
opening paragraph, and the long-form section kept.

**`~19k lines of Python`** — 19,391 tracked non-test lines. **`752 tests`**
— 752, over 26 files and 9,919 lines.

---

## 3. Flagged, not fixed — the author's call

1. **PORTFOLIO's self-narration.** "I said so in the README rather than
   quietly fixing it" and "swept rather than argued about" are true and
   twice-told. One of them is conviction; two reads as a man marking his
   own homework. Consider cutting one.
2. **"Perfect perception was worth two cells" now sits one paragraph above
   the admission that a fair-inset oracle does worse than shipping.** The
   tension is real and honest, but a reader may want the two joined
   explicitly rather than left adjacent.
3. **`obstruction` parity is disclaimed in README but not in PORTFOLIO or
   CV_BULLETS.** Neither of those quotes the row, so nothing is wrong; but
   if either ever gains a per-class table, the shared-code artefact caveat
   has to travel with it.
4. **Four CV bullets carry no number** — the two "documented the limits"
   bullets, the receipts-discipline bullet, and the unreachable-model
   bullet. That looks deliberate (limitations should not be numeric
   theatre) and is left alone.
5. **Role differentiation is real, not cosmetic.** ML leads on the
   diagnosis, the nulls and the oracle; robotics on the envelope, the SKU
   finding, calibration and packing; general on contracts, defects and
   receipts. The only genuine overlap is the five-defects bullet (ML 5 /
   general 2) and the oracle (ML 6 / robotics 2) — acceptable across
   variants that are never sent together, but worth knowing.
6. **README's first screen is strong** and needs no change: what it is, the
   architecture, the one command that runs it with no hardware, then the
   headline and the four limits above the fold. An engineer knows within
   fifteen seconds whether to keep reading. The `[0, 12]` crown paragraph
   and the "narrow claim only" heading are the best writing in the repo.

---

## 4. The weakest claim that survived

**"The system is within two cells of what perfect perception achieves."**

Every correction above hardens it and none of them removes the underlying
problem: the two numbers were produced by different code at different
commits with different planner settings, on 30 instances, and the
difference between them is 2. The oracle was measured at `9bfc25f` with
centre-pixel occupancy; the shipping figure is post-`0d7d204` whole-cell
occupancy plus the bad-box extent guard. The evidence that this is safe —
"on predicted masks that change moved no cell counts" — is real but is
evidence about the *other* arm. Two of the oracle's 27 cells (`c70` at
64.7 mm and `c76` at 65.1 mm against 65.0 needed) exist only through the
quantisation optimism that the whole-cell rule removes, so a re-run could
plausibly move the oracle to 25 and the gap to zero — which would make the
"we stopped because only two cells were left" story *stronger*, not
weaker, but nobody has run it.

The fix is one scratch re-run of the oracle at HEAD. Until then the
honest form is the one now in the README: a ceiling measured at a named
earlier commit, with the direction of the residual uncertainty stated.
An interviewer who reads `placement-feasibility` §5 will ask this
question, and it is the only place in the three documents where the
answer is "not measured" rather than a number.

> **Resolved, 2026-08-12** (added after this pass; the analysis above is
> unaltered). The re-run was done at `83348fa` and is recorded as the
> addendum to `2026-08-11-placement-feasibility.md`. The prediction in
> this section is confirmed: the oracle moves to **25 cells in 11
> cartridges** at inset 0.0 and the gap to shipping is **zero cells**.
> `c70` (64.7 mm) is one of the two lost, as predicted; `c76` (65.1 mm)
> survives, and the second lost cell is one of `c36`'s. At inset 4.25 the
> oracle is unchanged at 24 / 10, so §1.3's finding stands. The three
> documents now quote the same-code-state figures.

---

## 5. Student ID removed from `README.md`

Handed over by the concurrent public-release audit, which had already
removed the same number from `docs/FDR.md` and `docs/FDR_v3.md` at
`5ad9c85`. `README.md:5` was the last tracked file carrying it. The
parenthesised number is gone; the name, institution and supervisor
attribution are kept, because the author chose those and they are
appropriate for a public repository.

**Not fixed, and it cannot be fixed from the working copy:** the number is
present in the tree of all 168 commits, so `git log -S REDACTED` and any
`git show <old-commit>:README.md` still return it. Removing it from the
published history needs a rewrite (`git filter-repo` or equivalent) before
the first push to a public remote. That is the author's decision and was
not taken here.

---

## 6. Verification

* `pytest -q` — **752 passed**, unchanged. No code, config, metric
  definition or receipt was touched; the diff is three Markdown files.
* Every figure quoted in §1 and §2 above was read out of the named receipt
  or spec in this pass, not carried over from the documents under review.
* `docs/receipts/`, `docs/README.md`, `docs/archive/`, `.gitignore` and
  `LICENSE` were not touched — another agent owns them.
