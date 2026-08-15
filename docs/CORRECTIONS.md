# Corrections

Every figure this repository has published and then withdrawn, in one place.

The repository `README.md` used to carry its own errata inline: a score of
parenthetical and bolded passages, most opening `Corrected <date>:`, sitting
beside the numbers they retracted. That was the right instinct and the wrong
place — a reader arriving at a figure had to parse a paragraph of history before
reaching the current value, the corrections were scattered across sections that
had nothing else in common, and three of them were nested inside a single
sentence. They are consolidated here without loss: for each one, what the figure
**read**, what it **reads now**, **when** it changed, and **why**. The provenance
is the point. A project whose central claim is measurement discipline does not get
to quietly restate a number.

Where a correction is load-bearing for understanding the figure beside it, the
README keeps one short clause and links here. Where the surrounding sentence
reads correctly on its own, the passage was removed outright and lives only in
this file.

**Ordered newest first.** Dates are the date the correction was made, not the
date the superseded figure was first published. A few entries are anchored to a
commit rather than to a `Corrected:` datestamp, and say so. Several figures have
been corrected more than once; those are recorded as a chain rather than as a
single before-and-after, because the direction of travel is itself evidence.

---

## 2026-08-15 — the oracle's cells, cartridges and denominator

*README, "The headline result, and what it is not".*

**Read:** **11 of 30 and 25 cells** at the zero wall inset, and **10 of 30 and 24
cells** at the 4.25 mm inset.

**Reads now:** **9 of 29 open cartridges and 23 cells**, and that same 9 / 23 at
*every* wall inset from the production 4.25 mm down to 0.00 mm.

**Why**, verbatim:

> (**Corrected 2026-08-15: this read 11 of 30 and 25 cells at the zero inset, and
> 10 of 30 and 24 cells at the 4.25 mm inset.** Two separate things moved it and they
> should not be run together. *A real geometry defect was fixed:*
> `Planner._pack_cartridge` sized its packing strip from `pr.width × scale` while
> checking placements against the occupancy grid rasterised from the same region, so
> the strip stood up to one 1.5 mm cell wider per axis than the measured placeable
> floor and a marginal fit could be seated up to 1.5 mm past that floor. The strip is
> now sized from the grid itself — `occ.cols × res_mm` by `occ.rows × res_mm` — and
> the fits that lived on the unmeasured fringe are gone: one cartridge and one cell at
> the 4.25 mm inset, two cartridges and two cells at 0.00 mm. *And the denominator
> changed for an unrelated reason:* 29 rather than 30 because the figure now comes from
> a receipt that enumerates ground-truth open cartridges directly instead of from a
> spec table that paired 30 detector-matched instances. The two censuses differ by
> three units and all three place zero cells either way — `placement_feasibility.txt`
> §5 reconciles them line by line.
> `docs/superpowers/specs/2026-08-11-placement-feasibility.md` §3, §5 and its
> 2026-08-12 addendum are superseded on these totals.)

This is the correction that also made the figure regenerable for the first time:
`python scripts/placement_feasibility.py` writes
[`receipts/placement_feasibility.txt`](receipts/placement_feasibility.txt), and
`--check` fails on drift. Every earlier oracle figure in this file was
spec-table-derived and could not be re-run.

**And the shipping half was withdrawn, not restated.** The same edit withdrew the
shipping pipeline's *12 of 30 cartridges and 25 cells* pending re-measurement,
because it never had a regenerable receipt and every packing figure adjacent to it
moved when the strip was fixed. The sentence this README carried for three days —
*perfect perception is worth zero net cells on this corpus* — is a statement about
the difference between two numbers, only one of which has been re-run at HEAD, and
is not quoted at the new figures. See "Still standing" at the foot of this file.

The withdrawal reached one more place: the third of the README's four stated limits
led with **"12 of 30 is a ceiling, not a shortfall"** until 2026-08-15, and now leads
with "whatever the shipping pipeline scores on this corpus". The limit itself never
depended on the figure — one SKU cannot accept a cell at any perception accuracy, and
that is a property of the CAD and the nominal footprint, so it does not move when a
packing figure does.

---

## 2026-08-15 — the oracle figures became receipt-backed, and the retraction ran the other way

*README, "Where to look".*

**Read:** "both oracle figures (25 cells / 11 cartridges, and the 24 / 10 at the
shipping wall inset) trace to a results table in a spec, not to a receipt".

**Reads now:** the oracle is regenerated end to end from the committed CAD, config
and ground-truth annotations by `scripts/placement_feasibility.py`, guarded by
`tests/test_placement_feasibility_receipt.py`, with a `--check` mode that fails on
drift. It is the **shipping** side — 25 cells across 12 cartridges — that traces to
a spec table and to no receipt.

**Why:** the statement stopped being true at `b69bcb3`, which committed the receipt
and its generator. This one is worth flagging because it runs opposite to every
other entry in this file: the README was *under*-claiming its own rigour, which is
the error it is least entitled to make.

---

## 2026-08-15 — `AnkerPowerCore10000`, and what the zero wall inset recovers

*README, "A cartridge that cannot be certified".*

**The chain, in order:** the *measured* count went **2 → 1 → 0** in four days, on
two unrelated correctness fixes. The *published wording* moved three times, because
one of those moves was a prose correction rather than a measurement.

* **two → one**, at `b93bbd3` (2026-08-11), which stopped calling a half-wall grid
  cell free. The instance it removed measured **64.7 mm** of free strip against the
  65.0 mm the cell needs, and existed only because the occupancy grid quantised in
  the optimistic direction.
* **the wording, not the measurement**, at `cc3fce1` (2026-08-14). The README had
  published "recovers 0" while the receipt said 1. Nothing was re-measured; the
  sentence was corrected *upward* to match the evidence already on disk.
* **one → none**, on 2026-08-15. The surviving instance was `scene_00049/item5`, a
  bay measuring **65.10 mm** turned 0.297°. It was the last fit anywhere in this SKU
  living on the fringe the packer was not measuring — the same
  `Planner._pack_cartridge` strip-sizing defect as the oracle entry above. Sizing the
  strip from the grid takes it away, and `placement_feasibility.txt` §2's per-instance
  table now reads 0 cells for that unit at all three insets.

**Why it is recorded as a chain:** three corrections in four days, running in both
directions, on a figure that is either 0, 1 or 2. The safe reading is not that the
count is settled but that a zero-margin fit is decided by rasteriser rounding — which
is exactly the section's own thesis, arrived at the hard way.

Corpus-wide, the SKU used to place 6 cells across 4 of its 47 open instances and now
places 0 across 0. The rotation band the README quoted alongside it — **"no instance
turned more than 0.28° takes a cell"** — is retired: it no longer names anything,
because no instance takes a cell at any rotation.

---

## 2026-08-15 — the committed receipt count, for the fourth time

*README, "Where to look".*

**The chain:** **thirty-four** → **thirty-seven** → **thirty-nine** → **forty**. The
count of receipts added since FDR Appendix C was written moved with it, from "the
three" to "the **five** added since" to "the **six** added since".

**Why**, verbatim from the last correction:

> (**Corrected 2026-08-15: this said thirty-nine and "the five added since".**
> `git ls-files docs/receipts | wc -l` counts **40**. This is the third iteration of
> the same error — the line has read thirty-four, thirty-seven and thirty-nine, each
> corrected one commit after the receipt that invalidated it. The count is one shell
> command; it should be re-run whenever a receipt is added rather than carried forward
> by arithmetic.)

and, from the correction before it:

> (**Corrected 2026-08-14: this said thirty-seven, which was the 34 + 3 arithmetic
> written before the two real-photo receipts landed.** `git ls-files docs/receipts`
> counts 39.)

The sixteen-without-a-generator numerator has never moved. Only the denominator has,
four times, and always by the same mechanism: a receipt was committed and the prose
total was carried forward by arithmetic instead of being re-counted.

---

## 2026-08-15 — the seeding receipt's training numerics

*README, "Running it".*

**Read:** **1.8164 / 0.3590** twice at seed 20260812, against **1.7972 / 0.3426** at
seed 20260813.

**Reads now:** **1.7619 / 0.3817** twice at seed 20260812, both hashing to
`192d868597ab3ad1`, against **1.8099 / 0.3330** at seed 20260813.

**Why:** `recog/seg_dataset.py::_rng_for_worker` now mixes `torch.initial_seed()`
into the augmentation stream, fixing a bug in which the crop-jitter multiset repeated
byte-for-byte from epoch 2 onward. Training numerics legitimately moved with it. The
property the receipt exists to certify is untouched — the same-seed pair is still
bit-identical and the different seed still differs — and the gap between seeds
widened from 0.0192 / 0.0164 to 0.0480 / 0.0487.

---

## 2026-08-15 — a test named as weak that had already been made strong

*README, "Five defects, and none of them was the model".*

**Read:** "**Two** more are still standing on purpose — `tests/test_bay.py` pins
'zero seated cells is fine' and `tests/test_packing_move.py` asserts only that a
function exists."

**Reads now:** **one** is still standing on purpose: `tests/test_bay.py`.

**Why:** the claim about `tests/test_packing_move.py` stopped being true at
`a00a19f`, which inverted the file. Its test is now
`test_pack_cartridge_is_gone_from_both_modules` and it asserts `not hasattr(bin_packing,
"pack_cartridge")`, the same for `common.packing`, and that the name is out of
`__all__` — it guards a *deletion*. Its docstring records why: the adapter it used to
keep alive re-armed `mm_per_px: float = 0.38`, the placeholder scale that under-read
24 of 30 cartridges by 27 % at the median. A sibling test,
`test_no_module_re_arms_the_placeholder_scale`, walks every exported packing function
and fails if any carries an `mm_per_px` default at all.

Getting this wrong understated the project's own rigour in the middle of a passage
about not overstating it, which is the direction of error that paragraph exists to
catch.

---

## 2026-08-15 — the forbidden-mask-aware FFDH arm's timings

*README, "How the packer picks".*

**Read:** **0.32 ms at 2.5 % coverage, peaking at 1.05 ms at 15 %** — itself the
2026-08-12 re-reading (see the 2026-08-14 entry below).

**Reads now:** **0.35 ms at 2.5 % coverage, peaking at 1.07 ms at 10 %**. The
`aware us` column that produced it reads 109.9 / 354.2 / 598.7 / 1065.1 / 1060.8 /
839.1 µs, mean 0.67 ms, against the 2026-08-14 run's 85.9 / 321.7 / 568.7 / 1008.8 /
1050.0 / 904.8 µs, mean 0.66 ms.

**Why:** the microsecond columns are wall-clock and move on every regeneration; the
receipt says so. The **cell counts did not move** — `forbidden_bench.csv` and the
eight-seed sweep in `forbidden_bench_seeds.txt` came back byte-identical through the
2026-08-15 packing-strip fix, which is what exonerates `common/packing.py` from that
fix's movement elsewhere.

---

## 2026-08-14 — segmenter boundary displacement and placeable-area error

*README, "The headline result, and what it is not".*

**Read:** boundary displacement **0.949 mm**, placeable-area error optimistic by
**51.5 mm²** per crop.

**Reads now:** **1.226 mm** over the 35 bay-carrying crops the boundary row
scores, and **79.2 mm²** per crop over all 126 validation crops.

**Why**, verbatim from the passage as it stood:

> **Corrected 2026-08-14: this sentence read 0.949 mm and 51.5 mm² until today,
> and the 0.949 contradicted the 1.226 this same README quotes in the segmenter
> section below.** Both superseded figures were computed by multiplying pixel
> distances by the generator's *nominal* 0.625 mm/px — the framing at margin 1.0,
> zoom 1.0, which describes no frame in this corpus, because
> `recog/synth3d/world.py` randomises both per scene. Every crop is now converted
> at its own frame's ground sample distance, a median 1.31× larger; a length moves
> by that factor and an area, being an area, by its square.
> `docs/receipts/seg_eval.txt`:37 and :52 are the rows. FDR §13.2.1 retires 0.949
> explicitly and says the same thing at greater length.

Note what made this one detectable: the README was internally inconsistent, quoting
0.949 in one section and 1.226 in another for the same quantity.

---

## 2026-08-14 — the `segmentation*.yaml` config family

*README, "Running it".*

**Read:** "the ten `segmentation*.yaml` configs".

**Reads now:** 11 files match the glob, 9 of them are training configs, all 9 are
byte-identical apart from three path values, and 8 of those 9 are the
generalisation runs.

**Why**, verbatim:

> **Corrected 2026-08-14: this read "the ten `segmentation*.yaml` configs", and
> none of that was right.** The glob matches **11** files, not ten. Two of the
> eleven are not training configs and are not in the family, each by its own
> declaration: `configs/segmentation_cad_test.yaml` is an evaluation config at
> `train_val_split: 0.0`, and `configs/segmentation_seedcheck.yaml` is a one-epoch
> seeding probe whose header says it "is not a training config, and not a member of
> the … generalisation family". That leaves nine, and all nine are byte-identical
> apart from the three paths. The **eight** that FDR Appendix C and that header both
> name is the eight *generalisation* runs and is still correct as scoped — it
> excludes `configs/segmentation_anchored_crown.yaml`, which belongs to the
> sealed-unit experiment rather than the generalisation sweep, and which is
> nonetheless byte-identical to the rest on the same three keys. So: 11 files, 9
> training configs, 9 in the byte-identical set, 8 of them generalisation runs.
> Detail: FDR Appendix C.

Three different counts — 11, 9, 8 — are all correct under their own scoping, which
is why the wrong one survived as long as it did.

---

## 2026-08-14 — the adversarial audit count

*README, "Where to look".*

**Read:** "six … run on 2026-08-12".

**Reads now:** **nineteen** adversarial reviews — sixteen run on 2026-08-12 (A–P)
and three on 2026-08-14 (T, U, V).

**Why**, verbatim:

> (Corrected 2026-08-14: this said "six … run on 2026-08-12", the count at the
> time A–F were the only ones. `docs/README.md` and
> `docs/superpowers/specs/README.md` were brought up to nineteen on 2026-08-14;
> this line was missed.)

A stale count that survived because two of the three places carrying it were
updated and the third was not.

---

## 2026-08-14 — the FFDH-alone timing comparison

*README, "How the packer picks".*

**Read:** "0.9 ms for FFDH alone", and the shipping packer described as inside the
8 ms O3 budget by 3.4 ms at the worst mask measured.

**Read next:** the forbidden-mask-aware FFDH arm at **0.32 ms at 2.5 % coverage,
peaking at 1.05 ms at 15 %**, and the shipping packer inside the budget by
**0.6 ms at the worst mask measured**. The aware-arm half was re-measured again on
2026-08-15 (entry above); the 0.6 ms is current.

**Why**, verbatim:

> (**Corrected 2026-08-14: the comparison figure read "0.9 ms for FFDH alone",
> which reproduces no statistic in the receipt.** `forbidden_bench.txt`'s `aware us`
> column runs 85.9 / 321.7 / 568.7 / 1008.8 / 1050.0 / 904.8 µs across its six
> coverage levels — mean 0.66 ms, max 1.05 ms — so 0.9 ms is neither the mean nor
> the peak, only a near-miss for the 25 %-coverage row. It survives from a passage
> already struck in FDR §6.3.1; the corrected aware-arm figures quoted here are that
> same subsection's, re-read from this receipt on 2026-08-12.)

---

## 2026-08-13 — real-photo receipts landed

Not a correction of a published figure, but the event that made thirty-seven
wrong: `real_photo_eval.txt` and `real_photo_eval_include_empty.txt` were
committed at `9b38de9`, taking the receipt count from 37 to 39. Recorded here
because the receipt-count chain above is unintelligible without it.

---

## 2026-08-12 — the demo's run counts

*README, "Running it".*

**Read:** 37 cartridges, 77 batteries, 33 placement areas and 41 queue poses over
10 cycles, "identical in all ten runs".

**Reads now:** 1 cycle, 3 cartridge detections, 5 loose cells, 3 placement areas,
3 queued poses, 2 released reservations, 0 placement disagreements, 0 bad detector
boxes — byte-identical across five consecutive runs on a fresh clone.

**Why**, verbatim:

> **Corrected 2026-08-12: this section previously claimed 37 cartridges, 77
> batteries, 33 placement areas and 41 queue poses over 10 cycles, "identical in all
> ten runs". No reader could ever have reproduced that, and the numbers are withdrawn
> rather than restated.** They were measured in a working tree holding
> `recog/checkpoints/best.pt` with torch installed, so `recog.inference.load_detector`
> returned the trained Faster R-CNN. `recog/checkpoints/` is gitignored and **no `.pt`
> is tracked anywhere in this repository**, so on any clone that same call logs
> `No checkpoint at recog/checkpoints/best.pt (or torch unavailable). Using
> HeuristicDetector fallback` and the pure-OpenCV detector runs instead — which is
> exactly what "torch-free by design" means, and the counts above are its counts.
> Frame count was not the variable: the trained detector returns the same 37 / 77 / 41
> on the 50 frames step 2 generates as on the 100 an older tree happened to hold. The
> receipt names `detector : HeuristicDetector` on its eighth line so the two paths can
> never again be confused for one another. Diagnosis:
> `docs/superpowers/specs/2026-08-12-fix-first-impression.md`.

The withdrawn numbers were not wrong about the machine they were measured on. They
were unreproducible by anyone else, which is the same thing as wrong here.

---

## 2026-08-12 — the packer's own timing

*README, "How the packer picks".*

**Read:** "3.4 ms mean / 4.6 ms worst".

**Reads now:** **2.8 ms mean / 3.9 ms p95 / 7.4 ms worst observed** on bench masks.

**Why**, verbatim:

> (Corrected 2026-08-12: "3.4 ms mean / 4.6 ms worst" was not a statistic of the
> data. The mean was pessimistic and the worst case optimistic; the second error is
> the one that mattered. Recomputed over all 240 runs of the `us_best` column, which
> lives in the deliberately-gitignored `docs/receipts/forbidden_bench_timings.csv`
> and is regenerated by `python scripts/forbidden_bench.py`.)

The direction is the finding: an optimistic worst case is the error that lets a
latency budget be declared met when it is not.

---

## 2026-08-12 — the oracle's cell and cartridge counts

*README, "The headline result, and what it is not", and "A cartridge that cannot
be certified".*

**Read:** the ground-truth oracle at **27 cells** and **12 cartridges**; the
zero-wall-inset relaxation recovering **two** `AnkerPowerCore10000` instances.

**Read next:** **25 cells** and **11 cartridges**; the relaxation recovering
**one**. Both were superseded again on 2026-08-15 and now stand at 23 cells and
9 of 29, with the relaxation recovering none. The full chains — 27 → 25 → 23 for
the oracle, and 2 → 1 → 0 for the recovered instances — are in the 2026-08-15
entries above. They are recorded here in full because the README no longer carries
any of the earlier values.

**Why**, verbatim from the two passages:

> The 27-cell figure this README carried until 2026-08-12 was measured at
> `ce1d9cd`, when a grid cell counted as free if its *centre pixel* was free;
> `b93bbd3` made a cell free only if all of it is, and re-running the oracle at HEAD
> costs it two of those 27 cells and one of its twelve cartridges.

> A second instance at 64.7 mm recovered too until `b93bbd3` stopped calling a
> half-wall grid cell free — it existed only because the occupancy grid quantised in
> the optimistic direction, and re-measuring the oracle at HEAD removes it.

One code change, `b93bbd3` (2026-08-11), moved both. The recovered instance that
survives is 65.1 mm of free strip against the 65.0 mm the cell needs; the one that
did not was 64.7 mm, and existed only because the grid rounded in the generous
direction. The correction existed to put both sides of the headline comparison on
the same code state — a goal the 2026-08-15 entry reopened, because the shipping
side is now the one awaiting re-measurement.

---

## 2026-08-12 — the `demo_seg` receipt

*README, "The same loop with the trained segmenter in it".*

**Read:** *6 poses queued / 2 pick-and-places*.

**Reads now:** 4 poses queued, 1 pick-and-place.

**Why**, verbatim:

> The previous figures, *6 poses queued / 2 pick-and-places*, counted two poses the
> arm could never have reached.

`planning.camera.workspace_bounds_mm` was parsed and compared against nothing until
2026-08-12. The receipt could not be regenerated at all between 2026-08-11 and
2026-08-12, because the loop aborted on an out-of-reach candidate rather than
skipping and counting it —
`docs/superpowers/specs/2026-08-12-fix-demo-workspace.md`. On the trained-detector
path the same fix took queue poses from 62 to 41.

---

## 2026-08-12 — the heartbeat that was never sent

*README, "How it works".*

**Read:** execution offering "retries, heartbeat and E-stop".

**Reads now:** **there is no heartbeat.**

**Why**, verbatim:

> this line said "retries, heartbeat and E-stop" until 2026-08-12, and
> `OpCode.HEARTBEAT` exists only as an enum value and a simulator dispatch arm;
> nothing sends one, and neither end runs a watchdog.

FDR §7.5 states what that costs: the E-stop covers a robot that misbehaves and a
link that degrades, and not a host that stops running. This is a withdrawn
*capability*, not a restated number, and it is the most consequential entry in this
file.

---

## 2026-08-12 — the batched-segmentation timing table

*README, "Two placement-area extractors, and only one is for real cartridges".*

**Read, in sequence:** 20.2 / 76.5 ms at commit `4e3c03e`; 21.2 / 88.0 ms at the
2026-08-11 scale correction; an intermediate **40.9 ms / 157.0 ms**; and an earlier
**16.7 ms / 60.0 ms** on the same hardware.

**Read next:** **16.6 / 57.1 ms** on 2026-08-12.

**Reads now:** **16.2 ms for 8 cartridges batched against 58.6 ms for the same 8 run
in a loop**, on an RTX 3060.

**Why:** the table is wall-clock and is re-taken every time
`docs/receipts/seg_eval.txt` is regenerated, so it moves a little each time — a
16.2–21.2 ms batched spread across six clean runs. The 40.9 / 157.0 reading is the outlier
and is superseded rather than averaged in: it was taken while the machine carried
substantial unrelated GPU load. FDR §13.2.1 has the measurement conditions. The
conclusion the table exists to support — batched inside the 50 ms end-to-end budget,
looped well outside it — holds at every reading.

---

## 2026-08-12 — the three inconsistent `tau` values

*README, "Two placement-area extractors, and only one is for real cartridges".*

**Read:** a confidence threshold `tau` quoted at three mutually inconsistent values
in three places — **code 0.85, YAML 0.7492, README 0.5715**.

**Reads now:** **there is no `tau`.** The IoU between the two placement estimates is
still computed and reported on `PlacementArea.consistency_iou`, but nothing gates on
it, and the constructor no longer accepts the argument, so code that still passes it
fails loudly instead of being silently ignored.

**Why:** the gate was retired on measurement, not taste — a gate needs agreement and
error to move in *opposite* directions, and they move together in all four SKUs
(defect 2 and the first negative result in the README; FDR §13.2.1,
`docs/receipts/tau_independence_correlation.txt`). `recog/calibrate_tau.py` and its
receipt are kept as the record of the measurement that retired it.

---

## Undated — the two denominators on the segmenter's crops

*README, "The same loop with the trained segmenter in it".*

**Read:** pooled `bay` IoU and mean boundary displacement both attributed to the
same **36** validation crops.

**Reads now:** IoU over **36** bay-carrying crops, boundary displacement over
**35** of those 36 (`docs/receipts/seg_eval.txt`:20 and :37).

**Why:** displacement is only defined on a crop where a predicted `bay` boundary
exists to measure. One crop carries a true bay the segmenter predicted nothing for,
so it scores an IoU and no displacement. No datestamp was recorded on this
correction when it was made.

---

## Still standing

Four things in this file are not closed and should not read as if they were.

* **The shipping pipeline's own placement totals are withdrawn.** The oracle side
  became receipt-backed on 2026-08-15; the shipping side did not, and re-deriving it
  needs the detector-plus-segmenter instance-pairing pass re-run at HEAD. Until then
  no *difference* between the two may be quoted, which is what the withdrawn
  "perfect perception is worth zero net cells" sentence was.
* **The torch-free test count is unmeasured at this commit and deliberately not
  restated.** It last read "1,228 of 1,276 pass, 20 skipped"; the 2026-08-15 fixes
  added tests on both sides of the torch boundary and the clean-install run has not
  been re-done, so the pair is marked for re-measurement rather than scaled by
  guesswork.
* The oracle figures have already been corrected twice (2026-08-12 and 2026-08-15),
  and `AnkerPowerCore10000`'s recovered-instance count three times. They are the
  figures most likely to move again.
* Sixteen of the forty committed receipts are inherited from before this
  repository's history and have no surviving generator, one of which —
  `tau_independence_correlation.txt` — is current and load-bearing. A figure with
  no regenerable receipt cannot be corrected by re-running anything.

FDR Appendix C enumerates the receipt coverage; the repository
[`README.md`](../README.md) carries the current value of every figure named here.
