# Audit M — What annotating `recog/realtest/` would and would not buy

Read-only exploration. HEAD `f1989e9`, branch `feat/blender-synth-dataset`.
Nothing staged, nothing committed, no repo file modified. All experiments ran
against copies in a scratchpad directory.

**Verdict in one line:** the tooling works, the annotation is cheap, and the
result would be a *floor test* — it can prove the segmenter fails on real
pixels, it cannot prove it transfers. Do it, but the reframing in the brief is
not a discovery: FDR §13.2.2 already names the missing polygons as the gap, and
annotating them does not change the subsection's substance.

---

## 1. What is actually in the seven photographs

I opened all seven and cropped every one of the 20 `Cartridge` boxes into a
contact sheet. The corpus is much smaller than 7 images / 80 objects suggests.

### 1.1 Scene inventory

| Image | What it shows | Anns |
|---|---|---|
| `IMG_4426.jpg` | Blue jig plate, 6 power-bank cases + loose-cell storage strip at top | 6 Bat, 5 Cart |
| `IMG_4429.jpg` | **Same scene, same session**, slightly wider/shifted framing | 12 Bat, 5 Cart |
| `IMG_4434.jpg` | **Same scene again**, one case's contents swapped (PCB removed, paper+tape+blob added) | 5 Cart |
| `IMG_4427.jpg` | Same bench panned right: 21-cell storage jig + 2 of the same cases clipped at the left edge | 21 Bat, 2 Cart |
| `IMG_4433.jpg` | **Same as 4427**, marginally different angle | 21 Bat, 2 Cart |
| `IMG_4435.jpg` | Close-up of one case on the bare bench (the swapped one from 4434) | 1 Cart |
| `IMG_4428.jpg` | Near-empty jig plate with white plastic cylinders; cases and a cell strip clipped at the right edge | **0** |

One camera, one session, one bench, one lighting setup, one backdrop, and
handheld oblique geometry throughout.

### 1.2 The 20 cartridges are 5 physical objects

Matching the contact-sheet crops by content, the 20 `Cartridge` instances
resolve to **five distinct physical cartridges**, one of which appears in two
content states:

| Specimen | Description | Appears as |
|---|---|---|
| A1 | Long black ribbed bay, PCB module at one end, small white label on floor | `#2, #37, #76, #33, #60` (5×) |
| A2 | Square-ish black bay, empty but for a white tape cross | `#1, #36, #75, #34, #59` (5×) |
| A3 | Long black ribbed bay, PCB module + heavy translucent adhesive smears | `#10, #38, #78` (3×) |
| A4 | Small case — PCB + yellow tape in 4426/4429; white paper + yellow tape + black blob in 4434/4435 | `#9, #35` / `#77, #80` (4×) |
| A5 | Long grey ribbed bay, essentially empty, one small printed label | `#11, #39, #79` (3×) |

**Every one of the 20 annotated cartridges is OPEN.** Sealed, featureless
black shells are clearly visible in 4426/4429/4434 (bottom rows) and are **not
annotated**. `ANNOTATION_PROTOCOL.md` §9 explicitly instructs the annotator to
label one of them ("a smooth, featureless black block ... a sealed `cartridge`,
one outline, nothing else"), so a mask pass would be labelling objects the box
set does not contain. The mask set would not be a superset of the box set.

### 1.3 Zero cells are seated in a bay — verified, not eyeballed

Every `Battery` box is a loose 18650 in the jig's own storage holes or on the
bench. Checked programmatically: **0 of 60 battery boxes lie inside a cartridge
box, and 0 of 60 overlap one even partially.**

This is the single most consequential finding in this audit. The task the
segmenter exists to perform is "which floor is free, given what is sitting in
the bay", and the modal `placement_area` rule (protocol §1.1, "the most
important rule in this document") is specifically about floor occluded *by a
seated cell*. `recog/check_annotations.py` singles out the
`placement_area`/`battery` overlap pair by name as "the single most likely
mistake". **That condition occurs zero times in this corpus.** The real set can
only exercise the empty-bay and module-occupied-bay regimes.

### 1.4 Are all five classes present and annotatable? Yes — with caveats

- `cartridge` — yes, abundant, open and sealed. Walls and rim are clearly
  visible (the oblique angle actually *helps* here; a top-down shot would show
  less wall).
- `placement_area` — yes, and with useful range: A5's ribbed floor is almost
  entirely free; A3's is almost entirely covered in adhesive. A legitimately
  near-zero `placement_area` is a valuable real case.
- `electronics_module` — yes, 3–4 distinct boards (3 seated in bays, plus what
  protocol §9 reads as a loose module in a jig pocket). ~10 instances.
- `obstruction` — yes, and richer than the synthetic generator's: white tape
  crosses, printed labels, yellow masking tape, a black blob, and translucent
  stringy adhesive smears.
- `battery` — yes, 60+, but all loose. Not one in a bay (§1.3).

Two genuine ambiguities the protocol does not resolve: the **white plastic
cylinders** scattered across 4428 and 4426 (dummy cells? spacers? locating
pins?) — they are 18650-ish but plainly not cells; and whether 4428 is
"legitimately empty" or simply unlabelled. Protocol §2 fixes rules for
ambiguous *regions*, not for unmodelled *objects*.

---

## 2. The realistic n

`BaySegDataset` was run against a converted sidecar built from the existing
boxes (§5). It produces **exactly 20 crops** — one per cartridge unit; loose
battery units are correctly dropped because a unit is kept only if it carries a
cartridge-related annotation.

So the headline denominator for a mask metric is:

- **crops-with-a-bay: 20** — all open, all annotatable for `placement_area`.
- **statistically independent units: 5** (six if A4's two content states count
  separately). The 20 crops are 5 objects photographed 3–5 times each from
  near-identical viewpoints. Within-specimen correlation is close to 1: a
  segmenter that fails on A3 fails on all three A3 crops.
- `electronics_module`: ~10 instances, **3–4 independent**.
- `obstruction`: ~25 instances, **~8 independent**.
- `battery`: 60 instances, ~28 unique cells, **all out-of-bay**.

**Does mask annotation settle the heuristic-vs-segmenter comparison?** No. The
tray-interior spec's judgement — "It should not be expected to resolve on
n = 20, and a shift either way at that sample size is not evidence" — was
already generous, because it treated n as 20. The real n for a *generalisation*
claim is 5. Mask IoU replaces a proxy quantity (placeable fraction of ROI) with
the right quantity, measured on the same five objects. **It gives the same
underpowered comparison a better-named metric.** The 0.211 / 0.232 / 0.318
series moved by more than the effect it exists to detect; there is no reason to
expect a mask-IoU series over the same five objects to be more stable, and the
metric change gives no additional statistical power whatsoever.

---

## 3. What the result could and could not claim

### 3.1 Could

1. **A negative result is genuinely defensible.** If mask IoU on real crops
   comes back at 0.05–0.25 against 0.80 on synthetic validation, that is strong
   evidence the segmenter does not transfer, and it survives n = 5: a model
   that works would not fail on all five. **The asymmetry is the whole value
   proposition** — a bad number is informative, a good number is not.
2. **First real-domain mask measurement of any kind exists.** Design spec §9's
   headline framing ("placeable-area IoU against human polygons") becomes
   computable instead of deferred, and `seg_ablation`'s stand-in metric can be
   retired or reported alongside.
3. **The dormant annotation tooling gets exercised on real data** — though see
   §5: I have now done this, and it works.
4. **A per-condition qualitative decomposition** across five well-characterised
   bays (empty / module / adhesive-covered / near-free) that the synthetic
   metrics cannot give.

### 3.2 Could not

1. **Any transfer validation.** n = 5 objects, one session, one lighting, one
   backdrop, one camera.
2. **Any per-class or per-SKU figure.** `electronics_module` rests on 3–4
   independent instances. Per-SKU is worse — no SKU is recorded for any of
   these, and A1/A3/A5 may or may not be distinct products.
3. **Any claim about the deployed geometry.** These are handheld oblique phone
   shots. `docs/superpowers/specs/2026-08-09-spec3-realism-decisions.md` fixed
   the deployed camera as near top-down, 0–10° off vertical, ~400 mm. Protocol
   §7.3 is explicit: these seven "stay useful — as a **stress set**, not as
   part of this collection", and §9 says IMG_4426 is "part of the *existing*
   oblique set, **not something to re-annotate into the new collection**".
   **The project's own protocol argues against this exercise.**
4. **Anything about seated-cell occlusion** (§1.3, n = 0) — the dominant real
   condition and the one the modal rule exists for.
5. **Anything about sealed cartridges** (currently n = 0 annotated), which the
   FDR reports a synthetic decomposition for.
6. **Anything about model iteration.** No checkpoint comparison on this set can
   be read as improvement (§2).

### 3.3 Suggested FDR wording

§13.2.2 should **not** be softened. Its substance — sim-to-real transfer is
unvalidated and cannot be validated here — is unchanged by mask annotation,
because the constraint was never "no images" (the subsection already says
"7 photographs, 20 annotated cartridges, 80 boxes and **zero segmentation
polygons**"). If the annotation is done, add to §13.2.1, not §13.2.2:

> The seven photographs in `recog/realtest/` have since been annotated with
> five-class polygons, and mask IoU against them is reported here for the first
> time. **This is a smoke test, not a transfer measurement, and the sample is
> smaller than it appears.** The 20 cartridge crops are five physical
> cartridges photographed three to five times each in a single session, under
> one lighting setup, on one backdrop, with a handheld phone at oblique angles
> — not the near-top-down fixed mount the system is specified for
> (`docs/ANNOTATION_PROTOCOL.md` §7.3 designates this set a stress set for
> exactly that reason). No cell is seated in any bay in any of the seven
> photographs, so the modal `placement_area` rule — the free floor being
> occluded by a cell, which is the condition the segmenter exists to resolve —
> is exercised zero times. A low IoU here would be evidence that the segmenter
> does not transfer; a high IoU here would not be evidence that it does. §13.2.2
> stands unchanged.

---

## 4. What the annotation would cost

Five classes, seven images, in LabelMe, by one person who has read the protocol.

| Item | Hours |
|---|---|
| Read protocol (29 KB, ten sections of rulings) + LabelMe setup | 0.75 |
| 4426 / 4429 / 4434 — ~6 cartridges each incl. sealed, 3–4 modules/obstructions, 6–12 loose cells | 3.0 (1.0 each) |
| 4427 / 4433 — 21 cells each + 2 clipped cartridges | 1.2 (0.6 each) |
| 4428 — resolve the white-cylinder question, clipped cases, cell strip | 0.4 |
| 4435 — single close-up, paper + tape + blob | 0.3 |
| Convert → validate → fix loop (expect 1–2 rounds of overlap errors) | 1.0–2.0 |
| **Total human clicking + rework** | **6.5–8.5 h** |

Two corrections to the brief's expectation:

**`placement_area` is *not* the expensive class here, and its worst failure
mode is absent.** Protocol §4's paint-order rule means the annotator draws
`placement_area` as one generous rectangle over the whole floor and lets the
`electronics_module` and `obstruction` polygons cut the holes automatically —
6 to 10 clicks, not a careful trace. And because no cell is seated in any bay
(§1.3), the hardest modal call — floor under and behind a cell — never arises.
The residual modal risk is real but narrow: floor under a module and under a
label.

**The actual label-noise hotspot is A3's translucent adhesive smears.** They
are wispy, semi-transparent, and have no defensible boundary; they appear in
three crops; and `obstruction` is late in the paint order, so an over-generous
tracing is *not* corrected and eats `placement_area` directly. Expect the
largest annotator-to-annotator disagreement here, not on the floor. Second
hotspot: the 60 cell outlines, since `battery` wins all ties and nothing
rescues a sloppy one.

**Not counted above: the polygon-IoU comparison does not exist in code.**
`recog/seg_ablation.py` checks `has_ground_truth_polygons` at run time and, in
the branch where polygons *are* present, prints only:

> "this run still reports placeable-fraction-of-ROI for continuity with the
> measured 0.218 baseline, not polygon IoU — a later revision could add the
> polygon-IoU comparison now that ground truth exists."

`_load_real_cartridges` also hardcodes `annotations/instances_default.json` and
reads it through `recog.dataset.parse_coco_json` (a 2-class map), so the 5-class
sidecar would need plumbing. Add **3–6 h of development plus a receipt**. Total
honest cost: **10–15 h**.

---

## 5. Does the tooling work on this input? Yes — verified, first try

I generated LabelMe JSON from the 80 existing boxes (rectangles as 4-vertex
polygons, one `group_id` per cartridge, none for loose cells), then ran the
real pipeline against real 3024×4032 JPEGs:

```
$ python -m recog.labelme_to_seg <scratch>/images <scratch>/annotations/instances_seg.json \
      --images-dir <scratch>/images
note: 80 of 80 annotation(s) declare no SKU, so recog.seg_evaluate --per-sku will
group them under a single 'None' asset.
wrote 80 annotation(s) across 7 image(s)

$ python -m recog.check_annotations <scratch>
  ... [OK] placement_area_battery_overlap: 0 px (×7 images)
  [ERROR] no_annotations: 'IMG_4428.jpg' has an image record but zero annotations
  [WARN] empty_class: 'electronics_module' / 'placement_area' / 'obstruction'
  RESULT: FAIL

$ BaySegDataset(instances_seg.json, images, train=False)
  crops: 20     img torch.Size([3,256,256])   tgt torch.Size([256,256])
```

Findings:

- **The converter accepts real photographs unmodified.** No resolution limit,
  no aspect-ratio assumption, no crash on 12 MP portrait JPEGs.
- **The validator produces exactly the right complaints** — including catching
  the IMG_4428 gap it was written to catch, and correctly reporting the three
  unlabelled classes.
- **`BaySegDataset` builds 20 crops** from the result with zero code changes.
  The path from human polygons to a scored crop is real and unbroken.
- Class ids are compatible: `CLASS_MAP` (`battery`:1, `cartridge`:2) and
  `seg_class_ids()` agree on 1 and 2, so a 5-class sidecar is also readable by
  the box-level reader (classes 3–5 are skipped as unknown). The only blocker
  is the hardcoded `instances_default.json` filename in `seg_ablation`.

**Can the boxes seed a LabelMe starting point?** Technically yes — the script
above is a working seeder and took ten minutes. But it saves **20–30 %, not
half**:

- The boxes carry **nothing** for `placement_area`, `electronics_module` or
  `obstruction` — the three new classes and the entire point of the exercise.
- 20 `cartridge` rectangles give a usable outer-rim start; the inner cavity
  boundary must still be drawn.
- 60 `battery` rectangles are the biggest saving. Because every loose cell sits
  on the *unlabelled* blue jig, an over-generous battery polygon costs nothing
  there — the corners just need rounding. ~45 min drops to ~25 min.
- **Risk:** anchoring. An annotator correcting a rectangle produces boxier
  masks than one drawing free. Real masks systematically boxier than synthetic
  ones would bias the IoU in an undocumented direction. If seeding is used,
  seed `battery` and `cartridge` only, and say so in the receipt.

---

## 6. Alternative uses of the same effort

| Option | Cost | What it buys |
|---|---|---|
| **(a) Fix the box-level gaps** — annotate IMG_4428 (0 boxes, photo has a full cell strip and clipped cases) and the 7th cell in IMG_4426 that protocol §1.5 counts but the COCO file does not | **0.5 h** | Corrects an *already-published* real number. `recog/eval_real.py`'s own docstring records that IMG_4428 "carries zero boxes ... while the photograph itself is full of cells" and depresses whole-set AP. Highest value per hour in this audit, by a wide margin. |
| **(b) Qualitative failure-mode review** — run the segmenter over the 20 crops, dump overlays, write up what it gets wrong on real pixels | **2–3 h** | Converts §13.2.2's "unknown and not knowable" into *described*. Makes no measurement claim, so no claim can be misquoted. Would also tell you, before spending 10 h, whether the answer is obviously "it fails" — in which case the masks are unnecessary. |
| **(c) Full 5-class mask annotation** | **10–15 h** | A floor test (§3.1). Real but bounded. |
| **(d) 7 new rig-realistic photographs** at protocol §7.1 geometry | unavailable | Would be worth more than (c), but §13.2.2 records the hardware as unobtainable as of 2026-08-09. Moot unless that changed. |

(a) and (b) together cost under 4 h and are strictly prerequisite to (c):
(a) because a corpus with a known 0-annotation image should not have masks
layered on top of the same gap, and (b) because it is the cheap version of (c)'s
most likely finding.

---

## 7. Recommendation, and the case against it

**Recommendation: do (a) now, (b) next, and (c) only if (b) is inconclusive —
and report (c) as a smoke test with an explicit floor-test framing.**

The reasoning: the honest expected value of (c) is asymmetric. If the segmenter
transfers badly, five objects are enough to show it and the project gains a real
finding it currently lacks. If it transfers well, five objects prove nothing and
the number is a liability. (b) resolves which world you are in for a fifth of
the cost. That ordering is the recommendation, not "annotate" or "don't".

**The case against, stated properly:**

1. **This is not a discovered asset.** FDR §13.2.2 already enumerates the seven
   photographs, the 20 cartridges, the 80 boxes and the zero polygons, by name,
   in the same sentence. The limitation as written is already the narrow one
   ("no real *masks*"); the brief's reframing restates §13.2.2 rather than
   correcting it. Nothing in §13.2.2 becomes false if the annotation happens.
2. **The project's own annotation protocol argues against it.** §7.3 designates
   `recog/realtest/` a stress set and directs new work to `recog/realtest_rig/`;
   §9 says IMG_4426 is "not something to re-annotate into the new collection".
   Overriding a written, reasoned decision requires better grounds than "the
   images are already here".
3. **The corpus is missing the condition that matters.** Zero seated cells
   (§1.3) means the measurement covers the easy half of the task and is silent
   on the half the modal `placement_area` definition was written for.
4. **n = 5, not 20.** The three-point series that motivated all of this
   (0.211 / 0.232 / 0.318) moved because five objects photographed repeatedly
   cannot resolve the effect. A new metric on the same five objects inherits
   that exactly.
5. **Misquotation risk is the real cost.** This project has spent an entire FDR
   subsection refusing to state a transfer figure. Producing a real-photo mask
   IoU creates, for the first time, a number *shaped* like the one §13.2.2
   withholds. Every downstream reader — CV bullets, portfolio, a future
   maintainer — will be one careless sentence away from quoting it as the
   transfer result. §3.3's wording exists to blunt that, and wording is a weak
   defence.

If the decision is to skip (c) entirely, that is defensible and should be
recorded as a finding in its own right: *the seven existing photographs cannot
support a defensible transfer claim at any annotation effort, because five
physical cartridges at the wrong camera geometry with zero seated cells is not a
sample — and the question of whether to annotate them, open since the corpus
was committed on 2026-08-05, is therefore closed.*
