# Results summary — vision-guided 18650 kitting cell

**For:** Professor Svetan Ratchev FREng
**From:** Yousif Al-Haidary — MEng Individual Project, University of Nottingham
**Date:** 14 August 2026 · repository at `83a1383`

This is a results summary, not a second report. Every figure below is followed
by the receipt in `docs/receipts/` that produced it, and every figure carries
its scope in the same sentence, because the scopes differ sharply and the
difference is the substance of section 4.

---

## 1. What it set out to do, and where it landed

The brief was a vision-guided cell that recognises loose 18650 cells and
partly-filled cartridges from a fixed overhead camera, decides where the next
cell can legally go, and streams pick-and-place poses to a KUKA KR 6 R700 over
EthernetKRL. Perception, planning and execution were built and run end to end
in software. The robot was withdrawn for an external programme in mid-March
2026 and lab access was not regained, so the execution stage runs against a
mock controller and a protocol conformance suite rather than against hardware.

Of the six success criteria: three pass on measurement (O1 in-domain, O5, O6),
one passes on its threshold with its headline distribution withdrawn as
un-receipted (O3), one **fails** as an absolute bound once it was measured for
the first time (O2 — it had previously been carried as a pass citing a test
that measures nothing), and one is **not tested** for want of a robot (O4).

The single most useful thing the project produced is not on that list. It is a
geometric result about what a camera can and cannot certify, and it came out of
asking why one cartridge never accepted a cell.

---

## 2. The finding: an exact fit cannot be certified by a camera

One of the four CAD assemblies in the corpus, `AnkerPowerCore10000`, has a
placement region of **54.9 × 65.0 mm** — its `interior_mm` less its
`module_bay_mm`, read off the imported CAD in
`recog/synth3d/assets/catalog.json`. The planner reserves a nominal cell
footprint of **18.5 × 65.0 mm** (`configs/planning.yaml`: the 18650's
CAD-measured 18.3 mm diameter plus 0.2 mm of deliberate margin). The bay is
one cell long. Exactly: 65.0 against 65.0, **+0.00 mm** of longitudinal margin.

The placement rectangle is axis-aligned by construction, on the sound ground
that the camera mount is fixed. The *cartridge* is not: the scene generator
seats every unit at `quarter × 90° ± 2°` of jitter, and a real jig has
clearance of the same order. An axis-aligned strip of width `w` fits inside a
bay of height `H` rotated by θ only where `L(θ) = (H − w·sinθ)/cosθ ≥ 65.0`,
so the packer alone consumes **18.5·tanθ ≈ 0.32 mm of length per degree**
against 0.00 mm available. The first fraction of a degree is terminal, and
nothing is recovered until 31.8°, which no jig jitter reaches.

The consequence is unconditional on perception. Feeding **ground-truth** label
maps through the shipping extractor and packer at each frame's **true** scale —
perfect segmentation, perfect boxes, perfect calibration — that SKU places zero
cells in **10 of 10** instances at the production 4.25 mm wall inset. Tolerance
is not the lever either: taking the nominal 18.5 → 18.3 mm recovers **0**
instances, and relaxing the wall inset to 0.0 mm recovers **0**. The other
three SKUs carry +1.75, +70.2 and +75.75 mm of margin and are fine; the 13000
is spent by 6.9° of rotation.

**The general form is the transferable part: a bay packed to exact tolerance
cannot be certified by a vision system with non-zero measurement error, because
certification needs margin to absorb that error and an exact fit offers none.**
The corollary that follows is the sentence I would keep if I could keep only
one: **verify contact by force, not by camera.** No guard computed from a
prediction can detect that prediction's own error — every such guard agrees
with the number it was derived from — so the mitigations that work are reducing
boundary displacement or moving the confirmation onto a different physical
channel.

Its standing, stated honestly: the *geometry* is airtight and checkable from
two committed files in about a minute. The *measurement* is n = 10 on one SKU
in one synthetic corpus, recorded in
`docs/superpowers/specs/2026-08-11-placement-feasibility.md` §3 and §5 rather
than in a `docs/receipts/` artefact. That makes this a strong paragraph and a
design-review checklist item, not a general law.

---

## 3. Results

**Perception, in-domain.** The shipped Faster R-CNN detector scores
**mAP@0.50 = 0.9053** (battery 0.9046, cartridge 0.9061; recall 0.9544,
precision 0.9488 over 1,205 ground-truth boxes) on 150 held-out **synthetic**
frames at its shipped 0.70 confidence threshold — `detector_bench.txt` arm 3.
Localisation is the weak half: centroid error is 1.13 px at the median but
only 75.6 % of matched detections fall inside the 2 px bound the project set,
which is why O2 is now recorded as a failure.

**Perception, cross-distribution.** This is the result I would put forward as
the most interesting. A bay segmenter trained on **procedurally generated
trays only — no CAD in its training set at all** — scores **bay IoU 0.8755**
on the 836-crop held-out CAD test split (434 frames, four real Anker
assemblies), `seg_eval_anchored_crown_on_cad_test.txt`. The same procedural
recipe *without* one added generator field (a sampled fillet rolled onto the
lid's top edges, so sealed cartridges present shading structure rather than a
flat face) scores **0.6555** on the identical split,
`seg_eval_anchored_on_cad_test.txt`. Four leave-one-SKU-out controls trained
*on* CAD score **0.9032–0.9131** on the same split
(`seg_eval_cad_control_*_on_cad_test.txt`). So one config field closes most of
the gap between "trained on the real part" and "never saw the real part" —
on synthetic test data, against synthetic ground truth.

**Boundary accuracy and latency, on the shipping model's own synthetic
validation split** (`seg_eval.txt`, 126 crops): mean bay boundary displacement
**1.226 mm** over 35 crops, **3.14×** below the quantisation floor a 28×28
Mask R-CNN head would impose — the measurement that chose a per-ROI segmenter
over a mask head, and a ratio that is scale-invariant. Eight cartridges
segment in **16.6 ms** batched against **57.1 ms** one at a time, so batching
is what keeps the stage inside the 50 ms cycle budget rather than an
optimisation on top of it. The detector does not fit that budget on CPU at
all: median **437.4 ms**, p95 484.2 ms over 100 frames on an i7-12700H at two
threads (`frcnn_latency.txt`). On CPU this is a GPU-or-lighter-architecture
decision, not a tuning problem.

**Planning.** The packer treats obstructions as a forbidden mask rather than
rejection-sampling around them. Over 40 seeds per condition
(`forbidden_bench.txt`), at 5 % forbidden coverage the mask-aware arm places
**+3.25 cells** more than the naive arm (paired t = 15.18, 40 wins / 0 ties /
0 losses); the strategy the planner actually runs adds a further **+2.93
cells** at 10 % coverage (40/40 wins). The effect survives eight different
master seeds (`forbidden_bench_seeds.txt`). Allowing 90° rotation adds
+24.2 % and +57.1 % cells placed on two of four strip geometries and exactly
0.0 % on a third (`ffdh_ablation.txt`) — reported including the null.

**Reproducibility.** 1,210 tests pass, 2 skip; branch coverage is **93 %**
over `main.py`'s 19-module transitive import closure (1,845 statements) and
**67 %** over all 49 modules the coverage config resolves to (7,208
statements), the second figure below the 70 % threshold and on the record
because Blender-only renderers cannot be unit-tested outside Blender
(`pytest-cov.txt`). Training is seeded: the same seed twice gives bit-identical
weights, a different seed does not (`seed_reproducibility.txt`) — with the
caveat, stated in the receipt, that the published checkpoints predate seeding
and cannot be recovered.

**Two figures that moved downward on evidence, both kept visible.** Every
millimetre in every receipt published before 11 August was converted at a
nominal 0.6250 mm/px that describes *no frame in the corpus*; measured
per-frame ground sample distance runs 0.4903–1.0739 mm/px, median 0.8211, so
those figures understated real distances by a median 1.31×, and every receipt
now converts each crop at its own frame's scale (`seg_eval.txt`). Separately,
the confidence gate τ was **retired**: `tau_calibration.txt` shows that not one
of the 35 accepted cartridges admitted a cell at any observed IoU, so the 5 %
failure budget never bound and the returned threshold was simply the lowest IoU
the split happened to contain — a lower bound on where an unsafe boundary might
sit, not evidence that anything was safe.

---

## 4. What this project cannot claim

I would rather state this than have it found.

**There is no physical robot.** The KR 6 R700 was withdrawn in mid-March 2026.
Execution is validated against a mock controller and a protocol conformance
suite over two drivers and two encodings — enough to say the EthernetKRL frame
format and CRC are right, and nothing at all about pick reliability, force,
compliance or recovery. O4 is untested for that reason and is recorded as
untested.

**There is no real-photograph ground truth for the part of the stack that
matters.** Seven handheld phone photographs exist. They carry **80 bounding
boxes across two classes and zero segmentation polygons**. That is enough for a
detector smoke test — the shipped checkpoint scores **mAP@0.50 = 0.8484** on
six of the seven, one excluded for having no ground truth at all
(`real_photo_eval.txt`) — and it is nothing whatsoever for the five-class
segmentation the placement decision actually rests on. It is also the wrong
geometry: handheld, not the fixed near-vertical 400 mm mount the system is
designed around.

**Therefore every segmentation and placement figure in section 3 is
synthetic-to-synthetic**: measured on renders, against ground truth derived
from the same renders. The 0.8755, the 1.226 mm, the per-SKU tables — none of
them is evidence about photographs. The one real-photograph
segmentation comparison available is placeable area as a fraction of the
cartridge ROI over 20 cartridges in 6 images, where three checkpoints of the
same architecture have read **0.211, 0.232 and 0.318** against a heuristic
baseline of 0.217 (`seg_ablation.txt`). The series *is* the finding: run-to-run
variation exceeds the effect the comparison exists to detect, so no transfer
claim is made in either direction.

**And this cannot be fixed from inside the project.** Real photographs are not
obtainable here — confirmed, not deferred. So sim-to-real transfer is
unvalidated and, under this project's constraints, unvalidatable. Measured
against the Omnifactory's own stated business of transforming lab-proven
concepts into production-ready solutions, this sits below the rung where that
work starts. What it is, honestly, is a method and a design-time finding, not
a deployable capability.

---

## 5. What I think is worth offering Omnifactory Mini

Mini's declared envelope is assemblies up to 4 kg and 500 mm in x, y and z, and
its named use cases include the assembly of battery packs for aerospace
applications. That envelope is this project's scale to the point of
coincidence, and the problem class is shared — cylindrical cells placed into a
fixture at tight positional tolerance, at a mix and volume that will not
justify hard automation. I would claim the class and not the identity: an
aerospace pack is not an 18650 kitted into a consumer cartridge.

The offer is the **CAD-to-labelled-synthetic-data pipeline**. It takes a STEP
file, converts it to glTF with a scale guard that refuses parts outside a sane
size band (`recog/convert_cad.py` — a part imported 1000× too small renders
sub-pixel and produces hours of near-empty data with no error raised anywhere),
composes it into randomised scenes, and renders with an object-index pass that
gives **pixel-exact five-class masks** — `battery`, `cartridge`,
`electronics_module`, `placement_area`, `obstruction` — with zero overlapping
pixels by construction, because one index pass physically cannot paint two IDs
onto one pixel. Each dataset ships a `manifest.json` that *is* the resolved
generator config, seed, class map and measured per-class statistics, and the
calibration code prefers it over any authored default and raises rather than
falling back silently when the two disagree. Eleven datasets, 6,018 scenes,
roughly 8 GPU-hours of Cycles rendering on one desktop machine, all eleven
manifests and checksums committed (`docs/datasets/`).

I want to be straight about what is and is not novel here. **Synthetic training
data for manufacturing vision is a crowded field and I am not claiming the
idea.** What is on offer is a working, seeded, receipted implementation with
its limits measured and published, and the specific thing it is good for: a
new part goes from CAD to labelled training data in hours, without an
annotation campaign. That answers a problem your own group has stated in print
— Martínez-Arellano and Ratchev's frugal-industrial-AI framing, that building
a model per machine and per part is not cost-effective and needs expertise the
shop floor does not have. It also consumes an input an MBD shop already holds.
For a demonstrator whose named use cases span aerospace battery packs and
automotive gloveboxes — high mix by construction, where every new product is a
new annotation campaign — that is the cost worth attacking. The reconfigurable
floor already lets the cell change product; the perception stack still cannot.

One note on vocabulary, since the word is overloaded: what this project calls a
digital twin is a cell-level geometric occupancy model in `plan/scene.py` that
tracks which slots are filled and reserved. It is not a plant-level twin in the
sense the Omnifactory maintains, and I do not want it read as one.

---

## 6. The ask — the one experiment I could not run

Section 4 has an exact inverse. The single thing this project provably lacks is
the thing a facility with hardware has trivially: **you can photograph your
parts, and I could not.**

The apparatus for closing that loop is already built and tested, and has never
been run on a real photograph because none could be taken:

- **`docs/ANNOTATION_PROTOCOL.md`** — a 50–100 photograph labelling protocol
  written for the person holding the camera, not the developer. It fixes the
  acquisition geometry (near top-down, 0–10° off vertical, no roll, ~400 mm),
  defines all five classes with worked examples, and settles the rule that
  causes most disagreement: `placement_area` is the floor the camera can
  *currently see* as free, not the floor that would be free if the bay were
  empty.
- **`recog/labelme_to_seg.py`** — LabelMe polygons to the same COCO-RLE sidecar
  format the synthetic pipeline writes, so real and synthetic data are
  interchangeable downstream. Fully offline; no photograph of anyone's hardware
  leaves the machine.
- **`recog/check_annotations.py`** — a validator that re-measures on human
  polygons the zero-overlap invariant Blender's index pass gives for free, and
  singles out the `placement_area`/`battery` pair by name because that is the
  mistake a human is most likely to make.

Both tools are under test (`tests/test_labelme_to_seg.py`,
`tests/test_check_annotations.py`; 82 % and 93 % branch coverage respectively).

The experiment is small and its result is publishable either way: render a
Mini part from its CAD, train on renders alone, and score against 50–100
annotated photographs of the real part in the real fixture. If it transfers,
that is a measured statement about frugal industrial AI on a facility's own
hardware. If it does not, that is a measured statement about where synthetic
data stops — which is worth more to a testbed whose business is de-risking
than another paper reporting only the case that worked.

I am not asking for anything to be taken on trust. Everything above is in the
repository with the command that produced it, including the figures that got
worse when they were measured properly.

---

### Receipt index

| Claim | Receipt |
|---|---|
| Detector mAP@0.50 0.9053, in-domain, 150 synthetic frames | `docs/receipts/detector_bench.txt` arm 3 |
| Detector mAP@0.50 0.8484, 6 of 7 handheld phone photographs | `docs/receipts/real_photo_eval.txt` |
| Bay IoU 0.8755 (procedural + crown) on 836-crop held-out CAD split | `docs/receipts/seg_eval_anchored_crown_on_cad_test.txt` |
| Bay IoU 0.6555 (procedural, no crown), same split | `docs/receipts/seg_eval_anchored_on_cad_test.txt` |
| Bay IoU 0.9032–0.9131, four CAD-trained leave-one-SKU-out controls, same split | `docs/receipts/seg_eval_cad_control_*_on_cad_test.txt` |
| Boundary displacement 1.226 mm, 3.14× margin; 16.6 vs 57.1 ms at 8 cartridges; per-frame GSD 0.4903–1.0739 mm/px | `docs/receipts/seg_eval.txt` |
| Detector CPU latency, median 437.4 ms | `docs/receipts/frcnn_latency.txt` |
| Forbidden-mask packing, +3.25 cells at 5 % coverage, paired t 15.18 | `docs/receipts/forbidden_bench.txt`, `forbidden_bench_seeds.txt` |
| Rotation ablation, +24.2 % / +57.1 % / 0.0 % | `docs/receipts/ffdh_ablation.txt` |
| 1,210 passed / 2 skipped; 93 % and 67 % branch coverage | `docs/receipts/pytest-cov.txt` |
| Same-seed bit-identical training | `docs/receipts/seed_reproducibility.txt` |
| τ retired — the fail budget never bound | `docs/receipts/tau_calibration.txt` |
| Real-photo placeable fraction 0.318 vs heuristic 0.217, n = 20 | `docs/receipts/seg_ablation.txt` |
| Bay 54.9 × 65.0 mm vs an 18.5 × 65.0 mm cell (geometry) | `recog/synth3d/assets/catalog.json`, `configs/planning.yaml` |
| 10 of 10 zero-placement instances on ground-truth masks | `docs/receipts/placement_feasibility.txt` |
| Eleven datasets, 6,018 scenes, ~8 GPU-hours | `docs/datasets/README.md` |
