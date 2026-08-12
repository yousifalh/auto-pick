# What these documents are

This directory is the **working record** of the project's design and measurement work: one document per investigation, written at the time, kept as evidence rather than tidied afterwards. They are not tutorials and they are not the report — `docs/FDR_v3.md` is the report. These are what the report's numbers were derived from.

Read them if you want to see how a claim in the README or the FDR was arrived at, or what was tried and did not work.

## The shape of a document here

Most files follow the same trail:

**diagnosis → hypothesis with a threshold stated before the run → experiment → result, including when the result is null.**

Each one opens with its **baseline commit and test count**, states what it was and was not allowed to change, and names its receipt in `docs/receipts/`. Where a figure elsewhere in the repository is superseded, the document says which figure, by how much, and why — superseded values are left visible and named, not overwritten. Where a document turned out to be wrong, the correction is recorded in the successor document rather than by editing the original: `2026-08-11-placement-feasibility.md` §2 explicitly retracts a sentence from `2026-08-11-packing-ceiling.md` §3, and `2026-08-11-placement-safety.md` §2 records that the scale-calibration spec was right about the cause and wrong about the fix.

Negative results are kept at the same prominence as positive ones. Three of the investigations below concluded that the thing being tested does not work, and that is what they say.

## Index

### The 2026-08-11 measurement sequence

These six are the most self-contained, and are the ones cited from the README. Each acts on the one before it.

| Document | Question | Result |
|---|---|---|
| `2026-08-11-transfer-gap-diagnosis.md` | Why does a segmenter trained on procedural trays trail a CAD-trained control by 0.25 IoU? | It does not, on crops that contain the target class — it is within 0.021. **91.4 %** of the gap is false positives on *closed* cartridges. A cell-format hypothesis, pre-registered at ≥ 0.15, recovered **7.6 %** and is reported as null. |
| `2026-08-11-sealed-unit-experiment.md` | Is the missing shading structure on flat synthetic lids the mechanism? | Yes, with a monotone dose-response across five gradient quintiles. One config field takes false positives **21.8 % → 2.6 %** and pooled `bay` **0.6555 → 0.8755**. Includes six checks run *because* the result was favourable, one of which found a defect in the experiment's own tooling. |
| `2026-08-11-packing-ceiling.md` | Why did the packer place zero cells into a 93 %-free grid? | FFDH's shelf origin never scans in y. Fixed by competing three arms and taking the maximum, so no instance can regress by construction. |
| `2026-08-11-placement-feasibility.md` | Are the 23 cartridges that receive no cell a correct refusal? | No — the claim they were "too full" does not survive contact with ground truth; nothing in the corpus is more than half full. The dominant cause is `mm_per_px`. Underneath it, one SKU is **unplannable with perfect perception**. |
| `2026-08-11-scale-calibration.md` | Move scale onto the per-frame data contract. | 7 → 13 productive instances, 17 → 26 cells. **Its own acceptance criterion is not met**, and §4 says so before any result is quoted. |
| `2026-08-11-placement-safety.md` | Can the residual unsafe placements be caught inside the planner? | Two guards ship (5 of 26 → 2 of 25 overlaps, worst 100 % → 8.3 %). Two more were measured and **rejected**: one rejects nothing, the other costs up to 12 cells and creates a third overlap. |

Two supporting records from the same week: `2026-08-11-segmenter-integration.md` (the retired confidence gate that was still live in code, and wiring the segmenter into the end-to-end loop), `2026-08-11-scale-figures.md` (correcting every published millimetre figure that had been converted at a nominal scale no frame was rendered at), and `2026-08-11-doc-reconciliation.md` (six things that had become true in the code while remaining wrong in the documentation).

### Earlier design specs

Written before the work they describe, and kept as the record of what was decided and why.

- `2026-08-05-blender-synthetic-dataset-design.md` — the Blender/Cycles generator: CAD import, material and lighting randomisation, and deriving pixel-exact boxes from the object-index pass.
- `2026-08-06-scale-overlap-selection-design.md` — scale variety, overlapping parts, and honest checkpoint selection.
- `2026-08-06-segmentation-placement-area-design.md` — the per-ROI segmenter and the placement-area extractor; supersedes the FDR's original Mask R-CNN proposal, with the resolution argument that motivated the change.
- `2026-08-08-tray-interior-design.md` — modelling the open cartridge interior.
- `2026-08-09-spec3-realism-decisions.md`, `2026-08-10-generalisation-decisions.md` — decisions recorded *before* the corresponding spec was written, so the reasoning was not re-derived after seeing results.
- `2026-08-10-generalisation-groundwork.md`, `2026-08-10-generalisation-design.md` — the procedural-tray generalisation study: train on synthetic trays, test on real measured CAD. §12 states the comparisons in advance, "so none of them get chosen after seeing a result".
- `2026-08-10-tau-difficulty-design.md` — why the arbitration confidence threshold was not calibratable, including the algebraic argument that its two "independent" estimates are one `argmax` read twice.

## One caveat on all of it

**Every performance figure in this directory is synthetic-to-synthetic** — measured on renders, against ground truth derived from the same renders. Real photographs of this project's cells and cartridges are not obtainable, so sim-to-real transfer is unvalidated and cannot be validated here. The dedicated statement is `docs/FDR_v3.md` §13.2.2, and each document repeats it.
