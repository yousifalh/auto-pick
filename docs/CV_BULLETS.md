# CV and LinkedIn material — `auto-pick`

Yousif Al-Haidary, MEng, University of Nottingham. Every figure below is checked against a receipt in `docs/receipts/` or a spec in `docs/superpowers/specs/`, and is **synthetic-to-synthetic**: sim-to-real transfer is unvalidated and cannot be validated in this project (FDR §13.2.2), and there is no physical robot. Do not quote any of it as a deployment or real-world result.

---

## Project blurb (2–3 sentences)

### For an ML engineering audience

> **auto-pick — vision-guided robotic cell (MEng individual project).** Built a CAD-to-robot perception stack — Blender-rendered synthetic training data, Faster R-CNN detection, per-ROI semantic segmentation, occupancy-grid bin packing, KUKA protocol streaming — and then spent most of the effort measuring it. Diagnosed a 0.25 IoU generalisation gap to its mechanism (a missing shading cue in the training distribution) and closed it with a single change, cutting false positives from 21.8 % to 2.6 %; also published three null results, one against a numeric threshold registered in advance, and stopped the work when a ground-truth oracle showed further perception effort was worth zero net cells. All figures are synthetic-to-synthetic; sim-to-real transfer is stated as unvalidated.

### For a robotics / perception audience

> **auto-pick — vision-guided pick-and-place cell (MEng individual project).** A full perception → digital twin → 2-D packing → KUKA EthernetKRL pipeline for loading 18650 cells into protective cartridges, running end to end against a mock controller that speaks the real binary protocol. The engineering result is an operating envelope: on a 30-cartridge corpus the shipping system places 25 cells against a ground-truth oracle's 25, and one cartridge SKU was shown unplannable through the shipping planner at *any* perception accuracy because its bay is exactly one cell long. No physical robot; all perception figures are synthetic-to-synthetic.

### For a general software engineering audience

> **auto-pick — vision-guided robotic cell (MEng individual project, ~19k lines of Python plus a 752-test suite).** Designed and built a three-module pipeline with frozen dataclass contracts at every boundary, a torch-free demo path that runs the whole loop with no GPU or hardware, and tool-generated receipts for every published number. Found and fixed five silent defects — including a retired feature still live in code and a calibration constant that was wrong in three derived quantities at once — none of which any test had caught.

---

## Bullets — ML engineer

- Diagnosed a **0.25 IoU** generalisation gap to its mechanism rather than retraining: decomposition showed **91.4 %** of it was false positives on *closed* cartridges, not segmentation quality — on crops containing the target class the model was already within **0.021** of a ceiling model.
- Traced the cause to a measurable training-distribution gap (real lids carry an 11.10 mm crown; synthetic lids were flat, giving **10× less** internal shading structure) and confirmed it with a **monotone dose-response across five luminance-gradient quintiles, 6.4 % → 39.2 %**, stable within all seven lighting conditions.
- Closed it with **one config field**: hallucinated detections **21.8 % → 2.6 %**, pooled class IoU **0.6555 → 0.8755** against a same-data ceiling of **0.9009**, with a pre-registered falsifier ("the model just got shy") checked and not triggered.
- Ran and published **three negative results**, each comparison named before the run — a confidence gate whose IoU correlates with error in the **wrong direction in all 4 SKUs**; two training distributions separated by **0.0007**; and, the one carrying a numeric threshold registered in advance, a cell-format hypothesis pre-registered at ≥ 0.15 that recovered **7.6 %** of its target gap.
- Found **five defects that silently degraded output and that no test caught**, across a suite that grew **621 → 752** and never went red — including inverted CAD geometry in the render pipeline that had corrupted every label in the training set, requiring a full dataset regeneration and retrain from scratch.
- Built an **oracle evaluation** (ground-truth masks and boxes through the shipping code) showing the shipping system placed **25 cells against the oracle's 25** on a 30-cartridge corpus, and used it to stop further model work on evidence rather than intuition — then caught that the two sides had been measured at different commits, re-ran the oracle, and published the corrected figure.
- Enforced a hard latency budget by architecture: batched per-ROI inference at **21.2 ms for 8 crops** against **88.0 ms** unbatched, keeping planning at **~2 ms/cartridge** inside an 8 ms budget.
- Stated the **limits in the artefact itself**: every figure labelled synthetic-to-synthetic, with a dedicated report section explaining why sim-to-real transfer cannot be measured under the project's constraints.

## Bullets — robotics / perception

- Built a complete CAD-to-robot pipeline — Blender/Cycles synthetic data, Faster R-CNN detection, per-ROI DeepLabv3 segmentation, occupancy-grid packing, **16-byte EthernetKRL framing with CRC-16/MODBUS** — running end to end against a mock KUKA controller.
- Established the system's **operating envelope by measurement**: against a ground-truth oracle the shipping pipeline places **25 cells across 12 of 30 cartridges** versus the oracle's **25 across 11** — perfect perception is worth zero net cells on this corpus.
- Showed one cartridge SKU **unplannable at any perception accuracy** through the shipping planner: its bay is exactly the nominal cell length (65.0 vs 65.0 mm) and an axis-aligned packer loses **0.32 mm per degree** against ±2° jig jitter, so it takes **zero cells in 10 of 10** ground-truth instances through the shipping extractor — a specification finding, not an accuracy one.
- Fixed a calibration defect where `mm_per_px` was a constant while the camera framing varied per frame, under-reading the scene by **27 % at the median** and, on 6 of 30 instances, reserving a footprint **smaller than the cell being placed**; moved scale onto the per-frame contract so extractor and planner agree by construction.
- Cut unsafe commanded placements from **5 of 26 (worst 100 %) to 2 of 25 (worst 8.3 %)** with two changes — rejecting detector boxes whose placeable region exceeds the largest cataloged cartridge (a physical bound needing no tuned tolerance), and calling an occupancy cell free only when *all* of it is free — at a cost of exactly **one** cell, the one being driven into bare backdrop.
- Swept and **rejected** the obvious safety fix on evidence: a clearance margin costs up to **12 cells** and at 1.5 mm *creates* a third overlap; a free-floor guard rejects nothing, because 4 of 5 offending placements sit **100.0 %** inside the predicted free floor.
- Diagnosed a strip packer that placed **zero cells into a 93 %-free grid** because its shelf origin never scanned in y; fixed by competing three arms and taking the maximum, so **no instance can regress by construction** (8 → 17 cells on real frames; 1.9 ms on the worst real cartridge and 3.4 ms mean on oversized bench masks, against an 8 ms budget).
- Documented the limits first: **no physical robot** (protocol validated against a mock only) and **sim-to-real transfer unvalidated and unvalidatable** under the project's constraints.

## Bullets — general software engineering

- Delivered a **752-test**, three-module Python system with frozen dataclass contracts at every boundary and a **torch-free demo path** that runs the full loop with no GPU, camera or controller.
- Found and fixed **five defects that degraded output silently and that no test caught**, while the suite grew from **621 to 752 tests** and never went red — including a feature documented as retired but still live in code, holding three mutually inconsistent values across three files, whose only test asserted the retired behaviour.
- That retired gate was **rejecting 8 of 8** valid inputs at the project's own configured scale; deleted the constructor argument outright so any caller still passing it raises `TypeError` rather than being silently ignored.
- Discovered the trained model was **unreachable from the end-to-end entry point under any configuration**, wired it in, and corrected the overstated claim in the public README rather than quietly shipping the fix.
- Replaced a config constant with a per-frame value carried on the data contract, eliminating a hand-maintained coupling that had made **one wrong scalar wrong three times in the same direction**.
- Made correctness cheaper than optimism: replacing a centre-pixel occupancy test with a whole-cell test over a summed-area table was both **safer and faster** (2.97 ms vs 4.25 ms), and a latency test caught the naive 15.1 ms version before it shipped.
- Built a **receipts discipline** — every published number regenerated by committed tooling, never hand-edited, with superseded values left visible and named rather than overwritten.

---

## Notes for adapting these

- **Never drop the limits.** "Synthetic-to-synthetic", "no physical robot" and "12 of 30 is a ceiling, not a shortfall" are load-bearing. An interviewer who finds them after you have implied otherwise will discount everything else.
- **12 of 30 / 25 cells is not a low score.** Present it against the oracle's 11 / 25, or it reads as a failure. The remaining 18 cartridges are sealed, genuinely occupied, or physically unable to accept a cell.
- **Know the oracle's caveats before you are asked.** Net zero is four cells moving each way, not agreement: the two sets share nine cartridges, shipping places into three ground truth refuses and misses two it allows. The oracle runs with the wall inset at zero, which is a planner relaxation shipping does not get — give it the shipping 4.25 mm and it falls to 24 cells in 10 cartridges, *below* what ships. Have the explanation ready: the predicted bay is more permissive than the true one (≈1.3 mm of boundary displacement, 51.5 mm² of optimistic area per crop), so some cells are placed on floor that is really tray wall — the same defect as the 2 residual unsafe placements, counted from the other end. Say "zero net cells" or "at the ceiling"; never a percentage, and never "the system beats ground truth".
- **The oracle figure was 27 until 2026-08-12, and how it moved is worth telling.** The two sides had been measured at different commits — the oracle under a centre-pixel occupancy rule, shipping under the stricter whole-cell rule that replaced it. Re-running the oracle at the shipping commit cost it two cells. Volunteering that is stronger than the number: it is a same-code-state comparison now, and the correction went against the direction that would have flattered the system.
- **The lid-crown result is a mechanism result, not a transfer result.** The crown range was chosen after measuring real lids. Say "a measured coverage gap was the mechanism"; never "procedural training transfers".
- **Do not upgrade the nulls to pre-registration.** Only the cell-format null had a numeric threshold (≥ 0.15) fixed before the run. The `tau` criterion was directional and anchored-vs-wide was a comparison named in advance with no effect size attached. "Named before the run" is the accurate phrase for all three; "pre-registered" is accurate for one.
- **"No test caught them" is the claim, not "the tests held".** The suite never went red because the regression tests arrived with each fix; twice it was green because it asserted the defect. Volunteer this — an interviewer who works it out first will read the original phrasing as spin.
- Sources: README, `docs/PORTFOLIO.md`, `docs/FDR_v3.md` §3 / §13.1.1 / §13.2.1 / §13.2.2, and `docs/superpowers/specs/README.md` for the diagnosis trail.
