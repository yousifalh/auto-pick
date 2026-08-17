# Autonomous Recognition, Pick and Place

> ### One of the four cartridges cannot be automated by a vision system.
>
> Not by this one — by any of them. The bay is **65.0 mm**; the cell is **65.0 mm**. Fed perfect
> masks, perfect boxes and true scale, it places **zero cells in 47 of 47 instances**.
>
> **A bay packed to exact tolerance cannot be certified by a vision system: certification needs
> margin to absorb measurement error, and an exact fit offers none.**
>
> A design constraint, not an accuracy problem.
> **[The derivation ↓](#a-cartridge-that-cannot-be-certified)**

**A vision-guided robotic cell that picks loose 18650 / 21700 lithium-ion cells from a camera view and places them into protective cartridges — perception, digital twin, 2-D bin packing and KUKA command streaming, running end to end in software.**

MEng Individual Project — Yousif Al-Haidary, University of Nottingham, supervised by Professor Svetan Ratchev FREng.

**What this repository is for.** Many people have built a CAD-to-robot perception stack. The part
worth reading is the measurement discipline: three pre-registered predictions came out null and are
published as null, one success criterion is recorded as a **Fail**, and one headline figure is
currently **withdrawn** rather than restated.

```
Camera ─▶ Recognition ─▶ Planning ─▶ Execution ─▶ KUKA controller
         (Faster R-CNN   (digital   (EthernetKRL 3.1,
          + per-cartridge  twin +    CRC-16/MODBUS)
          bay segmenter)  packing)
```

Training data is path-traced in Blender from real measured CAD. The robot is a mock that speaks the real KUKA binary protocol, so `python main.py --config configs/demo.yaml` runs the whole loop with no GPU, no camera and no controller.

![Block diagram of the pipeline. An RGB frame enters Recognition, which holds recog/inference.py (Faster R-CNN, ResNet-34 FPN), recog/bay_segmenter.py (DeepLabv3 + MobileNetV3-Large, run per ROI and batched once per frame) and recog/calibration.py. Recognition emits a Snapshot — detections, masks and mm_per_px — to Planning, which holds plan/scene.py, plan/placement_area.py, plan/arbitration.py and common/packing.py's pack_best_effort. Planning emits a PickPlacePose to Execution, which holds execution/protocol.py, execution/execution.py and execution/mock_kuka_server.py, and streams it over TCP as EthernetKRL 3.1. A RobotStatus arrow returns from Execution to Planning each cycle. The real KUKA KR 6 R700 is drawn in a dashed box marked withdrawn.](docs/figures/fig11_architecture.png)

**The shipping architecture, drawn from the code as of 2026-08-12.** Nothing on it is measured: the only quantity is the 8 ms per-cartridge planning budget, which is a requirement (FDR O3), not a result. It is drawn fresh rather than reused because `docs/figures/fig1_architecture.png`, the version in the submitted FDR, predates both the segmenter's arrival in Recognition and the planner's move off bare FFDH — see `docs/superpowers/specs/2026-08-12-figures-audit.md`.

![Six panels in two rows of three. The top row shows three path-traced synthetic frames cropped to their detections, with the Faster R-CNN's boxes drawn on: cyan boxes labelled cartridge with confidences 0.92 to 1.00, and magenta boxes on loose cylindrical cells. The middle frame shows one large open cartridge beside eight loose cells in a row, all eight boxed. The bottom row shows the same three frames with the per-ROI segmenter's six-class label map painted back into each detected cartridge: blue for the cartridge shell, green for the open bay, orange for the electronics module and red for a small obstruction lying in the bay. The middle cartridge's bay is a large unbroken green region; the two closed cartridges in the outer frames are painted flat blue with no bay at all.](docs/figures/fig12_detections.png)

**Both recognition stages on the same frames.** Top: the Faster R-CNN. Bottom: the per-ROI
segmenter's label map composited back into each detected cartridge — green is the `bay` the packer
is allowed to use, red is an obstruction lying in it. Note the two *closed* cartridges, painted flat
blue with no bay: refusing to hallucinate a bay onto a sealed lid is the failure mode that
[the crown work ↓](#one-gap-diagnosed-to-mechanism-and-closed) was about.

The detector trained on `recog/dataset3d` and the segmenter on `recog/dataset3d_seg`; these frames
come from `recog/dataset3d_seg_cad_test`, so neither model was fitted to them. **The frames are
selected, and the rule is mechanical**: the first three in sorted filename order carrying a detected
cartridge with ≥ 2,000 predicted `bay` pixels — 3 of the first 7 scanned qualified. The crop is a
display choice; an 18 mm cell is ~25 px in a full 1280×720 frame. **These are renders, not
photographs**, and nothing here is a measurement — the numbers are in
[`docs/receipts/`](docs/receipts/). Regenerate with `python scripts/figure_detections.py`.

**Start here** — 15 minutes, no GPU:

1. The story, in 5 minutes: [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md)
2. The models — what they are, what they score, where they fail: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). Nine checkpoints in one comparison table, every figure generated from a receipt, with the failure modes and the scope limits in the same document. The datasets behind them: [`docs/datasets/`](docs/datasets/README.md)
3. Run it, in 2: `pip install -e . && python -m recog.synth_dataset --out recog/dataset --n 50 && python main.py --config configs/demo.yaml` — output pinned in [`docs/receipts/main_run.txt`](docs/receipts/main_run.txt)
4. How every number was arrived at: [`docs/superpowers/specs/README.md`](docs/superpowers/specs/README.md) — one document per investigation, written at the time. The adversarial reviews those answer are in [`docs/superpowers/audit/`](docs/superpowers/audit/)
5. The report, and what is authoritative vs. history: [`docs/FDR_v3.md`](docs/FDR_v3.md) is the current revision — start at §13.2.2 for what this project can and cannot claim. [`docs/README.md`](docs/README.md) indexes it and everything else in `docs/`, and says which documents are superseded

---

## The headline result, and what it is not

Feeding the shipping extractor and the shipping packer **ground-truth masks and ground-truth boxes** at each frame's true scale — an oracle with perception removed from the question — puts at least one cell into **9 of 29 open cartridges and places 23 cells**, and it places that same 9 / 23 at *every* wall inset from the production 4.25 mm down to 0.00 mm. That is the oracle side of the ceiling analysis, and it is now regenerable: `python scripts/placement_feasibility.py` writes [`docs/receipts/placement_feasibility.txt`](docs/receipts/placement_feasibility.txt), whose §3 sweep is the source of both figures, and `--check` fails on drift.

**Both of those figures were revised downward on 2026-08-15**, and by two separate things that should not be run together: a real geometry defect in `Planner._pack_cartridge`'s packing strip, and — unrelatedly — a change of denominator from 30 detector-matched instances to 29 ground-truth open cartridges. What they read before and the full mechanism are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md); `placement_feasibility.txt` §5 reconciles the two censuses line by line, and `docs/superpowers/specs/2026-08-11-placement-feasibility.md` §3, §5 and its 2026-08-12 addendum are superseded on these totals.

**The shipping-pipeline half of the comparison — 12 of 30 cartridges and 25 cells — is withdrawn pending re-measurement, not restated.** It never had a regenerable receipt: it comes from a results table in `docs/superpowers/specs/2026-08-11-placement-safety.md` §4, and re-deriving it needs the detector-plus-segmenter instance-pairing pass re-run at HEAD. Every packing figure adjacent to it moved when the strip was fixed, so the sentence this README carried — *perfect perception is worth zero net cells on this corpus* — cannot be quoted at numbers measured on the old strip and is not quoted here. What survives it, and is the transferable half, is the geometric result below: one SKU cannot accept a cell at any perception accuracy, and that is unchanged and receipt-backed.

**If the shipping figure does survive re-measurement, a shipping system at or above its own ground-truth oracle still needs explaining, and the explanation is not skill.** The predicted bay is slightly *more permissive* than the true one, so some cells are placed on floor that ground truth calls wall: `docs/receipts/seg_eval.txt` measures the segmenter's bay-boundary displacement at **1.226 mm** (over the 35 bay-carrying crops the boundary row scores) and its placeable-area error as optimistic by **79.2 mm²** per crop (over all 126 validation crops). Both of those figures are unchanged — the geometry fix touched the packer's strip, not the segmenter, and `seg_eval.txt` regenerated byte-identical on every non-timing row. Both are, however, 2026-08-14 revisions of a smaller pair computed against the generator's *nominal* mm/px rather than each frame's own — [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md), and FDR §13.2.1 at greater length. That is the same defect as the two residual unsafe placements below — 8.3 % and 5.2 % of a footprint on one cartridge's left tray wall — read from the other end. The reading this README used to draw from it, that **perception has stopped being the binding constraint on this corpus**, goes back to being a hypothesis until the shipping side is re-measured: it was a statement about a *difference* between two numbers, and only one of the two has been re-run at HEAD.

Four limits, stated here rather than buried:

* **Sim-to-real transfer is unvalidated and cannot be validated in this project.** Real photographs of these cells and cartridges are not obtainable (owner-confirmed, 2026-08-09). Every segmentation, placement-area and packing figure in this repository is **synthetic-to-synthetic** — measured on renders, against ground truth derived from the same renders. This is not "not yet measured"; the data that would settle it will not be collected. The dedicated limitation statement is FDR §13.2.2.
* **There is no physical robot.** The KR 6 R700 was withdrawn. Execution is `execution/mock_kuka_server.py`, which implements the real 16-byte framing and CRC-16/MODBUS, so the protocol code is exercised but is unvalidated against hardware (FDR §10.3).
* **Whatever the shipping pipeline scores on this corpus, it is a ceiling and not a shortfall.** One SKU — a third of the corpus by instance count — cannot accept a cell at *any* perception accuracy through the shipping extractor, and a second clears by 1.75 mm. That is a property of the CAD and the nominal footprint, so it does not move when a packing figure does. See "A cartridge that cannot be certified" below.
* **Nothing here is deployed.**

---

## Running it

```bash
pip install -e .                                            # 1. install
python -m recog.synth_dataset --out recog/dataset --n 50    # 2. make a few frames
python main.py --config configs/demo.yaml                   # 3. run the loop
```

That path is torch-free by design and is what the reproducibility claim rests on. The loop logs per-cycle perception / planning latencies, cartridge / mask / queue counts, and a placed / pick-failed / place-failed summary at exit.

**What that run does and does not promise.** Those three commands produce, deterministically, **1 cycle, 3 cartridge detections, 5 loose cells, 3 placement areas, 3 queued poses, 2 released reservations, 0 placement disagreements, 0 bad detector boxes** — byte-identical across five consecutive runs on a fresh clone, because `recog/synth_dataset.py` is seeded (`--seed`, default `0`) and every reader therefore gets the same 50 frames. The full summary is pinned in [`docs/receipts/main_run.txt`](docs/receipts/main_run.txt), which `python main.py --config configs/demo.yaml --receipt docs/receipts/main_run.txt` regenerates. The run ends at cycle 1 of `max_cycles: 10` because `stop_on_empty_queue` defaults to `True` and the second frame yields no executable pose — the demo's designed exit, not a fault. So what is promised is the *shape* of a cycle — detection → digital twin → packing → reachability → CRC-framed TCP round trip to the mock controller — plus the latency profile and "it does not crash". **It is a wiring demonstration on three green rectangles, not a result**; every quantitative claim in this repository comes from the trained models, which need a checkpoint this repository does not ship. The placed / pick-failed split is not promised either: `execution/mock_kuka_server.py` injects a deliberate simulated vacuum failure at `simulation.drop_probability: 0.02` from an unseeded RNG, so a cycle fails with probability 0.0298 and roughly one run in thirty-four reports `pick_failed: 1` instead of the `placed: 1` above.

**The counts above are the heuristic detector's, and an earlier edition of this section published a trained detector's instead.** `recog/checkpoints/` is gitignored and **no `.pt` is tracked anywhere in this repository**, so on any clone `recog.inference.load_detector` logs `No checkpoint at recog/checkpoints/best.pt (or torch unavailable). Using HeuristicDetector fallback` and the pure-OpenCV detector runs — which is exactly what "torch-free by design" means. The receipt names `detector : HeuristicDetector` on its eighth line so the two paths can never again be confused for one another. The withdrawn counts are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md); diagnosis in `docs/superpowers/specs/2026-08-12-fix-first-impression.md`.

**The workspace envelope is enforced, and the demo counts what it declines.** `planning.camera.workspace_bounds_mm` was once parsed and compared against nothing, so the loop commanded pick and place points beyond the arm's declared ±350 mm envelope. It is enforced now, and because a fixed-mount camera legitimately sees further than an arm reaches, an out-of-reach candidate is skipped and **counted** rather than aborting the run: the run above declines 4 place targets, 2 loose cells and 1 whole cartridge, and says so in both the run summary and the receipt. On the trained-detector path the same fix took queue poses from 62 to 41. Detail: `docs/superpowers/specs/2026-08-12-fix-demo-workspace.md`.

**Reproducing the trained results costs about 8 GPU-hours and will not return these exact numbers — the published checkpoints were trained before seeding existed.** Until 2026-08-12 training in this repository was genuinely unseeded: no `torch.manual_seed`, no `use_deterministic_algorithms`, and `DataLoader(shuffle=True)` with no `generator=`, so model initialisation and epoch order differed per process. Two one-epoch runs of the same command on the same data gave selected mean IoU 0.4111 and 0.3957. Every checkpoint here comes from that era and **cannot be recovered**, so a from-scratch reproduction of a published figure returns *a sample from the same distribution*, and a number that comes back 0.01 off is not a discrepancy.

**Training is seeded now, and the seeding has its own receipt.** `recog/seeding.py` fixes Python's `random`, NumPy, torch on CPU and CUDA, the DataLoader `generator` and `worker_init_fn`, the albumentations pipeline and the crop jitter, from `training.seed` (default 20260812, `--seed` overrides); the resolved seed is logged and written into every checkpoint, and any step that could silently fail to seed raises instead. Seeding the RNGs was **not enough on its own** — with the kernels unconstrained, two same-seed runs still diverged into 0.0197 / 0.0076 of mean loss and 0.0101 / 0.0409 of selected IoU — so `training.deterministic` defaults to `warn` (`cudnn.deterministic` + `use_deterministic_algorithms(warn_only=True)`), under which two runs at one seed produce **bit-identical weights**: 1.7619 / 0.3817 twice at seed 20260812 (both hashing to `192d868597ab3ad1`), against 1.8099 / 0.3330 at seed 20260813. (All four of those numerics moved on 2026-08-15, when `recog/seg_dataset.py::_rng_for_worker` began mixing `torch.initial_seed()` into the augmentation stream; the property this receipt exists to certify is untouched, and the superseded values are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).) `strict` raises rather than trains — `nll_loss2d_forward_out_cuda_template` has no deterministic implementation. The honest claim is **reproducible on the same machine and toolchain**, not reproducible unqualified: a different GPU, driver, cuDNN or torch build can still move the arithmetic. Receipt: `docs/receipts/seed_reproducibility.txt`, regenerated by `python scripts/seed_check.py` in about 90 seconds.

What *is* exact is the evaluation: given the checkpoints and datasets, the eleven `seg_eval*.txt` receipts regenerate byte-identically on every metric. And the comparisons the conclusions rest on are between models trained under identical conditions — **all nine `segmentation*.yaml` training configs are byte-identical apart from three path values** (`dataset.coco_path`, `dataset.img_dir` and `training.checkpoint_dir`), verified mechanically by stripping comments and normalising those three keys — which is what makes them robust to the missing seeds. **Nine is the number this claim needs**: it is exactly the set of models the comparisons span — one per row of the model card's training-data table, one per committed segmenter checkpoint — so no compared pair differs in architecture, schedule, augmentation, split seed, crop jitter or `training.seed`.

**Nine, eleven and eight are all correct under their own scoping, which is why a wrong count survived here for a while.** The glob itself matches **11** files; two are not training configs and say so in their own headers — `configs/segmentation_cad_test.yaml` is an evaluation config at `train_val_split: 0.0`, and `configs/segmentation_seedcheck.yaml` is a one-epoch seeding probe. The **eight** that FDR Appendix C names is the *generalisation* subset, excluding `configs/segmentation_anchored_crown.yaml`, which belongs to the sealed-unit experiment and is byte-identical to the rest on the same three keys anyway. So: 11 files, 9 training configs, 9 in the byte-identical set, 8 of them generalisation runs. This paragraph read "ten" until 2026-08-14 — [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md). Detail: FDR Appendix C.

### The same loop with the trained segmenter in it

`configs/demo.yaml` runs the **heuristic** placement-area extractor and no segmenter — that is what keeps it torch-free, and it is the only thing it claims. `configs/demo_seg.yaml` is the other half: it puts the trained bay segmenter into recognition as a second stage and plans from its label maps.

```bash
python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt
```

The `mode.segmentation` block selects that path, and it selects it in both places at once — the detector gets the segmenter, the planner gets `SegmentationPlacementAreaExtractor`. They cannot be configured apart, because either half alone is silent: a segmenter with no consumer runs for nothing, and the segmentation extractor with no segmenter raises on every cartridge into a blanket `except Exception`. For the same reason a missing checkpoint raises instead of falling back, and a run that completes having produced **zero** placement areas is treated as a failed run, not a quiet one.

The frames matter as much as the checkpoint. `demo_seg.yaml` reads Blender renders (`recog/dataset3d_seg`), because the segmenter predicts no `bay` at all on `synth_dataset.py`'s flat green rectangles — pointed at those, it would complete cleanly and demonstrate nothing. Last generated run (`docs/receipts/main_seg_run.txt`, 2026-08-12): 15 frames, 26 cartridges detected, 26 segmented, **7 placement areas**, 4 poses queued, 1 pick-and-place executed, **1 detector box rejected** as not describing a single cartridge, and 15 of 15 frames carrying their own scale. Note the pick count: the loop executes at most one pose per cycle and a pose also needs a loose battery in the same frame — see `docs/superpowers/specs/2026-08-11-segmenter-integration.md` for the measurement. **And note what the arm declined**: 57 of the 78 loose cells, 2 place targets and 1 whole cartridge lay outside `planning.camera.workspace_bounds_mm`, because these renders span up to 1338 × 752 mm against a 700 × 700 mm envelope. A camera legitimately sees further than an arm reaches; the planner skips those candidates and reports the count rather than aborting, and it used to abort — this receipt could not be regenerated at all between 2026-08-11 and 2026-08-12 (`docs/superpowers/specs/2026-08-12-fix-demo-workspace.md`). The earlier *6 / 2* figures that fix retired are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md). Those frames are the segmenter's own training corpus, so this receipt is evidence that the **wiring** works, not a generalisation measurement. Held-out numbers live in `docs/receipts/seg_eval_*_on_cad_test.txt` and FDR §13.1.1.

![Fifteen panels in three rows of five. The top row shows five 256-pixel synthetic camera crops of open power-bank cases under varied lighting, one of them very dark. The middle row shows the segmenter's predicted six-class label map for each crop; the bottom row shows the corresponding ground truth. Predicted and ground-truth maps agree closely on the large green bay region and on the orange electronics module, with the predicted bay running slightly wide into the blue cartridge wall and the predicted electronics boundary rounded where ground truth is square. Per-crop bay IoU runs 0.900 to 0.979.](docs/figures/fig10_segmenter.png)

**What the segmenter actually produces, on crops it never trained on — and these are renders, not photographs.** Middle row is the prediction, bottom row the ground truth. Measured on the 126-crop validation split of `recog/dataset3d_seg`, held out by the seeded 0.85/0.15 split that `recog/checkpoints/seg/best.pt` — the checkpoint `configs/demo_seg.yaml` loads — was selected against; `recog.seg_evaluate`'s split guard confirms the partition matches the one recorded in the checkpoint. The five shown are the first five bay-carrying crops in split order, **not hand-picked**, and they are an illustration rather than a summary: pooled `bay` IoU over all 36 bay-carrying validation crops is **0.8903**, with mean boundary displacement **1.226 mm** over **35** of those 36 (`docs/receipts/seg_eval.txt`:20 and :37). The two figures have different denominators: 36 is the IoU instance count, and the boundary row scores 35, because displacement is only defined on a crop where a predicted `bay` boundary exists to measure. The predicted bay running slightly wide at the walls is the same optimistic-boundary defect the headline section above is about, seen directly. **Synthetic-to-synthetic**: Blender/Cycles renders of the measured Anker CAD, scored against ground truth derived from the same renders. Nothing here supports a sim-to-real claim.

---

## How it works

A strict sequential flow between three loosely-coupled modules, with frozen dataclass contracts at every boundary. Each arrow is a well-defined type: `Snapshot` from recognition, `PickPlacePose` from planning, `RobotStatus` back from execution.

* **Recognition** turns pixels into a `Snapshot`. A Faster R-CNN (ResNet-34 FPN) finds cartridges and loose cells; a per-ROI DeepLabv3 + MobileNetV3-Large segmenter then labels each cartridge crop into six classes (`bay`, `battery`, `electronics`, `obstruction`, `cartridge`, background). Segmentation runs **once per frame, batched**, because the 8 ms per-cartridge planning budget cannot absorb a forward pass — 16.2 ms for 8 crops batched against 58.6 ms for the same 8 in a loop (`docs/receipts/seg_eval.txt`).
* **Planning** maintains a digital twin (`plan/scene.py`), converts each cartridge's label map into a placeable region, rasterises it to an occupancy grid, and packs 18.5 × 65.0 mm cell footprints into it. Planning does mask arithmetic only, measured at 2.0–2.2 ms per cartridge.
* **Execution** streams poses over TCP as 16-byte EthernetKRL packets with CRC-16/MODBUS, with retries and a host-initiated E-stop. The frame layout and the CRC were verified on 2026-08-12 against three independent CRC implementations over 20 000 vectors, with zero mismatches. **There is no heartbeat** — this line claimed one until 2026-08-12 ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md)), and `OpCode.HEARTBEAT` exists only as an enum value and a simulator dispatch arm; nothing sends one, and neither end runs a watchdog. FDR §7.5 states what that costs: the E-stop covers a robot that misbehaves and a link that degrades, and not a host that stops running.

Scale is a property of the **frame**, not of a config file: `Snapshot.mm_per_px` is filled by the image source and resolved once per cycle, so the extractor and the planner agree by construction rather than by convention. An uncalibrated frame with no configured fallback raises `UnknownScale` rather than reverting to a constant.

The full rationale and trade-off analysis is `docs/FDR_v3.md` — that is the current revision, and **every `§` reference in this README is to it** unless it is prefixed by another document's name (`PPR`, or a spec filename). `docs/FDR.md` and `docs/FDR_v2.md` are superseded earlier drafts, kept for history.

---

## What this project is actually about

Many people have built a CAD-to-robot perception stack. The part of this repository worth a reader's time is the measurement discipline around it — what was checked, what was reported as null, and what was stopped.

### Five defects, and none of them was the model

Each of these degraded output silently, and **no test caught any of them.** That is the point of the section, and it is worth stating exactly rather than flatteringly: the suite grew from **621 to 752 tests** over defects 2–5 (defect 1 predates the 621 anchor and was fixed at 533–570 tests) and never went red, because for each defect the tests that would have caught it did not exist until the fix shipped with them. Twice the green suite was *asserting the defect* — one test required the retired `tau` gate to fire, another asserted that the run receipt printed the hardcoded `mm_per_px`; both were deleted by the fix. Every one of the five was found by running the pipeline, rendering frames, or measuring against ground truth.

**And it has since happened five more times, which is the honest way to leave that number.** Six adversarial audits on 2026-08-12 (`docs/superpowers/audit/`) found five further tests that pinned the behaviour they were supposed to guard: `test_cycle_marks_cells_planned` asserted `planned_count() == len(queue)`, i.e. exactly one occupancy cell per 13 × 44-cell battery; two `confirm_placement` tests checked the anchor cell only, and passed just as happily when the other 571 cells stayed PLANNED for ever; and two `ExecutionConfig` tests asserted values of `approach_height_mm`, a field no client method reads. **One** is still standing on purpose: `tests/test_bay.py` pins "zero seated cells is fine". (It said "two" until 2026-08-15, and named a second file that had since been inverted into a real guard — [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).) The suite collects **1,397** tests now, **1,396** of them passing and one skipped for `bpy` (at HEAD with torch installed; it was 814 when this paragraph was written, 1,074 and then 1,211 on 2026-08-12, 1,221 before the 2026-08-14 execution-layer work, and 1,277 as this line last read it — a figure already stale by six at the start of the 2026-08-15 audit, which measured 1,283 before its own fixes added the rest), and the pattern of it going green over a defect is the recurring one in this project, not a solved problem. Reproduce the total with `python -m pytest -p no:cacheprovider` and read the summary line; do **not** append `-q`, because `pyproject.toml`'s `addopts` already carries one and the resulting `-qq` suppresses both the collected count and the summary. **The torch-free figure is unmeasured at this commit and is deliberately not restated here.** It last read "1,228 of 1,276 pass on `pip install -e ".[dev]"`, 20 skipped"; the 2026-08-15 fixes added tests on both sides of the torch boundary and the clean-install run has not been re-done, so the pair is marked for re-measurement rather than scaled by guesswork. What is unchanged is the *shape* of the skip set, which is by design and not a thinner run that went unnoticed: the torch-free skips are individual `importorskip` tests plus `tests/test_bay_segmenter.py`, which calls `pytest.importorskip("torch")` at module level and so takes its whole file out of collection, and two more that need `bpy` and the `.[cad]` STEP converter.

| # | Defect | Effect | Fixed at |
|---|---|---|---|
| 1 | Blender's glTF importer maps `(x,y,z) → (x,−z,y)`; this CAD's up-axis is Y with the cavity opening toward −Y. `lay_flat` chose *which* axis was vertical but had no notion of which **end** was up. | Every `open_case` cartridge rendered **upside down and closed** — the electronics module and the bay plane painted on the outside of a lid. Proof was exact: the shell measured z ∈ [11.1, 22.2] mm against the CAD's [0, 11.1] mm, a mirror about the lid's own mid-plane. Dataset deleted and re-rendered, checkpoint retrained from scratch. | `a31ac28`..`043e92d` |
| 2 | A confidence gate (`tau`) documented as retired, still live in `plan/placement_area.py`, with three mutually inconsistent values in three files. | At `configs/planning.yaml`'s own `mm_per_px` of 0.38 the wider erosion pushed every observed IoU (0.639–0.848) below the 0.85 code default: the gate **rejected 8 of 8 plannable cartridges**, one `except PlacementDisagreement: continue` at a time. | `cdd97fc` |
| 3 | `load_detector` took no `segmenter` argument and `_build_planner` hardcoded the heuristic extractor. | The trained segmenter was **unreachable from `main.py` under any configuration**. Any earlier claim that the pipeline demonstrated it end to end was overstated. | `f40cc1b` |
| 4 | FFDH opens its first shelf at `y = 0` and every later shelf at the top of the previous one — it **never scans the shelf origin in y** — and `_next_free_x` collapses a whole row band. | One blocked row killed an entire pack. On `scene_00005` the packer was handed a **93 %-free** grid containing a clear 48 × 112 mm rectangle and placed **zero** cells, when **79 of the 80** admissible shelf origins would have accepted one. | `562ca75` |
| 5 | `mm_per_px` was a config constant while the renderer randomised margin and zoom per frame. | True ground sample distance runs 0.490–1.045 mm/px; the pipeline used 0.625 for all of it. On 24 of 30 instances the planner **under-read the scene by 27 % at the median**; on the other 6 it over-read and reserved a footprint *smaller than the cell it was about to place*. | `380e7d5` |

Defect 1 is the one to take seriously as a lesson: `recog/synth3d/world.py` imports `bpy` and is not unit-testable, so the only thing that ever caught it was rendering frames and comparing them against the source CAD. Defect 5 is the one to take seriously as a bug class: a single wrong scalar was wrong three times in the same direction, because the erosion radius, the occupancy grid stride and the pixels-to-millimetres conversion are all derived from it.

### Negative results, reported as negative

Three comparisons were named before the result existed, and three came out null. They are in the repository at the same prominence as the positive ones. Only one of the three carried a *numeric* threshold fixed in advance — the cell-format null's ≥ 0.15. The `tau` criterion was directional (a gate needs the correlation to be negative; the sign was fixed in advance, the magnitude was not), and anchored-vs-wide was named as one of four comparisons in the design spec before any run existed but with no effect size attached. That distinction is worth keeping: a pre-named comparison is weaker evidence than a pre-registered threshold.

* **The confidence gate cannot work, structurally.** A gate needs agreement and error to move in *opposite* directions. Measured per SKU, the two placement estimates' IoU correlates with optimistic error **positively in all four SKUs** (Pearson 0.76 / 0.34 / 0.65 / 0.53, Spearman agreeing in sign), and area normalisation does not rescue it. The load-bearing evidence is the sign pattern, not any coefficient: n is 8–10 crops per SKU and the receipt says so. The mechanism explains the sign: `P_direct` and `P_derived` are the same `argmax` label map read twice, one with an erosion band, so they are not independent estimates at all. `P_safe = P_direct ∩ P_derived` is **retained** as a hard geometric constraint — that is a different claim, and the two are kept separate deliberately (FDR §13.2.1, `docs/receipts/tau_independence_correlation.txt`).
* **Widening the training distribution did nothing.** Two procedural tray distributions that differ across *every* sampled parameter — `anchored` and `wide` — score **0.6801 and 0.6794** selected-mean IoU on the same 836 held-out CAD test crops. A difference of **0.0007**, with no per-SKU per-class difference the instance counts support. The comparison was one of four stated in advance in the design spec, before any run existed.
* **A suspected cell-format mismatch recovered 7.6 % of its gap.** The prediction, recorded verbatim before the render started, was that restricting procedural trays to 18650 would move `battery` materially from 0.5593 toward the CAD control's 0.78, with ≥ 0.15 named as the resolvable effect. It moved **+0.017 of the 0.224 available**, the same order as the project's own dataset-to-dataset spread. Reported as the null it is (`docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md`).

### One gap diagnosed to mechanism, and closed

The procedural segmenter's `bay` IoU on held-out CAD trailed the CAD-trained control by 0.25. Decomposing the pooled metric showed it was **not a gap in segmenting bays**: on the 213 crops that contain a bay, procedural training was already within **0.021** of the control. **91.4 % of the gap** was `bay` painted onto *closed* cartridges — 136 of 623 sealed crops, against the control's 2.

The cause was measured, not guessed. Procedural lids were planar cuboids; all four real Anker lids are barrel-crowned with an **11.10 mm** long-edge fillet — equal to the entire lid height — and **89 % of each lid's upward-facing polygons sit below a 0.95 z-normal**, against 0 % for the procedural lid. Rendered, that is 10× less internal luminance structure. The model had learned *"featureless flat top ⇒ closed"*, which was true in 614 of 614 sealed training examples.

The evidence that this was the mechanism, rather than a plausible story, is a **monotone dose-response**: hallucination rate by quintile of the sealed shell's own luminance gradient ran 6.4 / 14.5 / 22.4 / 26.6 / **39.2 %** — a 6× spread — and stayed monotone within every one of the seven lighting rigs and within every brightness tercile.

Rolling a sampled fillet onto the procedural lid's top edges, as the single change:

| procedural model, `bay` | pooled (all 836 crops) | present-only (the 213 crops with a real bay) | sealed crops given a hallucinated bay |
| --- | ---: | ---: | ---: |
| flat lid (as first published) | 0.6555 | **0.8801** | **136 / 623 = 21.8 %** |
| crowned lid | **0.8755** | **0.8856** | **16 / 623 = 2.6 %** |
| CAD-trained control | 0.9009 | 0.9013 | 2 / 623 = 0.3 % |

The gradient dependence collapsed to 0.0 / 1.6 / 3.2 / 4.0 / 4.0 % — flat, which is the signature of a covered distribution rather than of a threshold move. Present-only `bay` **rose** (0.8801 → 0.8856), so the pre-registered falsifier — "the model just became reluctant to predict `bay`" — did not fire. `obstruction`, which lives inside open bays where the crown cannot reach, did not move (0.6306 → 0.6360). The crowned model is very slightly **worse** on its own validation split (0.7273 vs 0.7322), which is the shape a real out-of-distribution result has.

**The narrow claim only.** The `[0, 12]` mm crown range was chosen *after* measuring the real Anker lids, so this is **not** evidence that "procedural training transfers" and must not be quoted that way. What it shows is that a measured coverage gap was the *mechanism* behind a measured synthetic gap. It is domain randomisation informed by a measurement — and it is still synthetic-to-synthetic. Full record: `docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md` and `2026-08-11-sealed-unit-experiment.md`.

### The ceiling analysis that ended the work

Perception work stopped when the shipping pipeline reached 25 cells in 12 cartridges against an oracle's 25 in 11 — further accuracy was measured to be worth **nothing net** on this corpus. **Both halves of that comparison were measured before the 2026-08-15 packing-strip fix, and only the oracle half has been re-run since.** The oracle now measures 23 cells in 9 of 29 (`docs/receipts/placement_feasibility.txt` §3, and see the headline section for why it moved); the shipping half has no regenerable receipt and is not restated. The decision to stop is left standing because it was taken on the evidence available at the time and the geometry finding below is what actually bounds this corpus — but "worth nothing net" is a difference of two numbers and it is unsafe to quote until both are at HEAD. The same caveat applies to the sweep below, whose absolute cell counts were measured on the old strip; what is cited from it is the *ordering*, and that too wants re-running:

| clearance margin | instances with ≥ 1 cell | cells | overlaps > 5 % | worst |
| ---: | ---: | ---: | ---: | ---: |
| **0.0 mm (ships)** | **12** | **25** | **2** | **8.3 %** |
| 1.0 mm | 11 | 23 | 2 | 8.3 % |
| 1.5 mm | 10 | 21 | **3** | 7.8 % |
| 3.0 mm | 5 | 13 | 2 | 8.5 % |

3.0 mm is strictly dominated — twelve fewer cells for the same two overlaps — and 1.5 mm gives up four cells while *creating* a third overlap. The other proposed guard, rejecting a placement not fully inside the predicted free floor, rejects nothing: four of the five original offenders sit **100.0 %** inside it. Details in `docs/superpowers/specs/2026-08-11-placement-safety.md` §2.3–§2.4 and FDR §13.2.1.

### A cartridge that cannot be certified

`AnkerPowerCore10000`'s placement region is 54.9 × **65.0 mm** and the planner's nominal cell is 18.5 × **65.0 mm**. Not approximately — exactly. The placement rectangle is axis-aligned (sound: the camera mount is fixed) but the cartridge is not; `layout.plan` seats every unit at `quarter × 90° + jitter` with **±2°** of jitter, and a real jig has clearance of the same order. An axis-aligned strip of width `w` fits a bay of height `H` rotated by θ only where `L(θ) = (H − w·sinθ)/cosθ ≥ 65.0`, so the packer alone consumes **18.5·tanθ ≈ 0.32 mm per degree** against 0.00 mm of margin.

Fed **ground-truth** label maps at each frame's true scale — perfect segmentation, perfect boxes, perfect calibration — this SKU places zero cells in **10 of 10** instances through the shipping extractor, and **47 of 47** over the whole 502-scene corpus. Relaxing the extractor's 4.25 mm wall inset to zero recovers **none** of them: the sweep reads 0/10 (0 cells) at 4.25, 3.00 and 0.00 mm alike, at both the configured 18.5 mm nominal and the CAD's 18.3 mm. Tolerance is not the lever either — 18.5 → 18.3 mm frees 0.2 mm on the diameter and nothing at all on the length, and it is the length this bay is short of. Receipt: `docs/receipts/placement_feasibility.txt` §2 and §3.

**The measured count went 2 → 1 → 0 in four days, on two unrelated correctness fixes**, and the published wording moved three times because one of those moves was a prose correction rather than a measurement: `b93bbd3` removed a recovered instance at 64.7 mm of free strip; `cc3fce1` on 2026-08-14 changed this sentence *from* "recovers 0" *to* "recovers 1" because the receipt contradicted the published claim; and the 2026-08-15 packing-strip fix removed the remaining instance at 65.1 mm, which puts the measurement back at 0 and the sentence back to where it started. So the safe reading is not that the number is settled but that a zero-margin fit is decided by which way the rasteriser rounds, which is the point this section makes. The three revisions, each with its mechanism, are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

Over the whole corpus the same thing happened, larger: the SKU used to place 6 cells across 4 of its 47 open instances, and now places 0 across 0, with **no** counter-example left anywhere and no rotation band left to quote, because no instance takes a cell at any rotation ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md)).

The general statement is FDR §3's, and it is the transferable part: **a bay packed to exact tolerance cannot be certified by a vision system, because certification needs margin to absorb a non-zero measurement error and an exact fit offers none.** The available responses are specification changes, not accuracy work.

### The structural insight, learned twice

**You cannot detect your own prediction error using that same prediction.**

It retired the confidence gate: two masks derived from one `argmax` label map cannot disagree informatively, so their IoU carries no information about whether the label map is right. Later, independently, it rejected a clearance margin proposed to fix placement overlaps: every guard the planner has — `P_safe`, a clearance inset, an overlap test against the predicted classes — is computed from the same prediction that is wrong. Where the segmenter labels wall as floor, every downstream check computed from that label map agrees with it. The two mitigations that *would* work are the two using information the prediction does not contain: reduce the segmenter's own boundary displacement, or verify contact at placement time by force rather than by camera.

### Where to look

`docs/superpowers/specs/` holds the diagnosis → experiment → result trail for the work above, one document per investigation, each naming its own baseline commit and test count. `docs/superpowers/specs/README.md` indexes them. `docs/superpowers/audit/` holds **nineteen** adversarial reviews of this work — sixteen run on 2026-08-12 (A–P) and three on 2026-08-14 (T, U, V) — and the `2026-08-12-fix-*` specs record what was done about them.

`docs/receipts/` holds tool-generated, never-hand-edited output — **for most figures, not for every one, and the exception is named rather than glossed.** Everything scored by `recog.seg_evaluate`, `recog.seg_ablation`, `recog.calibrate_tau`, `scripts/forbidden_bench.py` and `scripts/seed_check.py` has a receipt regenerated by committed tooling. Three groups of quoted figures do not. The gap decomposition above — present-only `bay` 0.8801 / 0.8856 / 0.9013, the composite ceiling 0.9009, "91.4 % of the gap", and the 136/623 and 16/623 sealed-crop rates — came from a **scratch diagnostic that was never committed and emitted no receipt**; its anchor, pooled `bay` 0.6555, *is* the receipt's. The second group is the **shipping** side of the oracle comparison — 25 cells across 12 cartridges — which traces to a results table in `docs/superpowers/specs/2026-08-11-placement-safety.md` and to no receipt; the headline section withdraws it pending re-measurement. (Until 2026-08-15 this paragraph named the *oracle* side as the receiptless one; `b69bcb3` made it regenerable and the retraction is in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).) And sixteen of the **forty** committed receipts are inherited from before this repository's history and have no surviving generator at all — one of which, `tau_independence_correlation.txt`, is current and load-bearing. FDR Appendix C enumerates all of it, and was written when the count was thirty-four; the **six** added since — `detector_bench.txt`, `seed_reproducibility.txt`, `main_run.txt`, `real_photo_eval.txt` and `real_photo_eval_include_empty.txt` at `9b38de9`, and `placement_feasibility.txt` at `b69bcb3` — each have a committed generator (`scripts/detector_bench.py`, `scripts/seed_check.py`, `main.py --receipt`, `recog/eval_real.py`, `scripts/placement_feasibility.py`), so the sixteen-without-a-generator numerator is unchanged and only the denominator moves. (This total has now read thirty-four, thirty-seven, thirty-nine and forty, corrected each time one commit after the receipt that invalidated it — [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md). It is one shell command, `git ls-files docs/receipts | wc -l`, and should be re-run whenever a receipt is added rather than carried forward by arithmetic.)

---

## Repository layout

```
auto-pick/
├── common/           Shared types (BBox, Detection, PickPlacePose, ...),
│                     small utilities (YAML config loader, logger), and
│   └── packing.py          FFDH + the two arms `pack_best_effort` competes
├── recog/            Recognition module
│   ├── model.py            Faster R-CNN + ResNet-34 FPN factory
│   ├── dataset.py          VOC + COCO loaders, VOC / real-photo datasets
│   ├── augmentation.py     Albumentations pipeline + numpy fallback
│   ├── training.py         Training loop (cosine LR, frozen BN, ckpts)
│   ├── inference.py        FasterRCNNDetector + HeuristicDetector
│   ├── evaluate.py         Pure-numpy VOC 11-point mAP + pose errors
│   ├── eval_real.py        Held-out real-photo evaluation (mAP + overlays)
│   ├── calibration.py      Per-frame ground sample distance from the sidecar
│   ├── bay_segmenter.py    DeepLabv3+MobileNetV3 per-cartridge segmenter
│   ├── seg_dataset.py      Per-ROI crop dataset for the bay segmenter
│   ├── seg_training.py     Bay segmenter training loop (CE + Dice)
│   ├── seg_evaluate.py     Bay segmenter metrics (IoU, boundary, area error)
│   ├── calibrate_tau.py    Calibrated the (now retired) arbitration threshold
│   ├── realtest/           7 annotated real photographs (COCO, committed)
│   ├── synth_dataset.py    Procedural cv2 scene generator (demo / tests)
│   ├── synth3d/            Blender synthetic-scene package (assets, layout,
│   │                       materials, render, world, scene, annotate)
│   ├── generate3d.py       Blender entry point -> flat Pascal-VOC dataset
│   └── verify3d.py         Draws the generated boxes onto the renders
├── plan/             Planning module
│   ├── scene.py            Digital twin (entities + occupancy grids)
│   ├── placement_area.py   Heuristic (demo) + segmentation-based valid-area
│   │                       extractors — see "Two placement-area extractors"
│   ├── arbitration.py      Reconciles the segmenter's two placement estimates
│   ├── bin_packing.py      2-D strip packing — see "How the packer picks"
│   └── planner.py          Orchestrator emitting PickPlacePose queues
├── execution/        Execution module
│   ├── protocol.py         16-byte binary packet + CRC-16/MODBUS
│   ├── execution.py        KukaClient (TCP, retries, E-stop)
│   ├── mock_kuka_server.py Software-only KUKA simulator
│   └── krl_prog/           Reference KRL routines for the real controller
├── configs/          YAML configs per module + top-level demo.yaml
├── tests/            Pytest suite (1,396 passing, 1 skipped; the
│                     torch-free count is pending re-measurement)
├── docs/             Final Design Report, specs, receipts — index: docs/README.md
├── main.py           End-to-end integration loop
└── pyproject.toml    Project metadata and coverage config
```

## How the packer picks

`plan/bin_packing.py` and `common/packing.py` hold the 2-D strip packer. `first_fit_decreasing` is the forbidden-mask-aware shelf FFDH that is the project's principal algorithmic contribution (FDR §6.3.1), and it is unchanged and frozen — `recog/synth3d` lays out synthetic scenes with it, so touching it would silently redraw a training corpus.

It is no longer what the planner runs on its own, for the reason in defect 4 above. Both planner call sites now use `common.packing.pack_best_effort`, which competes unmodified FFDH against a shelf-origin-scanning arm and a shelf-free grid-greedy arm and returns whichever placed most. Ties go to FFDH, so `best ≥ FFDH` holds **by construction** — no instance can regress, which matters because the failure mode this project kept hitting is a change that lifts an average and quietly regresses a case nobody re-measured.

Both new arms earn their place and neither would do on its own: each scores 16 to the combination's 17 on the real instances. On 30 real packing instances the 7 that can hold a cell at all go from **8 to 17 cells**; on the published benchmark the shipping packer places 14.55 at 2.5 % forbidden coverage against FFDH's 14.28, with the real movement at 10–15 % coverage (2.60 → 5.53, 0.57 → 2.85). Cost: **2.8 ms mean / 3.9 ms p95 / 7.4 ms worst observed** on bench masks, against the forbidden-mask-aware FFDH arm's **0.35 ms at 2.5 % coverage, peaking at 1.07 ms at 10 %** — inside the 8 ms O3 budget, but by **0.6 ms at the worst mask measured**. (The microsecond columns are wall-clock and the receipt says so; they move on every regeneration, and the **cell counts do not** — `forbidden_bench.csv` and the eight-seed sweep in `forbidden_bench_seeds.txt` came back byte-identical through the 2026-08-15 packing-strip fix, which is what exonerates `common/packing.py` from that fix's movement. Every timing figure in this paragraph has been corrected at least once — on 2026-08-12, 2026-08-14 and 2026-08-15 — and the superseded readings are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).) Diagnosis and per-arm results: `docs/superpowers/specs/2026-08-11-packing-ceiling.md`; receipt `docs/receipts/forbidden_bench.txt`.

## Two synthetic generators, two purposes

There are deliberately two synthetic-scene generators, and they are not interchangeable:

| | `recog/synth_dataset.py` | `recog/generate3d.py` |
| --- | --- | --- |
| Renderer | OpenCV rectangles | Blender / Cycles path tracer |
| Interpreter | system Python | Blender's bundled Python |
| Speed | ~ms per image | ~3.5 s per 1280×720 frame on an RTX 3060 |
| Fidelity | flat coloured primitives | real CAD geometry, randomised PBR materials, physical lighting |
| Used by | `main.py`'s software-only demo (`configs/demo.yaml`), the unit tests | training the Faster R-CNN detector and the bay segmenter; `configs/demo_seg.yaml`'s demo frames |

`synth_dataset.py` exists so the end-to-end demo and the test suite can run in seconds with no GPU, no Blender and no CAD. It is a stand-in for a camera, not a source of training data — a detector trained on it learns "grey rectangle on grey background".

`generate3d.py` is the real training-data source. It imports the converted `.glb` assemblies from `recog/synth3d/assets/`, randomises materials, backdrop, lighting, layout (loose scatter or a fixture jig) and camera, path-traces a beauty pass, and derives pixel-exact boxes from Cycles' object-index pass rather than from projected 3-D corners. Presets live in `configs/synth3d.yaml`; Blender's bundled Python has no PyYAML, so run `python -m recog.sync_config` to transcribe it to the JSON sidecar Blender reads.

Both write the same flat Pascal-VOC layout (`images/` + `annotations/`), so `recog.dataset.BatteryCartridgeDataset` reads either one and `recog/training.py` owns the train/val split. Point `configs/recognition.yaml`'s `dataset` block at whichever you want to train on.

The demo's frames are configured separately, in `configs/demo.yaml`'s `mode.img_dir` — it defaults to `recog/dataset/images` and falls back to `recognition.dataset.img_dir` when unset. Repointing the training set at `recog/dataset3d` therefore leaves `python main.py --config configs/demo.yaml` working.

```bash
python -m recog.sync_config
BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BLENDER" -b --python recog/generate3d.py -- --n 2000 --out recog/dataset3d --device GPU --resume
python -m recog.verify3d --data recog/dataset3d --n 16     # then LOOK at contact_sheet.png
```

`--resume` makes a long run interruptible. `verify3d` runs in system Python because it needs Pillow, which Blender does not ship — that render/inspect split is why generation is two commands.

**A dataset rendered today will not look like the committed one.** A swallowed `except` in `recog/synth3d/materials.py` had been discarding both the drawn `roughness` and the drawn `wear` and wiring the raw noise ramp straight into Principled's Roughness — on 100 % of surfaces, while `meta["materials"]` recorded the values that never arrived. It was fixed on 2026-08-12, so new renders carry the roughness the sidecars always claimed. **Labels are unaffected**: boxes and masks come from the object-index pass, which materials never touch, so every committed annotation stands and no dataset was regenerated. But a redrawn corpus is a different *visual* distribution — do not mix pre- and post-fix renders in one training set — and `MIN_LUMA_DELTA` and `configs/synth3d.yaml`'s `luma_ref` table were both measured on the old path and should be re-measured before a redraw is trusted. `docs/superpowers/blender-dataset-known-issues.md` item 0 is the full note.

## Two placement-area extractors, and only one is for real cartridges

`plan/placement_area.py` exposes two extractors behind the same `extract(image_rgb, cartridge_bbox, ...) -> PlacementArea` contract, and they are not interchangeable:

* **`HeuristicPlacementAreaExtractor`** — Otsu threshold on the green channel, largest contour, inset, subtract a dark PCB blob. It is the **demo-only path**: it assumes a light tray with a dark interior module (PPR §5.3.2), which matches `recog/synth_dataset.py`'s flat green rectangles but not the real, black cartridges — measured at zero placeable area on 7 of the 20 cartridges annotated in the held-out real photographs (`recog/realtest/`). `main.py`'s software-only demo uses it deliberately, because it has no model to load and keeps the demo torch-free; it warns at construction time so its scope limit can't be missed.
* **`SegmentationPlacementAreaExtractor`** — the path for real imagery. It consumes a trained segmenter's per-pixel label map (`Snapshot.cartridge_masks`, populated by `recog.inference.attach_cartridge_masks`) and intersects two placement estimates via `plan/arbitration.py` — the network's own `bay` channel, and one derived from the eroded, centre-connected interior. `P_safe = P_direct ∩ P_derived` is applied **unconditionally**: it is a geometric constraint (nothing outside the visible cavity is ever placeable) and it has no threshold.

  **There is no `tau`.** The IoU between the two estimates is still computed and reported on `PlacementArea.consistency_iou`, but nothing gates on it, and the constructor no longer accepts the argument, so code that still passes it fails loudly instead of being silently ignored. The gate was retired on measurement, not taste — see defect 2 and the first negative result above (FDR §13.2.1, `docs/receipts/tau_independence_correlation.txt`). The three mutually inconsistent values this paragraph used to quote are all gone ([`docs/CORRECTIONS.md`](docs/CORRECTIONS.md)); `recog/calibrate_tau.py` and its receipt are kept as the record of the measurement that retired it.

  Two guards remain, both on the *contents* of a crop rather than on the model's confidence in it. `BadDetectorBox` fires when the crop centre lands on background, and again when the placeable region's rotated short axis exceeds **81.7 mm** — the largest cataloged cartridge's outer footprint, so the bound only ever fires on the physically impossible and needs no tuned tolerance. And an occupancy cell is FREE only if *all* of it is free, not merely its centre pixel; the honest version is also the faster one (2.97 ms against 4.25 ms on the budget test's crop), because it is computed over a summed-area table.

The split between the modules exists because of a hard latency budget, not preference: FDR O3 caps planning's queue rebuild at 8 ms per cartridge, and a single segmenter forward pass alone costs roughly that much on its own. Segmentation therefore runs once per frame, batched, in Recognition (`recog/bay_segmenter.py`) — measured on an RTX 3060 at 16.2 ms for 8 cartridges batched, against 58.6 ms for the same 8 run in a loop (`docs/receipts/seg_eval.txt`), inside vs. well outside the separate 50 ms end-to-end budget respectively (the table is wall-clock and is re-taken every time that receipt is regenerated, so it moves a little each time — a 16.2–21.2 ms batched spread across six clean runs. The six readings, including one taken under unrelated GPU load and superseded rather than averaged in, are in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md); FDR §13.2.1 has the measurement conditions). Planning only ever does mask arithmetic on the already-computed masks (`plan/arbitration.py`, measured ~2 ms per cartridge). `tests/test_planner.py::test_segmentation_extract_arithmetic_stays_under_the_o3_budget` pins that arithmetic-only cost (`extract()` alone — arbitration + rasterisation, not the packing pass that follows it) against the 8 ms budget.

## What the bay segmenter is measured on — and what cannot be measured here

**Real photographs of this project's cells and cartridges are not obtainable** (owner-confirmed, 2026-08-09). The direct consequence, stated before any number below: **sim-to-real transfer is unvalidated and cannot be validated under this constraint** — not "not yet measured". **Every segmentation, placement-area and packing figure in this repository is synthetic-to-synthetic**: measured on renders, against ground truth derived from the same renders. The dedicated limitation statement is FDR §13.2.2.

What *is* answerable without a photograph: does a segmenter trained on **procedurally generated** cartridge trays — shapes it has never seen a real example of — transfer to the four **real measured Anker CAD assemblies** it never trained on? Eight models — two procedural sampling bands, the 18650-only and crowned-lid variants, and four leave-one-SKU-out CAD controls — all scored on the same 836 held-out CAD test crops from a disjoint 500-scene render (`docs/receipts/seg_eval_*_on_cad_test.txt`, FDR §13.1.1; the FDR's "all six" predates the last two models). The decomposition, the mechanism and the fix are in "One gap diagnosed to mechanism, and closed" above.

One row of the per-class table is not evidence at all: `obstruction` parity between procedural and CAD models is a **shared-code artefact**, since `world.build_obstructions` has one call site executed identically by both pipelines, so parity would hold under any hypothesis. `cartridge` (0.9120 against the control composite's 0.9382) and `electronics` (0.7819 against 0.8530) are the largest honest remaining shortfalls, with no hallucination component left to explain them.

## The real-photo held-out set

Validation mAP on renders only says the detector learned the renderer. `recog/realtest/` is the closest this project gets to the real question, and it is a **box-level** check on the detector, not a segmentation or placement one: seven photographs of the actual cells and cartridges (3024×4032, phone camera), annotated in CVAT and exported as COCO — 80 boxes, 60 battery and 20 cartridge, and **zero segmentation polygons**. It is committed to the repository, never trained on, and never used to select a checkpoint. It is also the entire real-image corpus this project will ever have (see the limitation above), so at n = 20 cartridges it is a smoke test and a source of qualitative diagnostics — never a quantitative transfer claim. Three same-recipe segmenter checkpoints scored 0.211, 0.232 and 0.318 placeable fraction on it against the heuristic's fixed 0.217: run-to-run variation exceeds the effect the comparison exists to detect, and that number series, not any point in it, is the finding (FDR §13.2.1, §13.2.2).

```bash
python -m recog.eval_real --checkpoint recog/checkpoints/best.pt --save-overlays /tmp/real
```

The report gives per-class AP and mAP at IoU 0.50 and 0.75 (the same `recog.evaluate` 11-point protocol as the synthetic numbers), plus per-class GT and prediction counts and the confidence threshold used. `--save-overlays` writes ground truth and predictions onto each photo in different colours: a bare mAP cannot distinguish "found nothing" from "found everything at the wrong scale", and the overlays can.

## Other entry points

Task | Command
--- | ---
Train Faster R-CNN (needs torch + GPU) | `python -m recog.training --config configs/recognition.yaml`
Start the mock KUKA server standalone | `python -m execution.mock_kuka_server`
Generate synthetic scenes | `python -m recog.synth_dataset --out recog/dataset --n 200`
Sync synth3d config to Blender | `python -m recog.sync_config`
Generate 3-D synthetic scenes (Blender) | `blender -b --python recog/generate3d.py -- --n 200 --out recog/dataset3d`
Inspect generated boxes | `python -m recog.verify3d --data recog/dataset3d --n 12`
Evaluate on the real photographs | `python -m recog.eval_real --checkpoint recog/checkpoints/best.pt`
Sweep lighting presets | `blender -b --python recog/generate3d.py -- --sweep lighting --out recog/sweeps`
Train the bay segmenter (needs torch + GPU) | `python -m recog.seg_training --config configs/segmentation.yaml`
Evaluate the bay segmenter (IoU, boundary, area error) | `python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml`
Re-measure the retired arbitration threshold tau (historical) | `python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml`
Check that training is still reproducible (3 × 1 epoch, ~90 s) | `python scripts/seed_check.py`
Run the unit tests | `pytest -q`
Run with coverage | `pytest -q --cov`

## Requirements

- Python 3.10+
- NumPy 1.24+, PyYAML 6.0+, **`opencv-python-headless` 4.8+**
- Albumentations 1.4+ (recognition augmentation — a numpy-only fallback is used if unavailable)
- PyTorch 2.x + torchvision (required only for training and loading a learned detector; the heuristic detector runs without torch)
- **Pillow 10.3+** (dataset image loading)

Two of those floors moved on 2026-08-12 after a security audit, and the reasons are worth stating because they are floors and this project ships no lockfile — a constrained or reproducible resolve can legitimately land *at* the floor. `pillow>=10.0` admitted **CVE-2023-4863** (libwebp heap buffer overflow, critical, fixed in 10.0.1) and **CVE-2023-50447** (arbitrary code execution via `PIL.ImageMath.eval`, fixed in 10.2.0); Pillow decodes images throughout `recog/`, so the floor is now 10.3. And `opencv-python` was swapped for `opencv-python-headless`: a grep over the whole highgui surface — `imshow`, `waitKey`, `namedWindow`, `setMouseCallback` and eighteen others — returns zero hits, so the GUI build was pulling a Qt/GTK/X11 stack that is never entered, for install weight and an `ImportError: libGL.so.1` class of container failure. The one videoio call, `cv2.VideoCapture`, is present in the headless build.

Relatedly, `recog/inference.py`'s detector loader now passes `weights_only=True` to `torch.load`. No `.pt` is committed anywhere in this repository, so **anyone following the `--checkpoint recog/checkpoints/best.pt` commands above has obtained that file from somewhere else** — and `torch.load`'s default runs the pickle machinery, which makes loading a checkpoint arbitrary code execution by whoever produced it. `recog/bay_segmenter.py` had carried that rule in a comment since it was written; it had not been applied one module over, on the more prominent path. That is now every `torch.load` in the tree except `seg_training.py`'s `--resume` of its own local optimiser state, which is deliberate and justified in place.

Dev extras (`pip install -e '.[dev]'`) add pytest, pytest-cov.

## Design Report

The full Final Design Report — requirements, literature review, detailed design, test strategy, risk assessment, and AHEP-4 learning-outcome mapping — is in [`docs/FDR_v3.md`](docs/FDR_v3.md) (`docs/FDR.md` and `docs/FDR_v2.md` are the superseded earlier revisions). Start at §13.2.2 for what this project can and cannot claim.

A narrative account of the measurement work, written for a general engineering reader, is [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md).

## Authors

Yousif Al-Haidary, supervised by Professor Svetan Ratchev FREng, Cripps Professor of Production Engineering, University of Nottingham.

## Trademarks

Anker and PowerCore are trademarks of their respective owner. This project is not affiliated with, endorsed by or sponsored by them. The four PowerCore power banks named in asset filenames, in `recog/synth3d/assets/catalog.json`, in the photographs under `recog/realtest/` and throughout this documentation are retail units used as measurement subjects for academic research; the product names appear only to identify which physical unit a given figure was measured on.
