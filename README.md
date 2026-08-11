# Autonomous Recognition, Pick and Place

MEng Individual Project — Yousif Al-Haidary (REDACTED), University of Nottingham, supervised by Dr Svetan Ratchev.

A software-only realisation of a vision-guided robotic cell that picks loose cylindrical 18650 / 21700 lithium-ion cells from a camera view and places them into protective cartridges on a KUKA KR 6 R700 industrial robot. The full pipeline — perception, digital twin, 2-D bin-packing, and KUKA Ethernet KRL command streaming — runs end-to-end without any physical hardware via a mock robot simulator that speaks the real binary protocol.

## Architecture

A strict sequential flow between three loosely-coupled modules, with frozen dataclass contracts at every boundary:

```
Camera ─▶ Recognition ─▶ Planning ─▶ Execution ─▶ KUKA controller
         (Faster R-CNN   (digital   (EKRL 3.1,
          + optional      twin +     CRC-16)
          bay segmenter)  packing)
```

Each arrow is a well-defined type: `Snapshot` from recognition, `PickPlacePose` from planning, `RobotStatus` back from execution. See the Final Design Report in `docs/FDR_v3.md` for the full rationale and trade-off analysis — that is the current revision, and every `§` reference in this README is to it. `docs/FDR.md` and `docs/FDR_v2.md` are the superseded earlier drafts, kept for history.

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
├── tests/            Pytest suite (708 tests)
├── docs/             Final Design Report (FDR.md) and supporting material
├── main.py           End-to-end integration loop
└── pyproject.toml    Project metadata and coverage config
```

## Running the software-only demo

The default configuration runs with synthetic images and the mock robot, so no GPU / camera / KUKA is needed:

```bash
# 1. Install runtime dependencies
pip install -e .

# 2. Generate a small synthetic dataset (if you haven't already)
python -m recog.synth_dataset --out recog/dataset --n 50

# 3. Run the full perception → plan → execute loop
python main.py --config configs/demo.yaml
```

The loop logs per-cycle perception / planning latencies, the cartridge / mask / queue counts, and a summary of placed / pick-failed / place-failed counts at exit.

### ...and the same loop with the trained segmenter in it

`configs/demo.yaml` runs the **heuristic** placement-area extractor and no segmenter — that is what keeps it torch-free, and it is the only thing it claims. `configs/demo_seg.yaml` is the other half: it puts the trained bay segmenter into recognition as a second stage and plans from its label maps.

```bash
python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt
```

The `mode.segmentation` block is what selects that path, and it selects it in both places at once — the detector gets the segmenter, the planner gets `SegmentationPlacementAreaExtractor`. They cannot be configured apart, because either half alone is silent: a segmenter with no consumer runs for nothing, and the segmentation extractor with no segmenter raises on every cartridge into a blanket `except Exception`. For the same reason a missing checkpoint raises instead of falling back, and a run that completes having produced **zero** placement areas is treated as a failed run, not a quiet one.

The frames matter as much as the checkpoint. `demo_seg.yaml` reads Blender renders (`recog/dataset3d_seg`), because the segmenter predicts no `bay` at all on `synth_dataset.py`'s flat green rectangles — pointed at those, it would complete cleanly and demonstrate nothing. Last generated run (`docs/receipts/main_seg_run.txt`): 15 frames, 26 cartridges detected, 26 segmented, **8 placement areas**, 1 pick-and-place executed. Note the last number: the loop executes at most one pose per cycle and a pose also needs a loose battery in the same frame — see `docs/superpowers/specs/2026-08-11-segmenter-integration.md` for the measurement.

**Until commit `12134c2` the segmenter was not in this loop at all**, and could not be put in it from any config: `load_detector` took no `segmenter` argument and `_build_planner` hardcoded the heuristic extractor. Any earlier claim that the pipeline demonstrated the segmenter end to end was overstated. It is true now — of `demo_seg.yaml`. `demo.yaml` still runs the heuristic and is still torch-free, and that is the path the reproducibility claim rests on; the two must not be conflated. The receipt above is evidence that the **wiring** works: those frames are the segmenter's own training corpus, so it is not a generalisation measurement. Held-out numbers live in `docs/receipts/seg_eval_*_on_cad_test.txt` and FDR §13.1.1.

## How the packer picks

`plan/bin_packing.py` and `common/packing.py` hold the 2-D strip packer. `first_fit_decreasing` is the forbidden-mask-aware shelf FFDH that is the project's principal algorithmic contribution (FDR §6.3.1), and it is unchanged and frozen — `recog/synth3d` lays out synthetic scenes with it, so touching it would silently redraw a training corpus.

It is no longer what the planner runs on its own. FFDH opens its first shelf at `y = 0` and every later shelf at the top of the previous one — **it never scans the shelf origin in y** — and `_next_free_x` collapses a candidate shelf's whole row band, so one mostly-blocked row poisons every column in it. A forbidden region in the first shelf's band therefore kills the whole pack: on a real frame (`scene_00005`) the packer was handed a 93 %-free grid containing a clear 112 × 48 mm rectangle and placed **zero** 18.5 × 65 mm cells, when 79 of the 80 admissible shelf origins on that grid would have accepted one.

Both planner call sites now use `common.packing.pack_best_effort`, which competes unmodified FFDH against a shelf-origin-scanning arm and a shelf-free grid-greedy arm and returns whichever placed most. Ties go to FFDH, so `best ≥ FFDH` holds **by construction** — no instance can regress. On 30 real packing instances the 7 that can hold a cell at all go from **8 to 17 cells**; on the published benchmark the shipping packer places 14.55 at 2.5 % forbidden coverage against FFDH's 14.28, with the real movement at 10–15 % coverage (2.60 → 5.53, 0.57 → 2.85). Cost: 3.4 ms mean / 4.6 ms worst on bench masks against 0.9 ms for FFDH alone, still inside the 8 ms O3 budget. Diagnosis and per-arm results: `docs/superpowers/specs/2026-08-11-packing-ceiling.md`; receipt `docs/receipts/forbidden_bench.txt`.

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

## Two placement-area extractors, and only one is for real cartridges

`plan/placement_area.py` exposes two extractors behind the same `extract(image_rgb, cartridge_bbox, ...) -> PlacementArea` contract, and they are not interchangeable:

* **`HeuristicPlacementAreaExtractor`** — Otsu threshold on the green channel, largest contour, inset, subtract a dark PCB blob. It is the **demo-only path**: it assumes a light tray with a dark interior module (PPR §5.3.2), which matches `recog/synth_dataset.py`'s flat green rectangles but not the real, black cartridges — measured at zero placeable area on 7 of 20 held-out real photographs (`recog/realtest/`). `main.py`'s software-only demo uses it deliberately, because it has no model to load and keeps the demo torch-free; it warns at construction time so its scope limit can't be missed.
* **`SegmentationPlacementAreaExtractor`** — the path for real imagery. It consumes a trained segmenter's per-pixel label map (`Snapshot.cartridge_masks`, populated by `recog.inference.attach_cartridge_masks`) and intersects two placement estimates via `plan/arbitration.py` — the network's own `bay` channel, and one derived from the eroded, centre-connected interior. `P_safe = P_direct ∩ P_derived` is applied **unconditionally**: it is a geometric constraint (nothing outside the visible cavity is ever placeable) and it has no threshold.

  **There is no `tau`.** The IoU between the two estimates is still computed and reported on `PlacementArea.consistency_iou`, but nothing gates on it, and the constructor no longer accepts the argument, so code that still passes it fails loudly instead of being silently ignored. The gate was retired on measurement, not taste: the two estimates are the same `argmax` read twice (one with an erosion band), and their IoU correlates with placement error in the *wrong* direction — positive in all four cataloged SKUs, raw and area-normalised (FDR §13.2.1, `docs/receipts/tau_independence_correlation.txt`). It also had a real cost. On 15 `recog/dataset3d_seg` frames through the trained detector and segmenter — 26 cartridge crops, 8 with a predicted bay — the gate admitted **3 of those 8** at its code default 0.85, and **0 of 8** at `configs/planning.yaml`'s own `mm_per_px` of 0.38, where the wider erosion pushes every IoU below the threshold. Without it, all 8 are plannable at either scale (`docs/superpowers/specs/2026-08-11-segmenter-integration.md`). The three mutually inconsistent values this paragraph used to quote — code 0.85, YAML 0.7492, README 0.5715 — are all gone; `recog/calibrate_tau.py` and its receipt are kept as the record of the measurement that retired it.

The split exists because of a hard latency budget, not preference: FDR O3 caps planning's queue rebuild at 8 ms per cartridge, and a single segmenter forward pass alone costs roughly that much on its own. Segmentation therefore runs once per frame, batched, in Recognition (`recog/bay_segmenter.py`) — measured on an RTX 3060 at 20.2 ms for 8 cartridges batched, against 76.5 ms for the same 8 run in a loop (`docs/receipts/seg_eval.txt`, regenerated at commit `390836b`), inside vs. well outside the separate 50 ms end-to-end budget respectively (up from an earlier 16.7 ms / 60.0 ms measurement on the same hardware, a real but modest increase — an intermediate reading of 40.9 ms / 157.0 ms taken while this machine carried substantial unrelated GPU load was superseded by the clean re-measurement quoted here; see FDR §13.2.1 for the measurement conditions). Planning only ever does mask arithmetic on the already-computed masks (`plan/arbitration.py`, measured ~2 ms per cartridge). `tests/test_planner.py::test_segmentation_extract_arithmetic_stays_under_the_o3_budget` pins that arithmetic-only cost (`extract()` alone — arbitration + rasterisation, not the packing pass that follows it) against the 8 ms budget.

## What the bay segmenter is measured on — and what cannot be measured here

**Real photographs of this project's cells and cartridges are not obtainable** (owner-confirmed, 2026-08-09). The direct consequence, stated before any number below: **sim-to-real transfer is unvalidated and cannot be validated under this constraint** — not "not yet measured". **Every segmentation, placement-area and packing figure in this repository is synthetic-to-synthetic**: measured on renders, against ground truth derived from the same renders. The dedicated limitation statement is FDR §13.2.2.

What *is* answerable without a photograph: does a segmenter trained on **procedurally generated** cartridge trays — shapes it has never seen a real example of — transfer to the four **real measured Anker CAD assemblies** it never trained on? Six models, all scored on the same 836 held-out CAD test crops from a disjoint 500-scene render (`docs/receipts/seg_eval_*_on_cad_test.txt`, FDR §13.1.1). Two things must be read together, because the published pooled `bay` figure conflates them:

| procedural model, `bay` | pooled (all 836 crops) | present-only (the 213 crops with a real bay) | sealed crops given a hallucinated bay |
| --- | ---: | ---: | ---: |
| flat lid (as first published) | 0.6555 | **0.8801** | **136 / 623 = 21.8 %** |
| crowned lid | 0.8755 | **0.8856** | **16 / 623 = 2.6 %** |
| CAD-trained control | 0.9009 | **0.9013** | 2 / 623 = 0.3 % |

On crops that contain a bay, procedural training was already within 0.021 of the CAD-trained ceiling; **91.4 % of the apparent gap was `bay` painted onto *closed* cartridges.** The cause was measured: procedural lids were planar cuboids while all four Anker lids are barrel-crowned, giving 10× less internal shading structure, so the model had learned "featureless flat top ⇒ closed". Adding a sampled lid crown as the single change closed 92 % of those false positives.

**The narrow claim only.** The `[0, 12]` mm crown range was chosen *after* measuring the real Anker lids, so this is **not** evidence that "procedural training transfers" and must not be quoted that way. What it shows is that the missing shading-structure coverage was the *mechanism* behind the transfer gap, and closing it recovers most of the gap — domain randomisation informed by a measured gap. One row of the per-class table is not evidence at all: `obstruction` parity between procedural and CAD models is a **shared-code artefact**, since `world.build_obstructions` has one call site executed identically by both pipelines, so parity would hold under any hypothesis. Full record: `docs/superpowers/specs/2026-08-11-transfer-gap-diagnosis.md` and `2026-08-11-sealed-unit-experiment.md`.

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
Run the unit tests | `pytest -q`
Run with coverage | `pytest -q --cov`

## Requirements

- Python 3.10+
- NumPy, PyYAML, OpenCV-Python 4.8+
- Albumentations 1.4+ (recognition augmentation — a numpy-only fallback is used if unavailable)
- PyTorch 2.x + torchvision (required only for training and loading a learned detector; the heuristic detector runs without torch)
- Pillow (dataset image loading)

Dev extras (`pip install -e '.[dev]'`) add pytest, pytest-cov.

## Design Report

The full Final Design Report — requirements, literature review, detailed design, test strategy, risk assessment, and AHEP-4 learning-outcome mapping — is in `docs/FDR_v3.md` (`docs/FDR.md` and `docs/FDR_v2.md` are the superseded earlier revisions). Start at §13.2.2 for what this project can and cannot claim.

## Authors

Yousif Al-Haidary, supervised by Dr Svetan Ratchev.
