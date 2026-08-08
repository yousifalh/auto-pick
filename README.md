# Autonomous Recognition, Pick and Place

MEng Individual Project — Yousif Al-Haidary (REDACTED), University of Nottingham, supervised by Dr Svetan Ratchev.

A software-only realisation of a vision-guided robotic cell that picks loose cylindrical 18650 / 21700 lithium-ion cells from a camera view and places them into protective cartridges on a KUKA KR 6 R700 industrial robot. The full pipeline — perception, digital twin, 2-D bin-packing, and KUKA Ethernet KRL command streaming — runs end-to-end without any physical hardware via a mock robot simulator that speaks the real binary protocol.

## Architecture

A strict sequential flow between three loosely-coupled modules, with frozen dataclass contracts at every boundary:

```
Camera ─▶ Recognition ─▶ Planning ─▶ Execution ─▶ KUKA controller
         (Faster R-CNN)  (digital   (EKRL 3.1,
                          twin +     CRC-16)
                          FFDH)
```

Each arrow is a well-defined type: `Snapshot` from recognition, `PickPlacePose` from planning, `RobotStatus` back from execution. See the Final Design Report in `docs/FDR.md` for the full rationale and trade-off analysis.

## Repository layout

```
auto-pick/
├── common/           Shared types (BBox, Detection, PickPlacePose, ...)
│                     and small utilities (YAML config loader, logger)
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
│   ├── calibrate_tau.py    Calibrates the arbitration disagreement threshold
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
│   ├── bin_packing.py      Shelf-based FFDH 2-D strip packer
│   └── planner.py          Orchestrator emitting PickPlacePose queues
├── execution/        Execution module
│   ├── protocol.py         16-byte binary packet + CRC-16/MODBUS
│   ├── execution.py        KukaClient (TCP, retries, E-stop)
│   ├── mock_kuka_server.py Software-only KUKA simulator
│   └── krl_prog/           Reference KRL routines for the real controller
├── configs/          YAML configs per module + top-level demo.yaml
├── tests/            Pytest suite (~100 tests)
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

The loop logs per-cycle perception / planning latencies, the queue length, and a summary of placed / pick-failed / place-failed counts at exit.

## Two synthetic generators, two purposes

There are deliberately two synthetic-scene generators, and they are not interchangeable:

| | `recog/synth_dataset.py` | `recog/generate3d.py` |
| --- | --- | --- |
| Renderer | OpenCV rectangles | Blender / Cycles path tracer |
| Interpreter | system Python | Blender's bundled Python |
| Speed | ~ms per image | ~3.5 s per 1280×720 frame on an RTX 3060 |
| Fidelity | flat coloured primitives | real CAD geometry, randomised PBR materials, physical lighting |
| Used by | `main.py`'s software-only demo, the unit tests | training the Faster R-CNN detector |

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
* **`SegmentationPlacementAreaExtractor`** — the path for real imagery. It consumes a trained segmenter's per-pixel label map (`Snapshot.cartridge_masks`, populated by `recog.inference.attach_cartridge_masks`) and arbitrates two *independent* placement estimates via `plan/arbitration.py` — the network's own `bay` channel, and an estimate derived from the other channels it doesn't use. Their agreement (IoU) is a confidence signal, gated at a calibrated `tau` (`recog/calibrate_tau.py`, `configs/planning.yaml`); a cartridge whose two estimates disagree below `tau` is skipped for that cycle rather than planned on an untrustworthy mask.

The split exists because of a hard latency budget, not preference: FDR O3 caps planning's queue rebuild at 8 ms per cartridge, and a single segmenter forward pass alone costs roughly that much on its own. Segmentation therefore runs once per frame, batched, in Recognition (`recog/bay_segmenter.py`) — measured on an RTX 3060 at ~17-19 ms for 8 cartridges batched, against 76-101 ms for the same 8 run in a loop, well inside vs. well outside the separate 50 ms end-to-end budget respectively. Planning only ever does mask arithmetic on the already-computed masks (`plan/arbitration.py`, measured ~2 ms per cartridge). `tests/test_planner.py::test_planning_stays_under_the_o3_budget_with_masks_supplied` pins that arithmetic-only cost against the 8 ms budget.

## The real-photo held-out set

Validation mAP on renders only says the detector learned the renderer. `recog/realtest/` is the set that answers the real question: seven photographs of the actual cells and cartridges (3024×4032, phone camera), annotated in CVAT and exported as COCO — 80 boxes, 60 battery and 20 cartridge. It is committed to the repository, never trained on, and never used to select a checkpoint.

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
Calibrate the arbitration threshold tau | `python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml`
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

The full Final Design Report — requirements, literature review, detailed design, test strategy, risk assessment, and AHEP-4 learning-outcome mapping — is in `docs/FDR.md`.

## Authors

Yousif Al-Haidary, supervised by Dr Svetan Ratchev.
