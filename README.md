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
│   ├── dataset.py          Pascal VOC loader + BatteryCartridgeDataset
│   ├── augmentation.py     Albumentations pipeline + numpy fallback
│   ├── training.py         Training loop (cosine LR, frozen BN, ckpts)
│   ├── inference.py        FasterRCNNDetector + HeuristicDetector
│   ├── evaluate.py         Pure-numpy VOC 11-point mAP + pose errors
│   └── synth_dataset.py    Procedural synthetic-scene generator
├── plan/             Planning module
│   ├── scene.py            Digital twin (entities + occupancy grids)
│   ├── placement_area.py   Green-channel + PCB-aware valid-area extractor
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

## Other entry points

Task | Command
--- | ---
Train Faster R-CNN (needs torch + GPU) | `python -m recog.training --config configs/recognition.yaml`
Start the mock KUKA server standalone | `python -m execution.mock_kuka_server`
Generate synthetic scenes | `python -m recog.synth_dataset --out recog/dataset --n 200`
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
