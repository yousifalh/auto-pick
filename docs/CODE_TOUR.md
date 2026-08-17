# A reading route through this codebase

~19,000 lines of source across 56 modules. Read in dependency order, not directory order, and
follow **one frame** all the way through before going deep on anything.

**The technique that matters here:** read every module's docstring before its code. This codebase
puts the *reasoning* in docstrings and comments rather than in a wiki — `common/types.py` is 408
lines for five dataclasses, and the ratio is the point. Several files are arguments with code
attached.

**Second technique:** the tests are the specification. `tests/conformance.py` defines what a robot
driver *is*, more precisely than `driver.py` does.

---

## Session 0 — the contracts (30 min)

Everything else moves these objects around. Read them first and the rest stops being surprising.

| File | Lines | What to take away |
|---|---|---|
| `common/types.py` | 408 | `BBox`, `Detection`, `Snapshot`, `PickPlacePose`, `RobotStatus`, `ClassLabel`. Note the box convention comment: 0-based, exclusive max — *not* VOC's. Note why `Snapshot` is deliberately **not** frozen. |
| `common/config.py` | 216 | `load_yaml`, `load_demo_config`, and the validators that **refuse** a config key nothing reads. |

**Question to answer before moving on:** what exactly crosses each of the three module boundaries?

---

## Session 1 — the spine (1 hour)

| File | Lines | Read |
|---|---|---|
| `main.py` | 708 | `run()`, top to bottom. Skim the builders on the way past. |

Trace a single cycle and write the sequence down yourself:

```
frame + its own mm_per_px          _synthetic_source
  → detector.detect(rgb)           → Snapshot (boxes)
  → attach_cartridge_masks(...)    → Snapshot (+ label maps)
  → planner.update(snapshot)       → PickPlacePose queue
  → client.execute(pose)           → RobotStatus
```

**The three builders are where configuration becomes behaviour** — `_build_segmenter`,
`_build_planner`, `load_detector`. Note that `mode.segmentation` selects the segmenter *and* the
segmentation extractor together, and why splitting them would be silent.

After this session you should be able to answer: *where does the loop decide to stop?*

---

## Session 2 — Recognition (2 hours)

Read in this order — the wrappers are small, the interesting parts are the seams.

| File | Lines | Why it's here |
|---|---|---|
| `recog/model.py` | 138 | Faster R-CNN factory. Small — the architecture is torchvision's. |
| `recog/bay_segmenter.py` | 131 | **Read the module docstring in full.** The whole precision/latency ablation is in it. |
| `recog/inference.py` | 358 | The `Detector` base, both implementations, `attach_cartridge_masks`, `load_detector`. |
| `recog/calibration.py` | 157 | Per-frame ground sample distance. Small file, large consequences. |

**Seams to understand, not skim:**
- `attach_cartridge_masks` keys by *index into `snapshot.detections`*, not by position in the
  cartridge subset. The docstring says why.
- `load_detector` falls back to `HeuristicDetector` when there is no checkpoint — that fallback is
  what makes the demo torch-free, and it is also what once made a README figure unreproducible.
- `weights_only=True` on every `torch.load`. Find the one exception and its justification.

---

## Session 3 — Planning (3–4 hours, the densest part)

This is where your design decisions actually live. Do not start it tired.

| Order | File | Lines | Focus |
|---|---|---|---|
| 1 | `plan/arbitration.py` | 163 | Smallest file, biggest idea. `P_safe = P_direct ∩ P_derived`, and no threshold. |
| 2 | `plan/placement_area.py` | 742 | The two extractors behind one contract. `_resolve_scale` is the honest scale guard. |
| 3 | `plan/bin_packing.py` | 56 | Tiny — it re-exports. The frozen FFDH matters more than its size suggests. |
| 4 | `common/packing.py` | 589 | `pack_best_effort`: three arms, max wins, ties to FFDH. |
| 5 | `plan/scene.py` | 933 | The digital twin. Tracking, reservations, `visible_cartridges()`. |
| 6 | `plan/planner.py` | 1009 | The orchestrator that ties all of the above together. |

**The five things to be able to explain afterwards:**
1. Why an occupancy cell is FREE only if *all* of it is free — and why that's also faster.
2. Why `first_fit_decreasing` is frozen. (`recog/synth3d` lays out training scenes with it.)
3. Why ties go to FFDH, and what that guarantees *by construction*.
4. Why only `visible_cartridges()` may generate a pose.
5. Why greedy matching over globally sorted pairs, and not Hungarian.

---

## Session 4 — Execution (2 hours)

The most self-contained module, and the best-designed one. Good session to do when short on time.

| Order | File | Lines | Focus |
|---|---|---|---|
| 1 | `execution/protocol.py` | 348 | The 16-byte frame and CRC-16/MODBUS. Read the struct format characters one at a time. |
| 2 | `execution/task.py` | 116 | Small. What a unit of work is. |
| 3 | `execution/driver.py` | 961 | The sealed template method. Find `__init_subclass__` and the list of sealed names. |
| 4 | `execution/execution.py` | 431 | `KukaClient` — one implementation. |
| 5 | `execution/json_driver.py` | 322 | The second implementation. Different encoding, same suite. |
| 6 | `tests/conformance.py` | — | **The actual specification.** Read it as a document. |
| 7 | `execution/mock_kuka_server.py` | 393 | What the other end does, including the deliberate drop probability. |

Then read `execution/krl_prog/*.src` — it's KRL, not Python, and it's what would run on the
controller.

---

## Session 5 — Synthetic data (3 hours, the largest surface)

Save for last: it is the biggest subsystem and nothing else depends on reading it.

**Read the pure modules first — they are testable and they hold the decisions:**

| File | Lines | |
|---|---|---|
| `recog/synth3d/config.py` | 330 | Parameter space. |
| `recog/synth3d/catalog.py` | 309 | Class rules, the geometric liner split. |
| `recog/synth3d/bay.py` | 1016 | Interior geometry, `needs_flip`, the seating ladder. **The most decision-dense file in the project.** |
| `recog/synth3d/layout.py` | 229 | Scene layout, the ±2° jitter that the certifiability finding turns on. |

**Then the impure ones — they import `bpy` and cannot be unit-tested:**

| File | Lines | |
|---|---|---|
| `recog/synth3d/assets.py` | 584 | `lay_flat`, `flip_if_inverted`. Where defect 1 lived. |
| `recog/synth3d/world.py` | 1448 | The largest file in the repo. Scene construction. |
| `recog/synth3d/materials.py` | 227 | Where the swallowed exception fed noise into Roughness. |
| `recog/synth3d/annotate.py` | 589 | The hand-rolled column-major COCO RLE. |
| `recog/synth3d/render.py` | 399 | Cycles settings, the object-index pass. |
| `recog/generate3d.py` | 487 | The Blender entry point. |

**The boundary is the lesson:** decisions in the pure modules, construction in the `bpy` ones, and a
test enforces the line. Find that test.

---

## Session 6 — Training and evaluation (2 hours, optional)

Read only if you want to answer questions about how the numbers were produced.

`recog/seeding.py` (511) · `recog/seg_dataset.py` (387) · `recog/seg_training.py` (583) ·
`recog/seg_evaluate.py` (1090) · `recog/training.py` (453) · `recog/evaluate.py` (245) ·
`recog/augmentation.py` (306)

`seg_evaluate.py` is worth it on its own: it is where every published segmenter figure comes from,
and it carries the per-crop scale conversion that superseded the nominal-mm/px figures.

---

## The scar-tissue tour (45 min, do this last)

Nine places where the code encodes a defect that actually happened. Read the comment above each and
you will have most of the project's history.

| Where | What it remembers |
|---|---|
| `recog/synth3d/assets.py` — `lay_flat` / `flip_if_inverted` | Every cartridge rendered upside down and closed |
| `recog/synth3d/bay.py` — `needs_flip` | Why the *decision* was moved out of the bpy module |
| `recog/synth3d/materials.py` | A swallowed `except` that changed 100% of surfaces |
| `plan/placement_area.py` — the τ paragraphs | A gate that could not work, and why the intersection stayed |
| `common/packing.py` — `pack_best_effort` | A 93%-free grid that packed zero cells |
| `recog/calibration.py` | One scalar wrong in three places, all in the same direction |
| `execution/driver.py` — `__init_subclass__` | Sealing 17 names at class-definition time |
| `plan/scene.py` — the matching function | Why not Hungarian |
| `recog/synth3d/_gate_orientation.py` | The check that would have caught defect 1, now in CI |

---

## Rough budget

| Session | Hours |
|---|---|
| 0–1 contracts + spine | 1.5 |
| 2 recognition | 2 |
| 3 planning | 3–4 |
| 4 execution | 2 |
| 5 synthetic data | 3 |
| 6 training/eval | 2 |
| scar tissue | 0.75 |

**~14 hours for the whole thing.** Sessions 0, 1, 3 and 4 are the ones an interviewer will probe —
about 7 hours, and they cover the parts you designed.
