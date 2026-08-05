# Blender synthetic dataset generator for the recognition module

Design spec — 2026-08-05

## 1. Motivation

The recogniser currently trains on [`recog/synth_dataset.py`](../../../recog/synth_dataset.py):
flat OpenCV rectangles and cylinders drawn with `cv2.rectangle`. It has no
perspective, no shading, no occlusion, no material variation, and its geometry
is invented rather than measured. A detector trained on it learns colour priors,
not shape.

Real annotated data exists but is tiny: seven phone photos in
`d:/dev/rb/recognition/ann_btt_ctrdge/` (COCO, `Battery` / `Cartridge`, 80
annotations total). That is a validation set, not a training set.

CAD does exist. `d:/dev/partsgen_pipeline/` holds a working Blender pipeline
built around four Anker PowerCore STEP assemblies, already converted to glTF
(`assets/*.glb` + `catalog.json`). It renders photoreal top-down scenes with
pixel-exact boxes derived from an object-index pass.

This spec ports that pipeline into auto-pick as a higher-fidelity replacement
for the cv2 generator, adapted to auto-pick's classes, annotation format and
configuration conventions.

## 2. Goals and non-goals

**Goals**

- Offline batch generation of a photorealistic, pixel-exact-labelled detection
  dataset from CAD, written as Pascal-VOC XML into a flat directory that
  [`recog/dataset.py`](../../../recog/dataset.py) already reads.
- Scene content matching the real deployment: loose 18650 cells, open Anker case
  shells with visible PCB, sealed units, on both randomized backdrops and a
  pocketed jig tray.
- All scene parameters — lighting rigs, backdrops, material palettes, sampling
  weights — tunable from YAML without touching Python.
- A tight tuning loop: render one fixed scene across every lighting rig into a
  side-by-side comparison sheet.
- Full reproducibility from `(seed, index)`.

**Non-goals**

- Replacing the image source in [`main.py`](../../../main.py). The end-to-end
  demo continues to use the cv2 generator, so
  [`plan/placement_area.py`](../../../plan/placement_area.py) and the planner
  stay working and tested. The two generators coexist.
- COCO output, or reading the seven real photos. VOC only.
- Live rendering during training.
- Any change to `execution/`. The only change to `plan/` is a behaviour-preserving
  move of the shelf-packing algorithm into `common/` (see §6), with
  `plan/bin_packing.py` re-exporting it so no caller changes.

## 3. Class mapping

The identifiers already agree, so no remapping layer is needed anywhere.

| CAD sub-part role | auto-pick class | id |
| --- | --- | --- |
| `cell` (18650, 18.3 × 18.3 × 65.0 mm) | `battery` | 1 |
| `case` (Anker shell, receives cells) | `cartridge` | 2 |

`CLASSES = ["battery", "cartridge"]` derives `battery=1, cartridge=2`, identical
to `recog.dataset.CLASS_MAP` minus the `background: 0` entry. A unit test asserts
this equality so the two vocabularies cannot drift.

**Roles stay named `cell` and `case`.** They describe CAD geometry and the
regexes in `CLASS_RULES` match the real sub-part names (`004695_A;1-Cell_18651`,
`004697_A;2-Case10000_top`). `Variant.label_roles` maps role to class. This means
`CLASS_RULES` is carried over unchanged.

The interpretation "cartridge = the Anker case shell that receives cells" is
confirmed against the real photos: in IMG_4426, `Cartridge` boxes cover the black
shells with internal cell ribs, and `Battery` boxes cover the loose cylinders.
The partsgen HANDOFF's claim that no cartridge geometry exists was written
without sight of those photos and is superseded.

## 4. Architecture

New package `recog/synth3d/`, preserving partsgen's bpy / no-bpy split. Four
modules import no Blender API, so the entire dataset logic is testable under
plain pytest.

```
recog/synth3d/
  __init__.py
  config.py       dataclasses, CLASSES, CLASS_RULES, VARIANTS, YAML loading   pure
  catalog.py      catalog.json load, role_of() regex classification           pure
  layout.py       scatter solver + jig pocket packing                         pure
  annotate.py     mask -> boxes -> VOC XML, group merging                     pure
  assets.py       glTF import, instancing, variant construction               bpy
  materials.py    randomized Principled surfaces from YAML palettes           bpy
  world.py        backdrop, jig plate, PCB, lighting, overhead camera         bpy
  render.py       Cycles config, object-index pass, mask readback             bpy
  scene.py        orchestration                                               bpy
  convert_cad.py  STEP -> glTF + catalog (offline, for future CAD)            pure
  assets/         AnkerPowerCore{10000,13000,20100,26800}.glb + catalog.json

recog/generate3d.py   entry point, runs under `blender -b`; renders sweep frames
recog/verify3d.py     system Python; tiles frames into contact / sweep sheets
configs/synth3d.yaml  all presets and sampling ranges
tests/test_synth3d.py pytest, no Blender required
```

**Do not add `import bpy` to config, catalog, layout or annotate.** That boundary
is what makes the test suite possible without a Blender install.

### 4.1 Port table

| Source | Destination | Change |
| --- | --- | --- |
| `partsgen/config.py` | `recog/synth3d/config.py` | `CLASSES` renamed; `MATERIALS`, `ROLE_MATERIALS`, `BACKDROPS`, `LIGHTING`, `PARAM_SPACE` move to YAML; dataclasses and `CLASS_RULES` unchanged |
| `partsgen/catalog.py` | `recog/synth3d/catalog.py` | unchanged |
| `partsgen/layout.py` | `recog/synth3d/layout.py` | adds `Pocket`, `plan_jig()` |
| `partsgen/annotate.py` | `recog/synth3d/annotate.py` | COCO writer replaced by VOC writer; `split_of`, `coco_skeleton`, `append_to_coco` dropped |
| `partsgen/assets.py` | `recog/synth3d/assets.py` | unchanged |
| `partsgen/materials.py` | `recog/synth3d/materials.py` | palettes read from YAML; purple and grey cell wraps added |
| `partsgen/world.py` | `recog/synth3d/world.py` | adds `build_jig()`, `build_pcb()`; camera made aspect-aware |
| `partsgen/render.py` | `recog/synth3d/render.py` | unchanged |
| `partsgen/scene.py` | `recog/synth3d/scene.py` | dispatches on `layout_mode` |
| `generate.py` | `recog/generate3d.py` | VOC output, flat directory, `--sweep` |
| `verify_boxes.py` | `recog/verify3d.py` | reads VOC; gains sweep sheet |
| `convert_cad.py` | `recog/synth3d/convert_cad.py` | unchanged |
| `test_partsgen.py` | `tests/test_synth3d.py` | pytest idiom, three new tests |
| `frcnn_data.py`, `train_frcnn.py`, `smoke_test.py` | dropped | superseded by `recog/dataset.py`, `recog/training.py`, `recog/evaluate.py` |

## 5. Scene content

### 5.1 Variants

Three presentations, sampled by weight. Each maps onto something visible in the
real photos.

| Variant | Roles kept | Labelling | Weight | Real-world analogue |
| --- | --- | --- | --- | --- |
| `assembled` | cell, case | one merged `cartridge` box | 3.0 | sealed black shells, lower half of IMG_4426 |
| `cells_only` | cell | per-cell `battery` boxes, 30 mm explode | 2.0 | loose 18650s in the top rows |
| `open_case` | cell, case | `cartridge` + `battery` boxes, 45 mm explode | 1.0 | opened units in the middle pockets |

In `assembled`, cells are sealed inside the shell and contribute zero pixels to
the index pass, so they never appear in `np.unique(mask)` and are dropped with no
special-case code. Occlusion needs no handling anywhere in the pipeline.

### 5.2 Procedural PCB

The CAD contains no PCB, but the green board is the single most distinctive
feature inside an open case in the real photos. `world.build_pcb()` adds a green
rounded plane with a handful of extruded block components, parented into the
`open_case` variant.

It is **unlabelled**: `pass_index = 0`, so it merges with background in the index
map and produces no annotation, while still correctly occluding whatever sits
behind it. This shrinks the case's visible silhouette exactly as a real PCB
would, and stops the detector learning that cartridge interiors are uniformly
black.

### 5.3 Materials

Carried over, with the cell wrap palette widened to include the purple and grey
wraps visible in IMG_4426 alongside the existing green, blue, black and nickel.
All ranges live in `configs/synth3d.yaml` and are drawn per instance, so no two
renders share a surface.

Materials are assigned to the **object**, not the mesh. Instances share mesh
data; assigning to `obj.data` would recolour every instance together.

## 6. Layout

`layout.py` stays pure and gains a second mode. The packing decision is computed
in plain Python; Blender only realises the result as geometry.

```python
plan(footprints, cfg, rng)     -> list[Placement | None]
plan_jig(footprints, cfg, rng) -> tuple[list[Placement | None], list[Pocket]]

@dataclass(frozen=True)
class Pocket:
    x: float; y: float; w: float; h: float; depth: float   # metres, plate-local
```

**Scatter** (existing): rejection-sampled non-overlapping placement with
rotations restricted to k·90° ± 2°. That restriction is not cosmetic — it keeps
every footprint axis-aligned, so the overlap test is an exact AABB comparison and
non-overlap is guaranteed rather than approximated.

**Jig**: shelf-packs the given footprints, then emits one `Pocket` per placed
footprint, inflated by `jig_clearance`. Deriving pockets *from* the parts rather
than packing parts into a fixed grid guarantees fit by construction.

Within jig mode there is one special case, not a separately sampled mode: a
`cells_only` item contributes a single pocket sized for all its cells, and the
cells are laid inside it in regular rows rather than each getting its own pocket.
That reproduces the top-row photo, where seven cells sit in one tray recess.

`world.build_jig(pockets, rng)` extrudes the plate: a slab with recessed
pockets, blue plastic, randomized fillet radius and layer-line bump. The plate is
unlabelled (`pass_index = 0`), like the PCB.

`layout_mode` is sampled per scene, weighted **0.7 scatter / 0.3 jig**. Scatter
dominates because free-scattered parts on randomized backdrops force the detector
to learn shape rather than context; the jig minority supplies in-distribution
coverage of the deployment scene.

### Decision: lift shelf packing into `common/packing.py`

Jig pocket packing is shelf packing, and
[`plan/bin_packing.py`](../../../plan/bin_packing.py) already contains a tested,
generic shelf-based FFDH implementation — plain floats, optional forbidden mask,
no dependency on `plan.scene`. Duplicating it would be worse than reusing it.

Reusing it directly, however, would create an import edge
`recog.synth3d.layout -> plan.bin_packing`: a back-edge against the runtime data
flow (recog feeds plan) across the module boundary the README emphasises.

So the algorithm moves rather than being imported across:

- **`common/packing.py`** (new) receives `Item`, `PackedItem`, `PackResult`,
  `first_fit_decreasing`, `_try_place_item`, `_overlaps_forbidden`. numpy only —
  consistent with `common/`'s stated "no heavy dependencies" rule.
- **`plan/bin_packing.py`** re-exports those four public names and keeps
  `pack_cartridge`, which genuinely depends on `plan.scene.Cartridge`.
- **`recog/synth3d/layout.py`** imports from `common.packing`.

Both modules then depend only on `common/`, which both already depend on. No new
cross-module edge, no duplication, one tested implementation.

This is behaviour-preserving and requires no edits to callers:
[`plan/planner.py:28`](../../../plan/planner.py) and
[`tests/test_bin_packing.py:18`](../../../tests/test_bin_packing.py) both import
`Item, PackedItem, PackResult, first_fit_decreasing` from `plan.bin_packing`, and
the re-export keeps those imports valid. `tests/test_bin_packing.py` must pass
unchanged — that is the acceptance criterion for the move.

`first_fit_decreasing` documents its units as millimetres while synth3d works in
metres; `plan_jig` converts at the boundary (×1000 in, ÷1000 out) rather than
changing the shared function.

## 7. Configuration and tuning

### 7.1 `configs/synth3d.yaml`

Backdrops, lighting rigs, material palettes, and the sampling space move out of
Python into YAML, loaded through the existing [`common/config.py`](../../../common/config.py)
loader — matching how every other auto-pick module is configured. `config.py`
retains only the dataclasses and `CLASS_RULES`.

```yaml
render:   {res: [1280, 720], samples: 192, denoise: true, device: GPU, view_transform: AgX}
layout:   {area: [0.80, 0.45], pad: 0.008, jitter_deg: 2.0, jig_clearance: 0.004}
camera:   {ortho: true, height: 0.90, margin_range: [1.02, 1.10]}
filter:   {min_px: 80, min_side: 6, min_visibility: 0.25}

param_space:
  n_assemblies: [1, 4]
  layout_mode:  {scatter: 0.7, jig: 0.3}
  backdrop:     [concrete, brushed_metal, fabric, paper, conveyor_belt]
  lighting:     [overcast_softbox, harsh_inspection, warm_indoor]

lighting:
  overcast_softbox:
    kind: camera_softbox
    energy: [120.0, 260.0]
    size: [0.9, 1.6]
    kelvin: [5600, 6800]
    world_strength: [0.25, 0.55]
  # ... one block per rig
```

Adding a lighting rig or shifting a colour temperature is a YAML edit. Unknown
keys are rejected at load with a clear error rather than silently ignored.

**Blender's bundled Python has no PyYAML** (verified on the installed 5.0.0:
numpy present, `yaml` and `PIL` absent). Since `config.py` is imported inside
Blender, YAML cannot be the format read at runtime. So:

- `configs/synth3d.yaml` is the authored source of truth.
- `python -m recog.sync_config` (system Python, which has PyYAML — `common/config.py`
  already depends on it) transcribes it to `configs/synth3d.json`.
- `config.load_config()` uses `yaml` when importable and the JSON sidecar
  otherwise. If the sidecar is missing or older than the YAML it raises,
  naming the exact command to run. Stale config fails loudly, never silently.

### 7.2 CLI overrides

Any sampled axis can be pinned for a run, which is how you isolate one condition:

```
--lighting harsh_inspection  --backdrop concrete  --variant open_case
--layout-mode jig  --seed N  --res W H
```

### 7.3 Sweep mode

```bash
blender -b --python recog/generate3d.py -- --sweep lighting --seed 7 --out sweeps/
python -m recog.verify3d --sweep sweeps/ --out sweeps/lighting_sheet.png
```

`generate3d.py --sweep <axis>` renders **one fixed scene** — same seed, same
layout, same materials, same camera — once per entry in that axis.
`verify3d.py --sweep` then tiles those frames into a single labelled comparison
sheet. Edit YAML, re-run, see every rig side by side. This is the tuning loop;
without it, lighting work is guesswork.

Two commands rather than one because tiling needs Pillow, which Blender's bundled
Python does not ship. Blender renders; system Python composites. The same split
already applies to the ordinary contact sheet.

Holding the scene fixed while varying only the swept axis is the whole point, so
the RNG must be drawn once and reused across the sweep rather than re-seeded per
entry.

### 7.4 Interactive look-dev

`--save-blend look.blend` writes the built scene to disk. Open it in the Blender
GUI, tune lights interactively, copy the numbers back into YAML.

## 8. Output contract

```
recog/dataset3d/
  images/scene_00000.png         flat — matches BatteryCartridgeDataset
  annotations/scene_00000.xml    Pascal-VOC: battery / cartridge
  meta/scene_00000.json          full parameter draw for this sample
  masks/scene_00000.png          instance ids, only with --save-masks
  manifest.json                  config, catalog, box-size statistics
```

Flat, not split into train/val/test: [`recog/training.py`](../../../recog/training.py)
already performs a seeded `random_split` over one `images/` + `annotations/`
pair. partsgen's split logic is therefore dropped rather than ported.

VOC XML matches the dialect `parse_voc_xml` reads: `<filename>`, `<size>`,
and one `<object>` per instance with `<name>` and `<bndbox>`.

`meta/` carries backdrop, lighting rig, variant, asset and the full material draw
per scene. This is what makes it possible to slice validation mAP by rendering
condition and find where the model is weak.

### Invariants

These four fail silently rather than loudly. Preserve them.

1. **The index pass must render at `cycles.filter_width = 0.01`, 1 sample,
   denoising off.** At Blender's default 1.5 px reconstruction filter,
   neighbouring object indices are blended at silhouette edges into fractional
   ids that decode to the wrong instance. Note the property is
   `scene.cycles.filter_width`, **not** `scene.render.filter_width` — the latter
   does not exist in Blender 5.0 (see §11.1).
2. **Box max edges are exclusive.** A one-pixel object yields a 1×1 box. A
   zero-area box makes Faster R-CNN's box regression loss go NaN.
3. **Blender's pixel buffer is bottom-up; image files are top-down.**
   `render.read_index_exr` applies `np.flipud`. Any new pixel readback must too.
4. **Labels are 1-based**; 0 is the background head. `num_classes = 3`.

## 9. Integration

One edit to [`configs/recognition.yaml`](../../../configs/recognition.yaml):

```yaml
dataset:
  img_dir: recog/dataset3d/images
  ann_dir: recog/dataset3d/annotations
```

Nothing in `recog/`, `plan/`, `execution/` or `main.py` changes. `synth_dataset.py`
stays exactly as it is and continues to serve `main.py`'s demo.

Anchor sizing is worth one check after the first real run. `layout.area` is set
to `[0.80, 0.45] m` to match the 16:9 render aspect, so the camera frames the
whole layout region with no wasted margin. At that scale a 180 mm power bank
spans 40% of the 720 px short axis, roughly 290 px, and its box diagonal lands
near 350 px — inside the FPN default anchor range of 32–512 px.
`generate3d.py` writes `box_diag_px` p05/p50/p95 into `manifest.json`; if p95
approaches 512 the range is being exceeded, and the fix is to enlarge `area` or
widen `anchor_scales` in `configs/recognition.yaml`.

## 10. Testing

`tests/test_synth3d.py`, pytest, no Blender required. Ports partsgen's 22 checks:

- role classification against all 33 real CAD sub-part names
- 300 layout scenes with real asset footprints: zero overlaps, zero
  out-of-bounds, rotation constraint always satisfied
- mask → box correctness, exclusive edges, silhouette area
- the sealed-cell case: a cell inside an assembled shell yields no annotation
- box merging into a single assembly box
- truncation flagging and small-instance filtering

Plus three new tests specific to this integration:

1. **VOC round-trip.** Generated XML parses through the real
   `recog.dataset.parse_voc_xml` and yields back the boxes and labels that went
   in. Tests the actual contract, not a reimplementation of it.
2. **Class vocabulary agreement.** `synth3d.config.class_ids()` equals
   `recog.dataset.CLASS_MAP` minus `background`. Catches drift at import time.
3. **Jig packing.** Every pocket contains its item with at least `jig_clearance`
   margin; no two pockets overlap; unplaced items return `None`.

The bpy modules stay untested here — they need a Blender process. `verify3d`'s
contact sheet is the verification mechanism for those, and inspecting it is a
mandatory step, not an optional one.

## 11. Running it

```bash
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"

# small dev run
"$BLENDER" -b --python recog/generate3d.py -- --n 20 --out recog/dev3d --res 640 360

# inspect — do not skip
python -m recog.verify3d --data recog/dev3d --n 12

# tune lighting
"$BLENDER" -b --python recog/generate3d.py -- --sweep lighting --seed 7 --out sweeps/

# full run
"$BLENDER" -b --python recog/generate3d.py -- --n 2000 --out recog/dataset3d \
        --device GPU --resume
```

Per sample the order is: build scene, **index pass** (1 spp), annotate, **beauty
render** (full Cycles). Labels are produced before the expensive render, so
`--no-render` yields a complete annotation pass in seconds — use it when
iterating on layout or class rules.

**Target Blender 5.0.0** (Python 3.11.13). The `Blender 4.2` directory on this
machine contains only a leftover config folder with no `blender.exe`; 5.0 is the
only runnable install. partsgen was written against 4.2–4.5, so the port must
carry the API deltas in §11.1.

Because the pure modules are imported by both Blender's Python 3.11 and the
system Python 3.14, they must stay **3.11-compatible**: no PEP 695 generics, no
`type` statements.

Hardware on this machine: RTX 3060, 16 cores. `enable_gpu()` selects OPTIX and
the card is detected. Expect a few seconds per frame, so ~2000 images is a couple
of hours, and `--resume` makes it interruptible.

### 11.1 Blender 5.0 API deltas — verified, not guessed

partsgen's `render.py` fails on 5.0 in four places. All four were reproduced
against the installed build and the replacements confirmed working.

| partsgen (4.2) | Blender 5.0 | Consequence if unported |
| --- | --- | --- |
| `scene.render.filter_width = 0.01` | `scene.cycles.filter_width = 0.01` | `AttributeError`; the property does not exist. `scene.render.filter_size` also exists but Cycles reads its own. |
| `scene.node_tree` | `scene.compositing_node_group`, assigned a tree from `bpy.data.node_groups.new(name, "CompositorNodeTree")` | `AttributeError` on every index pass |
| `rl.outputs["IndexOB"]` | `rl.outputs["Object Index"]` | `KeyError`; socket renamed |
| `image_settings.file_format = "OPEN_EXR_MULTILAYER"` | removed from the enum | n/a — the File Output node path is used instead, which still works |

The dangerous one is the second. **`scene.use_nodes = True` still succeeds
silently in 5.0**, so nothing fails until `scene.node_tree` is dereferenced. Code
that merely sets `use_nodes` looks correct and is not.

Port both spellings behind `hasattr`/`getattr` guards rather than hard-coding 5.0
names, so the pipeline still runs if someone installs 4.2 later. `materials.py`
already uses this idiom for the Principled `Coat Weight` / `Clearcoat` rename,
and that rename is confirmed present on 5.0 (`Coat Weight`).

## 12. Known gaps and risks

**Environment, outside this design's scope but blocking on first run:**

- `cv2` is not installed in the active Python (3.14.3). Measured effect: `pytest -q`
  aborts with 4 collection errors (`test_inference`, `test_main_integration`,
  `test_placement_area`, `test_planner`). Excluding those four,
  **85 pass and 1 skips** — that is the green baseline this work must preserve.
  `pip install opencv-python` fixes it (5.0.0.93 is on PyPI). `verify3d` needs
  only Pillow and numpy, so it is unaffected.
- `torch` is a CPU-only build (`2.13.0+cpu`), so training will not use the 3060.
  Needs a CUDA wheel reinstall.
- `cascadio` and `trimesh` are absent, needed only by `convert_cad.py`. Not on
  the critical path — the four `.glb` files are already converted.

**Technical risks, in rough order of likelihood:**

1. **Camera aspect.** partsgen's ortho camera framing assumes a square render and
   a square layout area. This design uses 1280 × 720 with a `[0.80, 0.45] m`
   area, so `world.setup_camera` needs explicit aspect handling — set
   `ortho_scale` from the longer layout axis and let Blender derive the shorter
   one from the render aspect. Get this wrong and parts are cropped or the frame
   is mostly empty backdrop. Verify on sample 0 before any long run.
2. **Compositor File Output naming.** `render_index_map` expects
   `{stem}_0001.exr` and falls back to a directory scan. The 5.0 compositor
   rewrite (§11.1) is handled, but the written filename is the remaining
   unverified part of that path — keep the directory-scan fallback.
3. **glTF orientation.** `assets.lay_flat()` rotates each part so its smallest
   extent runs along Z. Check with `--save-blend` on sample 0: power banks should
   lie flat and cells on their sides, not upright.
4. **Jig realism.** A procedurally generated blue plate will not match the real
   3D-printed fixture exactly. It is domain randomization, not a digital twin,
   and only 30% of scenes use it.
5. **Sim-to-real gap is unmeasured.** With COCO reading out of scope, there is no
   number in this design for how a synthetic-trained detector performs on the
   seven real photos. That measurement is the obvious follow-up.

## 13. Out of scope, and the natural follow-ups

Deliberately excluded here, listed so they are not forgotten:

- **Reading the seven real photos as a held-out real-image test set** (needs a
  COCO reader in `recog/dataset.py`). Decided out, deliberately: the format and
  scope choices for this work were VOC-only and training-data-only, it blocks
  nothing here, and folding it in widens an already-large plan. It remains the
  single most valuable follow-up — mAP on real photos from a model trained only
  on synthetic renders is the result that justifies the whole exercise — and it
  is cleanly separable as its own spec.
- Exporting a ground-truth placement-area mask from CAD so
  `plan/placement_area.py` can drop its green-channel Otsu heuristic.
- Live rendering during training.
- Modelling the blue jig from measured dimensions rather than procedurally.
