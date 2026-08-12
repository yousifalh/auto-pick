# Audit E — silent failures

**Date** 2026-08-12 · **HEAD** `fa7a4f0` · **Scope** read-only. Nothing modified,
nothing staged. No Blender run, nothing rendered.

**Hunting** the project's characteristic defect: code that stops doing its job
*without erroring*, producing plausible-looking wrong output while a green suite
says nothing. Not "every defensive `try` is a bug" — the only class that counts
is *if this path is taken, does the system produce believable wrong output
rather than an error?*

Everything below is labelled **VERIFIED** (read end to end, and where marked,
measured by running the bpy-free half) or **SPECULATIVE** (the trigger is
plausible but I could not exercise it).

---

## Summary — top five, ranked by consequence × plausibility

| # | Site | What it does silently | Status |
|---|------|----------------------|--------|
| 1 | `plan/planner.py:402` | Marks **one** 1.5 mm occupancy cell per 18.5 × 65 mm battery, so the next cycle legally packs a cell ~17 mm into one already placed | **VERIFIED**, fires on every multi-cycle run; `tests/test_planner.py:80` cements it |
| 2 | `recog/synth3d/bay.py:647` | Frozen FFDH collapses on obstructed bays: 19–78 % of them seat **zero** cells, recorded as a legitimate zero | **VERIFIED + MEASURED**, live today; `tests/test_bay.py:657` cements it |
| 3 | `execution/execution.py:145-165` + `mock_kuka_server.py:71` | `pose.place.z_mm` never reaches the wire; the mock's third parameter is named `place_z` but receives the **pick** approach height, so it "inserts" at 60 mm and returns SUCCESS | **VERIFIED** |
| 4 | `recog/synth3d/materials.py:99-104` | A swallowed `except` discards both drawn `roughness` and `wear` and links raw 0→1 noise into Roughness — on 100 % of surfaces | **VERIFIED** by reading; trigger SPECULATIVE (Blender version) |
| 5 | `main.py:153-160` + `plan/scene.py:162` | `WorkspaceBounds` is parsed, stored and **never compared against anything** — a declared robot-workspace safety envelope that enforces nothing | **VERIFIED**; `tests/test_planner.py:171` passes with a deliberately tiny bound, which is the proof |

Runner-up, and the one I would fix first if #1–#5 were all resolved:
`recog/synth3d/render.py:219-222`, a swallowed `TypeError` that can re-tone-map
an entire dataset while `manifest.json` records the requested value.

**Two regions the brief expected to be dirty are genuinely clean** — the
synth3d config surface and `plan/arbitration.py` / `placement_area.py`'s scale
handling. See "Regions that are clean". Do not spend time there.

---

# Part A — the bpy blind spot

`recog/synth3d/{world,scene,assets,render,materials}.py` import bpy and are
confirmed outside pytest's reach by `tests/test_synth3d.py:126`. Everything in
this part is from careful reading unless a measurement is quoted.

## A1. The synthetic seated-cell packer collapses on obstructed bays *(overall #2)*

**File** `recog/synth3d/bay.py:637-659` (`seated_cell_poses`) → `common/packing.py:169`
(`first_fit_decreasing`). Call site `recog/synth3d/scene.py:343-365`.

**What has to be true.** An obstruction anywhere in the bay. That's it.

`first_fit_decreasing` pins its first shelf to `y = 0` and never scans its origin
in y, and `_next_free_x` collapses the shelf's whole row band with
`.any(axis=0)` — one blocked row poisons every column in the band. This is the
*exact* defect diagnosed in `docs/superpowers/specs/2026-08-11-packing-ceiling.md`
and fixed for the planner at `d6c46ac` by moving it to `pack_best_effort`.
`recog/synth3d` was deliberately left on frozen FFDH so the training corpus
would not be redrawn — but the *cost* of leaving it there was never measured.

It bites harder here than in the planner: an 18650 is 65.0 mm long and the CAD
bays are 65.0–140.8 mm deep, so on the two smaller SKUs the shelf's row band is
the **entire bay** and a single adhesive blob blocks its columns floor to ceiling.

**Measured.** Read-only probe reproducing `scene.py:343-357` call-for-call
against the committed `configs/synth3d.yaml` and
`recog/synth3d/assets/catalog.json`; 4000 seeded bays per SKU.

| SKU | bay (mm) | clean bays seating 0 | **obstructed bays seating 0** | cells seated / requested |
|---|---|---|---|---|
| AnkerPowerCore10000 | 54.9 × 65.0 | 0.0 % | **78.1 %** | 52 % |
| AnkerPowerCore13000 | 73.2 × 66.8 | 0.0 % | **54.9 %** | 61 % |
| AnkerPowerCore20100 | 54.9 × 135.2 | 0.0 % | **46.4 %** | 63 % |
| AnkerPowerCore26800 | 73.2 × 140.8 | 0.0 % | **19.3 %** | 72 % |

`obstruction.p_none = 0.40`, so 60 % of bays are obstructed. Swapping the same
call to the already-in-tree `pack_best_effort` recovers most of it —
26800 **19.3 % → 1.0 %**, 20100 **46.4 % → 16.8 %**, 13000 **54.9 % → 48.5 %** —
while 10000 is unchanged (its bay is exactly one cell tall, so no y-scan can
help; that part is real geometry, not a packer defect).

**Observable symptom.** None. Nothing raises. `scene.py:364` writes
`meta["seated_cells"] = {"asset": …, "n": len(item.seated_objects)}` — the
*outcome* count only. `want` and `len(local_seats)` are never recorded, so a
zero is indistinguishable from a legitimately-full bay, and
`bay.seated_cell_poses`'s own docstring pre-authorises it ("including a bay so
densely obstructed that none fit at all, which is correct behaviour, not a
failure").

**Would a test catch it?** No — the suite **actively asserts the defect**:
`tests/test_bay.py:657 test_dense_obstruction_coverage_yields_fewer_or_no_seated_cells_without_raising`
pins "zero seated cells is fine". `tests/test_packing_ceiling.py` exercises
`pack_best_effort`, which this call site never reaches.

**Consequence.** The corpus encodes a spurious correlation: *obstruction present
⇒ bay empty*. Nearly half of all requested seated cells are never built, and the
missing ones are concentrated precisely on the obstructed bays — inverting the
confusion the `obstruction` class was added to prevent (`annotate.py:441-459`).
`placement_area` ground truth in those frames labels the whole bay floor free
while the scene was supposed to show a partly-filled one.

**Fix (one line).** Record `want`/`len(local_seats)` alongside `n` at
`scene.py:364` so a zero is attributable — *then* decide whether to move
`bay.py:647` to `pack_best_effort` and accept the corpus redraw.

## A2. A swallowed `except` silently replaces every material's roughness *(overall #4)*

**File** `recog/synth3d/materials.py:96-104`.

```python
mix = nt.nodes.new("ShaderNodeMix")
mix.data_type = "FLOAT"
mix.inputs["Factor"].default_value = drawn["wear"]
try:
    mix.inputs[2].default_value = drawn["roughness"]
    nt.links.new(ramp.outputs["Color"], mix.inputs[3])
    nt.links.new(mix.outputs[0], bsdf.inputs["Roughness"])
except Exception:
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
```

**What has to be true.** `ShaderNodeMix`'s positional socket indices 2/3 (the
FLOAT A/B pair) move, or the socket set differs on the running build.
`ShaderNodeMix` is the node that *replaced* `ShaderNodeMixRGB` in 4.0 and carries
several typed socket pairs behind one interface — index-based access to it is
exactly the thing that has moved before. **SPECULATIVE** on whether it fires;
**VERIFIED** on what happens if it does.

**Observable symptom.** None. The fallback links the raw noise ramp (positions
0.35/0.75, a 0→1 swing) *directly* into `Roughness`. Both `drawn["roughness"]`
and `drawn["wear"]` stop reaching the shader entirely: every surface renders
with extreme, high-contrast, all-or-nothing specularity instead of a
wear-weighted mix. **Every preset in `configs/synth3d.yaml` has `wear ≥ 0.05 >
0.01`**, so this branch runs for every material on every object — all-or-nothing
across 100 % of the dataset, not a rare edge.

The receipt then lies: `meta["materials"]` records the drawn `roughness`/`wear`
that were discarded, and the whole `MIN_LUMA_DELTA` table (`materials.py:22-39`,
measured from renders) was calibrated under the correct path, so the contrast
gate would silently be scoring against the wrong appearance too.

**Would a test catch it?** No — bpy-only. `_gate_orientation.py` checks
transforms and nothing else (verified: its ten `check(...)` calls are all bbox
extents and placement poses).

**Fix.** Set the sockets by identifier rather than index and let a failure raise —
a fallback that discards two of the three drawn parameters is not a degraded
render, it is a different one.

## A3. The view transform can be silently discarded for a whole dataset

**File** `recog/synth3d/render.py:219-223`.

```python
try:
    s.view_settings.view_transform = cfg.view_transform
except TypeError:
    pass
s.view_settings.exposure = cfg.exposure if exposure is None else exposure
```

**What has to be true.** `cfg.view_transform` (`"AgX"`) is not a valid enum item
on the running build — Blender raises `TypeError` on an invalid enum assignment,
which is exactly what this catches. The valid set has moved twice recently
(Filmic removed in 4.0, AgX added in 4.0, Khronos PBR Neutral later); this
module already carries four documented helpers for the same class of drift.
**SPECULATIVE** on the trigger, **VERIFIED** on the consequence.

**Observable symptom.** The scene still renders. The exposure on the very next
line is applied *regardless*, and `param_space.exposure` samples −5.2 … −3.2 — a
band tuned specifically against AgX's response (`configs/synth3d.yaml`'s measured
clipping/spread table). Under a different transform that band is simply wrong and
every image in the run is uniformly mis-tone-mapped: plausible pictures, wrong
distribution. `manifest.json` records `cfg.to_dict()` — the *requested* `AgX`,
never the applied value. Nothing anywhere records what the shader actually used.
`--sweep` sheets look internally consistent because every entry is wrong the
same way.

**Fix.** Assert `s.view_settings.view_transform == cfg.view_transform` after the
assignment and record the effective value into `meta`.

## A4. `seat_cells` accepts N seats and returns zero without a word

**File** `recog/synth3d/world.py:1443-1447`.

```python
if not seats:
    return []
templates = (library._templates.get(asset) or {}).get("cell")
if not templates:
    return []
```

**What has to be true.** The asset's template dict has no `"cell"` role — one
rename away. `_load_template` (`assets.py:476`) re-derives every role from
`catalog.role_of(o.name)`, **discarding the authoritative `by_role` dict
`world.build_procedural_tray` already returned at `world.py:1031`**. If
`ProcCell_0` ever stops matching `CLASS_RULES`' `Cell[_ ]?\d+`, `role_of` falls
back to `ROLE_FALLBACK = "case"`, `_classify_case_liner` re-labels it
`case_liner` (its footprint differs from the case's by far more than the 1 %
margin, so `classify_case_parts` does *not* raise), the `open_case` variant drops
`case_liner`, and every procedural cartridge silently loses its cells — loose
*and* seated.

The lid version of this is protected (two same-footprint `case` objects make
`classify_case_parts` raise). **The cell version is not.**

**Observable symptom.** None. `seat_cells` throws away every entry in `seats`
and returns `[]`; `scene.py:364` records `n: 0`, which reads as a legitimately
full or heavily obstructed bay — i.e. **indistinguishable from A1 in the
receipt**.

**Would a test catch it?** Partially. `tests/test_bay.py`'s
`test_procedural_object_names_classify_correctly_via_role_of` pins the names, so
a rename fails. Nothing pins the `seat_cells` no-op itself; nothing covers the
CAD path.

**Fix.** `if seats and not templates: raise` — a caller that asked for N seats
and got 0 objects has hit a bug, not an empty bay.

## A5. A missing backdrop image renders generic grey and the sidecar says otherwise

**File** `recog/synth3d/world.py:136` and `150-161`. `drawn["source"]` is
computed from the **spec** (`"image" if spec["image"] else f"proc:{spec['proc']}"`)
while line 150 branches on `spec["image"] and os.path.exists(spec["image"])`. A
declared-but-missing image therefore renders procedurally — falling through
`.get(kind, 20.0)` and `_FALLBACK_PALETTE.get(kind, …)` to a generic grey if
`proc` is also absent — while the meta sidecar reports `source: "image"`.

Contrast `setup_lighting:278-283`, which handles the identical situation for a
missing HDRI by printing a warning and setting `drawn["hdri"] = None`. The
backdrop path is the inconsistent one. **Currently latent** — all five backdrops
have `image: null`; it goes live the moment anyone adds a photographic backdrop.

**Fix.** Compute `drawn["source"]` from the branch taken, and warn on a declared
image that is missing — mirroring the HDRI branch three functions down.

## A6-A10. Lower-ranked, still real

6. **`recog/synth3d/render.py:291-295`** — `img.colorspace_settings.name =
   "Non-Color"` under `except Exception: pass`, on the **index-pass EXR
   readback**: the one line between `np.rint(arr)` and a colour-managed float
   buffer. **SPECULATIVE on both halves** — I could not confirm whether a linear
   32-bit EXR's `Image.pixels` actually changes under the default colorspace on
   5.0 (it may be purely protective), nor how the assignment would fail. Flagged
   only because the consequence if it does is every mask decoding to wrong
   instance ids, and the `pass` guarantees nobody finds out. *Fix: let it raise
   here; the sibling at `:422` (mask PNG, diagnostic) can keep the `pass`.*

7. **`recog/synth3d/render.py:314-315`** — a mask whose resolution disagrees with
   `cfg.res` prints `[warn]` and continues. Boxes are then computed in the mask's
   pixel space while `write_voc_xml` stamps the *configured* `W, H` into the XML
   header: a coordinate-space lie with a correct-looking header. Low
   plausibility (both configure paths set `resolution_percentage = 100`).
   *Fix: raise.*

8. **`recog/synth3d/scene.py:220-224`** — `layout.plan` returns `None` for an item
   it could not place after `max_tries` and `scene.build` deletes it silently with
   no counter. `params["n_assemblies"]` vs `len(meta["items"])` is the only trace
   and nothing compares them.

9. **`recog/synth3d/world.py:305/335`** — `setup_lighting` branches on
   `spec["kind"]` with no `else`. A typo'd kind matches neither branch, so the
   scene is lit by world background alone: flat, shadowless, entirely plausible.
   `tests/test_synth3d.py:160` only asserts `off_axis` and `camera_softbox` each
   appear *somewhere*, so a typo on one of seven presets passes. Recoverable from
   the sidecar (`drawn["kind"]` records the typo).
   *Fix: `else: raise ValueError(f"unknown lighting kind …")`.*

10. **`recog/synth3d/scene.py:46-49`** — `reset_scene`'s `coll.remove(item)` under
    `except Exception: pass`. A surviving datablock renders into the next scene at
    `pass_index = 0`, i.e. unlabelled furniture that occludes real parts and
    shrinks their boxes; `_check_id_meta_covers_scene` only catches **non-zero**
    pass indices so it cannot see this. Low plausibility —
    `read_factory_settings(use_empty=True)` runs first.

---

# Part B — `plan/`, `execution/`, `common/`, `main.py`, `recog/` top level

## B1. One occupancy cell is marked per battery *(overall #1)*

**File** `plan/planner.py:400-402`.

```python
row, col = self._xy_mm_to_cell(ctg, placement.x, placement.y)
ctg.mark_cell(row, col, CellState.PLANNED)
```

**VERIFIED end to end.** An 18.5 × 65 mm battery at
`occupancy_grid.resolution_mm_per_cell: 1.5` spans ~12 × 43 cells. Exactly one —
the top-left corner — is marked. `_pack_cartridge` (`planner.py:353-355`) rebuilds
`forbidden = ctg.occupancy.mask_of(FORBIDDEN, PLANNED, PLACED)` and re-packs on
**every** frame (`planner.py:195`), and `common/packing.py:126` blocks a
candidate only where its own cell block covers a marked cell.

**Trigger.** Any second planning cycle on a cartridge that already holds a
planned or placed cell. Fires on every multi-cycle run.

**Symptom.** The next pack legally seats an item one cell (1.5 mm) from the
previous one — ~17 mm of physical overlap with a battery already in the tray. The
queue looks normal, the `placed` counter increments, the mock returns SUCCESS.

**Test coverage.** `tests/test_planner.py:80` asserts
`planned_count() == len(queue)` — the suite **cements** the defect, and nothing
checks the marked region against the footprint. This is the fourth instance of
the pattern in the brief.

**Fix.** Mark the whole `ceil(w/res) × ceil(h/res)` block, not `(row, col)`.

## B2. `place.z_mm` never reaches the wire, and the mock's parameter name hides it *(overall #3)*

**Files** `execution/execution.py:145-165`, `execution/mock_kuka_server.py:71-108`,
`execution/protocol.py:38`.

The 16-byte command packet carries one xyz triple. `pick_and_place` sends
`MOVE_TO(place.x, place.y, transport_height)` then
`PICK_AND_PLACE(pick.x, pick.y, **pick.z**, aux)`. `execution.py`'s own docstring
says this is intentional — the place target is latched by the preceding MOVE_TO.
That part is honest.

**What is not honest** is the other end. `mock_kuka_server.pick_and_place`'s
third parameter is **named `place_z`**, and step 5 does
`self.move_to(place_x, place_y, place_z)` — so the mock "inserts" the battery at
whatever the client sent, which is `pick.z_mm` = `pick_approach_height_mm` = 60 mm,
not `place_insert_height_mm` = 2 mm. It then returns SUCCESS. **VERIFIED.**

Meanwhile `plan/planner.py:411-415` computes and returns
`place.z_mm = cfg.place_insert_height_mm`, `tests/test_planner.py:71` asserts it,
and `execution/krl_prog/routines.src` step 4 does `LIN place_pos` — needing a
place Z the wire format cannot deliver.

**Symptom.** The configured `insert_height_mm` has no effect anywhere in the
system, on the mock or on hardware. `tests/test_execution.py`'s pick-and-place
test asserts status and final **x/y** only (`≤ 1 mm`); z is never asserted.

**Fix.** Either extend the packet with a place-Z field, or delete `place.z_mm`
and rename the mock's parameter so it stops reading as load-bearing.

## B3. `WorkspaceBounds` is a safety envelope that enforces nothing *(overall #5)*

**Files** `main.py:153-160` (parse), `plan/scene.py:162-175` (type),
`plan/planner.py:129-133` (store).

**VERIFIED by grep across the whole tree:** `x_min_mm` / `x_max_mm` / `y_min_mm` /
`y_max_mm` are never compared against a pose. `configs/planning.yaml:37` declares
`workspace_bounds_mm: {x_min: -350, …}` and nothing reads it into a check.
`main.py:157,159`'s `.get(..., 350)` defaults are plausible values, so a renamed
key yields an equally-inert ±350 mm envelope.

**Proof it is inert:** `tests/test_planner.py:171,206` construct a deliberately
tiny `WorkspaceBounds(-100, 100, …)` and still pass. A miscalibrated `mm_per_px`
or a bad `origin_offset` sends out-of-envelope coordinates and the one thing that
would have caught it does nothing.

**Fix.** Clamp-or-raise in `Planner._build_pose` against `self.env.workspace`, or
delete the type so it stops reading as a live interlock.

## B4. The planner reads a `motion:` block `planning.yaml` does not have

**File** `plan/planner.py:87-90`. **VERIFIED:** `configs/planning.yaml` contains
`battery`, `cartridge`, `occupancy_grid`, `packing`, `queue`, `camera` — and no
`motion:`. So `pick_approach_height_mm` and `place_insert_height_mm` are
permanently the in-code defaults 60.0 / 2.0. The real values live in
`configs/execution.yaml:14-18`, which the planner never sees.

The sting is four lines up, where the same constructor says of `mm_per_px`:
*"Absent key -> no fallback, not a default. A config that says nothing about the
camera scale has not calibrated the camera."* The reasoning was applied to one
key and not to the two below it.

Currently benign — the defaults happen to equal execution.yaml's — but it means
editing `execution.yaml: motion.insert_height_mm` to fix a crash changes the
executor's config and not the Z in the pose. `tests/test_planner.py:145-146`
passes a hand-built dict that *contains* a `motion` block, so it never notices.

**Fix.** Read both heights from the execution config, or add `motion:` to
`planning.yaml` and assert the two agree at load.

## B5. An unrecognised robot status code becomes `TIMEOUT`

**File** `execution/execution.py:227-230`.

```python
try:    code = RobotStatusCode(s["code"])
except ValueError:  code = RobotStatusCode.TIMEOUT
```

A code outside 0–6 — a KRL revision, a renumbering, a corrupted-but-CRC-valid
byte — is aliased to a real status. `main.py:504-508` routes anything that is not
SUCCESS/PICK_FAILED to `confirm_placement(..., False)` → `planner.py:492` reverts
the cell PLANNED → FREE. If the robot actually completed the place, the twin now
believes that cell is empty and the next cycle plans another battery into it —
compounding B1. Stats report a plausible `place_failed` count. No test covers it.
Related: `execution/protocol.py:158` validates the protocol version on **commands**
(`:123-124`) but discards it as `_version` on **status** packets.

**Fix.** Raise on an unknown status code rather than aliasing it to a real one;
apply the same version check in `unpack_status`.

## B6. `px_per_cell` clamp makes `OccupancyGrid.resolution_mm` a lie

**File** `plan/placement_area.py:163-167`. `px_per_cell = max(1.0, mm_per_cell /
mm_per_px)` binds when `mm_per_px > mm_per_cell`, but the grid is still built at
`resolution_mm = mm_per_cell`. `planner.py:366` then hands that value to
`pack_best_effort` as the forbidden mask's mm-per-cell while the strip dimensions
come from the true `pr.width * scale` — the two disagree by a constant factor and
misalignment accumulates along the strip (~6 mm over a 140 mm cartridge). No
error.

Not currently binding: the measured corpus GSD reaches 1.045 mm/px
(`recog/calibration.py:29`) against `resolution_mm_per_cell: 1.5`, ~30 %
headroom. Lowering that key to 1.0 — a perfectly reasonable edit — makes it bind
immediately. Nothing in `tests/test_placement_area.py` references the clamp.
*Fix: set `resolution_mm = px_per_cell * mm_per_px`, or raise when the clamp
binds.*

## B7. The detector silently downgrades to the heuristic

**File** `recog/inference.py:361-366`. When the checkpoint is missing **and** no
segmenter was passed, `load_detector` emits `log.warning` and returns
`HeuristicDetector()` — HSV thresholds tuned for `synth_dataset.py`'s flat green
rectangles, documented as not expected to generalise. `main.py:327-331` builds
the path from `recog_cfg.get("training", {}).get("checkpoint_dir",
"recog/checkpoints")`, so a renamed key, a moved file, or a torch import failure
all land here and the run reports normal-looking statistics.

Only the *segmentation* path is protected by a raise (`inference.py:352`), and
`tests/test_main_integration.py:243` covers only that one. **Fix:** raise unless
the config explicitly opts into the heuristic.

## B8. `crop_size` defaults silently when `mode.segmentation.config` is absent

**File** `main.py:247-260`. `_build_segmenter`'s docstring promises "Every failure
below raises". `num_classes` would raise on a state-dict shape mismatch and
`half` self-corrects on CPU — but **`crop_size` is pure inference-time
preprocessing and never touches the state dict**. A checkpoint trained at 384
loaded at 256 produces perfectly-shaped, plausible, systematically wrong label
maps. `tests/test_main_integration.py:300` always passes an explicit `config`, so
the empty-`model_cfg` branch is untested. *Fix: require `config` whenever
`checkpoint` is set.*

## B9. `recog/model.py:81` — the in-code anchor default is the value the config calls broken

`model_cfg.get("anchor_scales", [4, 8, 16, 32])`. `configs/recognition.yaml:21`
says of exactly that list: *"re-using the old anchors would put 15 % of boxes
below 0.5 best-IoU — the same failure the old [4, 8, 16, 32] caused."* A renamed
or absent `model.anchor_scales` trains and infers normally and produces a
detector with a measured 15.4 % of boxes unmatched by any anchor. Same shape for
`anchor_ratios` (`model.py:76`). *Fix: make both required — there is no safe
default.*

## B10. `recog/bay_segmenter.py:124-130` — the `except` wraps the resize, not the import

The `# pragma: no cover - optional` label is wrong: cv2 is a **hard** dependency
(`arbitration.py:32`, `placement_area.py:49`, `main.py:48`, `inference.py:29` all
raise ImportError without it). The only realistic trigger is `cv2.resize` itself
failing — non-contiguous array, unsupported dtype, zero-size crop from a
degenerate box — and the fallback silently swaps the segmenter's **input** from
bilinear to nearest-neighbour subsampling. Masks come back the right shape and
look plausible; `P_safe` shifts by a pixel or two, which `_rasterise_mask` turns
into placement cells. No log line, no counter, no test.
*Fix: hoist `import cv2` to module scope and let `cv2.resize` raise.*

## B11. `recog/dataset.py:109,176,192` — a renamed class silently empties the dataset

`parse_coco_json` correctly **raises** when a known name has a mismatched id
(180-185). But a *renamed* category (`battery` → `cell`) fails `name not in
lower_map` and is skipped, leaving `cat_to_class` empty; every annotation is then
dropped at 192 and every image comes back with empty `boxes`/`labels`. Same shape
at `parse_voc_xml:109`. Training runs to completion on an all-background dataset
and produces a detector that detects nothing; the loss curve looks plausible.
*Fix: raise if zero categories matched.*

## B12. `common/config.py:56-58` — a mistyped sub-config name yields `{}`

Misspell `planning:` in `demo.yaml` and the planning config becomes empty.
`mm_per_px` then correctly resolves to `None` → `UnknownScale` (that path is
properly protected). But `execution: {}` silently defaults to host
`172.31.1.147`, and `recognition: {}` builds the detector on `model.py`'s
broken-anchor defaults (B9). *Fix: raise when a `_SUB_KEYS` entry is present but
is neither `str` nor `dict`, or absent entirely.*

## B13. `configs/segmentation_cad_test.yaml:33` — checkpoint dir collides with production

`checkpoint_dir: recog/checkpoints/seg`, identical to `configs/segmentation.yaml:44`,
while every other `segmentation_*.yaml` uses a distinct dir and its header claims
the runs are separated by exactly that key. `configs/demo_seg.yaml:77` loads
`recog/checkpoints/seg/best.pt`, so a CAD-test training run silently overwrites
the checkpoint the demo evaluates — while `demo_seg.yaml:81` still reads model
params from `configs/segmentation.yaml`. *Fix: give it its own dir.*

## B14. `plan/bin_packing.py:56,91` — the placeholder scale survives as a default

`pack_cartridge(..., mm_per_px: float = 0.38)`. That is `planning.yaml`'s
explicitly-labelled **placeholder** ("Replace with real intrinsics when
available") baked in as a silent default — the exact class of value
`placement_area.UnknownScale` was written to abolish, surviving in a sibling
module. No non-test caller; `tests/test_packing_move.py:30` asserts only
`hasattr(bin_packing, "pack_cartridge")`, a green test guarding nothing.
*Fix: make `mm_per_px` required, or delete the function.*

## B15. `plan/planner.py:305` — the blanket handler, mostly mitigated

Credit where due: `UnknownScale` re-raises (270), `BadDetectorBox` and
`PlacementDisagreement` are counted (278, 287), and `_accepts_label_map` exists
specifically to keep a `TypeError` out of here. What still lands here uncounted:
`cv2.error` from a malformed mask, `ValueError` from `label_map is None`
(`placement_area.py:677`), shape-mismatch `IndexError`/`AttributeError`,
`MemoryError`. `main.py:397` catches the total-zero case **only when a segmenter
is configured**, so a systematic extractor failure on the heuristic path reports
a clean run. *Fix: add an `extractor_error_count` and extend `main.py:397`'s
zero-check to the heuristic path.*

---

## `except Exception` census

**27 real clauses** in `recog/`, `plan/`, `common/`, `execution/`, `main.py`
(the "32" in the brief counts five prose mentions inside `plan/planner.py`'s and
`recog/inference.py`'s comments and docstrings, which are not clauses —
verified by `grep -c '^\s*except Exception'`).

**Can swallow a real failure and continue with degraded output — 6:**

| Site | Degrades to |
|---|---|
| `recog/synth3d/materials.py:103` | every material's roughness becomes raw noise (A2) |
| `recog/synth3d/render.py:294` | index-pass EXR read possibly colour-transformed → wrong instance ids (A6, speculative) |
| `recog/synth3d/render.py:204` | denoiser config silently unapplied → noisier images than the config claims |
| `recog/synth3d/scene.py:48` | a datablock survives into the next scene as an unlabelled occluder (A10) |
| `recog/bay_segmenter.py:127` | segmenter input resampled nearest-neighbour instead of bilinear (B10) |
| `plan/planner.py:305` | a cartridge left unplanned by an uncounted exception type (B15) |

Of these, `plan/planner.py:305` is the only one already documented, countered and
covered (`tests/test_main_integration.py:89` asserts neither skip-counter fires);
it is correct as it stands modulo B15's residue.

The other **21 are correct and deliberate**: optional-import guards
(`augmentation.py:56`, `dataset.py:45/62`, `inference.py:36`, `model.py:30`,
`seg_dataset.py:59`), hard-dependency guards that **re-raise** (`main.py:48`,
`inference.py:29`, `arbitration.py:32`, `placement_area.py:49`,
`labelme_to_seg.py:56`, `training.py:165`, `seg_training.py:107`,
`convert_cad.py:452`), an exactly-equivalent numeric fallback
(`seg_evaluate.py:279`, scipy EDT → brute force — slow, not wrong), and
warning/diagnostic paths that print (`generate3d.py:111`,
`_gate_orientation.py:201`, `render.py:422`, `eval_real.py:433/452`,
`eval_real.py:155`).

---

## Config keys read by nothing, and defaults that mask

### Dead keys — present in YAML, zero hits in any `.py`

`configs/planning.yaml`: `cartridge.green_channel_thresh` (10),
`cartridge.pcb_exclusion_required` (14), `occupancy_grid.dtype` (18),
`packing.algorithm` / `.rotation_allowed` / `.deterministic` / `.worst_case_bound`
/ `.max_ms_budget` (21-25), `queue.fill_order` / `.assignment` (28-29),
`camera.mm_per_px_y` (34). Two of these are worse than merely dead:
`packing.rotation_allowed: [0, 90]` is contradicted by `planner.py:91`'s
hard-coded `allow_rotation=True`, and `cartridge.morph_close_ksize` /
`morph_open_ksize` (11-12) reach *only* the constructor defaults at
`placement_area.py:214-215` because `main._build_planner` never passes them —
i.e. tuning them in the YAML does nothing.

`configs/execution.yaml`: `kuka.protocol` (5), `kuka.command_length_bytes` (6),
`kuka.crc_polynomial` (7), `kuka.stop_category` (12), `motion.grasp_height_mm`
(16), `motion.default_velocity_mm_s` (19), **`motion.safety_max_velocity_mm_s`
(20)**. The last is the notable one: a declared velocity **safety cap** that no
Python enforces — the 150 mm/s in `krl_prog/routines.src` is a hard-coded KRL
literal, not derived from this key.

`configs/recognition.yaml`: `model.backbone` (4), `dataset.class_map` (84-86),
`training.log_dir` (175), `evaluation.iou_thresholds` (179),
`evaluation.centroid_error_target_px` (180), `evaluation.edge_error_target_px`
(181).

`configs/demo.yaml:10` / `demo_seg.yaml:17`: `mode.log_level` — `main.py:61` calls
`get_logger("autopick.main")` at import with no level, so the key never applies.

`camera.workspace_bounds_mm` (planning.yaml:37) is *read* but never enforced — B3.

**The previously-reported `arbitration.tau` is genuinely gone**, and
`configs/planning.yaml`'s trailing comment now documents why. That one is closed.

### Keys the code reads that are absent from the YAML, so a plausible default silently applies

* **`motion.approach_height_mm` / `motion.insert_height_mm`** in the planning
  config — B4. The whole block is missing.
* **`cartridge.wall_inset_mm`** — `main.py:203-204` reads it; `planning.yaml`'s
  `cartridge:` block does not define it, so the inset is permanently
  `_DEFAULT_WALL_INSET_MM = 4.25` (`placement_area.py:472`), the catalog
  **maximum**, and is not tunable from config in practice. Conservative
  direction, but a thin-walled SKU over-erodes real placeable floor with no way
  to say so.
* **`max_cartridge_extent_mm`** — `placement_area.py:552` accepts it, no YAML
  carries it, `main._build_planner` never passes it; the bad-box interlock is
  hard-coded to `(81.7, 180.0)`. (The constructor's *validation* of the value at
  569-584 is exemplary — the problem is only that nothing supplies one.)

### The synth3d config surface — clean, and I want that on the record

The brief expected dead keys here. There are none, because the surface is
defended in three layers:

* `config.load_config` (`config.py:372-385`) **raises** on any unknown top-level
  key *and* on any unknown key inside every dataclass section.
* The five unvalidated passthrough sections are pinned by tests instead:
  `param_space.exposure` (`test_synth3d.py:256`), `.zoom` (`:330`),
  `.overlap_prob` (`:304`), every `materials`/`backdrops` `luma_ref` (`:271`),
  every backdrop `color` (`:293`), every lighting rig reachable from
  `param_space` (`:169`), every `off_axis` key (`:181`), the all-or-nothing
  fill-lamp key set (`:195`).
* Every catalog.json key `scene.py` guards with `entry.get(...)` is pinned
  present and positive by `tests/test_bay.py:807` (`tray_outer_mm`,
  `tray_floor_mm`, `interior_mm`, `case_wall_mm`, `module_bay_mm`) and `:959`
  for the procedural path. The JSON sidecars are all newer than their YAML and
  consistent on the fields that matter; `_read_raw` raises on a stale one.

Two **plausible defaults that would mask but are currently held up by those
tests** — listed so nobody removes the defence without knowing what it carries:

* `recog/synth3d/scene.py:264-266` — `floor_z = hi.z` when
  `entry.get("tray_floor_mm")` is falsy. `hi.z` is the **top** of the case: this
  is exactly the pre-task-3 bug `world.build_pcb`'s docstring describes ("the
  board was drawn on the outside of a shut box"). Falsy-tested, so a
  `tray_floor_mm` of `0.0` triggers it too. Held up by `tests/test_bay.py:819`.
* `recog/synth3d/scene.py:287-288` — one `if` gates the **bay proxy, the
  obstructions and the seated cells** together. If `module_bay_mm` or
  `interior_mm` were absent, two of the five `SEG_CLASSES` would vanish from the
  corpus with no error, and `catalog.inspect_glb:233` only writes those keys
  `if cell_union and tray_outer` — so a `CLASS_RULES` change that stops matching
  `Cell_\d+` removes them at the next `convert_cad`. Held up by
  `tests/test_bay.py:807`. **`generate3d.py` records `per_seg_class_kept` in
  `manifest.json` but never asserts any class is non-zero** — a run with two
  empty classes prints a clean summary and exits 0. That is the one gap worth
  closing here.

---

## Regions that are clean

Stated positively so the next audit does not re-till them.

* **`recog/synth3d/bay.py`** — every geometric decision raises rather than
  returning a plausible rectangle (`module_bay_from_bounds`,
  `case_wall_from_bounds`, `interior_from_tray`, `_bay_edge`), including on ties
  and containment violations. `sample_tray`'s two clamps are derived from
  quantities drawn above them and are each commented with the measurement that
  motivated them. The only defect reachable from this file is A1, and it lives in
  `common/packing.py`.
* **`recog/synth3d/lightrig.py`** — raises on out-of-range elevation and
  non-positive distance; the aiming derivation is re-derived independently in the
  tests rather than reused. Nothing to find.
* **`recog/synth3d/annotate.py`** — no bpy, fully tested, and the two places it
  could silently drop an instance (`_FILTER_EXEMPT`, the `masks_from_index` merge
  asymmetry) are both documented with the measurement behind them.
* **`recog/synth3d/world.py`'s procedural-tray path** —
  `_assert_procedural_tray_geometry` (626-856) is eight numeric assertions run on
  **every tray built, not in a test**. It is the correct response to this exact
  defect class and the model the rest of the bpy region should follow; A2, A3 and
  A5 are all places with no equivalent.
* **`plan/arbitration.py`** — genuinely careful. `centre_component`'s refusal to
  guess (48-97), `mask_iou`'s union-zero rule, and `admits_a_cell`'s
  morphological (not areal) test are correct and their docstrings match the code.
* **`plan/placement_area.py`'s scale handling** — `_resolve_scale` (105-126),
  `UnknownScale`, `placeable_extent_mm`'s raise-on-empty (512-517),
  `reject_if_not_one_cartridge_floor`'s zero-extent guard (621-628) and
  `max_cartridge_extent_mm` validation (569-584) are exactly the right shape.
  Only the `px_per_cell` clamp (B6) is a problem.
* **`plan/planner.py`'s scale plumbing** — `frame_scale`,
  `_drop_areas_measured_at_another_scale`, and the deliberate two-scale split in
  `_build_pose` (378-388) are correct and well reasoned. This file's problems are
  the occupancy marking and the `motion` block, not the scale work.
* **`recog/calibration.py`** — clean throughout; raises rather than guessing at
  every branch, and `frame_mm_per_px_for_image` returning `None` is a properly
  distinguished third state that its caller handles.
* **`common/types.py`, `common/logging.py`, `execution/protocol.py`'s command
  path** — no defects found beyond the status-side version check (B5).

---

## Method note

Part A is careful reading of bpy-only modules, labelled VERIFIED/SPECULATIVE
throughout. The A1 figures come from a read-only probe importing only the
bpy-free half (`bay.py`, `common/packing.py`, `config.py`) and reproducing
`scene.py:343-357` call-for-call; it was written to a scratch directory outside
the repo. Part B was researched in parallel and its five highest-ranked items
(B1–B5) were independently re-verified by reading the cited lines and grepping
the tree before inclusion. Nothing in the working tree was modified; `git status`
shows only this file.
