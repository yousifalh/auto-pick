# Groundwork for spec #2 (generalisation): what the code actually requires

Investigation only — 2026-08-10. No generator code, dataset, or checkpoint was
touched. This document establishes ground truth for the spec the owner will
write next (`docs/superpowers/specs/2026-08-08-tray-interior-design.md` §8,
"2 — Generalisation"); it does not design the procedural tray family itself.

**A parallel document already exists and answers some of what this one was
asked to leave open**: `docs/superpowers/specs/2026-08-10-generalisation-decisions.md`,
recorded the same day, carries the owner's Decision 1 (train procedural, test
on held-out CAD), Decision 2 (anchored vs. wide procedural sets) and Decision 3
(three cell formats: 18650/21700/26650). Where this document's own
measurements bear on those decisions, both are cited; where they'd conflict,
the measurement in *this* document wins per that document's own instruction
("where it does, the measurement wins and this document should be corrected,
not defended") — no such conflict was found.

Throughout: **(measured)** means read directly from the code, `catalog.json`,
or a command run during this investigation. **(judgement)** means an
assessment or recommendation, not a fact. Line numbers are current as of
commit `e23c97e`.

---

## 1. The parameter surface a procedural tray would need to supply

### 1.1 What `catalog.json` actually holds, per asset (measured)

Read directly from `recog/synth3d/assets/catalog.json` (all four Anker
assemblies) and `recog/synth3d/catalog.py`'s `inspect_glb`:

| Field | Type | Example (`AnkerPowerCore10000`) | Origin |
|---|---|---|---|
| `tray_outer_mm` | `[x0,y0,x1,y1,zmax]` (5 floats, mm) | `[-31.45,-45.45,31.45,45.45,11.1]` | **measured from mesh** — AABB of the `case`-role geometry, after `_split_case_liner` removes the liner |
| `cell_union_mm` | `[x0,y0,x1,y1,zmax]` (5 floats, mm) | `[-27.45,-43.0,27.45,22.0,20.25]` | **measured from mesh** — AABB of the `cell`-role geometry |
| `case_wall_mm` | float, mm | `4.0` | **derived** — `bay.case_wall_from_bounds(tray_outer, cell_union)`, the non-bay-axis gap |
| `interior_mm` | `[x0,y0,x1,y1]` (4 floats, mm) | `[-27.45,-43.0,27.45,41.45]` | **derived** — `bay.interior_from_tray(tray_outer, cell_union, wall)` |
| `module_bay_mm` | `[x0,y0,x1,y1]` (4 floats, mm) | `[-27.45,22.0,27.45,41.45]` | **derived** — `bay.module_bay_from_bounds(interior, cell_union)` |
| `tray_floor_mm` | float, mm | `1.95` | **measured from mesh** — min z of `cell`-role geometry (`_role_zmin`), because the cells rest on the floor in assembled pose |

This confirms the six-field list the brief supplied is complete for what
`catalog.json` currently carries, **and adds a distinction the brief's list
doesn't make**: only three of the six are ever read again after conversion
time.

### 1.2 Only three fields are actually consumed at scene-build time (measured)

Grepping `recog/synth3d/scene.py` and `world.py` for every one of the six
field names: `scene.py`'s `build()` reads exactly `entry["tray_floor_mm"]`,
`entry["interior_mm"]`, `entry["module_bay_mm"]` (lines 265–312). `tray_outer_mm`,
`case_wall_mm` and `cell_union_mm` are **never read outside `catalog.py`** —
they exist only so `inspect_glb` can derive the three consumed fields from raw
CAD geometry, and are kept in `catalog.json` afterwards purely as audit
provenance.

This matters directly for a procedural tray. `catalog.inspect_glb`'s three
`bay.py` derivation calls (`case_wall_from_bounds`, `interior_from_tray`,
`module_bay_from_bounds`) exist to **reverse-engineer** a wall thickness and a
bay location from a CAD assembly that doesn't publish either as a number — they
infer wall thickness from the gap between measured cells and a measured outer
box, and infer which edge is the bay from which gap is largest. A procedural
generator is the opposite kind of thing: it **chooses** footprint, wall
thickness and bay depth directly (that's what "sampled" means in the spec's
own wording), so it does not need to re-derive them from a cell layout it
also invented. It can write `interior_mm`/`module_bay_mm`/`tray_floor_mm`
directly from its own sampled parameters and skip the inference machinery
entirely — **judgement**, but a fairly firm one, since `bay.py`'s consumption
functions (`module_rect_local`, `placement_rect_local`,
`module_world_placement`, `placement_world_placement`, `_bay_edge`,
`sample_obstructions`, `seated_cell_poses`) only ever take `interior_mm` /
`bay_mm` as opaque rectangles — nothing downstream cares whether they were
measured or chosen.

One consequence worth flagging: **`module_bay_mm` is not a free rectangle**.
`bay._bay_edge` requires it to be a full-span strip flush against exactly one
edge of `interior_mm` (spans the interior's full width on one axis, flush
against one edge on the other), and raises `ValueError` otherwise. Any
procedural sampler that wants to reuse `module_rect_local`/
`placement_rect_local` unchanged must satisfy this invariant by construction —
it is a real constraint on the sampler, not just documentation.

### 1.3 A parameter with no existing precedent at all (measured absence)

The wall/rim **height** — how far the cavity floor sits below the case's own
top rim, what the spec calls "bay depth" — has **no catalog.json field**.
Today it is implicit in the imported CAD mesh's own z-extent (all four Anker
assemblies happen to share an 11.1 mm case half-height) and is never extracted
as a number by `catalog.py`, never read by `scene.py`, and never a `bay.py`
argument. `config.py`'s `open_case` `Variant` comment mentions "9.15mm deep,
half an 18650" but that's descriptive prose, not a value read anywhere.

A procedural tray, having no mesh to measure, needs bay depth as an **explicit
sampled parameter with no existing analogue to reuse** — this is new surface,
not a rename of something that already exists. Likewise the tray's own outer
footprint (an equivalent of `tray_outer_mm`) is needed by whichever module
actually constructs the standing wall geometry (§2–3 below), even though
`scene.py` itself never reads `tray_outer_mm` today.

### 1.4 Net parameter list for a procedural tray (judgement, built on 1.1–1.3)

To reach parity with what `scene.py`/`world.py` currently consume per asset,
a procedural tray generator needs, per generated tray:

- `interior_mm: Rect` (mm) — direct sample or footprint-minus-wall arithmetic
- `module_bay_mm: Rect` (mm) — direct sample, constrained to be a full-span
  strip on one edge of `interior_mm` (§1.2)
- `tray_floor_mm: float` (mm)
- an outer footprint (`tray_outer_mm`-equivalent) and a wall **height** (§1.3,
  new) — needed to build the wall mesh itself, not read by `scene.py`
  directly but required by whichever code constructs the geometry
- a cell format selection (§4) feeding the seated-cell footprint, independent
  of the tray's own geometry

`case_wall_mm` and `cell_union_mm` are **not** needed as procedural-path
inputs under this reading — they are artifacts of the reverse-engineering
step a generative pipeline doesn't have to perform. If the design instead
wants to author a procedural tray by placing cells first and inferring the
bay from their layout (mirroring the CAD pipeline exactly, for symmetry) it
could still call the existing `case_wall_from_bounds`/`module_bay_from_bounds`
functions unchanged — they're generic rect arithmetic, not CAD-specific
despite living next to CAD-derived docstrings. That would be a design choice,
not a code constraint.

---

## 2. Where the bpy boundary falls

**Finding: nothing about a procedural tray inherently forces a geometric
decision into bpy-space, provided the same discipline the existing furniture
builders already follow is kept.** (judgement, but grounded in a direct
pattern match below)

`build_jig` (`world.py:509`) is the existing precedent for exactly this
shape of problem: `layout.plan_jig` (bpy-free) decides every pocket's
position and size; `build_jig` (bpy) does nothing but call
`primitive_cube_add` and boolean-difference at the numbers it's given. No
geometric judgement happens inside `build_jig` itself. `build_pcb` and
`build_bay_proxy` follow the identical split against `bay.py`'s
`module_world_placement`/`placement_world_placement`. A procedural tray's
wall construction can follow the same shape:

**Stays bpy-free (new work, but on the existing side of the line):**
- A sampler for the five spec axes (footprint, wall thickness, bay depth,
  cell count, pitch) — pure RNG and arithmetic, the same shape as
  `bay.sample_obstructions`.
- A forward-construction function producing `interior_mm`/`module_bay_mm`/
  `tray_floor_mm`/outer-footprint/wall-height from those samples — the
  mirror image of `catalog.inspect_glb`, and arguably simpler: no
  `classify_case_parts` disambiguation, no `needs_flip` ambiguity, no tie
  cases, because a generative pipeline chooses the answer instead of
  discovering it.
- A per-cell-format footprint lookup (§4) — bpy-free, same shape as the
  existing `SEAT_CELL_FOOTPRINT_M` constant but keyed by format.
- Everything `bay.py` already exports for consumption
  (`module_rect_local`, `placement_rect_local`, `*_world_placement`,
  `sample_obstructions`, `seated_cell_poses`, `obstruction_forbidden_mask`)
  needs **no changes** — all of it already operates on opaque
  `interior_mm`/`bay_mm` rectangles, never on anything CAD-specific.

**Forced into bpy-space, unavoidably:**
- Actually instancing geometry in the scene, whichever of §3's two options
  is chosen — either building primitives directly (`world.py`) or importing
  a generated glTF (`assets.py`'s existing `AssetLibrary` path). This is
  true today for every asset and isn't specific to procedural trays.

**The one discipline risk worth flagging**, not a certainty: if procedural
walls get any cosmetic detail beyond a plain rectangular boolean pocket
(chamfers, draft angle, corner fillets — anything spec #3's realism work
might eventually want), that shaping choice has to be pre-computed as
bpy-free parameters too (e.g. a fillet radius passed in, not decided by
eyeballing inside `world.py`), or it quietly becomes an ungoverned bpy-side
decision the way `build_jig`'s plain rectangular pockets are not. Nothing
currently proposed requires this; it's a boundary to watch if realism work
lands on top of the procedural tray later.

---

## 3. How a procedural tray would actually be built

Two realistic options, matching the brief's framing:

**Option A — Blender primitives at scene-build time.** A new bpy-side
builder (structurally identical to `build_jig`): `primitive_cube_add` for
the tray's outer block, a second cube for the cavity, boolean-difference,
using only numbers computed by §2's bpy-free sampler.

**Option B — glTF assets generated offline, fed through the existing
catalog path.** A new bpy-free generator (structurally identical to
`catalog.build_catalog`, but authoring geometry with `trimesh` instead of
converting STEP), writing `.glb` + a `catalog.json`-shaped entry, imported
through the existing `AssetLibrary`/`assets.py` machinery unchanged.

### Recommendation: Option A (judgement)

**Reliability, not elegance, decides this**, per the brief's own steer, and
on reliability Option A is the clear choice.

All three of the import-path surprises the brief names — the inverted
up-axis, the fused two-material object, the inner liner sharing a role — are
**specific to round-tripping through glTF authored by an external tool and
re-interpreted by Blender's importer**:

- The inverted up-axis (`bay.needs_flip`, `assets.flip_if_inverted`,
  task-3c) is Blender's glTF importer remapping `(x,y,z) → (x,-z,y)`
  combined with the *source* STEP file's own up-axis convention (Y-up) —
  an artifact of the CAD authoring tool, not of glTF or Blender alone.
- The fused two-material object (`assets._split_multi_material_case`,
  task-3b) is `cascadio`'s STEP→glTF tessellation merging two CAD bodies
  into one glTF node with two material primitives — an artifact of the STEP
  conversion step specifically.
- The inner liner sharing a role (`catalog.classify_case_parts`,
  task-3b) is a naming collision in the *source* CAD's own part names
  (`Case*_btm` for both the shell and the liner) that a regex can't
  disambiguate — again an artifact of the external CAD, not of the pipeline.

A procedural generator that authors its own glTF has no external CAD
naming/tessellation/up-axis conventions to inherit — it controls all three at
the source, so none of these three specific failure modes can recur *for the
same reason they occurred*. But Option B still round-trips through
`bpy.ops.import_scene.gltf`, which carries its own general landmine
independent of source (the QUATERNION rotation-mode issue `assets.lay_flat`'s
docstring documents at length) — already correctly handled by the existing
whole-matrix-write pattern, so not a new risk, but a reminder that Option B
doesn't fully exit the import machinery, only the CAD-specific quirks within
it.

Option A exits the import machinery **entirely**. It also naturally enforces
§2's bpy-free/bpy discipline the same way `build_jig` already does, with zero
new failure surface: no tessellation tolerance to choose, no glTF node/mesh
naming scheme to invent, no need to touch (or bypass)
`_split_multi_material_case`/`classify_case_parts`/`_bay_edge`'s malformed-
input paths at all — a plain rectangular pocket has no shell/liner
ambiguity to resolve in the first place. Option B is not unreasonable — it
reuses more existing code and would look more like a "real asset" to
downstream tooling that expects `catalog.json` entries — but it reopens a
surface that has cost this project three separate render cycles to debug,
for a benefit (mesh realism) the generalisation measurement doesn't
obviously need over a plain boolean-cut box.

**Cost note (judgement, not measured):** the cosmetic ceiling of Option A is
lower — chamfers, wall draft, moulding seams are much easier to add in a mesh
authored with `trimesh`/CAD-style tooling than in Blender boolean primitives.
If spec #3's realism work later wants that detail on procedural trays,
revisiting Option B (or a hybrid — Option A for the tray body, `trimesh` only
for cosmetic surface detail) would be reasonable. That tradeoff belongs to
whoever writes the spec, not to this document.

---

## 4. Cell formats: the hardcoded-18650-dimension audit

### 4.1 The stated figures (measured)

`catalog.json`: every `Cell_*` subpart across all four assemblies reports
`extents_mm: [18.3, 18.3, 65.0]` — confirms the brief's 18.3 × 65 mm figure
exactly. `docs/superpowers/specs/2026-08-10-generalisation-decisions.md`
Decision 3 and `configs/planning.yaml`'s own comment ("For 21700 override in
runtime: diameter=21, length=70") confirm 21700 = 21 × 70 mm and 26650 = 26 ×
65 mm, matching the brief.

### 4.2 Can the pipeline accept a parametric cylinder? Yes (judgement,
strongly grounded)

The 18650 "cell" is not special-cased anywhere as a mesh feature — it is a
plain cylinder CAD part, imported and instanced through the same
`AssetLibrary`/`clone`/`lay_flat` path as every other role. There is nothing
in `assets.py` that requires cell geometry to originate from STEP; a
Blender `primitive_cylinder_add` at the sampled radius/length would satisfy
every downstream consumer identically. This directly **updates a stale
assumption** in `docs/superpowers/specs/2026-08-06-segmentation-placement-
area-design.md` (lines 270–271): *"18.3 × 65 mm cells exclusively... no
21700 geometry in `cad/`. Extending the class to 21700 requires new CAD and
is out of scope here."* That was true as a statement about that task's
scope, but false as a statement about the pipeline's capability — no new CAD
is required, a parametric primitive suffices.

### 4.3 The full site list (measured — this is the deliverable)

Fifteen sites carry an 18650-specific figure in logic (not counting
comment-only mentions, listed separately below). Grouped by how they'd need
to change:

**No override path at all (worst offenders):**

1. `recog/synth3d/world.py:960` — `SEAT_CELL_FOOTPRINT_M = (0.0183, 0.065)`.
   Drives `scene.py`'s seated-cell capacity estimate (line 344–348) and the
   `cell_w`/`cell_h` handed to `bay.seated_cell_poses`'s packer for **every**
   scene, every asset, unconditionally. No config key, no parameter, no
   per-format lookup — this is the single most central hardcode in the
   pipeline, because it runs at every render.
2. `recog/seg_ablation.py:61-62` — `CELL_W_MM = 18.3`, `CELL_H_MM = 65.0`.
   Feeds the ablation harness's own `first_fit_decreasing` packer call
   (line 116-117). **Independently duplicated** from item 3 below — same
   figure, two unconnected constants, a drift risk on its own even before
   new formats are considered.
3. `recog/synth3d/_gate_orientation.py:95-107` — the orientation smoke gate
   literally asserts `abs(e.z*MM - 18.3) < 0.5` and
   `abs(max(e.x,e.y)*MM - 65.0) < 0.5` for every `role == "cell"` object in
   every imported template. This would **false-fail** against any CAD cell
   asset (or, if extended to gate procedural cylinders too) of a different
   size — it needs to become per-format or per-asset before any 21700/26650
   CAD or procedural cylinder is added to a template it runs against.

**Have an override path, but the default is single-format and nothing
selects between formats at the scene level:**

4. `recog/calibrate_tau.py:57-58` — `CELL_W_MM = 18.3`, `CELL_H_MM = 65.0`,
   with `--cell-w-mm`/`--cell-h-mm` CLI flags (line 358-359) that do allow a
   different figure **per invocation** of the whole calibration run — but
   there's no notion of a *mixed-format validation split* within one run.
5. `plan/planner.py:50-51,65-66` + `configs/planning.yaml:4-5` —
   `PlannerConfig.battery_width_mm`/`battery_length_mm` default to 18.5/65.0
   mm (note: **18.5, not 18.3** — the nominal spec figure, not the
   CAD-measured one; `calibrate_tau.py`'s own comment at line 51-54 already
   flags this exact discrepancy and deliberately uses the smaller,
   CAD-measured figure instead, reasoning that a worst-case test should use
   the smaller structuring element). Config-driven, but one global battery
   entry — the planner's data model has no per-cartridge cell format.

**Config values whose calibration assumed one cell scale, not a literal
mm constant in code:**

6. `configs/recognition.yaml:56` — `anchor_scales: [40, 64, 96, 144]`,
   explicitly documented in `world.py`'s `setup_camera` docstring (line
   410-414) as coupled to the 18650's measured pixel footprint at the
   current zoom range; the docstring itself warns that widening zoom without
   retuning anchors already once put 20% of boxes below 0.5 best-IoU. A
   larger cell format changes the footprint distribution the same way
   widening zoom does.
7. `configs/synth3d.yaml:141` — `filter.max_aspect: 4.0`, calibrated
   (`annotate.py:33`'s comment) from the 18650's own 65.0/18.3 = 3.55 aspect
   ratio plus rotation-jitter headroom (measured ceiling 3.68 over 1245
   boxes). 21700's aspect is 70/21 ≈ 3.33 and 26650's is 65/26 = 2.5 — both
   comfortably inside the existing band, so this one is not urgent, but its
   derivation assumed a single geometry and should be re-checked once real
   figures exist for the new formats.
8. `configs/synth3d.yaml:125` — `filter.min_px: 500`, calibrated
   (`annotate.py:391-397`'s comment) as "a CELL-sized threshold" against the
   18650's ~29×104 px footprint at 1.6 px/mm. All three candidate formats
   are the same size or larger, so this stays conservative, not broken.

**Widest tolerance, lowest risk:**

9. `recog/inference.py:265` — the heuristic detector's aspect-ratio gate
   `2.0 <= aspect <= 5.0` (comment: "18650/21700 cells are tall cylinders").
   Already spans two formats by construction; 26650's 2.5 fits too.

**Comment-only (no logic depends on the literal, but worth knowing they
exist since they document *why* the numbers above were chosen):**
`recog/synth3d/annotate.py:33` and `:393`, `recog/synth3d/layout.py:86`,
`recog/synth3d/assets.py:127`, `recog/synth3d/world.py:946-959,992,1015`.

### 4.4 What does *not* need to change (measured — corrects an open
question in the parallel decisions doc)

- **`admits_a_cell` itself** (`plan/arbitration.py:162`) is already fully
  parametric — `cell_w_px`/`cell_h_px` are caller-supplied arguments, not
  constants inside the function. The hardcoding lives one layer up, in the
  callers (items 2 and 4 above). The function needs no change; its two
  callers do.
- **`recog/seg_dataset.py`** needs no change. It operates on already-
  rendered per-instance ROI crops, resized to a fixed `out_size × out_size`
  grid (default 256, `__init__` line 210) regardless of the source object's
  physical mm size, and labels by the format-agnostic `SEG_CLASSES` taxonomy
  (`"battery"` covers any cylindrical cell). It was named in the parallel
  decisions doc as a site "to check" (hedged, not asserted) — checked: no
  hardcoded dimension found in it.
- **The class taxonomy** (`recog.dataset.CLASS_MAP`, `config.CLASSES`,
  `config.SEG_CLASSES`) needs no new classes. `"battery"` is already
  format-agnostic by design — nothing in the taxonomy encodes a size. What
  *does* need attention is the per-class filter/anchor **configuration**
  (items 6-8 above), which is a different thing from the class definitions
  the brief's wording could be read to mean.
- **The packer** (`common/packing.py`, `plan/bin_packing.py`) needs no
  change — `Item(id, width, height)` is already generic; every hardcode is
  in what gets passed in, not in the packer itself.

---

## 5. What a disjoint train/test split could look like

The brief asks this be enumerated, not decided — decision deferred to the
owner. **Note:** the owner's parallel decision (Decision 1,
`2026-08-10-generalisation-decisions.md`) already picked one of these three
options; it's included below for completeness of the option landscape, not
because this document is re-opening it.

All options below share one caveat that must travel with every result they'd
produce: **none of them is a sim-to-real measurement.** Per
`docs/NEXT_STEPS.md`'s "The constraint this plan works around," that
measurement is not obtainable on this project at all; every option here
answers a synthetic-to-synthetic generalisation question instead, and must
never be reported as more than that.

**A — Hold out SKUs.** Train on 3 of the 4 Anker assemblies, test on the
4th. Buildable today with a small filter in `generate3d.py`/`AssetLibrary`
(no new generator code). **Weak disjointness**: all four SKUs share the same
generative process (real CAD, same two-piece-shell-plus-liner-plus-lid
construction, wall thickness in a narrow 2.4–5.9 mm band, the same
largest-gap bay-placement rule). This tests interpolation between four
known points in one small family, not robustness to an unseen distribution
— closer to a leave-one-out sanity check than the generalisation measurement
spec #2 is meant to produce.

**B — Train procedural, test on CAD** (the owner's Decision 1). Requires
the procedural tray generator to exist first — not buildable today.
Strongest disjointness of the three: training geometry is entirely
parametric, test geometry is real measured hardware never seen in any form.
Carries the ambiguity the decisions doc itself already names: a poor score
can't distinguish "model doesn't generalise" from "the procedural
distribution doesn't cover the CAD's true parameter values" without the
CAD-trained control model the same document proposes.

**C — Train CAD, test on procedural** (the reverse). Also needs the
procedural generator to exist, but only for the **test** side — the
**training** data and checkpoint are what exist today, so once even a first
cut of the generator lands, this is the cheapest of the three to run: no
retrain, just new held-out scenes scored against `recog/checkpoints/seg/
best.pt`. Tells you whether the *current, already-shipped* model already
generalises to a wider distribution or is narrowly fit to four known shapes
— a useful, fast first data point that could land before B's full 2×2 is
built out.

**D — Format holdout (not in the brief's three, but supported by the same
mechanism).** Train on a subset of {18650, 21700, 26650}, test on the
excluded format(s) — orthogonal to A/B/C since tray geometry and cell format
are independent sampling axes in the current design. Decision 3 already
settles that all three formats are in the training mix, so this doesn't look
like the chosen axis, but the code has no structural reason it couldn't
support it alongside whichever of A/B/C is picked.

---

## 6. Corrections to existing docs found during this investigation

- **`docs/NEXT_STEPS.md` Step 5** still lists "harden `tests/test_synth3d.py`'s
  bpy-boundary check... it is a substring grep for `import bpy`" as an open
  item. **This is stale.** Commit `b7bb57b` ("test: replace the bpy-boundary
  substring grep with an AST check") already landed and is an ancestor of
  the current HEAD (confirmed via `git merge-base --is-ancestor`); the check
  in `tests/test_synth3d.py` (lines 43-83) is an AST walk that explicitly
  handles `from bpy import context`, aliased imports, and dynamic
  `importlib.import_module("bpy")`, with its own regression tests
  (`test_bpy_detector_rejects_every_known_evasion`,
  `test_bpy_detector_accepts_a_docstring_that_merely_mentions_it`). Relevant
  here because item 2 above leans on this boundary being real and enforced —
  it is, more thoroughly than NEXT_STEPS currently says.
- **`docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md`**
  (lines 270-271) states extending to 21700 "requires new CAD and is out of
  scope." Per §4.2 above, this was accurate as a scope statement for that
  task but is not an accurate statement of pipeline capability — a
  parametric cylinder needs no CAD at all.
- No contradiction found with `2026-08-08-tray-interior-design.md` §8 itself
  — its two-sentence sketch of spec #2 (cell formats + procedural family,
  Anker assemblies retained as anchors) is consistent with everything
  measured here.

---

## 7. Verification

No code changed. `python -m pytest -q` run in full during this investigation:
all tests pass, matching `docs/NEXT_STEPS.md`'s "621 tests" baseline at
`8744947` (current HEAD `e23c97e` is a docs-only commit ahead of it, so the
suite is unaffected). No retrain, no dataset regeneration, no generator
edit was made.
