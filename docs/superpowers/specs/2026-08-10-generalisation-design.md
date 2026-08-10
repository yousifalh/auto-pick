# Spec #2 (generalisation) — design 2026-08-10

Design only. No generator code, dataset, or checkpoint is touched by this
document. It turns `2026-08-10-generalisation-decisions.md` (the owner's
settled decisions — not re-opened here) and
`2026-08-10-generalisation-groundwork.md` (the measured investigation) into
a buildable design, and settles the two questions the decisions doc left
open. Line numbers and commit references are current as of `55e71b0`.

Throughout: **(measured)** is read directly from the code, `catalog.json`,
or a receipt. **(judgement)** is this document's own design choice, not a
fact anyone measured. Where this document's judgement conflicts with, or
completes something the groundwork left unstated, that is called out
explicitly rather than silently assumed.

---

## 0. What this spec is answering, restated once

Real photographs are unobtainable (`docs/NEXT_STEPS.md`, "The constraint
this plan works around"). Train-procedural / test-disjoint-CAD (Decision 1)
is therefore the project's primary robustness evidence, and it is **not** a
sim-to-real measurement — every number this spec produces is
synthetic-to-synthetic and must be reported as such.

---

## 1. What is being built

### 1.1 The procedural cartridge-tray family

A generator that samples, per tray:

| axis | what it controls | precedent |
|---|---|---|
| footprint | outer XY extent | new — see §5.2 |
| wall thickness | the 3-sides margin between outer footprint and cavity | `bay.case_wall_from_bounds` range 3.7–4.25mm (measured, all 4 SKUs) |
| bay depth (height) | how far the cavity floor sits below the case's own top rim | **no existing analogue** (groundwork §1.3) — new surface |
| cell count, pitch | how many cells of the sampled format the tray is sized around | new — see §5.2 for how this derives footprint rather than existing as a free axis |

Plus the outer footprint's own height split (bottom shell vs. lid, §4.4) and
the electronics-module edge (§6.1).

**Anchored** samples within/slightly beyond what the four Anker assemblies
span; every tray stays plausible as real hardware. **Wide** samples well
outside that range. Kept as two separate, separately-scored sets (Decision
2) — never blended, never trained as one model.

The measured anchor points, read from `catalog.json` directly, for every
axis that has one:

| SKU | outer footprint (mm) | wall (mm) | interior/cavity (mm) | module-bay depth (mm) | case half-height (mm) |
|---|---|---|---|---|---|
| PowerCore10000 | 62.9 × 90.9 | 4.00 | 54.9 × 84.45 | 19.45 | 11.1 |
| PowerCore13000 | 80.7 × 97.0 | 3.75 | 73.2 × 89.5 | 22.75 | 11.1 |
| PowerCore20100 | 62.3 × 167.8 | 3.70 | 54.9 × 160.4 | 25.20 | 11.1 |
| PowerCore26800 | 81.7 × 180.0 | 4.25 | 73.2 × 171.5 | 30.75 | 11.1 |

(all measured from `recog/synth3d/assets/catalog.json`; `tray_floor_mm` is
1.95mm on all four too.)

**One axis has no observed range at all (judgement, building on groundwork
§1.3's measured-absence finding):** case half-height is 11.1mm on **all
four** SKUs — n=1, not a 4-point spread. Decision 2's "anchored: within and
slightly beyond the range the four Anker assemblies span" is well-formed
for footprint and wall thickness, which have real 4-point ranges above, but
degenerate for bay depth/height — there is no range to sit "within." The
anchored sampler should treat this axis as "near the single observed value"
(e.g. a modest jitter band, not a wide draw), while the wide sampler is
free to depart from it; this is worth stating explicitly rather than
quietly reusing the same "within the observed range" language for an axis
where that phrase doesn't parse against the data.

### 1.2 Cell formats

Three formats in the training mix (Decision 3, "all three" reading — see
§8 for why): 18650 (18.3 × 65.0mm, **measured**, the existing CAD),
21700 (21 × 70mm), 26650 (26 × 65mm) (both from Decision 3 and
`configs/planning.yaml`'s own "For 21700 override in runtime" comment).
CAD assemblies always render 18650 cells (their cell mesh *is* an 18650,
cloned not generated — `world.seat_cells`'s docstring). Procedural trays
sample among all three, independent of tray geometry (§5.2).

---

## 2. The build approach: Blender primitives at scene-build time

**Decision, not left open: Option A** (the groundwork's own recommendation,
adopted here for the reason the groundwork gives — reliability over
elegance, per the owner's own steer in the decisions doc).

Restated because it is load-bearing: all three of this project's glTF
import surprises — the inverted up-axis (`bay.needs_flip` /
`assets.flip_if_inverted`), the fused two-material case object
(`assets._split_multi_material_case`), and the inner liner sharing a role
with the outer shell (`catalog.classify_case_parts`) — are artifacts of
**round-tripping through an externally-authored STEP file, tessellated to
glTF by `cascadio`, and re-interpreted by Blender's importer**. Each cost a
render cycle to diagnose. A procedural generator that builds primitives
directly in Blender (`primitive_cube_add` + boolean-difference, exactly
`world.build_jig`'s shape) never enters that pipeline at all: no source
CAD up-axis convention to invert, no multi-material tessellation to
re-split, no ambiguous liner name to disambiguate. These are not risks that
get smaller with a procedural glTF path (Option B); they disappear
structurally, because they were never about glTF or Blender in general —
only about *this specific external CAD's* conventions leaking through both.

Cost accepted knowingly: the cosmetic ceiling of boolean-cut primitives is
lower than a `trimesh`-authored mesh (no chamfers, no draft angle, no
moulding seams without extra work). That is spec #3's problem if it ever
needs it, not this one's — the generalisation measurement needs correct
labels on plausible-enough geometry, not visual realism, and Option A gives
correct labels with zero new failure surface.

---

## 3. Where the bpy boundary falls

`world.py`/`scene.py` import bpy and cannot be unit-tested; `bay.py`,
`catalog.py`, `config.py` (plus `layout.py`, `annotate.py`, `lightrig.py`)
are bpy-free and are the only modules `tests/test_synth3d.py`'s AST-based
boundary check (`_BPY_FREE_CANDIDATES`, lines 38–40) verifies stay that
way. A procedural tray is *generated* geometry — every prior asset was
either imported (CAD) or trivially primitive (the jig plate) — so this is
the boundary a careless implementation is most likely to blur. Three
pieces, three homes, matching the groundwork's own module split
(bay.py/catalog.py/config.py) with a specific division of labour:

### 3.1 `config.py` (bpy-free) — the tunables

A new config section (same shape as `ObstructionCfg`/`LayoutCfg`): sampling
ranges for footprint, wall thickness, bay-depth, cell count and pitch, kept
**separately** for anchored and wide (two range-sets, not one config with a
"how wide" knob — the separation has to be structural or the "kept
separable" requirement in Decision 2 has nothing to enforce it). Plus a
`CELL_FORMATS` table (`{"18650": (0.0183, 0.065), "21700": (0.021, 0.070),
"26650": (0.026, 0.065)}`, metres) replacing the single hardcoded
`SEAT_CELL_FOOTPRINT_M` (§6.1). No bpy import; loads through the existing
YAML/JSON-sidecar path `load_config` already provides.

### 3.2 `bay.py` (bpy-free) — the sampler

A new pure-RNG function, same shape as `sample_obstructions`: given the
config ranges and an rng, draws the five axes and returns a plain
dataclass (`TraySample` or similar) — footprint, wall, bay depth, cell
format, cell count/pitch. No Blender call anywhere in this function; it is
exactly as testable as `sample_obstructions` already is.

### 3.3 `catalog.py` (bpy-free) — the forward-construction

A new function, structurally the mirror image of `inspect_glb` (which
already lives here): given a `TraySample`, emit a catalog-entry-**shaped**
dict — `interior_mm`, `module_bay_mm`, `tray_floor_mm`, plus the two fields
`inspect_glb` never had to produce because `scene.py` never reads them
(outer footprint, wall height) that the bpy builder in §3.4 needs. Simpler
than `inspect_glb`: no `classify_case_parts` disambiguation, no
`needs_flip` question, no tie cases (groundwork §2) — a generative pipeline
**chooses** the bay edge instead of discovering it, and §5.2 below shows
exactly how, so `module_bay_from_bounds`'s full-span-strip invariant is
satisfied by construction, not asserted after the fact. This function is
what a caller uses to get something to register into an `AssetLibrary`
namespace (§4) — it is never written to `catalog.json` on disk (§4.2).

### 3.4 `world.py` (bpy, unavoidable) — the geometry

A new builder, structurally identical to `build_jig`: primitive cubes,
boolean-difference, at numbers computed entirely by §3.2/§3.3 above. No
geometric judgement happens here — see §4.4 for exactly what it builds and
why that shape needs no new bpy-side decisions either.

**The one discipline risk, restated from the groundwork because it is real
and easy to violate by accident:** if this builder ever grows cosmetic
detail (chamfer, draft angle, corner fillet), that has to be a parameter
computed in §3.2, not a number chosen by eyeballing inside `world.py` — the
same rule `build_jig`'s plain rectangular pockets already follow. Nothing
in this design needs it yet; flagged so it isn't quietly violated when
spec #3's realism work eventually touches this builder.

**No new bpy-free module file is needed.** All three homes above
(`bay.py`, `catalog.py`, `config.py`) are already in
`_BPY_FREE_CANDIDATES`; the AST boundary check needs no edits, and
`test_every_bpy_free_module_is_actually_checked` (which would fail loudly
if a new module were added without updating that list) never fires. This
is a real, checkable reason to prefer extending these three files over
introducing a fourth — one less place the boundary can be forgotten.

---

## 4. Coexistence with catalog-derived assets: no second code path

This is the constraint most likely to be answered vaguely, so it is made
concrete here.

### 4.1 The problem, precisely

`AssetLibrary.__init__` loads `catalog.json` into `self.assets: Dict[name,
dict]`. `scene.build()` calls `library.names()` (uniform `rng.choice` over
those keys — **measured**, `scene.py:171`), `library.instantiate(asset,
variant, rng)`, and `library.catalog_entry(item.asset)`, and reads exactly
three fields off the returned dict (`interior_mm`, `module_bay_mm`,
`tray_floor_mm` — groundwork §1.2). None of that code is CAD-specific; it
operates on whatever dict `self.assets[name]` holds. The only place that
*is* CAD-specific is `AssetLibrary._load_template`, which unconditionally
calls `bpy.ops.import_scene.gltf(filepath=self.assets[name]["file"])`.

### 4.2 The design: a merged namespace, one branch point (judgement)

Extend `self.assets` to hold both CAD entries (`"kind": "cad"`, from
`catalog.json` as today) and procedural entries (`"kind": "procedural"`,
built **in memory** at generation time by §3.2+§3.3, never written to
`catalog.json`). `catalog.json`'s own docstring already scopes it as "the
only thing the Blender side reads **about the CAD**" — keeping procedural
entries off disk preserves that meaning instead of turning the file into a
mixed audit trail of real measurements and ephemeral samples that would
need regenerating identically to stay meaningful.

`_load_template(name)` gets exactly **one** new branch: if
`self.assets[name]["kind"] == "procedural"`, call the §3.4 bpy builder
instead of `import_scene.gltf`; either way, return the same `by_role: Dict[
str, list]` shape `_load_template` already returns, with objects tagged
`ROLE_PROP` the same way. Every downstream consumer —
`AssetLibrary.instantiate`, `scene.build()`'s three-field read,
`bay.py`'s entire consumption surface — is **unchanged**, because none of
it was ever aware `_load_template` imports a file; it only ever consumed
the `by_role` dict and the catalog-entry dict, both of which now have two
possible origins behind one interface. This is the whole answer to "no
second code path": the fork is one `if` inside one already-CAD-specific
function, not a parallel `scene.py`/`bay.py`/`AssetLibrary` for procedural
data.

### 4.3 What does NOT need a fork

`bay.module_rect_local`, `placement_rect_local`, `*_world_placement`,
`sample_obstructions`, `seated_cell_poses`, `obstruction_forbidden_mask`
(groundwork §2) — all already operate on opaque `interior_mm`/`bay_mm`
rectangles. `world.build_pcb`, `build_bay_proxy`, `build_obstructions` all
take generic bounds/placements/poses computed on the bpy-free side
(**measured**, their docstrings and signatures at `world.py:612, 782,
857`) — none of them know or care whether the tray behind those numbers
was imported or built. `scene.py`'s entire `open_case` block (lines
253–360) needs zero changes, provided §4.2's catalog entry supplies the
same three fields a CAD entry does.

### 4.4 The one place variant selection has to be designed in, not assumed (judgement — this is not covered by either input document)

CAD's `keep_roles` mechanism (`config.Variant`) is cheap because a CAD
template already contains a *pre-built* `case` (bottom shell, with the
cavity) and `case_lid` (solid top) as two separate mesh objects from the
STEP file — `keep_roles` just chooses which pre-existing objects to link.
A procedural generator has nothing pre-existing to choose from; it has to
build whatever geometry the drawn variant needs.

The resolution: **build the procedural tray as the same two-piece
structure the CAD already uses**, not a single variant-specific mesh:

- **`case`** — the bottom shell, built by boolean-differencing the cavity
  (sized by bay depth, wall thickness, footprint) out of a lower block.
  This is the piece that carries the cavity when `open_case` links it.
- **`case_lid`** — a solid slab covering the same footprint at the case's
  own top, no cavity, no boolean operation needed.
- **`cell`** — `primitive_cylinder_add` at the sampled format's
  radius/length (§6.1), one template per format actually used.

With this shape, the *existing* `Variant.keep_roles` values
(`assembled: ("cell","case","case_lid")`, `open_case: ("cell","case")`,
`cells_only: ("cell",)`) apply to a procedural asset **completely
unchanged**: `open_case` reveals the cavity-cut shell exactly as it does
for CAD; `assembled` links the shell *and* the solid lid, rendering sealed,
identically to how a real CAD assembly renders sealed; `cells_only` links
neither tray piece. No `case_liner` role is produced — procedural trays
have no CAD liner-vs-shell ambiguity to model, and `keep_roles`/`kept =
{r: ... if r in variant.keep_roles}` already tolerates an absent role
(nothing in the code requires every role to be present).

This matters beyond elegance: `config.VARIANTS`' sampling weights are
`assembled=3.0, cells_only=2.0, open_case=1.0` — **assembled is the
majority of every scene today**. If the procedural builder only knew how
to build an open cavity, roughly half of every procedural-only training
run (Decision 1: procedural is the *entire* training set, not a
supplement) would either render wrong (an unsealed tray labelled and
weighted as if sealed) or need a variant-restricted procedural sampler,
which would itself be an undocumented, silent narrowing of what the model
trains on relative to the CAD-based baseline. Building the two-piece
structure above is what avoids that without inventing new `Variant`
plumbing.

Total case height (bottom shell + lid) needs its own value to split
between the two pieces. **Recommendation (judgement, low-stakes): split it
evenly**, mirroring the CAD's own pattern (11.1mm/11.1mm on all four
SKUs — a 50/50 split, not a coincidence worth second-guessing without
data suggesting otherwise). Easy to revisit if spec #3's realism work
finds a reason to.

---

## 5. The parameter surface

### 5.1 The two-tier fact, restated (measured, groundwork §1.1–1.2)

Only `interior_mm`, `module_bay_mm`, `tray_floor_mm` are read at
scene-build time (`scene.py:265-312`); `tray_outer_mm`, `case_wall_mm`,
`cell_union_mm` exist purely so `inspect_glb` can *derive* the first three
from a CAD mesh that publishes none of them directly, and are kept in
`catalog.json` afterward as audit provenance the Blender side never reads
again.

### 5.2 How a procedural tray satisfies the scene-build tier (judgement, resolving what "cell count, pitch" mean operationally)

Groundwork §1.4's net parameter list does not include "cell count" as an
independent field scene.py needs — and it shouldn't be one. Sampling
footprint *and* cell count *and* pitch as three independent free draws
risks physically impossible trays (a footprint too small for the sampled
cell count at the sampled pitch), which would need rejection sampling to
avoid. Instead:

1. Sample cell format (§1.2), an n × m cell grid, and pitch — this is
   `cell_union`: an n×m layout of that format's footprint at that pitch,
   the exact same arithmetic `bay.seated_cell_poses`'s packer already
   performs, just run once at generation time instead of per-scene.
2. Sample wall thickness. `tray_outer = cell_union` inset outward by wall
   on three sides.
3. Sample (or, for the anchored set, jitter near 11.1mm — §1.1) bay depth
   on the fourth side. This directly gives `module_bay_mm`: a full-span
   strip against that one edge, by construction — never inferred, never a
   tie (groundwork §2's judgement, now made concrete).
4. `interior_mm = tray_outer` (the cavity IS the full interior here, since
   there's no separate liner to inset further — a procedural tray's wall
   *is* the wall the cavity sits inside of, matching `interior_from_tray`'s
   post-conditions without needing to call it).
5. `tray_floor_mm`: sampled or fixed near 1.95mm (measured constant across
   all four SKUs — treat as anchored-near, wide-free, same reasoning as
   §1.1's bay-depth caveat).

This produces `interior_mm`/`module_bay_mm`/`tray_floor_mm` directly, with
`_bay_edge`'s full-span-strip invariant true by construction rather than
validated after the fact, and it is what makes "cell count and pitch" a
real, load-bearing part of the sampler rather than a decorative fifth axis
disconnected from the other four.

`case_wall_mm`/`cell_union_mm` are not needed as *outputs* of this
process the way they are for CAD (groundwork §1.4) — they're consumed
internally by step 1/2 above, never written into the catalog-entry dict,
because nothing downstream ever reads them off a procedural entry either.

### 5.3 No second code path here either

`scene.py`'s three-field read (`entry["tray_floor_mm"]`,
`entry["interior_mm"]`, `entry["module_bay_mm"]`) is satisfied identically
whether `entry` came from `inspect_glb` (measured from a mesh) or §5.2
above (chosen directly) — this was groundwork §1.2's own point, restated
here as a design conclusion rather than an observation: nothing downstream
of `catalog.json`'s three consumed fields cares which kind of process
produced them.

---

## 6. Cell formats: the sites that need to change

Groundwork's full audit (§4.3 of the groundwork doc) stands; not repeated
here in full. What matters for this design:

### 6.1 The one no-override-path site that runs in every scene

`recog/synth3d/world.py:960` — `SEAT_CELL_FOOTPRINT_M = (0.0183, 0.065)` —
feeds `scene.py`'s seated-cell capacity estimate and `bay.seated_cell_
poses`'s packer call, for **every** scene, every asset, with no config key
and no per-asset lookup. This is replaced by the `CELL_FORMATS` table
(§3.1), keyed by whichever format the given item — CAD (always 18650) or
procedural (sampled) — actually uses. `world._assert_seat_cell_footprint`
(the invariant check next to it, lines 965–984) needs the same
generalisation: it currently asserts a cloned cell's measured footprint
against one global constant; it needs to assert against the format the
clone was drawn for. Both are real code changes for the implementation
plan to make — not made here.

### 6.2 The other worst-offender: the orientation gate

`recog/synth3d/_gate_orientation.py:95-107` asserts every `role == "cell"`
object measures `18.3mm`/`65.0mm` **literally**, for every imported
template the gate runs against. This will false-fail the instant a 21700
or 26650 cylinder (CAD or procedural) enters a template the gate checks.
It needs to become per-format (checking against whichever format the
object under test was built as) before any new-format asset exists in a
gated template — this has to land *before* new formats do, not alongside
them, or the gate breaks the build rather than catching a real defect.

### 6.3 Everything already fine (measured, groundwork §4.4 — not re-derived here)

`admits_a_cell` (caller-supplied `cell_w_px`/`cell_h_px`, already generic),
`seg_dataset.py` (fixed-size ROI crops, format-agnostic taxonomy),
the class taxonomy (`"battery"` already covers any cylindrical cell), and
the packer (`Item(id, width, height)` already generic). No change needed
at any of these.

### 6.4 Config calibrations to re-check, not re-derive now

`configs/recognition.yaml:56`'s `anchor_scales` and `configs/synth3d.yaml
:141`'s `max_aspect` are both calibrated against the 18650's own measured
footprint/aspect (3.55, with a measured ceiling of 3.68 over 1245 boxes).
21700's aspect (70/21 ≈ 3.33) and 26650's (65/26 = 2.5) both sit inside the
existing 4.0 band — not urgent — but `anchor_scales`'s docstring warning
(widening the size distribution without retuning anchors once put 15.37%
of boxes below 0.5 best-IoU, `configs/recognition.yaml`'s own measured
table) applies just as much to widening the *size* distribution via new
cell formats as it does to widening it via zoom. This needs a real
re-check once actual per-format pixel figures exist from rendered data —
named here as a required verification step for the implementation, not
performed in this document.

---

## 7. The two procedural sets and the 2×2

Restating Decision 2's table because §9.2 below extends it with the
control:

| trained on | tested on | tells you |
|---|---|---|
| anchored | held-out CAD | does plausible synthetic geometry transfer? |
| wide | held-out CAD | does extra variation help or hurt? |

Two independently trained models (`seg_training.py` run twice, once per
set), each evaluated on the **same** held-out CAD crops, reported as two
separate numbers. Never pooled into one "procedural" figure — pooling
would destroy the exact comparison Decision 2 exists to enable.

---

## 8. Interpretation note carried forward, flagged once more

Decision 3's own text notes the owner's answer selected "21700, 26650 and
'keep 18650 only'" together, and reads that as "all three formats, 18650
retained." This design follows that reading (three formats in the
training mix) and, like the decisions doc, flags it for correction if
wrong — repeated here rather than silently assumed, because it is the
kind of thing that would be expensive to discover wrong after the sampler
and `CELL_FORMATS` table are built around it.

---

## 9. The two open questions

### 9.1 Electronics module position: sampled, but scoped by the anchored/wide split (judgement — recommendation)

**Recommendation: fix it to a short edge for the anchored set; sample
freely among all four edges for the wide set.**

Reasoning:

- The measured CAD ground truth is unanimous but small: **all four** SKUs
  put `module_bay_mm` flush against a short edge (the module_bay spans the
  full *short* axis and sits at one end of the *long* axis — verified
  directly against `catalog.json`'s four `module_bay_mm` entries, e.g.
  PowerCore10000's cavity is 54.9mm (short) × 84.45mm (long) and its
  module bay spans the full 54.9mm width, sitting at one end of the 84.45mm
  length). n=4, not a large sample, but unanimous.
- `bay._bay_edge`/`module_bay_from_bounds` already handle **any** of the
  four edges with no code change — this is a pure sampling-time choice,
  not a capability the code lacks (unlike bay depth, which needed new
  surface). The cost of supporting "sampled" is zero.
- Decision 2 already establishes the right place to put this choice:
  anchored stays "plausible as real hardware" (short edge, matching 4/4
  observed instances); wide is explicitly where "some trays will not
  resemble any real product" lives, and an electronics module on a long
  edge is exactly that kind of unrealistic-but-instructive variation —
  it is not a third, undecided axis, it's an application of a decision
  already made to a parameter Decision 2's own text flagged as open
  ("Real cartridges vary; the current CAD does not").
- This keeps the 2×2 clean: the *only* difference between what anchored
  and wide are allowed to produce is governed by one consistent
  principle (plausible-range vs. deliberately-out-of-range) applied
  uniformly across every axis including this one, rather than one axis
  having its own separate free/fixed rule invented on the side.

### 9.2 Sizing relative to the 502-scene baseline, and CAD-test-set adequacy (judgement — recommendation, most exposed to being wrong)

**Recommendation: render anchored and wide at roughly the same scale as
the existing 502-scene / 841-crop baseline each** (i.e., two independent
~500-scene renders, not the 502-scene budget split between them), and
**render a dedicated CAD-only test set large enough to report a per-SKU
number**, not just a pooled one — concretely, aim for the same per-class
instance density the project already treats as trustworthy (36 `bay` / 36
`electronics` / 24 `obstruction` instances, `docs/receipts/seg_eval.txt`)
**per SKU**, not pooled across all four.

Reasoning, laid out because this is the part most likely to need revising
against real render-time cost once it's tried:

- **Why not split 502 between anchored and wide.** Decision 2 makes these
  two *separately* trained, separately scored models, not two halves of
  one training run. Each has to independently support a full 40-epoch
  schedule at a scale that has already demonstrated a working segmenter
  (the existing 0.8032–0.8126 selected mean IoU, `docs/receipts/
  seg_eval.txt`) — halving the budget (251 scenes each) risks the CAD-test
  score being poor for a reason indistinguishable from Decision 1's own
  named ambiguity ("does the model fail to generalise, or is the
  procedural distribution unrealistic") **plus a third, confounding
  possibility this design would rather not introduce**: "or there just
  wasn't enough procedural training data." Parity with the existing
  working baseline removes that third confound from the outset.
  `docs/NEXT_STEPS.md`'s own documented next-scale-up command
  (`--n 1000`) is already on record as the project's next step past 502
  when more scale is warranted — cited here as a real, already-decided
  ceiling to grow into, not a number invented for this document. Start at
  ~502-scene parity per set; treat any further scale-up as its own
  explicitly labelled step, the same way the 502-scene baseline itself was
  (`docs/NEXT_STEPS.md`, "Step 4 — DONE").
- **Why the CAD test set needs to be bigger than "the 4 SKUs exist."**
  "Is 4 SKUs enough to separate them" is not really a question of *how
  many* SKUs (four is fixed — no more real CAD is coming, ever, per the
  constraint this whole plan works around) but of *how many crops per
  SKU* the per-SKU number is computed over. Today's per-SKU breakdown
  (`docs/receipts/tau_independence_correlation.txt`) runs on 35 crops
  across 4 SKUs — roughly 8-9 per SKU — adequate for a correlation *sign*
  check, not for a confident per-SKU IoU this spec's whole argument now
  rests on. Since decision 1 makes the CAD set purely an eval target (no
  training-time cost to rendering more of it — it never gets trained on),
  the marginal cost of rendering it larger is pure render time, not
  dataset-management complexity. Recommend sizing it so each of the 4 SKUs
  independently reaches the density (~24-36 instances per relevant class)
  the project's own existing receipts already treat as reportable, which
  by the crops-per-scene ratio the 502-scene set already demonstrates
  (~126 validation crops → ~24-36 instances per class, pooled across 4
  SKUs) implies roughly **4× that per-SKU density**, i.e. a CAD-only test
  render on the order of 150-200 scenes, not 20-40. This is a real,
  possibly-significant addition to render budget and is flagged as such
  rather than hidden inside a smaller, more convenient-looking number.

---

## 10. The control the decisions doc requires

The decisions doc names the instrument (a model trained directly on the
CAD assemblies) but not the mechanism. This design uses groundwork §5's
**Option A** (hold out one SKU, train on the other three), repurposed:
not as the main generalisation claim (groundwork already critiques Option
A there as "weak disjointness... closer to a leave-one-out sanity check"),
but exactly as a **ceiling reference** for what a model with *any* exposure
to real CAD geometry can score on the held-out CAD test's masks.

Concretely: four leave-one-SKU-out folds (train on 3, test on the 4th,
repeated for each SKU), reusing the existing `AssetLibrary`/`generate3d.py`
machinery with "a small filter... no new generator code" (groundwork §5's
own phrase for Option A) — no new code beyond the existing catalog
mechanism. Every fold's held-out SKU is scored on the **same** CAD test
crops the anchored- and wide-trained models are scored on for that SKU, so
all three models (anchored, wide, CAD-control) produce directly comparable
per-SKU, per-class numbers on identical test data.

**Why weak disjointness is not a problem for a control the way it would be
for a main claim.** Option A's weakness (all four SKUs share one generative
process, a narrow wall-thickness band, the same bay-placement rule) makes
it an *easy*, generous ceiling — which is exactly the right property for a
reference the procedural-trained models are being compared against. A
control that is too hard to beat would be uninformative in the other
direction. Groundwork's critique of Option A applies to using it as *the*
generalisation measurement; it does not apply to using it as *a ceiling
the generalisation measurement is judged against*, which is a different
use of the same mechanism.

**How this resolves the ambiguity, before a number exists (what §11 states
formally):** if the CAD-control ceiling and the procedural-trained score
are both low on a given class/SKU, that implicates the model's own
capacity (the ceiling itself is low, so no training distribution could do
better). If the ceiling is high but the procedural-trained score is low,
that implicates the procedural distribution specifically (a model *can*
score well on this SKU's masks — it just wasn't taught the shapes that let
it). This is exactly the distinction the decisions doc's Decision 1 risk
paragraph asks for, made mechanical rather than a post-hoc judgement call.

---

## 11. What must not regress

- **Five-class disjointness at 0 overlapping pixels.** Currently
  measured across 3280 mask pairs over the full 502-scene set (Plan B,
  `docs/NEXT_STEPS.md`). Procedural masks are built through the *same*
  occlusion-by-geometry mechanism (`build_bay_proxy` + `seat_cells` +
  `build_obstructions` all sit on the same `floor_z` with the same
  sub-millimetre lift ordering, §4.3 — unchanged) that produces this
  guarantee for CAD scenes today, so it is expected to carry over
  structurally, but it is a **data-quality gate to re-measure**, not an
  assumption to inherit silently: run the same disjointness check against
  the first procedural render before trusting it as training data, exactly
  as Plan B did for CAD.
- **The `assembled` variant.** §4.4 above is the design that keeps it
  working identically for procedural trays (case + case_lid, both linked,
  solid-looking, no cavity visible) — not optional, since it carries 50%
  of the current sampling weight.
- **`unit_id` grouping and unit-scoped VOC boxes** (`scene.py:140-165,
  371-503`). Every annotation belonging to one physical unit — shell,
  module, bay proxy, obstructions, seated cells — shares one `unit_id`;
  VOC boxes merge shell+module+bay+obstructions (not seated cells) into
  one box per unit. Nothing in this design changes how items are grouped
  or labelled; a procedural `Item` flows through the exact same
  `scene.build()` per-item loop a CAD `Item` does (§4.2), so this
  mechanism applies unchanged.
- **The torch-free demo, `python main.py --config configs/demo.yaml`.**
  Nothing in this design touches `main.py`, `configs/demo.yaml`, or the
  synthetic-boxes path the demo runs against.
- **The bpy-free line.** §3 above is the explicit design for keeping every
  new geometric decision on the bpy-free side of it; §3.4's closing note
  is the one discipline risk worth re-checking at implementation time.

---

## 12. How success is measured

Comparisons stated in advance, before any run exists, so none of them get
chosen after seeing a result:

1. **Anchored-trained vs. wide-trained**, both scored on the same held-out
   CAD crops, per class, per SKU. (Decision 2's own question: does extra
   variation help or hurt.)
2. **Anchored-trained vs. CAD-control**, and **wide-trained vs.
   CAD-control**, per class per SKU, on the same crops. (Decision 1's own
   question, made answerable per §10: model-capacity failure vs.
   distribution-realism failure.)
3. **In-distribution vs. out-of-distribution, per procedural model**: each
   of anchored/wide scored on its own held-out procedural validation split
   (same distribution it trained on) versus the CAD test set. The size of
   that drop is a direct, CAD-independent measure of how much the
   procedural-to-CAD gap costs, usable even before the control finishes.
4. **Regression checks**, pass/fail, not comparative: five-class
   disjointness at 0 overlapping pixels on the first procedural render
   (§11); `obstruction`/`battery` IoU on the CAD test set at or above the
   existing 0.6579/0.6907 floor (`docs/receipts/seg_eval.txt`) — a floor,
   not a target, since the CAD test set is now disjoint from training and
   a drop is possible without indicating a bug; the full 621-test suite
   staying green; the torch-free demo still running unmodified.

Every comparison above is stated at the **per-SKU, per-class** level
first, with any pooled/mean figure reported alongside it, never in place
of it — pooling four SKUs of very different footprint (62.9×90.9mm to
81.7×180mm) into one mean is exactly the kind of averaging that would hide
a SKU-specific procedural-distribution gap the whole point of §10's control
is to catch.

**What must never be reported, regardless of any of the numbers above:** a
sim-to-real transfer claim. Every comparison in this section is
synthetic-to-synthetic. This is restated once more here, at the point
where it would be easiest to forget, because it is the constraint the
entire spec exists to work within.

---

## 13. Corrections and disagreements found while writing this

- **`docs/NEXT_STEPS.md`'s "Step 1" and `2026-08-08-tray-interior-design.
  md` §8 are stale against Decision 1, not merely imprecise.** Both state
  spec #2 keeps "the four Anker assemblies... in the mix as real-CAD
  anchors" (NEXT_STEPS.md, dated 2026-08-09) — describing them as part of
  the *training* mix. Decision 1 (dated 2026-08-10, explicitly settled)
  instead holds out **all four** CAD assemblies as a pure test set: "the
  model never sees real measured geometry during training." These are not
  reconcilable as written — "kept in the mix" and "never sees... during
  training" describe different training sets. Per the decisions doc's own
  instruction ("where it does, the measurement wins and this document
  should be corrected, not defended") and per Decision 1 being the later,
  explicit, settled call, this design follows Decision 1 and treats
  NEXT_STEPS.md's phrasing as superseded, not as a second requirement to
  satisfy alongside it. NEXT_STEPS.md's own Step 1 section should be
  corrected to match Decision 1 the next time that file is touched; not
  done here, since this document is design-only.
- **No disagreement found with the groundwork's own recommendations**
  (Option A for the build approach, the bay.py/catalog.py/config.py
  split, the "cell count is not a free field" implication of its §1.4
  parameter list). Where this design goes further than the groundwork
  (§4.2's merged-namespace integration, §4.4's case/case_lid mirror, §5.2's
  cell-count-derives-footprint construction, §9's two open questions) it
  is because the groundwork explicitly left those for "whoever writes the
  spec" (its own words, §3's closing paragraph) — not because anything it
  measured was wrong.

---

## 14. Verification

No code, dataset, or checkpoint changed while writing this document.
`python -m pytest -q` run at the end of this task: 621 passed, matching
`docs/NEXT_STEPS.md`'s baseline and the groundwork document's own
verification, at current HEAD `55e71b0` (this document's own commit is
docs-only, one commit ahead).
