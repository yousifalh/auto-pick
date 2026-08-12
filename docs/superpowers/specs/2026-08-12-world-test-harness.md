# Reaching `recog/synth3d/world.py` from pytest

Date: 2026-08-12
Baseline: `a1f87a2`. Suite at the end of this work: **966 passed, 0 failed**
(`--ignore=tests/test_seeding.py`, which is another agent's in-flight file).
Commits: `1b7969a` (harness, 136 tests), `0d7f742` (one genuine bug),
`080f449` (one loud assertion, 13 more tests). **149 tests in
`tests/test_synth3d_world.py`.**

Files changed: `tests/test_synth3d_world.py` (new) and
`recog/synth3d/world.py`. Nothing else.

---

## 1. The problem this closes

`world.py` is 1,495 lines — the largest file in the project — and until now
**no test in the suite could execute one line of it**, because it imports
`bpy`. `scene.py` and `assets.py` share the constraint.

Every serious silent failure this project has had lived in, or was masked
by, that region:

| defect | why no test caught it |
| --- | --- |
| glTF importer inverted every cartridge; `lay_flat` had no notion of which end was up, so `placement_area` was painted on the outside of a closed lid | orientation is decided in `assets.py`/`world.py`, both bpy-only |
| a two-material object arrived fused and the splitting helper silently did not split it | same |
| an inner cell holder shared a role with the shell and closed every "open" cartridge | role naming is checked bpy-free, but the *objects* being named are built bpy-side |
| a renamed catalog key made a guarded `entry.get(...)` quietly stop building geometry | the guard is in a bpy-only builder |
| a swallowed exception discarded the drawn roughness on 100% of surfaces while the manifest recorded the discarded values | `materials.build`, bpy-only — this is the one that produced `_assert_wear_mix_took` |

All five render plausibly. Four of the five were found by looking at
pixels, months later.

`tests/test_synth3d_materials.py` (added in `d4a1497`) proved the way out:
load the module under a stub `bpy` and assert real behaviour. This extends
that to `world.py`.

## 2. The harness

Same discipline as `test_synth3d_materials.py`, not a second approach: the
module is executed under a private name (`recog.synth3d._world_stubbed`,
with `__package__` set so its relative imports resolve) and every stub is
removed from `sys.modules` — and from the `recog.synth3d` package object —
afterwards. A leaked `recog.synth3d.assets` would be importable and quietly
wrong for every other test in the run; `test_stub_leaks_nothing_into_sys_modules`
pins that it is not, and running this file *first* against
`test_synth3d.py` / `test_bay.py` / `test_scene.py` / `test_synth3d_materials.py`
is green (486 tests).

Two things `materials.py` did not need:

* **`mathutils` is not installable outside Blender either.** `Vector` and
  `Matrix` are stubbed over numpy, so the linear algebra — matrix
  composition, `inverted()` — is not hand-rolled.
* **Objects have geometry.** They are modelled as an axis-aligned local box
  plus a basis matrix composed exactly as Blender composes one:

  ```
  matrix_basis = T(location) @ R(rotation_euler) @ S(scale)
  matrix_world = parent.matrix_world @ matrix_parent_inverse @ basis
  ```

  `transform_apply` bakes the requested components into the box and resets
  them. That is enough to drive the **real** `assets.group_bbox`,
  `assets.lay_flat`, `assets.clone` and `materials.for_role` rather than
  fakes of them.

Because everything downstream is read through the stub, **the stub's own
semantics are pinned first**, against hand-computed values: rotation about
each axis, `mat4 @ vec3` applying translation (a point transform, which is
what `group_bbox` depends on), inverse round-trip, pivot conjugation, the
scale-then-`transform_apply` idiom every builder uses, and the parenting
displacement below. A test that fails is then a statement about `world.py`,
not about an unexamined fake.

## 3. What the 149 tests pin

Not coverage — the properties whose violation is silent.

| § | n | what it pins |
| --- | ---: | --- |
| stub | 7 | the harness itself, against hand-computed values |
| pure decisions | 11 | `_frame_extent` (incl. the property, checked independently of the formula, that the returned extent really does contain the area on both axes), `kelvin_to_rgb` normalisation/clamping, `_lamp_color` tint pulling green off the Planckian locus |
| lighting | 9 | the optional-fill-lamp early return records *nothing*; the fill azimuth is an **offset** from the key's, wrapped; the `off_axis`-with-no-`energy` `ValueError`; a missing HDRI records `hdri: None` and no strength |
| camera | 8 | `ortho_scale = extent x margin x zoom`; **larger zoom = wider frame** (inverted sense, coupled to `model.anchor_scales`); zero/negative zoom raises; shift reaches both object and manifest |
| seating ladder | 5 | see §5 |
| `build_jig` | 6 | the punch-through guard over 5 depths x 25 seeds; plate follows the **pockets**, not `layout_cfg.area`; every cutter applied and removed; `pass_index 0` |
| `build_pcb` | 8 | anchored vs centred fallback and which one the manifest claims; board rests **on** the floor argument; children keep their built world position; board and children turn rigidly about the board's own centre |
| proxy/obstructions | 8 | proxy geometry and rotation; each kind's shape; alpha **with** a matching `blend_method`; the kind vocabulary still matches `bay.sample_obstructions` |
| `seat_cells` | 8 | both early returns; seat position, lift, spin; per-**object**-linked materials so clones sharing one mesh do not share a draw; the footprint check fires, is format-keyed, and its memo does not disable it for later assets |
| tray assertions | 22 | §4 |
| `build_procedural_tray` | 6 | builder and self-check agree; names still classify to their roles |
| `_crown_lid` | 4 | the zero early return (with `bmesh` absent from `sys.modules` entirely); the 4-vert/4-edge guard; **which** edges reach the bevel; only non-axis-aligned faces are smoothed |
| `build_backdrop` | 8 | built at the `z` it was given and records it; procedural fallback; config colour overrides the palette; `uv_scale` **multiplies** the mapping (preserving brushed anisotropy); the bump threshold |
| `_set_recorded` | 13 | §6 |
| source-level | 13 | no builder has grown an exception handler; whole-module `try` budget; the seating constants are defined in exactly one place |

## 4. Historical defects now regression-tested

`_assert_procedural_tray_geometry` runs on every procedural tray built and
**nobody has ever seen it fail**. An assertion nobody has seen fail is not
evidence, so each defect it exists for is reproduced and the assertion is
*required* to fire:

* **an inverted assembly** — a lid *below* the case, i.e. the exact
  geometry of the months-long orientation defect;
* **a lid floating off the rim** — a 1mm light leak into the bay,
  invisible from overhead;
* **a cell off the cavity floor**, floating or buried through the base;
* **a cell standing on its end** — right diameter, right length, wrong
  axes; a circle where the packer reserved a rectangle;
* **a zero-wall entry** — the fused-two-material shape, where the helper
  that should have produced a separate wall silently did not;
* **a renamed catalog key** — parametrised over all eight keys the
  function reads, each required to raise `KeyError`, not default;
* **a bevel that silently did nothing** — `bmesh.ops.bevel` is a **no-op
  in this harness**, and `test_a_crown_whose_bevel_silently_did_nothing_fails_the_build`
  requires the whole build to stop rather than return a flat lid while the
  manifest records a 1.5mm crown. That is the "helper silently did not run"
  class, made reproducible.

Also pinned: `_assert_seat_cell_footprint` fires on a wrong-sized template,
is keyed on the format it was *drawn for* (a 21700 template passes as 21700
and fails as 18650), and its memo does not silence it for a later asset —
otherwise the first correct cartridge in a run would disable the check for
every one after it.

And the parenting bug `build_pcb` documents (a component measured 339mm off
its board) is reproduced in the stub *first*, so the test that the fix holds
is not vacuous.

## 5. Geometric decisions living on the wrong side of the bpy line

**Yes — and it is a clean split: `bay.py` decides XY, `world.py` decides Z.**

Every seating height is a `world.py` literal. Their *mutual ordering* is the
entire mechanism `placement_area` rests on — anything in the bay must sit
strictly above the proxy plane that carries the label, or the mask keeps
reporting that floor as free while an object visibly occupies it, and
nothing downstream can detect it:

```
pcb 0.0008 < proxy 0.0009 < tape/label 0.0011 < adhesive 0.0012
           = SEATED_CELL_LIFT 0.0012 < foam 0.0022
```

Three docstrings restate that invariant in prose. Nothing enforced it.
`test_the_seating_ladder_is_strictly_ordered_and_stays_sub_millimetre` now
does — but the *decision* still lives bpy-side, spread over three functions
and two module constants. I did not move it, per brief.

Three more, in decreasing severity:

1. **`build_jig` makes its own geometric judgements with its own rng
   draws**: plate margin `uniform(0.010, 0.030)`, thickness
   `max(uniform(0.010, 0.018), deepest + 0.004)` — the punch-through guard
   — and the plate footprint from the pocket bounding box. Contrast
   `build_procedural_tray`, whose docstring says outright *"No geometric
   judgement happens here (design spec Sec3.4)"* because every number comes
   from `bay.sample_tray` + `catalog.build_tray_entry`. `build_jig` is the
   same class of object with none of that discipline; `layout.plan_jig`
   produces the pockets but not the plate. A `bay.sample_jig_plate` would
   put it on the testable side.
2. **`build_pcb`'s un-anchored fallback** draws the board's whole rectangle
   (`0.55–0.80` of width, `0.20–0.38` of height, plus centre jitter) in
   `world.py`.
3. **`build_obstructions` decides the third dimension of every
   obstruction** — the adhesive blob's Z squash `uniform(0.35, 0.7)` and
   the foam pad's thickness `uniform(0.002, 0.005)` — while `bay.py`
   decides w/h. Same XY/Z split.

## 6. The bug the harness found — and the assertion added

### 6.1 A perspective render fabricated its own calibration (`0d7f742`)

`setup_camera` recorded `getattr(cam.data, "ortho_scale", None)`.
`bpy.types.Camera.ortho_scale` is an **unconditional** property of the
camera datablock — Blender only *hides* it in the UI when `type != 'ORTHO'`
— so the guard never fired. With `cfg.ortho: false` every sidecar carried
Blender's untouched **6.0** default as though it were a measurement.

That is not a manifest wart. `recog.calibration.frame_mm_per_px` derives
each frame's true ground sample distance as `ortho_scale * 1000 / width`
and hands it to `plan.planner` to size real placements. It raises when the
key is absent *precisely because* a perspective camera has no single scalar
mm_per_px — the scale varies with depth — so substituting one "would be a
fabricated calibration, not a fallback" (its own words). The old line gave
it 6.0 to fabricate from: **4.69 mm/px against a real 0.49–1.05**.

**Latent, not historic.** `CameraConfig.ortho` defaults `True` and every
shipped config sets `ortho: true`, so no rendered dataset carries the wrong
number. It was one config flag from doing so silently — the same shape as
the guarded `.get` that once stopped a builder building. Fixed in its own
commit; both new tests fail against the old line.

### 6.2 `world.py` may not record a value that never reached the shader (`080f449`)

`materials.set_input` returns `False` and continues when a socket is
missing. That tolerance is deliberate and stays — Principled socket names
moved in Blender 4.x, and `materials.build` relies on it to try
`Coat Weight` then `Clearcoat`.

But all seven of `world.py`'s colour/roughness call sites **also write that
value into the `drawn` dict they return**, which `scene.py` puts straight
into the manifest. A silent `False` renders a Blender default while the
manifest states the drawn value — bit for bit the defect
`_assert_wear_mix_took` was written for. `world.py` had no equivalent.

`_set_recorded` raises instead, naming the socket and what the running
build *does* offer. Scoped to `Base Color` and `Roughness`, the names that
have not moved across 3.x/4.x/5.x; `Metallic`, `Alpha` and the obstruction
colours are recorded in no manifest and keep plain `set_input` and its
tolerance — a test pins that distinction. All four builders are driven with
the socket renamed under the stub and required to stop; all eight tests
fail against the plain form. A source-level test requires any future
`set_input` whose value mentions `drawn[...]` to use the checked form.

### 6.3 Two smaller manifest/render disagreements — reported, not fixed

Both are pinned as **current behaviour** with an explicit note in the test,
because fixing them changes what a manifest says and the brief scopes this
work to test-and-assert:

* **`build_backdrop`'s `drawn["source"]`** is computed from the config
  alone — `"image" if spec["image"] else f"proc:{spec['proc']}"` — *before*
  the `os.path.exists` check that actually selects the branch. A configured
  but missing image records `source: "image"` while the procedural branch
  rendered. Note that the HDRI path two functions away gets this right
  (`drawn.update(hdri=None)` on the missing-file branch), so the fix is a
  one-liner with a precedent. `test_a_missing_backdrop_image_falls_back_to_the_procedural_texture`.
* **`drawn["bump"]`** is recorded even when it is `<= 0.005` and no bump
  node is built, so the manifest states a surface displacement that reached
  no shader. Same shape, much smaller.

One structural observation, not a defect: in
`_assert_procedural_tray_geometry` the `module_bay_mm ⊂ interior_mm`
containment check is **exact** while the adjacent wall check tolerates
0.02mm of rounding. That is safe only because `catalog.build_tray_entry`
rounds a flush edge's two copies from the same float and then calls
`bay_edge` on the rounded values itself. An entry builder that derived the
two independently would trip it. Pinned in
`test_the_bay_containment_check_has_no_tolerance_of_its_own`.

## 7. What this harness cannot detect

Stated at the top of the test file as well, because the misreading is
exactly how the earlier defects survived a green suite.

**A stub `bpy` tests our logic, not Blender's.** These 149 green tests say
nothing about:

* **any bpy API change** — a renamed socket, a moved operator, a changed
  `transform_apply` default. The stub would go on implementing the old
  contract quite happily. (§6.2's assertion is the partial answer for
  sockets specifically, and it works at *runtime*, not here.)
* **renderer behaviour** — z-fighting, shadow terminators, the coplanar
  black-plate failure `JIG_LIFT` exists for (whose index pass is *identical*
  either way, so no mask-level gate can see it), whether a bevel looks
  right.
* **`bmesh.ops.bevel` and boolean modifiers** — stubbed as, respectively, a
  no-op and an AABB-preserving no-op. `_crown_lid`'s actual fillet is never
  exercised; only its guard, its early return, and *which geometry and
  radius it passes* are.
* **whether Blender's glTF importer still orients CAD the way
  `flip_if_inverted` assumes.** The inverted-cartridge defect is
  regression-tested *as a geometric configuration* — the assertion fires
  when handed an inverted assembly — but nothing here can tell you the
  importer has started producing one again.
* **anything in `scene.py`** (511 lines) **or `assets.py`** (601), which
  remain unreached. `assets.py`'s helpers are exercised incidentally through
  `world.py`'s calls; its own decision points are not.

The honest summary: this converts "no test can reach 1,495 lines" into
"the arithmetic and control flow of those 1,495 lines are tested, and the
Blender boundary is not". The boundary still needs a render to check, which
is why the loud in-build assertions matter more than the tests do.

## 8. Follow-ups

1. `build_backdrop`'s `drawn["source"]` and `drawn["bump"]` (§6.3) — small,
   with an in-repo precedent for the fix.
2. Move the seating ladder (§5) to `bay.py` as a single ordered table, so
   the Z decisions sit beside the XY ones and the ordering is structural
   rather than asserted after the fact.
3. `bay.sample_jig_plate`, so `build_jig` stops making geometric judgements
   — the same move `build_procedural_tray` already had made for it.
4. The same harness now applies to `scene.py` and `assets.py` unchanged.
   `assets.lay_flat` / `flip_if_inverted` are the highest-value remaining
   targets by the same argument that motivated this work.
