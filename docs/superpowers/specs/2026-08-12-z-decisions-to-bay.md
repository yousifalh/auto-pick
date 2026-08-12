# Moving the Z decisions to `bay.py`

Date: 2026-08-12
Baseline: `82cff22`, 1032 passing. Suite at the end of this work:
**1074 passed, 0 failed** (`+37` in `tests/test_bay.py`, `108 -> 145`; `+5`
in `tests/test_synth3d_world.py`, `149 -> 154`, four of which are rewrites of
tests that asserted the old structure).

Files changed: `recog/synth3d/bay.py`, `recog/synth3d/world.py`,
`tests/test_bay.py`, `tests/test_synth3d_world.py`. Nothing else.

---

## 1. The line, and where it was broken

This project holds one architectural line: geometric decisions live in
bpy-free modules (`bay.py`, `catalog.py`, `config.py`) so they can be
unit-tested; `world.py` and `scene.py` import bpy and merely *build* what was
decided. The `2026-08-12-world-test-harness` pass found it **held for XY and
not for Z** - `bay.py` decided every footprint, `world.py` decided every
height, including the six offsets that ARE the `placement_area` occlusion
mechanism:

```
pcb 0.0008 < proxy 0.0009 < tape/label 0.0011 < adhesive 0.0012
           = SEATED_CELL_LIFT 0.0012 < foam 0.0022
```

`build_bay_proxy` builds the plane carrying the label, and
`annotate.boxes_from_mask` reports only the pixels of it that stayed *visible*
in the index pass. So an object seated at or below the proxy stops occluding
it and the mask goes on reporting that floor as free while the object sits on
it - in the render, the index pass and the manifest alike, with nothing
downstream able to tell. Every `placement_area` label in every dataset rests
on those six numbers being in that order. Three docstrings restated the
ordering in prose; until the harness pass nothing enforced it, and it was
enforced on the untestable side of the line.

## 2. What moved

Four things, all into `bay.py`, all verbatim.

| moved | to | why it is a decision |
| --- | --- | --- |
| the six seating offsets + `SEATED_CELL_LIFT` | `bay.SEATING_LADDER`, reached via `bay.seat_z(floor_z, rung)` | they decide what occludes what; §1 |
| `build_pcb`'s board thickness `0.0016` | `bay.PCB_THICKNESS_M` | not a free parameter - it is what makes the `pcb` rung's position *below* the proxy legal (§3) |
| `build_pcb`'s un-anchored fallback rectangle | `bay.fallback_module_placement` | it is the un-anchored half of the same decision `bay.module_world_placement` already makes; `build_pcb` consumes the two interchangeably |
| `build_obstructions`' adhesive Z squash and foam thickness | `bay.obstruction_z_scale` | `sample_obstructions` already decided w and h; this was the third axis of the same object, drawn 300 lines away in a module no test could reach |

What stayed in `world.py` is construction: which primitive, which material,
which parent, which pivot. `build_obstructions` now chooses a shape and a
surface and nothing else.

`bay.py` stays bpy-free - `tests/test_synth3d.py`'s AST check is green, and
nothing added here imports anything new.

## 3. The ordering invariant

`bay.SEATING_LADDER` is an ordered tuple of `(rung, offset)`. The offset is
the Z at which world.py builds that object's **origin** (not its base - the
board and the obstruction primitives all straddle their own origins;
`seated_cell` is the one rung that is a base, because `seat_cells` measures
the clone it just laid flat and translates its lowest point onto the seat).

`bay.assert_seating_ladder_ordered` raises `ValueError` - not `assert`,
because `python -O` strips assertions and an ordering that silently stops
being checked is the failure this exists to rule out - unless:

1. no rung is named twice and `bay_proxy` appears exactly once;
2. the table is non-decreasing **as written**, so reading it top to bottom is
   reading the physical stack bottom to top. Ties are legal and two are real
   (tape/label, adhesive/seated_cell): coplanar objects never occlude *each
   other* and nothing asks them to;
3. every offset is in `(0, MAX_SEAT_OFFSET_M = 0.003)` - a clearance, not a
   stand-off;
4. **every rung above `bay_proxy` clears it strictly.** This is the one the
   label rests on, and it is a tie, not an inversion, that loses it;
5. every rung below it is strictly below;
6. `pcb` - the only such rung - is at `PCB_THICKNESS_M / 2`, which is what
   rests the board *on* the cavity floor, and its top face at
   `PCB_THICKNESS_M` clears the proxy, which is what makes the board occlude
   the proxy along the edge they share rather than the reverse.

Clause 6 is why `PCB_THICKNESS_M` came along. Without it the ladder has one
rung that looks like a violation and a paragraph of prose explaining that it
is not; with it, that explanation is a check.

### Enforced in three places, deliberately

* **`bay.assert_seating_ladder_ordered`**, called by **`seat_z` on every
  call**. Seven comparisons against a seven-row table, a handful of times per
  bay. A perturbed table stops the first build.
* **`world._assert_clears_bay_proxy`**, on every obstruction and every seated
  cell built, and **`world._assert_board_straddles_bay_proxy`** on every
  board. These re-derive the proxy plane from the ladder and check the number
  that actually reached the geometry. A bpy-free function returning correct
  numbers does not help if `world.py` stops calling it - which is exactly what
  it used to do - so this is the `_assert_*` precedent applied to the thing
  built, not the input it came from.
* **`test_every_bay_builder_asks_bay_for_its_height`** (source-level, AST):
  each of `build_pcb`, `build_bay_proxy`, `build_obstructions`, `seat_cells`
  must contain a `bay.seat_z` call. A builder that went back to a literal
  would still pass every geometric test - they would simply be testing the
  literal.

### What the ordering check does NOT do

It pins the **order**, not the **values**. Nudging `foam` up by one ULP leaves
every relation intact and passes, by design: a check that also pinned values
would be two checks wearing one name, and the one that fired would not say
which property broke. The values are pinned separately and literally by
`test_the_seating_ladder_is_exactly_the_six_offsets_it_has_always_been`, which
writes the table out rather than deriving it - a diff on those lines is the
intended way to notice that the ground truth of every dataset just changed.
`test_the_ordering_check_pins_the_ORDER_and_not_the_VALUES` states this
outright, because believing otherwise is the obvious misreading.

The world-side expected offsets are also written out as literals rather than
read from `bay.SEATING_LADDER`: read from the table they would agree with it
by construction, and a retune would have to be made in two files that do not
import each other before anything goes quiet.

## 4. Proof that no height changed

Not asserted - measured. `z_probe.py` loads `world.py` under the existing
stub-bpy harness and drives every affected builder, recording each result as
**`float.hex()`**, so a one-ULP difference is a textual difference:

| builder | cases |
| --- | ---: |
| `build_pcb` (40 seeds x anchored/fallback x 4 floors x 5 rotations) | 1600 |
| `build_bay_proxy` (20 seeds x 3 floors x 3 rotations) | 180 |
| `build_obstructions` (9 mixed obstructions, 40 seeds x 3 floors; plus each kind alone, 10 seeds x 2 floors) | 200 |
| `seat_cells` (20 seeds x 3 cell formats x 3 floors) | 180 |
| `build_jig` (30 seeds - unchanged, included as a control) | 30 |

Each record carries every object in the scene by name (board, 3-7 components,
1-4 ports, inductor, proxy, obstructions, cells), its world AABB, its full 16
`matrix_world` floats, the builder's whole `drawn` dict, **and a three-draw
fingerprint of the rng state afterwards** - so a change in how many draws a
path takes, or in what order, shows up even where the geometry happens to
coincide.

**2190 records, 0 differing.** Before: `82cff22` working tree. After: this
change. Re-run at the end of the work, still 0.

The probe is not vacuous - it is ULP-sensitive to every quantity that moved:

| perturbation | result |
| --- | --- |
| `foam` rung +1 ULP | 90 / 2190 records differ |
| `foam` z-scale +1e-9 | 140 / 2190 differ |
| `foam` rung +0.0001 (the brief's figure) | 140 / 2190 differ |
| `fallback_module_placement`'s last two draws swapped | 800 / 2190 differ |
| `pcb` rung +1 ULP | **build stops** - `ValueError` from clause 6 |

No render, no dataset regeneration, no retraining.

## 5. The one behavioural difference, on unreachable input

`build_obstructions`' shape dispatch ends in a bare `else:  # label`, so a
fifth kind added to `sample_obstructions` without a matching branch would
render as a printed label - silently and plausibly, the same shape as the
renamed catalog key that once made a guarded `.get` stop building geometry.
`bay.obstruction_z_scale` raises `ValueError` on an unknown kind and is called
*before* that dispatch, so the build stops instead.

This is the only input/output difference in the change, and it is on input
`sample_obstructions` cannot currently produce - pinned by the pre-existing
`test_the_obstruction_kind_vocabulary_still_matches_bays` and by a new
`test_every_kind_sample_obstructions_produces_has_a_z_scale`, which sweeps 600
seeds and requires the two functions' kind vocabularies to be the same set.

## 6. What deliberately did not move

**`build_jig`'s plate margin, thickness and footprint.** The harness report
suggested `bay.sample_jig_plate`. That would be a category error, twice over:

* `bay.py` is "every geometric decision about a **cartridge's interior**". A
  jig plate is neither a cartridge nor an interior - it is a fixture that
  holds loose parts for a jig scene. Moving it there for symmetry with
  `build_procedural_tray` would make `bay.py` mean "the bpy-free module",
  which is a build-system fact, not a subject.
* Its real home is **`layout.py`**, which is already bpy-free and already owns
  this exact problem: `layout.plan_jig` packs the parts, emits the pockets the
  plate is sized from, and **draws the pocket depths itself**
  (`depth=rng.uniform(*cfg.jig_depth)`) - the very quantity the punch-through
  guard `max(uniform(0.010, 0.018), deepest + 0.004)` defends against. The
  margin, the thickness and the pocket bounding box all read from the same
  `layout_cfg` section (`jig_depth`, `jig_wall`, `jig_clearance`). A
  `layout.plan_jig_plate` would sit beside `plan_jig` with the same config and
  the same units.

It is also the weakest case on testability grounds now: the punch-through
guard is already covered under the stub harness over 5 depths x 25 seeds
(`test_the_plate_is_never_thinner_than_its_deepest_pocket`), and the plate
footprint by `test_the_plate_follows_the_pockets_not_the_layout_area`. The
argument for moving it is about ownership, not reach - and the right owner is
out of this task's file set. **Recommended follow-up: `layout.plan_jig_plate`,
not `bay.sample_jig_plate`.**

`JIG_LIFT` and `JIG_BACKDROP_GAP` stay in `world.py` for the same reason and a
stronger one: they are not bay geometry at all but a *rendering* artefact - a
plate coplanar with the backdrop renders entirely black while its index pass
is identical, which is a fact about Blender's shadow rays, not about
placement. `test_each_seating_constant_is_defined_in_exactly_one_place` now
pins that split explicitly: `SEATING_LADDER` / `PCB_THICKNESS_M` /
`MAX_SEAT_OFFSET_M` / `BAY_PROXY_RUNG` in `bay.py` only, `JIG_LIFT` /
`JIG_BACKDROP_GAP` in `world.py` only, and `SEATED_CELL_LIFT` nowhere.

## 7. Tests that asserted the old structure

Four, all in `tests/test_synth3d_world.py`, all fixed rather than deleted:

1. `test_a_seated_cell_clears_the_proxy_too` compared `W.SEATED_CELL_LIFT`
   against a literal - a comparison between two constants. Now builds a cell
   through `seat_cells` and measures its bottom against the proxy.
2. `test_the_seating_ladder_is_strictly_ordered_and_stays_sub_millimetre`
   asserted a hand-written list of the offsets was sorted, i.e. that a literal
   list in the test file was sorted. Replaced by
   `test_the_seating_ladder_holds_in_the_geometry_world_py_actually_builds`,
   which builds one of everything at a common floor and reads the ordering
   back out of six real objects. The table's own ordering is now `test_bay`'s
   subject.
3. `test_seated_cells_rest_on_the_lift_above_the_floor` read
   `W.SEATED_CELL_LIFT`; now a local literal, per §3.
4. `test_world_is_the_only_place_these_seating_constants_are_defined` asserted
   `SEATED_CELL_LIFT` was defined in `world.py`. Renamed and rewritten per §6;
   it also now walks `AnnAssign` as well as `Assign`, without which an
   annotated `SEATING_LADDER: Tuple[...] = (...)` would have been invisible to
   it - the check had a hole that only an annotated constant would find.

`test_every_obstruction_kind_rests_strictly_above_the_proxy` was not broken
but was near-tautological: it took the object's bbox and then asserted against
a hardcoded per-kind dict, never reading the geometry. It now measures the
built object's own origin.

## 8. Found, not changed

* **Foam pads and adhesive blobs sink below the cavity floor.** A foam pad is
  a cube seated at `+0.0022` with a thickness drawn from `0.002-0.005`, so at
  the thick end its underside is at `floor - 0.0003`; an adhesive blob is a
  sphere of radius `w/2` seated at `+0.0012` and squashed by `0.35-0.7`, so a
  10mm blob's underside is 0.55mm to 2.3mm *below* the floor. Harmless for
  occlusion (the ladder
  orders **origins**, and every one of these has its top well clear of the
  proxy) and invisible from an overhead ortho camera, but it means "seated
  on the floor" is not literally true for the solid kinds, and a future
  side-view or a physically-lit render would show foreign matter interpenetrating
  the tray. Left exactly as it was; flagged because the ladder's docstring now
  has to say "origin, not base" to be accurate.
* **`MAX_SEAT_OFFSET_M = 0.003` is a fossil of the old test's `< 0.003`**, and
  `foam` at `0.0022` is already 2.2mm - the constant's own docstring calls
  these sub-millimetre clearances and one of them is not. The bound is
  preserved bit-for-bit from the test it came from rather than retuned.
* The two `2026-08-12-world-test-harness` §6.3 follow-ups
  (`build_backdrop`'s `drawn["source"]` and `drawn["bump"]`) are still open and
  untouched by this work.

## 9. Unrelated flake seen once

`tests/test_execution.py::test_protocol_mismatch_is_fatal_without_retrying`
failed for two parameters in one full-suite run and passed in every run
before and after (three further full-suite runs, six isolated runs of that
test). It is a socket/timing race in the fake KUKA controller; nothing in
`test_execution.py` imports anything under `recog/synth3d`, so it is not this
change. Worth a look in its own right.
