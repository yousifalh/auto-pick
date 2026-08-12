# Fix: sub-millimetre truncation, four orphans, an unrun gate, and the
# contracts that were frozen but never checked

**Scope.** Findings 1-5 of `docs/superpowers/audit/2026-08-12-L-reachability.md`.
Branch `feat/blender-synth-dataset`, on top of `a239794`.

**What changed, in one line each.**

1. `execution/execution.py` quantises millimetres to the wire with `round`, not `int`.
2. Four dead functions deleted: `pack_cartridge`, `attach_placement_area`,
   `catalog.build_catalog`, `types.iter_labels`. `assets.object_cell_format` kept.
3. `.github/workflows/ci.yml` runs the Blender orientation gate, and a test asserts it does.
4. `common/types.py` gained four `__post_init__` validators; the cost was measured.
5. Two docstring-only invariants — `RobotStatusCode` ↔ KRL, and reservation
   quantisation ↔ `_overlaps_forbidden` — now have tests.

Suite: **1210 passed, 1 skipped** (the skip is the gate, which needs `bpy`).
The five files I own went from 73 to 126 collected tests, +53.

---

## 1. Sub-millimetre truncation on every commanded pose

### What it was

`WorkspacePoint` declares floats. The frame carries signed int32 millimetres. The
conversion was `int()` at seven call sites, and `int()` truncates **toward zero**:

```
    12.9 mm -> int() =  12      (0.9 mm lost, inward)
   -12.9 mm -> int() = -12      (0.9 mm lost, inward — the OTHER way)
     0.6 mm -> int() =   0
   349.9 mm -> int() = 349
```

Three separate problems, not one:

* **Magnitude.** Up to 0.999 mm on every pick, place and transport pose. The
  shipping-wall inset is 4.25 mm, so that is 23 % of the margin, and the README's
  two residual unsafe placements are 8.3 % and 5.2 % of an 18.5 mm footprint —
  ≈1.5 mm and ≈1.0 mm. The truncation was the same order of magnitude as the
  defect this project treats as its headline.
* **Sign.** The bias reverses at zero. Two cartridges on opposite sides of the
  origin were both pulled *inward, toward each other*. A systematic error that
  changes sign in the middle of the workspace is not a calibration offset anyone
  can back out.
* **Ordering.** It was applied *after* `WorkspaceBounds.require` validated the
  float, so the value that was checked was never the value that was commanded.

### What it is now

One helper, `execution.execution.wire_mm`, at all seven sites:

```python
def wire_mm(v: float) -> int:
    return int(round(v))
```

`round`, not `floor(v + 0.5)`. Half-up is asymmetric at zero
(`floor(-1.5 + 0.5) == -1` against `floor(1.5 + 0.5) == 2`), which would have
reintroduced the sign discontinuity in a subtler costume. Python's
round-half-to-even is symmetric — `round(-x) == -round(x)` for every `x` — and
unbiased over a population of poses. The error is now bounded by 0.5 mm,
everywhere, on both signs.

Nothing clamps. A coordinate outside int32 still raises `struct.error` in
`pack_command`, which the client turns into an E-stop; saturating to 2³¹-1 would
command a motion nobody asked for. `test_out_of_range_coordinate_fires_the_estop`
still covers that path.

### What `require` validates — the decision

**`require` still validates the float, in the planner. I did not move it.** The
audit is right that "the validated value is not the commanded one" is the
uncomfortable part, so here is the reasoning rather than a shrug.

* Moving the check onto the integer means the *planner* has to quantise, which
  means `plan/` has to know the transport layer's frame width. That is a layering
  inversion in the direction this codebase has been careful to avoid — it is the
  same reason `first_fit_decreasing` lives in `common/packing.py` rather than
  `plan/`, to keep `recog` from acquiring a back-edge into `plan`.
* The hazard the audit names — "with an asymmetric envelope not containing the
  origin, truncation could move a pose that passed `require` outside the bounds it
  was checked against" — was a consequence of an **unbounded-in-one-direction,
  sign-flipping** error of up to 0.999 mm. With `round` it becomes a bounded,
  symmetric ±0.5 mm, which is the property a margin can actually absorb.
* So rather than move the check, I **asserted the property the check is entitled
  to assume**: `test_no_commanded_coordinate_is_displaced_by_more_than_half_a_mm`
  drives 81 poses through a recording server and asserts, against the bytes on the
  wire, that `|wire − float| ≤ 0.5` for every coordinate. Previously that
  relationship was neither bounded nor stated anywhere.

If a future deployment has an envelope whose margin is under 0.5 mm, this
decision should be revisited — and the test above is where the assumption is
written down, so it will be found.

The reverse direction is now documented rather than left implicit: `_read_status`
builds `WorkspacePoint` from wire integers into float fields, so a *reported* pose
has whole-millimetre resolution and cannot distinguish "the arm is at exactly
10 mm" from "it is at 10.4 mm". A comment says so at the construction site.

### Tests added (`tests/test_execution.py`)

* `test_wire_mm_rounds_and_is_symmetric_about_zero` — the boundary and both signs:
  the exact `12.9 / -12.9` pair the audit measured, the tie cases
  (`0.5 → 0`, `1.5 → 2`, `2.5 → 2`), symmetry over ten values, `|error| ≤ 0.5`
  over 8001 coordinates in 0.125 mm steps spanning both signs, and integers left
  untouched.
* `test_move_to_rounds_the_commanded_coordinate` and
  `test_pick_and_place_rounds_every_coordinate_it_sends` — asserted against the
  parsed command frames, not the helper, so a future refactor that stops calling
  `wire_mm` fails. `_ScriptedServer` now records whole commands, not just opcodes.

---

## 2. Four orphans, deleted

None was called from production. Each was a *weaker duplicate* of live code, which
is the property that makes an orphan dangerous rather than merely dead.

| Deleted | LOC | Why it was worse than the live path |
|---|---:|---|
| `plan.bin_packing.pack_cartridge` | 55 | `mm_per_px: float = 0.38` — the placeholder whose removal is recorded in `2026-08-11-scale-calibration.md` as having under-read 24 of 30 cartridges by 27 % at the median and produced 3 unsafe placements. `Planner._pack_cartridge` resolves scale through `_resolve_scale`, which raises `UnknownScale` rather than guessing. |
| `plan.placement_area.attach_placement_area` | 20 | No `try/except`, so `UnknownScale` / `BadDetectorBox` / `PlacementDisagreement` escaped to a caller with no reason to expect them; and no participation in `_drop_areas_measured_at_another_scale`, so the cartridge carried a rectangle whose scale-consistency invariant nothing maintained. |
| `recog.synth3d.catalog.build_catalog` | 35 | Wrote `{"units": "m"}` unconditionally: no STEP length-unit detection, no implausible-extent guard, no unit suggestion, no incremental merge. It also clobbered the existing catalog. `recog/convert_cad.py` does all four. |
| `common.types.iter_labels` | 9 | Zero callers, not even a test, and its documented contract was to drop an unrecognised label **silently** — in the module that declares itself the boundary between all three subsystems. |

Three of the four were exported in `__all__`, which is what made them read as the
supported way to do the thing. Each deletion leaves a comment at the site saying
what was there, why it went, and what to use instead — an orphan that comes back
because nobody recorded why it left is the same defect twice.

**What broke: exactly one test, and it was asserting the defect.**
`tests/test_packing_move.py::test_pack_cartridge_stays_in_plan` asserted
`hasattr(bin_packing, "pack_cartridge")`. Audit E already named it "a green test
guarding nothing"; it was in fact worse than nothing, because it kept the 0.38
default alive and would have blocked its removal. Replaced with:

* `test_pack_cartridge_is_gone_from_both_modules` — the deletion, pinned.
* `test_no_module_re_arms_the_placeholder_scale` — walks `__all__` of both
  `plan.bin_packing` and `common.packing` and fails any exported function that
  takes an `mm_per_px` parameter at all. This is the general form of the defect:
  scale is a property of the frame, and a packing entry point that defaults it is
  guessing invisibly at the call site. That is how 0.38 survived the first time.

Nothing else changed. `pack_cartridge` had no callers; `attach_placement_area` had
none (removing it left `Cartridge` unused in `plan/placement_area.py`, so that
import went too); `build_catalog` had none — `convert_cad.py` imports
`convert_step` / `inspect_glb` from the same module and those stay; `iter_labels`
had none, and its removal made `typing.Iterable` unused in `common/types.py`.

**`first_fit_decreasing` was not touched.** Its freeze is deliberate and correct:
`recog/synth3d/layout.py` and `bay.py` import it directly and datasets on disk
encode its exact output.

### `assets.object_cell_format` — kept, and the reason changed

The audit judged it neutral scaffolding. It is now **live**, because of item 3:
`recog/synth3d/_gate_orientation.py:101` is its only caller, and that file is no
longer wired to nothing. Deleting it would break the gate the same commit turns
on. Keeping it is not a preference; it is a consequence.

### One stale reference I could not fix

`docs/FDR_v3.md:814` names `plan.bin_packing.pack_cartridge` as one of "both
planner call sites". That file is out of my scope for this change. The line now
refers to a function that does not exist and should be reduced to the single live
call site, `plan.planner.Planner._pack_cartridge`.

---

## 3. `_gate_orientation.py` — the guard nobody ran

206 lines, and the only executable check against the `lay_flat` / `place_item`
no-op class of failure: both were complete no-ops for their entire life, every
cartridge rendered unturned, and nothing noticed **because the scene still
rendered**. Its check 2 — two `rot_deg` values must give two different bounding
boxes — is the assertion that would have caught it. It called itself a CI gate.
`ci.yml` ran `pytest` and the demo loop and nothing else.

### It can run in CI, and now does

It needs Blender, but it does **not** need a `blender` binary or a display: `bpy`
is the Blender Foundation's own PyPI wheel, Blender-as-a-module. The gate imports
`recog.synth3d.assets` and loads the four `.glb` templates that are already
committed under `recog/synth3d/assets/`, so it needs no dataset generation, no
GPU, and no network beyond the wheel.

New job `orientation-gate` in `.github/workflows/ci.yml`:

* `ubuntu-latest`, Python **3.11** (the `bpy` wheel is cp311-only, which is why it
  is a separate job rather than a step in the 3.10/3.12/3.14 matrix).
* `apt-get` the X/GL shared libraries the wheel links against even headless.
* `pip install -e . && pip install bpy`, then
  `python recog/synth3d/_gate_orientation.py`.
* **Not** `continue-on-error`, and **not** conditioned on the diff touching
  `recog/synth3d/`. The failure it guards against was invisible in the renders, so
  "the diff looked unrelated" is exactly the evidence that does not count here.

### And a lock so it cannot quietly come undone

The gate was disarmed for months by *absence* — no code was wrong, a workflow step
simply did not exist, and nothing could notice. A CI job cannot detect its own
deletion. So `tests/test_orientation_gate.py` runs in the ordinary torch-free
suite and asserts:

* `ci.yml` names `_gate_orientation.py` in a step that runs it, and the job
  carries no `continue-on-error`;
* the script keeps the properties the wheel invocation depends on (its own
  `sys.path` bootstrap, `__main__` block, `sys.exit(code)`);
* and — skipped unless `bpy` imports — it **runs the gate for real** and requires
  exit 0.

The first of those is the one that would have caught the original state. It is
deterministic, needs no Blender, and turns "someone deleted the CI step" from
invisible into a red suite.

The gate's own header docstring now states that CI runs it and points at the test,
because "this works as a CI gate" was true as a description of the file and false
as a description of the world.

**Honest limitation.** I cannot execute this job from here: there is no `bpy` and
no Blender in this environment, so the workflow YAML is unverified against a real
runner. The apt package list and the cp311 pin are the two things most likely to
need adjustment on first run. That is a smaller risk than the status quo, which
was a guarantee with nothing behind it at all — and the failure mode is a loud red
job, not a silent pass.

---

## 4. Contracts: frozen, and now validated

`common/types.py` had **zero** `__post_init__`. Frozen guarantees a value will not
change; it guarantees nothing about the value being possible.

| Type | Check | What used to construct |
|---|---|---|
| `BBox` | `xmin <= xmax and ymin <= ymax` | `BBox(100, 100, 0, 0)`, width −100 |
| `Detection` | `0.0 <= confidence <= 1.0` | `17.5` and `−3.0` |
| `PickPlacePose` | `grid_row`, `grid_col`, `cartridge_id` non-negative | `−5, −5, −99` |
| `RobotStatus` | `cycle_time_ms >= 0.0` | `−42.0` |

`BBox` is the one that mattered most: `area` clamps with `max(0.0, ...)` and `iou`
returns `0.0`, so an inverted box **laundered itself into a plausible zero-area
value** and propagated into `Detection`, `Cartridge`, `Battery` and
`PlacementArea.rectangle` with nothing downstream able to tell it from a
legitimately empty box. Zero-area stays legal — the docstring calls it a valid
value and `iou` round-trips it — so the check is `<=`, not `<`. NaN fails it,
deliberately: a NaN coordinate is a detector that has diverged and should stop at
the boundary rather than at the first arithmetic that propagates it quietly.

`dataclasses.replace` re-runs `__init__` and therefore re-runs the validation;
that is pinned for `BBox` and `Detection`, because `replace` is the usual way
validation gets skipped and it was moot here only because there was nothing to
skip.

### What is deliberately NOT checked

* **`WorkspacePoint`** — any finite triple is a physically meaningful pose. What
  makes one unreachable is the deployment's envelope (`WorkspaceBounds`), which is
  configuration, not a property of the type. Range-checking here would hardcode
  one cell's geometry into the shared contract. `WorkspacePoint` also served as the
  **control** for the cost measurement below: it gained no `__post_init__`.
* **`PickPlacePose.battery_detection_id`** — `-1` is its documented "no battery"
  sentinel.
* **`Snapshot`** — and this is a decision, not an omission. It is the one type here
  that is not frozen, and every real writer of `mm_per_px` sets it by **attribute
  assignment after construction** (`main.py:540`, `recog.inference`), which
  `__post_init__` cannot see. A constructor check would read as protection while
  being structurally unable to fire on the only path that sets the field. The
  honest guard is `_resolve_scale`, which already rejects `<= 0.0` with "a pixel
  cannot span zero or negative millimetres", and it sits where the value is used.

### Cost, measured

Per construction, mean of two 200 000-iteration runs each side:

| type | before | after | delta |
|---|---:|---:|---:|
| `BBox` | 284 ns | 317 ns | +33 ns |
| `Detection` | 224 ns | 257 ns | +33 ns |
| `PickPlacePose` | 514 ns | 543 ns | +29 ns |
| `RobotStatus` | 408 ns | 442 ns | +34 ns |
| `WorkspacePoint` *(control, unvalidated)* | 219 ns | 213 ns | −6 ns |

The control's −6 ns is the noise floor of this machine.

Then the number that actually decides it. Instrumented, **one `Planner.cycle` on
the planner's own fixture** (one cartridge, 20 batteries) constructs **21** of
these types in total: 1 `BBox` and 20 `PickPlacePose`. At +33 ns that is

> **0.7 microseconds on an 8.4 ms cycle — 0.008 %.**

`Planner.cycle` itself measures 7.96 / 8.15 ms median before and 8.03 / 8.37 /
8.46 ms after, with a 7.9–9.9 ms run-to-run spread dominated by OpenCV and the
packer. The validation cost is four orders of magnitude below that spread, which
is why the end-to-end timing cannot resolve it and why the per-construction
measurement plus the construction count is the honest way to state it. **Not
material.** Detections are built once per frame by the recogniser, tens per frame,
against a ~12.6 ms segmentation forward pass.

The audit found no production code bypasses anything — zero hits for
`object.__setattr__`, `dataclasses.replace` and `__dict__[` outside tests — so
this is about catching bad input at the boundary, not defending against internal
misuse. Nothing in the suite or either demo produced a value that now fails.

---

## 5. Two docstring-only invariants, enforced

### `RobotStatusCode` ↔ `krl_prog/routines.src`

The enum says "Values must not be renumbered without matching updates in
`execution.protocol` and the KRL subroutine". `protocol.py` names the enum only in
prose and packs a raw `int`. The KRL side was a bare literal:

```
      RETURN 2                 ; PICK_FAILED
```

in a file no test read and no import touched. Renumbering the enum silently
redefined what the real controller means by `2` — on hardware, with nothing to
notice until a cell was dropped.

Three tests in `tests/test_execution.py`:

* `test_krl_subroutine_returns_the_numbers_this_enum_declares` parses every
  `RETURN n ; NAME` line and asserts `RobotStatusCode[NAME].value == n`.
* `test_the_krl_coupling_test_is_not_vacuous` — because the regex *is* the whole
  mechanism. If a reformat of the `.src` stops it matching, the test above passes
  over an empty list and the coupling is silently gone again, which is precisely
  the failure being repaired. So it asserts at least two matches, including both
  `SUCCESS` and `PICK_FAILED`.
* `test_codes_the_real_controller_never_emits_are_named_as_such` pins the other
  half of the audit's observation: `UNSUPPORTED_COMMAND` and `VERSION_MISMATCH`
  are simulator-only. The reasoning for their existence is sound; the real
  controller does not participate, and that gap is now asserted rather than left
  to be rediscovered.

### Reservation quantisation ↔ `_overlaps_forbidden`

`plan/scene.py:79` and `plan/planner.py:939` both claim the footprint quantises
outward "matching `common.packing._overlaps_forbidden` exactly", because "any
other rounding lets the mask be tested at one size and marked at another" and a
footprint "marked smaller than it is tested against is exactly how a battery gets
packed into space another one already occupies". Both sides were tested
separately. **Their agreement — the only thing the docstring actually claims — was
asserted nowhere.**

Two tests in `tests/test_packing_forbidden.py` assert it as a set equality, in
both directions, against the real `Planner._cell_block_for` and the real
`_overlaps_forbidden`, over eight footprints: both orientations of an 18.5 × 65 mm
cell at offsets deliberately chosen *not* to be multiples of the 1.5 mm cell,
since whole-cell footprints agree trivially and sub-cell ones are where a
floor/ceil mismatch shows.

* *marked ⊆ tested*: for every cell in the reservation's block, a mask containing
  only that cell must make `_overlaps_forbidden` return `True`. A block larger
  than the tested region fails here.
* *tested ⊆ marked*: a mask that is forbidden **everywhere except** the block must
  make `_overlaps_forbidden` return `False`. If the packer looks at any cell
  outside the block, it sees a 1 here and the test fails.

Verified sensitive rather than assumed so: flooring the far edge instead of
ceiling it (the exact mismatch the docstring forbids) shrinks the block from
`rows 1..46, cols 0..14` to `1..45, 0..13`, and the inverted-mask direction then
returns `True` and fails.

Invariant #2 in the audit's table (`BBox` ordering) is covered by item 4 above.

---

## 6. One flaky test, found and fixed

`test_out_of_range_coordinate_fires_the_estop` failed intermittently during this
work. I measured it before assuming it was mine: **1 failure in 15 runs at
`HEAD`**, on an untouched copy of the file from `git show`. It is a race, not a
regression — `KukaClient.estop` sends the packet and closes in a `finally`, so
`__exit__` can return before the test server's reader thread has appended the
opcode. The CRITICAL log line in the failing run proved the E-stop had gone out.

Fixed with `_ScriptedServer.wait_for_op(op, timeout=3.0)`, a bounded poll applied
at the six sites that assert an E-stop reached the wire. It weakens nothing: the
opcode still has to arrive, it is just given a bounded chance to. 12 consecutive
full runs of the file green afterwards.

---

## 7. Verification

### Suite

**1210 passed, 1 skipped.** The skip is `test_orientation_gate_passes`, which
needs `bpy`; it is a real skip with a stated reason, not a filtered-out test. The
five files I own went 73 → 126 collected, **+53 tests**.

### Demos, before and after

`configs/demo.yaml` — every deterministic statistic is unchanged:

| | before | after |
|---|---:|---:|
| cycles | 10 | 10 |
| cartridges_detected | 37 | 37 |
| batteries_detected | 77 | 77 |
| placement_areas | 29 | 29 |
| queue_poses | 41 | 41 |
| released_reservations | 29 | 29 |
| unreachable_place_targets | 17 | 17 |
| unreachable_batteries | 15 | 15 |
| unreachable_cartridges | 3 | 3 |
| track_dropouts / reacquired / expired | 29 / 5 / 11 | 29 / 5 / 11 |
| expired_placed_cells | 1716 | 1716 |
| geometry_refreshes | 5 | 5 |
| cross_cartridge_conflicts | 2 | 2 |
| duplicate_detections | 0 | 0 |
| **placed** | **9** | **9 / 9 / 10 / 10** |
| **pick_failed** | **1** | **0 / 1 / 0 / 0** |
| **reprojected_placed_batteries** | **2** | **3 / 3 / 3 / 3** |

The three bold rows are the only ones that move, and they move **between runs of
the same build**: `configs/demo.yaml:54-56` documents that the mock server draws a
failed grip from the *unseeded* module-global `random` at
`simulation.drop_probability: 0.02`. Four post-change runs are shown to make that
visible rather than asserted. `placed` and `pick_failed` are directly downstream
of that draw, and `reprojected_placed_batteries` counts placed batteries carried
onto a re-measured grid, so it follows which pick failed. Everything the planner
decides is byte-identical.

`configs/demo_seg.yaml` — **identical, every field**: cycles 1, placed 1,
pick_failed 0, empty_queue 14, cartridges_detected 26, batteries_detected 78,
cartridge_masks 26, placement_areas 7, queue_poses 4, frames_with_scale 15,
bad_detector_boxes 1, released_reservations 3, unreachable_place_targets 2,
unreachable_batteries 57, unreachable_cartridges 1, track_dropouts 25,
tracks_expired 16, expired_placed_cells 572, expired_placed_batteries 1, all
remaining counters 0. That run places one cell and never exercises the drop draw
again, so it is deterministic end to end and is the stronger of the two
comparisons.

That the rounding change moves no planner statistic is expected and worth stating:
`wire_mm` acts on the integers leaving for the controller, and nothing the planner
decides reads back from `RobotStatus.current_pose`.

### Not changed

No metric definition, no dataset, no model. `first_fit_decreasing` untouched.
`docs/FDR_v3.md`, `docs/NEXT_STEPS.md`, `recog/seg_dataset.py` and
`recog/check_annotations.py` untouched.

`plan/placement_area.py` carries one change that is not mine and is deliberately
included: a comment block at `_MAX_CARTRIDGE_EXTENT_MM` recording that the
constant also bounds packing latency (a floor at that bound packs in 2.04 ms
against requirement O3's 8 ms budget; 158 × 314 mm breaches at 8.16 ms). Its other
half landed in `common/packing.py` in `a239794`. Dropping it here would have lost
documentation of a real coupling.

### Tests found asserting a defect

**One**, and it resisted exactly as the brief predicted:
`test_pack_cartridge_stays_in_plan` required the dead `pack_cartridge` to keep
existing, and with it the `mm_per_px = 0.38` default. Corrected as described in
item 2.
