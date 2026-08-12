# Fixing the occupancy footprint and enforcing the workspace envelope

**Date** 2026-08-12 · **Fixes** `docs/superpowers/audit/2026-08-12-E-silent-failures.md`
findings **1** (`plan/planner.py:402`, one occupancy cell marked per battery) and
**5** (`WorkspaceBounds` parsed and never compared against a pose).
**Files** `plan/planner.py`, `plan/scene.py`, `tests/test_planner.py`.

---

## 1. The defect, reproduced

`_build_pose` marked exactly one 1.5 mm grid cell per placed 18.5 × 65 mm
battery:

```python
row, col = self._xy_mm_to_cell(ctg, placement.x, placement.y)
ctg.mark_cell(row, col, CellState.PLANNED)
```

`_pack_cartridge` rebuilds `forbidden = ctg.occupancy.mask_of(FORBIDDEN,
PLANNED, PLACED)` and re-packs from that grid **every frame**, so 571 of the
572 cells a battery covers read FREE on the next cycle.

Measured on `tests/test_planner.py`'s fixture (800 × 600 px at 0.38 mm/px, a
125 × 175 cell grid at 1.5 mm), cycle 1 places one cell and confirms it, cycle 2
plans three more:

| | cycle-2 pose | offset from the placed cell |
|---|---|---|
| **before** | (30.89, 52.64) mm | **dx 1.5 mm, dy 0.0** — ~17 mm of an 18.5 mm-wide battery inside one already in the tray |
| after | (48.89, 52.64) mm | dx 19.5 mm, dy 0.0 — clear |

Nothing raised in the "before" row: the queue looked normal, the `placed`
counter incremented, the mock returned SUCCESS.

## 2. What the occupancy grid now marks

The **whole block the footprint covers, in the orientation it was placed at**:
`floor(x / res) … ceil((x + w) / res)` by `floor(y / res) … ceil((y + h) / res)`,
which is 13 × 44 cells for an 18650 at 1.5 mm.

* **Rotation.** `common.packing.PackedItem.width` / `.height` already report the
  *placed* orientation (they swap when `rotated`), and `_build_pose` was already
  reading them for the centre — so `_reservation_for` reads the same properties.
  A 90° placement reserves 13 rows × 44 columns, not 44 × 13. This matters:
  `pack_best_effort` returns both orientations on the test fixture's 30-item
  pack, and reserving the nominal footprint would leave 31 cells of every
  rotated cell's long axis unmarked — the same defect in a rarer costume.
* **Rounding outward, deliberately.** Near edge floors, far edge ceils — the
  exact convention `common.packing._overlaps_forbidden` uses to *test* a
  candidate. The two have to agree; a footprint marked smaller than it is tested
  against is how this class of bug gets in. Rounding a footprint down is the
  unsafe direction (`_grid_greedy` says so in the same words).
* **Clipping is bounded and checked.** The strip is `pr.height * scale` mm while
  the grid is `int(height_px / px_per_cell)` cells, so the far edge can legally
  sit up to one cell past the last row. More than one cell means the strip and
  the grid describe different rectangles, and `_reservation_for` raises.

A `Reservation` (`plan/scene.py`) records both the cell block and the **mm**
footprint, keyed by the anchor `(row, col)` that travels on the `PickPlacePose`.
`confirm_placement` resolves that anchor back to the whole block: the anchor
cell alone would leave the other 571 cells of a placed battery PLANNED for
ever on success, and free none of them on failure.

### Two interlocks, at two resolutions

`Cartridge.reserve` raises `PlacementCollision` (never absorbs, never counts and
continues) on:

* a block covering any **FORBIDDEN** cell — checked *in cells*, exactly, because
  `_overlaps_forbidden` guarantees a placement that reaches here cannot cover
  one. If it does, a battery is going onto the PCB;
* a footprint overlapping any live reservation — checked **in millimetres**. A
  cell check would be wrong here: cells round outward, so neighbours in one pack
  legitimately share a boundary cell without sharing any physical space. Only
  the mm footprints answer "would these two collide?".

Neither can fire while the packer is correct. That is the point — they are the
alarm for the day it is not, and this project's every serious bug so far has
been silent.

### Releasing the reservations of a queue that was discarded

`cycle` rebuilds the queue from scratch every frame, and `main` executes exactly
one pose per cycle before re-planning. Reservations still PLANNED when the next
cycle starts therefore belong to a queue that is being thrown away, and are
released (`Planner._release_stale_reservations`, counted in
`released_reservation_count`).

This is **required by**, not incidental to, the footprint fix: with the whole
block marked and nothing released, a cartridge fills with batteries nobody
picked within a few frames, the packer then places nothing, and the run reports
"queue empty, job done" over a nearly empty tray — the same silent-wrong-output
class the fix is closing. PLACED reservations are physical and survive.

Ten cycles of the real plan → execute-one → confirm loop now fill a cartridge
left to right at a 19.5 mm pitch with zero overlap, `released_reservation_count`
climbing by 2 per cycle (queue of 3, one executed).

### The cost, stated

Across cycles the pitch is 19.5 mm rather than 18.5 mm: outward rounding
reserves up to one cell (1.5 mm) more than the battery on each axis, so a
following cell starts a cell later. That is ~1 fewer cell per 264 mm row in the
multi-cycle regime. Within a single pack nothing changes — positions there stay
continuous and the placements are byte-identical. Paying 1.5 mm of pitch to stop
17 mm of physical overlap is the right side of that trade, and it is the same
rounding the forbidden-mask test has always applied.

## 3. Tests corrected

**`test_cycle_marks_cells_planned` → `test_cycle_marks_the_whole_footprint_of_every_queued_cell`.**
It asserted `ctg.occupancy.planned_count() == len(queue)` — one grid cell per
battery, i.e. the defect, ratified — the latest in a run of tests in this
project found asserting the bug they cover (`tests/test_bay.py:657` pins "zero
seated cells is fine"; `tests/test_packing_move.py:30` asserts only that a
function exists).

It now asserts:

1. one reservation per queued pose;
2. every reservation covers `13 × 44` cells — the battery's real footprint,
   whichever way round it was placed;
3. the PLANNED region is **exactly the union** of the reservation blocks (a
   union, not a sum: adjacent batteries share a boundary column, and asserting a
   sum would be wrong and would drift);
4. `planned_count() >= len(queue) * 12 * 43`, which is false by three orders of
   magnitude under the old behaviour.

**`test_confirm_placement_success_marks_placed`** and
**`test_confirm_placement_failure_reverts_to_free`** checked the anchor cell
only, which passed just as happily when the other 571 cells were left PLANNED
for ever. They now assert the whole reserved block flipped, and that a failed
place drops the reservation.

**`_scaled_planner`'s workspace fixture widened, ±350 → ±500 mm.** Not an
assertion — a fixture value that was inert until the envelope started being
enforced. Those tests deliberately plan the same 800 × 600 px image at up to
0.50 mm/px, which puts the far edge of the cartridge at ~370 mm; that is a
property of the fixture's framing, not of the scale handling they measure. The
envelope's own behaviour is pinned by the new workspace tests instead.

`tests/test_planner.py:171,206`'s deliberately tiny `WorkspaceBounds(-100, 100,
…)` — the audit's proof that the envelope was inert — are left as they are: both
use extractors that raise before any pose is built, so they never reach the
check either way.

## 4. Tests added

| Test | Proves |
|---|---|
| `test_a_later_cycle_never_plans_a_cell_into_one_already_placed` | **The regression test.** Two cycles, one cell placed in the first, no cell in the second overlapping it — measured in mm, two ways |
| `test_a_rotated_placement_reserves_the_rotated_block` | 44 × 13 upright, 13 × 44 turned |
| `test_unexecuted_reservations_are_released_when_the_queue_is_rebuilt` | 2 released, the PLACED block untouched |
| `test_a_reservation_over_a_forbidden_cell_raises` | the cell-resolution interlock |
| `test_reserving_over_an_existing_footprint_raises` | the mm-resolution interlock, *and* that abutting stays legal |
| `test_a_place_target_outside_the_workspace_raises` | envelope, place side |
| `test_a_pick_point_outside_the_workspace_raises` | envelope, pick side |
| `test_poses_inside_the_envelope_are_emitted_unchanged` | the guard is not a clamp in disguise |
| `test_an_empty_workspace_envelope_is_rejected_at_construction` | an inverted bound reads as a typo, not as a broken planner |

The regression test measures overlap in **millimetres against the real
footprint**, never in grid cells — the grid is the thing under suspicion, so
asking it whether the cells collide would re-ask the code that got it wrong. It
asserts twice:

* **(a) from the poses alone.** An 18.5 mm square centred on a place target lies
  inside that battery's real footprint at *either* rotation, so if two such
  squares intersect the two batteries intersect: `max(|dx|, |dy|) >= 18.5`. This
  is the assertion that fails on the old planner, at dx = 1.5 mm.
* **(b) exactly**, against the reserved footprints, with the intersection area
  re-derived from raw floats in the test file — a collision test that asks the
  code under test whether it collided proves nothing.

Verified before/after by monkeypatching `_reservation_for` back to the old
one-cell behaviour and running the new tests unmodified:

```
FAILED as required: test_a_later_cycle_never_plans_a_cell_into_one_already_placed
    pose at (30.9, 52.6) mm is 1.5 mm / 0.0 mm from the cell placed last cycle …
FAILED as required: test_cycle_marks_the_whole_footprint_of_every_queued_cell
    1 cells for a 1.5 x 1.5 mm footprint at 1.5 mm — the block does not cover the battery
passes with the fix: (both)
```

---

## 5. Finding 5 — the workspace envelope: **raise**, not clamp

`WorkspaceBounds` is now enforced in `_build_pose`, on **both** points of every
pose, via `WorkspaceBounds.require` → `OutOfWorkspace`. The place target is
checked *before* a battery is consumed, so an out-of-envelope target does not
also silently spend one.

**Why raise.**

1. **Clamping produces a wrong motion, not a safe one.** A place target is where
   a cartridge slot physically *is*. Moving it onto the envelope's edge does not
   make it reachable — it inserts the cell into a wall, and the twin then records
   that slot PLACED. A pick point is where a battery physically *lies*; a clamped
   pick grasps empty table, or the neighbour. In both directions clamping
   converts a planning/calibration bug into a slightly-wrong motion that nothing
   downstream can distinguish from a correct one. That is precisely the
   silent-degradation class this audit is closing; it would be a new instance of
   it, added by the fix for an old one.
2. **The trigger is a whole-run configuration error, not a per-frame
   transient.** An out-of-envelope pose means `mm_per_px`, `origin_offset_*` or
   the bounds themselves are wrong — the same shape as `UnknownScale`, which
   this module already re-raises past its own blanket handler for exactly that
   reason (`_ensure_placement_areas`'s comment).
3. **Nothing else in the system will catch it.** A separate audit found the mock
   controller accepts `MOVE_TO(5000, 5000, 5000)` on a 706 mm-reach arm, and the
   16-byte wire format carries no envelope. This is the only place that can say
   no.

**The one judgement call.** An out-of-envelope *pick* is arguably a scene
condition — a battery lying outside the arm's reach — rather than a bug, and
skipping it instead of raising is defensible. It raises here because with a
fixed-mount camera the envelope is meant to describe the region the camera
images, so a detection outside it means the envelope or the origin offset is
wrong; and because silently skipping would make "queue shorter than expected"
indistinguishable from "no batteries", which is the same silent class again. If
a real cell ever does image beyond the arm's reach, the right change is an
explicit, **counted** filter on `available_batteries` — not a clamp, and not a
silent skip.

`WorkspaceBounds.__post_init__` rejects an inverted or zero-extent envelope at
construction, where the numbers came from: such an envelope would reject every
pose and read as a broken planner rather than as the typo it is.

`main.py:153-160`'s `.get(..., 350)` defaults remain (that file belongs to
another change in flight): a renamed key still yields a plausible ±350 mm
envelope, but it is now an envelope that *does something*.

---

## 6. Suite

811 passed. The two red tests in the run
(`tests/test_execution.py::test_mid_frame_close_is_fatal_and_the_stop_is_attempted`,
`::test_handshake_timeout_retries_then_escalates`) pass in isolation — they are
port-collision flakes from another agent's concurrent run against the same
hard-coded mock-robot ports, in a file this change does not touch.

## 7. Concerns left open

* **`main.py` cannot see `released_reservation_count`.** The counter exists on
  the planner; `main.run` copies the other three planner counters into `stats`
  and would want this one too. `main.py` belongs to another change in flight.
* **B5 compounds with this.** An unrecognised robot status code becomes
  `TIMEOUT` → `confirm_placement(..., False)` → the whole footprint reverts to
  FREE. If the robot actually completed the place, the twin now believes a
  571-cell region is empty and the next cycle plans into it. The blast radius of
  B5 is larger after this fix than before it, because the reverted region is now
  the real one. Being fixed separately.
* **B6 (`px_per_cell` clamp) interacts with the new clip check.** Under the
  clamp the grid is *larger* than the strip, so `_reservation_for`'s
  "more than one cell past the edge" guard cannot fire in that direction; it is
  a guard against the opposite disagreement only.
* **`EnvironmentModel.summary()["placed_cells"]` changed meaning by 572×** — it
  counts grid cells and always did, but it read like a battery count while a
  battery marked one cell. Documented in place; nothing outside `tests/` reads
  it.
