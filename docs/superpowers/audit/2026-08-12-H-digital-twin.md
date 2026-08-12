# Audit H — the digital twin (`plan/scene.py`)

**Scope.** Cross-frame identity, occupancy state machine, the new
whole-block reservation model, battery assignment, coordinate frames.
**HEAD** `39429a4`. **Read-only** — nothing in the tree was modified.

Findings marked **[X]** were **verified by executing** a multi-frame
scenario through the real `Planner`/`EnvironmentModel`; findings marked
**[R]** were **inferred by reading** only.

**Verdict: the twin is not sound.** The occupancy grid is authoritative
for safety decisions and it is destroyed by two ordinary perception
events (a one-frame dropout, and a cartridge that moves), in both cases
silently and in both cases producing a *plausible-looking* queue aimed at
physically occupied space. The reservation/release change (finding 6) is
itself correct for `main`'s one-pose-per-cycle loop; it is the
frame-to-frame identity layer underneath it that does not hold.

---

## Ranked findings

### 1. A cartridge missing for ONE frame comes back empty, and the next queue aims at the slots that are already full **[X]**

`plan/scene.py:506-509` (`_match_or_insert_cartridges`, the drop loop) —
`del self.cartridges[cid]` for anything not matched this frame.

The delete takes the whole `Cartridge` object, so it takes **everything**:
the `OccupancyGrid` (every PLACED cell), the `reservations` dict (every
PLACED footprint — the only input to the millimetre collision interlock),
`placeable_rectangle`, `pcb_mask`, `mm_per_px`, and the ID itself. On
return the cartridge is re-inserted at `plan/scene.py:498-504` with a
**fresh** `_next_cartridge_id` and no components; the planner then
re-extracts a virgin all-FREE grid.

Executed, three frames, real planner and heuristic extractor:

```
frame1: id=0  placed_cells=572  reservations=4
frame2 (cartridge not detected): cartridges=[]      queue=0
frame3 (identical box returns):  ids=[1]  placed_cells=0
        first place target (29.4, 52.6) mm  ==  frame1's (29.4, 52.6) mm
        same slot re-planned? True
```

**Symptom: indistinguishable from success.** The queue is full-length,
every pose is inside the envelope, `stats["placed"]` increments, the run
reports a clean finish. The arm drives an 18650 into a slot that already
contains one. Nothing raises — the mm interlock (`Cartridge.reserve`,
`plan/scene.py:227-234`) compares against `self.reservations`, which was
deleted with the cartridge, so it has nothing to compare against. No
counter moves: `released_reservation_count`, `rescaled_area_drop_count`
and the three unreachable counters are all untouched by this path.

A single dropped frame is the *ordinary* case, not the adversarial one:
one low-confidence frame, one gripper arm across the box, one partial
occlusion below the IoU floor. And the loss is unbounded — a cartridge
five batteries deep loses all five.

Related, same root cause: an **in-flight pose** whose cartridge vanished
from the next snapshot is confirmed into nothing. `plan/planner.py:764-766`
returns silently when `cartridge_id not in self.env.cartridges`, while
`main.py:582-586` still counts `stats["placed"] += 1`. Verified: the
placement is recorded nowhere and the counter says it happened.

### 2. A cartridge that MOVES keeps its old geometry — place targets frozen at last-frame millimetres **[X]**

`plan/scene.py:492-494` updates `bbox` and `confidence` on a match but
deliberately keeps the components; `plan/planner.py:447-449`
(`_ensure_placement_areas`) then skips any cartridge that already has a
`placeable_rectangle`. **The rectangle is extracted exactly once and never
again.** `_drop_areas_measured_at_another_scale` only re-extracts when the
*scale* changes — a cartridge that moves at constant scale is never
re-measured.

Executed, two frames, cartridge translated 100 px:

```
IoU(frame1 box, frame2 box) = 0.753  -> same id, components kept
cartridge moved 100 px = 38.0 mm
place target moved                    0.00 mm
ERROR +38.0 mm     rescaled_area_drop_count = 0
```

Because IoU 0.5 tolerates a translation of up to one third of the box
width (verified: 30 % of width → IoU 0.538, still matched), the twin will
keep commanding the old millimetres through a drift of ~90 mm on the test
fixture. **Symptom: looks like success** — the pose is in-envelope, the
grid marks the cell PLACED, the counters increment; the battery goes into
the cartridge wall, onto the PCB, or onto the table.

The same mechanism covers the workflow case the machine exists for:
**pull a full cartridge and drop an empty one into the same fixture.**
If the new box lands at IoU ≥ 0.5 the twin keeps the *old* cartridge's
full grid — the empty cartridge reads full, the packer places nothing,
and the run reports "queue empty, job done" over an empty tray.

### 3. Two twin entries for one physical cartridge double-book the same space, and no interlock fires **[X]**

`plan/scene.py:496-504` — a detection that matches nothing at IoU ≥ 0.5
becomes a new cartridge, unconditionally. A segmenter that splits one
cartridge into two boxes, or a duplicate that survives NMS at 45 % IoU,
therefore produces two `Cartridge` objects over the same physical object,
each with its own rectangle, its own grid and **its own `reservations`
dict**. `Cartridge.reserve`'s mm interlock is per-cartridge, so it cannot
see the other one.

Executed (`IoU(cart, dup) = 0.479`, 40 loose batteries):

```
queue=40  per-cartridge: {0: 38, 1: 2}
overlapping target pairs across the two entries: 8
   ctg0 (66.39, 52.64)  vs  ctg1 (67.39, 83.04)  -> dx=1.0  dy=30.4 mm
   ctg0 (84.89, 52.64)  vs  ctg1 (85.89, 83.04)  -> dx=1.0  dy=30.4 mm
no PlacementCollision raised
```

An 18.5 × 65 mm footprint at dx = 1.0 mm, dy = 30.4 mm overlaps by roughly
17.5 × 34.6 mm. **Symptom: success**, twice, into the same hole.

### 4. Re-identification is first-come greedy over detection order, not best-first **[X]**

`plan/scene.py:481-494`. For each detection **in snapshot order**, take the
best IoU over the not-yet-matched cartridges; accept at ≥ 0.5. No
global assignment (no Hungarian), no sort by IoU, no motion model, no
track age, no hysteresis, and — verified — the 0.5 threshold is a
**default parameter that no caller and no config ever sets**
(`plan/planner.py:281` calls `update_from_snapshot(snapshot)` bare; nothing
in `configs/` mentions it).

Consequences established concretely:

* **Identity steal on a tie or on order.** Two overlapping cartridges,
  detections re-ordered: verified a detection with IoU 0.667 against
  *both* took `id=0` (tie broken by dict insertion order), inheriting
  cartridge 0's PLACED cells while wearing cartridge 1's box; the true
  cartridge-0 detection (IoU 1.000 with id 0) then found it taken, fell
  below threshold against id 1, and was inserted as a **new empty
  `id=2`**. One frame, and the twin's memory of which cartridge holds
  what is transposed.
* **Both cartridges lost at once.** Two adjacent cartridges merged into
  one box by the segmenter scores below 0.5 against either → both are
  deleted (finding 1) and one empty entry replaces them.
* **Occlusion floor.** Verified thresholds: identity survives a 30 %
  translation and a shrink to 75 % linear (56 % of area), and is lost at
  34 % translation / 70 % linear (49 % of area). A cartridge whose box
  is occluded to just under half its area is a *new object* to the twin.

**What happens to occupancy on a mis-match is the answer to the question
posed:** the mis-identified cartridge *inherits the other's grid wholesale*
— PLACED cells, reservations, rectangle and scale — because the match
branch replaces only `bbox`/`confidence`/`detection_index`. There is no
consistency check that the inherited grid still describes the box it is
now attached to.

### 5. `PLACED → PLANNED` and `PLACED → FREE` are both permitted **[X]**

Enumerated transitions and the paths that allow them:

| transition | path | permitted? |
|---|---|---|
| FREE → PLANNED | `reserve` → `set_block` | yes, intended |
| PLANNED → PLACED | `confirm(True)` | yes, intended |
| PLANNED → FREE | `confirm(False)`, `release_planned` (`only_from=PLANNED`) | yes, intended |
| FORBIDDEN → anything | blocked in `reserve` (`scene.py:221-226`); **not** blocked in `set_block`/`mark_cell` | partially |
| **PLACED → PLANNED** | `reserve` of an abutting block: `set_block` at `scene.py:236-237` passes **no `only_from`** | **yes — should be impossible** |
| **PLACED → FREE** | the above, then `release_planned` or `confirm(False)`, which now see the cell as PLANNED | **yes — should be impossible** |
| double-reserve same anchor | caught by the mm interlock | no (good) |
| confirm a cell nobody planned | `KeyError` (`scene.py:250-252`) | no (good) |
| **confirm(True) then confirm(False) on the same anchor** | second call deletes the PLACED `Reservation` while leaving the grid PLACED | **yes — the mm interlock silently loses that battery's footprint** |

Executed:

```
A reserved cols[0:13], confirmed PLACED   -> placed=572, col12=PLACED
B reserved abutting at x=18.5 (cols 12:25)-> col12=PLANNED   placed=528
release_planned()                          -> col12=FREE      placed=528
   (A's Reservation still present, state=PLACED)
same via confirm(B, success=False)         -> col12=FREE      placed=528
```

44 cells of a physically placed battery revert to FREE. **In the current
pipeline this is latent, not live**: `_pack_cartridge` masks
FORBIDDEN|PLANNED|PLACED and `_overlaps_forbidden` rounds outward, so the
packer will not propose a block that touches a PLACED cell — verified in
a six-cycle end-to-end run where `placed_cells` grew strictly monotonically
(572 → 3432) and every PLACED reservation stayed 572/572 cells intact. It
is an invariant gap rather than a today-bug, but it is the invariant that
`reserve`'s whole docstring is about, and `set_block` accepting an
unrestricted write over PLACED is the mechanism. Note also that
`OccupancyGrid.set` / `Cartridge.mark_cell` (`scene.py:313-314`, `181-184`)
take *any* state pair with no guard at all.

### 6. The new reservation / `release_planned` interaction — sound for `main`, fragile by construction **[X]**

Verdict on the specific questions asked:

* **Can a release free cells another reservation still owns?** Yes, one
  shared boundary column, by the mechanism in finding 5 — but only
  between a PLANNED block and a *PLACED* neighbour, which the packer
  cannot produce. Between two PLANNED neighbours the cell is freed and
  the next cycle releases both anyway, so it is self-healing.
* **Can a partially-executed queue leave the twin inconsistent?** Not
  under `main.py:575` (one pose per cycle, then re-plan): verified over
  six cycles, `released_reservation_count` = 45, `placed_cells` monotone,
  targets advancing 19.5 mm each cycle with no repeats. The correctness
  depends on that one-pose discipline, which `cycle()`'s list return type
  does not enforce — a caller that executed the whole queue would exercise
  the finding-5 boundary case.
* **Is `tol=1e-6` right in both directions?** Yes. Verified: exactly
  abutting (dx = 18.5) → no overlap; a 0.5 nm overlap → no overlap; a
  0.1 µm overlap → overlap; a 0.5 nm gap → no overlap. The tolerance is
  permissive in the correct direction (it forgives abutment, which is
  legitimate) and 1 nm is far below any real overlap. No finding.

### 7. Batteries: no double assignment, but the pose's battery ID is meaningless past the cycle **[X]**

`_replace_batteries` (`scene.py:457-465`) wipes and re-IDs every battery
each frame from a monotonic counter. Within a cycle, `_build_pose` does
`available.remove(bat)` **and** `bat.assigned_to_pose = True`
(`planner.py:597-598`) against a single shared list, so one battery cannot
be assigned twice — verified, 4 poses, 4 distinct IDs. Across cycles the
same four physical batteries came back as IDs 4-7 with no overlap against
cycle 1's 0-3, and `assigned_to_pose` resets to False. That is consistent
with the ephemeral contract only because the queue is also rebuilt each
cycle; a queue held across cycles would carry `battery_detection_id`s that
identify nothing. No live defect.

### 8. Coordinate frames — one real mismatch, benign today **[X]**

Conversions checked: `_image_to_workspace` (px → mm, this frame's scale),
`_cell_to_workspace` (cartridge-local mm → workspace mm, the cartridge's
own scale), `_nearest_battery` (mm → px, cartridge's scale),
`_xy_mm_to_cell` (mm → cell, floor + clip), `_reservation_for` (floor near
edge, ceil far edge). The near/far rounding matches
`common.packing._overlaps_forbidden` exactly — floor `int(x/res)`, ceil
`int(np.ceil((x+w)/res))`, both clipped to the mask — so the mask is
tested and marked at the same size. `_place_target` is shared by the
reachability filter and the envelope invariant, so they cannot disagree.

The one mismatch, measured on the live fixture:

```
rect 694x494 px at 0.38 mm/px -> strip 263.720 x 187.720 mm
grid 125 x 175 cells at 1.5 mm ->      262.500 x 187.500 mm
strip is +1.220 mm wider / +0.220 mm taller than the grid
```

The packer may legally place up to `strip_width`, i.e. into 1.22 mm the
grid cannot represent; `_reservation_for` clips `col1` to `occ.cols`
(`planner.py:737-738`, guarded to ≤ 1 cell of slop by the `RuntimeError`
above it). So the last 0.8 cell of such a placement is unmarked. Bounded
at under one cell and it applies equally to the forbidden test, so it
cannot open a gap larger than the quantisation — noted, not ranked.

---

## Would any existing test have caught the top finding?

**No — the top finding is asserted as correct behaviour.**
`tests/test_scene.py:100-106`
(`test_update_drops_disappeared_cartridges`) asserts `env.cartridges == {}`
after a missing frame. There is no test anywhere that brings a cartridge
*back*. `tests/test_scene.py` contains no test of `reserve`, `confirm`,
`release_planned` or `Reservation` at all; the reservation tests live in
`tests/test_planner.py:826-866` and are all single-frame. Every one of
`test_planner.py`'s 34 tests builds a fresh planner and calls `cycle()` at
most twice on the *same* snapshot. `tests/test_main_integration.py:132-141`
documents the drop explicitly and *works around* it by using a
single-scene fixture so the counter it is asserting on stays meaningful.
Finding 2 (frozen rectangle) has no test either — no test ever moves a
cartridge between frames. Findings 3, 4 and 5 likewise have no coverage.

## Suggested test bar for any fix (not implemented — audit is read-only)

1. Three frames: present / absent / present, identical box. Assert the
   returning cartridge retains its PLACED cells, or that the queue does
   not contain a target within one battery footprint of a PLACED one.
2. Two frames with the cartridge translated 100 px at constant scale.
   Assert the place target moves by `100 * mm_per_px`.
3. One frame with two cartridge detections over one object. Assert the
   twin holds one entry, or that no two queued targets overlap in mm.
4. `reserve` a block abutting a PLACED block; assert it raises, or assert
   `placed_count()` is unchanged after `release_planned()`.
