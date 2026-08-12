# The workspace envelope at the right altitude, and the demo that runs again

**Date** 2026-08-12 · **Fixes** the abort recorded in
`docs/superpowers/specs/2026-08-12-seeded-training.md` §1, introduced by
`2026-08-12-fix-planner-occupancy.md` §5 (audit E finding 5).
**Files** `plan/planner.py`, `main.py`, `tests/test_planner.py`,
`tests/test_main_integration.py`, `docs/receipts/main_seg_run.txt`,
README, FDR v3 §8 / §13.2.1, NEXT_STEPS.

---

## 1. The defect, reproduced

```
$ python main.py --config configs/demo.yaml
plan.scene.OutOfWorkspace: place target for cartridge 0 at (358.9, 145.0) mm
is outside the robot workspace x [-350.0, 350.0] y [-350.0, 350.0] mm
```

`configs/demo_seg.yaml` aborts the same way at cycle 7. Both were the
`OutOfWorkspace` raise in `Planner._build_pose`, firing on the first
frame whose field of view reached past 350 mm.

**The raise was right and stays.** Clamping a place target inserts a
cell into a wall and records it PLACED; clamping a pick point grasps
empty table. Nothing was reverted and nothing became a clamp.

**What it found is real too**: `origin_offset_{x,y}_mm = 0` puts the
whole field of view in +x/+y, the envelope is ±350 mm, and the frames
span 486 × 274 mm (`demo.yaml`, 1280 × 720 at 0.38 mm/px) to
1338 × 752 mm (`demo_seg.yaml`, at the sidecar's own 0.490–1.045 mm/px).

## 2. The diagnosis: two conditions, one check

The check was at the wrong **altitude**, not in the wrong place. Two
different things were being answered by one exception:

| | what it is | right response |
|---|---|---|
| a cartridge slot or a loose cell **lying** outside the envelope | a **scene condition** — a fixed-mount camera images more table than a 706 mm-reach arm can serve | decline to serve it, **count it**, carry on |
| a **commanded pose** outside the envelope | a **planning bug** — `mm_per_px`, `origin_offset_*` or the bounds are wrong | refuse to command the arm. Fatal |

The first was being treated as the second, so an ordinary frame killed
the run.

### 2.1 Why not just centre `origin_offset`

Centring is arithmetically sufficient for `demo.yaml` (486 mm centred
spans ±243, inside ±350) and **cannot** work for `demo_seg.yaml`: 1338 mm
centred spans ±669 against a ±350 envelope. No offset makes a 1.34 m
field of view fit a 0.7 m envelope. So a filter is required *regardless*
of the offset — which makes changing the offset an independent decision
about what the demo depicts (how big the table is, where the base sits),
exactly the design call `2026-08-12-seeded-training.md` §1 declined to
make on a receipt regeneration. **`configs/planning.yaml` is unchanged.**
With offset 0 the reachable region is px 0–921 of 1280 in x and all 720
rows in y at 0.38 mm/px, i.e. 72 % of `demo.yaml`'s frame width, so the
demo still has plenty to serve and the honest count is reported rather
than engineered.

## 3. What changed

### 3.1 The filter, in `Planner.cycle`

* **Batteries.** `_reachable_batteries` maps each loose cell's centre to
  workspace mm and drops the ones outside the envelope **before**
  `available` is built — so an unreachable cell cannot be assigned by
  `_nearest_battery` and consume a cartridge slot on the way out. This is
  the "explicit, **counted** filter on `available_batteries`" that
  `2026-08-12-fix-planner-occupancy.md` §5 named as the right change if a
  real cell ever imaged beyond the arm's reach. It does.
* **Place targets.** Every packed placement's centre is tested and the
  unreachable ones are skipped.

Reachability is decided **for the whole cartridge before any pose is
built**, deliberately: it is a property of the cartridge's geometry
against the envelope and has nothing to do with how many batteries are
left, so deciding it inside the pose loop would let the
ran-out-of-batteries early return report a cartridge as unreachable when
it was merely unvisited.

### 3.2 The invariant, still in `_build_pose`

Both `WorkspaceBounds.require` calls stay exactly as they were. After
the filter they cannot fire in normal operation — **which is the
point**. `_build_pose` is the only thing in the pipeline that constructs
a `PickPlacePose`, and `main._run_one_cycle` hands `queue[0]` to
`KukaClient.pick_and_place` unmodified, so every coordinate that reaches
the socket came through those two lines.

`Planner._reaches` exists as a **named seam**: the filter calls it, the
invariant does not. They ask the same question of the same envelope and
must agree, but they are separate code paths so a test can disable the
filter and watch the invariant fire. A guard whose only evidence is that
nothing has tripped it is not a proven guard.

`_place_target` was extracted for the same reason in the other
direction: the filter and the invariant compute the target with **one**
function. A filter that computes it one way and a guard that computes it
another is how a pose passes the first and trips the second on a
perfectly ordinary scene.

### 3.3 The counts are reported, not implied

Three counters on `Planner`, copied into `main.run`'s stats, printed in
the run summary, in the generated receipt, and — as a **per-cycle
delta** — in the per-cycle log line, so `queue=0 unreachable=0` (empty
scene) and `queue=0 unreachable=31` (unreachable scene) are
distinguishable while the run is happening:

| counter | counts |
|---|---|
| `unreachable_place_target_count` | packed placements whose centre is outside the envelope |
| `unreachable_battery_count` | loose cells whose pick point is outside it |
| `unreachable_cartridge_count` | cartridges the packer found room in where **not one** slot was reachable |

All three sum over cycles, on the same convention as
`cartridges_detected`: the same physical slot in two frames counts twice.
The cartridge counter is a roll-up of the same skips the place-target
counter sees, so `main._unreachable_total` deliberately does not add it
in.

### 3.4 One new hard failure

A run that queued **zero** poses while declining a non-zero number as
unreachable now raises. That is not an empty scene, it is a scene the
arm cannot reach any part of — `workspace_bounds_mm`, `origin_offset_*`
or `mm_per_px` is wrong — and it is indistinguishable from "no batteries
today" in every other statistic the run prints. Same shape as the
existing "segmentation produced no placement area" raise.

The threshold is **zero and not a ratio**, on purpose: any ratio would be
a number nobody measured, and a partly-reachable scene is a legitimate
cell layout that the counters already describe.

## 4. Proving no out-of-envelope pose reaches the wire

`test_no_coordinate_outside_the_envelope_reaches_the_wire`
(`tests/test_main_integration.py`) runs the full `main.run` loop and
intercepts `KukaClient._send` — the last call before `socket.sendall` —
parsing every packet back with the protocol's own `unpack_command`. What
is asserted is the **int32 millimetres the controller would receive**,
not the float the planner computed. Asking `planner.cycle`'s return
value instead would re-ask the module that does the filtering whether it
filtered.

**The fixture's envelope is a LOWER bound, and that is load-bearing.**
`main` executes `queue[0]` and only `queue[0]`, and row-major fill makes
that the pose with the *smallest* coordinates — so an envelope that cuts
off only the far edge could never put a violation on the wire however
broken the filter was, and the test would pass while asserting nothing.
The first fixture attempt (`x_max = y_max = 180`) did exactly that. It
was caught by falsifying it: with `Planner._reaches` forced `True` **and**
`WorkspaceBounds.require` stubbed out, that fixture still recorded
**0** violations. Cutting off the near edge instead (`x_min = y_min =
80`) records **4** — `MOVE_TO(25, 48)`, `PICK_AND_PLACE(42, 152)`,
`MOVE_TO(31, 112)`, `PICK_AND_PLACE(156, 32)` — and the test fails as it
should.

The invariant is separately pinned by
`test_the_envelope_guard_fires_when_the_reachability_filter_is_bypassed`,
which disables `_reaches` only and requires `OutOfWorkspace`. If someone
deletes those `require` calls because "nothing reaches them", that test
catches the deletion.

**Not added: a second envelope check inside `KukaClient`.** It would need
the envelope either configured twice in `configs/` — the drift that
`ExecutionConfig`'s deleted `approach_height_mm` / `insert_height_mm`
keys were — or as an optional constructor argument that no-ops in 22 of
its 23 construction sites, which is the failure mode this project keeps
closing. The property is instead *proved* end to end at the socket, and
the single producer keeps the single check. Noted in §7 as a deliberate
non-change.

## 5. Tests corrected and added

**Corrected (2, both asserting the old behaviour).**

* `test_a_place_target_outside_the_workspace_raises` →
  `..._is_skipped_and_counted`. It asserted that an out-of-envelope
  *candidate* aborts the cycle, which is the conflation being undone. It
  now asserts the queue is empty, `unreachable_place_target_count > 0`
  and `unreachable_cartridge_count == 1` — the counter being what makes
  "cannot reach it" distinguishable from "the cartridge is full", since
  both are the same empty queue and want opposite responses.
* `test_a_pick_point_outside_the_workspace_raises` →
  `..._is_skipped_and_counted`. Now also asserts
  `unreachable_cartridge_count == 0`: the cartridge was reachable and
  only the battery was not, and collapsing the two would be the same
  conflation again.

**Added (6).**

| Test | Proves |
|---|---|
| `test_a_camera_that_sees_further_than_the_arm_reaches_still_plans` | the regression test — part served, part declined, run continues, **every** emitted pose inside the envelope |
| `test_the_envelope_guard_fires_when_the_reachability_filter_is_bypassed` | the invariant is still live with the filter off |
| `test_an_unreachable_slot_does_not_consume_a_battery` | skipping happens before assignment: one battery, low-x slots unreachable, queue length still 1 |
| `test_no_coordinate_outside_the_envelope_reaches_the_wire` | **the safety property**, on the bytes at `_send` |
| `test_run_reports_what_the_arm_declined_to_serve` | the counts reach `stats` *and* the generated receipt |
| `test_a_run_that_can_reach_nothing_is_a_failed_run` | §3.4's raise, on an envelope disjoint from the field of view |

Suite: **1032 passed** (1026 + 6).

## 6. The numbers

### 6.1 `configs/demo.yaml` — runs to completion again

Ten runs, 2026-08-12, after the fix. Every perception and planning
number identical in all ten — including all three new counters — and
only the placed / pick-failed split moves (9 or 10 placed, 0 or 1
pick-failed), for the reason `configs/demo.yaml` documents: the mock's
unseeded 2 % simulated vacuum drop.

| | before the envelope was enforced | now |
|---|---|---|
| cartridges detected | 37 | 37 |
| batteries detected | 77 | 77 |
| placement areas | 33 | 33 |
| **queue poses** | **62** | **41** |
| cycles | 10 | 10 |
| unreachable place targets | – | 14 |
| unreachable batteries | – | 15 |
| unreachable cartridges | – | 2 |

**41, not 62, is the honest number.** The missing 21 poses were pick or
place points beyond ±350 mm that the run commanded anyway, back when the
envelope compared against nothing. Perception is untouched — the same 37
cartridges and 77 batteries are found — and so is the placement-area
count; what changed is only which of them the arm agrees to serve.

### 6.2 `docs/receipts/main_seg_run.txt` — regenerated

From its own header command,
`python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt`,
which had not completed since 2026-08-11.

| | 2026-08-11 receipt | regenerated |
|---|---|---|
| cartridges detected / segmented | 26 / 26 | 26 / 26 |
| placement areas | 7 | 7 |
| bad detector boxes | 1 | 1 |
| frames with own scale | 15 of 15 | 15 of 15 |
| **poses queued** | **6** | **4** |
| **placed** | **2** | **1** |
| empty queue | 13 | 14 |
| unreachable place targets / batteries / cartridges | – | 2 / 57 / 1 |

**57 of 78 loose cells are out of reach**, which is the whole story of
this corpus: the renders span 627 × 353 to 1338 × 752 mm against a
700 × 700 mm envelope. Two of the six poses the old receipt reported
were never reachable, and one of the two pick-and-places was executed to
a point the arm could not have gone to.

Two smaller receipt corrections went in with it, both because a
regenerated receipt should not carry a false parenthetical:

* `reservations released : 0 (… - expected non-zero)` said the opposite
  of the number beside it. It is **0 correctly** here: this corpus cycles
  unrelated scenes, so the twin drops each cartridge whole and its
  reservations go with it, uncounted (the same effect
  `tests/test_main_integration.py::test_run_surfaces_released_reservations`
  documents for `demo_config`). The receipt now says when a zero is
  expected instead of asserting it never is.

### 6.3 Documents updated

`README.md` (both the 10-run demo table and the `demo_seg` paragraph),
`docs/FDR_v3.md` §8 and §13.2.1, `docs/NEXT_STEPS.md`. Each states the
superseded figure and why it moved, rather than replacing it silently.

## 7. Concerns left open

* **No second envelope check at `KukaClient`.** §4. The property is
  proved at the socket and enforced at the single producer; it is not
  enforced *twice*. If a second producer of `PickPlacePose` ever appears,
  this needs revisiting — and the honest place for the check would then
  be `common.types`, next to `WorkspacePoint`, so `execution` can reach
  it without importing `plan`.
* **The mock controller still accepts anything.** Audit F finding 2:
  `MOVE_TO(5000, 5000, 5000)` returns `SUCCESS` on a 706 mm-reach arm,
  and the 16-byte frame carries no envelope. The planner remains the only
  thing in the system that can say no. Unchanged here.
* **`origin_offset_{x,y}_mm = 0` is still a strange cell.** The envelope
  is symmetric about the origin while the camera images only +x/+y, so
  three quarters of the declared envelope is never imaged and can never
  be picked from. That is a description of the cell, not a bug in the
  code, and choosing better numbers is a design decision about the demo —
  §2.1. It is now *visible* rather than fatal, which is the change that
  makes deciding it possible.
* **`unreachable_*` counts sum over cycles**, so on a repeating scene the
  same physical unreachable cell is counted once per frame. Consistent
  with `cartridges_detected` and stated in the receipt, but it means the
  counters are a rate, not an inventory.
