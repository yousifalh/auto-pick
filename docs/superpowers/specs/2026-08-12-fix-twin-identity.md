# Cross-frame identity: the twin remembers, bounded and counted

**Date** 2026-08-12 · **Fixes** `docs/superpowers/audit/2026-08-12-H-digital-twin.md`
findings 1, 2, 3, 4 and 5.
**Files** `plan/scene.py`, `plan/planner.py`, `main.py`,
`configs/planning.yaml`, `tests/test_scene.py`, `tests/test_planner.py`,
`tests/test_main_integration.py`.

---

## 1. The defect: one root cause wearing four costumes

The twin is the only memory the system has of which cells in which
cartridge are FREE / PLANNED / PLACED. It was rebuilt from a fresh set of
detections every frame, and identity was decided by per-frame IoU with no
continuity. Audit H executed four ordinary perception events against it
and all four produced a full-length, in-envelope, clean-reporting queue
aimed at physically occupied space:

| # | event | measured before |
|---|---|---|
| 1 | cartridge undetected for ONE frame | 572 placed cells → new empty ID; next place target byte-identical to one already executed; interlock compared against reservations deleted with the cartridge |
| 2 | cartridge moves | moved 38 mm, place target moved **0.00 mm** (rectangle extracted once, ever) |
| 3 | one cartridge detected twice | two twin entries, 8 pairs of overlapping place targets, no `PlacementCollision` (the mm interlock is per-entry) |
| 4 | two detections competing for one track | a detection at IoU **0.667** took id 0 on dict order; the IoU **1.000** detection became a new empty id |

The common cause is one sentence: **identity was tracked by per-frame IoU
with no continuity, and losing identity silently discarded physical
state.** Every symptom above is indistinguishable from success in
`stats`, in the logs and in the robot's status.

## 2. The design, and where it differs from the brief's reading

The brief's reading — *the twin needs track persistence; a cartridge that
stops being detected should become **not currently visible** rather than
cease to exist* — is right, and is what was built. Three things had to be
added around it, because persistence alone re-opens two of the other
findings:

**A track is memory; a detection is evidence.** Retaining a track is not
the same as being willing to act on it. `Planner.cycle` now iterates
`env.visible_cartridges()`, and `_ensure_placement_areas` and
`_drop_areas_measured_at_another_scale` both skip the invisible. Memory
is enough to *refuse* a slot; commanding the arm into a box that is not
in this frame's snapshot is planning against a photograph. Without this
rule, persistence would have turned a dropout into a queue built from a
stale image.

**Memory is bounded, and the bound is the loudest event in the module.**
`tracking.max_missing_frames` (5) is the lifetime; expiry counts the
tracks, the PLACED cells and the placed batteries it forgets, and logs
them at WARNING. Unbounded retention grows one grid per cartridge ever
seen; silent expiry is the original defect with a delay fuse.

**Two entries over one object are prevented AND detected**, because
either alone is insufficient (§4).

### What is stored

`Cartridge.frames_since_seen` (0 = in this frame) and `.visible`.
`Cartridge.invalidate_geometry()` splits the entity in half along the
line that matters: `placeable_rectangle` / `occupancy` / `pcb_mask` /
`mm_per_px` are **measurements of a frame**; `reservations` are
**batteries**. A cartridge that moves loses the first and keeps the
second.

## 3. Matching: greedy over globally sorted pairs, not Hungarian

Every (track, detection) pair is scored once, filtered to
`iou_match_threshold`, sorted by `(-iou, track id, detection index)`, and
consumed greedily.

Not a Hungarian assignment, deliberately: Hungarian maximises the **sum**
of accepted IoUs and will trade the single best pair away to do it, which
is the exact failure being fixed. This order guarantees the
highest-scoring pair in the frame is honoured first, then the next.
Ties break deterministically, so one frame always yields one assignment.

Finding 4's scenario, as a test
(`test_matching_is_globally_best_not_first_come`): a detection tying
against both tracks arrives FIRST and an exact-match detection arrives
second. Before: the tie took id 0 on dict order and the IoU-1.000
detection was inserted as a new empty entry. After: id 0 goes to the
exact match, the ambiguous detection takes the other track, and no new ID
is minted.

## 4. A moved cartridge: geometry follows, the match is not rejected

`EnvironmentModel._box_shift_mm` measures corner displacement in
millimetres (using the scale the geometry was measured at); past
`tracking.geometry_refresh_mm` (1.5 mm = one occupancy cell, the finest
distinction the grid can represent) the measurement is thrown away and
re-extracted at the new position, and
`Planner._reproject_placed_reservations` re-quantises every PLACED
footprint onto the fresh grid.

**Why re-measure rather than reject the match.** The twin cannot
distinguish "the same cartridge slid" from "a different cartridge was
swapped into the fixture", so the choice is which error to make:

* re-measure, keep the batteries — if it slid, the record stays true; if
  it was swapped, the twin reads a full cartridge, places nothing and
  stalls. **Over-conservative.**
* reject the match, fresh empty grid — if it was swapped, correct; if it
  slid, the planner packs straight into batteries the cartridge already
  holds. **A collision.**

The first is the only one whose failure mode is idleness. It is also
visible: `geometry_refreshes` and `reprojected_placed_batteries` are
counted, and a footprint the new rectangle cannot hold is **kept** (it
still answers the mm interlock) and counted as
`unrepresentable_placed_batteries`.

## 5. Two entries for one object: suppressed, then interlocked

*Prevention.* A detection that matched nothing but overlaps a cartridge
**already observed this frame** by `tracking.duplicate_iou_threshold`
(0.3) is a second view, not a second cartridge, and does not become a
twin entry. Only cartridges observed this frame are candidates —
suppressing against a stale remembered box would let a dying track
swallow a real new cartridge.

*Detection.* Suppression cannot catch every case (a nested box scores
below 0.3 and is legitimately a second entry), so `Reservation` now
carries `wx_mm` / `wy_mm`: the same footprint in the **workspace** frame,
which is the only frame in which two cartridges are comparable —
`x_mm`/`y_mm` are measured from each cartridge's own rectangle.
`EnvironmentModel.conflicting_reservation` asks the cross-entry question
there. It is the same filter/invariant pairing as the workspace envelope:
`Planner._conflicts_with_another_cartridge` declines the slot and counts
it (a split cartridge box is a perception artefact, not a reason to abort
a run); `_build_pose` raises `PlacementCollision` if one ever gets that
far, and `test_the_cross_cartridge_guard_fires_when_its_filter_is_bypassed`
disables the filter to prove the guard is still live.

`wx_mm`/`wy_mm` are **required** dataclass fields, not defaulted: a
cross-cartridge check that silently skips a reservation carrying `None`
is not a check. Three test fixtures that built a cartridge with a grid
and no rectangle were corrected rather than accommodated.

## 6. The configuration

`configs/planning.yaml` gains a `tracking:` block —
`iou_match_threshold`, `max_missing_frames`, `duplicate_iou_threshold`,
`geometry_refresh_mm`. `TrackingConfig` validates the relationships
(a duplicate floor above the match threshold would suppress detections
that should have matched) and refuses unknown keys by name. The 0.5
threshold was previously a function default that no caller and no config
ever set.

## 7. Finding 5, guarded

`reserve` now writes `only_from=CellState.FREE`, so PLACED → PLANNED is
unreachable and the PLACED → FREE that followed it goes with it. The
post-condition became "no cell of this footprint is still FREE" rather
than "every cell is PLANNED", because under `only_from` an abutting
neighbour's boundary cell legitimately stays PLACED — the mm interlock
has already proved the two do not overlap physically. `confirm` also
refuses a second report on an anchor already PLACED, which used to delete
the reservation while leaving its cells PLACED. Both were latent; both
now have tests
(`test_a_reservation_abutting_a_placed_battery_cannot_unplace_it`,
`test_confirm_success_then_failure_on_one_anchor_is_refused`).

## 8. What is counted

Nothing is discarded silently. All of these reach `stats` and the
receipt: `track_dropouts`, `tracks_reacquired`, `tracks_expired`,
`expired_placed_cells`, `expired_placed_batteries`,
`duplicate_detections`, `geometry_refreshes`,
`reprojected_placed_batteries`, `unrepresentable_placed_batteries`,
`rescale_dropped_placed_batteries`, `cross_cartridge_conflicts`,
`untracked_confirmations`.

The last one is finding 1's tail: an executor result whose cartridge is
no longer tracked used to be a bare `return` while `main` counted
`stats["placed"] += 1`. It is now counted and logged at ERROR — the
battery really was put down, and the twin has no record of where.

## 9. Tests corrected

| test | was | now |
|---|---|---|
| `test_scene.py::test_update_drops_disappeared_cartridges` | asserted `env.cartridges == {}` after one missing frame — the defect's own regression test | renamed `test_update_keeps_an_undetected_cartridge_as_not_visible`: retained, `frames_since_seen == 1`, placed cells intact, `detection_index == -1`, dropout counted |
| `test_main_integration.py::test_run_surfaces_released_reservations` | needed a single-scene fixture *because* the twin deleted cartridges on unrelated scenes and their reservations went uncounted | runs on the ordinary 3-scene `demo_config`; the workaround and its explanation are gone |
| `test_planner.py` ×3 (`_bare_cartridge`) | built a cartridge with an occupancy grid and no rectangle | build both, which is the only state the planner produces; `_reservation_for` now refuses the other |

`tests/test_scene.py` had **zero** coverage of `reserve`, `confirm` and
`release_planned`; it now has ten tests of the state machine in cells,
with no packer in the way, plus the tracking tests above.

## 10. Verification

Suite: **1128 passed**, 0 failed (1074 at HEAD `39429a4`).

The audit's own three-frame scenario, executed through the real planner:

```
before:  frame1 id=0 placed_cells=572
         frame2 (undetected) cartridges=[]  queue=0
         frame3 ids=[1] placed_cells=0  first target (29.4, 52.6) mm == frame1's
after:   frame1 id=0 placed_cells=572
         frame2 (undetected) tracked=[0] visible=0 queue=0  placed_cells=572
         frame3 ids=[0] placed_cells=572  first target (48.9, 52.6) mm
         every queued target >= 18.5 mm from the executed one
```

A cartridge translated 40 px at 0.38 mm/px: place target moves
**15.20 mm** against an expected 15.20 (it moved 0.00 before), and the
battery already in it survives the re-measurement.

### The two demos

```
demo.yaml      cycles 10  placed 10  queue_poses 41  cartridges 37  batteries 77
  before  placement_areas 33  released_reservations  2  unreachable 14/15/2
  after   placement_areas 29  released_reservations 29  unreachable 17/15/3
          track_dropouts 29  reacquired 5  expired 11  expired_placed_cells 1716
          geometry_refreshes 5  reprojected 3  cross_cartridge_conflicts 2

demo_seg.yaml  cycles 1  placed 1  empty_queue 14  placement_areas 7  queue_poses 4
  before  released_reservations 0
  after   released_reservations 3
          track_dropouts 25  reacquired 0  expired 16  expired_placed_cells 572
```

Read the deltas as follows. `released_reservations` 2 → 29 is not new
leakage: those releases were happening before, inside a `del` that took
the whole cartridge and counted nothing. `placement_areas` 33 → 29
because a re-acquired cartridge keeps its ID instead of being counted as
a new one. `expired_placed_cells` 1716 (3 batteries) is the honest cost
of a synthetic corpus that cycles unrelated scenes: cartridges really do
leave, and the twin now says so five frames later instead of forgetting
them instantly and silently. `cross_cartridge_conflicts` 2 is two
placements declined because a second twin entry held that space — before,
both were queued.
