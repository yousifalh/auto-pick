# Seeding training, and the receipt that could not be regenerated

**Date** 2026-08-12 · **At** `51f3590`
**Job 1** regenerate `docs/receipts/main_seg_run.txt` — **blocked, see §1**.
**Job 2** make training reproducible — in the next commit; §2 onward of
this file.
**Files** `recog/seeding.py` (new), `recog/seg_training.py`,
`recog/training.py`, `configs/segmentation*.yaml`, `configs/recognition.yaml`,
`scripts/seed_check.py` (new), `tests/test_seeding.py` (new),
`docs/receipts/seed_reproducibility.txt` (new), README, FDR v3 Appendix C,
NEXT_STEPS, PORTFOLIO, CV_BULLETS.

---

## 1. Job 1: `main_seg_run.txt` cannot currently be regenerated at all

The receipt was to be refreshed because two behaviour-changing fixes landed
after it was last written (`a649f43`, whole-footprint reservation;
`d48dcbc`, latching E-stop and the removal of `place.z_mm`). It was not
refreshed, and it was not hand-edited. **Its own command aborts.**

```
$ python main.py --config configs/demo_seg.yaml --receipt docs/receipts/main_seg_run.txt
...
2026-08-12 16:14:58 [INFO] autopick.main: cycle=7 perc=1651.5ms plan=4.5ms cartridges=4 masks=4 queue=0
plan.scene.OutOfWorkspace: place target for cartridge 14 at (618.9, 399.2) mm
is outside the robot workspace x [-350.0, 350.0] y [-350.0, 350.0] mm
```

`configs/demo.yaml`, the torch-free demo the README's reproducibility claim
rests on, aborts the same way (`place target for cartridge 0 at (358.9,
145.0) mm`). Both were run at `51f3590` with a clean tree, and `demo.yaml`
was re-checked at `ad1796c` (after another agent's `01001b7`, which changed
the pick Z) and aborts identically.

**The cause is understood and is one of the two named fixes**, so this is not
a surprise in the "some third thing moved" sense. `a649f43`'s second half
(audit E finding 5) made `WorkspaceBounds` load-bearing for the first time:
every pose is now checked and an out-of-envelope pose **raises rather than
clamps**, deliberately, because clamping a place target inserts a cell into a
wall and records it PLACED. The check is right. What it has found is that the
demo configuration was never internally consistent:

| quantity | value at `51f3590` |
|---|---|
| `planning.camera.origin_offset_{x,y}_mm` | `0.0`, so pixel (0, 0) maps to the robot origin and the whole field of view lies in +x/+y |
| `planning.camera.workspace_bounds_mm` | ±350 mm about that origin |
| `demo.yaml` frame extent | 1280 × 720 px at 0.38 mm/px = **486 × 274 mm** |
| `demo_seg.yaml` frame extent | 1280 × 720 px at the sidecar's own 0.490–1.045 mm/px = **627 × 353 to 1338 × 752 mm** |

No offset makes the second row fit the fourth: a 0.7 m envelope cannot contain
a 1.34 m field of view, centred or not. So the envelope, the origin offset and
the frames disagree, and until `a649f43` nothing compared them —
`configs/planning.yaml` even labels that block "Replace with real intrinsics
when available".

**Not fixed here, and deliberately.** Choosing new numbers decides what the
demo depicts (how big the table is, where the robot base sits), which is a
design decision about the cell and not a receipt regeneration; and quietly
widening an envelope until the run completes is the exact move this project
refuses everywhere else. The commit's own message shows the author widening
`_scaled_planner`'s test fixture from ±350 to ±500 mm for the same reason —
"a property of the framing rather than of the scale handling those tests
measure" — so the mismatch was seen in the fixture and not carried through to
the shipped configs.

Also relevant to whoever picks this up: the working tree was being edited
concurrently by another agent (`plan/planner.py`, `execution/protocol.py`,
`main.py`, three test files) while this was measured. A receipt regenerated
against a tree in that state would correspond to no commit, which is a second
reason to leave it.

**Consequences, unrecorded elsewhere as of this writing:**

1. `docs/receipts/main_seg_run.txt` (2 picks / 7 areas / 6 poses) describes a
   run that no longer completes. Nothing in the docs quoting it — README, FDR
   §13.1.1 area, NEXT_STEPS — has been changed, because the numbers *were*
   true when measured and the code that would produce new ones is being
   edited right now.
2. The README's 10-run demo table (37 cartridges, 77 batteries, 33 placement
   areas, 62 queue poses) is in the same position: measured 2026-08-12, on a
   command that now aborts partway.

---

## 2. Job 2 — seeded training

Follows in the next commit, which appends the rest of this document.
