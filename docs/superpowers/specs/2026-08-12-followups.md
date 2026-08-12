# Clearing the recorded follow-ups

Written 2026-08-12 against `51f3590`. Four items, each found by an audit or a fix
pass and recorded rather than fixed at the time. Two were real defects, one was
half real, and one was already gone by the time it was opened.

Suite: **817 passing** (814 at `51f3590`, plus three added here), 0 failures,
excluding `tests/test_synth3d_world.py` and `tests/test_seeding.py` — new,
untracked and mid-edit by another agent in this tree, 13 failing there and none
of it reachable from anything below.

---

## 1. `plan/planner.py` put an approach height in the pick Z — real, fixed

**The mock's warning is right.** Evidence, in the order it decides the question:

`execution/krl_prog/routines.src`, `PickAndPlace`, step 1:

```krl
   ; 1. Approach — 60 mm above the target
   approach_pos = pick_pos
   approach_pos.Z = pick_pos.Z + 60
   PTP approach_pos
   LIN pick_pos
   ; 2. Grasp
   VacuumControl(1, 1, TRUE, vacuum_pct)
   IF $IN[10] == FALSE THEN
      RETURN 2                 ; PICK_FAILED
```

The controller takes the commanded pick pose as the point to **close the gripper
at** and derives its own approach from it. The wire Z is therefore a **grasp**
height by definition, and `_build_pose` was sending `pick_approach_height_mm` =
60 mm. On real hardware that approaches at +120 mm, descends to 60 mm — clear of
an 18.5 mm cell lying on the table — closes the vacuum on air, reads `$IN[10]`
false and returns `PICK_FAILED`. Every cycle. `mock_kuka_server` cannot fail that
grasp because it models no parts, which is exactly why it warns instead.

**The deleted keys do not change the answer, and the check was worth doing.**
`ExecutionConfig.insert_height_mm` / `approach_height_mm` / `grasp_height_mm` and
their `configs/execution.yaml` entries were deleted because *no `KukaClient`
method read them* — the 16-byte frame has one Z field and no place for a second.
That is an argument about what the **execution config** can carry. It says
nothing about what the one Z field that does exist should contain, and the answer
to that is fixed by `routines.src` above.

**The fix.** `PlannerConfig.pick_approach_height_mm` → `pick_grasp_height_mm`,
default **5.0** — the value `configs/execution.yaml` declared as
`motion.grasp_height_mm` before deletion, and the number `mock_kuka_server`
hardcoded. There is deliberately no approach height in `PlannerConfig` any more:
the approach is controller-side.

`motion.approach_height_mm` is now **refused by name** rather than ignored.
Accepting and dropping it would leave a key that looks like it still sets the
pick height and does not — the precise defect the five deleted `execution.yaml`
motion keys were, re-created in a new file.

`place.z_mm` is unchanged and still not transmitted. That is deliberate and
documented at `KukaClient.pick_and_place`; a comment at the construction site now
says so too, since that is where a reader meets the asymmetry.

**A test was asserting the defect — the eighth instance in this project.**
`test_cycle_produces_poses` did

```python
assert p.pick.z_mm == planner.cfg.pick_approach_height_mm
```

which compares the pose against the same field the pose was built from. It passes
for any value whatsoever. Replaced with a cross-module assertion against
`mock_kuka_server._GRASP_BAND_MM` itself, imported rather than copied, so the
planner and the simulator's band cannot drift apart.

**Not fixed, not mine:** `configs/execution.yaml`'s deletion note says "The
planner's own approach/insert heights live in planning.yaml's `motion:` block,
which `plan.PlannerConfig.from_dict` reads." `configs/planning.yaml` has no
`motion:` block (audit E §B4), so the planner has always used its in-code
defaults, and the sentence points a reader at a block that does not exist. Now
also stale in its key name.

## 2. `released_reservation_count` was not surfaced — real, fixed

Now `stats["released_reservations"]`, and a line in the generated receipt,
labelled as expected non-zero so it is not read as an error count.

**What testing it surfaced.** The obvious test — run `demo_config`, assert the
count is non-zero — fails, and not because the release is broken.
`EnvironmentModel.update_from_snapshot` matches cartridges across frames by IoU
and **drops any it does not see again**. `demo_config` cycles three *unrelated*
synthetic scenes, so each frame's cartridge is discarded whole and its stale
reservations go with it, uncounted: `released_reservations` reads 0 there while
nothing is leaking. The counter describes a **persisting** cartridge, which is
the fixed-mount-camera case it exists for. The test therefore uses a new
one-scene fixture, where the frame repeats, the cartridge survives, and the
release fires as designed.

Worth knowing for anyone reading a run's output: on a scene-cycling corpus this
statistic is structurally 0 and proves nothing. On a real cell it should be
roughly `queue_poses - cycles`.

## 3. Three wrong source comments — one dissolved, two fixed

### 3a. "Two disjoint crop populations" — **already correct, nothing to fix**

Both files were checked line by line at `51f3590`. Neither asserts disjointness;
both already state the opposite and compute the number:

* `recog/seg_dataset.py:26` — *"This is no longer the disjoint case the paragraph
  above describes for the pre-fix generator"*, followed by the 210-of-841 /
  176-of-502 snapshot and an explicit instruction not to treat it as current
  fact.
* `recog/seg_evaluate.py:395` — *"this used to be assumed disjoint in prose; it no
  longer is … so `format_report` reads the real count instead of restating an
  assumption"*, and `evaluate()` recomputes the four-way contingency every run.
* `recog/seg_evaluate.py:775` — emits `DISJOINT`/`OVERLAPPING` from the measured
  `overlap["both"]`, never from a constant.

The finding was recorded against an earlier state; `4e3c03e` ("compute the
cartridge/bay overlap instead of asserting it") fixed the source as well as the
receipt text. **Left alone.**

### 3b. Dead commit SHAs — fixed, and there were more than two

`docs/superpowers/specs/2026-08-12-sha-remap.md` repaired 141 documentation
citations and listed 17 in code its brief forbade it to touch. A `git cat-file -e
<sha>^{commit}` sweep of every 7-hex-character token in the tree found the same
set still dangling. Remapped from that document's own table:

| File | was | now |
| --- | --- | --- |
| `recog/calibrate_tau.py:518` | `138105d` | `75db46a` |
| `recog/calibrate_tau.py:519` | `58dd21d` | `380e7d5` |
| `recog/seg_dataset.py:9, 239` | `27cbd97..9fcf136` | `a31ac28..043e92d` |
| `recog/seg_dataset.py:29` | `43aa607` | `cd86d1f` |
| `plan/placement_area.py:616` | `58dd21d` | `380e7d5` |

**Still dangling, owned elsewhere:** `recog/seg_training.py:522` (`dedf700` →
`0ac6c5b`), `tests/test_synth3d.py:1979` (`ac54743` → `f4596e8`),
`.github/workflows/ci.yml:5` (`5ad9c85` → `3fedcb6`), the 22 citations inside
generated `docs/receipts/seg_eval*.txt` bodies (the generator string is already
fixed — they clear on the next regeneration, not by hand), and the audit/spec
documents written before the remap. `recog/seg_ablation.py:620`, listed in the
remap table, is already correct.

### 3c. `execution/protocol.py`'s CRC description — fixed

The claim was that CRC-16/MODBUS "is the standard integrity check on the KUKA
EthernetKRL XML transport (PPR §7.3, R4)". It is refutable from inside the
repository: `execution/krl_prog/laptop-comm.xml` declares an XML payload schema
with **no checksum element**. The CRC is not part of anything KUKA publishes — it
is part of a binary framing this repository defines, and FDR §7.1 already had the
accurate wording ("modelled on … and **augmented with** a CRC-16/MODBUS
trailer"), now adopted.

The framing and the CRC are correct and unchanged — audit F §1 verified them
against three independent implementations. What the docstring now also states,
having read the surrounding code:

* it covers **bytes 0..13 of one frame**, checked before any field is
  interpreted (`unpack_command` verifies CRC, *then* version, *then* opcode);
* it does **not** delimit or resynchronise the stream. No length prefix, no sync
  word — framing is positional, readers take exactly 16 bytes, and a stream that
  loses a byte fails CRC on every subsequent frame rather than recovering (audit
  F §1.6, observed, not hypothetical);
* it is **integrity, not authentication**. Anyone who can reach the socket can
  emit a correctly-CRC'd command.

`test_crc16_modbus_known_vectors` gains the CRC catalogue's published **check
value** for this variant, `crc16_modbus(b"123456789") == 0x4B37` — the one
constant that distinguishes CRC-16/MODBUS from the other CRC-16s sharing
polynomial 0xA001, so the framing claim is now checkable against a specification
rather than against this repository.

## 4. The two receipts — investigated, not touched

`docs/receipts/` belongs to another agent. Findings only.

### 4a. `pytest-cov.txt` — regenerable in one command, but the 86 % needs a scope decision first

**Measured, not estimated.** `pytest -q --cov` over the current tree, same
`[tool.coverage.run]` config, 29 s, no GPU and no Blender:

| Scope | Stmts | Cover |
| --- | ---: | ---: |
| The 18 modules the 2026-04 receipt actually listed (plus `common/packing.py`, which superseded `plan/bin_packing.py`) | 1 868 | **90 %** |
| What `source = ["recog", "plan", "execution", "common"]` resolves to *today* — 50 modules | 6 858 | **55 %** |

**So: the 86 % claim is still approximately true, and slightly conservative, for
the scope it was measured over.** It is not eroding. The 55 % is entirely a scope
effect — `source` now sweeps in modules that cannot execute in any test
environment: `recog/synth3d/world.py` (543 stmts), `scene.py`, `render.py`,
`assets.py`, `recog/generate3d.py`, `recog/verify3d.py` all sit at **0 %** because
they need Blender's `bpy`, and the torch CLIs land at 18–49 %.

What regenerating actually takes:

1. **A scope decision, before the command.** Run as-is, the receipt prints 55 %
   and Appendix E's O6 row ("Branch coverage ≥ 70 % — Pass, 86 %") becomes a
   **Fail** for a reason that has nothing to do with test quality. Either `omit`
   the `bpy`-gated modules (defensible and honest — they are unreachable without
   Blender), or restate O6 against a named scope. Not a mechanical refresh.
2. **The file is not purely generated.** Its last 9 lines are a hand-appended
   *"Post-rewrite end-to-end smoke test (2026-04-20)"* block that `pytest --cov`
   does not produce, quoting a 10-cycle demo run. A blind regeneration silently
   drops it; re-appending it means re-running `main.py --config
   configs/demo.yaml`, which now reports different statistics.
3. **`main.py` is not in `source` at all** — the top-level orchestrator is
   excluded from every coverage figure the project has ever published.
4. **Right now the suite is not clean**: 13 failures in another agent's
   in-progress `tests/test_synth3d_world.py`. A receipt taken today records them.

Everything else is trivial: `coverage` 7.13.5 and `pytest-cov` are installed and
working, the run is Windows / Python 3.14 against the receipt's Ubuntu / Python
3.10, and the test count would go 102 → 817.

### 4b. `frcnn_latency.txt` — the damage is diagnosed exactly and is losslessly repairable; the *measurement* is not regenerable

**The damage.** The first 64 lines are `(title + "\n" + "=") * 64` where the
intent was `title + "\n" + "=" * 64` — a header multiplied along with its
underline. Verified byte-exactly:

```python
t = "Faster R-CNN inference latency (CPU, 2 threads, 320x512 input)"   # 62 chars
s.startswith((t + "\n" + "=") * 64)     # True
s.count(t) == 64 and s.count("=") == 64 # True
```

**No data was lost.** The bug touched only the header; everything after it is
intact and internally consistent (`frames: 100`, mean 446.0 > median 437.4, p95
484.2 < p99 638.3 < max 689.5, min 406.4 < median — all coherent), including the
§10.4 comparison against the heuristic detector. Restoring the intended two-line
header is a pure deletion of duplicated text that touches no number, which is a
different act from hand-editing a measurement and is the only repair I would
recommend.

**Re-measuring is a different job and should not be confused with repairing the
file.** No producing script is in the tree (FDR Appendix E already lists this
among the sixteen receipts with no surviving tool; `git log --diff-filter=D --
'*.py'` returns nothing, so none was ever committed and deleted). The *inputs*
survive — `recog/checkpoints/best.pt` is present and torch 2.13.0 is installed —
so someone could write a benchmark and produce numbers. They would not be these
numbers: the original was CPU-bound on the 2026-04 Linux box, as was the
`median 3.3 ms / p95 5.5 ms` heuristic baseline it is compared against in the same
file. Re-measuring one without the other makes the comparison meaningless, and
re-measuring both on this Windows machine produces a new result, not a
regenerated receipt. Out of scope by the brief's "do not regenerate anything",
and flagged as a judgement call for whoever owns the directory.

---

## Concerns

* **`configs/planning.yaml` has no `motion:` block**, so `pick_grasp_height_mm`
  and `place_insert_height_mm` are permanently their in-code defaults (5.0 / 2.0)
  and `configs/execution.yaml`'s note claims otherwise. Audit E §B4's finding, one
  key name staler now. `configs/` is another agent's.
* **The grasp height 5.0 mm is inherited, not measured.** It is what the project
  declared and what the simulator hardcodes, so the fix is faithful to the
  project's own number — but nothing in the tree derives it from the tool
  geometry or the 18.5 mm cell diameter, and a real cell would need it measured.
  The test pins it inside the simulator's 25 mm band, which bounds the error
  rather than confirming the value.
* **Three tests now depend on `execution.mock_kuka_server` from `plan`/`main`
  test modules.** That is deliberate — it is what stops the planner and the
  simulator's grasp band drifting apart — but it is a new direction of coupling
  in the test suite.
