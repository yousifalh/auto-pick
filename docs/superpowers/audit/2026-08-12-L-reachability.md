# Audit L — reachability and contract enforcement

**Scope.** Read-only audit at HEAD `39429a4`, branch `feat/blender-synth-dataset`, 2026-08-12.
Nothing was staged, committed, edited or deleted.

**Two questions.** (1) How much of this codebase actually runs from a production entry
point, as opposed to running only under pytest or not at all? (2) Are the boundary
dataclass contracts in `common/types.py` — which the FDR describes as "frozen dataclass
contracts at every boundary" — actually frozen and actually enforced?

**The headline, up front.** Both answers are better than the brief anticipated.
**98.0 % of the non-test Python is reachable from a production entry point**, and the
single largest dead artefact is a deliberate Blender CI gate, not rot. The contracts are
frozen where they claim to be. But **frozen is the only thing they are**: there is not one
`__post_init__` in `common/types.py`, so every boundary type accepts values that are
physically impossible, and the one field that does validate (`mm_per_px`) validates in a
different file. And one unit boundary silently discards up to 0.999 mm from every
coordinate the robot is ever commanded to.

---

## Method, and what each conclusion rests on

Every finding below is tagged with how it was reached. The distinction matters because
the brief asked for it, and because two of the three methods disagree with each other in
instructive ways.

| Tag | Method |
|---|---|
| `[IMPORT]` | AST import graph over all 54 non-test `.py` files; relative and absolute imports resolved to files; transitive closure from each entry point. |
| `[CALLGRAPH]` | Module-aware call graph over 598 top-level functions, classes and methods. Names resolved through each file's own top-level defs and its own import table — *not* by global name matching. |
| `[GREP]` | Targeted `ripgrep` over source, configs, `.src` and YAML. |
| `[EXEC]` | Executed against HEAD in a Python REPL, output quoted. |
| `[READ]` | Read from source and reasoned about; no execution. |
| `[COV]` | Read from the `.coverage` database already present at HEAD (1074 passing tests). Not re-run. |

**Direction of error in the call graph.** Method resolution is deliberately
over-approximated: any method whose bare name is called anywhere counts as reachable
everywhere. That makes the call graph *generous* toward liveness. Consequently **every
item reported dead below is definitely dead; the true dead set may be slightly larger.**
This was chosen on purpose — a reachability audit that produces false accusations of
death is worse than useless, because someone acts on it.

**Entry points used.** Three, as the brief specifies:

* **`main.py`** — the demo loop, `python main.py --config configs/demo.yaml`.
* **The CLIs** — 16 modules with an `if __name__ == "__main__"` block that a human or CI
  actually invokes: `recog/{training,seg_training,evaluate,seg_evaluate,eval_real,
  seg_ablation,calibrate_tau,synth_dataset,labelme_to_seg,check_annotations,convert_cad,
  sync_config,verify3d}.py`, `execution/mock_kuka_server.py`,
  `scripts/{seed_check,forbidden_bench}.py`.
* **`recog/generate3d.py`** — the Blender generator, `blender -b --python`.

**One correction to the brief's figures.** The brief says 20,808 lines across 51 modules.
Measured at HEAD `[EXEC]`: **22,158 lines across 54 non-test `.py` files** (54 files = 49
modules + 5 package `__init__.py`). The 1,350-line gap is `scripts/` (614) and the
`__init__` shims (73) plus drift since the figure was taken. All percentages below use
22,158 as the denominator, so they are directly checkable.

---

# Part 1 — how much of it actually runs

## 1.1 Modules by reachability class

`[IMPORT]` for the class assignment, `[EXEC]` for the line counts.

| Reachability class | Modules | LOC | % |
|---|---:|---:|---:|
| `main.py` only | 4 | 2,135 | 9.6 % |
| `main.py` + CLIs | 12 | 3,458 | 15.6 % |
| `main.py` + CLIs + Blender | 1 | 624 | 2.8 % |
| CLIs only | 19 | 8,685 | 39.2 % |
| CLIs + Blender | 4 | 2,572 | 11.6 % |
| Blender only | 9 | 4,413 | 19.9 % |
| Package `__init__` shims (run as import side-effect) | 4 | 65 | 0.3 % |
| **Reachable from some production entry** | **53** | **21,952** | **99.1 %** |
| **Orphaned module** | **1** | **206** | **0.9 %** |

The one orphaned module is `recog/synth3d/_gate_orientation.py`. Every other file in the
repository is imported, transitively, by `main.py`, a CLI, or the Blender generator.

Which modules sit in which class:

* **`main.py` only** — `main.py` (736), `plan/planner.py` (771), `execution/execution.py`
  (513), `plan/bin_packing.py` (115).
* **`main.py` + CLIs** — `plan/placement_area.py` (797), `plan/scene.py` (551),
  `execution/mock_kuka_server.py` (450), `recog/inference.py` (384),
  `common/types.py` (276), `execution/protocol.py` (251), `plan/arbitration.py` (197),
  `recog/calibration.py` (189), `recog/bay_segmenter.py` (140), `recog/model.py` (127),
  `common/config.py` (61), `common/logging.py` (35).
* **All three** — `common/packing.py` (624). It is the only module the demo loop, the
  training CLIs and the Blender generator all reach, which is exactly why
  `first_fit_decreasing` living there rather than in `plan/` is the right call: it avoids a
  back-edge from `recog` into `plan`.
* **Blender only** — `recog/synth3d/{world,assets,scene,render,materials,layout,lightrig,
  __init__}.py` and `recog/generate3d.py`.

## 1.2 The plain figure

`[CALLGRAPH]`, `[EXEC]`.

Of **598** top-level functions, classes and methods defined outside `tests/`, **574 (96.0 %)
are reachable from a production entry point.** 24 are not.

Converting that to lines, and not double-counting the dead functions that live inside the
one dead module:

| | LOC | % of 22,158 |
|---|---:|---:|
| Reachable from `main.py`, a CLI, or Blender | **21,708** | **98.0 %** |
| Dead: reachable only from `tests/` | 112 | 0.5 % |
| Dead: reachable from nothing | 338 | 1.5 % |
| **Total dead** | **450** | **2.0 %** |

**98.0 % of this codebase runs in production.** That is a good result and it should be
stated as one. The 20,808-line figure a reader might discount as portfolio padding is,
to within 2 %, load-bearing code. There is no stratum of abandoned subsystems here.

## 1.3 Why coverage would have told you the opposite

`[COV]` against `[CALLGRAPH]`. This is the part of the brief that matters most, because
the two methods do not merely differ in precision — **they invert each other.**

Overall coverage at HEAD is **60 %**. Reachability is **98 %**. The 38-point gap is not
untested-because-dead; it is untested-because-untestable-under-pytest, and it runs in
production every time someone generates a dataset.

**Live in production, 0 % covered** `[COV]` + `[IMPORT]`:

| Module | Stmts | Cover | Runs from |
|---|---:|---:|---|
| `recog/generate3d.py` | 248 | 0 % | Blender entry point |
| `recog/synth3d/render.py` | 210 | 0 % | Blender |
| `recog/synth3d/scene.py` | 199 | 0 % | Blender |
| `recog/verify3d.py` | 122 | 0 % | CLI |
| `recog/sync_config.py` | 22 | 0 % | CLI |
| | **801** | | |

Plus `recog/synth3d/bay.py` at 22 % (284 stmts) and `assets.py` at 24 % (254 stmts) — both
squarely in the Blender path. **801 statements are simultaneously production-critical and
completely unexercised**, because they need `bpy` and the torch-free test environment does
not have it. Coverage reports these as the worst code in the repository. Reachability
reports them as among the most load-bearing. Reachability is right.

**Covered, but dead in production** — the inverse, and the specific distinction the brief
asked for:

| Item | Its module's coverage | Reachable from |
|---|---:|---|
| `recog/dataset.py::RealPhotoDataset` (75 LOC) | 88 % | `tests/` only |
| `recog/evaluate.py::centroid_error_px`, `edge_error_px` (11 LOC) | 98 % | `tests/` only |
| `recog/seg_evaluate.py::per_class_iou` (14 LOC) | 49 % | `tests/` only |
| `recog/augmentation.py::apply` (8 LOC) | 68 % | `tests/` only |
| `recog/synth3d/bay.py::occludes_bay_proxy` (9 LOC) | 22 % | `tests/` only |
| `recog/synth3d/config.py::config_to_dict` (3 LOC) | 96 % | `tests/` only |

`recog/evaluate.py` reports **98 % coverage** while containing two public functions that
nothing in production calls. A reader auditing by coverage would rate that file among the
best-verified in the repository and would never learn that a fifth of its public surface
is exercised exclusively by its own unit tests. **That is the whole point: coverage
measures test-driven execution, and test-driven execution is not production liveness.**

## 1.4 The dead code, classified

Six items reach production from nothing at all. Each is classified per the brief's
taxonomy, with a verdict on deletion. **Nothing was deleted.**

### (a) `recog/synth3d/_gate_orientation.py` — 206 LOC — **deliberate alternative path**

`[IMPORT]` no module imports it. `[GREP]` `.github/workflows/ci.yml` runs `pytest -q` and
the demo loop, and nothing else.

Its own header says what it is:

> ```
> blender -b --python recog/synth3d/_gate_orientation.py
> ```
> Exits 0 if every check passes, 1 otherwise, so it works as a CI gate.

It exists because `lay_flat` and `place_item` were **complete no-ops for their entire life**
and nothing caught it — the scene still rendered, the parts were just never turned. Check 2
("two `rot_deg` values must give two different bounding boxes") is the assertion that would
have caught it, and the file says it is committed rather than thrown away for exactly that
reason.

**Deleting it would be a clear loss.** It is the only executable guard against a class of
silent failure that has already happened once in this repository. But it is a *CI gate that
no CI runs* — the one thing about it that is genuinely wrong. `[GREP]` confirms zero
references in `.github/`. The honest description is not "orphaned" but "wired to nothing";
the fix is a workflow step on a Blender image, not a deletion. Until then, its 0 % coverage
`[COV]` is accurate and its guarantee is aspirational.

### (b) `plan/bin_packing.py::pack_cartridge` — 55 LOC — **superseded**

`[CALLGRAPH]` confirms it remains dead at HEAD; the prior audit's finding still holds.
`Planner._pack_cartridge` (`plan/planner.py:511`) is the live path and does the same job
correctly.

The two are not equivalent, and the difference is the hazard:

```python
# plan/bin_packing.py:51 — DEAD
def pack_cartridge(cartridge, battery_width_mm, battery_length_mm,
                   allow_rotation=True, mm_per_px=0.38):
```

```python
# plan/planner.py:519 — LIVE
scale = _resolve_scale(ctg.mm_per_px, self.cfg.mm_per_px, f"cartridge {ctg.id}")
```

The live path resolves scale through `_resolve_scale`, which raises `UnknownScale` rather
than guessing and rejects non-positive values `[READ]`. The dead path **defaults to
`0.38`** — `planning.yaml`'s placeholder, the one marked "Replace with real intrinsics when
available", the exact constant whose removal is documented in
`docs/superpowers/specs/2026-08-11-scale-calibration.md` as having under-read 24 of 30
cartridges by 27 % at the median and produced 3 unsafe placements.

**Deleting `pack_cartridge` would be an improvement.** It is a re-armed version of a trap
this project has already been caught by, sitting in `__all__` where it reads as supported
API. Its harmlessness depends entirely on nobody calling it, and that is not a property
anyone maintains on purpose. The rest of `plan/bin_packing.py` — the re-exports — must
stay: `first_fit_decreasing` is deliberately frozen and `plan/planner.py` and `common/packing.py`
both depend on the module's export surface.

### (c) `plan/placement_area.py::attach_placement_area` — 20 LOC — **superseded**

`[CALLGRAPH]` dead. It duplicates, line for line, the assignment block inside
`Planner._ensure_placement_areas` (`plan/planner.py:500-507`) — the four fields
`placeable_rectangle`, `occupancy`, `pcb_mask`, `mm_per_px` set from a `PlacementArea`.

What the dead copy lacks `[READ]`:

* No `try/except` around `extract`, so `UnknownScale`, `BadDetectorBox` and
  `PlacementDisagreement` propagate to a caller that has no reason to expect them — the
  live path catches all three and continues.
* No participation in `_drop_areas_measured_at_another_scale`. A cartridge populated
  through this helper carries a rectangle whose scale-consistency invariant nothing
  maintains.

**Deleting it would be an improvement.** It is exported in `__all__`, it looks like the
supported way to attach a placement area, and it is the *unsafe* way. Two implementations
of one safety-critical assignment, one of them silently the weaker, is precisely the
drift-by-duplication that a frozen-contract project should not carry.

### (d) `recog/synth3d/catalog.py::build_catalog` — 35 LOC — **superseded**

`[CALLGRAPH]` dead. `[GREP]` `recog/convert_cad.py` is the live replacement and imports
`convert_step` / `inspect_glb` from this very module (`convert_cad.py:55`). `catalog.py`'s
own module docstring (line 15) concedes the point: convert_cad "wraps `convert_step` /
`inspect_glb` with the things a bare `build_catalog` lacks".

Those things are not cosmetic `[READ]`: STEP length-unit detection
(`parse_step_length_unit`), an implausible-extent guard (`implausible`), unit suggestion
(`suggest_units`), and incremental merge (`merge_catalog`). `build_catalog` writes
`{"units": "m"}` unconditionally and asserts nothing about the geometry it just imported.

**Deleting it would be an improvement**, for the same reason as `pack_cartridge`: it is a
dead sibling of a live path that added safety checks specifically because the unsafe
version was wrong. The rest of `catalog.py` is live and must stay.

### (e) `common/types.py::iter_labels` — 9 LOC — **genuinely orphaned**

```python
def iter_labels(names: Iterable[str]) -> list[ClassLabel]:
    """Parse a sequence of label strings, silently dropping unknown ones."""
```

`[CALLGRAPH]` nothing calls it, not even a test. It is the only dead code in
`common/types.py`, and it is the only function in that file whose documented behaviour is
to fail silently. In a module whose docstring opens by declaring itself the boundary
between all three subsystems, a helper that turns an unrecognised class label into nothing
at all — no exception, no log — is a defect waiting for its first caller.

**Deleting it would be an improvement.** Nine lines, no callers, and its contract is
"silently wrong on bad input".

### (f) `recog/synth3d/assets.py::object_cell_format` — 5 LOC — **scaffolding**

Its own docstring `[READ]`:

> every object never tagged is a CAD template's own cell, which is always 18650

and the comment above it: "this task changes no CAD behaviour by itself". It is forward
scaffolding for multi-format cells. **Deleting it is neutral** — no safety argument either
way, and it is honest about being a placeholder.

### Correctly frozen, correctly kept

`[GREP]` verifies the brief's premise about `first_fit_decreasing`, and it holds. Both
`recog/synth3d/layout.py:34` and `recog/synth3d/bay.py:924` import it directly, and
`bay.py:901` states why: "Positions come from the SAME FFDH packer `common.packing`
exposes to the real planner, so the synthetic partly-filled bay" matches what the planner
will see. Datasets on disk encode its exact output. **Freezing it is correct, the
docstrings justifying the freeze are accurate, and it should not be touched.**

---

# Part 2 — are the contracts honoured?

## 2.1 Frozen: yes. Validated: no. Not once.

`[EXEC]`, against HEAD:

```
inverted BBox accepted: BBox(xmin=100, ymin=100, xmax=0, ymax=0) width= -100
confidence 17.5 accepted
confidence -3.0 accepted
Detection frozen OK: FrozenInstanceError
object.__setattr__ bypass works: 99.0
replace gives: -1.0
negative grid accepted: -5 -5
negative cycle_time_ms accepted: -42.0
```

`[GREP]` for `__post_init__` across the whole repository returns **two hits, both in
`plan/scene.py`** (`OccupancyGrid`, `WorkspaceBounds`). **`common/types.py` — the file whose
docstring declares itself the only thing that crosses `Recognition → Planning` and
`Planning → Execution` — contains zero validation of any kind.**

So the FDR's claim splits cleanly in two. *Frozen* is true and enforced by the language:
`BBox`, `Detection`, `WorkspacePoint`, `PickPlacePose` and `RobotStatus` all raise
`FrozenInstanceError` on assignment `[EXEC]`. *Contract* is not true. Immutability
guarantees a value will not change; it guarantees nothing about the value being possible.

The bypasses the brief asked about `[EXEC]`:

* `object.__setattr__(d, 'confidence', 99.0)` — works. Unavoidable in Python and not a
  finding on its own.
* `dataclasses.replace(d, confidence=-1.0)` — works, returns `-1.0`. Normally this is the
  dangerous one, because `replace` skips nothing that `__init__` does but *is* often used
  to route around validation. **Here it is moot: there is no validation to skip.** That
  is the finding.
* `[GREP]` for `object.__setattr__`, `dataclasses.replace` and `__dict__[` across all
  non-test source: **zero hits.** No production code attempts a bypass. The contracts are
  not being circumvented — they simply do not exist.

`Snapshot` is deliberately not frozen, and says so:

> ``Snapshot`` is deliberately *not* frozen: the recogniser may append to a working
> snapshot during inference. The planner consumes it as read-only; nothing downstream
> mutates it.

`[GREP]` verifies the first half and **falsifies the second**. Three mutation sites exist
outside tests: `recog/inference.py:328` (`snapshot.cartridge_masks[i] = mask`, the
recogniser, permitted), `main.py:523` (`snap.mm_per_px = frame_mm_per_px`, the loop, after
recognition), and `plan/planner.py:455` — **a read of `snapshot.cartridge_masks`, in the
planner.** The read is benign, but "nothing downstream mutates it" is a claim about a
mutable object protected by convention and a comment, verified by nobody. `[EXEC]`
`Snapshot()` accepts `mm_per_px = -1.0`, `image_shape = (0, 0)` and a plain string appended
to `detections`; `to_dict()` then dies with `AttributeError: 'str' object has no attribute
'to_dict'` — at serialisation time, far from the mistake.

## 2.2 Contract violations, ranked by consequence

### 1 — Sub-millimetre truncation on every commanded coordinate. `[EXEC]` `[READ]`

`WorkspacePoint` declares `x_mm: float`, `y_mm: float`, `z_mm: float`. The wire protocol
carries signed integers. The conversion, at `execution/execution.py:306, 337-345`:

```python
int(target.x_mm), int(target.y_mm), int(target.z_mm)
...
int(pose.place.x_mm), int(pose.place.y_mm), int(self.cfg.transport_height_mm),
...
int(pose.pick.x_mm), int(pose.pick.y_mm), int(pose.pick.z_mm),
```

`int()`, not `round()`. Measured `[EXEC]`:

```
    10.9 mm -> int() =    10   (loss 0.9 mm, toward origin)
   -10.9 mm -> int() =   -10   (loss 0.9 mm, toward origin)
     0.6 mm -> int() =     0   (loss 0.6 mm, toward origin)
   349.9 mm -> int() =   349   (loss 0.9 mm, toward origin)
```

**Every pick point, every place point and every transport pose is displaced by up to
0.999 mm, always toward the workspace origin.** Python's `int()` truncates toward zero, so
the bias is not a wash — it **reverses sign at x = 0**, meaning two cartridges on opposite
sides of the origin are pulled toward each other. `round()` would have been unbiased and
capped the error at 0.5 mm.

Why this is ranked first. The shipping wall inset is 4.25 mm; a 0.999 mm bias is 23 % of it.
The README documents two residual unsafe placements at **8.3 % and 5.2 % of a footprint** —
on an 18.5 mm cell that is ≈1.5 mm and ≈1.0 mm. **The truncation error is the same order of
magnitude as the placement errors this project already treats as its headline defect**, and
it is introduced *after* every check that could have caught it. `[READ]`
`WorkspaceBounds.require` (`plan/scene.py:400+`) validates the float pose; the truncated
integer is what actually goes on the wire. Here that is safe only by luck — the configured
envelope is ±350 mm, symmetric about zero, and truncation moves toward zero. **With an
asymmetric envelope not containing the origin, truncation could move a pose that passed
`require` outside the bounds it was checked against.**

The round trip loses the same precision coming back: `execution/execution.py:508` builds
`WorkspacePoint(s["x_mm"], s["y_mm"], s["z_mm"])` from wire integers into float-typed
fields, so a reported pose is silently integral and nothing distinguishes "the robot is at
exactly 10 mm" from "the robot is at 10.9 mm and we truncated".

This is a unit-preserving, precision-destroying boundary crossing — the name promises
millimetres and delivers millimetres, but the type promises float and delivers int. The
type system does not catch it because Python's `int` and `float` both satisfy a `float`
annotation at runtime.

### 2 — `BBox` states an ordering invariant and enforces nothing. `[EXEC]`

> ``xmin``/``ymin`` are inclusive and ``xmax``/``ymax`` are exclusive.

`[EXEC]` `BBox(100, 100, 0, 0)` constructs cleanly with `width = -100`. `area` returns
`0.0` because it clamps with `max(0.0, ...)`, and `iou` returns `0.0` — so an inverted box
**launders itself into a plausible zero-area value** and propagates. This is the
foundational geometric type: `Detection`, `Cartridge`, `Battery` and
`PlacementArea.rectangle` all carry one. A four-line `__post_init__` asserting
`xmin <= xmax and ymin <= ymax` would close it, and would have to be checked against
`plan/placement_area.py`'s degenerate-rectangle handling before being added.

### 3 — `Detection.confidence` is unbounded. `[EXEC]`

`17.5` and `-3.0` both construct. Confidence feeds score thresholds and NMS ordering
throughout `recog/inference.py`; a value outside `[0, 1]` sorts first or last
unconditionally. Nothing between the model output and the planner asserts the range.

### 4 — `PickPlacePose` grid indices and `RobotStatus.cycle_time_ms` unbounded. `[EXEC]`

`grid_row = -5, grid_col = -5, cartridge_id = -99` all construct. Negative grid indices
are the classic numpy silent-wraparound hazard. `plan/scene.py` defends itself well
here — `OccupancyGrid.set_block` raises `IndexError` on out-of-range blocks with an
explicit bounds check, and its docstring explains that numpy's "forgiving negative /
past-the-end indexing" would "silently write a smaller region than the caller asked for"
`[READ]`. **That defence is in the consumer, not the contract.** Any future consumer
re-derives it or does not. Likewise `cycle_time_ms = -42.0` constructs and would flow into
latency statistics as a negative duration.

### 5 — `RobotStatusCode` numbering is coupled to a `.src` file by a bare literal. `[GREP]`

The enum docstring:

> Values must not be renumbered without matching updates in :mod:`execution.protocol` and
> the KRL subroutine.

`[GREP]` `execution/protocol.py` mentions `RobotStatusCode` **only in a prose docstring**;
it packs and unpacks a raw `int`. `execution/krl_prog/routines.src:81` contains:

```
      RETURN 2                 ; PICK_FAILED
```

A magic literal in a file no test reads and no import touches. Renumbering the enum
silently redefines what the real controller means by `2`. Nothing — no test, no assertion,
no build step — couples the two.

The client side is genuinely well done and deserves saying so `[READ]`:
`execution/execution.py:498` refuses an unknown code rather than substituting one, with a
comment explaining that substituting `TIMEOUT` "made an unknown code from a future firmware
indistinguishable from an ordinary placement failure". The rigour is on the decode path;
it is the *numbering* that is uncoupled.

One related observation: codes 7 and 8 (`UNSUPPORTED_COMMAND`, `VERSION_MISMATCH`) exist,
per their comment, "because a controller that answers 'I could not parse that' must be able
to say WHY". `[GREP]` neither appears anywhere in `routines.src`. They are emitted by
`mock_kuka_server.py` only. The reasoning is sound; the real controller does not
participate in it.

### 6 — `camera.mm_per_px_y` is declared and read by nothing. `[GREP]`

`configs/planning.yaml:34` declares `mm_per_px_y: 0.38`. `PlannerConfig.from_dict`
(`plan/planner.py:142`) reads **only** `mm_per_px_x`, and `_image_to_workspace`
(`planner.py:669`) applies that single scalar to both axes:

```python
return (px * mm_per_px + self.cfg.origin_offset_x_mm,
        py * mm_per_px + self.cfg.origin_offset_y_mm)
```

Setting `mm_per_px_y` to anything at all changes nothing; the y axis silently adopts the x
scale. **Already found** — `docs/superpowers/audit/2026-08-12-F-execution-and-config.md:363`
and `2026-08-12-E-silent-failures.md:558` both name it. Recorded here because it is the
only *frame/unit* discrepancy this audit found in the config surface, and because it is the
same defect class as the `workspace_bounds_mm` key the README describes as "parsed and
compared against nothing" — a declared calibration that calibrates nothing.

### 7 — `Snapshot.image_shape` defaults to a lie. `[READ]`

`image_shape: tuple[int, int] = (1080, 1920)` with a `# (H, W)` comment. A default-constructed
`Snapshot` claims a specific resolution it has no knowledge of. `[GREP]` both real producers
set it correctly (`recog/inference.py:154` `image_rgb.shape[:2]`, `:202` `(h, w)`) and the
`(H, W)` order is used consistently — **no frame-order bug found**. But the default is a
plausible wrong answer rather than an absent one, the same failure shape as the
`mm_per_px = 0.38` constant that was removed for exactly this reason. `None` would be
honest, at the cost of every consumer handling it.

## 2.3 Optional fields and `None`

`[GREP]` + `[READ]`. The brief's hypothesis was that `mm_per_px` got rigour and the others
did not. That is **half right**, and the good half is better than expected.

**`mm_per_px` — rigorous, and the model to copy.** Three layers `[READ]`:

1. `_resolve_scale` (`plan/placement_area.py:105`) is "the only place that precedence is
   decided". It raises `UnknownScale` when neither the frame's own scale nor a configured
   fallback exists, and **also** raises when the resolved value is `<= 0.0` — "a pixel
   cannot span zero or negative millimetres". Range validation, not just null-checking.
2. `PlacementArea.mm_per_px` is typed non-`Optional` `float` (`placement_area.py:75`) and
   is always populated from `_resolve_scale`'s output, so it is structurally impossible for
   it to be `None`.
3. `_drop_areas_measured_at_another_scale` (`planner.py:387`) invalidates any cartridge
   whose stored scale differs from the current frame's, *and* releases its reservations,
   because "their rows, columns AND millimetres were all measured against a rectangle that
   no longer applies".

**`Cartridge.mm_per_px` — the docstring claims a biconditional, and it happens to hold.**
The claim is "None exactly while the rectangle is None". `[READ]` nothing enforces it, but
it cannot currently be violated: the only live writer (`planner.py:501-507`) sets both from
a `PlacementArea` whose `mm_per_px` is non-`Optional`, and the only invalidator
(`planner.py:394-396`) clears both together. The dead `attach_placement_area` also preserves
it. **This one is safe by construction rather than by check** — which is fine until someone
adds a third writer.

**The rest of the `Optional` surface is honestly handled.** `plan/scene.py`'s `Cartridge`
fields (`pcb_mask`, `placeable_rectangle`, `occupancy`) are `None` until the extractor runs,
and every consumer checks: `Cartridge.mark_cell`, `reserve` and `confirm` all raise
`RuntimeError("occupancy not initialised")`; `planner.py:293` and `:388` guard on
`placeable_rectangle is None`. `common/packing.py`'s `forbidden_mask: Optional[np.ndarray]`
is checked at every use site. **No unguarded `Optional` dereference was found on any
production path.**

The one field matching the brief's "usually populated, works until it doesn't" profile is
**`Snapshot.cartridge_masks`** — a `dict` defaulting to empty, consumed at
`planner.py:455`. It is empty for the heuristic detector by design and populated for the
segmenter path. `[READ]` `planner.py:168` records the intended discipline —
"if ``cartridge_masks`` is ever non-empty while it is the selected [heuristic] extractor" —
and `_accepts_label_map` dispatches on extractor capability rather than on the dict's
contents, which is the correct shape. Not a violation; worth naming as the field most
likely to become one.

## 2.4 Docstring-only invariants worth enforcing

`[GREP]` + `[READ]`. Ordering, range, non-emptiness and inter-field claims asserted in
prose and checked by no code. Ranked by what breaking one costs.

| # | Invariant | Stated at | Enforced |
|---|---|---|---|
| 1 | `RobotStatusCode` values must not be renumbered without matching `execution.protocol` and the KRL subroutine | `common/types.py:225` | **No.** KRL side is a bare `RETURN 2` literal at `routines.src:81`; no test reads that file. |
| 2 | `xmin`/`ymin` inclusive, `xmax`/`ymax` exclusive — implying `xmin <= xmax` | `common/types.py:27` | **No.** `[EXEC]` inverted box constructs, `area` and `iou` launder it to `0.0`. |
| 3 | "The planner consumes it as read-only; nothing downstream mutates it" | `common/types.py:118` | **No.** Convention only. `Snapshot` is mutable and reachable from the planner. |
| 4 | `Cartridge.mm_per_px` is "None exactly while the rectangle is None" | `plan/scene.py:170` | **No** — but currently true by construction. A one-line assert would keep it true. |
| 5 | Reservation cells quantise outward (floor near, ceil far) "matching `_overlaps_forbidden` exactly" — "any other rounding lets the mask be tested at one size and marked at another" | `plan/scene.py:79`, `plan/planner.py:712` | **Partially.** Both sites implement it and `tests/test_packing_ceiling.py` / `test_packing_forbidden.py` assert `_overlaps_forbidden`'s behaviour, but **no test asserts the two implementations agree** — which is what the docstring actually claims. |
| 6 | `first_fit_decreasing` is "deliberately FROZEN"; `recog.synth3d` depends on its exact output | `common/packing.py:296`, `plan/bin_packing.py:27` | **No golden lock.** `[GREP]` `tests/test_bin_packing.py` asserts behaviour (fits, rotations, forbidden cells) but pins no reference output. The dependency is real and correctly described `[GREP]`; the freeze is guarded by intent, not by a fixture. |
| 7 | `_sweep` returns "Every candidate tau, ascending" | `recog/calibrate_tau.py:81` | **Yes, by construction** (`sorted({...})`) — listed as a true positive of the grep, not a defect. |
| 8 | "A label map must never be interpolated" | `recog/bay_segmenter.py:134`, `recog/seg_dataset.py:125` | **Yes** — `INTER_NEAREST` at both sites `[GREP]`. Correct. |
| 9 | `pairwise_class_overlaps` keys are "alphabetically sorted pairs, independent of `SEG_CLASSES`' own ordering" | `recog/check_annotations.py:208` | **By construction**, not asserted. Low consequence. |

**The three worth acting on are 1, 2 and 5.** #1 because a renumber corrupts the meaning of
robot status on real hardware and nothing would notice until a cell was dropped. #2 because
`BBox` is the foundational geometric type and its violation is self-concealing. #5 because
the docstring at `plan/scene.py:79` states, correctly, that a footprint "marked smaller than
it is tested against is exactly how a battery gets packed into space another one already
occupies" — and that agreement is the one property the tests do not check.

---

## What this audit did not do

* Did not execute the Blender path (no `bpy` in this environment). Everything about
  `recog/synth3d/*` and `generate3d.py` is `[IMPORT]`, `[CALLGRAPH]`, `[GREP]` or `[READ]`.
* Did not re-run the test suite or regenerate coverage; `[COV]` figures are read from the
  `.coverage` database present at HEAD.
* Did not trace dynamic dispatch through config strings exhaustively. The call graph
  over-approximates method reachability, so the dead set reported is a **lower bound**.
* Changed nothing. No file outside this report was written, staged or committed.
