# Spec #2 (generalisation) Implementation Plan

*What this is: the task-by-task implementation plan a design spec was executed from, kept as part of this project's working record rather than written as documentation for a reader. The note that follows is tooling direction for the coding agent that executed the plan. For what these documents are, how they were used and what came of them, see [`../specs/README.md`](../specs/README.md).*

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the procedural cartridge-tray family and the two new cell
formats (21700, 26650) that `docs/superpowers/specs/2026-08-10-generalisation-design.md`
designs, then run the generalisation measurement the design exists to
produce: two procedural-trained segmenters (anchored, wide), a held-out
CAD-only test set spanning all four Anker SKUs, and a CAD-trained control,
all scored on the same crops and reported per-SKU per-class.

**Architecture.** A procedural tray is *generated* geometry, sampled
bpy-free and built in Blender only from numbers the bpy-free side already
decided (design spec §2, §3). Three existing bpy-free modules gain the new
surface: `config.py` (tunable ranges + the cell-format table), `bay.py`
(the pure-RNG sampler), `catalog.py` (forward-construction into a
catalog-entry-shaped dict, the mirror of `inspect_glb`). One bpy module
(`world.py`) gains a builder structurally identical to `build_jig`.
`assets.AssetLibrary` gains exactly one new branch in `_load_template`
(design spec §4.2's "merged namespace, one branch point") so every
existing consumer — `scene.py`, `bay.py`'s whole consumption surface,
`AssetLibrary.instantiate` — is unchanged. Cell-format generalisation
(21700/26650) is a second, independent axis: a lookup table replacing one
hardcoded 18650 constant, threaded through the two call sites that read it
unconditionally.

**Tech stack.** Python 3.10+, Blender 5.0 bundled Python (no PyYAML, no
pycocotools) for anything importing `bpy`, ordinary CPython + pytest for
everything else, NumPy, PyTorch for training/eval.

**What deliberately gets no task, because the design spec and the
groundwork investigation already confirmed it needs no change** (design
spec §6.3, groundwork §4.4): `plan/arbitration.py`'s `admits_a_cell`
(caller-supplied `cell_w_px`/`cell_h_px`, already generic);
`recog/seg_dataset.py`'s crop/label machinery (fixed-size ROI crops,
format-agnostic — Task 12 below adds asset-tracking to this file, which
is a *different* capability than the format-hardcode audit that cleared
it); the class taxonomy (`recog.dataset.CLASS_MAP`, `config.CLASSES`,
`config.SEG_CLASSES` — `"battery"` already covers any cylindrical cell);
`common/packing.py`/`plan/bin_packing.py` (`Item(id, width, height)`
already generic); `recog/inference.py`'s aspect-ratio gate (`2.0 <=
aspect <= 5.0` already spans 21700's 3.33 and 26650's 2.5).

---

## Global constraints

State these verbatim; every task is written to respect them.

- Every geometric decision lives in bpy-free `bay.py`/`catalog.py`/`config.py`;
  `world.py` and `scene.py` import bpy and cannot be unit-tested. Enforced by
  an AST check (`tests/test_synth3d.py`'s `_imports_bpy`/
  `_BPY_FREE_CANDIDATES`).
- `plan/arbitration.py` stays torch-free (it already imports only `numpy`
  and `cv2` — nothing in this plan changes that).
- `CLASSES` and `SEG_CLASSES` do not change.
- Must not regress: five-class disjointness at 0 overlapping pixels; the
  `assembled` variant; `unit_id` grouping and unit-scoped VOC boxes; the
  torch-free demo `python main.py --config configs/demo.yaml`.
- Run `python -m recog.sync_config` after any YAML change.
- Long renders run in the **foreground**.
- Cell dimensions now come from `config.CELL_W_MM`/`CELL_H_MM` (commit
  `f4596e8`). `plan/planner.py` deliberately uses 18.5 mm nominal rather
  than the 18.3 mm CAD figure — do **not** unify it.
- Documentation only was the constraint on the two *planning* documents;
  this plan itself is followed by real code changes, and its own
  verification bar is `python -m pytest -q` staying green after every task,
  starting from 621 passed at `f4596e8`.

## Resolving two places the design spec's prose is genuinely ambiguous

Both are called out here, once, rather than silently guessed at inside a
task — the same discipline the design spec itself uses for its own
judgement calls.

**1. "Bay depth" names two different quantities in the design spec, with
two different anchor numbers.** §1.1's axis table defines "bay depth
(height)" as "how far the cavity floor sits below the case's own top rim"
— a **Z-axis** quantity, anchored (per that section's own text and per
`groundwork.md` §1.3) near the measured **11.1 mm** case half-height. But
§5.2 step 3 says "jitter near 11.1 mm — bay depth on the fourth side...
this directly gives `module_bay_mm`", using the *same name* for what is
structurally an **XY-plane** inset (how far the module bay strip cuts into
the interior) — a quantity §1.1's own anchor table separately measures at
**19.45–30.75 mm** ("module-bay depth (mm)" column), not 11.1 mm. Applying
an 11.1 mm anchor to an XY inset would produce anchored trays whose bays
are smaller than every real SKU's, which contradicts Decision 2's "every
[anchored] tray stays plausible as real hardware." This plan therefore
uses **two** separate sampled fields — `case_half_height_mm` (Z, anchored
near 11.1 mm) and `bay_margin_mm` (XY, anchored near 19.45–30.75 mm) — see
Task 5.

**2. `interior_mm = tray_outer` (§5.2 step 4) cannot mean the cavity has
zero wall thickness.** Read literally, step 4 sets `interior_mm` equal to
the very rectangle step 2 calls `tray_outer` (cell union inset by wall on
three sides, by bay margin on the fourth) — but if that rectangle were also
the *physical case's own outer surface*, the wall would have no thickness
at all anywhere `world.py` could carve a cavity out of it. Reading `§5.1`
together with this ("`tray_outer_mm`... exist purely so `inspect_glb` can
derive [the three consumed fields]... kept... as audit provenance the
Blender side never reads again" and §3.3's "an outer footprint... needed
by the bpy builder in §3.4"), the resolution used here: `interior_mm` (what
`scene.py` reads) is the **cavity's** boundary — exactly step 2+3's
rectangle. A **separate**, larger rectangle — this plan calls it
`case_outer_mm`, never read by `scene.py`, used only by
`world.build_procedural_tray` — is `interior_mm` expanded outward by
`wall_mm` on **all four sides**, including the bay side, so the physical
shell has real wall thickness everywhere. See Task 6.

---

## File structure

| File | Responsibility |
|---|---|
| `recog/synth3d/config.py` | `CELL_FORMATS` table; `TrayRangeCfg` dataclass; `Config.tray_anchored`/`tray_wide`. |
| `recog/synth3d/bay.py` | `TraySample` + `sample_tray` (the pure-RNG sampler); public `bay_edge` wrapper. |
| `recog/synth3d/catalog.py` | `build_tray_entry`, `build_procedural_pool`, `exclude_assets`. |
| `recog/synth3d/world.py` | `build_procedural_tray`; per-format seated-cell footprint/assertion. |
| `recog/synth3d/assets.py` | `CELL_FORMAT_PROP`/`object_cell_format`; `AssetLibrary.register_procedural_pool`; one new `_load_template` branch. |
| `recog/synth3d/scene.py` | Reads `entry["cell_format"]` for the seated-cell packer call. No new arithmetic. |
| `recog/synth3d/_gate_orientation.py` | Per-format (not hardcoded 18.3/65.0) cell-size assertion. |
| `recog/generate3d.py` | `--tray-set {cad,anchored,wide}`, `--n-procedural`, `--exclude-asset` CLI. |
| `recog/synth3d/annotate.py` | `write_coco_json` carries `asset` through the sidecar. |
| `recog/seg_dataset.py` | `BaySegDataset.sample_assets` — per-crop SKU tracking. |
| `recog/seg_evaluate.py` | `group_indices_by_asset`, `format_per_sku_table`, `--per-sku` CLI. |
| `configs/synth3d.yaml` | `tray_anchored:`/`tray_wide:` sections. |
| `configs/segmentation_anchored.yaml`, `configs/segmentation_wide.yaml`, `configs/segmentation_cad_control_holdout_<SKU>.yaml` (×4), `configs/segmentation_cad_test.yaml` | New training/eval configs, one dataset each (Task 18/19). |
| `tests/test_synth3d.py`, `tests/test_bay.py`, `tests/test_seg_dataset.py`, `tests/test_annotate_masks.py`, `tests/test_bay_segmenter.py` | New unit tests, appended. |

---

# Group A — Cell-format generalisation (independent; do first or in parallel)

### Task 1: `CELL_FORMATS` table

**Files:**
- Modify: `recog/synth3d/config.py`
- Test: `tests/test_synth3d.py` (append)

**Interfaces:**
- Produces: `CELL_FORMATS: Dict[str, Tuple[float, float]]` — `{"18650": (short_m, long_m), "21700": (0.021, 0.070), "26650": (0.026, 0.065)}`, metres, keyed by a format name every later task (world.py, catalog.py, bay.py, scene.py) imports.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synth3d.py`:

```python
def test_cell_formats_18650_matches_the_authoritative_constant():
    """CELL_W_MM/CELL_H_MM is the one authoritative 18650 figure (commit
    f4596e8) - CELL_FORMATS must derive from it, not restate a third copy."""
    from recog.synth3d.config import CELL_FORMATS, CELL_H_MM, CELL_W_MM
    assert CELL_FORMATS["18650"] == pytest.approx(
        (CELL_W_MM / 1000.0, CELL_H_MM / 1000.0))


def test_cell_formats_has_all_three_decision_3_formats():
    from recog.synth3d.config import CELL_FORMATS
    assert set(CELL_FORMATS) == {"18650", "21700", "26650"}
    assert CELL_FORMATS["21700"] == pytest.approx((0.021, 0.070))
    assert CELL_FORMATS["26650"] == pytest.approx((0.026, 0.065))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_synth3d.py -v -k cell_formats`
Expected: FAIL — `ImportError: cannot import name 'CELL_FORMATS'`.

- [ ] **Step 3: Add the table**

In `recog/synth3d/config.py`, immediately below the existing
`CELL_W_MM`/`CELL_H_MM` block:

```python
# Decision 3 (2026-08-10-generalisation-decisions.md): three cell formats
# in the training mix. "18650" is CELL_W_MM/CELL_H_MM itself, converted to
# metres rather than restated a third time - the same "one authoritative
# dimension" discipline that consolidation established. 21700/21x70mm and
# 26650/26x65mm have no CAD (groundwork.md Sec4.2: a parametric cylinder
# needs no CAD); they exist here as radii/lengths a Blender primitive_
# cylinder_add can be built from directly.
CELL_FORMATS: Dict[str, Tuple[float, float]] = {
    "18650": (CELL_W_MM / 1000.0, CELL_H_MM / 1000.0),
    "21700": (0.021, 0.070),
    "26650": (0.026, 0.065),
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_synth3d.py -v -k cell_formats`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recog/synth3d/config.py tests/test_synth3d.py
git commit -m "feat(synth3d): add the CELL_FORMATS table (Decision 3)

18650 (from the existing CELL_W_MM/CELL_H_MM), 21700 and 26650, in
metres, keyed by name. Nothing reads this yet - the two call sites that
hardcode a single-format footprint are generalised in the next task."
```

---

### Task 2: tag every cell object with the format it was built as

**Files:**
- Modify: `recog/synth3d/assets.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `CELL_FORMAT_PROP = "synth3d_cell_format"` (a custom ID
  property, alongside the existing `ROLE_PROP`); `object_cell_format(o) ->
  str`, defaulting to `"18650"` for any object never tagged (every CAD
  template's cell, unconditionally, both before and after this task).

**No pytest possible — `assets.py` imports `bpy` at module level and
cannot be imported outside Blender.** Verified by Task 4's gate script and
a dev render (Task 11).

- [ ] **Step 1: Add the tag and accessor**

In `recog/synth3d/assets.py`, next to the existing `ROLE_PROP`:

```python
# Same pattern as ROLE_PROP: a custom ID property survives `clone()`'s
# `src.copy()`, so every instance of a cell template carries the format it
# was built as. Only "cell"-role objects are ever tagged; object_cell_format
# below defaults to "18650" for anything that isn't, which is every CAD
# cell today (CAD never varies format) - so this task changes no CAD
# behaviour by itself.
CELL_FORMAT_PROP = "synth3d_cell_format"


def object_cell_format(o) -> str:
    """The cell format `_load_template` tagged `o` with, defaulting to
    "18650" - every object never tagged is a CAD template's own cell,
    which is always 18650."""
    return o.get(CELL_FORMAT_PROP) or "18650"
```

- [ ] **Step 2: Tag cell objects in `_load_template`**

In `_load_template`, the existing tagging loop:

```python
        for role, objs in by_role.items():
            for o in objs:
                o[ROLE_PROP] = role
```

becomes:

```python
        cell_format = self.assets[name].get("cell_format", "18650")
        for role, objs in by_role.items():
            for o in objs:
                o[ROLE_PROP] = role
                if role == "cell":
                    o[CELL_FORMAT_PROP] = cell_format
```

`self.assets[name].get("cell_format", "18650")` is safe for every entry
that exists today: CAD entries never carry a `"cell_format"` key, so this
is `"18650"` unconditionally until Task 9 registers procedural entries
that do carry one.

- [ ] **Step 3: Commit**

```bash
git add recog/synth3d/assets.py
git commit -m "feat(synth3d): tag cell templates with the format they were built as

CELL_FORMAT_PROP mirrors ROLE_PROP. Every CAD cell defaults to 18650
(the only format that exists today); nothing reads the tag yet."
```

---

### Task 3: per-format seated-cell footprint

**Files:**
- Modify: `recog/synth3d/world.py`
- Modify: `recog/synth3d/scene.py`

**Interfaces:**
- Consumes: `config.CELL_FORMATS` (Task 1), `entry.get("cell_format",
  "18650")` (a catalog entry field every CAD entry lacks today, defaulting
  safely; procedural entries will carry it from Task 6 onward).
- Produces: `world.seat_cells(library, asset, seats, floor_z, rng, cfg,
  cell_format="18650", backdrop_luma=None)` — new keyword, defaulting to
  today's only format. `world._assert_seat_cell_footprint(asset:
  str, cell_format: str, lo, hi) -> None` — was `(asset, lo, hi)`.

No pytest possible (bpy). Verified by Task 4's gate script and a dev
render of the *existing* CAD assets, confirming byte-for-byte-equivalent
seated-cell placement (default `cell_format="18650"` must reproduce
current behaviour exactly).

- [ ] **Step 1: Generalise `world.py`'s footprint constant**

`recog/synth3d/world.py` currently:

```python
from .config import CELL_H_MM, CELL_W_MM
...
SEAT_CELL_FOOTPRINT_M = (CELL_W_MM / 1000, CELL_H_MM / 1000)

_seat_cell_footprint_checked: set = set()


def _assert_seat_cell_footprint(asset: str, lo: Vector, hi: Vector) -> None:
    if asset in _seat_cell_footprint_checked:
        return
    _seat_cell_footprint_checked.add(asset)
    got = sorted((hi.x - lo.x, hi.y - lo.y))
    want = sorted(SEAT_CELL_FOOTPRINT_M)
    assert all(math.isclose(g, w, abs_tol=5e-4) for g, w in zip(got, want)), (...)
```

Replace with:

```python
from .config import CELL_FORMATS, CELL_H_MM, CELL_W_MM
...
# SEAT_CELL_FOOTPRINT_M is retired: it named one hardcoded format. Every
# call site now looks up CELL_FORMATS[cell_format] directly, keyed by
# whichever format the item being seated actually is (2026-08-10,
# spec #2 cell-format generalisation).

_seat_cell_footprint_checked: set = set()


def _assert_seat_cell_footprint(asset: str, cell_format: str,
                                lo: Vector, hi: Vector) -> None:
    """Assert a freshly lay_flat'd (or, for a procedural tray, directly
    built) cell clone's measured XY footprint matches
    CELL_FORMATS[cell_format], once per (asset, cell_format).

    Generalised from a single hardcoded 18650 check (design spec Sec6.1):
    it now asserts against the format the clone was DRAWN FOR, not a
    single global constant - so a mismatch between what scene.py believes
    an item's format is and what its actual template measures still
    fails loudly, for every format, not just 18650.
    """
    key = (asset, cell_format)
    if key in _seat_cell_footprint_checked:
        return
    _seat_cell_footprint_checked.add(key)
    got = sorted((hi.x - lo.x, hi.y - lo.y))
    want = sorted(CELL_FORMATS[cell_format])
    assert all(math.isclose(g, w, abs_tol=5e-4) for g, w in zip(got, want)), (
        f"{asset}'s seated-cell template measures "
        f"{got[0] * 1000:.2f}x{got[1] * 1000:.2f}mm after lay_flat, not the "
        f"{want[0] * 1000:.2f}x{want[1] * 1000:.2f}mm {cell_format!r} "
        f"CELL_FORMATS entry - scene.py's packer call already sized and "
        f"placed this cell's seat against the assumed figure, so the "
        f"packing and the rendered geometry have desynced.")
```

`_seat_cell_footprint_checked`'s cache key changes from `asset` to `(asset,
cell_format)` — cheap, and removes a hidden assumption that one asset name
never seats two different formats (true today, but no longer
structurally guaranteed once procedural assets exist).

- [ ] **Step 2: Thread `cell_format` through `seat_cells`**

```python
def seat_cells(library, asset: str, seats, floor_z: float, rng: random.Random,
               cfg, cell_format: str = "18650", backdrop_luma=None):
```

Inside the loop, replace the call:

```python
        lo, hi_obj = A.group_bbox([dup])
        _assert_seat_cell_footprint(asset, lo, hi_obj)
```

with:

```python
        lo, hi_obj = A.group_bbox([dup])
        _assert_seat_cell_footprint(asset, cell_format, lo, hi_obj)
```

- [ ] **Step 3: `scene.py` reads the entry's own format**

`recog/synth3d/scene.py` imports `from .config import Config, VARIANTS,
Variant`. Add `CELL_FORMATS` to that import. In the seated-cells block
(the `if rng.random() < cfg.layout.p_seated:` branch), replace:

```python
                    cell_w, cell_h = W.SEAT_CELL_FOOTPRINT_M
```

with:

```python
                    cell_format = entry.get("cell_format", "18650")
                    cell_w, cell_h = CELL_FORMATS[cell_format]
```

and the `W.seat_cells(...)` call gains `cell_format=cell_format`:

```python
                    item.seated_objects = W.seat_cells(
                        library, item.asset, world_seats, floor_z, rng,
                        cfg, cell_format=cell_format,
                        backdrop_luma=backdrop_luma)
```

`entry` is already in scope (the `library.catalog_entry(item.asset)` bound
earlier in the same `open_case` block). `entry.get("cell_format",
"18650")` is safe for CAD entries (no such key today) and for procedural
entries (Task 6 onward, which always sets it).

- [ ] **Step 4: Verify unchanged CAD behaviour**

```bash
python -m recog.sync_config
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/synth3d/_gate_orientation.py
"$BL" -b --python recog/generate3d.py -- --n 8 --out recog/dev3d --device GPU --variant open_case
python -m recog.verify3d --data recog/dev3d --n 8 --masks
```

Expected: gate reports `ORIENTATION GATE OK`; the dev render's seated
cells look identical to a pre-Task-3 render (still 18650, still the same
pitch) — nothing in this task changes what format any *existing* asset
seats, only what format a *future* one can.

- [ ] **Step 5: Commit**

```bash
git add recog/synth3d/world.py recog/synth3d/scene.py
git commit -m "feat(synth3d): seat cells at the item's own cell format, not a global constant

SEAT_CELL_FOOTPRINT_M named one hardcoded 18650 footprint used for every
scene, every asset (groundwork.md Sec4.3, worst-offender #1). Replaced by
a per-item CELL_FORMATS lookup keyed off entry['cell_format'] (defaulting
to 18650 for every CAD entry, unchanged). _assert_seat_cell_footprint now
checks the format the clone was actually drawn for."
```

---

### Task 4: per-format orientation-gate check

**Files:**
- Modify: `recog/synth3d/_gate_orientation.py`

**Interfaces:**
- Consumes: `assets.object_cell_format` (Task 2), `config.CELL_FORMATS`
  (Task 1).

Design spec §6.2: this must land **before** any new-format cell enters a
template the gate checks, "or the gate breaks the build rather than
catching a real defect." No procedural or new-format CAD cell exists yet
(Group B builds the first one), so this task is purely prophylactic —
correct now, exercised for real once Task 11's dev library includes a
non-18650 cell.

- [ ] **Step 1: Generalise the hardcoded check**

In `_gate_orientation.py`'s `check_lay_flat`, replace:

```python
        for role, o in parts:
            if role != "cell":
                continue
            e, _, _ = extent([o])
            check(abs(e.z * MM - config.CELL_W_MM) < 0.5,
                  f"{name}/cell height {round(e.z * MM, 1)}mm ~= "
                  f"{config.CELL_W_MM}mm ({config.CELL_H_MM}mm would mean "
                  f"it is standing on end)")
            check(abs(max(e.x, e.y) * MM - config.CELL_H_MM) < 0.5,
                  f"{name}/cell long axis {round(max(e.x, e.y) * MM, 1)}mm "
                  f"~= {config.CELL_H_MM}mm and lies in the XY plane")
```

with:

```python
        for role, o in parts:
            if role != "cell":
                continue
            fmt = assets.object_cell_format(o)
            want_w_mm, want_h_mm = (v * MM for v in config.CELL_FORMATS[fmt])
            e, _, _ = extent([o])
            check(abs(e.z * MM - want_w_mm) < 0.5,
                  f"{name}/cell ({fmt}) height {round(e.z * MM, 1)}mm ~= "
                  f"{want_w_mm:.1f}mm ({want_h_mm:.1f}mm would mean it is "
                  f"standing on end)")
            check(abs(max(e.x, e.y) * MM - want_h_mm) < 0.5,
                  f"{name}/cell ({fmt}) long axis "
                  f"{round(max(e.x, e.y) * MM, 1)}mm ~= {want_h_mm:.1f}mm "
                  f"and lies in the XY plane")
```

- [ ] **Step 2: Run the gate**

```bash
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/synth3d/_gate_orientation.py
```

Expected: `ORIENTATION GATE OK` — identical result to before this task,
since every existing cell is still 18650 and `object_cell_format` defaults
to `"18650"`.

- [ ] **Step 3: Commit**

```bash
git add recog/synth3d/_gate_orientation.py
git commit -m "fix(synth3d): make the orientation gate's cell-size check per-format

Was a literal 18.3/65.0mm assertion against every role=='cell' object in
every template the gate checks. Design spec Sec6.2: this has to land
before any 21700/26650 (CAD or procedural) cell exists, or the gate
false-fails on the first one instead of catching a real defect."
```

---

# Group B — Procedural tray builder (must precede anything that renders with it)

### Task 5: `TrayRangeCfg` — the anchored/wide sampling ranges

**Files:**
- Modify: `recog/synth3d/config.py`
- Modify: `configs/synth3d.yaml`
- Test: `tests/test_synth3d.py` (append)

**Interfaces:**
- Produces: `TrayRangeCfg` dataclass; `Config.tray_anchored: TrayRangeCfg`,
  `Config.tray_wide: TrayRangeCfg`; `_SECTIONS`/`_TUPLE_FIELDS` updated so
  `load_config` parses both.

Decision 2's "kept as separate, separately-scored sets" (design spec
§3.1: "two range-sets, not one config with a 'how wide' knob") is enforced
structurally here: `tray_anchored`/`tray_wide` are two **separate
instances** of one dataclass, never one config with a widening factor.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synth3d.py`:

```python
# ------------------------------------------------------ procedural tray --

def test_tray_anchored_and_wide_are_separate_instances():
    from recog.synth3d.config import Config
    cfg = Config()
    assert cfg.tray_anchored is not cfg.tray_wide
    assert cfg.tray_anchored.free_bay_edge is False
    assert cfg.tray_wide.free_bay_edge is True


def test_tray_anchored_wall_range_brackets_the_measured_four_skus():
    """Measured (design spec Sec1.1, catalog.json): wall 3.70-4.25mm
    across all four Anker assemblies. Decision 2: anchored stays "within
    and slightly beyond" that span."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.wall_mm_range
    assert lo <= 3.70 and hi >= 4.25


def test_tray_anchored_bay_margin_range_brackets_the_measured_four_skus():
    """Measured module-bay depth: 19.45-30.75mm (design spec Sec1.1's
    anchor table) - the XY quantity Sec5.2 step 3 builds module_bay_mm
    from. See this plan's header note on why it is a separate field from
    case_half_height_mm."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.bay_margin_mm_range
    assert lo <= 19.45 and hi >= 30.75


def test_tray_anchored_case_half_height_is_a_modest_jitter_not_a_wide_draw():
    """Design spec Sec1.1: case half-height is 11.1mm on all four SKUs -
    n=1, not a 4-point spread. The anchored sampler treats this as "near
    the single observed value", not a wide draw."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.case_half_height_mm_range
    assert lo < 11.1 < hi
    assert (hi - lo) <= 2.0, f"range {hi - lo}mm is not a modest jitter band"


def test_tray_wide_ranges_are_strictly_wider_than_anchored():
    from recog.synth3d.config import Config
    cfg = Config()
    for field in ("n_cols_range", "n_rows_range", "pitch_mm_range",
                  "wall_mm_range", "bay_margin_mm_range",
                  "case_half_height_mm_range", "tray_floor_mm_range"):
        a_lo, a_hi = getattr(cfg.tray_anchored, field)
        w_lo, w_hi = getattr(cfg.tray_wide, field)
        assert w_lo <= a_lo and w_hi >= a_hi, field


def test_load_config_accepts_tray_sections(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "tray_anchored: {wall_mm_range: [3.5, 4.5]}\n"
        "tray_wide: {free_bay_edge: true}\n", encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg.tray_anchored.wall_mm_range == (3.5, 4.5)
    assert cfg.tray_wide.free_bay_edge is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_synth3d.py -v -k tray`
Expected: FAIL — `AttributeError: 'Config' object has no attribute
'tray_anchored'`.

- [ ] **Step 3: Add `TrayRangeCfg` and wire it into `Config`**

In `recog/synth3d/config.py`, below `ObstructionCfg`:

```python
@dataclass
class TrayRangeCfg:
    """One procedural sampling range-set - either the anchored or the
    wide population (Decision 2). config.Config.tray_anchored and
    .tray_wide are two SEPARATE instances of this dataclass, never one
    config with a "how wide" knob, because the separation has to be
    structural or Decision 2's "kept separable" requirement has nothing
    to enforce it. See bay.sample_tray for how each field is used.

    Two distinct depth-ish quantities, deliberately not one field (see
    this plan's header note): `case_half_height_mm_range` is the Z-axis
    case half-height (measured 11.1mm, all four SKUs); `bay_margin_mm_
    range` is the XY-plane depth of the module bay strip (measured
    19.45-30.75mm).
    """
    cell_formats: Tuple[str, ...] = ("18650", "21700", "26650")
    n_cols_range: Tuple[int, int] = (3, 5)
    n_rows_range: Tuple[int, int] = (1, 2)
    pitch_mm_range: Tuple[float, float] = (0.5, 2.0)
    wall_mm_range: Tuple[float, float] = (3.3, 4.7)
    bay_margin_mm_range: Tuple[float, float] = (17.0, 34.0)
    case_half_height_mm_range: Tuple[float, float] = (10.5, 11.7)
    tray_floor_mm_range: Tuple[float, float] = (1.6, 2.3)
    # Design spec Sec9.1: anchored fixes the bay to a short edge (matching
    # 4/4 measured SKUs); wide samples freely among all four edges,
    # including physically implausible ones. False = anchored's rule.
    free_bay_edge: bool = False


def _wide_tray_defaults() -> "TrayRangeCfg":
    """Decision 2's "well outside" population - unusual aspect ratios,
    thicker/thinner walls, denser packing, and (free_bay_edge) module
    positions no real SKU shows."""
    return TrayRangeCfg(
        n_cols_range=(2, 8), n_rows_range=(1, 4),
        pitch_mm_range=(0.3, 6.0), wall_mm_range=(1.5, 9.0),
        bay_margin_mm_range=(6.0, 60.0),
        case_half_height_mm_range=(4.0, 28.0),
        tray_floor_mm_range=(0.8, 4.5), free_bay_edge=True)
```

Add fields to `Config`:

```python
    tray_anchored: TrayRangeCfg = field(default_factory=TrayRangeCfg)
    tray_wide: TrayRangeCfg = field(default_factory=_wide_tray_defaults)
```

Wire into the loader:

```python
_SECTIONS = {"render": RenderCfg, "layout": LayoutCfg,
             "camera": CameraCfg, "filter": FilterCfg,
             "obstruction": ObstructionCfg,
             "tray_anchored": TrayRangeCfg, "tray_wide": TrayRangeCfg}
_TUPLE_FIELDS = {"res", "area", "margin_range", "shift_range", "jig_depth",
                 "n_adhesive", "n_foam", "n_tape", "n_label",
                 "adhesive_frac", "foam_frac", "tape_frac", "label_frac",
                 "seated_frac", "cell_formats", "n_cols_range",
                 "n_rows_range", "pitch_mm_range", "wall_mm_range",
                 "bay_margin_mm_range", "case_half_height_mm_range",
                 "tray_floor_mm_range"}
```

`_wide_tray_defaults` cannot be a lambda passed to `field(default_factory=
...)` directly inline if it needs to be referenced by name in a docstring,
but a plain module-level function works identically — used exactly as
`RenderCfg`/`LayoutCfg` are, via `field(default_factory=...)`.

- [ ] **Step 4: Document the ranges in YAML**

Append to `configs/synth3d.yaml` (after `obstruction:`):

```yaml
# Procedural cartridge-tray family (spec #2, design spec Sec3.1/Sec5.2).
# Anchored: "within and slightly beyond" what the four Anker assemblies
# span (Decision 2). Measured anchors (catalog.json, all four SKUs):
# wall 3.70-4.25mm, module-bay depth 19.45-30.75mm, case half-height
# 11.1mm (all four, n=1 - a modest jitter band, not a wide draw),
# tray_floor_mm 1.95mm (all four).
tray_anchored:
  cell_formats: [18650, 21700, 26650]
  n_cols_range: [3, 5]
  n_rows_range: [1, 2]
  pitch_mm_range: [0.5, 2.0]
  wall_mm_range: [3.3, 4.7]
  bay_margin_mm_range: [17.0, 34.0]
  case_half_height_mm_range: [10.5, 11.7]
  tray_floor_mm_range: [1.6, 2.3]
  free_bay_edge: false

# Wide: well outside the anchored span - unusual aspect ratios, thicker
# and thinner walls, denser packing, and a module bay allowed on any
# edge, including ones no real SKU shows.
tray_wide:
  cell_formats: [18650, 21700, 26650]
  n_cols_range: [2, 8]
  n_rows_range: [1, 4]
  pitch_mm_range: [0.3, 6.0]
  wall_mm_range: [1.5, 9.0]
  bay_margin_mm_range: [6.0, 60.0]
  case_half_height_mm_range: [4.0, 28.0]
  tray_floor_mm_range: [0.8, 4.5]
  free_bay_edge: true
```

Run `python -m recog.sync_config`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_synth3d.py -v -k tray`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add recog/synth3d/config.py configs/synth3d.yaml configs/synth3d.json tests/test_synth3d.py
git commit -m "feat(synth3d): TrayRangeCfg - the anchored/wide procedural sampling ranges

Two separate dataclass instances (Config.tray_anchored / .tray_wide), not
one config with a widening knob, per Decision 2's 'kept separable'
requirement. Ranges anchored to catalog.json's own measured four-SKU
span; case_half_height_mm_range is a modest jitter band (n=1 observed),
not a wide draw, per design spec Sec1.1."
```

---

### Task 6: `bay.py` — `TraySample` and the pure-RNG sampler

**Files:**
- Modify: `recog/synth3d/bay.py`
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `config.TrayRangeCfg` (Task 5), `config.CELL_FORMATS` (Task 1).
- Produces: `TraySample` (frozen dataclass); `sample_tray(cfg:
  TrayRangeCfg, rng: random.Random) -> TraySample`; public `bay_edge(interior_mm:
  Rect, bay_mm: Rect, tol: float = 1e-6) -> str` (a thin wrapper around the
  existing private `_bay_edge`, so a cross-module caller — `catalog.py`,
  Task 7 — has a non-underscore entry point). Consumed by Task 7's
  `build_tray_entry`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay.py`:

```python
# ------------------------------------------------ procedural tray sampler --
import random as _random

from recog.synth3d.config import CELL_FORMATS, Config


def test_bay_edge_public_wrapper_matches_the_private_validator():
    from recog.synth3d.bay import _bay_edge, bay_edge
    interior = (0.0, 0.0, 60.0, 90.0)
    bay = (0.0, 66.0, 60.0, 90.0)
    assert bay_edge(interior, bay) == _bay_edge(interior, bay) == "+y"


def test_sample_tray_is_deterministic_given_the_same_rng_state():
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    assert sample_tray(cfg, _random.Random(7)) == sample_tray(cfg, _random.Random(7))


def test_sample_tray_cell_format_is_always_a_configured_one():
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    for seed in range(30):
        s = sample_tray(cfg, _random.Random(seed))
        assert s.cell_format in CELL_FORMATS


def test_sample_tray_module_bay_is_a_full_span_strip_on_the_chosen_edge():
    """module_bay_from_bounds's own invariant must hold by construction -
    bay.bay_edge (the existing, tested validator) is reused directly
    rather than re-implementing the check, so the two can never drift
    apart (design spec Sec3.3: 'satisfied by construction, not asserted
    after the fact')."""
    from recog.synth3d.bay import bay_edge, sample_tray
    cfg = Config().tray_anchored
    for seed in range(50):
        s = sample_tray(cfg, _random.Random(seed))
        assert bay_edge(s.interior_mm, s.module_bay_mm) == s.bay_edge


def test_sample_tray_anchored_restricts_the_bay_to_the_long_axiss_ends():
    """Design spec Sec9.1: anchored fixes the module bay to a short edge -
    i.e. the bay axis is the tray's LONGER footprint axis, matching all
    four measured SKUs (e.g. PowerCore10000: 54.9mm short x 84.45mm long,
    bay flush against a long-axis end)."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    for seed in range(50):
        s = sample_tray(cfg, _random.Random(seed))
        ix0, iy0, ix1, iy1 = s.interior_mm
        long_axis_is_y = (iy1 - iy0) >= (ix1 - ix0)
        assert s.bay_edge in (("-y", "+y") if long_axis_is_y else ("-x", "+x"))


def test_sample_tray_wide_can_put_the_bay_on_the_short_axis():
    """The anchored restriction must be a real behavioural difference,
    not decoration - wide has to actually exercise an edge anchored
    would never draw, over enough seeds."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_wide
    saw_short_axis_bay = False
    for seed in range(300):
        s = sample_tray(cfg, _random.Random(seed))
        ix0, iy0, ix1, iy1 = s.interior_mm
        long_axis_is_y = (iy1 - iy0) >= (ix1 - ix0)
        short_axis_edges = ("-x", "+x") if long_axis_is_y else ("-y", "+y")
        if s.bay_edge in short_axis_edges:
            saw_short_axis_bay = True
            break
    assert saw_short_axis_bay, "wide never drew a short-axis bay in 300 seeds"


def test_sample_tray_case_outer_encloses_the_interior_with_real_wall():
    """case_outer_mm must be interior_mm expanded OUTWARD by wall_mm on
    ALL FOUR sides (this plan's header note #2) - not equal to
    interior_mm, or world.build_procedural_tray would have no wall
    thickness to build."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    s = sample_tray(cfg, _random.Random(0))
    ix0, iy0, ix1, iy1 = s.interior_mm
    ox0, oy0, ox1, oy1 = s.case_outer_mm
    assert ox0 == pytest.approx(ix0 - s.wall_mm)
    assert oy0 == pytest.approx(iy0 - s.wall_mm)
    assert ox1 == pytest.approx(ix1 + s.wall_mm)
    assert oy1 == pytest.approx(iy1 + s.wall_mm)


def test_sample_tray_anchored_footprint_roughly_brackets_the_measured_skus():
    """Not exact per-draw (independent axes multiply out wider than any
    single SKU) but the population should land in the measured 62.9x90.9
    - 81.7x180mm neighbourhood, not somewhere wildly different."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    widths, heights = [], []
    for seed in range(200):
        s = sample_tray(cfg, _random.Random(seed))
        ox0, oy0, ox1, oy1 = s.case_outer_mm
        widths.append(ox1 - ox0)
        heights.append(oy1 - oy0)
    widths.sort(); heights.sort()
    assert 40.0 <= widths[len(widths) // 2] <= 130.0
    assert 60.0 <= heights[len(heights) // 2] <= 220.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bay.py -v -k sample_tray`
Expected: FAIL — `ImportError: cannot import name 'sample_tray'`.

- [ ] **Step 3: Add `TraySample`, `sample_tray`, `bay_edge`**

Append to `recog/synth3d/bay.py`:

```python
def bay_edge(interior_mm: Rect, bay_mm: Rect, tol: float = 1e-6) -> str:
    """Public wrapper around `_bay_edge`, for cross-module callers
    (catalog.build_tray_entry) that need to validate a chosen bay edge
    without reaching into a leading-underscore name."""
    return _bay_edge(interior_mm, bay_mm, tol)


@dataclass(frozen=True)
class TraySample:
    """One procedurally-sampled cartridge tray, fully resolved to plain
    numbers - the pure-RNG half of a procedural asset (design spec
    Sec3.2). Millimetres throughout, matching catalog.json's own
    convention.

    `interior_mm` is the CAVITY's own boundary (what scene.py reads).
    `case_outer_mm` is the PHYSICAL shell's outer footprint - interior_mm
    expanded outward by wall_mm on all four sides, including the bay
    side - needed only by world.build_procedural_tray to give the wall
    real thickness; scene.py never reads it (see this plan's header note
    #2 for why this is a separate field rather than aliasing interior_mm
    the way an over-literal reading of design spec Sec5.2 step 4 might
    suggest).
    """
    interior_mm: Rect
    module_bay_mm: Rect
    case_outer_mm: Rect
    wall_mm: float
    case_half_height_mm: float
    tray_floor_mm: float
    cell_format: str
    bay_edge: str


def sample_tray(cfg, rng: random.Random) -> TraySample:
    """Draw one procedural tray from `cfg` (config.Config.tray_anchored
    or .tray_wide - the caller picks which). Pure RNG and arithmetic,
    same shape as sample_obstructions: no Blender call anywhere in this
    function.

    Design spec Sec5.2's five steps, in order:
      1. cell format + an n_cols x n_rows grid at a sampled pitch ->
         cell_union, centred on (0, 0).
      2/3. wall_mm on the three non-bay sides, bay_margin_mm on the
         fourth (the edge chosen by cfg.free_bay_edge - fixed to the
         tray's LONGER axis for anchored, matching design spec Sec9.1's
         "short edge" reading of all four measured SKUs; free among all
         four edges for wide) -> interior_mm, with module_bay_mm falling
         out as the full-span strip between cell_union and interior_mm
         on that one edge, by construction (never inferred, never a tie -
         see bay_edge's own use in the caller as a loud cross-check).
      4/5. tray_floor_mm and case_half_height_mm sampled directly from
         cfg's own ranges.

    See this plan's header note #1 for why case_half_height_mm (Z) and
    the bay_margin used above (XY) are two distinct sampled quantities
    rather than the design spec's single "bay depth" name.
    """
    from .config import CELL_FORMATS

    cell_format = rng.choice(cfg.cell_formats)
    cell_w_mm, cell_h_mm = (v * 1000.0 for v in CELL_FORMATS[cell_format])
    n_cols = rng.randint(*cfg.n_cols_range)
    n_rows = rng.randint(*cfg.n_rows_range)
    pitch_mm = rng.uniform(*cfg.pitch_mm_range)

    union_w = n_cols * cell_w_mm + (n_cols + 1) * pitch_mm
    union_h = n_rows * cell_h_mm + (n_rows + 1) * pitch_mm
    cx0, cy0, cx1, cy1 = -union_w / 2, -union_h / 2, union_w / 2, union_h / 2

    wall_mm = rng.uniform(*cfg.wall_mm_range)
    bay_margin_mm = rng.uniform(*cfg.bay_margin_mm_range)

    long_axis_is_y = union_h >= union_w
    if cfg.free_bay_edge:
        edge = rng.choice(("-x", "+x", "-y", "+y"))
    else:
        edge = rng.choice(("-y", "+y") if long_axis_is_y else ("-x", "+x"))

    if edge == "-x":
        interior = (cx0 - bay_margin_mm, cy0 - wall_mm, cx1 + wall_mm, cy1 + wall_mm)
        module_bay = (interior[0], interior[1], cx0, interior[3])
    elif edge == "+x":
        interior = (cx0 - wall_mm, cy0 - wall_mm, cx1 + bay_margin_mm, cy1 + wall_mm)
        module_bay = (cx1, interior[1], interior[2], interior[3])
    elif edge == "-y":
        interior = (cx0 - wall_mm, cy0 - bay_margin_mm, cx1 + wall_mm, cy1 + wall_mm)
        module_bay = (interior[0], interior[1], interior[2], cy0)
    else:  # "+y"
        interior = (cx0 - wall_mm, cy0 - wall_mm, cx1 + wall_mm, cy1 + bay_margin_mm)
        module_bay = (interior[0], cy1, interior[2], interior[3])

    ix0, iy0, ix1, iy1 = interior
    case_outer = (ix0 - wall_mm, iy0 - wall_mm, ix1 + wall_mm, iy1 + wall_mm)

    return TraySample(
        interior_mm=interior, module_bay_mm=module_bay, case_outer_mm=case_outer,
        wall_mm=wall_mm, case_half_height_mm=rng.uniform(*cfg.case_half_height_mm_range),
        tray_floor_mm=rng.uniform(*cfg.tray_floor_mm_range),
        cell_format=cell_format, bay_edge=edge)
```

Add `from dataclasses import dataclass` if not already imported (it is —
`bay.py` already imports `dataclass` for `ObstructionPose`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_bay.py -v -k "sample_tray or bay_edge"`
Expected: PASS. If `test_sample_tray_anchored_footprint_roughly_brackets_the_measured_skus`
fails, the `n_cols_range`/`n_rows_range`/`pitch_mm_range` defaults in Task
5 need adjusting (widen or narrow) before continuing — this is the
calibration check the header note flags as a starting point, not a
guaranteed-correct guess.

- [ ] **Step 5: Commit**

```bash
git add recog/synth3d/bay.py tests/test_bay.py
git commit -m "feat(synth3d): TraySample and sample_tray - the procedural tray's pure-RNG sampler

Same shape as sample_obstructions: bpy-free, deterministic given an rng.
module_bay_mm is a full-span strip by construction (verified against the
existing bay_edge validator, not re-derived); case_outer_mm is a new,
separate field from interior_mm - see the plan's header note on why."
```

---

### Task 7: `catalog.py` — forward construction and the procedural pool

**Files:**
- Modify: `recog/synth3d/catalog.py`
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `bay.TraySample`, `bay.sample_tray`, `bay.bay_edge` (Task 6).
- Produces: `build_tray_entry(sample: TraySample) -> dict` (a
  catalog-entry-shaped dict, never written to `catalog.json`);
  `build_procedural_pool(n: int, sample_fn, cfg, seed: int, name_prefix:
  str = "proc") -> Dict[str, dict]`; `exclude_assets(assets: Dict[str,
  dict], names: Sequence[str]) -> Dict[str, dict]` (pure dict filter, used
  by Task 10's `--exclude-asset` CLI). Consumed by Task 9
  (`AssetLibrary.register_procedural_pool`) and Task 8
  (`world.build_procedural_tray` reads the dict `build_tray_entry`
  produces).
- **Also pins a naming contract Task 8 must honour**: `role_of` (this
  file) classifies an object by regex-searching its *name* — see Step 3
  below for exactly which names `world.build_procedural_tray` must use.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay.py`:

```python
# --------------------------------------------- procedural catalog entries --

def test_build_tray_entry_carries_the_three_fields_scene_reads():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_tray_entry
    s = sample_tray(Config().tray_anchored, _random.Random(0))
    entry = build_tray_entry(s)
    assert entry["kind"] == "procedural"
    for key in ("interior_mm", "module_bay_mm", "tray_floor_mm"):
        assert key in entry
    assert entry["interior_mm"] == [round(v, 2) for v in s.interior_mm]
    assert entry["cell_format"] == s.cell_format


def test_build_tray_entry_raises_loudly_on_a_malformed_sample():
    """A TraySample whose module_bay isn't a genuine full-span strip must
    fail HERE, at registration time - not silently degrade a scene's
    labels the way scene.py's `entry.get(...)` guard would if a bad
    procedural entry ever reached it (that guard exists for the CAD's
    legitimate no-measurement case, not for a fully-computed procedural
    entry, which has no such excuse)."""
    from recog.synth3d.bay import TraySample
    from recog.synth3d.catalog import build_tray_entry
    bad = TraySample(
        interior_mm=(0.0, 0.0, 60.0, 90.0),
        module_bay_mm=(10.0, 10.0, 50.0, 80.0),   # not flush against any edge
        case_outer_mm=(-4.0, -4.0, 64.0, 94.0), wall_mm=4.0,
        case_half_height_mm=11.1, tray_floor_mm=1.95, cell_format="18650",
        bay_edge="+y")
    with pytest.raises(ValueError):
        build_tray_entry(bad)


def test_build_procedural_pool_makes_n_uniquely_named_entries():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    pool = build_procedural_pool(10, sample_tray, Config().tray_anchored, seed=0)
    assert len(pool) == 10
    assert all(e["kind"] == "procedural" for e in pool.values())


def test_build_procedural_pool_is_reproducible_for_the_same_seed():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    a = build_procedural_pool(5, sample_tray, Config().tray_anchored, seed=3)
    b = build_procedural_pool(5, sample_tray, Config().tray_anchored, seed=3)
    assert a == b


def test_build_procedural_pool_names_do_not_collide_with_cad_names():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    pool = build_procedural_pool(3, sample_tray, Config().tray_anchored,
                                 seed=0, name_prefix="anchored")
    assert all(name.startswith("anchored_") for name in pool)


def test_exclude_assets_drops_only_the_named_keys_without_mutating_input():
    from recog.synth3d.catalog import exclude_assets
    assets = {"A": {}, "B": {}, "C": {}}
    out = exclude_assets(assets, ["B"])
    assert set(out) == {"A", "C"}
    assert set(assets) == {"A", "B", "C"}


def test_exclude_assets_raises_on_an_unknown_name():
    from recog.synth3d.catalog import exclude_assets
    with pytest.raises(KeyError):
        exclude_assets({"A": {}}, ["nope"])


# ---------------------------------------- naming contract Task 8 relies on --

def test_procedural_object_names_classify_correctly_via_role_of():
    """world.build_procedural_tray (bpy-only, no direct pytest coverage)
    names its objects to satisfy CLASS_RULES' EXISTING regexes, so
    _load_template's shared role-tagging tail needs no procedural-aware
    branch. A silent misclassification here would tag the lid as `case`
    too (both fall to ROLE_FALLBACK if the name doesn't match "_top"),
    rendering every open procedural cartridge CLOSED - the exact
    pre-tray-interior-fix defect, reincarnated."""
    from recog.synth3d.catalog import role_of
    assert role_of("ProcCase_btm") == "case"
    assert role_of("ProcCase_top") == "case_lid"
    assert role_of("ProcCell_0") == "cell"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bay.py -v -k "build_tray_entry or build_procedural_pool or exclude_assets or naming"`
Expected: FAIL — `ImportError: cannot import name 'build_tray_entry'`
(`test_procedural_object_names_classify_correctly_via_role_of` PASSES
already, since it only exercises the existing `role_of`/`CLASS_RULES` —
that is deliberate: it is a contract test, pinning behaviour Task 8 must
match, not new behaviour of this task).

- [ ] **Step 3: Add `build_tray_entry`, `build_procedural_pool`, `exclude_assets`**

`catalog.py` currently imports only `from typing import List`; change it
to `from typing import Dict, List` (the three new functions below all
return `Dict[str, dict]` or take one — `from __future__ import
annotations` at the top of this file means this isn't a runtime
requirement, but every existing annotation in this module names an
imported type, and these should too).

Append to `recog/synth3d/catalog.py`:

```python
def build_tray_entry(sample) -> dict:
    """Forward-construct a catalog-entry-shaped dict from a
    `bay.TraySample` - the mirror image of `inspect_glb`: chooses
    `interior_mm`/`module_bay_mm`/`tray_floor_mm` directly instead of
    measuring them from a mesh (design spec Sec3.3). Never written to
    `catalog.json` - a caller (AssetLibrary.register_procedural_pool,
    Task 9) registers the returned dict straight into
    `AssetLibrary.assets`.

    `bay.bay_edge` is called here as a LOUD post-condition, not trusted
    silently: a malformed TraySample raises ValueError at THIS point
    (registration time), never inside scene.py's per-scene build loop,
    where a missing/malformed field would instead fall through the
    existing `entry.get(...)` guards into the pre-tray-interior-fix
    fallback (hi.z-anchored labels) with no error at all - correct
    behaviour for a CAD entry that genuinely has no measurement, wrong
    for a procedural entry that is fully computed and has no such
    excuse.
    """
    from .bay import bay_edge

    entry = {
        "kind": "procedural",
        "interior_mm": [round(v, 2) for v in sample.interior_mm],
        "module_bay_mm": [round(v, 2) for v in sample.module_bay_mm],
        "tray_floor_mm": round(sample.tray_floor_mm, 2),
        "case_outer_mm": [round(v, 2) for v in sample.case_outer_mm],
        "case_half_height_mm": round(sample.case_half_height_mm, 2),
        "case_wall_mm": round(sample.wall_mm, 2),
        "cell_format": sample.cell_format,
    }
    bay_edge(tuple(entry["interior_mm"]), tuple(entry["module_bay_mm"]))
    return entry


def build_procedural_pool(n: int, sample_fn, cfg, seed: int,
                          name_prefix: str = "proc") -> Dict[str, dict]:
    """`n` independently-sampled procedural tray entries, keyed by a
    unique name (`f"{name_prefix}_{i:04d}"`). `sample_fn` is
    `bay.sample_tray`; passed in rather than imported directly so this
    function has no import-time dependency on `bay.py` beyond what
    `build_tray_entry` already needs.

    Same per-index RNG recipe as `scene.scene_generator`
    (`(seed * 1_000_003) ^ (i + 1)`), so a pool built with the same
    `(n, cfg, seed)` is byte-for-byte reproducible.
    """
    import random as _random

    pool: Dict[str, dict] = {}
    for i in range(n):
        rng = _random.Random((seed * 1_000_003) ^ (i + 1))
        entry = build_tray_entry(sample_fn(cfg, rng))
        pool[f"{name_prefix}_{i:04d}"] = entry
    return pool


def exclude_assets(assets: Dict[str, dict], names) -> Dict[str, dict]:
    """A copy of `assets` with every name in `names` removed. Raises
    KeyError naming the missing key if any `names` entry is not present -
    a typo'd `--exclude-asset` must fail loudly, not silently no-op and
    leave the "excluded" SKU in the training set (design spec Sec10's
    leave-one-SKU-out control depends on this actually excluding what it
    says it excludes).
    """
    out = dict(assets)
    for name in names:
        if name not in out:
            raise KeyError(
                f"exclude_assets: {name!r} is not a known asset "
                f"({sorted(assets)})")
        del out[name]
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_bay.py -v -k "build_tray_entry or build_procedural_pool or exclude_assets or naming"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recog/synth3d/catalog.py tests/test_bay.py
git commit -m "feat(synth3d): forward-construct procedural catalog entries

build_tray_entry mirrors inspect_glb in reverse: chooses interior_mm/
module_bay_mm/tray_floor_mm from a TraySample instead of measuring them,
validated loudly via bay.bay_edge rather than silently trusted.
build_procedural_pool batches N of them; exclude_assets is the small
filter generate3d.py's --exclude-asset (Task 10) and the leave-one-SKU-out
CAD-control folds (Task 17) both use."
```

---

### Task 8: `world.py` — the bpy builder

**Files:**
- Modify: `recog/synth3d/world.py`

**Interfaces:**
- Consumes: a `build_tray_entry`-shaped dict (Task 7):
  `case_outer_mm`, `case_wall_mm`, `case_half_height_mm`,
  `tray_floor_mm`, `cell_format`; `config.CELL_FORMATS` (Task 1).
- Produces: `build_procedural_tray(entry: dict) -> Dict[str, list]`,
  shaped exactly like `_load_template`'s existing CAD return value
  (`{"case": [obj], "case_lid": [obj], "cell": [obj]}`). Consumed by Task
  9's `AssetLibrary._load_template` branch.

**Discipline note carried forward from the design spec (§3.4's closing
paragraph), not acted on in this task because nothing here needs it**: if
this builder ever grows cosmetic detail (a chamfer, a draft angle, a
corner fillet — spec #3's realism work, not this plan's), that shaping
choice has to be a parameter computed in `bay.sample_tray`/
`catalog.build_tray_entry`, never a number chosen by eyeballing inside
this function — the same rule `build_jig`'s plain rectangular pockets
already follow. `build_procedural_tray` below adds nothing beyond plain
boxes and one cylinder, so this is not yet a live constraint; it is
recorded here so it is not quietly violated later.

No pytest possible (`world.py` imports bpy). One piece of this task **is**
pytest-covered: Task 7's `test_procedural_object_names_classify_correctly_via_role_of`
already pins the naming contract this task must follow — `role_of`
regex-searches an object's *name*, so the objects below are named
`ProcCase_btm` / `ProcCase_top` / `ProcCell_0` specifically so
`_load_template`'s existing, unmodified role-tagging loop classifies them
correctly with **zero** procedural-aware code in that loop.

- [ ] **Step 1: Add the builder**

Append to `recog/synth3d/world.py` (after `build_jig`, before `build_pcb`):

```python
def build_procedural_tray(entry: dict) -> Dict[str, list]:
    """Bare boolean-cut geometry for one procedural tray: `case` (a
    boolean-differenced shell), `case_lid` (a solid slab, same
    footprint), and `cell` (one cylinder at the sampled format's
    radius/length) - structurally identical to build_jig: every number
    comes from `entry` (bay.sample_tray + catalog.build_tray_entry). No
    geometric judgement happens here (design spec Sec3.4).

    No lay_flat, no flip_if_inverted: those exist ONLY to correct an
    imported CAD file's own up-axis/orientation ambiguity (design spec
    Sec2). A primitive built directly in Blender has neither to correct -
    `_load_template`'s shared re-centre step (unchanged, run on whatever
    this function returns) is what establishes the (0,0)-centred local
    frame bay.py's module docstring requires, exactly as it already does
    for CAD.

    Object names (`ProcCase_btm`/`ProcCase_top`/`ProcCell_0`) are chosen
    to satisfy catalog.CLASS_RULES' existing regexes - see
    tests/test_bay.py's `test_procedural_object_names_classify_correctly_
    via_role_of` (Task 7) for the pinned contract. Get these names wrong
    and `_load_template`'s shared tail silently tags the lid as `case`
    too, re-closing every open procedural cartridge.

    Returns a `by_role` dict shaped exactly like `_load_template`'s
    existing CAD return value.
    """
    x0, y0, x1, y1 = entry["case_outer_mm"]
    w_m, h_m = (x1 - x0) / 1000.0, (y1 - y0) / 1000.0
    cx_m, cy_m = (x0 + x1) / 2000.0, (y0 + y1) / 2000.0
    half_h = entry["case_half_height_mm"] / 1000.0
    wall = entry["case_wall_mm"] / 1000.0
    floor = entry["tray_floor_mm"] / 1000.0

    # --- case: solid block, then a cavity boolean-differenced out of it -
    # exactly build_jig's own cube-plus-cutter shape.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx_m, cy_m, half_h / 2))
    case = bpy.context.active_object
    case.name = "ProcCase_btm"
    case.scale = (w_m, h_m, half_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    cav_w, cav_h = w_m - 2 * wall, h_m - 2 * wall
    cav_z_h = (half_h - floor) + 0.001     # +1mm so the cutter clears the open top
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(cx_m, cy_m, floor + cav_z_h / 2))
    cutter = bpy.context.active_object
    cutter.name = "_proc_cavity_cutter"
    cutter.scale = (cav_w, cav_h, cav_z_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mod = case.modifiers.new(name="cavity", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    bpy.context.view_layer.objects.active = case
    bpy.ops.object.modifier_apply(modifier="cavity")
    bpy.data.objects.remove(cutter, do_unlink=True)

    # --- case_lid: a solid slab, no boolean op needed - resting exactly
    # on the shell's own top rim (design spec Sec4.4's two-piece split).
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(cx_m, cy_m, half_h + half_h / 2))
    lid = bpy.context.active_object
    lid.name = "ProcCase_top"
    lid.scale = (w_m, h_m, half_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # --- cell: one cylinder template at the sampled format's own
    # radius/length, resting ON THE CAVITY FLOOR - not parked outside the
    # case/lid footprint. _load_template's shared re-centre step measures
    # group_bbox(meshes) over every object this function returns,
    # INCLUDING this one; a cell parked outside the case/lid's own
    # footprint or height would silently skew that measurement and
    # mis-centre case and case_lid too, with no error anywhere short of
    # Task 9's dedicated assertion.
    diam_m, len_m = CELL_FORMATS[entry["cell_format"]]
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diam_m / 2, depth=len_m,
        location=(cx_m, cy_m, floor + diam_m / 2))
    cell = bpy.context.active_object
    cell.name = "ProcCell_0"
    # primitive_cylinder_add's own axis is Z (standing on end); rotate 90
    # degrees about X so it rests on its SIDE, long axis horizontal - the
    # same resting pose lay_flat gives every imported CAD cell.
    cell.rotation_euler = (math.radians(90), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bpy.context.view_layer.update()
    return {"case": [case], "case_lid": [lid], "cell": [cell]}
```

Add `CELL_FORMATS` to the existing `from .config import CELL_H_MM,
CELL_W_MM` import line.

- [ ] **Step 2: Run the one pytest-able piece**

Run: `python -m pytest tests/test_bay.py -v -k naming`
Expected: PASS (this passed before this task too — it is a contract test
against `catalog.role_of`, not against `world.py`, and confirms the
literal names typed into Step 1 above are the ones the test pins).

- [ ] **Step 3: Commit**

```bash
git add recog/synth3d/world.py
git commit -m "feat(synth3d): build_procedural_tray - the bpy builder for a procedural asset

Boolean-cut case + solid lid + one cell cylinder, structurally identical
to build_jig: every number comes from catalog.build_tray_entry, no
geometric judgement happens here. No lay_flat/flip_if_inverted - those
only correct import-specific surprises this builder never has. Object
names satisfy catalog.CLASS_RULES's existing regexes so no new role-
detection code is needed downstream."
```

(Not independently render-verified yet — Task 9 wires it into
`AssetLibrary`, and Task 11 is the first actual render.)

---

### Task 9: `assets.py` — the merged namespace

**Files:**
- Modify: `recog/synth3d/assets.py`

**Interfaces:**
- Consumes: `world.build_procedural_tray` (Task 8, imported lazily — see
  below), `entry["kind"] == "procedural"` (Task 7).
- Produces: `AssetLibrary.register_procedural_pool(self, pool: Dict[str,
  dict]) -> None`; one new branch inside `_load_template`. Every other
  `AssetLibrary` method (`instantiate`, `catalog_entry`, `names`) is
  **unchanged** — design spec §4.2's "one branch point."

No pytest possible (bpy). Verified by Task 11's dev render, which
exercises `assembled`/`open_case`/`cells_only` on a registered procedural
asset and inspects the contact sheet by eye.

- [ ] **Step 1: Add `register_procedural_pool`**

In `AssetLibrary`, alongside `catalog_entry`:

```python
    def register_procedural_pool(self, pool: Dict[str, dict]) -> None:
        """Merge `pool` (catalog.build_procedural_pool's output) into
        `self.assets` - design spec Sec4.2's "merged namespace, one
        branch point": every entry already carries `"kind": "procedural"`,
        and `_load_template`'s one new branch (below) is what makes
        `instantiate`/`catalog_entry`/every bay.py consumer treat it
        identically to a CAD entry from here on.

        Raises ValueError on any name collision with an existing entry
        (CAD or a previously registered pool) - a silent overwrite would
        drop whichever asset lost the collision with no error, which is
        exactly the kind of silent degradation this plan has to guard
        against.
        """
        collisions = set(pool) & set(self.assets)
        if collisions:
            raise ValueError(
                f"register_procedural_pool: name(s) already registered: "
                f"{sorted(collisions)}")
        self.assets.update(pool)
```

- [ ] **Step 2: Branch `_load_template`**

Replace the **whole method body** with the version below — this shows
the complete post-edit function, not a partial diff, because the change
touches its start (the new branch), its middle (a new loud assertion) and
its end (Task 2's `CELL_FORMAT_PROP` tagging, already landed, shown here
in its final position so the two tasks' edits are never applied
out of order):

```python
    def _load_template(self, name: str) -> Dict[str, list]:
        if name in self._templates:
            return self._templates[name]

        entry = self.assets[name]
        cell_format = entry.get("cell_format", "18650")

        if entry.get("kind") == "procedural":
            # A procedural tray is GENERATED, not imported - it has no
            # external CAD up-axis convention to invert, no glTF
            # multi-material tessellation to re-split, no case/liner name
            # collision to disambiguate (design spec Sec2, Sec3.4). The
            # import + lay_flat + flip_if_inverted prefix below exists
            # ONLY to correct those import-specific surprises, so it is
            # skipped entirely; everything from the re-centre step
            # onward is shared with the CAD path, unmodified.
            from . import world as W   # local import: world.py imports
                                        # THIS module at its own top level
                                        # (`from . import assets as A`), so
                                        # a module-level import here would
                                        # be circular.
            meshes = [o for lst in W.build_procedural_tray(entry).values()
                     for o in lst]
        else:
            path = os.path.join(self.dir, entry["file"])
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=path)
            new = [o for o in bpy.data.objects if o not in before]
            meshes = [o for o in new if o.type == "MESH"]

            meshes = _split_multi_material_case(meshes)

            for o in meshes:
                if o.parent:
                    world_mat = o.matrix_world.copy()
                    o.parent = None
                    o.matrix_world = world_mat
            for o in new:
                if o.type != "MESH":
                    bpy.data.objects.remove(o, do_unlink=True)

            bpy.context.view_layer.update()
            lay_flat(meshes)
            flip_if_inverted(meshes)

        # --- shared tail: re-centre, role bookkeeping, loud post-
        # conditions - IDENTICAL for either branch above.
        lo, hi = group_bbox(meshes)
        centre = (lo + hi) / 2
        for o in meshes:
            o.location -= Vector((centre.x, centre.y, lo.z))
        bpy.context.view_layer.update()

        if entry.get("kind") == "procedural":
            # Loud, not a log line: a cell parked outside the case/lid
            # group (world.build_procedural_tray's own risk, see its
            # docstring) would silently skew the recentre offset above
            # and mis-place case/case_lid relative to the interior_mm/
            # module_bay_mm rects bay.py already computed for THIS exact
            # entry - corrupting every label with no error anywhere else.
            case_objs = [o for o in meshes if role_of(o.name) == "case"]
            if case_objs:
                clo, chi = group_bbox(case_objs)
                ex0, ey0, ex1, ey1 = entry["case_outer_mm"]
                ew, eh = (ex1 - ex0) / 1000.0, (ey1 - ey0) / 1000.0
                gw, gh = chi.x - clo.x, chi.y - clo.y
                if abs(gw - ew) > 1e-4 or abs(gh - eh) > 1e-4:
                    raise RuntimeError(
                        f"{name}: built case measures "
                        f"{gw * 1000:.2f}x{gh * 1000:.2f}mm after "
                        f"re-centring, not the {ew * 1000:.2f}x"
                        f"{eh * 1000:.2f}mm case_outer_mm the entry asked "
                        f"for - build_procedural_tray and "
                        f"catalog.build_tray_entry have desynced (a "
                        f"mis-placed cell template is the likely cause; "
                        f"see build_procedural_tray's own docstring)")

        coll = _template_collection()
        by_role: Dict[str, list] = {}
        for o in meshes:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
            o.hide_render = True
            by_role.setdefault(role_of(o.name), []).append(o)

        _classify_case_liner(by_role)

        case_objs = by_role.get("case") or []
        lid_objs = by_role.get("case_lid") or []
        if case_objs and lid_objs:
            case_lo, case_hi = group_bbox(case_objs)
            lid_lo, lid_hi = group_bbox(lid_objs)
            if lid_lo.z < case_hi.z - 1e-4:
                raise RuntimeError(
                    f"{name}: the lid (case_lid, z=[{lid_lo.z * 1000:.3f},"
                    f"{lid_hi.z * 1000:.3f}]mm) does not sit at/above the "
                    f"shell's own top (case, z=[{case_lo.z * 1000:.3f},"
                    f"{case_hi.z * 1000:.3f}]mm) - the assembly is upside "
                    f"down (or a procedural build put the lid below the "
                    f"shell). This is exactly the silent failure mode "
                    f"task-3c exists to close off, for BOTH CAD and "
                    f"procedural assets.")

        for role, objs in by_role.items():
            for o in objs:
                o[ROLE_PROP] = role
                if role == "cell":
                    o[CELL_FORMAT_PROP] = cell_format

        bpy.context.view_layer.update()
        self._templates[name] = by_role
        return by_role
```

Note the existing lid-above-case `RuntimeError` check (previously CAD-only
in practice, since it was the only path) now applies **unmodified** to
procedural assets too, for free — it was already generic over `by_role`,
never CAD-specific in its own logic. This is a second, independent guard
against the same class of "silently inverted assembly" defect this
project has already hit once (task-3c).

- [ ] **Step 2: Verify existing CAD behaviour is unchanged**

```bash
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/synth3d/_gate_orientation.py
```

Expected: `ORIENTATION GATE OK` — the CAD branch's code is textually
identical to before this task, only re-indented under an `else:`.

- [ ] **Step 3: Commit**

```bash
git add recog/synth3d/assets.py
git commit -m "feat(synth3d): merge procedural entries into AssetLibrary, one branch point

_load_template gets exactly one new branch: a 'kind: procedural' entry
calls world.build_procedural_tray instead of import_scene.gltf; every
downstream consumer (instantiate, catalog_entry, all of bay.py) is
unchanged, because none of it was ever aware _load_template imports a
file. Adds a loud RuntimeError if a procedural build's own case geometry
doesn't match the entry it was built from - a silent desync here would
corrupt every downstream label with no other symptom."
```

---

### Task 10: `generate3d.py` — CLI to select which population renders

**Files:**
- Modify: `recog/generate3d.py`

**Interfaces:**
- Consumes: `catalog.build_procedural_pool`, `catalog.exclude_assets`,
  `bay.sample_tray`, `cfg.tray_anchored`/`cfg.tray_wide`,
  `AssetLibrary.register_procedural_pool` (Tasks 6, 7, 9).
- Produces: `--tray-set {cad,anchored,wide}` (default `cad`, preserving
  today's behaviour exactly), `--n-procedural INT` (defaults to `--n`),
  `--exclude-asset NAME` (repeatable). Consumed operationally by Tasks
  14–17 (the render commands).

No pytest possible (bpy script). Verified by Task 11's dev renders.

- [ ] **Step 1: Add the CLI flags**

In `parse_args`, alongside the existing `--variant`:

```python
    p.add_argument("--tray-set", choices=["cad", "anchored", "wide"],
                   default="cad",
                   help="cad (default): today's four Anker assemblies, "
                        "unchanged. anchored/wide: replace the library "
                        "with an in-memory procedural pool (Decision 1: "
                        "CAD is never mixed into a procedural render).")
    p.add_argument("--n-procedural", type=int, default=None,
                   help="size of the procedural pool for --tray-set "
                        "anchored/wide; defaults to --n.")
    p.add_argument("--exclude-asset", action="append", default=[],
                   help="drop a cataloged asset by name before "
                        "generation; repeatable. Used for the "
                        "leave-one-SKU-out CAD-control folds (design "
                        "spec Sec10).")
```

- [ ] **Step 2: Wire it in after `library = AssetLibrary(assets_dir)`**

```python
    library = AssetLibrary(assets_dir)
    print(f"assets: {library.names()}")

    if a.tray_set != "cad":
        from recog.synth3d.bay import sample_tray
        from recog.synth3d.catalog import build_procedural_pool

        # Decision 1: the model never sees real measured geometry during
        # training - .clear() before registering the pool, not merely
        # ADDING it, so a procedural render can never accidentally
        # sample a CAD name too.
        library.assets.clear()
        range_cfg = cfg.tray_anchored if a.tray_set == "anchored" else cfg.tray_wide
        pool = build_procedural_pool(
            a.n_procedural or a.n, sample_tray, range_cfg, seed=a.seed,
            name_prefix=a.tray_set)
        library.register_procedural_pool(pool)
        print(f"[tray-set={a.tray_set}] registered {len(pool)} procedural "
              f"assets, CAD assets cleared")

    if a.exclude_asset:
        from recog.synth3d.catalog import exclude_assets
        library.assets = exclude_assets(library.assets, a.exclude_asset)
        print(f"[exclude-asset] dropped {a.exclude_asset}; "
              f"{len(library.assets)} asset(s) remain: "
              f"{sorted(library.assets)}")
```

- [ ] **Step 3: Run the one pytest-able piece indirectly**

This task has no code of its own outside a bpy script; its correctness is
`catalog.exclude_assets`/`catalog.build_procedural_pool`'s own tests
(already passing from Task 7) plus Task 11's render.

Run: `python -m pytest tests/test_bay.py -v -k "exclude_assets or build_procedural_pool"`
Expected: PASS (already passing — confirms nothing in this task broke the
functions it calls).

- [ ] **Step 4: Commit**

```bash
git add recog/generate3d.py
git commit -m "feat(recog): --tray-set / --exclude-asset on generate3d.py

--tray-set anchored/wide clears the CAD names and registers an in-memory
procedural pool instead (Decision 1: never mixed). --exclude-asset drops
a named CAD asset before generation, for the leave-one-SKU-out
CAD-control folds (Task 17). Default --tray-set cad reproduces today's
behaviour exactly - no existing invocation changes."
```

---

### Task 11: first procedural render — regression re-check

**Files:** none (render + verification only)

**Interfaces:**
- Consumes: Tasks 1–10 in full.
- Produces: confidence the procedural path is a data-quality-safe
  training input, before any multi-hour render is spent on it.

Design spec §11: "Procedural masks are... expected to carry over
structurally, but it is a **data-quality gate to re-measure**, not an
assumption to inherit silently." This task is that re-measurement, at a
small, fast scale, **before** Group D's real renders.

- [ ] **Step 1: Render a small anchored smoke set**

```bash
python -m recog.sync_config
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/generate3d.py -- --n 32 --out recog/dev3d_anchored \
    --device GPU --tray-set anchored --save-masks
python -m recog.verify3d --data recog/dev3d_anchored --n 32 --masks
```

- [ ] **Step 2: Look at the contact sheet**

Open `recog/dev3d_anchored/contact_sheet.png`. Confirm, the same way
`docs/superpowers/plans/2026-08-09-tray-interior.md` Task 3 required for
the CAD tray fix:

- [ ] An `open_case` procedural cartridge reads as a recessed tray —
  visible walls, visible depth, self-shadowing under directional lighting
- [ ] The electronics module sits inside the cavity at one end of the
  tray's longer axis (anchored's short-edge rule)
- [ ] `assembled` procedural cartridges render **sealed** — case and lid
  both visible, no cavity showing
- [ ] `cells_only` procedural cells render as loose cylinders at the
  sampled format's own proportions
- [ ] At least one 21700 or 26650 cell appears among the render (check
  `manifest.json`'s per-variant stats, or eyeball a visibly-thicker
  cylinder)

If the cartridge reads as a flat closed box, `open_case`'s `keep_roles`
did not exclude `case_lid` — check the object names Task 8 built actually
satisfy Task 7's naming-contract test in the running Blender process (not
just in the pytest environment).

- [ ] **Step 3: Re-run the five-class disjointness sweep**

```bash
python -c "
import json, numpy as np
from recog.synth3d.annotate import rle_decode

with open('recog/dev3d_anchored/instances_seg.json') as f:
    doc = json.load(f)

by_image = {}
for a in doc['annotations']:
    by_image.setdefault(a['image_id'], []).append(a)

names = {c['id']: c['name'] for c in doc['categories']}
pairs = [('placement_area', 'battery'), ('placement_area', 'obstruction'),
         ('placement_area', 'electronics_module'),
         ('battery', 'obstruction'), ('battery', 'electronics_module'),
         ('obstruction', 'electronics_module')]

n_pairs = n_overlap = 0
for img_id, anns in by_image.items():
    by_cls = {}
    for a in anns:
        by_cls.setdefault(names[a['category_id']], []).append(a)
    for c1, c2 in pairs:
        for a1 in by_cls.get(c1, []):
            m1 = rle_decode(a1['segmentation']).astype(bool)
            for a2 in by_cls.get(c2, []):
                m2 = rle_decode(a2['segmentation']).astype(bool)
                n_pairs += 1
                if (m1 & m2).any():
                    n_overlap += 1
                    print(f'OVERLAP image={img_id} {c1}#{a1[\"id\"]} vs {c2}#{a2[\"id\"]}')

print(f'{n_pairs} mask pairs checked, {n_overlap} overlapping')
"
```

Expected: `0 overlapping` — the SAME occlusion-by-geometry mechanism
(`build_bay_proxy`/`seat_cells`/`build_obstructions` on the same
`floor_z`, unchanged by this plan) that guarantees this for CAD scenes.
**If this is not 0, stop — do not proceed to Group D's multi-hour renders
until it is.** The most likely cause is a `floor_z`/offset mismatch
between `build_procedural_tray`'s cavity geometry and the fixed
sub-millimetre lift offsets `build_bay_proxy`/`build_obstructions`/
`seat_cells` already use.

- [ ] **Step 4: Repeat for `--tray-set wide`**

```bash
"$BL" -b --python recog/generate3d.py -- --n 32 --out recog/dev3d_wide \
    --device GPU --tray-set wide --save-masks
python -m recog.verify3d --data recog/dev3d_wide --n 32 --masks
```

Repeat Steps 2–3 against `recog/dev3d_wide`. Wide trays are expected to
look unrealistic (that is the point) but must pass every disjointness and
`assembled`-sealing check the same way anchored does — Decision 2's "wide"
is about variation *within* the same correctness guarantees, not a looser
bar.

- [ ] **Step 5: Check the anchor-scale/aspect calibration (design spec §6.4)**

Read the "Anchor check" block `generate3d.py` prints at the end of both
runs above. If `p05`/`p95` (in `box_sqrt_area_px`) falls outside the
`floor`–`ceil` band it reports, note it — this is the "required
verification step for the implementation" the design spec names but does
not perform; `configs/recognition.yaml`'s `anchor_scales` or
`configs/synth3d.yaml`'s `filter.max_aspect` may need retuning before
Group D's real training renders (not before this smoke test).

- [ ] **Step 6: Delete the smoke-test renders**

```bash
rm -rf recog/dev3d_anchored recog/dev3d_wide
```

They are gitignored dev output, not the training data — Group D renders
fresh, full-scale sets.

No commit — this task changes no tracked files. Record the outcome
(pass/fail on each checklist item, and any anchor-check note) in the
final report handed back after this plan is executed.

---

# Group C — Per-SKU evaluability

### Task 12: carry `asset` through the COCO sidecar

**Files:**
- Modify: `recog/synth3d/annotate.py`
- Modify: `recog/seg_dataset.py`
- Test: `tests/test_annotate_masks.py` (append), `tests/test_seg_dataset.py` (append)

**Interfaces:**
- Consumes: `masks_from_index`'s existing per-annotation `"asset"` key
  (already present in-memory, `recog/synth3d/annotate.py:503`; only
  dropped at the `write_coco_json` serialisation boundary).
- Produces: every COCO sidecar annotation now carries `"asset"`;
  `BaySegDataset.sample_assets: List[Optional[str]]`, index-parallel to
  `self.samples`. Consumed by Task 13.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_annotate_masks.py`, right after
`test_write_coco_json_round_trips_unit_id`:

```python
def test_write_coco_json_round_trips_asset(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "AnkerPowerCore10000",
                "variant": "v", "unit_id": "item3"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    out = tmp_path / "instances_seg.json"
    write_coco_json(
        str(out),
        images=[{"id": 0, "file_name": "x.png", "width": 20, "height": 20}],
        annotations=[dict(a, image_id=0, id=1) for a in anns],
        seg_class_ids=SEG_IDS,
    )
    doc = json.loads(out.read_text())
    assert doc["annotations"][0]["asset"] == "AnkerPowerCore10000"
```

Append to `tests/test_seg_dataset.py`:

```python
def test_dataset_tracks_which_asset_each_crops_unit_belongs_to(tmp_path):
    """design spec Sec7/Sec10/Sec12 need per-SKU, per-class numbers on
    the CAD test set - this is what makes a crop's own SKU queryable
    without re-deriving it from the raw annotations at report time."""
    from recog.seg_dataset import BaySegDataset

    H = W = 40
    cart = np.zeros((H, W), np.uint8)
    cart[5:35, 5:35] = 1
    coco = {
        "categories": [{"id": 2, "name": "cartridge"}],
        "images": [{"id": 0, "file_name": "x.png", "width": W, "height": H}],
        "annotations": [{
            "id": 1, "image_id": 0, "category_id": 2,
            "bbox": [5, 5, 30, 30], "area": 900,
            "segmentation": rle_encode(cart), "iscrowd": 0,
            "unit_id": "item0", "asset": "AnkerPowerCore10000",
        }],
    }
    coco_path = tmp_path / "instances_seg.json"
    coco_path.write_text(json.dumps(coco))

    ds = BaySegDataset(str(coco_path), str(tmp_path))
    assert ds.sample_assets == ["AnkerPowerCore10000"]


def test_dataset_asset_is_none_when_absent():
    """A hand-built fixture that never sets `asset` (every other test in
    this file) must not raise - .get() the same way unit_id already
    does."""
    from recog.seg_dataset import BaySegDataset

    H = W = 40
    cart = np.zeros((H, W), np.uint8)
    cart[5:35, 5:35] = 1
    coco = {
        "categories": [{"id": 2, "name": "cartridge"}],
        "images": [{"id": 0, "file_name": "x.png", "width": W, "height": H}],
        "annotations": [{
            "id": 1, "image_id": 0, "category_id": 2,
            "bbox": [5, 5, 30, 30], "area": 900,
            "segmentation": rle_encode(cart), "iscrowd": 0,
        }],
    }
    coco_path = tmp_path / "instances_seg.json"
    coco_path.write_text(json.dumps(coco))

    ds = BaySegDataset(str(coco_path), str(tmp_path))
    assert ds.sample_assets == [None]
```

Add `import json` and `from recog.synth3d.annotate import rle_encode` to
`tests/test_seg_dataset.py` if not already present at the top (`rle_encode`
is already imported there for other fixtures — check before adding a
duplicate).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_annotate_masks.py tests/test_seg_dataset.py -v -k asset`
Expected: FAIL — `KeyError: 'asset'` / `AttributeError: 'BaySegDataset'
object has no attribute 'sample_assets'`.

- [ ] **Step 3: Carry `asset` through `write_coco_json`**

In `recog/synth3d/annotate.py`'s `write_coco_json`, the per-annotation
dict:

```python
        "annotations": [
            {"id": a["id"], "image_id": a["image_id"],
             "category_id": a["category_id"],
             "bbox": a["bbox_xywh"], "area": a["area"],
             "segmentation": a["segmentation"], "iscrowd": a.get("iscrowd", 0),
             "unit_id": a.get("unit_id"), "asset": a.get("asset")}
            for a in annotations
        ],
```

(one key added: `"asset": a.get("asset")`, immediately after `unit_id`,
same `.get()` pattern for the same reason — a caller that builds
annotations by hand should get `None`, not a `KeyError`.)

- [ ] **Step 4: Track it in `BaySegDataset`**

In `recog/seg_dataset.py`'s `BaySegDataset.__init__`, alongside
`self.samples: List[...] = []`:

```python
        self.samples: List[Tuple[dict, List[dict], Tuple[int, int, int, int]]] = []
        self.sample_assets: List[Optional[str]] = []
```

In the `for uid, unit in by_unit.items():` loop, right before
`self.samples.append((images[img_id], anns, box))`:

```python
                self.sample_assets.append(unit[0].get("asset"))
```

(`unit[0]` is that unit's own first annotation — every annotation sharing
one `unit_id` shares one `asset`, per `scene.build`'s own construction, so
the first is representative.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_annotate_masks.py tests/test_seg_dataset.py -v -k asset`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: 621 + new tests passed, 0 failed — this task touches a
widely-used serialisation function; a shape mismatch anywhere else would
show up here.

- [ ] **Step 7: Commit**

```bash
git add recog/synth3d/annotate.py recog/seg_dataset.py tests/test_annotate_masks.py tests/test_seg_dataset.py
git commit -m "feat(recog): carry asset (SKU) through the COCO sidecar and BaySegDataset

masks_from_index already computed 'asset' per annotation; write_coco_json
was the one place it got dropped before reaching disk. BaySegDataset.
sample_assets is index-parallel to .samples, giving seg_evaluate (Task 13)
a crop's own SKU without re-deriving it from raw annotations."
```

---

### Task 13: `seg_evaluate.py` — per-SKU grouping and report

**Files:**
- Modify: `recog/seg_evaluate.py`
- Test: `tests/test_bay_segmenter.py` (append — the existing home for
  `seg_evaluate` unit tests)

**Interfaces:**
- Consumes: `BaySegDataset.sample_assets` (Task 12), the existing
  `evaluate(segmenter, full_dataset, val_indices, mm_per_px, num_classes)`
  (unchanged — called once per SKU with a filtered `val_indices`).
- Produces: `group_indices_by_asset(full_dataset, val_indices) ->
  Dict[Optional[str], List[int]]`; `format_per_sku_table(per_sku_results:
  Dict[str, dict]) -> str`; `--per-sku` CLI flag. Consumed operationally
  by Task 19.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay_segmenter.py`:

```python
def test_group_indices_by_asset_partitions_by_sku():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A", None]

    out = group_indices_by_asset(_FakeDataset(), [0, 1, 2, 3])
    assert out == {"A": [0, 2], "B": [1], None: [3]}


def test_group_indices_by_asset_only_includes_requested_indices():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A"]

    out = group_indices_by_asset(_FakeDataset(), [0, 1])
    assert out == {"A": [0], "B": [1]}


def test_format_per_sku_table_lists_every_sku_with_its_crop_count():
    from recog.seg_evaluate import format_per_sku_table

    results = {
        "AnkerPowerCore10000": {"n_val_crops": 12,
                                "ious": {"bay": 0.80, "obstruction": 0.60}},
        "AnkerPowerCore13000": {"n_val_crops": 9,
                                "ious": {"bay": 0.75, "obstruction": 0.55}},
    }
    table = format_per_sku_table(results)
    assert "AnkerPowerCore10000" in table and "12" in table
    assert "AnkerPowerCore13000" in table and "9" in table
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bay_segmenter.py -v -k "per_sku or group_indices"`
Expected: FAIL — `ImportError: cannot import name 'group_indices_by_asset'`.

- [ ] **Step 3: Add the two functions**

In `recog/seg_evaluate.py`, near `compute_val_instance_counts`:

```python
def group_indices_by_asset(full_dataset, val_indices: Sequence[int]
                           ) -> Dict[Optional[str], List[int]]:
    """`val_indices` partitioned by which catalog asset (SKU) each crop's
    unit belongs to, via `BaySegDataset.sample_assets` (Task 12).

    Design spec Sec7/Sec10/Sec12: every comparison this measurement makes
    has to be reported per-SKU first, pooled figure alongside it, never
    in place of it - this is what makes that grouping possible by
    calling the EXISTING `evaluate()` once per group, rather than
    threading a new SKU-aware code path through it.
    """
    out: Dict[Optional[str], List[int]] = {}
    for idx in val_indices:
        out.setdefault(full_dataset.sample_assets[idx], []).append(idx)
    return out


def format_per_sku_table(per_sku_results: Dict[str, Dict[str, Any]]) -> str:
    """A compact per-SKU IoU table, one row per asset, for the classes in
    SELECT_ON plus 'battery' (design spec Sec12's regression floor names
    both explicitly). Appended to the same report `format_report`
    produces - not folded into it, to keep the well-tested pooled report
    untouched by this addition.
    """
    cols = SELECT_ON + ["battery"]
    lines = ["", "Per-SKU IoU (design spec Sec7/Sec10/Sec12):",
            "  " + "asset".ljust(24) + "n_crops".rjust(9)
            + "".join(c.rjust(14) for c in cols)]
    for asset, res in sorted(per_sku_results.items(),
                             key=lambda kv: (kv[0] is None, kv[0])):
        ious = res.get("ious", {})
        row = ("  " + str(asset).ljust(24)
              + str(res.get("n_val_crops", 0)).rjust(9)
              + "".join(f"{ious.get(c, float('nan')):.4f}".rjust(14)
                        for c in cols))
        lines.append(row)
    return "\n".join(lines)
```

`recog/seg_evaluate.py` already imports `Any, Dict, List, Optional,
Sequence, Tuple` from `typing` (line 30) — no import change needed.

- [ ] **Step 4: Wire the CLI flag**

In `build_arg_parser`:

```python
    ap.add_argument("--per-sku", action="store_true",
                    help="also report per-catalog-asset (SKU) IoU - "
                         "design spec Sec12. Only meaningful against a "
                         "dataset whose sidecar carries 'asset' per "
                         "annotation (Task 12).")
```

In `main`, after the existing `results = evaluate(...)` call (the exact
call producing the pooled report — confirm the local variable name
against the current file, since this task doesn't show the surrounding
lines):

```python
    if args.per_sku:
        by_asset = group_indices_by_asset(full_dataset, val_indices)
        per_sku_results = {
            asset: evaluate(segmenter, full_dataset, idxs, mm_per_px,
                            num_classes=num_classes)
            for asset, idxs in by_asset.items() if idxs
        }
        report += "\n" + format_per_sku_table(per_sku_results)
```

(`report` here stands for whatever variable `main` already builds from
`format_report(...)` before writing `args.out` — match the existing
variable name in place at edit time.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_bay_segmenter.py -v -k "per_sku or group_indices"`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: 621 + new tests passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add recog/seg_evaluate.py tests/test_bay_segmenter.py
git commit -m "feat(recog): --per-sku on seg_evaluate.py

group_indices_by_asset partitions val_indices by BaySegDataset.
sample_assets; the existing evaluate() is called once per group (no
change to evaluate() itself). format_per_sku_table renders a compact
per-SKU IoU table, appended to the existing report. Design spec Sec12
requires every comparison stated per-SKU first, pooled alongside it."
```

---

# Group D — Dataset generation and training (multi-hour; last)

**Everything below is an operational task, not a code task.** Each render
is on the order of 500 scenes at the existing 502-scene baseline's own
measured throughput; each training run is the existing full 40-epoch
schedule. Together these are **many hours of wall-clock time**, not
something to run inside one sitting — use `seg_training.py`'s `--resume`
and `generate3d.py`'s `--resume` exactly as
`docs/superpowers/plans/2026-08-09-tray-interior.md` Task 4 already
established, and run every long command in the **foreground**.

### Task 14: render the anchored procedural training set

```bash
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/generate3d.py -- --n 502 \
    --out recog/dataset3d_seg_anchored --device GPU \
    --tray-set anchored --resume
```

- [ ] Reaches `[done] ... wrote N segmentation annotations`. The Bash tool
  caps at 600000 ms; expect multiple `--resume` invocations, exactly as
  the existing 502-scene CAD set needed (`docs/NEXT_STEPS.md`, Step 4).
- [ ] Re-run the Task 11 Step 3 disjointness sweep against
  `recog/dataset3d_seg_anchored/instances_seg.json` — must stay at 0
  overlapping pairs at this full scale, not just the 32-scene smoke test.
- [ ] Read the printed "Anchor check" block; note whether
  `anchor_scales`/`max_aspect` need retuning (design spec §6.4) before
  Task 18 trains on this set. If retuning is needed, do it now (edit
  `configs/recognition.yaml`/`configs/synth3d.yaml`, `python -m
  recog.sync_config`) and re-render — training against uncalibrated
  anchors would confound the eventual per-class IoU comparison with a
  detector-config artefact.

No commit (dataset output is gitignored, matching every prior generation
task in this project).

---

### Task 15: render the wide procedural training set

```bash
"$BL" -b --python recog/generate3d.py -- --n 502 \
    --out recog/dataset3d_seg_wide --device GPU \
    --tray-set wide --resume
```

Same acceptance checklist as Task 14, against `recog/dataset3d_seg_wide`.

---

### Task 16: render the CAD-only held-out test set

Design spec §9.2: sized for a per-SKU, not pooled, reportable instance
density — roughly 4× the existing baseline's per-class density, i.e.
150–200 scenes, not 20–40.

```bash
"$BL" -b --python recog/generate3d.py -- --n 180 \
    --out recog/dataset3d_seg_cad_test --device GPU --resume
```

(`--tray-set` omitted — defaults to `cad`, i.e. the existing four Anker
assemblies, exactly as every prior render in this project.)

- [ ] Reaches `[done]`.
- [ ] Run the Task 11 Step 3 disjointness sweep against this set too —
  it is a **test** set, but its labels are still ground truth that every
  model's IoU is measured against; a disjointness bug here would corrupt
  every comparison Task 19 makes.
- [ ] Confirm per-SKU crop density: run

```bash
python -c "
import json
from recog.seg_dataset import BaySegDataset
ds = BaySegDataset('recog/dataset3d_seg_cad_test/instances_seg.json',
                   'recog/dataset3d_seg_cad_test/images', out_size=64, train=False)
from collections import Counter
print(dict(Counter(ds.sample_assets)))
"
```

  and confirm each of the four SKUs individually reaches roughly the
  ~24–36-instance-per-class density `docs/receipts/seg_eval.txt` already
  treats as reportable (design spec §9.2). If any SKU is thin, `--n`
  needs to grow and this task re-run before Task 19.

No commit.

---

### Task 17: render the four leave-one-SKU-out CAD-control training sets

Design spec §10: "a small filter... no new generator code" — this is
exactly `--exclude-asset` (Task 10), applied four times, once per held-out
SKU.

```bash
for SKU in AnkerPowerCore10000 AnkerPowerCore13000 AnkerPowerCore20100 AnkerPowerCore26800; do
  "$BL" -b --python recog/generate3d.py -- --n 502 \
      --out "recog/dataset3d_seg_cad_control_holdout_${SKU}" \
      --device GPU --exclude-asset "$SKU" --resume
done
```

- [ ] Each reaches `[done]`.
- [ ] Confirm the excluded SKU never appears: `grep -c '"asset": "SKU_NAME"'
  recog/dataset3d_seg_cad_control_holdout_<SKU>/instances_seg.json`
  reports `0` for its own held-out SKU.
- [ ] Run the disjointness sweep against each of the four.

This is the largest single render cost in this plan (4 × 502 scenes on
top of Tasks 14–16's 3 × ~500) — flagged explicitly, per this plan's
brief, as a multi-hour cost the implementer should budget for rather than
discover mid-run.

No commit.

---

### Task 18: train six models

New configs, one dataset each — copy `configs/segmentation.yaml` and edit
only `dataset.coco_path`/`dataset.img_dir`/`training.checkpoint_dir`
(everything else — epochs, augmentation, `select_on` — stays identical
across all six, so training conditions are comparable):

| Config | `dataset.coco_path` / `img_dir` | `checkpoint_dir` |
|---|---|---|
| `configs/segmentation_anchored.yaml` | `recog/dataset3d_seg_anchored/...` | `recog/checkpoints/seg_anchored` |
| `configs/segmentation_wide.yaml` | `recog/dataset3d_seg_wide/...` | `recog/checkpoints/seg_wide` |
| `configs/segmentation_cad_control_holdout_<SKU>.yaml` (×4) | `recog/dataset3d_seg_cad_control_holdout_<SKU>/...` | `recog/checkpoints/seg_cad_control_<SKU>` |

```bash
python -m recog.seg_training --config configs/segmentation_anchored.yaml [--resume]
python -m recog.seg_training --config configs/segmentation_wide.yaml [--resume]
for SKU in AnkerPowerCore10000 AnkerPowerCore13000 AnkerPowerCore20100 AnkerPowerCore26800; do
  python -m recog.seg_training \
      --config "configs/segmentation_cad_control_holdout_${SKU}.yaml" [--resume]
done
```

- [ ] Each run completes the full 40-epoch schedule (`training.epochs: 40`,
  unchanged from `configs/segmentation.yaml`) — use `--resume` across
  multiple invocations exactly as the existing baseline checkpoint
  required.
- [ ] Each writes `best.pt` to its own `checkpoint_dir`.

No commit (checkpoints are gitignored).

---

### Task 19: evaluate all six models on the CAD test set, per-SKU per-class

The CAD test set (Task 16) is evaluated **six times** — once per model —
against the **same** crops each time, using a shared eval config with
`dataset.train_val_split: 0.0` (puts every crop in the validation split;
`_split_dataset`'s `n_train = round(train_val_split * n)`, so `0.0` means
`n_train = 0`) so nothing is excluded as "training data" on a set that was
never trained on:

`configs/segmentation_cad_test.yaml` — copy `configs/segmentation.yaml`,
set `dataset.coco_path`/`img_dir` to
`recog/dataset3d_seg_cad_test/...` and `dataset.train_val_split: 0.0`.

```bash
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_anchored/best.pt \
    --config configs/segmentation_cad_test.yaml --per-sku \
    --out docs/receipts/seg_eval_anchored_on_cad_test.txt

python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_wide/best.pt \
    --config configs/segmentation_cad_test.yaml --per-sku \
    --out docs/receipts/seg_eval_wide_on_cad_test.txt

for SKU in AnkerPowerCore10000 AnkerPowerCore13000 AnkerPowerCore20100 AnkerPowerCore26800; do
  python -m recog.seg_evaluate \
      --checkpoint "recog/checkpoints/seg_cad_control_${SKU}/best.pt" \
      --config configs/segmentation_cad_test.yaml --per-sku \
      --out "docs/receipts/seg_eval_cad_control_${SKU}_on_cad_test.txt"
done
```

Also evaluate anchored/wide against their **own** held-out procedural
validation split (design spec §12 comparison 3 — in-distribution vs.
out-of-distribution):

```bash
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_anchored/best.pt \
    --config configs/segmentation_anchored.yaml \
    --out docs/receipts/seg_eval_anchored_on_anchored_val.txt

python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg_wide/best.pt \
    --config configs/segmentation_wide.yaml \
    --out docs/receipts/seg_eval_wide_on_wide_val.txt
```

- [ ] Every receipt written; `--per-sku` output present in the two
  CAD-test comparisons that need it (anchored/wide) and useful, though not
  strictly required, on each control fold's own receipt (its own held-out
  SKU is the only one whose control-model score is meaningful as a
  ceiling — design spec §10).
- [ ] Assemble design spec §12's four comparisons from the receipts above:
  1. anchored-trained vs. wide-trained, per class per SKU, on the CAD test set.
  2. anchored-trained vs. CAD-control, and wide-trained vs. CAD-control
     (each control fold's own held-out SKU only), per class per SKU.
  3. in-distribution (own val split) vs. out-of-distribution (CAD test),
     per procedural model.
  4. Regression checks: five-class disjointness (already re-verified in
     Tasks 14–17); `obstruction`/`battery` IoU on the CAD test set at or
     above the existing 0.6579/0.6907 floor
     (`docs/receipts/seg_eval.txt`) — a floor, not a target, since this
     CAD test set is now disjoint from training; `python -m pytest -q`
     staying green; `python main.py --config configs/demo.yaml` still
     running.
- [ ] Update `docs/NEXT_STEPS.md` and `docs/FDR_v3.md` with the new
  figures, and **state explicitly, next to every number, that this is a
  synthetic-to-synthetic generalisation result, never a sim-to-real
  measurement** — design spec §0 and §12's closing line, repeated once
  more here because it is the constraint the whole plan exists to work
  within.

- [ ] **Commit**

```bash
git add docs/receipts/ docs/NEXT_STEPS.md docs/FDR_v3.md configs/segmentation_*.yaml
git commit -m "docs: publish the generalisation measurement (spec #2)

Anchored-trained, wide-trained and four leave-one-SKU-out CAD-control
models, all scored on the same held-out CAD test crops, per-SKU per-class.
Synthetic-to-synthetic throughout - not a sim-to-real claim (design spec
Sec0, Sec12)."
```

---

## Acceptance

- [ ] `python -m pytest -q` passes with no regressions, starting from 621
  passed at `f4596e8`.
- [ ] Five-class disjointness stays at 0 overlapping pixels — re-verified
  fresh on the anchored, wide and CAD-test renders (Tasks 11, 14–16), not
  assumed to carry over from the CAD-only guarantee.
- [ ] The `assembled` variant renders sealed for procedural trays exactly
  as it does for CAD (Task 11's contact-sheet check).
- [ ] `unit_id` grouping and unit-scoped VOC boxes are untouched — no task
  in this plan modifies `scene.py`'s pass-index loop.
- [ ] The torch-free demo, `python main.py --config configs/demo.yaml`,
  still runs.
- [ ] Every geometric decision for the procedural tray lives in
  `bay.py`/`catalog.py`/`config.py`; `tests/test_synth3d.py`'s AST bpy
  check still passes with no new bpy-free module added (Tasks 1–7 only
  extend the three files already in `_BPY_FREE_CANDIDATES`).
- [ ] `plan/arbitration.py` still imports only `numpy`/`cv2` — unchanged
  by this plan.
- [ ] `CLASSES`/`SEG_CLASSES` unchanged.
- [ ] Six models trained, all evaluated on the same CAD test crops,
  per-SKU per-class, with the sim-to-real caveat restated next to every
  published number.
