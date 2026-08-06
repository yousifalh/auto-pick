# Segmentation Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Blender generator to emit five-class pixel-exact segmentation ground truth — `battery`, `cartridge`, `electronics_module`, `placement_area`, `obstruction` — as a COCO-RLE sidecar, without changing the two-class Pascal-VOC output the detector trains on.

**Architecture:** Every *geometric decision* (where the module sits, what rectangle the bay proxy covers, where obstructions go, where seated cells go) lives in a new bpy-free module `recog/synth3d/bay.py` so it can be unit-tested outside Blender. `world.py` and `scene.py` — which import `bpy` and cannot be tested — only *apply* those decisions. Measurements come from `catalog.json`, computed once at CAD-conversion time, so no dimension is hardcoded twice.

**Tech Stack:** Python 3.10+, Blender 5.0 bundled Python (no PyYAML, no Pillow, no pycocotools), NumPy, trimesh + cascadio (conversion only, system Python), pytest.

## Global Constraints

- Units are **metres** throughout `recog/synth3d/`; the CAD is millimetres and `catalog.json` records `extents_mm`. `MM = 1000.0`.
- Blender's bundled Python has **no PyYAML and no pycocotools**. Config is read from the JSON sidecar via `recog.sync_config`; RLE must be encoded by hand (Task 7 gives the code).
- `CLASSES = ["battery", "cartridge"]` must not change. A test pins it to `recog.dataset.CLASS_MAP`.
- `SEG_CLASSES = ["battery", "cartridge", "electronics_module", "placement_area", "obstruction"]`. FasterRCNN reserves 0 for background, so ids start at 1 in both sets.
- Every object with a non-zero `pass_index` must have an `id_meta` entry, or `scene._check_id_meta_covers_scene` raises.
- Boxes use **exclusive** max edges: a one-pixel object is a 1×1 box, never zero-area.
- Modules importing `bpy`: `world.py`, `scene.py`, `assets.py`, `render.py`, `lightrig.py`, `materials.py`. Modules that must stay bpy-free and testable: `config.py`, `catalog.py`, `layout.py`, `annotate.py`, and the new `bay.py`.
- `placement_area` is the **currently free** floor (spec ruling 4) — modal, what the camera sees.

---

## File Structure

| File | Responsibility |
|---|---|
| `recog/synth3d/bay.py` | **New, bpy-free.** All bay geometry: module rect, proxy rect, obstruction poses, seated-cell poses. Pure functions over numbers. |
| `recog/synth3d/catalog.py` | Gains cell-union AABB and module-bay measurement in `inspect_glb`. |
| `recog/synth3d/config.py` | Gains `SEG_CLASSES`, `seg_class_ids()`, `ObstructionCfg`. |
| `recog/synth3d/world.py` | `build_pcb` anchored + labelled; new `build_bay_proxy`, `build_obstructions`. Applies `bay.py` decisions. |
| `recog/synth3d/scene.py` | Wires new objects into `pass_index` / `id_meta`; seats cells in bays. |
| `recog/synth3d/annotate.py` | Gains `rle_encode`, `masks_from_index`, `write_coco_json`; filter exemption for `placement_area`. |
| `recog/generate3d.py` | Writes the COCO sidecar alongside VOC. |
| `recog/verify3d.py` | Draws masks, not just boxes. |
| `tests/test_bay.py` | **New.** Unit tests for `bay.py`. |
| `tests/test_annotate_masks.py` | **New.** RLE round-trip, filter exemption, COCO shape. |

---

### Task 1: Measure the module bay at CAD-conversion time

**Files:**
- Modify: `recog/synth3d/catalog.py` (`inspect_glb`, lines 58-84)
- Test: `tests/test_bay.py` (create)

**Interfaces:**
- Consumes: `role_of(subpart_name) -> str` (existing, `catalog.py:36`), trimesh scene graph
- Produces: two new keys in each `catalog.json` asset entry —
  - `"cell_union_mm": [x0, y0, x1, y1, z_top]` — world-space AABB of every `cell`-role sub-part in assembled pose, millimetres
  - `"case_interior_mm": [x0, y0, x1, y1, z_top]` — world-space AABB of every `case`-role sub-part
  - `"module_bay_mm": [x0, y0, x1, y1]` — the free strip: `case_interior` minus `cell_union`, taken on the side with the largest gap

Expected values, already measured from the four committed assemblies — these are the acceptance bar:

| Asset | `case_interior_mm` x,y extent | `cell_union_mm` x,y extent | bay depth |
|---|---|---|---|
| AnkerPowerCore10000 | 62.9 × 90.9 | 54.9 × 65.0 | 23.5 |
| AnkerPowerCore13000 | 80.7 × 97.0 | 73.2 × 65.0 | 26.5 |
| AnkerPowerCore20100 | 62.3 × 167.8 | 54.9 × 133.0 | 28.9 |
| AnkerPowerCore26800 | 81.7 × 180.0 | 73.2 × 140.0 | 35.0 |

- [ ] **Step 1: Write the failing test**

Create `tests/test_bay.py`:

```python
"""Bay geometry — pure functions, no bpy, no Blender."""
from __future__ import annotations

import json
import os

import pytest

from recog.synth3d.bay import module_bay_from_bounds

ASSETS = os.path.join(os.path.dirname(__file__), "..",
                      "recog", "synth3d", "assets")


def test_module_bay_picks_the_largest_gap_side():
    # Interior 0..60 x 0..90; cells fill 4..56 x 4..66.
    # Gaps: -x 4, +x 4, -y 4, +y 24. The +y gap wins.
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 66.0, 60.0, 90.0))


def test_module_bay_spans_the_full_interior_width():
    """The module runs wall to wall across the short side."""
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    x0, y0, x1, y1 = module_bay_from_bounds(interior, cells)
    assert (x0, x1) == (0.0, 60.0)


def test_module_bay_handles_the_gap_on_the_minus_y_side():
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 24.0, 56.0, 86.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 0.0, 60.0, 24.0))


def test_module_bay_handles_the_gap_on_an_x_side():
    interior = (0.0, 0.0, 90.0, 60.0)
    cells = (24.0, 4.0, 86.0, 56.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 0.0, 24.0, 60.0))


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,depth", [
    ("AnkerPowerCore10000", 23.5),
    ("AnkerPowerCore13000", 26.5),
    ("AnkerPowerCore20100", 28.9),
    ("AnkerPowerCore26800", 35.0),
])
def test_catalog_records_the_measured_bay_depth(name, depth):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert "module_bay_mm" in entry, "re-run: python -m recog.convert_cad"
    x0, y0, x1, y1 = entry["module_bay_mm"]
    assert max(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6) or \
           min(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.synth3d.bay'`

- [ ] **Step 3: Create `bay.py` with the measurement function**

Create `recog/synth3d/bay.py`:

```python
"""
recog.synth3d.bay - every geometric decision about a cartridge's interior.

No bpy import, so all of it is unit-testable outside Blender. world.py and
scene.py import bpy and cannot be tested; they call into here for the numbers
and only apply the result. Keeping the arithmetic on this side of the line is
what makes the bay geometry checkable at all.

Units follow the caller. `module_bay_from_bounds` is used on millimetre CAD
bounds at conversion time and on metre scene bounds at render time; it is scale
free.
"""

from __future__ import annotations

from typing import Tuple

Rect = Tuple[float, float, float, float]     # x0, y0, x1, y1


def module_bay_from_bounds(interior: Rect, cells: Rect) -> Rect:
    """The strip of interior the cells do not occupy, on the widest side.

    The four Anker assemblies all leave 2.4-5.9 mm on three sides (wall
    thickness) and 23.5-35.0 mm on one short side. That large gap is the
    electronics bay: the CAD has no PCB part, but it reserves the space.

    Returns the bay as a rectangle spanning the interior's full width on
    the chosen axis, because the real module runs wall to wall.
    """
    ix0, iy0, ix1, iy1 = interior
    cx0, cy0, cx1, cy1 = cells

    gaps = {
        "-x": cx0 - ix0,
        "+x": ix1 - cx1,
        "-y": cy0 - iy0,
        "+y": iy1 - cy1,
    }
    side = max(gaps, key=lambda k: gaps[k])

    if side == "-x":
        return (ix0, iy0, cx0, iy1)
    if side == "+x":
        return (cx1, iy0, ix1, iy1)
    if side == "-y":
        return (ix0, iy0, ix1, cy0)
    return (ix0, cy1, ix1, iy1)
```

- [ ] **Step 4: Run the pure-geometry tests**

Run: `pytest tests/test_bay.py -v -k "not catalog"`
Expected: PASS, 4 tests. The catalog test still fails — Step 5 fixes it.

- [ ] **Step 5: Record the measurement in `inspect_glb`**

In `recog/synth3d/catalog.py`, replace the body of `inspect_glb` (lines 58-84) with a version that walks the scene *graph* rather than bare geometry, so world transforms are applied. The existing version reads `g.extents`, which is the untransformed mesh extent and cannot tell you where a part sits.

```python
def inspect_glb(path: str) -> dict:
    """Measure a converted asset and classify its sub-parts.

    Walks the scene GRAPH, not `scene.geometry`, because a sub-part's
    position is carried by its node transform. `g.extents` alone says how
    big a cell is but not where it sits, and where it sits is what
    reveals the electronics bay.
    """
    import numpy as np
    import trimesh

    from .bay import module_bay_from_bounds

    scene = trimesh.load(path)

    subparts, counts = [], {}
    by_role_bounds = {}

    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        g = scene.geometry[gname]
        corners = trimesh.transform_points(
            trimesh.bounds.corners(g.bounds), transform)
        lo, hi = corners.min(axis=0) * MM, corners.max(axis=0) * MM

        role = role_of(node)
        counts[role] = counts.get(role, 0) + 1
        by_role_bounds.setdefault(role, []).append((lo, hi))

        subparts.append({
            "name": node,
            "role": role,
            "extents_mm": [round(float(v) * MM, 2) for v in g.extents],
            "triangles": int(len(g.faces)),
            "volume_mm3": round(float(g.volume) * MM ** 3, 1)
            if g.is_volume else None,
        })

    def _aabb(role):
        if role not in by_role_bounds:
            return None
        los = np.array([b[0] for b in by_role_bounds[role]])
        his = np.array([b[1] for b in by_role_bounds[role]])
        lo, hi = los.min(axis=0), his.max(axis=0)
        return [round(float(v), 2) for v in
                (lo[0], lo[1], hi[0], hi[1], hi[2])]

    cell_union = _aabb("cell")
    case_interior = _aabb("case")

    out = {
        "extents_mm": [round(float(v) * MM, 2)
                       for v in (scene.bounds[1] - scene.bounds[0])],
        "triangles": int(sum(len(g.faces) for g in scene.geometry.values())),
        "subparts": sorted(subparts, key=lambda s: (s["role"], s["name"])),
        "role_counts": counts,
        "cell_union_mm": cell_union,
        "case_interior_mm": case_interior,
    }
    if cell_union and case_interior:
        out["module_bay_mm"] = [
            round(v, 2) for v in module_bay_from_bounds(
                tuple(case_interior[:4]), tuple(cell_union[:4]))
        ]
    return out
```

Note `role_of(node)` uses the **node** name, not the geometry name. Node names carry the instance suffix NX generates (`004695_A;1-Cell_18650_18652`), which is what `CLASS_RULES`' `Cell[_ ]?\d+` pattern is written against.

- [ ] **Step 6: Regenerate the catalog and verify**

Run:
```bash
pip install cascadio trimesh
python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/
pytest tests/test_bay.py -v
```
Expected: PASS, 8 tests. The four parametrised catalog tests confirm the measured depths.

If `convert_cad` refuses an asset on its extents plausibility guard, that guard is doing its job — read the message rather than bypassing it.

- [ ] **Step 7: Commit**

```bash
git add recog/synth3d/bay.py recog/synth3d/catalog.py recog/synth3d/assets/catalog.json tests/test_bay.py
git commit -m "feat(synth3d): measure the electronics bay from CAD at conversion time

inspect_glb walked scene.geometry, which gives each sub-part's size but
not its position, so it could not see that every assembly reserves a
23-35mm strip on one short side for the electronics module. It now walks
the scene graph and records cell_union_mm, case_interior_mm and the
derived module_bay_mm.

The bay is where the PCB goes and its complement is the battery
placement area, so both now come from one measurement rather than two
hardcoded constants."
```

---

### Task 2: Split the class sets

**Files:**
- Modify: `recog/synth3d/config.py:35-39`
- Test: `tests/test_synth3d.py` (append)

**Interfaces:**
- Consumes: `recog.dataset.CLASS_MAP` (existing)
- Produces:
  - `CLASSES: List[str]` — unchanged, `["battery", "cartridge"]`
  - `SEG_CLASSES: List[str]` — `["battery", "cartridge", "electronics_module", "placement_area", "obstruction"]`
  - `class_ids() -> Dict[str, int]` — unchanged
  - `seg_class_ids() -> Dict[str, int]` — `{name: i+1 for i, name in enumerate(SEG_CLASSES)}`

The detector's head is sized by `CLASS_MAP`. Growing `CLASSES` would invalidate every committed checkpoint and every published number in the FDR, so the segmentation labels live in a parallel set that only the COCO sidecar carries.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_synth3d.py`:

```python
def test_detector_class_set_is_unchanged():
    """Growing CLASSES would resize the detector head and invalidate
    every committed checkpoint. The segmentation classes go elsewhere."""
    from recog.dataset import CLASS_MAP
    from recog.synth3d.config import CLASSES, class_ids

    assert CLASSES == ["battery", "cartridge"]
    assert class_ids() == CLASS_MAP


def test_seg_class_set_extends_the_detector_set_in_order():
    from recog.synth3d.config import CLASSES, SEG_CLASSES, seg_class_ids

    assert SEG_CLASSES[:len(CLASSES)] == CLASSES, (
        "SEG_CLASSES must start with CLASSES so a shared id means a "
        "shared class between the VOC and COCO outputs")
    assert SEG_CLASSES == ["battery", "cartridge", "electronics_module",
                           "placement_area", "obstruction"]
    assert seg_class_ids() == {
        "battery": 1, "cartridge": 2, "electronics_module": 3,
        "placement_area": 4, "obstruction": 5,
    }
    assert 0 not in seg_class_ids().values(), "0 is reserved for background"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_synth3d.py -v -k class_set`
Expected: FAIL — `ImportError: cannot import name 'SEG_CLASSES'`

- [ ] **Step 3: Add the second class set**

In `recog/synth3d/config.py`, replace lines 35-39:

```python
CLASSES: List[str] = ["battery", "cartridge"]

# The segmentation label set. Deliberately SEPARATE from CLASSES: the
# detector's classification head is sized by recog.dataset.CLASS_MAP, and
# growing that invalidates every committed checkpoint and every published
# number. The VOC output stays two-class; these five go only to the COCO
# sidecar that the segmenter reads.
#
# Order matters. SEG_CLASSES starts with CLASSES so that ids 1 and 2 mean
# the same thing in both files.
SEG_CLASSES: List[str] = [
    "battery", "cartridge", "electronics_module",
    "placement_area", "obstruction",
]


def class_ids() -> Dict[str, int]:
    return {c: i + 1 for i, c in enumerate(CLASSES)}


def seg_class_ids() -> Dict[str, int]:
    return {c: i + 1 for i, c in enumerate(SEG_CLASSES)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_synth3d.py -v -k class_set`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add recog/synth3d/config.py tests/test_synth3d.py
git commit -m "feat(synth3d): add SEG_CLASSES alongside CLASSES

The detector head is sized by CLASS_MAP, so growing CLASSES from 2 to 5
would invalidate every committed checkpoint. The segmentation labels get
their own set, carried only by the COCO sidecar; VOC output stays
two-class and the detector cannot regress."
```

---

### Task 3: Anchor the electronics module to the short side and label it

**Files:**
- Modify: `recog/synth3d/bay.py` (add `module_rect_in_footprint`)
- Modify: `recog/synth3d/world.py:569-640` (`build_pcb`)
- Modify: `recog/synth3d/scene.py:213-218` (the `build_pcb` call site)
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `module_bay_from_bounds` (Task 1), `catalog.json`'s `module_bay_mm` and `case_interior_mm`
- Produces:
  - `bay.module_rect_in_footprint(footprint, bay_mm, interior_mm) -> Rect` — given a placed cartridge's world footprint `(x0, y0, x1, y1)` in metres and the catalog's millimetre rectangles, returns the module's rectangle in the same world metres, anchored to the correct side and scaled proportionally.
  - `world.build_pcb(bounds_xy, z, rng, module_rect=None) -> (board_obj, meta)` — gains the `module_rect` parameter; when given, the board is built there instead of centred. Returns the board object so the caller can assign a `pass_index`.

`build_pcb` currently centres the board (`cx = (x0+x1)/2`, `cy = (y0+y1)/2 ± 10 mm`). Every real photograph and every CAD assembly puts it against one short side. This is an appearance defect, not a box defect — a centred board punches a hole that `boxes_from_mask`'s min/max bounding ignores — but appearance is what the network learns from.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bay.py`:

```python
from recog.synth3d.bay import module_rect_in_footprint


def test_module_rect_anchors_to_the_plus_y_side_and_scales():
    # Catalog: interior 0..60 x 0..90 mm, bay is the +y strip 66..90.
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    # The cartridge is placed in the scene at 0.1..0.16 x 0.2..0.29 m,
    # i.e. the same proportions, 1000x smaller.
    footprint = (0.100, 0.200, 0.160, 0.290)
    x0, y0, x1, y1 = module_rect_in_footprint(footprint, bay_mm, interior_mm)
    assert (x0, x1) == pytest.approx((0.100, 0.160))       # full width
    assert (y0, y1) == pytest.approx((0.266, 0.290))       # +y strip


def test_module_rect_anchors_to_the_minus_x_side():
    interior_mm = (0.0, 0.0, 90.0, 60.0)
    bay_mm = (0.0, 0.0, 24.0, 60.0)
    footprint = (0.0, 0.0, 0.090, 0.060)
    x0, y0, x1, y1 = module_rect_in_footprint(footprint, bay_mm, interior_mm)
    assert (x0, x1) == pytest.approx((0.0, 0.024))
    assert (y0, y1) == pytest.approx((0.0, 0.060))


def test_module_rect_is_a_strict_subset_of_the_footprint():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.100, 0.200, 0.160, 0.290)
    mx0, my0, mx1, my1 = module_rect_in_footprint(
        footprint, bay_mm, interior_mm)
    fx0, fy0, fx1, fy1 = footprint
    assert fx0 - 1e-9 <= mx0 < mx1 <= fx1 + 1e-9
    assert fy0 - 1e-9 <= my0 < my1 <= fy1 + 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bay.py -v -k module_rect`
Expected: FAIL — `ImportError: cannot import name 'module_rect_in_footprint'`

- [ ] **Step 3: Implement the mapping**

Append to `recog/synth3d/bay.py`:

```python
def _lerp_rect(rect: Rect, src: Rect, dst: Rect) -> Rect:
    """Map `rect` from the `src` box's frame into the `dst` box's frame."""
    sx0, sy0, sx1, sy1 = src
    dx0, dy0, dx1, dy1 = dst
    sw = (sx1 - sx0) or 1.0
    sh = (sy1 - sy0) or 1.0
    fx = (dx1 - dx0) / sw
    fy = (dy1 - dy0) / sh
    x0, y0, x1, y1 = rect
    return (dx0 + (x0 - sx0) * fx, dy0 + (y0 - sy0) * fy,
            dx0 + (x1 - sx0) * fx, dy0 + (y1 - sy0) * fy)


def module_rect_in_footprint(footprint: Rect, bay_mm: Rect,
                             interior_mm: Rect) -> Rect:
    """Where the electronics module sits inside a placed cartridge.

    `footprint` is the cartridge's world AABB in metres; `bay_mm` and
    `interior_mm` come from catalog.json in millimetres. The bay keeps
    its proportion of the interior, so the module lands against the same
    short side at the same relative depth whatever the scene scale.

    The caller is responsible for the cartridge's rotation: this works in
    the footprint's own axis-aligned frame, which is what `layout.plan`
    produces after a k*90 rotation.
    """
    return _lerp_rect(bay_mm, interior_mm, footprint)


def placement_rect_in_footprint(footprint: Rect, bay_mm: Rect,
                                interior_mm: Rect) -> Rect:
    """The battery placement area: the interior minus the module bay.

    The complement of `module_rect_in_footprint` within the footprint,
    taken on whichever axis the bay occupies.
    """
    fx0, fy0, fx1, fy1 = footprint
    mx0, my0, mx1, my1 = module_rect_in_footprint(
        footprint, bay_mm, interior_mm)

    # The bay spans one full axis; the placement area is what is left on
    # the other. Compare against the footprint edges to find which.
    tol = 1e-9
    if abs(mx0 - fx0) < tol and abs(mx1 - fx1) < tol:
        # Bay spans full width -> it took a y side.
        return (fx0, my1, fx1, fy1) if abs(my0 - fy0) < tol \
            else (fx0, fy0, fx1, my0)
    return (mx1, fy0, fx1, fy1) if abs(mx0 - fx0) < tol \
        else (fx0, fy0, mx0, fy1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bay.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Add a placement-rect test and re-run**

Append to `tests/test_bay.py`:

```python
def test_placement_rect_is_the_complement_of_the_module_rect():
    from recog.synth3d.bay import (module_rect_in_footprint,
                                   placement_rect_in_footprint)
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.0, 0.0, 0.060, 0.090)

    m = module_rect_in_footprint(footprint, bay_mm, interior_mm)
    p = placement_rect_in_footprint(footprint, bay_mm, interior_mm)

    # Disjoint, adjacent, and together they tile the footprint.
    assert p[3] == pytest.approx(m[1])
    assert (p[0], p[1], p[2]) == pytest.approx((0.0, 0.0, 0.060))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    assert area(m) + area(p) == pytest.approx(0.060 * 0.090)
```

Run: `pytest tests/test_bay.py -v`
Expected: PASS, 12 tests

- [ ] **Step 6: Rebuild `build_pcb` against the module rect**

In `recog/synth3d/world.py`, change `build_pcb`'s signature and its position arithmetic. Replace lines 569-597 (the docstring through the `drawn = {...}` assignment) with:

```python
def build_pcb(bounds_xy, z: float, rng: random.Random, module_rect=None):
    """
    Green PCB with extruded components, for the open_case variant.

    The CAD has no PCB part, but it reserves the space: every assembly
    leaves a 23-35mm strip on one short side (catalog.json's
    module_bay_mm). `module_rect` is that strip mapped into this
    cartridge's world footprint by bay.module_rect_in_footprint; pass it
    and the board lands where the hardware puts it.

    Omit it and the board is centred, which is what this function did
    before and what every real photograph contradicts. Centring is
    box-safe - boxes_from_mask takes min/max of visible pixels, so a hole
    in the middle of a case does not move its box - but it is not what
    the detector should be learning to see.

    `z` is the shell's TOP: the board is laid on top of the shell rather
    than modelled inside it. There is no interior geometry in these
    assemblies at all. From the near-orthographic bird's-eye camera the
    two read the same.

    Returns (board_object, drawn_meta). The caller assigns pass_index.
    """
    x0, y0, x1, y1 = bounds_xy
    if module_rect is not None:
        mx0, my0, mx1, my1 = module_rect
        w, h = mx1 - mx0, my1 - my0
        cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
    else:
        w = (x1 - x0) * rng.uniform(0.55, 0.80)
        h = (y1 - y0) * rng.uniform(0.20, 0.38)
        cx = (x0 + x1) / 2 + rng.uniform(-0.004, 0.004)
        cy = (y0 + y1) / 2 + rng.uniform(-0.010, 0.010)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z + 0.0008))
    board = bpy.context.active_object
    board.name = "PCB"
    board.scale = (w, h, 0.0016)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    board.pass_index = 0          # scene.py overrides this

    drawn = {"w": w, "h": h,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.16, 0.30),
                       rng.uniform(0.04, 0.10)],
             "roughness": rng.uniform(0.25, 0.50),
             "n_components": rng.randint(3, 7),
             "anchored": module_rect is not None}
```

Then at the end of the function, change the return to hand back the board and its child components:

```python
    return board, drawn
```

Check the existing return statement at the end of `build_pcb` and make it match — the current signature returns `(board, drawn)` already at the call site (`_, pcb_meta = W.build_pcb(...)`), so only the parameter is new.

- [ ] **Step 7: Add gold connectors and a copper inductor**

Still in `build_pcb`, after the component loop, add two distinctive features. In IMG_4426 the gold USB shells and the copper inductor are what make the module unmistakable; the current board has only dark cuboids.

```python
    # Gold USB shells along the board's outward edge, and one copper
    # inductor. These are the module's two most recognisable features in
    # the real photographs and the dark cuboids above have neither.
    gold = bpy.data.materials.new("PCBGold")
    gold.use_nodes = True
    gb = gold.node_tree.nodes.get("Principled BSDF")
    set_input(gb, "Base Color", (0.75, 0.60, 0.22, 1.0))
    set_input(gb, "Metallic", 1.0)
    set_input(gb, "Roughness", rng.uniform(0.20, 0.35))

    n_ports = rng.randint(1, 4)
    port_w, port_h, port_z = 0.013, 0.006, 0.005
    for k in range(n_ports):
        span = n_ports * port_w * 1.3
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx - span / 2 + port_w * 1.3 * (k + 0.5),
                      cy + h / 2 - port_h,
                      z + 0.0016 + port_z / 2))
        p = bpy.context.active_object
        p.name = f"PCBPort{k}"
        p.scale = (port_w, port_h, port_z)
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)
        p.pass_index = 0
        p.data.materials.append(gold)
        p.parent = board
        p.matrix_parent_inverse = board.matrix_world.inverted()

    copper = bpy.data.materials.new("PCBCopper")
    copper.use_nodes = True
    cb2 = copper.node_tree.nodes.get("Principled BSDF")
    set_input(cb2, "Base Color", (0.72, 0.35, 0.18, 1.0))
    set_input(cb2, "Metallic", 1.0)
    set_input(cb2, "Roughness", rng.uniform(0.35, 0.55))
    bpy.ops.mesh.primitive_cylinder_add(
        radius=rng.uniform(0.004, 0.007), depth=0.006,
        location=(cx + rng.uniform(-w / 4, w / 4),
                  cy + rng.uniform(-h / 4, h / 4), z + 0.0016 + 0.003))
    ind = bpy.context.active_object
    ind.name = "PCBInductor"
    ind.pass_index = 0
    ind.data.materials.append(copper)
    ind.parent = board
    ind.matrix_parent_inverse = board.matrix_world.inverted()

    drawn["n_ports"] = n_ports
```

The `matrix_parent_inverse` line is not optional. `world.py:628-634` records the measured consequence of omitting it: a component built at (0.32, 0.16, 0.06) under a board at (0.30, 0.15, 0.05) ended up 339 mm away.

- [ ] **Step 8: Pass the module rect from the call site**

In `recog/synth3d/scene.py`, replace lines 213-218:

```python
    for item in items:
        if item.variant == "open_case" and any(
                role_of(o.name) == "case" for o in item.objects):
            lo, hi = A.group_bbox(item.objects)
            entry = library.catalog_entry(item.asset)
            module_rect = None
            if entry and entry.get("module_bay_mm") \
                    and entry.get("case_interior_mm"):
                module_rect = B.module_rect_in_footprint(
                    (lo.x, lo.y, hi.x, hi.y),
                    tuple(entry["module_bay_mm"]),
                    tuple(entry["case_interior_mm"][:4]),
                )
            board, pcb_meta = W.build_pcb(
                (lo.x, lo.y, hi.x, hi.y), hi.z, rng, module_rect=module_rect)
            item.module_object = board
            meta.setdefault("pcbs", []).append(pcb_meta)
```

Add `from . import bay as B` to `scene.py`'s imports. Add a `module_object` field defaulting to `None` on `assets.Item`. Add `AssetLibrary.catalog_entry(name) -> Optional[dict]` returning the matching entry from the loaded catalog.

- [ ] **Step 9: Render a check and look at it**

Run:
```bash
python -m recog.sync_config
BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BLENDER" -b --python recog/generate3d.py -- --n 6 --out recog/dev3d --variant open_case --device GPU
python -m recog.verify3d --data recog/dev3d --n 6
```
Then **open `recog/dev3d/contact_sheet.png` and look at it.** The board must sit against one short side of each case, not in the middle, with visible gold ports along its outer edge. A rendering bug that a test cannot catch is caught here or not at all.

- [ ] **Step 10: Commit**

```bash
git add recog/synth3d/bay.py recog/synth3d/world.py recog/synth3d/scene.py recog/synth3d/assets.py tests/test_bay.py
git commit -m "fix(synth3d): anchor the PCB to the short side it occupies in reality

build_pcb centred the board. Every real photograph and all four CAD
assemblies put it hard against one short side, in the 23-35mm strip
catalog.json now records as module_bay_mm.

This is an appearance fix, not a box fix: boxes_from_mask takes min/max
of visible pixels, so a hole in the middle of a case never moved its
box. But appearance is what the detector learns from, and a centred
board contradicts every photograph it will be deployed on.

Also adds gold USB shells and a copper inductor, which are the module's
most recognisable real features."
```

---

### Task 4: The bay proxy

**Files:**
- Modify: `recog/synth3d/world.py` (add `build_bay_proxy`)
- Modify: `recog/synth3d/scene.py` (call it, assign `pass_index`, extend `id_meta`)
- Test: `tests/test_bay.py` (already covers `placement_rect_in_footprint` from Task 3)

**Interfaces:**
- Consumes: `bay.placement_rect_in_footprint` (Task 3), `seg_class_ids()` (Task 2)
- Produces: `world.build_bay_proxy(placement_rect, z, rng) -> (proxy_obj, meta)` — a thin plane covering the placement rectangle, sitting fractionally above the shell top so it renders in front of it.

The proxy is what carries the `placement_area` label. It is coplanar with the PCB — on the shell **top**, not an interior floor, because these assemblies have no interior geometry. Under ruling 4 the label is the *currently free* floor, and occlusion does that automatically: any cell, obstruction or the module resting on the proxy hides those pixels, and `boxes_from_mask` reports only what is visible. No second render pass and no set arithmetic.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bay.py`:

```python
def test_proxy_and_module_rects_do_not_overlap():
    """The proxy must not be drawn under the module, or the two labels
    would fight for the same pixels and the winner would be z-order."""
    from recog.synth3d.bay import (module_rect_in_footprint,
                                   placement_rect_in_footprint)
    interior_mm = (0.0, 0.0, 62.9, 90.9)
    bay_mm = (0.0, 67.4, 62.9, 90.9)
    footprint = (-0.0315, -0.0455, 0.0315, 0.0455)

    m = module_rect_in_footprint(footprint, bay_mm, interior_mm)
    p = placement_rect_in_footprint(footprint, bay_mm, interior_mm)

    ox = min(m[2], p[2]) - max(m[0], p[0])
    oy = min(m[3], p[3]) - max(m[1], p[1])
    assert ox <= 1e-9 or oy <= 1e-9, "module and placement rects overlap"
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `pytest tests/test_bay.py -v -k proxy_and_module`
Expected: PASS — `placement_rect_in_footprint` was written as the complement in Task 3, so this is a guard against a future regression rather than a red test. Keep it; a test that documents an invariant earns its place even when it starts green.

- [ ] **Step 3: Implement `build_bay_proxy`**

Add to `recog/synth3d/world.py`, after `build_pcb`:

```python
def build_bay_proxy(placement_rect, z: float, rng: random.Random):
    """
    The plane that carries the `placement_area` label.

    Coplanar with the PCB, on the shell TOP - these assemblies have no
    interior geometry at all, so there is no floor to sit on. Under the
    bird's-eye camera the two read the same, which is the same trick
    build_pcb already relies on.

    Sits 0.1mm above the PCB's own offset so it never z-fights with the
    shell. It is DELIBERATELY a visible, shaded object rather than an
    index-only helper: the placement area has to look like the inside of
    a cartridge, because the segmenter is asked to recognise it from
    appearance.

    Occlusion does the rest. A cell seated on this plane, an obstruction
    stuck to it, or the module beside it all hide their own pixels, and
    boxes_from_mask reports only what stayed visible - so the label is
    the CURRENTLY FREE floor with no set arithmetic anywhere.
    """
    x0, y0, x1, y1 = placement_rect
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, z + 0.0009))
    proxy = bpy.context.active_object
    proxy.name = "BayProxy"
    proxy.scale = (x1 - x0, y1 - y0, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    proxy.pass_index = 0          # scene.py overrides

    drawn = {"w": x1 - x0, "h": y1 - y0,
             "color": [rng.uniform(0.02, 0.05)] * 3,
             "roughness": rng.uniform(0.55, 0.85)}

    mat = bpy.data.materials.new("BayFloor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    set_input(bsdf, "Roughness", drawn["roughness"])
    proxy.data.materials.append(mat)
    return proxy, drawn
```

- [ ] **Step 4: Wire it into the scene with a pass index**

In `scene.py`, inside the same `for item in items:` loop from Task 3 Step 8, after building the board:

```python
            placement_rect = None
            if module_rect is not None:
                placement_rect = B.placement_rect_in_footprint(
                    (lo.x, lo.y, hi.x, hi.y),
                    tuple(entry["module_bay_mm"]),
                    tuple(entry["case_interior_mm"][:4]),
                )
                proxy, proxy_meta = W.build_bay_proxy(
                    placement_rect, hi.z, rng)
                item.bay_object = proxy
                meta.setdefault("bays", []).append(proxy_meta)
```

Then in the pass-index assignment block (`scene.py:220-239`), after the existing per-object loop, assign indices to the new objects and record their `id_meta`:

```python
    seg_ids = seg_class_ids()
    for gi, item in enumerate(items):
        for obj, cls in (
            (getattr(item, "module_object", None), "electronics_module"),
            (getattr(item, "bay_object", None), "placement_area"),
        ):
            if obj is None:
                continue
            pid += 1
            obj.pass_index = pid
            id_meta[pid] = {"class": cls, "asset": item.asset,
                            "variant": item.variant, "role": cls}
            objects_by_id[pid] = [obj]
```

The PCB's child ports and inductor keep `pass_index = 0`. They sit *on* the board, so they occlude it and shrink its visible region — which is what a real connector does to a real board's silhouette. If you want them counted as module pixels instead, give them the board's `pid` rather than a new one; do not give them `0` and then wonder why the module mask has holes.

Add `bay_object` and `module_object` fields to `assets.Item`, both defaulting to `None`. Import `seg_class_ids` in `scene.py`.

- [ ] **Step 5: Verify `_check_id_meta_covers_scene` still passes**

Run the same 6-scene render as Task 3 Step 9. Expected: no `ValueError: objects carry a non-zero pass_index with no id_meta entry`. If it raises, an object got an index without an `id_meta` row — that guard exists precisely to catch this and it is doing its job.

- [ ] **Step 6: Look at the contact sheet**

Run: `python -m recog.verify3d --data recog/dev3d --n 6`
Open `recog/dev3d/contact_sheet.png`. Each open case must show a dark bay plane beside the green board, with a box around it.

- [ ] **Step 7: Commit**

```bash
git add recog/synth3d/world.py recog/synth3d/scene.py recog/synth3d/assets.py tests/test_bay.py
git commit -m "feat(synth3d): add the bay proxy carrying the placement_area label

A shaded plane on the shell top, complementary to the PCB, covering the
strip the cells occupy in assembled pose.

It carries a pass_index, so occlusion alone produces the label: anything
resting on it hides its own pixels and boxes_from_mask reports only what
is visible. That makes placement_area the currently-free floor by
construction, with no second render pass and no mask arithmetic."
```

---

### Task 5: Obstructions

**Files:**
- Modify: `recog/synth3d/bay.py` (add `sample_obstructions`)
- Modify: `recog/synth3d/config.py` (add `ObstructionCfg`, register in `_SECTIONS`)
- Modify: `configs/synth3d.yaml` (add the `obstruction` block)
- Modify: `recog/synth3d/world.py` (add `build_obstructions`)
- Modify: `recog/synth3d/scene.py` (call, index, label)
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `bay.placement_rect_in_footprint`
- Produces:
  - `bay.ObstructionPose` — dataclass `(kind: str, x: float, y: float, w: float, h: float, rot_deg: float)`
  - `bay.sample_obstructions(placement_rect, cfg, rng) -> List[ObstructionPose]`
  - `world.build_obstructions(poses, z, rng) -> List[Tuple[obj, meta]]`

Adhesive, foam, tape and labels are what the real bays contain (IMG_4426) and what §1.1's measurement shows the current extractor latching onto. A placement mask trained on clean bays would site a cell on a glue blob.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bay.py`:

```python
import random

from recog.synth3d.bay import ObstructionPose, sample_obstructions


class _ObsCfg:
    p_none = 0.4
    n_adhesive = (0, 6)
    n_foam = (0, 1)
    n_tape = (0, 2)
    n_label = (0, 1)
    adhesive_frac = (0.04, 0.14)
    foam_frac = (0.15, 0.35)
    tape_frac = (0.05, 0.12)
    label_frac = (0.10, 0.22)


RECT = (0.0, 0.0, 0.055, 0.065)


def test_every_obstruction_lies_inside_the_placement_rect():
    cfg = _ObsCfg()
    for seed in range(300):
        poses = sample_obstructions(RECT, cfg, random.Random(seed))
        for p in poses:
            assert p.x - p.w / 2 >= RECT[0] - 1e-9, f"seed {seed}: {p}"
            assert p.y - p.h / 2 >= RECT[1] - 1e-9, f"seed {seed}: {p}"
            assert p.x + p.w / 2 <= RECT[2] + 1e-9, f"seed {seed}: {p}"
            assert p.y + p.h / 2 <= RECT[3] + 1e-9, f"seed {seed}: {p}"


def test_roughly_forty_percent_of_bays_are_clean():
    cfg = _ObsCfg()
    empty = sum(1 for s in range(2000)
                if not sample_obstructions(RECT, cfg, random.Random(s)))
    assert 0.33 < empty / 2000 < 0.47, (
        "the network must see clean bays too, or it learns that every "
        "bay contains adhesive")


def test_sampling_is_deterministic_for_a_seed():
    cfg = _ObsCfg()
    a = sample_obstructions(RECT, cfg, random.Random(42))
    b = sample_obstructions(RECT, cfg, random.Random(42))
    assert a == b


def test_all_four_kinds_can_be_produced():
    cfg = _ObsCfg()
    kinds = set()
    for s in range(500):
        kinds.update(p.kind for p in sample_obstructions(
            RECT, cfg, random.Random(s)))
    assert kinds == {"adhesive", "foam", "tape", "label"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bay.py -v -k obstruction or clean or kinds`
Expected: FAIL — `ImportError: cannot import name 'ObstructionPose'`

- [ ] **Step 3: Implement the sampler**

Append to `recog/synth3d/bay.py`:

```python
import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ObstructionPose:
    """A piece of foreign matter in a bay. Centre-based, caller's units."""
    kind: str            # adhesive | foam | tape | label
    x: float
    y: float
    w: float
    h: float
    rot_deg: float = 0.0


def sample_obstructions(placement_rect: Rect, cfg,
                        rng: random.Random) -> List[ObstructionPose]:
    """Draw the foreign matter sitting in one bay.

    IMG_4426 shows thermal adhesive, foam pads, tape crosses and printed
    labels in the real bays. None of it is in the CAD, and a placement
    mask that ignores it would site a cell on a glue blob.

    `cfg.p_none` of bays come back empty so the network also sees clean
    ones. Sizes are fractions of the SHORTER bay edge, so an obstruction
    scales with the cartridge rather than being absolute.
    """
    x0, y0, x1, y1 = placement_rect
    bw, bh = x1 - x0, y1 - y0
    short = min(bw, bh)

    if rng.random() < cfg.p_none:
        return []

    out: List[ObstructionPose] = []
    for kind, count_range, frac_range in (
        ("adhesive", cfg.n_adhesive, cfg.adhesive_frac),
        ("foam", cfg.n_foam, cfg.foam_frac),
        ("tape", cfg.n_tape, cfg.tape_frac),
        ("label", cfg.n_label, cfg.label_frac),
    ):
        for _ in range(rng.randint(*count_range)):
            w = short * rng.uniform(*frac_range)
            h = w * rng.uniform(0.6, 1.8) if kind != "tape" \
                else short * rng.uniform(0.5, 0.95)
            w = min(w, bw)
            h = min(h, bh)
            out.append(ObstructionPose(
                kind=kind,
                x=rng.uniform(x0 + w / 2, x1 - w / 2),
                y=rng.uniform(y0 + h / 2, y1 - h / 2),
                w=w, h=h,
                rot_deg=rng.uniform(-180, 180) if kind != "tape"
                else rng.choice([0.0, 90.0]) + rng.uniform(-4, 4),
            ))
    return out
```

Note the clamps: `w = min(w, bw)` and `h = min(h, bh)` before sampling the centre. Without them `rng.uniform(x0 + w/2, x1 - w/2)` gets an inverted range when an obstruction is wider than the bay, and Python's `uniform` does not raise on that — it silently returns a value outside the rect. That is exactly the bug the first test catches.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bay.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Add the config block**

In `recog/synth3d/config.py`, add the dataclass and register it:

```python
@dataclass
class ObstructionCfg:
    """Foreign matter in a cartridge bay. See bay.sample_obstructions."""
    p_none: float = 0.40           # fraction of bays left clean
    n_adhesive: Tuple[int, int] = (0, 6)
    n_foam: Tuple[int, int] = (0, 1)
    n_tape: Tuple[int, int] = (0, 2)
    n_label: Tuple[int, int] = (0, 1)
    adhesive_frac: Tuple[float, float] = (0.04, 0.14)
    foam_frac: Tuple[float, float] = (0.15, 0.35)
    tape_frac: Tuple[float, float] = (0.05, 0.12)
    label_frac: Tuple[float, float] = (0.10, 0.22)
```

Add `obstruction` to `Config`, to `_SECTIONS`, and add every tuple field name to `_TUPLE_FIELDS` — `load_config` coerces lists to tuples only for names listed there, and a YAML list left as a list will break `rng.randint(*count_range)` in a way that only shows up mid-render.

Add to `configs/synth3d.yaml`:

```yaml
obstruction:
  # Foreign matter in the bays. IMG_4426 has thermal adhesive, foam pads,
  # tape crosses and printed labels in every opened case; none of it is in
  # the CAD. p_none keeps clean bays in the distribution too.
  p_none: 0.40
  n_adhesive: [0, 6]
  n_foam: [0, 1]
  n_tape: [0, 2]
  n_label: [0, 1]
  adhesive_frac: [0.04, 0.14]
  foam_frac: [0.15, 0.35]
  tape_frac: [0.05, 0.12]
  label_frac: [0.10, 0.22]
```

Then run `python -m recog.sync_config` — Blender reads the JSON sidecar, and `_read_raw` raises if the sidecar is older than the YAML.

- [ ] **Step 6: Build the geometry**

Add to `recog/synth3d/world.py`:

```python
def build_obstructions(poses, z: float, rng: random.Random):
    """Realise ObstructionPoses as geometry. Returns [(obj, meta), ...].

    Each gets its own pass_index from scene.py, so each is an instance.
    They sit ON the bay proxy and therefore occlude it, which is what
    makes placement_area the currently-free floor rather than the
    nominal one.
    """
    made = []
    for i, p in enumerate(poses):
        if p.kind == "adhesive":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=p.w / 2, location=(p.x, p.y, z + 0.0012))
            o = bpy.context.active_object
            o.scale = (1.0, p.h / max(p.w, 1e-9), rng.uniform(0.35, 0.7))
            base, rough, alpha = (0.92, 0.92, 0.90), 0.30, 0.85
        elif p.kind == "foam":
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(p.x, p.y, z + 0.0022))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, rng.uniform(0.002, 0.005))
            base, rough, alpha = (0.88, 0.88, 0.86), 0.95, 1.0
        elif p.kind == "tape":
            bpy.ops.mesh.primitive_plane_add(
                size=1, location=(p.x, p.y, z + 0.0011))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, 1.0)
            base, rough, alpha = (0.95, 0.95, 0.93), 0.45, 1.0
        else:  # label
            bpy.ops.mesh.primitive_plane_add(
                size=1, location=(p.x, p.y, z + 0.0011))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, 1.0)
            base, rough, alpha = (0.90, 0.89, 0.84), 0.70, 1.0

        o.name = f"Obs{i}_{p.kind}"
        o.rotation_euler = (0.0, 0.0, math.radians(p.rot_deg))
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)
        o.pass_index = 0          # scene.py overrides

        mat = bpy.data.materials.new(f"Obs{p.kind}")
        mat.use_nodes = True
        b = mat.node_tree.nodes.get("Principled BSDF")
        set_input(b, "Base Color", base + (1.0,))
        set_input(b, "Roughness", rough)
        if alpha < 1.0:
            set_input(b, "Alpha", alpha)
            mat.blend_method = "BLEND"
        o.data.materials.append(mat)
        made.append((o, {"kind": p.kind, "w": p.w, "h": p.h,
                         "rot_deg": p.rot_deg}))
    return made
```

Add `import math` to `world.py` if it is not already imported.

- [ ] **Step 7: Wire into the scene**

In `scene.py`, inside the open-case loop after building the proxy:

```python
                poses = B.sample_obstructions(placement_rect,
                                              cfg.obstruction, rng)
                item.obstruction_objects = [
                    o for o, _ in W.build_obstructions(poses, hi.z, rng)]
                meta.setdefault("obstructions", []).extend(
                    m for _, m in W.build_obstructions.__wrapped__(poses, hi.z, rng)
                ) if False else meta.setdefault("obstructions", []).extend(
                    [{"kind": p.kind, "w": p.w, "h": p.h} for p in poses])
```

Simplify that to a single call — build once, keep both halves:

```python
                poses = B.sample_obstructions(placement_rect,
                                              cfg.obstruction, rng)
                built = W.build_obstructions(poses, hi.z, rng)
                item.obstruction_objects = [o for o, _ in built]
                meta.setdefault("obstructions", []).extend(m for _, m in built)
```

And in the pass-index block, extend the loop from Task 4 to cover the list:

```python
        for obj in getattr(item, "obstruction_objects", None) or []:
            pid += 1
            obj.pass_index = pid
            id_meta[pid] = {"class": "obstruction", "asset": item.asset,
                            "variant": item.variant, "role": "obstruction"}
            objects_by_id[pid] = [obj]
```

Add `obstruction_objects` to `assets.Item`, defaulting to `None`.

- [ ] **Step 8: Render and look**

Run the 6-scene render and `verify3d` again. Roughly 60 % of the open cases should show white blobs, pads or strips on the bay plane, each with its own box.

- [ ] **Step 9: Commit**

```bash
git add recog/synth3d/bay.py recog/synth3d/config.py recog/synth3d/world.py recog/synth3d/scene.py recog/synth3d/assets.py configs/synth3d.yaml configs/synth3d.json tests/test_bay.py
git commit -m "feat(synth3d): add bay obstructions as a labelled class

Adhesive blobs, foam pads, tape strips and printed labels, sampled per
bay with 40% left clean. All four are in every opened case in IMG_4426
and none is in the CAD.

They sit on the bay proxy and occlude it, so they subtract themselves
from placement_area without any mask arithmetic."
```

---

### Task 6: Seat cells in bays

**Files:**
- Modify: `recog/synth3d/bay.py` (add `seated_cell_poses`)
- Modify: `recog/synth3d/scene.py` (seat a subset of cells)
- Modify: `recog/synth3d/config.py` (`LayoutCfg.p_seated`, `seated_frac`)
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `common.packing.first_fit_decreasing`, `bay.placement_rect_in_footprint`
- Produces: `bay.seated_cell_poses(placement_rect, cell_w, cell_h, n, rng) -> List[Tuple[float, float, float]]` — up to `n` `(x, y, rot_deg)` centres for cells seated in the bay at the pitch the packer would choose.

The deployed system fills cartridges one cell at a time, so the camera sees partly-filled bays for most of every run. `layout.plan` already lets parts overlap and lifts them via `Placement.z` (`max_overlap_iou: 0.20` in `configs/synth3d.yaml`), so cells already land on cartridges — but at random positions and angles. What is missing is the *seated* case: axis-aligned, in the bay, at packer pitch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bay.py`:

```python
from recog.synth3d.bay import seated_cell_poses

CELL_W, CELL_H = 0.0183, 0.065          # 18650 in metres


def test_seated_cells_lie_inside_the_placement_rect():
    rect = (0.0, 0.0, 0.055, 0.065)
    for seed in range(100):
        poses = seated_cell_poses(rect, CELL_W, CELL_H, 3,
                                  random.Random(seed))
        for x, y, rot in poses:
            hw, hh = (CELL_W / 2, CELL_H / 2) if rot % 180 == 0 \
                else (CELL_H / 2, CELL_W / 2)
            assert rect[0] - 1e-9 <= x - hw and x + hw <= rect[2] + 1e-9
            assert rect[1] - 1e-9 <= y - hh and y + hh <= rect[3] + 1e-9


def test_seated_cells_do_not_overlap_each_other():
    rect = (0.0, 0.0, 0.055, 0.065)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 3, random.Random(0))
    boxes = []
    for x, y, rot in poses:
        hw, hh = (CELL_W / 2, CELL_H / 2) if rot % 180 == 0 \
            else (CELL_H / 2, CELL_W / 2)
        boxes.append((x - hw, y - hh, x + hw, y + hh))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            assert ox <= 1e-9 or oy <= 1e-9


def test_requesting_more_cells_than_fit_returns_only_what_fits():
    rect = (0.0, 0.0, 0.055, 0.065)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 99, random.Random(0))
    assert 0 < len(poses) <= 3          # 55mm / 18.3mm = 3 across


def test_zero_requested_returns_empty():
    rect = (0.0, 0.0, 0.055, 0.065)
    assert seated_cell_poses(rect, CELL_W, CELL_H, 0, random.Random(0)) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bay.py -v -k seated`
Expected: FAIL — `ImportError: cannot import name 'seated_cell_poses'`

- [ ] **Step 3: Implement**

Append to `recog/synth3d/bay.py`:

```python
def seated_cell_poses(placement_rect: Rect, cell_w: float, cell_h: float,
                      n: int, rng: random.Random):
    """Up to `n` cell centres seated in the bay at the packer's pitch.

    The deployed system fills a cartridge one cell at a time, so the
    camera sees partly-filled bays for most of every run. A segmenter
    trained only on empty bays would first meet a partly-filled one
    during the operation it exists to support.

    Positions come from the SAME FFDH packer the planner uses, so the
    synthetic partly-filled bay matches what the planner would actually
    produce rather than an invented arrangement.

    Returns [(x, y, rot_deg), ...] with rot_deg in {0, 90}.
    """
    from common.packing import Item as _PackItem
    from common.packing import first_fit_decreasing

    if n <= 0:
        return []

    x0, y0, x1, y1 = placement_rect
    strip_w, strip_h = x1 - x0, y1 - y0

    items = [_PackItem(i, cell_w, cell_h) for i in range(n)]
    res = first_fit_decreasing(items, strip_w, strip_h, allow_rotation=True)

    out = []
    for p in res.placements:
        out.append((
            x0 + p.x + p.width / 2,
            y0 + p.y + p.height / 2,
            90.0 if p.rotated else 0.0,
        ))
    rng.shuffle(out)
    return out[:n]
```

Units are the caller's and must be consistent — this passes metres straight to `first_fit_decreasing`, whose docstring says millimetres. That is fine because the function is scale-free, but do not mix the two in one call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bay.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Seat cells in the scene**

Add to `LayoutCfg` in `config.py`:

```python
    # Fraction of open cartridges that get cells seated in the bay, and how
    # full those bays are. The deployed camera sees partly-filled cartridges
    # for most of every run; a set of only-empty bays would not contain the
    # case the segmenter exists to handle.
    p_seated: float = 0.5
    seated_frac: Tuple[float, float] = (0.15, 0.85)
```

Add `seated_frac` to `_TUPLE_FIELDS`. Add both to `configs/synth3d.yaml`'s `layout` block and re-run `python -m recog.sync_config`.

In `scene.py`, after obstructions are built, seat cells by cloning the asset's own cell template so the seated cells are real 18650 geometry with the same materials, and give them the `battery` class:

```python
                if rng.random() < cfg.layout.p_seated:
                    cap = max(1, int(
                        (placement_rect[2] - placement_rect[0]) *
                        (placement_rect[3] - placement_rect[1]) /
                        (0.0183 * 0.065)))
                    want = max(1, int(cap * rng.uniform(*cfg.layout.seated_frac)))
                    seats = B.seated_cell_poses(
                        placement_rect, 0.0183, 0.065, want, rng)
                    item.seated_objects = W.seat_cells(
                        library, item.asset, seats, hi.z, rng)
```

`W.seat_cells` clones one `cell`-role object from the asset template per seat, places it at the given centre with the given Z rotation, and rests it on `hi.z`. Reuse `assets.clone` and `assets.lay_flat`; do not build a new cylinder, or the seated cells will not match the loose ones the detector already sees.

Give each seated object its own `pass_index` with `{"class": "battery", ...}` in the same pass-index block.

- [ ] **Step 6: Render and look**

Run the 6-scene render with `--variant open_case` and inspect the contact sheet. About half the open cases should show cells sitting in the bay, axis-aligned, with the bay's own box shrinking to the free floor around them.

**That shrinking is the whole point of this plan.** If the `placement_area` box still spans the full bay with cells sitting on it, the proxy is rendering in front of the cells rather than behind them — check the Z offsets in `build_bay_proxy` against `seat_cells`.

- [ ] **Step 7: Commit**

```bash
git add recog/synth3d/bay.py recog/synth3d/config.py recog/synth3d/scene.py recog/synth3d/world.py configs/synth3d.yaml configs/synth3d.json tests/test_bay.py
git commit -m "feat(synth3d): seat cells in bays at the packer's own pitch

layout.plan already lets parts overlap and lifts them via Placement.z,
so cells already landed on cartridges - but at random positions and
angles. The deployed system produces cells seated in the bay,
axis-aligned, at the pitch FFDH chose.

Positions come from the same packer the planner uses, so a synthetic
partly-filled bay matches what the planner would really produce. Seated
cells occlude the bay proxy, which shrinks placement_area to the free
floor - the case the segmenter exists to handle and the one the training
set previously did not contain."
```

---

### Task 7: RLE masks and the COCO sidecar

**Files:**
- Modify: `recog/synth3d/annotate.py`
- Modify: `recog/generate3d.py`
- Test: `tests/test_annotate_masks.py` (create)

**Interfaces:**
- Consumes: the `(H, W) int32` index map from `render.render_index_map`, `id_meta`, `seg_class_ids()`
- Produces:
  - `annotate.rle_encode(mask: np.ndarray) -> dict` — `{"size": [h, w], "counts": [int, ...]}`, column-major run lengths starting with a zero run, matching the COCO uncompressed RLE convention.
  - `annotate.rle_decode(rle: dict) -> np.ndarray`
  - `annotate.masks_from_index(ids, id_meta, seg_class_ids, cfg) -> Tuple[List[dict], List[dict]]` — like `boxes_from_mask` but each annotation also carries `"segmentation"`.
  - `annotate.write_coco_json(path, images, annotations, seg_class_ids) -> None`

Blender's bundled Python has no pycocotools, so RLE is encoded by hand. Column-major is not a stylistic choice — it is what the COCO format specifies, and a row-major encoding will decode transposed in every downstream reader.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_annotate_masks.py`:

```python
"""RLE encoding and the COCO sidecar. No bpy, no Blender."""
from __future__ import annotations

import json

import numpy as np
import pytest

from recog.synth3d.annotate import (masks_from_index, rle_decode, rle_encode,
                                    write_coco_json)


class _Cfg:
    min_px = 80
    min_side = 6
    min_visibility = 0.25
    drop_truncated = False


SEG_IDS = {"battery": 1, "cartridge": 2, "electronics_module": 3,
           "placement_area": 4, "obstruction": 5}


def test_rle_round_trips_an_empty_mask():
    m = np.zeros((7, 5), dtype=np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_round_trips_a_full_mask():
    m = np.ones((7, 5), dtype=np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_round_trips_a_patterned_mask():
    rng = np.random.default_rng(3)
    for _ in range(50):
        m = (rng.random((13, 17)) < 0.4).astype(np.uint8)
        assert np.array_equal(rle_decode(rle_encode(m)), m)


def test_rle_counts_start_with_a_zero_run_when_the_first_pixel_is_set():
    """COCO's convention: counts always begin with a background run."""
    m = np.ones((2, 2), dtype=np.uint8)
    assert rle_encode(m)["counts"][0] == 0


def test_rle_is_column_major():
    """COCO RLE runs down columns. A row-major encoder decodes
    transposed everywhere downstream and the error is silent."""
    m = np.zeros((2, 3), dtype=np.uint8)
    m[0, 0] = 1                       # single pixel, top-left
    counts = rle_encode(m)["counts"]
    assert counts[:2] == [0, 1], counts


def test_masks_carry_segmentation_alongside_boxes():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:35, 5:35] = 1
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1
    a = anns[0]
    assert a["bbox_xyxy"] == [5, 5, 35, 35]
    assert a["category_id"] == 4
    assert np.array_equal(rle_decode(a["segmentation"]), (ids == 1))


def test_placement_area_is_exempt_from_the_size_filters():
    """A nearly-full cartridge has a small, thin sliver of free floor.
    That is exactly what min_px / min_side / min_visibility discard, and
    it is exactly the cartridge where remaining room matters most."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[10:13, 10:12] = 1              # 6 px, 2 px on the short side
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert len(anns) == 1, "placement_area must survive the filters"
    assert dropped == []


def test_other_classes_still_obey_the_size_filters():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[10:13, 10:12] = 1
    meta = {1: {"class": "obstruction", "asset": "A", "variant": "v"}}
    anns, dropped = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns == []
    assert dropped and dropped[0]["class"] == "obstruction"


def test_a_fully_occluded_bay_yields_no_instance():
    """Zero visible pixels means no free floor, which is correct."""
    ids = np.zeros((40, 40), dtype=np.int32)
    meta = {1: {"class": "placement_area", "asset": "A", "variant": "v"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    assert anns == []


def test_coco_json_has_the_five_categories_and_parses(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:18, 2:18] = 1
    meta = {1: {"class": "cartridge", "asset": "A", "variant": "v"}}
    anns, _ = masks_from_index(ids, meta, SEG_IDS, _Cfg())
    out = tmp_path / "instances_seg.json"
    write_coco_json(
        str(out),
        images=[{"id": 0, "file_name": "x.png", "width": 20, "height": 20}],
        annotations=[dict(a, image_id=0, id=i + 1)
                     for i, a in enumerate(anns)],
        seg_class_ids=SEG_IDS,
    )
    doc = json.loads(out.read_text())
    assert [c["name"] for c in doc["categories"]] == list(SEG_IDS)
    assert doc["annotations"][0]["segmentation"]["size"] == [20, 20]
    assert doc["annotations"][0]["iscrowd"] == 0
    assert doc["annotations"][0]["bbox"] == [2, 2, 16, 16]   # xywh
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_annotate_masks.py -v`
Expected: FAIL — `ImportError: cannot import name 'rle_encode'`

- [ ] **Step 3: Implement RLE**

Append to `recog/synth3d/annotate.py`:

```python
def rle_encode(mask: np.ndarray) -> Dict[str, object]:
    """Uncompressed COCO RLE for a binary mask.

    Blender's bundled Python has no pycocotools, so this is hand-rolled.

    COCO runs are COLUMN-major and always start with a background run,
    which is why a mask whose first pixel is set begins `[0, ...]`. A
    row-major encoder produces a file that every reader accepts and every
    reader decodes transposed - silently.
    """
    m = np.asarray(mask, dtype=np.uint8)
    flat = m.flatten(order="F")
    counts: List[int] = []
    last = 0
    run = 0
    for v in flat:
        if v == last:
            run += 1
        else:
            counts.append(run)
            last = v
            run = 1
    counts.append(run)
    if m.size and flat[0] == 1:
        counts.insert(0, 0)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_decode(rle: Dict[str, object]) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    value = 0
    for run in rle["counts"]:
        flat[pos:pos + run] = value
        pos += run
        value ^= 1
    return flat.reshape((h, w), order="F")
```

The `counts.insert(0, 0)` runs only when the mask is non-empty and starts set; encoding an all-zero mask must not gain a spurious leading zero, which the empty-mask round-trip test checks.

- [ ] **Step 4: Implement `masks_from_index` with the filter exemption**

Append to `recog/synth3d/annotate.py`:

```python
# Classes exempt from the size filters. See the spec's ruling 4: under the
# modal definition a nearly-full cartridge has a small, thin, mostly-occluded
# strip of free floor, which is exactly what min_px / min_side /
# min_visibility discard - and exactly the cartridge where knowing the
# remaining room matters most. Filtering it would teach the segmenter that a
# nearly-full bay has NO placement area rather than a small one.
#
# A bay with zero visible free floor still yields nothing, because it never
# appears in np.unique. That case is correct: there is no room.
_UNFILTERED = frozenset({"placement_area"})


def masks_from_index(ids: np.ndarray, id_meta: Dict[int, dict],
                     class_ids: Dict[str, int], cfg,
                     full_areas: Dict[int, int] = None
                     ) -> Tuple[List[dict], List[dict]]:
    """Like :func:`boxes_from_mask`, but each annotation carries an RLE.

    Kept separate rather than folded into boxes_from_mask so the VOC
    detector path is provably untouched: it still calls the old function
    with the two-class map and gets byte-identical output.
    """
    H, W = ids.shape
    anns: List[dict] = []
    dropped: List[dict] = []

    for pid in np.unique(ids):
        pid = int(pid)
        if pid <= 0:
            continue
        meta = id_meta.get(pid)
        if meta is None:
            continue
        cls = meta.get("class")
        if cls not in class_ids:
            dropped.append({"pass_index": pid, "class": cls,
                            "reason": "unmapped"})
            continue

        inst = (ids == pid)
        ys, xs = np.nonzero(inst)
        visible_px = int(xs.size)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        truncated = bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H)

        vis_frac = None
        if full_areas and full_areas.get(pid):
            vis_frac = round(visible_px / max(1, full_areas[pid]), 4)

        reason = None
        if cls not in _UNFILTERED:
            if visible_px < cfg.min_px:
                reason = f"visible_px<{cfg.min_px}"
            elif min(w, h) < cfg.min_side:
                reason = f"side<{cfg.min_side}"
            elif cfg.drop_truncated and truncated:
                reason = "truncated"
            elif vis_frac is not None and vis_frac < cfg.min_visibility:
                reason = f"visibility<{cfg.min_visibility}"
        if reason:
            dropped.append({"pass_index": pid, "class": cls,
                            "reason": reason, "visible_px": visible_px})
            continue

        anns.append({
            "pass_index": pid,
            "class": cls,
            "category_id": class_ids[cls],
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_xywh": [x0, y0, w, h],
            "segmentation": rle_encode(inst),
            "area": visible_px,
            "truncated": truncated,
            "visible_fraction": vis_frac,
            "asset": meta.get("asset"),
            "variant": meta.get("variant"),
            "iscrowd": 0,
        })
    return anns, dropped


def write_coco_json(path: str, images: Sequence[dict],
                    annotations: Sequence[dict],
                    seg_class_ids: Dict[str, int]) -> None:
    """Write the segmentation sidecar.

    Sits ALONGSIDE the Pascal-VOC output rather than replacing it. VOC
    has no mask field, and the detector's training path must keep reading
    the two-class VOC files unchanged.
    """
    doc = {
        "info": {"description": "auto-pick synthetic segmentation set"},
        "licenses": [],
        "categories": [{"id": i, "name": n, "supercategory": ""}
                       for n, i in sorted(seg_class_ids.items(),
                                          key=lambda kv: kv[1])],
        "images": list(images),
        "annotations": [
            {"id": a["id"], "image_id": a["image_id"],
             "category_id": a["category_id"],
             "bbox": a["bbox_xywh"], "area": a["area"],
             "segmentation": a["segmentation"], "iscrowd": a.get("iscrowd", 0)}
            for a in annotations
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
```

Add `import json` to `annotate.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_annotate_masks.py -v`
Expected: PASS, 11 tests

- [ ] **Step 6: Emit the sidecar from `generate3d.py`**

In the generation loop, alongside the existing `annotate.boxes_from_mask` call at `recog/generate3d.py:224`, add the segmentation pass and accumulate across scenes:

```python
            anns, dropped = annotate.boxes_from_mask(mask, id_meta, ids,
                                                     cfg.filter, full_areas)
            seg_anns, _ = annotate.masks_from_index(
                mask, id_meta, seg_ids, cfg.filter, full_areas)
```

Accumulate `coco_images` and `coco_annotations` lists across the loop with running integer ids, and after the loop write `os.path.join(root, "instances_seg.json")` via `annotate.write_coco_json`.

Under `--resume`, a skipped scene contributes nothing to the accumulators, so a resumed run would emit a partial sidecar. Guard it: either re-derive the sidecar rows from the per-scene `meta` JSON that `--resume` reloads, or refuse to write the sidecar when any scene was skipped and print a message saying so. Silently writing a short file is the failure mode to avoid.

- [ ] **Step 7: Generate and validate**

Run:
```bash
"$BLENDER" -b --python recog/generate3d.py -- --n 24 --out recog/dev3d --device GPU
python - <<'PY'
import json
d = json.load(open('recog/dev3d/instances_seg.json'))
from collections import Counter
print('images', len(d['images']), 'anns', len(d['annotations']))
names = {c['id']: c['name'] for c in d['categories']}
print(Counter(names[a['category_id']] for a in d['annotations']))
PY
```
Expected: all five category names present, with `placement_area` and `electronics_module` counts roughly equal to the number of open cases.

- [ ] **Step 8: Commit**

```bash
git add recog/synth3d/annotate.py recog/generate3d.py tests/test_annotate_masks.py
git commit -m "feat(synth3d): emit a COCO-RLE segmentation sidecar

masks_from_index mirrors boxes_from_mask but attaches hand-rolled
column-major RLE (Blender's Python has no pycocotools). It is a separate
function so the VOC detector path is provably byte-identical.

placement_area is exempt from min_px / min_side / min_visibility. Under
the modal definition a nearly-full cartridge has a small thin sliver of
free floor - exactly what those filters discard, and exactly the
cartridge where the remaining room matters most. A bay with zero visible
floor still yields nothing, which is correct."
```

---

### Task 8: Visual verification

**Files:**
- Modify: `recog/verify3d.py`
- Modify: `recog/synth3d/render.py` (correct the stale `isolated_areas` docstring)

**Interfaces:**
- Consumes: `instances_seg.json`, `annotate.rle_decode`
- Produces: `contact_sheet.png` with per-class mask overlays

A box around a mask proves nothing about the mask. `verify3d` currently draws boxes; masks need their own look.

- [ ] **Step 1: Add mask overlay to `verify3d`**

Add a `--masks` flag. When set, read `instances_seg.json` instead of the VOC XML, decode each RLE with `annotate.rle_decode`, and alpha-blend a per-class colour over the image:

```python
SEG_COLOURS = {
    "battery": (255, 96, 96),
    "cartridge": (96, 160, 255),
    "electronics_module": (96, 255, 128),
    "placement_area": (255, 208, 64),
    "obstruction": (255, 96, 255),
}
```

Blend at alpha 0.45 so the underlying render stays readable — a fully opaque overlay hides exactly the boundary errors you are looking for.

- [ ] **Step 2: Generate a contact sheet and inspect it**

Run:
```bash
"$BLENDER" -b --python recog/generate3d.py -- --n 32 --out recog/dev3d --device GPU
python -m recog.verify3d --data recog/dev3d --n 16 --masks
```

**Open `recog/dev3d/contact_sheet.png` and check each of these by eye.** None is testable and each has a specific failure mode:

- [ ] Yellow (`placement_area`) covers only the *free* bay floor. If it covers seated cells too, the proxy is rendering in front of them.
- [ ] Green (`electronics_module`) sits on one short side, never centred, never straddling the middle.
- [ ] Yellow and green never overlap. If they do, `placement_rect_in_footprint` is not returning the complement.
- [ ] Magenta (`obstruction`) blobs lie inside the yellow region's outer boundary and punch holes in it.
- [ ] Sealed cartridges show blue only — no yellow, no green.
- [ ] Loose modules outside any cartridge are still green.
- [ ] Roughly 40 % of open bays have no magenta at all.

- [ ] **Step 3: Correct the stale docstring**

`render.isolated_areas`'s docstring (lines 352-365) still claims `--visibility` yields no signal because nothing labelled can occlude anything labelled. That stopped being true when `layout.max_overlap_iou` went live at 0.20, and this plan makes it emphatically false — obstructions sit on the bay, cells sit on the bay, ports sit on the board. Rewrite it to say what is now true, and note that `filter.min_visibility` is live for every class except `placement_area`.

An earlier draft of the design spec re-derived a wrong conclusion from this docstring. Leaving it will cost the next reader the same way.

- [ ] **Step 4: Commit**

```bash
git add recog/verify3d.py recog/synth3d/render.py
git commit -m "feat(synth3d): draw segmentation masks in verify3d

A box around a mask proves nothing about the mask. --masks decodes the
COCO RLE and blends a per-class colour at 0.45 alpha, which is the only
way several of this plan's invariants can be checked at all.

Also corrects isolated_areas' docstring, which still claimed nothing
labelled can occlude anything labelled. That has been false since
max_overlap_iou went live at 0.20, and an earlier draft of the design
spec re-derived a wrong conclusion from it."
```

---

## Acceptance

- [ ] `pytest -q` passes with no regressions against Plan A's baseline.
- [ ] `recog/synth3d/assets/catalog.json` carries `module_bay_mm` for all four assemblies, matching §1.2's measured depths within 0.6 mm.
- [ ] `CLASSES == ["battery", "cartridge"]` and `class_ids() == CLASS_MAP` — the detector cannot regress.
- [ ] A generated `instances_seg.json` contains all five categories with non-zero counts.
- [ ] `placement_area` survives a 6 px, 2 px-wide sliver; `obstruction` at the same size does not.
- [ ] The contact-sheet checklist in Task 8 Step 2 passes by eye, every item.
