# Blender Synthetic Dataset Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give auto-pick's recogniser a photorealistic, CAD-derived training dataset generated offline in Blender, written as Pascal-VOC XML that the existing `BatteryCartridgeDataset` reads with no changes.

**Architecture:** A new `recog/synth3d/` package ported from `d:/dev/partsgen_pipeline/`, preserving its bpy / no-bpy split — `config`, `catalog`, `layout` and `annotate` import no Blender API and are fully unit-tested under pytest; `assets`, `materials`, `world`, `render` and `scene` run only inside Blender. Entry points are `recog/generate3d.py` (runs under `blender -b`) and `recog/verify3d.py` (system Python, draws contact sheets).

**Tech Stack:** Python 3.11 (Blender bundled) / 3.14 (system), Blender 5.0.0 + Cycles + OPTIX, numpy, Pillow, PyYAML (system only).

**Spec:** `docs/superpowers/specs/2026-08-05-blender-synthetic-dataset-design.md`

**Source being ported:** `d:/dev/partsgen_pipeline/` — read files there directly; they are the reference implementation.

## Global Constraints

- **Python 3.11 compatibility is mandatory** for `recog/synth3d/*`. Blender 5.0 bundles Python 3.11.13; the system Python is 3.14.3. No PEP 695 generics (`def f[T]()`), no `type X = ...` statements. Use `from __future__ import annotations` at the top of every module.
- **Never add `import bpy` to** `recog/synth3d/config.py`, `catalog.py`, `layout.py`, `annotate.py`. That boundary is what makes the test suite runnable without Blender. A test enforces it.
- **`recog/synth3d/__init__.py` must not import any submodule.** Docstring and `__version__` only. Importing `assets`/`world`/`render` there would drag `bpy` into pytest and break everything.
- **Labels are 1-based**: `battery=1`, `cartridge=2`. 0 is Faster R-CNN's background head.
- **Box max edges are exclusive.** A 1-pixel object yields a 1×1 box. Zero-area boxes make Faster R-CNN's regression loss go NaN.
- **Blender's pixel buffer is bottom-up, image files are top-down.** Every pixel readback applies `np.flipud`.
- **Blender 5.0 API deltas** (verified against the installed build — see spec §11.1). Guard with `hasattr` so a 4.2 install still works:
  - `scene.cycles.filter_width`, **not** `scene.render.filter_width` (does not exist in 5.0)
  - `scene.compositing_node_group` (a tree from `bpy.data.node_groups.new(name, "CompositorNodeTree")`), **not** `scene.node_tree`
  - render-layer socket is `"Object Index"` in 5.0, `"IndexOB"` in 4.2 — try both
  - `scene.use_nodes = True` still succeeds silently in 5.0 and does nothing useful. Do not rely on it as a signal.
- **Blender's Python has no PyYAML and no Pillow** (verified; numpy is present). Anything running inside Blender reads JSON, not YAML, and cannot composite images.
- **Blender path on this machine:** `C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`. The `Blender 4.2` folder has no executable.
- **Green baseline:** `python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py` → **85 passed, 1 skipped**. Those four files fail to collect because `cv2` is not installed; that is pre-existing and not caused by this work. Every task must keep this baseline green.
- **This project is not a git repository.** `git commit` steps are therefore written as "run the verification command" checkpoints. If you run `git init` first, treat each checkpoint as a commit point.

---

## File Structure

**Created:**

| Path | Responsibility |
| --- | --- |
| `common/packing.py` | Shelf-based FFDH strip packing, moved out of `plan/bin_packing.py` so both `plan/` and `recog/` can use it |
| `configs/synth3d.yaml` | Authored source of truth for render/layout/camera/filter settings and all appearance presets |
| `configs/synth3d.json` | Machine-generated sidecar, read inside Blender where PyYAML is absent |
| `recog/sync_config.py` | Transcribes the YAML to the JSON sidecar |
| `recog/synth3d/__init__.py` | Docstring only — no submodule imports |
| `recog/synth3d/config.py` | Dataclasses, `CLASSES`, `CLASS_RULES`, `VARIANTS`, config loading |
| `recog/synth3d/catalog.py` | `role_of()` regex classification, catalog load, STEP→glTF conversion |
| `recog/synth3d/layout.py` | Scatter placement solver + jig pocket packing |
| `recog/synth3d/annotate.py` | mask → boxes → Pascal-VOC XML |
| `recog/synth3d/assets.py` | glTF import, instancing, variant construction (bpy) |
| `recog/synth3d/materials.py` | Randomized Principled surfaces (bpy) |
| `recog/synth3d/world.py` | Backdrop, jig plate, PCB, lighting, camera (bpy) |
| `recog/synth3d/render.py` | Cycles config, object-index pass, mask readback (bpy) |
| `recog/synth3d/scene.py` | Per-sample orchestration (bpy) |
| `recog/synth3d/assets/` | 4 `.glb` files + `catalog.json`, copied from partsgen |
| `recog/generate3d.py` | Entry point, runs under `blender -b` |
| `recog/verify3d.py` | Contact sheet and sweep sheet, system Python |
| `tests/test_synth3d.py` | Unit tests for the four pure modules |
| `tests/test_packing_move.py` | Guards the `common/packing.py` extraction |

**Modified:**

| Path | Change |
| --- | --- |
| `plan/bin_packing.py` | Algorithm body removed; re-exports from `common/packing.py`; keeps `pack_cartridge` |
| `configs/recognition.yaml` | `img_dir` / `ann_dir` repointed at `recog/dataset3d` |
| `README.md` | New entry-point rows and a dataset-generation section |

---

## Task 1: Lift shelf packing into `common/packing.py`

Pure refactor, no behaviour change. Doing it first means Task 4 has a stable import target.

**Files:**
- Create: `common/packing.py`
- Modify: `plan/bin_packing.py`
- Test: `tests/test_packing_move.py`
- Must keep passing untouched: `tests/test_bin_packing.py`

**Interfaces:**
- Produces: `common.packing.Item(id: int, width: float, height: float)`,
  `common.packing.PackedItem(item, x, y, rotated=False)` with `.width`/`.height` properties,
  `common.packing.PackResult(placements, unplaced_ids, shelf_heights)` with `.count`,
  `common.packing.first_fit_decreasing(items, strip_width, strip_height, allow_rotation=True, forbidden_mask=None, mm_per_cell=1.5, tol=1e-6) -> PackResult`
- Consumes: nothing

- [ ] **Step 1: Write the failing test**

Create `tests/test_packing_move.py`:

```python
"""The FFDH algorithm lives in common/ so both plan/ and recog/ can use it."""
import numpy as np


def test_algorithm_importable_from_common():
    from common.packing import Item, PackedItem, PackResult, first_fit_decreasing
    res = first_fit_decreasing([Item(0, 10, 10)], 100, 100)
    assert res.count == 1
    assert isinstance(res.placements[0], PackedItem)
    assert isinstance(res, PackResult)


def test_plan_reexports_the_same_objects():
    """plan.bin_packing must re-export, not redefine — identity, not equality."""
    from common import packing
    from plan import bin_packing
    for name in ("Item", "PackedItem", "PackResult", "first_fit_decreasing"):
        assert getattr(bin_packing, name) is getattr(packing, name), name


def test_common_packing_does_not_import_plan():
    import common.packing
    src = open(common.packing.__file__, encoding="utf-8").read()
    assert "import plan" not in src
    assert "from plan" not in src


def test_pack_cartridge_stays_in_plan():
    from plan import bin_packing
    assert hasattr(bin_packing, "pack_cartridge")
    import common.packing
    assert not hasattr(common.packing, "pack_cartridge")


def test_forbidden_mask_still_honoured():
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :] = True
    from common.packing import Item, first_fit_decreasing
    res = first_fit_decreasing([Item(0, 5, 5)], 30, 30,
                               forbidden_mask=mask, mm_per_cell=1.5)
    assert res.count == 0
    assert res.unplaced_ids == [0]
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python -m pytest tests/test_packing_move.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'common.packing'`

- [ ] **Step 3: Create `common/packing.py`**

Move — do not retype — these from `plan/bin_packing.py`: the module docstring's algorithm description, `Item`, `PackedItem`, `PackResult`, `_overlaps_forbidden`, `_Shelf`, `first_fit_decreasing`, `_try_place_item`. Keep the bodies byte-identical. Header:

```python
"""Shelf-based First-Fit Decreasing Height (FFDH) 2-D strip packing.

Lives in ``common`` rather than ``plan`` because two callers need it:
the planner packs batteries into a cartridge's placement rectangle, and
:mod:`recog.synth3d.layout` packs part footprints into jig pockets when
generating synthetic scenes. Neither should import the other, so the
shared algorithm sits below both.

Units are whatever the caller uses consistently; the planner works in
millimetres. Properties (Berkey & Wang 1987; Martello, Pisinger & Toth
2000): deterministic, worst-case 1.7 x OPT, O(n log n).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
```

- [ ] **Step 4: Reduce `plan/bin_packing.py` to a re-export plus `pack_cartridge`**

Replace the moved definitions with a re-export. Keep the existing module docstring, adding a line that the algorithm now lives in `common.packing`. Keep `pack_cartridge` and its imports exactly as they are.

```python
from common.packing import (  # noqa: F401  (re-exported for existing callers)
    Item,
    PackedItem,
    PackResult,
    _overlaps_forbidden,
    _try_place_item,
    first_fit_decreasing,
)

__all__ = [
    "Item",
    "PackedItem",
    "PackResult",
    "first_fit_decreasing",
    "pack_cartridge",
]
```

`plan/planner.py:28` imports `Item, PackResult, first_fit_decreasing` from `plan.bin_packing` and `tests/test_bin_packing.py:18` imports those plus `PackedItem`. The re-export keeps both working with zero edits. Do not edit either file.

- [ ] **Step 5: Verify — new tests pass AND the old suite is untouched**

```bash
python -m pytest tests/test_packing_move.py tests/test_bin_packing.py -q
```

Expected: PASS, and `tests/test_bin_packing.py` must show the same count it did before (20 tests). If you changed `tests/test_bin_packing.py` to make it pass, you did this wrong — revert and fix the re-export.

- [ ] **Step 6: Checkpoint — full baseline**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **90 passed, 1 skipped** (85 baseline + 5 new).

---

## Task 2: `recog/synth3d/config.py` and `configs/synth3d.yaml`

**Files:**
- Create: `recog/synth3d/__init__.py`, `recog/synth3d/config.py`, `configs/synth3d.yaml`, `recog/sync_config.py`
- Test: `tests/test_synth3d.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CLASSES: list[str] = ["battery", "cartridge"]`, `class_ids() -> dict[str, int]`
  - `CLASS_RULES: list[tuple[str, str]]`, `ROLE_FALLBACK: str`
  - `Variant(name, keep_roles, label, label_roles, explode, weight)`, `VARIANTS: list[Variant]`
  - `RenderCfg`, `LayoutCfg`, `CameraCfg`, `FilterCfg`, `Config`
  - `Config` fields: `render`, `layout`, `camera`, `filter`, `param_space`, `backdrops`, `lighting`, `materials`, `role_materials`
  - `load_config(path=None) -> Config`
  - `default_config_path() -> pathlib.Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synth3d.py`:

```python
"""Unit tests for the Blender-free half of recog.synth3d.

Every module touched here must import without bpy, so this file runs in
the ordinary pytest environment.
"""
import json
from pathlib import Path

import pytest

from recog.synth3d import config as C


ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ classes ----

def test_class_ids_are_one_based():
    ids = C.class_ids()
    assert ids == {"battery": 1, "cartridge": 2}


def test_class_vocabulary_matches_the_voc_loader():
    """Drift between synth3d and recog.dataset would silently mislabel data."""
    from recog.dataset import CLASS_MAP
    expected = {k: v for k, v in CLASS_MAP.items() if v != 0}
    assert C.class_ids() == expected


def test_num_classes_with_background():
    assert len(C.CLASSES) + 1 == 3


# ------------------------------------------------------- bpy boundary ----

@pytest.mark.parametrize("mod", ["config", "catalog", "layout", "annotate"])
def test_pure_modules_never_import_bpy(mod):
    src = (ROOT / "recog" / "synth3d" / f"{mod}.py").read_text(encoding="utf-8")
    assert "import bpy" not in src, f"{mod}.py must stay Blender-free"


def test_package_init_imports_no_submodules():
    src = (ROOT / "recog" / "synth3d" / "__init__.py").read_text(encoding="utf-8")
    assert "import" not in src.replace("__version__", "")


# ------------------------------------------------------------- config ----

def test_load_config_from_yaml():
    cfg = C.load_config()
    assert cfg.render.res == (1280, 720)
    assert cfg.layout.area == (0.80, 0.45)
    assert "scatter" in cfg.param_space["layout_mode"]
    assert "jig" in cfg.param_space["layout_mode"]
    assert set(cfg.role_materials) == {"case", "cell"}
    assert cfg.lighting["overcast_softbox"]["kind"] == "camera_softbox"


def test_layout_area_matches_render_aspect():
    """A square area under a 16:9 render wastes ~44% of every frame."""
    cfg = C.load_config()
    aspect = cfg.render.res[0] / cfg.render.res[1]
    assert cfg.layout.area[0] / cfg.layout.area[1] == pytest.approx(aspect, rel=0.02)


def test_unknown_top_level_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("render: {res: [8, 8]}\nnonsense_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense_key"):
        C.load_config(p)


def test_json_sidecar_is_used_when_yaml_unavailable(tmp_path, monkeypatch):
    """Inside Blender there is no PyYAML, so the .json sidecar is the path."""
    src = C.load_config()
    j = tmp_path / "synth3d.json"
    j.write_text(json.dumps(C.config_to_dict(src)), encoding="utf-8")
    monkeypatch.setattr(C, "_HAVE_YAML", False)
    cfg = C.load_config(tmp_path / "synth3d.yaml")
    assert cfg.render.res == src.render.res
    assert cfg.lighting.keys() == src.lighting.keys()


def test_stale_sidecar_raises_with_the_fix_command(tmp_path, monkeypatch):
    y = tmp_path / "synth3d.yaml"
    j = tmp_path / "synth3d.json"
    j.write_text("{}", encoding="utf-8")
    y.write_text("render: {}\n", encoding="utf-8")   # written after the json
    import os, time
    os.utime(j, (time.time() - 100, time.time() - 100))
    monkeypatch.setattr(C, "_HAVE_YAML", False)
    with pytest.raises(RuntimeError, match="recog.sync_config"):
        C.load_config(y)


# ----------------------------------------------------------- variants ----

def test_variants_cover_the_three_real_presentations():
    names = {v.name for v in C.VARIANTS}
    assert names == {"assembled", "cells_only", "open_case"}


def test_assembled_labels_the_whole_unit_cartridge():
    v = next(v for v in C.VARIANTS if v.name == "assembled")
    assert v.label == "cartridge"
    assert set(v.keep_roles) == {"cell", "case"}


def test_open_case_labels_roles_separately():
    v = next(v for v in C.VARIANTS if v.name == "open_case")
    assert v.label is None
    assert v.label_roles == {"cell": "battery", "case": "cartridge"}


def test_every_variant_label_is_a_real_class():
    valid = set(C.CLASSES)
    for v in C.VARIANTS:
        if v.label is not None:
            assert v.label in valid, v.name
        for cls in v.label_roles.values():
            assert cls in valid, v.name
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python -m pytest tests/test_synth3d.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'recog.synth3d'`

- [ ] **Step 3: Create the package marker**

`recog/synth3d/__init__.py` — this exact content, nothing more:

```python
"""Blender-based synthetic detection-dataset generator.

Submodules deliberately are NOT imported here: config, catalog, layout and
annotate must stay importable without Blender, while assets, materials,
world, render and scene require bpy. Import what you need directly.
"""

__version__ = "1.0.0"
```

- [ ] **Step 4: Write `configs/synth3d.yaml`**

```yaml
# Synthetic 3-D scene generation. Authored here; `python -m recog.sync_config`
# transcribes this to configs/synth3d.json, which is what Blender reads
# (Blender's bundled Python has no PyYAML).

render:
  res: [1280, 720]          # matches recog/dataset so both sources interchange
  samples: 192
  adaptive_threshold: 0.01
  denoise: true
  device: GPU               # OPTIX on the RTX 3060; falls back to CPU
  view_transform: AgX
  exposure: 0.0
  max_bounces: 12
  clamp_indirect: 10.0
  film_transparent: false
  persistent_data: true

layout:
  area: [0.80, 0.45]        # metres; 16:9 to match render aspect, no wasted frame
  mode: scatter
  pad: 0.008
  max_tries: 500
  jitter_deg: 2.0           # hard cap on off-axis tilt; keeps footprints AABB
  allow_90s: true
  jig_clearance: 0.004      # metres of pocket margin around each part
  jig_jitter_deg: 1.0       # smaller than scatter: must stay inside the pocket
  jig_depth: [0.006, 0.012]

camera:
  ortho: true
  height: 0.90
  margin_range: [1.02, 1.10]
  shift_range: [-0.006, 0.006]
  focal: 50.0

filter:
  min_px: 80
  min_side: 6
  min_visibility: 0.25
  drop_truncated: false

param_space:
  n_assemblies: [1, 4]
  layout_mode:
    scatter: 0.7            # domain randomization: learn shape, not context
    jig: 0.3                # in-distribution coverage of the real fixture
  backdrop: [concrete, brushed_metal, fabric, paper, conveyor_belt]
  lighting: [overcast_softbox, harsh_inspection, warm_indoor]

# Which materials each CAD role may draw from.
role_materials:
  case: [shell_white, shell_black, shell_navy, shell_alu]
  cell: [cell_green, cell_blue, cell_black, cell_nickel, cell_purple, cell_grey]

# Ranges are [low, high], sampled per instance, so no two renders share a
# surface. Colours are [[r,g,b] low, [r,g,b] high] in linear space.
materials:
  shell_white:
    color: [[0.68, 0.68, 0.67], [0.86, 0.86, 0.84]]
    metallic: [0.0, 0.10]
    roughness: [0.30, 0.55]
    coat: [0.15, 0.55]
    wear: [0.05, 0.30]
  shell_black:
    color: [[0.015, 0.015, 0.018], [0.055, 0.055, 0.06]]
    metallic: [0.0, 0.15]
    roughness: [0.28, 0.60]
    coat: [0.15, 0.60]
    wear: [0.05, 0.40]
  shell_navy:
    color: [[0.03, 0.05, 0.13], [0.07, 0.11, 0.24]]
    metallic: [0.0, 0.12]
    roughness: [0.30, 0.55]
    coat: [0.15, 0.55]
    wear: [0.05, 0.30]
  shell_alu:
    color: [[0.55, 0.56, 0.58], [0.74, 0.75, 0.77]]
    metallic: [0.85, 1.0]
    roughness: [0.20, 0.45]
    coat: [0.0, 0.0]
    wear: [0.10, 0.45]
  cell_green:
    color: [[0.06, 0.30, 0.07], [0.14, 0.42, 0.13]]
    metallic: [0.05, 0.25]
    roughness: [0.28, 0.48]
    coat: [0.20, 0.60]
    wear: [0.05, 0.35]
  cell_blue:
    color: [[0.04, 0.12, 0.38], [0.10, 0.22, 0.55]]
    metallic: [0.05, 0.25]
    roughness: [0.26, 0.46]
    coat: [0.20, 0.60]
    wear: [0.05, 0.35]
  cell_black:
    color: [[0.02, 0.02, 0.024], [0.06, 0.06, 0.065]]
    metallic: [0.0, 0.20]
    roughness: [0.32, 0.58]
    coat: [0.10, 0.50]
    wear: [0.05, 0.40]
  cell_nickel:
    color: [[0.62, 0.63, 0.65], [0.80, 0.81, 0.83]]
    metallic: [0.95, 1.0]
    roughness: [0.12, 0.35]
    coat: [0.0, 0.0]
    wear: [0.05, 0.45]
  # Purple and grey wraps are both visible in the real photos (IMG_4426).
  cell_purple:
    color: [[0.26, 0.16, 0.42], [0.42, 0.28, 0.62]]
    metallic: [0.05, 0.25]
    roughness: [0.28, 0.50]
    coat: [0.20, 0.60]
    wear: [0.05, 0.35]
  cell_grey:
    color: [[0.38, 0.38, 0.40], [0.58, 0.58, 0.60]]
    metallic: [0.05, 0.30]
    roughness: [0.30, 0.55]
    coat: [0.10, 0.50]
    wear: [0.05, 0.40]

# "image": path to a texture, or null to use the procedural generator.
backdrops:
  concrete:
    image: null
    proc: concrete
    uv_scale: [1.5, 4.0]
    brightness: [-0.06, 0.06]
    roughness: [0.55, 0.85]
    bump: [0.05, 0.25]
  brushed_metal:
    image: null
    proc: brushed
    uv_scale: [2.0, 6.0]
    brightness: [-0.04, 0.08]
    roughness: [0.15, 0.40]
    bump: [0.02, 0.10]
  fabric:
    image: null
    proc: fabric
    uv_scale: [6.0, 18.0]
    brightness: [-0.05, 0.05]
    roughness: [0.70, 0.95]
    bump: [0.10, 0.35]
  paper:
    image: null
    proc: paper
    uv_scale: [1.0, 3.0]
    brightness: [-0.03, 0.10]
    roughness: [0.60, 0.90]
    bump: [0.01, 0.08]
  conveyor_belt:
    image: null
    proc: belt
    uv_scale: [3.0, 8.0]
    brightness: [-0.08, 0.02]
    roughness: [0.50, 0.80]
    bump: [0.15, 0.45]

lighting:
  overcast_softbox:
    kind: camera_softbox
    energy: [120.0, 260.0]
    size: [0.9, 1.6]
    kelvin: [5600, 6800]
    offset: [0.0, 0.10]
    world_strength: [0.25, 0.55]
    world_kelvin: [6000, 8000]
    hdri: null
    hdri_strength: [0, 0]
    hdri_rotation: [0, 0]
  harsh_inspection:
    kind: camera_softbox
    energy: [400.0, 750.0]
    size: [0.10, 0.25]
    kelvin: [5000, 6000]
    offset: [0.0, 0.05]
    world_strength: [0.02, 0.12]
    world_kelvin: [5500, 7000]
    hdri: null
    hdri_strength: [0, 0]
    hdri_rotation: [0, 0]
  warm_indoor:
    kind: camera_softbox
    energy: [140.0, 300.0]
    size: [0.4, 0.9]
    kelvin: [2900, 3800]
    offset: [0.05, 0.22]
    world_strength: [0.15, 0.40]
    world_kelvin: [3200, 4500]
    hdri: null
    hdri_strength: [0, 0]
    hdri_rotation: [0, 0]
```

- [ ] **Step 5: Write `recog/synth3d/config.py`**

Port from `d:/dev/partsgen_pipeline/partsgen/config.py`. Keep the dataclasses; replace module-level preset dicts with YAML/JSON loading. `CLASS_RULES` carries over verbatim — roles stay `cell`/`case` because they name CAD geometry; `Variant.label_roles` maps role to class.

```python
"""
recog.synth3d.config - every tunable in one place. No bpy import, so this
module can be read and tested outside Blender.

Units are METRES throughout. The CAD is millimetres; the converter writes
glTF in metres and records that in catalog.json.

Presets live in configs/synth3d.yaml. Blender's bundled Python has no
PyYAML, so a JSON sidecar (configs/synth3d.json, written by
`python -m recog.sync_config`) is read instead when yaml is unavailable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _HAVE_YAML = True
except ImportError:            # Blender's bundled Python
    _HAVE_YAML = False


# =========================================================================== #
#  CLASSES
#
#  FasterRCNN reserves label 0 for background, so ids start at 1. These must
#  stay identical to recog.dataset.CLASS_MAP - a test enforces it.
# =========================================================================== #

CLASSES: List[str] = ["battery", "cartridge"]


def class_ids() -> Dict[str, int]:
    return {c: i + 1 for i, c in enumerate(CLASSES)}


# Sub-part name -> semantic ROLE (not class). Matched in order, first hit wins.
# Roles describe CAD geometry and match the real sub-part names, e.g.
# "004695_A;1-Cell_18651" and "004697_A;2-Case10000_top".
CLASS_RULES: List[Tuple[str, str]] = [
    # NX increments instance names, so cells appear as Cell_18650, Cell_18651,
    # Cell_18650_18652 ... - match "Cell_" + digits, not a literal.
    (r"Cell[_ ]?\d+", "cell"),
    (r"Case.*_(top|btm)", "case"),
]

ROLE_FALLBACK = "case"          # unmatched sub-parts are treated as shell


@dataclass
class Variant:
    """
    How one CAD assembly is presented in a scene.

    keep_roles   which sub-part roles are linked into the scene
    label        class for the whole assembly, or None to label each visible
                 sub-part by its own role
    label_roles  role -> class, used when label is None
    weight       relative sampling probability
    """
    name: str
    keep_roles: Tuple[str, ...] = ("cell", "case")
    label: Optional[str] = None
    label_roles: Dict[str, str] = field(default_factory=dict)
    explode: float = 0.0        # metres of separation for loose sub-parts
    weight: float = 1.0


VARIANTS: List[Variant] = [
    # Sealed unit. Cells are inside the shell and contribute no visible pixels,
    # so the mask pass drops them automatically - no special casing anywhere.
    # Matches the closed black shells in the lower half of IMG_4426.
    Variant("assembled", keep_roles=("cell", "case"), label="cartridge",
            weight=3.0),

    # Shell removed: loose 18650 cells, scattered individually. Matches the
    # top rows of cells in the real photos.
    Variant("cells_only", keep_roles=("cell",), label=None,
            label_roles={"cell": "battery"}, explode=0.030, weight=2.0),

    # Opened unit: shell present, cells visible beside it. Matches the middle
    # pockets of IMG_4426.
    Variant("open_case", keep_roles=("cell", "case"), label=None,
            label_roles={"cell": "battery", "case": "cartridge"},
            explode=0.045, weight=1.0),
]


# =========================================================================== #
#  SCENE CONFIG
# =========================================================================== #

@dataclass
class RenderCfg:
    res: Tuple[int, int] = (1280, 720)
    samples: int = 192
    adaptive_threshold: float = 0.01
    denoise: bool = True
    device: str = "GPU"                  # "CPU" | "GPU"
    view_transform: str = "AgX"
    exposure: float = 0.0
    max_bounces: int = 12
    clamp_indirect: float = 10.0
    film_transparent: bool = False
    persistent_data: bool = True


@dataclass
class LayoutCfg:
    area: Tuple[float, float] = (0.80, 0.45)
    mode: str = "scatter"
    pad: float = 0.008
    max_tries: int = 500
    jitter_deg: float = 2.0
    allow_90s: bool = True
    jig_clearance: float = 0.004
    jig_jitter_deg: float = 1.0
    jig_depth: Tuple[float, float] = (0.006, 0.012)


@dataclass
class CameraCfg:
    ortho: bool = True
    height: float = 0.90
    margin_range: Tuple[float, float] = (1.02, 1.10)
    shift_range: Tuple[float, float] = (-0.006, 0.006)
    focal: float = 50.0


@dataclass
class FilterCfg:
    min_px: int = 80
    min_side: int = 6
    min_visibility: float = 0.25
    drop_truncated: bool = False


@dataclass
class Config:
    render: RenderCfg = field(default_factory=RenderCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    param_space: Dict[str, Any] = field(default_factory=dict)
    backdrops: Dict[str, dict] = field(default_factory=dict)
    lighting: Dict[str, dict] = field(default_factory=dict)
    materials: Dict[str, dict] = field(default_factory=dict)
    role_materials: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def config_to_dict(cfg: Config) -> dict:
    """Round-trippable plain-data form, used to write the JSON sidecar."""
    return cfg.to_dict()


# =========================================================================== #
#  LOADING
# =========================================================================== #

_SECTIONS = {"render": RenderCfg, "layout": LayoutCfg,
             "camera": CameraCfg, "filter": FilterCfg}
_PASSTHROUGH = ("param_space", "backdrops", "lighting",
                "materials", "role_materials")
_TUPLE_FIELDS = {"res", "area", "margin_range", "shift_range", "jig_depth"}


def default_config_path() -> Path:
    """configs/synth3d.yaml, resolved relative to the project root."""
    return Path(__file__).resolve().parents[2] / "configs" / "synth3d.yaml"


def _read_raw(path: Path) -> dict:
    """YAML when available, else the JSON sidecar next to it."""
    if _HAVE_YAML and path.suffix in (".yaml", ".yml") and path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise RuntimeError(
            f"{sidecar} not found and PyYAML is unavailable in this "
            f"interpreter. Run:  python -m recog.sync_config"
        )
    if path.is_file() and sidecar.stat().st_mtime < path.stat().st_mtime:
        raise RuntimeError(
            f"{sidecar} is older than {path.name}; the config is stale. "
            f"Run:  python -m recog.sync_config"
        )
    with sidecar.open("r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def load_config(path: "str | os.PathLike | None" = None) -> Config:
    """Build a Config from configs/synth3d.yaml (or its JSON sidecar)."""
    p = Path(path) if path is not None else default_config_path()
    raw = _read_raw(p)

    unknown = set(raw) - set(_SECTIONS) - set(_PASSTHROUGH)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {p.name}: {sorted(unknown)}. "
            f"Valid keys: {sorted(set(_SECTIONS) | set(_PASSTHROUGH))}"
        )

    kwargs: Dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        valid = {f for f in cls.__dataclass_fields__}
        bad = set(section) - valid
        if bad:
            raise ValueError(f"unknown key(s) in {p.name}:{name}: {sorted(bad)}")
        coerced = {k: (tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v)
                   for k, v in section.items()}
        kwargs[name] = cls(**coerced)
    for name in _PASSTHROUGH:
        kwargs[name] = raw.get(name) or {}

    return Config(**kwargs)
```

- [ ] **Step 6: Write `recog/sync_config.py`**

```python
"""Transcribe configs/synth3d.yaml to the JSON sidecar Blender reads.

Blender's bundled Python has numpy but no PyYAML, so the generator cannot
read the authored YAML directly. Run this after every edit to synth3d.yaml:

    python -m recog.sync_config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from recog.synth3d.config import default_config_path


def sync(yaml_path: Path) -> Path:
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out = yaml_path.with_suffix(".json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    src = Path(args.config) if args.config else default_config_path()
    out = sync(src)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Generate the sidecar and run the tests**

```bash
python -m recog.sync_config
python -m pytest tests/test_synth3d.py -q
```

Expected: PASS, 14 tests. `configs/synth3d.json` exists.

- [ ] **Step 8: Checkpoint**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **104 passed, 1 skipped**.

---

## Task 3: `recog/synth3d/catalog.py` and the CAD assets

**Files:**
- Create: `recog/synth3d/catalog.py`, `recog/synth3d/assets/` (copied binaries)
- Modify: `tests/test_synth3d.py` (append)

**Interfaces:**
- Consumes: `config.CLASS_RULES`, `config.ROLE_FALLBACK`
- Produces: `role_of(subpart_name: str) -> str`, `load_catalog(assets_dir: str) -> dict`, `build_catalog(src_dir, out_dir, ...) -> dict`, `convert_step(src, dst, ...)`, `inspect_glb(path) -> dict`

- [ ] **Step 1: Copy the pre-converted assets**

```bash
mkdir -p recog/synth3d/assets
cp d:/dev/partsgen_pipeline/assets/*.glb recog/synth3d/assets/
cp d:/dev/partsgen_pipeline/assets/catalog.json recog/synth3d/assets/
ls -la recog/synth3d/assets/
```

Expected: 4 `.glb` files (~210 KB each) and `catalog.json`.

- [ ] **Step 2: Write the failing tests — append to `tests/test_synth3d.py`**

```python
# ---------------------------------------------------------- catalog ----

from recog.synth3d import catalog as CAT

ASSETS = ROOT / "recog" / "synth3d" / "assets"


def test_all_four_assets_present():
    cat = CAT.load_catalog(str(ASSETS))
    names = {a["name"] for a in cat["assets"]}
    assert names == {"AnkerPowerCore10000", "AnkerPowerCore13000",
                     "AnkerPowerCore20100", "AnkerPowerCore26800"}
    assert cat["units"] == "m"


def test_role_of_classifies_every_real_subpart_name():
    """All 33 sub-part names from the real CAD must classify correctly."""
    cat = CAT.load_catalog(str(ASSETS))
    seen = 0
    for asset in cat["assets"]:
        for sp in asset["subparts"]:
            role = CAT.role_of(sp["name"])
            assert role == sp["role"], f"{sp['name']}: {role} != {sp['role']}"
            assert role in ("cell", "case")
            seen += 1
    assert seen == 33, f"expected 33 sub-parts, catalogued {seen}"


def test_cell_regex_matches_nx_incremented_names():
    """NX renames instances; a literal 'Cell_18650' match misses these."""
    for name in ("004695_A;1-Cell_18650", "004695_A;2-Cell_18651",
                 "004695_A;3-Cell_18650_18652", "Cell_99999"):
        assert CAT.role_of(name) == "cell", name


def test_case_names_classify_as_case():
    for name in ("004697_A;2-Case10000_top", "004697_A;1-Case26800_btm"):
        assert CAT.role_of(name) == "case", name


def test_unknown_subpart_falls_back_to_case():
    assert CAT.role_of("something_unrecognised") == "case"


def test_expected_cell_counts_per_asset():
    """Cell count is the CAD's ground truth: 3 / 4 / 6 / 8."""
    cat = CAT.load_catalog(str(ASSETS))
    counts = {a["name"]: a["role_counts"].get("cell", 0) for a in cat["assets"]}
    assert counts == {"AnkerPowerCore10000": 3, "AnkerPowerCore13000": 4,
                      "AnkerPowerCore20100": 6, "AnkerPowerCore26800": 8}
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_synth3d.py -q -k catalog or role or cell or case or asset
```

Expected: FAIL — no module `recog.synth3d.catalog`.

- [ ] **Step 4: Create `recog/synth3d/catalog.py`**

Copy `d:/dev/partsgen_pipeline/partsgen/catalog.py` verbatim, changing only the module docstring's header line to `recog.synth3d.catalog` and the import to `from .config import CLASS_RULES, ROLE_FALLBACK`. The `role_of`, `convert_step`, `inspect_glb`, `build_catalog` and `load_catalog` bodies are unchanged.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_synth3d.py -q
```

Expected: PASS, 20 tests.

If `test_role_of_classifies_every_real_subpart_name` fails, the regex and the catalog disagree — the catalog was built by the same `role_of`, so a mismatch means `CLASS_RULES` was altered during the port. Restore it exactly.

- [ ] **Step 6: Checkpoint**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **110 passed, 1 skipped**.

---

## Task 4: `recog/synth3d/layout.py` — scatter and jig

**Files:**
- Create: `recog/synth3d/layout.py`
- Modify: `tests/test_synth3d.py` (append)

**Interfaces:**
- Consumes: `common.packing.Item`, `common.packing.first_fit_decreasing`, `config.LayoutCfg`
- Produces:
  - `Placement(x: float, y: float, quarter: int, rot_deg: float)` with `.as_dict()`
  - `Pocket(x: float, y: float, w: float, h: float, depth: float)` with `.as_dict()`
  - `footprint_after_rotation(fx, fy, quarter) -> tuple[float, float]`
  - `plan(footprints, cfg, rng) -> list[Placement | None]`
  - `plan_jig(footprints, cfg, rng) -> tuple[list[Placement | None], list[Pocket]]`
  - `cluster_offsets(n, spread, rng) -> list[tuple[float, float]]`

- [ ] **Step 1: Write the failing tests — append to `tests/test_synth3d.py`**

```python
# ----------------------------------------------------------- layout ----

import random

from recog.synth3d import layout as L


def _real_footprints():
    """Footprints in metres, from the real catalog extents."""
    cat = CAT.load_catalog(str(ASSETS))
    out = []
    for a in cat["assets"]:
        ex = a["extents_mm"]
        out.append((ex[0] / 1000.0, ex[1] / 1000.0))
    return out


def _aabb(fp, plc, cfg):
    ex, ey = L.footprint_after_rotation(fp[0], fp[1], plc.quarter)
    return (plc.x - ex / 2, plc.y - ey / 2, plc.x + ex / 2, plc.y + ey / 2)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_scatter_never_overlaps_over_300_scenes():
    cfg = C.load_config().layout
    fps = _real_footprints()
    total = 0
    for seed in range(300):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs = L.plan(chosen, cfg, rng)
        boxes = [_aabb(f, p, cfg) for f, p in zip(chosen, plcs) if p is not None]
        total += len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j]), f"seed {seed}"
    assert total > 500, f"only {total} items placed across 300 scenes"


def test_scatter_stays_inside_the_area():
    cfg = C.load_config().layout
    W, H = cfg.area
    fps = _real_footprints()
    for seed in range(200):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        for f, p in zip(chosen, L.plan(chosen, cfg, rng)):
            if p is None:
                continue
            x0, y0, x1, y1 = _aabb(f, p, cfg)
            assert x0 >= -W / 2 - 1e-6 and x1 <= W / 2 + 1e-6, f"seed {seed}"
            assert y0 >= -H / 2 - 1e-6 and y1 <= H / 2 + 1e-6, f"seed {seed}"


def test_rotation_constraint_always_satisfied():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(200):
        rng = random.Random(seed)
        for p in L.plan(fps, cfg, rng):
            if p is None:
                continue
            assert p.quarter in (0, 1, 2, 3)
            off = abs(p.rot_deg - p.quarter * 90)
            assert off <= cfg.jitter_deg + 1e-9, f"tilt {off} exceeds cap"


def test_plan_is_deterministic_for_a_seed():
    cfg = C.load_config().layout
    fps = _real_footprints()
    a = [p.as_dict() if p else None for p in L.plan(fps, cfg, random.Random(7))]
    b = [p.as_dict() if p else None for p in L.plan(fps, cfg, random.Random(7))]
    assert a == b


# -------------------------------------------------------------- jig ----

def test_jig_pockets_contain_their_items():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs, pockets = L.plan_jig(chosen, cfg, rng)
        placed = [(f, p) for f, p in zip(chosen, plcs) if p is not None]
        assert len(pockets) == len(placed), f"seed {seed}"
        for (f, p), pk in zip(placed, pockets):
            ix0, iy0, ix1, iy1 = _aabb(f, p, cfg)
            px0, py0 = pk.x - pk.w / 2, pk.y - pk.h / 2
            px1, py1 = pk.x + pk.w / 2, pk.y + pk.h / 2
            assert px0 <= ix0 + 1e-6 and ix1 <= px1 + 1e-6, f"seed {seed} x"
            assert py0 <= iy0 + 1e-6 and iy1 <= py1 + 1e-6, f"seed {seed} y"


def test_jig_pockets_never_overlap():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(2, 4))]
        _, pockets = L.plan_jig(chosen, cfg, rng)
        boxes = [(p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2)
                 for p in pockets]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j]), f"seed {seed}"


def test_jig_pockets_stay_inside_the_area():
    cfg = C.load_config().layout
    W, H = cfg.area
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        _, pockets = L.plan_jig(chosen, cfg, rng)
        for p in pockets:
            assert p.x - p.w / 2 >= -W / 2 - 1e-6
            assert p.x + p.w / 2 <= W / 2 + 1e-6
            assert p.y - p.h / 2 >= -H / 2 - 1e-6
            assert p.y + p.h / 2 <= H / 2 + 1e-6


def test_jig_clearance_is_actually_applied():
    cfg = C.load_config().layout
    rng = random.Random(0)
    fp = [(0.06, 0.09)]
    plcs, pockets = L.plan_jig(fp, cfg, rng)
    assert plcs[0] is not None and len(pockets) == 1
    pk = pockets[0]
    ex, ey = L.footprint_after_rotation(0.06, 0.09, plcs[0].quarter)
    assert pk.w >= ex + 2 * cfg.jig_clearance - 1e-9
    assert pk.h >= ey + 2 * cfg.jig_clearance - 1e-9


def test_jig_rotation_uses_the_tighter_jitter_cap():
    """Parts sit inside pockets; scatter's 2 deg would sweep past the clearance."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(50):
        rng = random.Random(seed)
        plcs, _ = L.plan_jig(fps, cfg, rng)
        for p in plcs:
            if p is None:
                continue
            assert abs(p.rot_deg - p.quarter * 90) <= cfg.jig_jitter_deg + 1e-9


def test_jig_returns_none_for_items_that_do_not_fit():
    cfg = C.load_config().layout
    rng = random.Random(0)
    plcs, pockets = L.plan_jig([(5.0, 5.0)], cfg, rng)
    assert plcs == [None]
    assert pockets == []


def test_cluster_offsets_ring():
    rng = random.Random(3)
    offs = L.cluster_offsets(6, 0.03, rng)
    assert len(offs) == 6
    for dx, dy in offs:
        assert 0.0 < (dx * dx + dy * dy) ** 0.5 < 0.03 * 1.5
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_synth3d.py -q -k "layout or jig or scatter or cluster or rotation or plan"
```

Expected: FAIL — no module `recog.synth3d.layout`.

- [ ] **Step 3: Create `recog/synth3d/layout.py`**

Copy `Placement`, `footprint_after_rotation`, `plan` and `cluster_offsets` verbatim from `d:/dev/partsgen_pipeline/partsgen/layout.py`, then add `Pocket` and `plan_jig`. Header and new code:

```python
"""
recog.synth3d.layout - axis-aligned placement with guaranteed non-overlap.

No bpy import: this is pure geometry and is unit-tested outside Blender.

Rotations are restricted to k*90 plus a small jitter. That is not just a
stylistic constraint - because every footprint stays axis-aligned, the overlap
test is an exact AABB comparison rather than an approximation, so "no two parts
intersect" is a guarantee rather than a hope.

Two modes:
  plan()      free scatter, for domain randomization
  plan_jig()  shelf-packed pockets, reproducing the real 3-D-printed fixture

plan_jig derives pockets FROM the parts rather than fitting parts into a fixed
grid, so a pocket always fits its part by construction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from common.packing import Item as _PackItem
from common.packing import first_fit_decreasing

MM = 1000.0        # common.packing documents its units as millimetres
```

Then, after `Placement`:

```python
@dataclass(frozen=True)
class Pocket:
    """A recess in the jig plate. Centre-based, metres, layout-local."""
    x: float
    y: float
    w: float
    h: float
    depth: float

    def as_dict(self):
        return {"x": round(self.x, 5), "y": round(self.y, 5),
                "w": round(self.w, 5), "h": round(self.h, 5),
                "depth": round(self.depth, 5)}
```

And the new solver:

```python
def plan_jig(footprints: Sequence[Tuple[float, float]], cfg,
             rng: random.Random
             ) -> Tuple[List[Optional[Placement]], List[Pocket]]:
    """
    Shelf-pack footprints, then emit one Pocket around each placed part.

    Reuses the FFDH packer the planner uses for batteries-into-cartridges;
    a jig plate is the same problem. Deriving pockets from the packed parts
    guarantees fit, unlike packing parts into a fixed pocket grid.

    Returns (placements, pockets). pockets is parallel to the non-None
    placements in input order.
    """
    W, H = cfg.area
    clear = cfg.jig_clearance
    jit = cfg.jig_jitter_deg

    # Inflate by clearance on all sides before packing, so pockets tile without
    # touching. Convert to millimetres for common.packing.
    items = [_PackItem(i, (fx + 2 * clear) * MM, (fy + 2 * clear) * MM)
             for i, (fx, fy) in enumerate(footprints)]
    res = first_fit_decreasing(items, W * MM, H * MM,
                               allow_rotation=cfg.allow_90s)

    placements: List[Optional[Placement]] = [None] * len(footprints)
    pockets_by_index = {}

    for packed in res.placements:
        i = packed.item.id
        # PackedItem.x/.y is the top-left corner in mm, y growing downward.
        # Convert to the centred, y-up frame the rest of the pipeline uses.
        w_m = packed.width / MM
        h_m = packed.height / MM
        cx = -W / 2 + (packed.x / MM) + w_m / 2
        cy = H / 2 - (packed.y / MM) - h_m / 2

        quarter = 1 if packed.rotated else 0
        placements[i] = Placement(
            x=cx, y=cy, quarter=quarter,
            rot_deg=quarter * 90 + rng.uniform(-jit, jit))
        pockets_by_index[i] = Pocket(
            x=cx, y=cy, w=w_m, h=h_m,
            depth=rng.uniform(*cfg.jig_depth))

    pockets = [pockets_by_index[i] for i in sorted(pockets_by_index)]
    return placements, pockets


def rows_in_pocket(n: int, cell_fp: Tuple[float, float], pocket: Pocket,
                   rng: random.Random) -> List[Placement]:
    """
    Lay n identical cells in regular rows inside one pocket.

    Reproduces the top rows of the real photos, where seven cells sit in a
    single tray recess rather than one recess each. Returns at most n
    placements - fewer if the pocket cannot hold them all.
    """
    fx, fy = cell_fp
    cols = max(1, int(pocket.w // fx))
    rows = max(1, int(pocket.h // fy))
    out: List[Placement] = []
    for k in range(min(n, cols * rows)):
        r, c = divmod(k, cols)
        out.append(Placement(
            x=pocket.x - pocket.w / 2 + fx * (c + 0.5),
            y=pocket.y + pocket.h / 2 - fy * (r + 0.5),
            quarter=0,
            rot_deg=rng.uniform(-0.5, 0.5)))
    return out
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_synth3d.py -q
```

Expected: PASS, 31 tests.

If `test_jig_pockets_contain_their_items` fails on the y axis, the top-left-to-centre conversion is wrong: `common.packing` places with y growing **downward** from the strip top, which is why `cy` subtracts from `H/2`.

- [ ] **Step 5: Checkpoint**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **121 passed, 1 skipped**.

---

## Task 5: `recog/synth3d/annotate.py` — mask to boxes to VOC

**Files:**
- Create: `recog/synth3d/annotate.py`
- Modify: `tests/test_synth3d.py` (append)

**Interfaces:**
- Consumes: `config.FilterCfg`
- Produces:
  - `boxes_from_mask(ids, id_meta, class_ids, cfg, full_areas=None) -> tuple[list[dict], list[dict]]`
  - `merge_group_boxes(anns, groups, class_ids, cfg) -> list[dict]`
  - `write_voc_xml(path, filename, width, height, anns, depth=3) -> None`
  - Annotation dicts carry: `pass_index`, `class`, `category_id`, `bbox_xyxy`, `bbox_xywh`, `area`, `truncated`, `visible_fraction`, `asset`, `variant`, `iscrowd`

- [ ] **Step 1: Write the failing tests — append to `tests/test_synth3d.py`**

```python
# --------------------------------------------------------- annotate ----

import numpy as np

from recog.synth3d import annotate as A


def _meta(**over):
    base = {"class": "battery", "asset": "AnkerPowerCore10000",
            "variant": "cells_only", "role": "cell"}
    base.update(over)
    return base


def test_box_edges_are_exclusive():
    """A 1-pixel object must yield a 1x1 box, never a zero-area one."""
    ids = np.zeros((10, 10), dtype=np.int32)
    ids[4, 5] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert len(anns) == 1
    assert anns[0]["bbox_xyxy"] == [5, 4, 6, 5]
    assert anns[0]["bbox_xywh"] == [5, 4, 1, 1]


def test_no_annotation_has_zero_area():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 4, size=(64, 64)).astype(np.int32)
    meta = {i: _meta() for i in (1, 2, 3)}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    for a in anns:
        x0, y0, x1, y1 = a["bbox_xyxy"]
        assert x1 > x0 and y1 > y0


def test_area_is_silhouette_pixels_not_box_area():
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:12, 2:12] = 1          # 100 px box
    ids[2:4, 2:4] = 0            # knock out 4 -> 96 visible
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns[0]["area"] == 96
    x0, y0, x1, y1 = anns[0]["bbox_xyxy"]
    assert (x1 - x0) * (y1 - y0) == 100


def test_sealed_cell_produces_no_annotation():
    """A cell inside an assembled shell contributes zero pixels."""
    ids = np.zeros((32, 32), dtype=np.int32)
    ids[4:28, 4:28] = 1                      # the case
    meta = {1: _meta(**{"class": "cartridge", "role": "case"}),
            2: _meta()}                      # cell id 2 is never drawn
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert [a["pass_index"] for a in anns] == [1]


def test_truncation_flag_on_frame_edge():
    ids = np.zeros((16, 16), dtype=np.int32)
    ids[0:5, 3:9] = 1                        # touches y = 0
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns[0]["truncated"] is True


def test_small_instances_are_dropped_with_a_reason():
    ids = np.zeros((32, 32), dtype=np.int32)
    ids[1, 1] = 1
    cfg = C.FilterCfg(min_px=80, min_side=6)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"].startswith("visible_px<")


def test_merge_collapses_an_assembly_into_one_box():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1                      # shell top
    ids[20:30, 20:30] = 2                    # shell bottom
    meta = {1: _meta(), 2: _meta()}
    for m in meta.values():
        m["class"] = "cartridge"
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), cfg)
    assert len(merged) == 1
    assert merged[0]["bbox_xyxy"] == [5, 5, 30, 30]
    assert merged[0]["area"] == 200          # union of silhouettes, not box area


def test_unmapped_class_is_dropped():
    ids = np.zeros((16, 16), dtype=np.int32)
    ids[2:10, 2:10] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, dropped = A.boxes_from_mask(
        ids, {1: _meta(**{"class": "widget"})}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"] == "unmapped"


# ------------------------------------------------------- VOC output ----

def test_voc_xml_round_trips_through_the_real_loader(tmp_path):
    """The contract is recog.dataset.parse_voc_xml, not a reimplementation."""
    from recog.dataset import CLASS_MAP, parse_voc_xml

    ids = np.zeros((60, 80), dtype=np.int32)
    ids[10:30, 10:40] = 1
    ids[35:50, 50:70] = 2
    meta = {1: _meta(), 2: _meta()}
    meta[2]["class"] = "cartridge"
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)

    xml = tmp_path / "scene_00000.xml"
    A.write_voc_xml(str(xml), "scene_00000.png", 80, 60, anns)

    parsed = parse_voc_xml(xml, CLASS_MAP)
    assert parsed.filename == "scene_00000.png"
    assert parsed.width == 80 and parsed.height == 60
    assert parsed.labels == [1, 2]
    assert parsed.boxes == [(10.0, 10.0, 40.0, 30.0), (50.0, 35.0, 70.0, 50.0)]


def test_voc_survives_an_empty_annotation_list(tmp_path):
    from recog.dataset import CLASS_MAP, parse_voc_xml
    xml = tmp_path / "empty.xml"
    A.write_voc_xml(str(xml), "empty.png", 32, 32, [])
    parsed = parse_voc_xml(xml, CLASS_MAP)
    assert parsed.boxes == [] and parsed.labels == []


def test_voc_writes_integers_not_floats(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:10, 3:12] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    xml = tmp_path / "i.xml"
    A.write_voc_xml(str(xml), "i.png", 20, 20, anns)
    text = xml.read_text(encoding="utf-8")
    assert "." not in text.split("<bndbox>")[1].split("</bndbox>")[0]
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_synth3d.py -q -k "box or voc or merge or mask or truncat or sealed"
```

Expected: FAIL — no module `recog.synth3d.annotate`.

- [ ] **Step 3: Create `recog/synth3d/annotate.py`**

Copy `boxes_from_mask` and `merge_group_boxes` verbatim from `d:/dev/partsgen_pipeline/partsgen/annotate.py`. Do **not** port `split_of`, `coco_skeleton` or `append_to_coco` — `recog/training.py` owns splitting and the output format is VOC. Add the VOC writer:

```python
"""
recog.synth3d.annotate - instance mask -> boxes -> Pascal-VOC XML.

No bpy import, so the whole annotation path is unit-testable outside Blender.
The mask comes from render.py (Blender's object-index pass), but every decision
about what becomes a label is made here.

Output is the VOC dialect recog.dataset.parse_voc_xml reads: <filename>,
<size>/<width|height|depth>, and one <object> per instance with <name> and
<bndbox>. Splitting is NOT done here - recog/training.py random_splits a flat
directory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np
```

Then append:

```python
def write_voc_xml(path: str, filename: str, width: int, height: int,
                  anns: Sequence[dict], depth: int = 3) -> None:
    """
    Write one Pascal-VOC annotation file.

    Coordinates are written as integers because parse_voc_xml casts through
    int(); a float there would raise. Boxes keep their exclusive max edges,
    so a 1-pixel object stays a 1x1 box and never degenerates to zero area
    (which makes FasterRCNN's regression loss go NaN).
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(width))
    ET.SubElement(size, "height").text = str(int(height))
    ET.SubElement(size, "depth").text = str(int(depth))

    for a in anns:
        x0, y0, x1, y1 = a["bbox_xyxy"]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = a["class"]
        ET.SubElement(obj, "truncated").text = "1" if a.get("truncated") else "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x0))
        ET.SubElement(bnd, "ymin").text = str(int(y0))
        ET.SubElement(bnd, "xmax").text = str(int(x1))
        ET.SubElement(bnd, "ymax").text = str(int(y1))

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_synth3d.py -q
```

Expected: PASS, 43 tests.

- [ ] **Step 5: Checkpoint**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **133 passed, 1 skipped**. This is the last task with pytest coverage; Tasks 6–8 are gated by Blender runs.

---

## Task 6: Port the bpy modules — `assets`, `materials`, `render`

The Blender 5.0 API deltas land here. `render.py` is the risky one.

**Files:**
- Create: `recog/synth3d/assets.py`, `recog/synth3d/materials.py`, `recog/synth3d/render.py`

**Interfaces:**
- Consumes: `catalog.load_catalog`, `catalog.role_of`, `config.Variant`, `config.Config`
- Produces:
  - `assets.Item(objects, labels, merge, asset, variant, footprint, local_offsets)`
  - `assets.AssetLibrary(assets_dir)` with `.names()`, `.instantiate(name, variant, rng) -> list[Item]`, `._templates`
  - `assets.group_bbox(objects) -> (Vector, Vector)`, `assets.place_item(item, placement, rng)`, `assets.lay_flat(objects)`, `assets.drop_to_floor(objects)`
  - `materials.set_input(node, name, value) -> bool`, `materials.rng_range(rng, span) -> float`, `materials.rng_color(rng, span)`, `materials.build(preset_name, rng, cfg, name=None)`, `materials.apply_to_object(obj, mat)`, `materials.for_role(role, rng, cfg)`
  - `render.configure_beauty(cfg, out_png)`, `render.enable_gpu()`, `render.render_index_map(cfg, mask_dir, stem) -> np.ndarray`, `render.render_beauty(cfg, out_png)`, `render.save_mask_png(ids, path)`, `render.isolated_areas(cfg, mask_dir, objects_by_id)`, `render.read_index_exr(path, expect_res=None)`

- [ ] **Step 1: Copy `assets.py` unchanged**

Copy `d:/dev/partsgen_pipeline/partsgen/assets.py` to `recog/synth3d/assets.py`. Change only:
- the docstring header to `recog.synth3d.assets`
- `TEMPLATE_COLLECTION = "_synth3d_templates"`

Imports (`from .catalog import load_catalog, role_of`, `from .config import Variant`) already resolve.

- [ ] **Step 2: Port `materials.py` to take config explicitly**

Copy `d:/dev/partsgen_pipeline/partsgen/materials.py`. Remove `from .config import MATERIALS, ROLE_MATERIALS` — those globals no longer exist. Change two signatures to receive the loaded config:

```python
def build(preset_name: str, rng: random.Random, cfg, name: str = None):
    """Return (material, drawn_parameters). cfg is a config.Config."""
    p = cfg.materials[preset_name]
    ...


def for_role(role: str, rng: random.Random, cfg):
    """Draw a material appropriate to a sub-part role."""
    presets = cfg.role_materials.get(role) or cfg.role_materials["case"]
    return build(rng.choice(presets), rng, cfg)
```

Everything else — `set_input`, `rng_range`, `rng_color`, `apply_to_object`, the wear-noise node graph — is unchanged. `set_input`'s `Coat Weight` / `Clearcoat` fallback is confirmed correct on 5.0 (`Coat Weight` is present).

Note `rng_color` returns a 4-tuple and `p["color"]` arrives from JSON/YAML as nested **lists**, not tuples. `zip(lo, hi)` works on lists, so no change needed.

- [ ] **Step 3: Port `render.py` with the Blender 5.0 fixes**

Copy `d:/dev/partsgen_pipeline/partsgen/render.py`, then apply exactly these four changes.

**(a) Filter width — add a helper and use it in both places.** `scene.render.filter_width` does not exist in Blender 5.0.

```python
def _set_filter_width(scene, value: float):
    """
    Blender 5.0 removed scene.render.filter_width; Cycles reads
    scene.cycles.filter_width. Set whichever exists so 4.2 still works.

    This is Invariant #1: at the default 1.5px reconstruction filter Blender
    blends neighbouring object indices at silhouette edges into fractional ids
    that decode to the WRONG instance. Getting this wrong corrupts labels
    silently - the render still looks fine.
    """
    done = False
    if hasattr(scene.cycles, "filter_width"):
        scene.cycles.filter_width = value
        done = True
    if hasattr(scene.render, "filter_width"):
        scene.render.filter_width = value
        done = True
    if not done:
        raise RuntimeError(
            "no filter-width property found; the index pass would blend "
            "object indices at silhouettes and mislabel instances")
```

In `configure_beauty`, replace `s.render.filter_width = 1.5` with `_set_filter_width(s, 1.5)`.
In `_configure_index_render`, replace `s.render.filter_width = 0.01` with `_set_filter_width(s, 0.01)`.

**(b) Compositor — `scene.node_tree` is gone in 5.0.** Replace the body of `_enable_index_output`:

```python
INDEX_NODE_GROUP = "_synth3d_index_comp"


def _compositor_tree(scene):
    """
    Get a writable compositor node tree across Blender 4.x and 5.0.

    4.2: scene.use_nodes = True, then scene.node_tree.
    5.0: the scene compositor became a node GROUP; scene.node_tree no longer
         exists. Note scene.use_nodes = True still SUCCEEDS silently on 5.0
         and does nothing useful, so it cannot be used to detect the version.
    """
    if hasattr(scene, "compositing_node_group"):          # Blender 5.0+
        ng = bpy.data.node_groups.get(INDEX_NODE_GROUP)
        if ng is None:
            ng = bpy.data.node_groups.new(INDEX_NODE_GROUP, "CompositorNodeTree")
        scene.compositing_node_group = ng
        return ng
    scene.use_nodes = True                                 # Blender 4.x
    return scene.node_tree


def _object_index_socket(rl_node):
    """The render-layer socket was renamed 'IndexOB' -> 'Object Index' in 5.0."""
    for key in ("Object Index", "IndexOB"):
        if key in rl_node.outputs:
            return rl_node.outputs[key]
    raise RuntimeError(
        f"no object-index output on the render-layer node; "
        f"available: {[o.name for o in rl_node.outputs]}")


def _enable_index_output(mask_dir: str, stem: str):
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    vl.use_pass_object_index = True
    vl.use_pass_combined = True

    nt = _compositor_tree(scene)
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    out = nt.nodes.new("CompositorNodeOutputFile")
    out.base_path = os.path.abspath(mask_dir)
    out.format.file_format = "OPEN_EXR"
    out.format.color_mode = "BW"
    out.format.color_depth = "32"
    out.format.exr_codec = "ZIP"
    out.file_slots.clear()
    out.file_slots.new(stem + "_")
    nt.links.new(_object_index_socket(rl), out.inputs[0])
    return out
```

Add `import bpy` is already present at the top of the file.

**(c) Detach the compositor before the beauty render.** In `render_beauty`, `scene.use_nodes = False` does nothing on 5.0. Replace the first line:

```python
def render_beauty(cfg, out_png: str):
    scene = bpy.context.scene
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None      # Blender 5.0
    scene.use_nodes = False                      # Blender 4.x
    bpy.context.view_layer.use_pass_object_index = False
    configure_beauty(cfg, out_png)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    bpy.ops.render.render(write_still=True)
```

Leaving the File Output node attached would write a stray EXR on every beauty render.

**(d) Keep the EXR filename fallback.** `render_index_map`'s directory scan when `{stem}_0001.exr` is absent stays exactly as written — the 5.0 File Output slot naming is the one part of this path not yet verified.

- [ ] **Step 4: Verify the three modules import inside Blender**

```bash
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --python-expr "
import sys; sys.path.insert(0, r'd:/dev/auto-pick')
from recog.synth3d import assets, materials, render, config
cfg = config.load_config()
print('IMPORT OK', cfg.render.res, len(cfg.materials), 'materials')
render.enable_gpu()
"
```

Expected: `IMPORT OK (1280, 720) 10 materials` and a `[gpu] backend=OPTIX devices=['NVIDIA GeForce RTX 3060', ...]` line.

If it reports `RuntimeError: configs/synth3d.json not found`, run `python -m recog.sync_config` first.

- [ ] **Step 5: Verify the index-pass plumbing on a trivial scene**

```bash
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --python-expr "
import sys, tempfile, numpy as np
sys.path.insert(0, r'd:/dev/auto-pick')
import bpy
from recog.synth3d import render, config
cfg = config.load_config(); cfg.render.res = (128, 128); cfg.render.device = 'CPU'
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0)); bpy.context.active_object.pass_index = 7
bpy.ops.object.camera_add(location=(0,0,5)); bpy.context.scene.camera = bpy.context.active_object
ids = render.render_index_map(cfg.render, tempfile.mkdtemp(), 'probe')
u = sorted(np.unique(ids).tolist())
print('MASK SHAPE', ids.shape, 'IDS', u)
assert ids.shape == (128,128), ids.shape
assert u == [0, 7], f'expected [0,7] got {u} - filter width or socket wrong'
print('INDEX PASS OK')
"
```

Expected: `MASK SHAPE (128, 128) IDS [0, 7]` then `INDEX PASS OK`.

This is the single most important gate in the plan. A result like `[0, 3, 4, 5, 6, 7]` means the reconstruction filter is blending indices — fix `_set_filter_width`. A `KeyError` means the socket name lookup failed. An `AttributeError` on `node_tree` means `_compositor_tree` was not applied.

- [ ] **Step 6: Confirm the pure tests are still green**

```bash
python -m pytest tests/test_synth3d.py -q
```

Expected: **43 passed** — adding bpy modules must not break the boundary test.

---

## Task 7: `world.py` and `scene.py` — jig, PCB, aspect-aware camera, orchestration

**Files:**
- Create: `recog/synth3d/world.py`, `recog/synth3d/scene.py`

**Interfaces:**
- Consumes: `materials.set_input`, `materials.rng_range`, `materials.for_role`, `layout.plan`, `layout.plan_jig`, `layout.Pocket`, `assets.AssetLibrary`, `assets.place_item`, `assets.group_bbox`, `catalog.role_of`, `config.Config`, `config.VARIANTS`
- Produces:
  - `world.kelvin_to_rgb(k) -> tuple`
  - `world.build_backdrop(name, rng, cfg, size=3.0) -> (plane, drawn)`
  - `world.setup_lighting(preset_name, rng, cam_loc, cfg) -> dict`
  - `world.setup_camera(camera_cfg, layout_cfg, res, rng, top_z=0.0) -> (cam, meta)`
  - `world.build_jig(pockets, layout_cfg, rng) -> (plate_obj, drawn)`
  - `world.build_pcb(bounds_xy, z, rng) -> (obj, drawn)`
  - `scene.reset_scene()`, `scene.pick_variant(rng)`, `scene.sample_params(rng, cfg, overrides=None) -> dict`, `scene.scene_generator(n, seed, cfg, overrides=None)`, `scene.build(params, rng, library, cfg) -> (id_meta, groups, meta)`

- [ ] **Step 1: Port `world.py` — presets from config, aspect-aware camera**

Copy `d:/dev/partsgen_pipeline/partsgen/world.py`. Remove `from .config import BACKDROPS, LIGHTING`. Thread `cfg` through:

```python
def build_backdrop(name: str, rng: random.Random, cfg, size: float = 3.0):
    spec = cfg.backdrops[name]
    ...


def setup_lighting(preset_name: str, rng: random.Random, cam_loc, cfg):
    spec = cfg.lighting[preset_name]
    ...
```

Bodies otherwise unchanged.

**Replace `setup_camera` entirely.** The original assumes a square layout area; this design uses a 16:9 area under a 16:9 render, and getting it wrong either crops parts or wastes half the frame:

```python
def setup_camera(cfg, layout_cfg, res, rng: random.Random, top_z: float = 0.0):
    """
    Bird's-eye camera. A camera with zero rotation already looks down -Z, so a
    top-down view needs no aiming.

    Framing derives from the layout AREA, not from the objects, so scale stays
    consistent across the dataset - a power bank is the same number of pixels in
    every image, which is what detection training wants.

    Blender's ortho_scale always describes the LONGER sensor axis and derives
    the shorter one from the render aspect. So the required scale is whichever
    of (area_w, area_h * aspect) is larger, where aspect = res_x / res_y.
    With area [0.80, 0.45] at 1280x720 both terms are 0.80 - the frame is used
    exactly, with no wasted backdrop.
    """
    margin = rng_range(rng, cfg.margin_range)
    shift_x = rng_range(rng, cfg.shift_range)
    shift_y = rng_range(rng, cfg.shift_range)

    bpy.ops.object.camera_add(location=(shift_x, shift_y, top_z + cfg.height),
                              rotation=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = "TopCam"
    bpy.context.scene.camera = cam

    area_w, area_h = layout_cfg.area
    res_x, res_y = res
    aspect = res_x / res_y

    if cfg.ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = max(area_w, area_h * aspect) * margin
    else:
        cam.data.type = "PERSP"
        cam.data.lens = cfg.focal
        fov = 2 * math.atan(cam.data.sensor_width / (2 * cfg.focal))
        half = max(area_w, area_h * aspect) / 2 * margin
        cam.location.z = top_z + half / math.tan(fov / 2)

    cam.data.clip_start, cam.data.clip_end = 0.001, 100.0
    return cam, {"margin": margin, "shift_x": shift_x, "shift_y": shift_y,
                 "ortho": cfg.ortho, "height": cfg.height,
                 "ortho_scale": getattr(cam.data, "ortho_scale", None)}
```

Then append the two new geometry builders:

```python
def build_jig(pockets, layout_cfg, rng: random.Random):
    """
    Blue 3-D-printed fixture plate with a recess per pocket.

    Built by boolean-differencing pocket cubes out of a slab. The plate is
    UNLABELLED (pass_index 0): it merges with background in the index map and
    produces no annotation, while still correctly occluding what sits behind
    it - the same trick the PCB uses.
    """
    area_w, area_h = layout_cfg.area
    thickness = rng.uniform(0.010, 0.018)
    margin = rng.uniform(0.010, 0.030)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -thickness / 2))
    plate = bpy.context.active_object
    plate.name = "JigPlate"
    plate.scale = (area_w + 2 * margin, area_h + 2 * margin, thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    plate.pass_index = 0

    cutters = []
    for i, pk in enumerate(pockets):
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(pk.x, pk.y, -pk.depth / 2 + 0.0005))
        c = bpy.context.active_object
        c.name = f"_pocket{i}"
        c.scale = (pk.w, pk.h, pk.depth + 0.001)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        cutters.append(c)

    for c in cutters:
        mod = plate.modifiers.new(name=c.name, type="BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.object = c
    bpy.context.view_layer.objects.active = plate
    for c in cutters:
        bpy.ops.object.modifier_apply(modifier=c.name)
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)

    drawn = {"thickness": thickness, "margin": margin,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.06, 0.16),
                       rng.uniform(0.35, 0.62)],
             "roughness": rng.uniform(0.45, 0.75),
             "layer_bump": rng.uniform(0.15, 0.45),
             "n_pockets": len(pockets)}

    mat = bpy.data.materials.new("JigPlastic")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    set_input(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 0.0)

    # 3-D-print layer lines: a striped wave along Z, bumped into the normal.
    coord = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = rng.uniform(180.0, 420.0)
    nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = drawn["layer_bump"]
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    plate.data.materials.append(mat)
    return plate, drawn


def build_pcb(bounds_xy, z: float, rng: random.Random):
    """
    Green PCB with a few extruded components, for the open_case variant.

    The CAD has no PCB, but it is the most distinctive thing inside an opened
    case in the real photos. UNLABELLED (pass_index 0) - it is scene content,
    not a class, and it correctly shrinks the case's visible silhouette the way
    a real board would.
    """
    x0, y0, x1, y1 = bounds_xy
    w = (x1 - x0) * rng.uniform(0.55, 0.80)
    h = (y1 - y0) * rng.uniform(0.20, 0.38)
    cx = (x0 + x1) / 2 + rng.uniform(-0.004, 0.004)
    cy = (y0 + y1) / 2 + rng.uniform(-0.010, 0.010)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z + 0.0008))
    board = bpy.context.active_object
    board.name = "PCB"
    board.scale = (w, h, 0.0016)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    board.pass_index = 0

    drawn = {"w": w, "h": h,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.16, 0.30),
                       rng.uniform(0.04, 0.10)],
             "roughness": rng.uniform(0.25, 0.50),
             "n_components": rng.randint(3, 7)}

    mat = bpy.data.materials.new("PCBGreen")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    set_input(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 0.15)
    board.data.materials.append(mat)

    comp_mat = bpy.data.materials.new("PCBComponent")
    comp_mat.use_nodes = True
    cb = comp_mat.node_tree.nodes.get("Principled BSDF")
    set_input(cb, "Base Color", (0.05, 0.05, 0.055, 1.0))
    set_input(cb, "Roughness", 0.55)

    for k in range(drawn["n_components"]):
        cw = w * rng.uniform(0.06, 0.20)
        ch = h * rng.uniform(0.15, 0.45)
        cz = rng.uniform(0.0015, 0.0045)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx + rng.uniform(-w / 2 + cw, w / 2 - cw),
                      cy + rng.uniform(-h / 2 + ch, h / 2 - ch),
                      z + 0.0016 + cz / 2))
        c = bpy.context.active_object
        c.name = f"PCBComp{k}"
        c.scale = (cw, ch, cz)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        c.pass_index = 0
        c.data.materials.append(comp_mat)
        c.parent = board

    return board, drawn
```

Add `ShaderNodeTexWave` availability is not in the probe list — if `wave.bands_direction` raises on 5.0, drop that line; the default is `X` and still produces stripes.

- [ ] **Step 2: Port `scene.py` with layout-mode dispatch**

Copy `d:/dev/partsgen_pipeline/partsgen/scene.py`. Changes:

```python
from . import assets as A
from . import layout as L
from . import materials as M
from . import world as W
from .catalog import role_of
from .config import VARIANTS, Variant, class_ids
```

`sample_params` now reads the param space off the config and adds `layout_mode`:

```python
def sample_params(rng: random.Random, cfg, overrides: dict = None) -> dict:
    ps = cfg.param_space
    modes = ps["layout_mode"]
    total = sum(modes.values())
    r, acc, chosen = rng.uniform(0, total), 0.0, list(modes)[-1]
    for name, weight in modes.items():
        acc += weight
        if r <= acc:
            chosen = name
            break
    p = {
        "n_assemblies": rng.randint(*ps["n_assemblies"]),
        "backdrop": rng.choice(ps["backdrop"]),
        "lighting": rng.choice(ps["lighting"]),
        "layout_mode": chosen,
    }
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def scene_generator(n: int, seed: int, cfg, overrides: dict = None):
    """Yields (index, params, rng). Same seed + index => identical scene."""
    for i in range(n):
        rng = random.Random((seed * 1_000_003) ^ (i + 1))
        yield i, sample_params(rng, cfg, overrides), rng
```

In `build`, three edits. Material draws pass `cfg`:

```python
        mat, drawn = M.for_role(role, rng, cfg)
```

Placement dispatches on mode and builds the jig:

```python
    # ---- placement -------------------------------------------------------- #
    pockets = []
    if params.get("layout_mode") == "jig":
        placements, pockets = L.plan_jig([it.footprint for it in items],
                                         cfg.layout, rng)
    else:
        placements = L.plan([it.footprint for it in items], cfg.layout, rng)
```

And after the kept-items loop, before pass indices are assigned, add the unlabelled scene geometry:

```python
    # ---- unlabelled scene geometry ---------------------------------------- #
    # Both carry pass_index 0, so they never produce an annotation but do
    # occlude correctly. Built BEFORE pass indices are assigned so the
    # "for o in bpy.data.objects: o.pass_index = 0" reset covers them.
    if pockets:
        _, jig_meta = W.build_jig(pockets, cfg.layout, rng)
        meta["jig"] = jig_meta

    for item in items:
        if item.variant == "open_case" and any(
                role_of(o.name) == "case" for o in item.objects):
            lo, hi = A.group_bbox(item.objects)
            _, pcb_meta = W.build_pcb((lo.x, lo.y, hi.x, hi.y), hi.z, rng)
            meta.setdefault("pcbs", []).append(pcb_meta)
```

Backdrop and lighting calls take `cfg`:

```python
    _, bd = W.build_backdrop(params["backdrop"], rng, cfg, size=plane_size)
    ...
    meta["lighting"] = W.setup_lighting(params["lighting"], rng, cam.location, cfg)
```

Also update `library._templates.clear()`'s comment and `reset_scene` unchanged.

- [ ] **Step 3: Gate — build two scenes with no rendering**

```bash
cd d:/dev/auto-pick
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --python-expr "
import sys, random
sys.path.insert(0, r'd:/dev/auto-pick')
from recog.synth3d import config, scene as S, assets as A
cfg = config.load_config()
lib = A.AssetLibrary(r'd:/dev/auto-pick/recog/synth3d/assets')
for mode in ('scatter', 'jig'):
    rng = random.Random(1)
    params = S.sample_params(rng, cfg, {'layout_mode': mode})
    id_meta, groups, meta = S.build(params, rng, lib, cfg)
    classes = sorted({m['class'] for m in id_meta.values() if m['class']})
    print(mode, 'instances=', len(id_meta), 'classes=', classes,
          'jig=' + str('jig' in meta))
    assert id_meta, mode + ': nothing placed'
    assert all(c in ('battery','cartridge') for c in classes), classes
print('SCENE BUILD OK')
"
```

Expected: two lines then `SCENE BUILD OK`. The `jig` line must report `jig=True`, the `scatter` line `jig=False`.

- [ ] **Step 4: Gate — the mask actually decodes to the right instances**

```bash
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" -b --python-expr "
import sys, random, tempfile, numpy as np
sys.path.insert(0, r'd:/dev/auto-pick')
from recog.synth3d import config, scene as S, assets as A, render, annotate
cfg = config.load_config(); cfg.render.res = (320, 180); cfg.render.device = 'CPU'
lib = A.AssetLibrary(r'd:/dev/auto-pick/recog/synth3d/assets')
rng = random.Random(4)
params = S.sample_params(rng, cfg, None)
id_meta, groups, meta = S.build(params, rng, lib, cfg)
mask = render.render_index_map(cfg.render, tempfile.mkdtemp(), 'g')
present = set(np.unique(mask).tolist()) - {0}
print('mask ids', sorted(present), 'declared', sorted(id_meta))
assert present <= set(id_meta) | {0}, 'mask has ids no object declared'
anns, dropped = annotate.boxes_from_mask(mask, id_meta, config.class_ids(),
                                         cfg.filter)
anns = annotate.merge_group_boxes(anns, groups, config.class_ids(), cfg.filter)
print('boxes', [(a['class'], a['bbox_xyxy']) for a in anns])
assert anns, 'no annotations produced'
print('MASK->BOX OK')
"
```

Expected: mask ids are a subset of the declared ids, at least one box, `MASK->BOX OK`.

`mask has ids no object declared` means index blending is back — recheck `_set_filter_width`.

---

## Task 8: Entry points, wiring, and the first real dataset

**Files:**
- Create: `recog/generate3d.py`, `recog/verify3d.py`
- Modify: `configs/recognition.yaml`, `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: `recog/dataset3d/{images,annotations,meta}/`, `manifest.json`

- [ ] **Step 1: Write `recog/generate3d.py`**

Adapt `d:/dev/partsgen_pipeline/generate.py`. Differences: flat output (no split subdirectories — `recog/training.py` random_splits), VOC instead of COCO, `--sweep`, config from YAML/JSON, non-square `--res`.

```python
"""
Generate the synthetic 3-D dataset. Runs inside Blender.

    BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
    "$BLENDER" -b --python recog/generate3d.py -- --n 20 --out recog/dev3d --res 640 360
    "$BLENDER" -b --python recog/generate3d.py -- --n 2000 --out recog/dataset3d --device GPU --resume

Output is flat Pascal-VOC that recog.dataset.BatteryCartridgeDataset reads
directly; recog/training.py owns the train/val split.

Presets come from configs/synth3d.yaml. Blender has no PyYAML, so run
`python -m recog.sync_config` after editing it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402

from recog.synth3d import annotate, render, scene as S  # noqa: E402
from recog.synth3d.assets import AssetLibrary  # noqa: E402
from recog.synth3d.config import (CLASSES, VARIANTS, class_ids,  # noqa: E402
                                  load_config)


def parse_args(cfg):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=None, help="defaults to recog/synth3d/assets")
    p.add_argument("--out", default="recog/dataset3d")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="scene")
    p.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--device", choices=["CPU", "GPU"], default=None)
    p.add_argument("--backdrop", choices=list(cfg.backdrops), default=None)
    p.add_argument("--lighting", choices=list(cfg.lighting), default=None)
    p.add_argument("--layout-mode", choices=["scatter", "jig"], default=None)
    p.add_argument("--variant", choices=[v.name for v in VARIANTS], default=None)
    p.add_argument("--sweep", choices=["lighting", "backdrop"], default=None,
                   help="render ONE fixed scene once per entry in that axis")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--visibility", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def _dirs(root, save_masks):
    subs = ["images", "annotations", "meta"] + (["masks"] if save_masks else [])
    for s in subs:
        os.makedirs(os.path.join(root, s), exist_ok=True)
    tmp = os.path.join(root, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def run_sweep(a, cfg, library, ids, root, tmp):
    """
    One fixed scene, rendered once per entry in the swept axis.

    Same seed, same layout, same materials, same camera - only the swept axis
    moves. The RNG is redrawn from the SAME seed for each entry so the scene is
    literally identical; re-using a single advanced RNG would drift.
    """
    axis = a.sweep
    entries = list(cfg.lighting if axis == "lighting" else cfg.backdrops)
    print(f"sweeping {axis}: {entries}")
    for entry in entries:
        _, params, rng = next(iter(S.scene_generator(1, a.seed, cfg)))
        params[axis] = entry
        stem = f"sweep_{axis}_{entry}"
        id_meta, groups, meta = S.build(params, rng, library, cfg)
        if not id_meta:
            print(f"[{entry}] nothing placed, skipping")
            continue
        png = os.path.join(root, "images", stem + ".png")
        if not a.no_render:
            render.render_beauty(cfg.render, png)
        meta.update({"sweep_axis": axis, "sweep_entry": entry,
                     "image": stem + ".png"})
        with open(os.path.join(root, "meta", stem + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[sweep] {entry} -> {png}")
    print(f"[done] {root}\nNow tile them:\n"
          f"  python -m recog.verify3d --sweep {root} "
          f"--out {os.path.join(root, axis + '_sheet.png')}")


def main():
    cfg = load_config()
    a = parse_args(cfg)
    if a.config:
        cfg = load_config(a.config)
    if a.res:
        cfg.render.res = (a.res[0], a.res[1])
    if a.samples:
        cfg.render.samples = a.samples
    if a.device:
        cfg.render.device = a.device
    if a.variant:
        VARIANTS[:] = [v for v in VARIANTS if v.name == a.variant]

    ids = class_ids()
    root = os.path.abspath(a.out)
    tmp = _dirs(root, a.save_masks)

    assets_dir = a.assets or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synth3d", "assets")
    library = AssetLibrary(assets_dir)
    print(f"assets: {library.names()}")

    if a.sweep:
        run_sweep(a, cfg, library, ids, root, tmp)
        return

    W, H = cfg.render.res
    overrides = {"backdrop": a.backdrop, "lighting": a.lighting,
                 "layout_mode": a.layout_mode}
    ws, hs = [], []
    per_class = {c: 0 for c in CLASSES}
    per_variant, per_mode, n_drop, n_images = {}, {}, 0, 0

    for i, params, rng in S.scene_generator(a.n, a.seed, cfg, overrides):
        stem = f"{a.prefix}_{i:05d}"
        png = os.path.join(root, "images", stem + ".png")
        xml = os.path.join(root, "annotations", stem + ".xml")
        meta_path = os.path.join(root, "meta", stem + ".json")

        if a.resume and os.path.exists(meta_path) and os.path.exists(xml):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            id_meta, groups, meta = S.build(params, rng, library, cfg)
            if not id_meta:
                print(f"[{i}] nothing placed, skipping")
                continue

            mask = render.render_index_map(cfg.render, tmp, stem)

            full_areas = None
            if a.visibility:
                objs = {int(pid): [bpy.data.objects[n] for n in names
                                   if n in bpy.data.objects]
                        for pid, names in meta["objects_by_id"].items()}
                full_areas = render.isolated_areas(cfg.render, tmp, objs)

            anns, dropped = annotate.boxes_from_mask(mask, id_meta, ids,
                                                     cfg.filter, full_areas)
            anns = annotate.merge_group_boxes(anns, groups, ids, cfg.filter)

            if a.save_masks:
                render.save_mask_png(
                    mask, os.path.join(root, "masks", stem + ".png"))
            if a.save_blend and i == 0:
                bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
            if not a.no_render:
                render.render_beauty(cfg.render, png)

            annotate.write_voc_xml(xml, stem + ".png", W, H, anns)

            meta.pop("objects_by_id", None)
            meta.update({"index": i, "seed": a.seed, "image": stem + ".png",
                         "width": W, "height": H, "annotations": anns,
                         "dropped": dropped, "classes": CLASSES,
                         "class_to_id": ids})
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        n_images += 1
        for ann in meta["annotations"]:
            ws.append(ann["bbox_xywh"][2])
            hs.append(ann["bbox_xywh"][3])
            per_class[ann["class"]] = per_class.get(ann["class"], 0) + 1
            per_variant[ann.get("variant")] = per_variant.get(ann.get("variant"), 0) + 1
        mode = meta["params"].get("layout_mode")
        per_mode[mode] = per_mode.get(mode, 0) + 1
        n_drop += len(meta.get("dropped", []))

        print(f"[{i + 1}/{a.n}] {stem}  {len(meta['annotations'])} boxes  "
              f"({meta['params']['backdrop']}/{meta['params']['lighting']}"
              f"/{mode})")

    stats = {
        "n_images": n_images,
        "n_boxes": len(ws),
        "per_class": per_class,
        "per_variant": per_variant,
        "per_layout_mode": per_mode,
        "dropped_instances": n_drop,
        "box_w_px": {"min": min(ws, default=0), "max": max(ws, default=0),
                     "mean": round(statistics.fmean(ws), 1) if ws else 0},
        "box_h_px": {"min": min(hs, default=0), "max": max(hs, default=0),
                     "mean": round(statistics.fmean(hs), 1) if hs else 0},
    }
    if ws:
        diags = sorted(math.hypot(w, h) for w, h in zip(ws, hs))
        stats["box_diag_px"] = {
            "p05": round(diags[int(0.05 * (len(diags) - 1))], 1),
            "p50": round(diags[len(diags) // 2], 1),
            "p95": round(diags[int(0.95 * (len(diags) - 1))], 1)}

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"classes": CLASSES, "class_to_id": ids,
                   "num_classes_with_background": len(CLASSES) + 1,
                   "seed": a.seed, "config": cfg.to_dict(),
                   "variants": [v.name for v in VARIANTS],
                   "catalog": library.catalog, "stats": stats}, f, indent=2)

    try:
        os.rmdir(tmp)
    except OSError:
        pass

    print(json.dumps(stats, indent=2))
    if stats.get("box_diag_px"):
        d = stats["box_diag_px"]
        print(f"\nAnchor check: FPN defaults cover 32-512px diagonals; "
              f"yours are p05={d['p05']} p50={d['p50']} p95={d['p95']}")
        if d["p95"] > 480:
            print("  WARNING: p95 near the top of the anchor range. Enlarge "
                  "layout.area or widen anchor_scales in configs/recognition.yaml")
    print(f"[done] {root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `recog/verify3d.py`**

Runs in **system** Python — Blender has no Pillow. Reads VOC, draws boxes, tiles.

```python
"""
Draw the generated boxes onto the renders so you can actually look at them.

    python -m recog.verify3d --data recog/dev3d --n 12
    python -m recog.verify3d --sweep sweeps/ --out sweeps/lighting_sheet.png

Inspecting the contact sheet is not optional. A silently-wrong mask pass
produces boxes that look plausible in JSON and are obviously wrong on screen.

System Python only: Blender's bundled interpreter has no Pillow.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

from recog.dataset import CLASS_MAP, parse_voc_xml

COLOURS = {"battery": (60, 220, 90), "cartridge": (255, 90, 60)}
INV = {v: k for k, v in CLASS_MAP.items()}


def draw_one(img_path: Path, xml_path: Path, thick: int = 3) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    if not xml_path.exists():
        return im
    ann = parse_voc_xml(xml_path, CLASS_MAP)
    d = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), label in zip(ann.boxes, ann.labels):
        name = INV.get(label, "?")
        d.rectangle([x0, y0, x1, y1],
                    outline=COLOURS.get(name, (255, 255, 0)), width=thick)
        d.text((x0 + 4, max(0, y0 - 14)), name, fill=COLOURS.get(name))
    return im


def tile(images, cols: int, out: Path, label_texts=None):
    if not images:
        raise SystemExit("nothing to tile")
    rows = math.ceil(len(images) / cols)
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    pad = 26 if label_texts else 4
    sheet = Image.new("RGB", (cols * w, rows * (h + pad)), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        sheet.paste(im, (c * w, r * (h + pad) + pad))
        if label_texts:
            d.text((c * w + 6, r * (h + pad) + 6), label_texts[k],
                   fill=(235, 235, 240))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({len(images)} panels, {sheet.width}x{sheet.height})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="dataset root with images/ + annotations/")
    ap.add_argument("--sweep", default=None, help="sweep root produced by --sweep")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thick", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.sweep:
        root = Path(a.sweep)
        pngs = sorted((root / "images").glob("sweep_*.png"))
        if not pngs:
            raise SystemExit(f"no sweep renders in {root / 'images'}")
        labels = [p.stem.replace("sweep_", "") for p in pngs]
        imgs = [Image.open(p).convert("RGB") for p in pngs]
        out = Path(a.out) if a.out else root / "sweep_sheet.png"
        tile(imgs, min(a.cols, len(imgs)), out, labels)
        return

    if not a.data:
        raise SystemExit("pass --data or --sweep")
    root = Path(a.data)
    pngs = sorted((root / "images").glob("*.png"))[:a.n]
    if not pngs:
        raise SystemExit(f"no images in {root / 'images'}")
    imgs, labels = [], []
    n_boxes = 0
    for p in pngs:
        x = root / "annotations" / (p.stem + ".xml")
        imgs.append(draw_one(p, x, a.thick))
        if x.exists():
            n_boxes += len(parse_voc_xml(x, CLASS_MAP).boxes)
        labels.append(p.stem)
    out = Path(a.out) if a.out else root / "contact_sheet.png"
    tile(imgs, a.cols, out, labels)
    print(f"{n_boxes} boxes across {len(imgs)} images "
          f"({n_boxes / max(1, len(imgs)):.1f} per image)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate a small dev set with no rendering (fast label check)**

```bash
cd d:/dev/auto-pick
BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BLENDER" -b --python recog/generate3d.py -- --n 12 --out recog/dev3d --res 640 360 --no-render
```

Expected: 12 lines of `[k/12] scene_000kk  N boxes (backdrop/lighting/mode)`, then a stats block. `per_class` must contain **both** `battery` and `cartridge` with non-zero counts, and `per_layout_mode` must contain both `scatter` and `jig`. Runs in seconds.

- [ ] **Step 4: Render the dev set and LOOK at it**

```bash
"$BLENDER" -b --python recog/generate3d.py -- --n 12 --out recog/dev3d --res 640 360 --device GPU
python -m recog.verify3d --data recog/dev3d --n 12
```

Then open `recog/dev3d/contact_sheet.png`. Check, in this order:

1. Boxes hug the parts tightly. Loose boxes mean the mask pass is wrong.
2. Power banks lie **flat**, cells lie **on their sides** — not upright. If upright, `assets.lay_flat()` needs checking with `--save-blend`.
3. Cells sealed inside assembled units have **no** box.
4. Roughly a third of scenes show the blue jig plate with parts in recesses.
5. Open cases show a green PCB, and the PCB has no box of its own.

This is a human gate. Do not proceed on green tests alone.

- [ ] **Step 5: Verify the dataset loads through the real training loader**

```bash
python -c "
from recog.dataset import BatteryCartridgeDataset
ds = BatteryCartridgeDataset('recog/dev3d/images', 'recog/dev3d/annotations')
print('images:', len(ds))
img, tgt = ds[0]
print('image:', type(img).__name__, getattr(img, 'shape', None))
print('boxes:', tgt['boxes'].shape if hasattr(tgt['boxes'],'shape') else len(tgt['boxes']))
print('labels:', sorted(set(tgt['labels'].tolist() if hasattr(tgt['labels'],'tolist') else tgt['labels'])))
assert len(ds) == 12
"
```

Expected: 12 images, a CHW float tensor, labels drawn from `{1, 2}`. This proves the whole contract end to end.

- [ ] **Step 6: Tune-loop smoke test**

```bash
"$BLENDER" -b --python recog/generate3d.py -- --sweep lighting --seed 7 --out recog/sweeps --res 640 360 --device GPU
python -m recog.verify3d --sweep recog/sweeps --out recog/sweeps/lighting_sheet.png
```

Expected: three panels (`overcast_softbox`, `harsh_inspection`, `warm_indoor`) showing the **same scene** under different light. If the layouts differ between panels, the RNG is not being redrawn from the same seed per entry.

- [ ] **Step 7: Wire the config and document it**

`configs/recognition.yaml` — repoint the dataset block:

```yaml
dataset:
  img_dir: recog/dataset3d/images
  ann_dir: recog/dataset3d/annotations
  class_map:
    battery: 1
    cartridge: 2
  train_val_split: 0.85
```

`README.md` — add to the "Other entry points" table:

```markdown
Sync synth3d config to Blender | `python -m recog.sync_config`
Generate 3-D synthetic scenes (Blender) | `blender -b --python recog/generate3d.py -- --n 200 --out recog/dataset3d`
Inspect generated boxes | `python -m recog.verify3d --data recog/dataset3d --n 12`
Sweep lighting presets | `blender -b --python recog/generate3d.py -- --sweep lighting --out recog/sweeps`
```

And a short section after "Running the software-only demo" explaining that `recog/synth_dataset.py` (cv2, fast, used by `main.py`'s demo) and `recog/generate3d.py` (Blender, photoreal, used for training) are two generators serving different purposes.

- [ ] **Step 8: Final verification**

```bash
python -m pytest -q --ignore=tests/test_inference.py --ignore=tests/test_main_integration.py --ignore=tests/test_placement_area.py --ignore=tests/test_planner.py
```

Expected: **133 passed, 1 skipped**.

- [ ] **Step 9: The real run**

Only after the contact sheet looks right.

```bash
"$BLENDER" -b --python recog/generate3d.py -- --n 2000 --out recog/dataset3d --device GPU --resume
python -m recog.verify3d --data recog/dataset3d --n 16
```

Check the printed anchor line: `box_diag_px` p95 should sit well under 512. If it approaches it, enlarge `layout.area` in `configs/synth3d.yaml`, re-run `python -m recog.sync_config`, and regenerate.

`--resume` makes this interruptible; re-running the same command continues where it stopped.

---

## Self-Review

**Spec coverage.** §3 classes → Task 2. §4 architecture and the port table → Tasks 2–7. §5.1 variants → Task 2. §5.2 PCB → Task 7. §5.3 materials incl. purple/grey wraps → Task 2 YAML. §6 layout and the `common/packing` decision → Tasks 1 and 4. §7 config, CLI, sweep, save-blend → Tasks 2 and 8. §8 output contract and all four invariants → Tasks 5, 6, 8. §9 integration and the anchor check → Task 8 Steps 7 and 9. §10 testing → Tasks 1–5. §11 running, and §11.1's four API deltas → Task 6 Step 3. §12 environment gaps → Global Constraints.

Two spec items intentionally have no task: `convert_cad.py` (§4.1) is not ported because the four `.glb` files already exist and `cascadio`/`trimesh` are not installed — Task 3 copies the converted assets instead, and the spec lists STEP conversion as an offline path for future CAD. If new CAD arrives, copy `partsgen/convert_cad.py` and the `catalog.build_catalog` entry point is already ported in Task 3. And §13's deferred COCO reader is out of scope by decision.

**Type consistency.** `Placement(x, y, quarter, rot_deg)` is produced by both `plan` and `plan_jig` and consumed by `assets.place_item` — matches partsgen's existing contract. `Pocket(x, y, w, h, depth)` is centre-based in metres, produced by `plan_jig` and consumed by `world.build_jig`; the tests assert the centre convention. `cfg` threading is consistent: `materials.build/for_role`, `world.build_backdrop/setup_lighting` and `scene.sample_params/scene_generator` all gained a `cfg` parameter, and every call site in Task 7 passes it. Annotation dicts use `bbox_xyxy` throughout; `write_voc_xml` reads that key and nothing else.

**Known risk carried, not resolved.** Task 7 Step 1 uses `ShaderNodeTexWave` with `bands_direction`, which was not in the verified probe set. The step says to drop that one line if it raises. Everything else in the bpy path was probed against the installed Blender 5.0.

---

## Execution notes

Tasks 1–5 are gated by pytest and are safe to hand to a subagent with only this document. Tasks 6–8 need the Blender binary and a human looking at a contact sheet; the gates are written as runnable commands with explicit expected output, and Task 8 Step 4 is explicitly a human judgement gate.

Task ordering is a hard dependency chain: 1 → 4 (packing), 2 → all (config), 3 → 4 (real footprints in tests), 5 → 8 (VOC writer), 6 → 7 → 8.
