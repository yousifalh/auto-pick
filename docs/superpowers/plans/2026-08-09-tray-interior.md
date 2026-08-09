# Open-cartridge tray interior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an `open_case` cartridge render as a real recessed tray — lid removed, electronics module and `placement_area` plane seated on the cavity floor — instead of a flat decal painted on the outside of a closed lid.

**Architecture:** The CAD already contains the tray; the generator discards it. A new `case_lid` role separates the top shell from the tray so `open_case` can drop it, `catalog.py` measures the cavity properly (the existing `case_interior_mm` is the assembly's *outer* AABB), and `bay.py` derives the module and placement rects from that real interior. `world.py`/`scene.py` only apply the result, as they have across ten prior tasks.

**Tech Stack:** Python 3.10+, Blender 5.0 bundled Python (no PyYAML, no pycocotools), trimesh + cascadio (system Python, conversion only), NumPy, pytest.

## Global Constraints

- Units are **metres** in `recog/synth3d/`; `catalog.json` is millimetres. `MM = 1000.0`.
- **Every geometric decision belongs in bpy-free `recog/synth3d/bay.py` or `catalog.py`.** `world.py` and `scene.py` import bpy and cannot be unit-tested. This line has held across ten tasks; it is what makes the geometry checkable at all.
- The bpy-boundary guard is now an **AST check** (`tests/test_synth3d.py`), so a docstring mentioning bpy is safe, but a real import in a listed module fails.
- `CLASSES` = `["battery", "cartridge"]` and `SEG_CLASSES` = the five segmentation classes. Neither changes.
- Every non-zero `pass_index` needs an `id_meta` entry or `scene._check_id_meta_covers_scene` raises.
- The **`assembled` variant must be unchanged** — geometry, labels and VOC boxes. A sealed cartridge is a real scene state and its labels are already correct.
- **Five-class pixel disjointness must stay at 0 overlapping pixels.** It is the design's central invariant.
- `unit_id` grouping and unit-scoped VOC boxes (Plan B Task 9) must be preserved.
- The torch-free demo, `python main.py --config configs/demo.yaml`, must keep working.
- Boxes keep exclusive max edges; a one-pixel object is a 1×1 box, never zero-area.

---

## File Structure

| File | Responsibility |
|---|---|
| `recog/synth3d/config.py` | Gains a `case_lid` role in `CLASS_RULES`; `open_case`/`assembled` `keep_roles` updated. |
| `recog/synth3d/catalog.py` | Measures the tray cavity: floor height, interior rect, wall thickness. Replaces the misnamed `case_interior_mm`. |
| `recog/synth3d/bay.py` | Bay/module rects derived from the **interior**, not the outer extent. Stays bpy-free. |
| `recog/synth3d/world.py` | `build_pcb` / `build_bay_proxy` / `seat_cells` take a floor height rather than the assembly top. |
| `recog/synth3d/scene.py` | Passes the interior and floor from the catalog. No arithmetic. |
| `configs/synth3d.yaml` | `role_materials` gains `case_lid`. |
| `tests/test_bay.py`, `tests/test_synth3d.py` | Interior measurement and role-split tests. |

---

### Task 1: Split the lid from the tray

**Files:**
- Modify: `recog/synth3d/config.py` (`CLASS_RULES`, `VARIANTS`)
- Modify: `configs/synth3d.yaml` (`role_materials`)
- Test: `tests/test_synth3d.py` (append)

**Interfaces:**
- Produces: role `"case_lid"` for sub-parts matching `Case.*_top`; role `"case"` continues to mean the tray. `VARIANTS`' `assembled` keeps `("cell", "case", "case_lid")`; `open_case` keeps `("cell", "case")`.

`CLASS_RULES` currently maps both halves to one role with `(r"Case.*_(top|btm)", "case")`, so `open_case` cannot drop the lid — that is the whole defect. Note there are **two** `btm` parts per assembly (an outer shell and an inner liner); both stay `case`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synth3d.py`:

```python
def test_lid_and_tray_are_distinct_roles():
    """open_case cannot drop the lid while both halves share one role."""
    from recog.synth3d.catalog import role_of

    assert role_of("004697_A;2-Case10000_top") == "case_lid"
    assert role_of("004696_A;2-Case10000_btm") == "case"
    assert role_of("004696_A;2-Case10000_btm_1") == "case"
    assert role_of("004695_A;1-Cell_18650") == "cell"


def test_open_case_drops_the_lid_and_assembled_keeps_it():
    from recog.synth3d.config import VARIANTS

    by_name = {v.name: v for v in VARIANTS}
    assert "case_lid" not in by_name["open_case"].keep_roles, (
        "an open cartridge must show its tray, not a closed assembly")
    assert "case" in by_name["open_case"].keep_roles
    assert "case_lid" in by_name["assembled"].keep_roles, (
        "a sealed cartridge must keep both halves")
    assert "case" in by_name["assembled"].keep_roles


def test_open_case_labels_the_tray_as_cartridge():
    from recog.synth3d.config import VARIANTS

    oc = {v.name: v for v in VARIANTS}["open_case"]
    assert oc.label_roles.get("case") == "cartridge"
    assert oc.label_roles.get("cell") == "battery"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_synth3d.py -v -k "lid_and_tray or drops_the_lid or labels_the_tray"`
Expected: FAIL — `role_of(...) == "case"` for the top part, and `case_lid` absent from every `keep_roles`.

- [ ] **Step 3: Split the role**

In `recog/synth3d/config.py`, replace the `CLASS_RULES` case entry:

```python
CLASS_RULES: List[Tuple[str, str]] = [
    # NX increments instance names, so cells appear as Cell_18650, Cell_18651,
    # Cell_18650_18652 ... - match "Cell_" + digits, not a literal.
    (r"Cell[_ ]?\d+", "cell"),
    # The lid gets its OWN role so `open_case` can drop it. Both halves shared
    # one role until now, which is why every "open" cartridge rendered closed
    # and the bay was painted on the outside of the lid. Order matters: the
    # `_top` rule must precede the general one.
    (r"Case.*_top", "case_lid"),
    (r"Case.*_btm", "case"),
]
```

`ROLE_FALLBACK` stays `"case"` — an unmatched sub-part is tray, not lid, so a new asset degrades toward showing more geometry rather than less.

Then update `VARIANTS`:

```python
VARIANTS: List[Variant] = [
    # Sealed unit: both shell halves, cells inside contributing no visible
    # pixels, so the mask pass drops them automatically.
    Variant("assembled", keep_roles=("cell", "case", "case_lid"),
            label="cartridge", weight=3.0),

    # Shell removed: loose 18650 cells, scattered individually.
    Variant("cells_only", keep_roles=("cell",), label=None,
            label_roles={"cell": "battery"}, weight=2.0),

    # Opened unit: the TRAY only, lid dropped, so the cavity is visible and
    # the module and bay proxy sit inside it rather than on a closed lid.
    Variant("open_case", keep_roles=("cell", "case"), label=None,
            label_roles={"cell": "battery", "case": "cartridge"},
            weight=1.0),
]
```

- [ ] **Step 4: Give the lid a material palette**

`materials.for_role` looks the role up in `role_materials`; an unmapped role would fall through. In `configs/synth3d.yaml`, extend `role_materials` so `case_lid` draws from the same shell palette:

```yaml
role_materials:
  case: [shell_white, shell_black, shell_navy, shell_alu]
  case_lid: [shell_white, shell_black, shell_navy, shell_alu]
  cell: [cell_green, cell_blue, cell_black, cell_nickel, cell_purple, cell_grey]
```

Then run `python -m recog.sync_config` — Blender's bundled Python reads the JSON sidecar and `_read_raw` raises if it is older than the YAML.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_synth3d.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 6: Commit**

```bash
git add recog/synth3d/config.py configs/synth3d.yaml configs/synth3d.json tests/test_synth3d.py
git commit -m "feat(synth3d): give the lid its own role so open_case can drop it

CLASS_RULES mapped Case.*_top and Case.*_btm to a single `case` role, so
open_case's keep_roles could not distinguish them and every 'open' cartridge
linked both halves and rendered CLOSED. build_pcb and build_bay_proxy then
anchored the module and placement_area plane to the assembly's hi.z - the
outer surface of the lid.

The lid is now `case_lid`. assembled keeps both; open_case keeps the tray."
```

---

### Task 2: Measure the tray cavity

**Files:**
- Modify: `recog/synth3d/catalog.py` (`inspect_glb`)
- Modify: `recog/synth3d/bay.py` (add `interior_from_tray`)
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: role `"case"` now meaning tray only (Task 1)
- Produces, per asset in `catalog.json`:
  - `tray_outer_mm: [x0, y0, x1, y1, z_top]` — AABB of the `case`-role (tray) meshes
  - `tray_floor_mm: float` — the cavity floor height, i.e. the top of the tray's solid base
  - `interior_mm: [x0, y0, x1, y1]` — the cavity footprint, tray outer inset by the wall
  - `case_wall_mm: float` — measured wall thickness
  - `module_bay_mm` — unchanged in meaning, but now derived from `interior_mm`
- `bay.interior_from_tray(tray_outer, cells_union, wall) -> Rect`

**`case_interior_mm` is misnamed and must go.** It is the AABB of *all* case-role meshes — the assembly's **outer** extent — and `module_bay_mm`, `case_wall_mm` and every label inherit that error. It has already caused one bug. Replace it; do not leave a field whose name asserts something false.

**Derive the floor from the mesh, not from a constant.** For `AnkerPowerCore10000` the tray spans z −0.0 → 11.1 mm and the cells sit at z 1.9 → 20.3, so the cavity floor is at **z = 1.95** — the cells rest on it, measured across all four assemblies. Use the cell union's `z_min` in assembled pose as the measurement, and cross-check it against the tray's own vertex distribution.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay.py`:

```python
def test_interior_is_the_tray_inset_by_the_wall():
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    # wall 4.0 -> interior runs 4..56 x 4..86
    assert interior_from_tray(tray, cells, 4.0) == pytest.approx(
        (4.0, 4.0, 56.0, 86.0))


def test_interior_never_exceeds_the_tray():
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    x0, y0, x1, y1 = interior_from_tray(tray, cells, 4.0)
    assert x0 >= tray[0] and y0 >= tray[1]
    assert x1 <= tray[2] and y1 <= tray[3]


def test_interior_contains_the_cells():
    """The cells demonstrably fit inside the cavity in assembled pose, so an
    interior that excludes them is measured wrong."""
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    ix0, iy0, ix1, iy1 = interior_from_tray(tray, cells, 4.0)
    assert ix0 <= cells[0] and iy0 <= cells[1]
    assert ix1 >= cells[2] and iy1 >= cells[3]


def test_interior_raises_when_the_wall_would_swallow_the_cavity():
    from recog.synth3d.bay import interior_from_tray

    with pytest.raises(ValueError):
        interior_from_tray((0.0, 0.0, 20.0, 20.0),
                           (2.0, 2.0, 18.0, 18.0), 12.0)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name", [
    "AnkerPowerCore10000", "AnkerPowerCore13000",
    "AnkerPowerCore20100", "AnkerPowerCore26800",
])
def test_catalog_records_a_tray_cavity(name):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)

    for key in ("tray_outer_mm", "tray_floor_mm", "interior_mm",
                "case_wall_mm", "module_bay_mm"):
        assert key in entry, f"{name} missing {key}; re-run recog.convert_cad"

    assert "case_interior_mm" not in entry, (
        "case_interior_mm was the assembly's OUTER extent despite its name "
        "and must not survive alongside a real interior measurement")

    tx0, ty0, tx1, ty1, _ = entry["tray_outer_mm"]
    ix0, iy0, ix1, iy1 = entry["interior_mm"]
    assert tx0 <= ix0 < ix1 <= tx1
    assert ty0 <= iy0 < iy1 <= ty1
    assert entry["tray_floor_mm"] > 0.0


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,expected_floor", [
    ("AnkerPowerCore10000", 1.95),
    ("AnkerPowerCore13000", 1.95),
    ("AnkerPowerCore20100", 1.95),
    ("AnkerPowerCore26800", 1.95),
])
def test_tray_floor_matches_where_the_cells_rest(name, expected_floor):
    """The cells sit ON the cavity floor in assembled pose, so the floor is
    where their lowest point is. Hand-measured from the CAD: every assembly
    rests its cells at z = 1.95 mm. If Step 4's derivation disagrees, the
    floor is measured wrong and every label sits at the wrong height.

    If a regenerated catalog reports a materially different value for an
    assembly, do NOT relax this test - re-measure that asset and find out
    why it differs, exactly as the bay depths were established.
    """
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert entry["tray_floor_mm"] == pytest.approx(expected_floor, abs=0.3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_bay.py -v -k "interior or tray"`
Expected: FAIL — `ImportError: cannot import name 'interior_from_tray'`, and the catalog tests fail on the missing keys.

- [ ] **Step 3: Add the interior derivation**

Append to `recog/synth3d/bay.py`:

```python
def interior_from_tray(tray_outer: Rect, cells_union: Rect,
                       wall: float) -> Rect:
    """The tray's cavity footprint: its outer rectangle inset by the wall.

    `case_interior_mm` used to hold the AABB of every case mesh - the
    assembly's OUTER extent - despite its name, and module_bay_mm,
    case_wall_mm and every downstream label inherited that error. This is
    the real thing: the space a cell can actually occupy.

    The cavity is widened if necessary to contain the cells, which demonstrably
    fit in assembled pose. A wall measurement that excludes them is wrong, and
    trusting it would shrink every placement area.
    """
    tx0, ty0, tx1, ty1 = tray_outer
    cx0, cy0, cx1, cy1 = cells_union

    ix0, iy0 = tx0 + wall, ty0 + wall
    ix1, iy1 = tx1 - wall, ty1 - wall

    if ix1 - ix0 <= 0.0 or iy1 - iy0 <= 0.0:
        raise ValueError(
            f"interior_from_tray: wall {wall} swallows the cavity of "
            f"{tray_outer}")

    # The cells sit inside the cavity in the CAD. If the inset excludes any of
    # them the wall is over-measured, so widen rather than propagate the error.
    ix0, iy0 = min(ix0, cx0), min(iy0, cy0)
    ix1, iy1 = max(ix1, cx1), max(iy1, cy1)

    return (max(ix0, tx0), max(iy0, ty0), min(ix1, tx1), min(iy1, ty1))
```

- [ ] **Step 4: Measure the tray in `inspect_glb`**

In `recog/synth3d/catalog.py`, the role AABB helper already exists from the bay work. Change the case measurement to use the **tray only** (role `"case"`, which after Task 1 excludes the lid), and emit the new fields.

Replace the block that computes `case_interior_mm` and `module_bay_mm` with:

```python
    cell_union = _aabb("cell")
    tray_outer = _aabb("case")          # tray only - the lid is `case_lid`

    out = {
        "extents_mm": [round(float(v) * MM, 2)
                       for v in (scene.bounds[1] - scene.bounds[0])],
        "triangles": int(sum(len(g.faces) for g in scene.geometry.values())),
        "subparts": sorted(subparts, key=lambda s: (s["role"], s["name"])),
        "role_counts": counts,
        "cell_union_mm": cell_union,
        "tray_outer_mm": tray_outer,
    }

    if cell_union and tray_outer:
        # Wall thickness: the smallest clearance between the cells and the
        # tray on the three non-bay sides. Measured, not assumed - the four
        # assemblies give 3.7-4.25 mm.
        gaps = [cell_union[0] - tray_outer[0], tray_outer[2] - cell_union[2],
                cell_union[1] - tray_outer[1], tray_outer[3] - cell_union[3]]
        wall = round(min(g for g in gaps if g > 0.0), 2)

        interior = interior_from_tray(
            tuple(tray_outer[:4]), tuple(cell_union[:4]), wall)

        out["case_wall_mm"] = wall
        out["interior_mm"] = [round(v, 2) for v in interior]
        out["module_bay_mm"] = [
            round(v, 2) for v in module_bay_from_bounds(
                interior, tuple(cell_union[:4]))
        ]
        # The cells rest ON the cavity floor in assembled pose, so their
        # minimum z IS the floor. `_aabb` returns z_max last; capture z_min
        # here so the floor is a measurement rather than a constant.
        out["tray_floor_mm"] = round(_role_zmin("cell") * 1.0, 2)

    return out
```

Add a `_role_zmin(role)` helper beside `_aabb`, returning the minimum world-space z in millimetres over that role's meshes. Add `interior_from_tray` to the `from .bay import ...` line.

`_aabb` returns `z_max` as its last element, so **do not read a resting height out of `cell_union_mm`** — add the dedicated `_role_zmin` helper instead. `tray_floor_mm` is the only field that carries the floor, and Step 1's test checks it against the hand-measured 1.95 mm rather than against another catalog slot, so the two cannot drift into agreeing with each other while both being wrong.

- [ ] **Step 5: Regenerate the catalog and verify**

Run:
```bash
python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/
python -m pytest tests/test_bay.py -v
```
Expected: PASS. Report the measured `case_wall_mm`, `tray_floor_mm` and `interior_mm` for all four assets. The previous wall values were 4.0 / 3.75 / 3.7 / 4.25 mm; a large departure means the tray-only measurement is picking up something unexpected and is worth investigating before continuing.

- [ ] **Step 6: Commit**

```bash
git add recog/synth3d/catalog.py recog/synth3d/bay.py recog/synth3d/assets/catalog.json tests/test_bay.py
git commit -m "feat(synth3d): measure the real tray cavity, retire case_interior_mm

case_interior_mm was the AABB of every case mesh - the assembly's OUTER
extent - despite its name, and module_bay_mm, case_wall_mm and every label
inherited that error.

With the lid split off, the tray can be measured properly: tray_outer_mm,
a floor height taken from where the cells actually rest, a wall thickness
from the smallest cell-to-tray clearance, and an interior_mm the module and
bay rects are now derived from."
```

---

### Task 3: Seat the module, proxy and cells on the cavity floor

**Files:**
- Modify: `recog/synth3d/world.py` (`build_pcb`, `build_bay_proxy`, `build_obstructions`, `seat_cells`)
- Modify: `recog/synth3d/scene.py` (the `open_case` block)
- Test: `tests/test_bay.py` (append)

**Interfaces:**
- Consumes: `interior_mm`, `tray_floor_mm`, `module_bay_mm` from Task 2
- Produces: no new public names. The four `world` builders take a `floor_z` in metres instead of deriving one from the group's `hi.z`.

`build_pcb`'s docstring records the simplification being removed here: *"the board is laid on top of the shell rather than modelled inside it… this needs no interior geometry."* Delete that reasoning along with the behaviour, and say what it does now.

**Z ordering inside the cavity must be preserved.** The existing offsets are relative to a surface and stay relative to the floor: proxy `+0.0009`, adhesive `+0.0012`, tape/label `+0.0011`, foam `+0.0022`, seated cells `SEATED_CELL_LIFT = 0.0012`. Everything in the bay must still sit **above** the proxy or occlusion breaks and `placement_area` silently stops meaning "currently free floor".

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bay.py`:

```python
def test_module_and_placement_rects_derive_from_the_interior():
    """Both rects must tile the CAVITY, not the tray's outer extent - the
    walls are cartridge, not placeable floor."""
    from recog.synth3d.bay import module_rect_local, placement_rect_local

    interior = (4.0, 4.0, 56.0, 86.0)
    bay = (4.0, 66.0, 56.0, 86.0)
    fp = (-26.0, -41.0, 26.0, 41.0)

    m = module_rect_local(fp, bay, interior)
    p = placement_rect_local(fp, bay, interior)

    for r in (m, p):
        assert fp[0] - 1e-9 <= r[0] < r[2] <= fp[2] + 1e-9
        assert fp[1] - 1e-9 <= r[1] < r[3] <= fp[3] + 1e-9

    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    assert area(m) + area(p) == pytest.approx(
        (fp[2] - fp[0]) * (fp[3] - fp[1]))
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `python -m pytest tests/test_bay.py -v -k derive_from_the_interior`
Expected: PASS. `module_rect_local` and `placement_rect_local` already map a bay rect through an interior rect — Task 2 changed *which* interior they are handed, not their arithmetic. This test pins that the pair still tiles exactly, so a later edit cannot reintroduce a gap or an overlap. Keep it; a test documenting an invariant earns its place even when it starts green.

- [ ] **Step 3: Take a floor height in the world builders**

In `recog/synth3d/world.py`, change each builder's `z` parameter from "the shell's top" to "the cavity floor", and rewrite the docstrings. For `build_pcb`:

```python
def build_pcb(bounds_xy, floor_z: float, rng: random.Random,
              module_rect=None):
    """
    Green PCB with extruded components, seated in the tray's cavity.

    `floor_z` is the CAVITY FLOOR, from catalog.json's tray_floor_mm - not
    the assembly's top. Until the lid was split into its own role this was
    handed `hi.z`, the outer surface of a closed lid, so the board was drawn
    on the outside of a shut box. The cells rest on this same floor in the
    CAD, which is where the measurement comes from.
    """
```

The body is unchanged apart from the parameter name: every `z + <offset>` already reads from the passed height.

Make the same substitution in `build_bay_proxy`, `build_obstructions` and `seat_cells`. Do not change any offset — they are relative to the surface and must stay in their current order.

- [ ] **Step 4: Pass the floor and interior from the scene**

In `recog/synth3d/scene.py`'s `open_case` block, read the new catalog fields and convert millimetres to metres:

```python
            entry = library.catalog_entry(item.asset)
            module_rect = placement_rect = None
            floor_z = hi.z          # fallback: an asset with no measurement
            if entry and entry.get("module_bay_mm") and entry.get("interior_mm"):
                module_rect = B.module_rect_local(
                    footprint, tuple(entry["module_bay_mm"]),
                    tuple(entry["interior_mm"]))
                placement_rect = B.placement_rect_local(
                    footprint, tuple(entry["module_bay_mm"]),
                    tuple(entry["interior_mm"]))
                # tray_floor_mm is measured from the asset's own origin, and
                # lay_flat rests the item on z = 0, so the floor sits that far
                # above the item's base.
                floor_z = lo.z + entry["tray_floor_mm"] / 1000.0
```

Then pass `floor_z` where `hi.z` was passed, to all four builders. `scene.py` must contain **no arithmetic beyond this unit conversion** — the rects come from `bay.py`.

- [ ] **Step 5: Render and look at it**

Run:
```bash
python -m recog.sync_config
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/generate3d.py -- --n 8 --out recog/dev3d --device GPU --variant open_case
python -m recog.verify3d --data recog/dev3d --n 8 --masks
```

Then **open `recog/dev3d/contact_sheet.png` and look at it.** `world.py` and `scene.py` cannot be unit-tested; this is the only evidence the render is right. Check each, and report what you actually saw:

- [ ] An open cartridge reads as a **recessed tray** — visible walls, visible depth
- [ ] The module sits **inside** the cavity at one short end, not on a surface
- [ ] Self-shadowing appears in the cavity under directional lighting
- [ ] Seated cells lie **below the rim**
- [ ] Yellow `placement_area` covers the cavity floor, not the wall tops
- [ ] Sealed (`assembled`) cartridges are visually unchanged

If the cartridge still reads as a flat closed box, the lid is still being linked — check `keep_roles` reached `assets.instantiate` rather than assuming Task 1 took effect.

- [ ] **Step 6: Commit**

```bash
git add recog/synth3d/world.py recog/synth3d/scene.py tests/test_bay.py
git commit -m "fix(synth3d): seat the module, proxy and cells on the cavity floor

The four bay builders took the assembly's hi.z - the outer face of a closed
lid - and drew the electronics module, placement_area plane, obstructions and
seated cells on it. They now take the measured cavity floor.

Z offsets are unchanged and stay relative to the surface, so obstructions and
seated cells still render above the proxy and continue to subtract themselves
from placement_area by occlusion."
```

---

### Task 4: Regenerate, retrain and refresh every published number

**Files:**
- Modify: `docs/FDR_v3.md`, `docs/NEXT_STEPS.md`, `docs/receipts/*`
- No source changes

The spec states this is part of the work, not a follow-on: every label geometry moved, so the dataset and checkpoint are invalid.

- [ ] **Step 1: Regenerate the dataset**

```bash
BL="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
"$BL" -b --python recog/generate3d.py -- --n 502 --out recog/dataset3d_seg --device GPU --resume
```

**Delete `recog/dataset3d_seg` first** — `--resume` skips scenes whose files already exist, and every one of them now has stale geometry. Resuming onto the old set would silently mix two label conventions.

The Bash tool caps at 600000 ms, so expect several `--resume` calls after the first. Run them in the **foreground**; backgrounding has stalled several tasks on this project. The run must reach `[done]` for the COCO sidecar to be written — it is emitted at the end, and an interrupted run leaves a stale one.

- [ ] **Step 2: Verify the regenerated labels**

```bash
python -c "
import json
from collections import Counter
from recog.seg_dataset import BaySegDataset
d=json.load(open('recog/dataset3d_seg/instances_seg.json'))
n={c['id']:c['name'] for c in d['categories']}
print('images', len(d['images']), 'anns', len(d['annotations']))
print(dict(Counter(n[a['category_id']] for a in d['annotations'])))
print('all unit_id:', all('unit_id' in a for a in d['annotations']))
ds=BaySegDataset('recog/dataset3d_seg/instances_seg.json','recog/dataset3d_seg/images',out_size=64,train=False)
print('crops', len(ds))
"
```

Then re-run the pixel-exact disjointness sweep — `placement_area` against `battery` / `obstruction` / `electronics_module` across all images. **It must stay at 0 overlapping pixels.** Anything else means occlusion ordering broke when the surfaces moved.

Report the `cartridge` pixel count per open unit against the previous run. It should **rise**: the walls are now real geometry rather than a rim preserved by an artificial inset.

- [ ] **Step 3: Retrain**

```bash
python -m recog.seg_training --config configs/segmentation.yaml
```

`seg_training.py` has a `--resume` flag, added after a run was lost to the command cap. Use it rather than restarting, and run the full schedule to completion.

- [ ] **Step 4: Regenerate every receipt**

```bash
python -m recog.seg_evaluate  --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
python -m recog.seg_ablation
```

Report each against its current value: selected mean IoU **0.8045**; boundary displacement **1.299 / 1.085 / 1.633 mm**; τ **0.3180** with the budget never binding and the largest optimistic error at **79.4 %** of a cell; Δcells **+0.032** with 1 of 126 negative; and the real-photograph comparison, currently **unresolved** — two same-recipe checkpoints scored 0.211 and 0.232 against the heuristic's 0.217.

**Do not expect the real-photo number to resolve, and do not present a shift as evidence.** At n = 20 it cannot settle either way; that is recorded in the spec and in `NEXT_STEPS.md`. The tray fix is justified on physical correctness.

Watch whether the **largest optimistic error** grows past one cell's footprint. If it does, τ becomes calibratable for the first time and that is a genuine result worth stating.

- [ ] **Step 5: Update the documents**

Correct every superseded figure in `docs/FDR_v3.md` and `docs/NEXT_STEPS.md`. Stale numbers have caught this project three times — `grep -n` for each old value rather than trusting a read-through, and check the README, which duplicates the τ and latency figures.

Add to FDR §13.2.1 that the generator drew the bay and module on the outside of a closed lid until this change, and what moved as a result. It is the kind of finding that section exists to record.

- [ ] **Step 6: Commit**

```bash
git add docs/ recog/synth3d/assets/catalog.json
git commit -m "docs: regenerate and retrain on real tray interiors

Every label geometry moved when the module and bay proxy came off the lid and
into the cavity, so the dataset was regenerated and the segmenter retrained.
All published figures refreshed from the new receipts."
```

---

## Acceptance

- [ ] `python -m pytest -q` passes with no regressions.
- [ ] An open cartridge renders as a recessed tray with visible walls, depth and self-shadowing — confirmed by eye on a contact sheet.
- [ ] The module sits inside the cavity; seated cells lie below the rim.
- [ ] `assembled` cartridges are unchanged, geometry and labels alike.
- [ ] Five-class pixel disjointness stays at 0 overlapping pixels.
- [ ] `cartridge` pixels per open unit rise relative to the previous dataset.
- [ ] `case_interior_mm` no longer exists; `interior_mm`, `tray_outer_mm`, `tray_floor_mm` and `case_wall_mm` are measured per asset.
- [ ] The torch-free demo still runs.
- [ ] Every published figure in the FDR, README and `NEXT_STEPS.md` matches a regenerated receipt.
