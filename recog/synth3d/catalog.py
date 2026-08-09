"""
recog.synth3d.catalog - STEP -> glTF conversion and asset cataloguing.

Runs OUTSIDE Blender (needs cascadio + trimesh, which Blender's bundled Python
does not have). Blender cannot read STEP at all, so this is a required
preprocessing step; the catalog it writes is the only thing the Blender side
reads about the CAD.

These are the primitives. The command-line entry point is recog.convert_cad,
which is what you should actually run to import CAD:

    pip install cascadio trimesh
    python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/

It wraps `convert_step`/`inspect_glb` with the things a bare `build_catalog`
call does not do: it MERGES into the existing catalog instead of clobbering
it, reads each file's declared length unit, and refuses to write an entry
whose extents are implausible for this domain. Prefer it.

`build_catalog` below converts every .stp/.step in the source directory and
REWRITES assets/catalog.json from scratch, dropping anything already in it.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

from .config import CLASS_RULES, ROLE_FALLBACK

MM = 1000.0        # glTF is metres; CAD is millimetres


def role_of(subpart_name: str) -> str:
    """Map a CAD sub-part name to a semantic role via CLASS_RULES."""
    for pattern, role in CLASS_RULES:
        if re.search(pattern, subpart_name, flags=re.IGNORECASE):
            return role
    return ROLE_FALLBACK


def classify_case_parts(xy_areas: dict) -> dict:
    """Decide, among a group of same-named-role `case` parts, which is the
    outer half-shell (`"case"`) and which are inner cell-holder liners
    (`"case_liner"`) - purely by XY footprint area. The outer shell always
    has the larger footprint (~4mm of margin on every real asset, against
    a 6-7mm z-height difference in the other direction - not a close
    call); a regex or an index cannot make this distinction at all, since
    both parts share one name prefix and appear in a DIFFERENT ORDER per
    asset (the 20100 and 26800 list the holder first). See
    task-3b-brief.md.

    `xy_areas` is `{id: area_mm2}` for every part sharing the ambiguous
    role - `id` is whatever the caller uses to identify a part.
    `catalog.inspect_glb` calls this with subpart names (at STEP-
    conversion time, via `_split_case_liner` below); `assets.AssetLibrary`
    calls it a second time with Blender objects (at Blender-import time),
    because Blender's glTF importer merges the shell and the liner into
    ONE object - they are two differently-materialled PRIMITIVES of one
    glTF node, not two nodes - so assets.py has to split that object apart
    itself before this same decision applies to it. Keeping the decision
    rule itself here, in bpy-free catalog.py, is what keeps it unit-tested
    even though it necessarily runs twice at two different layers.

    Returns `{id: "case"}` for the single largest, `{id: "case_liner"}`
    for every other id. A no-op (`{only_id: "case"}`) when 0 or 1 ids are
    given - nothing to disambiguate.

    Raises ValueError if the largest and second-largest areas are within
    1% of each other: that would mean the geometric discriminator is not
    reliable for this asset, and a silent mis-split puts a solid body
    back over the bay - worth failing loudly rather than costing another
    render cycle to find.
    """
    if len(xy_areas) <= 1:
        return {k: "case" for k in xy_areas}

    ranked = sorted(xy_areas.items(), key=lambda kv: kv[1], reverse=True)
    (_, largest_area), (_, second_area) = ranked[0], ranked[1]
    if largest_area <= 0 or \
            (largest_area - second_area) / largest_area < 0.01:
        raise ValueError(
            f"classify_case_parts: largest and second-largest XY areas "
            f"are within 1% of each other ({largest_area:.2f} vs "
            f"{second_area:.2f} mm^2 among {list(xy_areas)}) - the "
            f"geometric discriminator cannot reliably tell the outer "
            f"shell from an inner liner for this assembly")

    out = {ranked[0][0]: "case"}
    out.update({k: "case_liner" for k, _ in ranked[1:]})
    return out


def _split_case_liner(subparts: List[dict]) -> None:
    """Geometric post-pass: among an assembly's `case`-role parts, separate
    the outer half-shell from the inner cell holder via `classify_case_
    parts` (see its docstring), and mutate `subparts` in place, rewriting
    the loser(s)' `role` to `case_liner`; the winner keeps `role ==
    "case"`.

    A no-op when 0 or 1 `case`-role parts are present - nothing to split.

    Asserts the discriminator rather than trusting it: raises ValueError
    if the assembly does not end up with exactly one `case` part after
    the split (on top of `classify_case_parts`'s own 1%-margin check) -
    belt and suspenders, since a silent mis-split here is exactly the
    failure this function exists to rule out.
    """
    cases = [s for s in subparts if s["role"] == "case"]
    if len(cases) <= 1:
        return

    areas = {i: s["extents_mm"][0] * s["extents_mm"][1]
             for i, s in enumerate(cases)}
    roles = classify_case_parts(areas)
    for i, s in enumerate(cases):
        s["role"] = roles[i]

    remaining = [s for s in subparts if s["role"] == "case"]
    if len(remaining) != 1:
        raise ValueError(
            f"_split_case_liner: expected exactly one case part after the "
            f"split, got {len(remaining)}: {[s['name'] for s in remaining]}")


def convert_step(src: str, dst: str, tol_linear: float = 0.05,
                 tol_angular: float = 0.3) -> None:
    """
    Tessellate a STEP file to glTF.

    tol_linear is the max chord deviation in millimetres. 0.05mm keeps the
    silhouette sub-pixel at 1k render resolution while cutting triangle count
    by ~3x versus 0.02mm.
    """
    import cascadio
    cascadio.step_to_glb(src, dst, tol_linear=tol_linear,
                         tol_angular=tol_angular)


def inspect_glb(path: str) -> dict:
    """Measure a converted asset and classify its sub-parts.

    Walks the scene GRAPH, not `scene.geometry`, because a sub-part's
    position is carried by its node transform. `g.extents` alone says how
    big a cell is but not where it sits, and where it sits is what
    reveals the electronics bay.
    """
    import numpy as np
    import trimesh

    from .bay import (case_wall_from_bounds, interior_from_tray,
                      module_bay_from_bounds)

    scene = trimesh.load(path)

    subparts = []
    node_bounds = {}          # subpart name -> (lo, hi), millimetres

    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        g = scene.geometry[gname]
        corners = trimesh.transform_points(
            trimesh.bounds.corners(g.bounds), transform)
        lo, hi = corners.min(axis=0) * MM, corners.max(axis=0) * MM
        node_bounds[node] = (lo, hi)

        subparts.append({
            "name": node,
            "role": role_of(node),
            "extents_mm": [round(float(v) * MM, 2) for v in g.extents],
            "triangles": int(len(g.faces)),
            "volume_mm3": round(float(g.volume) * MM ** 3, 1)
            if g.is_volume else None,
        })

    # Geometric refinement: the name-based role_of() above cannot tell the
    # outer shell from the inner cell holder - both are `Case*_btm*` with
    # opaque hash suffixes, ordered differently per asset. This rewrites
    # the loser(s)' role to `case_liner` in place. Recompute counts/bounds
    # from `subparts` AFTER this, not before, or `by_role_bounds["case"]`
    # would still include the liner and tray_outer_mm would stay wrong.
    _split_case_liner(subparts)

    counts = {}
    by_role_bounds = {}
    for s in subparts:
        role = s["role"]
        counts[role] = counts.get(role, 0) + 1
        by_role_bounds.setdefault(role, []).append(node_bounds[s["name"]])

    def _aabb(role):
        if role not in by_role_bounds:
            return None
        los = np.array([b[0] for b in by_role_bounds[role]])
        his = np.array([b[1] for b in by_role_bounds[role]])
        lo, hi = los.min(axis=0), his.max(axis=0)
        return [round(float(v), 2) for v in
                (lo[0], lo[1], hi[0], hi[1], hi[2])]

    def _role_zmin(role):
        """Minimum world-space z, in millimetres, over a role's meshes.

        `by_role_bounds` is already in millimetres (see the `* MM` above),
        so this is a direct min - no further scaling. `_aabb` returns
        z_max as its last element and never z_min, which is why this is a
        separate helper rather than a read out of `cell_union_mm`.
        """
        if role not in by_role_bounds:
            return None
        los = np.array([b[0] for b in by_role_bounds[role]])
        return float(los.min(axis=0)[2])

    cell_union = _aabb("cell")
    # Outer half-shell only, now that _split_case_liner has run - the lid is
    # `case_lid` and the inner cell holder is `case_liner`. The liner is
    # narrower than the shell in XY (see _split_case_liner), so this AABB's
    # x0/y0/x1/y1 are unchanged from before the split; only z (index 4,
    # the liner's own top) moves down to the shell's own, shorter height.
    tray_outer = _aabb("case")

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
        # Wall thickness. The plan's literal Step 4 (docs/superpowers/plans/
        # 2026-08-09-tray-interior.md) takes this as the plain
        # min(gap for gap in the four cell-to-tray clearances if gap > 0).
        # For AnkerPowerCore10000 (the smallest assembly) that picks up the
        # near/closed-end gap (2.45mm) instead of the true side wall
        # (4.0mm), because that gap happens to be SMALLER than the wall on
        # this one asset - exactly the "spring-contact recess" case
        # `case_wall_from_bounds`'s docstring documents ("plausibly a
        # spring-contact recess ... at least on the smallest case").
        # Task-2's brief (Amendment C) anticipates this and says to stop
        # and report rather than accept a wall value that has moved by
        # more than ~0.5mm from the previously-measured 4.0/3.75/3.7/4.25mm
        # figures. `case_wall_from_bounds` already exists, is tested against
        # those exact four figures, and deliberately excludes the bay-axis
        # near side for this reason - reuse it instead of a naive min() that
        # would silently corrupt interior_mm/module_bay_mm for this asset.
        wall = round(case_wall_from_bounds(
            tuple(tray_outer[:4]), tuple(cell_union[:4])), 2)

        interior = interior_from_tray(
            tuple(tray_outer[:4]), tuple(cell_union[:4]), wall)

        out["case_wall_mm"] = wall
        out["interior_mm"] = [round(v, 2) for v in interior]
        out["module_bay_mm"] = [
            round(v, 2) for v in module_bay_from_bounds(
                interior, tuple(cell_union[:4]))
        ]
        # The cells rest ON the cavity floor in assembled pose, so their
        # minimum z IS the floor. `by_role_bounds` is already in
        # millimetres (the `* MM` above), so `_role_zmin` must NOT be
        # scaled again here.
        out["tray_floor_mm"] = round(_role_zmin("cell"), 2)

    return out


def build_catalog(src_dir: str, out_dir: str, tol_linear: float = 0.05,
                  tol_angular: float = 0.3,
                  patterns=(".stp", ".step")) -> dict:
    """Convert every STEP file in src_dir and write assets/catalog.json."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir)
                   if f.lower().endswith(patterns))
    if not files:
        raise FileNotFoundError(f"no STEP files in {src_dir}")

    assets: List[dict] = []
    for f in files:
        stem = os.path.splitext(f)[0]
        # "004708_A_2-AnkerPowerCore26800" -> "AnkerPowerCore26800"
        pretty = stem.split("-", 1)[-1] if "-" in stem else stem
        glb = os.path.join(out_dir, pretty + ".glb")
        convert_step(os.path.join(src_dir, f), glb, tol_linear, tol_angular)

        info = inspect_glb(glb)
        info.update({"name": pretty, "source": f,
                     "file": os.path.basename(glb)})
        assets.append(info)
        print(f"  {pretty:26s} {info['extents_mm']}  "
              f"{info['triangles']:6d} tris  {info['role_counts']}")

    catalog = {
        "units": "m",
        "note": "glTF geometry is in metres; extents_mm are millimetres",
        "tol_linear_mm": tol_linear,
        "tol_angular": tol_angular,
        "assets": assets,
    }
    with open(os.path.join(out_dir, "catalog.json"), "w") as fh:
        json.dump(catalog, fh, indent=2)
    return catalog


def load_catalog(assets_dir: str) -> dict:
    with open(os.path.join(assets_dir, "catalog.json")) as fh:
        return json.load(fh)
