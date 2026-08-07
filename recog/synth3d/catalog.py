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

    from .bay import case_wall_from_bounds, module_bay_from_bounds

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
        out["case_wall_mm"] = round(case_wall_from_bounds(
            tuple(case_interior[:4]), tuple(cell_union[:4])), 2)
        out["module_bay_mm"] = [
            round(v, 2) for v in module_bay_from_bounds(
                tuple(case_interior[:4]), tuple(cell_union[:4]))
        ]
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
