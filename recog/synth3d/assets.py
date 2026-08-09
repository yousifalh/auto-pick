"""
recog.synth3d.assets - loading converted CAD into Blender and turning it into
placeable scene items.

Requires bpy. The CAD itself is never read here: catalog.py has already
produced .glb + catalog.json, and this module only consumes those.

Each asset is imported ONCE into a hidden template collection. Instances are
linked duplicates that share mesh data, so eight copies of a 50k-triangle
assembly cost 50k triangles, not 400k. Materials are linked to the OBJECT
rather than the mesh, which is what lets shared-data instances still carry
independently randomized surfaces.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector

from .catalog import classify_case_parts, load_catalog, role_of
from .config import Variant


TEMPLATE_COLLECTION = "_synth3d_templates"

# Custom ID property `_load_template` tags onto every template object with
# its final role, so scene.py can read a role WITHOUT re-deriving it from
# the object's (Blender-mangled, e.g. ".001"-suffixed) name. `clone()`'s
# `src.copy()` carries custom properties over to every instance, so the tag
# survives instancing. See `object_role` below for why this exists: a plain
# name regex cannot tell `case` from `case_liner` apart (see
# catalog.classify_case_parts), and after `_split_multi_material_case`
# splits them into two Blender objects they still share the same name
# prefix - only the geometry the tag was computed from knows which is which.
ROLE_PROP = "synth3d_role"


@dataclass
class Item:
    """One independently placeable thing: an assembly, or a single loose part."""
    objects: List[object] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)   # obj.name -> class
    merge: bool = False          # collapse member boxes into one
    asset: str = ""
    variant: str = ""
    footprint: Tuple[float, float] = (0.0, 0.0)
    # The electronics-module board scene.py built for an open_case item, or
    # None. Not in `objects`/`labels`: it is not a CAD sub-part, so it gets
    # its own pass_index and id_meta entry in scene.py rather than going
    # through the per-item labelling loop.
    module_object: object = None
    # The placement_area proxy plane scene.py built for an open_case item,
    # or None. Same reasoning as module_object: scene content, not a CAD
    # sub-part, so it gets its own pass_index and id_meta entry too.
    bay_object: object = None
    # The obstruction objects (adhesive/foam/tape/label) scene.py built on
    # this item's bay, or None. Same reasoning as bay_object: scene content,
    # not a CAD sub-part. Unlike bay_object there can be several, and each
    # is its OWN instance with its own pass_index and id_meta entry.
    obstruction_objects: Optional[List[object]] = None
    # Cells seated in this item's bay (the packer's own pitch, axis-aligned
    # - see bay.seated_cell_poses/world.seat_cells), or None. Same reasoning
    # as obstruction_objects: scene content built here, not a CAD sub-part,
    # so each gets its own pass_index/id_meta entry rather than going
    # through the per-item labelling loop.
    seated_objects: Optional[List[object]] = None
    # The rigid transform `layout.Placement` applied to this item - its
    # `rot_deg` and `(x, y)` - captured once the item is placed.
    # bay.module_world_placement needs both to put the electronics module
    # exactly where the case itself ended up: rotating the module's local
    # centre by the same angle and translating by the same offset, rather
    # than approximating from the case's rotation-inflated world AABB.
    rot_deg: float = 0.0
    placed_xy: Tuple[float, float] = (0.0, 0.0)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def group_bbox(objects) -> Tuple[Vector, Vector]:
    pts = []
    for o in objects:
        pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return (Vector((min(p.x for p in pts), min(p.y for p in pts),
                    min(p.z for p in pts))),
            Vector((max(p.x for p in pts), max(p.y for p in pts),
                    max(p.z for p in pts))))


def _template_collection():
    coll = bpy.data.collections.get(TEMPLATE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(TEMPLATE_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
        coll.hide_render = True
        coll.hide_viewport = True
    return coll


def lay_flat(objects):
    """
    Rotate the group so its smallest extent runs along Z, i.e. the part rests
    on the backdrop the way it would in reality.

    Worth doing even though glTF import already converts Y-up to Z-up: it makes
    the pipeline independent of how any individual CAD file was modelled. For a
    22mm-thick power bank this puts the slab flat; for an 18650 cell it lays the
    65mm axis horizontal.

    DO NOT "simplify" this back into `o.rotation_euler += ...`. Two reasons,
    both of which fail SILENTLY - the scene still renders, just wrong:

    1. The glTF importer sets `rotation_mode = 'QUATERNION'` on every object it
       creates. Blender IGNORES `rotation_euler` in quaternion mode, so euler
       writes land in the property and never reach `matrix_world`. Measured:
       after writing 90 degrees the property read back 1.5708 while the world
       extents stayed byte-identical at [18.3, 18.3, 65.0]mm. Every cell stood
       on its end and every shell stood on its long edge.
    2. Even with the rotation mode forced to XYZ, per-object euler writes rotate
       each object about ITS OWN origin, not about the group pivot, which pulls
       an assembly apart (measured [114.7, 90.9, 52.2]mm instead of
       [62.9, 90.9, 22.2]mm). A single pivot-conjugated matrix applied to every
       member is what keeps the assembly rigid.

    Assigning `matrix_world` is immune to both: it is rotation-mode agnostic and
    it moves the whole group as one body. `_gate_orientation.py` asserts this.
    """
    lo, hi = group_bbox(objects)
    ext = hi - lo
    axis = min(range(3), key=lambda i: ext[i])
    if axis == 2:
        return
    R = Matrix.Rotation(math.radians(90), 4, "X" if axis == 1 else "Y")
    pivot = (lo + hi) / 2
    M = Matrix.Translation(pivot) @ R @ Matrix.Translation(-pivot)
    for o in objects:
        o.matrix_world = M @ o.matrix_world
    bpy.context.view_layer.update()


def drop_to_floor(objects):
    lo, _ = group_bbox(objects)
    for o in objects:
        o.location.z -= lo.z


def clone(src, target=None):
    """Linked duplicate of `src`: shares mesh data (see the module
    docstring on template import cost) but gets independent per-instance
    materials, because material slots are linked to the OBJECT rather than
    the mesh.

    `target` defaults to the active scene collection. Module-level (not a
    closure inside `AssetLibrary.instantiate`) so `world.seat_cells` can
    clone straight from a `library._templates[asset]["cell"]` template too,
    the same way `instantiate` clones every other role - a seated cell has
    to be the asset's own 18650 geometry, not a new primitive, or it will
    not match the loose cells the detector already sees.
    """
    if target is None:
        target = bpy.context.scene.collection
    dup = src.copy()                 # linked duplicate: shares mesh data
    dup.data = src.data
    target.objects.link(dup)
    dup.hide_render = False
    dup.location = src.location.copy()
    dup.rotation_euler = src.rotation_euler.copy()
    for slot in dup.material_slots:
        slot.link = "OBJECT"
    return dup


def _separate_by_material(o):
    """`bpy.ops.mesh.separate(type="MATERIAL")` on `o`, returning every
    resulting piece: the original object (now holding only the polygons of
    its first material) plus one new object per additional material.
    """
    before = set(bpy.data.objects)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    o.select_set(False)
    new = [x for x in bpy.data.objects if x not in before]
    return [o] + new


def _split_multi_material_case(meshes):
    """Split apart any name-role `case` object Blender's glTF importer
    merged from multiple materially-distinct primitives.

    `Case*_btm` bundles the outer half-shell and the inner cell holder as
    TWO PRIMITIVES OF ONE glTF NODE (confirmed against the raw glTF: one
    node, one mesh, two `primitives`, each with its OWN material) - not
    two nodes. That means trimesh's own names for them are not even
    stable across two loads of the same file (a random disambiguating
    hash, regenerated per load - `catalog.json`'s subpart names for these
    two cannot be joined back onto a Blender object by name at all), and
    Blender's importer merges the two primitives into ONE object with two
    material slots. The only thing that can single the liner back out is
    splitting that merged object the way it was two bodies in the CAD,
    which is what this does - via Blender's own material-slot boundary,
    confirmed to land exactly on the same two triangle counts
    catalog.json records for the shell and the liner (1180 vs. 712 on
    AnkerPowerCore10000).

    Only touches objects whose NAME classifies as `case` (not `case_lid`,
    not `cell` - across all four real assets neither ever carries more
    than one material) and that actually carry more than one material
    slot; everything else passes through unchanged. Returns a new list
    with each split target replaced by its resulting pieces.
    """
    out = []
    for o in meshes:
        if role_of(o.name) == "case" and len(o.material_slots) > 1:
            out.extend(_separate_by_material(o))
        else:
            out.append(o)
    return out


def _classify_case_liner(by_role):
    """Disambiguate a `by_role["case"]` group of more than one object -
    the outer shell plus the inner liner(s) `_split_multi_material_case`
    just separated - into `"case"`/`"case_liner"`, in place on `by_role`.

    Uses `catalog.classify_case_parts`: the SAME rule (largest XY
    footprint wins `"case"`) and the SAME 1%-margin assertion
    `catalog._split_case_liner` applies to catalog.json at conversion
    time. See that function's docstring for why the decision has to be
    made a second time here, at Blender-import time, rather than just
    read back out of catalog.json.

    A no-op when `by_role["case"]` holds 0 or 1 objects.
    """
    case_objs = by_role.get("case") or []
    if len(case_objs) <= 1:
        return

    bpy.context.view_layer.update()
    areas = {}
    for i, o in enumerate(case_objs):
        lo, hi = group_bbox([o])
        areas[i] = (hi.x - lo.x) * (hi.y - lo.y)

    roles = classify_case_parts(areas)
    by_role["case"] = [case_objs[i] for i, r in roles.items() if r == "case"]
    liners = [case_objs[i] for i, r in roles.items() if r == "case_liner"]
    if liners:
        by_role.setdefault("case_liner", []).extend(liners)


def _connected_face_islands(bm):
    """Group `bm.faces` into connected components via shared edges."""
    visited = [False] * len(bm.faces)
    islands = []
    for seed in bm.faces:
        if visited[seed.index]:
            continue
        stack, comp = [seed], []
        visited[seed.index] = True
        while stack:
            f = stack.pop()
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if not visited[lf.index]:
                        visited[lf.index] = True
                        stack.append(lf)
        islands.append(comp)
    return islands


def _oversized_near_top_islands(bm, mw, hi_z, footprint_area,
                                top_tol=0.002, flatness_tol=0.0006,
                                area_frac=0.15):
    """Which of `bm`'s connected face-islands (see `_connected_face_
    islands`) are (a) FLAT - the island's own z-spread is under
    `flatness_tol` metres, so this excludes a real wall (which spans most
    of the case's own height, e.g. floor-to-rim) even though its highest
    point also reaches near the top, (b) sitting within `top_tol` metres
    of `hi_z` (world space) at their highest point, and (c) together cover
    more than `area_frac` of `footprint_area`. See `_strip_phantom_cap`
    for what this is for. Returns a list of `(faces, area)` pairs.

    Measured (task-3c): the phantom cap is actually TWO coincident-ish
    flat quads ~1.5mm apart (paired top/bottom faces of a zero-thickness
    "plate"), not one - `top_tol` has to be wide enough to reach the
    lower of the two (a plain z-nearness check on JUST the island's own
    max would also flag a real full-height wall whose top happens to
    reach the rim; the flatness check is what rules that out instead).
    """
    bad = []
    for comp in _connected_face_islands(bm):
        zs = [(mw @ v.co).z for f in comp for v in f.verts]
        z_lo, z_hi = min(zs), max(zs)
        if (z_hi - z_lo) > flatness_tol:
            continue                      # spans real height - a wall, not a cap
        if hi_z - z_hi > top_tol:
            continue                      # flat, but not near the top - a floor/step
        area = sum(f.calc_area() for f in comp)
        if area > area_frac * footprint_area:
            bad.append((comp, area))
    return bad


def _strip_phantom_cap(case_obj, area_frac=0.15, top_tol=0.002,
                       flatness_tol=0.0006):
    """Remove disconnected, flat face island(s) sitting at or near
    `case_obj`'s own top (world z_max) that span a large fraction of its
    own XY footprint - a tessellation/CAD artifact sharing the outer
    shell's material, so `_split_multi_material_case`'s material boundary
    cannot separate it from the true shell.

    Measured directly (task-3c-report.md), not guessed: on all four real
    assets, the "case" role piece `_classify_case_liner` selects is not
    one connected shape - it is ~30 disconnected face-islands sharing one
    material index (confirmed via bmesh connected-component analysis, and
    independently confirmed by rendering the isolated, already-correctly-
    split, already-correctly-classified shell alone: both a straight-down
    and a 35-degree oblique render come back a flat, featureless image -
    exactly task-3b's "solid, closed, rounded block from every angle"
    finding). On every asset, the by-far-largest islands are a pair of
    disconnected quads ~1.5mm apart, both near the shell's own world
    z_max, spanning 50-90% of the footprint, with zero shared edges to
    the walls or floor - not a rim lip (those are real and present too,
    but tiny: under 3% of footprint on every asset measured here, versus
    this pair's 50-90%). These faces, not a failed split, are what make a
    correctly-split, correctly-classified shell render solid: every
    straight-down camera ray hits one of them before ever reaching the
    module, the placement-area proxy, or the true cavity floor below.

    `area_frac`/`top_tol`/`flatness_tol` are deliberately generous
    relative to that measured gap, so a legitimate rim/lip (small area)
    or a real wall (spans most of the case's own height, so fails the
    flatness check even though its top also reaches near the rim) is
    never touched. Raises if a large near-top flat island still remains
    after stripping - the removal must not silently leave the exact
    defect it exists to fix. bpy-only geometry edit; per this task's
    constraints it has no unit test, only this assertion and the render
    (see task-3c-report.md).
    """
    bpy.context.view_layer.update()
    mw = case_obj.matrix_world.copy()
    lo, hi = group_bbox([case_obj])
    footprint_area = (hi.x - lo.x) * (hi.y - lo.y)
    if footprint_area <= 0:
        return

    bm = bmesh.new()
    bm.from_mesh(case_obj.data)
    bm.faces.ensure_lookup_table()
    bad = _oversized_near_top_islands(bm, mw, hi.z, footprint_area,
                                      top_tol, flatness_tol, area_frac)
    if bad:
        bmesh.ops.delete(bm, geom=[f for comp, _ in bad for f in comp],
                         context='FACES')
        bm.to_mesh(case_obj.data)
        case_obj.data.update()
    bm.free()
    if not bad:
        return

    bpy.context.view_layer.update()
    bm2 = bmesh.new()
    bm2.from_mesh(case_obj.data)
    bm2.faces.ensure_lookup_table()
    still_bad = _oversized_near_top_islands(bm2, mw, hi.z, footprint_area,
                                            top_tol, flatness_tol, area_frac)
    bm2.free()
    if still_bad:
        worst = max(a for _, a in still_bad)
        raise RuntimeError(
            f"_strip_phantom_cap: {case_obj.name!r} still has a near-top "
            f"flat island covering {worst / footprint_area:.1%} of its "
            f"own footprint after stripping - the removal did not clear "
            f"the defect it exists to fix.")


def object_role(o) -> str:
    """The role `_load_template` tagged `o` with (see `ROLE_PROP`),
    falling back to the name-based `catalog.role_of` for anything the tag
    is absent from - an object `world.py`/`scene.py` built directly (a
    PCB board, a bay proxy, an obstruction, a seated cell) rather than
    cloned from a template never gets tagged, and does not need to: those
    are labelled by scene.py explicitly, not looked up by role.

    This is the ONLY correct way for scene.py to ask a CAD-derived
    object's role: plain `catalog.role_of(o.name)` cannot tell `case`
    from `case_liner` apart (see `catalog.classify_case_parts`), and would
    silently misclassify a `case_liner` object as `case` since both keep
    the same name prefix after `_split_multi_material_case`.
    """
    role = o.get(ROLE_PROP)
    return role if role else role_of(o.name)


# --------------------------------------------------------------------------- #
#  library
# --------------------------------------------------------------------------- #

class AssetLibrary:
    def __init__(self, assets_dir: str):
        self.dir = os.path.abspath(assets_dir)
        self.catalog = load_catalog(self.dir)
        self.assets = {a["name"]: a for a in self.catalog["assets"]}
        if self.catalog.get("units") != "m":
            print(f"[warn] catalog units are {self.catalog.get('units')}, "
                  "expected metres")
        self._templates: Dict[str, Dict[str, list]] = {}

    def names(self) -> List[str]:
        return sorted(self.assets)

    def catalog_entry(self, name: str) -> Optional[dict]:
        """The raw catalog.json entry for `name`, or None if unknown."""
        return self.assets.get(name)

    # ---- import once ------------------------------------------------------ #
    def _load_template(self, name: str) -> Dict[str, list]:
        if name in self._templates:
            return self._templates[name]

        path = os.path.join(self.dir, self.assets[name]["file"])
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]

        # Un-merge the case's outer shell from its inner liner BEFORE
        # anything else touches these objects - see
        # _split_multi_material_case's docstring. Must run first: the
        # parent-detach and lay_flat/recentre steps below have to see
        # both resulting pieces, not the one merged object.
        meshes = _split_multi_material_case(meshes)

        # detach from any imported empties so transforms are independent
        for o in meshes:
            if o.parent:
                world = o.matrix_world.copy()
                o.parent = None
                o.matrix_world = world
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)

        bpy.context.view_layer.update()
        lay_flat(meshes)

        # re-centre on the origin so instances place predictably
        lo, hi = group_bbox(meshes)
        centre = (lo + hi) / 2
        for o in meshes:
            o.location -= Vector((centre.x, centre.y, lo.z))

        coll = _template_collection()
        by_role: Dict[str, list] = {}
        for o in meshes:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
            o.hide_render = True
            by_role.setdefault(role_of(o.name), []).append(o)

        # Geometric refinement: role_of() alone still can't tell the two
        # split-off case pieces apart (same name prefix), so decide by XY
        # footprint area - see _classify_case_liner. Then tag every object
        # with its FINAL role so scene.py never has to re-derive one from
        # a (possibly ".001"-suffixed) name; object_role() reads this.
        _classify_case_liner(by_role)

        # Strip any phantom cap merged into the shell via a shared
        # material - see _strip_phantom_cap. Must run AFTER
        # _classify_case_liner (needs the final "case" piece, in the
        # already lay_flat'd + recentred world frame) and BEFORE this
        # template is cached/cloned, so every instance gets the stripped
        # mesh, not just the first one built.
        for o in by_role.get("case", []):
            _strip_phantom_cap(o)

        for role, objs in by_role.items():
            for o in objs:
                o[ROLE_PROP] = role

        bpy.context.view_layer.update()
        self._templates[name] = by_role
        return by_role

    # ---- instantiate ------------------------------------------------------ #
    def instantiate(self, name: str, variant: Variant, rng: random.Random
                    ) -> List[Item]:
        """
        Build scene items for one asset under one presentation variant.

        Returns one Item for an intact assembly, or several when the variant
        explodes it into separately placeable pieces.
        """
        by_role = self._load_template(name)
        target = bpy.context.scene.collection

        kept = {r: [clone(o, target) for o in objs]
                for r, objs in by_role.items() if r in variant.keep_roles}
        if not any(kept.values()):
            return []

        items: List[Item] = []

        if variant.label is not None:
            # one rigid object: keep the relative transforms from the CAD
            objs = [o for lst in kept.values() for o in lst]
            labels = {o.name: variant.label for o in objs}
            bpy.context.view_layer.update()
            lo, hi = group_bbox(objs)
            items.append(Item(objects=objs, labels=labels, merge=True,
                              asset=name, variant=variant.name,
                              footprint=(hi.x - lo.x, hi.y - lo.y)))
            return items

        # exploded: each sub-part (or role group) becomes its own item
        for role, objs in kept.items():
            cls = variant.label_roles.get(role)
            if cls is None:
                for o in objs:
                    bpy.data.objects.remove(o, do_unlink=True)
                continue
            if role == "case":
                # shell halves stay together as one object
                bpy.context.view_layer.update()
                lo, hi = group_bbox(objs)
                items.append(Item(objects=objs,
                                  labels={o.name: cls for o in objs},
                                  merge=True, asset=name, variant=variant.name,
                                  footprint=(hi.x - lo.x, hi.y - lo.y)))
            else:
                for o in objs:
                    o.rotation_euler = (0, 0, 0)
                    o.location = (0, 0, 0)
                    bpy.context.view_layer.update()
                    lay_flat([o])
                    lo, hi = group_bbox([o])
                    items.append(Item(objects=[o], labels={o.name: cls},
                                      merge=False, asset=name,
                                      variant=variant.name,
                                      footprint=(hi.x - lo.x, hi.y - lo.y)))
        return items


def place_item(item: Item, placement, rng: random.Random):
    """
    Apply a Placement: rotate about Z, move into position, sit on
    `placement.z` (0 for every non-overlapping placement).

    Composes one world matrix and applies it to every member, for the same two
    reasons lay_flat does - `o.rotation_euler.z += rot` is a no-op under the
    importer's QUATERNION rotation mode, which silently threw away every
    rotation layout.py asked for and left the dataset with zero rotation
    diversity. The gate asserts two rot_deg values give two different bboxes.

    The lift is applied AFTER `drop_to_floor`, so it is measured from the
    part's own lowest point rather than from wherever the import left it.
    `layout.plan` sets it to the top of whatever the item overlaps: two parts
    sharing ground area cannot both rest on the floor without occupying the
    same space, and a render of interpenetrating solids is not a render of one
    part lying across another.

    Calls `bpy.context.view_layer.update()` UNCONDITIONALLY after
    `drop_to_floor`, not only inside the `if lift:` branch (task-3c: the
    previous code only refreshed the depsgraph when a lift was applied,
    so for the common `lift == 0` case, `drop_to_floor`'s
    `o.location.z -= lo.z` write sat un-evaluated - `matrix_world`/
    `bound_box` kept reporting the PRE-drop position to any caller that
    read them before some UNRELATED later object happened to trigger an
    update. `scene.py`'s open_case block is exactly such a caller
    (`floor_z = lo.z + tray_floor_mm/1000`, read straight after
    placement): it silently computed `floor_z` as if the case's floor sat
    at its own TOP (an 11.1mm offset, exactly the shell's own height) for
    every case that was not itself lifted, seating the module and bay
    proxy ~11mm above the case's true rim - floating in open air, clear
    of the case entirely. That is why some renders "worked": the module
    was never inside the tray to be occluded. See task-3c-report.md.

    rng is unused; it is kept for signature stability with the callers and with
    the other placement helpers.
    """
    bpy.context.view_layer.update()
    lo, hi = group_bbox(item.objects)
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, 0.0))
    R = Matrix.Rotation(math.radians(placement.rot_deg), 4, "Z")
    M = (Matrix.Translation(Vector((placement.x, placement.y, 0.0)))
         @ R @ Matrix.Translation(-centre))
    for o in item.objects:
        o.matrix_world = M @ o.matrix_world
    bpy.context.view_layer.update()
    drop_to_floor(item.objects)
    bpy.context.view_layer.update()

    lift = float(getattr(placement, "z", 0.0) or 0.0)
    if lift:
        for o in item.objects:
            o.location.z += lift
        bpy.context.view_layer.update()

    # Loud post-condition (task-3c): the group's own z-min must now sit at
    # exactly `lift` (0 for the common non-overlapping case). Trusting
    # drop_to_floor's arithmetic without checking is exactly what let the
    # stale-update bug above go unnoticed for three render cycles - assert
    # the RESULT, not that the code ran.
    lo_final, _ = group_bbox(item.objects)
    if abs(lo_final.z - lift) > 5e-4:
        raise RuntimeError(
            f"place_item: post-condition failed for "
            f"{[o.name for o in item.objects]} - expected the group's "
            f"z-min to settle at {lift * 1000:.3f}mm after drop_to_floor "
            f"(+lift), but matrix_world/bound_box still reports "
            f"{lo_final.z * 1000:.3f}mm. This is the stale-transform "
            f"signature: a bpy.context.view_layer.update() was skipped "
            f"somewhere between the transform write and this read.")
