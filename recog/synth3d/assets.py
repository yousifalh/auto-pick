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

import bpy
from mathutils import Matrix, Vector

from .bay import needs_flip
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


def flip_if_inverted(objects):
    """Correct an upside-down import: after `lay_flat` has picked WHICH
    axis is vertical, this decides WHICH END of that axis is up - a
    question `lay_flat` has no way to answer (see its docstring; it only
    compares extents, never orientation).

    Task-3c: Blender's glTF importer maps the source (x, y, z) to
    (x, -z, y). For this CAD the raw file's up-axis is Y and the cavity
    opens toward -Y, so -Y lands on -Z - the assembly imports with its
    tray facing the ground. A top-down camera then sees the shell's own
    solid, featureless outer underside: exactly the "solid closed box"
    symptom three prior rounds chased into the split/material/geometry
    layer, when the true fault was orientation, decided nowhere.

    Detects it the same way `bay.needs_flip` documents: the lid
    (`case_lid`) must sit ABOVE the shell (`case` - both split pieces,
    shell and liner, since either reliably indicates which side the case
    body is on). A no-op when `objects` has no `case` or no `case_lid`
    (e.g. the `cells_only` variant's template has neither) - nothing to
    orient relative to.

    Applies the SAME pivot-conjugated whole-group rotation `lay_flat`
    uses (a single `matrix_world` write per object, about the group's own
    pivot) for the SAME two reasons documented on `lay_flat`: per-object
    `rotation_euler` writes are silently ignored under the importer's
    QUATERNION rotation mode, and even forced to XYZ they rotate each
    object about its own origin, tearing the assembly apart rather than
    turning it over as one rigid body.
    """
    cases = [o for o in objects if role_of(o.name) == "case"]
    lids = [o for o in objects if role_of(o.name) == "case_lid"]
    if not cases or not lids:
        return

    case_lo, case_hi = group_bbox(cases)
    lid_lo, lid_hi = group_bbox(lids)
    case_z = (case_lo.z + case_hi.z) / 2
    lid_z = (lid_lo.z + lid_hi.z) / 2
    if not needs_flip(case_z, lid_z):
        return

    lo, hi = group_bbox(objects)
    pivot = (lo + hi) / 2
    R = Matrix.Rotation(math.radians(180), 4, "X")
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

    # ---- import once ------------------------------------------------------ #
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

    lift = float(getattr(placement, "z", 0.0) or 0.0)
    if lift:
        for o in item.objects:
            o.location.z += lift
        bpy.context.view_layer.update()
