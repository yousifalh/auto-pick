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
from typing import Dict, List, Tuple

import bpy
from mathutils import Matrix, Vector

from .catalog import load_catalog, role_of
from .config import Variant


TEMPLATE_COLLECTION = "_synth3d_templates"


@dataclass
class Item:
    """One independently placeable thing: an assembly, or a single loose part."""
    objects: List[object] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)   # obj.name -> class
    merge: bool = False          # collapse member boxes into one
    asset: str = ""
    variant: str = ""
    footprint: Tuple[float, float] = (0.0, 0.0)
    local_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)


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

    # ---- import once ------------------------------------------------------ #
    def _load_template(self, name: str) -> Dict[str, list]:
        if name in self._templates:
            return self._templates[name]

        path = os.path.join(self.dir, self.assets[name]["file"])
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]

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

        def clone(src):
            dup = src.copy()                 # linked duplicate: shares mesh data
            dup.data = src.data
            target.objects.link(dup)
            dup.hide_render = False
            dup.location = src.location.copy()
            dup.rotation_euler = src.rotation_euler.copy()
            # per-instance materials despite shared mesh data
            for slot in dup.material_slots:
                slot.link = "OBJECT"
            return dup

        kept = {r: [clone(o) for o in objs]
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
            centre = (lo + hi) / 2
            items.append(Item(objects=objs, labels=labels, merge=True,
                              asset=name, variant=variant.name,
                              footprint=(hi.x - lo.x, hi.y - lo.y),
                              local_offsets={o.name: (o.location.x - centre.x,
                                                      o.location.y - centre.y)
                                             for o in objs}))
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
    Apply a Placement: rotate about Z, move into position, sit on z=0.

    Composes one world matrix and applies it to every member, for the same two
    reasons lay_flat does - `o.rotation_euler.z += rot` is a no-op under the
    importer's QUATERNION rotation mode, which silently threw away every
    rotation layout.py asked for and left the dataset with zero rotation
    diversity. The gate asserts two rot_deg values give two different bboxes.

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
