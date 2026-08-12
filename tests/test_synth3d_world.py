"""`recog/synth3d/world.py` under a stub `bpy` - the decisions and guards.

world.py is the largest file in the project (1495 lines) and, because it
imports `bpy`, no test in this suite could reach a single line of it. Every
serious silent failure this project has had lived in, or was masked by,
exactly that region: an inverted assembly, a two-material object that arrived
fused, a renamed catalog key that quietly stopped building geometry, a
swallowed exception that discarded the drawn roughness on 100% of surfaces
while the manifest recorded the discarded values. None was caught by a test,
because no test could run the code.

This file loads world.py under stub `bpy`/`bmesh`/`mathutils` modules and
drives its real functions, following `tests/test_synth3d_materials.py`'s
approach (private module name, stubs removed from `sys.modules` afterwards)
rather than inventing a second harness.

WHAT THIS HARNESS CAN DETECT
    Logic errors in world.py's own arithmetic and control flow: a seating
    height that stops clearing the proxy it must occlude, a guard that stops
    firing, an early return that silently builds nothing, a fallback that
    changes shape, a manifest field that stops matching the geometry it
    describes, and - via `_assert_procedural_tray_geometry` and
    `_assert_seat_cell_footprint`, driven here against deliberately broken
    input - whether world.py's own loud assertions still fire on the exact
    historical defects they were written for.

WHAT IT CANNOT DETECT
    Anything about Blender. The `bpy` here is roughly 400 lines of stub that
    models objects as transformable axis-aligned boxes; it is NOT Blender and
    a green run here says nothing about:
      * a bpy API change (a renamed socket, a moved operator, a changed
        `transform_apply` default) - the stub would keep implementing the old
        contract quite happily;
      * renderer behaviour - z-fighting, shadow terminators, the coplanar
        black-plate failure JIG_LIFT exists for, whether a bevel looks right;
      * `bmesh.ops.bevel` and boolean modifiers, which are stubbed as, in
        turn, a no-op and an AABB-preserving no-op. `_crown_lid`'s actual
        fillet is NOT exercised. Only its guard clause and its early return
        are.
      * whether Blender's glTF importer still orients CAD the way
        `flip_if_inverted` assumes.
    A stub bpy tests OUR logic, not Blender's behaviour. Reading these tests
    as broader assurance than that is precisely how the earlier defects
    survived a green suite.

The stub's own semantics (matrix composition, parenting, `transform_apply`
baking) are pinned by the first section below, so a test that fails is a
statement about world.py and not about an unexamined fake.
"""

import ast
import contextlib
import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORLD_PY = ROOT / "recog" / "synth3d" / "world.py"


# =========================================================================== #
#  stub mathutils - numpy-backed, so the linear algebra is not hand-rolled
# =========================================================================== #

class Vector:
    """A 3-component `mathutils.Vector` stand-in."""

    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self._v = np.asarray(tuple(seq), dtype=float)

    # -- component access; the setters matter because world.py and assets.py
    #    both write through them (`cam.location.z = ...`, `o.location.z -= ...`)
    @property
    def x(self):
        return float(self._v[0])

    @x.setter
    def x(self, val):
        self._write(0, val)

    @property
    def y(self):
        return float(self._v[1])

    @y.setter
    def y(self, val):
        self._write(1, val)

    @property
    def z(self):
        return float(self._v[2])

    @z.setter
    def z(self, val):
        self._write(2, val)

    def _write(self, i, val):
        self._v[i] = float(val)

    def __getitem__(self, i):
        return float(self._v[i])

    def __setitem__(self, i, val):
        self._write(i, val)

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(float(c) for c in self._v)

    def __add__(self, other):
        return Vector(self._v + np.asarray(tuple(other), dtype=float))

    def __sub__(self, other):
        return Vector(self._v - np.asarray(tuple(other), dtype=float))

    def __neg__(self):
        return Vector(-self._v)

    def __mul__(self, k):
        return Vector(self._v * float(k))

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vector(self._v / float(k))

    def copy(self):
        return Vector(self._v)

    def __repr__(self):
        return f"Vector(({self.x:.6g}, {self.y:.6g}, {self.z:.6g}))"


class _BoundVector(Vector):
    """The vector `Object.location` hands back.

    In Blender `obj.location.z = v` mutates the object, not a copy. Both
    `world.setup_camera` (`cam.location.z = ...`) and `assets.drop_to_floor`
    (`o.location.z -= lo.z`) depend on that, so the stub has to reproduce it
    rather than quietly dropping the write.
    """

    def __init__(self, seq, owner):
        super().__init__(seq)
        self._owner = owner

    def _write(self, i, val):
        self._v[i] = float(val)
        self._owner._set_translation(self._v)


class Matrix:
    """A 4x4 `mathutils.Matrix` stand-in over numpy."""

    def __init__(self, m=None):
        self._m = np.eye(4) if m is None else np.asarray(m, dtype=float)

    @classmethod
    def Identity(cls, size=4):
        assert size == 4
        return cls()

    @classmethod
    def Translation(cls, v):
        m = np.eye(4)
        m[:3, 3] = np.asarray(tuple(v), dtype=float)[:3]
        return cls(m)

    @classmethod
    def Rotation(cls, angle, size, axis):
        assert size == 4, "only the 4x4 form is used by this codebase"
        c, s = math.cos(angle), math.sin(angle)
        m = np.eye(4)
        if axis == "X":
            m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
        elif axis == "Y":
            m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
        elif axis == "Z":
            m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        else:
            raise ValueError(f"bad axis {axis!r}")
        return cls(m)

    @classmethod
    def Diagonal(cls, v):
        m = np.eye(4)
        m[0, 0], m[1, 1], m[2, 2] = (float(c) for c in tuple(v)[:3])
        return cls(m)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self._m @ other._m)
        if isinstance(other, Vector):
            # Blender treats `mat4 @ vec3` as a POINT transform - the
            # translation column applies. group_bbox depends on that.
            h = np.append(np.asarray(tuple(other), dtype=float), 1.0)
            return Vector((self._m @ h)[:3])
        return NotImplemented

    def inverted(self):
        return Matrix(np.linalg.inv(self._m))

    def copy(self):
        return Matrix(self._m.copy())

    def to_translation(self):
        return Vector(self._m[:3, 3])

    def __repr__(self):
        return f"Matrix({np.array2string(self._m, precision=5)})"


# =========================================================================== #
#  stub bpy
#
#  Objects are modelled as an axis-aligned local box (`bound_box`, 8 corners)
#  plus a basis matrix, composed exactly as Blender composes one:
#
#      matrix_basis = T(location) @ R(rotation_euler) @ S(scale)
#      matrix_world = parent.matrix_world @ matrix_parent_inverse @ basis
#
#  `transform_apply` bakes the requested components into the local box and
#  resets them, which is what Blender does to the mesh data.
# =========================================================================== #

_CUBE_CORNERS = np.array([(sx, sy, sz)
                          for sx in (-0.5, 0.5)
                          for sy in (-0.5, 0.5)
                          for sz in (-0.5, 0.5)], dtype=float)

# Outward normals of the six axis-aligned faces of a box, in the order
# `_face_polys` builds them.
_BOX_FACE_NORMALS = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                              (0, 1, 0), (0, 0, -1), (0, 0, 1)], dtype=float)


class _Vert:
    def __init__(self, co):
        self.co = Vector(co)


class _Poly:
    def __init__(self, normal):
        self.normal = Vector(normal)
        self.use_smooth = False


class _MaterialSlots(list):
    def append(self, mat):          # bpy's `mesh.materials.append`
        super().append(mat)


class _Mesh:
    """Just enough mesh for `bound_box`, the crown check, and material slots.

    `vertices` are the eight box corners and `polygons` the six axis faces -
    the real thing for the cuboids world.py builds, an AABB approximation for
    the cylinder and sphere primitives (which is all `group_bbox` reads of
    them anyway, since `bound_box` is itself an AABB in Blender too).
    """

    def __init__(self, corners):
        self.materials = _MaterialSlots()
        self._set_corners(corners)
        self.updated = 0

    def _set_corners(self, corners):
        self._corners = np.asarray(corners, dtype=float)
        self.vertices = [_Vert(c) for c in self._corners]
        self.polygons = [_Poly(n) for n in _BOX_FACE_NORMALS]

    def update(self):
        self.updated += 1


class _Light:
    def __init__(self):
        self.energy = 10.0
        self.shape = "SQUARE"
        self.size = 0.25
        self.color = (1.0, 1.0, 1.0)


class _Camera:
    def __init__(self):
        self.type = "PERSP"
        self.lens = 50.0
        self.sensor_width = 36.0      # Blender's default
        self.ortho_scale = 6.0
        self.clip_start = 0.1
        self.clip_end = 100.0


class _Modifier:
    def __init__(self, name, mtype):
        self.name = name
        self.type = mtype
        self.operation = None
        self.object = None


class _Modifiers:
    def __init__(self):
        self._mods = []

    def new(self, name, type):
        m = _Modifier(name, type)
        self._mods.append(m)
        return m

    def get(self, name):
        for m in self._mods:
            if m.name == name:
                return m
        return None

    def remove(self, mod):
        self._mods.remove(mod)

    def __iter__(self):
        return iter(self._mods)

    def __len__(self):
        return len(self._mods)


class _Slot:
    """A material slot. `link` decides whether the material is read off the
    OBJECT or the mesh DATA - the distinction `assets.clone` and
    `materials.apply_to_object` turn on, so that linked duplicates sharing one
    mesh still get independent materials."""

    def __init__(self, obj, index):
        self._obj = obj
        self._index = index
        self.link = "DATA"
        self._material = None

    @property
    def material(self):
        if self.link == "OBJECT":
            return self._material
        return self._obj.data.materials[self._index]

    @material.setter
    def material(self, mat):
        if self.link == "OBJECT":
            self._material = mat
        else:
            self._obj.data.materials[self._index] = mat


class Object:
    def __init__(self, name, corners, data=None):
        self.name = name
        self.data = _Mesh(corners) if data is None else data
        self.pass_index = 0
        self.parent = None
        self.matrix_parent_inverse = Matrix()
        self.modifiers = _Modifiers()
        self.applied_modifiers = []
        self.hide_render = False
        self._slots = []
        self._loc = np.zeros(3)
        self._rot = np.zeros(3)
        self._scale = np.ones(3)
        self._basis_override = None

    @property
    def material_slots(self):
        want = len(self.data.materials)
        while len(self._slots) < want:
            self._slots.append(_Slot(self, len(self._slots)))
        del self._slots[want:]
        return self._slots

    def copy(self):
        """`bpy.types.Object.copy()` - a new object sharing the same mesh."""
        dup = Object(self.name, self.data._corners.copy(), data=self.data)
        dup._loc = self._loc.copy()
        dup._rot = self._rot.copy()
        dup._scale = self._scale.copy()
        if self._basis_override is not None:
            dup._basis_override = self._basis_override.copy()
        dup.pass_index = self.pass_index
        _register(dup, self.name)
        return dup

    # -- basis ---------------------------------------------------------- #

    def _set_translation(self, v):
        self._loc = np.asarray(v, dtype=float).copy()
        if self._basis_override is not None:
            self._basis_override._m[:3, 3] = self._loc

    def _compose(self):
        if self._basis_override is not None:
            return self._basis_override
        R = (Matrix.Rotation(self._rot[2], 4, "Z")
             @ Matrix.Rotation(self._rot[1], 4, "Y")
             @ Matrix.Rotation(self._rot[0], 4, "X"))
        return (Matrix.Translation(self._loc) @ R
                @ Matrix.Diagonal(self._scale))

    @property
    def location(self):
        return _BoundVector(self._loc, self)

    @location.setter
    def location(self, v):
        self._set_translation(tuple(v)[:3])

    @property
    def rotation_euler(self):
        return Vector(self._rot)

    @rotation_euler.setter
    def rotation_euler(self, v):
        self._rot = np.asarray(tuple(v)[:3], dtype=float)
        self._basis_override = None
        self._set_translation(self._loc)

    @property
    def scale(self):
        return Vector(self._scale)

    @scale.setter
    def scale(self, v):
        self._scale = np.asarray(tuple(v)[:3], dtype=float)
        self._basis_override = None
        self._set_translation(self._loc)

    @property
    def matrix_basis(self):
        return self._compose()

    @property
    def matrix_world(self):
        basis = self._compose()
        if self.parent is None:
            return basis
        return self.parent.matrix_world @ self.matrix_parent_inverse @ basis

    @matrix_world.setter
    def matrix_world(self, m):
        # Parentless is the only case world.py assigns through, which is what
        # Blender's own `matrix_world = M` shortcut assumes too.
        assert self.parent is None, (
            "the stub only models `matrix_world =` on a parentless object; "
            "world.py never assigns it on a parented one")
        self._basis_override = Matrix(m._m.copy())
        self._loc = m._m[:3, 3].copy()

    # -- geometry ------------------------------------------------------- #

    @property
    def bound_box(self):
        """Blender's `bound_box`: the LOCAL-space AABB corners of the mesh."""
        c = self.data._corners
        lo, hi = c.min(axis=0), c.max(axis=0)
        return [(x, y, z) for x in (lo[0], hi[0])
                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]

    def world_bbox(self):
        pts = np.array([tuple(self.matrix_world @ Vector(c))
                        for c in self.bound_box])
        return pts.min(axis=0), pts.max(axis=0)


def _register(obj, base_name):
    """Put `obj` in `bpy.data.objects` under a uniquified name, as Blender
    does when an object is created or copied."""
    store = _STUB_BPY.data.objects
    key, n = base_name, 1
    while key in store:
        key, n = f"{base_name}.{n:03d}", n + 1
    obj.name = key
    store[key] = obj
    return obj


class _ObjectsLink(list):
    """A collection's `.objects`, which is linked into rather than appended."""

    def link(self, obj):
        if obj not in self:
            self.append(obj)

    def unlink(self, obj):
        if obj in self:
            self.remove(obj)


class _Collection(dict):
    """`bpy.data.<x>` - a name-keyed store with `.new()` and iteration."""

    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def new(self, name, *a, **kw):
        obj = self._factory(name, *a, **kw)
        # Blender uniquifies clashing names with a .001 suffix; reproduce
        # that so a test can tell two same-named materials apart.
        key, n = name, 1
        while key in self:
            key, n = f"{name}.{n:03d}", n + 1
        obj.name = key
        self[key] = obj
        return obj

    def __iter__(self):
        return iter(list(self.values()))

    def get(self, name, default=None):
        return super().get(name, default)


# ------------------------------------------------------------ node trees ----

class _Socket:
    def __init__(self, name, default=0.0):
        self.name = name
        self.identifier = name
        self.default_value = default
        self.links = []

    @property
    def is_linked(self):
        return bool(self.links)


class _Sockets:
    def __init__(self, names_and_defaults):
        self._s = [_Socket(n, d) for n, d in names_and_defaults]

    def __iter__(self):
        return iter(self._s)

    def __contains__(self, name):
        return any(s.name == name for s in self._s)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._s[key]
        for s in self._s:
            if s.name == key:
                return s
        raise KeyError(
            f"no socket named {key!r}; this node offers "
            f"{[s.name for s in self._s]}")


class _RampElement:
    def __init__(self, position):
        self.position = position
        self.color = (0.0, 0.0, 0.0, 1.0)


class _ColorRamp:
    def __init__(self):
        self.elements = [_RampElement(0.0), _RampElement(1.0)]


# Socket tables for every node type world.py creates. A type that is not
# listed raises, so a new `nodes.new(...)` in world.py forces this harness to
# be updated rather than silently accepting anything.
_NODE_SOCKETS = {
    "ShaderNodeTexCoord": ([], ["Generated", "Normal", "UV", "Object",
                                "Camera", "Window", "Reflection"]),
    "ShaderNodeMapping": ([("Vector", (0.0, 0.0, 0.0)),
                           ("Location", (0.0, 0.0, 0.0)),
                           ("Rotation", (0.0, 0.0, 0.0)),
                           ("Scale", (1.0, 1.0, 1.0))], ["Vector"]),
    "ShaderNodeTexNoise": ([("Vector", (0.0, 0.0, 0.0)), ("Scale", 5.0),
                            ("Detail", 2.0), ("Roughness", 0.5)],
                           ["Fac", "Color"]),
    "ShaderNodeValToRGB": ([("Fac", 0.5)], ["Color", "Alpha"]),
    "ShaderNodeTexImage": ([("Vector", (0.0, 0.0, 0.0))], ["Color", "Alpha"]),
    "ShaderNodeBrightContrast": ([("Color", (0.0,) * 4), ("Bright", 0.0),
                                  ("Contrast", 0.0)], ["Color"]),
    "ShaderNodeBump": ([("Strength", 1.0), ("Distance", 1.0), ("Height", 1.0),
                        ("Normal", (0.0, 0.0, 0.0))], ["Normal"]),
    "ShaderNodeTexWave": ([("Vector", (0.0, 0.0, 0.0)), ("Scale", 5.0),
                           ("Distortion", 0.0)], ["Fac", "Color"]),
    "ShaderNodeTexEnvironment": ([("Vector", (0.0, 0.0, 0.0))], ["Color"]),
    # The two nodes a fresh node tree already contains.
    "ShaderNodeBsdfPrincipled": (
        [("Base Color", (0.8, 0.8, 0.8, 1.0)), ("Metallic", 0.0),
         ("Roughness", 0.5), ("IOR", 1.45), ("Alpha", 1.0),
         ("Normal", (0.0, 0.0, 0.0)), ("Emission Color", (1.0,) * 4),
         ("Emission Strength", 0.0)], ["BSDF"]),
    "ShaderNodeBackground": ([("Color", (0.05,) * 4), ("Strength", 1.0)],
                             ["Background"]),
    "ShaderNodeOutputMaterial": ([("Surface", None)], []),
    "ShaderNodeOutputWorld": ([("Surface", None)], []),
}

_DEFAULT_NODE_NAME = {
    "ShaderNodeBsdfPrincipled": "Principled BSDF",
    "ShaderNodeBackground": "Background",
    "ShaderNodeOutputMaterial": "Material Output",
    "ShaderNodeOutputWorld": "World Output",
}


class _Node:
    def __init__(self, ntype, name):
        if ntype not in _NODE_SOCKETS:
            raise KeyError(
                f"the stub bpy has no socket table for node type {ntype!r}; "
                f"world.py has grown a node this harness does not model")
        ins, outs = _NODE_SOCKETS[ntype]
        self.bl_idname = ntype
        self.name = name
        self.inputs = _Sockets(ins)
        self.outputs = _Sockets([(n, None) for n in outs])
        if ntype == "ShaderNodeValToRGB":
            self.color_ramp = _ColorRamp()
        if ntype in ("ShaderNodeTexImage", "ShaderNodeTexEnvironment"):
            self.image = None
            self.extension = "REPEAT"
        if ntype == "ShaderNodeTexWave":
            self.wave_type = "BANDS"
            self.bands_direction = "X"


class _Nodes(list):
    def new(self, ntype):
        n = _Node(ntype, _DEFAULT_NODE_NAME.get(ntype, ntype))
        self.append(n)
        return n

    def get(self, name, default=None):
        for n in self:
            if n.name == name:
                return n
        return default


class _Link:
    def __init__(self, from_socket, to_socket):
        self.from_socket = from_socket
        self.to_socket = to_socket


class _Links(list):
    def new(self, from_socket, to_socket):
        if from_socket is None or to_socket is None:
            raise ValueError("cannot link a None socket")
        link = _Link(from_socket, to_socket)
        from_socket.links.append(link)
        to_socket.links.append(link)
        self.append(link)
        return link


class _NodeTree:
    def __init__(self, kind):
        self.nodes = _Nodes()
        self.links = _Links()
        if kind == "material":
            self.nodes.new("ShaderNodeBsdfPrincipled")
            self.nodes.new("ShaderNodeOutputMaterial")
        else:
            self.nodes.new("ShaderNodeBackground")
            self.nodes.new("ShaderNodeOutputWorld")


class _NodeHolder:
    """Shared by Material and World: `use_nodes = True` builds the tree."""
    _KIND = "material"

    def __init__(self, name):
        self.name = name
        self._use_nodes = False
        self.node_tree = None
        self.blend_method = "OPAQUE"

    @property
    def use_nodes(self):
        return self._use_nodes

    @use_nodes.setter
    def use_nodes(self, val):
        self._use_nodes = bool(val)
        if val and self.node_tree is None:
            self.node_tree = _NodeTree(self._KIND)


class Material(_NodeHolder):
    _KIND = "material"


class World(_NodeHolder):
    _KIND = "world"


class Image:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = Path(filepath).name
        self.colorspace_settings = types.SimpleNamespace(name="sRGB")


def _make_bpy():
    """Build the stub `bpy` module. One module, reset between tests."""
    bpy = types.ModuleType("bpy")

    bpy.data = types.SimpleNamespace()
    bpy.context = types.SimpleNamespace()
    bpy.ops = types.SimpleNamespace()
    bpy.ops.mesh = types.SimpleNamespace()
    bpy.ops.object = types.SimpleNamespace()

    def reset():
        bpy.data.objects = _Collection(lambda name, *a: Object(name, a[0]))
        bpy.data.materials = _Collection(Material)
        bpy.data.worlds = _Collection(World)
        bpy.data.images = types.SimpleNamespace(
            load=lambda path, **kw: Image(path))
        bpy.data.collections = _Collection(
            lambda name: types.SimpleNamespace(
                name=name, objects=[], children=[],
                hide_render=False, hide_viewport=False))

        def _remove(obj, do_unlink=False):
            for k, v in list(bpy.data.objects.items()):
                if v is obj:
                    del bpy.data.objects[k]
                    return
            raise ReferenceError(f"{obj.name} is not in bpy.data.objects")

        bpy.data.objects.remove = _remove

        scene = types.SimpleNamespace(camera=None, world=None,
                                      collection=types.SimpleNamespace(
                                          children=types.SimpleNamespace(
                                              link=lambda c: None),
                                          objects=_ObjectsLink()))
        view_layer = types.SimpleNamespace(
            objects=types.SimpleNamespace(active=None),
            update=lambda: None)
        bpy.context.scene = scene
        bpy.context.view_layer = view_layer
        bpy.context.active_object = None
        bpy.data.objects_created = []

    def _spawn(name, corners, location=(0, 0, 0), rotation=(0, 0, 0),
               data=None):
        obj = Object(name, corners, data=data)
        obj.location = location
        obj.rotation_euler = rotation
        _register(obj, name)
        bpy.context.active_object = obj
        bpy.context.view_layer.objects.active = obj
        bpy.data.objects_created.append(obj)
        return obj

    # -- mesh primitives; corner sets match Blender's own primitives ----- #

    def primitive_cube_add(size=2.0, location=(0, 0, 0), **kw):
        _spawn("Cube", _CUBE_CORNERS * size, location)

    def primitive_plane_add(size=2.0, location=(0, 0, 0), **kw):
        c = _CUBE_CORNERS.copy() * size
        c[:, 2] = 0.0
        _spawn("Plane", c, location)

    def primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0),
                               **kw):
        c = _CUBE_CORNERS.copy()
        c[:, 0] *= 2 * radius
        c[:, 1] *= 2 * radius
        c[:, 2] *= depth
        _spawn("Cylinder", c, location)

    def primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0), **kw):
        _spawn("Sphere", _CUBE_CORNERS * (2 * radius), location)

    bpy.ops.mesh.primitive_cube_add = primitive_cube_add
    bpy.ops.mesh.primitive_plane_add = primitive_plane_add
    bpy.ops.mesh.primitive_cylinder_add = primitive_cylinder_add
    bpy.ops.mesh.primitive_uv_sphere_add = primitive_uv_sphere_add

    def light_add(type="POINT", location=(0, 0, 0), **kw):
        obj = _spawn("Light", _CUBE_CORNERS * 0.0, location, data=_Light())
        obj.light_type = type

    def camera_add(location=(0, 0, 0), rotation=(0, 0, 0), **kw):
        _spawn("Camera", _CUBE_CORNERS * 0.0, location, rotation,
               data=_Camera())

    bpy.ops.object.light_add = light_add
    bpy.ops.object.camera_add = camera_add

    def transform_apply(location=False, rotation=False, scale=False, **kw):
        """Bake the requested basis components into the mesh, as Blender does.

        Blender recomputes `bound_box` from the transformed vertices, so a
        baked rotation gives the AABB OF the rotated box. For the axis-aligned
        90-degree rotations world.py bakes that is exact.
        """
        obj = bpy.context.active_object
        assert obj is not None, "transform_apply with no active object"
        M = Matrix()
        if scale:
            M = Matrix.Diagonal(obj._scale) @ M
        if rotation:
            R = (Matrix.Rotation(obj._rot[2], 4, "Z")
                 @ Matrix.Rotation(obj._rot[1], 4, "Y")
                 @ Matrix.Rotation(obj._rot[0], 4, "X"))
            M = R @ M
        if location:
            M = Matrix.Translation(obj._loc) @ M
        pts = np.array([tuple(M @ Vector(c)) for c in obj.data._corners])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        obj.data._set_corners(np.array(
            [(x, y, z) for x in (lo[0], hi[0])
             for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
        if scale:
            obj._scale = np.ones(3)
        if rotation:
            obj._rot = np.zeros(3)
        if location:
            obj._loc = np.zeros(3)
        obj._basis_override = None
        obj._set_translation(obj._loc)

    bpy.ops.object.transform_apply = transform_apply

    def modifier_apply(modifier=None, **kw):
        """Apply a modifier to the ACTIVE object.

        Geometry is left alone: every boolean world.py applies is a DIFFERENCE
        with a cutter that is either strictly inside the target in XY or open
        through one face, neither of which changes the target's AABB. What
        this DOES model faithfully is which object the modifier was applied
        to and whether it existed - both real failure modes, since Blender
        applies to whatever `view_layer.objects.active` happens to be.
        """
        obj = bpy.context.view_layer.objects.active
        assert obj is not None, "modifier_apply with no active object"
        mod = obj.modifiers.get(modifier)
        if mod is None:
            raise RuntimeError(
                f"modifier {modifier!r} not found on {obj.name!r} - Blender "
                f"raises here too")
        obj.applied_modifiers.append((modifier, mod.operation, mod.object))
        obj.modifiers.remove(mod)

    bpy.ops.object.modifier_apply = modifier_apply

    def duplicate(**kw):            # used by assets.clone
        src = bpy.context.active_object
        _spawn(src.name, src.data._corners.copy(), tuple(src._loc))

    bpy.ops.object.duplicate = duplicate

    bpy.reset = reset
    reset()
    return bpy


def _make_bmesh():
    """A `bmesh` stub whose `ops.bevel` does NOTHING.

    That is deliberate, and it is a test in its own right: a bevel that
    silently fails to run is exactly the "helper that should have split it
    silently did not" defect this project has already had, and
    `_assert_procedural_tray_geometry`'s crown checks must catch it. See
    `test_a_bevel_that_silently_does_nothing_is_caught`.
    """
    bmesh = types.ModuleType("bmesh")

    class _BMVert:
        def __init__(self, co):
            self.co = Vector(co)

    class _BMEdge:
        def __init__(self, verts):
            self.verts = verts

    class _Table(list):
        def ensure_lookup_table(self):
            return None

    class _BMesh:
        def __init__(self):
            self.verts = _Table()
            self.edges = _Table()
            self.freed = False
            self.bevel_calls = []

        def from_mesh(self, mesh):
            self.verts = _Table(_BMVert(c) for c in mesh._corners)
            # The 12 edges of a box: corner pairs differing on ONE axis.
            self.edges = _Table()
            for i, a in enumerate(self.verts):
                for b in self.verts[i + 1:]:
                    if sum(1 for k in range(3)
                           if abs(a.co[k] - b.co[k]) > 1e-12) == 1:
                        self.edges.append(_BMEdge([a, b]))

        def to_mesh(self, mesh):
            pass

        def free(self):
            self.freed = True

    bmesh.new = _BMesh
    bmesh.bevel_calls = []
    bmesh.ops = types.SimpleNamespace(
        bevel=lambda bm, **kw: bmesh.bevel_calls.append(kw))
    bmesh.types = types.SimpleNamespace(BMesh=_BMesh)
    return bmesh


# =========================================================================== #
#  loading world.py under the stubs
# =========================================================================== #

_STUB_BPY = _make_bpy()
_STUB_BMESH = _make_bmesh()


def _load_world():
    """Execute world.py with stub `bpy`/`bmesh`/`mathutils` installed.

    Loaded under `recog.synth3d._world_stubbed` with `__package__` set, so its
    relative imports resolve - which means `assets.py` and `materials.py` are
    the REAL modules, running under the same stubs, not fakes of their own.

    Everything the load pulled in under a stub is removed from `sys.modules`
    and from the `recog.synth3d` package object afterwards, so no stubbed
    module leaks into the rest of the suite (a leaked `recog.synth3d.assets`
    would be importable and quietly wrong for every other test).
    """
    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = Matrix
    mathutils.Vector = Vector

    stubs = {"bpy": _STUB_BPY, "bmesh": _STUB_BMESH, "mathutils": mathutils}
    tainted = ("recog.synth3d.assets", "recog.synth3d.materials",
               "recog.synth3d._world_stubbed")

    saved = {name: sys.modules.get(name)
             for name in tuple(stubs) + tainted}
    for name, mod in stubs.items():
        sys.modules[name] = mod
    for name in tainted:
        sys.modules.pop(name, None)

    try:
        spec = importlib.util.spec_from_file_location(
            "recog.synth3d._world_stubbed", WORLD_PY)
        world = importlib.util.module_from_spec(spec)
        world.__package__ = "recog.synth3d"
        sys.modules["recog.synth3d._world_stubbed"] = world
        spec.loader.exec_module(world)
    finally:
        import recog.synth3d as pkg
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        for attr in ("assets", "materials", "_world_stubbed"):
            if getattr(pkg, attr, None) is not None \
                    and f"recog.synth3d.{attr}" not in sys.modules:
                delattr(pkg, attr)
    return world


@pytest.fixture(scope="module")
def W():
    return _load_world()


@pytest.fixture(autouse=True)
def clean_bpy(request):
    """Fresh scene per test, and reset world.py's own memoised assertion set."""
    _STUB_BPY.reset()
    if "W" in request.fixturenames:
        request.getfixturevalue("W")._seat_cell_footprint_checked.clear()
    yield


def rng(seed=0):
    return random.Random(seed)


@contextlib.contextmanager
def bmesh_available():
    """`_crown_lid` imports bmesh LAZILY, at call time, so the stub has to be
    back in `sys.modules` for the few tests that reach it - and gone again
    afterwards, like every other stub here."""
    prev = sys.modules.get("bmesh")
    sys.modules["bmesh"] = _STUB_BMESH
    try:
        yield
    finally:
        if prev is None:
            sys.modules.pop("bmesh", None)
        else:
            sys.modules["bmesh"] = prev


# =========================================================================== #
#  0. the harness itself
#
#  Every geometric assertion below is read through this stub, so a wrong stub
#  would make the rest of the file meaningless. These pin its semantics
#  against hand-computed values.
# =========================================================================== #

def test_stub_matrix_rotation_matches_hand_computed_values():
    p = Matrix.Rotation(math.radians(90), 4, "Z") @ Vector((1, 0, 0))
    assert (round(p.x, 12), round(p.y, 12), round(p.z, 12)) == (0.0, 1.0, 0.0)
    p = Matrix.Rotation(math.radians(90), 4, "X") @ Vector((0, 1, 0))
    assert (round(p.x, 12), round(p.y, 12), round(p.z, 12)) == (0.0, 0.0, 1.0)
    p = Matrix.Rotation(math.radians(90), 4, "Y") @ Vector((0, 0, 1))
    assert (round(p.x, 12), round(p.y, 12), round(p.z, 12)) == (1.0, 0.0, 0.0)


def test_stub_matrix_at_vector_applies_translation():
    """`mat4 @ vec3` is a POINT transform in Blender. `assets.group_bbox`
    reads every corner through it, so a vector-style transform that dropped
    the translation would put every measured bbox at the origin."""
    p = Matrix.Translation(Vector((1, 2, 3))) @ Vector((0, 0, 0))
    assert tuple(p) == (1.0, 2.0, 3.0)


def test_stub_matrix_inverse_round_trips():
    M = (Matrix.Translation(Vector((0.3, -0.2, 0.1)))
         @ Matrix.Rotation(0.7, 4, "Z") @ Matrix.Diagonal((2.0, 3.0, 4.0)))
    back = M.inverted() @ (M @ Vector((0.11, 0.22, 0.33)))
    assert all(abs(a - b) < 1e-12
               for a, b in zip(tuple(back), (0.11, 0.22, 0.33)))


def test_stub_pivot_conjugation_rotates_about_the_pivot_not_the_origin():
    """The composition `T(p) @ R @ T(-p)` that world.py uses in three places.
    A point AT the pivot must not move; a point off it must swing."""
    pivot = Vector((0.5, 0.25, 0.0))
    M = (Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(90), 4, "Z")
         @ Matrix.Translation(-pivot))
    fixed = M @ pivot
    assert all(abs(a - b) < 1e-12 for a, b in zip(tuple(fixed), tuple(pivot)))
    moved = M @ Vector((0.6, 0.25, 0.0))
    assert abs(moved.x - 0.5) < 1e-12 and abs(moved.y - 0.35) < 1e-12


def test_stub_object_scale_then_transform_apply_bakes_into_the_bbox():
    """`primitive_cube_add(size=1)` + `scale = (w, h, d)` + `transform_apply`
    must measure w x h x d at the given location - the exact idiom every
    builder in world.py uses."""
    _STUB_BPY.ops.mesh.primitive_cube_add(size=1, location=(0.1, 0.2, 0.3))
    o = _STUB_BPY.context.active_object
    o.scale = (0.4, 0.6, 0.02)
    _STUB_BPY.ops.object.transform_apply(location=False, rotation=False,
                                         scale=True)
    lo, hi = o.world_bbox()
    assert np.allclose(hi - lo, (0.4, 0.6, 0.02))
    assert np.allclose((lo + hi) / 2, (0.1, 0.2, 0.3))
    assert np.allclose(tuple(o.scale), (1, 1, 1)), "scale must be reset"


def test_stub_parenting_without_a_parent_inverse_displaces_the_child():
    """The measured 339mm bug `build_pcb` documents: a bare `c.parent = board`
    leaves `matrix_parent_inverse` at identity, so the child inherits the
    parent's translation on top of its own. The stub must reproduce that, or
    the regression test for the fix below proves nothing."""
    _STUB_BPY.ops.mesh.primitive_cube_add(size=1, location=(0.30, 0.15, 0.05))
    board = _STUB_BPY.context.active_object
    _STUB_BPY.ops.mesh.primitive_cube_add(size=1, location=(0.32, 0.16, 0.06))
    child = _STUB_BPY.context.active_object

    child.parent = board
    lo, hi = child.world_bbox()
    assert np.allclose((lo + hi) / 2, (0.62, 0.31, 0.11)), (
        "the stub must show the un-inverted parenting displacement")

    child.matrix_parent_inverse = board.matrix_world.inverted()
    lo, hi = child.world_bbox()
    assert np.allclose((lo + hi) / 2, (0.32, 0.16, 0.06)), (
        "with the parent inverse set the child must not move at all")


def test_stub_leaks_nothing_into_sys_modules():
    """`_load_world` must leave no stubbed module importable. A leaked
    `recog.synth3d.assets` would be handed to every later test in the run."""
    _load_world()
    import recog.synth3d as pkg
    for name in ("bpy", "bmesh", "mathutils"):
        assert name not in sys.modules, f"{name} leaked into sys.modules"
    for attr in ("assets", "materials"):
        assert f"recog.synth3d.{attr}" not in sys.modules
        assert getattr(pkg, attr, None) is None, (
            f"recog.synth3d.{attr} is still reachable as a package attribute")


# =========================================================================== #
#  1. pure decisions - kelvin_to_rgb, _lamp_color, _frame_extent
# =========================================================================== #

def test_frame_extent_landscape_uses_the_longer_render_axis(W):
    """The measured case in `setup_camera`'s docstring: area [0.80, 0.45] at
    1280x720 must frame to exactly 0.80, using the frame edge to edge. If this
    returned the shorter axis instead, every scene would be cropped by 16:9 -
    silently, since a cropped render still looks like a render."""
    assert W._frame_extent(0.80, 0.45, 1280 / 720) == pytest.approx(0.80)


def test_frame_extent_grows_for_an_area_taller_than_the_frame(W):
    """A 4:3 area under a 16:9 camera is height-limited: the extent must come
    from `area_h * aspect`, not from `area_w`."""
    assert W._frame_extent(0.40, 0.40, 16 / 9) == pytest.approx(0.40 * 16 / 9)


def test_frame_extent_portrait_swaps_which_term_binds(W):
    """`ortho_scale` still describes the LONGER axis when the render is
    portrait, so the width term must be divided by the aspect, not multiplied."""
    assert W._frame_extent(0.80, 0.45, 0.5) == pytest.approx(1.60)
    assert W._frame_extent(0.20, 0.90, 0.5) == pytest.approx(0.90)


@pytest.mark.parametrize("w,h,aspect", [
    (0.80, 0.45, 16 / 9), (0.40, 0.40, 16 / 9), (0.80, 0.45, 0.5),
    (0.20, 0.90, 0.5), (1.0, 1.0, 1.0), (0.5, 0.9, 1.3),
])
def test_frame_extent_always_actually_contains_the_area(W, w, h, aspect):
    """The property the formula exists for, checked independently of it: the
    frame the returned extent describes must cover the whole layout area on
    BOTH axes."""
    s = W._frame_extent(w, h, aspect)
    frame_w = s if aspect >= 1.0 else s * aspect
    frame_h = s / aspect if aspect >= 1.0 else s
    assert frame_w >= w - 1e-12 and frame_h >= h - 1e-12


def test_kelvin_to_rgb_is_normalised_and_clamped(W):
    for k in (500, 1000, 2700, 4000, 5500, 6500, 12000, 40000, 90000):
        rgb = W.kelvin_to_rgb(k)
        assert len(rgb) == 3
        assert all(0.0 <= c <= 1.0 for c in rgb), f"{k}K left the unit cube"
        assert max(rgb) == pytest.approx(1.0, abs=1e-9), (
            f"{k}K is not peak-normalised (max {max(rgb)}), so lamp energy "
            f"and lamp colour would no longer be independent knobs")


def test_kelvin_to_rgb_warm_is_red_dominant_and_cool_is_blue_dominant(W):
    warm, cool = W.kelvin_to_rgb(2700), W.kelvin_to_rgb(9000)
    assert warm[0] > warm[2], "2700K must be red-dominant"
    assert cool[2] >= cool[0], "9000K must be blue-dominant"
    assert W.kelvin_to_rgb(3000)[2] < W.kelvin_to_rgb(6000)[2]


def test_kelvin_to_rgb_clamps_outside_its_valid_range(W):
    """Below 1000K and above 40000K must saturate rather than run off the end
    of Helland's fit - the config is free to draw either."""
    assert W.kelvin_to_rgb(1) == W.kelvin_to_rgb(1000)
    assert W.kelvin_to_rgb(1e9) == W.kelvin_to_rgb(40000)


def test_lamp_color_without_a_tint_is_the_blackbody_colour(W):
    assert W._lamp_color(5000, None) == W.kelvin_to_rgb(5000)


def test_lamp_color_applies_the_tint_per_channel(W):
    """The fluorescent green spike the docstring is about: without this the
    illuminant is always Planckian and every real fluorescent frame is
    off-model."""
    base = W.kelvin_to_rgb(4000)
    got = W._lamp_color(4000, (0.9, 1.0, 0.85))
    assert got[0] == pytest.approx(min(1.0, base[0] * 0.9))
    assert got[1] == pytest.approx(min(1.0, base[1] * 1.0))
    assert got[2] == pytest.approx(min(1.0, base[2] * 0.85))
    assert got != base, (
        "a tint that changes nothing is a tint that is not wired up")
    # The property that matters: green is pulled OFF the Planckian locus,
    # relative to the other two channels, which is what no colour temperature
    # alone can reproduce.
    assert got[1] / got[0] > base[1] / base[0]
    assert got[1] / got[2] > base[1] / base[2]


def test_lamp_color_clamps_a_tint_that_overshoots(W):
    assert all(0.0 <= c <= 1.0 for c in W._lamp_color(6500, (4.0, 4.0, 4.0)))


# =========================================================================== #
#  2. lighting - the optional-lamp guard and the off_axis key-lamp guard
# =========================================================================== #

_OFF_AXIS = {
    "kind": "off_axis", "hdri": None,
    "world_kelvin": [5000, 5000], "world_strength": [0.2, 0.2],
    "energy": [40, 40], "size": [0.5, 0.5], "kelvin": [4500, 4500],
    "elevation": [35, 35], "distance": [2.0, 2.0], "azimuth": [90, 90],
}

_FILL = {
    "fill_energy": [10, 10], "fill_size": [0.3, 0.3],
    "fill_kelvin": [3000, 3000], "fill_elevation": [25, 25],
    "fill_distance": [1.5, 1.5], "fill_azimuth_offset": [150, 150],
}


def _cfg(lighting=None, backdrops=None, materials=None):
    return types.SimpleNamespace(lighting=lighting or {},
                                 backdrops=backdrops or {},
                                 materials=materials or {})


def test_an_unconfigured_optional_lamp_returns_none_and_builds_nothing(W):
    """The fill lamp is optional and most rigs omit it. The early return must
    build no lamp at all, not a lamp at some default energy."""
    drawn = {}
    got = W._off_axis_lamp(dict(_OFF_AXIS), rng(), "FillLight", W._FILL_LAMP,
                           drawn, "fill_")
    assert got is None
    assert drawn == {}, "an absent lamp must record nothing in the manifest"
    assert not [o for o in _STUB_BPY.data.objects.values()
                if isinstance(o.data, _Light)]


def test_a_configured_lamp_is_built_and_fully_recorded(W):
    drawn = {}
    az = W._off_axis_lamp(dict(_OFF_AXIS), rng(), "KeyLight", W._KEY_LAMP,
                          drawn, "")
    assert az == pytest.approx(90.0)
    lamps = [o for o in _STUB_BPY.data.objects.values()
             if isinstance(o.data, _Light)]
    assert len(lamps) == 1 and lamps[0].name == "KeyLight"
    assert lamps[0].data.energy == 40 and lamps[0].data.size == 0.5
    for key in ("energy", "size", "kelvin", "azimuth", "elevation",
                "distance", "shadow_dir"):
        assert key in drawn, f"{key} is built but not recorded in the manifest"


def test_the_fill_lamp_azimuth_is_an_offset_from_the_key_lamp(W):
    """Two independent draws from [0, 360) land on top of each other often
    enough to hollow a mixed-illuminant rig out. The fill azimuth must be the
    key azimuth PLUS the configured offset, wrapped."""
    spec = dict(_OFF_AXIS, **_FILL)
    drawn = {}
    key_az = W._off_axis_lamp(spec, rng(), "KeyLight", W._KEY_LAMP, drawn, "")
    W._off_axis_lamp(spec, rng(), "FillLight", W._FILL_LAMP, drawn, "fill_",
                     azimuth_base=key_az)
    assert drawn["fill_azimuth"] == pytest.approx((key_az + 150.0) % 360.0)
    assert abs(drawn["fill_azimuth"] - key_az) > 30.0


def test_the_fill_azimuth_wraps_rather_than_exceeding_360(W):
    spec = dict(_OFF_AXIS, azimuth=[300, 300], **_FILL)
    drawn = {}
    key_az = W._off_axis_lamp(spec, rng(), "KeyLight", W._KEY_LAMP, drawn, "")
    W._off_axis_lamp(spec, rng(), "FillLight", W._FILL_LAMP, drawn, "fill_",
                     azimuth_base=key_az)
    assert drawn["fill_azimuth"] == pytest.approx(90.0)


def test_shadow_direction_is_recorded_opposite_the_lamp(W):
    """The property the whole off_axis rig exists to vary. A lamp in the +X
    direction must cast its shadows toward -X."""
    drawn = {}
    W._off_axis_lamp(dict(_OFF_AXIS, azimuth=[0, 0]), rng(), "KeyLight",
                     W._KEY_LAMP, drawn, "")
    sx, sy = drawn["shadow_dir"]
    assert sx == pytest.approx(-1.0) and sy == pytest.approx(0.0, abs=1e-12)


def test_an_off_axis_preset_with_no_key_lamp_raises_rather_than_rendering_dark(W):
    """The guard `setup_lighting` carries: a preset that declares kind
    'off_axis' but configures no `energy` would build NO key lamp and render
    lit only by the world background, silently ignoring every other key the
    preset sets."""
    broken = {k: v for k, v in _OFF_AXIS.items() if k != "energy"}
    cfg = _cfg(lighting={"rig": broken})
    with pytest.raises(ValueError, match="off_axis"):
        W.setup_lighting("rig", rng(), (0, 0, 1.0), cfg)


def test_a_complete_off_axis_preset_builds_key_and_fill(W):
    cfg = _cfg(lighting={"rig": dict(_OFF_AXIS, **_FILL)})
    drawn = W.setup_lighting("rig", rng(), (0, 0, 1.0), cfg)
    names = sorted(o.name for o in _STUB_BPY.data.objects.values()
                   if isinstance(o.data, _Light))
    assert names == ["FillLight", "KeyLight"]
    assert drawn["kind"] == "off_axis" and drawn["lighting"] == "rig"


def test_a_missing_hdri_falls_back_to_grey_sky_and_records_the_fallback(W):
    """The fallback exists so a missing file does not abort a 1000-image run.
    What matters is that the manifest says `hdri: None` rather than naming a
    file that was never loaded - a manifest that lies about its own inputs is
    the exact failure this project has already had."""
    spec = {"kind": "hdri", "hdri": str(ROOT / "no_such_file_1234.exr"),
            "hdri_rotation": [0, 0], "hdri_strength": [1, 1],
            "world_kelvin": [5000, 5000], "world_strength": [1, 1]}
    drawn = W.setup_lighting("rig", rng(), (0, 0, 1.0),
                             _cfg(lighting={"rig": spec}))
    assert drawn["hdri"] is None
    assert "hdri_strength" not in drawn, (
        "a strength recorded for an HDRI that was never loaded describes a "
        "render that did not happen")
    bg = _STUB_BPY.context.scene.world.node_tree.nodes.get("Background")
    assert bg.inputs["Color"].default_value == (0.5, 0.5, 0.5, 1.0)


def test_a_softbox_rig_hangs_the_lamp_between_the_camera_and_the_parts(W):
    """`cam_loc[2] - 0.06`. Above the camera and it lights the back of it;
    at the parts and it is in shot."""
    spec = {"kind": "camera_softbox", "hdri": None,
            "world_kelvin": [5000, 5000], "world_strength": [1, 1],
            "energy": [100, 100], "size": [1.0, 1.0], "kelvin": [5000, 5000],
            "offset": [0.2, 0.2]}
    W.setup_lighting("rig", rng(), (0.0, 0.0, 1.2), _cfg(lighting={"rig": spec}))
    lamp = next(o for o in _STUB_BPY.data.objects.values()
                if isinstance(o.data, _Light))
    assert lamp.location.z == pytest.approx(1.2 - 0.06)
    assert math.hypot(lamp.location.x, lamp.location.y) == pytest.approx(0.2)


# =========================================================================== #
#  3. camera
# =========================================================================== #

_CAM = types.SimpleNamespace(margin_range=[1.0, 1.0], shift_range=[0.0, 0.0],
                             ortho=True, focal=50.0, height=1.2)
_LAYOUT = types.SimpleNamespace(area=[0.80, 0.45])


def test_ortho_scale_is_the_framed_extent_times_margin_and_zoom(W):
    cam, meta = W.setup_camera(_CAM, _LAYOUT, (1280, 720), rng(), zoom=1.0)
    assert cam.data.type == "ORTHO"
    assert cam.data.ortho_scale == pytest.approx(0.80)
    assert meta["ortho_scale"] == pytest.approx(0.80)


@pytest.mark.parametrize("zoom", [0.8, 1.0, 1.25])
def test_larger_zoom_means_a_wider_view_and_smaller_parts(W, zoom):
    """The sense of `zoom` is load-bearing and inverted from the everyday
    photographic meaning: it multiplies the framed extent, so larger zoom is a
    WIDER frame. Inverting it would silently move the whole dataset's object
    scale distribution - the one property this knob was added to widen - and
    it is COUPLED to model.anchor_scales."""
    cam, meta = W.setup_camera(_CAM, _LAYOUT, (1280, 720), rng(), zoom=zoom)
    assert cam.data.ortho_scale == pytest.approx(0.80 * zoom)
    assert meta["zoom"] == zoom


def test_a_zero_or_negative_zoom_raises_rather_than_collapsing_the_frame(W):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="zoom"):
            W.setup_camera(_CAM, _LAYOUT, (1280, 720), rng(), zoom=bad)


def test_the_camera_sits_above_the_scene_top_and_looks_straight_down(W):
    cam, _ = W.setup_camera(_CAM, _LAYOUT, (1280, 720), rng(), top_z=0.05)
    assert cam.location.z == pytest.approx(0.05 + 1.2)
    assert tuple(cam.rotation_euler) == (0.0, 0.0, 0.0), (
        "a zero-rotation camera already looks down -Z; any rotation here "
        "would tilt the whole dataset")
    assert _STUB_BPY.context.scene.camera is cam


def test_a_perspective_camera_is_lifted_to_frame_the_same_extent(W):
    """The ortho path sets `ortho_scale`; the perspective path has to move the
    camera instead. Both must frame the same area."""
    persp = types.SimpleNamespace(**{**_CAM.__dict__, "ortho": False})
    cam, meta = W.setup_camera(persp, _LAYOUT, (1280, 720), rng(), top_z=0.0)
    assert cam.data.type == "PERSP" and cam.data.lens == 50.0
    fov = 2 * math.atan(cam.data.sensor_width / (2 * 50.0))
    assert cam.location.z == pytest.approx((0.80 / 2) / math.tan(fov / 2))
    assert meta["ortho"] is False


def test_a_perspective_render_records_no_ortho_scale_to_calibrate_from(W):
    """A perspective camera has NO single scalar mm_per_px - the scale varies
    with depth - so the manifest must carry no `ortho_scale` at all rather
    than a number the planner would size real placements against.

    This was a live defect. The line read
    `getattr(cam.data, "ortho_scale", None)`, and
    `bpy.types.Camera.ortho_scale` is an unconditional property of the camera
    datablock (Blender only hides it in the UI when the type is not ORTHO), so
    the guard never fired and a perspective render recorded Blender's
    untouched 6.0 default as though it were a measurement."""
    persp = types.SimpleNamespace(**{**_CAM.__dict__, "ortho": False})
    _, meta = W.setup_camera(persp, _LAYOUT, (1280, 720), rng())
    assert _STUB_BPY.context.scene.camera.data.ortho_scale == 6.0, (
        "the stub must keep Blender's unconditional ortho_scale default, or "
        "this test cannot reproduce the defect at all")
    assert meta["ortho_scale"] is None


def test_calibration_refuses_the_sidecar_a_perspective_render_writes(W):
    """The consumer half, stated against the real module: with the fix,
    `frame_mm_per_px` raises on a perspective frame's metadata instead of
    fabricating a ground sample distance from a default nobody set."""
    from recog.calibration import frame_mm_per_px
    persp = types.SimpleNamespace(**{**_CAM.__dict__, "ortho": False})
    _, persp_meta = W.setup_camera(persp, _LAYOUT, (1280, 720), rng())
    with pytest.raises(ValueError, match="ortho_scale"):
        frame_mm_per_px({"camera": persp_meta, "width": 1280})

    _STUB_BPY.reset()
    _, ortho_meta = W.setup_camera(_CAM, _LAYOUT, (1280, 720), rng())
    assert frame_mm_per_px({"camera": ortho_meta, "width": 1280}) == \
        pytest.approx(0.80 * 1000 / 1280)


def test_the_camera_shift_reaches_both_the_object_and_the_manifest(W):
    shifted = types.SimpleNamespace(**{**_CAM.__dict__,
                                       "shift_range": [0.05, 0.05]})
    cam, meta = W.setup_camera(shifted, _LAYOUT, (1280, 720), rng())
    assert (meta["shift_x"], meta["shift_y"]) == (0.05, 0.05)
    assert (cam.location.x, cam.location.y) == pytest.approx((0.05, 0.05)), (
        "annotate's pixel projection subtracts the recorded shift, so a shift "
        "in the manifest that never reached the camera moves every box")


# =========================================================================== #
#  4. the seating-height ladder
#
#  The ladder itself now lives in `bay.SEATING_LADDER`, and `tests/test_bay.py`
#  pins the table and its ordering invariant directly. What is left for THIS
#  file is the half a bpy-free test cannot reach: that the geometry world.py
#  actually builds lands on those rungs. `build_bay_proxy` carries the
#  `placement_area` label; anything placed in the bay must rest STRICTLY above
#  it or the label keeps claiming the floor is free while an object visibly
#  sits on it - and nothing downstream can detect that from the mask alone.
#
#  The expected offsets below are deliberately written out as literals rather
#  than read from `bay.SEATING_LADDER`. Reading them from the table would make
#  every test here agree with the table by construction; as literals, a change
#  to the table has to be made in two files that do not import each other
#  before any of this goes quiet.
# =========================================================================== #

_PROXY_LIFT = 0.0009        # bay.SEATING_LADDER's "bay_proxy" rung
_PCB_LIFT = 0.0008          # ... "pcb": build_pcb's board, about its centre
_SEATED_CELL_LIFT = 0.0012  # ... "seated_cell": the cell's BASE
_OBSTRUCTION_LIFT = {"adhesive": 0.0012, "foam": 0.0022,
                     "tape": 0.0011, "label": 0.0011}


def test_the_bay_proxy_plane_sits_between_the_board_s_base_and_its_top(W):
    """The 0.1mm the docstrings turn on. The board is 1.6mm thick and built
    centred on `floor_z + 0.0008`, so it RESTS on the cavity floor; the proxy
    plane is 0.9mm up. That puts the proxy above the floor it labels (so it is
    never coplanar with the tray) and below the board's top face (so along the
    edge the two rectangles share, the board occludes the proxy rather than
    the other way round)."""
    floor = 0.010
    board, _ = W.build_pcb((0, 0, 0.1, 0.06), floor, rng(),
                           module_placement=(0.02, 0.03, 0.02, 0.02))
    proxy, _ = W.build_bay_proxy((0.07, 0.03, 0.02, 0.02), floor, rng())
    b_lo, b_hi = board.world_bbox()
    p_lo, p_hi = proxy.world_bbox()
    assert b_lo[2] == pytest.approx(floor, abs=1e-9), (
        "the board must rest ON the cavity floor, not float above or sink "
        "below it")
    assert p_lo[2] == pytest.approx(p_hi[2]), "the proxy is a flat plane"
    assert floor < p_lo[2] < b_hi[2], (
        f"the proxy plane at {p_lo[2]:.5f} must sit strictly above the tray "
        f"floor {floor} and strictly below the board's top {b_hi[2]:.5f}")
    assert p_lo[2] - floor == pytest.approx(_PROXY_LIFT, abs=1e-9)


@pytest.mark.parametrize("kind", ["adhesive", "foam", "tape", "label"])
def test_every_obstruction_kind_rests_strictly_above_the_proxy(W, kind):
    """An obstruction at or below `floor_z + 0.0009` loses the z-fight and
    placement_area keeps reporting that floor as free while glue visibly sits
    on it. Silently - the mask cannot show it.

    Measured off the object world.py built, not off the table it read: the
    point of the move to `bay.SEATING_LADDER` is that world.py now ASKS for
    each height, and a builder that asked and then ignored the answer would
    still satisfy any check made against the table alone."""
    floor = 0.010
    pose = types.SimpleNamespace(kind=kind, x=0.0, y=0.0, w=0.01, h=0.008,
                                 rot_deg=0.0)
    built = W.build_obstructions([pose], floor, rng())
    assert len(built) == 1
    origin_z = built[0][0].location[2]
    assert origin_z - floor == pytest.approx(_OBSTRUCTION_LIFT[kind],
                                             abs=1e-12)
    assert origin_z > floor + _PROXY_LIFT, (
        f"{kind} is seated at +{origin_z - floor:.4f}m, not strictly above "
        f"the bay proxy's +{_PROXY_LIFT}m")


def test_a_seated_cell_clears_the_proxy_too(W):
    """The entire mechanism `seat_cells` exists for: a cell at or below the
    proxy's own rung would not occlude it, and placement_area would claim the
    floor it sits on is free. Driven through the builder rather than compared
    against a constant, because the constant no longer lives here."""
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    floor = 0.010
    made = W.seat_cells(lib, "AnkerX", [(0.02, 0.01, 0.0)], floor, rng(),
                        _MAT_CFG)
    lo, _ = made[0].world_bbox()
    assert lo[2] > floor + _PROXY_LIFT
    assert lo[2] - floor == pytest.approx(_SEATED_CELL_LIFT, abs=1e-12)


def test_the_seating_ladder_holds_in_the_geometry_world_py_actually_builds(W):
    """The whole ladder, as one statement, measured off six real objects.

    `bay.assert_seating_ladder_ordered` proves the TABLE is ordered;
    `tests/test_bay.py` proves it fails when perturbed. Neither says anything
    about whether world.py still reads it - which is exactly the failure mode
    this move introduces, and exactly what the old literals were. So this
    builds one of everything at a common floor and reads the ordering back out
    of the geometry."""
    floor = 0.010
    _STUB_BPY.reset()
    board, _ = W.build_pcb((0, 0, 0.1, 0.06), floor, rng(),
                           module_placement=(0.02, 0.03, 0.02, 0.02))
    proxy, _ = W.build_bay_proxy((0.07, 0.03, 0.02, 0.02), floor, rng())
    obs = {m["kind"]: o for o, m in W.build_obstructions(
        [types.SimpleNamespace(kind=k, x=0.0, y=0.0, w=0.01, h=0.008,
                               rot_deg=0.0)
         for k in ("tape", "label", "adhesive", "foam")], floor, rng())}
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    cell = W.seat_cells(lib, "AnkerX", [(0.02, 0.01, 0.0)], floor, rng(),
                        _MAT_CFG)[0]

    ladder = [
        ("pcb", board.location[2]),
        ("bay_proxy", proxy.location[2]),
        ("tape", obs["tape"].location[2]),
        ("label", obs["label"].location[2]),
        ("adhesive", obs["adhesive"].location[2]),
        ("seated_cell", cell.world_bbox()[0][2]),
        ("foam", obs["foam"].location[2]),
    ]
    zs = [z for _, z in ladder]
    assert zs == sorted(zs), (
        f"the built geometry is out of ladder order: "
        f"{[(n, z - floor) for n, z in ladder]}")
    assert all(0 < z - floor < 0.003 for z in zs), (
        "these are clearances, not stand-offs; a millimetre-scale gap would "
        "be visible from overhead against an 18mm cell")
    proxy_z = proxy.location[2]
    for name, z in ladder[2:]:
        assert z > proxy_z, (
            f"{name} was built at {z}, not strictly above the proxy plane at "
            f"{proxy_z} that carries the placement_area label")


def test_an_obstruction_kind_world_py_cannot_shape_stops_the_build(W):
    """`build_obstructions`' shape dispatch ends in a bare `else:  # label`,
    so a fifth kind added to `bay.sample_obstructions` without a matching
    branch here would render as a printed label - silently and plausibly, the
    same shape as the renamed catalog key that once stopped a builder
    building. `bay.obstruction_z_scale` is called BEFORE that dispatch and
    raises, so the build stops instead.

    This is the one behavioural difference the ladder move makes, and it is on
    input `sample_obstructions` cannot currently produce (pinned by
    `test_the_obstruction_kinds_match_bay_sample_obstructions`)."""
    pose = types.SimpleNamespace(kind="gasket", x=0.0, y=0.0, w=0.01, h=0.008,
                                 rot_deg=0.0)
    with pytest.raises(ValueError, match="not an obstruction kind"):
        W.build_obstructions([pose], 0.010, rng())


@pytest.mark.parametrize("func", ["build_pcb", "build_bay_proxy",
                                  "build_obstructions", "seat_cells"])
def test_every_bay_builder_asks_bay_for_its_height(func):
    """Source-level, because this is the property the move exists to create:
    world.py BUILDS, bay.py DECIDES. Each of these four used to carry its own
    float literal. A builder that quietly goes back to one would still pass
    every geometric test above - they would simply be testing the literal."""
    fn = _function(WORLD_PY, func)
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "seat_z"
             and getattr(n.func.value, "id", None) == "bay"]
    assert calls, (
        f"world.{func} no longer calls bay.seat_z; its seating height has "
        f"drifted back onto the untestable side of the bpy line")


def test_the_jig_backdrop_sits_below_the_plate_it_supports(W):
    """`build_jig` returns `backdrop_z` for the caller to put the ground plane
    at. Above the plate underside and the plate is buried; AT it and the two
    are coplanar, which renders the ENTIRE PLATE BLACK (measured, see
    JIG_LIFT's note) with an identical index pass, so no mask-level gate can
    catch it."""
    pockets = [types.SimpleNamespace(x=0.0, y=0.0, w=0.05, h=0.03, depth=0.008)]
    _, drawn = W.build_jig(pockets, rng())
    assert drawn["backdrop_z"] < drawn["underside_z"], "backdrop is not below"
    assert drawn["underside_z"] - drawn["backdrop_z"] == pytest.approx(
        W.JIG_BACKDROP_GAP)
    assert W.JIG_LIFT > 0.0, (
        "a zero lift makes the plate top coplanar with a z=0 backdrop, which "
        "renders the plate black rather than z-fighting visibly")


# =========================================================================== #
#  5. build_jig
# =========================================================================== #

def _pocket(x, y, w, h, depth):
    return types.SimpleNamespace(x=x, y=y, w=w, h=h, depth=depth)


@pytest.mark.parametrize("depth", [0.006, 0.009, 0.012, 0.017, 0.030])
def test_the_plate_is_never_thinner_than_its_deepest_pocket(W, depth):
    """`jig_depth` (6-12mm) and the sampled thickness range (10-18mm) overlap,
    so a plate punched clean through is not hypothetical - and a pocket that
    became a HOLE shows the backdrop through the plate rather than a floor."""
    for seed in range(25):
        _STUB_BPY.reset()
        _, drawn = W.build_jig([_pocket(0, 0, 0.05, 0.03, depth)], rng(seed))
        assert drawn["thickness"] >= depth + 0.004 - 1e-12, (
            f"thickness {drawn['thickness']} does not clear a {depth}m pocket")
        assert drawn["underside_z"] < W.JIG_LIFT - depth, (
            "the plate underside must sit below the deepest pocket floor")


def test_the_plate_follows_the_pockets_not_the_layout_area(W):
    """Sizing the plate to `layout_cfg.area` gave a plate that filled the frame
    and hid the backdrop in every jig scene, even one holding a single small
    part. It must bound the pockets it was GIVEN."""
    pockets = [_pocket(-0.10, 0.0, 0.04, 0.02, 0.008),
               _pocket(0.10, 0.0, 0.04, 0.02, 0.008)]
    _, drawn = W.build_jig(pockets, rng())
    span_x = (0.10 + 0.02) - (-0.10 - 0.02)
    assert drawn["w"] == pytest.approx(span_x + 2 * drawn["margin"])
    assert drawn["h"] == pytest.approx(0.02 + 2 * drawn["margin"])
    assert drawn["cx"] == pytest.approx(0.0) and drawn["cy"] == pytest.approx(0.0)


def test_the_built_plate_measures_what_the_manifest_says(W):
    pockets = [_pocket(0.02, -0.01, 0.05, 0.03, 0.009)]
    plate, drawn = W.build_jig(pockets, rng(3))
    lo, hi = plate.world_bbox()
    assert (hi - lo)[:2] == pytest.approx((drawn["w"], drawn["h"]))
    assert hi[2] == pytest.approx(W.JIG_LIFT)
    assert lo[2] == pytest.approx(drawn["underside_z"])


def test_the_plate_is_unlabelled_furniture(W):
    """pass_index 0 is what makes the plate occlude correctly while producing
    no annotation. Any NON-zero id with no id_meta entry is dropped silently
    and without an audit trail."""
    plate, _ = W.build_jig([_pocket(0, 0, 0.05, 0.03, 0.008)], rng())
    assert plate.pass_index == 0


def test_every_pocket_becomes_its_own_applied_boolean_and_the_cutters_go(W):
    """A cutter left in the scene is an unlabelled solid sitting in the shot;
    a boolean applied to the wrong active object cuts the wrong mesh."""
    pockets = [_pocket(-0.03, 0, 0.02, 0.02, 0.008),
               _pocket(0.03, 0, 0.02, 0.02, 0.010),
               _pocket(0.0, 0.04, 0.02, 0.02, 0.006)]
    plate, drawn = W.build_jig(pockets, rng())
    assert drawn["n_pockets"] == 3
    assert len(plate.applied_modifiers) == 3
    assert all(op == "DIFFERENCE" for _, op, _ in plate.applied_modifiers)
    assert not [o for o in _STUB_BPY.data.objects.values()
                if o.name.startswith("_pocket")], "a cutter survived the build"
    assert len(plate.modifiers) == 0


# =========================================================================== #
#  6. build_pcb
# =========================================================================== #

def test_an_anchored_board_lands_on_the_module_bay_not_in_the_middle(W):
    """`module_placement` is the real strip the hardware reserves. Ignoring it
    centres the board, which is box-safe and therefore invisible to every
    downstream check, but it is not what the detector should be learning."""
    board, drawn = W.build_pcb((0.0, 0.0, 0.10, 0.06), 0.005, rng(),
                               module_placement=(0.08, 0.012, 0.02, 0.018))
    lo, hi = board.world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.08)
    assert (lo + hi)[1] / 2 == pytest.approx(0.012)
    assert (hi - lo)[:2] == pytest.approx((0.02, 0.018))
    assert drawn["anchored"] is True
    assert (drawn["w"], drawn["h"]) == (0.02, 0.018)


def test_the_unanchored_fallback_centres_the_board_and_says_so(W):
    """The fallback is legitimate - scene.py uses it for an asset with no
    catalog measurement - but `anchored` must record which path ran, or the
    manifest cannot distinguish a real bay from a guess."""
    board, drawn = W.build_pcb((0.0, 0.0, 0.10, 0.06), 0.005, rng(7))
    lo, hi = board.world_bbox()
    assert drawn["anchored"] is False
    assert abs((lo + hi)[0] / 2 - 0.05) < 0.005
    assert abs((lo + hi)[1] / 2 - 0.03) < 0.011


def test_the_board_is_seated_on_the_cavity_floor_it_was_given(W):
    """Until task 3 this took the assembly's `hi.z` - the outer surface of a
    closed lid - so the board was drawn on the OUTSIDE of a shut box, which is
    also where the placement_area label went. The floor argument is the whole
    fix; a board that ignored it would reproduce the defect exactly."""
    for floor in (0.0, 0.004, 0.011):
        _STUB_BPY.reset()
        board, _ = W.build_pcb((0, 0, 0.1, 0.06), floor, rng(),
                               module_placement=(0.05, 0.03, 0.02, 0.02))
        lo, hi = board.world_bbox()
        assert lo[2] == pytest.approx(floor, abs=1e-9), (
            "the board must rest ON the floor it was handed - it tracks the "
            "argument, so a caller passing the assembly's top would put it "
            "on the outside of a shut lid again")
        assert (hi - lo)[2] == pytest.approx(0.0016, abs=1e-9)


def test_every_child_component_keeps_the_world_position_it_was_built_at(W):
    """The measured 339mm defect: a bare `c.parent = board` leaves
    matrix_parent_inverse at identity and the child jumps by the board's own
    translation. Components carry pass_index 0, so strays would occlude
    LABELLED parts and shrink their boxes with no audit trail at all."""
    board, drawn = W.build_pcb((0, 0, 0.10, 0.06), 0.005, rng(1),
                               module_placement=(0.30, 0.15, 0.03, 0.02))
    kids = [o for o in _STUB_BPY.data.objects.values() if o.parent is board]
    assert kids, "the board has no children at all"
    board_lo, board_hi = board.world_bbox()
    for k in kids:
        lo, hi = k.world_bbox()
        cx, cy = (lo + hi)[0] / 2, (lo + hi)[1] / 2
        assert board_lo[0] - 0.02 <= cx <= board_hi[0] + 0.02, (
            f"{k.name} is at x={cx:.4f}, off a board spanning "
            f"{board_lo[0]:.4f}..{board_hi[0]:.4f} - the parent-inverse is "
            f"not pinning it")
        assert board_lo[1] - 0.02 <= cy <= board_hi[1] + 0.02
        assert k.pass_index == 0


def test_the_board_turns_with_its_cartridge_about_its_own_centre(W):
    """`rot_deg` is the cartridge's own placement rotation. Rotating about the
    board's centre keeps the anchored position from `module_placement` and
    changes only the orientation - rotating about the world origin instead
    would fling the board across the scene."""
    place = (0.20, 0.10, 0.030, 0.012)
    board, _ = W.build_pcb((0, 0, 0.3, 0.2), 0.005, rng(2),
                           module_placement=place, rot_deg=90.0)
    lo, hi = board.world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.20, abs=1e-9)
    assert (lo + hi)[1] / 2 == pytest.approx(0.10, abs=1e-9)
    assert (hi - lo)[0] == pytest.approx(0.012, abs=1e-9), (
        "a 90-degree turn must swap the board's measured width and height")
    assert (hi - lo)[1] == pytest.approx(0.030, abs=1e-9)


def test_the_children_turn_rigidly_with_the_board(W):
    """Ports and the inductor are parented, so the board's rotation must carry
    them. A port left axis-aligned on a turned board sits crosswise off the
    edge - and at pass_index 0 it would occlude labelled parts silently."""
    place = (0.20, 0.10, 0.030, 0.012)
    straight, _ = W.build_pcb((0, 0, 0.3, 0.2), 0.005, rng(5),
                              module_placement=place, rot_deg=0.0)
    offsets_0 = _child_offsets(straight)
    _STUB_BPY.reset()
    turned, _ = W.build_pcb((0, 0, 0.3, 0.2), 0.005, rng(5),
                            module_placement=place, rot_deg=90.0)
    offsets_90 = _child_offsets(turned)
    assert len(offsets_0) == len(offsets_90) and offsets_0
    for (dx0, dy0), (dx9, dy9) in zip(offsets_0, offsets_90):
        assert dx9 == pytest.approx(-dy0, abs=1e-9)
        assert dy9 == pytest.approx(dx0, abs=1e-9)


def _child_offsets(board):
    out = []
    b_lo, b_hi = board.world_bbox()
    bcx, bcy = (b_lo + b_hi)[0] / 2, (b_lo + b_hi)[1] / 2
    for o in _STUB_BPY.data.objects.values():
        if o.parent is board:
            lo, hi = o.world_bbox()
            out.append(((lo + hi)[0] / 2 - bcx, (lo + hi)[1] / 2 - bcy))
    return out


def test_the_board_carries_a_placeholder_pass_index_for_scene_to_override(W):
    board, _ = W.build_pcb((0, 0, 0.1, 0.06), 0.005, rng())
    assert board.pass_index == 0


# =========================================================================== #
#  7. build_bay_proxy and build_obstructions
# =========================================================================== #

def test_the_proxy_is_built_at_the_placement_rectangle_it_was_given(W):
    proxy, drawn = W.build_bay_proxy((0.03, -0.02, 0.040, 0.025), 0.006, rng())
    lo, hi = proxy.world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.03)
    assert (lo + hi)[1] / 2 == pytest.approx(-0.02)
    assert (hi - lo)[:2] == pytest.approx((0.040, 0.025))
    assert (drawn["w"], drawn["h"]) == (0.040, 0.025)


def test_the_proxy_turns_with_the_cartridge_about_its_own_centre(W):
    proxy, drawn = W.build_bay_proxy((0.03, -0.02, 0.040, 0.020), 0.006,
                                     rng(), rot_deg=90.0)
    lo, hi = proxy.world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.03, abs=1e-9)
    assert (lo + hi)[1] / 2 == pytest.approx(-0.02, abs=1e-9)
    assert (hi - lo)[0] == pytest.approx(0.020, abs=1e-9)
    assert drawn["rot_deg"] == 90.0


def test_the_proxy_is_a_shaded_object_not_an_index_only_helper(W):
    """It has to LOOK like the inside of a cartridge: the segmenter is asked
    to recognise placement area from appearance, not from a hidden id."""
    proxy, drawn = W.build_bay_proxy((0.0, 0.0, 0.04, 0.02), 0.006, rng())
    assert len(proxy.data.materials) == 1
    bsdf = proxy.data.materials[0].node_tree.nodes.get("Principled BSDF")
    assert bsdf.inputs["Base Color"].default_value == \
        tuple(drawn["color"]) + (1.0,)
    assert bsdf.inputs["Roughness"].default_value == drawn["roughness"]


def test_obstruction_geometry_matches_the_pose_it_was_given(W):
    poses = [types.SimpleNamespace(kind="foam", x=0.01, y=-0.005,
                                   w=0.008, h=0.006, rot_deg=0.0)]
    (obj, meta), = W.build_obstructions(poses, 0.006, rng())
    lo, hi = obj.world_bbox()
    assert (hi - lo)[:2] == pytest.approx((0.008, 0.006))
    assert meta == {"kind": "foam", "w": 0.008, "h": 0.006, "rot_deg": 0.0}


def test_the_obstruction_kind_vocabulary_still_matches_bays(W):
    """`build_obstructions` dispatches on `kind` with a bare `else` that turns
    ANY unrecognised kind into a printed label. A new kind added to
    `bay.sample_obstructions` would therefore render as a label, silently, and
    the manifest would still record the new name. This pins the two together
    so that drift fails here instead of in a dataset."""
    src = (ROOT / "recog" / "synth3d" / "bay.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "sample_obstructions")
    sampled = {n.value for n in ast.walk(fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value in {"adhesive", "foam", "tape", "label"}}
    world_src = WORLD_PY.read_text(encoding="utf-8")
    wfn = next(n for n in ast.walk(ast.parse(world_src))
               if isinstance(n, ast.FunctionDef)
               and n.name == "build_obstructions")
    handled = {n.comparators[0].value for n in ast.walk(wfn)
               if isinstance(n, ast.Compare)
               and isinstance(n.comparators[0], ast.Constant)}
    assert sampled == {"adhesive", "foam", "tape", "label"}
    assert sampled - handled - {"label"} == set(), (
        f"bay.sample_obstructions draws {sorted(sampled)} but "
        f"build_obstructions only branches on {sorted(handled)}; the rest "
        f"fall into the `else` and render as printed labels")


def test_a_tape_obstruction_is_flat_and_a_foam_pad_is_not(W):
    """Tape is a plane; foam is a pad with real thickness. Building foam flat
    would leave nothing for the light to catch and the segmenter would see a
    printed rectangle where the reference photo has a raised pad."""
    flat = types.SimpleNamespace(kind="tape", x=0, y=0, w=0.01, h=0.01,
                                 rot_deg=0.0)
    pad = types.SimpleNamespace(kind="foam", x=0, y=0, w=0.01, h=0.01,
                                rot_deg=0.0)
    (t, _), (f, _) = W.build_obstructions([flat, pad], 0.006, rng())
    t_lo, t_hi = t.world_bbox()
    f_lo, f_hi = f.world_bbox()
    assert (t_hi - t_lo)[2] == pytest.approx(0.0)
    assert 0.002 <= (f_hi - f_lo)[2] <= 0.005


def test_a_translucent_obstruction_gets_an_alpha_and_a_blend_mode(W):
    """Adhesive is the one translucent kind. Setting Alpha without switching
    `blend_method` renders it fully opaque - the value is in the material and
    has no effect, which is the same shape as the discarded-roughness bug."""
    pose = types.SimpleNamespace(kind="adhesive", x=0, y=0, w=0.006, h=0.005,
                                 rot_deg=0.0)
    (obj, _), = W.build_obstructions([pose], 0.006, rng())
    mat = obj.data.materials[0]
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    assert bsdf.inputs["Alpha"].default_value < 1.0
    assert mat.blend_method == "BLEND", (
        "an alpha below 1 with an OPAQUE blend method renders opaque")


def test_an_opaque_obstruction_is_left_opaque(W):
    pose = types.SimpleNamespace(kind="label", x=0, y=0, w=0.006, h=0.005,
                                 rot_deg=0.0)
    (obj, _), = W.build_obstructions([pose], 0.006, rng())
    assert obj.data.materials[0].blend_method == "OPAQUE"
    assert obj.data.materials[0].node_tree.nodes.get(
        "Principled BSDF").inputs["Alpha"].default_value == 1.0


def test_no_obstructions_builds_nothing(W):
    assert W.build_obstructions([], 0.006, rng()) == []


# =========================================================================== #
#  8. seat_cells - the "renamed key silently stops building geometry" shape
# =========================================================================== #

class _FakeLibrary:
    def __init__(self, templates):
        self._templates = templates


def _cell_template(diam=0.0183, length=0.065):
    """A template shaped like an already-lay_flat'd 18650: long axis in Y."""
    _STUB_BPY.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    o = _STUB_BPY.context.active_object
    o.name = "Cell_template"
    o.scale = (diam, length, diam)
    _STUB_BPY.ops.object.transform_apply(location=False, rotation=False,
                                         scale=True)
    return o


# `wear` is held at 0 so `materials.build` takes its no-wear path: the wear
# mix graph is `tests/test_synth3d_materials.py`'s subject, not this file's.
_MAT_CFG = types.SimpleNamespace(
    materials={"cell_steel": {"color": [[0.5] * 3, [0.5] * 3],
                              "roughness": [0.4, 0.4], "metallic": [1.0, 1.0],
                              "coat": [0.0, 0.0], "wear": [0.0, 0.0],
                              "luma_ref": 0.4}},
    role_materials={"cell": ["cell_steel"]})


def test_no_seats_builds_no_cells(W):
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    assert W.seat_cells(lib, "AnkerX", [], 0.006, rng(), _MAT_CFG) == []


def test_a_missing_asset_template_returns_empty_rather_than_raising(W):
    """`(library._templates.get(asset) or {}).get("cell")` is a guarded
    lookup, and a guarded `.get` on a renamed key is precisely how this
    project once had a builder quietly stop building geometry.

    This pins the CURRENT behaviour rather than asserting it is right: a
    missing template silently seats nothing. It is survivable here only
    because scene.py records `n: len(item.seated_objects)` from the same
    return value, so the manifest agrees with the empty bay - see the report
    for why that agreement is what makes this a design choice rather than a
    repeat of the defect."""
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    seats = [(0.0, 0.0, 0.0)]
    assert W.seat_cells(lib, "NotAnAsset", seats, 0.006, rng(), _MAT_CFG) == []
    assert W.seat_cells(_FakeLibrary({"AnkerX": {}}), "AnkerX", seats, 0.006,
                        rng(), _MAT_CFG) == []
    assert W.seat_cells(_FakeLibrary({"AnkerX": {"cell": []}}), "AnkerX",
                        seats, 0.006, rng(), _MAT_CFG) == []


def test_seated_cells_rest_on_the_lift_above_the_floor(W):
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    floor = 0.006
    made = W.seat_cells(lib, "AnkerX", [(0.02, 0.01, 0.0), (0.04, 0.01, 0.0)],
                        floor, rng(), _MAT_CFG)
    assert len(made) == 2
    for obj in made:
        lo, _ = obj.world_bbox()
        assert lo[2] == pytest.approx(floor + _SEATED_CELL_LIFT, abs=1e-9)


def test_a_seated_cell_lands_on_the_seat_it_was_given(W):
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    made = W.seat_cells(lib, "AnkerX", [(0.021, -0.013, 0.0)], 0.006, rng(),
                        _MAT_CFG)
    lo, hi = made[0].world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.021, abs=1e-9)
    assert (lo + hi)[1] / 2 == pytest.approx(-0.013, abs=1e-9)


def test_a_seated_cell_spins_about_its_own_seat_not_the_origin(W):
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    made = W.seat_cells(lib, "AnkerX", [(0.021, -0.013, 90.0)], 0.006, rng(),
                        _MAT_CFG)
    lo, hi = made[0].world_bbox()
    assert (lo + hi)[0] / 2 == pytest.approx(0.021, abs=1e-9)
    assert (lo + hi)[1] / 2 == pytest.approx(-0.013, abs=1e-9)
    assert (hi - lo)[0] == pytest.approx(0.065, abs=1e-6), (
        "a 90-degree spin must put the cell's 65mm length along X")
    assert (hi - lo)[2] == pytest.approx(0.0183, abs=1e-6), (
        "a Z-axis spin must leave the resting height alone")


def test_a_seated_cell_gets_its_own_material(W):
    """A template object carries no material - it is never rendered directly,
    only cloned - so seat_cells must draw and apply one explicitly or every
    seated cell renders with Blender's default grey."""
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template()]}})
    made = W.seat_cells(lib, "AnkerX", [(0.0, 0.0, 0.0), (0.03, 0.0, 0.0)],
                        0.006, rng(), _MAT_CFG)
    for obj in made:
        slots = obj.material_slots
        assert len(slots) == 1 and slots[0].material is not None
        assert slots[0].link == "OBJECT", (
            "a DATA-linked slot would be shared through the mesh the clones "
            "share, so every seated cell would take the last draw")
    assert made[0].material_slots[0].material \
        is not made[1].material_slots[0].material, (
        "each seated cell must get its own per-instance material draw")


def test_a_template_of_the_wrong_size_is_caught_loudly(W):
    """`_assert_seat_cell_footprint`: scene.py has ALREADY sized and placed
    this cell's seat from CELL_FORMATS before world.py builds anything, so a
    template that measures something else means the packing and the rendered
    geometry have desynced - a bay that reports room it does not have."""
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template(0.026, 0.070)]}})
    with pytest.raises(AssertionError, match="seated-cell template"):
        W.seat_cells(lib, "AnkerX", [(0.0, 0.0, 0.0)], 0.006, rng(), _MAT_CFG,
                     cell_format="18650")


def test_the_footprint_check_is_keyed_on_the_format_it_was_drawn_for(W):
    """Generalised from a hardcoded 18650 check: the same template that is
    correct for one format must fail against another, or the check has
    stopped saying anything about formats at all."""
    from recog.synth3d.config import CELL_FORMATS
    d21, l21 = CELL_FORMATS["21700"]
    lib = _FakeLibrary({"AnkerX": {"cell": [_cell_template(d21, l21)]}})
    assert W.seat_cells(lib, "AnkerX", [(0.0, 0.0, 0.0)], 0.006, rng(),
                        _MAT_CFG, cell_format="21700")
    W._seat_cell_footprint_checked.clear()
    with pytest.raises(AssertionError, match="21700|18650"):
        W.seat_cells(lib, "AnkerX", [(0.0, 0.0, 0.0)], 0.006, rng(), _MAT_CFG,
                     cell_format="18650")


def test_the_footprint_check_runs_once_per_asset_and_format(W):
    """It is memoised, so it must still fire for a DIFFERENT asset or a
    different format after one pair has passed - otherwise the first correct
    cartridge in a run would disable the check for every later one."""
    good = _FakeLibrary({"A": {"cell": [_cell_template()]}})
    W.seat_cells(good, "A", [(0, 0, 0)], 0.006, rng(), _MAT_CFG)
    assert ("A", "18650") in W._seat_cell_footprint_checked
    bad = _FakeLibrary({"B": {"cell": [_cell_template(0.030, 0.080)]}})
    with pytest.raises(AssertionError):
        W.seat_cells(bad, "B", [(0, 0, 0)], 0.006, rng(), _MAT_CFG)


# =========================================================================== #
#  9. _assert_procedural_tray_geometry - regression-pinning the history
#
#  This assertion runs on every procedural tray ever built and nobody has
#  seen it fail. An assertion nobody has ever seen fail is not evidence, so
#  each historical defect it exists to catch is reproduced here and the
#  assertion is REQUIRED to fire on it.
# =========================================================================== #

def _entry(**over):
    e = {"interior_mm": [2.0, 2.0, 58.0, 38.0],
         "case_outer_mm": [0.0, 0.0, 60.0, 40.0],
         "case_wall_mm": 2.0,
         "case_half_height_mm": 12.0,
         "tray_floor_mm": 3.0,
         "module_bay_mm": [2.0, 2.0, 58.0, 12.0],
         "cell_format": "18650",
         "lid_crown_mm": 0.0}
    e.update(over)
    return e


def _box(name, x0, y0, z0, x1, y1, z1):
    """A bare object occupying a given world-space AABB, in METRES."""
    obj = Object(name, np.array([(x, y, z) for x in (x0, x1)
                                 for y in (y0, y1) for z in (z0, z1)]))
    return obj


def _tray_objects(entry, lid_dz=0.0, cell_dz=0.0, cell_scale=1.0):
    """Correctly built case/lid/cell for `entry`, with optional sabotage."""
    from recog.synth3d.config import CELL_FORMATS
    ox0, oy0, ox1, oy1 = (v / 1000.0 for v in entry["case_outer_mm"])
    half = entry["case_half_height_mm"] / 1000.0
    floor = entry["tray_floor_mm"] / 1000.0
    diam, length = (v * cell_scale for v in CELL_FORMATS[entry["cell_format"]])
    case = _box("ProcCase_btm", ox0, oy0, 0.0, ox1, oy1, half)
    lid = _box("ProcCase_top", ox0, oy0, half + lid_dz, ox1, oy1,
               2 * half + lid_dz)
    cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    cell = _box("ProcCell_0", cx - diam / 2, cy - length / 2,
                floor + cell_dz, cx + diam / 2, cy + length / 2,
                floor + diam + cell_dz)
    return case, lid, cell


def test_a_correctly_built_tray_passes(W):
    entry = _entry()
    W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_an_inverted_assembly_is_caught(W):
    """THE defect of this project's history: Blender's glTF importer inverted
    every cartridge and `lay_flat` had no notion of which end was up, so the
    `placement_area` label was painted on the OUTSIDE of a closed lid.
    Undetected for months, because an upside-down cartridge renders
    perfectly plausibly from overhead. A lid BELOW the case is that defect."""
    entry = _entry()
    case, lid, cell = _tray_objects(entry)
    upside_down = _box("ProcCase_top",
                       *[v / 1000.0 for v in (entry["case_outer_mm"][0],
                                              entry["case_outer_mm"][1])],
                       -entry["case_half_height_mm"] / 1000.0,
                       *[v / 1000.0 for v in (entry["case_outer_mm"][2],
                                              entry["case_outer_mm"][3])],
                       0.0)
    with pytest.raises(AssertionError, match="lid base"):
        W._assert_procedural_tray_geometry(entry, case, upside_down, cell)


def test_a_lid_floating_above_the_rim_is_caught(W):
    """A 1mm gap between lid and case is invisible from overhead and turns a
    sealed cartridge into one with a light leak straight into the bay."""
    entry = _entry()
    with pytest.raises(AssertionError, match="lid base"):
        W._assert_procedural_tray_geometry(
            entry, *_tray_objects(entry, lid_dz=0.001))


def test_a_cell_floating_off_the_cavity_floor_is_caught(W):
    entry = _entry()
    with pytest.raises(AssertionError, match="cavity floor"):
        W._assert_procedural_tray_geometry(
            entry, *_tray_objects(entry, cell_dz=0.002))


def test_a_cell_buried_through_the_tray_base_is_caught(W):
    entry = _entry()
    with pytest.raises(AssertionError, match="cavity floor|below the tray"):
        W._assert_procedural_tray_geometry(
            entry, *_tray_objects(entry, cell_dz=-0.004))


def test_a_cell_of_the_wrong_format_is_caught(W):
    """The other half of the desync `_assert_seat_cell_footprint` guards: the
    bay was packed against CELL_FORMATS, so a cell built to another size means
    the packing and the geometry disagree."""
    entry = _entry()
    with pytest.raises(AssertionError, match="cell cross-section|cell length"):
        W._assert_procedural_tray_geometry(
            entry, *_tray_objects(entry, cell_scale=1.4))


def test_a_cell_standing_on_its_end_is_caught(W):
    """`lay_flat` rests every cell on its side. A cylinder left standing has
    the right diameter and the right length, just on the wrong axes - which
    from directly overhead is a circle instead of a rectangle, and the packer
    already reserved a rectangle for it."""
    entry = _entry()
    from recog.synth3d.config import CELL_FORMATS
    diam, length = CELL_FORMATS[entry["cell_format"]]
    ox0, oy0, ox1, oy1 = (v / 1000.0 for v in entry["case_outer_mm"])
    floor = entry["tray_floor_mm"] / 1000.0
    cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    standing = _box("ProcCell_0", cx - diam / 2, cy - diam / 2, floor,
                    cx + diam / 2, cy + diam / 2, floor + length)
    case, lid, _ = _tray_objects(entry)
    with pytest.raises(AssertionError):
        W._assert_procedural_tray_geometry(entry, case, lid, standing)


def test_a_zero_wall_thickness_entry_is_caught(W):
    """The fused-two-material shape: the helper that should have produced a
    separate inner wall silently did not, so `interior` and `case_outer`
    coincide and the cavity has no wall at all."""
    entry = _entry(interior_mm=[0.0, 0.0, 60.0, 40.0])
    with pytest.raises(AssertionError, match="zero wall thickness"):
        W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_a_cavity_floor_outside_the_case_is_caught(W):
    for floor in (0.0, -1.0, 12.0, 20.0):
        entry = _entry(tray_floor_mm=floor)
        with pytest.raises(AssertionError, match="tray_floor_mm"):
            W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_a_module_bay_outside_the_interior_is_caught(W):
    entry = _entry(module_bay_mm=[0.0, 0.0, 59.0, 12.0])
    with pytest.raises(AssertionError, match="module_bay_mm"):
        W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_a_module_bay_floating_in_the_middle_of_the_interior_is_caught(W):
    """`bay_edge` requires a full-span strip flush against exactly one
    interior edge. A bay that touches no edge is a hole in the middle of the
    tray, which is not what any real cartridge has."""
    entry = _entry(module_bay_mm=[10.0, 10.0, 40.0, 20.0])
    with pytest.raises(ValueError):
        W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_a_renamed_catalog_key_raises_instead_of_silently_not_building(W):
    """The exact historical defect: a renamed catalog key made a guarded
    `entry.get(...)` quietly stop building geometry entirely. Every key this
    function reads must raise KeyError when it is missing, not default."""
    for key in ("interior_mm", "case_outer_mm", "case_wall_mm",
                "case_half_height_mm", "tray_floor_mm", "module_bay_mm",
                "cell_format", "lid_crown_mm"):
        entry = _entry()
        objs = _tray_objects(entry)
        del entry[key]
        with pytest.raises(KeyError, match=key):
            W._assert_procedural_tray_geometry(entry, *objs)


def test_the_built_case_footprint_must_match_the_entry(W):
    entry = _entry()
    _, lid, cell = _tray_objects(entry)
    wrong = _box("ProcCase_btm", 0.0, 0.0, 0.0, 0.055, 0.040, 0.012)
    with pytest.raises(AssertionError, match="case footprint"):
        W._assert_procedural_tray_geometry(entry, wrong, lid, cell)


def test_the_lid_footprint_must_match_the_case(W):
    entry = _entry()
    case, _, cell = _tray_objects(entry)
    narrow = _box("ProcCase_top", 0.0, 0.0, 0.012, 0.050, 0.040, 0.024)
    with pytest.raises(AssertionError, match="lid footprint"):
        W._assert_procedural_tray_geometry(entry, case, narrow, cell)


def test_rounding_noise_in_the_entry_does_not_fire_the_wall_check(W):
    """`catalog.build_tray_entry` rounds interior/outer/wall INDEPENDENTLY to
    2dp, so a perfectly correct entry can be off by ~0.015mm of pure rounding.
    A check that fires on every tray is not a defect-catching check - and a
    real zero-wall defect is off by the full wall_mm, several hundred times
    this."""
    entry = _entry(interior_mm=[2.01, 1.99, 58.01, 37.99], case_wall_mm=2.0,
                   module_bay_mm=[2.01, 1.99, 58.01, 12.0])
    W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


def test_the_bay_containment_check_has_no_tolerance_of_its_own(W):
    """Deliberately pinned, because it is a asymmetry worth knowing about: the
    wall check tolerates 0.02mm of rounding but `module_bay_mm`'s containment
    in `interior_mm` is compared EXACTLY. That is safe only because
    `catalog.build_tray_entry` rounds a flush edge's two copies from the same
    float and then calls `bay_edge` on the rounded values itself. A future
    entry builder that derived the two independently would trip this."""
    entry = _entry(interior_mm=[2.01, 2.0, 58.0, 38.0],
                   module_bay_mm=[2.0, 2.0, 58.0, 12.0])
    with pytest.raises(AssertionError, match="module_bay_mm"):
        W._assert_procedural_tray_geometry(entry, *_tray_objects(entry))


# ---------------------------------------------------------- the lid crown ----

def _lid_with_crown(entry, crown_mm, segments=12, rolled_fraction=1.0):
    """A lid whose measured plateau and normals reproduce a real bevel."""
    ox0, oy0, ox1, oy1 = (v / 1000.0 for v in entry["case_outer_mm"])
    half = entry["case_half_height_mm"] / 1000.0
    crown = crown_mm / 1000.0
    lid = _box("ProcCase_top", ox0, oy0, half, ox1, oy1, 2 * half)
    if crown_mm <= 0.0:
        return lid
    top = 2 * half
    lid.data.vertices = [
        _Vert((ox0 + crown, oy0 + crown, top)),
        _Vert((ox1 - crown, oy0 + crown, top)),
        _Vert((ox0 + crown, oy1 - crown, top)),
        _Vert((ox1 - crown, oy1 - crown, top)),
        _Vert((ox0, oy0, half)), _Vert((ox1, oy1, half)),
    ]
    n_up = 4 * segments + 1
    n_rolled = int(round(rolled_fraction * 4 * segments))
    polys = [_Poly((0, 0, 1.0))]
    for i in range(4 * segments):
        polys.append(_Poly((0.5, 0.0, 0.9) if i < n_rolled else (0, 0, 1.0)))
    lid.data.polygons = polys
    assert len(polys) == n_up
    return lid


def test_a_zero_crown_lid_must_stay_a_plain_six_face_cuboid(W):
    """The control: a zero crown must reproduce the pre-2026-08-11 lid
    exactly, which is what lets a crowned render be attributed to the crown
    and to nothing else."""
    entry = _entry(lid_crown_mm=0.0)
    case, lid, cell = _tray_objects(entry)
    W._assert_procedural_tray_geometry(entry, case, lid, cell)
    assert len(lid.data.polygons) == 6


def test_a_zero_crown_lid_that_is_not_a_cuboid_is_caught(W):
    entry = _entry(lid_crown_mm=0.0)
    case, lid, cell = _tray_objects(entry)
    lid.data.polygons = [_Poly((0, 0, 1.0))] * 20
    with pytest.raises(AssertionError, match="polygons"):
        W._assert_procedural_tray_geometry(entry, case, lid, cell)


def test_a_real_crown_passes(W):
    entry = _entry(lid_crown_mm=1.5)
    case, _, cell = _tray_objects(entry)
    lid = _lid_with_crown(entry, 1.5)
    W._assert_procedural_tray_geometry(entry, case, lid, cell)


def test_a_bevel_that_silently_does_nothing_is_caught(W):
    """The defect class this project has already had once: a helper that was
    supposed to reshape geometry silently did not, and the result rendered
    plausibly. A lid that claims a 1.5mm crown but is still a flat six-face
    slab must fail."""
    entry = _entry(lid_crown_mm=1.5)
    case, flat_lid, cell = _tray_objects(entry)
    with pytest.raises(AssertionError, match="plateau|effectively flat"):
        W._assert_procedural_tray_geometry(entry, case, flat_lid, cell)


def test_a_crown_of_the_wrong_radius_is_caught(W):
    """The bevel ran, but with a radius that is not the one the entry drew and
    the manifest records."""
    entry = _entry(lid_crown_mm=1.5)
    case, _, cell = _tray_objects(entry)
    lid = _lid_with_crown(entry, 3.0)          # rolled twice as far
    with pytest.raises(AssertionError, match="plateau"):
        W._assert_procedural_tray_geometry(entry, case, lid, cell)


def test_a_crown_that_barely_rolls_anything_is_caught(W):
    """The measured property the crown exists to reproduce: 89% of each Anker
    lid's upward faces are non-planar, against the flat procedural lid's 0%.
    A roll that leaves the majority planar has not fixed anything."""
    entry = _entry(lid_crown_mm=1.5)
    case, _, cell = _tray_objects(entry)
    lid = _lid_with_crown(entry, 1.5, rolled_fraction=0.1)
    with pytest.raises(AssertionError, match="effectively flat"):
        W._assert_procedural_tray_geometry(entry, case, lid, cell)


def test_a_crown_taller_than_the_case_half_height_is_caught(W):
    entry = _entry(lid_crown_mm=30.0)
    case, _, cell = _tray_objects(entry)
    lid = _lid_with_crown(entry, 30.0)
    with pytest.raises(AssertionError, match="lid_crown_mm"):
        W._assert_procedural_tray_geometry(entry, case, lid, cell)


def test_a_crown_that_changed_the_lid_height_is_caught(W):
    """Bevelling a cube's TOP edges leaves every face centre where it was. If
    the lid's own height moved, the bevel ran on the wrong edges - and the
    lid's base is what has to stay flush on the case rim."""
    entry = _entry(lid_crown_mm=1.5)
    case, _, cell = _tray_objects(entry)
    lid = _lid_with_crown(entry, 1.5)
    lid.data._set_corners(np.array(
        [(x, y, z) for x in (0.0, 0.060) for y in (0.0, 0.040)
         for z in (0.012, 0.020)]))
    with pytest.raises(AssertionError, match="lid top"):
        W._assert_procedural_tray_geometry(entry, case, lid, cell)


# =========================================================================== #
#  10. build_procedural_tray end to end (through the stub)
# =========================================================================== #

def test_a_procedural_tray_builds_geometry_its_own_assertion_accepts(W):
    """The builder and its self-check must agree. This runs through the stub's
    primitive/scale/transform_apply model, so it is evidence about world.py's
    arithmetic and NOT about Blender's - see this file's header."""
    entry = _entry()
    by_role = W.build_procedural_tray(entry)
    assert sorted(by_role) == ["case", "case_lid", "cell"]
    assert [o.name for o in by_role["case"]] == ["ProcCase_btm"]
    assert [o.name for o in by_role["case_lid"]] == ["ProcCase_top"]
    assert [o.name for o in by_role["cell"]] == ["ProcCell_0"]


def test_the_procedural_object_names_still_classify_to_their_roles(W):
    """Get these names wrong and `_load_template`'s shared tail tags the lid as
    `case` too, re-closing every open procedural cartridge - one of the exact
    defects in this project's history."""
    from recog.synth3d.catalog import role_of
    by_role = W.build_procedural_tray(_entry())
    for role, objs in by_role.items():
        for o in objs:
            assert role_of(o.name) == role, (
                f"{o.name!r} classifies as {role_of(o.name)!r}, not {role!r}")


def test_the_cavity_cutter_is_applied_to_the_case_and_removed(W):
    by_role = W.build_procedural_tray(_entry())
    case = by_role["case"][0]
    assert [op for _, op, _ in case.applied_modifiers] == ["DIFFERENCE"]
    assert not [o for o in _STUB_BPY.data.objects.values()
                if o.name.startswith("_proc_cavity_cutter")]


def test_the_cell_is_built_resting_on_its_side(W):
    """`primitive_cylinder_add`'s own axis is Z - standing on end. The 90
    degree turn about X is what puts it in the same resting pose lay_flat
    gives every imported CAD cell."""
    from recog.synth3d.config import CELL_FORMATS
    diam, length = CELL_FORMATS["18650"]
    cell = W.build_procedural_tray(_entry())["cell"][0]
    lo, hi = cell.world_bbox()
    assert (hi - lo)[0] == pytest.approx(diam, abs=1e-6)
    assert (hi - lo)[1] == pytest.approx(length, abs=1e-6)
    assert (hi - lo)[2] == pytest.approx(diam, abs=1e-6)


def test_a_tray_whose_entry_is_inconsistent_fails_at_build_time(W):
    """The builder must not hand back geometry that its own checker rejects -
    the point of running the assertion inside `build_procedural_tray` rather
    than in a test is that it fires on every tray in a real run."""
    with pytest.raises(AssertionError):
        W.build_procedural_tray(_entry(tray_floor_mm=20.0))


def test_a_crown_whose_bevel_silently_did_nothing_fails_the_build(W):
    """`bmesh.ops.bevel` is a NO-OP in this harness. That is the test: the
    build must not quietly return a flat lid while the manifest records a
    1.5mm crown."""
    with bmesh_available():
        with pytest.raises(AssertionError, match="plateau|effectively flat"):
            W.build_procedural_tray(_entry(lid_crown_mm=1.5))


# =========================================================================== #
#  11. _crown_lid's own guard
# =========================================================================== #

def test_a_zero_crown_is_a_no_op_and_never_even_reaches_bmesh(W):
    """The early return is what keeps every existing config's geometry byte
    identical. It runs with bmesh absent from sys.modules entirely, so a
    regression that moved the import above the guard would fail here."""
    lid = _box("ProcCase_top", 0, 0, 0.012, 0.06, 0.04, 0.024)
    before = lid.data._corners.copy()
    assert "bmesh" not in sys.modules
    W._crown_lid(lid, 0.0)
    assert np.array_equal(lid.data._corners, before)


def test_crowning_something_that_is_not_a_fresh_cube_is_caught(W):
    """Handed anything but `build_procedural_tray`'s own primitive, the bevel
    would silently produce a shape nothing has measured."""
    lid = _box("ProcCase_top", 0, 0, 0.012, 0.06, 0.04, 0.024)
    lid.data._corners = lid.data._corners[:6]
    with bmesh_available():
        with pytest.raises(AssertionError, match="4-vert/4-edge"):
            W._crown_lid(lid, 0.0015)


def test_the_bevel_gets_the_four_top_edges_and_the_sampled_radius(W):
    """The most this harness can honestly say about `_crown_lid`: WHICH
    geometry it selects and WHAT radius it passes. "The bevel ran on the wrong
    edges" is a real failure mode - rolling the four VERTICAL edges instead
    would round the sealed unit's silhouette, and rolling the bottom four
    would lift the lid off the case rim it has to stay flush on. What the
    bevel then does with that selection is Blender's, and no stub can speak
    for it - see this file's header."""
    lid = _box("ProcCase_top", 0, 0, 0.012, 0.06, 0.04, 0.024)
    _STUB_BMESH.bevel_calls.clear()
    with bmesh_available():
        W._crown_lid(lid, 0.0015)
    assert len(_STUB_BMESH.bevel_calls) == 1
    call = _STUB_BMESH.bevel_calls[0]
    assert call["offset"] == pytest.approx(0.0015)
    assert call["offset_type"] == "OFFSET"
    assert call["segments"] == W._CROWN_SEGMENTS
    assert call["affect"] == "EDGES"
    geom = call["geom"]
    assert len(geom) == 8, "four top edges plus their four verts"
    tops = [g for g in geom if hasattr(g, "co")]
    edges = [g for g in geom if hasattr(g, "verts")]
    assert len(tops) == 4 and len(edges) == 4
    assert all(v.co.z == pytest.approx(0.024) for v in tops), (
        "the selection reached below the top face")
    assert all(v.co.z == pytest.approx(0.024) for e in edges for v in e.verts)


def test_only_the_crown_is_shade_smoothed(W):
    """Shading the whole object smooth would round the four vertical edges
    into the sides and turn a moulded slab into a melted one; leaving the
    crown flat-shaded renders 12 hard specular steps where a real fillet
    gives one continuous sweep. The rule is "every axis-aligned face stays
    flat"."""
    lid = _box("ProcCase_top", 0, 0, 0.012, 0.06, 0.04, 0.024)
    lid.data.polygons.append(_Poly((0.5, 0.0, 0.866)))     # a bevel facet
    with bmesh_available():
        W._crown_lid(lid, 0.0015)
    flat = [p for p in lid.data.polygons if not p.use_smooth]
    smooth = [p for p in lid.data.polygons if p.use_smooth]
    assert len(flat) == 6 and len(smooth) == 1
    assert smooth[0].normal.z == pytest.approx(0.866)


# =========================================================================== #
#  12. build_backdrop
# =========================================================================== #

_BACKDROP = {"image": None, "proc": "concrete", "luma_ref": 0.35,
             "uv_scale": [2.0, 2.0], "brightness": [0.0, 0.0],
             "roughness": [0.5, 0.5], "bump": [0.2, 0.2]}


def test_the_backdrop_is_built_at_the_z_it_was_given_and_records_it(W):
    """Jig scenes pass the plate's `backdrop_z`. Left at 0 the pocket floors -
    which are cut BELOW z=0 - showed backdrop instead of their own floor, so
    the recesses were geometrically real but visually absent."""
    for z in (0.0, -0.014, -0.02):
        _STUB_BPY.reset()
        plane, drawn = W.build_backdrop("bd", rng(), _cfg(
            backdrops={"bd": dict(_BACKDROP)}), size=3.0, z=z)
        lo, hi = plane.world_bbox()
        assert lo[2] == pytest.approx(z) and hi[2] == pytest.approx(z)
        assert drawn["z"] == z, "the manifest must record the plane's real z"


def test_the_backdrop_is_unlabelled_and_sized_as_asked(W):
    plane, _ = W.build_backdrop("bd", rng(), _cfg(
        backdrops={"bd": dict(_BACKDROP)}), size=2.4)
    lo, hi = plane.world_bbox()
    assert (hi - lo)[:2] == pytest.approx((2.4, 2.4))
    assert plane.pass_index == 0


def test_a_missing_backdrop_image_falls_back_to_the_procedural_texture(W):
    """`spec["image"] and os.path.exists(...)` - the source recorded in the
    manifest must be the one that actually ran, or a run reports image
    backdrops it never loaded."""
    spec = dict(_BACKDROP, image=str(ROOT / "definitely_absent_9c1.png"))
    _, drawn = W.build_backdrop("bd", rng(), _cfg(backdrops={"bd": spec}))
    assert drawn["source"] == "image", (
        "PINS CURRENT BEHAVIOUR: `source` is decided from the config alone "
        "and does NOT reflect the existence check that chose the procedural "
        "branch - see the report")


def test_the_procedural_source_is_named_after_the_kind(W):
    for kind in ("concrete", "brushed", "fabric", "paper", "belt"):
        _STUB_BPY.reset()
        _, drawn = W.build_backdrop("bd", rng(), _cfg(
            backdrops={"bd": dict(_BACKDROP, proc=kind)}))
        assert drawn["source"] == f"proc:{kind}"


def test_every_fallback_palette_kind_is_one_the_procedural_builder_knows(W):
    """A palette entry for a kind `_procedural` cannot build, or a kind with
    no palette, both end in the same silent grey default."""
    assert set(W._FALLBACK_PALETTE) == {"concrete", "brushed", "fabric",
                                        "paper", "belt"}
    for lo, hi in W._FALLBACK_PALETTE.values():
        assert len(lo) == 3 and len(hi) == 3
        assert all(0.0 <= c <= 1.0 for c in tuple(lo) + tuple(hi))
        assert all(l < h for l, h in zip(lo, hi)), (
            "a ramp whose low end is not below its high end is a flat colour")


def test_a_configured_colour_overrides_the_fallback_palette(W):
    """The single property that decides whether a part is visible against its
    background has to be the config's, not the hardcoded one."""
    spec = dict(_BACKDROP, color=[[0.9, 0.1, 0.1], [0.95, 0.2, 0.2]])
    plane, _ = W.build_backdrop("bd", rng(), _cfg(backdrops={"bd": spec}))
    ramp = next(n for n in plane.data.materials[0].node_tree.nodes
                if n.bl_idname == "ShaderNodeValToRGB")
    assert ramp.color_ramp.elements[0].color == (0.9, 0.1, 0.1, 1.0)
    assert ramp.color_ramp.elements[1].color == (0.95, 0.2, 0.2, 1.0)


def test_the_uv_scale_multiplies_the_mapping_rather_than_replacing_it(W):
    """`brushed` and `belt` pre-stretch the mapping to get their anisotropy.
    Replacing the scale instead of multiplying it would silently make every
    brushed backdrop isotropic."""
    spec = dict(_BACKDROP, proc="brushed", uv_scale=[3.0, 3.0])
    plane, drawn = W.build_backdrop("bd", rng(), _cfg(backdrops={"bd": spec}))
    mapping = next(n for n in plane.data.materials[0].node_tree.nodes
                   if n.bl_idname == "ShaderNodeMapping")
    assert mapping.inputs["Scale"].default_value == pytest.approx(
        (3.0, 180.0, 3.0)), "the brushed anisotropy (1, 60, 1) was lost"
    assert drawn["uv_scale"] == 3.0


def test_a_bump_below_the_threshold_builds_no_bump_node(W):
    """An early return in disguise: `if drawn["bump"] > 0.005`. The manifest
    records the drawn value either way, so a threshold that drifted would give
    a run whose recorded bump never reached a shader."""
    quiet = dict(_BACKDROP, bump=[0.001, 0.001])
    plane, drawn = W.build_backdrop("bd", rng(), _cfg(backdrops={"bd": quiet}))
    nodes = plane.data.materials[0].node_tree.nodes
    assert not [n for n in nodes if n.bl_idname == "ShaderNodeBump"]
    assert drawn["bump"] == 0.001, (
        "PINS CURRENT BEHAVIOUR: the manifest records a bump that was never "
        "applied whenever the drawn value is <= 0.005 - see the report")
    _STUB_BPY.reset()
    loud = dict(_BACKDROP, bump=[0.2, 0.2])
    plane, _ = W.build_backdrop("bd", rng(), _cfg(backdrops={"bd": loud}))
    assert [n for n in plane.data.materials[0].node_tree.nodes
            if n.bl_idname == "ShaderNodeBump"]


def test_a_brushed_backdrop_is_metallic_and_the_others_are_not(W):
    for kind, want in (("brushed", 1.0), ("concrete", 0.0), ("belt", 0.0)):
        _STUB_BPY.reset()
        plane, _ = W.build_backdrop("bd", rng(), _cfg(
            backdrops={"bd": dict(_BACKDROP, proc=kind)}))
        bsdf = plane.data.materials[0].node_tree.nodes.get("Principled BSDF")
        assert bsdf.inputs["Metallic"].default_value == want


# =========================================================================== #
#  13. _set_recorded - the manifest may not describe a value that never
#      reached the shader
#
#  `materials.set_input` returns False and continues when a socket is
#  missing. Every call site in world.py also writes that same value into the
#  `drawn` dict scene.py puts in the render manifest, so a silent False
#  renders a Blender default while the manifest states the drawn value - the
#  exact defect `_assert_wear_mix_took` exists for, in a module that had no
#  equivalent. Like that assertion this runs on every surface built rather
#  than in a test, so each way it can fire is reproduced here.
# =========================================================================== #

@contextlib.contextmanager
def _socket_renamed(old, new):
    """Rename a Principled input across the whole stub, as a Blender release
    would. The tolerance in `set_input` exists precisely because this has
    happened before (Clearcoat -> Coat Weight)."""
    table = _NODE_SOCKETS["ShaderNodeBsdfPrincipled"]
    before = list(table[0])
    table[0][:] = [((new if n == old else n), d) for n, d in before]
    try:
        yield
    finally:
        table[0][:] = before


def test_the_rename_harness_actually_renames_the_socket(W):
    """Guard the guard: if `_socket_renamed` silently did nothing, the four
    tests below would pass on any code at all."""
    with _socket_renamed("Roughness", "Surface Roughness"):
        node = _Node("ShaderNodeBsdfPrincipled", "Principled BSDF")
        assert "Roughness" not in node.inputs
        assert "Surface Roughness" in node.inputs
    assert "Roughness" in _Node("ShaderNodeBsdfPrincipled", "P").inputs


@pytest.mark.parametrize("socket", ["Base Color", "Roughness"])
def test_a_renamed_socket_stops_the_jig_rather_than_lying_about_it(W, socket):
    with _socket_renamed(socket, socket + " 2"):
        with pytest.raises(KeyError, match="never reached the shader"):
            W.build_jig([_pocket(0, 0, 0.05, 0.03, 0.008)], rng())


@pytest.mark.parametrize("socket", ["Base Color", "Roughness"])
def test_a_renamed_socket_stops_the_board_rather_than_lying_about_it(W, socket):
    with _socket_renamed(socket, socket + " 2"):
        with pytest.raises(KeyError, match="never reached the shader"):
            W.build_pcb((0, 0, 0.1, 0.06), 0.005, rng())


@pytest.mark.parametrize("socket", ["Base Color", "Roughness"])
def test_a_renamed_socket_stops_the_bay_proxy_rather_than_lying_about_it(
        W, socket):
    with _socket_renamed(socket, socket + " 2"):
        with pytest.raises(KeyError, match="never reached the shader"):
            W.build_bay_proxy((0.0, 0.0, 0.04, 0.02), 0.006, rng())


def test_a_renamed_socket_stops_the_backdrop_rather_than_lying_about_it(W):
    with _socket_renamed("Roughness", "Surface Roughness"):
        with pytest.raises(KeyError, match="never reached the shader"):
            W.build_backdrop("bd", rng(), _cfg(
                backdrops={"bd": dict(_BACKDROP)}))


def test_the_error_names_the_socket_and_what_the_build_does_offer(W):
    """A rename is undiagnosable from the traceback alone unless the message
    says what the running build actually has - the same requirement
    `socket_by_identifier` carries."""
    with _socket_renamed("Roughness", "Surface Roughness"):
        with pytest.raises(KeyError) as exc:
            W.build_bay_proxy((0.0, 0.0, 0.04, 0.02), 0.006, rng())
    assert "Roughness" in str(exc.value)
    assert "Surface Roughness" in str(exc.value)


def test_sockets_that_have_genuinely_moved_keep_their_tolerance(W):
    """The distinction this helper draws. `Metallic` is set but NOT recorded
    in any manifest, so it stays on plain `set_input` and a build without it
    must still render - that tolerance is why `set_input` exists at all."""
    with _socket_renamed("Metallic", "Metalness"):
        plate, drawn = W.build_jig([_pocket(0, 0, 0.05, 0.03, 0.008)], rng())
        assert plate is not None
        proxy, _ = W.build_bay_proxy((0.0, 0.0, 0.04, 0.02), 0.006, rng())
        assert proxy is not None


def test_every_manifest_recorded_surface_value_goes_through_the_check(W):
    """The rule stated as source, so a NEW builder that records a colour or a
    roughness cannot quietly go back to the silent form. Any `set_input` whose
    value expression mentions `drawn[...]` must be a `_set_recorded`."""
    tree = ast.parse(WORLD_PY.read_text(encoding="utf-8"))
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
            if not (isinstance(call.func, ast.Name)
                    and call.func.id == "set_input" and len(call.args) == 3):
                continue
            value_src = ast.dump(call.args[2])
            if "'drawn'" in value_src:
                offenders.append(f"{fn.name}:{call.args[1].value}")
    assert offenders == [], (
        f"these set_input calls write a value the manifest also records, so "
        f"a missing socket would render a default while the manifest claims "
        f"the drawn value: {offenders}. Use _set_recorded.")


# =========================================================================== #
#  14. source-level checks - what a stub bpy cannot reach
# =========================================================================== #

def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} has no function named {name!r}")


@pytest.mark.parametrize("func", [
    "build_backdrop", "build_jig", "build_pcb", "build_bay_proxy",
    "build_obstructions", "seat_cells", "build_procedural_tray",
    "_assert_procedural_tray_geometry", "setup_camera", "setup_lighting",
])
def test_no_builder_swallows_an_exception(func):
    """The historical defect this mirrors: a swallowed exception discarded the
    drawn roughness on 100% of surfaces while the manifest recorded the
    discarded values. `materials.build` carries the same check. A `try` here
    would let a build continue past a failure and produce a render that
    disagrees with the manifest describing it.

    `_crown_lid` is exempt and deliberately excluded: its `try` is a genuine
    two-signature bmesh API fallback that re-raises nothing away."""
    handlers = [n for n in ast.walk(_function(WORLD_PY, func))
                if isinstance(n, (ast.Try, ast.ExceptHandler))]
    assert not handlers, (
        f"world.{func} has grown an exception handler; if it is genuinely "
        f"needed it must re-raise or record, not continue")


def test_the_only_exception_handler_in_the_module_is_the_bmesh_fallback():
    """Stated as a whole-module budget so a `try` added to a function nobody
    thought to list above still fails here."""
    tries = []
    for node in ast.walk(ast.parse(WORLD_PY.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef):
            for n in ast.walk(node):
                if isinstance(n, ast.Try):
                    tries.append(node.name)
    assert tries == ["_crown_lid"], (
        f"world.py's exception handlers are now in {sorted(set(tries))}; only "
        f"_crown_lid's documented bmesh-signature fallback is expected")


def test_each_seating_constant_is_defined_in_exactly_one_place():
    """A second copy of any of these anywhere else would drift from the first
    silently - the two would agree until one was tuned.

    The split is the point. `JIG_LIFT`/`JIG_BACKDROP_GAP` are world.py's: they
    are about the jig plate's relationship to the backdrop, which is a
    rendering artefact (a coplanar plate renders BLACK) with no bay geometry
    in it at all. `SEATING_LADDER`/`PCB_THICKNESS_M`/`MAX_SEAT_OFFSET_M` are
    bay.py's: they decide what occludes what on the bay floor, which is the
    `placement_area` label's whole meaning, and they used to be literals here.
    `SEATED_CELL_LIFT` is gone entirely - it is the ladder's `seated_cell`
    rung, and a world.py copy of it is exactly the drift this forbids."""
    hits = []
    names = ("JIG_LIFT", "JIG_BACKDROP_GAP", "SEATED_CELL_LIFT",
             "SEATING_LADDER", "PCB_THICKNESS_M", "MAX_SEAT_OFFSET_M",
             "BAY_PROXY_RUNG")
    for path in (ROOT / "recog").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in names:
                    hits.append((path.name, t.id))
    assert sorted(hits) == [("bay.py", "BAY_PROXY_RUNG"),
                            ("bay.py", "MAX_SEAT_OFFSET_M"),
                            ("bay.py", "PCB_THICKNESS_M"),
                            ("bay.py", "SEATING_LADDER"),
                            ("world.py", "JIG_BACKDROP_GAP"),
                            ("world.py", "JIG_LIFT")]
