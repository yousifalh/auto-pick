"""
Orientation gate for recog.synth3d.assets. Requires bpy.

    blender -b --python recog/synth3d/_gate_orientation.py

Exits 0 if every check passes, 1 otherwise, so it works as a CI gate.

Why this exists
---------------
`lay_flat` and `place_item` were both complete no-ops for their entire life in
the ported code, and nothing caught it, because the failure is invisible: the
scene still renders, the parts are just never turned. Two causes, verified
against Blender 5.0.0:

  1. The glTF importer sets `rotation_mode = 'QUATERNION'`, and Blender ignores
     `rotation_euler` in that mode. Writes landed in the property and never
     reached `matrix_world`.
  2. Per-object euler writes rotate each object about its own origin, so even
     with the rotation mode forced they pull an assembly apart.

Check 2 below - two rot_deg values must give two different bounding boxes - is
the one that would have caught the original bug, and is the reason this file is
committed rather than being a throwaway probe. `layout.py` computes a rotation
for every placement; before the fix none of it reached the scene and the whole
dataset had zero rotation diversity.
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bpy                                                 # noqa: E402

from recog.synth3d import assets, config                   # noqa: E402
from recog.synth3d.layout import Placement                 # noqa: E402

ASSETS_DIR = os.path.join(_HERE, "assets")
TOL = 1e-6          # metres; bbox arithmetic is float32-clean well below this
MM = 1000.0

_failures: list = []


def check(ok: bool, msg: str):
    print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        _failures.append(msg)


def mm(v):
    return [round(float(c) * MM, 1) for c in v]


def extent(objs):
    lo, hi = assets.group_bbox(objs)
    return (hi - lo), lo, hi


def fresh_library():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return assets.AssetLibrary(ASSETS_DIR)


# --------------------------------------------------------------------------- #
#  1. lay_flat really lays parts flat
# --------------------------------------------------------------------------- #

def check_lay_flat():
    print("\n[1] lay_flat: smallest world extent must run along Z")
    lib = fresh_library()
    for name in lib.names():
        lib2 = fresh_library()
        by_role = lib2._load_template(name)
        parts = [(role, o) for role, objs in by_role.items() for o in objs]
        group = [o for _, o in parts]

        ext, _, _ = extent(group)
        check(ext.z <= min(ext.x, ext.y) + TOL,
              f"{name}: group {mm(ext)}mm - Z is the smallest extent")
        check(ext.z < max(ext.x, ext.y) - TOL,
              f"{name}: group {mm(ext)}mm - not standing on end")

        for role, o in parts:
            e, _, _ = extent([o])
            check(e.z <= min(e.x, e.y) + TOL,
                  f"{name}/{role} {o.name[:26]}: {mm(e)}mm - Z smallest")

        # An 18650 is 65.0 x 18.3 x 18.3mm. Lying down means a ~18.3mm height,
        # whichever of X/Y the long axis ends up on. Standing on end - the
        # pre-fix behaviour - gives a 65mm height and fails here.
        for role, o in parts:
            if role != "cell":
                continue
            e, _, _ = extent([o])
            check(abs(e.z * MM - 18.3) < 0.5,
                  f"{name}/cell height {round(e.z * MM, 1)}mm ~= 18.3mm "
                  f"(65.0mm would mean it is standing on end)")
            check(abs(max(e.x, e.y) * MM - 65.0) < 0.5,
                  f"{name}/cell long axis {round(max(e.x, e.y) * MM, 1)}mm "
                  f"~= 65.0mm and lies in the XY plane")


# --------------------------------------------------------------------------- #
#  2. place_item's rotation actually reaches the scene
# --------------------------------------------------------------------------- #

def _assembled_item(lib, name, rng):
    variant = next(v for v in config.VARIANTS if v.name == "assembled")
    items = lib.instantiate(name, variant, rng)
    assert len(items) == 1, f"expected one merged item, got {len(items)}"
    return items[0]


def check_rotation_is_live():
    print("\n[2] place_item: different rot_deg must give different bboxes")
    name = "AnkerPowerCore10000"
    boxes = {}
    for rot in (0.0, 90.0, 37.0):
        lib = fresh_library()
        item = _assembled_item(lib, name, random.Random(0))
        assets.place_item(item, Placement(x=0.0, y=0.0, quarter=0,
                                          rot_deg=rot), random.Random(0))
        ext, _, _ = extent(item.objects)
        boxes[rot] = ext
        print(f"        rot={rot:6.1f} -> ext {mm(ext)}mm")

    e0, e90, e37 = boxes[0.0], boxes[90.0], boxes[37.0]

    check(abs(e0.x - e90.x) * MM > 1.0,
          f"rot 0 vs 90 differ in X ({round(e0.x * MM, 1)} vs "
          f"{round(e90.x * MM, 1)}mm) - THE assertion that catches a no-op")
    # Deliberately the exact-swap form, not merely "the extents differ". The
    # pre-fix code translated members individually while the rotation was a
    # no-op, which smeared the group's Y extent (22.2 -> 54.9mm) and would have
    # satisfied a loose "they differ" check while the assembly was being pulled
    # apart rather than turned.
    check(abs(e0.x - e90.y) * MM < 0.1 and abs(e0.y - e90.x) * MM < 0.1,
          "a 90 degree turn swaps the X and Y extents exactly")
    check(abs(e0.x - e37.x) * MM > 1.0,
          f"an arbitrary angle also takes effect (rot 37 X = "
          f"{round(e37.x * MM, 1)}mm vs {round(e0.x * MM, 1)}mm)")
    check(e37.x > e0.x - TOL and e37.y > e0.y - TOL,
          "an off-axis turn grows the axis-aligned bbox, as a real rotation must")
    check(abs(e0.z - e90.z) * MM < 0.1 and abs(e0.z - e37.z) * MM < 0.1,
          "a Z rotation leaves the height unchanged")


# --------------------------------------------------------------------------- #
#  3. place_item lands the group where it was asked to
# --------------------------------------------------------------------------- #

def check_placement_pose():
    print("\n[3] place_item: centre on (x, y), group resting on z = 0")
    name = "AnkerPowerCore10000"
    for x, y, rot in ((0.123, -0.045, 37.0), (-0.080, 0.210, 90.0),
                      (0.0, 0.0, 0.0)):
        lib = fresh_library()
        item = _assembled_item(lib, name, random.Random(1))
        assets.place_item(item, Placement(x=x, y=y, quarter=0, rot_deg=rot),
                          random.Random(1))
        _, lo, hi = extent(item.objects)
        cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2
        check(abs(cx - x) < TOL and abs(cy - y) < TOL,
              f"rot={rot}: centre ({round(cx, 6)}, {round(cy, 6)}) == "
              f"requested ({x}, {y})")
        check(abs(lo.z) < TOL,
              f"rot={rot}: rests on the floor (z_min = {round(lo.z, 9)})")


def main():
    print(f"orientation gate - Blender {bpy.app.version_string}")
    print(f"assets: {ASSETS_DIR}")
    check_lay_flat()
    check_rotation_is_live()
    check_placement_pose()

    print()
    if _failures:
        print(f"ORIENTATION GATE FAILED - {len(_failures)} check(s):")
        for f in _failures:
            print(f"   - {f}")
        return 1
    print("ORIENTATION GATE OK")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nORIENTATION GATE FAILED - unhandled exception")
        code = 1
    sys.exit(code)
