"""
recog.synth3d.bay - every geometric decision about a cartridge's interior.

This module has no Blender dependencies, so all of it is unit-testable outside
Blender. world.py and scene.py depend on Blender and cannot be tested; they
call into here for the numbers and only apply the result. Keeping the arithmetic
on this side of the line is what makes the bay geometry checkable at all.

Units follow the caller. `module_bay_from_bounds` is used on millimetre CAD
bounds at conversion time and on metre scene bounds at render time; it is scale
free.
"""

from __future__ import annotations

import math
from typing import Tuple

Rect = Tuple[float, float, float, float]     # x0, y0, x1, y1


def module_bay_from_bounds(interior: Rect, cells: Rect) -> Rect:
    """The strip of interior the cells do not occupy, on the widest side.

    The four Anker assemblies all leave 2.4-5.9 mm on three sides (wall
    thickness) and 23.5-35.0 mm on one short side. That large gap is the
    electronics bay: the CAD has no PCB part, but it reserves the space.

    Returns the bay as a rectangle spanning the interior's full width on
    the chosen axis, because the real module runs wall to wall.

    Raises ValueError if `cells` is not contained in `interior` (a bad
    upstream measurement would otherwise produce a negative-width rectangle
    that propagates silently into the placement area), or if two or more
    sides tie for the largest gap (an exactly-centred cell union has no
    unambiguous bay side; resolving that by dict/iteration order would be
    an accident, not a decision - callers should look at the geometry
    rather than get a silently arbitrary answer).
    """
    ix0, iy0, ix1, iy1 = interior
    cx0, cy0, cx1, cy1 = cells

    if not (ix0 <= cx0 <= cx1 <= ix1 and iy0 <= cy0 <= cy1 <= iy1):
        raise ValueError(
            f"module_bay_from_bounds: cells {cells} are not contained in "
            f"interior {interior}")

    gaps = {
        "-x": cx0 - ix0,
        "+x": ix1 - cx1,
        "-y": cy0 - iy0,
        "+y": iy1 - cy1,
    }
    best = max(gaps.values())
    tied = [k for k, v in gaps.items() if v == best]
    if len(tied) > 1:
        raise ValueError(
            f"module_bay_from_bounds: ambiguous - sides {tied} tie for the "
            f"largest gap ({best:g}); cells {cells} are not off-centre "
            f"enough in interior {interior} to pick a bay side")
    side = tied[0]

    if side == "-x":
        return (ix0, iy0, cx0, iy1)
    if side == "+x":
        return (cx1, iy0, ix1, iy1)
    if side == "-y":
        return (ix0, iy0, ix1, cy0)
    return (ix0, cy1, ix1, iy1)


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


def module_rect_local(footprint: Tuple[float, float], bay_mm: Rect,
                      interior_mm: Rect) -> Rect:
    """Where the electronics module sits, in the cartridge's OWN local
    frame - centred on the cartridge's own pivot, before any placement
    rotation or translation.

    `footprint` is `assets.Item.footprint` (metres): the cartridge's
    un-rotated (size_x, size_y), the same numbers `layout.plan` packs
    with. `bay_mm`/`interior_mm` come from catalog.json in millimetres.

    A cartridge's own axis-aligned bbox `[lo, hi]` is, trivially, centred
    on `(lo+hi)/2` - that is what "centre" means - so representing it as
    `(-w/2, -h/2, w/2, h/2)` is exact, not an approximation, for ANY
    cartridge shape. `_lerp_rect` only uses fractional position within
    each rect's own span, so it does not matter where interior_mm's
    absolute origin sits either. The bay keeps its proportion of the
    interior, so the module lands against the same short side at the same
    relative depth whatever the scene scale - and, because this never
    touches a rotated world AABB, at the cartridge's true physical size
    regardless of how the placement later rotates it. Use
    `module_world_placement` for the world-space centre and size after a
    `layout.Placement` is applied.
    """
    fw, fh = footprint
    local = (-fw / 2, -fh / 2, fw / 2, fh / 2)
    return _lerp_rect(bay_mm, interior_mm, local)


def placement_rect_local(footprint: Tuple[float, float], bay_mm: Rect,
                         interior_mm: Rect) -> Rect:
    """The battery placement area, in the same local frame as
    `module_rect_local`: the interior minus the module bay, taken on
    whichever axis the bay occupies.
    """
    fw, fh = footprint
    fx0, fy0, fx1, fy1 = (-fw / 2, -fh / 2, fw / 2, fh / 2)
    mx0, my0, mx1, my1 = module_rect_local(footprint, bay_mm, interior_mm)

    # The bay spans one full axis; the placement area is what is left on
    # the other. Compare against the footprint edges to find which.
    tol = 1e-9
    if abs(mx0 - fx0) < tol and abs(mx1 - fx1) < tol:
        # Bay spans full width -> it took a y side.
        return (fx0, my1, fx1, fy1) if abs(my0 - fy0) < tol \
            else (fx0, fy0, fx1, my0)
    return (mx1, fy0, fx1, fy1) if abs(mx0 - fx0) < tol \
        else (fx0, fy0, mx0, fy1)


def module_world_placement(footprint: Tuple[float, float], bay_mm: Rect,
                           interior_mm: Rect, rot_deg: float,
                           translate: Tuple[float, float]
                           ) -> Tuple[float, float, float, float]:
    """The module's centre and TRUE size in world space.

    Returns `(cx, cy, w, h)` in metres. `rot_deg` and `translate` are the
    SAME `layout.Placement.rot_deg` and `(x, y)` the cartridge itself was
    placed with - the caller (world.py) still has to rotate the board
    MESH by `rot_deg` about `(cx, cy)` to match the cartridge's actual
    orientation; this only gets the centre and size right.

    This works in `module_rect_local`'s frame and applies ONE rigid
    rotate-then-translate to the module's centre - the same composition
    `assets.place_item` applies to the cartridge itself. Rotating a
    rectangle does not change its own width or height, so `w, h` come out
    exact for ANY `rot_deg`, including the jitter `layout.plan` adds on
    top of every k*90 turn.

    That last point is the reason this function exists rather than a
    single `bay_mm -> world AABB` lerp: `layout.plan` rotates by
    `quarter*90 + jitter` (jitter up to a few degrees), and the AABB of a
    rotated rectangle is LARGER than the rectangle - its diagonal projects
    wider than either side. Mapping the bay proportionally into that
    AABB (as a naive world-footprint lerp does) inflates and mislocates
    the module by several percent even at a couple of degrees of jitter,
    enough to visibly overhang the case - measured on an 81.7x180mm case
    at 2 degrees: the AABB is 87.9x182.8, a 7.6% inflation on the short
    axis. Rotating one POINT (the centre) instead of re-bounding a whole
    rotated RECTANGLE has no such effect: a point has no extent to
    inflate.
    """
    lx0, ly0, lx1, ly1 = module_rect_local(footprint, bay_mm, interior_mm)
    w, h = lx1 - lx0, ly1 - ly0
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    wcx = translate[0] + lcx * cos_t - lcy * sin_t
    wcy = translate[1] + lcx * sin_t + lcy * cos_t
    return wcx, wcy, w, h
