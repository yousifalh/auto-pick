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
