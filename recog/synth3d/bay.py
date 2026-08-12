"""
recog.synth3d.bay - every geometric decision about a cartridge's interior.

This module has no Blender dependencies, so all of it is unit-testable outside
Blender. world.py and scene.py depend on Blender and cannot be tested; they
call into here for the numbers and only apply the result. Keeping the arithmetic
on this side of the line is what makes the bay geometry checkable at all.

That line held for XY and not for Z until 2026-08-12: this module decided every
footprint while world.py decided every height, including the seating offsets
that ARE the `placement_area` occlusion mechanism. See the SEATING LADDER
section below - the ladder, `fallback_module_placement` and
`obstruction_z_scale` are the Z half, moved here so the same tests can reach
them. What stayed in world.py is construction: which primitive, which
material, which parent.

Units follow the caller. `module_bay_from_bounds` is used on millimetre CAD
bounds at conversion time and on metre scene bounds at render time; it is scale
free.

Task-10 amendment note (superseded by task 3): `case_interior_mm` used to be
the case meshes' OUTER AABB, not a true interior, so the module and bay proxy
tiled it completely - leaving zero `cartridge` (shell) pixels on an open
unit. A `wall_mm` parameter threaded through `module_rect_local`/
`placement_rect_local`/`module_world_placement`/`placement_world_placement`
used to inset an artificial margin, via a lerp from `interior_mm`'s frame
onto the cartridge's own outer footprint, to fake a rim.

Task 2 replaced `case_interior_mm` with a REAL `interior_mm` - the tray's
actual cavity, already inset by the measured wall (`interior_from_tray`).
That made the lerp both unnecessary and wrong: it was only an identity
mapping while `interior_mm` and the footprint were (near enough) the same
rectangle, which stopped being true once `interior_mm` became a genuinely
smaller, real cavity - the lerp then stretched the module/placement rects
back OUT across the whole footprint, including the wall, instead of
confining them to the cavity.

Task 3 removes the lerp and `wall_mm` entirely. `tray_outer_mm` (and so
`interior_mm` and `module_bay_mm`) are measured centred on (0, 0) in the
CAD's own frame, and an item's local frame (`assets.place_item`'s pivot) is
centred the same way - they are the SAME frame, differing only by
millimetres vs. metres. `module_rect_local`/`placement_rect_local` now take
`interior_mm`/`bay_mm` directly and only convert units; no footprint
argument, no `wall_mm`, no lerp.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]     # x0, y0, x1, y1


def needs_flip(case_centroid_z: float, lid_centroid_z: float) -> bool:
    """Is the imported assembly upside down?

    Task-3c: Blender's glTF importer maps the source (x, y, z) to
    (x, -z, y) - for this CAD, the raw file's up-axis is Y and the
    cavity opens toward -Y, so -Y lands on -Z: the tray ends up facing
    the ground. `assets.lay_flat` only picks WHICH axis is vertical (the
    group's smallest extent); it has no notion of which END of that axis
    is "up", so it cannot catch this - it correctly lays the assembly
    flat, just sometimes flat and inverted.

    The lid must sit ABOVE the shell in a right-side-up assembly (the lid
    caps the cavity from above). If it does not - the shell's own
    centroid is at or above the lid's - the assembly is inverted and
    needs a 180-degree flip (about a horizontal axis) to correct it.

    Pass world-space Z centroids (e.g. the mean of each role's own
    group-bbox lo/hi) computed in the SAME frame, before any translation
    that would make the comparison meaningless (rotation-only stages are
    fine - centroid order is rotation-invariant about the flip axis by
    construction, since both objects rotate together).
    """
    return case_centroid_z >= lid_centroid_z


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


def case_wall_from_bounds(interior: Rect, cells: Rect) -> float:
    """The case's wall thickness, in the same units as `interior`/`cells`
    (millimetres, at the CAD-conversion call site this is used from).

    `module_bay_from_bounds` already establishes that the single largest of
    the four edge gaps is the module bay - a deliberate void the CAD
    reserves for the PCB, not wall material (see its own docstring). The
    OTHER axis has no such void: both of its gaps are unbroken shell wall,
    and for the four Anker assemblies they agree with each other to within
    CAD tessellation noise (4.0/4.0, 3.75/3.75, 3.7/3.7, 4.25/4.25 mm) even
    though the assemblies themselves range from 63 to 82 mm wide - which is
    what makes "the wall thickness" a single well-defined number per asset
    rather than four different ones.

    Deliberately does NOT look at the bay axis's own near-side gap (the
    "closed end" opposite the module bay): measured across the four
    assemblies that one is a DIFFERENT, unrelated figure (2.45/5.5/5.9/5.0
    mm against walls of 4.0/3.75/3.7/4.25 mm) - plausibly a spring-contact
    recess rather than plain wall on at least the smallest case - so
    folding it into this average would understate or overstate the wall
    depending on the asset. Trusting only the axis with no such ambiguity
    is a decision, not an oversight; see the amendment brief this function
    was added for.

    Raises ValueError under the same "cells not contained in interior" and
    "ambiguous tied gap" conditions as `module_bay_from_bounds` (a bad
    upstream measurement or an exactly-centred cell union should fail
    loudly here too, for the same reasons), and additionally if the two
    non-bay-axis gaps disagree by more than tessellation noise should
    allow - that would mean this asset's wall is not the uniform thickness
    this function assumes, and a silently-averaged number would be worse
    than an error surfaced once, at conversion time.
    """
    ix0, iy0, ix1, iy1 = interior
    cx0, cy0, cx1, cy1 = cells

    if not (ix0 <= cx0 <= cx1 <= ix1 and iy0 <= cy0 <= cy1 <= iy1):
        raise ValueError(
            f"case_wall_from_bounds: cells {cells} are not contained in "
            f"interior {interior}")

    gaps = {
        "-x": cx0 - ix0, "+x": ix1 - cx1,
        "-y": cy0 - iy0, "+y": iy1 - cy1,
    }
    best = max(gaps.values())
    tied = [k for k, v in gaps.items() if v == best]
    if len(tied) > 1:
        raise ValueError(
            f"case_wall_from_bounds: ambiguous - sides {tied} tie for the "
            f"largest gap ({best:g}); cells {cells} are not off-centre "
            f"enough in interior {interior} to pick a bay side")

    bay_axis_is_y = tied[0] in ("-y", "+y")
    a, b = (gaps["-x"], gaps["+x"]) if bay_axis_is_y \
        else (gaps["-y"], gaps["+y"])
    if not math.isclose(a, b, rel_tol=0.02, abs_tol=0.05):
        raise ValueError(
            f"case_wall_from_bounds: the non-bay-axis gaps disagree "
            f"({a:g} vs {b:g}) by more than CAD tessellation noise should "
            f"allow - this asset's wall is not the uniform thickness this "
            f"function assumes, so averaging them would silently invent a "
            f"number")
    return (a + b) / 2.0


def interior_from_tray(tray_outer: Rect, cells_union: Rect,
                       wall: float) -> Rect:
    """The tray's cavity footprint: its outer rectangle inset by the wall.

    `case_interior_mm` used to hold the AABB of every case mesh - the
    assembly's OUTER extent - despite its name. This is the real thing: the
    space a cell can actually occupy.

    The cavity is widened if necessary to contain the cells, which
    demonstrably fit in assembled pose. A wall measurement that excludes
    them is wrong, and trusting it would shrink every placement area.
    """
    tx0, ty0, tx1, ty1 = tray_outer
    cx0, cy0, cx1, cy1 = cells_union

    ix0, iy0 = tx0 + wall, ty0 + wall
    ix1, iy1 = tx1 - wall, ty1 - wall

    if ix1 - ix0 <= 0.0 or iy1 - iy0 <= 0.0:
        raise ValueError(
            f"interior_from_tray: wall {wall} swallows the cavity of "
            f"{tray_outer}")

    # The cells sit inside the cavity in the CAD. If the inset excludes any
    # of them the wall is over-measured, so widen rather than propagate the
    # error.
    ix0, iy0 = min(ix0, cx0), min(iy0, cy0)
    ix1, iy1 = max(ix1, cx1), max(iy1, cy1)

    return (max(ix0, tx0), max(iy0, ty0), min(ix1, tx1), min(iy1, ty1))


def _bay_edge(interior_mm: Rect, bay_mm: Rect, tol: float = 1e-6) -> str:
    """Which single edge of `interior_mm` the strip `bay_mm` is flush
    against, as one of `"-x"`, `"+x"`, `"-y"`, `"+y"`.

    `bay_mm` (catalog.json's `module_bay_mm`, from `module_bay_from_bounds`)
    is always a rectangle spanning `interior_mm`'s FULL width on one axis
    and flush against one edge on the other - "the module runs wall to
    wall" (see that function's docstring). This checks that shape
    explicitly rather than assuming it, because both `module_rect_local`
    and `placement_rect_local` need to know which edge to build the
    complement against, and a bad upstream measurement (module_bay_mm not
    actually a full-span strip) should fail loudly here rather than
    silently return a plausible-looking rectangle - the way a wrong
    `case_wall_mm` upstream already once corrupted `interior_mm` unnoticed.

    Raises ValueError if `bay_mm` is not a full-span strip flush against
    exactly one edge of `interior_mm` (including the degenerate case where
    `bay_mm` spans both axes fully, i.e. equals `interior_mm` itself).
    """
    ix0, iy0, ix1, iy1 = interior_mm
    bx0, by0, bx1, by1 = bay_mm

    x_full = math.isclose(bx0, ix0, abs_tol=tol) and \
        math.isclose(bx1, ix1, abs_tol=tol)
    y_full = math.isclose(by0, iy0, abs_tol=tol) and \
        math.isclose(by1, iy1, abs_tol=tol)

    candidates = []
    if x_full:
        if math.isclose(by0, iy0, abs_tol=tol) and \
                not math.isclose(by1, iy1, abs_tol=tol):
            candidates.append("-y")
        if math.isclose(by1, iy1, abs_tol=tol) and \
                not math.isclose(by0, iy0, abs_tol=tol):
            candidates.append("+y")
    if y_full:
        if math.isclose(bx0, ix0, abs_tol=tol) and \
                not math.isclose(bx1, ix1, abs_tol=tol):
            candidates.append("-x")
        if math.isclose(bx1, ix1, abs_tol=tol) and \
                not math.isclose(bx0, ix0, abs_tol=tol):
            candidates.append("+x")

    if len(candidates) != 1:
        raise ValueError(
            f"bay {bay_mm} is not a full-span strip flush against exactly "
            f"one edge of interior {interior_mm} (candidates: {candidates})"
            f" - module_bay_mm should always run wall to wall on one axis; "
            f"a bad upstream measurement must fail loudly rather than "
            f"silently return a plausible rectangle")
    return candidates[0]


def module_rect_local(interior_mm: Rect, bay_mm: Rect) -> Rect:
    """Where the electronics module sits, in the cartridge's OWN local
    frame - centred on the cartridge's own pivot, before any placement
    rotation or translation. Returns metres.

    `interior_mm`/`bay_mm` are catalog.json's `interior_mm`/`module_bay_mm`,
    in millimetres. No lerp and no footprint argument: `tray_outer_mm` (and
    therefore `interior_mm` and `bay_mm`, both derived from it) is measured
    centred on (0, 0) in the CAD's own frame, and an item's local frame
    (`assets.place_item`'s pivot) is centred on the SAME point - they are
    the same frame, differing only by millimetres vs. metres. `bay_mm`
    already IS the module's true position within the cavity; it needs
    unit conversion, not remapping. (See the module docstring for why this
    replaces a lerp through the cartridge's outer footprint, which stopped
    being correct once `interior_mm` became a real, smaller cavity.)

    Use `module_world_placement` for the world-space centre and size after
    a `layout.Placement` is applied.
    """
    _bay_edge(interior_mm, bay_mm)          # validate; raises if malformed
    x0, y0, x1, y1 = bay_mm
    return (x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0)


def placement_rect_local(interior_mm: Rect, bay_mm: Rect) -> Rect:
    """The battery placement area, in the same local frame as
    `module_rect_local`: `interior_mm`'s complement of `bay_mm`, on
    whichever edge `bay_mm` is flush against - found generically via
    `_bay_edge` rather than assumed to be any particular side. Returns
    metres.
    """
    edge = _bay_edge(interior_mm, bay_mm)
    ix0, iy0, ix1, iy1 = interior_mm
    bx0, by0, bx1, by1 = bay_mm
    rect_mm = {
        "-x": (bx1, iy0, ix1, iy1),
        "+x": (ix0, iy0, bx0, iy1),
        "-y": (ix0, by1, ix1, iy1),
        "+y": (ix0, iy0, ix1, by0),
    }[edge]
    return tuple(v / 1000.0 for v in rect_mm)


def module_world_placement(interior_mm: Rect, bay_mm: Rect, rot_deg: float,
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

    That last point is the reason this function rotates a POINT rather
    than re-bounding a rotated RECTANGLE: `layout.plan` rotates by
    `quarter*90 + jitter` (jitter up to a few degrees), and the AABB of a
    rotated rectangle is LARGER than the rectangle - its diagonal projects
    wider than either side. Mapping the bay proportionally into that AABB
    (a naive world-footprint lerp) inflates and mislocates the module by
    several percent even at a couple of degrees of jitter, enough to
    visibly overhang the case - measured on an 81.7x180mm case at 2
    degrees: the AABB is 87.9x182.8, a 7.6% inflation on the short axis.
    Rotating one POINT (the centre) instead has no such effect: a point
    has no extent to inflate.
    """
    lx0, ly0, lx1, ly1 = module_rect_local(interior_mm, bay_mm)
    w, h = lx1 - lx0, ly1 - ly0
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    wcx = translate[0] + lcx * cos_t - lcy * sin_t
    wcy = translate[1] + lcx * sin_t + lcy * cos_t
    return wcx, wcy, w, h


def placement_world_placement(interior_mm: Rect, bay_mm: Rect,
                              rot_deg: float,
                              translate: Tuple[float, float]
                              ) -> Tuple[float, float, float, float]:
    """The placement area's centre and TRUE size in world space.

    Identical contract and identical reasoning to `module_world_placement`
    - same `(cx, cy, w, h)` shape, same rotate-the-centre-POINT approach for
    the same reason (a rotated world AABB would inflate this rectangle just
    as it would the module's) - applied to `placement_rect_local` instead
    of `module_rect_local`. This is the proxy's half of the interior: the
    module and the placement area are complementary rectangles in the same
    local frame, so carrying each through this same one-rotate-one-translate
    pipeline keeps them exact AND keeps them adjacent-not-overlapping after
    a `layout.plan` jitter, because a rigid transform applied identically to
    two disjoint rectangles cannot make them overlap.
    """
    lx0, ly0, lx1, ly1 = placement_rect_local(interior_mm, bay_mm)
    w, h = lx1 - lx0, ly1 - ly0
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    wcx = translate[0] + lcx * cos_t - lcy * sin_t
    wcy = translate[1] + lcx * sin_t + lcy * cos_t
    return wcx, wcy, w, h


def fallback_module_placement(bounds_xy: Rect, rng: random.Random
                              ) -> Tuple[float, float, float, float]:
    """The module board's centre and size when there is no measured bay.

    Returns `(cx, cy, w, h)` in the SAME frame and the SAME shape as
    `module_world_placement`, so `world.build_pcb` consumes the two
    interchangeably: the anchored path gets the strip the CAD actually
    reserves, this one gets a plausible board drawn inside `bounds_xy` (the
    cartridge's own world-space footprint).

    This is what `build_pcb` did before `module_bay_mm` existed, and what it
    still does for any asset whose catalog entry has no bay measurement. Every
    real photograph contradicts it - the module is against one short wall, not
    in the middle - so it is a fallback, not an alternative. It stays because
    a centred board is box-safe (`annotate.boxes_from_mask` takes min/max of
    visible pixels, so a hole in the middle of a case does not move its box),
    which makes it survivable in a way a wrongly-POSITIONED board would not be.

    The four draws are taken in this exact order - width fraction, height
    fraction, X jitter, Y jitter - because `build_pcb` shares one `rng` with
    every other draw in the scene: reordering them, or taking one draw where
    there used to be four, silently resamples the board colour, its component
    count and everything downstream in that scene's stream. Only the
    un-anchored path draws at all; the anchored path must consume nothing.
    """
    x0, y0, x1, y1 = bounds_xy
    w = (x1 - x0) * rng.uniform(0.55, 0.80)
    h = (y1 - y0) * rng.uniform(0.20, 0.38)
    cx = (x0 + x1) / 2 + rng.uniform(-0.004, 0.004)
    cy = (y0 + y1) / 2 + rng.uniform(-0.010, 0.010)
    return cx, cy, w, h


# =========================================================================== #
#  THE SEATING LADDER - what occludes what on the bay floor
#
#  `world.build_bay_proxy` builds a flat plane carrying the `placement_area`
#  label, and `annotate.boxes_from_mask` reports only the pixels of it that
#  stayed VISIBLE in the index pass. That is the whole mechanism: anything
#  resting in the bay hides its own patch of proxy and so subtracts itself
#  from `placement_area`, with no mask arithmetic anywhere - which is why the
#  label means "the currently FREE floor" rather than "the nominal bay".
#
#  That mechanism is nothing but an ORDERING. Every object in the bay is built
#  at a small offset above the cavity floor (`floor_z` - catalog.json's
#  tray_floor_mm, converted to metres by scene.py), and an object seated at or
#  below the proxy's own offset loses the z-fight, or loses outright, and
#  stops occluding it. The mask then keeps reporting that floor as free while
#  the object visibly sits on it. Nothing downstream can detect that: the
#  render, the index pass and the manifest all agree with each other, and all
#  three are wrong. Every `placement_area` label in every dataset rests on
#  these six numbers being in this order.
#
#  Which is why they live here. They used to be six literals spread across
#  three bpy-only builders plus a module constant in world.py, with the
#  ordering restated in three docstrings and enforced nowhere - decided on the
#  side of the bpy line no test can reach, while bay.py decided every
#  footprint. This is one table, checked by `assert_seating_ladder_ordered`
#  on every `seat_z` call, so a perturbed offset stops the build instead of
#  quietly re-labelling every dataset generated afterwards.
#
#  These are CLEARANCES, not stand-offs: fractions of a millimetre, invisible
#  from overhead against an 18mm cell (see world.JIG_LIFT's note for what a
#  clearance of ZERO costs). They are hand-tuned. Do not round them off.
# =========================================================================== #

# Ordered from the cavity floor up. Each value is the offset in metres above
# `floor_z` at which world.py builds that object's ORIGIN - not its base:
# `pcb` is a slab built about its own centre, and the obstruction primitives
# are spheres, cubes and planes that straddle their origins too. `seated_cell`
# is the exception the comment beside it names, and it is the one rung whose
# offset IS a base, because `world.seat_cells` measures the clone it just laid
# flat and translates its lowest point onto the seat.
SEATING_LADDER: Tuple[Tuple[str, float], ...] = (
    ("pcb",         0.0008),   # world.build_pcb's board  (its CENTRE)
    ("bay_proxy",   0.0009),   # world.build_bay_proxy - the labelled plane
    ("tape",        0.0011),   # world.build_obstructions
    ("label",       0.0011),   # world.build_obstructions
    ("adhesive",    0.0012),   # world.build_obstructions
    ("seated_cell", 0.0012),   # world.seat_cells         (the cell's BASE)
    ("foam",        0.0022),   # world.build_obstructions
)

# The rung carrying the `placement_area` label. Every rung ABOVE it must clear
# it strictly; the one rung below it is `pcb`, deliberately - see below.
BAY_PROXY_RUNG = "bay_proxy"

# world.build_pcb's board thickness, in metres. Load-bearing for the ladder,
# not a free parameter: the board is built about its centre, so seating it at
# `pcb` = PCB_THICKNESS_M / 2 is exactly what rests it ON the cavity floor
# rather than floating it or sinking it, and its TOP face at PCB_THICKNESS_M
# is what has to clear the proxy so that along the edge the board and the
# placement rectangle share, the board occludes the proxy and not the reverse.
# That pair of relations is why `pcb` is the one rung BELOW the proxy and not
# a violation of the ordering; `assert_seating_ladder_ordered` checks both
# rather than leaving them to the prose.
PCB_THICKNESS_M = 0.0016

# Largest offset that still counts as a clearance rather than a stand-off. A
# millimetre-scale gap between a cell and the floor it is meant to be resting
# on would be visible from overhead against an 18mm cell, and would be a
# modelling error dressed up as a z-fighting fix.
MAX_SEAT_OFFSET_M = 0.003


def assert_seating_ladder_ordered(ladder: Optional[
        Tuple[Tuple[str, float], ...]] = None) -> None:
    """Raise ValueError unless the seating ladder still holds.

    `ValueError`, not `assert`: `python -O` strips assertions, and an
    ordering that silently stops being checked is the failure this exists to
    rule out. `world.py`'s build-time `_assert_*` helpers keep the module's
    own `assert` idiom and check the numbers actually reaching the geometry;
    this checks the table they come from. Both, deliberately - a correct
    table does not help if world.py stops asking for it, and a build-time
    check on world.py's side does not help if the table itself is perturbed.

    The invariants, in the order they are checked:

      1. no rung is named twice, and `bay_proxy` appears exactly once - the
         ladder is keyed by name and a duplicate would make `seat_offset`
         return whichever copy came first;
      2. the table is non-decreasing AS WRITTEN, so reading it top to bottom
         is reading the physical stack bottom to top. Ties are legal and two
         are real (tape/label, adhesive/seated_cell): coplanar objects at the
         same offset never occlude EACH OTHER, and nothing requires them to;
      3. every offset is positive and under MAX_SEAT_OFFSET_M;
      4. every rung above `bay_proxy` clears it STRICTLY. This is the one the
         `placement_area` label rests on;
      5. every rung below `bay_proxy` is strictly below it;
      6. `pcb`, the only such rung, is at PCB_THICKNESS_M / 2 - resting the
         board on the cavity floor - and its top face clears the proxy.
    """
    ladder = SEATING_LADDER if ladder is None else ladder
    names = [n for n, _ in ladder]
    zs = [z for _, z in ladder]

    if len(set(names)) != len(names):
        raise ValueError(
            f"the seating ladder names a rung twice ({names}); seat_offset "
            f"would return whichever copy came first")
    if names.count(BAY_PROXY_RUNG) != 1:
        raise ValueError(
            f"the seating ladder must contain exactly one {BAY_PROXY_RUNG!r} "
            f"rung - it is the plane every other rung is ordered against - "
            f"but it has {names.count(BAY_PROXY_RUNG)}")
    if zs != sorted(zs):
        raise ValueError(
            f"the seating ladder is out of order as written: {list(ladder)}. "
            f"It is read top to bottom as the physical stack bottom to top, "
            f"so a table that does not sort is a table nobody can read")
    bad = [(n, z) for n, z in ladder
           if not (0.0 < z < MAX_SEAT_OFFSET_M)]
    if bad:
        raise ValueError(
            f"seating offsets must be clearances in (0, {MAX_SEAT_OFFSET_M}) "
            f"metres, not stand-offs: {bad}")

    i = names.index(BAY_PROXY_RUNG)
    proxy_z = zs[i]
    not_clearing = [(n, z) for n, z in ladder[i + 1:] if not z > proxy_z]
    if not_clearing:
        raise ValueError(
            f"{not_clearing} do not rest STRICTLY above the bay proxy at "
            f"+{proxy_z}m, so they would not occlude it - `placement_area` "
            f"would keep reporting that floor as free while they visibly sit "
            f"on it, in the render, the index pass and the manifest alike")
    not_below = [(n, z) for n, z in ladder[:i] if not z < proxy_z]
    if not_below:
        raise ValueError(
            f"{not_below} are listed below the bay proxy but are not below "
            f"it at +{proxy_z}m")

    pcb_z = dict(ladder).get("pcb")
    if pcb_z is not None:
        if pcb_z != PCB_THICKNESS_M / 2.0:
            raise ValueError(
                f"the pcb rung is +{pcb_z}m but the board is "
                f"{PCB_THICKNESS_M}m thick and built about its CENTRE, so "
                f"anything other than +{PCB_THICKNESS_M / 2.0}m floats it "
                f"above the cavity floor or sinks it through")
        if not pcb_z + PCB_THICKNESS_M / 2.0 > proxy_z:
            raise ValueError(
                f"the board's top face at "
                f"+{pcb_z + PCB_THICKNESS_M / 2.0}m does not clear the bay "
                f"proxy at +{proxy_z}m, so along the edge the board and the "
                f"placement rectangle share, the proxy would occlude the "
                f"board rather than the other way round")


def seat_offset(name: str) -> float:
    """`name`'s rung of the seating ladder, in metres above the cavity floor.

    Raises ValueError on an unknown rung rather than returning a default. A
    guarded `.get(...)` on a renamed key is how this project once had a
    builder quietly stop building geometry; here it would seat an object on
    the floor itself, where it would z-fight with the tray and stop occluding
    the proxy - the exact silent failure the ladder exists to prevent.
    """
    for rung, z in SEATING_LADDER:
        if rung == name:
            return z
    raise ValueError(
        f"{name!r} is not a rung of the seating ladder; known rungs are "
        f"{[n for n, _ in SEATING_LADDER]}")


def seat_z(floor_z: float, name: str) -> float:
    """The world Z at which world.py builds `name`, given the cavity floor.

    This is the call world.py makes instead of knowing a number: `floor_z`
    is catalog.json's tray_floor_mm in metres (scene.py converts it), and the
    return value is where the primitive's origin goes.

    Checks the whole ladder on every call. It is seven comparisons against a
    seven-row table, taken a handful of times per bay, and it means a
    perturbed offset stops the FIRST build rather than the first person to
    look at pixels months later - which is how four of this project's five
    historical silent defects were actually found.
    """
    assert_seating_ladder_ordered()
    return floor_z + seat_offset(name)


def occludes_bay_proxy(name: str) -> bool:
    """Whether `name` is seated strictly above the plane carrying the
    `placement_area` label, and therefore subtracts itself from that label.

    True for every rung except `bay_proxy` itself and `pcb` - the board is
    seated below the proxy because it rests ON the floor, and occludes
    through its thickness instead (see PCB_THICKNESS_M).
    """
    return seat_offset(name) > seat_offset(BAY_PROXY_RUNG)


# =========================================================================== #
#  OBSTRUCTIONS
#
#  IMG_4426 shows thermal adhesive, foam pads, tape crosses and printed
#  labels sitting in every opened real bay; none of it is in the CAD, and a
#  placement mask trained only on clean bays would site a cell on a glue
#  blob. They sit ON the bay proxy and occlude it (world.build_obstructions,
#  scene.py), so they subtract themselves from `placement_area` with no mask
#  arithmetic anywhere - the proxy's label is already "whatever is currently
#  free floor", not "the nominal bay".
# =========================================================================== #

@dataclass(frozen=True)
class ObstructionPose:
    """A piece of foreign matter in a bay. Centre-based, caller's units."""
    kind: str            # adhesive | foam | tape | label
    x: float
    y: float
    w: float
    h: float
    rot_deg: float = 0.0


def sample_obstructions(placement_rect: Rect, cfg,
                        rng: random.Random) -> List[ObstructionPose]:
    """Draw the foreign matter sitting in one bay, in the SAME frame as
    `placement_rect`.

    This is a pure function of a rect and a config - same pattern as
    `module_rect_local`/`placement_rect_local` - so it works equally well
    called on `placement_rect_local`'s LOCAL-frame result (the cartridge's
    own pivot at the origin, before any placement rotation) as on a
    world-space rect. The caller wants the local-frame case: `sample_
    obstructions` never sees the cartridge's rotation, and `obstruction_
    world_poses` below carries its output into world space afterwards,
    exactly as `module_world_placement`/`placement_world_placement` do for
    the module and placement rectangles.

    IMG_4426 shows thermal adhesive, foam pads, tape crosses and printed
    labels in the real bays. None of it is in the CAD, and a placement
    mask that ignores it would site a cell on a glue blob.

    `cfg.p_none` of bays come back empty so the network also sees clean
    ones. Sizes are fractions of the SHORTER bay edge, so an obstruction
    scales with the cartridge rather than being absolute.
    """
    x0, y0, x1, y1 = placement_rect
    bw, bh = x1 - x0, y1 - y0
    short = min(bw, bh)

    if rng.random() < cfg.p_none:
        return []

    out: List[ObstructionPose] = []
    for kind, count_range, frac_range in (
        ("adhesive", cfg.n_adhesive, cfg.adhesive_frac),
        ("foam", cfg.n_foam, cfg.foam_frac),
        ("tape", cfg.n_tape, cfg.tape_frac),
        ("label", cfg.n_label, cfg.label_frac),
    ):
        for _ in range(rng.randint(*count_range)):
            w = short * rng.uniform(*frac_range)
            h = w * rng.uniform(0.6, 1.8) if kind != "tape" \
                else short * rng.uniform(0.5, 0.95)
            # Clamp BEFORE sampling the centre below: `rng.uniform(a, b)`
            # with a > b does not raise, it silently returns a value outside
            # [a, b] - and an obstruction wider/taller than the bay produces
            # exactly that inverted range on the next two lines otherwise.
            w = min(w, bw)
            h = min(h, bh)
            out.append(ObstructionPose(
                kind=kind,
                x=rng.uniform(x0 + w / 2, x1 - w / 2),
                y=rng.uniform(y0 + h / 2, y1 - h / 2),
                w=w, h=h,
                rot_deg=rng.uniform(-180, 180) if kind != "tape"
                else rng.choice([0.0, 90.0]) + rng.uniform(-4, 4),
            ))
    return out


def obstruction_world_poses(poses: List[ObstructionPose], rot_deg: float,
                            translate: Tuple[float, float]
                            ) -> List[ObstructionPose]:
    """Carry LOCAL obstruction poses into world space.

    Same contract and same reasoning as `module_world_placement` /
    `placement_world_placement`: each obstruction's centre is a POINT in
    the cartridge's own local frame (the frame `placement_rect_local`
    returns), so rotating that point about the local origin by `rot_deg`
    and then translating by `translate` is exact for ANY angle - a point
    has no extent for the rotation to inflate, unlike lerping into a
    rotated world AABB. `rot_deg` and `translate` are the SAME `layout.
    Placement.rot_deg` and `(x, y)` the cartridge, the module board and the
    placement proxy were placed with, so an obstruction rotates WITH its
    cartridge: a glue blob drawn near one edge of a bay stays near that
    same physical edge after the cartridge turns 90 degrees, instead of
    staying at a fixed world (x, y) while the bay rotates out from under it.

    `w`/`h` are unchanged - rotating a rectangle does not change its own
    width or height - and each obstruction's own `rot_deg` (its tilt within
    the bay) is added to the cartridge's, so a tape cross laid crosswise on
    the local placement rect is still crosswise after the turn.
    """
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    tx, ty = translate
    out: List[ObstructionPose] = []
    for p in poses:
        wx = tx + p.x * cos_t - p.y * sin_t
        wy = ty + p.x * sin_t + p.y * cos_t
        out.append(ObstructionPose(kind=p.kind, x=wx, y=wy, w=p.w, h=p.h,
                                   rot_deg=(p.rot_deg + rot_deg) % 360.0))
    return out


def obstruction_z_scale(kind: str, rng: random.Random) -> float:
    """The Z scale `world.build_obstructions` gives `kind`'s primitive - the
    third dimension of a piece of foreign matter, the one `ObstructionPose`
    does not carry.

    `sample_obstructions` decides w and h; this decides the remaining axis,
    so both halves of an obstruction's geometry are drawn on the bpy-free
    side. The QUANTITY is the same in every case - the z component of the
    object's `scale` - but what it multiplies differs with the primitive,
    which is why this returns a scale rather than a height:

      * `adhesive` is a UV sphere of radius w/2, so this is a dimensionless
        SQUASH factor: a blob is a flattened dome, not a ball, and 0.35-0.7
        keeps its height between about a third and two thirds of its width;
      * `foam` is a unit cube, so this is a THICKNESS in metres - a 2-5mm
        pad, matching the ones in IMG_4426;
      * `tape` and `label` are flat planes with no third dimension at all,
        so their scale is exactly 1.0 and NO random draw is taken. The draw
        count per obstruction is part of this function's contract:
        `build_obstructions` shares one `rng` with the rest of the scene, so
        a draw taken for a kind that used to take none silently resamples
        everything downstream of it in that stream.

    Raises ValueError on an unknown kind rather than falling through to the
    plane case. `build_obstructions`' shape dispatch ends in a bare
    `else:  # label`, so a fifth kind added to `sample_obstructions` without
    a matching branch there would render as a printed label - silently and
    plausibly, the same shape as the renamed catalog key that once made a
    guarded `.get` stop building geometry. `build_obstructions` calls this
    BEFORE that dispatch, so the build stops instead.
    """
    if kind == "adhesive":
        return rng.uniform(0.35, 0.7)
    if kind == "foam":
        return rng.uniform(0.002, 0.005)
    if kind in ("tape", "label"):
        return 1.0
    raise ValueError(
        f"{kind!r} is not an obstruction kind sample_obstructions produces "
        f"(adhesive, foam, tape, label), so world.build_obstructions has no "
        f"shape for it and would silently build it as a printed label")


# =========================================================================== #
#  SEATED CELLS
#
#  The deployed robot fills a cartridge one cell at a time, so its camera
#  sees PARTLY-FILLED bays for most of every run - not the empty-or-sealed
#  cases the generator produced before this. `layout.plan` already lets parts
#  overlap and lifts them via Placement.z (max_overlap_iou in
#  configs/synth3d.yaml), so cells already landed on cartridges, but at
#  random positions and angles. What was missing is the SEATED case:
#  axis-aligned, inside the bay, at the pitch the packer itself would choose.
#
#  Positions come from the SAME FFDH packer common.packing exposes to the
#  real planner, so the synthetic partly-filled bay matches what the packer
#  would actually produce rather than an invented arrangement. Seated cells
#  sit ON the bay proxy and occlude it (world.seat_cells, scene.py), exactly
#  the mechanism sample_obstructions/obstruction_world_poses established:
#  placement_area shrinks to the free floor with no mask arithmetic anywhere.
#
#  Obstructions and seated cells are sampled against the SAME placement_rect,
#  independently - so a bay containing both must keep them apart, or a cell
#  would render sitting spatially on top of physical foreign matter, which is
#  impossible and undermines the whole reason obstructions exist (a mask that
#  never saw a clean bay-vs-glue-blob distinction would learn nothing from
#  one). `obstruction_forbidden_mask` rasterises obstruction footprints into
#  the SAME forbidden-cell grid `first_fit_decreasing` already knows how to
#  pack around (it now advances past a forbidden cell and keeps packing the
#  shelf, rather than abandoning it - see common.packing's own docstring),
#  so `seated_cell_poses` seats cells on the free floor around foreign
#  matter, matching what the real packer would have to do too.
# =========================================================================== #

# Forbidden-grid resolution for seated_cell_poses's obstruction avoidance, in
# the SAME metres this module uses throughout - NOT first_fit_decreasing's
# own default of 1.5 taken literally, which is calibrated for the REAL
# planner working in millimetres; passed straight through here that would
# mean a 1.5-METRE grid cell, dozens of times larger than the whole bay.
# 1.5mm (0.0015) keeps the SAME physical resolution the real planner already
# uses for the identical algorithm: fine enough that a seated cell's
# stand-off from a rasterised obstruction is sub-millimetre quantisation
# error, not centimetres, and coarse enough that a bay's grid stays a few
# dozen cells on a side (0.055m / 0.0015m ~ 37 columns) rather than
# thousands.
SEAT_MM_PER_CELL = 0.0015


def obstruction_forbidden_mask(poses: List[ObstructionPose],
                               placement_rect: Rect,
                               mm_per_cell: float = SEAT_MM_PER_CELL
                               ) -> np.ndarray:
    """Rasterise obstruction footprints into a `first_fit_decreasing`
    forbidden-cell grid over `placement_rect`, so `seated_cell_poses` can be
    told to seat cells around foreign matter instead of on top of it.

    `poses` must be in the SAME frame as `placement_rect` - the local frame
    `sample_obstructions` returns, before `obstruction_world_poses` carries
    them into world space. `seated_cell_poses` samples in that same local
    frame for the same reason (see its own docstring), so the two compose
    directly with no extra conversion.

    Each obstruction is rasterised as its AXIS-ALIGNED BOUNDING BOX, not its
    exact rotated footprint. `first_fit_decreasing`'s forbidden_mask is
    itself a grid of axis-aligned cells, so an exact rotated polygon could
    only ever be approximated by that same grid anyway - and a bounding box
    is a conservative (never smaller) approximation: a cell packed clear of
    an obstruction's AABB is provably clear of its exact rotated shape too.
    The cost is that a cell may stand off a heavily-rotated obstruction by a
    bit more than strictly necessary; the alternative (a tighter but
    non-conservative approximation) could let a cell corner clip a rotated
    obstruction's real footprint, which is the exact failure this function
    exists to rule out.

    Grid indexed `[row, col]` the way `first_fit_decreasing` expects: row is
    the Y axis, col is the X axis, both aligned to `placement_rect`'s own
    `(x0, y0)` origin - the SAME origin `seated_cell_poses` already measures
    the packer's strip from.
    """
    x0, y0, x1, y1 = placement_rect
    strip_w, strip_h = x1 - x0, y1 - y0
    n_cols = max(1, int(math.ceil(strip_w / mm_per_cell)))
    n_rows = max(1, int(math.ceil(strip_h / mm_per_cell)))
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for p in poses:
        # Half-extent of the rotated rectangle's AABB: the standard
        # |cos|*w/2 + |sin|*h/2 projection, applied on both axes.
        theta = math.radians(p.rot_deg)
        cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
        half_w = (p.w * cos_t + p.h * sin_t) / 2
        half_h = (p.w * sin_t + p.h * cos_t) / 2
        c0 = max(0, int(math.floor((p.x - half_w - x0) / mm_per_cell)))
        c1 = min(n_cols, int(math.ceil((p.x + half_w - x0) / mm_per_cell)))
        r0 = max(0, int(math.floor((p.y - half_h - y0) / mm_per_cell)))
        r1 = min(n_rows, int(math.ceil((p.y + half_h - y0) / mm_per_cell)))
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    return mask


def seated_cell_poses(placement_rect: Rect, cell_w: float, cell_h: float,
                      n: int, rng: random.Random,
                      forbidden_mask: Optional[np.ndarray] = None,
                      mm_per_cell: float = SEAT_MM_PER_CELL
                      ) -> List[Tuple[float, float, float]]:
    """Up to `n` cell centres seated in the bay at the packer's own pitch.

    Same frame contract as `sample_obstructions`: this is a pure function of
    whatever rect it is given, so the caller passes `placement_rect_local`'s
    LOCAL-frame result (the cartridge's own pivot at the origin, before any
    placement rotation) and carries the output into world space afterwards
    with `seated_cell_world_poses`, exactly as `obstruction_world_poses`
    does for obstructions.

    Positions come from the SAME FFDH packer `common.packing` exposes to the
    real planner (`first_fit_decreasing`), so the synthetic partly-filled bay
    matches what the packer would actually produce rather than an invented
    arrangement.

    `forbidden_mask`, if given, is `obstruction_forbidden_mask`'s output
    over this SAME `placement_rect` - cells are then packed around it rather
    than on top of it, using the packer's own obstacle-advancing behaviour
    (it advances past a forbidden cell and keeps packing the same shelf,
    rather than abandoning it - see common.packing's docstring). Omit it
    (the default) and every shelf is a clean strip of `placement_rect`, as
    before obstructions existed. `mm_per_cell` must match whatever
    resolution `forbidden_mask` was built at; the default is
    `SEAT_MM_PER_CELL`.

    Returns `[(x, y, rot_deg), ...]` with `rot_deg` in {0, 90} - the cell's
    OWN pitch orientation from the packer, in the same LOCAL frame as
    `placement_rect`. Fewer than `n` come back when the bay cannot hold that
    many - including a bay so densely obstructed that none fit at all, which
    is correct behaviour, not a failure: `first_fit_decreasing` reports the
    rest as unplaced rather than overlapping them or raising.
    """
    from common.packing import Item as _PackItem
    from common.packing import first_fit_decreasing

    if n <= 0:
        return []

    x0, y0, x1, y1 = placement_rect
    strip_w, strip_h = x1 - x0, y1 - y0

    items = [_PackItem(i, cell_w, cell_h) for i in range(n)]
    res = first_fit_decreasing(items, strip_w, strip_h, allow_rotation=True,
                               forbidden_mask=forbidden_mask,
                               mm_per_cell=mm_per_cell)

    out = []
    for p in res.placements:
        out.append((
            x0 + p.x + p.width / 2,
            y0 + p.y + p.height / 2,
            90.0 if p.rotated else 0.0,
        ))
    rng.shuffle(out)
    return out[:n]


def bay_edge(interior_mm: Rect, bay_mm: Rect, tol: float = 1e-6) -> str:
    """Public wrapper around `_bay_edge`, for cross-module callers
    (catalog.build_tray_entry) that need to validate a chosen bay edge
    without reaching into a leading-underscore name."""
    return _bay_edge(interior_mm, bay_mm, tol)


@dataclass(frozen=True)
class TraySample:
    """One procedurally-sampled cartridge tray, fully resolved to plain
    numbers - the pure-RNG half of a procedural asset (design spec
    Sec3.2). Millimetres throughout, matching catalog.json's own
    convention.

    `interior_mm` is the CAVITY's own boundary (what scene.py reads).
    `case_outer_mm` is the PHYSICAL shell's outer footprint - interior_mm
    expanded outward by wall_mm on all four sides, including the bay
    side - needed only by world.build_procedural_tray to give the wall
    real thickness; scene.py never reads it (see this plan's header note
    #2 for why this is a separate field rather than aliasing interior_mm
    the way an over-literal reading of design spec Sec5.2 step 4 might
    suggest).
    """
    interior_mm: Rect
    module_bay_mm: Rect
    case_outer_mm: Rect
    wall_mm: float
    case_half_height_mm: float
    tray_floor_mm: float
    cell_format: str
    bay_edge: str
    # Fillet radius rolled onto the lid's four TOP edges. Defaults to 0.0
    # (a planar cuboid lid, the geometry every render before 2026-08-11
    # used) so a caller that predates the field builds exactly what it
    # always built. See sample_tray and config.TrayRangeCfg.
    lid_crown_mm: float = 0.0


# Minimum clearance (mm) sample_tray guarantees between tray_floor_mm and
# case_half_height_mm - see that function's docstring for why this is a
# structural requirement, not a plausibility tuning knob.
MIN_CAVITY_RIM_CLEARANCE_MM = 1.0

# Largest fraction of the lid's own SHORTEST outer side that the top-edge
# crown fillet may consume. The bevel rolls in from both opposing edges,
# so at 0.5 the top plateau collapses to a ridge and beyond it the bevel
# is self-intersecting - a degenerate mesh, the same failure class as a
# zero-height cavity cutter, not merely an unusual tray. 0.45 leaves a
# real plateau on every draw. The measured Anker lids sit at 0.35/0.28/
# 0.36/0.27 of their own half-width, i.e. 0.18-0.13 of the full short
# side, comfortably inside this.
MAX_LID_CROWN_FOOTPRINT_FRAC = 0.45

# Minimum clearance (mm) sample_tray guarantees between a cell's own
# diameter and the ASSEMBLED shell's full height (2 * case_half_height_mm
# - case + lid together, the envelope world.build_procedural_tray's
# "assembled" variant actually seals the cell inside). See sample_tray's
# docstring for why this coupling exists at all.
MIN_CELL_HEIGHT_CLEARANCE_MM = 1.0


def sample_tray(cfg, rng: random.Random) -> TraySample:
    """Draw one procedural tray from `cfg` (config.Config.tray_anchored
    or .tray_wide - the caller picks which). Pure RNG and arithmetic,
    same shape as sample_obstructions: no Blender call anywhere in this
    function.

    Design spec Sec5.2's five steps, in order:
      1. cell format + an n_cols x n_rows grid at a sampled pitch ->
         cell_union, centred on (0, 0).
      2/3. wall_mm on the three non-bay sides, bay_margin_mm on the
         fourth (the edge chosen by cfg.free_bay_edge - fixed to the
         tray's LONGER axis for anchored, matching design spec Sec9.1's
         "short edge" reading of all four measured SKUs; free among all
         four edges for wide) -> interior_mm, with module_bay_mm falling
         out as the full-span strip between cell_union and interior_mm
         on that one edge, by construction (never inferred, never a tie -
         see bay_edge's own use in the caller as a loud cross-check).
      4/5. tray_floor_mm and case_half_height_mm sampled directly from
         cfg's own ranges.

    See this plan's header note #1 for why case_half_height_mm (Z) and
    the bay_margin used above (XY) are two distinct sampled quantities
    rather than the design spec's single "bay depth" name.
    """
    from .config import CELL_FORMATS

    cell_format = rng.choice(cfg.cell_formats)
    cell_w_mm, cell_h_mm = (v * 1000.0 for v in CELL_FORMATS[cell_format])
    n_cols = rng.randint(*cfg.n_cols_range)
    n_rows = rng.randint(*cfg.n_rows_range)
    pitch_mm = rng.uniform(*cfg.pitch_mm_range)

    union_w = n_cols * cell_w_mm + (n_cols + 1) * pitch_mm
    union_h = n_rows * cell_h_mm + (n_rows + 1) * pitch_mm
    cx0, cy0, cx1, cy1 = -union_w / 2, -union_h / 2, union_w / 2, union_h / 2

    wall_mm = rng.uniform(*cfg.wall_mm_range)
    bay_margin_mm = rng.uniform(*cfg.bay_margin_mm_range)

    long_axis_is_y = union_h >= union_w
    if cfg.free_bay_edge:
        edge = rng.choice(("-x", "+x", "-y", "+y"))
    else:
        edge = rng.choice(("-y", "+y") if long_axis_is_y else ("-x", "+x"))

    if edge == "-x":
        interior = (cx0 - bay_margin_mm, cy0 - wall_mm, cx1 + wall_mm, cy1 + wall_mm)
        module_bay = (interior[0], interior[1], cx0, interior[3])
    elif edge == "+x":
        interior = (cx0 - wall_mm, cy0 - wall_mm, cx1 + bay_margin_mm, cy1 + wall_mm)
        module_bay = (cx1, interior[1], interior[2], interior[3])
    elif edge == "-y":
        interior = (cx0 - wall_mm, cy0 - bay_margin_mm, cx1 + wall_mm, cy1 + wall_mm)
        module_bay = (interior[0], interior[1], interior[2], cy0)
    else:  # "+y"
        interior = (cx0 - wall_mm, cy0 - wall_mm, cx1 + wall_mm, cy1 + bay_margin_mm)
        module_bay = (interior[0], cy1, interior[2], interior[3])

    ix0, iy0, ix1, iy1 = interior
    case_outer = (ix0 - wall_mm, iy0 - wall_mm, ix1 + wall_mm, iy1 + wall_mm)

    # case_half_height_mm and tray_floor_mm are independently drawn from
    # cfg's own ranges (design spec Sec5.2 steps 4/5) - but world.
    # build_procedural_tray's cavity cutter has height (case_half_height_mm
    # - tray_floor_mm), and TrayRangeCfg.tray_wide's own ranges overlap
    # (tray_floor_mm_range up to 4.5mm, case_half_height_mm_range down to
    # 4.0mm): an unconstrained pair of draws can put the floor at or above
    # the rim roughly 1 draw in 400 (measured), giving that cutter zero or
    # negative height - a degenerate boolean cut (the same failure class as
    # "a hole punched through the wrong mesh face"), not merely an unusual
    # tray. This is a structural validity requirement for ANY tray, so it
    # is enforced here by construction rather than left to chance; nudging
    # case_half_height_mm UP (never shrinking tray_floor_mm) keeps the more
    # physically-anchored quantity (the floor, anchored near the measured
    # 1.95mm) exactly as drawn.
    case_half_height_mm = rng.uniform(*cfg.case_half_height_mm_range)
    tray_floor_mm = rng.uniform(*cfg.tray_floor_mm_range)
    case_half_height_mm = max(
        case_half_height_mm, tray_floor_mm + MIN_CAVITY_RIM_CLEARANCE_MM)

    # A SECOND, independent coupling case_half_height_mm_range was never
    # calibrated against: it is anchored to the real, 18650-only measured
    # assemblies (11.1mm), predating Group A's 21700/26650 cell formats.
    # The "assembled" variant seals a cell inside the CASE+LID envelope,
    # whose total height is 2 * case_half_height_mm - and the cell rests
    # not at z=0 but at z=tray_floor_mm (the cavity floor), so its own
    # top is tray_floor_mm + diameter, not diameter alone. A 26650 cell
    # (26mm diameter) on a ~2mm floor does not fit inside a shell sized
    # 22.2mm tall (2 * 11.1), and this was found the hard way, twice:
    # world.build_procedural_tray's own numeric self-check (Task 8)
    # raised on the very first render, and the FIRST version of this fix
    # (diameter alone, forgetting the floor offset) still raised on the
    # second procedural tray built. Nudging case_half_height_mm UP (never
    # shrinking the drawn floor or cell) keeps every format physically
    # sealable; for 18650 (the format the anchored range was calibrated
    # for) this is a no-op in practice - 2*11.1=22.2mm already clears a
    # ~2mm floor plus an 18.3mm cell (~20.3mm) with room to spare.
    case_half_height_mm = max(
        case_half_height_mm,
        (tray_floor_mm + cell_w_mm + MIN_CELL_HEIGHT_CLEARANCE_MM) / 2.0)

    # DRAWN LAST, deliberately and load-bearingly. The crowned procedural
    # set built for the sealed-unit experiment must differ from `anchored`
    # in the crown and NOTHING else; catalog.build_procedural_pool gives
    # each tray its own seeded Random, so taking this draw after every
    # other one leaves the whole preceding stream bit-identical between a
    # (0, 0) config and a (0, 12) one. Move it earlier and every
    # downstream tray parameter silently resamples, turning a one-variable
    # experiment into a two-variable one with no error anywhere - pinned
    # by test_lid_crown_is_drawn_LAST_so_every_other_tray_field_is_identical.
    #
    # Clamped rather than range-restricted, for the same reason
    # case_half_height_mm is: the two limits depend on quantities drawn
    # ABOVE (the lid's own height and its outer footprint), so no static
    # range can express them. A crown deeper than the lid would roll past
    # its own base; one wider than half the shortest side self-intersects.
    lid_crown_mm = rng.uniform(*cfg.lid_crown_mm_range)
    ox0, oy0, ox1, oy1 = case_outer
    lid_crown_mm = min(lid_crown_mm, case_half_height_mm,
                       MAX_LID_CROWN_FOOTPRINT_FRAC
                       * min(ox1 - ox0, oy1 - oy0))

    return TraySample(
        interior_mm=interior, module_bay_mm=module_bay, case_outer_mm=case_outer,
        wall_mm=wall_mm, case_half_height_mm=case_half_height_mm,
        tray_floor_mm=tray_floor_mm,
        cell_format=cell_format, bay_edge=edge,
        lid_crown_mm=lid_crown_mm)


def seated_cell_world_poses(poses: List[Tuple[float, float, float]],
                            rot_deg: float, translate: Tuple[float, float]
                            ) -> List[Tuple[float, float, float]]:
    """Carry LOCAL `seated_cell_poses` output into world space.

    Same contract and same reasoning as `obstruction_world_poses`: each
    (x, y) is a POINT in the cartridge's own local frame (the frame
    `placement_rect_local` returns, which is what `seated_cell_poses`'s
    `placement_rect` argument must be called against), so rotating that
    point about the local origin by `rot_deg` and translating by `translate`
    is exact for ANY angle - a point has no extent for the rotation to
    inflate, unlike lerping into a rotated world AABB. `rot_deg` and
    `translate` are the SAME `layout.Placement.rot_deg` and `(x, y)` the
    cartridge, the module board, the bay proxy and any obstructions were
    placed with, so a seated cell turns WITH its cartridge: it stays
    axis-aligned to the BAY it was packed into rather than to the world,
    instead of staying axis-aligned to the world while the bay rotates out
    from under it.

    Each cell's own pitch orientation from the packer (0 or 90) is added to
    the cartridge's `rot_deg`, the same composition
    `obstruction_world_poses` uses for a tape cross's own tilt.
    """
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    tx, ty = translate
    out = []
    for x, y, r in poses:
        wx = tx + x * cos_t - y * sin_t
        wy = ty + x * sin_t + y * cos_t
        out.append((wx, wy, (r + rot_deg) % 360.0))
    return out
