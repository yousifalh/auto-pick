"""
recog.synth3d.bay - every geometric decision about a cartridge's interior.

This module has no Blender dependencies, so all of it is unit-testable outside
Blender. world.py and scene.py depend on Blender and cannot be tested; they
call into here for the numbers and only apply the result. Keeping the arithmetic
on this side of the line is what makes the bay geometry checkable at all.

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
