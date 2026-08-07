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
import random
from dataclasses import dataclass
from typing import List, Tuple

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


def placement_world_placement(footprint: Tuple[float, float], bay_mm: Rect,
                              interior_mm: Rect, rot_deg: float,
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
    lx0, ly0, lx1, ly1 = placement_rect_local(footprint, bay_mm, interior_mm)
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
# =========================================================================== #

def seated_cell_poses(placement_rect: Rect, cell_w: float, cell_h: float,
                      n: int, rng: random.Random) -> List[Tuple[float, float, float]]:
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
    arrangement. No `forbidden_mask` is passed, so the packer's obstacle-
    advancing behaviour never engages here - every shelf is a clean strip of
    `placement_rect`.

    Returns `[(x, y, rot_deg), ...]` with `rot_deg` in {0, 90} - the cell's
    OWN pitch orientation from the packer, in the same LOCAL frame as
    `placement_rect`. Fewer than `n` come back when the bay cannot hold that
    many; `first_fit_decreasing` reports the rest as unplaced rather than
    overlapping them.
    """
    from common.packing import Item as _PackItem
    from common.packing import first_fit_decreasing

    if n <= 0:
        return []

    x0, y0, x1, y1 = placement_rect
    strip_w, strip_h = x1 - x0, y1 - y0

    items = [_PackItem(i, cell_w, cell_h) for i in range(n)]
    res = first_fit_decreasing(items, strip_w, strip_h, allow_rotation=True)

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
