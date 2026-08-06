"""
recog.synth3d.layout - axis-aligned placement with a bounded overlap.

No bpy import: this is pure geometry and is unit-tested outside Blender.

Rotations are restricted to k*90 plus a small jitter. That is not just a
stylistic constraint - because every footprint stays axis-aligned, the overlap
test is an exact AABB comparison rather than an approximation, so how much two
parts may share is a guarantee rather than a hope.

`LayoutCfg.max_overlap_iou` is that bound, in padded-footprint IoU. At its
default of 0 the solvers reproduce the exact non-overlap they always had.
Above it, `plan` lets parts touch and occlude and lifts the upper one onto the
lower via `Placement.z`, because two solids resting on the same floor and
sharing ground area would render as interpenetrating rather than as one lying
across the other.

Two modes:
  plan()      free scatter, for domain randomization
  plan_jig()  shelf-packed pockets, reproducing the real 3-D-printed fixture

plan_jig derives pockets FROM the parts rather than fitting parts into a fixed
grid, so a pocket always fits its part by construction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from common.packing import Item as _PackItem
from common.packing import first_fit_decreasing

MM = 1000.0        # common.packing documents its units as millimetres


@dataclass
class Placement:
    x: float
    y: float
    quarter: int          # 0..3, the k in k*90 degrees
    rot_deg: float        # quarter*90 + jitter
    # Metres above the floor to rest on. 0 for every non-overlapping placement,
    # which is all of them until `max_overlap_iou` is raised. Two parts that
    # share ground area cannot both rest on z = 0: they would occupy the same
    # space and render as interpenetrating solids, which is not what a part
    # lying across another looks like. `assets.place_item` drops the item to
    # the floor and then adds this.
    z: float = 0.0

    def as_dict(self):
        return {"x": round(self.x, 5), "y": round(self.y, 5),
                "quarter": self.quarter, "rot_deg": round(self.rot_deg, 3),
                "z": round(self.z, 5)}


@dataclass(frozen=True)
class Pocket:
    """A recess in the jig plate. Centre-based, metres, layout-local."""
    x: float
    y: float
    w: float
    h: float
    depth: float

    def as_dict(self):
        return {"x": round(self.x, 5), "y": round(self.y, 5),
                "w": round(self.w, 5), "h": round(self.h, 5),
                "depth": round(self.depth, 5)}


def footprint_after_rotation(fx: float, fy: float, quarter: int):
    """90 and 270 degree turns swap the extents."""
    return (fx, fy) if quarter % 2 == 0 else (fy, fx)


def padded_iou(x1, y1, hx1, hy1, x2, y2, hx2, hy2) -> float:
    """
    IoU of two centred, axis-aligned, half-extent boxes.

    The half-extents callers pass in already include `cfg.pad / 2` on each
    side, so this is the IoU of the padded FOOTPRINTS, not of the parts. That
    matters when reading `max_overlap_iou`: `pad` is 8 mm, so for an 18650
    (65.0 x 18.3 mm, padded to 73.0 x 26.3) a padded IoU of about 0.16 is
    where the raw cells stop having a gap and start touching, and real
    interpenetration only begins above it. Defining the threshold on the
    padded box is what makes 0.0 reproduce the old strict-disjointness rule
    exactly: `iou > 0` is true precisely when the padded AABBs intersect,
    which was the old test.
    """
    ix = min(x1 + hx1, x2 + hx2) - max(x1 - hx1, x2 - hx2)
    iy = min(y1 + hy1, y2 + hy2) - max(y1 - hy1, y2 - hy2)
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    inter = ix * iy
    return inter / (4 * hx1 * hy1 + 4 * hx2 * hy2 - inter)


def plan(footprints: Sequence[Tuple[float, float]], cfg, rng: random.Random,
         heights: Optional[Sequence[float]] = None,
         max_overlap_iou: Optional[float] = None
         ) -> List[Optional[Placement]]:
    """
    footprints: (size_x, size_y) per item at zero rotation, in metres.
    Returns a Placement per item, or None where the item would not fit.
    Large items are placed first, which materially improves packing.

    `max_overlap_iou` is the largest padded-footprint IoU a candidate may have
    with any already-placed item; None reads `cfg.max_overlap_iou`. At 0 a
    candidate is rejected as soon as it intersects anything at all, which is
    byte-for-byte the rule this solver has always used - the same rng draws,
    the same accepts, the same rejects.

    Above 0, parts may touch and occlude. That is the point: real trays hold
    cells shoulder to shoulder, and while the layout guaranteed exact
    non-overlap no labelled object could occlude another, which made
    `filter.min_visibility` a threshold with nothing to threshold and made the
    synthetic validation metric saturate at mAP 1.0 within a few epochs.

    `heights` (metres, per item) is what keeps overlap physical. An item that
    shares ground area with one already placed gets `Placement.z` set to the
    top of the tallest thing it sits on, so it lies ACROSS it rather than
    through it. Omit `heights` and every lift is 0 - correct as long as
    `max_overlap_iou` is 0 too, which is the default.
    """
    W, H = cfg.area
    pad = cfg.pad
    jit = cfg.jitter_deg
    quarters = [0, 1, 2, 3] if cfg.allow_90s else [0]
    max_iou = (getattr(cfg, "max_overlap_iou", 0.0)
               if max_overlap_iou is None else max_overlap_iou) or 0.0

    result: List[Optional[Placement]] = [None] * len(footprints)
    order = sorted(range(len(footprints)), key=lambda i: -max(footprints[i]))

    if cfg.mode == "grid":
        cols = max(1, math.ceil(math.sqrt(len(footprints))))
        rows = max(1, math.ceil(len(footprints) / cols))
        cw, ch = W / cols, H / rows
        for n, i in enumerate(order):
            q = rng.choice(quarters)
            result[i] = Placement(
                x=-W / 2 + cw * (n % cols + 0.5) + rng.uniform(-cw * .1, cw * .1),
                y=H / 2 - ch * (n // cols + 0.5) + rng.uniform(-ch * .1, ch * .1),
                quarter=q, rot_deg=q * 90 + rng.uniform(-jit, jit))
        return result

    # (x, y, half_x, half_y, top_z) per accepted placement.
    placed: List[Tuple[float, float, float, float, float]] = []
    for i in order:
        fx, fy = footprints[i]
        hgt = float(heights[i]) if heights is not None else 0.0
        for _ in range(cfg.max_tries):
            q = rng.choice(quarters)
            ex, ey = footprint_after_rotation(fx, fy, q)
            hx, hy = ex / 2 + pad / 2, ey / 2 + pad / 2
            if hx * 2 > W or hy * 2 > H:
                break                      # cannot fit in the area at all
            x = rng.uniform(-W / 2 + hx, W / 2 - hx)
            y = rng.uniform(-H / 2 + hy, H / 2 - hy)
            lift, rejected = 0.0, False
            for px, py, phx, phy, ptop in placed:
                v = padded_iou(x, y, hx, hy, px, py, phx, phy)
                if v > max_iou:
                    rejected = True
                    break
                if v > 0.0:
                    lift = max(lift, ptop)
            if rejected:
                continue
            placed.append((x, y, hx, hy, lift + hgt))
            result[i] = Placement(x=x, y=y, quarter=q,
                                  rot_deg=q * 90 + rng.uniform(-jit, jit),
                                  z=lift)
            break
    return result


def plan_jig(footprints: Sequence[Tuple[float, float]], cfg,
             rng: random.Random
             ) -> Tuple[List[Optional[Placement]], List[Pocket]]:
    """
    Shelf-pack footprints, then emit one Pocket around each placed part.

    Reuses the FFDH packer the planner uses for batteries-into-cartridges;
    a jig plate is the same problem. Deriving pockets from the packed parts
    guarantees fit, unlike packing parts into a fixed pocket grid.

    Returns (placements, pockets). pockets is parallel to the non-None
    placements in input order.
    """
    W, H = cfg.area
    clear = cfg.jig_clearance
    jit = cfg.jig_jitter_deg
    wall = cfg.jig_wall

    # Inflate by clearance plus wall on all sides before packing, so packed
    # cells tile without touching. The wall margin is trimmed back off the
    # pocket below, leaving plate material between adjacent pockets - without
    # it, packed cells (and therefore pockets) would abut with zero material,
    # and Task 7's per-pocket boolean cutters would merge into one trough
    # instead of leaving walled recesses. Convert to millimetres for
    # common.packing.
    items = [_PackItem(i, (fx + 2 * clear + wall) * MM, (fy + 2 * clear + wall) * MM)
             for i, (fx, fy) in enumerate(footprints)]
    res = first_fit_decreasing(items, W * MM, H * MM,
                               allow_rotation=cfg.allow_90s)

    placements: List[Optional[Placement]] = [None] * len(footprints)
    pockets_by_index = {}

    for packed in res.placements:
        i = packed.item.id
        # PackedItem.x/.y is the top-left corner in mm, y growing downward.
        # Convert to the centred, y-up frame the rest of the pipeline uses.
        w_m = packed.width / MM
        h_m = packed.height / MM
        cx = -W / 2 + (packed.x / MM) + w_m / 2
        cy = H / 2 - (packed.y / MM) - h_m / 2

        quarter = 1 if packed.rotated else 0
        placements[i] = Placement(
            x=cx, y=cy, quarter=quarter,
            rot_deg=quarter * 90 + rng.uniform(-jit, jit))
        pockets_by_index[i] = Pocket(
            x=cx, y=cy, w=w_m - wall, h=h_m - wall,
            depth=rng.uniform(*cfg.jig_depth))

    if pockets_by_index:
        dx, dy = _jig_centering_offset(pockets_by_index, W, H, rng)
        for i, pk in pockets_by_index.items():
            pockets_by_index[i] = Pocket(x=pk.x + dx, y=pk.y + dy,
                                         w=pk.w, h=pk.h, depth=pk.depth)
            placements[i].x += dx
            placements[i].y += dy

    pockets = [pockets_by_index[i] for i in sorted(pockets_by_index)]
    return placements, pockets


def _jig_centering_offset(pockets_by_index, W: float, H: float,
                          rng: random.Random) -> Tuple[float, float]:
    """
    FFDH shelf-packs from the top-left corner, so an unmodified pack always
    sits pinned in the top-left of the plate - a real fixture's parts are
    spread across it. Compute the translation that recentres the packed
    block's bounding box on the plate, plus a per-scene jitter so different
    scenes do not all land on exactly the same offset.

    The jitter is drawn from, and clamped to, the slack between the centred
    block and the plate edges, so a jittered pocket can still never cross the
    plate boundary - the "pockets stay inside the area" guarantee holds for
    any jitter draw, not just in expectation.
    """
    xs0 = [pk.x - pk.w / 2 for pk in pockets_by_index.values()]
    xs1 = [pk.x + pk.w / 2 for pk in pockets_by_index.values()]
    ys0 = [pk.y - pk.h / 2 for pk in pockets_by_index.values()]
    ys1 = [pk.y + pk.h / 2 for pk in pockets_by_index.values()]
    lo_x, hi_x = min(xs0), max(xs1)
    lo_y, hi_y = min(ys0), max(ys1)
    bbox_cx, bbox_cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2

    # Half the room left over once the block is centred: how far the centred
    # block can still move before a pocket would cross the plate edge.
    slack_x = max(0.0, (W - (hi_x - lo_x)) / 2)
    slack_y = max(0.0, (H - (hi_y - lo_y)) / 2)
    jitter_x = min(slack_x, max(-slack_x, rng.uniform(-slack_x, slack_x)))
    jitter_y = min(slack_y, max(-slack_y, rng.uniform(-slack_y, slack_y)))

    return -bbox_cx + jitter_x, -bbox_cy + jitter_y
