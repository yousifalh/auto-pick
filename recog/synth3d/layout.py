"""
recog.synth3d.layout - axis-aligned placement with guaranteed non-overlap.

No bpy import: this is pure geometry and is unit-tested outside Blender.

Rotations are restricted to k*90 plus a small jitter. That is not just a
stylistic constraint - because every footprint stays axis-aligned, the overlap
test is an exact AABB comparison rather than an approximation, so "no two parts
intersect" is a guarantee rather than a hope.

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

    def as_dict(self):
        return {"x": round(self.x, 5), "y": round(self.y, 5),
                "quarter": self.quarter, "rot_deg": round(self.rot_deg, 3)}


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


def plan(footprints: Sequence[Tuple[float, float]], cfg, rng: random.Random
         ) -> List[Optional[Placement]]:
    """
    footprints: (size_x, size_y) per item at zero rotation, in metres.
    Returns a Placement per item, or None where the item would not fit.
    Large items are placed first, which materially improves packing.
    """
    W, H = cfg.area
    pad = cfg.pad
    jit = cfg.jitter_deg
    quarters = [0, 1, 2, 3] if cfg.allow_90s else [0]

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

    placed: List[Tuple[float, float, float, float]] = []
    for i in order:
        fx, fy = footprints[i]
        for _ in range(cfg.max_tries):
            q = rng.choice(quarters)
            ex, ey = footprint_after_rotation(fx, fy, q)
            hx, hy = ex / 2 + pad / 2, ey / 2 + pad / 2
            if hx * 2 > W or hy * 2 > H:
                break                      # cannot fit in the area at all
            x = rng.uniform(-W / 2 + hx, W / 2 - hx)
            y = rng.uniform(-H / 2 + hy, H / 2 - hy)
            if any(abs(x - px) < (hx + phx) and abs(y - py) < (hy + phy)
                   for px, py, phx, phy in placed):
                continue
            placed.append((x, y, hx, hy))
            result[i] = Placement(x=x, y=y, quarter=q,
                                  rot_deg=q * 90 + rng.uniform(-jit, jit))
            break
    return result


def cluster_offsets(n: int, spread: float, rng: random.Random
                    ) -> List[Tuple[float, float]]:
    """
    Offsets for the sub-parts of an exploded assembly: a loose ring so cells
    land near their case rather than scattered across the whole frame.
    """
    if n <= 0:
        return []
    out = []
    base = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        ang = base + 2 * math.pi * k / n + rng.uniform(-0.25, 0.25)
        r = spread * rng.uniform(0.6, 1.25)
        out.append((r * math.cos(ang), r * math.sin(ang)))
    return out


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

    pockets = [pockets_by_index[i] for i in sorted(pockets_by_index)]
    return placements, pockets


def rows_in_pocket(n: int, cell_fp: Tuple[float, float], pocket: Pocket,
                   rng: random.Random) -> List[Placement]:
    """
    Lay n identical cells in regular rows inside one pocket.

    Reproduces the top rows of the real photos, where seven cells sit in a
    single tray recess rather than one recess each. Returns at most n
    placements - fewer if the pocket cannot hold them all.
    """
    fx, fy = cell_fp
    cols = max(1, int(pocket.w // fx))
    rows = max(1, int(pocket.h // fy))
    out: List[Placement] = []
    for k in range(min(n, cols * rows)):
        r, c = divmod(k, cols)
        out.append(Placement(
            x=pocket.x - pocket.w / 2 + fx * (c + 0.5),
            y=pocket.y + pocket.h / 2 - fy * (r + 0.5),
            quarter=0,
            rot_deg=rng.uniform(-0.5, 0.5)))
    return out
