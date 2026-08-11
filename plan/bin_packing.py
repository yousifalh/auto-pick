"""2-D orthogonal strip-packing for the planner.

The strip is the cartridge's placement rectangle; items are identical
battery footprints (each cartridge type permits at most two rotations,
0° or 90°). PPR §5.3.3 selected shelf-based FFDH:

1. Sort items by decreasing height.
2. Place each on the leftmost "shelf" that can still take it (first-fit).
3. If no shelf works, open a new shelf above the last one.

Deterministic, 1.7 × OPT worst case, O(n log n) — well under the
50 ms / 8 ms budgets in the PPR (Berkey & Wang 1987; Martello, Pisinger
& Toth 2000).

**The planner no longer runs FFDH alone.** Shelves span the full strip
width and their origins never scan in y, so a forbidden region that
crosses the strip anywhere inside the first shelf's row band kills the
whole pack: measured on frame ``scene_00005``, zero cells placed on a
93 %-free grid containing a clear 48 × 112 mm rectangle. Across 30 real
cartridge instances FFDH placed 8 cells where 18 were demonstrably
achievable. :func:`pack_cartridge` therefore calls
:func:`common.packing.pack_best_effort`, which competes FFDH against two
obstacle-tolerant arms and returns whichever placed most — never fewer
than FFDH alone. See ``docs/superpowers/specs/
2026-08-11-packing-ceiling.md`` for the diagnosis and the measurements.

:func:`first_fit_decreasing` itself is unchanged and still exported; the
synthetic-scene generators in :mod:`recog.synth3d` depend on its exact
output and are deliberately NOT switched.

The algorithms live in :mod:`common.packing` so that
:mod:`recog.synth3d.layout` can use them too without creating a back-edge
from ``recog`` into ``plan``; this module re-exports them for existing
callers.
"""
from __future__ import annotations

from common.packing import (  # noqa: F401  (re-exported for existing callers)
    Item,
    PackedItem,
    PackResult,
    _overlaps_forbidden,
    _try_place_item,
    first_fit_decreasing,
    pack_best_effort,
)


# ---------------------------------- adapter for the digital twin ----

def pack_cartridge(
    cartridge,
    battery_width_mm: float,
    battery_length_mm: float,
    allow_rotation: bool = True,
    mm_per_px: float = 0.38,
) -> PackResult:
    """Build a packing instance for ``cartridge`` and solve it.

    The cartridge's placement rectangle (in pixels) is converted to a
    strip in millimetres. The forbidden mask is derived from the
    cartridge's occupancy grid, unioning FORBIDDEN / PLACED / PLANNED
    cells so already-assigned positions aren't packed over.

    Solved with :func:`common.packing.pack_best_effort` rather than FFDH
    directly — see the module docstring.
    """
    pr = cartridge.placeable_rectangle
    if pr is None:
        raise ValueError(
            "cartridge.placeable_rectangle is None — "
            "extract placement area first"
        )

    strip_w_mm = pr.width * mm_per_px
    strip_h_mm = pr.height * mm_per_px

    # Build an upper-bound number of candidate items. We over-estimate
    # so FFDH has enough identical items to saturate the strip.
    n_max_est = max(
        4,
        int((strip_w_mm * strip_h_mm)
            / (battery_width_mm * battery_length_mm)) * 2,
    )
    items = [
        Item(id=i, width=battery_width_mm, height=battery_length_mm)
        for i in range(n_max_est)
    ]

    forbidden = None
    mm_per_cell = 1.5
    if cartridge.occupancy is not None:
        from plan.scene import CellState

        forbidden = cartridge.occupancy.mask_of(
            CellState.FORBIDDEN, CellState.PLACED, CellState.PLANNED,
        )
        mm_per_cell = cartridge.occupancy.resolution_mm

    return pack_best_effort(
        items, strip_w_mm, strip_h_mm,
        allow_rotation=allow_rotation,
        forbidden_mask=forbidden,
        mm_per_cell=mm_per_cell,
    )


__all__ = [
    "Item",
    "PackedItem",
    "PackResult",
    "first_fit_decreasing",
    "pack_best_effort",
    "pack_cartridge",
]
