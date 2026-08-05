"""Shelf-based First-Fit Decreasing Height (FFDH) bin-packing.

This implements the 2-D orthogonal strip-packing variant selected in
PPR §5.3.3. The strip is the cartridge's placement rectangle; items
are identical battery footprints (each cartridge type permits at most
two rotations, 0° or 90°). The algorithm sorts items by decreasing
height, then for each item:

1. Tries the leftmost "shelf" that can still accommodate it (first-fit).
2. If no shelf works, opens a new shelf above the last one.

Properties (Berkey & Wang 1987; Martello, Pisinger & Toth 2000):

* Deterministic and reproducible across runs.
* Worst-case packing ratio: 1.7 × OPT.
* Runs in O(n log n) — well under the 50 ms / 8 ms budgets in the PPR.

The module also exposes a convenience adapter,
:func:`pack_cartridge`, that pulls the strip geometry and forbidden
mask directly off a :class:`plan.scene.Cartridge` and invokes FFDH.

The FFDH algorithm itself now lives in :mod:`common.packing` so that
:mod:`recog.synth3d.layout` can use it too without creating a back-edge
from ``recog`` into ``plan``; this module re-exports it for existing
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
)


# ---------------------------------- adapter for the digital twin ----

def pack_cartridge(
    cartridge,
    battery_width_mm: float,
    battery_length_mm: float,
    allow_rotation: bool = True,
    mm_per_px: float = 0.38,
) -> PackResult:
    """Build an FFDH instance for ``cartridge`` and run it.

    The cartridge's placement rectangle (in pixels) is converted to a
    strip in millimetres. The forbidden mask is derived from the
    cartridge's occupancy grid, unioning FORBIDDEN / PLACED / PLANNED
    cells so already-assigned positions aren't packed over.
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

    return first_fit_decreasing(
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
    "pack_cartridge",
]
