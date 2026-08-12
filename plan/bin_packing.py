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
achievable. :meth:`plan.planner.Planner._pack_cartridge` therefore calls
:func:`common.packing.pack_best_effort`, which competes FFDH against two
obstacle-tolerant arms and returns whichever placed most — never fewer
than FFDH alone. See ``docs/superpowers/specs/
2026-08-11-packing-ceiling.md`` for the diagnosis and the measurements.

This module used to also carry a ``pack_cartridge(cartridge, ...)``
adapter. It was dead — nothing but a test named it — and it was a weaker
twin of the live ``Planner._pack_cartridge``: it took
``mm_per_px: float = 0.38``, re-arming the exact placeholder constant
whose removal is recorded in ``docs/superpowers/specs/
2026-08-11-scale-calibration.md`` as having under-read 24 of 30
cartridges by 27 % at the median and produced 3 unsafe placements. The
live path resolves scale through ``_resolve_scale``, which raises
``UnknownScale`` rather than guessing. Exporting the guessing version in
``__all__`` made it read as the supported API. Deleted 2026-08-12.

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


__all__ = [
    "Item",
    "PackedItem",
    "PackResult",
    "first_fit_decreasing",
    "pack_best_effort",
]
