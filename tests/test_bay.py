"""Bay geometry — pure functions, no bpy, no Blender."""
from __future__ import annotations

import json
import os

import pytest

from recog.synth3d.bay import module_bay_from_bounds

ASSETS = os.path.join(os.path.dirname(__file__), "..",
                      "recog", "synth3d", "assets")


def test_module_bay_picks_the_largest_gap_side():
    # Interior 0..60 x 0..90; cells fill 4..56 x 4..66.
    # Gaps: -x 4, +x 4, -y 4, +y 24. The +y gap wins.
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 66.0, 60.0, 90.0))


def test_module_bay_spans_the_full_interior_width():
    """The module runs wall to wall across the short side."""
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    x0, y0, x1, y1 = module_bay_from_bounds(interior, cells)
    assert (x0, x1) == (0.0, 60.0)


def test_module_bay_handles_the_gap_on_the_minus_y_side():
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 24.0, 56.0, 86.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 0.0, 60.0, 24.0))


def test_module_bay_handles_the_gap_on_the_minus_x_side():
    # Gaps: -x 24, +x 4, -y 4, +y 4. The -x gap wins.
    interior = (0.0, 0.0, 90.0, 60.0)
    cells = (24.0, 4.0, 86.0, 56.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((0.0, 0.0, 24.0, 60.0))


def test_module_bay_handles_the_gap_on_the_plus_x_side():
    # Interior 0..90 x 0..60; cells fill 4..66 x 4..56.
    # Gaps: -x 4, +x 24, -y 4, +y 4. The +x gap wins, so the bay is the
    # strip from the cells' far edge (x=66) to the interior's far wall
    # (x=90), spanning the full y range (0..60) because the module runs
    # wall to wall across the short side.
    interior = (0.0, 0.0, 90.0, 60.0)
    cells = (4.0, 4.0, 66.0, 56.0)
    bay = module_bay_from_bounds(interior, cells)
    assert bay == pytest.approx((66.0, 0.0, 90.0, 60.0))


def test_module_bay_rejects_an_ambiguous_tie():
    # Cells exactly centred: all four gaps equal 4. No side is unambiguously
    # the bay, so this must raise rather than silently pick one by dict
    # order.
    interior = (0.0, 0.0, 60.0, 60.0)
    cells = (4.0, 4.0, 56.0, 56.0)
    with pytest.raises(ValueError):
        module_bay_from_bounds(interior, cells)


def test_module_bay_rejects_cells_outside_the_interior():
    # A bad upstream measurement puts the cell union outside the case
    # interior on the +x side; this must not silently produce a
    # negative-width rectangle.
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 66.0, 66.0)
    with pytest.raises(ValueError):
        module_bay_from_bounds(interior, cells)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,depth", [
    ("AnkerPowerCore10000", 23.5),
    ("AnkerPowerCore13000", 26.5),
    ("AnkerPowerCore20100", 28.9),
    ("AnkerPowerCore26800", 35.0),
])
def test_catalog_records_the_measured_bay_depth(name, depth):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert "module_bay_mm" in entry, "re-run: python -m recog.convert_cad"
    x0, y0, x1, y1 = entry["module_bay_mm"]
    assert max(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6) or \
           min(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6)
