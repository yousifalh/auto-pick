"""Bay geometry — pure functions, no bpy, no Blender."""
from __future__ import annotations

import json
import math
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


from recog.synth3d.bay import (module_rect_local, module_world_placement,
                               placement_rect_local, placement_world_placement)

# `footprint` for these is `assets.Item.footprint`: the cartridge's own
# UN-ROTATED (size_x, size_y) in metres - not a world AABB. See
# `module_world_placement`'s docstring for why that distinction is load
# bearing: a world AABB is inflated by rotation, an un-rotated footprint is
# not.


def test_module_rect_local_anchors_to_the_plus_y_side_and_scales():
    # Catalog: interior 0..60 x 0..90 mm, bay is the +y strip 66..90.
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)         # same proportions, 1000x smaller
    x0, y0, x1, y1 = module_rect_local(footprint, bay_mm, interior_mm)
    assert (x0, x1) == pytest.approx((-0.030, 0.030))       # full width
    assert (y0, y1) == pytest.approx((0.021, 0.045))        # +y strip


def test_module_rect_local_anchors_to_the_minus_x_side():
    interior_mm = (0.0, 0.0, 90.0, 60.0)
    bay_mm = (0.0, 0.0, 24.0, 60.0)
    footprint = (0.090, 0.060)
    x0, y0, x1, y1 = module_rect_local(footprint, bay_mm, interior_mm)
    assert (x0, x1) == pytest.approx((-0.045, -0.021))
    assert (y0, y1) == pytest.approx((-0.030, 0.030))


def test_module_rect_local_is_a_strict_subset_of_the_local_footprint():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    mx0, my0, mx1, my1 = module_rect_local(footprint, bay_mm, interior_mm)
    fw, fh = footprint
    fx0, fy0, fx1, fy1 = -fw / 2, -fh / 2, fw / 2, fh / 2
    assert fx0 - 1e-9 <= mx0 < mx1 <= fx1 + 1e-9
    assert fy0 - 1e-9 <= my0 < my1 <= fy1 + 1e-9


def test_placement_rect_local_is_the_complement_of_the_module_rect():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)

    m = module_rect_local(footprint, bay_mm, interior_mm)
    p = placement_rect_local(footprint, bay_mm, interior_mm)

    # Disjoint, adjacent, and together they tile the footprint.
    assert p[3] == pytest.approx(m[1])
    fw, fh = footprint
    assert (p[0], p[1], p[2]) == pytest.approx((-fw / 2, -fh / 2, fw / 2))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    assert area(m) + area(p) == pytest.approx(fw * fh)


# ---- module_world_placement: `layout.plan` rotates by quarter*90 + a few
# degrees of jitter, and a naive lerp into the ROTATED case's world AABB
# (what `module_rect_local`'s predecessor did) inflates the module, because
# the AABB of a rotated rectangle is larger than the rectangle. These pin
# down that the fix - rotating the LOCAL centre as a point, which has no
# extent to inflate - keeps the module's true size for ANY rotation angle,
# not just the k*90 multiples the old quarter-only fix handled.

def test_module_world_placement_size_is_invariant_to_rotation_angle():
    # AnkerPowerCore26800's real numbers: 81.7mm-wide interior, a 35.0mm
    # bay on the +y end - the case measured on the contact sheet as
    # overhanging by 3.3% at ~2 degrees of jitter under the old lerp.
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 91.7, 137.0, 180.0, 270.0, 358.0):
        _, _, w, h = module_world_placement(
            footprint, bay_mm, interior_mm, rot_deg, (0.0, 0.0))
        assert w == pytest.approx(0.0817, abs=1e-9)
        assert h == pytest.approx(0.035, abs=1e-9)


def test_module_world_placement_matches_the_local_case_at_zero_rotation():
    footprint = (0.060, 0.090)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    # A cartridge placed with its centre at (0.130, 0.245) sits at world
    # AABB 0.100..0.160 x 0.200..0.290 - the same case the un-rotated
    # module_rect tests above use.
    cx, cy, w, h = module_world_placement(
        footprint, bay_mm, interior_mm, 0.0, (0.130, 0.245))
    assert (w, h) == pytest.approx((0.060, 0.024))
    assert (cx, cy) == pytest.approx((0.130, 0.278))


def test_module_world_placement_centre_rotates_about_the_translate_point():
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    cx, cy, w, h = module_world_placement(
        footprint, bay_mm, interior_mm, 90.0, (0.0, 0.0))
    # Local centre is (0, 0.0725); a +90 degree turn sends +y to -x.
    assert (cx, cy) == pytest.approx((-0.0725, 0.0), abs=1e-6)
    assert (w, h) == pytest.approx((0.0817, 0.035))


# ---- proxy geometry: the placement_area label is the complementary
# rectangle to the module, carried through the same local-frame ->
# rotate-the-centre-point pipeline `module_world_placement` uses. See that
# function's docstring for why a rotated world AABB is not an option here
# either: the same 7.6%-at-2-degrees inflation would oversize the proxy and
# claim placement space the case does not physically have.

def test_module_and_placement_local_rects_do_not_overlap():
    """The proxy must not be drawn under the module, or the two labels
    would fight for the same pixels and the winner would be z-order."""
    interior_mm = (0.0, 0.0, 62.9, 90.9)
    bay_mm = (0.0, 67.4, 62.9, 90.9)
    footprint = (0.0629, 0.0909)

    m = module_rect_local(footprint, bay_mm, interior_mm)
    p = placement_rect_local(footprint, bay_mm, interior_mm)

    ox = min(m[2], p[2]) - max(m[0], p[0])
    oy = min(m[3], p[3]) - max(m[1], p[1])
    assert ox <= 1e-9 or oy <= 1e-9, "module and placement rects overlap"


def test_placement_world_placement_size_is_invariant_to_rotation_angle():
    # Same fixture as module_world_placement's rotation-invariance test:
    # AnkerPowerCore26800, 81.7mm interior, a 35.0mm bay on the +y end.
    # The complement is the 90 - 35 = 55.0mm strip on the -y side.
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 91.7, 137.0, 180.0, 270.0, 358.0):
        _, _, w, h = placement_world_placement(
            footprint, bay_mm, interior_mm, rot_deg, (0.0, 0.0))
        assert w == pytest.approx(0.0817, abs=1e-9)
        assert h == pytest.approx(0.145, abs=1e-9)


def test_placement_world_placement_matches_the_local_case_at_zero_rotation():
    footprint = (0.060, 0.090)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    cx, cy, w, h = placement_world_placement(
        footprint, bay_mm, interior_mm, 0.0, (0.130, 0.245))
    assert (w, h) == pytest.approx((0.060, 0.066))
    assert (cx, cy) == pytest.approx((0.130, 0.233))


def test_placement_world_placement_centre_rotates_about_the_translate_point():
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    cx, cy, w, h = placement_world_placement(
        footprint, bay_mm, interior_mm, 90.0, (0.0, 0.0))
    # placement_rect_local's centre here is (0, -0.0175); a +90 degree turn
    # sends +y to -x, so (lcx, lcy) = (0, -0.0175) -> (0.0175, 0).
    assert (cx, cy) == pytest.approx((0.0175, 0.0), abs=1e-6)
    assert (w, h) == pytest.approx((0.0817, 0.145))


def test_placement_and_module_world_rects_do_not_overlap_after_rotation():
    """A rigid rotate-then-translate applied identically to two disjoint
    local rectangles cannot make them overlap; this pins that down at the
    world-placement level, not just the local-frame level."""
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 137.0, 358.0):
        mcx, mcy, mw, mh = module_world_placement(
            footprint, bay_mm, interior_mm, rot_deg, (0.3, 0.2))
        pcx, pcy, pw, ph = placement_world_placement(
            footprint, bay_mm, interior_mm, rot_deg, (0.3, 0.2))
        # Both rectangles keep their true, un-rotated size; the join across
        # their shared edge is invariant to the rigid transform, so the
        # distance between centres along the local bay axis must equal
        # half of each rect's extent on that axis, for any rotation.
        dist = math.hypot(mcx - pcx, mcy - pcy)
        assert dist == pytest.approx((mh + ph) / 2 if
                                     abs(mw - pw) < 1e-9 else (mw + pw) / 2)
