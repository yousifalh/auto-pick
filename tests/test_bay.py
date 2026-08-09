"""Bay geometry — pure functions, no bpy, no Blender."""
from __future__ import annotations

import json
import math
import os

import pytest

from recog.synth3d.bay import case_wall_from_bounds, module_bay_from_bounds

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


# ---- case_wall_from_bounds: the amendment's new wall-thickness measurement.
# Fixture mirrors test_module_bay_picks_the_largest_gap_side's interior/cells
# (bay on +y, symmetric 4.0mm ±x walls) so the two functions are pinned
# against the SAME geometry.

def test_case_wall_from_bounds_averages_the_non_bay_axis_gaps():
    # Interior 0..60 x 0..90; cells 4..56 x 4..66. Bay is +y (24mm gap);
    # the non-bay (x) axis gaps are 4.0 and 4.0.
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    assert case_wall_from_bounds(interior, cells) == pytest.approx(4.0)


def test_case_wall_from_bounds_ignores_the_bay_axis_near_side():
    # Same as above but the near (-y) gap is a DIFFERENT figure (2.45mm,
    # like the real AnkerPowerCore10000) from the x walls (4.0mm) - the
    # result must still be 4.0, not an average involving 2.45.
    interior = (0.0, -2.45, 60.0, 87.55)
    cells = (4.0, 0.0, 56.0, 62.0)
    assert case_wall_from_bounds(interior, cells) == pytest.approx(4.0)


def test_case_wall_from_bounds_handles_a_bay_on_the_x_axis():
    # Gaps: -x 24, +x 4, -y 4, +y 4 (mirrors
    # test_module_bay_handles_the_gap_on_the_minus_x_side). Bay is -x, so
    # the wall is averaged from the y-axis gaps.
    interior = (0.0, 0.0, 90.0, 60.0)
    cells = (24.0, 4.0, 86.0, 56.0)
    assert case_wall_from_bounds(interior, cells) == pytest.approx(4.0)


def test_case_wall_from_bounds_matches_the_measured_anker_figures():
    # The four real assemblies, catalog.json's rounded cell_union_mm /
    # case_interior_mm (x, y only). Expected values from the amendment
    # brief: 4.0 / 3.8 / 3.7 / 4.2 mm.
    cases = [
        ((-31.45, -45.45, 31.45, 45.45), (-27.45, -43.0, 27.45, 22.0), 4.0),
        ((-40.35, -48.5, 40.35, 48.5), (-36.6, -43.0, 36.6, 22.0), 3.75),
        ((-31.15, -83.9, 31.15, 83.9), (-27.45, -78.0, 27.45, 55.0), 3.7),
        ((-40.85, -90.0, 40.85, 90.0), (-36.6, -85.0, 36.6, 55.0), 4.25),
    ]
    for interior, cells, expected in cases:
        assert case_wall_from_bounds(interior, cells) == \
            pytest.approx(expected, abs=0.01)


def test_case_wall_from_bounds_rejects_an_ambiguous_tie():
    interior = (0.0, 0.0, 60.0, 60.0)
    cells = (4.0, 4.0, 56.0, 56.0)
    with pytest.raises(ValueError):
        case_wall_from_bounds(interior, cells)


def test_case_wall_from_bounds_rejects_cells_outside_the_interior():
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 66.0, 66.0)
    with pytest.raises(ValueError):
        case_wall_from_bounds(interior, cells)


def test_case_wall_from_bounds_rejects_an_asymmetric_wall():
    # x walls of 4.0 and 10.0 disagree far beyond tessellation noise - the
    # asset would not have the uniform wall this function assumes.
    interior = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 50.0, 66.0)
    with pytest.raises(ValueError):
        case_wall_from_bounds(interior, cells)


# Task 2 (2026-08-09-tray-interior): module_bay_mm is now derived from
# interior_mm - the tray's outer AABB inset by case_wall_mm - rather than
# from the raw (un-inset) outer AABB the pre-task-2 catalog used. The bay's
# far edge used to run all the way to the tray's outer wall, i.e. it
# covered the wall tops too (see bay.py's module docstring and the task-2
# brief); insetting by the wall correctly shrinks the depth by exactly
# case_wall_mm per asset: 23.45-4.0=19.45, 26.5-3.75=22.75, 28.9-3.7=25.2,
# 35.0-4.25=30.75. These are the physically-correct depths, not a
# regression - the old figures described the bug this task fixes.
@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,depth", [
    ("AnkerPowerCore10000", 19.45),
    ("AnkerPowerCore13000", 22.75),
    ("AnkerPowerCore20100", 25.2),
    ("AnkerPowerCore26800", 30.75),
])
def test_catalog_records_the_measured_bay_depth(name, depth):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert "module_bay_mm" in entry, "re-run: python -m recog.convert_cad"
    x0, y0, x1, y1 = entry["module_bay_mm"]
    assert max(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6) or \
           min(x1 - x0, y1 - y0) == pytest.approx(depth, abs=0.6)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,wall", [
    ("AnkerPowerCore10000", 4.0),
    ("AnkerPowerCore13000", 3.8),
    ("AnkerPowerCore20100", 3.7),
    ("AnkerPowerCore26800", 4.2),
])
def test_catalog_records_the_measured_case_wall(name, wall):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert "case_wall_mm" in entry, "re-run: python -m recog.convert_cad"
    assert entry["case_wall_mm"] == pytest.approx(wall, abs=0.1)


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


# ---- wall_mm: the amendment's inset. Defaults to 0.0 everywhere (every
# test above calls these functions with no wall_mm and is therefore an
# unchanged regression pin for the pre-amendment behaviour); these tests
# cover the non-zero case directly.

def test_module_rect_local_wall_mm_zero_is_the_default_behaviour():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    assert module_rect_local(footprint, bay_mm, interior_mm, 0.0) == \
        pytest.approx(module_rect_local(footprint, bay_mm, interior_mm))


def test_module_rect_local_insets_by_wall_mm():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    x0, y0, x1, y1 = module_rect_local(footprint, bay_mm, interior_mm,
                                       wall_mm=3.0)
    assert (x0, x1) == pytest.approx((-0.027, 0.027))
    assert (y0, y1) == pytest.approx((0.0196, 0.042))


def test_placement_rect_local_insets_by_wall_mm():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    x0, y0, x1, y1 = placement_rect_local(footprint, bay_mm, interior_mm,
                                          wall_mm=3.0)
    assert (x0, x1) == pytest.approx((-0.027, 0.027))
    assert (y0, y1) == pytest.approx((-0.042, 0.0196))


def test_wall_mm_leaves_a_rim_around_all_four_sides_of_the_true_footprint():
    """The whole point of the amendment: module + placement no longer tile
    the cartridge's true physical footprint completely, so the case's own
    shell mesh shows through on every side by exactly wall_mm."""
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    wall_mm = 3.0
    wall_m = wall_mm / 1000.0
    fw, fh = footprint
    true_fx0, true_fy0 = -fw / 2, -fh / 2
    true_fx1, true_fy1 = fw / 2, fh / 2

    m = module_rect_local(footprint, bay_mm, interior_mm, wall_mm)
    p = placement_rect_local(footprint, bay_mm, interior_mm, wall_mm)

    # Both sides of the x axis are inset (module and placement share the
    # same x span, since the bay is on the y axis here).
    assert m[0] == p[0] == pytest.approx(true_fx0 + wall_m)
    assert m[2] == p[2] == pytest.approx(true_fx1 - wall_m)
    # module owns the far (+y) edge, placement owns the near (-y) edge.
    assert m[3] == pytest.approx(true_fy1 - wall_m)
    assert p[1] == pytest.approx(true_fy0 + wall_m)


def test_module_and_placement_tile_only_the_inset_footprint_with_wall_mm():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    wall_mm = 3.0
    wall_m = wall_mm / 1000.0
    fw, fh = footprint

    m = module_rect_local(footprint, bay_mm, interior_mm, wall_mm)
    p = placement_rect_local(footprint, bay_mm, interior_mm, wall_mm)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    inset_area = (fw - 2 * wall_m) * (fh - 2 * wall_m)
    assert area(m) + area(p) == pytest.approx(inset_area)
    # ... and strictly less than the un-inset footprint, i.e. some area was
    # actually reserved for the shell rim.
    assert area(m) + area(p) < fw * fh


def test_wall_mm_too_large_for_the_footprint_raises():
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (0.0, 66.0, 60.0, 90.0)
    footprint = (0.060, 0.090)
    with pytest.raises(ValueError):
        module_rect_local(footprint, bay_mm, interior_mm, wall_mm=31.0)


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


def test_module_world_placement_wall_mm_shrinks_size_but_stays_rotation_invariant():
    # Same AnkerPowerCore26800 fixture, with its measured case_wall_mm
    # (4.2mm). The x axis is inset by the wall on both sides exactly
    # (0.0817 - 2*0.0042 = 0.0733: the module always spans interior's full
    # perpendicular axis, which is now the inset local width). The y axis
    # shrinks too, but not by a simple 2*wall - it is the SAME proportional
    # lerp as before, now mapping into a smaller destination span; 0.03337
    # is that lerp's exact result, not a rounded approximation. Either way
    # w, h stay invariant to rotation - the point-rotation argument does
    # not care how "local" was computed.
    footprint = (0.0817, 0.180)
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 137.0, 358.0):
        _, _, w, h = module_world_placement(
            footprint, bay_mm, interior_mm, rot_deg, (0.0, 0.0),
            wall_mm=4.2)
        assert w == pytest.approx(0.0733, abs=1e-9)
        assert h == pytest.approx(0.033366666666666656, abs=1e-9)


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


import random

from recog.synth3d.bay import ObstructionPose, obstruction_world_poses, \
    sample_obstructions


class _ObsCfg:
    p_none = 0.4
    n_adhesive = (0, 6)
    n_foam = (0, 1)
    n_tape = (0, 2)
    n_label = (0, 1)
    adhesive_frac = (0.04, 0.14)
    foam_frac = (0.15, 0.35)
    tape_frac = (0.05, 0.12)
    label_frac = (0.10, 0.22)


RECT = (0.0, 0.0, 0.055, 0.065)


def test_every_obstruction_lies_inside_the_placement_rect():
    cfg = _ObsCfg()
    hits = 0
    for seed in range(300):
        poses = sample_obstructions(RECT, cfg, random.Random(seed))
        if poses:
            hits += 1
        for p in poses:
            assert p.x - p.w / 2 >= RECT[0] - 1e-9, f"seed {seed}: {p}"
            assert p.y - p.h / 2 >= RECT[1] - 1e-9, f"seed {seed}: {p}"
            assert p.x + p.w / 2 <= RECT[2] + 1e-9, f"seed {seed}: {p}"
            assert p.y + p.h / 2 <= RECT[3] + 1e-9, f"seed {seed}: {p}"

    # Anti-vacuity guard: a property test that iterates 300 seeds but never
    # gets a non-empty pose list back would pass with its assertion body
    # never once executed. Measured 176/300 non-empty draws with this cfg
    # and seed range; the floor below is well under that with headroom.
    assert hits > 100, f"test is vacuous: only {hits}/300 seeds produced any pose"


def test_roughly_forty_percent_of_bays_are_clean():
    cfg = _ObsCfg()
    empty = sum(1 for s in range(2000)
                if not sample_obstructions(RECT, cfg, random.Random(s)))
    assert 0.33 < empty / 2000 < 0.47, (
        "the network must see clean bays too, or it learns that every "
        "bay contains adhesive")


def test_sampling_is_deterministic_for_a_seed():
    cfg = _ObsCfg()
    a = sample_obstructions(RECT, cfg, random.Random(42))
    b = sample_obstructions(RECT, cfg, random.Random(42))
    assert a == b


def test_all_four_kinds_can_be_produced():
    cfg = _ObsCfg()
    kinds = set()
    for s in range(500):
        kinds.update(p.kind for p in sample_obstructions(
            RECT, cfg, random.Random(s)))
    assert kinds == {"adhesive", "foam", "tape", "label"}


# ---- obstruction_world_poses: same rotate-the-centre-point contract as
# module_world_placement/placement_world_placement, applied to individual
# obstruction poses instead of a whole rect - see that pair's docstrings
# for why a rotated world AABB is not an option (7.6% inflation measured at
# 2 degrees of jitter on an 81.7x180mm case). These pin down that an
# obstruction actually turns WITH its cartridge rather than staying fixed
# in world space while the bay rotates under it.

def test_obstruction_world_poses_is_identity_at_zero_rotation_and_no_translate():
    poses = [ObstructionPose(kind="adhesive", x=0.01, y=-0.02,
                             w=0.005, h=0.006, rot_deg=15.0)]
    out = obstruction_world_poses(poses, 0.0, (0.0, 0.0))
    assert out[0].x == pytest.approx(0.01)
    assert out[0].y == pytest.approx(-0.02)
    assert out[0].rot_deg == pytest.approx(15.0)
    assert (out[0].w, out[0].h) == (0.005, 0.006)


def test_obstruction_world_poses_translates_a_centred_obstruction():
    poses = [ObstructionPose(kind="foam", x=0.0, y=0.0, w=0.01, h=0.01)]
    out = obstruction_world_poses(poses, 0.0, (0.130, 0.245))
    assert (out[0].x, out[0].y) == pytest.approx((0.130, 0.245))


def test_obstruction_world_poses_rotates_the_local_point_about_the_origin():
    # A point at local (0.02, 0.0); a +90 degree turn sends +x to +y.
    poses = [ObstructionPose(kind="label", x=0.02, y=0.0, w=0.01, h=0.01,
                             rot_deg=0.0)]
    out = obstruction_world_poses(poses, 90.0, (0.0, 0.0))
    assert (out[0].x, out[0].y) == pytest.approx((0.0, 0.02), abs=1e-9)
    assert out[0].rot_deg == pytest.approx(90.0)


def test_obstruction_world_poses_composes_rotation_and_translation():
    poses = [ObstructionPose(kind="tape", x=0.02, y=0.0, w=0.01, h=0.03,
                             rot_deg=90.0)]
    out = obstruction_world_poses(poses, 90.0, (0.3, 0.2))
    # Local point rotates to (0, 0.02), then translates by (0.3, 0.2).
    assert (out[0].x, out[0].y) == pytest.approx((0.3, 0.22), abs=1e-9)
    # The obstruction's own tilt (90) plus the cartridge's turn (90) = 180.
    assert out[0].rot_deg == pytest.approx(180.0)


def test_obstruction_world_poses_keeps_size_invariant_to_rotation_angle():
    poses = [ObstructionPose(kind="adhesive", x=0.015, y=-0.01,
                             w=0.006, h=0.009, rot_deg=-40.0)]
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 137.0, 358.0):
        out = obstruction_world_poses(poses, rot_deg, (0.1, -0.2))
        assert (out[0].w, out[0].h) == (0.006, 0.009)


# ---- seated_cell_poses: cells the packer would seat in the bay itself,
# axis-aligned, at FFDH pitch - the deployed system's partly-filled case
# rather than the empty-or-sealed cases the generator produced before this.

from recog.synth3d.bay import seated_cell_poses, seated_cell_world_poses

CELL_W, CELL_H = 0.0183, 0.065          # 18650 in metres


def test_seated_cells_lie_inside_the_placement_rect():
    rect = (0.0, 0.0, 0.055, 0.065)
    for seed in range(100):
        poses = seated_cell_poses(rect, CELL_W, CELL_H, 3,
                                  random.Random(seed))
        for x, y, rot in poses:
            hw, hh = (CELL_W / 2, CELL_H / 2) if rot % 180 == 0 \
                else (CELL_H / 2, CELL_W / 2)
            assert rect[0] - 1e-9 <= x - hw and x + hw <= rect[2] + 1e-9
            assert rect[1] - 1e-9 <= y - hh and y + hh <= rect[3] + 1e-9


def test_seated_cells_do_not_overlap_each_other():
    rect = (0.0, 0.0, 0.055, 0.065)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 3, random.Random(0))
    boxes = []
    for x, y, rot in poses:
        hw, hh = (CELL_W / 2, CELL_H / 2) if rot % 180 == 0 \
            else (CELL_H / 2, CELL_W / 2)
        boxes.append((x - hw, y - hh, x + hw, y + hh))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            assert ox <= 1e-9 or oy <= 1e-9


def test_requesting_more_cells_than_fit_returns_only_what_fits():
    rect = (0.0, 0.0, 0.055, 0.065)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 99, random.Random(0))
    assert 0 < len(poses) <= 3          # 55mm / 18.3mm = 3 across


def test_zero_requested_returns_empty():
    rect = (0.0, 0.0, 0.055, 0.065)
    assert seated_cell_poses(rect, CELL_W, CELL_H, 0, random.Random(0)) == []


# ---- obstruction_forbidden_mask / seated_cell_poses(forbidden_mask=...):
# a bay can carry BOTH obstructions and seated cells (sampled independently
# against the same placement_rect), so a seated cell must be kept off any
# obstruction footprint - it is physically impossible for a real cell to
# rest on top of a glue blob, and the whole reason obstructions exist is to
# teach the segmenter that distinction. Rasterising obstructions into a
# forbidden grid and handing it to the SAME FFDH packer (which now advances
# past a forbidden cell instead of abandoning the shelf) keeps that
# guarantee without inventing a second collision system.

from recog.synth3d.bay import SEAT_MM_PER_CELL, obstruction_forbidden_mask


def _boxes_overlap(a, b, tol=1e-9):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox > tol and oy > tol


def test_seated_cells_avoid_a_forbidden_obstruction_footprint():
    rect = (0.0, 0.0, 0.055, 0.065)
    # Centred exactly on the first 18.3mm column - precisely where an
    # obstruction-unaware packer's first shelf cursor (x=0) would place the
    # first cell. Without the mask this WILL overlap (see the next test,
    # which pins that down as the pre-fix failure).
    obstruction = ObstructionPose(kind="foam", x=0.00915, y=0.0325,
                                  w=0.0183, h=0.065, rot_deg=0.0)
    obs_box = (obstruction.x - obstruction.w / 2, obstruction.y - obstruction.h / 2,
              obstruction.x + obstruction.w / 2, obstruction.y + obstruction.h / 2)

    mask = obstruction_forbidden_mask([obstruction], rect, SEAT_MM_PER_CELL)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 3, random.Random(0),
                              forbidden_mask=mask, mm_per_cell=SEAT_MM_PER_CELL)

    assert poses, "test is vacuous: the packer placed nothing at all"
    for x, y, rot in poses:
        hw, hh = (CELL_W / 2, CELL_H / 2) if rot % 180 == 0 \
            else (CELL_H / 2, CELL_W / 2)
        cell_box = (x - hw, y - hh, x + hw, y + hh)
        assert not _boxes_overlap(cell_box, obs_box), \
            f"seated cell at ({x}, {y}) overlaps the obstruction {obs_box}"


def test_without_the_mask_the_same_scene_actually_overlaps():
    """Pins down that the test above is not vacuous: the identical bay,
    packed with no forbidden_mask (the code's behaviour before this fix),
    really does seat a cell on top of the obstruction. If a future change
    makes this start failing, the fix above has stopped doing anything."""
    rect = (0.0, 0.0, 0.055, 0.065)
    obstruction = ObstructionPose(kind="foam", x=0.00915, y=0.0325,
                                  w=0.0183, h=0.065, rot_deg=0.0)
    obs_box = (obstruction.x - obstruction.w / 2, obstruction.y - obstruction.h / 2,
              obstruction.x + obstruction.w / 2, obstruction.y + obstruction.h / 2)

    poses = seated_cell_poses(rect, CELL_W, CELL_H, 3, random.Random(0))
    hits = [(x, y, rot) for x, y, rot in poses
            if _boxes_overlap(
                (x - (CELL_W / 2 if rot % 180 == 0 else CELL_H / 2),
                 y - (CELL_H / 2 if rot % 180 == 0 else CELL_W / 2),
                 x + (CELL_W / 2 if rot % 180 == 0 else CELL_H / 2),
                 y + (CELL_H / 2 if rot % 180 == 0 else CELL_W / 2)),
                obs_box)]
    assert hits, "expected the mask-unaware packer to overlap the obstruction"


def test_dense_obstruction_coverage_yields_fewer_or_no_seated_cells_without_raising():
    rect = (0.0, 0.0, 0.055, 0.065)
    # One obstruction covering the whole bay.
    obstruction = ObstructionPose(kind="foam", x=0.0275, y=0.0325,
                                  w=0.055, h=0.065, rot_deg=0.0)
    mask = obstruction_forbidden_mask([obstruction], rect, SEAT_MM_PER_CELL)
    poses = seated_cell_poses(rect, CELL_W, CELL_H, 3, random.Random(0),
                              forbidden_mask=mask, mm_per_cell=SEAT_MM_PER_CELL)
    assert poses == []


def test_obstruction_forbidden_mask_is_empty_for_no_obstructions():
    rect = (0.0, 0.0, 0.055, 0.065)
    mask = obstruction_forbidden_mask([], rect, SEAT_MM_PER_CELL)
    assert mask.any() == False
    assert mask.shape == (
        max(1, __import__("math").ceil(0.065 / SEAT_MM_PER_CELL)),
        max(1, __import__("math").ceil(0.055 / SEAT_MM_PER_CELL)),
    )


def test_obstruction_forbidden_mask_marks_the_obstructions_own_cells():
    rect = (0.0, 0.0, 0.055, 0.065)
    obstruction = ObstructionPose(kind="label", x=0.00915, y=0.0325,
                                  w=0.0183, h=0.01, rot_deg=0.0)
    mask = obstruction_forbidden_mask([obstruction], rect, SEAT_MM_PER_CELL)
    assert mask.any()
    # The centre cell of the obstruction must be forbidden.
    col = int(obstruction.x / SEAT_MM_PER_CELL)
    row = int(obstruction.y / SEAT_MM_PER_CELL)
    assert mask[row, col]
    # A far corner of the bay, well outside the obstruction, must be free.
    assert not mask[-1, -1]


# ---- seated_cell_world_poses: same rotate-the-centre-point contract as
# obstruction_world_poses, applied to seated_cell_poses's (x, y, rot_deg)
# tuples instead of an ObstructionPose - a seated cell must turn WITH its
# cartridge, staying axis-aligned to the BAY rather than to the world, or a
# rotated cartridge would show cells crossing its own bay walls.

def test_seated_cell_world_poses_is_identity_at_zero_rotation_and_no_translate():
    poses = [(0.01, -0.02, 90.0)]
    out = seated_cell_world_poses(poses, 0.0, (0.0, 0.0))
    assert out[0][:2] == pytest.approx((0.01, -0.02))
    assert out[0][2] == pytest.approx(90.0)


def test_seated_cell_world_poses_translates_a_centred_cell():
    poses = [(0.0, 0.0, 0.0)]
    out = seated_cell_world_poses(poses, 0.0, (0.130, 0.245))
    assert out[0][:2] == pytest.approx((0.130, 0.245))


def test_seated_cell_world_poses_rotates_the_local_point_about_the_origin():
    # A point at local (0.02, 0.0); a +90 degree turn sends +x to +y.
    poses = [(0.02, 0.0, 0.0)]
    out = seated_cell_world_poses(poses, 90.0, (0.0, 0.0))
    assert out[0][:2] == pytest.approx((0.0, 0.02), abs=1e-9)
    assert out[0][2] == pytest.approx(90.0)


def test_seated_cell_world_poses_composes_rotation_and_translation():
    poses = [(0.02, 0.0, 90.0)]
    out = seated_cell_world_poses(poses, 90.0, (0.3, 0.2))
    # Local point rotates to (0, 0.02), then translates by (0.3, 0.2).
    assert out[0][:2] == pytest.approx((0.3, 0.22), abs=1e-9)
    # The cell's own pitch orientation (90) plus the cartridge's turn (90).
    assert out[0][2] == pytest.approx(180.0)


def test_seated_cell_world_poses_empty_in_empty_out():
    assert seated_cell_world_poses([], 37.0, (0.1, 0.2)) == []


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


# ---- interior_from_tray: task 2 - the tray's cavity footprint (its outer
# rect inset by the wall), replacing case_interior_mm which held the
# assembly's OUTER extent despite its name.

def test_interior_is_the_tray_inset_by_the_wall():
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    # wall 4.0 -> interior runs 4..56 x 4..86
    assert interior_from_tray(tray, cells, 4.0) == pytest.approx(
        (4.0, 4.0, 56.0, 86.0))


def test_interior_never_exceeds_the_tray():
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    x0, y0, x1, y1 = interior_from_tray(tray, cells, 4.0)
    assert x0 >= tray[0] and y0 >= tray[1]
    assert x1 <= tray[2] and y1 <= tray[3]


def test_interior_contains_the_cells():
    """The cells demonstrably fit inside the cavity in assembled pose, so an
    interior that excludes them is measured wrong."""
    from recog.synth3d.bay import interior_from_tray

    tray = (0.0, 0.0, 60.0, 90.0)
    cells = (4.0, 4.0, 56.0, 66.0)
    ix0, iy0, ix1, iy1 = interior_from_tray(tray, cells, 4.0)
    assert ix0 <= cells[0] and iy0 <= cells[1]
    assert ix1 >= cells[2] and iy1 >= cells[3]


def test_interior_raises_when_the_wall_would_swallow_the_cavity():
    from recog.synth3d.bay import interior_from_tray

    with pytest.raises(ValueError):
        interior_from_tray((0.0, 0.0, 20.0, 20.0),
                           (2.0, 2.0, 18.0, 18.0), 12.0)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name", [
    "AnkerPowerCore10000", "AnkerPowerCore13000",
    "AnkerPowerCore20100", "AnkerPowerCore26800",
])
def test_catalog_records_a_tray_cavity(name):
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)

    for key in ("tray_outer_mm", "tray_floor_mm", "interior_mm",
                "case_wall_mm", "module_bay_mm"):
        assert key in entry, f"{name} missing {key}; re-run recog.convert_cad"

    assert "case_interior_mm" not in entry, (
        "case_interior_mm was the assembly's OUTER extent despite its name "
        "and must not survive alongside a real interior measurement")

    tx0, ty0, tx1, ty1, _ = entry["tray_outer_mm"]
    ix0, iy0, ix1, iy1 = entry["interior_mm"]
    assert tx0 <= ix0 < ix1 <= tx1
    assert ty0 <= iy0 < iy1 <= ty1
    assert entry["tray_floor_mm"] > 0.0


@pytest.mark.skipif(not os.path.isfile(os.path.join(ASSETS, "catalog.json")),
                    reason="catalog.json not built")
@pytest.mark.parametrize("name,expected_floor", [
    ("AnkerPowerCore10000", 1.95),
    ("AnkerPowerCore13000", 1.95),
    ("AnkerPowerCore20100", 1.95),
    ("AnkerPowerCore26800", 1.95),
])
def test_tray_floor_matches_where_the_cells_rest(name, expected_floor):
    """The cells sit ON the cavity floor in assembled pose, so the floor is
    where their lowest point is. Hand-measured from the CAD: every assembly
    rests its cells at z = 1.95 mm. If Step 4's derivation disagrees, the
    floor is measured wrong and every label sits at the wrong height.

    If a regenerated catalog reports a materially different value for an
    assembly, do NOT relax this test - re-measure that asset and find out
    why it differs, exactly as the bay depths were established.
    """
    with open(os.path.join(ASSETS, "catalog.json")) as fh:
        cat = json.load(fh)
    entry = next(a for a in cat["assets"] if a["name"] == name)
    assert entry["tray_floor_mm"] == pytest.approx(expected_floor, abs=0.3)
