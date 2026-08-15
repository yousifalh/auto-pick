"""Bay geometry — pure functions, no bpy, no Blender."""
from __future__ import annotations

import json
import math
import os

import pytest

from recog.synth3d.bay import (case_wall_from_bounds, module_bay_from_bounds,
                                needs_flip)

ASSETS = os.path.join(os.path.dirname(__file__), "..",
                      "recog", "synth3d", "assets")


def test_needs_flip_is_false_when_lid_sits_above_the_case():
    # Right-side-up: case centred at 5.55 (span 0..11.1), lid at 16.65
    # (span 11.1..22.2) - the lid is above the case, no flip needed.
    assert needs_flip(case_centroid_z=5.55, lid_centroid_z=16.65) is False


def test_needs_flip_is_true_when_case_sits_above_the_lid():
    # Inverted: mirroring 0..11.1 about the assembly mid-plane (11.1)
    # sends the case to 11.1..22.2 (centroid 16.65) while the lid stays
    # at 0..11.1 (centroid 5.55) - exactly task-3c's measured case.
    assert needs_flip(case_centroid_z=16.65, lid_centroid_z=5.55) is True


def test_needs_flip_ties_toward_flip():
    # Degenerate/adjacent-not-yet-separated centroids: treat as inverted
    # rather than silently trusting an ambiguous "already correct" read -
    # this function only ever gets called because something looked wrong.
    assert needs_flip(case_centroid_z=10.0, lid_centroid_z=10.0) is True


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

# Task 3 (2026-08-09-tray-interior, revised brief): `module_rect_local` /
# `placement_rect_local` used to lerp `bay_mm`'s fractional position within
# `interior_mm` onto `assets.Item.footprint` (a THIRD rect, the cartridge's
# outer un-rotated size) via a `wall_mm`-inset margin. That was only an
# identity mapping while `interior_mm` and the footprint were (near enough)
# the same rectangle - true only because `case_interior_mm` was misnamed
# and actually held the outer AABB. Task 2 made `interior_mm` a REAL,
# smaller (and sometimes asymmetric) cavity, which made the lerp wrong: it
# stretched the module/placement rects back OUT across the whole footprint,
# including the wall (see task-3-report.md for the measured numbers).
#
# The fix removes the lerp, the footprint argument and `wall_mm` entirely:
# `tray_outer_mm` (and so `interior_mm`/`bay_mm`) is measured centred on
# (0, 0) in the CAD's own frame, which IS the item's local pivot frame, so
# `bay_mm` needs only a millimetre -> metre conversion, no remapping.

CATALOG_FIXTURES = [
    # name, interior_mm, bay_mm - copied from recog/synth3d/assets/
    # catalog.json so these tests exercise the real measured geometry, not
    # an invented rectangle.
    ("AnkerPowerCore10000", (-27.45, -43.0, 27.45, 41.45),
     (-27.45, 22.0, 27.45, 41.45)),
    ("AnkerPowerCore13000", (-36.6, -44.75, 36.6, 44.75),
     (-36.6, 22.0, 36.6, 44.75)),
    ("AnkerPowerCore20100", (-27.45, -80.2, 27.45, 80.2),
     (-27.45, 55.0, 27.45, 80.2)),
    ("AnkerPowerCore26800", (-36.6, -85.75, 36.6, 85.75),
     (-36.6, 55.0, 36.6, 85.75)),
]


@pytest.mark.parametrize("name,interior_mm,bay_mm", CATALOG_FIXTURES)
def test_module_and_placement_rects_tile_the_interior_exactly(
        name, interior_mm, bay_mm):
    """The design spec's §7 acceptance criterion: the pair tiles the
    INTERIOR, not the cartridge's outer footprint. This is the test that
    would have caught the lerp-through-the-footprint bug (see
    task-3-report.md's Amendment C measurements)."""
    m = module_rect_local(interior_mm, bay_mm)
    p = placement_rect_local(interior_mm, bay_mm)

    ox = min(m[2], p[2]) - max(m[0], p[0])
    oy = min(m[3], p[3]) - max(m[1], p[1])
    assert ox <= 1e-9 or oy <= 1e-9, f"{name}: module and placement overlap"

    ix0, iy0, ix1, iy1 = (v / 1000.0 for v in interior_mm)
    union = (min(m[0], p[0]), min(m[1], p[1]), max(m[2], p[2]), max(m[3], p[3]))
    assert union == pytest.approx((ix0, iy0, ix1, iy1))

    def area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    assert area(m) + area(p) == pytest.approx((ix1 - ix0) * (iy1 - iy0))


@pytest.mark.parametrize("name,interior_mm,bay_mm", CATALOG_FIXTURES)
def test_neither_rect_exceeds_the_interior(name, interior_mm, bay_mm):
    ix0, iy0, ix1, iy1 = (v / 1000.0 for v in interior_mm)
    for r in (module_rect_local(interior_mm, bay_mm),
             placement_rect_local(interior_mm, bay_mm)):
        assert ix0 - 1e-9 <= r[0] < r[2] <= ix1 + 1e-9
        assert iy0 - 1e-9 <= r[1] < r[3] <= iy1 + 1e-9


@pytest.mark.parametrize("name,interior_mm,bay_mm", CATALOG_FIXTURES)
def test_module_rect_equals_bay_mm_converted_to_metres(name, interior_mm, bay_mm):
    m = module_rect_local(interior_mm, bay_mm)
    assert m == pytest.approx(tuple(v / 1000.0 for v in bay_mm))


def test_bay_not_flush_against_any_interior_edge_raises():
    """A bay floating in the middle of the interior (touching no edge) is a
    bad upstream measurement; both rect functions must fail loudly on it
    rather than silently return a plausible-looking rectangle."""
    interior_mm = (0.0, 0.0, 60.0, 90.0)
    bay_mm = (10.0, 10.0, 50.0, 50.0)
    with pytest.raises(ValueError):
        module_rect_local(interior_mm, bay_mm)
    with pytest.raises(ValueError):
        placement_rect_local(interior_mm, bay_mm)


def test_the_10000s_asymmetric_interior_survives_into_the_placement_rect():
    """interior_from_tray widened AnkerPowerCore10000's y0 edge to -43.0mm
    to contain the cells - NOT a symmetric wall inset, which would have put
    it at -41.45mm (= tray_outer's -45.45 + case_wall_mm's 4.0). The old
    lerp-through-the-footprint destroyed exactly this asymmetry (it can only
    reproduce a UNIFORM inset of the footprint); pin the true value
    explicitly so it cannot silently regress."""
    interior_mm = (-27.45, -43.0, 27.45, 41.45)
    bay_mm = (-27.45, 22.0, 27.45, 41.45)
    p = placement_rect_local(interior_mm, bay_mm)
    assert p[1] == pytest.approx(-0.0430, abs=1e-6)
    assert p[1] != pytest.approx(-0.04145, abs=1e-4)


@pytest.mark.parametrize("interior,bay,edge", [
    ((0.0, 0.0, 60.0, 90.0), (0.0, 66.0, 60.0, 90.0), "+y"),
    ((0.0, 0.0, 60.0, 90.0), (0.0, 0.0, 60.0, 24.0), "-y"),
    ((0.0, 0.0, 90.0, 60.0), (66.0, 0.0, 90.0, 60.0), "+x"),
    ((0.0, 0.0, 90.0, 60.0), (0.0, 0.0, 24.0, 60.0), "-x"),
])
def test_bay_edge_detected_generically_on_all_four_sides(interior, bay, edge):
    """`_bay_edge` must not hardcode any one side - every real asset
    happens to have its bay on +y, but nothing in the geometry guarantees
    that for a future one, and the brief explicitly warns against assuming
    it."""
    from recog.synth3d.bay import _bay_edge
    assert _bay_edge(interior, bay) == edge

    m = module_rect_local(interior, bay)
    p = placement_rect_local(interior, bay)
    def area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    ix0, iy0, ix1, iy1 = (v / 1000.0 for v in interior)
    assert area(m) + area(p) == pytest.approx((ix1 - ix0) * (iy1 - iy0))


# ---- module_world_placement: `layout.plan` rotates by quarter*90 + a few
# degrees of jitter, and a naive lerp into the ROTATED case's world AABB
# inflates the module, because the AABB of a rotated rectangle is larger
# than the rectangle. These pin down that rotating the LOCAL centre as a
# point, which has no extent to inflate, keeps the module's true size for
# ANY rotation angle, not just the k*90 multiples a quarter-only fix would
# handle. Unchanged by this task other than dropping `footprint`/`wall_mm`
# from the call - "do not change how they rotate" (task-3-brief.md).

def test_module_world_placement_size_is_invariant_to_rotation_angle():
    # AnkerPowerCore26800's real numbers: 81.7mm-wide interior, a 35.0mm
    # bay on the +y end - the case measured on the contact sheet as
    # overhanging by 3.3% at ~2 degrees of jitter under the old naive lerp.
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 91.7, 137.0, 180.0, 270.0, 358.0):
        _, _, w, h = module_world_placement(
            interior_mm, bay_mm, rot_deg, (0.0, 0.0))
        assert w == pytest.approx(0.0817, abs=1e-9)
        assert h == pytest.approx(0.035, abs=1e-9)


def test_module_world_placement_matches_the_local_case_at_zero_rotation():
    # A synthetic but CENTRED (item-pivot-frame) 60x90mm interior, bay on
    # +y - the real assets' shape, at round numbers.
    bay_mm = (-30.0, 21.0, 30.0, 45.0)
    interior_mm = (-30.0, -45.0, 30.0, 45.0)
    cx, cy, w, h = module_world_placement(
        interior_mm, bay_mm, 0.0, (0.130, 0.245))
    assert (w, h) == pytest.approx((0.060, 0.024))
    assert (cx, cy) == pytest.approx((0.130, 0.278))


def test_module_world_placement_centre_rotates_about_the_translate_point():
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    cx, cy, w, h = module_world_placement(
        interior_mm, bay_mm, 90.0, (0.0, 0.0))
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
    interior_mm = (-31.45, -45.45, 31.45, 45.45)
    bay_mm = (-31.45, 22.6, 31.45, 45.45)

    m = module_rect_local(interior_mm, bay_mm)
    p = placement_rect_local(interior_mm, bay_mm)

    ox = min(m[2], p[2]) - max(m[0], p[0])
    oy = min(m[3], p[3]) - max(m[1], p[1])
    assert ox <= 1e-9 or oy <= 1e-9, "module and placement rects overlap"


def test_placement_world_placement_size_is_invariant_to_rotation_angle():
    # Same fixture as module_world_placement's rotation-invariance test:
    # AnkerPowerCore26800, 81.7mm interior, a 35.0mm bay on the +y end.
    # The complement is the 90 - 35 = 55.0mm strip on the -y side.
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 91.7, 137.0, 180.0, 270.0, 358.0):
        _, _, w, h = placement_world_placement(
            interior_mm, bay_mm, rot_deg, (0.0, 0.0))
        assert w == pytest.approx(0.0817, abs=1e-9)
        assert h == pytest.approx(0.145, abs=1e-9)


def test_placement_world_placement_matches_the_local_case_at_zero_rotation():
    bay_mm = (-30.0, 21.0, 30.0, 45.0)
    interior_mm = (-30.0, -45.0, 30.0, 45.0)
    cx, cy, w, h = placement_world_placement(
        interior_mm, bay_mm, 0.0, (0.130, 0.245))
    assert (w, h) == pytest.approx((0.060, 0.066))
    assert (cx, cy) == pytest.approx((0.130, 0.233))


def test_placement_world_placement_centre_rotates_about_the_translate_point():
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    cx, cy, w, h = placement_world_placement(
        interior_mm, bay_mm, 90.0, (0.0, 0.0))
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
    assert not mask.any()
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
    bay_mm = (-40.85, 55.0, 40.85, 90.0)
    interior_mm = (-40.85, -90.0, 40.85, 90.0)
    for rot_deg in (0.0, 2.0, 47.3, 90.0, 137.0, 358.0):
        mcx, mcy, mw, mh = module_world_placement(
            interior_mm, bay_mm, rot_deg, (0.3, 0.2))
        pcx, pcy, pw, ph = placement_world_placement(
            interior_mm, bay_mm, rot_deg, (0.3, 0.2))
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


# ------------------------------------------------ procedural tray sampler --
import random as _random

from recog.synth3d.config import CELL_FORMATS, Config


def test_bay_edge_public_wrapper_matches_the_private_validator():
    from recog.synth3d.bay import _bay_edge, bay_edge
    interior = (0.0, 0.0, 60.0, 90.0)
    bay = (0.0, 66.0, 60.0, 90.0)
    assert bay_edge(interior, bay) == _bay_edge(interior, bay) == "+y"


def test_sample_tray_is_deterministic_given_the_same_rng_state():
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    assert sample_tray(cfg, _random.Random(7)) == sample_tray(cfg, _random.Random(7))


def test_sample_tray_cell_format_is_always_a_configured_one():
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    for seed in range(30):
        s = sample_tray(cfg, _random.Random(seed))
        assert s.cell_format in CELL_FORMATS


def test_sample_tray_module_bay_is_a_full_span_strip_on_the_chosen_edge():
    """module_bay_from_bounds's own invariant must hold by construction -
    bay.bay_edge (the existing, tested validator) is reused directly
    rather than re-implementing the check, so the two can never drift
    apart (design spec Sec3.3: 'satisfied by construction, not asserted
    after the fact')."""
    from recog.synth3d.bay import bay_edge, sample_tray
    cfg = Config().tray_anchored
    for seed in range(50):
        s = sample_tray(cfg, _random.Random(seed))
        assert bay_edge(s.interior_mm, s.module_bay_mm) == s.bay_edge


def test_sample_tray_anchored_restricts_the_bay_to_the_long_axiss_ends():
    """Design spec Sec9.1: anchored fixes the module bay to a short edge -
    i.e. the bay axis is the tray's LONGER footprint axis, matching all
    four measured SKUs (e.g. PowerCore10000: 54.9mm short x 84.45mm long,
    bay flush against a long-axis end)."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    for seed in range(50):
        s = sample_tray(cfg, _random.Random(seed))
        ix0, iy0, ix1, iy1 = s.interior_mm
        long_axis_is_y = (iy1 - iy0) >= (ix1 - ix0)
        assert s.bay_edge in (("-y", "+y") if long_axis_is_y else ("-x", "+x"))


def test_sample_tray_wide_can_put_the_bay_on_the_short_axis():
    """The anchored restriction must be a real behavioural difference,
    not decoration - wide has to actually exercise an edge anchored
    would never draw, over enough seeds."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_wide
    saw_short_axis_bay = False
    for seed in range(300):
        s = sample_tray(cfg, _random.Random(seed))
        ix0, iy0, ix1, iy1 = s.interior_mm
        long_axis_is_y = (iy1 - iy0) >= (ix1 - ix0)
        short_axis_edges = ("-x", "+x") if long_axis_is_y else ("-y", "+y")
        if s.bay_edge in short_axis_edges:
            saw_short_axis_bay = True
            break
    assert saw_short_axis_bay, "wide never drew a short-axis bay in 300 seeds"


def test_sample_tray_case_outer_encloses_the_interior_with_real_wall():
    """case_outer_mm must be interior_mm expanded OUTWARD by wall_mm on
    ALL FOUR sides (this plan's header note #2) - not equal to
    interior_mm, or world.build_procedural_tray would have no wall
    thickness to build."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    s = sample_tray(cfg, _random.Random(0))
    ix0, iy0, ix1, iy1 = s.interior_mm
    ox0, oy0, ox1, oy1 = s.case_outer_mm
    assert ox0 == pytest.approx(ix0 - s.wall_mm)
    assert oy0 == pytest.approx(iy0 - s.wall_mm)
    assert ox1 == pytest.approx(ix1 + s.wall_mm)
    assert oy1 == pytest.approx(iy1 + s.wall_mm)


def test_sample_tray_anchored_footprint_roughly_brackets_the_measured_skus():
    """Not exact per-draw (independent axes multiply out wider than any
    single SKU) but the population should land in the measured 62.9x90.9
    - 81.7x180mm neighbourhood, not somewhere wildly different."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    widths, heights = [], []
    for seed in range(200):
        s = sample_tray(cfg, _random.Random(seed))
        ox0, oy0, ox1, oy1 = s.case_outer_mm
        widths.append(ox1 - ox0)
        heights.append(oy1 - oy0)
    widths.sort()
    heights.sort()
    assert 40.0 <= widths[len(widths) // 2] <= 130.0
    assert 60.0 <= heights[len(heights) // 2] <= 220.0


# --------------------------------------------- procedural catalog entries --

def test_build_tray_entry_carries_the_three_fields_scene_reads():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_tray_entry
    s = sample_tray(Config().tray_anchored, _random.Random(0))
    entry = build_tray_entry(s)
    assert entry["kind"] == "procedural"
    for key in ("interior_mm", "module_bay_mm", "tray_floor_mm"):
        assert key in entry
    assert entry["interior_mm"] == [round(v, 2) for v in s.interior_mm]
    assert entry["cell_format"] == s.cell_format


def test_build_tray_entry_raises_loudly_on_a_malformed_sample():
    """A TraySample whose module_bay isn't a genuine full-span strip must
    fail HERE, at registration time - not silently degrade a scene's
    labels the way scene.py's `entry.get(...)` guard would if a bad
    procedural entry ever reached it (that guard exists for the CAD's
    legitimate no-measurement case, not for a fully-computed procedural
    entry, which has no such excuse)."""
    from recog.synth3d.bay import TraySample
    from recog.synth3d.catalog import build_tray_entry
    bad = TraySample(
        interior_mm=(0.0, 0.0, 60.0, 90.0),
        module_bay_mm=(10.0, 10.0, 50.0, 80.0),   # not flush against any edge
        case_outer_mm=(-4.0, -4.0, 64.0, 94.0), wall_mm=4.0,
        case_half_height_mm=11.1, tray_floor_mm=1.95, cell_format="18650",
        bay_edge="+y")
    with pytest.raises(ValueError):
        build_tray_entry(bad)


def test_build_procedural_pool_makes_n_uniquely_named_entries():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    pool = build_procedural_pool(10, sample_tray, Config().tray_anchored, seed=0)
    assert len(pool) == 10
    assert all(e["kind"] == "procedural" for e in pool.values())


def test_build_procedural_pool_is_reproducible_for_the_same_seed():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    a = build_procedural_pool(5, sample_tray, Config().tray_anchored, seed=3)
    b = build_procedural_pool(5, sample_tray, Config().tray_anchored, seed=3)
    assert a == b


def test_build_procedural_pool_names_do_not_collide_with_cad_names():
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_procedural_pool
    pool = build_procedural_pool(3, sample_tray, Config().tray_anchored,
                                 seed=0, name_prefix="anchored")
    assert all(name.startswith("anchored_") for name in pool)


def test_exclude_assets_drops_only_the_named_keys_without_mutating_input():
    from recog.synth3d.catalog import exclude_assets
    assets = {"A": {}, "B": {}, "C": {}}
    out = exclude_assets(assets, ["B"])
    assert set(out) == {"A", "C"}
    assert set(assets) == {"A", "B", "C"}


def test_exclude_assets_raises_on_an_unknown_name():
    from recog.synth3d.catalog import exclude_assets
    with pytest.raises(KeyError):
        exclude_assets({"A": {}}, ["nope"])


# ---------------------------------------- naming contract Task 8 relies on --

def test_procedural_object_names_classify_correctly_via_role_of():
    """world.build_procedural_tray (bpy-only, no direct pytest coverage)
    names its objects to satisfy CLASS_RULES' EXISTING regexes, so
    _load_template's shared role-tagging tail needs no procedural-aware
    branch. A silent misclassification here would tag the lid as `case`
    too (both fall to ROLE_FALLBACK if the name doesn't match "_top"),
    rendering every open procedural cartridge CLOSED - the exact
    pre-tray-interior-fix defect, reincarnated."""
    from recog.synth3d.catalog import role_of
    assert role_of("ProcCase_btm") == "case"
    assert role_of("ProcCase_top") == "case_lid"
    assert role_of("ProcCell_0") == "cell"


def test_sample_tray_case_half_height_always_clears_the_floor():
    """world.build_procedural_tray's cavity cutter has height
    (case_half_height_mm - tray_floor_mm); TrayRangeCfg.tray_wide's own
    ranges overlap (tray_floor_mm_range up to 4.5mm, case_half_height_mm_
    range down to 4.0mm) so an UNCONSTRAINED independent draw of both can
    put the floor AT OR ABOVE the rim, giving that cutter zero or negative
    height - a degenerate boolean cut, not merely an unusual tray. This is
    a structural validity requirement for ANY tray (anchored or wide), not
    a plausibility judgement, so sample_tray must guarantee it by
    construction rather than leave it to chance."""
    from recog.synth3d.bay import sample_tray
    for cfg in (Config().tray_anchored, Config().tray_wide):
        for seed in range(5000):
            s = sample_tray(cfg, _random.Random(seed))
            assert s.case_half_height_mm > s.tray_floor_mm, (
                f"seed={seed} half_height={s.case_half_height_mm} "
                f"floor={s.tray_floor_mm}")


def test_sample_tray_case_half_height_always_fits_the_drawn_cells_diameter():
    """The 'assembled' variant seals a cell inside the CASE+LID envelope
    (total height 2*case_half_height_mm) - found via world.
    build_procedural_tray's own numeric self-check raising on the very
    first render: case_half_height_mm_range is anchored to the real,
    18650-only measured assemblies (11.1mm) and predates Group A's
    21700/26650 formats, so an unconstrained draw could pick a 26mm-
    diameter 26650 cell for a shell sized 2*10.5=21mm tall. The cell
    rests at z=tray_floor_mm, not z=0, so its own top is
    tray_floor_mm + diameter - a first version of this fix that forgot
    the floor offset still raised on the SECOND procedural tray built.
    sample_tray must guarantee every format is physically sealable by
    construction."""
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.config import CELL_FORMATS
    for cfg in (Config().tray_anchored, Config().tray_wide):
        for seed in range(2000):
            s = sample_tray(cfg, _random.Random(seed))
            diam_mm = CELL_FORMATS[s.cell_format][0] * 1000.0
            cell_top_mm = s.tray_floor_mm + diam_mm
            assert 2.0 * s.case_half_height_mm > cell_top_mm, (
                f"seed={seed} cell_format={s.cell_format} "
                f"floor={s.tray_floor_mm} diam={diam_mm} "
                f"cell_top={cell_top_mm} "
                f"2*half_height={2.0 * s.case_half_height_mm}")


# ------------------------------------------------- lid crown (sealed unit) --
#
# Why this exists: measured 2026-08-11, the four Anker lids are BARREL-
# CROWNED - long-edge fillet radius 11.10mm = the entire lid height, and
# 89% of each lid's upward-facing polygons have z-normal < 0.95 - while
# world.build_procedural_tray's lid is a planar cuboid (0%). A sealed
# procedural cartridge therefore renders as a featureless flat rectangle
# (internal luminance p95-p05 median 0.0272) where a sealed CAD one has a
# dark edge falloff framing a bright crown (0.2719, 10x). See
# docs/superpowers/specs/2026-08-11-sealed-unit-experiment.md.

def test_lid_crown_range_defaults_to_zero_so_existing_configs_are_unchanged():
    """A config written before the crown existed must sample exactly the
    geometry it always did - the default range is degenerate, so the
    drawn crown is 0.0 and world.build_procedural_tray's bevel is skipped
    entirely."""
    from recog.synth3d.bay import sample_tray
    cfg = Config().tray_anchored
    assert cfg.lid_crown_mm_range == (0.0, 0.0)
    for seed in range(200):
        assert sample_tray(cfg, _random.Random(seed)).lid_crown_mm == 0.0


def test_lid_crown_is_drawn_LAST_so_every_other_tray_field_is_identical():
    """The experiment's whole design rests on this: the crowned procedural
    set must differ from `anchored` in the crown and NOTHING else. If the
    crown draw were taken anywhere but last it would shift the rng stream
    and silently resample every downstream tray parameter, turning a
    one-variable experiment into a two-variable one with no error
    anywhere."""
    import dataclasses
    from recog.synth3d.bay import sample_tray
    flat = Config().tray_anchored
    crowned = dataclasses.replace(flat, lid_crown_mm_range=(0.0, 12.0))
    moved = 0
    for seed in range(500):
        a = sample_tray(flat, _random.Random(seed))
        b = sample_tray(crowned, _random.Random(seed))
        for f in dataclasses.fields(a):
            if f.name == "lid_crown_mm":
                continue
            assert getattr(a, f.name) == getattr(b, f.name), (
                f"seed={seed}: {f.name} moved when only the crown range "
                f"changed - the crown draw is not last")
        moved += b.lid_crown_mm > 0.0
    assert moved > 400, (
        f"only {moved}/500 draws produced a non-zero crown - the range "
        f"is not actually being sampled")


def test_lid_crown_is_clamped_to_the_lid_height_and_its_own_footprint():
    """A bevel wider than the lid is deep or half its shortest side is a
    degenerate boolean-shaped failure, not an unusual tray - the same
    class world.build_procedural_tray's cavity-cutter clamp already
    guards. Clamp here, bpy-free, where it can be tested."""
    import dataclasses
    from recog.synth3d.bay import MAX_LID_CROWN_FOOTPRINT_FRAC, sample_tray
    for base in (Config().tray_anchored, Config().tray_wide):
        cfg = dataclasses.replace(base, lid_crown_mm_range=(0.0, 40.0))
        for seed in range(2000):
            s = sample_tray(cfg, _random.Random(seed))
            ox0, oy0, ox1, oy1 = s.case_outer_mm
            short = min(ox1 - ox0, oy1 - oy0)
            assert 0.0 <= s.lid_crown_mm <= s.case_half_height_mm + 1e-9, (
                f"seed={seed} crown={s.lid_crown_mm} exceeds the lid's own "
                f"height {s.case_half_height_mm}")
            assert s.lid_crown_mm <= MAX_LID_CROWN_FOOTPRINT_FRAC * short + 1e-9, (
                f"seed={seed} crown={s.lid_crown_mm} exceeds "
                f"{MAX_LID_CROWN_FOOTPRINT_FRAC} x the lid's shortest side "
                f"{short}")


def test_build_tray_entry_carries_lid_crown_mm_to_the_bpy_side():
    """world.build_procedural_tray reads the entry, not the TraySample -
    a field that stops at the dataclass boundary is this project's
    documented silent-degradation shape (a renamed catalog key that
    quietly stopped building geometry)."""
    import dataclasses
    from recog.synth3d.bay import sample_tray
    from recog.synth3d.catalog import build_tray_entry
    cfg = dataclasses.replace(Config().tray_anchored,
                              lid_crown_mm_range=(0.0, 12.0))
    s = sample_tray(cfg, _random.Random(3))
    entry = build_tray_entry(s)
    assert "lid_crown_mm" in entry
    assert entry["lid_crown_mm"] == round(s.lid_crown_mm, 2)
    flat = build_tray_entry(sample_tray(Config().tray_anchored,
                                        _random.Random(3)))
    assert flat["lid_crown_mm"] == 0.0


# ============================================================ seating ladder ==
#
# The ladder is what makes `placement_area` mean "the currently free floor".
# Every object in a bay is built at a small offset above the cavity floor, and
# `bay_proxy` is the plane carrying the label; anything at or below it stops
# occluding it, and the mask goes on reporting that floor as free while the
# object visibly sits on it - in the render, the index pass and the manifest
# alike, with nothing downstream able to tell.
#
# Until 2026-08-12 these were six literals in three bpy-only builders plus a
# world.py constant, with the ordering restated in three docstrings and
# enforced nowhere. `tests/test_synth3d_world.py` checks that world.py still
# builds ON these rungs; this checks the rungs.

from recog.synth3d.bay import (BAY_PROXY_RUNG, MAX_SEAT_OFFSET_M,
                               PCB_THICKNESS_M, SEATING_LADDER,
                               assert_seating_ladder_ordered,
                               fallback_module_placement, obstruction_z_scale,
                               occludes_bay_proxy, seat_offset, seat_z)


def test_the_seating_ladder_is_exactly_the_six_offsets_it_has_always_been():
    """Written out, not derived. This is the one test in the suite whose
    failure means "the ground truth of every rendered dataset just changed" -
    so it states the numbers rather than recomputing them, and a diff on these
    lines is the intended way to notice."""
    assert SEATING_LADDER == (
        ("pcb",         0.0008),
        ("bay_proxy",   0.0009),
        ("tape",        0.0011),
        ("label",       0.0011),
        ("adhesive",    0.0012),
        ("seated_cell", 0.0012),
        ("foam",        0.0022),
    )
    assert PCB_THICKNESS_M == 0.0016
    assert BAY_PROXY_RUNG == "bay_proxy"


def test_the_shipped_ladder_satisfies_its_own_invariant():
    assert assert_seating_ladder_ordered() is None


def _reordered(name: str, z: float):
    """`SEATING_LADDER` with `name` moved to `z` and the table re-sorted.

    Re-sorting on purpose. Moving a rung usually leaves the table unsorted,
    and the "not sorted as written" clause would then fire for every case
    below - which would test the table's readability seven times over and its
    OCCLUSION rule not at all. Sorting first makes each case reach the clause
    it is about."""
    L = sorted(((n, z if n == name else zz) for n, zz in SEATING_LADDER),
               key=lambda r: r[1])
    return tuple(L)


@pytest.mark.parametrize("name,z", [
    # each dropped onto the proxy's own plane: coplanar, so no longer
    # reliably occluding it - the tie-break would be the renderer's to make
    ("tape", 0.0009), ("label", 0.0009), ("adhesive", 0.0009),
    ("seated_cell", 0.0009), ("foam", 0.0009),
    # and the proxy raised onto theirs, which breaks it from the other side
    ("bay_proxy", 0.0012),
])
def test_a_rung_that_stops_clearing_the_proxy_is_caught(name, z):
    """The invariant every `placement_area` label in every dataset rests on.
    `>=` where `>` was meant is enough to lose it, so each case here is a TIE,
    not an inversion."""
    with pytest.raises(ValueError, match="STRICTLY"):
        assert_seating_ladder_ordered(_reordered(name, z))


def test_the_board_lifted_off_the_cavity_floor_is_caught():
    """`pcb` is the one rung below the proxy, and it is legal only because it
    is exactly half the board's thickness - which is what rests the board ON
    the floor. One ULP is enough: the coupling has no tolerance of its own."""
    for z in (0.00085, math.nextafter(0.0008, 1.0),
              math.nextafter(0.0008, 0.0)):
        L = tuple((n, z if n == "pcb" else zz) for n, zz in SEATING_LADDER)
        with pytest.raises(ValueError, match="built about its CENTRE"):
            assert_seating_ladder_ordered(L)


def test_swapping_two_rungs_is_caught():
    """The failure the ordering exists to prevent, stated directly: foam
    seated below the proxy would stop occluding it."""
    L = list(SEATING_LADDER)
    L[1], L[6] = L[6], L[1]
    with pytest.raises(ValueError, match="out of order"):
        assert_seating_ladder_ordered(tuple(L))


def test_the_ordering_check_pins_the_ORDER_and_not_the_VALUES():
    """Stated rather than left to be discovered. Nudging `foam` up by one ULP
    leaves every ordering relation intact, so `assert_seating_ladder_ordered`
    passes - and should: an ordering check that also pinned values would be
    two checks wearing one name, and the one that fired would not say which
    property broke.

    The VALUES are pinned by
    `test_the_seating_ladder_is_exactly_the_six_offsets_it_has_always_been`
    above. That is the test that fails when the ground truth of every rendered
    dataset changes; this one is the test that fails when the mechanism does.
    Reading the ordering check as protection against a retuned offset is the
    misreading this docstring exists to prevent."""
    L = list(SEATING_LADDER)
    L[6] = (L[6][0], math.nextafter(L[6][1], 1.0))
    assert tuple(L) != SEATING_LADDER
    assert assert_seating_ladder_ordered(tuple(L)) is None


def test_ties_between_rungs_on_the_same_side_stay_legal():
    """tape/label and adhesive/seated_cell are genuinely coplanar pairs.
    Coplanar objects never occlude EACH OTHER and nothing asks them to, so a
    check that forbade all ties would forbid the shipped table."""
    assert [z for _, z in SEATING_LADDER].count(0.0011) == 2
    assert [z for _, z in SEATING_LADDER].count(0.0012) == 2
    assert_seating_ladder_ordered()


def test_a_duplicated_rung_name_is_caught():
    """`seat_offset` scans the table in order, so a duplicate would silently
    resolve to whichever copy came first."""
    L = list(SEATING_LADDER) + [("foam", 0.0025)]
    with pytest.raises(ValueError, match="twice"):
        assert_seating_ladder_ordered(tuple(L))


def test_a_ladder_with_no_proxy_rung_is_caught():
    L = [(n, z) for n, z in SEATING_LADDER if n != BAY_PROXY_RUNG]
    with pytest.raises(ValueError, match="exactly one"):
        assert_seating_ladder_ordered(tuple(L))


def test_a_stand_off_rather_than_a_clearance_is_caught():
    """A millimetre-scale gap under a cell is a modelling error dressed up as
    a z-fighting fix, and it is visible from overhead against an 18mm cell."""
    L = [(n, 0.02) if n == "foam" else (n, z) for n, z in SEATING_LADDER]
    with pytest.raises(ValueError):
        assert_seating_ladder_ordered(tuple(L))
    assert all(0.0 < z < MAX_SEAT_OFFSET_M for _, z in SEATING_LADDER)


def test_the_board_rests_on_the_floor_and_its_top_clears_the_proxy():
    """The coupling that makes `pcb` the one rung BELOW the proxy legal. The
    board is built about its centre, so its seat must be half its thickness -
    otherwise it floats above the cavity floor or sinks through it - and its
    top face must clear the proxy, or along the edge the board and the
    placement rectangle share, the proxy would occlude the board."""
    import recog.synth3d.bay as _bay
    assert seat_offset("pcb") == PCB_THICKNESS_M / 2.0
    assert seat_offset("pcb") + PCB_THICKNESS_M / 2.0 \
        > seat_offset(BAY_PROXY_RUNG)

    saved = _bay.PCB_THICKNESS_M
    try:
        # A board only as thick as the proxy's own offset rests on the floor
        # correctly and still fails: its top face no longer clears the plane.
        _bay.PCB_THICKNESS_M = 2 * seat_offset("pcb")
        assert_seating_ladder_ordered()          # unchanged: 0.0016
        _bay.PCB_THICKNESS_M = 0.0009
        with pytest.raises(ValueError):
            assert_seating_ladder_ordered()
    finally:
        _bay.PCB_THICKNESS_M = saved
    assert_seating_ladder_ordered()


def test_seat_z_is_the_floor_plus_the_rung_to_the_last_bit():
    """`floor_z + offset`, not a re-derivation. world.py used to add these as
    literals at the call site; a difference of one ULP is still a different
    rendered scene, so the arithmetic has to be the same arithmetic."""
    for floor in (0.0, 0.00195, 0.0045, 0.012, 0.03):
        for name, z in SEATING_LADDER:
            assert seat_z(floor, name).hex() == float(floor + z).hex()


def test_seat_z_rechecks_the_ladder_on_every_call():
    """A table nobody validates is a table nobody validates. `seat_z` is the
    only way world.py reaches these numbers, so checking there means a
    perturbed ladder stops the FIRST build rather than the first person to
    look at pixels months later."""
    import recog.synth3d.bay as _bay
    saved = _bay.SEATING_LADDER
    try:
        L = list(saved)
        L[1], L[6] = L[6], L[1]
        _bay.SEATING_LADDER = tuple(L)
        with pytest.raises(ValueError):
            _bay.seat_z(0.012, "foam")
    finally:
        _bay.SEATING_LADDER = saved
    assert seat_z(0.012, "foam") == pytest.approx(0.0142)


def test_an_unknown_rung_raises_rather_than_defaulting_to_the_floor():
    """A guarded `.get(...)` on a renamed key is how this project once had a
    builder quietly stop building geometry. Here it would seat the object on
    the floor itself, where it would z-fight with the tray and stop occluding
    the proxy - the exact failure the ladder exists to prevent."""
    with pytest.raises(ValueError, match="not a rung"):
        seat_offset("bay-proxy")
    with pytest.raises(ValueError, match="not a rung"):
        seat_z(0.012, "pcb_board")


@pytest.mark.parametrize("name,expected", [
    ("pcb", False), ("bay_proxy", False), ("tape", True), ("label", True),
    ("adhesive", True), ("seated_cell", True), ("foam", True)])
def test_occludes_bay_proxy_names_the_rungs_that_subtract_from_the_label(
        name, expected):
    assert occludes_bay_proxy(name) is expected


# ------------------------------------------- the module board's fallback ----

def test_fallback_module_placement_matches_the_rectangle_world_py_drew():
    """Moved verbatim out of `world.build_pcb`. The values, and the ORDER the
    four draws are taken in, are what has to be preserved: `build_pcb` shares
    one rng with the rest of the scene, so a reordering silently resamples the
    board colour, the component count and everything after them."""
    bounds = (-0.04, -0.09, 0.04, 0.09)
    r = _random.Random(7)
    cx, cy, w, h = fallback_module_placement(bounds, r)

    ref = _random.Random(7)
    w_ref = (0.04 - -0.04) * ref.uniform(0.55, 0.80)
    h_ref = (0.09 - -0.09) * ref.uniform(0.20, 0.38)
    cx_ref = (-0.04 + 0.04) / 2 + ref.uniform(-0.004, 0.004)
    cy_ref = (-0.09 + 0.09) / 2 + ref.uniform(-0.010, 0.010)
    assert (cx, cy, w, h) == (cx_ref, cy_ref, w_ref, h_ref)
    assert r.getstate() == ref.getstate(), "four draws, in that order"


def test_the_fallback_board_always_lands_inside_the_bounds_it_was_given():
    """A board hanging off the cartridge would be sitting on the backdrop, and
    `build_pcb`'s board is not occlusion-only furniture - scene.py gives it a
    real class id and an `electronics_module` id_meta entry."""
    bounds = (-0.04, -0.09, 0.04, 0.09)
    x0, y0, x1, y1 = bounds
    for seed in range(500):
        cx, cy, w, h = fallback_module_placement(bounds, _random.Random(seed))
        assert x0 <= cx - w / 2 and cx + w / 2 <= x1
        assert y0 <= cy - h / 2 and cy + h / 2 <= y1
        assert 0.55 * (x1 - x0) <= w <= 0.80 * (x1 - x0)
        assert 0.20 * (y1 - y0) <= h <= 0.38 * (y1 - y0)


def test_the_fallback_returns_the_same_tuple_shape_as_the_anchored_path():
    """`build_pcb` consumes the two interchangeably - both are
    `(cx, cy, w, h)` - so a fallback returning `(w, h, cx, cy)` would build a
    board the size of its own jitter, at the jitter's position, with no error
    anywhere. Checked by giving both a frame centred on the origin: a centre
    then has to be near zero and a size cannot be."""
    interior, bay_mm = (-30.0, -50.0, 30.0, 50.0), (-30.0, 30.0, 30.0, 50.0)
    anchored = module_world_placement(interior, bay_mm, 0.0, (0.0, 0.0))
    fallback = fallback_module_placement((-0.04, -0.09, 0.04, 0.09),
                                         _random.Random(0))
    for cx, cy, w, h in (anchored, fallback):
        assert abs(cx) <= 0.045 and abs(cy) <= 0.045, "first two are a centre"
        assert w > 0.02 and h > 0.01, "last two are a size"


# ------------------------------------------- the obstructions' third axis ---

@pytest.mark.parametrize("kind,lo,hi", [
    ("adhesive", 0.35, 0.7),        # dimensionless squash of a w/2 sphere
    ("foam", 0.002, 0.005),         # a thickness in metres, on a unit cube
])
def test_obstruction_z_scale_stays_in_the_range_world_py_drew_from(kind, lo,
                                                                  hi):
    for seed in range(400):
        v = obstruction_z_scale(kind, _random.Random(seed))
        assert lo <= v <= hi


@pytest.mark.parametrize("kind", ["tape", "label"])
def test_a_flat_obstruction_takes_no_draw_at_all(kind):
    """Part of the contract, not an implementation detail: world.
    build_obstructions shares one rng with the rest of the scene, so a draw
    taken for a kind that used to take none silently resamples everything
    downstream of it."""
    r = _random.Random(3)
    before = r.getstate()
    assert obstruction_z_scale(kind, r) == 1.0
    assert r.getstate() == before


@pytest.mark.parametrize("kind", ["adhesive", "foam"])
def test_a_solid_obstruction_takes_exactly_one_draw(kind):
    r = _random.Random(3)
    obstruction_z_scale(kind, r)
    ref = _random.Random(3)
    ref.random()
    assert r.getstate() == ref.getstate()


def test_an_unknown_obstruction_kind_stops_the_build():
    """`world.build_obstructions`' shape dispatch ends in a bare
    `else:  # label`, so a fifth kind added to `sample_obstructions` without a
    matching branch there would render as a printed label - silently and
    plausibly. This is called first, so it stops instead."""
    with pytest.raises(ValueError, match="not an obstruction kind"):
        obstruction_z_scale("gasket", _random.Random(0))


def test_every_kind_sample_obstructions_produces_has_a_z_scale():
    """The two halves of an obstruction's geometry are decided in two
    functions; a kind known to one and not the other is the desync this
    pairing exists to rule out."""
    cfg = Config().obstruction
    kinds = set()
    for seed in range(600):
        for p in sample_obstructions((-0.03, -0.02, 0.03, 0.02), cfg,
                                     _random.Random(seed)):
            kinds.add(p.kind)
    assert kinds == {"adhesive", "foam", "tape", "label"}
    for k in kinds:
        assert obstruction_z_scale(k, _random.Random(0)) > 0.0
