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
