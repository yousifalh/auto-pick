"""recog.calibration - millimetres per pixel, nominal and per-frame.

The distinction these tests pin is the one that cost 9 cells and put 3
placements onto ground-truth non-floor material: the generator's NOMINAL
framing and an individual frame's TRUE ground sample distance are
different numbers on a corpus that randomises margin and zoom, and the
planner needs the second one.
"""
from __future__ import annotations

import json

import pytest

from recog.calibration import (frame_mm_per_px, frame_mm_per_px_for_image,
                               mm_per_px_from_extent, resolve_mm_per_px)


# ---------------------------------------------------------- arithmetic --

def test_mm_per_px_from_extent_is_metres_to_millimetres_over_pixels():
    assert mm_per_px_from_extent(0.8, 1280) == pytest.approx(0.625)


def test_zero_width_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="positive"):
        mm_per_px_from_extent(0.8, 0)


# ------------------------------------------------------------- nominal --

def test_resolve_mm_per_px_reads_the_generator_framing():
    cfg = {"layout": {"area": [0.8, 0.45]}, "render": {"res": [1280, 720]}}
    assert resolve_mm_per_px(cfg) == pytest.approx(0.625)


def test_seg_evaluate_re_exports_the_same_function_not_a_second_copy():
    """One definition, imported twice - not two definitions that agree
    today. `recog.calibrate_tau` and `recog.seg_ablation` both import
    this name from `recog.seg_evaluate` and quote the result in shipped
    receipts; a divergent second copy would move those receipts without
    anything failing.
    """
    import recog.calibration as calibration
    import recog.seg_evaluate as seg_evaluate

    assert seg_evaluate.resolve_mm_per_px is calibration.resolve_mm_per_px
    assert seg_evaluate.load_synth_config is calibration.load_synth_config


# ----------------------------------------------------------- per-frame --

def test_frame_mm_per_px_is_ortho_scale_over_render_width():
    meta = {"camera": {"ortho_scale": 0.9510}, "width": 1280, "height": 720}
    assert frame_mm_per_px(meta) == pytest.approx(0.9510 * 1000 / 1280)


def test_a_zoomed_frame_does_not_have_the_nominal_scale():
    """THE defect, as a test.

    `recog/synth3d/world.py:setup_camera` sets
    `ortho_scale = need * margin * zoom`, with margin in [1.02, 1.10] and
    zoom drawn from `param_space.zoom`. A frame rendered at zoom 1.5 has
    half again the ground sample distance of the nominal framing, and
    planning it at the nominal number under-reads every distance in the
    scene by a third. If these two ever compare equal, either the
    generator stopped randomising the framing or one of them stopped
    describing what its name says.
    """
    cfg = {"layout": {"area": [0.8, 0.45]}, "render": {"res": [1280, 720]}}
    nominal = resolve_mm_per_px(cfg)

    margin, zoom = 1.05, 1.5
    meta = {"camera": {"ortho_scale": 0.8 * margin * zoom},
            "width": 1280, "height": 720}
    true_gsd = frame_mm_per_px(meta)

    assert true_gsd == pytest.approx(nominal * margin * zoom)
    assert true_gsd > nominal * 1.5
    # The direction that matters: planning at `nominal` would treat a
    # 65 mm cell as covering far more of the bay than it does.
    assert 65.0 / nominal > 65.0 / true_gsd


def test_a_perspective_frame_raises_rather_than_inventing_a_scale():
    """`setup_camera` leaves `ortho_scale` None for a PERSP camera, which
    has no single scalar mm_per_px at all. Substituting one would be a
    fabricated calibration."""
    meta = {"camera": {"ortho_scale": None, "focal": 50}, "width": 1280}
    with pytest.raises(ValueError, match="ortho_scale"):
        frame_mm_per_px(meta)


def test_missing_width_raises():
    with pytest.raises(ValueError, match="width"):
        frame_mm_per_px({"camera": {"ortho_scale": 0.8}})


# -------------------------------------------------------- sidecar I/O --

def _write_frame(root, stem, ortho_scale, width=1280):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    img = root / "images" / f"{stem}.png"
    img.write_bytes(b"")
    (root / "meta" / f"{stem}.json").write_text(
        json.dumps({"camera": {"ortho_scale": ortho_scale},
                    "width": width, "height": 720}),
        encoding="utf-8")
    return img


def test_sidecar_is_found_beside_the_image_directory(tmp_path):
    img = _write_frame(tmp_path / "ds", "scene_00007", 0.6291412115097046)
    assert frame_mm_per_px_for_image(img) == pytest.approx(
        0.6291412115097046 * 1000 / 1280)


def test_two_frames_of_one_dataset_can_have_different_scales(tmp_path):
    """Not a hypothetical: over recog/dataset3d_seg the true GSD runs
    0.490-1.045 mm/px. A per-DATASET scale cannot describe that, which is
    why the calibration is read per frame."""
    root = tmp_path / "ds"
    a = _write_frame(root, "scene_00005", 0.6291)
    b = _write_frame(root, "scene_00052", 1.3377)
    assert frame_mm_per_px_for_image(a) != frame_mm_per_px_for_image(b)


def test_no_sidecar_returns_none_rather_than_a_default(tmp_path):
    """A photograph has no render sidecar. `None` says so; it does not
    say "use 0.625". Who decides what an unknown scale is worth is the
    caller's problem, and plan.planner.Planner raises unless a fallback
    was configured on purpose."""
    (tmp_path / "images").mkdir()
    lonely = tmp_path / "images" / "IMG_4426.png"
    lonely.write_bytes(b"")
    assert frame_mm_per_px_for_image(lonely) is None


def test_a_present_but_unusable_sidecar_raises_instead_of_returning_none(
        tmp_path):
    """Missing metadata is a fallback case; BROKEN metadata is a bug.
    Collapsing the two would let a corrupted sidecar silently plan every
    frame at the fallback while the run looked healthy."""
    root = tmp_path / "ds"
    img = _write_frame(root, "scene_00000", 0.8)
    (root / "meta" / "scene_00000.json").write_text(
        json.dumps({"width": 1280}), encoding="utf-8")
    with pytest.raises(ValueError):
        frame_mm_per_px_for_image(img)
