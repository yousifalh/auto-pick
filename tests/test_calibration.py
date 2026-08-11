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


# ------------------------------------------- the receipts' own tooling --
#
# `recog.seg_evaluate` and `recog.calibrate_tau` publish millimetres into
# docs/receipts/, and both converted at the nominal 0.625 until
# 2026-08-11 - understating every millimetre figure they shipped by a
# median 1.37x, including the 0.949 mm bay boundary displacement quoted
# in the FDR and NEXT_STEPS as Plan C's headline result. These tests fail
# if either tool goes back to a constant.

_W = _H = 40


def _rle(mask):
    from recog.synth3d.annotate import rle_encode
    return rle_encode(mask)


def _stub_dataset(tmp_path, ortho_scales, with_sidecar=True):
    """A real on-disk split: PNG frames, COCO-RLE annotations, and (by
    default) a render sidecar per frame carrying its own ortho_scale.

    Only what `evaluate` and `collect_records` actually touch, so these
    tests exercise the real functions rather than a re-implementation.
    """
    import numpy as np
    from PIL import Image

    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    truth = np.zeros((_H, _W), dtype=np.uint8)
    truth[10:30, 10:30] = 1                     # the ground-truth bay

    samples = []
    for i, ortho in enumerate(ortho_scales):
        stem = f"scene_{i:05d}"
        Image.fromarray(np.zeros((_H, _W, 3), np.uint8)).save(
            root / "images" / f"{stem}.png")
        if with_sidecar:
            (root / "meta" / f"{stem}.json").write_text(
                json.dumps({"camera": {"ortho_scale": ortho},
                            "width": _W, "height": _H}), encoding="utf-8")
        samples.append(({"file_name": f"{stem}.png"},
                        [{"class": "placement_area",
                          "segmentation": _rle(truth)}],
                        (0, 0, _W, _H)))

    class _Dataset:
        img_dir = str(root / "images")

    ds = _Dataset()
    ds.samples = samples
    ds.sample_assets = [None] * len(samples)
    return ds


class _StubSegmenter:
    """Predicts a bay shifted two pixels right of the truth, identically
    on every crop - so any difference between two runs is the scale, not
    the model. Shifted rather than merely shrunk so BOTH signed area
    errors are non-zero: two columns of truth are missed (conservative)
    and two columns outside it are claimed (optimistic)."""

    def segment_batch(self, crops):
        import numpy as np
        pred = np.zeros((_H, _W), dtype=np.int64)
        pred[10:30, 12:32] = 2                  # SEG_CHANNELS["bay"]
        return [pred for _ in crops]


def test_seg_evaluate_reads_each_crops_scale_from_its_own_frame(tmp_path):
    from recog.seg_evaluate import resolve_frame_scales

    ds = _stub_dataset(tmp_path, [_W * 0.4915 / 1000.0, _W * 1.0915 / 1000.0])
    scales, provenance = resolve_frame_scales(ds, [0, 1])

    assert scales[0] == pytest.approx(0.4915)
    assert scales[1] == pytest.approx(1.0915)
    assert provenance == {"n_measured": 2, "n_fallback": 0,
                          "n_frames": 2, "fallback": None}


def test_seg_evaluate_millimetres_move_with_the_scale_but_iou_does_not(
        tmp_path):
    """THE regression, as a test.

    The same pixels, the same predictions, scored twice: once at each
    frame's own ground sample distance and once at the nominal 0.625 the
    receipts used to convert at. Every MILLIMETRE figure must differ.
    Every PIXEL-space figure - IoU, crop counts - must be bit-identical,
    because it never multiplies by this number; if IoU moves, the change
    has done something other than fix the scale.
    """
    from recog.seg_evaluate import evaluate, resolve_frame_scales

    ds = _stub_dataset(tmp_path, [_W * 0.4915 / 1000.0, _W * 1.0915 / 1000.0])
    scales, _ = resolve_frame_scales(ds, [0, 1])

    per_frame = evaluate(_StubSegmenter(), ds, [0, 1], scales)
    constant = evaluate(_StubSegmenter(), ds, [0, 1], {0: 0.625, 1: 0.625})

    # `== pytest.approx(..., nan_ok=True)` because an absent class scores
    # NaN by design (per_class_iou's docstring) and NaN != NaN.
    assert per_frame["ious"] == pytest.approx(constant["ious"], nan_ok=True)
    assert per_frame["instance_counts"] == constant["instance_counts"]

    # The true GSDs here average 0.7915, so the millimetres go UP - the
    # direction that matters, because the published figures were the
    # understated ones.
    assert (per_frame["boundary_mm"]["bay"]
            > constant["boundary_mm"]["bay"] * 1.2)
    # Area is a SQUARE of the scale, so it moves further than the length.
    assert (per_frame["area_opt_mm2_mean"]["bay"]
            > constant["area_opt_mm2_mean"]["bay"] * 1.4)
    assert per_frame["mm_per_px"]["median"] != pytest.approx(0.625)


def test_the_mask_head_comparison_is_scale_invariant(tmp_path):
    """The architecture verdict must not move when the calibration does.

    Boundary displacement and a 28x28 head's quantisation are both pixel
    counts times mm_per_px, so their RATIO - the whole content of the
    §13.2.1 argument - is a property of the model, not of the framing.
    Judging the corrected millimetres against the 2.9 mm figure frozen at
    the nominal 0.625 would flip that verdict on scale alone.
    """
    from recog.seg_evaluate import (MASK_HEAD_QUANTISATION_PX, evaluate,
                                    resolve_frame_scales)

    ds = _stub_dataset(tmp_path, [_W * 0.4915 / 1000.0, _W * 1.0915 / 1000.0])
    scales, _ = resolve_frame_scales(ds, [0, 1])

    per_frame = evaluate(_StubSegmenter(), ds, [0, 1], scales)
    constant = evaluate(_StubSegmenter(), ds, [0, 1], {0: 0.625, 1: 0.625})

    def clears_by(res):
        return (MASK_HEAD_QUANTISATION_PX[0]
                * res["boundary_mm_per_px"]["bay"]) / res["boundary_mm"]["bay"]

    assert clears_by(per_frame) == pytest.approx(clears_by(constant))


def test_seg_evaluate_refuses_one_constant_for_the_whole_split():
    """The signature itself must not accept the thing that went wrong."""
    from recog.seg_evaluate import evaluate

    with pytest.raises(TypeError, match="per-crop"):
        evaluate(_StubSegmenter(), None, [], 0.625)


def test_an_uncalibrated_frame_raises_rather_than_reverting_to_a_constant(
        tmp_path):
    from plan.placement_area import UnknownScale
    from recog.seg_evaluate import resolve_frame_scales

    ds = _stub_dataset(tmp_path, [0.8], with_sidecar=False)
    with pytest.raises(UnknownScale, match="no fallback"):
        resolve_frame_scales(ds, [0])


def test_a_deliberate_fallback_is_honoured_and_recorded_as_one(tmp_path):
    """A real fixed-mount camera IS one calibrated scale. The fallback
    serves it - but the receipt has to be able to tell that run apart
    from a per-frame-calibrated one, which is precisely what the old
    single-number header could not do."""
    from recog.seg_evaluate import resolve_frame_scales

    ds = _stub_dataset(tmp_path, [0.8], with_sidecar=False)
    scales, provenance = resolve_frame_scales(ds, [0], fallback=0.38)

    assert scales[0] == pytest.approx(0.38)
    assert provenance["n_measured"] == 0
    assert provenance["n_fallback"] == 1
    assert provenance["fallback"] == pytest.approx(0.38)


def test_calibrate_tau_erodes_the_wall_inset_per_frame(tmp_path):
    """Not only a reporting fix here. The 4.25 mm wall inset and the
    18650's footprint are converted to pixels and then FED to the
    measurement - `arbitrate`'s erosion radius and `admits_a_cell`'s
    structuring element - so a constant scale eroded too deep and tested
    with a cell of the wrong size on every frame at once.
    """
    from recog.calibrate_tau import collect_records
    from recog.seg_evaluate import resolve_frame_scales

    ds = _stub_dataset(tmp_path, [_W * 0.4915 / 1000.0, _W * 1.0915 / 1000.0])
    scales, _ = resolve_frame_scales(ds, [0, 1])
    records = collect_records(_StubSegmenter(), ds, [0, 1], scales,
                              wall_inset_mm=4.25, cell_w_mm=18.3,
                              cell_h_mm=65.0)

    assert [r["mm_per_px"] for r in records] == [pytest.approx(0.4915),
                                                 pytest.approx(1.0915)]
    # 4.25/0.4915 -> 9 px; 4.25/1.0915 -> 4 px. One constant cannot be
    # both, and 0.625 gives 7 for each.
    assert records[0]["wall_inset_px"] == 9
    assert records[1]["wall_inset_px"] == 4
    assert records[0]["cell_px"][1] > records[1]["cell_px"][1]


def test_calibrate_tau_refuses_one_constant_for_the_whole_split():
    from recog.calibrate_tau import collect_records

    with pytest.raises(TypeError, match="per-crop"):
        collect_records(_StubSegmenter(), None, [], 0.625, 4.25, 18.3, 65.0)
