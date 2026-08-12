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


# ----------------------------------- delta_cells, the end-to-end number --
#
# `recog.seg_ablation` was in NEITHER of the two passes that took the
# nominal constant out of `seg_evaluate` and `calibrate_tau`, and it is
# the tool that publishes the figure FDR 13.2.1 calls "the figure that
# matters for safety". Here the scale is not a reporting unit at all: it
# sets the wall-inset erosion RADIUS, the strip's size in millimetres and
# the occupancy grid's stride, so it changes what the packer does rather
# than how the answer is labelled.
#
# audit/2026-08-12-A-measurement-tools.md section 7 asks for
# "`_pack_count` at two different mm_per_px on one label map must not
# change the conclusion". The first test below is that test, with one
# correction it needs to be true: `_pack_count` CANNOT be invariant to
# mm_per_px, and should not be. The same pixel rectangle at a larger GSD
# is a larger PHYSICAL rectangle, and 18650s are packed in millimetres -
# so the count must rise, and an assertion that the two counts agree
# asserts something physically false. What the caveat on record claimed,
# and what is genuinely testable, is that the CONCLUSION survives the
# wrong scale. It does not, and that is what this pins.

_A_W, _A_H = 131, 288          # the PowerCore26800 crop shape FDR 13.2.1 uses
_NOMINAL = 0.6250              # the generator's framing at margin=1, zoom=1
_GSD_LO = 0.4903               # the 126-crop val split's measured extremes
_GSD_MED = 0.8211              # ... and its median true GSD
_PCB_ROWS = 20


def _damage_case():
    """``(gt, pred)``: a bay with a PCB across its top, and a prediction
    that calls that PCB placeable. The damage direction, by construction.
    """
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS

    gt = np.zeros((_A_H, _A_W), np.int8)
    gt[5:_A_H - 5, 5:_A_W - 5] = CH_CARTRIDGE
    gt[12:_A_H - 12, 12:_A_W - 12] = CH_BAY
    gt[12:12 + _PCB_ROWS, 12:_A_W - 12] = CH_ELECTRONICS

    pred = gt.copy()
    pred[12:12 + _PCB_ROWS, 12:_A_W - 12] = CH_BAY
    return gt, pred


def test_the_pack_count_conclusion_is_not_invariant_to_mm_per_px():
    """The audit's headline test, and the one that falsifies the caveat.

    `specs/2026-08-11-scale-calibration.md` section 5 filed the nominal
    scale in seg_ablation as a MAGNITUDE caveat: "both sides are at the
    same wrong scale so the sign is trustworthy, but the magnitude is
    compressed". Packing is discrete and non-monotone in scale - it is
    not a multiplicative rescaling that cancels in a difference - so the
    sign is not trustworthy either. ONE label map, ONE prediction, two
    scales, and the conclusion changes from "the prediction and the truth
    agree" to "the prediction packed a cell where the truth forbids it".
    """
    from recog.seg_ablation import _pack_count, delta_cells

    gt, pred = _damage_case()

    # Not invariant, and correctly so: at a coarser GSD the same pixels
    # are more millimetres, and more fixed-size cells fit.
    assert _pack_count(gt, _NOMINAL) < _pack_count(gt, _GSD_MED)

    # The conclusion, which IS supposed to survive, does not.
    assert delta_cells(gt, pred, _NOMINAL) == 0        # "no difference"
    assert delta_cells(gt, pred, _GSD_LO) == 0         # ... at the low end too
    assert delta_cells(gt, pred, _GSD_MED) < 0         # a damage-risk event


def _ablation_dataset(tmp_path, gsds):
    """One frame per GSD, each carrying its own render sidecar, each
    holding the `_damage_case` geometry as ground truth."""
    import numpy as np
    from PIL import Image

    from recog.synth3d.annotate import rle_encode

    gt, _pred = _damage_case()

    root = tmp_path / "ablation_ds"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    def _rle_of(cls):
        return rle_encode((gt == cls).astype(np.uint8))

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS

    samples = []
    for i, gsd in enumerate(gsds):
        stem = f"scene_{i:05d}"
        Image.fromarray(np.zeros((_A_H, _A_W, 3), np.uint8)).save(
            root / "images" / f"{stem}.png")
        (root / "meta" / f"{stem}.json").write_text(
            json.dumps({"camera": {"ortho_scale": _A_W * gsd / 1000.0},
                        "width": _A_W, "height": _A_H}), encoding="utf-8")
        samples.append((
            {"file_name": f"{stem}.png"},
            [{"class": "cartridge", "segmentation": _rle_of(CH_CARTRIDGE)},
             {"class": "placement_area", "segmentation": _rle_of(CH_BAY)},
             {"class": "electronics_module",
              "segmentation": _rle_of(CH_ELECTRONICS)}],
            (0, 0, _A_W, _A_H)))

    class _Dataset:
        img_dir = str(root / "images")

    ds = _Dataset()
    ds.samples = samples
    ds.sample_assets = [None] * len(samples)
    return ds


class _OptimisticSegmenter:
    """Predicts the PCB as placeable - `_damage_case`'s prediction,
    identically on every crop, so any difference between two runs is the
    scale and not the model."""

    def segment_batch(self, crops):
        _gt, pred = _damage_case()
        return [pred.astype("int64") for _ in crops]


def test_seg_ablation_refuses_one_constant_for_the_whole_split():
    """The signature itself must not accept the thing that went wrong -
    the same guard `evaluate` and `collect_records` already carry, on the
    third tool that was left behind."""
    from recog.seg_ablation import evaluate_delta_cells

    with pytest.raises(TypeError, match="per-crop"):
        evaluate_delta_cells(_OptimisticSegmenter(), None, [], 0.625)


def test_seg_ablation_measures_each_crop_at_its_own_frames_scale(tmp_path):
    """THE regression, at the level the receipt publishes.

    Two frames, same pixels, same prediction, differing only in the GSD
    their own sidecars record. Scored at the nominal 0.6250 the split
    reports ZERO damage-direction crops; scored at the frames' own scales
    it reports one - and the receipt's whole safety claim is that count.
    """
    from recog.seg_ablation import evaluate_delta_cells
    from recog.seg_evaluate import resolve_frame_scales

    ds = _ablation_dataset(tmp_path, [_GSD_LO, _GSD_MED])
    scales, provenance = resolve_frame_scales(ds, [0, 1])
    assert provenance["n_measured"] == 2 and provenance["n_fallback"] == 0

    per_frame = evaluate_delta_cells(_OptimisticSegmenter(), ds, [0, 1], scales)
    constant = evaluate_delta_cells(_OptimisticSegmenter(), ds, [0, 1],
                                    {0: _NOMINAL, 1: _NOMINAL})

    assert [r["mm_per_px"] for r in per_frame["rows"]] == [
        pytest.approx(_GSD_LO, abs=1e-3), pytest.approx(_GSD_MED, abs=1e-3)]

    assert constant["n_negative"] == 0        # what the shipped receipt said
    assert per_frame["n_negative"] == 1       # what the frames actually say
    assert per_frame["mm_per_px"]["median"] != pytest.approx(_NOMINAL)


def test_seg_ablation_counts_val_instances_at_the_checkpoints_crop_size(
        tmp_path, monkeypatch):
    """`compute_val_instance_counts`' docstring says `out_size` MUST be
    the checkpoint's `model.crop_size` whenever the result is fed to
    `check_split_matches_checkpoint`. `seg_evaluate` and `calibrate_tau`
    obey it; `seg_ablation` did not, so its guard compared native counts
    against 256-rasterised ones and refused to run on any config where a
    single sliver of background disappears in the downsample - which is
    every non-default config in this repo.

    A CLI smoke test, so it also covers what all three of this module's
    historical breakages had in common: they were `main()`-only.
    """
    import numpy as np
    torch = pytest.importorskip("torch")
    import yaml
    from PIL import Image

    from recog.synth3d.annotate import rle_encode

    import recog.bay_segmenter as bay_segmenter
    import recog.seg_ablation as seg_ablation
    import recog.seg_evaluate as seg_evaluate

    # A unit whose crop is fully covered EXCEPT one background pixel -
    # the real split's own failure mode in miniature. It survives at
    # native resolution and is lost to the nearest-neighbour downsample,
    # so `background` is present natively and absent at crop_size.
    cart = np.ones((_A_H, _A_W), np.uint8)
    cart[1, 1] = 0
    bay = np.zeros((_A_H, _A_W), np.uint8)
    bay[12:_A_H - 12, 12:_A_W - 12] = 1
    pcb = np.zeros((_A_H, _A_W), np.uint8)
    pcb[12:12 + _PCB_ROWS, 12:_A_W - 12] = 1

    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    for i in range(4):
        stem = f"scene_{i:05d}"
        Image.fromarray(np.zeros((_A_H, _A_W, 3), np.uint8)).save(
            root / "images" / f"{stem}.png")
        (root / "meta" / f"{stem}.json").write_text(
            json.dumps({"camera": {"ortho_scale": _A_W * _GSD_MED / 1000.0},
                        "width": _A_W, "height": _A_H}), encoding="utf-8")
        images.append({"id": i, "file_name": f"{stem}.png",
                       "width": _A_W, "height": _A_H})
        for cid, mask in ((2, cart), (1, bay), (3, pcb)):
            annotations.append({
                "id": len(annotations) + 1, "image_id": i, "category_id": cid,
                "bbox": [0, 0, _A_W, _A_H], "area": int(mask.sum()),
                "segmentation": rle_encode(mask), "iscrowd": 0,
                "unit_id": f"item{i}"})

    coco_path = root / "instances_seg.json"
    coco_path.write_text(json.dumps({
        "categories": [{"id": 1, "name": "placement_area"},
                       {"id": 2, "name": "cartridge"},
                       {"id": 3, "name": "electronics_module"}],
        "images": images, "annotations": annotations}), encoding="utf-8")

    crop_size = 64
    config_path = tmp_path / "seg.yaml"
    config_path.write_text(yaml.safe_dump({
        "model": {"num_classes": 6, "crop_size": crop_size, "half": False},
        "dataset": {"coco_path": str(coco_path),
                    "img_dir": str(root / "images"),
                    "jitter_frac": 0.0, "split_seed": 0,
                    "train_val_split": 0.5}}), encoding="utf-8")

    # The checkpoint records what seg_training would have recorded: counts
    # off a DataLoader that rasterises at crop_size, NOT at native. That
    # is the whole point - a receipt tool that recounts natively compares
    # two different quantities.
    from recog.seg_dataset import BaySegDataset
    from recog.seg_training import _split_dataset

    ds = BaySegDataset(str(coco_path), str(root / "images"), out_size=crop_size,
                       jitter_frac=0.0, train=True, transform=None)
    _train, val = _split_dataset(ds, 0.5, seed=0)
    val_indices = list(val.indices)
    recorded = seg_evaluate.compute_val_instance_counts(
        ds, val_indices, num_classes=6, out_size=crop_size)
    native = seg_evaluate.compute_val_instance_counts(
        ds, val_indices, num_classes=6)
    # The fixture is only a test of anything if the two disagree.
    assert native["background"] != recorded["background"]

    ckpt_path = tmp_path / "best.pt"
    torch.save({"val_instance_counts": recorded, "coco_path": str(coco_path)},
               ckpt_path)

    class _Ckpt:
        device = "cpu"

        def __init__(self, **kw):
            pass

        def segment_batch(self, crops):
            _g, pred = _damage_case()
            return [pred.astype("int64") for _ in crops]

    monkeypatch.setattr(bay_segmenter, "BaySegmenter", _Ckpt)
    monkeypatch.setattr(seg_ablation, "heuristic_vs_segmenter",
                        lambda *a, **k: _EMPTY_REAL_RESULT)

    out = tmp_path / "receipt.txt"
    rc = seg_ablation.main([
        "--checkpoint", str(ckpt_path), "--config", str(config_path),
        "--device", "cpu", "--out", str(out)])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    # It reached a measurement rather than dying at the guard, and it did
    # NOT print one constant for the split.
    assert "validation crops" in text
    assert "mm_per_px=0.6250" not in text


_EMPTY_REAL_RESULT = {
    "n_cartridges": 0, "n_images": 0, "n_annotations_total": 0,
    "has_ground_truth_polygons": False, "wall_inset_mm": 4.25, "rows": [],
    "heuristic": {"mean": float("nan"), "median": float("nan"),
                  "n_zero": 0, "n_warned": 0},
    "segmenter": {"mean": float("nan"), "median": float("nan"), "n_zero": 0},
    "baseline_mean": 0.218, "baseline_n_zero": 7, "baseline_n": 20,
    "beats_baseline_mean": False, "has_zero_area_cartridges": False,
}
