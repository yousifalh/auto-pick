"""Bay segmenter model factory and inference wrapper."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_model_has_six_output_channels():
    from recog.bay_segmenter import build_segmenter

    m = build_segmenter(num_classes=6, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 64, 64))["out"]
    assert out.shape[1] == 6


def test_segment_batch_returns_one_map_per_crop_at_native_size():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    crops = [np.zeros((40, 90, 3), np.uint8),
             np.zeros((120, 55, 3), np.uint8)]
    out = seg.segment_batch(crops)

    assert len(out) == 2
    assert out[0].shape == (40, 90)
    assert out[1].shape == (120, 55)
    for m in out:
        assert m.dtype == np.int8
        assert int(m.max()) < 6


def test_segment_matches_segment_batch_of_one():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    crop = (np.arange(48 * 64 * 3, dtype=np.uint8) % 255).reshape(48, 64, 3)
    assert np.array_equal(seg.segment(crop), seg.segment_batch([crop])[0])


def test_empty_batch_returns_empty_list():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    assert seg.segment_batch([]) == []


def test_half_precision_is_refused_on_cpu():
    """fp16 on CPU is slower than fp32 and silently so. Better to say."""
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=True)
    assert seg.half is False


def test_baysegmenter_loads_a_checkpoint_the_training_loop_wrote(tmp_path):
    """The train -> save -> load -> infer round trip Plan D depends on.

    Training builds the model with `pretrained=True` (the aux classifier
    regularises), so its state_dict carries aux_classifier.* keys.
    BaySegmenter, given a checkpoint, builds with `pretrained=False` (no
    inference-time role for aux - see the module docstring), so those
    keys have nowhere to land and load_state_dict's strict check raises.
    None of the other tests in this file catch it: they all construct
    BaySegmenter(checkpoint=None), which never exercises loading a real
    checkpoint at all.
    """
    from recog.bay_segmenter import BaySegmenter, build_segmenter
    from recog.seg_training import checkpoint_state_dict

    # Mirrors exactly how recog.seg_training.train() builds and saves a
    # checkpoint - same constructor call, same save helper - so this test
    # exercises the production code path rather than a re-implementation
    # of it that could silently drift out of sync.
    model = build_segmenter(num_classes=6, pretrained=True)
    ckpt_path = tmp_path / "roundtrip.pt"
    torch.save({"model": checkpoint_state_dict(model)}, ckpt_path)

    seg = BaySegmenter(checkpoint=str(ckpt_path), device="cpu",
                       crop_size=64, half=False)
    crop = np.zeros((48, 64, 3), np.uint8)
    out = seg.segment_batch([crop])

    assert out[0].shape == (48, 64)
    assert out[0].dtype == np.int8
    assert 0 <= int(out[0].min()) and int(out[0].max()) < 6


# ------------------------------------------------- batch construction --
#
# `segment_batch` builds its input differently since the audit: uint8
# assigned into the float32 batch with one divide after the loop (the old
# `uint8 / 255.0` promoted to a 1,572,864-byte float64 temporary per crop,
# 12.6 MB per 8-crop frame, immediately narrowed back), and the fp16 cast
# folded into the transfer instead of a `.half()` on the device. Neither
# is allowed to move a single value, and these are what say so.

def _batch_the_model_received(seg, crops):
    """The tensor `segment_batch` actually hands the model."""
    seen = {}
    real = seg.model

    def spy(x):
        seen["x"] = x
        return real(x)

    seg.model = spy
    try:
        seg.segment_batch(crops)
    finally:
        seg.model = real
    return seen["x"]


def _crops(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (131, 288, 3), dtype=np.uint8),
            rng.integers(0, 256, (57, 40, 3), dtype=np.uint8)]


def test_the_batch_is_bit_identical_to_the_float64_intermediate_it_replaced():
    from recog.bay_segmenter import BaySegmenter, _resize_rgb

    seg = BaySegmenter(checkpoint=None, device="cpu", crop_size=64,
                       half=False, pretrained=False)
    crops = _crops()
    got = _batch_the_model_received(seg, crops)

    before_the_change = np.empty((len(crops), 3, 64, 64), dtype=np.float32)
    for i, c in enumerate(crops):
        before_the_change[i] = _resize_rgb(c, 64).transpose(2, 0, 1) / 255.0

    assert got.dtype == torch.float32
    assert np.array_equal(got.numpy(), before_the_change), (
        "dropping the float64 intermediate is only allowed to change what "
        "is allocated, not what is computed")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the fp16 transfer path only exists on CUDA")
def test_the_fp16_batch_is_bit_identical_to_transferring_fp32_and_halving():
    """`.to(device, dtype=fp16)` halves the bytes on the bus. It must not
    change the numbers: fp16 is round-to-nearest of the same fp32 values
    whichever side of the transfer it happens on."""
    from recog.bay_segmenter import BaySegmenter, _resize_rgb

    seg = BaySegmenter(checkpoint=None, device="cuda", crop_size=64,
                       half=True, pretrained=False)
    assert seg.half is True
    crops = _crops(1)
    got = _batch_the_model_received(seg, crops)

    before_the_change = np.empty((len(crops), 3, 64, 64), dtype=np.float32)
    for i, c in enumerate(crops):
        before_the_change[i] = _resize_rgb(c, 64).transpose(2, 0, 1) / 255.0
    before = torch.from_numpy(before_the_change).to(seg.device).half()

    assert got.dtype == torch.float16
    assert torch.equal(got, before)


def test_dice_loss_is_zero_for_a_perfect_prediction():
    from recog.seg_training import dice_loss

    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :4] = 2
    logits = torch.full((2, 6, 8, 8), -20.0)
    for b in range(2):
        for y in range(8):
            for x in range(8):
                logits[b, int(target[b, y, x]), y, x] = 20.0
    assert float(dice_loss(logits, target, 6)) < 0.02


def test_dice_loss_is_large_for_an_inverted_prediction():
    from recog.seg_training import dice_loss

    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :4] = 2
    logits = torch.full((2, 6, 8, 8), -20.0)
    logits[:, 5] = 20.0                    # predict class 5 everywhere
    assert float(dice_loss(logits, target, 6)) > 0.8


def test_dice_loss_ignores_classes_absent_from_the_batch():
    """A batch with no obstruction pixels must not be penalised for
    failing to predict obstruction - otherwise 40% of batches carry a
    constant gradient toward a class that is not there."""
    from recog.seg_training import dice_loss

    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 6, 4, 4), -20.0)
    logits[:, 0] = 20.0
    assert float(dice_loss(logits, target, 6)) < 0.02


def test_signed_area_error_separates_optimistic_from_conservative():
    """Optimistic error puts a cell where one cannot go - a damage
    event. Conservative error refuses where one can - a lost cell. Only
    the first is a safety issue and a single unsigned number hides it."""
    import numpy as np

    from recog.seg_evaluate import signed_area_error_mm2

    target = np.zeros((20, 20), np.int8)
    target[5:15, 5:15] = 2                 # 100 px of true bay
    pred = np.zeros((20, 20), np.int8)
    pred[5:15, 5:17] = 2                   # 20 px too many, all optimistic

    opt, cons = signed_area_error_mm2(pred, target, cls=2, mm_per_px=0.5)
    assert opt == pytest.approx(20 * 0.25)
    assert cons == pytest.approx(0.0)


def test_signed_area_error_reports_conservative_when_under_predicting():
    import numpy as np

    from recog.seg_evaluate import signed_area_error_mm2

    target = np.zeros((20, 20), np.int8)
    target[5:15, 5:15] = 2
    pred = np.zeros((20, 20), np.int8)
    pred[5:15, 5:13] = 2                   # 20 px too few

    opt, cons = signed_area_error_mm2(pred, target, cls=2, mm_per_px=0.5)
    assert opt == pytest.approx(0.0)
    assert cons == pytest.approx(20 * 0.25)


def test_boundary_displacement_is_zero_for_an_exact_match():
    import numpy as np

    from recog.seg_evaluate import boundary_displacement_mm

    m = np.zeros((30, 30), np.int8)
    m[8:22, 8:22] = 2
    assert boundary_displacement_mm(m, m, cls=2, mm_per_px=0.63) == \
        pytest.approx(0.0)


def test_boundary_displacement_scales_with_mm_per_px():
    import numpy as np

    from recog.seg_evaluate import boundary_displacement_mm

    target = np.zeros((30, 30), np.int8)
    target[8:22, 8:22] = 2
    pred = np.zeros((30, 30), np.int8)
    pred[8:22, 8:24] = 2

    a = boundary_displacement_mm(pred, target, cls=2, mm_per_px=1.0)
    b = boundary_displacement_mm(pred, target, cls=2, mm_per_px=2.0)
    assert b == pytest.approx(2.0 * a)
    assert a > 0.0


def test_boundary_displacement_reports_a_scipy_failure_instead_of_hiding_it():
    """`except Exception` used to wrap the scipy import AND the
    distance_transform_edt call it guards, so a fault inside scipy - a
    dtype it rejects, a MemoryError, a moved signature - was answered by
    the numpy fallback and the receipt printed a number no reader could
    tell apart from a scipy-computed one. Missing scipy is a
    configuration; scipy raising is a fault, and this module's whole
    argument is that silent degradation is what must not happen."""
    import scipy.ndimage

    from recog import seg_evaluate

    target = np.zeros((30, 30), np.int8)
    target[8:22, 8:22] = 2
    pred = np.zeros((30, 30), np.int8)
    pred[8:22, 8:24] = 2

    def boom(_x):
        raise RuntimeError("scipy is unwell")

    saved = scipy.ndimage.distance_transform_edt
    scipy.ndimage.distance_transform_edt = boom
    try:
        with pytest.raises(RuntimeError, match="scipy is unwell"):
            seg_evaluate.boundary_displacement_mm(pred, target, cls=2,
                                                  mm_per_px=0.63)
    finally:
        scipy.ndimage.distance_transform_edt = saved


def test_boundary_displacement_falls_back_when_scipy_is_absent(monkeypatch):
    """The fallback is still reached when scipy genuinely is not
    installed, and it must agree with scipy to the last bit - it is the
    same Euclidean distance, computed the long way."""
    import builtins

    from recog import seg_evaluate

    target = np.zeros((30, 30), np.int8)
    target[8:22, 8:22] = 2
    pred = np.zeros((30, 30), np.int8)
    pred[8:22, 8:24] = 2
    with_scipy = seg_evaluate.boundary_displacement_mm(pred, target, cls=2,
                                                       mm_per_px=0.63)

    real_import = builtins.__import__

    def no_scipy(name, *a, **kw):
        if name.startswith("scipy"):
            raise ImportError("no scipy here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without_scipy = seg_evaluate.boundary_displacement_mm(pred, target, cls=2,
                                                          mm_per_px=0.63)

    assert without_scipy == with_scipy
    assert with_scipy > 0.0, "fixture must have a non-zero displacement"


def test_the_scipy_free_distance_field_is_exact_at_every_chunk_size():
    """The fallback is chunked over boundary pixels because the one-shot
    (H, W, N) form it replaces cost ~302 MB per temporary and ~1 GB peak
    for a 131x288 crop with ~1000 boundary pixels, three times per crop.
    Chunking is only legitimate if it is EXACT: min-of-chunk-minima is
    min-over-all, and sqrt is monotone so taking it once at the end picks
    the same pixel. This asserts both, against scipy."""
    from scipy.ndimage import distance_transform_edt

    from recog.seg_evaluate import _distance_to_nearest

    rng = np.random.default_rng(4)
    tb = rng.random((23, 31)) < 0.08
    assert tb.sum() > 20, "fixture must have enough boundary pixels to chunk"

    reference = distance_transform_edt(~tb)
    one_shot = _distance_to_nearest(tb, max_elements=10 ** 9)

    np.testing.assert_array_equal(one_shot, reference)
    for max_elements in (1, 23 * 31, 23 * 31 * 3, 10 ** 6):
        np.testing.assert_array_equal(
            _distance_to_nearest(tb, max_elements=max_elements), reference,
            err_msg=f"chunking at max_elements={max_elements} changed the "
                    "distance field")


def test_per_class_iou_handles_a_class_absent_from_both():
    import numpy as np

    from recog.seg_evaluate import per_class_iou

    m = np.zeros((10, 10), np.int8)
    m[2:8, 2:8] = 2
    iou = per_class_iou(m, m, num_classes=6)
    assert iou["bay"] == pytest.approx(1.0)
    assert np.isnan(iou["obstruction"]), (
        "a class in neither prediction nor truth has no IoU; reporting "
        "0.0 would drag the mean down for a class that was never tested")


# ---------------------------------------------- decode-cached evaluate --
#
# `evaluate` used to decode the source frame once per CROP rather than
# once per FRAME, while `resolve_frame_scales` a few lines above already
# cached its sidecar per file for exactly that reason. It also called
# `segment_batch([crop])[0]` once per crop, and that half was measured
# and NOT taken: batch 8 disagrees with batch 1 on 128 of 2 217 730 label
# pixels over the real split and moves twelve published figures in their
# last digit, for 0.43 s of a ~20 s run (see SEG_EVAL_BATCH in
# recog/seg_evaluate.py). The chunking exists and is tested; the default
# stays at one crop per call.
#
# These pin the new path's outputs to the old one's, exactly.

def _seg_corpus(tmp_path):
    """Three frames and five crops, deliberately interleaved so the
    file-grouped traversal really does reorder them, and so two frames
    contribute more than one crop each (the redundant decode).

    Returns `(dataset, val_indices, scales)`.
    """
    from PIL import Image

    from recog.synth3d.annotate import rle_encode

    H, W = 40, 60
    rng = np.random.default_rng(11)
    layout = [("a.png", (0, 0, 20, 20)), ("b.png", (10, 10, 30, 30)),
              ("a.png", (20, 10, 50, 35)), ("c.png", (5, 5, 25, 25)),
              ("b.png", (30, 0, 55, 20))]

    for name in ("a.png", "b.png", "c.png"):
        pixels = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / name)

    bay = np.zeros((H, W), np.uint8)
    bay[8:30, 8:45] = 1
    obst = np.zeros((H, W), np.uint8)
    obst[12:18, 12:20] = 1
    anns = [{"class": "placement_area", "segmentation": rle_encode(bay)},
            {"class": "obstruction", "segmentation": rle_encode(obst)}]

    class _Corpus:
        img_dir = str(tmp_path)
        samples = [({"file_name": name}, anns, box) for name, box in layout]

    val_indices = list(range(len(layout)))
    scales = {i: 0.50 + 0.01 * i for i in val_indices}   # distinct per crop
    return _Corpus(), val_indices, scales


class _DeterministicSegmenter:
    """Label map is a pure function of the crop's own pixels.

    So batching cannot change the answer by construction, and any
    difference the equivalence test below sees is a real one - a crop
    paired with the wrong index, or a frame decoded from the wrong file -
    rather than kernel noise.
    """

    def __init__(self):
        self.batch_sizes = []

    def segment_batch(self, crops):
        self.batch_sizes.append(len(crops))
        return [(c[:, :, 0] % 6).astype(np.int8) for c in crops]


def _one_crop_per_call(segmenter, full_dataset, val_indices, batch_size=None):
    """`evaluate`'s inference exactly as it was before the change: one
    `segment_batch` call per crop, one decode per crop."""
    from pathlib import Path

    from PIL import Image

    from recog.seg_dataset import extract_crop

    out = []
    for idx in val_indices:
        img_meta, _anns, unit_box = full_dataset.samples[idx]
        image = np.asarray(Image.open(
            Path(full_dataset.img_dir) / img_meta["file_name"]
        ).convert("RGB"))
        crop = extract_crop(image, tuple(int(v) for v in unit_box),
                            out_size=None)
        out.append(segmenter.segment_batch([crop])[0])
    return out


def test_evaluate_reports_exactly_what_the_one_crop_at_a_time_path_reported(
    tmp_path, monkeypatch,
):
    """Every number in the receipt, unchanged - not approximately.

    `np.testing.assert_equal` recurses through the result dict and treats
    NaN as equal to NaN, which is what the absent-class IoUs need; it
    compares everything else exactly, which is what a claim of "pure
    efficiency" means.
    """
    from recog import seg_evaluate

    ds, val_indices, scales = _seg_corpus(tmp_path)

    monkeypatch.setattr(seg_evaluate, "predict_val_crops", _one_crop_per_call)
    before_the_change = seg_evaluate.evaluate(
        _DeterministicSegmenter(), ds, val_indices, scales)
    monkeypatch.undo()

    now = seg_evaluate.evaluate(_DeterministicSegmenter(), ds, val_indices,
                                scales)

    assert before_the_change["n_val_crops"] == 5
    assert not np.isnan(before_the_change["boundary_mm"]["bay"]), (
        "the fixture must actually produce a boundary displacement, or "
        "this test is comparing two NaNs and asserting nothing")
    np.testing.assert_equal(now, before_the_change)


def test_evaluate_still_sends_one_crop_per_call_by_default(tmp_path):
    """Not an oversight - a measured decision, pinned so nobody quietly
    reverses it. Raising SEG_EVAL_BATCH changes what the receipt prints
    (128 label pixels over the real split, twelve figures in their last
    digit) and belongs in the same commit as a regenerated
    docs/receipts/seg_eval.txt."""
    from recog import seg_evaluate

    assert seg_evaluate.SEG_EVAL_BATCH == 1

    ds, val_indices, scales = _seg_corpus(tmp_path)
    seg = _DeterministicSegmenter()
    seg_evaluate.evaluate(seg, ds, val_indices, scales)

    assert seg.batch_sizes == [1, 1, 1, 1, 1]


def test_predict_val_crops_chunks_at_the_requested_batch_size(tmp_path):
    """The chunking is real and correct, so raising the default is a
    one-line change rather than a rewrite."""
    from recog.seg_evaluate import predict_val_crops

    ds, val_indices, _scales = _seg_corpus(tmp_path)
    seg = _DeterministicSegmenter()
    preds = predict_val_crops(seg, ds, val_indices, batch_size=2)

    assert seg.batch_sizes == [2, 2, 1]
    assert [p.shape for p in preds] == \
        [p.shape for p in predict_val_crops(_DeterministicSegmenter(), ds,
                                            val_indices, batch_size=8)]


def test_each_frame_is_decoded_once_not_once_per_crop_it_carries(
    tmp_path, monkeypatch,
):
    """The dataset emits ~1.68 crops per frame, so ~40% of the decodes
    were of a frame already in hand. Five crops here come from three
    files."""
    import PIL.Image

    from recog import seg_evaluate

    ds, val_indices, scales = _seg_corpus(tmp_path)

    opened = []
    real_open = PIL.Image.open
    monkeypatch.setattr(PIL.Image, "open",
                        lambda p, *a, **kw: (opened.append(str(p)),
                                             real_open(p, *a, **kw))[1])

    seg_evaluate.evaluate(_DeterministicSegmenter(), ds, val_indices, scales)

    assert len(opened) == 3, (
        f"{len(opened)} decodes for 5 crops from 3 frames: {opened}")


def test_predict_val_crops_returns_predictions_in_the_callers_order(tmp_path):
    """The traversal is reordered by file; the RESULT must not be.

    `evaluate` reduces its per-crop boundary and area lists with np.mean
    and sum, and floating-point addition is not associative - accumulating
    in file-sorted order would move published millimetre figures in their
    last digits for a reason no reader of the receipt could reconstruct.
    """
    from pathlib import Path

    from PIL import Image

    from recog.seg_dataset import extract_crop
    from recog.seg_evaluate import predict_val_crops

    ds, val_indices, _scales = _seg_corpus(tmp_path)
    seg = _DeterministicSegmenter()
    preds = predict_val_crops(seg, ds, val_indices)

    for pos, idx in enumerate(val_indices):
        img_meta, _anns, unit_box = ds.samples[idx]
        image = np.asarray(Image.open(
            Path(ds.img_dir) / img_meta["file_name"]).convert("RGB"))
        crop = extract_crop(image, tuple(int(v) for v in unit_box),
                            out_size=None)
        np.testing.assert_array_equal(
            preds[pos], (crop[:, :, 0] % 6).astype(np.int8),
            err_msg=f"position {pos} (crop {idx}) got another crop's mask")


# ------------------------------------------------------ split guard --
#
# recog/dataset3d_seg is gitignored and came from a resumable generation
# run stopped part-way through. Resuming or regenerating it changes
# len(full_dataset), so random_split silently returns a DIFFERENT
# partition for the same seed and a checkpoint could be scored on crops
# it trained on - with nothing to signal it. seg_training already writes
# val_instance_counts into every checkpoint for exactly this check.

class _StubDataset:
    """Only `.samples` is read by compute_val_instance_counts."""

    def __init__(self, samples):
        self.samples = samples


def _one_crop_with_a_background_sliver():
    """A 40x40 crop that is ALL cartridge except a single background
    pixel at (1, 1).

    (1, 1) on purpose: a 40 -> 4 nearest resize samples rows/cols
    {0, 10, 20, 30} on the numpy path and {5, 15, 25, 35} on the cv2
    path, so this pixel is missed by BOTH and the fixture does not
    depend on which one is installed. (0, 0) would survive the numpy
    grid and quietly stop testing anything.
    """
    from recog.synth3d.annotate import rle_encode

    cart = np.ones((40, 40), np.uint8)
    cart[1, 1] = 0
    anns = [{"class": "cartridge", "category_id": 2,
             "segmentation": rle_encode(cart),
             "bbox_xyxy": [0, 0, 40, 40]}]
    return [({"file_name": "x.png"}, anns, (0, 0, 40, 40))]


def test_val_instance_counts_are_computed_at_the_training_loaders_resolution():
    """The split guard's fingerprint must be built the SAME way
    seg_training built the one stored in the checkpoint - and training
    counts off its val DataLoader, whose targets are rasterised at
    `model.crop_size` and nearest-downsampled to it, not at the crop's
    native resolution.

    Measured, not assumed: on `configs/segmentation_anchored.yaml`'s val
    split the two resolutions disagree by exactly one crop
    (background 124 native vs. 123 at 256), because a sliver of
    background survives at native size and is lost by the downsample.
    That one crop made the guard declare the dataset had changed when
    nothing had changed, and it blocked the in-distribution evaluation
    outright.
    """
    from recog.seg_evaluate import compute_val_instance_counts

    ds = _StubDataset(_one_crop_with_a_background_sliver())

    native = compute_val_instance_counts(ds, [0], out_size=None)
    downsampled = compute_val_instance_counts(ds, [0], out_size=4)

    assert native["background"] == 1
    assert downsampled["background"] == 0, (
        "the corner sliver must not survive the nearest-neighbour "
        "downsample - if it does, this fixture no longer exercises the "
        "resolution difference the guard tripped on")
    assert native["cartridge"] == downsampled["cartridge"] == 1


def test_check_split_matches_checkpoint_passes_when_counts_agree(tmp_path):
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    counts = {"background": 53, "cartridge": 37, "bay": 19,
             "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": counts,
               "coco_path": "recog/dataset3d_seg/instances_seg.json"},
              ckpt_path)

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
        counts)  # must not raise


def test_check_split_matches_checkpoint_raises_loudly_on_mismatch(tmp_path):
    """A resumed/regenerated dataset changes len(full_dataset), which
    changes what random_split returns for the same seed - this is the
    live failure mode the guard exists to catch, not a hypothetical."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg/instances_seg.json"},
              ckpt_path)

    recomputed_after_resume = dict(trained_on, bay=24)  # dataset grew

    with pytest.raises(SystemExit) as exc:
        check_split_matches_checkpoint(
            str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
            recomputed_after_resume)

    msg = str(exc.value)
    assert "19" in msg and "24" in msg, (
        "the error must name BOTH the checkpoint's recorded counts and "
        "the recomputed ones, not just say 'mismatch'")


def test_check_split_matches_checkpoint_warns_but_does_not_raise_on_old_checkpoint(
    tmp_path,
):
    """A checkpoint written before this field existed has nothing to
    check against - warn, do not fail a run that has no way to comply."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": {}}, ckpt_path)

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
        {"bay": 1})  # must not raise


def test_check_split_matches_checkpoint_skips_count_check_across_datasets(tmp_path):
    """Spec #2 (generalisation): a checkpoint trained on one dataset (e.g.
    the anchored procedural set) is deliberately evaluated against a
    DIFFERENT held-out dataset (the CAD test set) - that is the plan's
    entire Task 19, not a drifted split. The two datasets' val instance
    counts are expected to differ (different scenes entirely); this must
    not raise just because the checkpoint's own coco_path differs from
    the eval config's."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg_anchored/instances_seg.json"},
              ckpt_path)

    wildly_different = {"background": 400, "cartridge": 300, "bay": 0,
                        "electronics": 0, "obstruction": 90, "battery": 250}

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg_cad_test/instances_seg.json",
        wildly_different)  # must not raise - different dataset, by design


def test_check_split_matches_checkpoint_still_raises_when_coco_path_matches(tmp_path):
    """The cross-dataset skip must not become a blanket bypass: if the
    eval's coco_path is the SAME one the checkpoint recorded, a count
    mismatch is still the drifted-split bug and must still raise."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg_anchored/instances_seg.json"},
              ckpt_path)

    with pytest.raises(SystemExit):
        check_split_matches_checkpoint(
            str(ckpt_path), "recog/dataset3d_seg_anchored/instances_seg.json",
            dict(trained_on, bay=24))


# ------------------------------------------------- singleton last batch --

def test_drop_last_batch_true_when_the_final_batch_would_be_a_singleton():
    """A train split of 721 crops at batch_size 8 leaves a final batch of
    exactly 1. DeepLabV3's ASPP pooling branch reduces that to a
    [1, 256, 1, 1] tensor, and BatchNorm in TRAINING mode raises
    `ValueError: Expected more than 1 value per channel when training`.
    This is not hypothetical: it is what
    `configs/segmentation_anchored.yaml` (848 crops, 0.85 split -> 721
    train) did on its first epoch, and it kills the run outright.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(721, 8) is True


def test_drop_last_batch_false_when_the_final_batch_is_safe():
    """Every other remainder trains fine, so nothing is dropped - the
    existing baseline (361 crops -> 307 train, 307 % 8 == 3) must keep
    seeing every crop in every epoch, exactly as it did when
    `docs/receipts/seg_eval.txt` was produced.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(307, 8) is False
    assert drop_last_batch(720, 8) is False   # remainder 0
    assert drop_last_batch(722, 8) is False   # remainder 2


def test_drop_last_batch_false_when_batch_size_is_one():
    """batch_size 1 makes EVERY batch a singleton; dropping the last one
    would not help and dropping them all would train on nothing. Such a
    config is broken for BatchNorm regardless, and silently emptying the
    loader would hide that instead of letting it surface.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(9, 1) is False


# --------------------------------------------------------- latency exit --

def test_latency_within_budget_true_when_the_batch8_row_passes():
    from recog.seg_evaluate import latency_within_budget

    latency = [{"cartridges": 1, "within_50ms_budget": True},
              {"cartridges": 8, "within_50ms_budget": True}]
    assert latency_within_budget(latency) is True


def test_latency_within_budget_false_when_the_batch8_row_fails():
    """This is what main()'s exit code is derived from - previously only
    a log.warning fired here, so a CI job could not gate on the plan's
    latency acceptance criterion at all."""
    from recog.seg_evaluate import latency_within_budget

    latency = [{"cartridges": 1, "within_50ms_budget": True},
              {"cartridges": 8, "within_50ms_budget": False}]
    assert latency_within_budget(latency) is False


# --------------------------------------------------- checkpoint note --

def test_sibling_checkpoint_note_reports_both_ious_and_their_delta(tmp_path):
    from recog.seg_evaluate import _sibling_checkpoint_note

    counts = {"bay": 19, "electronics": 19, "obstruction": 11}
    torch.save({"selected_mean_iou": 0.8158, "val_instance_counts": counts},
              tmp_path / "best.pt")
    torch.save({"selected_mean_iou": 0.8140, "val_instance_counts": counts},
              tmp_path / "last.pt")

    note = _sibling_checkpoint_note(str(tmp_path / "best.pt"))
    assert note is not None
    assert "0.8158" in note and "0.8140" in note


def test_sibling_checkpoint_note_is_none_without_both_checkpoints(tmp_path):
    from recog.seg_evaluate import _sibling_checkpoint_note

    torch.save({"selected_mean_iou": 0.8}, tmp_path / "best.pt")
    assert _sibling_checkpoint_note(str(tmp_path / "best.pt")) is None


# ------------------------------------------------------------- per-SKU --

def test_group_indices_by_asset_partitions_by_sku():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A", None]

    out = group_indices_by_asset(_FakeDataset(), [0, 1, 2, 3])
    assert out == {"A": [0, 2], "B": [1], None: [3]}


def test_group_indices_by_asset_only_includes_requested_indices():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A"]

    out = group_indices_by_asset(_FakeDataset(), [0, 1])
    assert out == {"A": [0], "B": [1]}


def test_format_per_sku_table_lists_every_sku_with_its_crop_count():
    from recog.seg_evaluate import format_per_sku_table

    results = {
        "AnkerPowerCore10000": {"n_val_crops": 12,
                                "ious": {"bay": 0.80, "obstruction": 0.60}},
        "AnkerPowerCore13000": {"n_val_crops": 9,
                                "ious": {"bay": 0.75, "obstruction": 0.55}},
    }
    table = format_per_sku_table(results)
    assert "AnkerPowerCore10000" in table and "12" in table
    assert "AnkerPowerCore13000" in table and "9" in table
