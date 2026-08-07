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
                       crop_size=64, half=False)
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
                       crop_size=64, half=False)
    crop = (np.arange(48 * 64 * 3, dtype=np.uint8) % 255).reshape(48, 64, 3)
    assert np.array_equal(seg.segment(crop), seg.segment_batch([crop])[0])


def test_empty_batch_returns_empty_list():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False)
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

    target = np.zeros((30, 30), np.int8); target[8:22, 8:22] = 2
    pred = np.zeros((30, 30), np.int8); pred[8:22, 8:24] = 2

    a = boundary_displacement_mm(pred, target, cls=2, mm_per_px=1.0)
    b = boundary_displacement_mm(pred, target, cls=2, mm_per_px=2.0)
    assert b == pytest.approx(2.0 * a)
    assert a > 0.0


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
