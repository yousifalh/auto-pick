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
