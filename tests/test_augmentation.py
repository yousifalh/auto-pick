"""Augmentation pipeline tests (PPR §5.2.2).

Verifies that both the Albumentations and fallback paths:

* return the correct dict keys
* preserve image shape
* preserve bounding-box count for the (deterministic) validation transform
"""
from __future__ import annotations

import numpy as np
import pytest

from recog.augmentation import (
    apply, build_train_transform, build_val_transform,
)


AUG_CFG = {
    "brightness_limit": 0.40,
    "contrast_limit": 0.40,
    "gamma_limit": [60, 140],
    "hue_shift_limit": 15,
    "sat_shift_limit": 25,
    "val_shift_limit": 10,
    "rotation_limit": 4,
    "scale_limit": 0.1,
    "shadow_num_range": [1, 3],
    "p_photometric": 0.8,
    "p_geometric": 0.5,
}


def _img(h=120, w=160):
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_train_transform_returns_expected_keys():
    t = build_train_transform(AUG_CFG)
    out = apply(t, _img(), bboxes=[[10, 10, 50, 50]], class_labels=[1])
    assert set(out.keys()) >= {"image", "bboxes", "class_labels"}


def test_val_transform_preserves_bboxes():
    t = build_val_transform(AUG_CFG)
    boxes = [[10, 10, 50, 50], [70, 80, 120, 110]]
    out = apply(t, _img(), bboxes=boxes, class_labels=[1, 2])
    assert len(out["bboxes"]) == 2
    assert len(out["class_labels"]) == 2


def test_val_transform_shape_preserved():
    t = build_val_transform(AUG_CFG)
    img = _img(80, 100)
    out = apply(t, img, bboxes=[], class_labels=[])
    assert out["image"].shape == img.shape


def test_train_transform_image_dtype_preserved():
    t = build_train_transform(AUG_CFG)
    out = apply(t, _img(), bboxes=[[10, 10, 50, 50]], class_labels=[1])
    assert out["image"].dtype == np.uint8


def test_apply_is_deterministic_given_seed(monkeypatch):
    """The fallback path uses a fixed rng so results are reproducible."""
    from recog.augmentation import _FallbackTransform
    t = _FallbackTransform(AUG_CFG, train=True)
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    out1 = t(image=img.copy(), bboxes=[], class_labels=[])
    t2 = _FallbackTransform(AUG_CFG, train=True)
    out2 = t2(image=img.copy(), bboxes=[], class_labels=[])
    np.testing.assert_array_equal(out1["image"], out2["image"])
