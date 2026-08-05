"""Training- and validation-time augmentation for the recogniser.

Albumentations (when available) synchronously transforms the image
and its Pascal-VOC bounding boxes. The :func:`build_train_transform`
pipeline is deliberately aggressive — factory-floor lighting varies
more than typical lab conditions — so the photometric block stacks
brightness/contrast, gamma, hue-saturation-value and random shadows.

When Albumentations isn't installed (CPU-only CI), the module falls
back to :class:`_FallbackTransform`: a seeded, deterministic
numpy-only shim that still returns the expected dict shape so the
dataset pipeline can be unit-tested.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover - import guard
    import albumentations as A  # type: ignore

    _ALB_AVAILABLE = True
except Exception:  # pragma: no cover - only hit on CI containers
    _ALB_AVAILABLE = False


# ------------------------------------------------------------- builders --

def build_train_transform(cfg: Dict[str, Any]):
    """Aggressive training augmentation.

    See ``configs/recognition.yaml::augmentation`` for tunables.
    """
    if not _ALB_AVAILABLE:
        return _FallbackTransform(cfg, train=True)

    photometric = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.40),
            contrast_limit=cfg.get("contrast_limit", 0.40),
            p=0.9,
        ),
        A.RandomGamma(
            gamma_limit=tuple(cfg.get("gamma_limit", (60, 140))),
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("hue_shift_limit", 15),
            sat_shift_limit=cfg.get("sat_shift_limit", 25),
            val_shift_limit=cfg.get("val_shift_limit", 10),
            p=0.5,
        ),
        A.GaussNoise(
            # Albumentations 2.x uses `std_range` in normalised units.
            std_range=(0.05, 0.2),
            p=0.4,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=tuple(cfg.get("shadow_num_range", (1, 3))),
            shadow_dimension=5,
            p=0.4,
        ),
    ], p=cfg.get("p_photometric", 0.8))

    geometric = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.04,
            scale_limit=cfg.get("scale_limit", 0.10),
            rotate_limit=cfg.get("rotation_limit", 4),
            border_mode=0,
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
    ], p=cfg.get("p_geometric", 0.5))

    return A.Compose(
        [photometric, geometric],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )


def build_val_transform(cfg: Dict[str, Any]):
    """Validation-time transform: no randomisation."""
    if not _ALB_AVAILABLE:
        return _FallbackTransform(cfg, train=False)

    # PadIfNeeded with a trivial minimum is a safe identity; it keeps
    # the bbox pipeline engaged so downstream code can assume the same
    # dict keys regardless of whether any real augmentation fired.
    return A.Compose(
        [A.PadIfNeeded(min_height=1, min_width=1, p=1.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            # Albumentations does sub-pixel rounding during normalisation,
            # so a strict min_visibility=1.0 would drop boxes that haven't
            # actually changed. 1 % is a safe lower bound.
            min_visibility=0.01,
        ),
    )


def apply(
    transform,
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
) -> Dict[str, Any]:
    """Single entry-point wrapper — shields callers from minor API drift."""
    return transform(image=image, bboxes=bboxes, class_labels=class_labels)


# ---------------------------------------------------- fallback pathway --

class _FallbackTransform:
    """Numpy-only stand-in used when Albumentations is missing.

    Produces a valid dict with ``image``, ``bboxes`` and ``class_labels``
    so the dataset / augmentation tests pass without the optional
    dependency. Deterministic given construction arguments so the
    "identical seed → identical output" contract holds.
    """

    def __init__(self, cfg: Dict[str, Any], train: bool) -> None:
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(0)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
    ) -> Dict[str, Any]:
        img = image.astype(np.float32)

        if self.train:
            # Brightness (additive).
            b_lim = self.cfg.get("brightness_limit", 0.4)
            delta = self.rng.uniform(-b_lim, b_lim) * 255
            img = np.clip(img + delta, 0, 255)

            # Contrast (scale about the mean).
            c_lim = self.cfg.get("contrast_limit", 0.4)
            scale = 1.0 + self.rng.uniform(-c_lim, c_lim)
            mean = img.mean()
            img = np.clip((img - mean) * scale + mean, 0, 255)

            # Gaussian noise, firing 40 % of the time.
            if self.rng.random() < 0.4:
                img = np.clip(
                    img + self.rng.normal(0, 10.0, img.shape), 0, 255,
                )

        return {
            "image": img.astype(np.uint8),
            "bboxes": [list(b) for b in bboxes],
            "class_labels": list(class_labels),
        }


__all__ = ["apply", "build_train_transform", "build_val_transform"]
