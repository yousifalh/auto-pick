"""Training- and validation-time augmentation for the recogniser.

Albumentations (when available) synchronously transforms the image
and its Pascal-VOC bounding boxes. The :func:`build_train_transform`
pipeline is deliberately aggressive — factory-floor lighting varies
more than typical lab conditions — so the photometric block stacks
brightness/contrast, gamma, hue-saturation-value and random shadows.

The train pipeline is four blocks, applied in capture order:

  1. ``photometric``  — illumination: brightness/contrast, gamma, HSV,
     cast shadows. Gated by ``p_photometric``.
  2. ``dihedral``     — the eight symmetries of the square: horizontal
     flip, vertical flip, k*90° rotation. NOT gated by ``p_geometric``
     (see below).
  3. ``geometric``    — sub-pixel camera pose: a small affine shift /
     scale / rotate. Gated by ``p_geometric``.
  4. ``camera``       — sensor and optics degradation: motion blur,
     defocus, ISO/read noise. Gated by ``p_camera``.

Why the dihedral block is separate and ungated
----------------------------------------------
This domain is top-down and axis-aligned: the camera looks straight
down at parts lying flat on a surface, and nothing in the scene has a
canonical "up". A flip or a 90° rotation therefore maps a valid scene
to another equally valid scene, and maps an axis-aligned box to an
axis-aligned box with no area lost and no interpolation — it is exact,
free, and multiplies the effective dataset by eight. That is a much
better deal than the affine block, which resamples pixels and can push
a box off the frame, so the two do not belong behind the same switch.
Folding them together also meant ``HorizontalFlip(p=0.5)`` nested
inside ``Compose(p=0.5)`` fired on only a quarter of samples.

Real photos are handheld phone shots of a bench; the renders are
pin-sharp, noiseless and taken from a fixed height. Block 4 exists to
close that gap — without it the detector can key on render sharpness,
which no real photograph has.

When Albumentations isn't installed (CPU-only CI), the module falls
back to :class:`_FallbackTransform`: a seeded, deterministic
numpy-only shim that still returns the expected dict shape so the
dataset pipeline can be unit-tested. The fallback deliberately does
NOT implement blocks 2-4; it exists to keep the suite runnable, not to
be a second training pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover - import guard
    import albumentations as A  # type: ignore

    _ALB_AVAILABLE = True
except Exception:  # pragma: no cover - only hit on CI containers
    _ALB_AVAILABLE = False


# cv2.BORDER_CONSTANT. Named here rather than importing cv2 for one integer:
# the module must stay importable in the albumentations-free fallback
# environment, which has no cv2 either.
_BORDER_CONSTANT = 0


# ------------------------------------------------------------- builders --

def build_train_transform(cfg: Dict[str, Any]):
    """Aggressive training augmentation.

    See ``configs/recognition.yaml::augmentation`` for tunables.
    """
    if not _ALB_AVAILABLE:
        return _FallbackTransform(cfg, train=True)

    photometric = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", 0.55),
            contrast_limit=cfg.get("contrast_limit", 0.50),
            p=0.9,
        ),
        A.RandomGamma(
            # Gamma is what actually reaches "dim" without crushing the
            # black point the way a large negative brightness offset does:
            # it rescales the whole tone curve instead of subtracting a
            # constant, so shadow detail survives.
            gamma_limit=tuple(cfg.get("gamma_limit", (45, 190))),
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.get("hue_shift_limit", 15),
            sat_shift_limit=cfg.get("sat_shift_limit", 30),
            val_shift_limit=cfg.get("val_shift_limit", 20),
            p=0.5,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=tuple(cfg.get("shadow_num_range", (1, 3))),
            shadow_dimension=5,
            p=0.4,
        ),
    ], p=cfg.get("p_photometric", 0.85))

    # Label-exact, so ungated — see the module docstring.
    dihedral = A.Compose([
        A.HorizontalFlip(p=cfg.get("p_flip", 0.5)),
        A.VerticalFlip(p=cfg.get("p_flip", 0.5)),
        A.RandomRotate90(p=cfg.get("p_rot90", 0.5)),
    ], p=1.0)

    geometric = A.Compose([
        # Affine, not ShiftScaleRotate: albumentations 2.x deprecates the
        # latter as "a special case of Affine". Same three limits, same
        # meaning - shift_limit is a fraction of the side, scale_limit is
        # a delta about 1.0, rotate_limit is degrees either side of 0.
        A.Affine(
            translate_percent=_symmetric(cfg.get("shift_limit", 0.04)),
            scale=_about_one(cfg.get("scale_limit", 0.10)),
            rotate=_symmetric(cfg.get("rotation_limit", 4)),
            border_mode=_BORDER_CONSTANT,
            fill=0,
            p=0.5,
        ),
    ], p=cfg.get("p_geometric", 0.5))

    camera = A.Compose([
        A.OneOf([
            A.MotionBlur(blur_limit=tuple(cfg.get("motion_blur_limit", (3, 11))),
                         p=1.0),
            A.Defocus(radius=tuple(cfg.get("defocus_radius", (1, 5))),
                      alias_blur=(0.05, 0.35), p=1.0),
        ], p=cfg.get("p_blur", 0.35)),
        A.ISONoise(
            color_shift=tuple(cfg.get("iso_color_shift", (0.01, 0.06))),
            intensity=tuple(cfg.get("iso_noise_intensity", (0.1, 0.7))),
            p=0.35,
        ),
        A.GaussNoise(
            # Albumentations 2.x uses `std_range` in normalised units.
            std_range=tuple(cfg.get("gauss_noise_std_range", (0.02, 0.14))),
            p=0.35,
        ),
    ], p=cfg.get("p_camera", 0.6))

    return A.Compose(
        [photometric, dihedral, geometric, camera],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.25,
        ),
    )


def _symmetric(limit):
    """``0.04`` -> ``(-0.04, 0.04)``; an explicit pair passes through.

    ShiftScaleRotate took one-sided limits and mirrored them; Affine takes
    the range directly. This keeps the config keys meaning what they did.
    """
    if isinstance(limit, (list, tuple)):
        return (float(limit[0]), float(limit[1]))
    return (-abs(float(limit)), abs(float(limit)))


def _about_one(limit):
    """``0.10`` -> ``(0.9, 1.1)`` — ShiftScaleRotate's scale convention."""
    if isinstance(limit, (list, tuple)):
        return (float(limit[0]), float(limit[1]))
    return (1.0 - abs(float(limit)), 1.0 + abs(float(limit)))


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


# ------------------------------------------------------- mask-aware path --
#
# The detector's pipeline above transforms an image against Pascal-VOC
# boxes. The segmenter needs the same illumination coverage applied to an
# image against a dense label map instead - and a label map must never be
# interpolated: the mean of class 2 and class 4 is class 3, a different
# object entirely. Albumentations routes its `mask` target through
# nearest-neighbour for every geometric op, which is why it is used here
# rather than resampling the label by hand.
#
# The geometric block is deliberately narrower than the detector's: no
# Affine. A per-ROI crop is already produced from a jittered box (see
# recog.seg_dataset.jitter_box), which is how this pipeline models
# detector box error; stacking a shift/scale on top of that would
# double-count it. The label-exact dihedral group (flip / 90-degree
# rotation) stays, since it costs nothing - it is an exact permutation of
# pixels, not a resample.

def build_seg_train_transform(cfg: Dict[str, Any]):
    """Training augmentation for the bay segmenter.

    The photometric block is shared with the detector's pipeline - the
    same renders need the same illumination coverage. The geometric block
    is narrower: no Affine. A per-ROI crop is already produced by a
    jittered box (see recog.seg_dataset.jitter_box), so shifting and
    rescaling it again would double-count the detector error this is
    meant to model.
    """
    if not _ALB_AVAILABLE:
        return _FallbackSegTransform(cfg, train=True)

    return A.Compose([
        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("brightness_limit", 0.55),
                contrast_limit=cfg.get("contrast_limit", 0.50), p=0.9),
            A.RandomGamma(
                gamma_limit=tuple(cfg.get("gamma_limit", (45, 190))), p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=cfg.get("hue_shift_limit", 15),
                sat_shift_limit=cfg.get("sat_shift_limit", 30),
                val_shift_limit=cfg.get("val_shift_limit", 20), p=0.5),
        ], p=cfg.get("p_photometric", 0.85)),
        A.HorizontalFlip(p=cfg.get("p_flip", 0.5)),
        A.VerticalFlip(p=cfg.get("p_flip", 0.5)),
        A.RandomRotate90(p=cfg.get("p_rot90", 0.5)),
        A.Compose([
            A.MotionBlur(blur_limit=tuple(cfg.get("motion_blur_limit", (3, 11))),
                         p=0.5),
            A.GaussNoise(std_range=tuple(
                cfg.get("gauss_noise_std_range", (0.02, 0.14))), p=0.5),
        ], p=cfg.get("p_camera", 0.6)),
    ])


def build_seg_val_transform(cfg: Dict[str, Any]):
    """Validation: geometric identity. Any geometry here would make the
    reported IoU a property of the augmenter, not of the model."""
    if not _ALB_AVAILABLE:
        return _FallbackSegTransform(cfg, train=False)
    return A.Compose([])


def apply_with_mask(transform, image: np.ndarray,
                    mask: np.ndarray) -> Dict[str, Any]:
    """Apply ``transform`` to an image and its dense label map.

    Albumentations routes a `mask` target through nearest-neighbour
    resampling for every geometric op, which is the only correct choice:
    interpolating labels averages class 2 and class 4 into class 3, a
    different object.
    """
    out = transform(image=image, mask=mask)
    return {"image": out["image"], "mask": out["mask"]}


class _FallbackSegTransform:
    """Numpy-only stand-in, mirroring _FallbackTransform.

    Photometric only, plus the label-exact dihedral group. That is enough
    for the test suite to run without albumentations, which recog's
    existing pipeline already treats as optional.
    """

    def __init__(self, cfg: Dict[str, Any], train: bool) -> None:
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(0)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        img = image.astype(np.float32)
        out_mask = mask
        if self.train:
            if self.rng.random() < self.cfg.get("p_flip", 0.5):
                img, out_mask = img[:, ::-1], out_mask[:, ::-1]
            if self.rng.random() < self.cfg.get("p_flip", 0.5):
                img, out_mask = img[::-1], out_mask[::-1]
            b = self.cfg.get("brightness_limit", 0.4)
            img = np.clip(img + self.rng.uniform(-b, b) * 255, 0, 255)
        return {"image": np.ascontiguousarray(img).astype(np.uint8),
                "mask": np.ascontiguousarray(out_mask)}


__all__ = [
    "apply", "build_train_transform", "build_val_transform",
    "apply_with_mask", "build_seg_train_transform", "build_seg_val_transform",
]
