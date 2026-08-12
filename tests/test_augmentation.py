"""Augmentation pipeline tests (PPR §5.2.2).

Verifies that both the Albumentations and fallback paths:

* return the correct dict keys
* preserve image shape
* preserve bounding-box count for the (deterministic) validation transform

and that the label-exact dihedral group (horizontal flip, vertical flip,
90-degree rotation) moves a known box exactly where the geometry says it
should. Those three are the cheapest augmentation available in this
top-down axis-aligned domain, and a silently-wrong bbox mapping under any
of them would mislabel a large fraction of the training set while every
image still looked perfectly fine.
"""
from __future__ import annotations

import numpy as np
import pytest

from recog.augmentation import (
    _ALB_AVAILABLE, apply, build_train_transform, build_val_transform,
)


# Mirrors configs/recognition.yaml::augmentation.
AUG_CFG = {
    "brightness_limit": 0.55,
    "contrast_limit": 0.50,
    "gamma_limit": [45, 190],
    "hue_shift_limit": 15,
    "sat_shift_limit": 30,
    "val_shift_limit": 20,
    "shadow_num_range": [1, 3],
    "shift_limit": 0.04,
    "scale_limit": 0.1,
    "rotation_limit": 4,
    "motion_blur_limit": [3, 11],
    "defocus_radius": [1, 5],
    "iso_color_shift": [0.01, 0.06],
    "iso_noise_intensity": [0.1, 0.7],
    "gauss_noise_std_range": [0.02, 0.14],
    "p_photometric": 0.85,
    "p_geometric": 0.5,
    "p_flip": 0.5,
    "p_rot90": 0.5,
    "p_blur": 0.35,
    "p_camera": 0.6,
}

requires_alb = pytest.mark.skipif(
    not _ALB_AVAILABLE, reason="albumentations is not installed")


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


# ------------------------------------------------ label-exact geometry --
#
# HorizontalFlip, VerticalFlip and RandomRotate90 are the whole reason this
# domain is cheap to augment: top-down, axis-aligned, nothing has a canonical
# "up", so each of the eight square symmetries maps a valid scene to another
# valid scene and an axis-aligned box to an axis-aligned box with nothing
# lost. That only holds if albumentations actually moves the box with the
# pixels - and a wrong mapping is invisible, because the image still looks
# like a perfectly good factory photo, just with the label somewhere else.

import albumentations as A  # noqa: E402  (guarded by requires_alb below)

_BBOX_PARAMS = dict(format="pascal_voc", label_fields=["class_labels"],
                    min_visibility=0.25)


def _one_op(op):
    """One op, wrapped exactly the way build_train_transform wraps its blocks.

    The inner Compose carries no bbox_params of its own. That nesting is not
    incidental: if albumentations ever stopped propagating the outer
    processor into a nested Compose, every geometric op in the real pipeline
    would move pixels while leaving boxes behind, and these tests would be
    the only thing that noticed.
    """
    return A.Compose([A.Compose([op], p=1.0)],
                     bbox_params=A.BboxParams(**_BBOX_PARAMS))


def _painted(h, w, box):
    """Black image with the box's interior painted solid white."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = 255
    return img


def _white_extent(img):
    """Tight (x0, y0, x1, y1) around the non-zero pixels, exclusive edges."""
    ys, xs = np.nonzero(img.max(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@requires_alb
def test_horizontal_flip_maps_a_known_box_to_the_mirrored_column():
    t = _one_op(A.HorizontalFlip(p=1.0))
    out = t(image=_painted(100, 200, (10, 20, 50, 60)),
            bboxes=[[10, 20, 50, 60]], class_labels=[1])
    x0, y0, x1, y1 = [round(v) for v in out["bboxes"][0]]
    # width 200: x -> 200 - x, so [10, 50) becomes [150, 190).
    assert (x0, y0, x1, y1) == (150, 20, 190, 60)
    assert _white_extent(out["image"]) == (150, 20, 190, 60)


@requires_alb
def test_vertical_flip_maps_a_known_box_to_the_mirrored_row():
    t = _one_op(A.VerticalFlip(p=1.0))
    out = t(image=_painted(100, 200, (10, 20, 50, 60)),
            bboxes=[[10, 20, 50, 60]], class_labels=[1])
    x0, y0, x1, y1 = [round(v) for v in out["bboxes"][0]]
    # height 100: y -> 100 - y, so [20, 60) becomes [40, 80).
    assert (x0, y0, x1, y1) == (10, 40, 50, 80)
    assert _white_extent(out["image"]) == (10, 40, 50, 80)


@requires_alb
@pytest.mark.parametrize("seed", range(24))
def test_rotate90_box_still_bounds_the_pixels_it_labelled(seed):
    """RandomRotate90 draws k in 0..3, so pin the box to the CONTENT.

    Painting the box's interior and re-measuring it after the rotation
    checks the mapping for whichever k came up, without this test having to
    reach inside albumentations to force one.
    """
    box = (10, 20, 50, 60)
    t = A.Compose([A.Compose([A.RandomRotate90(p=1.0)], p=1.0)],
                  bbox_params=A.BboxParams(**_BBOX_PARAMS), seed=seed)
    out = t(image=_painted(100, 200, box), bboxes=[list(box)], class_labels=[1])
    got = tuple(round(v) for v in out["bboxes"][0])
    assert got == _white_extent(out["image"]), f"seed {seed}"
    # A quarter turn permutes the side lengths, never changes them.
    w, h = box[2] - box[0], box[3] - box[1]
    assert sorted((got[2] - got[0], got[3] - got[1])) == sorted((w, h))


@requires_alb
def test_rotate90_actually_transposes_a_landscape_frame():
    """Guards against RandomRotate90 silently drawing k = 0 every time."""
    box = [10, 20, 50, 60]
    shapes = set()
    for seed in range(24):
        t = A.Compose([A.RandomRotate90(p=1.0)],
                      bbox_params=A.BboxParams(**_BBOX_PARAMS), seed=seed)
        out = t(image=_painted(100, 200, tuple(box)), bboxes=[box],
                class_labels=[1])
        shapes.add(out["image"].shape[:2])
    assert shapes == {(100, 200), (200, 100)}, shapes


@requires_alb
def test_train_pipeline_carries_all_three_dihedral_ops():
    """They are the cheapest signal available here; losing one is silent."""
    names = _transform_names(build_train_transform(AUG_CFG))
    for op in ("HorizontalFlip", "VerticalFlip", "RandomRotate90"):
        assert op in names, f"{op} missing from the train pipeline"


@requires_alb
def test_train_pipeline_uses_affine_not_the_deprecated_shiftscalerotate():
    names = _transform_names(build_train_transform(AUG_CFG))
    assert "Affine" in names
    assert "ShiftScaleRotate" not in names


@requires_alb
def test_building_the_train_transform_emits_no_deprecation_warning():
    """albumentations 2.x warns on ShiftScaleRotate; a clean build proves
    the replacement is real rather than aliased."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_train_transform(AUG_CFG)
    msgs = [str(w.message) for w in caught
            if "special case of Affine" in str(w.message)]
    assert msgs == [], msgs


@requires_alb
def test_affine_preserves_the_configured_shift_scale_rotate_limits():
    """The ShiftScaleRotate -> Affine swap must not quietly retune anything."""
    aff = _find_transform(build_train_transform(AUG_CFG), "Affine")
    assert aff.translate_percent["x"] == (-0.04, 0.04)
    assert aff.translate_percent["y"] == (-0.04, 0.04)
    assert aff.scale["x"] == (0.9, 1.1)
    assert aff.rotate == (-4, 4)


@requires_alb
def test_camera_degradations_are_present():
    """Renders are pin-sharp and noiseless; real photos are neither."""
    names = _transform_names(build_train_transform(AUG_CFG))
    for op in ("MotionBlur", "Defocus", "ISONoise", "GaussNoise"):
        assert op in names, f"{op} missing from the train pipeline"


def _render_like_probe(h=96, w=160):
    """A stand-in for a render: mid-tones, structure, no clipped extremes.

    A flat mid-grey frame is useless as a probe here. RandomBrightnessContrast
    is ``img * alpha + beta * 255``, so a constant image saturates to 0 or 255
    under almost any draw and every configuration measures the same full-range
    spread - including the narrow one this test is meant to reject.
    """
    r = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 90 + 50 * (xx / w) + 30 * np.sin(yy / 9.0)
    img = np.repeat(base[:, :, None], 3, axis=2) + r.normal(0, 6, (h, w, 3))
    img[20:60, 30:80] += 55          # a bright part
    img[60:90, 100:150] -= 45        # a dark part
    return np.clip(img, 0, 255).astype(np.uint8)


@requires_alb
def test_photometric_range_actually_spans_dark_to_bright():
    """The point of the widening: measure it, do not assume it.

    Measured over 300 seeded draws of the probe above (mean 121, std 26):

        settings                              std of frame mean
        brightness/contrast 0.40, gamma 60-140      ~51
        brightness/contrast 0.55/0.50, gamma 45-190 ~67

    so the 60 threshold sits between the two and this assertion fails if the
    config is reverted to the old moderate ranges. Four disjoint 150-seed
    blocks measured 51.2 / 54.6 / 53.1 / 44.6 for the old settings against
    67.1 / 70.5 / 67.7 / 62.5 for the new, so the gap is not sampling noise.
    """
    probe = _render_like_probe()
    means = []
    for seed in range(300):
        t = build_train_transform(AUG_CFG)
        t.set_random_seed(seed)
        out = t(image=probe.copy(), bboxes=[[30, 20, 80, 60]],
                class_labels=[1])
        means.append(float(out["image"].mean()))
    spread = float(np.std(means))
    assert spread > 60.0, (
        f"frame-mean spread is only {spread:.1f}/255 - the photometric block "
        f"is back in the narrow band the widening was meant to leave")
    assert min(means) < 25.0, f"nothing dark: darkest mean {min(means):.1f}"
    assert max(means) > 230.0, f"nothing bright: brightest {max(means):.1f}"


def _flatten(t):
    for sub in getattr(t, "transforms", []):
        yield sub
        yield from _flatten(sub)


def _transform_names(t):
    return {type(s).__name__ for s in _flatten(t)}


def _find_transform(t, name):
    for s in _flatten(t):
        if type(s).__name__ == name:
            return s
    raise AssertionError(f"{name} not found in the pipeline")


# ------------------------------------------- the JOINT distribution of ops --
#
# Every other geometric test in this file is structurally blind to a
# correlation BETWEEN two ops, and that is how the worst defect in this
# module survived: they either force p=1.0 on one op (a correlation needs
# two) or rebuild the pipeline with a fresh seed per draw (which re-imposes
# the same correlation identically every time, so it cannot show up as a
# difference). The tests below are the shape that was missing - seed ONCE,
# draw a few thousand samples, and look at the ops jointly.
#
# What they would have caught: recog.seeding.seed_transform used to call
# albumentations' Compose.set_random_seed once, which in 2.0.8 hands all 17
# child transforms the SAME seed. HorizontalFlip and VerticalFlip are
# structurally identical - one p draw per call - so their streams never
# desynchronised and they fired in lockstep forever. Measured: 4 distinct
# orientations where 8 are possible, losing exactly the horizontal-only and
# vertical-only mirrors, and an Affine block that fired on 0.495 of flipped
# samples and 0.000 of unflipped ones.

_DIHEDRAL_SEED = 20260812


def _dihedral_images(img):
    """The eight symmetries of the square, as arrays. Exact, no resampling."""
    out = []
    for k in range(4):
        r = np.rot90(img, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def _geometry_only_cfg():
    """AUG_CFG with every resampling / pixel-value block switched off.

    The dihedral ops are exact permutations of pixels, so with the
    photometric, affine and camera blocks off, each output is bit-identical
    to one of the eight symmetries and the orientation can be identified by
    equality rather than inferred.
    """
    cfg = dict(AUG_CFG)
    cfg.update(p_photometric=0.0, p_geometric=0.0, p_camera=0.0)
    return cfg


def _fire_counter(pipeline, name):
    """Count how often one transform actually FIRES, without touching its RNG.

    Wrapping ``apply_with_params`` is the honest probe: albumentations calls
    it only after the transform's own ``p`` gate has passed, and the wrapper
    draws no random numbers of its own, so instrumenting the pipeline does
    not change the stream being measured.
    """
    node = _find_transform(pipeline, name)
    count = [0]
    original = node.apply_with_params

    def wrapped(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    node.apply_with_params = wrapped
    return count


@requires_alb
def test_one_seeding_call_leaves_all_eight_dihedral_orientations_reachable():
    """Seed once, draw many: the group must not collapse to a subgroup.

    The module docstring justifies the dihedral block on the grounds that it
    "multiplies the effective dataset by eight". This is the test that the
    eight are all actually reachable from a single seeding call - which is
    how the pipeline is seeded in training.py and seg_training.py, once,
    before the loop.
    """
    from recog.seeding import seed_transform

    t = build_train_transform(_geometry_only_cfg())
    seed_transform(t, _DIHEDRAL_SEED, "test")

    base = np.random.default_rng(7).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    orientations = _dihedral_images(base)
    counts = [0] * 8
    n = 2000
    for _ in range(n):
        out = t(image=base, bboxes=[[4, 4, 20, 30]], class_labels=[1])["image"]
        hit = [i for i, o in enumerate(orientations) if np.array_equal(o, out)]
        assert len(hit) == 1, (
            "output is not one of the eight symmetries; a non-exact op leaked "
            "into the geometry-only pipeline")
        counts[hit[0]] += 1

    missing = [i for i, c in enumerate(counts) if c == 0]
    assert not missing, (
        f"only {8 - len(missing)} of the 8 dihedral orientations occurred in "
        f"{n} draws from one seeding call (counts {counts}). The flips are "
        "firing in lockstep, so the group has collapsed to a subgroup and "
        "half the claimed augmentation does not exist.")
    # The four rotations carry 0.1875 each and the four reflections 0.0625
    # each (rot90's own gate fires at 0.5 and then picks k uniformly), so
    # the rarest expected count is 125. A floor of 40 is a wide margin on
    # sampling noise and still nowhere near the 0 the defect produced.
    assert min(counts) >= 40, counts


@requires_alb
def test_one_seeding_call_leaves_the_two_flips_independent():
    """The exact failure: HorizontalFlip and VerticalFlip fired together.

    Under the defect the (h, v) joint was {(0,0): 0.494, (1,1): 0.506} and
    the two off-diagonal cells - the horizontal-only and vertical-only
    mirrors, which are the ones that matter for cylinders lying at arbitrary
    angles - were empty.
    """
    from recog.seeding import seed_transform

    t = build_train_transform(_geometry_only_cfg())
    seed_transform(t, _DIHEDRAL_SEED, "test")
    h_count = _fire_counter(t, "HorizontalFlip")
    v_count = _fire_counter(t, "VerticalFlip")

    img = np.random.default_rng(7).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    n = 2000
    joint = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for _ in range(n):
        before = (h_count[0], v_count[0])
        t(image=img, bboxes=[[4, 4, 20, 30]], class_labels=[1])
        joint[(h_count[0] - before[0], v_count[0] - before[1])] += 1

    fractions = {k: v / n for k, v in joint.items()}
    for cell, frac in fractions.items():
        assert 0.20 <= frac <= 0.30, (
            f"P(hflip, vflip) = {fractions} — cell {cell} is at {frac:.4f} "
            "where two independent fair coins give 0.25. The two flips share "
            "an RNG stream.")


@requires_alb
def test_one_seeding_call_leaves_the_affine_gate_free_of_the_flip_gate():
    """The second-order consequence, on the block the config reasons about.

    configs/recognition.yaml spends fourteen lines tuning ``scale_limit`` on
    the premise that the affine block fires on ``p_geometric * Affine.p`` =
    25% of samples. Its MARGINAL rate was never the problem; under the
    defect it was 0.25 and the config looked satisfied. What was wrong was
    the conditional: affine fired on 0.495 of the samples that were flipped
    and 0.000 of the ones that were not, so no unflipped sample in the whole
    training set ever received a camera-pose perturbation.
    """
    from recog.seeding import seed_transform

    t = build_train_transform(AUG_CFG)
    seed_transform(t, _DIHEDRAL_SEED, "test")
    h_count = _fire_counter(t, "HorizontalFlip")
    a_count = _fire_counter(t, "Affine")

    img = np.random.default_rng(7).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    n = 2000
    with_flip = [0, 0]      # [affine fires, samples]
    without_flip = [0, 0]
    for _ in range(n):
        before = (h_count[0], a_count[0])
        t(image=img, bboxes=[[4, 4, 20, 30]], class_labels=[1])
        flipped = h_count[0] > before[0]
        affine = a_count[0] > before[1]
        bucket = with_flip if flipped else without_flip
        bucket[0] += int(affine)
        bucket[1] += 1

    assert min(with_flip[1], without_flip[1]) > 500, (with_flip, without_flip)
    rate_flipped = with_flip[0] / with_flip[1]
    rate_unflipped = without_flip[0] / without_flip[1]
    for label, rate in (("flipped", rate_flipped),
                        ("unflipped", rate_unflipped)):
        assert 0.19 <= rate <= 0.31, (
            f"P(affine | {label}) = {rate:.4f}, against the documented "
            f"p_geometric * Affine.p = 0.25. flipped={rate_flipped:.4f} "
            f"unflipped={rate_unflipped:.4f} — the affine gate is riding the "
            "flip gate's RNG stream.")


@requires_alb
def test_the_independent_child_seeds_are_still_perfectly_reproducible():
    """Independence must not have been bought with entropy.

    The fix for the lockstep gives every child transform its own seed. If
    those seeds came from anywhere but the run seed, augmentation would go
    back to being unreproducible - which is the defect recog.seeding exists
    to prevent, reintroduced by its own repair.
    """
    from recog.seeding import seed_transform

    def draw(seed, n=30):
        t = build_train_transform(AUG_CFG)
        seed_transform(t, seed, "test")
        img = np.random.default_rng(7).integers(
            0, 255, (48, 48, 3), dtype=np.uint8)
        return [int(t(image=img, bboxes=[[4, 4, 20, 30]],
                      class_labels=[1])["image"].sum()) for _ in range(n)]

    assert draw(_DIHEDRAL_SEED) == draw(_DIHEDRAL_SEED), (
        "two pipelines seeded identically produced different draws")
    assert draw(_DIHEDRAL_SEED) != draw(_DIHEDRAL_SEED + 1), (
        "two different seeds produced identical draws — the child seeds are "
        "not actually derived from the run seed")


# --------------------------------------------------------- mask-aware path --

def test_apply_with_mask_moves_image_and_mask_together():
    import numpy as np

    from recog.augmentation import (apply_with_mask,
                                    build_seg_train_transform)

    # A quadrant-coded image and a mask marking the same quadrant.
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32, :32] = 255
    mask = np.zeros((64, 64), np.int64)
    mask[:32, :32] = 2

    t = build_seg_train_transform({"p_flip": 1.0, "p_rot90": 0.0,
                                   "p_photometric": 0.0,
                                   "p_geometric": 0.0, "p_camera": 0.0})
    out = apply_with_mask(t, img, mask)
    bright = out["image"].reshape(-1, 3).max(axis=1) > 127
    labelled = out["mask"].reshape(-1) == 2
    agree = (bright == labelled).mean()
    assert agree > 0.97, f"image and mask diverged: {agree:.3f} agreement"


def test_apply_with_mask_never_invents_a_class():
    """The segmentation pipeline's geometric ops (flip / 90-degree
    rotation) are exact pixel permutations, so an invented class is
    unreachable by construction - that alone makes the class-membership
    assertion below near-vacuous. The variant-count assertion is what
    actually exercises the pipeline: without it this test would still
    pass even if apply_with_mask silently stopped touching the mask at
    all, which is exactly the failure mode this project has a documented
    history of (see the SDD ledger)."""
    import numpy as np

    from recog.augmentation import (apply_with_mask,
                                    build_seg_train_transform)

    img = np.zeros((64, 64, 3), np.uint8)
    mask = np.zeros((64, 64), np.int64)
    mask[:32] = 2
    mask[32:] = 4

    t = build_seg_train_transform({})
    variants = set()
    for _ in range(30):
        out = apply_with_mask(t, img, mask)
        assert set(np.unique(out["mask"])) <= {0, 2, 4}, (
            "an interpolated mask produced class 3, which is a different "
            "object entirely")
        variants.add(out["mask"].tobytes())

    assert len(variants) >= 2, (
        f"only {len(variants)} distinct mask variant(s) over 30 draws - "
        "the mask-invariant assertion above would pass even if the "
        "pipeline stopped moving the mask entirely")


def test_seg_val_transform_is_geometrically_identity():
    import numpy as np

    from recog.augmentation import apply_with_mask, build_seg_val_transform

    mask = np.zeros((32, 32), np.int64)
    mask[:16, :16] = 5
    img = np.zeros((32, 32, 3), np.uint8)
    out = apply_with_mask(build_seg_val_transform({}), img, mask)
    assert np.array_equal(out["mask"], mask)
