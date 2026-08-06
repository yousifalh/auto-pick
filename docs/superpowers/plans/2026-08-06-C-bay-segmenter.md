# Bay Segmenter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a six-channel per-ROI semantic segmenter that, given a cartridge crop, labels every pixel as background / cartridge / bay / electronics / obstruction / battery — and measure it against the metrics that decide whether it is worth shipping.

**Architecture:** Plan B's `instances_seg.json` is rasterised into per-cartridge crops at load time, with the crop box jittered to match the detector's measured box error. A `deeplabv3_mobilenet_v3_large` with six output channels trains on those crops under cross-entropy + Dice. Inference is fp16, batched, at 256², because Plan B's spec §3.2 measured that as the only configuration fitting the 50 ms budget.

**Tech Stack:** PyTorch 2.x + torchvision, NumPy, albumentations (with a numpy fallback, following `recog/augmentation.py`'s existing pattern), pytest.

## Global Constraints

- Channel order is fixed and load-bearing: `0 background, 1 cartridge, 2 bay, 3 electronics, 4 obstruction, 5 battery`. Plan D's arbitration indexes these directly.
- `bay` is the `placement_area` class from `SEG_CLASSES`; the channel is named for what it is in the crop.
- Inference must run **fp16, batched, at 256 × 256**. All three are load-bearing — spec §3.2 measured 18.5 ms for 8 cartridges in that configuration against 101 ms for fp32-unbatched-384².
- Training crops come from **jittered** boxes, never ground-truth boxes. At inference the crop comes from the detector.
- The detector, its checkpoints, `configs/recognition.yaml` and `recog/training.py` are not modified by this plan.
- Albumentations is optional. `recog/augmentation.py` ships a numpy fallback and the mask path must too, or `pytest` breaks in environments without it.

---

## File Structure

| File | Responsibility |
|---|---|
| `recog/seg_dataset.py` | **New.** Reads `instances_seg.json`, emits `(crop_uint8, label_map_int64)` pairs with jittered boxes. |
| `recog/augmentation.py` | Gains `apply_with_mask` and mask-aware transform builders. |
| `recog/bay_segmenter.py` | **New.** Model factory + `BaySegmenter` inference wrapper (fp16, batched). |
| `recog/seg_training.py` | **New.** Training loop, CE + Dice, checkpointing. |
| `recog/seg_evaluate.py` | **New.** Mask IoU, boundary displacement in mm, latency table. |
| `configs/segmentation.yaml` | **New.** Its own config; `recognition.yaml` is not touched. |
| `tests/test_seg_dataset.py` | **New.** |
| `tests/test_bay_segmenter.py` | **New.** |

---

### Task 1: Crop dataset from the COCO-RLE sidecar

**Files:**
- Create: `recog/seg_dataset.py`
- Test: `tests/test_seg_dataset.py`

**Interfaces:**
- Consumes: `recog.synth3d.annotate.rle_decode(rle) -> np.ndarray` (Plan B Task 7), `instances_seg.json`
- Produces:
  - `SEG_CHANNELS: Dict[str, int]` — `{"background": 0, "cartridge": 1, "bay": 2, "electronics": 3, "obstruction": 4, "battery": 5}`
  - `jitter_box(box, rng, frac) -> Tuple[int, int, int, int]`
  - `rasterise_crop(anns, box, out_size) -> np.ndarray` — `(out_size, out_size)` int64 label map
  - `BaySegDataset(coco_path, img_dir, out_size=256, jitter_frac=0.06, train=True)` — duck-typed dataset (`__len__` / `__getitem__`, not a `torch.utils.data.Dataset` subclass, so the module imports without torch) yielding `(image_chw_float32, label_hw_int64)`

The label map is painted in a fixed order so overlaps resolve deterministically: `cartridge` first, then `bay` over it, then `electronics`, `obstruction`, `battery` on top. Painting battery last is what makes a seated cell win over the bay floor it covers — which is the modal definition, encoded as a paint order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_seg_dataset.py`:

```python
"""Per-ROI crop dataset built from the COCO-RLE sidecar."""
from __future__ import annotations

import numpy as np
import pytest

from recog.seg_dataset import (SEG_CHANNELS, jitter_box, rasterise_crop)
from recog.synth3d.annotate import rle_encode


def _ann(cls, mask, cat_id):
    ys, xs = np.nonzero(mask)
    return {"class": cls, "category_id": cat_id,
            "segmentation": rle_encode(mask),
            "bbox_xyxy": [int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1]}


def test_channel_order_is_the_contract():
    """Plan D's arbitration indexes these directly. Reordering them
    silently changes which mask is subtracted from which."""
    assert SEG_CHANNELS == {
        "background": 0, "cartridge": 1, "bay": 2,
        "electronics": 3, "obstruction": 4, "battery": 5,
    }


def test_jitter_stays_within_the_requested_fraction():
    rng = np.random.default_rng(0)
    box = (100, 200, 200, 400)          # 100 x 200
    for _ in range(500):
        x0, y0, x1, y1 = jitter_box(box, rng, 0.10)
        assert abs(x0 - 100) <= 10 and abs(x1 - 200) <= 10
        assert abs(y0 - 200) <= 20 and abs(y1 - 400) <= 20
        assert x1 > x0 and y1 > y0


def test_zero_jitter_is_the_identity():
    rng = np.random.default_rng(0)
    assert jitter_box((10, 20, 30, 40), rng, 0.0) == (10, 20, 30, 40)


def test_rasterise_paints_battery_over_bay():
    """A cell seated in a bay must win. That paint order IS the modal
    definition of placement_area: the free floor is what is left."""
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[5:35, 5:35] = 1
    cell = np.zeros((H, W), np.uint8); cell[10:20, 10:20] = 1
    anns = [_ann("placement_area", bay, 4), _ann("battery", cell, 1)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=40)
    assert lab[15, 15] == SEG_CHANNELS["battery"]
    assert lab[30, 30] == SEG_CHANNELS["bay"]
    assert lab[0, 0] == SEG_CHANNELS["background"]


def test_rasterise_paints_electronics_and_obstruction_over_bay():
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:40, 0:40] = 1
    pcb = np.zeros((H, W), np.uint8); pcb[0:10, :] = 1
    glue = np.zeros((H, W), np.uint8); glue[20:24, 20:24] = 1
    anns = [_ann("placement_area", bay, 4),
            _ann("electronics_module", pcb, 3),
            _ann("obstruction", glue, 5)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=40)
    assert lab[5, 5] == SEG_CHANNELS["electronics"]
    assert lab[22, 22] == SEG_CHANNELS["obstruction"]
    assert lab[35, 35] == SEG_CHANNELS["bay"]


def test_rasterise_resizes_with_nearest_neighbour():
    """Labels must never be interpolated: averaging class 2 and 4 gives
    class 3, which is a different object."""
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:40, 0:20] = 1
    pcb = np.zeros((H, W), np.uint8); pcb[0:40, 20:40] = 1
    anns = [_ann("placement_area", bay, 4),
            _ann("electronics_module", pcb, 3)]

    lab = rasterise_crop(anns, (0, 0, W, H), out_size=8)
    assert set(np.unique(lab)) <= {SEG_CHANNELS["bay"],
                                   SEG_CHANNELS["electronics"]}


def test_crop_outside_the_annotation_is_all_background():
    H = W = 40
    bay = np.zeros((H, W), np.uint8); bay[0:10, 0:10] = 1
    lab = rasterise_crop([_ann("placement_area", bay, 4)],
                         (20, 20, 40, 40), out_size=16)
    assert (lab == SEG_CHANNELS["background"]).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_seg_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.seg_dataset'`

- [ ] **Step 3: Implement the crop machinery**

Create `recog/seg_dataset.py`:

```python
"""
recog.seg_dataset - per-ROI crops for the bay segmenter.

Reads the COCO-RLE sidecar recog/generate3d.py writes and turns each
cartridge annotation into one training crop: the image inside a jittered
cartridge box, plus a dense label map over the same window.

Crops come from JITTERED boxes on purpose. At inference the crop comes
from the detector, whose boxes are not ground truth; training on perfect
boxes would bake in a distribution shift the model first meets in
deployment.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recog.synth3d.annotate import rle_decode

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                       # pragma: no cover - optional
    _HAVE_CV2 = False


# Channel order is a CONTRACT. Plan D's arbitration indexes these
# directly, so reordering silently changes which mask is subtracted from
# which. `bay` is SEG_CLASSES' `placement_area`, named here for what it
# is inside a crop.
SEG_CHANNELS: Dict[str, int] = {
    "background": 0,
    "cartridge": 1,
    "bay": 2,
    "electronics": 3,
    "obstruction": 4,
    "battery": 5,
}

# Paint order. Later wins. Battery is painted last so a cell seated in a
# bay covers the floor beneath it - that paint order IS the spec's modal
# ruling 4, expressed as code rather than prose.
_PAINT_ORDER: Sequence[Tuple[str, str]] = (
    ("cartridge", "cartridge"),
    ("placement_area", "bay"),
    ("electronics_module", "electronics"),
    ("obstruction", "obstruction"),
    ("battery", "battery"),
)


def jitter_box(box, rng: np.random.Generator, frac: float):
    """Perturb each edge by up to ``frac`` of the box's own side length.

    Independent per edge, so the crop shifts AND rescales - which is what
    a detector's box error actually looks like.
    """
    x0, y0, x1, y1 = box
    if frac <= 0.0:
        return (int(x0), int(y0), int(x1), int(y1))
    w, h = x1 - x0, y1 - y0
    dx0, dx1 = rng.uniform(-frac, frac, 2) * w
    dy0, dy1 = rng.uniform(-frac, frac, 2) * h
    nx0, nx1 = int(round(x0 + dx0)), int(round(x1 + dx1))
    ny0, ny1 = int(round(y0 + dy0)), int(round(y1 + dy1))
    if nx1 <= nx0:
        nx1 = nx0 + 1
    if ny1 <= ny0:
        ny1 = ny0 + 1
    return (nx0, ny0, nx1, ny1)


def _resize_nearest(a: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize. Labels must never be interpolated:
    the mean of class 2 and class 4 is class 3, a different object."""
    if a.shape[0] == size and a.shape[1] == size:
        return a
    if _HAVE_CV2:
        return cv2.resize(a.astype(np.uint8), (size, size),
                          interpolation=cv2.INTER_NEAREST).astype(a.dtype)
    ys = (np.arange(size) * a.shape[0] / size).astype(int).clip(0, a.shape[0] - 1)
    xs = (np.arange(size) * a.shape[1] / size).astype(int).clip(0, a.shape[1] - 1)
    return a[ys][:, xs]


def rasterise_crop(anns: Sequence[dict], box, out_size: int) -> np.ndarray:
    """Dense label map for the window ``box``, resized to ``out_size``."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = y1 - y0, x1 - x0
    label = np.zeros((h, w), dtype=np.int64)

    by_class: Dict[str, List[dict]] = {}
    for a in anns:
        by_class.setdefault(a["class"], []).append(a)

    for cls, channel in _PAINT_ORDER:
        for a in by_class.get(cls, ()):
            full = rle_decode(a["segmentation"])
            fh, fw = full.shape
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(fw, x1), min(fh, y1)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            sub = full[sy0:sy1, sx0:sx1]
            label[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][sub > 0] = \
                SEG_CHANNELS[channel]

    return _resize_nearest(label, out_size)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_seg_dataset.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Add the Dataset class**

Append to `recog/seg_dataset.py`:

```python
class BaySegDataset:
    """One crop per cartridge annotation in the COCO sidecar.

    Not a torch.utils.data.Dataset subclass at import time: recog.dataset
    already carries a shim for the torch-free environment and this module
    follows the same rule, so the tests run without torch installed.
    """

    def __init__(self, coco_path: str, img_dir: str, out_size: int = 256,
                 jitter_frac: float = 0.06, train: bool = True,
                 transform=None, seed: int = 0) -> None:
        with open(coco_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        self.img_dir = img_dir
        self.out_size = int(out_size)
        self.jitter_frac = float(jitter_frac) if train else 0.0
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        names = {c["id"]: c["name"] for c in doc["categories"]}
        images = {im["id"]: im for im in doc["images"]}

        by_image: Dict[int, List[dict]] = {}
        for a in doc["annotations"]:
            rec = dict(a)
            rec["class"] = names[a["category_id"]]
            rec["bbox_xyxy"] = [a["bbox"][0], a["bbox"][1],
                                a["bbox"][0] + a["bbox"][2],
                                a["bbox"][1] + a["bbox"][3]]
            by_image.setdefault(a["image_id"], []).append(rec)

        # One sample per cartridge. A frame with three cartridges is
        # three crops, not one - the segmenter never sees a whole frame.
        self.samples: List[Tuple[dict, List[dict], dict]] = []
        for img_id, anns in by_image.items():
            for a in anns:
                if a["class"] == "cartridge":
                    self.samples.append((images[img_id], anns, a))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        img_meta, anns, cart = self.samples[idx]
        path = os.path.join(self.img_dir, img_meta["file_name"])
        image = np.asarray(Image.open(path).convert("RGB"))

        box = jitter_box(cart["bbox_xyxy"], self.rng, self.jitter_frac)
        label = rasterise_crop(anns, box, self.out_size)

        x0, y0, x1, y1 = box
        pad_t, pad_l = max(0, -y0), max(0, -x0)
        crop = image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if pad_t or pad_l or crop.shape[0] != y1 - y0 or crop.shape[1] != x1 - x0:
            padded = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            padded[pad_t:pad_t + crop.shape[0],
                   pad_l:pad_l + crop.shape[1]] = crop
            crop = padded
        crop = _resize_nearest(crop, self.out_size) if not _HAVE_CV2 else \
            cv2.resize(crop, (self.out_size, self.out_size),
                       interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            from recog.augmentation import apply_with_mask
            out = apply_with_mask(self.transform, crop, label)
            crop, label = out["image"], out["mask"]

        chw = np.ascontiguousarray(crop.transpose(2, 0, 1)).astype(np.float32)
        return torch.from_numpy(chw / 255.0), torch.from_numpy(label.astype(np.int64))
```

Note the crop resize uses **linear** for the image and **nearest** for the label. Using nearest on the image throws away detail; using linear on the label invents classes.

- [ ] **Step 6: Add an integration test against a real sidecar**

Append to `tests/test_seg_dataset.py`:

```python
import os

import pytest

DEV = os.path.join(os.path.dirname(__file__), "..", "recog", "dev3d")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(DEV, "instances_seg.json")),
    reason="run Plan B Task 7 first to generate recog/dev3d")
def test_dataset_yields_crops_with_every_channel_present_somewhere():
    torch = pytest.importorskip("torch")
    from recog.seg_dataset import BaySegDataset

    ds = BaySegDataset(os.path.join(DEV, "instances_seg.json"),
                       os.path.join(DEV, "images"), out_size=128)
    assert len(ds) > 0

    seen = set()
    for i in range(min(len(ds), 24)):
        img, lab = ds[i]
        assert img.shape == (3, 128, 128)
        assert lab.shape == (128, 128)
        assert img.dtype == torch.float32 and lab.dtype == torch.int64
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        assert int(lab.max()) < 6
        seen.update(int(v) for v in lab.unique())

    assert {0, 1, 2, 3} <= seen, f"missing channels; saw {sorted(seen)}"
```

- [ ] **Step 7: Commit**

```bash
git add recog/seg_dataset.py tests/test_seg_dataset.py
git commit -m "feat(recog): per-ROI crop dataset from the COCO-RLE sidecar

One crop per cartridge annotation, with the crop box jittered - at
inference the box comes from the detector, so training on ground-truth
boxes would bake in a shift the model first meets in deployment.

Labels are painted cartridge -> bay -> electronics -> obstruction ->
battery, later winning. Painting battery last is what makes a seated
cell cover the floor beneath it, which is the spec's modal ruling 4
expressed as code.

Label maps resize nearest-neighbour only: the mean of class 2 and class
4 is class 3, a different object."
```

---

### Task 2: Mask-aware augmentation

**Files:**
- Modify: `recog/augmentation.py`
- Test: `tests/test_augmentation.py` (append)

**Interfaces:**
- Consumes: existing `build_train_transform(cfg)` / `_FallbackTransform`
- Produces:
  - `build_seg_train_transform(cfg)` / `build_seg_val_transform(cfg)`
  - `apply_with_mask(transform, image, mask) -> {"image": np.ndarray, "mask": np.ndarray}`

The existing `apply()` takes `bboxes` and `class_labels`; a segmentation transform takes a `mask` target instead. Geometric ops must move both together, and the mask must never be interpolated.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_augmentation.py`:

```python
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
                                   "p_geometric": 0.0})
    out = apply_with_mask(t, img, mask)
    bright = out["image"].reshape(-1, 3).max(axis=1) > 127
    labelled = out["mask"].reshape(-1) == 2
    agree = (bright == labelled).mean()
    assert agree > 0.97, f"image and mask diverged: {agree:.3f} agreement"


def test_apply_with_mask_never_invents_a_class():
    import numpy as np

    from recog.augmentation import (apply_with_mask,
                                    build_seg_train_transform)

    img = np.zeros((64, 64, 3), np.uint8)
    mask = np.zeros((64, 64), np.int64)
    mask[:32] = 2
    mask[32:] = 4

    t = build_seg_train_transform({})
    for _ in range(30):
        out = apply_with_mask(t, img, mask)
        assert set(np.unique(out["mask"])) <= {0, 2, 4}, (
            "an interpolated mask produced class 3, which is a different "
            "object entirely")


def test_seg_val_transform_is_geometrically_identity():
    import numpy as np

    from recog.augmentation import apply_with_mask, build_seg_val_transform

    mask = np.zeros((32, 32), np.int64)
    mask[:16, :16] = 5
    img = np.zeros((32, 32, 3), np.uint8)
    out = apply_with_mask(build_seg_val_transform({}), img, mask)
    assert np.array_equal(out["mask"], mask)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_augmentation.py -v -k mask`
Expected: FAIL — `ImportError: cannot import name 'apply_with_mask'`

- [ ] **Step 3: Implement**

Append to `recog/augmentation.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_augmentation.py -v`
Expected: PASS, including the three new mask tests and every existing test unchanged.

- [ ] **Step 5: Commit**

```bash
git add recog/augmentation.py tests/test_augmentation.py
git commit -m "feat(recog): mask-aware augmentation for the bay segmenter

apply_with_mask routes a dense label map through the same photometric
pipeline the detector uses, with nearest-neighbour resampling for every
geometric op - interpolating labels averages class 2 and class 4 into
class 3, a different object.

No Affine in the segmentation geometric block: the crop already comes
from a jittered box, and shifting it again would double-count the
detector error the jitter exists to model.

Ships a numpy fallback, matching the existing optional-albumentations
contract."
```

---

### Task 3: Model factory and batched fp16 inference

**Files:**
- Create: `recog/bay_segmenter.py`
- Test: `tests/test_bay_segmenter.py`

**Interfaces:**
- Consumes: `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`
- Produces:
  - `build_segmenter(num_classes=6, pretrained=True) -> nn.Module`
  - `BaySegmenter(checkpoint, device="cuda", crop_size=256, half=True)`
  - `BaySegmenter.segment_batch(crops: Sequence[np.ndarray]) -> List[np.ndarray]` — one `(h, w)` int8 label map per crop, resized back to each crop's own size
  - `BaySegmenter.segment(crop: np.ndarray) -> np.ndarray` — single-crop convenience, calls `segment_batch`

`segment_batch` is the primary API and `segment` the convenience wrapper, not the other way round. Spec §3.2 measured 8 cartridges at 101 ms looped versus 18.5 ms batched; an API that makes the loop the easy path will get used that way.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bay_segmenter.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_bay_segmenter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.bay_segmenter'`

- [ ] **Step 3: Implement**

Create `recog/bay_segmenter.py`:

```python
"""
recog.bay_segmenter - per-ROI semantic segmentation of a cartridge crop.

Six channels, fixed order:
    0 background   1 cartridge   2 bay
    3 electronics  4 obstruction 5 battery

`bay` is SEG_CLASSES' `placement_area`. plan.placement_area's
SegmentationPlacementAreaExtractor indexes these numbers directly.

Inference is BATCHED and fp16 by default because the latency budget has
no slack. Measured on an RTX 3060 at 384x384: one crop 12.6 ms, eight
crops looped 101 ms, eight batched fp32 59.6 ms, eight batched fp16 at
256x256 18.5 ms - against FDR 10.4's 50 ms end-to-end budget. Only the
last configuration fits.

256 is not a compromise. A PowerCore26800 at the generator's framing is
about 131x288 px, so 256x256 is at or above native crop resolution; 384
was upsampling and paid 3x for it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def build_segmenter(num_classes: int = 6, pretrained: bool = True):
    """DeepLabv3 + MobileNetV3-Large with `num_classes` outputs."""
    import torch.nn as nn
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)

    weights = (DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
               if pretrained else None)
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    # Re-head for our class count. The aux classifier is retained during
    # training (it regularises) and ignored at inference.
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(10, num_classes, kernel_size=1)
    return model


class BaySegmenter:
    """Inference wrapper. `segment_batch` is the primary entry point."""

    def __init__(self, checkpoint: Optional[str] = None,
                 device: str = "cuda", crop_size: int = 256,
                 half: bool = True, num_classes: int = 6) -> None:
        import torch

        self.crop_size = int(crop_size)
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available())
            else "cpu")
        # fp16 on CPU is slower than fp32, and silently so. Refuse rather
        # than let a CPU fallback quietly halve throughput.
        self.half = bool(half) and self.device.type == "cuda"

        self.model = build_segmenter(num_classes=num_classes,
                                     pretrained=checkpoint is None)
        if checkpoint:
            # weights_only=True is not optional. The default unpickles
            # arbitrary Python objects, so loading a checkpoint becomes
            # arbitrary code execution by whoever produced the file. A
            # segmenter checkpoint holds tensors and nothing else.
            state = torch.load(checkpoint, map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model = self.model.to(self.device).eval()
        if self.half:
            self.model = self.model.half()

    def segment_batch(self, crops: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Label maps for `crops`, each resized back to its own size."""
        import torch
        import torch.nn.functional as F

        if not len(crops):
            return []

        n = self.crop_size
        batch = np.empty((len(crops), 3, n, n), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i] = _resize_rgb(c, n).transpose(2, 0, 1) / 255.0

        x = torch.from_numpy(batch).to(self.device)
        if self.half:
            x = x.half()

        with torch.no_grad():
            logits = self.model(x)["out"]
            pred = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()

        out: List[np.ndarray] = []
        for i, c in enumerate(crops):
            h, w = c.shape[:2]
            out.append(_resize_nearest_2d(pred[i], h, w).astype(np.int8))
        return out

    def segment(self, crop: np.ndarray) -> np.ndarray:
        """Single crop. Prefer `segment_batch` - see the module docstring."""
        return self.segment_batch([crop])[0]


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:                     # pragma: no cover - optional
        ys = (np.arange(size) * img.shape[0] / size).astype(int)
        xs = (np.arange(size) * img.shape[1] / size).astype(int)
        return img[ys.clip(0, img.shape[0] - 1)][:, xs.clip(0, img.shape[1] - 1)]


def _resize_nearest_2d(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest only. A label map must never be interpolated."""
    ys = (np.arange(h) * a.shape[0] / h).astype(int).clip(0, a.shape[0] - 1)
    xs = (np.arange(w) * a.shape[1] / w).astype(int).clip(0, a.shape[1] - 1)
    return a[ys][:, xs]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bay_segmenter.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add recog/bay_segmenter.py tests/test_bay_segmenter.py
git commit -m "feat(recog): bay segmenter model factory and batched inference

DeepLabv3 + MobileNetV3-Large, six channels in a fixed order that
plan.placement_area indexes directly.

segment_batch is the primary API and segment the convenience wrapper,
deliberately that way round: measured on an RTX 3060, eight crops looped
cost 101 ms against 18.5 ms batched at fp16/256, and the 50 ms
end-to-end budget has no room for the loop. An API whose easy path is
the slow one gets used the slow way.

fp16 is refused on CPU, where it is slower than fp32 and silently so."
```

---

### Task 4: Training

**Files:**
- Create: `recog/seg_training.py`, `configs/segmentation.yaml`
- Test: `tests/test_bay_segmenter.py` (append)

**Interfaces:**
- Consumes: `BaySegDataset` (Task 1), `build_seg_train_transform` (Task 2), `build_segmenter` (Task 3)
- Produces:
  - `dice_loss(logits, target, num_classes, eps=1.0) -> Tensor`
  - `combined_loss(logits, target, num_classes, dice_weight) -> Tensor`
  - `train(cfg: dict) -> None`
  - CLI: `python -m recog.seg_training --config configs/segmentation.yaml`

`obstruction` occupies a small area and is absent from ~40 % of samples, so plain cross-entropy buys almost nothing by predicting it and would rather not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay_segmenter.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_bay_segmenter.py -v -k dice`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.seg_training'`

- [ ] **Step 3: Implement the loss and training loop**

Create `recog/seg_training.py` with the loss first:

```python
"""
recog.seg_training - training loop for the bay segmenter.

Loss is cross-entropy plus Dice. `obstruction` covers a small area and is
absent from roughly 40% of crops, so plain cross-entropy can score well
while never predicting it: getting a rare small class wrong costs almost
nothing per pixel. Dice is computed per class over the batch, which
makes a class's contribution independent of how many pixels it occupies.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict


def dice_loss(logits, target, num_classes: int, eps: float = 1.0):
    """Mean soft Dice over the classes PRESENT in `target`.

    Classes absent from the batch are skipped rather than scored as a
    perfect miss. Without that, ~40% of batches would carry a constant
    gradient pushing toward an obstruction that is not in the image.
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)

    present = onehot.sum(dims) > 0
    if not bool(present.any()):
        return torch.zeros((), device=logits.device)
    return 1.0 - dice[present].mean()


def combined_loss(logits, target, num_classes: int,
                  dice_weight: float = 0.5):
    import torch.nn.functional as F
    return (F.cross_entropy(logits, target)
            + dice_weight * dice_loss(logits, target, num_classes))
```

- [ ] **Step 4: Run the loss tests**

Run: `pytest tests/test_bay_segmenter.py -v -k dice`
Expected: PASS, 3 tests

- [ ] **Step 5: Add the training loop and config**

Append the loop to `recog/seg_training.py`. Follow `recog/training.py`'s existing shape: `_require_torch()`, a train/val split by index with a fixed seed, cosine LR, per-epoch validation, and **always save `last.pt` as well as `best.pt`** — commit `dedf700` fixed exactly that bug in the detector's loop and the same bug is easy to reintroduce here.

Selection metric: **mean IoU over `bay`, `electronics` and `obstruction` only**. Including `background` and `cartridge` would let a model that gets the big easy regions right look good while failing the three classes the placement mask is built from.

Create `configs/segmentation.yaml`:

```yaml
# Bay segmenter. Separate from recognition.yaml on purpose: the detector's
# config, checkpoints and published numbers are not touched by this work.
model:
  num_classes: 6        # background, cartridge, bay, electronics, obstruction, battery
  pretrained: true
  crop_size: 256        # at or above native crop resolution; see bay_segmenter
  half: true            # fp16 - load-bearing for the 50 ms budget

dataset:
  coco_path: recog/dataset3d/instances_seg.json
  img_dir: recog/dataset3d/images
  train_val_split: 0.85
  # Jitter magnitude should match the detector's measured box error, not be
  # chosen. Run recog.eval_real and read the median IoU of matched cartridge
  # boxes; 0.06 corresponds to roughly IoU 0.88.
  jitter_frac: 0.06

training:
  epochs: 40
  batch_size: 8         # crops are 256px, far smaller than the detector's
  num_workers: 0        # Windows spawn; raise on Linux
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: cosine
  dice_weight: 0.5
  checkpoint_dir: recog/checkpoints/seg
  # Selection is on bay/electronics/obstruction IoU only. Including
  # background and cartridge lets a model that gets the big easy regions
  # right mask a failure on the three classes the placement mask needs.
  select_on: [bay, electronics, obstruction]

augmentation:
  brightness_limit: 0.55
  contrast_limit: 0.50
  gamma_limit: [45, 190]
  hue_shift_limit: 15
  sat_shift_limit: 30
  val_shift_limit: 20
  motion_blur_limit: [3, 11]
  gauss_noise_std_range: [0.02, 0.14]
  p_photometric: 0.85
  p_flip: 0.5
  p_rot90: 0.5
  p_camera: 0.6
```

- [ ] **Step 6: Train a short run and confirm it learns**

Run:
```bash
python -m recog.seg_training --config configs/segmentation.yaml
```

Expected: validation mean IoU over the three selected classes rises above 0.5 within the first ten epochs on a dataset of a few hundred crops. If it sits near zero, check the label maps first with `verify3d --masks` (Plan B Task 8) — a segmenter cannot learn labels that are wrong.

- [ ] **Step 7: Commit**

```bash
git add recog/seg_training.py configs/segmentation.yaml tests/test_bay_segmenter.py
git commit -m "feat(recog): bay segmenter training loop

Cross-entropy plus per-class Dice. obstruction is small and absent from
~40% of crops, so plain CE scores well while never predicting it.

Dice skips classes absent from the batch rather than scoring them as a
perfect miss - otherwise 40% of batches carry a constant gradient toward
an obstruction that is not in the image.

Checkpoint selection is on bay/electronics/obstruction IoU only.
Including background and cartridge would let a model that gets the big
easy regions right mask a failure on the three the placement mask needs.

Saves last.pt as well as best.pt - see dedf700 for the same bug in the
detector's loop."
```

---

### Task 5: Evaluation — the metrics that decide

**Files:**
- Create: `recog/seg_evaluate.py`
- Test: `tests/test_bay_segmenter.py` (append)

**Interfaces:**
- Produces:
  - `per_class_iou(pred, target, num_classes) -> Dict[str, float]`
  - `boundary_displacement_mm(pred, target, cls, mm_per_px) -> float` — mean distance from each predicted boundary pixel to the nearest ground-truth boundary pixel
  - `signed_area_error_mm2(pred, target, cls, mm_per_px) -> Tuple[float, float]` — `(optimistic_mm2, conservative_mm2)`
  - `latency_table(segmenter, counts=(1, 2, 4, 8)) -> List[dict]`
  - CLI: `python -m recog.seg_evaluate --checkpoint ... --config ...`

IoU alone hides the number that chose this architecture. Spec §3.1 rejected a Mask R-CNN head on a **2.9 × 6.4 mm** quantisation figure; reporting only IoU would make that argument unfalsifiable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bay_segmenter.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_bay_segmenter.py -v -k "area_error or boundary or per_class"`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.seg_evaluate'`

- [ ] **Step 3: Implement**

Create `recog/seg_evaluate.py`:

```python
"""
recog.seg_evaluate - the metrics that decide whether the segmenter ships.

Per-class IoU is table stakes. Three others carry the argument:

  boundary_displacement_mm  the quantity that chose this architecture.
                            Spec 3.1 rejected a Mask R-CNN head on a
                            2.9 x 6.4 mm quantisation figure; reporting
                            only IoU makes that argument unfalsifiable.

  signed_area_error_mm2     optimistic error sites a cell where one
                            cannot go, a damage event. Conservative
                            error refuses where one can, a lost cell.
                            One unsigned number hides the difference.

  latency_table             the 50 ms budget has no slack. See
                            bay_segmenter's module docstring.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Derived from the dataset's contract rather than restated. Two
# hand-written copies of the same order drift, and the drift is silent:
# the metrics would simply be reported under the wrong class names.
from recog.seg_dataset import SEG_CHANNELS

CHANNEL_NAMES = [name for name, _ in
                 sorted(SEG_CHANNELS.items(), key=lambda kv: kv[1])]


def per_class_iou(pred: np.ndarray, target: np.ndarray,
                  num_classes: int = 6) -> Dict[str, float]:
    """IoU per class. NaN where a class appears in neither array.

    NaN rather than 0.0: a class that was never present was never
    tested, and scoring it zero drags the mean down for a question that
    was not asked.
    """
    out: Dict[str, float] = {}
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = int((p | t).sum())
        out[CHANNEL_NAMES[c]] = (float(np.nan) if union == 0
                                 else float((p & t).sum()) / union)
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` with at least one 4-neighbour outside it."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    interior = (pad[:-2, 1:-1] & pad[2:, 1:-1] &
                pad[1:-1, :-2] & pad[1:-1, 2:])
    return m & ~interior


def boundary_displacement_mm(pred: np.ndarray, target: np.ndarray,
                             cls: int, mm_per_px: float) -> float:
    """Mean distance from each predicted boundary pixel to the nearest
    ground-truth boundary pixel, in millimetres."""
    pb = _boundary(pred == cls)
    tb = _boundary(target == cls)
    if not pb.any() or not tb.any():
        return float("nan")

    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~tb)
    except Exception:
        ty, tx = np.nonzero(tb)
        yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]]
        dist = np.min(
            np.sqrt((yy[..., None] - ty) ** 2 + (xx[..., None] - tx) ** 2),
            axis=-1)

    return float(dist[pb].mean() * mm_per_px)


def signed_area_error_mm2(pred: np.ndarray, target: np.ndarray,
                          cls: int, mm_per_px: float
                          ) -> Tuple[float, float]:
    """``(optimistic_mm2, conservative_mm2)`` for class ``cls``.

    Optimistic = predicted placeable where truth is not. That is the
    error that puts a cell on a PCB.
    """
    p, t = pred == cls, target == cls
    px_mm2 = mm_per_px * mm_per_px
    return (float((p & ~t).sum()) * px_mm2,
            float((~p & t).sum()) * px_mm2)


def latency_table(segmenter, counts: Sequence[int] = (1, 2, 4, 8),
                  crop_hw: Tuple[int, int] = (131, 288),
                  repeats: int = 20) -> List[dict]:
    """Wall-clock per batch size, against the 50 ms PPR budget."""
    import time

    rows: List[dict] = []
    for n in counts:
        crops = [np.zeros((crop_hw[0], crop_hw[1], 3), np.uint8)
                 for _ in range(n)]
        for _ in range(3):
            segmenter.segment_batch(crops)
        t0 = time.perf_counter()
        for _ in range(repeats):
            segmenter.segment_batch(crops)
        total_ms = (time.perf_counter() - t0) / repeats * 1000.0
        rows.append({"cartridges": n,
                     "total_ms": round(total_ms, 1),
                     "per_cartridge_ms": round(total_ms / n, 2),
                     "within_50ms_budget": total_ms <= 50.0})
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bay_segmenter.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Add the CLI and produce the receipt**

Add an `argparse` CLI to `recog/seg_evaluate.py` that loads a checkpoint and the validation split, reports per-class IoU with instance counts, boundary displacement in mm for `bay`/`electronics`/`obstruction`, signed area error, and the latency table. Write it to `docs/receipts/seg_eval.txt`, matching how `recog/eval_real.py` reports.

`mm_per_px` for synthetic data is the generator's framing: `layout.area[0] * 1000 / render.res[0]`, which at the defaults is `800 / 1280 = 0.625`. Compute it from config rather than hardcoding, or it silently goes wrong the first time anyone changes the framing.

Run:
```bash
python -m recog.seg_evaluate --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
```

**The latency table must show `within_50ms_budget: true` at 8 cartridges.** If it does not, the deployment configuration is wrong before Plan D starts — check `half` and `crop_size` before touching anything else.

- [ ] **Step 6: Commit**

```bash
git add recog/seg_evaluate.py docs/receipts/seg_eval.txt tests/test_bay_segmenter.py
git commit -m "feat(recog): bay segmenter evaluation

Per-class IoU, plus the three metrics that actually carry the argument:
boundary displacement in mm (the figure that chose a per-ROI segmenter
over a 28x28 mask head - IoU alone would make that unfalsifiable),
signed area error separating optimistic from conservative (only the
first can put a cell on a PCB), and a latency table against the 50 ms
budget.

IoU is NaN, not 0.0, for a class absent from both prediction and truth:
a class that was never present was never tested."
```

---

## Acceptance

- [ ] `pytest -q` passes with no regressions against Plan B's baseline.
- [ ] `SEG_CHANNELS` is exactly the documented order — Plan D depends on it.
- [ ] A trained checkpoint exists at `recog/checkpoints/seg/best.pt` **and** `last.pt`.
- [ ] Validation mean IoU over `bay`, `electronics`, `obstruction` is reported with per-class instance counts.
- [ ] Boundary displacement is reported **in millimetres**, and is below the 2.9 × 6.4 mm figure that rejected the Mask R-CNN head — otherwise the architecture choice in spec §3.1 is not supported by evidence and should be revisited rather than defended.
- [ ] `latency_table` shows `within_50ms_budget: true` at 8 cartridges.
- [ ] `configs/recognition.yaml`, `recog/training.py` and the detector's checkpoints are untouched.
