"""
recog.seg_dataset - per-ROI crops for the bay segmenter.

Reads the COCO-RLE sidecar recog/generate3d.py writes and turns each
physical UNIT into one training crop: the image inside a jittered union
box, plus a dense label map over the same window.

Grouping is by unit, not by `cartridge` annotation. An OPEN cartridge has
no surviving `cartridge` mask at all - the electronics module and bay
proxy cover the shell's entire top face, so the index pass reports no
shell pixels (see recog/synth3d/annotate.py's masks_from_index and
scene.py's unit_id comments). Cropping to `cartridge` annotations was
measured to yield 43 crops containing zero bay, electronics or
obstruction pixels - the segmenter would have trained on nothing. Every
annotation carries a `unit_id` linking one physical unit's parts
(cartridge, module, bay, obstructions, seated cells), and grouping by it
gives a crop that actually contains a bay.

Crops come from JITTERED boxes on purpose. At inference the crop comes
from the detector, whose boxes are not ground truth; training on perfect
boxes would bake in a distribution shift the model first meets in
deployment.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

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


class BaySegDataset:
    """One crop per physical UNIT in the COCO sidecar.

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

        # One sample per UNIT, not per `cartridge` annotation.
        #
        # This is not a stylistic choice. An OPEN cartridge has no
        # surviving `cartridge` mask at all: the electronics module and
        # bay proxy cover the shell's entire top face, so the index pass
        # reports no shell pixels. Cropping to `cartridge` annotations
        # therefore yields only SEALED units - measured before the fix as
        # 43 crops containing zero bay, electronics or obstruction
        # pixels. The segmenter would have trained on nothing.
        #
        # Every annotation now carries a `unit_id` linking the parts of
        # one physical unit - its cartridge, module, bay, obstructions
        # and any cells seated in it. A loose cell or loose module gets
        # its own. Grouping by it and cropping the union of the group's
        # boxes gives a crop that actually contains a bay: measured 6 of
        # 6 after the fix.
        #
        # A unit is kept if it has ANY cartridge-related annotation, not
        # only `placement_area`. A SEALED unit's group is, BY
        # CONSTRUCTION, exactly `{"cartridge"}` and nothing else -
        # annotate.py's extend_group_boxes docstring: "Sealed (assembled)
        # units never have a module/proxy/obstruction pid at all
        # (scene.build guards those on the open_case variant)" - so it
        # never carries a `placement_area` member to pass a
        # placement_area-only filter. Requiring `placement_area` here
        # would silently drop every sealed unit and make SEG_CHANNELS
        # ["cartridge"] unreachable in the whole dataset: measured over a
        # 24-scene regeneration, 174 units split cleanly into 132 pure
        # `{"battery"}` (loose cells, no cartridge context - excluded),
        # 31 pure `{"cartridge"}` (sealed shells - must be kept for the
        # channel to ever appear), and 11 combinations containing
        # `placement_area`; no unit ever mixes `cartridge` with anything
        # else. Only the pure-`{"battery"}` loose scatter has nothing to
        # segment; everything else is kept.
        _CARTRIDGE_RELATED = {"cartridge", "placement_area",
                              "electronics_module", "obstruction"}
        self.samples: List[Tuple[dict, List[dict], Tuple[int, int, int, int]]] = []
        for img_id, anns in by_image.items():
            by_unit: Dict[object, List[dict]] = {}
            for a in anns:
                by_unit.setdefault(a.get("unit_id"), []).append(a)
            for uid, unit in by_unit.items():
                if not any(a["class"] in _CARTRIDGE_RELATED for a in unit):
                    continue          # loose battery scatter; nothing to segment
                xs = [a["bbox_xyxy"] for a in unit]
                box = (min(b[0] for b in xs), min(b[1] for b in xs),
                       max(b[2] for b in xs), max(b[3] for b in xs))
                self.samples.append((images[img_id], anns, box))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        img_meta, anns, unit_box = self.samples[idx]
        path = os.path.join(self.img_dir, img_meta["file_name"])
        image = np.asarray(Image.open(path).convert("RGB"))

        box = jitter_box(unit_box, self.rng, self.jitter_frac)
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


__all__ = [
    "SEG_CHANNELS",
    "jitter_box",
    "rasterise_crop",
    "BaySegDataset",
]
