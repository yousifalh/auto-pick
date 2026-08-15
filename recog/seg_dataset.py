"""
recog.seg_dataset - per-ROI crops for the bay segmenter.

Reads the COCO-RLE sidecar recog/generate3d.py writes and turns each
physical UNIT into one training crop: the image inside a jittered union
box, plus a dense label map over the same window.

Grouping is by unit, not by `cartridge` annotation. BEFORE the
tray-interior fix (commits a31ac28..043e92d), an OPEN cartridge had no
surviving `cartridge` mask at all - the electronics module and bay proxy
sat flush on the outer top face and covered it entirely, so the index
pass reported no shell pixels, and cropping to `cartridge` annotations
was measured to yield 43 crops containing zero bay, electronics or
obstruction pixels (the segmenter would have trained on nothing). That
motivated grouping by `unit_id` instead of by annotation class - every
annotation carries a `unit_id` linking one physical unit's parts
(cartridge, module, bay, obstructions, seated cells), and grouping by it
gives a crop that actually contains a bay regardless of which classes
that unit happens to carry.

AFTER the fix, an open unit's tray walls are real standing geometry
(not a flat decal), so they are visible and DO carry a `cartridge`
annotation - and that annotation now typically shares its `unit_id`,
and therefore its crop, with the SAME unit's own bay/electronics/
obstruction annotations, because both come from the one physical
object. This is no longer the disjoint case the paragraph above
describes for the pre-fix generator: as a snapshot (measured
2026-08-09 on the 502-scene / 841-crop dataset that shipped in
`cd86d1f`, not a claim that holds for whatever dataset exists when you
read this), 210 of 841 units/crops carry a `cartridge` annotation
together with at least one of `placement_area`/`electronics_module`/
`obstruction` on the SAME `unit_id` (176 of 502 scenes at the coarser
per-image level); 630 carry `cartridge` alone, 1 carries a bay-class
annotation with no `cartridge`, and none carry neither. Do not treat
that snapshot as current fact - `recog.seg_evaluate` computes the
same contingency (at the rasterised-pixel level, which is what the
model is actually scored against) fresh every run and prints it in
`docs/receipts/seg_eval.txt`; read the number there, not here, since a
docstring cannot recompute itself when the dataset changes.

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


def _rng_for_worker(seed: int, worker_id: Optional[int],
                    torch_seed: Optional[int] = None) -> np.random.Generator:
    """A jitter RNG stream for one DataLoader worker.

    ``worker_id`` is None in the main process (num_workers=0, today's
    default). Under num_workers > 0 (configs/segmentation.yaml invites this
    on Linux), each worker is a fork/spawn of the SAME dataset object,
    carrying the SAME ``self.rng`` state it had at construction time -
    without folding the worker id in, every worker would draw an
    identical jitter stream, silently degrading augmentation the moment
    num_workers > 0. Combining the dataset's own seed with the worker id
    keeps each worker's stream distinct and each stream reproducible.

    ``torch_seed`` is ``torch.initial_seed()``, read INSIDE the worker,
    and it is what makes the stream differ per EPOCH rather than only per
    worker. ``(seed, worker_id)`` is a constant for the life of the run,
    and the workers are re-created from scratch at the start of every
    epoch out of a parent dataset object whose ``_worker_id`` is
    permanently None - so every epoch re-entered the re-seed branch and
    rebuilt the SAME generator, and the crop-jitter multiset repeated
    byte for byte from epoch 2 onwards. Augmentation that looks random
    and is actually one fixed draw is worse than no augmentation,
    because nothing says so.

    torch already derives each worker's own seed from the loader's
    ``generator`` afresh each epoch - ``recog/seeding.py``'s
    ``seed_worker`` documents and uses exactly this property to spread
    ``random`` and ``numpy.random`` across workers, and
    ``recog/seg_training.py`` passes the ``generator`` that drives it.
    The mechanism was already there and this stream was the one thing not
    reading from it. Keeping ``seed`` and ``worker_id`` in the tuple as
    well costs nothing and keeps two datasets constructed with different
    ``seed`` values distinct inside one worker.
    """
    if worker_id is None:
        return np.random.default_rng(seed)
    if torch_seed is None:
        return np.random.default_rng((seed, worker_id))
    return np.random.default_rng((seed, worker_id, torch_seed))


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


def rasterise_crop(anns: Sequence[dict], box,
                   out_size: Optional[int] = None) -> np.ndarray:
    """Dense label map for the window ``box``.

    Resized to ``out_size`` x ``out_size`` if given; left at the crop's own
    native resolution if ``out_size`` is None. The evaluator needs the
    native map (a jittered union box is not square, so a single scalar
    mm_per_px cannot describe a resized crop) - it calls this function
    directly with ``out_size=None`` rather than keeping its own copy of
    this painting loop, so a future change here (e.g. to `iscrowd`
    handling) cannot silently drift out of sync with what the receipt
    scores against.
    """
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

    return label if out_size is None else _resize_nearest(label, out_size)


def extract_crop(image: np.ndarray, box,
                 out_size: Optional[int] = None) -> np.ndarray:
    """The window ``box``'s own pixels, zero-padded where it runs off
    ``image``, resized to ``out_size`` x ``out_size`` if given (native
    resolution otherwise).

    Shared by BaySegDataset.__getitem__ and the evaluator's native-crop
    extraction for the same reason ``rasterise_crop`` takes an
    ``out_size`` argument instead of the evaluator hand-copying this
    loop: one definition, one place to fix.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    pad_t, pad_l = max(0, -y0), max(0, -x0)
    crop = image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if pad_t or pad_l or crop.shape[0] != y1 - y0 or crop.shape[1] != x1 - x0:
        padded = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        padded[pad_t:pad_t + crop.shape[0],
               pad_l:pad_l + crop.shape[1]] = crop
        crop = padded
    if out_size is None:
        return crop
    # No-cv2 fallback resizes with nearest rather than linear (verbatim
    # from the original __getitem__ path) - dead in practice since
    # opencv is a hard dependency, left as-is rather than "fixed".
    return (_resize_nearest(crop, out_size) if not _HAVE_CV2 else
            cv2.resize(crop, (out_size, out_size),
                      interpolation=cv2.INTER_LINEAR))


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
        self._seed = int(seed)
        self._worker_id: Optional[int] = None
        self.rng = _rng_for_worker(self._seed, None)

        names = {c["id"]: c["name"] for c in doc["categories"]}
        images = {im["id"]: im for im in doc["images"]}

        # Every annotation must name its unit BEFORE any grouping happens.
        #
        # Grouping is `by_image[image_id] -> by_unit[unit_id]`, and a dict
        # is perfectly happy to bucket on None: an annotation set whose
        # `unit_id` is missing or None therefore collapses every unit in
        # one image into a SINGLE crop spanning all of them - measured
        # (audit I §7b) on a three-annotation, two-unit image as
        # `crops=1, union_box=(10,20,260,80)` where the correct answer is
        # two crops at (10,20,140,80) and (180,20,260,80). No exception,
        # no warning: training simply proceeds on a crop whose scale and
        # content are wrong, and nothing downstream can tell.
        #
        # Neither live producer can reach this - `masks_from_index` always
        # writes the key and `labelme_to_seg._unit_id` never returns None,
        # and `unit_id` is present on all 43 449 annotations of all 10
        # datasets on disk. The reachable path is a hand-edited sidecar or
        # a third-party converter, which is exactly the path real-photograph
        # ground truth would arrive on. So: refuse it, loudly, rather than
        # silently produce one wrong crop per image.
        no_unit = [a.get("id") for a in doc["annotations"]
                   if a.get("unit_id") is None or a.get("unit_id") == ""]
        if no_unit:
            raise ValueError(
                f"{coco_path}: {len(no_unit)} of {len(doc['annotations'])} "
                f"annotations carry no `unit_id` (ids {no_unit[:10]}"
                f"{' ...' if len(no_unit) > 10 else ''}). Crops are grouped "
                f"by unit within each image, so annotations without one "
                f"would all share a single None bucket and collapse every "
                f"unit in their image into ONE crop spanning all of them - "
                f"silently, with the wrong scale and the wrong content. "
                f"Every annotation needs a `unit_id` naming the physical "
                f"unit it belongs to (see recog/labelme_to_seg.py and "
                f"docs/ANNOTATION_PROTOCOL.md Sec5); it need only be unique "
                f"within its own image.")

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
        # This is not a stylistic choice. BEFORE the tray-interior fix
        # (a31ac28..043e92d), an OPEN cartridge had no surviving
        # `cartridge` mask at all: the electronics module and bay proxy
        # covered the shell's entire top face, so the index pass reported
        # no shell pixels. Cropping to `cartridge` annotations therefore
        # yielded only SEALED units - measured before the fix as 43 crops
        # containing zero bay, electronics or obstruction pixels. The
        # segmenter would have trained on nothing. AFTER the fix an open
        # unit's tray walls are real geometry and DO carry a `cartridge`
        # annotation (see the module docstring), so this specific failure
        # mode no longer applies as stated - but grouping by `unit_id`
        # remains correct and is now load-bearing for a different reason:
        # it is what puts a unit's `cartridge` and bay-class annotations
        # in the SAME crop when both exist, rather than requiring a second
        # pass to merge them.
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
        # ["cartridge"] unreachable in the whole dataset.
        #
        # PRE-fix (measured over a 24-scene regeneration), 174 units split
        # cleanly into 132 pure `{"battery"}` (loose cells, no cartridge
        # context - excluded), 31 pure `{"cartridge"}` (sealed shells -
        # must be kept for the channel to ever appear), and 11
        # combinations containing `placement_area`; no unit mixed
        # `cartridge` with anything else, because an open unit's
        # `cartridge` mask did not survive at all back then. POST-fix
        # (measured 2026-08-09 on the current 502-scene dataset - a
        # snapshot, not a claim that holds for whatever dataset exists
        # when you read this) that is no longer true: of 841 kept units,
        # 630 are pure `{"cartridge"}` (sealed) but 210 mix `cartridge`
        # with `placement_area`/`electronics_module`/`obstruction` on the
        # SAME unit_id (open units, whose tray walls are now real
        # geometry - see the module docstring), and 1 carries a bay-class
        # annotation with no `cartridge` at all. Only the pure-
        # `{"battery"}` loose scatter has nothing to segment; everything
        # else is kept regardless of which classes it mixes.
        _CARTRIDGE_RELATED = {"cartridge", "placement_area",
                              "electronics_module", "obstruction"}
        self.samples: List[Tuple[dict, List[dict], Tuple[int, int, int, int]]] = []
        self.sample_assets: List[Optional[str]] = []
        for img_id, anns in by_image.items():
            # IMAGE FIRST, unit second, and that order is load-bearing.
            # `unit_id` is SCENE-LOCAL, not globally unique: scene.build
            # derives it from a per-scene counter, so "item0" names 252
            # different physical units across recog/dataset3d_seg's 502
            # images (69 distinct ids in total - audit I Sec7a). Bucketing
            # by image_id first is the only reason that is harmless.
            # Flattening this into a single `by_unit` over the whole
            # sidecar would merge every one of those 252 units into one
            # crop, and nothing would raise.
            # `tests/test_seg_dataset.py::
            # test_the_same_unit_id_in_two_images_stays_two_crops` pins it.
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
                self.sample_assets.append(unit[0].get("asset"))

    def __len__(self) -> int:
        return len(self.samples)

    def _reseed_for_worker(self) -> None:
        """Give this process its own jitter stream, once per fork.

        The guard fires exactly once per worker process: a worker is
        built from the parent object, whose ``_worker_id`` is None, so
        the first ``__getitem__`` in it re-seeds and every later one
        leaves the stream running. With ``num_workers=0`` (today's
        default) ``wid`` is None, matches the constructor's
        ``_worker_id``, and nothing here runs at all - that path is
        `default_rng(seed)` exactly as it always was.

        ``torch.initial_seed()`` is read here and not in ``__init__``
        because it is only per-worker and per-epoch when read from inside
        the worker; in the parent it is one constant. With
        ``persistent_workers=True`` a worker is NOT rebuilt between
        epochs, so this does not fire again and the stream simply keeps
        running - which is the same guarantee by a different route.
        """
        import torch

        worker = torch.utils.data.get_worker_info()
        if worker is None:
            wid = torch_seed = None
        else:
            wid, torch_seed = worker.id, int(torch.initial_seed())
        if wid != self._worker_id:
            self.rng = _rng_for_worker(self._seed, wid, torch_seed)
            self._worker_id = wid

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        self._reseed_for_worker()

        img_meta, anns, unit_box = self.samples[idx]
        path = os.path.join(self.img_dir, img_meta["file_name"])
        image = np.asarray(Image.open(path).convert("RGB"))

        box = jitter_box(unit_box, self.rng, self.jitter_frac)
        label = rasterise_crop(anns, box, self.out_size)
        crop = extract_crop(image, box, self.out_size)

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
    "extract_crop",
    "BaySegDataset",
]
