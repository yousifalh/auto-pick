"""Dataset loaders for the battery / cartridge recogniser.

Two annotation dialects are supported.

**Pascal VOC** (:func:`parse_voc_xml`, :class:`BatteryCartridgeDataset`)
— what the synthetic generators emit:

* ``<filename>``, ``<size>/<width>``, ``<size>/<height>``
* one or more ``<object>`` blocks with ``<name>`` and ``<bndbox>``

**COCO** (:func:`parse_coco_json`, :class:`RealPhotoDataset`) — what
CVAT exports, used for the held-out set of *real* photographs in
``recog/realtest``. This is the set that answers the only question
that matters about the synthetic pipeline: does a detector trained
purely on renders work on photographs?

Both readers are deliberately defensive in the same way. Objects
whose class label is not in :data:`CLASS_MAP` are silently skipped
(so unknown / experimental labels don't explode the loader), and so
are bounding boxes with a non-positive width or height — those are
degenerate annotations that bite detector training in subtle ways if
left in.

The module imports cleanly even when torch is not installed: the
datasets fall back to returning ``(numpy_image, {"boxes": [...],
"labels": [...]})`` tuples, which is what the unit tests exercise in
the CI container.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    # No bare `import torch` here: it bound a module global that nothing in
    # this file ever read. Every `torch.` reference lives inside
    # `_to_training_sample`, behind its own local `import torch` (:314), so
    # the global was dead weight ruff reported as F401. The guard is
    # unweakened - `from torch.utils.data import ...` imports the torch
    # package first, so a torch-free environment still lands in `except`.
    from torch.utils.data import Dataset  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - executed only on torch-free CI
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        """Minimal shim so the module imports without torch."""

        def __len__(self) -> int:  # pragma: no cover - shim
            return 0

        def __getitem__(self, idx: int):  # pragma: no cover - shim
            raise IndexError


try:  # pragma: no cover - import guard
    from PIL import Image  # type: ignore

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - handled by cv2 fallback
    _PIL_AVAILABLE = False


# Class id convention: 0 is reserved for the detector background head.
CLASS_MAP: Dict[str, int] = {
    "background": 0,
    "battery": 1,
    "cartridge": 2,
}
INV_CLASS_MAP: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}


# ---------------------------------------------------- parsing helpers ----

@dataclass(frozen=True)
class VocAnnotation:
    """One parsed VOC XML file."""

    filename: str
    width: int
    height: int
    boxes: List[Tuple[float, float, float, float]]
    labels: List[int]


def parse_voc_xml(
    xml_path: str | Path,
    class_map: Dict[str, int] = CLASS_MAP,
) -> VocAnnotation:
    """Parse a single Pascal-VOC XML file.

    Objects with an unknown class name are skipped, as are objects
    with a bounding box that has zero or negative width or height.
    The ``background`` pseudo-class in ``class_map`` is also skipped.
    """
    root = ET.parse(str(xml_path)).getroot()

    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    width = int(size.findtext("width") or 0) if size is not None else 0
    height = int(size.findtext("height") or 0) if size is not None else 0

    boxes: List[Tuple[float, float, float, float]] = []
    labels: List[int] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_map or class_map[name] == 0:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin") or 0)
        ymin = float(bnd.findtext("ymin") or 0)
        xmax = float(bnd.findtext("xmax") or 0)
        ymax = float(bnd.findtext("ymax") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(class_map[name])

    return VocAnnotation(filename, width, height, boxes, labels)


@dataclass(frozen=True)
class CocoImageAnnotation:
    """One image's worth of a parsed COCO ``instances_*.json``.

    ``boxes`` are ``(x0, y0, x1, y1)`` floats with *exclusive* max
    edges, matching :class:`common.types.BBox` and the VOC reader —
    not COCO's on-disk ``[x, y, w, h]``.
    """

    image_id: int
    file_name: str
    width: int
    height: int
    boxes: List[Tuple[float, float, float, float]]
    labels: List[int]


def parse_coco_json(
    path: str | Path,
    class_map: Dict[str, int] = CLASS_MAP,
) -> List[CocoImageAnnotation]:
    """Parse a COCO detection JSON into one record per image.

    Category names are matched against ``class_map``
    **case-insensitively**, so CVAT's ``"Battery"`` / ``"Cartridge"``
    line up with the project's lower-case :data:`CLASS_MAP` keys.

    Categories whose name is unknown to ``class_map`` (and the
    ``background`` pseudo-class) are skipped, along with every
    annotation referring to them. Boxes with a non-positive width or
    height are dropped, mirroring :func:`parse_voc_xml`.

    Where a category name *is* known, its COCO ``id`` must equal
    ``class_map[name]``. A re-export that renumbers the categories
    would otherwise silently mislabel the whole set, so the mismatch
    raises :class:`ValueError` instead.

    Images are returned in the order they appear in the file, each
    with its annotations in file order. Images with no annotations
    are still returned, with empty ``boxes`` and ``labels``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    lower_map = {name.lower(): cid for name, cid in class_map.items()}

    # category id → class id, for the categories we recognise.
    cat_to_class: Dict[int, int] = {}
    for cat in data.get("categories", []):
        name = str(cat.get("name", "")).strip().lower()
        if name not in lower_map or lower_map[name] == 0:
            continue
        cat_id = int(cat["id"])
        expected = lower_map[name]
        if cat_id != expected:
            raise ValueError(
                f"COCO category '{cat.get('name')}' has id {cat_id} but "
                f"CLASS_MAP says {expected}. Re-export the annotations "
                f"with matching ids, or pass an explicit class_map."
            )
        cat_to_class[cat_id] = expected

    boxes_by_image: Dict[int, List[Tuple[float, float, float, float]]] = {}
    labels_by_image: Dict[int, List[int]] = {}
    for ann in data.get("annotations", []):
        cls = cat_to_class.get(int(ann.get("category_id", -1)))
        if cls is None:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0.0 or h <= 0.0:
            continue
        img_id = int(ann["image_id"])
        boxes_by_image.setdefault(img_id, []).append((x, y, x + w, y + h))
        labels_by_image.setdefault(img_id, []).append(cls)

    records: List[CocoImageAnnotation] = []
    for img in data.get("images", []):
        img_id = int(img["id"])
        records.append(CocoImageAnnotation(
            image_id=img_id,
            file_name=str(img.get("file_name", "")),
            width=int(img.get("width", 0)),
            height=int(img.get("height", 0)),
            boxes=boxes_by_image.get(img_id, []),
            labels=labels_by_image.get(img_id, []),
        ))
    return records


# ------------------------------------------------------------- dataset ---

class BatteryCartridgeDataset(Dataset):
    """Directory-based dataset matching ``<stem>.png`` to ``<stem>.xml``.

    Parameters
    ----------
    img_dir, ann_dir:
        Directories containing images and VOC annotations respectively.
        Files are matched by stem, so ``images/foo.png`` pairs with
        ``annotations/foo.xml``.
    transforms:
        Optional Albumentations-style callable. Must accept
        ``image=..., bboxes=..., class_labels=...`` and return a dict
        with those same keys.
    class_map:
        Mapping of class name to integer id. Defaults to the module
        level :data:`CLASS_MAP`.

    Notes
    -----
    An image with no corresponding XML is returned with empty boxes
    and labels — this mirrors what production looks like when a
    dispatcher drops in a new photo before CVAT has been edited.
    """

    _IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        transforms: Optional[Callable] = None,
        class_map: Dict[str, int] = CLASS_MAP,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.class_map = class_map
        self.images: List[str] = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        )

    # ---- Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        fname = self.images[idx]
        img_path = self.img_dir / fname
        ann_path = self.ann_dir / (Path(fname).stem + ".xml")

        image = self._load_image(img_path)
        if ann_path.exists():
            ann = parse_voc_xml(ann_path, self.class_map)
        else:
            ann = VocAnnotation(
                filename=fname,
                width=image.shape[1],
                height=image.shape[0],
                boxes=[],
                labels=[],
            )

        bboxes = [list(b) for b in ann.boxes]
        labels = list(ann.labels)

        if self.transforms is not None:
            out = self.transforms(
                image=image, bboxes=bboxes, class_labels=labels,
            )
            image = out["image"]
            bboxes = out["bboxes"]
            labels = out["class_labels"]

        return self._to_training_sample(image, bboxes, labels, idx)

    # ---- Internals --------------------------------------------------------

    def _load_image(self, path: Path) -> np.ndarray:
        if _PIL_AVAILABLE:
            with Image.open(path) as im:
                return np.array(im.convert("RGB"))
        # Fallback path: OpenCV. Tested in environments where PIL is
        # missing but cv2 is available (some headless training images).
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2 could not read {path}")
        return bgr[:, :, ::-1].copy()  # BGR → RGB

    def _to_training_sample(self, image, bboxes, labels, idx):
        """Wrap an image + boxes pair in the shape the training loop wants."""
        if _TORCH_AVAILABLE:
            import torch  # type: ignore

            arr = image.astype(np.float32) / 255.0
            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
            img_t = torch.as_tensor(arr, dtype=torch.float32)

            if bboxes:
                boxes_t = torch.as_tensor(bboxes, dtype=torch.float32)
            else:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            if labels:
                labels_t = torch.as_tensor(labels, dtype=torch.int64)
            else:
                labels_t = torch.zeros((0,), dtype=torch.int64)

            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([idx], dtype=torch.int64),
            }
            return img_t, target

        return image, {"boxes": bboxes, "labels": labels, "image_id": [idx]}


class RealPhotoDataset(Dataset):
    """Held-out set of real photographs, annotated as COCO by CVAT.

    Unlike :class:`BatteryCartridgeDataset` this is driven by a single
    annotation *file* rather than a directory of per-image XML, but
    the ``__getitem__`` contract is identical: a float32 CHW image in
    ``[0, 1]`` plus a target dict with ``boxes``, ``labels`` and
    ``image_id`` (and the same numpy fallback when torch is absent).

    Parameters
    ----------
    img_dir:
        Directory containing the photographs. Records are matched by
        the annotation's ``file_name``.
    ann_path:
        Path to the COCO ``instances_*.json``.
    transforms:
        Optional Albumentations-style callable, as above.
    class_map:
        Mapping of class name to integer id. Defaults to
        :data:`CLASS_MAP`.

    Notes
    -----
    ``image_id`` carries the *COCO* image id, not the index into the
    dataset, so predictions can be joined back to the annotation file.
    Records whose image file is missing from ``img_dir`` are dropped
    at construction time — a partial export should evaluate on what it
    has rather than crash halfway through a run.
    """

    def __init__(
        self,
        img_dir: str,
        ann_path: str,
        transforms: Optional[Callable] = None,
        class_map: Dict[str, int] = CLASS_MAP,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.ann_path = Path(ann_path)
        self.transforms = transforms
        self.class_map = class_map
        self.records: List[CocoImageAnnotation] = [
            r for r in parse_coco_json(self.ann_path, class_map)
            if (self.img_dir / r.file_name).exists()
        ]

    # ---- Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = self._load_image(self.img_dir / rec.file_name)

        bboxes = [list(b) for b in rec.boxes]
        labels = list(rec.labels)

        if self.transforms is not None:
            out = self.transforms(
                image=image, bboxes=bboxes, class_labels=labels,
            )
            image = out["image"]
            bboxes = out["bboxes"]
            labels = out["class_labels"]

        return self._to_training_sample(image, bboxes, labels, rec.image_id)

    # ---- Internals --------------------------------------------------------

    # The loading / tensor-wrapping behaviour is identical to the VOC
    # dataset's; borrow it rather than maintaining a second copy.
    _load_image = BatteryCartridgeDataset._load_image
    _to_training_sample = BatteryCartridgeDataset._to_training_sample


def collate_fn(batch):
    """Object-detection-friendly collate: tuple-of-images, tuple-of-targets."""
    return tuple(zip(*batch))


__all__ = [
    "BatteryCartridgeDataset",
    "CLASS_MAP",
    "INV_CLASS_MAP",
    "CocoImageAnnotation",
    "RealPhotoDataset",
    "VocAnnotation",
    "collate_fn",
    "parse_coco_json",
    "parse_voc_xml",
]
