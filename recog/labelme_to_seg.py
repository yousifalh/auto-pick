"""
recog.labelme_to_seg - LabelMe polygon export -> this project's COCO-RLE
segmentation sidecar (the format `recog.seg_dataset.BaySegDataset` reads
and `recog.synth3d.annotate.masks_from_index` writes for synthetic data).

Why LabelMe: pip-installable (`pip install labelme`), fully offline (no
server, no account, no upload of real-hardware photographs anywhere), and
its one-JSON-per-image output carries a per-shape `group_id` that maps
directly onto this project's `unit_id` concept - the field
`recog.synth3d.annotate.masks_from_index` stamps on every synthetic
annotation so `recog.seg_dataset.BaySegDataset` can group one physical
cartridge's shell, floor, module and obstructions into a single crop.
See docs/ANNOTATION_PROTOCOL.md for the annotator-facing half of this:
which five labels to draw, and how to use "Edit > Group ID" to link a
unit's shapes together.

Reuses `recog.synth3d.annotate.rle_encode` / `write_coco_json` rather
than re-implementing COCO-RLE. A previous defect in a hand-rolled
encoder emitted a double leading zero and would have corrupted every
mask whose first pixel is set - reuse keeps this converter immune to a
class of bug it would otherwise be free to reintroduce.

**Independently-drawn polygons that share a boundary WILL share pixels
after rasterisation, even when traced correctly** - a standard property
of filling two adjacent simple polygons with the same edge coordinates
(``cv2.fillPoly`` fills boundary pixels on both sides of a shared edge),
not a sign of sloppy annotation. The synthetic pipeline never meets this
problem: a single Blender object-index render assigns each pixel
exactly one id by construction, so two labels can never physically
overlap there. Hand-drawn polygons have no such guarantee, so this
module reproduces the same guarantee here by applying
`recog.seg_dataset._PAINT_ORDER` - the identical later-wins precedence
`rasterise_crop` applies at training time - ONCE, globally, at
conversion time (:func:`resolve_paint_order`). This is also why an
annotator does not need pixel-perfect boundaries (see
docs/ANNOTATION_PROTOCOL.md): drawing the earlier-priority class a
little generously into a later one's territory is corrected
automatically; the reverse is not, so precision matters most for
whichever class is later in the order at a given boundary.

No torch import, so this runs in the same torch-free environment as
plan/arbitration.py and recog/check_annotations.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - hard dep, mirrors plan/arbitration.py
    raise ImportError("opencv-python is required") from exc

from recog.seg_dataset import _PAINT_ORDER
from recog.synth3d.annotate import rle_encode, write_coco_json
from recog.synth3d.config import seg_class_ids

# One source of truth for the five-class vocabulary and their ids -
# recog/synth3d/config.py, the same place the synthetic pipeline reads it.
SEG_CLASS_IDS: Dict[str, int] = seg_class_ids()

# The SAME later-wins precedence recog.seg_dataset.rasterise_crop applies
# at training time, read from its own _PAINT_ORDER rather than restated
# here - a second, independently-maintained copy of this list is exactly
# the kind of drift that would make a converted annotation resolve
# differently at conversion time than the model will ever see it resolved
# at training time.
_CLASS_ORDER: List[str] = [cls for cls, _channel in _PAINT_ORDER]
_CLASS_PRIORITY: Dict[str, int] = {cls: i + 1 for i, cls in enumerate(_CLASS_ORDER)}


class LabelmeConversionError(ValueError):
    """A LabelMe export is malformed or uses a label outside the fixed
    five-class vocabulary. Raised rather than silently skipped: a
    silently-dropped shape is a hole in the ground truth that nothing
    downstream can detect."""


def polygon_to_mask(points: Sequence[Sequence[float]], height: int,
                    width: int) -> np.ndarray:
    """Rasterise one LabelMe polygon into a dense ``(height, width)``
    uint8 mask.

    ``cv2.fillPoly`` needs integer vertices; LabelMe stores float pixel
    coordinates, so vertices are rounded rather than truncated - int()
    truncation would bias every polygon half a pixel toward the image
    origin, systematically shrinking every shape on its bottom/right
    edge.
    """
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image  # project hard dependency (pyproject.toml)

    with Image.open(path) as im:
        return im.width, im.height


def resolve_paint_order(
    shapes: Sequence[dict], height: int, width: int,
) -> Tuple[List[dict], List[dict]]:
    """Cut every shape down to the pixels it actually owns once the
    project's fixed later-wins precedence is applied, so two boundaries
    that were traced loosely (or deliberately generously - see the
    module docstring) can never both claim the same output pixel.

    ``shapes`` is a sequence of ``{"shape_index", "label", "mask", ...}``
    dicts in file order; extra keys ride through unchanged on kept
    shapes. Classes resolve in paint order (cartridge, placement_area,
    electronics_module, obstruction, battery - later wins). Within one
    class, paint order has nothing to say, so ties resolve by FILE
    ORDER: the shape listed first in the LabelMe export keeps a
    contested pixel.

    Returns ``(kept, dropped)``, mirroring
    `recog.synth3d.annotate.masks_from_index`'s ``(anns, dropped)``
    convention. A shape dropped here lost every one of its raw pixels to
    a higher-priority class or an earlier same-class shape - it is
    reported, not silently discarded, but it is not an error: drawing a
    class generously enough that a neighbour fully reclaims it is the
    intended, documented behaviour, not a mistake.
    """
    owner_class = np.zeros((height, width), dtype=np.int16)  # 0 = unclaimed
    for cls in _CLASS_ORDER:
        priority = _CLASS_PRIORITY[cls]
        for s in shapes:
            if s["label"] == cls:
                owner_class[s["mask"] > 0] = priority

    kept: List[dict] = []
    dropped: List[dict] = []
    for cls in _CLASS_ORDER:
        available = owner_class == _CLASS_PRIORITY[cls]
        for s in shapes:
            if s["label"] != cls:
                continue
            final = (s["mask"] > 0) & available
            if not final.any():
                dropped.append({"shape_index": s["shape_index"], "class": cls,
                                "reason": "fully_overlapped_by_paint_order"})
                continue
            available = available & ~final    # first-in-file-order wins ties
            out = dict(s)
            out["mask"] = final.astype(np.uint8)
            kept.append(out)

    kept.sort(key=lambda s: s["shape_index"])  # stable, file-order output
    return kept, dropped


def _unit_id(image_stem: str, group_id: Optional[int], shape_index: int) -> str:
    """One id per physical unit, matching the grouping key
    `recog.seg_dataset.BaySegDataset` reads.

    Shapes sharing a LabelMe ``group_id`` within one image share a
    unit - a cartridge's shell, its placement_area, its
    electronics_module and any obstructions/cells seated in it (see
    docs/ANNOTATION_PROTOCOL.md's "linking a unit" section). A shape
    with no group_id (a loose cell or loose module - design spec ruling
    3) gets a unit id of its own, keyed by its position in the file, so
    it is never silently merged with an unrelated ungrouped shape in
    the same image the way sharing a bare ``None`` key would.
    """
    if group_id is None:
        return f"{image_stem}#u{shape_index}"
    return f"{image_stem}#g{group_id}"


def convert_labelme_file(json_path: Path,
                         images_dir: Optional[Path] = None
                         ) -> Tuple[dict, List[dict], List[dict]]:
    """Convert one LabelMe ``.json`` into ``(image_record, annotations,
    dropped)``.

    ``image_record`` is ``{"file_name", "width", "height"}`` - no ``id``
    yet, ``annotations`` entries carry no ``id`` / ``image_id`` yet
    either: both are assigned once :func:`convert_labelme_dir` knows the
    full set being written together. ``dropped`` lists shapes
    :func:`resolve_paint_order` removed because a higher-priority class
    (or an earlier same-class shape) claimed every one of their pixels -
    see that function's docstring; it is not a sign anything is wrong.

    When ``images_dir`` is given and the referenced image file exists
    there, its *actual* on-disk dimensions are used (and checked against
    LabelMe's recorded ``imageWidth``/``imageHeight`` if present) rather
    than trusting the JSON blindly - a photo re-saved or re-exported at
    a different resolution after annotation would otherwise silently
    mis-scale every polygon.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    image_path_field = str(doc.get("imagePath") or "")
    file_name = Path(image_path_field).name or (json_path.stem + ".jpg")

    width = doc.get("imageWidth")
    height = doc.get("imageHeight")

    img_path = (images_dir / file_name) if images_dir is not None else None
    if img_path is not None and img_path.is_file():
        aw, ah = _image_size(img_path)
        if width and height and (int(width), int(height)) != (aw, ah):
            raise LabelmeConversionError(
                f"{json_path}: recorded image size {int(width)}x{int(height)} "
                f"does not match the actual file {img_path} ({aw}x{ah}). "
                f"Re-export from LabelMe against the current image."
            )
        width, height = aw, ah
    elif images_dir is not None:
        raise LabelmeConversionError(
            f"{json_path}: image file {file_name!r} not found under "
            f"{images_dir}."
        )

    if not width or not height:
        raise LabelmeConversionError(
            f"{json_path}: no image dimensions (imageWidth/imageHeight "
            f"missing from the JSON and no --images-dir given to read the "
            f"file directly)."
        )
    width, height = int(width), int(height)

    image_stem = json_path.stem
    raw_shapes: List[dict] = []
    for i, shape in enumerate(doc.get("shapes", [])):
        raw_label = shape.get("label", "")
        label = str(raw_label).strip().lower()
        if label not in SEG_CLASS_IDS:
            raise LabelmeConversionError(
                f"{json_path}: shape {i} has label {raw_label!r}, not one "
                f"of {sorted(SEG_CLASS_IDS)}. Fix the label in LabelMe and "
                f"re-export - see docs/ANNOTATION_PROTOCOL.md."
            )
        shape_type = shape.get("shape_type", "polygon")
        if shape_type != "polygon":
            raise LabelmeConversionError(
                f"{json_path}: shape {i} ({label}) is a {shape_type!r}, "
                f"not a polygon. Only polygons are supported - see "
                f"docs/ANNOTATION_PROTOCOL.md."
            )
        points = shape.get("points") or []
        if len(points) < 3:
            raise LabelmeConversionError(
                f"{json_path}: shape {i} ({label}) has only {len(points)} "
                f"vertices; a polygon needs at least 3."
            )

        mask = polygon_to_mask(points, height, width)
        if not mask.any():
            raise LabelmeConversionError(
                f"{json_path}: shape {i} ({label}) rasterises to zero "
                f"pixels - a degenerate or self-intersecting polygon, or "
                f"vertices entirely outside the image bounds."
            )
        raw_shapes.append({"shape_index": i, "label": label, "mask": mask,
                           "group_id": shape.get("group_id")})

    kept, dropped = resolve_paint_order(raw_shapes, height, width)

    anns: List[dict] = []
    for s in kept:
        mask = s["mask"]
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        anns.append({
            "class": s["label"],
            "category_id": SEG_CLASS_IDS[s["label"]],
            "segmentation": rle_encode(mask),
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            "area": int(mask.sum()),
            "iscrowd": 0,
            "unit_id": _unit_id(image_stem, s["group_id"], s["shape_index"]),
        })

    image_record = {"file_name": file_name, "width": width, "height": height}
    return image_record, anns, dropped


def convert_labelme_dir(labelme_dir: Path, out_path: Path,
                        images_dir: Optional[Path] = None
                        ) -> Tuple[int, int, List[dict]]:
    """Convert every ``*.json`` under ``labelme_dir`` into one COCO-RLE
    sidecar at ``out_path``. Returns ``(n_images, n_annotations, dropped)``,
    ``dropped`` being every shape :func:`resolve_paint_order` removed
    across the whole directory, each tagged with its source file.
    """
    json_paths = sorted(labelme_dir.glob("*.json"))
    if not json_paths:
        raise LabelmeConversionError(f"no .json files found under {labelme_dir}")

    images: List[dict] = []
    annotations: List[dict] = []
    all_dropped: List[dict] = []
    next_ann_id = 1
    for image_id, jp in enumerate(json_paths, start=1):
        image_record, anns, dropped = convert_labelme_file(jp, images_dir)
        image_record["id"] = image_id
        images.append(image_record)
        for a in anns:
            a["image_id"] = image_id
            a["id"] = next_ann_id
            next_ann_id += 1
            annotations.append(a)
        for d in dropped:
            all_dropped.append(dict(d, source=jp.name,
                                    file_name=image_record["file_name"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_coco_json(str(out_path), images=images, annotations=annotations,
                    seg_class_ids=SEG_CLASS_IDS)
    return len(images), len(annotations), all_dropped


# --------------------------------------------------------------- CLI ----

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.labelme_to_seg",
        description=(
            "Convert a directory of LabelMe polygon *.json exports into "
            "this project's COCO-RLE segmentation sidecar "
            "(recog.seg_dataset.BaySegDataset's input format)."
        ),
    )
    ap.add_argument("labelme_dir", help="directory of LabelMe *.json files")
    ap.add_argument("out", help="output COCO-RLE json path, e.g. "
                                "recog/realtest_seg/annotations/instances_seg.json")
    ap.add_argument("--images-dir", default=None,
                    help="directory holding the source photographs. Strongly "
                         "recommended: without it, image dimensions come from "
                         "LabelMe's own (sometimes stale) record instead of "
                         "the actual file.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    labelme_dir = Path(args.labelme_dir)
    if not labelme_dir.is_dir():
        raise SystemExit(f"error: not a directory: {labelme_dir}")

    images_dir = Path(args.images_dir) if args.images_dir else None
    if images_dir is not None and not images_dir.is_dir():
        raise SystemExit(f"error: not a directory: {images_dir}")

    try:
        n_images, n_anns, dropped = convert_labelme_dir(
            labelme_dir, Path(args.out), images_dir)
    except LabelmeConversionError as exc:
        raise SystemExit(f"error: {exc}")

    if dropped:
        print(f"note: {len(dropped)} shape(s) were fully claimed by a "
             f"higher-priority class and dropped (this is expected when a "
             f"class was drawn generously - see docs/ANNOTATION_PROTOCOL.md):")
        for d in dropped:
            print(f"  {d['file_name']}  shape {d['shape_index']} "
                 f"({d['class']}): {d['reason']}")

    print(f"wrote {n_anns} annotation(s) across {n_images} image(s) to "
         f"{args.out}")
    return 0


__all__ = [
    "LabelmeConversionError",
    "SEG_CLASS_IDS",
    "build_arg_parser",
    "convert_labelme_dir",
    "convert_labelme_file",
    "main",
    "polygon_to_mask",
    "resolve_paint_order",
]


if __name__ == "__main__":
    sys.exit(main())
