"""
Draw the generated boxes (or, with --masks, segmentation masks) onto the
renders so you can actually look at them.

    python -m recog.verify3d --data recog/dev3d --n 12
    python -m recog.verify3d --data recog/dev3d --n 16 --masks
    python -m recog.verify3d --sweep sweeps/ --out sweeps/lighting_sheet.png

Inspecting the contact sheet is not optional. A silently-wrong mask pass
produces boxes (or masks) that look plausible in JSON and are obviously wrong
on screen - and a box around a mask proves nothing about the mask's shape,
which is the whole reason --masks exists: it alpha-blends the actual RLE
region instead of just its bounding box.

System Python only: Blender's bundled interpreter has no Pillow.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from recog.dataset import CLASS_MAP, parse_voc_xml
from recog.synth3d.annotate import rle_decode

COLOURS = {"battery": (60, 220, 90), "cartridge": (255, 90, 60)}
INV = {v: k for k, v in CLASS_MAP.items()}

# Per-class overlay colours for --masks. Deliberately distinct from COLOURS
# above (the VOC box palette): the two modes are never shown together, but
# keeping them visually separate avoids any temptation to conflate a box
# colour with a mask colour while reading code.
SEG_COLOURS = {
    "battery": (255, 96, 96),
    "cartridge": (96, 160, 255),
    "electronics_module": (96, 255, 128),
    "placement_area": (255, 208, 64),
    "obstruction": (255, 96, 255),
}
MASK_ALPHA = 0.45  # fully opaque hides exactly the boundary errors this is for


def draw_one(img_path: Path, xml_path: Path, thick: int = 3) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    if not xml_path.exists():
        return im
    ann = parse_voc_xml(xml_path, CLASS_MAP)
    d = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), label in zip(ann.boxes, ann.labels):
        name = INV.get(label, "?")
        d.rectangle([x0, y0, x1, y1],
                    outline=COLOURS.get(name, (255, 255, 0)), width=thick)
        d.text((x0 + 4, max(0, y0 - 14)), name, fill=COLOURS.get(name))
    return im


def load_seg_index(root: Path):
    """Load instances_seg.json and index it for per-image lookup.

    Returns (image_id_by_filename, annotations_by_image_id, category_name_by_id).
    """
    path = root / "instances_seg.json"
    if not path.exists():
        raise SystemExit(f"no {path} - run generate3d.py first (--masks needs "
                          f"the segmentation sidecar, not the VOC XML)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in doc["categories"]}
    image_id_by_file = {img["file_name"]: img["id"] for img in doc["images"]}
    anns_by_image: dict = {}
    for a in doc["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    return image_id_by_file, anns_by_image, cat_names


def draw_masks_one(img_path: Path, anns: list, cat_names: dict,
                    alpha: float = MASK_ALPHA) -> Image.Image:
    """Alpha-blend each instance's decoded RLE over the render in its class colour.

    A box proves nothing about a mask's shape; this is the only view that
    can catch a mask bleeding under a seated cell or straddling two regions
    it shouldn't.
    """
    im = Image.open(img_path).convert("RGB")
    if not anns:
        return im
    arr = np.asarray(im, dtype=np.float32).copy()
    for a in anns:
        name = cat_names.get(a["category_id"])
        colour = SEG_COLOURS.get(name)
        if colour is None:
            continue
        mask = rle_decode(a["segmentation"]).astype(bool)
        c = np.array(colour, dtype=np.float32)
        arr[mask] = arr[mask] * (1 - alpha) + c * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def tile(images, cols: int, out: Path, label_texts=None):
    if not images:
        raise SystemExit("nothing to tile")
    rows = math.ceil(len(images) / cols)
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    pad = 26 if label_texts else 4
    sheet = Image.new("RGB", (cols * w, rows * (h + pad)), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        sheet.paste(im, (c * w, r * (h + pad) + pad))
        if label_texts:
            d.text((c * w + 6, r * (h + pad) + 6), label_texts[k],
                   fill=(235, 235, 240))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  ({len(images)} panels, {sheet.width}x{sheet.height})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="dataset root with images/ + annotations/")
    ap.add_argument("--sweep", default=None, help="sweep root produced by --sweep")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thick", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--masks", action="store_true",
                    help="overlay instances_seg.json's per-instance RLE masks "
                         f"(alpha {MASK_ALPHA}) instead of drawing VOC boxes; "
                         "--data only, needs the segmentation sidecar")
    a = ap.parse_args()

    if a.sweep:
        root = Path(a.sweep)
        pngs = sorted((root / "images").glob("sweep_*.png"))
        if not pngs:
            raise SystemExit(f"no sweep renders in {root / 'images'}")
        labels = [p.stem.replace("sweep_", "") for p in pngs]
        imgs = [Image.open(p).convert("RGB") for p in pngs]
        out = Path(a.out) if a.out else root / "sweep_sheet.png"
        tile(imgs, min(a.cols, len(imgs)), out, labels)
        return

    if not a.data:
        raise SystemExit("pass --data or --sweep")
    root = Path(a.data)
    pngs = sorted((root / "images").glob("*.png"))[:a.n]
    if not pngs:
        raise SystemExit(f"no images in {root / 'images'}")
    out = Path(a.out) if a.out else root / "contact_sheet.png"

    if a.masks:
        image_id_by_file, anns_by_image, cat_names = load_seg_index(root)
        imgs, labels = [], []
        counts: Counter = Counter()
        for p in pngs:
            img_id = image_id_by_file.get(p.name)
            anns = anns_by_image.get(img_id, []) if img_id is not None else []
            imgs.append(draw_masks_one(p, anns, cat_names))
            labels.append(p.stem)
            for a_ in anns:
                counts[cat_names.get(a_["category_id"], "?")] += 1
        tile(imgs, a.cols, out, labels)
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
        print(f"{total} mask instances across {len(imgs)} images ({breakdown})")
        return

    imgs, labels = [], []
    n_boxes = 0
    for p in pngs:
        x = root / "annotations" / (p.stem + ".xml")
        imgs.append(draw_one(p, x, a.thick))
        if x.exists():
            n_boxes += len(parse_voc_xml(x, CLASS_MAP).boxes)
        labels.append(p.stem)
    tile(imgs, a.cols, out, labels)
    print(f"{n_boxes} boxes across {len(imgs)} images "
          f"({n_boxes / max(1, len(imgs)):.1f} per image)")


if __name__ == "__main__":
    main()
