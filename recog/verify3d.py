"""
Draw the generated boxes onto the renders so you can actually look at them.

    python -m recog.verify3d --data recog/dev3d --n 12
    python -m recog.verify3d --sweep sweeps/ --out sweeps/lighting_sheet.png

Inspecting the contact sheet is not optional. A silently-wrong mask pass
produces boxes that look plausible in JSON and are obviously wrong on screen.

System Python only: Blender's bundled interpreter has no Pillow.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

from recog.dataset import CLASS_MAP, parse_voc_xml

COLOURS = {"battery": (60, 220, 90), "cartridge": (255, 90, 60)}
INV = {v: k for k, v in CLASS_MAP.items()}


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
    ap.add_argument("--data", default=None, help="dataset root with images/ + annotations/")
    ap.add_argument("--sweep", default=None, help="sweep root produced by --sweep")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thick", type=int, default=3)
    ap.add_argument("--out", default=None)
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
    imgs, labels = [], []
    n_boxes = 0
    for p in pngs:
        x = root / "annotations" / (p.stem + ".xml")
        imgs.append(draw_one(p, x, a.thick))
        if x.exists():
            n_boxes += len(parse_voc_xml(x, CLASS_MAP).boxes)
        labels.append(p.stem)
    out = Path(a.out) if a.out else root / "contact_sheet.png"
    tile(imgs, a.cols, out, labels)
    print(f"{n_boxes} boxes across {len(imgs)} images "
          f"({n_boxes / max(1, len(imgs)):.1f} per image)")


if __name__ == "__main__":
    main()
