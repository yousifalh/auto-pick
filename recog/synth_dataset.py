"""Procedural synthetic-scene generator for the recognition module.

Produces 2-D overhead views of cartridges (green rectangular trays with
a dark central PCB region) and loose 18650/21700-style battery
cylinders, plus a matching Pascal-VOC XML annotation for each image.

The generator is a *development aid*, not a substitute for the real
factory-floor dataset. It is used:

* in the test suite to produce reliable inputs for the heuristic
  detector and the VOC-XML parser, and
* as a synthetic image source for :func:`main.run` when no trained
  checkpoint is available.

Usage::

    python -m recog.synth_dataset --out recog/dataset --n 50
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - hard dep
    raise ImportError(
        "opencv-python is required for recog.synth_dataset"
    ) from exc


# Rectangle type: (xmin, ymin, xmax, ymax)
Rect = Tuple[int, int, int, int]


# ---------------------------------------------------------- primitives ----

def _rectangles_overlap(a: Rect, b: Rect) -> bool:
    """``True`` if the two rectangles share a strictly positive area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _draw_cartridge(img: np.ndarray, rng: random.Random) -> Rect:
    """Render one cartridge into ``img`` and return its bbox."""
    h_img, w_img = img.shape[:2]
    w = rng.randint(260, 440)
    h = rng.randint(180, 320)
    x = rng.randint(40, max(40, w_img - w - 40))
    y = rng.randint(40, max(40, h_img - h - 40))

    # Green tray body (channel order: B, G, R). The green channel is
    # deliberately the dominant one so that the heuristic detector's
    # HSV green-band mask picks the tray out.
    green = (
        rng.randint(70, 110),
        rng.randint(140, 190),
        rng.randint(70, 110),
    )
    cv2.rectangle(img, (x, y), (x + w, y + h), green, thickness=-1)
    cv2.rectangle(
        img, (x, y), (x + w, y + h),
        tuple(int(c * 0.85) for c in green), thickness=3,
    )

    # Central PCB — a dark rectangle with a sprinkling of soldermask
    # dots. This is what the placement-area extractor removes.
    pcb_w = int(w * rng.uniform(0.22, 0.35))
    pcb_h = int(h * rng.uniform(0.40, 0.60))
    px = x + (w - pcb_w) // 2
    py = y + (h - pcb_h) // 2
    cv2.rectangle(
        img, (px, py), (px + pcb_w, py + pcb_h),
        (30, 30, 30), thickness=-1,
    )
    for _ in range(rng.randint(6, 14)):
        cv2.circle(
            img,
            (rng.randint(px + 5, px + pcb_w - 5),
             rng.randint(py + 5, py + pcb_h - 5)),
            radius=2, color=(180, 180, 180), thickness=-1,
        )

    return x, y, x + w, y + h


def _draw_battery(
    img: np.ndarray,
    rng: random.Random,
    avoid: Sequence[Rect],
) -> Rect:
    """Render one battery cylinder, avoiding the given occupied rects."""
    h_img, w_img = img.shape[:2]

    bw = rng.randint(30, 60)
    bh = int(bw * rng.uniform(3.0, 3.8))  # 18650-ish 3.6:1 aspect ratio

    for _ in range(50):
        x = rng.randint(10, max(10, w_img - bw - 10))
        y = rng.randint(10, max(10, h_img - bh - 10))
        rect = (x, y, x + bw, y + bh)
        if not any(_rectangles_overlap(rect, a) for a in avoid):
            break

    body = (rng.randint(160, 210),) * 3  # brushed-metal grey
    cv2.rectangle(img, (x, y), (x + bw, y + bh), body, thickness=-1)
    # Positive-terminal cap
    cv2.rectangle(
        img, (x + 2, y), (x + bw - 2, y + 8),
        (60, 60, 60), thickness=-1,
    )
    # Vertical specular streak
    streak_x = x + int(bw * rng.uniform(0.30, 0.70))
    cv2.line(
        img, (streak_x, y + 4), (streak_x, y + bh - 4),
        (255, 255, 255), thickness=2,
    )
    return rect


# ------------------------------------------------------ annotation I/O ----

def _write_voc_xml(
    path: Path,
    filename: str,
    hwc: Tuple[int, int, int],
    objects: Sequence[Tuple[str, Rect]],
) -> None:
    """Emit a single Pascal-VOC annotation file at ``path``."""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(hwc[1])
    ET.SubElement(size, "height").text = str(hwc[0])
    ET.SubElement(size, "depth").text = str(hwc[2])
    for name, (x0, y0, x1, y1) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x0))
        ET.SubElement(bnd, "ymin").text = str(int(y0))
        ET.SubElement(bnd, "xmax").text = str(int(x1))
        ET.SubElement(bnd, "ymax").text = str(int(y1))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------- public entry point ---

def generate_dataset(
    out_dir: str | Path,
    n: int = 50,
    seed: int = 0,
    size: Tuple[int, int] = (720, 1280),
) -> int:
    """Generate ``n`` synthetic scenes under ``out_dir``.

    Images are written to ``out_dir/images`` and annotations to
    ``out_dir/annotations``. Returns the number of scenes produced
    (always ``n`` unless generation is interrupted).
    """
    rng = random.Random(seed)
    root = Path(out_dir)
    img_dir = root / "images"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    height, width = size

    for i in range(n):
        # Neutral grey background to simulate the factory floor with a
        # small per-image offset so the recogniser cannot cheat on a
        # constant-brightness prior.
        base = int(np.clip(180 + rng.randint(-20, 20), 0, 255))
        img = np.full((height, width, 3), base, dtype=np.uint8)

        placed: List[Rect] = []
        objects: List[Tuple[str, Rect]] = []

        # One to three cartridges, non-overlapping.
        for _ in range(rng.randint(1, 3)):
            for _attempt in range(20):
                rect = _draw_cartridge(img, rng)
                if not any(_rectangles_overlap(rect, p) for p in placed):
                    placed.append(rect)
                    objects.append(("cartridge", rect))
                    break

        # Three to ten loose batteries.
        for _ in range(rng.randint(3, 10)):
            objects.append(("battery", _draw_battery(img, rng, placed)))

        # Per-image gain to simulate uncontrolled factory lighting.
        gain = rng.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        name = f"synth_{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), img)
        _write_voc_xml(
            ann_dir / f"{name}.xml",
            f"{name}.png",
            (height, width, 3),
            objects,
        )
    return n


# ------------------------------------------------------------- CLI entry --

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic recognition dataset.")
    parser.add_argument("--out", default="recog/dataset")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    count = generate_dataset(args.out, n=args.n, seed=args.seed)
    print(f"Generated {count} synthetic scenes in {args.out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
