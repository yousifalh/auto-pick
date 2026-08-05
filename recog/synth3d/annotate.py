"""
recog.synth3d.annotate - instance mask -> boxes -> Pascal-VOC XML.

No bpy import, so the whole annotation path is unit-testable outside Blender.
The mask comes from render.py (Blender's object-index pass), but every decision
about what becomes a label is made here.

Output is the VOC dialect recog.dataset.parse_voc_xml reads: <filename>,
<size>/<width|height|depth>, and one <object> per instance with <name> and
<bndbox>. Splitting is NOT done here - recog/training.py random_splits a flat
directory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np


def boxes_from_mask(ids: np.ndarray, id_meta: Dict[int, dict], class_ids: Dict[str, int],
                    cfg, full_areas: Dict[int, int] = None
                    ) -> Tuple[List[dict], List[dict]]:
    """
    ids       (H, W) int32 instance map, 0 = background
    id_meta   pass_index -> {"class": str, "asset": str, "variant": str, ...}
    cfg       FilterCfg

    Boxes are [x0, y0, x1, y1] with EXCLUSIVE max edges: a one-pixel object
    yields a 1x1 box instead of a zero-area box. Zero-area boxes are what make
    FasterRCNN's box regression loss go NaN.

    Objects with no visible pixels never appear in np.unique, so a cell sealed
    inside an assembled shell is dropped automatically - occlusion needs no
    special handling anywhere.
    """
    H, W = ids.shape
    anns: List[dict] = []
    dropped: List[dict] = []

    for pid in np.unique(ids):
        pid = int(pid)
        if pid <= 0:
            continue
        meta = id_meta.get(pid)
        if meta is None:
            continue
        cls = meta.get("class")
        if cls not in class_ids:
            dropped.append({"pass_index": pid, "class": cls, "reason": "unmapped"})
            continue

        ys, xs = np.nonzero(ids == pid)
        visible_px = int(xs.size)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        truncated = bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H)

        vis_frac = None
        if full_areas and full_areas.get(pid):
            vis_frac = round(visible_px / max(1, full_areas[pid]), 4)

        reason = None
        if visible_px < cfg.min_px:
            reason = f"visible_px<{cfg.min_px}"
        elif min(w, h) < cfg.min_side:
            reason = f"side<{cfg.min_side}"
        elif cfg.drop_truncated and truncated:
            reason = "truncated"
        elif vis_frac is not None and vis_frac < cfg.min_visibility:
            reason = f"visibility<{cfg.min_visibility}"
        if reason:
            dropped.append({"pass_index": pid, "class": cls, "reason": reason,
                            "visible_px": visible_px})
            continue

        anns.append({
            "pass_index": pid,
            "class": cls,
            "category_id": class_ids[cls],
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_xywh": [x0, y0, w, h],
            "area": visible_px,
            "truncated": truncated,
            "visible_fraction": vis_frac,
            "asset": meta.get("asset"),
            "variant": meta.get("variant"),
            "iscrowd": 0,
        })
    return anns, dropped


def merge_group_boxes(anns: List[dict], groups: Dict[int, str],
                      class_ids: Dict[str, int], cfg) -> List[dict]:
    """
    Collapse the sub-part boxes of one assembly into a single box.

    Used by the "assembled" variant, where the label is the whole power bank
    rather than its shell halves: the union of the visible shell pieces is the
    object's true silhouette box.
    """
    by_group: Dict[str, List[dict]] = {}
    loose: List[dict] = []
    for a in anns:
        gid = groups.get(a["pass_index"])
        (by_group.setdefault(gid, []) if gid else loose).append(a)

    out = list(loose)
    for gid, members in by_group.items():
        if not members:
            continue
        cls = members[0]["class"]
        x0 = min(m["bbox_xyxy"][0] for m in members)
        y0 = min(m["bbox_xyxy"][1] for m in members)
        x1 = max(m["bbox_xyxy"][2] for m in members)
        y1 = max(m["bbox_xyxy"][3] for m in members)
        area = sum(m["area"] for m in members)
        if area < cfg.min_px or min(x1 - x0, y1 - y0) < cfg.min_side:
            continue
        out.append({
            "pass_index": members[0]["pass_index"],
            "class": cls,
            "category_id": class_ids[cls],
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            "area": area,
            "truncated": any(m["truncated"] for m in members),
            # Deliberately None, not a computed value: the members' fractions
            # are over their own isolated silhouettes and do not compose into
            # the group's. Consequence, documented in configs/synth3d.yaml:
            # filter.min_visibility never applies to a merged box, which is
            # every `cartridge`.
            "visible_fraction": None,
            "asset": members[0].get("asset"),
            "variant": members[0].get("variant"),
            "iscrowd": 0,
            "merged_from": [m["pass_index"] for m in members],
        })
    return out


def write_voc_xml(path: str, filename: str, width: int, height: int,
                  anns: Sequence[dict], depth: int = 3) -> None:
    """
    Write one Pascal-VOC annotation file.

    Coordinates are written through int(), which truncates rather than
    raises: a caller that hands this float bbox coordinates (e.g.
    [1.0, 1.9, 4.9, 4.1]) gets back a silently shrunk box ([1, 1, 4, 4]),
    not an error - so callers must keep boxes integer-valued themselves
    (boxes_from_mask always does). Width and height are the fields that
    matter for round-tripping: parse_voc_xml casts THEM through int(),
    which does raise on a float string such as "80.0"; its bndbox fields
    use float() and would silently accept either. Boxes keep their
    exclusive max edges, so a 1-pixel object stays a 1x1 box and never
    degenerates to zero area (which makes FasterRCNN's regression loss go
    NaN).
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(width))
    ET.SubElement(size, "height").text = str(int(height))
    ET.SubElement(size, "depth").text = str(int(depth))

    for a in anns:
        x0, y0, x1, y1 = a["bbox_xyxy"]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = a["class"]
        ET.SubElement(obj, "truncated").text = "1" if a.get("truncated") else "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x0))
        ET.SubElement(bnd, "ymin").text = str(int(y0))
        ET.SubElement(bnd, "xmax").text = str(int(x1))
        ET.SubElement(bnd, "ymax").text = str(int(y1))

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
