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

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _too_thin(w: int, h: int, cfg) -> bool:
    """
    True when a box is a longer, thinner strip than any WHOLE part can be.

    Only frame truncation produces these. A part cropped along its length
    leaves a sliver whose aspect ratio is a property of where the frame edge
    fell, not of the object - and no anchor can ever match it, because
    `model.anchor_ratios` is (correctly) tuned to the parts' own aspect
    ratios. Measured over 1245 boxes from 150 zoom-varied scenes, EVERY box
    with an aspect above 3.7 was truncated, and the widest un-cropped box was
    3.68 (the 18650's own 65.0/18.3 = 3.55 plus rotation jitter and
    antialiasing); the four above 4.0 ran 4.6, 5.1, 6.9 and 9.8 and scored
    0.34-0.45 best-centred-IoU against every anchor set tried, including sets
    that put every other box above 0.5.

    `max_aspect = 0` disables the check, which is the dataclass default: a
    config that predates this filter behaves exactly as it did.
    """
    limit = getattr(cfg, "max_aspect", 0.0) or 0.0
    if limit <= 0.0:
        return False
    lo, hi = min(w, h), max(w, h)
    return lo <= 0 or hi > limit * lo


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
        elif _too_thin(w, h, cfg):
            reason = f"aspect>{cfg.max_aspect}"
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
        if (area < cfg.min_px or min(x1 - x0, y1 - y0) < cfg.min_side
                or _too_thin(x1 - x0, y1 - y0, cfg)):
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
            # the group's - a shell top and a shell bottom occlude each other,
            # so summing them double-counts and biases the ratio low. A correct
            # group fraction needs a further isolated render of the whole
            # group. Consequence, documented in configs/synth3d.yaml:
            # filter.min_visibility never applies to a merged box, which is
            # every `cartridge`. Measured over 90 overlapping scatter scenes,
            # the deepest occlusion anywhere left 74.8% of a part visible
            # against a 0.25 threshold, so computing it would not have changed
            # a single label.
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


def rle_encode(mask: np.ndarray) -> Dict[str, object]:
    """Uncompressed COCO RLE for a binary mask.

    Blender's bundled Python has no pycocotools, so this is hand-rolled.

    COCO runs are COLUMN-major and always start with a background run,
    which is why a mask whose first pixel is set begins `[0, ...]`. A
    row-major encoder produces a file that every reader accepts and every
    reader decodes transposed - silently.
    """
    m = np.asarray(mask, dtype=np.uint8)
    flat = m.flatten(order="F")
    counts: List[int] = []
    last = 0
    run = 0
    for v in flat:
        if v == last:
            run += 1
        else:
            counts.append(run)
            last = v
            run = 1
    counts.append(run)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_decode(rle: Dict[str, object]) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    value = 0
    for run in rle["counts"]:
        flat[pos:pos + run] = value
        pos += run
        value ^= 1
    return flat.reshape((h, w), order="F")


# Per-class, per-filter exemptions. A filter tuned for one class's geometry
# does not automatically transfer to another's - see the two cases below,
# both found the same way: by measuring what a filter actually threw away
# against the real construction of the class it was hitting.
#
# placement_area is exempt from every filter (min_px, min_side, max_aspect,
# min_visibility, and drop_truncated). See the spec's ruling 4: under the
# modal definition a nearly-full cartridge has a small, thin, mostly-occluded
# strip of free floor, which is exactly what those filters discard - and
# exactly the cartridge where knowing the remaining room matters most.
# Filtering it would teach the segmenter that a nearly-full bay has NO
# placement area rather than a small one. A bay with zero visible free floor
# still yields nothing, because it never appears in np.unique; that case is
# correct, there is no room.
#
# obstruction is exempt from min_px and max_aspect ONLY - min_side and
# min_visibility stay in force.
#   min_px=500 (configs/synth3d.yaml) is a CELL-sized threshold: at the
#   production 1280x720 render over an 800mm layout (1.6 px/mm), a battery
#   (18.3x65mm) is ~29x104px = ~3000px^2, but an adhesive blob at
#   adhesive_frac 0.04-0.14 of a ~55mm bay edge is 2-8mm i.e. 3-13px, or
#   25-500px^2 - an order of magnitude smaller. Measured over the 27
#   obstructions a 24-scene run built: 8/27 passed min_px, 19/27 failed, so
#   min_px alone discarded ~85% of the class Task 5 exists to create, while
#   the object stayed fully RENDERED - teaching the segmenter that glue is
#   bay floor, the exact confusion the class was added to prevent. min_side
#   stays in force because, unlike min_px, 6px (3.75mm at 1.6px/mm) is a
#   legibility floor rather than a cell-sized one and rejects real noise on
#   its own: the same measurement found 24/27 obstructions clearing it.
#   max_aspect is tuned to 4.0 against battery cells' own aspect ratio (see
#   _too_thin's docstring), and does NOT transfer to obstruction: `bay.
#   sample_obstructions` draws a tape strip's width from 5-12% of the bay's
#   short edge and its height from 50-95% of the SAME edge - an aspect
#   ratio of ~4-19 BY CONSTRUCTION, not from frame truncation. Applying
#   max_aspect here would discard most real tape, the identical mistake
#   min_px made on the whole class, just for a different reason.
#
# battery and cartridge get no exemptions: SEG_CLASSES starts with CLASSES
# specifically so ids 1 and 2 mean the same thing in both the VOC and the
# seg files, and max_aspect applies to both here exactly as it does in
# boxes_from_mask, so the two paths never disagree on these two classes by
# omission.
#
# electronics_module also gets no exemptions. Unlike a scattered obstruction,
# its true (untruncated) shape is fixed by the CAD: catalog.json's
# module_bay_mm gives an aspect ratio of 2.16-3.05 across all four Anker
# assemblies, safely under 4.0, so max_aspect only ever fires on a genuine
# frame-truncation sliver here, the case it exists to catch.
_FILTER_EXEMPT: Dict[str, frozenset] = {
    "placement_area": frozenset(
        {"min_px", "min_side", "max_aspect", "min_visibility", "truncated"}),
    "obstruction": frozenset({"min_px", "max_aspect"}),
}


def _filter_exempt(cls: str, filt: str) -> bool:
    return filt in _FILTER_EXEMPT.get(cls, frozenset())


def masks_from_index(ids: np.ndarray, id_meta: Dict[int, dict],
                     class_ids: Dict[str, int], cfg,
                     full_areas: Dict[int, int] = None
                     ) -> Tuple[List[dict], List[dict]]:
    """Like :func:`boxes_from_mask`, but each annotation carries an RLE.

    Kept separate rather than folded into boxes_from_mask so the VOC
    detector path is provably untouched: it still calls the old function
    with the two-class map and gets byte-identical output.
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
            dropped.append({"pass_index": pid, "class": cls,
                            "reason": "unmapped"})
            continue

        inst = (ids == pid)
        ys, xs = np.nonzero(inst)
        visible_px = int(xs.size)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        truncated = bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H)

        vis_frac = None
        if full_areas and full_areas.get(pid):
            vis_frac = round(visible_px / max(1, full_areas[pid]), 4)

        reason = None
        if not _filter_exempt(cls, "min_px") and visible_px < cfg.min_px:
            reason = f"visible_px<{cfg.min_px}"
        elif not _filter_exempt(cls, "min_side") and min(w, h) < cfg.min_side:
            reason = f"side<{cfg.min_side}"
        elif not _filter_exempt(cls, "max_aspect") and _too_thin(w, h, cfg):
            reason = f"aspect>{cfg.max_aspect}"
        elif (not _filter_exempt(cls, "truncated")
              and cfg.drop_truncated and truncated):
            reason = "truncated"
        elif (not _filter_exempt(cls, "min_visibility")
              and vis_frac is not None and vis_frac < cfg.min_visibility):
            reason = f"visibility<{cfg.min_visibility}"
        if reason:
            dropped.append({"pass_index": pid, "class": cls,
                            "reason": reason, "visible_px": visible_px})
            continue

        anns.append({
            "pass_index": pid,
            "class": cls,
            "category_id": class_ids[cls],
            "bbox_xyxy": [x0, y0, x1, y1],
            "bbox_xywh": [x0, y0, w, h],
            "segmentation": rle_encode(inst),
            "area": visible_px,
            "truncated": truncated,
            "visible_fraction": vis_frac,
            "asset": meta.get("asset"),
            "variant": meta.get("variant"),
            "iscrowd": 0,
        })
    return anns, dropped


def write_coco_json(path: str, images: Sequence[dict],
                    annotations: Sequence[dict],
                    seg_class_ids: Dict[str, int]) -> None:
    """Write the segmentation sidecar.

    Sits ALONGSIDE the Pascal-VOC output rather than replacing it. VOC
    has no mask field, and the detector's training path must keep reading
    the two-class VOC files unchanged.
    """
    doc = {
        "info": {"description": "auto-pick synthetic segmentation set"},
        "licenses": [],
        "categories": [{"id": i, "name": n, "supercategory": ""}
                       for n, i in sorted(seg_class_ids.items(),
                                          key=lambda kv: kv[1])],
        "images": list(images),
        "annotations": [
            {"id": a["id"], "image_id": a["image_id"],
             "category_id": a["category_id"],
             "bbox": a["bbox_xywh"], "area": a["area"],
             "segmentation": a["segmentation"], "iscrowd": a.get("iscrowd", 0)}
            for a in annotations
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
