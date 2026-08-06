"""
Generate the synthetic 3-D dataset. Runs inside Blender.

    BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
    "$BLENDER" -b --python recog/generate3d.py -- --n 20 --out recog/dev3d --res 640 360
    "$BLENDER" -b --python recog/generate3d.py -- --n 2000 --out recog/dataset3d --device GPU --resume

Output is flat Pascal-VOC that recog.dataset.BatteryCartridgeDataset reads
directly; recog/training.py owns the train/val split.

Presets come from configs/synth3d.yaml. Blender has no PyYAML, so run
`python -m recog.sync_config` after editing it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402

from recog.synth3d import annotate, render, scene as S  # noqa: E402
from recog.synth3d.assets import AssetLibrary  # noqa: E402
from recog.synth3d.config import (CLASSES, VARIANTS, class_ids,  # noqa: E402
                                  load_config)


def parse_args(cfg):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=None, help="defaults to recog/synth3d/assets")
    p.add_argument("--out", default="recog/dataset3d")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="scene")
    p.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--device", choices=["CPU", "GPU"], default=None)
    p.add_argument("--backdrop", choices=list(cfg.backdrops), default=None)
    p.add_argument("--lighting", choices=list(cfg.lighting), default=None)
    p.add_argument("--layout-mode", choices=["scatter", "jig"], default=None)
    p.add_argument("--variant", choices=[v.name for v in VARIANTS], default=None)
    p.add_argument("--sweep", choices=["lighting", "backdrop"], default=None,
                   help="render ONE fixed scene once per entry in that axis")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--visibility", action="store_true",
                   help="measure each instance's unoccluded area (one extra "
                        "index render per instance, up to ~32 per sample). "
                        "Live since layout.max_overlap_iou, but small: "
                        "measured over 90 overlapping scatter scenes, 2.0% of "
                        "non-merged boxes came back under 1.0 and the deepest "
                        "was 0.748, against a filter.min_visibility of 0.25 - "
                        "so it changed no label. merge_group_boxes still "
                        "reports None for merged boxes, i.e. for every "
                        "cartridge. Diagnostic, not part of a production run.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def _anchor_range(path: str = None):
    """
    (smallest, largest) anchor size in pixels, read from the recognition config.

    torchvision's AnchorGenerator sizes are sqrt(anchor area), so this is
    directly comparable to sqrt(box area) - NOT to a box diagonal.
    `recog/model.py` appends `base_scales[-1] * 2` as the P6 level, so the real
    upper bound is twice the largest configured scale.

    Returns None if the file is missing or the value cannot be read. This feeds
    a warning; a warning must never stop a 2000-frame render. Blender's bundled
    Python has no PyYAML, hence the line-scan fallback.
    """
    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "recognition.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    scales = None
    try:
        import yaml
        scales = ((yaml.safe_load(text) or {}).get("model") or {}).get(
            "anchor_scales")
    except Exception:
        scales = None
    if not scales:
        m = re.search(r"^\s*anchor_scales:\s*\[([^\]]*)\]", text, re.M)
        if not m:
            return None
        scales = [v for v in m.group(1).split(",") if v.strip()]
    try:
        vals = sorted(float(s) for s in scales)
    except (TypeError, ValueError):
        return None
    if not vals:
        return None
    return vals[0], vals[-1] * 2.0


# The resolution filter.min_px is expressed at. It is a PIXEL count, so it
# does not travel with --res; see _filter_res_check.
_FILTER_TUNED_RES = (1280, 720)


def _filter_res_check(cfg):
    """
    Warn when `--res` makes `filter.min_px` throw away legitimate parts.

    min_px is a silhouette PIXEL count, so halving each side quarters every
    part's area while the threshold stays put. That did not matter while it was
    80 - nothing in the corpus came close - but it is now 500, chosen to floor
    the box size against frame-truncation slivers at the production resolution.
    MEASURED on a 48-scene 640x360 run, the same scenes yielded 365 boxes
    before and 266 after, every loss recorded in `dropped` with its reason.
    Production renders at 1280x720 and is unaffected (177 vs 185 boxes over 24
    scenes), so this is a dev-run artefact and a warning rather than a rescale:
    silently scaling the threshold would make a dev run disagree with
    production about which parts exist, which is the harder bug to find.
    """
    W, H = cfg.render.res
    rw, rh = _FILTER_TUNED_RES
    ratio = (W * H) / float(rw * rh)
    if ratio >= 0.9 or cfg.filter.min_px <= 0:
        return
    print(f"[warn] filter.min_px={cfg.filter.min_px} is expressed at "
          f"{rw}x{rh}; at {W}x{H} a part covers {ratio:.2f}x the pixels, so "
          f"this behaves like a threshold of {cfg.filter.min_px / ratio:.0f} "
          f"there and will drop parts a production run keeps. Every loss is "
          f"in each scene's `dropped` list and in manifest "
          f"stats.dropped_instances.")


def _dirs(root, save_masks):
    subs = ["images", "annotations", "meta"] + (["masks"] if save_masks else [])
    for s in subs:
        os.makedirs(os.path.join(root, s), exist_ok=True)
    tmp = os.path.join(root, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def run_sweep(a, cfg, library, root):
    """
    One fixed scene, rendered once per entry in the swept axis.

    Same seed, same layout, same materials, same camera - only the swept axis
    moves. The RNG is redrawn from the SAME seed for each entry so the scene is
    literally identical; re-using a single advanced RNG would drift.
    """
    axis = a.sweep
    entries = list(cfg.lighting if axis == "lighting" else cfg.backdrops)
    print(f"sweeping {axis}: {entries}")
    for entry in entries:
        _, params, rng = next(iter(S.scene_generator(1, a.seed, cfg)))
        params[axis] = entry
        stem = f"sweep_{axis}_{entry}"
        id_meta, groups, meta = S.build(params, rng, library, cfg)
        if not id_meta:
            print(f"[{entry}] nothing placed, skipping")
            continue
        png = os.path.join(root, "images", stem + ".png")
        if not a.no_render:
            # Same seed for every entry, so the drawn exposure is identical
            # across the sweep: the swept axis really is the only thing that
            # moves, which is the whole contract of a sweep sheet.
            render.render_beauty(cfg.render, png, params.get("exposure"))
        meta.update({"sweep_axis": axis, "sweep_entry": entry,
                     "image": stem + ".png"})
        with open(os.path.join(root, "meta", stem + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[sweep] {entry} -> {png}")
    print(f"[done] {root}\nNow tile them:\n"
          f"  python -m recog.verify3d --sweep {root} "
          f"--out {os.path.join(root, axis + '_sheet.png')}")


def main():
    cfg = load_config()
    a = parse_args(cfg)
    if a.config:
        cfg = load_config(a.config)
    if a.res:
        cfg.render.res = (a.res[0], a.res[1])
    if a.samples:
        cfg.render.samples = a.samples
    if a.device:
        cfg.render.device = a.device
    if a.variant:
        VARIANTS[:] = [v for v in VARIANTS if v.name == a.variant]
    if a.visibility:
        print("[warn] --visibility costs one extra index render per instance. "
              "It yields real but small signal now that layout.max_overlap_iou "
              "lets parts occlude: measured over 90 overlapping scatter "
              "scenes, 2.0% of non-merged boxes fell under 1.0 and the deepest "
              "was 0.748, so filter.min_visibility=0.25 dropped nothing. "
              "Merged boxes (every cartridge) still get no visible_fraction.")

    _filter_res_check(cfg)

    ids = class_ids()
    root = os.path.abspath(a.out)
    tmp = _dirs(root, a.save_masks)

    assets_dir = a.assets or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synth3d", "assets")
    library = AssetLibrary(assets_dir)
    print(f"assets: {library.names()}")

    if a.sweep:
        run_sweep(a, cfg, library, root)
        return

    W, H = cfg.render.res
    overrides = {"backdrop": a.backdrop, "lighting": a.lighting,
                 "layout_mode": a.layout_mode}
    ws, hs = [], []
    per_class = {c: 0 for c in CLASSES}
    per_variant, per_mode, n_drop, n_images = {}, {}, 0, 0

    for i, params, rng in S.scene_generator(a.n, a.seed, cfg, overrides):
        stem = f"{a.prefix}_{i:05d}"
        png = os.path.join(root, "images", stem + ".png")
        xml = os.path.join(root, "annotations", stem + ".xml")
        meta_path = os.path.join(root, "meta", stem + ".json")

        # The image must exist too, or a --no-render pass followed by a
        # --resume run leaves annotations with no pixels: the dataset lists
        # images/, so len(dataset) would be 0 while manifest.json claimed a
        # full image count. Under --no-render there is no PNG to demand.
        if (a.resume and os.path.exists(meta_path) and os.path.exists(xml)
                and (a.no_render or os.path.exists(png))):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            id_meta, groups, meta = S.build(params, rng, library, cfg)
            if not id_meta:
                print(f"[{i}] nothing placed, skipping")
                continue

            mask = render.render_index_map(cfg.render, tmp, stem)

            full_areas = None
            if a.visibility:
                objs = {int(pid): [bpy.data.objects[n] for n in names
                                   if n in bpy.data.objects]
                        for pid, names in meta["objects_by_id"].items()}
                full_areas = render.isolated_areas(cfg.render, tmp, objs)

            anns, dropped = annotate.boxes_from_mask(mask, id_meta, ids,
                                                     cfg.filter, full_areas)
            anns = annotate.merge_group_boxes(anns, groups, ids, cfg.filter)

            if a.save_masks:
                render.save_mask_png(
                    mask, os.path.join(root, "masks", stem + ".png"))
            if a.save_blend and i == 0:
                bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
            if not a.no_render:
                render.render_beauty(cfg.render, png, params.get("exposure"))

            annotate.write_voc_xml(xml, stem + ".png", W, H, anns)

            meta.pop("objects_by_id", None)
            meta.update({"index": i, "seed": a.seed, "image": stem + ".png",
                         "width": W, "height": H, "annotations": anns,
                         "dropped": dropped, "classes": CLASSES,
                         "class_to_id": ids})
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        n_images += 1
        for ann in meta["annotations"]:
            ws.append(ann["bbox_xywh"][2])
            hs.append(ann["bbox_xywh"][3])
            per_class[ann["class"]] = per_class.get(ann["class"], 0) + 1
            per_variant[ann.get("variant")] = per_variant.get(ann.get("variant"), 0) + 1
        mode = meta["params"].get("layout_mode")
        per_mode[mode] = per_mode.get(mode, 0) + 1
        n_drop += len(meta.get("dropped", []))

        print(f"[{i + 1}/{a.n}] {stem}  {len(meta['annotations'])} boxes  "
              f"({meta['params']['backdrop']}/{meta['params']['lighting']}"
              f"/{mode})")

    stats = {
        "n_images": n_images,
        "n_boxes": len(ws),
        "per_class": per_class,
        "per_variant": per_variant,
        "per_layout_mode": per_mode,
        "dropped_instances": n_drop,
        "box_w_px": {"min": min(ws, default=0), "max": max(ws, default=0),
                     "mean": round(statistics.fmean(ws), 1) if ws else 0},
        "box_h_px": {"min": min(hs, default=0), "max": max(hs, default=0),
                     "mean": round(statistics.fmean(hs), 1) if hs else 0},
    }
    if ws:
        def _pcts(vals):
            v = sorted(vals)
            return {"p05": round(v[int(0.05 * (len(v) - 1))], 1),
                    "p50": round(v[len(v) // 2], 1),
                    "p95": round(v[int(0.95 * (len(v) - 1))], 1)}

        stats["box_diag_px"] = _pcts(math.hypot(w, h) for w, h in zip(ws, hs))
        # sqrt(area) is the quantity anchor_scales is expressed in.
        stats["box_sqrt_area_px"] = _pcts(
            math.sqrt(w * h) for w, h in zip(ws, hs))

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"classes": CLASSES, "class_to_id": ids,
                   "num_classes_with_background": len(CLASSES) + 1,
                   "seed": a.seed, "config": cfg.to_dict(),
                   "variants": [v.name for v in VARIANTS],
                   "catalog": library.catalog, "stats": stats}, f, indent=2)

    try:
        os.rmdir(tmp)
    except OSError:
        pass

    print(json.dumps(stats, indent=2))
    _anchor_check(stats, cfg.render.res)
    print(f"[done] {root}")


# torchvision FasterRCNN's GeneralizedRCNNTransform defaults. recog/model.py
# deliberately does NOT override them (see the NOTE there), so these are what
# every render is resized to before the RPN ever sees it, in training as well
# as inference.
_TV_MIN_SIZE, _TV_MAX_SIZE = 800.0, 1333.0


def _transform_scale(res):
    """
    The factor GeneralizedRCNNTransform applies to a `res` render, and hence
    to its boxes, before they meet the anchors.

    This is why a low-res dev run no longer trips a false warning: torchvision
    resizes a 640x360 render to the SAME 1333x750 as a 1280x720 one, so a box
    is the same number of network pixels either way. Comparing raw rendered
    pixels against anchor_scales - which the previous version did - made every
    `--res 640 360` run look like a regression and taught the reader to ignore
    this check.
    """
    lo, hi = min(res), max(res)
    return min(_TV_MIN_SIZE / lo, _TV_MAX_SIZE / hi)


def _anchor_check(stats, res):
    """
    Warn when the box sizes fall off EITHER end of the configured anchor range.
    The original version hard-coded "32-512px" and only warned about boxes
    being too large, so it could not detect the inverted failure where anchors
    are far smaller than the boxes - which is exactly what happened when this
    dataset replaced the cv2 one.

    The band is the anchor range widened by sqrt(2) at each end, not the
    anchor range itself. That is the same criterion the anchor tuning uses:
    two same-aspect boxes a factor of sqrt(2) apart in sqrt(area) have a
    centred IoU of exactly 0.50, so a box within sqrt(2) of the nearest anchor
    still has an anchor that can match it. `param_space.zoom` deliberately
    puts the smallest boxes a little under the smallest anchor - the old
    strict test would fire on every single production run and mean nothing.
    """
    s = stats.get("box_sqrt_area_px")
    if not s:
        return
    rng = _anchor_range()
    if rng is None:
        print("\nAnchor check: skipped - could not read model.anchor_scales "
              "from configs/recognition.yaml")
        return
    lo, hi = rng
    k = _transform_scale(res)
    p05, p50, p95 = (s["p05"] * k, s["p50"] * k, s["p95"] * k)
    floor, ceil = lo / math.sqrt(2.0), hi * math.sqrt(2.0)
    print(f"\nAnchor check: configs/recognition.yaml covers "
          f"{lo:g}-{hi:g}px sqrt(area) (anchor_scales, plus the x2 P6 level "
          f"recog/model.py appends), matchable at IoU>=0.5 over "
          f"{floor:.0f}-{ceil:.0f}px; your boxes are p05={p05:.1f} "
          f"p50={p50:.1f} p95={p95:.1f} after torchvision resizes the "
          f"rendered {res[0]}x{res[1]} by {k:.3f} "
          f"(raw p05={s['p05']} p50={s['p50']} p95={s['p95']})")
    if p05 < floor:
        print(f"  WARNING: p05={p05:.1f} is below {floor:.0f}px, more than "
              f"sqrt(2) under the smallest anchor ({lo:g}px) - those boxes "
              f"cannot reach 0.5 IoU with any anchor. Lower anchor_scales in "
              f"configs/recognition.yaml, or narrow param_space.zoom.")
    if p95 > ceil:
        print(f"  WARNING: p95={p95:.1f} is above {ceil:.0f}px, more than "
              f"sqrt(2) over the largest anchor ({hi:g}px). Widen "
              f"anchor_scales in configs/recognition.yaml, or narrow "
              f"param_space.zoom.")


if __name__ == "__main__":
    main()
