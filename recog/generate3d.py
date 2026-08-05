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
    p.add_argument("--visibility", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def _dirs(root, save_masks):
    subs = ["images", "annotations", "meta"] + (["masks"] if save_masks else [])
    for s in subs:
        os.makedirs(os.path.join(root, s), exist_ok=True)
    tmp = os.path.join(root, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def run_sweep(a, cfg, library, ids, root, tmp):
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
            render.render_beauty(cfg.render, png)
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

    ids = class_ids()
    root = os.path.abspath(a.out)
    tmp = _dirs(root, a.save_masks)

    assets_dir = a.assets or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synth3d", "assets")
    library = AssetLibrary(assets_dir)
    print(f"assets: {library.names()}")

    if a.sweep:
        run_sweep(a, cfg, library, ids, root, tmp)
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
                render.render_beauty(cfg.render, png)

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
        diags = sorted(math.hypot(w, h) for w, h in zip(ws, hs))
        stats["box_diag_px"] = {
            "p05": round(diags[int(0.05 * (len(diags) - 1))], 1),
            "p50": round(diags[len(diags) // 2], 1),
            "p95": round(diags[int(0.95 * (len(diags) - 1))], 1)}

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
    if stats.get("box_diag_px"):
        d = stats["box_diag_px"]
        print(f"\nAnchor check: FPN defaults cover 32-512px diagonals; "
              f"yours are p05={d['p05']} p50={d['p50']} p95={d['p95']}")
        if d["p95"] > 480:
            print("  WARNING: p95 near the top of the anchor range. Enlarge "
                  "layout.area or widen anchor_scales in configs/recognition.yaml")
    print(f"[done] {root}")


if __name__ == "__main__":
    main()
