"""
recog.synth3d.render - Cycles configuration, the object-index pass, and mask
readback. Requires bpy.

Two renders per sample:
  1. index pass  - 1 sample, near-zero filter width, geometry only. Fast.
  2. beauty pass - full Cycles quality.

The index pass is what produces pixel-exact boxes. Projected 3D bounding-box
corners are far too loose for rounded or cylindrical parts, and they cannot
represent occlusion at all.

Blender 5.0 moved four things this module depends on. Each is handled by a
small helper that also still works on 4.x:

  scene.render.filter_width  -> scene.cycles.filter_width   (_set_filter_width)
  scene.node_tree            -> scene.compositing_node_group (_compositor_tree)
  RenderLayers "IndexOB"     -> "Object Index"        (_object_index_socket)
  File Output base_path/file_slots
                             -> directory/file_name/file_output_items
                                                     (_configure_file_output)

Note that `scene.use_nodes = True` still SUCCEEDS on 5.0 while doing nothing
useful, so it can neither switch the compositor on nor be used to detect the
version.
"""

from __future__ import annotations

import os

import bpy
import numpy as np


INDEX_NODE_GROUP = "_synth3d_index_comp"


# --------------------------------------------------------------------------- #
#  version-portability helpers
# --------------------------------------------------------------------------- #

def _set_filter_width(scene, value: float):
    """
    Blender 5.0 removed scene.render.filter_width; Cycles reads
    scene.cycles.filter_width. Set whichever exists so 4.2 still works.

    Intent (Invariant #1): a wide reconstruction filter can blend neighbouring
    object indices at silhouette edges into fractional ids that decode to the
    WRONG instance. That corrupts labels silently - the render still looks fine.

    Measured on Blender 5.0.0: Cycles does NOT filter the Object Index pass.
    Two adjacent cubes rendered at filter widths 0.01, 1.5 and 5.0, at both 1
    and 16 samples, produced byte-identical index maps with zero fractional
    pixels. So on 5.0 this call is defensive rather than load-bearing, and the
    index-pass gate below does not exercise it. It is kept because 4.x Cycles
    does filter this pass, and because a future engine change reverting to
    filtered data passes would otherwise reintroduce the corruption silently.
    """
    done = False
    cycles = getattr(scene, "cycles", None)          # None if addon disabled
    if cycles is not None and hasattr(cycles, "filter_width"):
        cycles.filter_width = value
        done = True
    if hasattr(scene.render, "filter_width"):
        scene.render.filter_width = value
        done = True
    if not done:
        raise RuntimeError(
            "no filter-width property found; the index pass would blend "
            "object indices at silhouettes and mislabel instances")


def _compositor_tree(scene):
    """
    Get a writable compositor node tree across Blender 4.x and 5.0.

    4.2: scene.use_nodes = True, then scene.node_tree.
    5.0: the scene compositor became a node GROUP; scene.node_tree no longer
         exists. Note scene.use_nodes = True still SUCCEEDS silently on 5.0
         and does nothing useful, so it cannot be used to detect the version.
    """
    if hasattr(scene, "compositing_node_group"):          # Blender 5.0+
        ng = bpy.data.node_groups.get(INDEX_NODE_GROUP)
        if ng is None:
            ng = bpy.data.node_groups.new(INDEX_NODE_GROUP, "CompositorNodeTree")
        scene.compositing_node_group = ng
        return ng
    scene.use_nodes = True                                 # Blender 4.x
    return scene.node_tree


_INDEX_SOCKET_NAMES = ("Object Index", "IndexOB")


def _object_index_socket(rl_node):
    """
    The render-layer socket was renamed 'IndexOB' -> 'Object Index' in 5.0.

    Keyed lookup on a node's socket collection only finds ENABLED sockets, so
    this doubles as a check that the object-index pass is actually being
    produced. On 5.0 the socket is present but disabled while the engine is
    EEVEE, which has no object-index pass - hence _configure_index_render must
    run (and select Cycles) before the compositor graph is built. Linking a
    disabled socket would yield an all-zero mask and silently drop every box,
    so a disabled socket is reported rather than used.
    """
    for key in _INDEX_SOCKET_NAMES:
        if key in rl_node.outputs:
            return rl_node.outputs[key]
    if any(o.name in _INDEX_SOCKET_NAMES for o in rl_node.outputs):
        scene = bpy.context.scene
        raise RuntimeError(
            "the object-index render-layer socket exists but is disabled, so "
            "the pass is not being rendered (engine="
            f"{scene.render.engine}, use_pass_object_index="
            f"{bpy.context.view_layer.use_pass_object_index}). "
            "The index pass requires the Cycles engine.")
    raise RuntimeError(
        f"no object-index output on the render-layer node; "
        f"available: {[o.name for o in rl_node.outputs]}")


def _configure_file_output(out, mask_dir: str, stem: str):
    """
    Aim a File Output node at a single-channel 32-bit EXR under mask_dir.

    Blender 5.0 rewrote this node: base_path -> directory + file_name, and
    file_slots -> file_output_items. The node-level `format` now only accepts
    OPEN_EXR_MULTILAYER, so the per-item format override is what selects a
    plain single-layer EXR.

    The 5.0 item name is deliberately EMPTY. It becomes the EXR layer prefix,
    and any non-empty prefix names the channel "<item>.V", which Blender then
    classifies as multilayer and refuses to hand back through Image.pixels -
    the readback comes out as a 0x0 image with 0 channels. Verified on 5.0.0:
    item name "probe_"/"V"/"Image" all read back empty, "" reads back correctly.

    The written file is {stem}_.exr on 5.0 (no frame number) but
    {stem}_0001.exr on 4.x, which is why render_index_map keeps a directory
    scan as a fallback.
    """
    directory = os.path.abspath(mask_dir)

    if hasattr(out, "file_output_items"):                   # Blender 5.0+
        out.directory = directory
        out.file_name = stem + "_"          # this, not the item, names the file
        out.save_as_render = False          # raw data: no view transform
        out.file_output_items.clear()
        item = out.file_output_items.new("FLOAT", "")
        item.save_as_render = False
        item.override_node_format = True
        fmt = item.format
    else:                                                   # Blender 4.x
        out.base_path = directory
        out.file_slots.clear()
        out.file_slots.new(stem + "_")
        fmt = out.format

    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "BW"
    fmt.color_depth = "32"
    fmt.exr_codec = "ZIP"


# --------------------------------------------------------------------------- #
#  beauty render settings
# --------------------------------------------------------------------------- #

def configure_beauty(cfg, out_png: str):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.render.resolution_x, s.render.resolution_y = cfg.res
    s.render.resolution_percentage = 100
    s.render.film_transparent = cfg.film_transparent
    _set_filter_width(s, 1.5)
    s.render.use_persistent_data = cfg.persistent_data
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.render.image_settings.color_depth = "8"
    s.render.dither_intensity = 1.0
    s.render.filepath = os.path.abspath(out_png)

    c = s.cycles
    c.device = cfg.device
    c.samples = cfg.samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = cfg.adaptive_threshold
    c.use_denoising = cfg.denoise
    for attr, val in (("denoiser", "OPENIMAGEDENOISE"),
                      ("denoising_input_passes", "RGB_ALBEDO_NORMAL"),
                      ("denoising_prefilter", "ACCURATE")):
        try:
            setattr(c, attr, val)
        except Exception:
            pass
    c.max_bounces = cfg.max_bounces
    c.diffuse_bounces = 4
    c.glossy_bounces = 8
    c.transmission_bounces = 8
    c.transparent_max_bounces = 8
    c.volume_bounces = 0
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.blur_glossy = 1.0
    c.sample_clamp_indirect = cfg.clamp_indirect
    if hasattr(c, "use_light_tree"):
        c.use_light_tree = True

    try:
        s.view_settings.view_transform = cfg.view_transform
    except TypeError:
        pass
    s.view_settings.exposure = cfg.exposure

    if cfg.device == "GPU":
        enable_gpu()


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        print("[warn] cycles addon preferences unavailable")
        return
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            break
        except TypeError:
            continue
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type != "CPU")
    print(f"[gpu] backend={prefs.compute_device_type} "
          f"devices={[d.name for d in prefs.devices if d.use]}")


# --------------------------------------------------------------------------- #
#  index pass
# --------------------------------------------------------------------------- #

def _enable_index_output(mask_dir: str, stem: str):
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    vl.use_pass_object_index = True
    vl.use_pass_combined = True

    nt = _compositor_tree(scene)
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    out = nt.nodes.new("CompositorNodeOutputFile")
    _configure_file_output(out, mask_dir, stem)
    nt.links.new(_object_index_socket(rl), out.inputs[0])
    return out


def _configure_index_render(cfg):
    """
    One sample per pixel and a near-zero reconstruction filter, so the object
    index values stay integral. At the default 1.5px filter Blender blends
    neighbouring object indices at silhouette edges, producing fractional ids
    that decode to the wrong instance.
    """
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.render.resolution_x, s.render.resolution_y = cfg.res
    s.render.resolution_percentage = 100
    _set_filter_width(s, 0.01)
    s.render.film_transparent = False
    s.render.use_persistent_data = cfg.persistent_data
    s.cycles.samples = 1
    s.cycles.use_adaptive_sampling = False
    s.cycles.use_denoising = False
    s.cycles.max_bounces = 0
    s.cycles.device = cfg.device


def read_index_exr(path: str, expect_res=None) -> np.ndarray:
    """Load the EXR back through Blender; return an int32 (H, W) id map."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    w, h = img.size
    nchan = img.channels
    if not (w and h and nchan):
        bpy.data.images.remove(img)
        raise RuntimeError(
            f"Blender read {path} as an empty image ({w}x{h}, {nchan} "
            f"channels). That is what a MULTILAYER EXR looks like from here: "
            f"a non-empty File Output item name becomes an EXR layer prefix "
            f"and Image.pixels then comes back empty. See "
            f"_configure_file_output.")
    buf = np.empty(w * h * nchan, dtype=np.float32)
    img.pixels.foreach_get(buf)          # much faster than list(img.pixels)
    arr = buf.reshape(h, w, nchan)[..., 0]
    bpy.data.images.remove(img)

    arr = np.flipud(arr)                 # Blender buffers are bottom-up
    ids = np.rint(arr).astype(np.int32)
    ids[ids < 0] = 0
    if expect_res and (w, h) != tuple(expect_res):
        print(f"[warn] mask {w}x{h} != configured {tuple(expect_res)}")
    return ids


def render_index_map(cfg, mask_dir: str, stem: str) -> np.ndarray:
    os.makedirs(mask_dir, exist_ok=True)
    # Order matters on Blender 5.0: the render-layer node only exposes the
    # object-index socket once the engine that produces that pass is active,
    # so the engine must be selected before the compositor graph is built.
    _configure_index_render(cfg)
    _enable_index_output(mask_dir, stem)

    scene = bpy.context.scene
    prev = scene.render.filepath
    scene.render.filepath = os.path.join(mask_dir, "_discard_")
    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)     # File Output writes the EXR
    scene.render.filepath = prev

    exr = os.path.join(os.path.abspath(mask_dir), f"{stem}_0001.exr")
    if not os.path.exists(exr):
        cands = [f for f in os.listdir(mask_dir)
                 if f.startswith(stem) and f.endswith(".exr")]
        if not cands:
            raise RuntimeError(f"index pass produced no EXR for {stem}")
        exr = os.path.join(os.path.abspath(mask_dir), sorted(cands)[-1])

    ids = read_index_exr(exr, cfg.res)
    os.remove(exr)
    return ids


def isolated_areas(cfg, mask_dir: str, objects_by_id):
    """
    True unoccluded silhouette area per instance, for a real visible-fraction.
    Costs one cheap render per instance - up to ~32 per sample.

    CURRENTLY YIELDS NO SIGNAL, for two independent reasons, so `--visibility`
    buys renders and nothing else:

      * `layout.plan` and `layout.plan_jig` both guarantee exact AABB
        non-overlap, so no LABELLED instance can occlude another. Everything
        that can occlude - the jig plate, the PCB, the backdrop - carries
        pass_index 0 and is not in `objects_by_id`, so every measured fraction
        comes back exactly 1.0.
      * `annotate.merge_group_boxes` hard-codes `visible_fraction = None` on
        merged boxes, and every `cartridge` is merged, so the measurement never
        reaches a cartridge annotation even in principle.

    The function itself is correct; enable it once a layout mode that actually
    piles parts on each other exists.
    """
    areas = {}
    all_objs = [o for objs in objects_by_id.values() for o in objs]
    saved = {o: o.hide_render for o in all_objs}
    for pid, objs in objects_by_id.items():
        for o in all_objs:
            o.hide_render = o not in objs
        ids = render_index_map(cfg, mask_dir, f"_iso{pid}")
        areas[pid] = int((ids == pid).sum())
    for o, h in saved.items():
        o.hide_render = h
    return areas


def render_beauty(cfg, out_png: str):
    scene = bpy.context.scene
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None      # Blender 5.0
    else:
        scene.use_nodes = False                  # Blender 4.x
    bpy.context.view_layer.use_pass_object_index = False
    configure_beauty(cfg, out_png)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    bpy.ops.render.render(write_still=True)


def save_mask_png(ids: np.ndarray, path: str):
    """8-bit greyscale instance-id map (up to 255 instances)."""
    H, W = ids.shape
    if ids.max() > 255:
        print("[warn] >255 instances; mask PNG ids will wrap")
    img = bpy.data.images.new("mask_out", width=W, height=H, alpha=False,
                              float_buffer=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    v = np.flipud(ids).astype(np.float32) / 255.0     # back to bottom-up
    rgba = np.stack([v, v, v, np.ones_like(v)], axis=-1).ravel()
    img.pixels.foreach_set(rgba)
    img.filepath_raw = os.path.abspath(path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
