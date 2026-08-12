"""
recog.synth3d.world - backdrop, lighting, the overhead camera, and the two
pieces of scene furniture (the jig plate and the PCB). Requires bpy.

Presets are not module globals: they are read off the loaded config, so
`build_backdrop` and `setup_lighting` take a `cfg` (a `config.Config`).

The jig plate carries `pass_index = 0` on purpose. It is scene content, not
a class: it must occlude correctly in the index pass but must never produce
an annotation. Keeping it at 0 is what makes that work -
`annotate.boxes_from_mask` skips id 0 as background, and any NON-zero id that
had no `id_meta` entry would be dropped silently and without an audit trail.

The PCB built by `build_pcb` also carries `pass_index = 0` here, but that is
a placeholder, not the final word: scene.py assigns it a real id and an
`electronics_module` id_meta entry once it knows the board object, so its
mask is real ground truth rather than occlusion-only furniture. Its child
ports and inductor are given the SAME id as the board, not left at 0 - see
scene.py's pass-index loop for why.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict

import bpy
from mathutils import Matrix, Vector

from . import assets as A
from . import bay
from .config import CELL_FORMATS, CELL_H_MM, CELL_W_MM
from .lightrig import off_axis_placement, shadow_direction
from .materials import apply_to_object, for_role, set_input, rng_range


# --------------------------------------------------------------------------- #
#  surface properties that are also written into the manifest
# --------------------------------------------------------------------------- #

def _set_recorded(node, name: str, value) -> None:
    """`materials.set_input`, but loud when the socket is not there.

    `set_input` returns False and continues if the named socket is missing.
    That tolerance is deliberate and stays: Principled socket names moved in
    Blender 4.x, and `materials.build` relies on it to try `Coat Weight` then
    `Clearcoat`. But every call site in THIS module writes the same value into
    the `drawn` dict it returns, and `scene.py` puts that dict straight into
    the render's manifest. A silent False therefore produces a render whose
    surface is Blender's default while the manifest states the value that
    never reached the shader - which is, exactly, the defect
    `materials._assert_wear_mix_took` was written for after it discarded the
    drawn roughness on 100% of surfaces.

    Nothing here can be reached from pytest, so like that assertion this runs
    on every surface built rather than in a test. The four socket names it
    guards - Base Color, Roughness - are the ones that have NOT moved across
    3.x/4.x/5.x; the ones that did move are set through plain `set_input` and
    keep their tolerance.
    """
    if not set_input(node, name, value):
        raise KeyError(
            f"this Blender build's {node.name!r} has no {name!r} input, so "
            f"{value!r} never reached the shader - while the manifest this "
            f"function returns records it as though it had. The build offers "
            f"{[s.name for s in node.inputs]}. A generator that renders "
            f"something other than what it recorded is worse than one that "
            f"stops, so this stops.")


# --------------------------------------------------------------------------- #
#  colour temperature
# --------------------------------------------------------------------------- #

def kelvin_to_rgb(k: float):
    """Tanner Helland's blackbody approximation, peak-normalized."""
    k = max(1000.0, min(40000.0, k)) / 100.0
    if k <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(k) - 161.1195681661 if k > 0 else 0.0
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60) ** -0.0755148492)
    if k >= 66:
        b = 255.0
    elif k <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(k - 10) - 305.0447927307
    clamp = lambda v: max(0.0, min(255.0, v)) / 255.0
    return (clamp(r), clamp(g), clamp(b))


# --------------------------------------------------------------------------- #
#  backdrop
# --------------------------------------------------------------------------- #

# Fallback albedo ramp per procedural kind, used only when the backdrop has no
# `color` in the config. The authored values live in configs/synth3d.yaml so
# that the single property that decides whether a part is visible against its
# background is visible - and tunable - from the config.
_FALLBACK_PALETTE = {
    "concrete": ((0.22, 0.22, 0.215), (0.46, 0.455, 0.44)),
    "brushed": ((0.30, 0.305, 0.31), (0.52, 0.525, 0.535)),
    "fabric": ((0.10, 0.12, 0.16), (0.24, 0.27, 0.33)),
    "paper": ((0.76, 0.75, 0.72), (0.90, 0.89, 0.86)),
    "belt": ((0.035, 0.035, 0.038), (0.115, 0.115, 0.12)),
}


def _procedural(nt, kind: str, rng: random.Random, color=None):
    """Returns (colour_socket, height_socket, mapping_node).

    `color` is [[r,g,b] low, [r,g,b] high] in LINEAR space, from the
    backdrop's config entry; None falls back to `_FALLBACK_PALETTE`.
    """
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    if kind == "brushed":
        mapping.inputs["Scale"].default_value = (1.0, 60.0, 1.0)
    elif kind == "belt":
        mapping.inputs["Scale"].default_value = (1.0, 24.0, 1.0)

    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = {
        "concrete": rng.uniform(8, 22), "brushed": rng.uniform(20, 60),
        "fabric": rng.uniform(120, 280), "paper": rng.uniform(180, 400),
        "belt": rng.uniform(10, 30)}.get(kind, 20.0)
    tex.inputs["Detail"].default_value = 6.0 if kind == "concrete" else 3.0
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    lo, hi = (color if color else
              _FALLBACK_PALETTE.get(kind, ((0.2, 0.2, 0.2), (0.5, 0.5, 0.5))))
    ramp.color_ramp.elements[0].color = tuple(lo)[:3] + (1.0,)
    ramp.color_ramp.elements[1].color = tuple(hi)[:3] + (1.0,)
    if kind == "concrete":
        ramp.color_ramp.elements[0].position = 0.30
        ramp.color_ramp.elements[1].position = 0.72
    # "Fac" is the socket IDENTIFIER on every Blender version; 5.0 renamed the
    # display NAME to "Factor" but bpy resolves the identifier first, so this
    # keyed lookup is version-portable. Verified on 5.0.0.
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    return ramp.outputs["Color"], tex.outputs["Fac"], mapping


def build_backdrop(name: str, rng: random.Random, cfg, size: float = 3.0,
                   z: float = 0.0):
    """
    The ground plane every scatter scene sits on.

    `z` defaults to 0, which is where scatter parts are dropped to, and must
    stay there for them. Jig scenes pass the plate's `backdrop_z` instead: the
    pocket floors are cut BELOW z = 0, so with the backdrop left at 0 a pocket
    showed backdrop rather than its own floor and the recesses were
    geometrically real but visually absent. See `build_jig`.
    """
    spec = cfg.backdrops[name]
    drawn = {
        "backdrop": name,
        "luma_ref": spec.get("luma_ref"),
        "uv_scale": rng_range(rng, spec["uv_scale"]),
        "brightness": rng_range(rng, spec["brightness"]),
        "roughness": rng_range(rng, spec["roughness"]),
        "bump": rng_range(rng, spec["bump"]),
        "rot90": rng.choice([0, 90, 180, 270]),
        "source": "image" if spec["image"] else f"proc:{spec['proc']}",
    }

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z))
    plane = bpy.context.active_object
    plane.name = "Backdrop"
    plane.pass_index = 0
    drawn["z"] = z

    mat = bpy.data.materials.new("Backdrop")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")

    if spec["image"] and os.path.exists(spec["image"]):
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        img = nt.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(os.path.abspath(spec["image"]))
        img.extension = "REPEAT"
        nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], img.inputs["Vector"])
        color_out = height_out = img.outputs["Color"]
    else:
        color_out, height_out, mapping = _procedural(nt, spec["proc"], rng,
                                                     spec.get("color"))

    mapping.inputs["Scale"].default_value = tuple(
        v * drawn["uv_scale"] for v in mapping.inputs["Scale"].default_value)
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(drawn["rot90"]))

    bright = nt.nodes.new("ShaderNodeBrightContrast")
    bright.inputs["Bright"].default_value = drawn["brightness"]
    nt.links.new(color_out, bright.inputs["Color"])
    nt.links.new(bright.outputs["Color"], bsdf.inputs["Base Color"])
    _set_recorded(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 1.0 if spec.get("proc") == "brushed" else 0.0)

    if drawn["bump"] > 0.005:
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = drawn["bump"]
        nt.links.new(height_out, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    plane.data.materials.append(mat)
    return plane, drawn


# --------------------------------------------------------------------------- #
#  lighting
# --------------------------------------------------------------------------- #

def _lamp_color(kelvin: float, tint):
    """
    Blackbody colour, optionally pulled off the Planckian locus.

    `tint` is a per-channel multiplier from the rig's config. It exists for
    one reason: real fluorescent tubes are NOT blackbody radiators. They are
    mercury-vapour discharge with a phosphor coating and a hard spike near
    546 nm, which is why a fluorescent-lit factory photograph has a green
    cast that no colour temperature can reproduce. Feeding the detector only
    Planckian illuminants teaches it that the illuminant lies on a
    one-parameter curve, and every real fluorescent frame is then off-model.
    """
    r, g, b = kelvin_to_rgb(kelvin)
    if tint:
        r, g, b = r * tint[0], g * tint[1], b * tint[2]
    return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)))


def _add_area_light(name: str, loc, rot, energy: float, size: float,
                    color, shape: str = "SQUARE"):
    bpy.ops.object.light_add(type="AREA", location=loc)
    lt = bpy.context.active_object
    lt.name = name
    lt.data.energy = energy
    lt.data.shape = shape
    lt.data.size = size
    lt.data.color = color
    lt.rotation_euler = rot
    return lt


def _off_axis_lamp(spec, rng: random.Random, name: str, keys: dict,
                   drawn: dict, out_prefix: str, azimuth_base: float = 0.0):
    """
    Draw one off-axis lamp from `spec` and build it.

    Returns the drawn azimuth, or None if the rig does not configure this
    lamp - the fill lamp is optional and most rigs omit it.

    `keys` maps each logical parameter to its config key, so the key lamp and
    the fill lamp share this code while reading `energy` / `fill_energy` and
    so on. `azimuth_base` is what makes the fill lamp's azimuth an OFFSET
    from the key lamp's rather than an independent draw: a mixed-illuminant
    rig is only mixed if the two illuminants actually arrive from different
    sides, and two independent draws from [0, 360) land on top of each other
    often enough to hollow the rig out.
    """
    if keys["energy"] not in spec:
        return None
    energy = rng_range(rng, spec[keys["energy"]])
    size = rng_range(rng, spec[keys["size"]])
    kelvin = rng_range(rng, spec[keys["kelvin"]])
    elevation = rng_range(rng, spec[keys["elevation"]])
    distance = rng_range(rng, spec[keys["distance"]])
    azimuth = (azimuth_base + rng_range(rng, spec[keys["azimuth"]])) % 360.0

    loc, rot = off_axis_placement(azimuth, elevation, distance)
    _add_area_light(name, loc, rot, energy, size,
                    _lamp_color(kelvin, spec.get(keys["tint"])))
    sx, sy = shadow_direction(azimuth)
    drawn.update({
        out_prefix + "energy": energy, out_prefix + "size": size,
        out_prefix + "kelvin": kelvin, out_prefix + "azimuth": azimuth,
        out_prefix + "elevation": elevation, out_prefix + "distance": distance,
        out_prefix + "shadow_dir": [sx, sy],
    })
    return azimuth


_KEY_LAMP = {"energy": "energy", "size": "size", "kelvin": "kelvin",
             "elevation": "elevation", "distance": "distance",
             "azimuth": "azimuth", "tint": "tint"}

_FILL_LAMP = {"energy": "fill_energy", "size": "fill_size",
              "kelvin": "fill_kelvin", "elevation": "fill_elevation",
              "distance": "fill_distance", "azimuth": "fill_azimuth_offset",
              "tint": "fill_tint"}


def setup_lighting(preset_name: str, rng: random.Random, cam_loc, cfg):
    spec = cfg.lighting[preset_name]
    drawn = {"lighting": preset_name, "kind": spec["kind"]}

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")

    if spec["kind"] in ("hdri", "hdri_plus_softbox") and spec["hdri"]:
        if not os.path.exists(spec["hdri"]):
            print(f"[warn] HDRI missing: {spec['hdri']} - using flat grey sky")
            set_input(bg, "Color", (0.5, 0.5, 0.5, 1.0))
            set_input(bg, "Strength", 1.0)
            drawn.update(hdri=None)
        else:
            rot = rng_range(rng, spec["hdri_rotation"])
            strength = rng_range(rng, spec["hdri_strength"])
            coord = nt.nodes.new("ShaderNodeTexCoord")
            mapping = nt.nodes.new("ShaderNodeMapping")
            mapping.inputs["Rotation"].default_value = (0, 0, math.radians(rot))
            env = nt.nodes.new("ShaderNodeTexEnvironment")
            env.image = bpy.data.images.load(os.path.abspath(spec["hdri"]))
            nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
            nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
            nt.links.new(env.outputs["Color"], bg.inputs["Color"])
            set_input(bg, "Strength", strength)
            drawn.update(hdri=spec["hdri"], hdri_strength=strength,
                         hdri_rotation=rot)
    else:
        wk = rng_range(rng, spec["world_kelvin"])
        ws = rng_range(rng, spec["world_strength"])
        set_input(bg, "Color", kelvin_to_rgb(wk) + (1.0,))
        set_input(bg, "Strength", ws)
        drawn.update(world_kelvin=wk, world_strength=ws)

    if spec["kind"] in ("camera_softbox", "hdri_plus_softbox"):
        energy = rng_range(rng, spec["energy"])
        size = rng_range(rng, spec["size"])
        kelvin = rng_range(rng, spec["kelvin"])
        off = rng_range(rng, spec["offset"])
        ang = rng.uniform(0, 2 * math.pi)

        loc = (cam_loc[0] + off * math.cos(ang),
               cam_loc[1] + off * math.sin(ang),
               cam_loc[2] - 0.06)
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.active_object
        lt.name = "CameraSoftbox"
        lt.data.energy = energy
        lt.data.shape = "SQUARE"
        lt.data.size = size
        lt.data.color = kelvin_to_rgb(kelvin)
        lt.rotation_euler = (0, 0, 0)      # area lights emit along -Z

        # The softbox hangs BETWEEN the camera and the parts (cam_z - 0.06) and
        # is up to 1.6m across, so under the top-down ortho camera it spans the
        # whole 0.8m frame. That looks like it should black out the shot, and it
        # does not: measured on Blender 5.0.0, an area light is not an occluder
        # for camera rays. Rendering the same scene with the light object's
        # `visible_camera` forced True and False gave byte-identical statistics
        # (mean 0.9823, std 0.0270) and the index pass was unchanged at 921456
        # px for a full-frame plane either way. `visible_camera` is False by
        # default for a light on 5.0 in any case, so nothing is set here.
        drawn.update(energy=energy, size=size, kelvin=kelvin, offset=off)

    elif spec["kind"] == "off_axis":
        # The whole point of this kind. `camera_softbox` hangs the lamp at the
        # camera, so under a top-down camera the light is coaxial with the
        # view: shading is flat and every cast shadow falls exactly behind the
        # object that cast it, where the camera cannot see it. Real factory
        # light arrives from overhead and to one side, and the reference
        # photos show the shadows to prove it. Azimuth is drawn over the full
        # circle and elevation over the rig's own band, so shadow DIRECTION
        # and LENGTH both vary across the dataset instead of being absent
        # from it.
        key_az = _off_axis_lamp(spec, rng, "KeyLight", _KEY_LAMP, drawn, "")
        if key_az is None:
            raise ValueError(
                f"lighting preset {preset_name!r} declares kind 'off_axis' "
                f"but configures no `energy`, so it would build no key lamp "
                f"and render lit only by the world background. That is a "
                f"config bug, not a dim rig: it silently ignores every other "
                f"key the preset sets.")
        _off_axis_lamp(spec, rng, "FillLight", _FILL_LAMP, drawn, "fill_",
                       azimuth_base=key_az)

    return drawn


# --------------------------------------------------------------------------- #
#  camera
# --------------------------------------------------------------------------- #

def _frame_extent(area_w: float, area_h: float, aspect: float) -> float:
    """
    Size, in metres, that must be given to the camera's LONGER sensor axis for
    the whole layout area to be inside the frame.

    Blender's `ortho_scale` (and, for a perspective camera, `sensor_width` under
    the default AUTO sensor fit) always describes the LONGER render axis and
    derives the shorter one by dividing by the render aspect. So:

      landscape (aspect >= 1): frame is S wide and S/aspect tall, hence
                               S >= area_w and S >= area_h * aspect
      portrait  (aspect <  1): frame is S tall and S * aspect wide, hence
                               S >= area_h and S >= area_w / aspect

    With area [0.80, 0.45] at 1280x720 both landscape terms are 0.80 - the frame
    is used exactly, with no wasted backdrop.
    """
    if aspect >= 1.0:
        return max(area_w, area_h * aspect)
    return max(area_w / aspect, area_h)


def setup_camera(cfg, layout_cfg, res, rng: random.Random, top_z: float = 0.0,
                 zoom: float = 1.0):
    """
    Bird's-eye camera. A camera with zero rotation already looks down -Z, so a
    top-down view needs no aiming.

    Framing derives from the layout AREA, not from the objects, so scale stays
    consistent across the dataset for a given `zoom`.

    `zoom` is the per-scene framing multiplier drawn from `param_space.zoom`.
    It multiplies the framed extent, so LARGER zoom means a WIDER view and
    SMALLER parts - the same sense as `margin`, which it stacks with. It
    defaults to 1.0, so a config without `param_space.zoom` frames exactly as
    it did before.

    It exists because a fixed framing made the dataset effectively
    single-scale: measured over the 1000-image set, battery sqrt(area) ran
    p05 50.9 / p95 56.2 px, a 1.10 ratio, against a real-photo span of
    43-65 px (1.51). The `cartridge` class is the control - it has four
    different CAD models and therefore a 1.69 training ratio, and it is the
    class whose AP does not collapse when scale shifts.

    At `zoom < 1.0` the frame is NARROWER than the layout area, so parts near
    the area boundary are cropped by the frame edge. That is deliberate: the
    old set had 0 of 8542 truncated annotations while real photos have 2 of 80.
    At `zoom > 1.0` there is spare backdrop around the layout, which is
    harmless - the backdrop plane is `max(area) * 6` across.

    NOTE this is COUPLED to `model.anchor_scales` in configs/recognition.yaml.
    Widening the zoom range without retuning the anchors puts the smallest
    boxes below the smallest anchor, which is the exact mismatch that once
    left 20% of boxes under 0.5 best-IoU. Change them together and re-run the
    anchor check `recog/generate3d.py` prints at the end of a run.

    The framing above is verified empirically, not trusted from the formula.
    On Blender 5.0.0 with area [0.80, 0.45], res 1280x720, margin forced to 1.0
    and shift 0, 4mm marker cubes rendered through the index pass landed at the
    centroids predicted by

        px = res_x/2 + (x - shift_x) / ortho_scale * res_x
        py = res_y/2 - (y - shift_y) / (ortho_scale / aspect) * res_y

    to within a pixel, in all four quadrants:

        layout (+0.30, +0.15) -> (1120.0, 120.0)   predicted (1120.0, 120.0)
        layout (-0.35, -0.20) -> (  80.0, 680.0)   predicted (  80.0, 680.0)
        layout ( 0.00,  0.00) -> ( 640.0, 360.0)   predicted ( 640.0, 360.0)
        layout (+0.20, -0.10) -> ( 960.0, 520.0)   predicted ( 960.0, 520.0)

    and a plane of exactly the layout area covered pixels x 0..1279, y 0..719
    solid - the whole frame, edge to edge, with nothing spare. That is the
    measurement that confirms `ortho_scale` is the LONGER (here horizontal) axis
    and that the vertical extent really is ortho_scale / aspect = 0.45m; had it
    described the shorter axis instead, the area would have overflowed the frame
    by 16:9 and every scene would have been cropped.
    """
    margin = rng_range(rng, cfg.margin_range)
    shift_x = rng_range(rng, cfg.shift_range)
    shift_y = rng_range(rng, cfg.shift_range)

    bpy.ops.object.camera_add(location=(shift_x, shift_y, top_z + cfg.height),
                              rotation=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = "TopCam"
    bpy.context.scene.camera = cam

    area_w, area_h = layout_cfg.area
    res_x, res_y = res
    aspect = res_x / res_y
    need = _frame_extent(area_w, area_h, aspect)

    zoom = float(zoom)
    if zoom <= 0.0:
        raise ValueError(
            f"camera zoom must be positive, got {zoom!r}: it multiplies the "
            f"framed extent, so zero or negative would collapse or mirror the "
            f"frame rather than merely widening it")

    if cfg.ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = need * margin * zoom
    else:
        cam.data.type = "PERSP"
        cam.data.lens = cfg.focal
        fov = 2 * math.atan(cam.data.sensor_width / (2 * cfg.focal))
        half = need / 2 * margin * zoom
        cam.location.z = top_z + half / math.tan(fov / 2)

    cam.data.clip_start, cam.data.clip_end = 0.001, 100.0
    # `ortho_scale` is recorded ONLY on the ortho branch, and the condition is
    # `cfg.ortho` rather than `getattr(cam.data, "ortho_scale", None)`, which
    # is what this line used to be. `bpy.types.Camera.ortho_scale` is an
    # unconditional property of the camera datablock - Blender only HIDES it
    # in the UI when `type != 'ORTHO'` - so the getattr never returned None
    # and a perspective render recorded Blender's untouched 6.0 default as
    # though it were a measurement.
    #
    # That is not a cosmetic manifest wart. `recog.calibration.frame_mm_per_px`
    # derives every frame's true ground sample distance as
    # `ortho_scale * 1000 / width` and feeds it to `plan.planner`, and it
    # raises when the key is absent precisely because a perspective camera has
    # NO single scalar mm_per_px - the scale varies with depth - so
    # substituting one would be a fabricated calibration rather than a
    # fallback. The old line handed it 6.0 to fabricate from. Latent, not
    # historic: `config.CameraConfig.ortho` defaults True and every shipped
    # config sets `ortho: true`, so no rendered dataset carries the wrong
    # number - but it was one config flag away from doing so silently.
    return cam, {"margin": margin, "shift_x": shift_x, "shift_y": shift_y,
                 "zoom": zoom,
                 "ortho": cfg.ortho, "height": cfg.height,
                 "ortho_scale": cam.data.ortho_scale if cfg.ortho else None}


# --------------------------------------------------------------------------- #
#  unlabelled scene furniture
# --------------------------------------------------------------------------- #

# The backdrop plane sits at exactly z = 0 and every part is dropped so its
# lowest point is z = 0 too. Building the plate as `location=(0, 0, -thickness/2)`
# with `scale.z = thickness` - which is what this task's brief specifies - puts
# its top face at exactly z = 0, perfectly coplanar with the backdrop over the
# plate's whole area.
#
# That does NOT show up as the speckled z-fighting you would expect. It is far
# worse and much quieter: every shadow ray leaving the plate's top face starts
# on the backdrop plane too, so it self-occludes immediately and the ENTIRE
# PLATE RENDERS BLACK. Measured on Blender 5.0.0, one jig scene, 64 samples:
# the plate's mean RGB was [0.0003, 0.0003, 0.0003] at lift 0 versus
# [0.869, 0.897, 0.992] at the lift below, and whole-image mean went 0.1355 ->
# 0.9231. The index pass is identical either way (parallel ortho camera rays
# tie-break consistently), so no mask- or box-level gate can catch this - only
# looking at a beauty render can.
#
# Lifting the plate a fraction of a millimetre puts its top face unambiguously
# above the backdrop. The parts then sit JIG_LIFT down into their pockets, which
# is invisible from overhead (0.6mm against an 18mm cell) and marginally more
# realistic than floating exactly level with the rim.
JIG_LIFT = 0.0006          # metres

# Clearance between the plate's underside and the backdrop. Small enough that
# the plate still reads as resting on the surface, large enough that the two
# are never coplanar (see the JIG_LIFT note above for what coplanar costs).
JIG_BACKDROP_GAP = 0.002   # metres


def build_jig(pockets, rng: random.Random):
    """
    Blue 3-D-printed fixture plate with a recess per pocket.

    Built by boolean-differencing pocket cubes out of a slab. The plate is
    UNLABELLED (pass_index 0): it merges with background in the index map and
    produces no annotation, while still correctly occluding what sits behind
    it - the same trick the PCB uses.

    `layout.plan_jig` leaves `layout_cfg.jig_wall` of plate material between
    adjacent pockets, so the cutters are never coplanar with each other and each
    recess stays a separate walled trough.

    The plate is sized to the bounding box of the pockets it was GIVEN, not to
    `layout_cfg.area`. Sizing it to the area produced a plate of 0.82-0.86m
    against a camera ortho_scale of 0.816-0.88m, so the plate filled the frame
    and hid the backdrop in every jig scene - even one holding a single small
    part. Following the packed block instead leaves real backdrop visible
    around the fixture, which is what the reference photos look like.

    Returns (plate, drawn); `drawn["backdrop_z"]` is where the caller must put
    the backdrop plane, BELOW the plate, so that looking into a pocket shows
    the pocket floor instead of the ground plane.
    """
    margin = rng.uniform(0.010, 0.030)
    deepest = max(pk.depth for pk in pockets)
    # A pocket floor sits at JIG_LIFT - depth and the plate's underside at
    # JIG_LIFT - thickness, so a plate thinner than its deepest pocket is
    # punched clean through and the "recess" becomes a hole. jig_depth
    # (6-12mm) and this range (10-18mm) overlap, so this is not hypothetical.
    thickness = max(rng.uniform(0.010, 0.018), deepest + 0.004)

    x0 = min(pk.x - pk.w / 2 for pk in pockets)
    x1 = max(pk.x + pk.w / 2 for pk in pockets)
    y0 = min(pk.y - pk.h / 2 for pk in pockets)
    y1 = max(pk.y + pk.h / 2 for pk in pockets)
    plate_w = (x1 - x0) + 2 * margin
    plate_h = (y1 - y0) + 2 * margin
    plate_cx, plate_cy = (x0 + x1) / 2, (y0 + y1) / 2

    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(plate_cx, plate_cy, JIG_LIFT - thickness / 2))
    plate = bpy.context.active_object
    plate.name = "JigPlate"
    plate.scale = (plate_w, plate_h, thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    plate.pass_index = 0

    cutters = []
    for i, pk in enumerate(pockets):
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(pk.x, pk.y, JIG_LIFT - pk.depth / 2 + 0.0005))
        c = bpy.context.active_object
        c.name = f"_pocket{i}"
        c.scale = (pk.w, pk.h, pk.depth + 0.001)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        cutters.append(c)

    for c in cutters:
        mod = plate.modifiers.new(name=c.name, type="BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.object = c
    bpy.context.view_layer.objects.active = plate
    for c in cutters:
        bpy.ops.object.modifier_apply(modifier=c.name)
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)

    drawn = {"thickness": thickness, "margin": margin, "lift": JIG_LIFT,
             "w": plate_w, "h": plate_h, "cx": plate_cx, "cy": plate_cy,
             "underside_z": JIG_LIFT - thickness,
             "backdrop_z": JIG_LIFT - thickness - JIG_BACKDROP_GAP,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.06, 0.16),
                       rng.uniform(0.35, 0.62)],
             "roughness": rng.uniform(0.45, 0.75),
             "layer_bump": rng.uniform(0.15, 0.45),
             "n_pockets": len(pockets)}

    mat = bpy.data.materials.new("JigPlastic")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    _set_recorded(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    _set_recorded(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 0.0)

    # 3-D-print layer lines: a striped wave along Z, bumped into the normal.
    coord = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = rng.uniform(180.0, 420.0)
    nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = drawn["layer_bump"]
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    plate.data.materials.append(mat)
    return plate, drawn


# Blender's own float/boolean-cut tessellation noise floor for a scene at
# this project's scale (parts tens to hundreds of millimetres) - the same
# order of magnitude _assert_seat_cell_footprint already uses (5e-4 m =
# 0.5mm). Geometry built directly (no boolean op at all) is checked to a
# tighter 0.1mm; the boolean-cut case gets the full 0.5mm because Blender's
# modifier_apply on a DIFFERENCE can leave sub-mm tessellation slop on the
# cut faces - the UNCUT outer faces this function actually measures are not
# themselves subject to that, but the tolerance is kept uniform rather than
# argued case-by-case.
_PROC_TRAY_TOL_MM = 0.5


def _assert_procedural_tray_geometry(entry: dict, case, lid, cell) -> None:
    """Numerically verify the geometry `build_procedural_tray` just built
    against the `TraySample`/`build_tray_entry` numbers it was built from.

    This is deliberately NOT a re-check of the bpy-free entry alone -
    `bay.sample_tray`/`catalog.build_tray_entry` already assert that class
    of thing at sample/registration time, bpy-free. This is the other
    half, and the one only bpy can answer: did world.py's OWN bpy calls
    actually build what the entry said, or did an offset/naming/axis
    mistake in THIS function silently build something else? That is
    exactly the failure class that has already cost this project a render
    cycle three times over - an inverted assembly, a hole cut through the
    wrong mesh face, a module floating in mid-air - and every one of them
    rendered plausibly and passed a green test suite. A loud
    AssertionError naming the exact mismatch, on every build, is the
    deliverable this function exists for.

    Checks, in order:
      1. Pure-number cross-check that `entry` itself is still internally
         consistent (defence against a desync between catalog.
         build_tray_entry's output and what reaches this function - the
         same "renamed catalog key silently stops building geometry"
         shape this project has already hit once).
      2. The BUILT case's measured world-space AABB: outer footprint ==
         case_outer_mm, base at z=0 (the tray's own base), rim at
         z=case_half_height_mm.
      3. The cavity floor sits strictly above the base and strictly below
         the rim, by the sampled depth (case_half_height_mm -
         tray_floor_mm) - measured via the CELL, which is built resting
         exactly on that floor; there is no separate floor object to
         measure directly, since the cavity is a boolean-cut hollow, not
         a mesh of its own.
      4. The BUILT lid rests exactly on the case's own measured rim, at
         the same footprint.
      5. The BUILT cell's measured diameter/length match
         config.CELL_FORMATS[entry['cell_format']] and it rests on the
         cavity floor, not floating or buried, and stays within the
         ASSEMBLED shell's own envelope (case + lid together, 2 *
         case_half_height_mm - the 'assembled' variant keeps the cell at
         this exact built position, sealed by the lid; a cell whose own
         diameter exceeds that combined height would poke through it).
    """
    ix0, iy0, ix1, iy1 = entry["interior_mm"]
    ox0, oy0, ox1, oy1 = entry["case_outer_mm"]
    wall_mm = entry["case_wall_mm"]
    half_h_mm = entry["case_half_height_mm"]
    floor_mm = entry["tray_floor_mm"]

    # ---- 1. entry self-consistency (pure numbers, no bpy) -------------- #
    assert 0.0 < floor_mm < half_h_mm, (
        f"tray_floor_mm ({floor_mm}) is not strictly between the case's "
        f"own base (0) and its own rim (case_half_height_mm={half_h_mm}) "
        f"- the cavity cutter would have zero or negative height")

    # catalog.build_tray_entry rounds interior_mm/case_outer_mm/
    # case_wall_mm INDEPENDENTLY to 2 decimal places, so their exact
    # algebraic identity (interior = case_outer +/- wall) can be off by
    # up to 3 * 0.005mm of pure rounding noise even for a perfectly
    # correct entry - too tight a tolerance here fires on every single
    # tray, which is not what a defect-catching assertion is for. A real
    # "zero wall thickness" defect is off by the FULL wall_mm (multiple
    # mm), so 0.02mm keeps enormous margin against a genuine mismatch
    # while tolerating the rounding this entry was deliberately built with.
    entry_round_tol = 0.02
    for side, (outer, inner) in (
        ("x0", (ox0, ix0)), ("y0", (oy0, iy0)),
    ):
        assert math.isclose(inner - outer, wall_mm, abs_tol=entry_round_tol), (
            f"interior_mm's {side} ({inner}) is not case_outer_mm's {side} "
            f"({outer}) + case_wall_mm ({wall_mm}) - this is the exact "
            f"'zero wall thickness' defect class this plan's header note "
            f"flags: interior_mm must be strictly inside case_outer_mm by "
            f"the sampled wall on every side")
    for side, (outer, inner) in (
        ("x1", (ox1, ix1)), ("y1", (oy1, iy1)),
    ):
        assert math.isclose(outer - inner, wall_mm, abs_tol=entry_round_tol), (
            f"interior_mm's {side} ({inner}) is not case_outer_mm's {side} "
            f"({outer}) - case_wall_mm ({wall_mm}) - this is the exact "
            f"'zero wall thickness' defect class this plan's header note "
            f"flags: interior_mm must be strictly inside case_outer_mm by "
            f"the sampled wall on every side")

    bx0, by0, bx1, by1 = entry["module_bay_mm"]
    assert ix0 <= bx0 <= bx1 <= ix1 and iy0 <= by0 <= by1 <= iy1, (
        f"module_bay_mm {entry['module_bay_mm']} is not contained in "
        f"interior_mm {entry['interior_mm']}")
    # bay.bay_edge() raises ValueError unless module_bay_mm is a full-span
    # strip flush against EXACTLY one edge of interior_mm - reaching the
    # next line already proves "shares exactly one interior edge".
    bay.bay_edge(tuple(entry["interior_mm"]), tuple(entry["module_bay_mm"]))

    # ---- 2/3/4/5. bpy-measured checks on the built geometry ------------ #
    tol = _PROC_TRAY_TOL_MM
    case_lo, case_hi = A.group_bbox([case])
    lid_lo, lid_hi = A.group_bbox([lid])
    cell_lo, cell_hi = A.group_bbox([cell])

    def mm(v: float) -> float:
        return v * 1000.0

    assert math.isclose(mm(case_lo.x), ox0, abs_tol=tol) \
        and math.isclose(mm(case_lo.y), oy0, abs_tol=tol) \
        and math.isclose(mm(case_hi.x), ox1, abs_tol=tol) \
        and math.isclose(mm(case_hi.y), oy1, abs_tol=tol), (
        f"built case footprint ({mm(case_lo.x):.2f},{mm(case_lo.y):.2f})-"
        f"({mm(case_hi.x):.2f},{mm(case_hi.y):.2f})mm != case_outer_mm "
        f"{entry['case_outer_mm']}")
    assert math.isclose(case_lo.z, 0.0, abs_tol=1e-4), (
        f"built case base z={case_lo.z * 1000:.3f}mm != 0 (the tray's own "
        f"base)")
    assert math.isclose(mm(case_hi.z), half_h_mm, abs_tol=tol), (
        f"built case rim z={mm(case_hi.z):.2f}mm != case_half_height_mm "
        f"{half_h_mm}")

    assert math.isclose(lid_lo.z, case_hi.z, abs_tol=1e-4), (
        f"lid base z={lid_lo.z * 1000:.3f}mm does not sit exactly on the "
        f"case's own measured rim z={case_hi.z * 1000:.3f}mm")
    assert math.isclose(mm(lid_lo.x), ox0, abs_tol=tol) \
        and math.isclose(mm(lid_lo.y), oy0, abs_tol=tol) \
        and math.isclose(mm(lid_hi.x), ox1, abs_tol=tol) \
        and math.isclose(mm(lid_hi.y), oy1, abs_tol=tol), (
        f"lid footprint ({mm(lid_lo.x):.2f},{mm(lid_lo.y):.2f})-"
        f"({mm(lid_hi.x):.2f},{mm(lid_hi.y):.2f})mm != case_outer_mm "
        f"{entry['case_outer_mm']}")

    diam_mm, len_mm = (v * 1000.0 for v in CELL_FORMATS[entry["cell_format"]])
    got_cross = sorted((mm(cell_hi.x - cell_lo.x), mm(cell_hi.z - cell_lo.z)))
    assert all(math.isclose(g, diam_mm, abs_tol=tol) for g in got_cross), (
        f"cell cross-section {got_cross}mm != {entry['cell_format']!r} "
        f"diameter {diam_mm:.2f}mm - is the cylinder really resting on "
        f"its side?")
    assert math.isclose(mm(cell_hi.y - cell_lo.y), len_mm, abs_tol=tol), (
        f"cell length {mm(cell_hi.y - cell_lo.y):.2f}mm != "
        f"{entry['cell_format']!r} length {len_mm:.2f}mm")

    assert math.isclose(mm(cell_lo.z), floor_mm, abs_tol=tol), (
        f"cell rests at z={mm(cell_lo.z):.2f}mm, not the cavity floor "
        f"tray_floor_mm={floor_mm}mm - a cell parked off the floor would "
        f"silently skew _load_template's shared re-centre step (see "
        f"build_procedural_tray's own docstring)")
    measured_depth = mm(case_hi.z) - mm(cell_lo.z)
    expected_depth = half_h_mm - floor_mm
    assert math.isclose(measured_depth, expected_depth, abs_tol=tol), (
        f"measured rim-to-floor depth {measured_depth:.2f}mm != the "
        f"sampled bay depth (case_half_height_mm - tray_floor_mm) = "
        f"{expected_depth:.2f}mm")
    assert case_lo.z - 1e-6 <= cell_lo.z, (
        "cell sits below the tray's own base - it would poke through the "
        "floor")
    # The OPEN cavity's own rim (half_h_mm) is not the cell's ceiling: the
    # 'assembled' variant keeps case + lid TOGETHER (bay.sample_tray
    # guarantees 2*half_h_mm clears the drawn cell's diameter with a
    # minimum clearance - see MIN_CELL_HEIGHT_CLEARANCE_MM), and it is
    # that combined envelope the cell must not poke through, not the bare
    # case half on its own (a cell resting on the open cavity's floor is
    # EXPECTED to rise above the open rim - that is what makes the cavity
    # useful headroom for a real, unsealed cell in open_case rendering,
    # which discards and independently re-seats this template cell
    # entirely; see assets.AssetLibrary.instantiate's exploded branch).
    assembled_top_mm = 2.0 * half_h_mm
    assert mm(cell_hi.z) <= assembled_top_mm + tol, (
        f"cell top z={mm(cell_hi.z):.2f}mm pokes above the assembled "
        f"case+lid envelope ({assembled_top_mm:.2f}mm = 2 * "
        f"case_half_height_mm) - it would not fit inside the sealed "
        f"'assembled' shell")

    # ---- 6/7/8. the lid's crown -------------------------------------- #
    #
    # `_crown_lid` is bpy-only and unreachable from pytest, so these three
    # are its entire evidence. They run on EVERY tray built, not in a test.
    crown_mm = entry["lid_crown_mm"]
    assert 0.0 <= crown_mm <= half_h_mm + tol, (
        f"lid_crown_mm ({crown_mm}) is not within [0, case_half_height_mm="
        f"{half_h_mm}] - bay.sample_tray's clamp did not reach the entry")

    # 6. The crown must not have moved the lid's AABB. Bevelling a cube's
    #    top edges leaves every face CENTRE where it was, so the footprint
    #    check above and this height check must both still bind exactly -
    #    if either moves, the bevel ran on the wrong edges.
    assert math.isclose(mm(lid_hi.z), assembled_top_mm, abs_tol=tol), (
        f"lid top z={mm(lid_hi.z):.2f}mm != 2 * case_half_height_mm "
        f"({assembled_top_mm:.2f}mm) - the crown moved the lid's own "
        f"height, which it must not")

    top_z = max(v.z for v in (lid.matrix_world @ v.co for v in lid.data.vertices))
    plateau = [lid.matrix_world @ v.co for v in lid.data.vertices
               if abs((lid.matrix_world @ v.co).z - top_z) <= 1e-6]
    px = mm(max(p.x for p in plateau) - min(p.x for p in plateau))
    py = mm(max(p.y for p in plateau) - min(p.y for p in plateau))
    upward = [p for p in lid.data.polygons if p.normal.z > 0.05]
    rolled = sum(1 for p in upward if p.normal.z < 0.95)

    if crown_mm <= 0.0:
        # 7. A zero crown must reproduce the pre-2026-08-11 lid EXACTLY -
        #    six faces, none of them rolled. This is what lets the crowned
        #    render be attributed to the crown and to nothing else.
        assert len(lid.data.polygons) == 6 and rolled == 0, (
            f"lid_crown_mm is 0 but the lid has {len(lid.data.polygons)} "
            f"polygons ({rolled} rolled), not the plain 6-face cuboid a "
            f"zero crown must leave untouched")
    else:
        # 8. The crown is REAL and has the sampled radius: the top plateau
        #    is inset by exactly `crown_mm` on all four sides, and the roll
        #    actually produced non-planar upward-facing geometry (the
        #    measured 89%-of-faces property the four Anker lids have and
        #    the flat procedural lid did not).
        want_x, want_y = (ox1 - ox0) - 2 * crown_mm, (oy1 - oy0) - 2 * crown_mm
        assert math.isclose(px, want_x, abs_tol=tol) \
            and math.isclose(py, want_y, abs_tol=tol), (
            f"crowned lid top plateau {px:.2f}x{py:.2f}mm != "
            f"{want_x:.2f}x{want_y:.2f}mm (case_outer inset by "
            f"lid_crown_mm={crown_mm}) - the bevel did not roll the "
            f"radius it was given")
        # Stated in the SAME terms as the measurement that motivated the
        # crown: 89% of each Anker lid's upward-facing polygons have a
        # z-normal below 0.95, against the planar procedural lid's 0%.
        # A fraction, not a face count - the first version of this
        # assertion demanded 4 * _CROWN_SEGMENTS and fired on the smoke
        # render at 40 of 49, because the one or two segments nearest the
        # plateau are within 0.95 of vertical by construction (a
        # 12-segment quarter-round steps 7.5 degrees at a time). The count
        # is also independent of the radius, so only the fraction says
        # anything about whether the roll is real.
        assert len(upward) and rolled >= 0.5 * len(upward) \
            and rolled >= 2 * _CROWN_SEGMENTS, (
            f"crowned lid has {rolled} non-planar upward-facing faces of "
            f"{len(upward)}; a genuine {_CROWN_SEGMENTS}-segment roll on "
            f"four edges leaves the majority of them rolled - the lid is "
            f"still effectively flat, which is the exact defect this "
            f"crown exists to remove")


# Segments in the lid's crown fillet. 12 puts each facet under a pixel at
# the generator's framing (a cartridge spans ~100-250 px and the crown
# covers at most 45% of its short side), so the roll reads as a smooth
# gradient rather than as banding once the bevel faces are shade-smoothed.
_CROWN_SEGMENTS = 12


def _crown_lid(lid, crown_m: float) -> None:
    """Roll a fillet of radius `crown_m` (metres) onto the lid's four TOP
    edges, in place. A no-op at 0.0, which is the pre-2026-08-11 geometry
    and the default every existing config still samples.

    Why the lid and only the lid: `config.VARIANTS` keeps `case_lid` in
    the `assembled` variant and drops it from `open_case`, so this touches
    the SEALED population and nothing else. An open procedural unit's
    pixels are unchanged by construction, which is what makes any move in
    present-only `bay` interpretable rather than confounded.

    Why the top edges and not all twelve: the lid's base has to stay flush
    on the case's own rim (`_assert_procedural_tray_geometry` check 4
    measures exactly that), and its four vertical edges are the sealed
    unit's silhouette, which the diagnosis already showed is not where the
    failure lives.

    Bevelling a cube's top edges leaves the AABB alone - the face centres
    do not move - so every existing footprint/height assertion still binds
    unchanged. `_assert_procedural_tray_geometry` adds three more that
    bind on THIS function's output specifically (see its checks 6-8),
    because nothing here is reachable from pytest.
    """
    if crown_m <= 0.0:
        return
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(lid.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    eps = 1e-6
    top_z = max(v.co.z for v in bm.verts)
    top_verts = [v for v in bm.verts if abs(v.co.z - top_z) <= eps]
    top_edges = [e for e in bm.edges
                 if all(abs(v.co.z - top_z) <= eps for v in e.verts)]
    # A freshly-added cube has exactly 8 verts and 12 edges, 4 of each on
    # the top face. Anything else means this function is being handed
    # something other than build_procedural_tray's own primitive, and
    # bevelling it would silently produce a shape nothing has measured.
    assert len(top_verts) == 4 and len(top_edges) == 4, (
        f"_crown_lid expected a 4-vert/4-edge top face on the lid "
        f"primitive, measured {len(top_verts)}/{len(top_edges)}")

    try:
        bmesh.ops.bevel(bm, geom=top_edges + top_verts, offset=crown_m,
                        offset_type="OFFSET", segments=_CROWN_SEGMENTS,
                        profile=0.5, affect="EDGES")
    except TypeError:               # pragma: no cover - older bmesh API
        bmesh.ops.bevel(bm, geom=top_edges + top_verts, offset=crown_m,
                        offset_type="OFFSET", segments=_CROWN_SEGMENTS,
                        profile=0.5)
    bm.to_mesh(lid.data)
    bm.free()

    # Smooth ONLY the crown. Shading the whole object smooth would round
    # the four vertical edges into the sides and turn a moulded slab into
    # a melted one; leaving the crown flat-shaded would render 12 hard
    # specular steps where a real fillet gives one continuous sweep.
    # Per-polygon `use_smooth` is the version-stable way to say that (the
    # auto-smooth operator/flag has moved twice across 4.x/5.x). The six
    # original faces are exactly the axis-aligned ones.
    for poly in lid.data.polygons:
        n = poly.normal
        poly.use_smooth = max(abs(n.x), abs(n.y), abs(n.z)) < 0.999
    lid.data.update()


def build_procedural_tray(entry: dict) -> Dict[str, list]:
    """Bare boolean-cut geometry for one procedural tray: `case` (a
    boolean-differenced shell), `case_lid` (a solid slab, same
    footprint), and `cell` (one cylinder at the sampled format's
    radius/length) - structurally identical to build_jig: every number
    comes from `entry` (bay.sample_tray + catalog.build_tray_entry). No
    geometric judgement happens here (design spec Sec3.4).

    No lay_flat, no flip_if_inverted: those exist ONLY to correct an
    imported CAD file's own up-axis/orientation ambiguity (design spec
    Sec2). A primitive built directly in Blender has neither to correct -
    `_load_template`'s shared re-centre step (unchanged, run on whatever
    this function returns) is what establishes the (0,0)-centred local
    frame bay.py's module docstring requires, exactly as it already does
    for CAD.

    Object names (`ProcCase_btm`/`ProcCase_top`/`ProcCell_0`) are chosen
    to satisfy catalog.CLASS_RULES' existing regexes - see
    tests/test_bay.py's `test_procedural_object_names_classify_correctly_
    via_role_of` (Task 7) for the pinned contract. Get these names wrong
    and `_load_template`'s shared tail silently tags the lid as `case`
    too, re-closing every open procedural cartridge.

    Before returning, `_assert_procedural_tray_geometry` numerically
    verifies the geometry just built against `entry`'s own numbers - see
    that function's docstring for exactly what it checks and why. This
    runs on EVERY procedural tray this function ever builds, not just in
    a test: a render that looks right is not evidence, on this project's
    own prior track record (see this plan's Task 8 note).

    Returns a `by_role` dict shaped exactly like `_load_template`'s
    existing CAD return value.
    """
    x0, y0, x1, y1 = entry["case_outer_mm"]
    w_m, h_m = (x1 - x0) / 1000.0, (y1 - y0) / 1000.0
    cx_m, cy_m = (x0 + x1) / 2000.0, (y0 + y1) / 2000.0
    half_h = entry["case_half_height_mm"] / 1000.0
    wall = entry["case_wall_mm"] / 1000.0
    floor = entry["tray_floor_mm"] / 1000.0

    # --- case: solid block, then a cavity boolean-differenced out of it -
    # exactly build_jig's own cube-plus-cutter shape.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx_m, cy_m, half_h / 2))
    case = bpy.context.active_object
    case.name = "ProcCase_btm"
    case.scale = (w_m, h_m, half_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    cav_w, cav_h = w_m - 2 * wall, h_m - 2 * wall
    cav_z_h = (half_h - floor) + 0.001     # +1mm so the cutter clears the open top
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(cx_m, cy_m, floor + cav_z_h / 2))
    cutter = bpy.context.active_object
    cutter.name = "_proc_cavity_cutter"
    cutter.scale = (cav_w, cav_h, cav_z_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mod = case.modifiers.new(name="cavity", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    bpy.context.view_layer.objects.active = case
    bpy.ops.object.modifier_apply(modifier="cavity")
    bpy.data.objects.remove(cutter, do_unlink=True)

    # --- case_lid: a solid slab, no boolean op needed - resting exactly
    # on the shell's own top rim (design spec Sec4.4's two-piece split).
    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(cx_m, cy_m, half_h + half_h / 2))
    lid = bpy.context.active_object
    lid.name = "ProcCase_top"
    lid.scale = (w_m, h_m, half_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _crown_lid(lid, entry["lid_crown_mm"] / 1000.0)

    # --- cell: one cylinder template at the sampled format's own
    # radius/length, resting ON THE CAVITY FLOOR - not parked outside the
    # case/lid footprint. _load_template's shared re-centre step measures
    # group_bbox(meshes) over every object this function returns,
    # INCLUDING this one; a cell parked outside the case/lid's own
    # footprint or height would silently skew that measurement and
    # mis-centre case and case_lid too, with no error anywhere short of
    # Task 9's dedicated assertion.
    diam_m, len_m = CELL_FORMATS[entry["cell_format"]]
    bpy.ops.mesh.primitive_cylinder_add(
        radius=diam_m / 2, depth=len_m,
        location=(cx_m, cy_m, floor + diam_m / 2))
    cell = bpy.context.active_object
    cell.name = "ProcCell_0"
    # primitive_cylinder_add's own axis is Z (standing on end); rotate 90
    # degrees about X so it rests on its SIDE, long axis horizontal - the
    # same resting pose lay_flat gives every imported CAD cell.
    cell.rotation_euler = (math.radians(90), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bpy.context.view_layer.update()
    _assert_procedural_tray_geometry(entry, case, lid, cell)
    return {"case": [case], "case_lid": [lid], "cell": [cell]}


# --------------------------------------------------------------------------- #
#  the seating ladder, checked at build time
#
#  `bay.SEATING_LADDER` decides every height in the bay and
#  `bay.assert_seating_ladder_ordered` re-checks the table on every
#  `bay.seat_z` call, so a perturbed offset raises there. That does not help
#  if THIS module stops asking and goes back to a literal - which is precisely
#  what it used to do, with the ordering restated in three docstrings and
#  enforced nowhere. So the two helpers below re-derive the proxy plane from
#  the ladder and check the number that actually reached the geometry, in the
#  same spirit as `_assert_procedural_tray_geometry` and
#  `_assert_seat_cell_footprint`: assert the built thing, not the input.
# --------------------------------------------------------------------------- #

def _assert_clears_bay_proxy(what: str, floor_z: float, z: float) -> None:
    """Assert `what` is seated strictly above the plane carrying the
    `placement_area` label, and so occludes it.

    Anything at or below the proxy loses the z-fight, or loses outright, and
    `annotate.boxes_from_mask` goes on reporting that floor as free while the
    object visibly sits on it. There is no downstream check for this: the
    render, the index pass and the manifest all agree with each other and all
    three are wrong. Runs on every obstruction and every seated cell built.
    """
    proxy_z = bay.seat_z(floor_z, bay.BAY_PROXY_RUNG)
    assert z > proxy_z, (
        f"{what} is seated at z={z:.6f} on a cavity floor of {floor_z:.6f}, "
        f"which is not strictly above the bay proxy plane at {proxy_z:.6f}. "
        f"It would not occlude the proxy, so placement_area would keep "
        f"claiming that floor is free while this object sits on it - "
        f"silently, in the render and the manifest alike. Heights come from "
        f"bay.seat_z; a literal here is how this drifts.")


def _assert_board_straddles_bay_proxy(floor_z: float, z: float,
                                      thickness: float) -> None:
    """Assert the module board rests ON the cavity floor and its top face
    clears the bay proxy.

    The board is the one thing in the bay seated BELOW the proxy, and that is
    deliberate: it is built about its own centre and has to rest on the floor
    like the cells do, so it occludes through its thickness rather than
    through its seat. Both halves of that have to hold - a board floating
    above the floor is visible from overhead, and a board thinner than the
    proxy's own offset would be occluded BY the placement area it borders.
    """
    proxy_z = bay.seat_z(floor_z, bay.BAY_PROXY_RUNG)
    assert math.isclose(z - thickness / 2.0, floor_z, abs_tol=1e-12), (
        f"the module board's underside is at {z - thickness / 2.0:.6f}, not "
        f"on the cavity floor at {floor_z:.6f}: a {thickness:.4f}m board "
        f"built about its centre rests on the floor only when it is seated "
        f"at floor + {thickness / 2.0:.4f}")
    assert z + thickness / 2.0 > proxy_z, (
        f"the module board's top face at {z + thickness / 2.0:.6f} does not "
        f"clear the bay proxy at {proxy_z:.6f}, so along the edge the board "
        f"and the placement rectangle share, the proxy occludes the board "
        f"rather than the other way round")


def build_pcb(bounds_xy, floor_z: float, rng: random.Random,
             module_placement=None, rot_deg: float = 0.0):
    """
    Green PCB with extruded components, seated in the tray's cavity, for the
    open_case variant.

    The CAD has no PCB part, but it reserves the space: every assembly
    leaves a 19-31mm-deep strip on one short side of the cavity
    (catalog.json's module_bay_mm, inset from the wall). `module_placement`
    is `(cx, cy, w, h)` - that strip's TRUE centre and size in world metres,
    from `bay.module_world_placement`. Pass it and the board lands where the
    hardware puts it, at its real, un-inflated size regardless of how the
    cartridge is rotated. `rot_deg` is the SAME `layout.Placement.rot_deg`
    the cartridge itself was placed with; the board (and everything
    parented to it) is rotated to match at the end of this function, via
    `matrix_world` rather than `rotation_euler` for the same reason
    `assets.place_item` does - see its own comment.

    Omit `module_placement` and the board is centred instead, from
    `bay.fallback_module_placement` - which draws that rectangle on the
    bpy-free side, beside the anchored path's `bay.module_world_placement`,
    rather than here. `rot_deg` still applies in this path, so the centred
    fallback also turns with the case rather than staying axis-aligned
    against a rotated one.

    The board's own Z is `bay.seat_z(floor_z, "pcb")` and its thickness
    `bay.PCB_THICKNESS_M`; the two are coupled (a board built about its
    centre rests on the floor only at half its thickness) and
    `_assert_board_straddles_bay_proxy` checks that coupling on every
    build. See `bay`'s SEATING LADDER section for why the board is the one
    thing in the bay seated BELOW the plane carrying `placement_area`.

    `floor_z` is the CAVITY FLOOR (catalog.json's tray_floor_mm, converted
    to metres by scene.py), not the assembly's top. Until task 3 this took
    the assembly's `hi.z` - the outer surface of a closed lid, since the
    lid had not yet been split into its own role and dropped - so the
    board was drawn on the outside of a shut box. The cells rest on this
    same floor in the CAD, which is where the measurement comes from.

    Returns (board_object, drawn_meta). The caller assigns pass_index.
    """
    if module_placement is not None:
        cx, cy, w, h = module_placement
    else:
        cx, cy, w, h = bay.fallback_module_placement(bounds_xy, rng)

    board_z = bay.seat_z(floor_z, "pcb")
    _assert_board_straddles_bay_proxy(floor_z, board_z, bay.PCB_THICKNESS_M)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, board_z))
    board = bpy.context.active_object
    board.name = "PCB"
    board.scale = (w, h, bay.PCB_THICKNESS_M)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    board.pass_index = 0          # scene.py overrides this

    drawn = {"w": w, "h": h,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.16, 0.30),
                       rng.uniform(0.04, 0.10)],
             "roughness": rng.uniform(0.25, 0.50),
             "n_components": rng.randint(3, 7),
             "anchored": module_placement is not None, "rot_deg": rot_deg}

    mat = bpy.data.materials.new("PCBGreen")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    _set_recorded(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    _set_recorded(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 0.15)
    board.data.materials.append(mat)

    comp_mat = bpy.data.materials.new("PCBComponent")
    comp_mat.use_nodes = True
    cb = comp_mat.node_tree.nodes.get("Principled BSDF")
    set_input(cb, "Base Color", (0.05, 0.05, 0.055, 1.0))
    set_input(cb, "Roughness", 0.55)

    # Everything mounted on the board sits on its TOP face. The board rests
    # on the cavity floor (see _assert_board_straddles_bay_proxy), so that is
    # the floor plus one board thickness - NOT another rung of the seating
    # ladder: these are relative to the board, and move with it.
    board_top_z = floor_z + bay.PCB_THICKNESS_M

    for k in range(drawn["n_components"]):
        cw = w * rng.uniform(0.06, 0.20)
        ch = h * rng.uniform(0.15, 0.45)
        cz = rng.uniform(0.0015, 0.0045)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx + rng.uniform(-w / 2 + cw, w / 2 - cw),
                      cy + rng.uniform(-h / 2 + ch, h / 2 - ch),
                      board_top_z + cz / 2))
        c = bpy.context.active_object
        c.name = f"PCBComp{k}"
        c.scale = (cw, ch, cz)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        c.pass_index = 0
        c.data.materials.append(comp_mat)
        # Parent for tidiness, but KEEP the world transform. A bare
        # `c.parent = board` leaves matrix_parent_inverse at identity, so the
        # child's world matrix becomes parent.matrix_world @ child.matrix_basis
        # and the component jumps by the board's OWN translation. Measured on
        # 5.0.0 with a board at (0.30, 0.15, 0.05): a component built at
        # (0.32, 0.16, 0.06) ended up at (0.62, 0.31, 0.11), 339mm away and
        # nowhere near the board. Components carry pass_index 0, so strays would
        # occlude LABELLED parts and shrink their boxes with no audit trail
        # whatsoever. Setting the parent inverse pins them where they were built
        # (measured displacement 0.0).
        c.parent = board
        c.matrix_parent_inverse = board.matrix_world.inverted()

    # Gold USB shells along the board's outward edge, and one copper
    # inductor. These are the module's two most recognisable features in
    # the real photographs and the dark cuboids above have neither.
    gold = bpy.data.materials.new("PCBGold")
    gold.use_nodes = True
    gb = gold.node_tree.nodes.get("Principled BSDF")
    set_input(gb, "Base Color", (0.75, 0.60, 0.22, 1.0))
    set_input(gb, "Metallic", 1.0)
    set_input(gb, "Roughness", rng.uniform(0.20, 0.35))

    n_ports = rng.randint(1, 4)
    port_w, port_h, port_z = 0.013, 0.006, 0.005
    for k in range(n_ports):
        span = n_ports * port_w * 1.3
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx - span / 2 + port_w * 1.3 * (k + 0.5),
                      cy + h / 2 - port_h,
                      board_top_z + port_z / 2))
        p = bpy.context.active_object
        p.name = f"PCBPort{k}"
        p.scale = (port_w, port_h, port_z)
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)
        p.pass_index = 0
        p.data.materials.append(gold)
        p.parent = board
        p.matrix_parent_inverse = board.matrix_world.inverted()

    copper = bpy.data.materials.new("PCBCopper")
    copper.use_nodes = True
    cb2 = copper.node_tree.nodes.get("Principled BSDF")
    set_input(cb2, "Base Color", (0.72, 0.35, 0.18, 1.0))
    set_input(cb2, "Metallic", 1.0)
    set_input(cb2, "Roughness", rng.uniform(0.35, 0.55))
    bpy.ops.mesh.primitive_cylinder_add(
        radius=rng.uniform(0.004, 0.007), depth=0.006,
        location=(cx + rng.uniform(-w / 4, w / 4),
                  cy + rng.uniform(-h / 4, h / 4), board_top_z + 0.003))
    ind = bpy.context.active_object
    ind.name = "PCBInductor"
    ind.pass_index = 0
    ind.data.materials.append(copper)
    ind.parent = board
    ind.matrix_parent_inverse = board.matrix_world.inverted()

    drawn["n_ports"] = n_ports

    # Rotate the board to match the cartridge's actual placement rotation.
    # The board (and its ports/inductor, built above at cx, cy with no
    # rotation of their own) was built axis-aligned; for a 90/180/270
    # degree turn - or the few degrees of jitter layout.plan adds on top
    # of every turn - an axis-aligned board would sit crosswise across a
    # turned case. This rotates ABOUT (cx, cy, z), the board's own centre
    # (also its own origin, since the cube was built there), so its
    # position from module_placement is preserved and only its
    # orientation changes. Ports and the inductor are parented to `board`
    # with matrix_parent_inverse already set, so they turn rigidly with
    # it through the normal parent/child transform - no separate rotation
    # needed for them.
    if rot_deg:
        pivot = Vector((cx, cy, 0.0))
        R = Matrix.Rotation(math.radians(rot_deg), 4, "Z")
        board.matrix_world = (Matrix.Translation(pivot) @ R
                              @ Matrix.Translation(-pivot) @ board.matrix_world)

    return board, drawn


def build_bay_proxy(placement, floor_z: float, rng: random.Random,
                    rot_deg: float = 0.0):
    """
    The plane that carries the `placement_area` label.

    Seated on the tray's cavity floor (`floor_z`, catalog.json's
    tray_floor_mm converted to metres by scene.py) - the real floor a cell
    rests on inside the tray, not the assembly's outer top.

    `placement` is `(cx, cy, w, h)` - the placement rectangle's TRUE centre
    and size in world metres, from `bay.placement_world_placement`. Same
    contract and same reason as `build_pcb`'s `module_placement`: computed
    in the cartridge's own local frame and rotated as a POINT rather than
    lerped into a rotated world AABB, so the proxy is exact at any
    `layout.plan` jitter angle instead of oversized by rotation - see
    `bay.module_world_placement`'s docstring for the measured 7.6%-at-2-
    degrees AABB error this sidesteps. `rot_deg` is the SAME
    `layout.Placement.rot_deg` the cartridge (and the module board) were
    placed with; the plane is built axis-aligned at `(cx, cy)` and then
    rotated about that same point via `matrix_world`, exactly as
    `build_pcb` rotates the board - so the proxy turns rigidly with the
    case and keeps sharing its edge with the module rectangle rather than
    drifting apart under rotation.

    Sits at `bay.seat_z(floor_z, "bay_proxy")` - the `bay_proxy` rung of
    `bay.SEATING_LADDER`, 0.9mm above the cavity floor, only 0.1mm above
    the board's own seat - so it never z-fights with the tray or the board.
    This is the rung every OTHER object placed in the bay - a seated cell,
    an obstruction, anything added later - must clear: it has to rest
    strictly ABOVE this plane to occlude it at all, which is what
    `bay.assert_seating_ladder_ordered` and `_assert_clears_bay_proxy`
    between them require. Something placed at or below it loses the
    z-fight (or loses outright), and
    `annotate.boxes_from_mask`/`masks_from_index` would keep reporting
    that floor as free placement area while the object sits visibly on top
    of it - silently, since nothing downstream can detect the failure from
    the mask alone. It is DELIBERATELY a visible, shaded object rather
    than an index-only helper: the placement area has to look like the
    inside of a cartridge, because the segmenter is asked to recognise it
    from appearance, not from a hidden id alone.

    Occlusion does the rest. A cell seated on this plane, an obstruction
    stuck to it, or the module beside it all hide their own pixels, and
    `annotate.boxes_from_mask` reports only what stayed visible - so the
    label is the CURRENTLY FREE floor with no set arithmetic anywhere.

    Returns (proxy_object, drawn_meta). The caller assigns pass_index.
    """
    cx, cy, w, h = placement
    bpy.ops.mesh.primitive_plane_add(
        size=1, location=(cx, cy, bay.seat_z(floor_z, bay.BAY_PROXY_RUNG)))
    proxy = bpy.context.active_object
    proxy.name = "BayProxy"
    proxy.scale = (w, h, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    proxy.pass_index = 0          # scene.py overrides

    drawn = {"w": w, "h": h,
             "color": [rng.uniform(0.02, 0.05)] * 3,
             "roughness": rng.uniform(0.55, 0.85), "rot_deg": rot_deg}

    mat = bpy.data.materials.new("BayFloor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    _set_recorded(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    _set_recorded(bsdf, "Roughness", drawn["roughness"])
    proxy.data.materials.append(mat)

    if rot_deg:
        pivot = Vector((cx, cy, 0.0))
        R = Matrix.Rotation(math.radians(rot_deg), 4, "Z")
        proxy.matrix_world = (Matrix.Translation(pivot) @ R
                              @ Matrix.Translation(-pivot) @ proxy.matrix_world)

    return proxy, drawn


def build_obstructions(poses, floor_z: float, rng: random.Random):
    """Realise `bay.ObstructionPose`s as geometry. Returns [(obj, meta), ...].

    `poses` are already in WORLD space - `bay.obstruction_world_poses` has
    rotated and translated them to match the cartridge's actual placement,
    so this function only has to build shapes at the given (x, y) and spin
    them by their own `rot_deg`; no pivot rotation is needed here the way
    `build_pcb`/`build_bay_proxy` need one, because that composition already
    happened on the bpy-free side.

    `floor_z` is the tray's cavity floor (catalog.json's tray_floor_mm,
    converted to metres by scene.py), the same surface `build_bay_proxy`
    seats its plane on. Each kind's seat and its Z scale both come from
    `bay` - `seat_z` off the seating ladder and `obstruction_z_scale`
    beside `sample_obstructions`, which already decided w and h - so this
    function chooses a primitive and a surface and nothing else. Every
    seat is checked against the proxy's own rung as it is built: an
    obstruction at or below it would lose the z-fight and `placement_area`
    would keep reporting that floor as free.

    Each gets its own pass_index from scene.py, so each is an instance.
    They sit ON the bay proxy and therefore occlude it, which is what makes
    placement_area the currently-free floor rather than the nominal one.
    """
    made = []
    for i, p in enumerate(poses):
        # Drawn BEFORE the shape dispatch below, and per kind: adhesive and
        # foam consume exactly one draw each, tape and label none. That is
        # `obstruction_z_scale`'s contract, and it is what keeps this scene's
        # shared rng stream identical to the literals this replaced.
        z_scale = bay.obstruction_z_scale(p.kind, rng)
        seat = bay.seat_z(floor_z, p.kind)
        _assert_clears_bay_proxy(f"obstruction {i} ({p.kind})", floor_z, seat)

        if p.kind == "adhesive":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=p.w / 2, location=(p.x, p.y, seat))
            o = bpy.context.active_object
            o.scale = (1.0, p.h / max(p.w, 1e-9), z_scale)
            base, rough, alpha = (0.92, 0.92, 0.90), 0.30, 0.85
        elif p.kind == "foam":
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(p.x, p.y, seat))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, z_scale)
            base, rough, alpha = (0.88, 0.88, 0.86), 0.95, 1.0
        elif p.kind == "tape":
            bpy.ops.mesh.primitive_plane_add(
                size=1, location=(p.x, p.y, seat))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, z_scale)
            base, rough, alpha = (0.95, 0.95, 0.93), 0.45, 1.0
        else:  # label
            bpy.ops.mesh.primitive_plane_add(
                size=1, location=(p.x, p.y, seat))
            o = bpy.context.active_object
            o.scale = (p.w, p.h, z_scale)
            base, rough, alpha = (0.90, 0.89, 0.84), 0.70, 1.0

        o.name = f"Obs{i}_{p.kind}"
        o.rotation_euler = (0.0, 0.0, math.radians(p.rot_deg))
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True)
        o.pass_index = 0          # scene.py overrides

        mat = bpy.data.materials.new(f"Obs{p.kind}")
        mat.use_nodes = True
        b = mat.node_tree.nodes.get("Principled BSDF")
        set_input(b, "Base Color", base + (1.0,))
        set_input(b, "Roughness", rough)
        if alpha < 1.0:
            set_input(b, "Alpha", alpha)
            mat.blend_method = "BLEND"
        o.data.materials.append(mat)
        made.append((o, {"kind": p.kind, "w": p.w, "h": p.h,
                         "rot_deg": p.rot_deg}))
    return made


# --------------------------------------------------------------------------- #
#  seated cells
#
#  The deployed robot fills a cartridge one cell at a time, so its camera
#  sees PARTLY-FILLED bays for most of every run. `bay.seated_cell_poses`
#  draws where the packer itself would put them; this realises those poses
#  as the asset's own 18650 geometry, cloned rather than a new cylinder, so
#  a seated cell matches the loose cells the detector already sees.
# --------------------------------------------------------------------------- #

# SEATED_CELL_LIFT is retired as a world.py constant: it is the
# `seated_cell` rung of `bay.SEATING_LADDER`, reached through
# `bay.seat_z(floor_z, "seated_cell")` like every other height in the bay.
# It sat here, as a literal, while bay.py decided the cell's footprint - the
# XY/Z split this module used to be on the wrong side of. Its value and its
# reason are unchanged: a seated cell's bottom has to sit strictly ABOVE the
# proxy that carries the placement_area label, because
# annotate.boxes_from_mask reports only what stayed VISIBLE in the id pass,
# and a cell at or below the proxy would leave placement_area claiming that
# floor is free while a cell visibly occupies it.

# SEAT_CELL_FOOTPRINT_M is retired: it named one hardcoded format. Every
# call site now looks up CELL_FORMATS[cell_format] directly, keyed by
# whichever format the item being seated actually is (2026-08-10,
# spec #2 cell-format generalisation).

_seat_cell_footprint_checked: set = set()


def _assert_seat_cell_footprint(asset: str, cell_format: str,
                                lo: Vector, hi: Vector) -> None:
    """Assert a freshly lay_flat'd (or, for a procedural tray, directly
    built) cell clone's measured XY footprint matches
    CELL_FORMATS[cell_format], once per (asset, cell_format).

    Generalised from a single hardcoded 18650 check (design spec Sec6.1):
    it now asserts against the format the clone was DRAWN FOR, not a
    single global constant - so a mismatch between what scene.py believes
    an item's format is and what its actual template measures still
    fails loudly, for every format, not just 18650.
    """
    key = (asset, cell_format)
    if key in _seat_cell_footprint_checked:
        return
    _seat_cell_footprint_checked.add(key)
    got = sorted((hi.x - lo.x, hi.y - lo.y))
    want = sorted(CELL_FORMATS[cell_format])
    assert all(math.isclose(g, w, abs_tol=5e-4) for g, w in zip(got, want)), (
        f"{asset}'s seated-cell template measures "
        f"{got[0] * 1000:.2f}x{got[1] * 1000:.2f}mm after lay_flat, not the "
        f"{want[0] * 1000:.2f}x{want[1] * 1000:.2f}mm {cell_format!r} "
        f"CELL_FORMATS entry - scene.py's packer call already sized and "
        f"placed this cell's seat against the assumed figure, so the "
        f"packing and the rendered geometry have desynced.")


def seat_cells(library, asset: str, seats, floor_z: float, rng: random.Random,
               cfg, cell_format: str = "18650", backdrop_luma=None):
    """Clone the asset's own 18650 cell template into the bay, seated.

    `seats` are `(x, y, rot_deg)` WORLD centres from
    `bay.seated_cell_world_poses` - already rotated and translated to match
    the cartridge's actual placement - so this only has to clone, orient and
    position each cell; no pivot rotation for the cartridge's own turn is
    needed here, the same simplification `build_obstructions` makes for the
    same reason (that composition already happened on the bpy-free side).

    Cloned from `library._templates[asset]["cell"]` - the asset's own
    already-imported 18650 mesh - via `assets.clone`, rather than a new
    primitive, so a seated cell matches the loose ("battery") cells the
    detector already sees: same geometry, and the same
    `materials.for_role("cell", ...)` draw every other cell instance gets
    (a template object carries no material of its own - it is never
    rendered directly, only cloned - so this must draw and apply one
    explicitly, the same call `scene.build`'s per-item materials loop makes
    for every other cell).

    Each clone is put through the SAME reset-then-lay_flat recipe
    `AssetLibrary.instantiate` uses to rest a standalone loose cell on its
    side (reset transform to identity, then `assets.lay_flat` on the single
    object) rather than inventing a new one - that recipe is what already
    correctly orients every loose cell in the dataset, and a seated cell has
    to end up in the exact same resting orientation.

    Rests each cell's bottom at `bay.seat_z(floor_z, "seated_cell")`,
    strictly above the bay proxy's own rung (`build_bay_proxy`) - see
    `bay`'s SEATING LADDER section for why that separation is the entire
    mechanism this function exists for, and `_assert_clears_bay_proxy`
    below for it being required of the seat actually used, not just of the
    table it came from. Occlusion does the rest:
    annotate.boxes_from_mask reports only what stayed visible in the id
    pass, so placement_area shrinks to the free floor around the seated
    cells with no mask arithmetic anywhere, the same trick
    build_obstructions already uses for foreign matter in the bay.

    Returns the list of new cell objects. The caller assigns pass_index.
    """
    if not seats:
        return []
    templates = (library._templates.get(asset) or {}).get("cell")
    if not templates:
        return []
    template = templates[0]

    seat = bay.seat_z(floor_z, "seated_cell")
    _assert_clears_bay_proxy("a seated cell", floor_z, seat)

    made = []
    for i, (x, y, rot_deg) in enumerate(seats):
        dup = A.clone(template)
        # Reset to identity and re-flatten the single object exactly as
        # AssetLibrary.instantiate does for a standalone loose cell - see
        # assets.lay_flat's docstring for why a bare rotation_euler write
        # would silently do nothing under the glTF importer's quaternion
        # rotation mode. This clone was just duplicated from a template
        # that carries the whole assembly's own lay_flat rotation baked in;
        # lay_flat has to measure THIS object's own raw extents, not the
        # assembly's, to rest it on its side correctly in isolation.
        dup.rotation_euler = (0.0, 0.0, 0.0)
        dup.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        A.lay_flat([dup])

        # Measure rather than assume where lay_flat left it, then translate
        # so the cell's own XY centroid lands at (x, y) and its bottom Z
        # lands exactly on the seat - the same drop-to-floor-then-lift
        # composition assets.place_item uses for a whole cartridge.
        lo, hi_obj = A.group_bbox([dup])
        _assert_seat_cell_footprint(asset, cell_format, lo, hi_obj)
        centre = Vector(((lo.x + hi_obj.x) / 2, (lo.y + hi_obj.y) / 2, 0.0))
        dup.location += Vector((x - centre.x, y - centre.y,
                               seat - lo.z))
        bpy.context.view_layer.update()

        # Spin about the cell's OWN centre (x, y) - a Z-axis rotation, so
        # the Z rest position just established is untouched - matching the
        # pivot-conjugated matrix_world composition assets.place_item and
        # build_pcb use for the same reason: a bare rotation_euler write
        # would be a silent no-op under the importer's quaternion mode.
        if rot_deg:
            pivot = Vector((x, y, 0.0))
            R = Matrix.Rotation(math.radians(rot_deg), 4, "Z")
            dup.matrix_world = (Matrix.Translation(pivot) @ R
                                @ Matrix.Translation(-pivot) @ dup.matrix_world)
            bpy.context.view_layer.update()

        mat, _drawn = for_role("cell", rng, cfg, avoid_luma=backdrop_luma)
        apply_to_object(dup, mat)

        dup.name = f"SeatedCell{i}"
        dup.pass_index = 0        # scene.py overrides
        made.append(dup)
    return made
