"""
recog.synth3d.world - backdrop, lighting, the overhead camera, and the two
pieces of UNLABELLED scene furniture (the jig plate and the PCB). Requires bpy.

Presets are not module globals: they are read off the loaded config, so
`build_backdrop` and `setup_lighting` take a `cfg` (a `config.Config`).

The jig plate and the PCB both carry `pass_index = 0` on purpose. They are
scene content, not classes: they must occlude correctly in the index pass but
must never produce an annotation. Keeping them at 0 is what makes that work -
`annotate.boxes_from_mask` skips id 0 as background, and any NON-zero id that
had no `id_meta` entry would be dropped silently and without an audit trail.
"""

from __future__ import annotations

import math
import os
import random

import bpy

from .materials import set_input, rng_range


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

def _procedural(nt, kind: str, rng: random.Random):
    """Returns (colour_socket, height_socket, mapping_node)."""
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
    palette = {
        "concrete": ((0.22, 0.22, 0.215, 1), (0.46, 0.455, 0.44, 1)),
        "brushed": ((0.30, 0.305, 0.31, 1), (0.52, 0.525, 0.535, 1)),
        "fabric": ((0.10, 0.12, 0.16, 1), (0.24, 0.27, 0.33, 1)),
        "paper": ((0.76, 0.75, 0.72, 1), (0.90, 0.89, 0.86, 1)),
        "belt": ((0.035, 0.035, 0.038, 1), (0.115, 0.115, 0.12, 1)),
    }.get(kind, ((0.2, 0.2, 0.2, 1), (0.5, 0.5, 0.5, 1)))
    ramp.color_ramp.elements[0].color = palette[0]
    ramp.color_ramp.elements[1].color = palette[1]
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
        color_out, height_out, mapping = _procedural(nt, spec["proc"], rng)

    mapping.inputs["Scale"].default_value = tuple(
        v * drawn["uv_scale"] for v in mapping.inputs["Scale"].default_value)
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(drawn["rot90"]))

    bright = nt.nodes.new("ShaderNodeBrightContrast")
    bright.inputs["Bright"].default_value = drawn["brightness"]
    nt.links.new(color_out, bright.inputs["Color"])
    nt.links.new(bright.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, "Roughness", drawn["roughness"])
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


def setup_camera(cfg, layout_cfg, res, rng: random.Random, top_z: float = 0.0):
    """
    Bird's-eye camera. A camera with zero rotation already looks down -Z, so a
    top-down view needs no aiming.

    Framing derives from the layout AREA, not from the objects, so scale stays
    consistent across the dataset - a power bank is the same number of pixels in
    every image, which is what detection training wants.

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

    if cfg.ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = need * margin
    else:
        cam.data.type = "PERSP"
        cam.data.lens = cfg.focal
        fov = 2 * math.atan(cam.data.sensor_width / (2 * cfg.focal))
        half = need / 2 * margin
        cam.location.z = top_z + half / math.tan(fov / 2)

    cam.data.clip_start, cam.data.clip_end = 0.001, 100.0
    return cam, {"margin": margin, "shift_x": shift_x, "shift_y": shift_y,
                 "ortho": cfg.ortho, "height": cfg.height,
                 "ortho_scale": getattr(cam.data, "ortho_scale", None)}


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
    set_input(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    set_input(bsdf, "Roughness", drawn["roughness"])
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


def build_pcb(bounds_xy, z: float, rng: random.Random):
    """
    Green PCB with a few extruded components, for the open_case variant.

    The CAD has no PCB, but it is the most distinctive thing inside an opened
    case in the real photos. `z` is the shell's TOP, so the board is laid on
    top of the shell rather than modelled inside it: from a bird's-eye camera
    the two read the same, and this needs no interior geometry. UNLABELLED
    (pass_index 0) - it is scene content, not a class, and it correctly
    shrinks the case's visible silhouette the way a real board would.
    """
    x0, y0, x1, y1 = bounds_xy
    w = (x1 - x0) * rng.uniform(0.55, 0.80)
    h = (y1 - y0) * rng.uniform(0.20, 0.38)
    cx = (x0 + x1) / 2 + rng.uniform(-0.004, 0.004)
    cy = (y0 + y1) / 2 + rng.uniform(-0.010, 0.010)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, z + 0.0008))
    board = bpy.context.active_object
    board.name = "PCB"
    board.scale = (w, h, 0.0016)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    board.pass_index = 0

    drawn = {"w": w, "h": h,
             "color": [rng.uniform(0.02, 0.06), rng.uniform(0.16, 0.30),
                       rng.uniform(0.04, 0.10)],
             "roughness": rng.uniform(0.25, 0.50),
             "n_components": rng.randint(3, 7)}

    mat = bpy.data.materials.new("PCBGreen")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", tuple(drawn["color"]) + (1.0,))
    set_input(bsdf, "Roughness", drawn["roughness"])
    set_input(bsdf, "Metallic", 0.15)
    board.data.materials.append(mat)

    comp_mat = bpy.data.materials.new("PCBComponent")
    comp_mat.use_nodes = True
    cb = comp_mat.node_tree.nodes.get("Principled BSDF")
    set_input(cb, "Base Color", (0.05, 0.05, 0.055, 1.0))
    set_input(cb, "Roughness", 0.55)

    for k in range(drawn["n_components"]):
        cw = w * rng.uniform(0.06, 0.20)
        ch = h * rng.uniform(0.15, 0.45)
        cz = rng.uniform(0.0015, 0.0045)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx + rng.uniform(-w / 2 + cw, w / 2 - cw),
                      cy + rng.uniform(-h / 2 + ch, h / 2 - ch),
                      z + 0.0016 + cz / 2))
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

    return board, drawn
