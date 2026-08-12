"""recog.synth3d.materials - randomized Principled surfaces. Requires bpy.

Presets are not module globals here: they come from the loaded config
(`config.load_config()`), so `build` and `for_role` take a `cfg` argument.

`for_role` draws the preset JOINTLY with the backdrop rather than
independently. Drawing them independently produced correctly labelled boxes
that a human cannot see: measured over 30 scenes, 15% of boxes differed from
their immediate surround by less than 0.05 display luminance, and the fifth
percentile was 0.018 - objects that are, to a detector, background wearing a
box. That teaches it to hallucinate. See `MIN_LUMA_DELTA`.
"""

from __future__ import annotations

import random
from typing import Tuple

import bpy


# Minimum separation, in display-referred luminance, between a material preset
# and the backdrop it will be seen against. Both sides come from `luma_ref` in
# configs/synth3d.yaml, which is MEASURED from renders rather than derived from
# the albedo - albedo does not predict the rendered value here, and not by a
# little. shell_navy (albedo 0.084) renders at 0.589 while cell_black (albedo
# 0.040) renders at 0.257: the clearcoat sheen on a flat shell facing an
# overhead softbox dominates its own base colour, and a cylinder's median pixel
# is dimmer than a flat top face of the same material. Any formula from albedo,
# metallic and coat got those two backwards, so the table is measured.
#
# 0.10 is chosen by measuring both ends of the trade. It takes the fraction of
# boxes within 0.05 luminance of their surround from 9.6% to 5.4% while leaving
# every (backdrop, role) combination at least two presets to draw from. 0.12
# scored no better (5.8%) and costs the `cell` role on fabric two of its four
# remaining presets; by 0.15 the `case` role on paper is down to shell_black
# alone, which swaps one bias (invisible objects) for a worse one (the backdrop
# predicting the object's colour).
MIN_LUMA_DELTA = 0.10


def set_input(node, name, value) -> bool:
    """Principled socket names moved in Blender 4.x; set only what exists."""
    if name in node.inputs:
        node.inputs[name].default_value = value
        return True
    return False


def socket_by_identifier(sockets, identifier):
    """A node socket addressed by its stable `identifier` - not name, not index.

    `ShaderNodeMix` is the node that replaced `ShaderNodeMixRGB` in Blender
    4.0, and it carries one typed A/B/Result triple per `data_type` behind a
    single interface. Every one of those triples is *named* "A"/"B"/"Result",
    and there are two sockets named "Factor". A name lookup on a bpy
    collection silently returns whichever duplicate comes first, and a
    positional index is the socket layout of one particular Blender build.
    The `identifier` - "Factor_Float", "A_Float", "B_Float", "Result_Float" -
    is the only key that actually says which typed socket is meant.

    Raises `KeyError` listing what the running build does offer, rather than
    returning None: a caller that cannot find the socket it asked for must
    stop, not quietly wire up a different one.
    """
    for s in sockets:
        if s.identifier == identifier:
            return s
    raise KeyError(
        f"no node socket with identifier {identifier!r}; this Blender build "
        f"offers {[s.identifier for s in sockets]}")


def rng_range(rng: random.Random, span: Tuple[float, float]) -> float:
    lo, hi = span
    return lo if lo == hi else rng.uniform(lo, hi)


def rng_color(rng: random.Random, span) -> Tuple[float, float, float, float]:
    lo, hi = span
    return tuple(rng.uniform(a, b) for a, b in zip(lo, hi)) + (1.0,)


# A bpy float socket stores single precision, so a value written from a
# Python float reads back with ~1e-7 relative error. This is a "did the write
# land at all" tolerance, not a physical one - the failure being guarded
# against leaves the socket at its 0.5 default or writes to a different
# socket entirely, both of which are orders of magnitude outside it.
_SOCKET_TOL = 1e-6


def _assert_wear_mix_took(mix, ramp, bsdf, drawn) -> None:
    """Verify the wear mix `build` just wired matches what `drawn` records.

    This module imports bpy, so pytest cannot reach it - `tests/
    test_synth3d.py` pins that boundary explicitly - and an assertion that
    runs on every material built is therefore the only evidence this code
    path can produce. That is the same treatment
    `world._assert_procedural_tray_geometry` gives the procedural tray, for
    the same reason: the failure it guards against renders plausibly and
    leaves a receipt saying the opposite, so a green test suite and a
    good-looking image both stay silent.

    The four checks, and the specific way each one can go wrong:

      1/2. The drawn `wear` and `roughness` landed on the FLOAT pair. A
           lookup that resolves to the Vector or Color pair instead - which
           is precisely what name- and index-based access to this node
           risks - writes somewhere harmless-looking and leaves the Float
           sockets at their defaults.
      3.   The noise ramp drives B rather than the shader directly. Linking
           the ramp into Roughness is the exact shape of the old fallback
           and the reason this function exists.
      4.   Principled's Roughness is driven by the mix RESULT. If it is fed
           by anything else, or by nothing, then `drawn["roughness"]` is not
           what renders while `meta["materials"]` says it is.
    """
    factor = socket_by_identifier(mix.inputs, "Factor_Float")
    a_in = socket_by_identifier(mix.inputs, "A_Float")
    b_in = socket_by_identifier(mix.inputs, "B_Float")
    result = socket_by_identifier(mix.outputs, "Result_Float")
    rough = bsdf.inputs["Roughness"]

    assert abs(factor.default_value - drawn["wear"]) < _SOCKET_TOL, (
        f"wear {drawn['wear']!r} did not reach the mix Factor_Float socket "
        f"(reads {factor.default_value!r}); meta['materials'] would record a "
        f"wear this render never used")
    assert abs(a_in.default_value - drawn["roughness"]) < _SOCKET_TOL, (
        f"roughness {drawn['roughness']!r} did not reach the mix A_Float "
        f"socket (reads {a_in.default_value!r}); meta['materials'] would "
        f"record a roughness this render never used")
    assert b_in.is_linked and b_in.links[0].from_node == ramp, (
        f"the wear noise ramp does not drive the mix B_Float socket "
        f"(linked={b_in.is_linked}); the surface would render uniformly "
        f"rough rather than worn")
    assert (rough.is_linked
            and rough.links[0].from_node == mix
            and rough.links[0].from_socket.identifier
            == result.identifier), (
        f"Principled BSDF Roughness is not driven by the mix result "
        f"(linked={rough.is_linked}); the drawn roughness/wear pair is not "
        f"what will render")


def build(preset_name: str, rng: random.Random, cfg, name: str = None):
    """Return (material, drawn_parameters). cfg is a config.Config."""
    p = cfg.materials[preset_name]
    drawn = {
        "preset": preset_name,
        "color": rng_color(rng, p["color"]),
        "metallic": rng_range(rng, p["metallic"]),
        "roughness": rng_range(rng, p["roughness"]),
        "coat": rng_range(rng, p["coat"]),
        "wear": rng_range(rng, p["wear"]),
    }

    mat = bpy.data.materials.new(name or f"{preset_name}_{rng.randrange(1 << 30)}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")

    set_input(bsdf, "Base Color", drawn["color"])
    set_input(bsdf, "Metallic", drawn["metallic"])
    set_input(bsdf, "Roughness", drawn["roughness"])
    # 4.x renamed Clearcoat -> Coat Weight
    (set_input(bsdf, "Coat Weight", drawn["coat"])
     or set_input(bsdf, "Clearcoat", drawn["coat"]))

    if drawn["wear"] > 0.01:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(60.0, 260.0)
        noise.inputs["Detail"].default_value = 4.0
        if "Roughness" in noise.inputs:
            noise.inputs["Roughness"].default_value = 0.6

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[1].position = 0.75
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

        # Roughness = mix(drawn roughness, noise ramp, factor=drawn wear).
        #
        # Every socket is addressed by identifier and nothing here is
        # wrapped in a try. The previous version set mix.inputs[2]/[3]
        # positionally under `except Exception:` and, on failure, linked the
        # raw ramp straight into Roughness - which discards BOTH
        # drawn["roughness"] and drawn["wear"] and renders every surface
        # with an all-or-nothing 0->1 specular swing, while meta["materials"]
        # still records the two values that never reached the shader. Every
        # preset in configs/synth3d.yaml has wear >= 0.05 > 0.01, so that
        # fallback sat on 100% of surfaces, and MIN_LUMA_DELTA above was
        # measured from renders made on the correct path. A generator that
        # renders something other than what it recorded is worse than one
        # that stops, so this stops.
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "FLOAT"
        socket_by_identifier(mix.inputs, "Factor_Float") \
            .default_value = drawn["wear"]
        socket_by_identifier(mix.inputs, "A_Float") \
            .default_value = drawn["roughness"]
        nt.links.new(ramp.outputs["Color"],
                     socket_by_identifier(mix.inputs, "B_Float"))
        nt.links.new(socket_by_identifier(mix.outputs, "Result_Float"),
                     bsdf.inputs["Roughness"])
        _assert_wear_mix_took(mix, ramp, bsdf, drawn)

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.02 + 0.10 * drawn["wear"]
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat, drawn


def apply_to_object(obj, mat):
    """
    Assign per-instance. Slots are linked to the OBJECT (see assets.py), so
    linked duplicates sharing one mesh still get independent materials.
    """
    if not obj.material_slots:
        obj.data.materials.append(None)
    for slot in obj.material_slots:
        slot.link = "OBJECT"
        slot.material = mat


def preset_luma(preset_name: str, cfg):
    """Measured display luminance of a preset, or None if unmeasured.

    Returning None rather than guessing is deliberate: a preset added to the
    config without a `luma_ref` must not silently be treated as though it sat
    at some derived luminance, because an albedo-derived number here would be
    wrong by up to a factor of two (see MIN_LUMA_DELTA) and would exclude the
    wrong presets rather than the colliding ones. An unmeasured preset simply
    stops participating in the rule.
    """
    return (cfg.materials.get(preset_name) or {}).get("luma_ref")


def eligible_presets(presets, cfg, avoid_luma, min_delta=None):
    """The presets far enough from `avoid_luma` to be visible against it.

    Falls back, in order:
      * no `avoid_luma` (backdrop unmeasured)  -> every preset, unchanged;
      * presets with no `luma_ref`             -> always kept, they opt out;
      * nothing far enough                     -> the single farthest one,
        because a scene still has to be built and the best available contrast
        is strictly better than a uniform draw.
    """
    if avoid_luma is None or len(presets) < 2:
        return list(presets)
    delta = MIN_LUMA_DELTA if min_delta is None else min_delta

    def gap(p):
        ref = preset_luma(p, cfg)
        return None if ref is None else abs(ref - avoid_luma)

    far = [p for p in presets if gap(p) is None or gap(p) >= delta]
    if far:
        return far
    return [max(presets, key=lambda p: gap(p) or 0.0)]


def for_role(role: str, rng: random.Random, cfg, avoid_luma=None,
             min_delta=None):
    """Draw a material appropriate to a sub-part role.

    `avoid_luma` is the backdrop's measured luminance; presets that would
    render within MIN_LUMA_DELTA of it are dropped from the draw. Exactly one
    rng draw is consumed either way, so a seed still reproduces a scene.
    """
    presets = cfg.role_materials.get(role) or cfg.role_materials["case"]
    return build(rng.choice(eligible_presets(presets, cfg, avoid_luma,
                                             min_delta)), rng, cfg)
