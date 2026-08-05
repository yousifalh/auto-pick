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


def rng_range(rng: random.Random, span: Tuple[float, float]) -> float:
    lo, hi = span
    return lo if lo == hi else rng.uniform(lo, hi)


def rng_color(rng: random.Random, span) -> Tuple[float, float, float, float]:
    lo, hi = span
    return tuple(rng.uniform(a, b) for a, b in zip(lo, hi)) + (1.0,)


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

        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "FLOAT"
        mix.inputs["Factor"].default_value = drawn["wear"]
        try:
            mix.inputs[2].default_value = drawn["roughness"]
            nt.links.new(ramp.outputs["Color"], mix.inputs[3])
            nt.links.new(mix.outputs[0], bsdf.inputs["Roughness"])
        except Exception:
            nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])

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
