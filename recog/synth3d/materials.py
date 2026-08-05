"""recog.synth3d.materials - randomized Principled surfaces. Requires bpy.

Presets are not module globals here: they come from the loaded config
(`config.load_config()`), so `build` and `for_role` take a `cfg` argument.
"""

from __future__ import annotations

import random
from typing import Tuple

import bpy


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


def for_role(role: str, rng: random.Random, cfg):
    """Draw a material appropriate to a sub-part role."""
    presets = cfg.role_materials.get(role) or cfg.role_materials["case"]
    return build(rng.choice(presets), rng, cfg)
