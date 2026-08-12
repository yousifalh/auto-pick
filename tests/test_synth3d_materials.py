"""The wear-mix wiring in recog/synth3d/materials.py.

`materials.py` imports bpy, so `tests/test_synth3d.py` deliberately keeps it
outside the bpy-free boundary and nothing in the suite imports it. That is
exactly why the module carries `_assert_wear_mix_took`, which runs on every
material built rather than in a test.

This file is the other half: it loads `materials.py` under a stub `bpy` (the
module only touches `bpy.data` inside `build`, which is never called here)
and drives `socket_by_identifier` and `_assert_wear_mix_took` against fake
nodes. An assertion nobody has ever seen fail is not evidence, so each of the
three ways the old code could silently mis-wire the graph is reproduced here
and the assertion is required to fire on it.

The module is executed under a private name and `bpy` is removed from
`sys.modules` again afterwards, so no stub leaks into the rest of the suite
and `recog.synth3d.materials` is never registered as importable.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MATERIALS_PY = ROOT / "recog" / "synth3d" / "materials.py"
RENDER_PY = ROOT / "recog" / "synth3d" / "render.py"


def _load_under_stub_bpy(path: Path, modname: str):
    stub = types.ModuleType("bpy")
    stub.data = types.SimpleNamespace()
    had = "bpy" in sys.modules
    prev = sys.modules.get("bpy")
    sys.modules["bpy"] = stub
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if had:
            sys.modules["bpy"] = prev
        else:
            sys.modules.pop("bpy", None)
    return mod


@pytest.fixture(scope="module")
def materials():
    return _load_under_stub_bpy(MATERIALS_PY, "_synth3d_materials_stubbed")


# ------------------------------------------------------- fake bpy nodes ----

class _Link:
    def __init__(self, from_node, from_socket):
        self.from_node = from_node
        self.from_socket = from_socket


class _Socket:
    def __init__(self, name, identifier, default=0.5):
        self.name = name
        self.identifier = identifier
        self.default_value = default
        self.links = []

    @property
    def is_linked(self):
        return bool(self.links)


class _Sockets:
    """bpy's socket collection: iterable, indexable, and keyed by NAME.

    The duplicate names are the whole point. `ShaderNodeMix` has two sockets
    called "Factor" and four each called "A", "B" and "Result", so both
    `inputs["A"]` and `inputs[2]` resolve to whichever the current build
    happens to put first.
    """

    def __init__(self, sockets):
        self._s = list(sockets)

    def __iter__(self):
        return iter(self._s)

    def __len__(self):
        return len(self._s)

    def __contains__(self, name):
        return any(s.name == name for s in self._s)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._s[key]
        for s in self._s:
            if s.name == key:
                return s
        raise KeyError(key)


class _Node:
    def __init__(self, name, inputs=(), outputs=()):
        self.name = name
        self.inputs = _Sockets(inputs)
        self.outputs = _Sockets(outputs)


def _mix_node(float_pair_first=True):
    """A ShaderNodeMix stand-in.

    `float_pair_first=True` mirrors the socket order this code was written
    against; `False` models the drift the old positional access assumed away,
    with the Color pair sitting where the Float pair used to be.
    """
    factor_f = _Socket("Factor", "Factor_Float")
    factor_v = _Socket("Factor", "Factor_Vector")
    a_f, b_f = _Socket("A", "A_Float"), _Socket("B", "B_Float")
    a_c, b_c = _Socket("A", "A_Color"), _Socket("B", "B_Color")
    res_f = _Socket("Result", "Result_Float")
    res_c = _Socket("Result", "Result_Color")
    if float_pair_first:
        ins = [factor_f, factor_v, a_f, b_f, a_c, b_c]
        outs = [res_f, res_c]
    else:
        ins = [factor_v, factor_f, a_c, b_c, a_f, b_f]
        outs = [res_c, res_f]
    return _Node("Mix", ins, outs)


def _graph(materials, wear=0.42, roughness=0.31):
    """A correctly wired (mix, ramp, bsdf, drawn) exactly as `build` wires it."""
    mix = _mix_node()
    ramp = _Node("Color Ramp", outputs=[_Socket("Color", "Color")])
    bsdf = _Node("Principled BSDF",
                 inputs=[_Socket("Roughness", "Roughness")])
    drawn = {"wear": wear, "roughness": roughness}

    materials.socket_by_identifier(
        mix.inputs, "Factor_Float").default_value = wear
    materials.socket_by_identifier(
        mix.inputs, "A_Float").default_value = roughness
    b_in = materials.socket_by_identifier(mix.inputs, "B_Float")
    b_in.links.append(_Link(ramp, ramp.outputs["Color"]))
    result = materials.socket_by_identifier(mix.outputs, "Result_Float")
    bsdf.inputs["Roughness"].links.append(_Link(mix, result))
    return mix, ramp, bsdf, drawn


# ------------------------------------------------- socket_by_identifier ----

def test_identifier_lookup_finds_the_float_pair_whatever_the_socket_order(
        materials):
    """The reason index and name access were wrong.

    On a build where the Color pair comes first, `inputs[2]` and `inputs["A"]`
    both hand back the Color socket while the drawn roughness is meant for the
    Float one. `socket_by_identifier` is unaffected by the order.
    """
    mix = _mix_node(float_pair_first=False)

    assert mix.inputs[2].identifier == "A_Color"        # what index gave
    assert mix.inputs["A"].identifier == "A_Color"      # what name gives
    assert mix.outputs[0].identifier == "Result_Color"

    assert materials.socket_by_identifier(
        mix.inputs, "A_Float").identifier == "A_Float"
    assert materials.socket_by_identifier(
        mix.inputs, "B_Float").identifier == "B_Float"
    assert materials.socket_by_identifier(
        mix.outputs, "Result_Float").identifier == "Result_Float"


def test_a_missing_identifier_raises_and_names_what_the_build_offers(
        materials):
    """It must not return None and let the caller wire up something else."""
    mix = _mix_node()
    with pytest.raises(KeyError) as exc:
        materials.socket_by_identifier(mix.inputs, "A_Quaternion")
    assert "A_Quaternion" in str(exc.value)
    assert "A_Float" in str(exc.value), (
        "the error has to say what the running build does offer, or a socket "
        "rename is undiagnosable from the traceback alone")


# ------------------------------------------------- _assert_wear_mix_took ----

def test_correctly_wired_graph_passes(materials):
    materials._assert_wear_mix_took(*_graph(materials))


def test_roughness_left_on_the_socket_default_is_caught(materials):
    """The headline symptom of the old fallback: `drawn["roughness"]` never
    reaches the shader while `meta["materials"]` records it anyway."""
    mix, ramp, bsdf, drawn = _graph(materials)
    materials.socket_by_identifier(mix.inputs, "A_Float").default_value = 0.5
    with pytest.raises(AssertionError, match="roughness"):
        materials._assert_wear_mix_took(mix, ramp, bsdf, drawn)


def test_wear_left_on_the_socket_default_is_caught(materials):
    mix, ramp, bsdf, drawn = _graph(materials)
    materials.socket_by_identifier(
        mix.inputs, "Factor_Float").default_value = 0.5
    with pytest.raises(AssertionError, match="wear"):
        materials._assert_wear_mix_took(mix, ramp, bsdf, drawn)


def test_the_old_fallback_wiring_is_caught(materials):
    """Reproduce the discarded branch exactly: raw ramp straight into
    Principled's Roughness, mix bypassed. Every surface then renders with an
    all-or-nothing 0->1 specular swing and neither drawn value is used."""
    mix, ramp, bsdf, drawn = _graph(materials)
    bsdf.inputs["Roughness"].links = [_Link(ramp, ramp.outputs["Color"])]
    with pytest.raises(AssertionError, match="Roughness"):
        materials._assert_wear_mix_took(mix, ramp, bsdf, drawn)


def test_an_unlinked_roughness_socket_is_caught(materials):
    mix, ramp, bsdf, drawn = _graph(materials)
    bsdf.inputs["Roughness"].links = []
    with pytest.raises(AssertionError, match="Roughness"):
        materials._assert_wear_mix_took(mix, ramp, bsdf, drawn)


def test_a_ramp_that_does_not_drive_the_mix_is_caught(materials):
    """Without the ramp on B the mix returns a constant and the wear texture
    is absent - a uniformly rough surface that still renders plausibly."""
    mix, ramp, bsdf, drawn = _graph(materials)
    materials.socket_by_identifier(mix.inputs, "B_Float").links = []
    with pytest.raises(AssertionError, match="ramp"):
        materials._assert_wear_mix_took(mix, ramp, bsdf, drawn)


# ----------------------------------------------------- no swallowed except ----

def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} has no function named {name!r}")


@pytest.mark.parametrize("path,func", [
    (MATERIALS_PY, "build"),
    (RENDER_PY, "configure_beauty"),
    (RENDER_PY, "read_index_exr"),
])
def test_these_functions_swallow_nothing(path, func):
    """All three used to continue past a failure and produce a render that
    disagreed with the manifest describing it: materials.build discarded the
    drawn roughness and wear, configure_beauty discarded the view transform
    while still applying an exposure tuned for it, and read_index_exr read
    the object-index pass back through whatever colour space it landed in.

    They are bpy-only, so nothing else in this suite can reach them. A source
    check is a weak test, but it is the one that would notice a `try` being
    reintroduced around these particular calls - which is how all three got
    here.
    """
    tries = [n for n in ast.walk(_function(path, func))
             if isinstance(n, (ast.Try, ast.ExceptHandler))]
    assert not tries, (
        f"{path.name}:{func} has grown an exception handler again; if it is "
        f"genuinely needed it must re-raise or record, not continue")
