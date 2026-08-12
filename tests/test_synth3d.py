"""Unit tests for the Blender-free half of recog.synth3d.

Every module touched here must import without bpy, so this file runs in
the ordinary pytest environment.
"""
import ast
import json
from pathlib import Path

import pytest

from recog.synth3d import config as C


ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ classes ----

def test_class_ids_are_one_based():
    ids = C.class_ids()
    assert ids == {"battery": 1, "cartridge": 2}


def test_class_vocabulary_matches_the_voc_loader():
    """Drift between synth3d and recog.dataset would silently mislabel data."""
    from recog.dataset import CLASS_MAP
    expected = {k: v for k, v in CLASS_MAP.items() if v != 0}
    assert C.class_ids() == expected


def test_num_classes_with_background():
    assert len(C.CLASSES) + 1 == 3


# ------------------------------------------------------- bpy boundary ----

_BPY_FREE_CANDIDATES = ["config", "catalog", "layout", "annotate", "lightrig", "bay"]
_BPY_FREE_MODS = [m for m in _BPY_FREE_CANDIDATES
                   if (ROOT / "recog" / "synth3d" / f"{m}.py").is_file()]


def _imports_bpy(src: str) -> bool:
    """Whether `src` actually imports (statically or dynamically) the
    `bpy` package or any of its submodules.

    An AST walk, not a substring grep: the previous `"import bpy" not in
    src` check let `from bpy import context`, `import bpy as b`, and
    `importlib.import_module("bpy")` all walk straight past it, and it
    false-positived once on a docstring that merely mentioned the phrase
    "import bpy" in prose. Parsing means a string appearing in a
    docstring, comment, or ordinary string constant is never mistaken for
    a real import, because it is never an `Import`/`ImportFrom`/`Call`
    node - only actual import statements and the two dynamic-import
    builtins (`importlib.import_module`, `__import__`) are inspected, and
    only when their first argument is a literal string.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "bpy" or a.name.startswith("bpy.")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and (
                    node.module == "bpy" or node.module.startswith("bpy.")):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                fname = func.attr
            elif isinstance(func, ast.Name):
                fname = func.id
            else:
                fname = None
            if fname in ("import_module", "__import__") and node.args:
                arg0 = node.args[0]
                if (isinstance(arg0, ast.Constant)
                        and isinstance(arg0.value, str)
                        and (arg0.value == "bpy"
                             or arg0.value.startswith("bpy."))):
                    return True
    return False


@pytest.mark.parametrize("src", [
    "import bpy\n",
    "import bpy as b\n",
    "import bpy.types\n",
    "import bpy.types as bt\n",
    "from bpy import context\n",
    "from bpy import context as c\n",
    "from bpy.types import Object\n",
    "import importlib\nimportlib.import_module('bpy')\n",
    "import importlib\nimportlib.import_module('bpy.types')\n",
    "import importlib as il\nil.import_module('bpy')\n",
    "__import__('bpy')\n",
])
def test_bpy_detector_rejects_every_known_evasion(src):
    """Each of these walks straight past a plain `\"import bpy\" not in
    src` substring check; the AST walk must catch all of them."""
    assert _imports_bpy(src), f"missed an evasion: {src!r}"


def test_bpy_detector_accepts_a_docstring_that_merely_mentions_it():
    """The exact false positive this check replaced: a module that talks
    ABOUT staying free of `import bpy` in prose, without ever actually
    importing it, must not fail."""
    src = ('"""This module must stay free of `import bpy` - see the '
           'package boundary note."""\nx = 1\n')
    assert not _imports_bpy(src)


def test_every_bpy_free_module_is_actually_checked():
    """The existence filter above was scaffolding from when these modules
    landed one task at a time. All six exist now, so it must be a no-op:
    without this assertion, renaming or moving a module would silently drop
    it from the boundary check below and the parametrised test would still
    pass, having tested one fewer file."""
    assert _BPY_FREE_MODS == _BPY_FREE_CANDIDATES, (
        f"these modules were skipped by the bpy-boundary check: "
        f"{sorted(set(_BPY_FREE_CANDIDATES) - set(_BPY_FREE_MODS))}")


@pytest.mark.parametrize("mod", _BPY_FREE_MODS)
def test_pure_modules_never_import_bpy(mod):
    src = (ROOT / "recog" / "synth3d" / f"{mod}.py").read_text(encoding="utf-8")
    assert not _imports_bpy(src), f"{mod}.py must stay Blender-free"


def test_package_init_imports_no_submodules():
    src = (ROOT / "recog" / "synth3d" / "__init__.py").read_text(encoding="utf-8")
    assert "import" not in src.replace("__version__", "")


# ------------------------------------------------------------- config ----

def test_load_config_from_yaml():
    cfg = C.load_config()
    assert cfg.render.res == (1280, 720)
    assert cfg.layout.area == (0.80, 0.45)
    assert "scatter" in cfg.param_space["layout_mode"]
    assert "jig" in cfg.param_space["layout_mode"]
    assert set(cfg.role_materials) == {"case", "case_lid", "case_liner", "cell"}
    assert cfg.lighting["overcast_softbox"]["kind"] == "camera_softbox"


# ----------------------------------------------------------- lighting ----
#
# The dataset's three original rigs were all `camera_softbox`, which puts the
# lamp at the camera. Under a top-down camera that makes the light coaxial
# with the view: shading is flat and every cast shadow hides exactly behind
# the object that cast it. The reference photos are full of directional
# shadows, so the training set was missing a cue the real images all carry.

_OFF_AXIS_KEYS = ("energy", "size", "kelvin", "azimuth", "elevation",
                  "distance", "world_strength", "world_kelvin")


def test_lighting_rigs_cover_more_than_one_kind():
    """All-camera_softbox is the state this task existed to leave."""
    kinds = {s["kind"] for s in C.load_config().lighting.values()}
    assert "off_axis" in kinds, (
        "no off-axis rig: every scene is lit coaxially with the camera, so "
        "the dataset contains no visible cast shadow at all")
    assert "camera_softbox" in kinds, "the original rigs must keep working"


def test_every_lighting_rig_is_reachable_from_the_param_space():
    """A rig absent from param_space.lighting is never sampled, so it is
    dead config that looks live - the single easiest way to add a lighting
    preset and have it change nothing whatsoever."""
    cfg = C.load_config()
    assert set(cfg.param_space["lighting"]) == set(cfg.lighting), (
        f"rigs defined but never sampled: "
        f"{sorted(set(cfg.lighting) - set(cfg.param_space['lighting']))}; "
        f"sampled but not defined: "
        f"{sorted(set(cfg.param_space['lighting']) - set(cfg.lighting))}")


def test_off_axis_rigs_carry_every_key_setup_lighting_reads():
    """world.setup_lighting indexes these directly, so a missing one is a
    KeyError thousands of renders into a dataset build, not at load time."""
    cfg = C.load_config()
    for name, spec in cfg.lighting.items():
        if spec["kind"] != "off_axis":
            continue
        for key in _OFF_AXIS_KEYS:
            assert key in spec, f"{name} has kind off_axis but no {key!r}"
        lo, hi = spec["elevation"]
        assert 0 < lo <= hi <= 90, f"{name} elevation {spec['elevation']}"
        assert spec["distance"][0] > 0, name


def test_a_fill_lamp_rig_configures_every_fill_key():
    """The fill lamp is optional and drawn all-or-nothing off `fill_energy`.
    A rig that sets fill_energy and forgets fill_kelvin gets a KeyError at
    render time; one that sets fill_kelvin and forgets fill_energy silently
    builds no fill lamp at all and quietly stops being mixed-illuminant."""
    cfg = C.load_config()
    fill_keys = ("fill_energy", "fill_size", "fill_kelvin",
                 "fill_azimuth_offset", "fill_elevation", "fill_distance")
    for name, spec in cfg.lighting.items():
        present = [k for k in fill_keys if k in spec]
        assert present in ([], list(fill_keys)), (
            f"{name} configures a partial fill lamp: has {present}, "
            f"missing {[k for k in fill_keys if k not in spec]}")


def test_the_dim_rig_is_actually_the_dimmest_and_lowest():
    """`dim_workshop` is the user-facing 'dark setting'. Its darkness must
    come from the light's geometry - a low elevation and almost no ambient -
    rather than from a small energy, because exposure is sampled per scene
    and multiplies energy while it cannot multiply an angle. An earlier draft
    got this backwards and rendered black frames at the bottom of the
    exposure range: median object-to-background difference under one 8-bit
    level, on annotated images."""
    cfg = C.load_config()
    dim = cfg.lighting["dim_workshop"]
    others = {n: s for n, s in cfg.lighting.items()
              if n != "dim_workshop" and s["kind"] == "off_axis"}

    # Midpoints, not disjoint bands: mixed_daylight is a window light and is
    # legitimately low too, so the bands overlap by design. What has to hold
    # is that dim_workshop is the lowest and least-ambient rig on average.
    def mid(span):
        return (span[0] + span[1]) / 2.0

    assert all(mid(dim["elevation"]) < mid(o["elevation"])
               for o in others.values()), (
        f"dim_workshop's mean elevation {mid(dim['elevation'])} is not the "
        f"lowest of the off-axis rigs "
        f"({ {n: mid(s['elevation']) for n, s in others.items()} }) - a low "
        f"lamp is what makes its shadows long")
    assert all(mid(dim["world_strength"]) < mid(o["world_strength"])
               for o in others.values()), (
        f"dim_workshop's mean ambient {mid(dim['world_strength'])} is not "
        f"the lowest of the off-axis rigs "
        f"({ {n: mid(s['world_strength']) for n, s in others.items()} }) - "
        f"low ambient is what makes its shadows deep")


def test_the_fluorescent_rig_is_off_the_planckian_locus():
    """Real tubes are mercury discharge behind a phosphor, not blackbody. If
    every illuminant in the set lies on the Planckian locus the detector
    learns a one-parameter illuminant model that no real fluorescent-lit
    photograph obeys."""
    tint = C.load_config().lighting["fluorescent_factory"]["tint"]
    assert len(tint) == 3
    assert tint[1] > tint[0] and tint[1] > tint[2], (
        f"fluorescent tint {tint} has no green bias, so it is just another "
        f"colour temperature")
    assert all(0.8 <= c <= 1.2 for c in tint), f"tint {tint} is not subtle"


def test_exposure_is_sampled_per_scene_not_fixed():
    """A single render.exposure tone-maps every scene identically. Measured
    over 20 scenes at the old fixed -3.5, the spread of mean frame luminance
    was 0.170 and essentially all of it came from the backdrop, not from any
    variation in light level."""
    cfg = C.load_config()
    lo, hi = cfg.param_space["exposure"]
    assert lo < hi, "exposure must be a real range, not a pinned value"
    assert hi - lo >= 1.0, f"exposure range {hi - lo:.2f} stops is token"
    # render.exposure survives as the documented fallback for a config that
    # omits param_space.exposure; it must stay inside the sampled band or the
    # two disagree about what a normal scene looks like.
    assert lo <= cfg.render.exposure <= hi


def test_every_material_and_backdrop_has_a_measured_luma_ref():
    """The contrast rule silently stops applying to anything unmeasured.

    materials.for_role keeps a part off a backdrop it would disappear into by
    comparing the two `luma_ref` values. A preset with no `luma_ref` opts out
    of that check entirely - deliberately, since guessing from the albedo gets
    it wrong by up to a factor of two - so adding a preset without measuring it
    would quietly reintroduce invisible objects. This is the reminder.
    """
    cfg = C.load_config()
    for name, spec in cfg.materials.items():
        assert isinstance(spec.get("luma_ref"), (int, float)), (
            f"material preset {name!r} has no measured luma_ref; see the "
            f"comment above `materials:` in configs/synth3d.yaml")
    for name, spec in cfg.backdrops.items():
        assert isinstance(spec.get("luma_ref"), (int, float)), (
            f"backdrop {name!r} has no measured luma_ref")
    # Both sides are display-referred luminance, so they share a scale.
    for name, spec in list(cfg.materials.items()) + list(cfg.backdrops.items()):
        assert 0.0 <= spec["luma_ref"] <= 1.0, name


def test_every_backdrop_carries_its_own_albedo_range():
    """The palette used to be hard-coded in world.py, where it could not be
    tuned or even seen from the config."""
    cfg = C.load_config()
    for name, spec in cfg.backdrops.items():
        lo, hi = spec["color"]
        assert len(lo) == 3 and len(hi) == 3, name
        assert all(0.0 <= c <= 1.0 for c in lo + hi), name
        assert all(h >= l for l, h in zip(lo, hi)), name


def test_overlap_is_configured_to_actually_happen():
    """A `max_overlap_iou` of 0 or an `overlap_prob` of 0 leaves the whole
    feature inert while looking live. Without real overlap, `min_visibility`
    still has nothing to threshold and the synthetic validation metric still
    saturates at mAP 1.0."""
    cfg = C.load_config()
    assert cfg.layout.max_overlap_iou > 0.0
    assert 0.0 < cfg.param_space["overlap_prob"] <= 1.0
    # Not so large that clean, unambiguous examples become the minority.
    assert cfg.param_space["overlap_prob"] <= 0.75


def test_jig_clearance_leaves_room_for_the_jitter_it_allows():
    """jig_jitter_deg turns a part inside its own pocket; too little clearance
    and the part pushes through the pocket wall it is supposed to sit in. The
    binding case is the longest asset on its short axis."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    theta = math.radians(cfg.jig_jitter_deg)
    worst = max(max(w, h) * math.sin(theta) / 2.0 for w, h in fps)
    assert cfg.jig_clearance >= worst, (
        f"jig_clearance {cfg.jig_clearance * 1000:.2f}mm is under the "
        f"{worst * 1000:.2f}mm a {cfg.jig_jitter_deg} degree turn adds to the "
        f"largest part's AABB")


def test_zoom_is_sampled_per_scene_and_actually_varies_scale():
    """Without this the training set is effectively single-scale. MEASURED on
    the previous 1000-image set: battery sqrt(area) p05 50.9 / p95 56.2, a
    ratio of 1.10, while the real photos span 43-65 px (1.51) and per-image AP
    tracks that gap monotonically down to 0.380 on the smallest."""
    cfg = C.load_config()
    lo, hi = cfg.param_space["zoom"]
    assert 0 < lo < hi, f"zoom {cfg.param_space['zoom']} is not a real range"
    # The real photos need 1.51 end to end; a range that cannot reach it is
    # decoration. Measured: 0.75-1.60 yields a 1.86 box-size ratio.
    assert hi / lo >= 1.51, (
        f"zoom range {lo}-{hi} spans {hi / lo:.2f}, under the 1.51 scale ratio "
        f"the real photos actually show")
    # zoom < 1 is what crops parts at the frame edge; the old set had 0 of
    # 8542 truncated annotations against 2 of 80 in the real photos.
    assert lo < 1.0, (
        f"zoom range starts at {lo} >= 1.0, so the frame always contains the "
        f"whole layout area and nothing is ever truncated")


def test_layout_area_matches_render_aspect():
    """A square area under a 16:9 render wastes ~44% of every frame."""
    cfg = C.load_config()
    aspect = cfg.render.res[0] / cfg.render.res[1]
    assert cfg.layout.area[0] / cfg.layout.area[1] == pytest.approx(aspect, rel=0.02)


def test_unknown_top_level_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("render: {res: [8, 8]}\nnonsense_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense_key"):
        C.load_config(p)


def test_json_sidecar_is_used_when_yaml_unavailable(tmp_path, monkeypatch):
    """Inside Blender there is no PyYAML, so the .json sidecar wins even when
    a real, parseable .yaml file sits right next to it - not merely in the
    case where no .yaml file exists at all. The two files carry different
    render.res values, so the assertion can only pass if the sidecar (not
    the decoy .yaml) was actually the one read."""
    import os, time
    src = C.load_config()
    y = tmp_path / "synth3d.yaml"
    j = tmp_path / "synth3d.json"
    y.write_text("render: {res: [1, 1]}\n", encoding="utf-8")   # decoy value
    j.write_text(json.dumps(C.config_to_dict(src)), encoding="utf-8")
    now = time.time()
    os.utime(y, (now - 100, now - 100))
    os.utime(j, (now, now))                                     # sidecar newer
    monkeypatch.setattr(C, "_HAVE_YAML", False)
    cfg = C.load_config(y)
    assert cfg.render.res == src.render.res   # from JSON, not the (1, 1) decoy
    assert cfg.lighting.keys() == src.lighting.keys()


def test_stale_sidecar_raises_with_the_fix_command(tmp_path, monkeypatch):
    y = tmp_path / "synth3d.yaml"
    j = tmp_path / "synth3d.json"
    j.write_text("{}", encoding="utf-8")
    y.write_text("render: {}\n", encoding="utf-8")   # written after the json
    import os, time
    os.utime(j, (time.time() - 100, time.time() - 100))
    monkeypatch.setattr(C, "_HAVE_YAML", False)
    with pytest.raises(RuntimeError, match="recog.sync_config"):
        C.load_config(y)


# ----------------------------------------------------------- variants ----

def test_variants_cover_the_three_real_presentations():
    names = {v.name for v in C.VARIANTS}
    assert names == {"assembled", "cells_only", "open_case"}


def test_assembled_labels_the_whole_unit_cartridge():
    v = next(v for v in C.VARIANTS if v.name == "assembled")
    assert v.label == "cartridge"
    assert set(v.keep_roles) == {"cell", "case", "case_lid", "case_liner"}


def test_open_case_labels_roles_separately():
    v = next(v for v in C.VARIANTS if v.name == "open_case")
    assert v.label is None
    assert v.label_roles == {"cell": "battery", "case": "cartridge"}


def test_every_variant_label_is_a_real_class():
    valid = set(C.CLASSES)
    for v in C.VARIANTS:
        if v.label is not None:
            assert v.label in valid, v.name
        for cls in v.label_roles.values():
            assert cls in valid, v.name


# ---------------------------------------------------------- catalog ----

from recog.synth3d import catalog as CAT

ASSETS = ROOT / "recog" / "synth3d" / "assets"


def test_all_four_assets_present():
    cat = CAT.load_catalog(str(ASSETS))
    names = {a["name"] for a in cat["assets"]}
    assert names == {"AnkerPowerCore10000", "AnkerPowerCore13000",
                     "AnkerPowerCore20100", "AnkerPowerCore26800"}
    assert cat["units"] == "m"


def test_role_of_classifies_every_real_subpart_name():
    """All 33 sub-part names from the real CAD must classify correctly by
    name, EXCEPT the case/case_liner split (task 3b): that split is
    geometric (catalog.classify_case_parts, by XY footprint area), not
    name-based, because the two `Case*_btm*` parts share one name prefix
    and appear in a different order per asset. A `case_liner` subpart's
    catalog.json role therefore legitimately differs from what role_of()
    alone would say for its name (which is still "case" - the same
    fallback role_of gives every `_btm` part) - that gap is exactly why
    the split had to move into catalog.py's post-pass instead of
    CLASS_RULES.
    """
    cat = CAT.load_catalog(str(ASSETS))
    seen = 0
    for asset in cat["assets"]:
        for sp in asset["subparts"]:
            role = CAT.role_of(sp["name"])
            if sp["role"] == "case_liner":
                assert role == "case", \
                    f"{sp['name']}: expected name-based case, got {role}"
            else:
                assert role == sp["role"], \
                    f"{sp['name']}: {role} != {sp['role']}"
            assert role in ("cell", "case", "case_lid")
            seen += 1
    assert seen == 33, f"expected 33 sub-parts, catalogued {seen}"


def test_cell_regex_matches_nx_incremented_names():
    """NX renames instances; a literal 'Cell_18650' match misses these."""
    for name in ("004695_A;1-Cell_18650", "004695_A;2-Cell_18651",
                 "004695_A;3-Cell_18650_18652", "Cell_99999"):
        assert CAT.role_of(name) == "cell", name


def test_case_names_classify_as_case():
    for name in ("004696_A;2-Case10000_btm", "004697_A;1-Case26800_btm"):
        assert CAT.role_of(name) == "case", name


def test_top_names_classify_as_case_lid():
    for name in ("004697_A;2-Case10000_top", "004710_A;2-Case26800_top"):
        assert CAT.role_of(name) == "case_lid", name


def test_unknown_subpart_falls_back_to_case():
    assert CAT.role_of("something_unrecognised") == "case"


# ------------------------------------------------- case/case_liner split ----

def test_classify_case_parts_picks_the_largest_xy_area():
    assert CAT.classify_case_parts({"a": 100.0, "b": 50.0}) == \
        {"a": "case", "b": "case_liner"}


def test_classify_case_parts_is_a_noop_for_a_single_part():
    assert CAT.classify_case_parts({"a": 42.0}) == {"a": "case"}


def test_classify_case_parts_is_a_noop_for_no_parts():
    assert CAT.classify_case_parts({}) == {}


def test_classify_case_parts_raises_when_areas_are_within_one_percent():
    with pytest.raises(ValueError):
        CAT.classify_case_parts({"a": 100.0, "b": 99.5})


def test_classify_case_parts_handles_more_than_two_parts():
    result = CAT.classify_case_parts({"a": 10.0, "b": 100.0, "c": 50.0})
    assert result == {"a": "case_liner", "b": "case", "c": "case_liner"}


def test_split_case_liner_reclassifies_the_smaller_of_two_case_parts():
    subparts = [
        {"name": "outer", "role": "case", "extents_mm": [62.9, 90.9, 11.1]},
        {"name": "inner", "role": "case", "extents_mm": [59.02, 90.9, 18.32]},
        {"name": "lid", "role": "case_lid", "extents_mm": [62.9, 90.9, 11.1]},
    ]
    CAT._split_case_liner(subparts)
    roles = {s["name"]: s["role"] for s in subparts}
    assert roles == {"outer": "case", "inner": "case_liner", "lid": "case_lid"}


def test_split_case_liner_is_a_noop_with_only_one_case_part():
    subparts = [{"name": "outer", "role": "case",
                "extents_mm": [62.9, 90.9, 11.1]}]
    CAT._split_case_liner(subparts)
    assert subparts[0]["role"] == "case"


def test_split_case_liner_raises_on_ambiguous_areas():
    subparts = [
        {"name": "a", "role": "case", "extents_mm": [62.9, 90.9, 11.1]},
        {"name": "b", "role": "case", "extents_mm": [62.6, 90.9, 18.32]},
    ]
    with pytest.raises(ValueError):
        CAT._split_case_liner(subparts)


def test_all_four_real_assets_split_to_exactly_one_case_and_one_liner():
    """Verified against the real CAD (task-3b-brief.md's own table): the
    outer shell always wins by several mm of XY margin."""
    cat = CAT.load_catalog(str(ASSETS))
    for asset in cat["assets"]:
        counts = asset["role_counts"]
        assert counts.get("case") == 1, asset["name"]
        assert counts.get("case_liner") == 1, asset["name"]
        case = next(s for s in asset["subparts"] if s["role"] == "case")
        liner = next(s for s in asset["subparts"]
                    if s["role"] == "case_liner")
        case_area = case["extents_mm"][0] * case["extents_mm"][1]
        liner_area = liner["extents_mm"][0] * liner["extents_mm"][1]
        assert case_area > liner_area, asset["name"]


def test_tray_outer_z_top_is_the_shell_alone_not_the_taller_liner():
    """task 3b: tray_outer_mm's z-top must come from the outer shell only
    (11.10mm on every real asset), not the taller inner liner (~18.2-
    18.3mm) that used to be lumped into the same `case` role's AABB."""
    cat = CAT.load_catalog(str(ASSETS))
    for asset in cat["assets"]:
        assert asset["tray_outer_mm"][4] == pytest.approx(11.1, abs=0.01), \
            asset["name"]


def test_interior_wall_and_floor_are_unmoved_by_the_liner_split():
    """The liner is narrower than the shell in XY (never contributed to
    tray_outer_mm's x0/y0/x1/y1), so everything derived from those - the
    wall thickness, the interior cavity, the module bay, the cell floor -
    must be byte-identical to their pre-task-3b values."""
    cat = CAT.load_catalog(str(ASSETS))
    expected = {
        "AnkerPowerCore10000": dict(
            case_wall_mm=4.0, tray_floor_mm=1.95,
            interior_mm=[-27.45, -43.0, 27.45, 41.45],
            module_bay_mm=[-27.45, 22.0, 27.45, 41.45]),
        "AnkerPowerCore13000": dict(
            case_wall_mm=3.75, tray_floor_mm=1.95,
            interior_mm=[-36.6, -44.75, 36.6, 44.75],
            module_bay_mm=[-36.6, 22.0, 36.6, 44.75]),
        "AnkerPowerCore20100": dict(
            case_wall_mm=3.7, tray_floor_mm=1.95,
            interior_mm=[-27.45, -80.2, 27.45, 80.2],
            module_bay_mm=[-27.45, 55.0, 27.45, 80.2]),
        "AnkerPowerCore26800": dict(
            case_wall_mm=4.25, tray_floor_mm=1.95,
            interior_mm=[-36.6, -85.75, 36.6, 85.75],
            module_bay_mm=[-36.6, 55.0, 36.6, 85.75]),
    }
    for asset in cat["assets"]:
        exp = expected[asset["name"]]
        assert asset["case_wall_mm"] == pytest.approx(exp["case_wall_mm"])
        assert asset["tray_floor_mm"] == pytest.approx(exp["tray_floor_mm"])
        assert asset["interior_mm"] == pytest.approx(exp["interior_mm"])
        assert asset["module_bay_mm"] == pytest.approx(exp["module_bay_mm"])


def test_expected_cell_counts_per_asset():
    """Cell count is the CAD's ground truth: 3 / 4 / 6 / 8."""
    cat = CAT.load_catalog(str(ASSETS))
    counts = {a["name"]: a["role_counts"].get("cell", 0) for a in cat["assets"]}
    assert counts == {"AnkerPowerCore10000": 3, "AnkerPowerCore13000": 4,
                      "AnkerPowerCore20100": 6, "AnkerPowerCore26800": 8}


# ----------------------------------------------------------- layout ----

import random

from recog.synth3d import layout as L


def _real_footprints():
    """Footprints in metres, from the real catalog extents."""
    cat = CAT.load_catalog(str(ASSETS))
    out = []
    for a in cat["assets"]:
        ex = a["extents_mm"]
        out.append((ex[0] / 1000.0, ex[1] / 1000.0))
    return out


def _aabb(fp, plc, cfg):
    ex, ey = L.footprint_after_rotation(fp[0], fp[1], plc.quarter)
    return (plc.x - ex / 2, plc.y - ey / 2, plc.x + ex / 2, plc.y + ey / 2)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _padded_half(f, plc, cfg):
    ex, ey = L.footprint_after_rotation(f[0], f[1], plc.quarter)
    return ex / 2 + cfg.pad / 2, ey / 2 + cfg.pad / 2


def test_scatter_never_overlaps_when_overlap_is_disabled_over_300_scenes():
    """max_overlap_iou = 0 is the dataclass default and must reproduce the
    exact non-overlap this solver guaranteed for its whole life - every config
    written before overlap existed depends on it."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    total = 0
    for seed in range(300):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs = L.plan(chosen, cfg, rng, max_overlap_iou=0.0)
        boxes = [_aabb(f, p, cfg) for f, p in zip(chosen, plcs) if p is not None]
        total += len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j]), f"seed {seed}"
        assert all(p.z == 0.0 for p in plcs if p is not None), f"seed {seed}"
    assert total > 500, f"only {total} items placed across 300 scenes"


def test_disabled_overlap_is_the_same_layout_the_old_solver_produced():
    """Not merely 'also non-overlapping': the same rng draws, hence the same
    placements. A rewrite that consumed the rng differently would silently
    change every scene in the corpus while still passing the test above."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in (0, 3, 11, 42):
        a = [p.as_dict() if p else None
             for p in L.plan(fps, cfg, random.Random(seed), max_overlap_iou=0.0)]
        b = [p.as_dict() if p else None
             for p in L.plan(fps, cfg, random.Random(seed), heights=[0.02] * len(fps),
                             max_overlap_iou=0.0)]
        assert a == b, f"seed {seed}: passing heights changed a disjoint layout"


def test_scatter_overlap_is_bounded_by_the_configured_iou():
    """Above 0 parts may touch and occlude, but never by more than asked."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    limit = 0.20
    n_overlapping = 0
    for seed in range(300):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(2, 5))]
        plcs = L.plan(chosen, cfg, rng, heights=[0.02] * len(chosen),
                      max_overlap_iou=limit)
        got = [(f, p) for f, p in zip(chosen, plcs) if p is not None]
        for i in range(len(got)):
            for j in range(i + 1, len(got)):
                (fi, pi), (fj, pj) = got[i], got[j]
                hxi, hyi = _padded_half(fi, pi, cfg)
                hxj, hyj = _padded_half(fj, pj, cfg)
                v = L.padded_iou(pi.x, pi.y, hxi, hyi, pj.x, pj.y, hxj, hyj)
                assert v <= limit + 1e-12, f"seed {seed}: IoU {v:.4f} > {limit}"
                n_overlapping += v > 0.0
    assert n_overlapping > 0, (
        "no pair overlapped at all across 300 scenes - the threshold is inert "
        "and filter.min_visibility would still have nothing to threshold")


def test_an_overlapping_part_is_lifted_onto_the_one_it_overlaps():
    """Two solids resting on the same floor and sharing ground area occupy the
    same space. Without the lift a render shows interpenetration rather than
    one part lying across another, and no mask- or box-level check can see it.
    """
    cfg = C.load_config().layout
    fps = _real_footprints()
    seen_lift = False
    for seed in range(300):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(2, 5))]
        heights = [0.02 + 0.001 * k for k in range(len(chosen))]
        plcs = L.plan(chosen, cfg, rng, heights=heights, max_overlap_iou=0.20)
        got = [(k, f, p) for k, (f, p) in enumerate(zip(chosen, plcs))
               if p is not None]
        for k, f, p in got:
            hx, hy = _padded_half(f, p, cfg)
            partners = [
                (k2, f2, p2) for k2, f2, p2 in got
                if k2 != k and L.padded_iou(
                    p.x, p.y, hx, hy, p2.x, p2.y,
                    *_padded_half(f2, p2, cfg)) > 0.0]
            if not partners:
                assert p.z == 0.0, f"seed {seed}: lifted with nothing under it"
                continue
            seen_lift = True
            # It rests on something, and never below the floor.
            assert p.z >= 0.0
            # One of the two must be on top of the other, not both on the
            # floor: at least one of every overlapping pair carries a lift.
            assert p.z > 0.0 or any(p2.z > 0.0 for _, _, p2 in partners), (
                f"seed {seed}: an overlapping pair both rest on z = 0")
    assert seen_lift, "no overlap occurred, so the lift was never exercised"


def test_scatter_stays_inside_the_area():
    cfg = C.load_config().layout
    W, H = cfg.area
    fps = _real_footprints()
    for seed in range(200):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        for f, p in zip(chosen, L.plan(chosen, cfg, rng)):
            if p is None:
                continue
            x0, y0, x1, y1 = _aabb(f, p, cfg)
            assert x0 >= -W / 2 - 1e-6 and x1 <= W / 2 + 1e-6, f"seed {seed}"
            assert y0 >= -H / 2 - 1e-6 and y1 <= H / 2 + 1e-6, f"seed {seed}"


def test_rotation_constraint_always_satisfied():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(200):
        rng = random.Random(seed)
        for p in L.plan(fps, cfg, rng):
            if p is None:
                continue
            assert p.quarter in (0, 1, 2, 3)
            off = abs(p.rot_deg - p.quarter * 90)
            assert off <= cfg.jitter_deg + 1e-9, f"tilt {off} exceeds cap"


def test_plan_is_deterministic_for_a_seed():
    cfg = C.load_config().layout
    fps = _real_footprints()
    a = [p.as_dict() if p else None for p in L.plan(fps, cfg, random.Random(7))]
    b = [p.as_dict() if p else None for p in L.plan(fps, cfg, random.Random(7))]
    assert a == b


# -------------------------------------------------------------- jig ----

def test_jig_pockets_contain_their_items():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs, pockets = L.plan_jig(chosen, cfg, rng)
        placed = [(f, p) for f, p in zip(chosen, plcs) if p is not None]
        assert len(pockets) == len(placed), f"seed {seed}"
        for (f, p), pk in zip(placed, pockets):
            ix0, iy0, ix1, iy1 = _aabb(f, p, cfg)
            px0, py0 = pk.x - pk.w / 2, pk.y - pk.h / 2
            px1, py1 = pk.x + pk.w / 2, pk.y + pk.h / 2
            assert px0 <= ix0 + 1e-6 and ix1 <= px1 + 1e-6, f"seed {seed} x"
            assert py0 <= iy0 + 1e-6 and iy1 <= py1 + 1e-6, f"seed {seed} y"


def test_jig_pockets_never_overlap():
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(2, 4))]
        _, pockets = L.plan_jig(chosen, cfg, rng)
        boxes = [(p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2)
                 for p in pockets]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j]), f"seed {seed}"


def test_jig_pockets_leave_a_wall_between_them():
    """Flush pockets boolean-difference into one trough, not a fixture plate."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(200):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(2, 4))]
        _, pockets = L.plan_jig(chosen, cfg, rng)
        for i in range(len(pockets)):
            for j in range(i + 1, len(pockets)):
                a, b = pockets[i], pockets[j]
                gx = max((b.x - b.w/2) - (a.x + a.w/2), (a.x - a.w/2) - (b.x + b.w/2))
                gy = max((b.y - b.h/2) - (a.y + a.h/2), (a.y - a.h/2) - (b.y + b.h/2))
                assert max(gx, gy) >= cfg.jig_wall - 1e-9, f"seed {seed}"


def test_jig_pockets_stay_inside_the_area():
    cfg = C.load_config().layout
    W, H = cfg.area
    fps = _real_footprints()
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        _, pockets = L.plan_jig(chosen, cfg, rng)
        for p in pockets:
            assert p.x - p.w / 2 >= -W / 2 - 1e-6
            assert p.x + p.w / 2 <= W / 2 + 1e-6
            assert p.y - p.h / 2 >= -H / 2 - 1e-6
            assert p.y + p.h / 2 <= H / 2 + 1e-6


def test_jig_clearance_is_actually_applied():
    cfg = C.load_config().layout
    rng = random.Random(0)
    fp = [(0.06, 0.09)]
    plcs, pockets = L.plan_jig(fp, cfg, rng)
    assert plcs[0] is not None and len(pockets) == 1
    pk = pockets[0]
    ex, ey = L.footprint_after_rotation(0.06, 0.09, plcs[0].quarter)
    assert pk.w >= ex + 2 * cfg.jig_clearance - 1e-9
    assert pk.h >= ey + 2 * cfg.jig_clearance - 1e-9


def test_jig_rotation_uses_the_tighter_jitter_cap():
    """Parts sit inside pockets; scatter's 2 deg would sweep past the clearance."""
    cfg = C.load_config().layout
    fps = _real_footprints()
    for seed in range(50):
        rng = random.Random(seed)
        plcs, _ = L.plan_jig(fps, cfg, rng)
        for p in plcs:
            if p is None:
                continue
            assert abs(p.rot_deg - p.quarter * 90) <= cfg.jig_jitter_deg + 1e-9


def test_jig_returns_none_for_items_that_do_not_fit():
    cfg = C.load_config().layout
    rng = random.Random(0)
    plcs, pockets = L.plan_jig([(5.0, 5.0)], cfg, rng)
    assert plcs == [None]
    assert pockets == []


# --------------------------------------------------------- annotate ----

import numpy as np

from recog.synth3d import annotate as A


def _meta(**over):
    base = {"class": "battery", "asset": "AnkerPowerCore10000",
            "variant": "cells_only", "role": "cell"}
    base.update(over)
    return base


def test_box_edges_are_exclusive():
    """A 1-pixel object must yield a 1x1 box, never a zero-area one."""
    ids = np.zeros((10, 10), dtype=np.int32)
    ids[4, 5] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert len(anns) == 1
    assert anns[0]["bbox_xyxy"] == [5, 4, 6, 5]
    assert anns[0]["bbox_xywh"] == [5, 4, 1, 1]


def test_no_annotation_has_zero_area():
    """min_side=0 deliberately disables the side filter, so a degenerate box
    can only be kept out by the +1 exclusive-edge arithmetic itself - not by
    a filter that would coincidentally reject it first."""
    ids = np.zeros((64, 64), dtype=np.int32)
    ids[10, 10] = 1              # 1x1 instance
    ids[30, 5:15] = 2            # 1-row instance: 10 px wide, 1 px tall
    meta = {1: _meta(), 2: _meta()}
    cfg = C.FilterCfg(min_px=1, min_side=0)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert len(anns) == 2
    for a in anns:
        x0, y0, x1, y1 = a["bbox_xyxy"]
        assert x1 > x0 and y1 > y0


def test_area_is_silhouette_pixels_not_box_area():
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:12, 2:12] = 1          # 100 px box
    ids[2:4, 2:4] = 0            # knock out 4 -> 96 visible
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns[0]["area"] == 96
    x0, y0, x1, y1 = anns[0]["bbox_xyxy"]
    assert (x1 - x0) * (y1 - y0) == 100


def test_sealed_cell_produces_no_annotation():
    """A cell inside an assembled shell contributes zero pixels."""
    ids = np.zeros((32, 32), dtype=np.int32)
    ids[4:28, 4:28] = 1                      # the case
    meta = {1: _meta(**{"class": "cartridge", "role": "case"}),
            2: _meta()}                      # cell id 2 is never drawn
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert [a["pass_index"] for a in anns] == [1]


def test_truncation_flag_on_frame_edge():
    ids = np.zeros((16, 16), dtype=np.int32)
    ids[0:5, 3:9] = 1                        # touches y = 0
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns[0]["truncated"] is True


def test_small_instances_are_dropped_with_a_reason():
    ids = np.zeros((32, 32), dtype=np.int32)
    ids[1, 1] = 1
    cfg = C.FilterCfg(min_px=80, min_side=6)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"].startswith("visible_px<")


def test_thin_sliver_is_dropped_for_short_side_not_low_area():
    """A large-area sliver must still be rejected on its SHORT side. Guards
    against min(w, h) being swapped for max(w, h): under max(), a 100x2
    sliver would pass and ship a 2px-tall box - exactly what a real
    min_side=6 exists to prevent."""
    ids = np.zeros((10, 100), dtype=np.int32)
    ids[0:2, :] = 1                          # 200 px, but only 2 px tall
    cfg = C.FilterCfg(min_px=80, min_side=6)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"] == "side<6"


def test_crop_sliver_is_dropped_for_an_impossible_aspect_ratio():
    """Frame truncation - new with param_space.zoom - produces strips whose
    aspect is a property of where the frame edge fell, not of the object. No
    anchor can ever match one: MEASURED, every box above aspect 3.7 in a
    1245-box sample was truncated, the widest un-cropped box was 3.68, and the
    four above 4.0 scored 0.34-0.45 best-centred-IoU against every anchor set
    tried."""
    ids = np.zeros((200, 40), dtype=np.int32)
    ids[0:180, 2:14] = 1                     # 12 x 180, aspect 15
    cfg = C.FilterCfg(min_px=1, min_side=1, max_aspect=4.0)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"] == "aspect>4.0"


def test_a_whole_cell_at_its_real_aspect_survives_the_filter():
    """The 18650's own 65.0/18.3 = 3.55 must stay comfortably inside the
    limit, or the filter deletes the majority class."""
    ids = np.zeros((200, 60), dtype=np.int32)
    ids[10:81, 10:30] = 1                    # 20 x 71, aspect 3.55
    cfg = C.FilterCfg(min_px=1, min_side=1, max_aspect=4.0)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert dropped == []
    assert len(anns) == 1


def test_max_aspect_zero_disables_the_filter():
    """The dataclass default, so a config predating the filter is unchanged."""
    ids = np.zeros((200, 40), dtype=np.int32)
    ids[0:180, 2:14] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1, max_aspect=0.0)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert len(anns) == 1 and dropped == []


def test_merged_box_is_also_checked_for_an_impossible_aspect():
    """A truncated `cartridge` is merged from its shell halves, and
    merge_group_boxes re-derives the box - so the filter has to be applied
    there too or every cartridge opts out of it."""
    ids = np.zeros((200, 60), dtype=np.int32)
    ids[0:20, 5:25] = 1                      # 20 x 20, aspect 1
    ids[160:180, 5:25] = 2                   # 20 x 20, aspect 1
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    cfg = C.FilterCfg(min_px=1, min_side=1, max_aspect=4.0)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert len(anns) == 2, "each half is square; only the union is a sliver"
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), C.FilterCfg(
                                     min_px=1, min_side=1, max_aspect=0.0))
    assert len(merged) == 1, "control: with the filter off the merge survives"
    assert A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                               C.class_ids(), cfg) == []


def test_the_configured_filters_bound_the_smallest_box_the_anchors_must_cover():
    """min_px and max_aspect were dead thresholds until zoom made truncation
    possible; together they floor the box size at sqrt(min_px), which is what
    lets model.anchor_scales have a smallest value at all. MEASURED: the
    smallest un-cropped silhouette is 1037 px, cropped ones reach 488."""
    f = C.load_config().filter
    assert f.max_aspect >= 3.7, (
        f"max_aspect {f.max_aspect} is at or under the 3.68 widest un-cropped "
        f"box measured, so it would delete whole cells")
    assert f.min_px > 400, (
        f"min_px {f.min_px} does not bound the box size: a corner crop can "
        f"leave a box far below the smallest anchor")
    assert f.min_px < 1037, (
        f"min_px {f.min_px} is at or above the smallest un-cropped silhouette "
        f"measured (1037 px), so it would start deleting whole parts")


def test_drop_truncated_flag_actually_drops():
    ids = np.zeros((16, 16), dtype=np.int32)
    ids[0:5, 3:9] = 1                        # touches y = 0
    cfg = C.FilterCfg(min_px=1, min_side=1, drop_truncated=True)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"] == "truncated"


def test_merge_collapses_an_assembly_into_one_box():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1                      # shell top
    ids[20:30, 20:30] = 2                    # shell bottom
    meta = {1: _meta(), 2: _meta()}
    for m in meta.values():
        m["class"] = "cartridge"
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), cfg)
    assert len(merged) == 1
    assert merged[0]["bbox_xyxy"] == [5, 5, 30, 30]
    assert merged[0]["area"] == 200          # union of silhouettes, not box area


def test_merge_preserves_loose_annotations_in_a_mixed_scene():
    """A realistic scene mixes grouped instances (assembled shells) with
    loose ones (cells_only batteries). Guards against `out = list(loose)`
    being reduced to `out = []`, which would silently delete every loose
    label - e.g. every battery - from the image."""
    ids = np.zeros((60, 60), dtype=np.int32)
    ids[5:15, 5:15] = 1                      # shell top (grouped)
    ids[20:30, 20:30] = 2                    # shell bottom (grouped)
    ids[40:50, 40:50] = 3                    # loose battery cell (ungrouped)
    meta = {1: _meta(**{"class": "cartridge"}),
            2: _meta(**{"class": "cartridge"}),
            3: _meta()}                      # class "battery" by default
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), cfg)
    assert sorted(a["class"] for a in merged) == ["battery", "cartridge"]


def test_merged_box_relabels_to_the_assembly_class():
    """Relabelling the assembly is the whole point of merging: check class
    and category_id directly rather than only the geometry."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1
    ids[20:30, 20:30] = 2
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), cfg)
    assert merged[0]["class"] == "cartridge"
    assert merged[0]["category_id"] == C.class_ids()["cartridge"]


def test_merged_box_below_min_px_is_dropped():
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:8, 5:8] = 1                        # 9 px
    ids[20:23, 20:23] = 2                    # 9 px -> merged area 18
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    cfg = C.FilterCfg(min_px=1, min_side=1)  # each individual box passes
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merge_cfg = C.FilterCfg(min_px=50, min_side=1)  # merged area 18 < 50
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), merge_cfg)
    assert merged == []


def test_merged_truncated_is_any_not_all():
    """One truncated member is enough to flag the merged box - guards
    against any() being weakened to all()."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[0:5, 5:15] = 1                       # touches y = 0 -> truncated
    ids[20:30, 20:30] = 2                    # interior -> not truncated
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"},
                                 C.class_ids(), cfg)
    assert merged[0]["truncated"] is True


def test_unmapped_class_is_dropped():
    ids = np.zeros((16, 16), dtype=np.int32)
    ids[2:10, 2:10] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, dropped = A.boxes_from_mask(
        ids, {1: _meta(**{"class": "widget"})}, C.class_ids(), cfg)
    assert anns == []
    assert dropped[0]["reason"] == "unmapped"


# ------------------------------------------------- unit-scoped cartridge
# boxes (Task 9): boxes_from_mask drops electronics_module/placement_area/
# obstruction as "unmapped" before merge_group_boxes ever runs, so an open
# unit's box has to be rebuilt from the raw mask, after merge_group_boxes -
# that is what extend_group_boxes does. It REBUILDS rather than grows an
# existing box because in practice there rarely IS one: the module/bay are
# rendered flush with the shell's own top surface and cover its whole
# interior (world.build_pcb), so an open unit's shell silhouette is at or
# below the size filter far more often than not - measured over a 32-scene
# run, 0 of 13 open units had a shell that individually passed
# boxes_from_mask on its own. -------------------------------------------- #

def test_merged_box_carries_its_group_id():
    """extend_group_boxes locates/replaces a merged box by `groups[pid]` ==
    `ann["group_id"]`; guard the field it depends on directly."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1
    ids[20:30, 20:30] = 2
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, {1: "item0", 2: "item0"}, C.class_ids(), cfg)
    assert merged[0]["group_id"] == "item0"


def test_extend_group_boxes_is_a_noop_for_a_sealed_cartridge():
    """A sealed unit never has a module/bay/obstruction pid mapped into
    `groups` at all (scene.build only builds those for open_case), so no gid
    ever qualifies as a rebuild target and extend_group_boxes must return
    the merged box exactly as merge_group_boxes produced it - the
    byte-identical-boxes requirement."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1
    ids[20:30, 20:30] = 2
    meta = {1: _meta(**{"class": "cartridge"}), 2: _meta(**{"class": "cartridge"})}
    groups = {1: "item0", 2: "item0"}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, groups, C.class_ids(), cfg)
    before_bbox, before_area = merged[0]["bbox_xyxy"], merged[0]["area"]
    extended = A.extend_group_boxes(merged, ids, meta, groups, C.class_ids(), cfg)
    assert extended is merged, "must return the SAME list object untouched"
    assert extended[0]["bbox_xyxy"] == before_bbox == [5, 5, 30, 30]
    assert extended[0]["area"] == before_area == 200


def test_open_cartridge_box_spans_its_module_bay_and_obstruction():
    """The required outcome, end to end: one cartridge box whose extent is
    the union of the shell, module, bay proxy and an obstruction - matching
    what a human draws in recog/realtest/ - not one box per contributing
    part, and not merely the shell's own (here, filter-failing) box."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:9, 5:9] = 1           # cartridge shell: 4x4=16px, fails min_px alone
    ids[5:12, 16:24] = 2        # electronics_module, OUTSIDE the shell box
    ids[16:24, 5:24] = 3        # placement_area (bay), also outside
    ids[2:4, 2:4] = 4           # obstruction, sticking out the other way
    meta = {
        1: _meta(**{"class": "cartridge"}),
        2: _meta(**{"class": "electronics_module"}),
        3: _meta(**{"class": "placement_area"}),
        4: _meta(**{"class": "obstruction"}),
    }
    groups = {1: "item0", 2: "item0", 3: "item0", 4: "item0"}
    cfg = C.FilterCfg(min_px=80, min_side=1)     # 16px shell alone fails this
    anns, dropped = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert {d["class"] for d in dropped} == {
        "cartridge", "electronics_module", "placement_area", "obstruction"}, (
        "control: the shell itself fails min_px on its own, and the other "
        "three are dropped from `anns` as unmapped - the VOC file must "
        "still contain only battery/cartridge, never their own boxes")
    merged = A.merge_group_boxes(anns, groups, C.class_ids(), cfg)
    assert merged == [], "control: no shell survived, so nothing to merge"

    extended = A.extend_group_boxes(merged, ids, meta, groups, C.class_ids(), cfg)
    assert len(extended) == 1, "one cartridge box per unit, not per part"
    assert extended[0]["class"] == "cartridge"
    assert extended[0]["category_id"] == C.class_ids()["cartridge"]
    assert extended[0]["bbox_xyxy"] == [2, 2, 24, 24]
    # explicitly the containment property the spec asks for
    mx0, my0, mx1, my1 = 16, 5, 24, 12          # module's own box
    bx0, by0, bx1, by1 = 5, 16, 24, 24          # bay's own box
    ex0, ey0, ex1, ey1 = extended[0]["bbox_xyxy"]
    assert ex0 <= mx0 and ey0 <= my0 and ex1 >= mx1 and ey1 >= my1
    assert ex0 <= bx0 and ey0 <= by0 and ex1 >= bx1 and ey1 >= by1


def test_open_cartridge_box_is_built_even_when_the_shell_is_fully_invisible():
    """The dominant real case (see the module comment above): the shell
    contributes ZERO pixels at all (never appears in `ids`, so it never even
    reaches `dropped`) because the module+bay proxy sit flush with its top
    and cover its whole interior. The unit's box must still be built, from
    the module and bay alone."""
    ids = np.zeros((40, 40), dtype=np.int32)
    # pid 1 (the shell) never appears in `ids` at all - fully occluded.
    ids[5:12, 16:24] = 2        # electronics_module
    ids[16:24, 5:24] = 3        # placement_area (bay)
    meta = {
        1: _meta(**{"class": "cartridge"}),
        2: _meta(**{"class": "electronics_module"}),
        3: _meta(**{"class": "placement_area"}),
    }
    groups = {1: "item0", 2: "item0", 3: "item0"}
    cfg = C.FilterCfg(min_px=80, min_side=1, max_aspect=4.0)
    anns, dropped = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert {d["pass_index"] for d in dropped} == {2, 3}, (
        "pid 1 (the shell) never appears in np.unique(ids) at all, so it is "
        "never even a `dropped` entry - unlike pid 2/3, which DO render and "
        "are dropped only as unmapped, this one is simply absent")
    merged = A.merge_group_boxes(anns, groups, C.class_ids(), cfg)
    assert merged == []

    extended = A.extend_group_boxes(merged, ids, meta, groups, C.class_ids(), cfg)
    assert len(extended) == 1
    assert extended[0]["class"] == "cartridge"
    assert extended[0]["bbox_xyxy"] == [5, 5, 24, 24]     # union of 2 and 3 only
    assert extended[0]["merged_from"] == [2, 3]


def test_a_loose_module_with_no_case_produces_no_cartridge_box():
    """A module (or bay/obstruction) whose gid has no cartridge-class member
    at all must not invent one. This is the guard against a loose module
    becoming a spurious cartridge - it cannot happen through scene.build
    (module/bay/obstruction pids are only ever added to `groups` alongside a
    real case item's own pids), but extend_group_boxes must not rely on
    that; it checks directly."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:15, 5:15] = 1
    meta = {1: _meta(**{"class": "electronics_module"})}
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, dropped = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    assert anns == [] and dropped[0]["reason"] == "unmapped"

    for groups in ({1: "item0"}, {}):
        merged = A.merge_group_boxes(anns, groups, C.class_ids(), cfg)
        assert merged == []
        extended = A.extend_group_boxes(merged, ids, meta, groups, C.class_ids(), cfg)
        assert extended == []


def test_extend_group_boxes_below_min_px_is_dropped():
    """The union-level filter still applies: a module+bay too small even
    combined must not produce a box, the same way merge_group_boxes' own
    merged-box filter already works for the shell-only case."""
    ids = np.zeros((40, 40), dtype=np.int32)
    ids[5:7, 5:7] = 2            # electronics_module: 4px
    ids[10:12, 10:12] = 3        # placement_area: 4px -> union area 8
    meta = {
        1: _meta(**{"class": "cartridge"}),   # never rendered - fine
        2: _meta(**{"class": "electronics_module"}),
        3: _meta(**{"class": "placement_area"}),
    }
    groups = {1: "item0", 2: "item0", 3: "item0"}
    cfg = C.FilterCfg(min_px=80, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)
    merged = A.merge_group_boxes(anns, groups, C.class_ids(), cfg)
    extended = A.extend_group_boxes(merged, ids, meta, groups, C.class_ids(), cfg)
    assert extended == []


def test_visible_fraction_is_visible_over_full_area():
    """Guards against the ratio being accidentally inverted to full/visible."""
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[0:10, 0:10] = 1                      # 100 visible px
    cfg = C.FilterCfg(min_px=1, min_side=1, min_visibility=0.0)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg,
                                full_areas={1: 200})
    assert anns[0]["visible_fraction"] == pytest.approx(0.5)


def test_low_visibility_instance_is_dropped():
    """A cell mostly hidden under a shell (assembled / open_case variants)
    must be dropped once visible_fraction falls below min_visibility."""
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[0:10, 0:10] = 1                      # 100 visible px of a 1000 px whole
    cfg = C.FilterCfg(min_px=1, min_side=1, min_visibility=0.25)
    anns, dropped = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg,
                                      full_areas={1: 1000})
    assert anns == []
    assert dropped[0]["reason"] == "visibility<0.25"


# ------------------------------------------------------- VOC output ----

def test_voc_xml_round_trips_through_the_real_loader(tmp_path):
    """The contract is recog.dataset.parse_voc_xml, not a reimplementation."""
    from recog.dataset import CLASS_MAP, parse_voc_xml

    ids = np.zeros((60, 80), dtype=np.int32)
    ids[10:30, 10:40] = 1
    ids[35:50, 50:70] = 2
    meta = {1: _meta(), 2: _meta()}
    meta[2]["class"] = "cartridge"
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, meta, C.class_ids(), cfg)

    xml = tmp_path / "scene_00000.xml"
    A.write_voc_xml(str(xml), "scene_00000.png", 80, 60, anns)

    parsed = parse_voc_xml(xml, CLASS_MAP)
    assert parsed.filename == "scene_00000.png"
    assert parsed.width == 80 and parsed.height == 60
    assert parsed.labels == [1, 2]
    assert parsed.boxes == [(10.0, 10.0, 40.0, 30.0), (50.0, 35.0, 70.0, 50.0)]


def test_voc_survives_an_empty_annotation_list(tmp_path):
    from recog.dataset import CLASS_MAP, parse_voc_xml
    xml = tmp_path / "empty.xml"
    A.write_voc_xml(str(xml), "empty.png", 32, 32, [])
    parsed = parse_voc_xml(xml, CLASS_MAP)
    assert parsed.boxes == [] and parsed.labels == []


def test_voc_writes_integers_not_floats(tmp_path):
    ids = np.zeros((20, 20), dtype=np.int32)
    ids[2:10, 3:12] = 1
    cfg = C.FilterCfg(min_px=1, min_side=1)
    anns, _ = A.boxes_from_mask(ids, {1: _meta()}, C.class_ids(), cfg)
    xml = tmp_path / "i.xml"
    A.write_voc_xml(str(xml), "i.png", 20, 20, anns)
    text = xml.read_text(encoding="utf-8")
    assert "." not in text.split("<bndbox>")[1].split("</bndbox>")[0]


# --------------------------------------------------------- PNG validity ----
# generate3d's `--resume` trusts `is_valid_png` to tell a genuinely
# finished render apart from the ~24KB stub a real OOM-mid-render crash
# left behind (measured - see is_valid_png's own docstring). These pin
# that contract directly, without going through Blender.

def _real_png_bytes(w=8, h=6) -> bytes:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGBA", (w, h), (10, 20, 30, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_is_valid_png_accepts_a_complete_file(tmp_path):
    p = tmp_path / "ok.png"
    p.write_bytes(_real_png_bytes(8, 6))
    assert A.is_valid_png(str(p))
    assert A.is_valid_png(str(p), (8, 6))


def test_is_valid_png_rejects_a_truncated_write(tmp_path):
    """The crash this exists for: `write_still` killed partway through
    leaves a file that starts exactly like a real one and simply stops -
    no IEND, because the encoder writes that chunk last."""
    data = _real_png_bytes(8, 6)
    assert len(data) > 40, "test PNG too small to prove truncation"
    p = tmp_path / "truncated.png"
    p.write_bytes(data[: len(data) - 20])
    assert not A.is_valid_png(str(p))


def test_is_valid_png_rejects_a_zero_byte_file(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    assert not A.is_valid_png(str(p))


def test_is_valid_png_rejects_wrong_dimensions(tmp_path):
    """A structurally complete PNG at the wrong resolution is still not
    the render this scene asked for - e.g. a stale file left by a run at
    a different --res."""
    p = tmp_path / "wrong_size.png"
    p.write_bytes(_real_png_bytes(8, 6))
    assert not A.is_valid_png(str(p), (99, 99))


def test_is_valid_png_rejects_a_missing_file(tmp_path):
    assert not A.is_valid_png(str(tmp_path / "does_not_exist.png"))


def test_is_valid_png_rejects_non_png_bytes(tmp_path):
    p = tmp_path / "not_a_png.png"
    p.write_bytes(b"this is not a png file at all, just plain text" * 4)
    assert not A.is_valid_png(str(p))


# ------------------------------------------------- jig positional bias ----

def test_jig_placements_are_spread_across_the_plate_not_pinned_to_the_top():
    """
    FFDH shelf-packing lays shelves from the top edge down and packs
    left-to-right, so an unrecentred pack pins every part in the plate's
    top-left corner. Before plan_jig recentred the packed block, this exact
    100-seed sweep put 100% of placements in the top half (mean y=+0.152m,
    std=0.020m against a 0.45m-tall plate) and 100% in the top-left quadrant
    - the whole jig subset would have placed every label in a thin band at
    the top of the frame. plan_jig now centres the packed block's bounding
    box on the plate (plus a clamped per-scene jitter), so placements should
    spread across both halves instead.
    """
    cfg = C.load_config().layout
    H = cfg.area[1]
    fps = _real_footprints()

    ys = []
    top_left = 0
    for seed in range(100):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs, _ = L.plan_jig(chosen, cfg, rng)
        for p in plcs:
            if p is None:
                continue
            ys.append(p.y)
            if p.y > 0 and p.x < 0:
                top_left += 1

    n = len(ys)
    assert n > 100, f"only {n} placements sampled across 100 seeds"

    mean_abs_y = sum(abs(y) for y in ys) / n
    # Pre-fix this is ~0.152m (mean y, since every placement is positive);
    # post-fix it measures ~0.073m. 0.45 * H/2 = 0.10125m sits clearly
    # between the two, well inside the plate rather than pinned to an edge.
    assert mean_abs_y < 0.45 * (H / 2), (
        f"mean|y|={mean_abs_y:.4f} is pinned near the plate edge "
        f"(H={H}, half-extent={H / 2})")

    n_top = sum(1 for y in ys if y > 0)
    n_bot = n - n_top
    # Pre-fix: n_bot == 0 - the bottom half is never used at all.
    assert n_top > 0 and n_bot > 0, (
        f"placements only appear in one half of the plate: "
        f"top={n_top} bottom={n_bot} (n={n})")
    assert n_bot / n > 0.2, (
        f"bottom half is only {n_bot}/{n} = {n_bot / n:.3f} of placements - "
        f"still mostly pinned to the top")

    # Pre-fix: top_left/n == 1.0 - every placement in one quadrant.
    assert top_left / n < 0.6, (
        f"top-left quadrant holds {top_left}/{n} = {top_left / n:.3f} of "
        f"placements - still corner-biased")


# ----------------------------------------------------- off-axis lamps ----
#
# The aiming is the one part of the off-axis rig that is easy to get silently
# wrong. A 90-degree error still lights *something*, the render still looks
# plausible, and only comparing shadow direction against the configured
# azimuth across a whole sweep would reveal it. So the rotation is re-derived
# here from independently written matrices rather than by reusing the
# module's own formula.

import math

from recog.synth3d import lightrig as LR


def _rot_xyz(rx, ry, rz):
    """Blender composes rotation_euler XYZ as Rz @ Ry @ Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@pytest.mark.parametrize("az", [0, 37, 90, 143, 180, 271, 359])
@pytest.mark.parametrize("el", [5, 25, 47, 70, 90])
def test_an_off_axis_lamp_points_at_the_thing_it_is_lighting(az, el):
    loc, rot = LR.off_axis_placement(az, el, 1.7)
    emitted = _rot_xyz(*rot) @ np.array(LR.EMIT_AXIS)
    wanted = -np.array(loc)
    wanted = wanted / np.linalg.norm(wanted)
    np.testing.assert_allclose(emitted, wanted, atol=1e-9)


@pytest.mark.parametrize("az", [0, 90, 210])
def test_an_off_axis_lamp_aims_at_a_target_that_is_not_the_origin(az):
    """Nothing uses a non-origin target today, but the offset is the easiest
    thing to drop when someone later aims a lamp at the jig plate."""
    target = (0.12, -0.05, 0.03)
    loc, rot = LR.off_axis_placement(az, 40.0, 1.4, target=target)
    emitted = _rot_xyz(*rot) @ np.array(LR.EMIT_AXIS)
    wanted = np.array(target) - np.array(loc)
    wanted = wanted / np.linalg.norm(wanted)
    np.testing.assert_allclose(emitted, wanted, atol=1e-9)


def test_lamp_position_matches_the_requested_spherical_coordinates():
    loc, _ = LR.off_axis_placement(0.0, 30.0, 2.0)
    assert loc[0] == pytest.approx(2.0 * math.cos(math.radians(30)))
    assert loc[1] == pytest.approx(0.0, abs=1e-12)
    assert loc[2] == pytest.approx(2.0 * math.sin(math.radians(30)))


def test_a_lamp_is_always_above_the_backdrop_plane():
    """Elevation is capped positive precisely so a lamp cannot end up level
    with or under the ground plane, where it would light nothing visible."""
    for el in (0.5, 15.0, 89.0, 90.0):
        loc, _ = LR.off_axis_placement(123.0, el, 1.5)
        assert loc[2] > 0.0, el


@pytest.mark.parametrize("bad", [0.0, -10.0, 90.1, 180.0])
def test_an_unusable_elevation_is_rejected_rather_than_rendered(bad):
    with pytest.raises(ValueError, match="elevation"):
        LR.off_axis_placement(0.0, bad, 1.5)


def test_a_non_positive_distance_is_rejected():
    with pytest.raises(ValueError, match="distance"):
        LR.off_axis_placement(0.0, 45.0, 0.0)


def test_straight_overhead_reduces_to_no_tilt():
    """el = 90 is the degenerate case that should behave like the old
    camera_softbox: lamp directly above, emitting straight down."""
    loc, rot = LR.off_axis_placement(217.0, 90.0, 1.5)
    assert loc[0] == pytest.approx(0.0, abs=1e-12)
    assert loc[1] == pytest.approx(0.0, abs=1e-12)
    assert rot[0] == pytest.approx(0.0)
    emitted = _rot_xyz(*rot) @ np.array(LR.EMIT_AXIS)
    np.testing.assert_allclose(emitted, [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize("az,expected", [
    (0.0, (-1.0, 0.0)), (90.0, (0.0, -1.0)),
    (180.0, (1.0, 0.0)), (270.0, (0.0, 1.0)),
])
def test_shadows_fall_away_from_the_lamp(az, expected):
    np.testing.assert_allclose(LR.shadow_direction(az), expected, atol=1e-12)


def test_azimuth_sweeps_shadows_through_every_direction():
    """The reason azimuth is drawn over the full circle: shadow DIRECTION has
    to vary across the dataset, or the detector can learn 'shadow is always
    down-left' as if it were a property of the parts."""
    dirs = [LR.shadow_direction(a) for a in range(0, 360, 15)]
    assert len({(round(x, 6), round(y, 6)) for x, y in dirs}) == 24
    xs = [d[0] for d in dirs]
    ys = [d[1] for d in dirs]
    assert min(xs) < -0.9 and max(xs) > 0.9
    assert min(ys) < -0.9 and max(ys) > 0.9


def test_seg_class_set_extends_the_detector_set_in_order():
    from recog.synth3d.config import CLASSES, SEG_CLASSES, seg_class_ids

    assert SEG_CLASSES[:len(CLASSES)] == CLASSES, (
        "SEG_CLASSES must start with CLASSES so a shared id means a "
        "shared class between the VOC and COCO outputs")
    assert SEG_CLASSES == ["battery", "cartridge", "electronics_module",
                           "placement_area", "obstruction"]
    assert seg_class_ids() == {
        "battery": 1, "cartridge": 2, "electronics_module": 3,
        "placement_area": 4, "obstruction": 5,
    }
    assert 0 not in seg_class_ids().values(), "0 is reserved for background"


# ------------------------------------------------------- CAD import ----
#
# recog.convert_cad is the entry point for getting new CAD into the asset
# library. Its two safety mechanisms are tested independently, because that
# is how they are meant to work: the unit parser is best-effort, and the
# plausibility guard is what actually stops a mis-scaled asset reaching a
# dataset.
#
# Nothing here converts a real STEP file - the fixtures are hand-written
# headers. cascadio is a native dependency and tessellation is slow, so the
# one test that would need it is skipped when it is absent.

import copy
import importlib.util
import os

from recog import convert_cad as CC


def _step(*unit_entities: str, context: str = "#18,#19,#20") -> str:
    """A minimal STEP file carrying only the entities the parser reads."""
    body = "\n".join(unit_entities)
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_NAME('part.stp','2026-01-01T00:00:00',(''),(''),'x','x','x');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#5= (GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((" + context + "))"
        "REPRESENTATION_CONTEXT('NONE','WORKSPACE'));\n"
        + body + "\n"
        "#19= (NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n"
        "#20= (NAMED_UNIT(*)SOLID_ANGLE_UNIT()SI_UNIT($,.STERADIAN.));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )


# NX / ST-Developer emit the SI form directly, as the four committed Anker
# assemblies do.
_SI_MM = "#18= (NAMED_UNIT(*)LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));"

# SolidWorks / Fusion / ACIS translators emit CONVERSION_BASED_UNIT, with
# generous whitespace and the factor held one reference away. This is the
# shape of the real Lug.stp header.
_CONV_MM = (
    "#18 = ( CONVERSION_BASED_UNIT( 'MILLIMETRE', #28 ) LENGTH_UNIT( ) "
    "NAMED_UNIT( * ) );\n"
    "#28 = LENGTH_MEASURE_WITH_UNIT( LENGTH_MEASURE( 1.000000000000000 ), "
    "#45 );\n"
    "#45 = ( LENGTH_UNIT( ) NAMED_UNIT( * ) SI_UNIT( .MILLI., .METRE. ) );"
)


def test_unit_parser_reads_the_si_declaration():
    unit = CC.parse_step_length_unit(_step(_SI_MM))
    assert unit.metres == pytest.approx(0.001)
    assert unit.label == "MILLIMETRE"
    assert unit.encoding == "SI_UNIT"


def test_unit_parser_reads_the_conversion_based_declaration():
    """The other real encoding: same unit, completely different syntax."""
    unit = CC.parse_step_length_unit(_step(_CONV_MM))
    assert unit.metres == pytest.approx(0.001)
    assert unit.label == "MILLIMETRE"
    assert unit.encoding == "CONVERSION_BASED_UNIT"


def test_both_encodings_of_millimetre_agree():
    """A mm file must scale identically whichever way it says 'mm'."""
    si = CC.parse_step_length_unit(_step(_SI_MM))
    conv = CC.parse_step_length_unit(_step(_CONV_MM))
    assert si.metres == pytest.approx(conv.metres)


def test_unit_parser_reads_bare_metres():
    unit = CC.parse_step_length_unit(
        _step("#18= (NAMED_UNIT(*)LENGTH_UNIT()SI_UNIT($,.METRE.));"))
    assert unit.metres == pytest.approx(1.0)
    assert unit.label == "METRE"


def test_unit_parser_reads_centimetres():
    unit = CC.parse_step_length_unit(
        _step("#18= (NAMED_UNIT(*)LENGTH_UNIT()SI_UNIT(.CENTI.,.METRE.));"))
    assert unit.metres == pytest.approx(0.01)


def test_unit_parser_applies_the_conversion_factor_not_just_the_base():
    """INCH is 25.4 MILLIMETRE; ignoring the factor mis-scales by 25x."""
    unit = CC.parse_step_length_unit(_step(
        "#18 = ( CONVERSION_BASED_UNIT( 'INCH', #28 ) LENGTH_UNIT( ) "
        "NAMED_UNIT( * ) );\n"
        "#28 = LENGTH_MEASURE_WITH_UNIT( LENGTH_MEASURE( 25.4 ), #45 );\n"
        "#45 = ( LENGTH_UNIT( ) NAMED_UNIT( * ) SI_UNIT( .MILLI., .METRE. ) );"))
    assert unit.metres == pytest.approx(0.0254)
    assert unit.label == "INCH"


def test_unit_parser_handles_a_conversion_factor_against_metres():
    unit = CC.parse_step_length_unit(_step(
        "#18 = ( CONVERSION_BASED_UNIT( 'MILLIMETRE', #28 ) LENGTH_UNIT( ) "
        "NAMED_UNIT( * ) );\n"
        "#28 = LENGTH_MEASURE_WITH_UNIT( LENGTH_MEASURE( 1.0E-03 ), #45 );\n"
        "#45 = ( LENGTH_UNIT( ) NAMED_UNIT( * ) SI_UNIT( $, .METRE. ) );"))
    assert unit.metres == pytest.approx(0.001)


def test_unit_parser_reads_entities_split_across_lines():
    """NX wraps long entities; the parser must not be line-oriented."""
    unit = CC.parse_step_length_unit(_step(
        "#18= (NAMED_UNIT(*)\n  LENGTH_UNIT()\n  SI_UNIT(.MILLI.,.METRE.));"))
    assert unit.metres == pytest.approx(0.001)


def test_unit_parser_ignores_semicolons_inside_strings():
    """A ';' in a part name must not terminate the entity early."""
    text = _step(_SI_MM).replace(
        "DATA;", "DATA;\n#99=PRODUCT('a;b','name;with;semis','',(#5));")
    assert CC.parse_step_length_unit(text).metres == pytest.approx(0.001)


def test_unit_parser_ignores_comments():
    text = _step("/* a comment with SI_UNIT(.KILO.,.METRE.) inside */\n"
                 + _SI_MM)
    assert CC.parse_step_length_unit(text).metres == pytest.approx(0.001)


def test_unit_parser_refuses_a_file_with_no_length_unit():
    """Guessing here would silently mis-scale a whole dataset."""
    with pytest.raises(CC.UnitError):
        CC.parse_step_length_unit(_step(
            "#18= (NAMED_UNIT(*)MASS_UNIT()SI_UNIT(.KILO.,.GRAM.));"))


def test_unit_parser_refuses_conflicting_declarations():
    text = _step(
        _SI_MM + "\n"
        "#21= (NAMED_UNIT(*)LENGTH_UNIT()SI_UNIT($,.METRE.));",
        context="#18,#19,#20,#21")
    with pytest.raises(CC.UnitError, match="conflicting"):
        CC.parse_step_length_unit(text)


def test_unit_parser_rejects_a_non_step_file():
    with pytest.raises(CC.UnitError):
        CC.parse_step_length_unit("this is not a STEP file at all")


def test_unit_parser_falls_back_when_there_is_no_global_context():
    """Some translators omit GLOBAL_UNIT_ASSIGNED_CONTEXT; scan everything."""
    text = ("ISO-10303-21;\nDATA;\n" + _SI_MM
            + "\nENDSEC;\nEND-ISO-10303-21;\n")
    assert CC.parse_step_length_unit(text).metres == pytest.approx(0.001)


# ------------------------------------------------- plausibility guard ----

def test_guard_rejects_a_part_scaled_1000x_too_small():
    """The real Lug.stp failure: 125x50x20mm imported as 0.12x0.05x0.02mm."""
    reason = CC.implausible([0.12, 0.05, 0.02])
    assert reason and "below" in reason


def test_guard_rejects_a_part_scaled_1000x_too_large():
    reason = CC.implausible([125000.0, 50000.0, 20000.0])
    assert reason and "above" in reason


def test_guard_accepts_a_real_cell_and_a_real_power_bank():
    assert CC.implausible([18.3, 18.3, 65.0]) is None       # 18650 cell
    assert CC.implausible([81.7, 180.0, 22.2]) is None      # PowerCore 26800


def test_guard_accepts_a_thin_jig_plate():
    """Guards the largest extent only - a 6mm-thick plate is legitimate."""
    assert CC.implausible([400.0, 300.0, 6.0]) is None


def test_guard_rejects_zero_sized_geometry():
    assert CC.implausible([0.0, 0.0, 0.0]) is not None


def test_guard_range_is_configurable():
    assert CC.implausible([600.0, 10.0, 10.0]) is not None
    assert CC.implausible([600.0, 10.0, 10.0], max_mm=1000.0) is None


def test_guard_suggests_the_unit_that_would_fix_the_scale():
    """The Lug case: declared mm, actually authored in metres."""
    declared = CC.LengthUnit(0.001, "MILLIMETRE", "CONVERSION_BASED_UNIT")
    assert "m" in CC.suggest_units([0.12, 0.05, 0.02], declared)


def test_guard_suggests_nothing_when_the_unit_is_unknown():
    assert CC.suggest_units([0.12, 0.05, 0.02], None) == []


# ------------------------------------------------------ role fallback ----

def test_unmatched_subparts_flags_names_no_rule_matches():
    """Silently classing everything as 'case' yields no 'battery' labels."""
    subparts = [{"name": "Cell_18650"}, {"name": "Case10000_top"},
                {"name": "SOLID_BODY_1"}, {"name": "Extrude3"}]
    assert CC.unmatched_subparts(subparts) == ["SOLID_BODY_1", "Extrude3"]


def test_unmatched_subparts_is_empty_for_the_real_anker_names():
    subparts = [{"name": "004695_A;1-Cell_18650"},
                {"name": "004697_A;2-Case10000_top"},
                {"name": "004696_A;2-Case10000_btm"}]
    assert CC.unmatched_subparts(subparts) == []


def test_unmatched_subparts_distinguishes_fallback_from_a_real_case_match():
    """role_of cannot: its fallback role is 'case', which rules also produce."""
    from recog.synth3d.catalog import role_of
    assert role_of("Case10000_btm") == role_of("SOLID_BODY_1") == "case"
    assert CC.unmatched_subparts([{"name": "SOLID_BODY_1"}]) == ["SOLID_BODY_1"]
    assert CC.unmatched_subparts([{"name": "Case10000_btm"}]) == []


# ---------------------------------------------------- catalog merging ----

def _fake_asset(name, extents=(50.0, 60.0, 20.0)):
    return {"name": name, "file": name + ".glb", "source": name + ".stp",
            "extents_mm": list(extents), "triangles": 100,
            "subparts": [], "role_counts": {"case": 1}}


def test_merge_appends_without_dropping_existing_assets():
    existing = {"units": "m", "assets": [_fake_asset("A"), _fake_asset("B")]}
    merged = CC.merge_catalog(existing, [_fake_asset("C")], 0.05, 0.3)
    assert [a["name"] for a in merged["assets"]] == ["A", "B", "C"]


def test_merge_replaces_in_place_rather_than_duplicating():
    existing = {"units": "m",
                "assets": [_fake_asset("A"), _fake_asset("B"), _fake_asset("C")]}
    updated = _fake_asset("B", extents=(11.0, 12.0, 13.0))
    merged = CC.merge_catalog(existing, [updated], 0.05, 0.3)

    names = [a["name"] for a in merged["assets"]]
    assert names == ["A", "B", "C"], "re-import must not duplicate or reorder"
    assert merged["assets"][1]["extents_mm"] == [11.0, 12.0, 13.0]


def test_merge_leaves_untouched_entries_byte_identical():
    existing = {"units": "m", "assets": [_fake_asset("A"), _fake_asset("B")]}
    before = copy.deepcopy(existing)
    merged = CC.merge_catalog(existing, [_fake_asset("C")], 0.05, 0.3)
    for i in (0, 1):
        assert json.dumps(merged["assets"][i], sort_keys=True) == \
            json.dumps(before["assets"][i], sort_keys=True)


def test_merge_preserves_unknown_top_level_keys():
    existing = {"units": "m", "note": "hand-written", "tol_linear_mm": 0.02,
                "assets": [_fake_asset("A")]}
    merged = CC.merge_catalog(existing, [_fake_asset("B")], 0.05, 0.3)
    assert merged["note"] == "hand-written"
    assert merged["tol_linear_mm"] == 0.02, "must not silently retune existing"


def test_merge_creates_a_catalog_when_none_exists():
    merged = CC.merge_catalog(None, [_fake_asset("A")], 0.05, 0.3)
    assert merged["units"] == "m"
    assert merged["tol_linear_mm"] == 0.05
    assert [a["name"] for a in merged["assets"]] == ["A"]


def test_merge_does_not_mutate_the_catalog_it_was_given():
    existing = {"units": "m", "assets": [_fake_asset("A")]}
    CC.merge_catalog(existing, [_fake_asset("B")], 0.05, 0.3)
    assert [a["name"] for a in existing["assets"]] == ["A"]


def test_importing_preserves_the_four_committed_anker_assets(tmp_path):
    """
    The end-to-end merge invariant, run against the real committed catalog.

    The dataset in flight depends on these four entries; an import that
    perturbed any of them would silently change every future render.
    """
    real = json.loads(
        (ROOT / "recog" / "synth3d" / "assets" / "catalog.json").read_text())
    original = copy.deepcopy(real["assets"])
    assert len(original) == 4

    out = tmp_path / "assets"
    out.mkdir()
    (out / "catalog.json").write_text(json.dumps(real, indent=2))

    loaded = CC.load_existing_catalog(str(out))
    merged = CC.merge_catalog(loaded, [_fake_asset("JigFixture")], 0.05, 0.3)
    (out / "catalog.json").write_text(json.dumps(merged, indent=2))

    after = json.loads((out / "catalog.json").read_text())
    assert [a["name"] for a in after["assets"]] == \
        [a["name"] for a in original] + ["JigFixture"]
    for old, new in zip(original, after["assets"]):
        assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)

    # ...and the real catalog on disk is still what it was.
    on_disk = json.loads(
        (ROOT / "recog" / "synth3d" / "assets" / "catalog.json").read_text())
    assert on_disk["assets"] == original


def test_load_existing_catalog_returns_none_for_a_fresh_directory(tmp_path):
    assert CC.load_existing_catalog(str(tmp_path)) is None


# ---------------------------------------------------- name derivation ----

def test_asset_name_strips_the_part_number_prefix():
    """Must reproduce the names already in the committed catalog."""
    assert CC.asset_name_for("004708_A_2-AnkerPowerCore26800.stp") == \
        "AnkerPowerCore26800"


def test_asset_name_passes_through_a_plain_filename():
    assert CC.asset_name_for("Lug.stp") == "Lug"
    assert CC.asset_name_for("JigFixture.step") == "JigFixture"


def test_asset_name_ignores_the_directory():
    assert CC.asset_name_for(os.path.join("cad", "Lug.stp")) == "Lug"


# ----------------------------------------------------------- CLI args ----

def test_cli_rejects_scale_together_with_assume_unit():
    with pytest.raises(SystemExit):
        CC.parse_args(["--src", "cad", "--assume-unit", "m", "--scale", "1000"])


def test_cli_defaults_point_at_the_committed_asset_directory():
    args = CC.parse_args(["--src", "cad"])
    assert Path(args.out) == Path("recog/synth3d/assets")
    assert args.force is False


def test_assume_unit_choices_cover_the_units_the_parser_reports():
    assert set(CC.KNOWN_UNITS) == {"m", "cm", "mm", "in"}
    assert CC.KNOWN_UNITS["in"] == pytest.approx(0.0254)


@pytest.mark.skipif(importlib.util.find_spec("cascadio") is None,
                    reason="cascadio is not installed")
def test_convert_step_is_importable_when_cascadio_is_present():
    """Smoke-check only - tessellating a real STEP file is far too slow here."""
    from recog.synth3d.catalog import convert_step
    assert callable(convert_step)


# --------------------------------------------------------- tray interior

def test_lid_and_tray_are_distinct_roles():
    """open_case cannot drop the lid while both halves share one role."""
    from recog.synth3d.catalog import role_of

    assert role_of("004697_A;2-Case10000_top") == "case_lid"
    assert role_of("004696_A;2-Case10000_btm") == "case"
    assert role_of("004696_A;2-Case10000_btm_1") == "case"
    assert role_of("004695_A;1-Cell_18650") == "cell"


def test_open_case_drops_the_lid_and_assembled_keeps_it():
    from recog.synth3d.config import VARIANTS

    by_name = {v.name: v for v in VARIANTS}
    assert "case_lid" not in by_name["open_case"].keep_roles, (
        "an open cartridge must show its tray, not a closed assembly")
    assert "case" in by_name["open_case"].keep_roles
    assert "case_lid" in by_name["assembled"].keep_roles, (
        "a sealed cartridge must keep both halves")
    assert "case" in by_name["assembled"].keep_roles


def test_open_case_labels_the_tray_as_cartridge():
    from recog.synth3d.config import VARIANTS

    oc = {v.name: v for v in VARIANTS}["open_case"]
    assert oc.label_roles.get("case") == "cartridge"
    assert oc.label_roles.get("cell") == "battery"


# ------------------------------------------------------- cell formats ----

def test_cell_formats_18650_matches_the_authoritative_constant():
    """CELL_W_MM/CELL_H_MM is the one authoritative 18650 figure (commit
    f4596e8) - CELL_FORMATS must derive from it, not restate a third copy."""
    from recog.synth3d.config import CELL_FORMATS, CELL_H_MM, CELL_W_MM
    assert CELL_FORMATS["18650"] == pytest.approx(
        (CELL_W_MM / 1000.0, CELL_H_MM / 1000.0))


def test_cell_formats_has_all_three_decision_3_formats():
    from recog.synth3d.config import CELL_FORMATS
    assert set(CELL_FORMATS) == {"18650", "21700", "26650"}
    assert CELL_FORMATS["21700"] == pytest.approx((0.021, 0.070))
    assert CELL_FORMATS["26650"] == pytest.approx((0.026, 0.065))


# ------------------------------------------------------ procedural tray --

def test_tray_anchored_and_wide_are_separate_instances():
    from recog.synth3d.config import Config
    cfg = Config()
    assert cfg.tray_anchored is not cfg.tray_wide
    assert cfg.tray_anchored.free_bay_edge is False
    assert cfg.tray_wide.free_bay_edge is True


def test_tray_anchored_wall_range_brackets_the_measured_four_skus():
    """Measured (design spec Sec1.1, catalog.json): wall 3.70-4.25mm
    across all four Anker assemblies. Decision 2: anchored stays "within
    and slightly beyond" that span."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.wall_mm_range
    assert lo <= 3.70 and hi >= 4.25


def test_tray_anchored_bay_margin_range_brackets_the_measured_four_skus():
    """Measured module-bay depth: 19.45-30.75mm (design spec Sec1.1's
    anchor table) - the XY quantity Sec5.2 step 3 builds module_bay_mm
    from. See this plan's header note on why it is a separate field from
    case_half_height_mm."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.bay_margin_mm_range
    assert lo <= 19.45 and hi >= 30.75


def test_tray_anchored_case_half_height_is_a_modest_jitter_not_a_wide_draw():
    """Design spec Sec1.1: case half-height is 11.1mm on all four SKUs -
    n=1, not a 4-point spread. The anchored sampler treats this as "near
    the single observed value", not a wide draw."""
    from recog.synth3d.config import Config
    lo, hi = Config().tray_anchored.case_half_height_mm_range
    assert lo < 11.1 < hi
    assert (hi - lo) <= 2.0, f"range {hi - lo}mm is not a modest jitter band"


def test_tray_wide_ranges_are_strictly_wider_than_anchored():
    from recog.synth3d.config import Config
    cfg = Config()
    for field in ("n_cols_range", "n_rows_range", "pitch_mm_range",
                  "wall_mm_range", "bay_margin_mm_range",
                  "case_half_height_mm_range", "tray_floor_mm_range"):
        a_lo, a_hi = getattr(cfg.tray_anchored, field)
        w_lo, w_hi = getattr(cfg.tray_wide, field)
        assert w_lo <= a_lo and w_hi >= a_hi, field


def test_load_config_accepts_tray_sections(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "tray_anchored: {wall_mm_range: [3.5, 4.5]}\n"
        "tray_wide: {free_bay_edge: true}\n", encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg.tray_anchored.wall_mm_range == (3.5, 4.5)
    assert cfg.tray_wide.free_bay_edge is True


def test_synth3d_yaml_cell_formats_load_as_strings_not_yaml_ints():
    """A `cell_formats: [18650, 21700, 26650]` YAML list is unquoted -
    without quotes, PyYAML parses those as ints, not the string keys
    `config.CELL_FORMATS`/`bay.sample_tray` index by. Found during Task
    11's first procedural render: `sample_tray` raised `KeyError: 18650`
    (the int) because the shipped configs/synth3d.yaml had exactly this
    defect. Loaded from the REAL project config, not a synthetic fixture,
    so this pins the actual file, not just the parsing rule."""
    from recog.synth3d.config import CELL_FORMATS, load_config
    cfg = load_config()
    for section in (cfg.tray_anchored, cfg.tray_wide):
        for fmt in section.cell_formats:
            assert isinstance(fmt, str), (
                f"{fmt!r} ({type(fmt).__name__}) is not a string - YAML "
                f"parsed an unquoted numeric-looking list entry as an int")
            assert fmt in CELL_FORMATS
