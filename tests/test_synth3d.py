"""Unit tests for the Blender-free half of recog.synth3d.

Every module touched here must import without bpy, so this file runs in
the ordinary pytest environment.
"""
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

# catalog.py, layout.py and annotate.py are added by later tasks (3-5); only
# test modules that actually exist yet, so this file stays green at every
# task boundary instead of erroring on files that haven't landed.
_BPY_FREE_CANDIDATES = ["config", "catalog", "layout", "annotate"]
_BPY_FREE_MODS = [m for m in _BPY_FREE_CANDIDATES
                   if (ROOT / "recog" / "synth3d" / f"{m}.py").is_file()]


@pytest.mark.parametrize("mod", _BPY_FREE_MODS)
def test_pure_modules_never_import_bpy(mod):
    src = (ROOT / "recog" / "synth3d" / f"{mod}.py").read_text(encoding="utf-8")
    assert "import bpy" not in src, f"{mod}.py must stay Blender-free"


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
    assert set(cfg.role_materials) == {"case", "cell"}
    assert cfg.lighting["overcast_softbox"]["kind"] == "camera_softbox"


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
    assert set(v.keep_roles) == {"cell", "case"}


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
    """All 33 sub-part names from the real CAD must classify correctly."""
    cat = CAT.load_catalog(str(ASSETS))
    seen = 0
    for asset in cat["assets"]:
        for sp in asset["subparts"]:
            role = CAT.role_of(sp["name"])
            assert role == sp["role"], f"{sp['name']}: {role} != {sp['role']}"
            assert role in ("cell", "case")
            seen += 1
    assert seen == 33, f"expected 33 sub-parts, catalogued {seen}"


def test_cell_regex_matches_nx_incremented_names():
    """NX renames instances; a literal 'Cell_18650' match misses these."""
    for name in ("004695_A;1-Cell_18650", "004695_A;2-Cell_18651",
                 "004695_A;3-Cell_18650_18652", "Cell_99999"):
        assert CAT.role_of(name) == "cell", name


def test_case_names_classify_as_case():
    for name in ("004697_A;2-Case10000_top", "004697_A;1-Case26800_btm"):
        assert CAT.role_of(name) == "case", name


def test_unknown_subpart_falls_back_to_case():
    assert CAT.role_of("something_unrecognised") == "case"


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


def test_scatter_never_overlaps_over_300_scenes():
    cfg = C.load_config().layout
    fps = _real_footprints()
    total = 0
    for seed in range(300):
        rng = random.Random(seed)
        chosen = [rng.choice(fps) for _ in range(rng.randint(1, 4))]
        plcs = L.plan(chosen, cfg, rng)
        boxes = [_aabb(f, p, cfg) for f, p in zip(chosen, plcs) if p is not None]
        total += len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j]), f"seed {seed}"
    assert total > 500, f"only {total} items placed across 300 scenes"


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


def test_cluster_offsets_ring():
    rng = random.Random(3)
    offs = L.cluster_offsets(6, 0.03, rng)
    assert len(offs) == 6
    for dx, dy in offs:
        assert 0.0 < (dx * dx + dy * dy) ** 0.5 < 0.03 * 1.5
