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

_BPY_FREE_CANDIDATES = ["config", "catalog", "layout", "annotate"]
_BPY_FREE_MODS = [m for m in _BPY_FREE_CANDIDATES
                   if (ROOT / "recog" / "synth3d" / f"{m}.py").is_file()]


def test_every_bpy_free_module_is_actually_checked():
    """The existence filter above was scaffolding from when these modules
    landed one task at a time. All four exist now, so it must be a no-op:
    without this assertion, renaming or moving a module would silently drop
    it from the boundary check below and the parametrised test would still
    pass, having tested one fewer file."""
    assert _BPY_FREE_MODS == _BPY_FREE_CANDIDATES, (
        f"these modules were skipped by the bpy-boundary check: "
        f"{sorted(set(_BPY_FREE_CANDIDATES) - set(_BPY_FREE_MODS))}")


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
    assert role_of("Case10000_top") == role_of("SOLID_BODY_1") == "case"
    assert CC.unmatched_subparts([{"name": "SOLID_BODY_1"}]) == ["SOLID_BODY_1"]
    assert CC.unmatched_subparts([{"name": "Case10000_top"}]) == []


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
