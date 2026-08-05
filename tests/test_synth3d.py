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
    """Inside Blender there is no PyYAML, so the .json sidecar is the path."""
    src = C.load_config()
    j = tmp_path / "synth3d.json"
    j.write_text(json.dumps(C.config_to_dict(src)), encoding="utf-8")
    monkeypatch.setattr(C, "_HAVE_YAML", False)
    cfg = C.load_config(tmp_path / "synth3d.yaml")
    assert cfg.render.res == src.render.res
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
