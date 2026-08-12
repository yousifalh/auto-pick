"""Checkpoint selection: the hard validation subset, and the last.pt rule.

`recog.training` keeps every torch import inside a function, so the selection
logic tested here runs in the ordinary pytest environment.
"""
import ast
import json
from pathlib import Path

import pytest

from recog import training as T

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- scene sidecars ---

def _meta(tmp_path, stem, exposure=-4.0, zoom=1.0, boxes=None, lift=0.0):
    """Write one meta/<stem>.json of the shape generate3d emits."""
    d = tmp_path / "meta"
    d.mkdir(exist_ok=True)
    payload = {
        "params": {"exposure": exposure, "zoom": zoom,
                   "backdrop": "concrete", "lighting": "warm_indoor"},
        "items": [{"asset": "A", "variant": "cells_only",
                   "placement": {"x": 0, "y": 0, "quarter": 0,
                                 "rot_deg": 0, "z": lift}}],
        "annotations": [{"class": "battery", "bbox_xyxy": list(b)}
                        for b in (boxes or [(0, 0, 10, 10)])],
    }
    (d / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def _corpus(tmp_path, n=40):
    """n scenes spanning exposure and zoom, with overlap on a known few."""
    d = None
    for i in range(n):
        d = _meta(
            tmp_path, f"scene_{i:05d}",
            exposure=-5.2 + 2.0 * (i / (n - 1)),
            zoom=0.75 + 0.85 * (i / (n - 1)),
            # every 7th scene has two boxes that intersect
            boxes=([(0, 0, 20, 20), (10, 10, 30, 30)] if i % 7 == 0
                   else [(0, 0, 20, 20), (40, 40, 60, 60)]),
        )
    return d


# ------------------------------------------------------------- hardness ----

def test_hardness_rewards_dark_small_and_occluded():
    """The three axes the design names, each on its own."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _meta(tmp, "easy", exposure=-3.2, zoom=0.75)
        _meta(tmp, "dark", exposure=-5.2, zoom=0.75)
        _meta(tmp, "small", exposure=-3.2, zoom=1.60)
        d = _meta(tmp, "occluded", exposure=-3.2, zoom=0.75,
                  boxes=[(0, 0, 20, 20), (10, 10, 30, 30)])
        stems = ["easy", "dark", "small", "occluded"]
        s = dict(zip(stems, T.hardness_scores(stems, d)))
    assert s["dark"] > s["easy"]
    assert s["small"] > s["easy"]
    assert s["occluded"] > s["easy"]
    assert s["easy"] == pytest.approx(0.0)


def test_a_scene_hard_on_every_axis_outranks_one_hard_on_any_single_axis(tmp_path):
    d = _meta(tmp_path, "easy", exposure=-3.2, zoom=0.75)
    _meta(tmp_path, "dark", exposure=-5.2, zoom=0.75)
    _meta(tmp_path, "all", exposure=-5.2, zoom=1.60,
          boxes=[(0, 0, 20, 20), (10, 10, 30, 30)])
    stems = ["easy", "dark", "all"]
    s = dict(zip(stems, T.hardness_scores(stems, d)))
    assert s["all"] > s["dark"] > s["easy"]
    assert s["all"] == pytest.approx(3.0)


def test_a_z_lift_counts_as_overlap_even_when_the_boxes_do_not_intersect(tmp_path):
    """A part resting ON another can be fully hidden, so its box may not be in
    the file at all - the placement lift is the direct record that the scene
    contains occlusion."""
    d = _meta(tmp_path, "flat", lift=0.0)
    _meta(tmp_path, "stacked", lift=0.0183)
    stems = ["flat", "stacked"]
    s = dict(zip(stems, T.hardness_scores(stems, d)))
    assert s["stacked"] - s["flat"] == pytest.approx(1.0)


def test_selection_takes_the_hardest_and_respects_the_fraction(tmp_path):
    d = _corpus(tmp_path, n=40)
    stems = [f"scene_{i:05d}" for i in range(40)]
    picks = T.select_hard_subset(stems, d, fraction=0.25)
    assert len(picks) == 10
    scores = T.hardness_scores(stems, d)
    chosen = {scores[i] for i in picks}
    rest = {scores[i] for i in range(40) if i not in picks}
    assert min(chosen) >= max(rest)


def test_selection_is_ordered_hardest_first(tmp_path):
    d = _corpus(tmp_path, n=40)
    stems = [f"scene_{i:05d}" for i in range(40)]
    picks = T.select_hard_subset(stems, d, fraction=0.5)
    scores = T.hardness_scores(stems, d)
    got = [scores[i] for i in picks]
    assert got == sorted(got, reverse=True)


def test_a_tiny_validation_split_still_yields_a_usable_subset(tmp_path):
    """0.25 of 12 scenes is 3, which is not an mAP. The floor is what stops
    the selection signal being a single image."""
    d = _corpus(tmp_path, n=12)
    stems = [f"scene_{i:05d}" for i in range(12)]
    picks = T.select_hard_subset(stems, d, fraction=0.25, min_scenes=8)
    assert len(picks) == 8
    # ...and never more scenes than exist.
    assert len(T.select_hard_subset(stems[:4], d, fraction=0.25,
                                    min_scenes=8)) == 4


def test_no_meta_directory_selects_nothing_rather_than_the_first_n(tmp_path):
    """Falling back to filename order would be worse than the saturating
    metric it replaced, because it would look like it was working."""
    assert T.select_hard_subset(["a", "b"], tmp_path / "absent") == []


def test_a_corpus_with_no_variation_selects_nothing(tmp_path):
    """An older dataset whose meta records no zoom, no exposure spread and no
    overlap has no hard subset to find, and must say so."""
    d = None
    for i in range(20):
        d = _meta(tmp_path, f"s{i}", exposure=-3.5, zoom=1.0)
    for p in d.glob("*.json"):
        m = json.loads(p.read_text())
        m["params"].pop("zoom")
        m["params"].pop("exposure")
        p.write_text(json.dumps(m))
    assert T.select_hard_subset([f"s{i}" for i in range(20)], d) == []


def test_an_unreadable_or_missing_sidecar_scores_zero_rather_than_raising(tmp_path):
    """One corrupt sidecar must not take a training run down at epoch 0."""
    d = _meta(tmp_path, "hard", exposure=-5.2, zoom=1.6)
    _meta(tmp_path, "easy", exposure=-3.2, zoom=0.75)
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    scores = T.hardness_scores(["hard", "easy", "broken", "absent"], d)
    assert scores[0] > 0.0
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == 0.0 and scores[3] == 0.0


def test_ties_do_not_hand_hardness_to_the_easiest_scenes(tmp_path):
    """Three scenes, two of them tied at the easy end: the tied pair must
    score 0 on that axis, not the 0.5 that position-in-sorted-order gives."""
    d = _meta(tmp_path, "a", exposure=-3.2, zoom=0.75)
    _meta(tmp_path, "b", exposure=-3.2, zoom=0.75)
    _meta(tmp_path, "c", exposure=-3.2, zoom=1.60)
    s = T.hardness_scores(["a", "b", "c"], d)
    assert s[0] == pytest.approx(0.0) and s[1] == pytest.approx(0.0)
    assert s[2] == pytest.approx(1.0)


# ------------------------------------------------------ the last.pt rule ----

def _train_body():
    tree = ast.parse((ROOT / "recog" / "training.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "train")


def _saves_of(fn):
    """(node, ancestors) for every torch.save call inside `fn`."""
    out = []

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "save"):
                out.append((child, list(chain)))
            walk(child, chain + [child])

    walk(fn, [])
    return out


def test_last_pt_is_saved_unconditionally_every_epoch():
    """
    This is the fix that made a +39% localisation gain recoverable, and it is
    structural rather than behavioural, so it is asserted structurally. The
    full-set synthetic metric saturates at mAP 1.0 within a couple of epochs;
    before this, every later epoch trained and was then thrown away. Putting
    the last.pt write back under any condition - including the new hard-subset
    metric, which is only a proxy for the real photos - would silently
    reintroduce it.
    """
    fn = _train_body()
    saves = [(c, chain) for c, chain in _saves_of(fn)
             if "last.pt" in ast.dump(c)]
    assert len(saves) == 1, f"expected exactly one last.pt write, got {len(saves)}"
    _, chain = saves[0]
    guards = [n for n in chain if isinstance(n, (ast.If, ast.Try, ast.While))]
    assert not guards, (
        f"the last.pt write sits inside {[type(g).__name__ for g in guards]}; "
        f"it must run every epoch regardless of any metric")
    assert any(isinstance(n, ast.For) for n in chain), (
        "the last.pt write is not inside the epoch loop")


def test_best_pt_is_the_one_that_is_conditional():
    """Control for the test above: if neither save were guarded, that test
    would pass while checkpoint selection had quietly stopped happening."""
    fn = _train_body()
    best = [(c, chain) for c, chain in _saves_of(fn)
            if "best.pt" in ast.dump(c)]
    assert len(best) == 1
    assert any(isinstance(n, ast.If) for n in best[0][1])


# --------------------------------------------------------- frozen BatchNorm ---
#
# There was no test of freeze_batchnorm at all, which is how it stayed
# one-way: `requires_grad` was set to False at epoch 0 and nothing anywhere
# set it back, so gamma/beta were pinned for all 35 epochs while
# configs/recognition.yaml said `frozen_bn_epochs: 8`. The knob has been
# removed rather than repaired — BN is now frozen, both halves, for the
# whole run — and these pin that decision so it cannot drift back into a
# half-live config key.


def test_freeze_batchnorm_freezes_both_halves_and_says_how_many():
    torch = pytest.importorskip("torch")
    from recog.model import freeze_batchnorm

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(), torch.nn.BatchNorm2d(4),
    )
    model.train()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(m.training and m.weight.requires_grad for m in bns), "premise"

    assert freeze_batchnorm(model) == 2, (
        "freeze_batchnorm must report how many modules it froze, so a caller "
        "can tell 'froze everything' from 'found nothing to freeze'")
    for m in bns:
        assert not m.training, "running statistics are still updating"
        assert not m.weight.requires_grad, "gamma is still receiving gradient"
        assert not m.bias.requires_grad, "beta is still receiving gradient"


def test_freeze_batchnorm_finding_nothing_to_freeze_is_visible():
    """A GroupNorm backbone would make this a silent no-op; the count says so."""
    torch = pytest.importorskip("torch")
    from recog.model import freeze_batchnorm

    assert freeze_batchnorm(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))) == 0


def test_train_one_epoch_refreezes_batchnorm_unconditionally_every_epoch():
    """``model.train()`` un-freezes BN, so the call must follow it, always.

    Structural, because the failure is a call that stops happening rather
    than one that returns the wrong thing — and because the version of this
    defect that shipped was exactly a freeze that ran under a condition.
    """
    src = (ROOT / "recog" / "training.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train_one_epoch")

    calls = []

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "freeze_batchnorm"):
                calls.append(chain)
            walk(child, chain + [child])

    walk(fn, [])
    assert len(calls) == 1, f"expected one freeze_batchnorm call, got {len(calls)}"
    guards = [n for n in calls[0] if isinstance(n, (ast.If, ast.While))]
    assert not guards, (
        f"the freeze sits inside {[type(g).__name__ for g in guards]}. A "
        "conditional freeze is what produced the half-frozen state audit G "
        "found: statistics thawing at epoch 8 while gamma/beta never did.")
    assert "frozen_bn" not in {a.arg for a in fn.args.args}, (
        "train_one_epoch still takes a frozen_bn flag")


def test_no_recognition_config_still_carries_the_removed_bn_knob():
    """A documented parameter that does nothing is worse than no parameter."""
    from common.config import load_yaml

    offenders = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = load_yaml(str(path)) or {}
        if "frozen_bn_epochs" in (cfg.get("training") or {}):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} set training.frozen_bn_epochs, which nothing reads. "
        "BatchNorm is frozen for the whole run — see recog/model.py::"
        "freeze_batchnorm.")
    # And nothing still READS the key. Checked against calls rather than raw
    # text, because the docstrings deliberately name it to explain why it
    # is gone — the point is that no code looks it up.
    tree = ast.parse((ROOT / "recog" / "training.py")
                     .read_text(encoding="utf-8"))
    lookups = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and any(isinstance(a, ast.Constant)
                       and a.value == "frozen_bn_epochs" for a in n.args)]
    assert not lookups, (
        f"recog/training.py still looks up frozen_bn_epochs at line(s) "
        f"{[n.lineno for n in lookups]}")


def test_the_configured_dataset_points_meta_dir_at_a_real_sidecar_directory():
    """A typo here degrades silently to the saturating metric plus a warning
    nobody reads."""
    from common.config import load_yaml
    cfg = load_yaml(str(ROOT / "configs" / "recognition.yaml"))
    ds = cfg["dataset"]
    assert ds.get("meta_dir"), "dataset.meta_dir is unset"
    assert 0.0 < float(ds.get("hard_val_fraction", 0.25)) <= 1.0
    # The sidecars live beside the images the same generator wrote.
    assert Path(ds["meta_dir"]).parent == Path(ds["img_dir"]).parent
