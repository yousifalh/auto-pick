"""recog.seeding — and the proof that the trainers actually use it.

The defect this module was written against is not "the seed is wrong",
it is "the seed is set and then nothing reads it". So roughly half of
what is asserted here is that a *missing* seeding step raises instead of
passing quietly, and the structural tests at the bottom check that the
two training loops still call every step — a `train()` that stops
splatting `dataloader_kwargs` into its DataLoader would otherwise go on
logging a resolved seed while shuffling from OS entropy, and every test
above would still be green.
"""
import ast
import random
from pathlib import Path

import numpy as np
import pytest

from recog.augmentation import _ALB_AVAILABLE, build_seg_train_transform
from recog.seeding import (DEFAULT_SEED, SeedingError, _rng_fingerprint,
                           _walk_pipeline, assert_loader_seeded,
                           capture_rng_state, dataloader_kwargs,
                           normalise_deterministic, resolve_deterministic,
                           resolve_seed, restore_rng_state, seed_everything,
                           seed_transform, seed_worker)

ROOT = Path(__file__).resolve().parents[1]

requires_alb = pytest.mark.skipif(
    not _ALB_AVAILABLE, reason="albumentations is not installed")


def _find(pipeline, name):
    """The first transform of class ``name`` anywhere in ``pipeline``."""
    for node in _walk_pipeline(pipeline):
        if type(node).__name__ == name:
            return node
    raise AssertionError(f"{name} not found in the pipeline")


# ------------------------------------------------------- resolve_seed ----

def test_a_config_with_no_seed_gets_the_explicit_default():
    assert resolve_seed({}) == DEFAULT_SEED
    assert resolve_seed({"training": {}}) == DEFAULT_SEED


def test_the_config_seed_is_read_and_the_override_beats_it():
    cfg = {"training": {"seed": 7}}
    assert resolve_seed(cfg) == 7
    assert resolve_seed(cfg, override=99) == 99


def test_seed_zero_is_a_seed_not_a_missing_value():
    """0 is falsy; a `or DEFAULT` implementation would silently ignore it."""
    assert resolve_seed({"training": {"seed": 0}}) == 0
    assert resolve_seed({}, override=0) == 0


def test_a_quoted_seed_is_accepted_and_a_nonsense_one_is_refused():
    assert resolve_seed({"training": {"seed": "1234"}}) == 1234
    with pytest.raises(SeedingError):
        resolve_seed({"training": {"seed": "the usual"}})


@pytest.mark.parametrize("bad", [True, 1.5, None, -1, 2 ** 32])
def test_a_seed_that_is_not_a_usable_integer_raises(bad):
    """Including the two silent-corruption cases: a bool (which is an int
    in Python) and a value NumPy's 32-bit global seed would wrap."""
    with pytest.raises(SeedingError):
        resolve_seed({"training": {"seed": bad}})


def test_the_determinism_mode_defaults_to_warn_and_is_overridable():
    """`warn`, not `off`: seeding alone leaves two same-seed runs
    diverging on CUDA (docs/receipts/seed_reproducibility.txt)."""
    assert resolve_deterministic({}) == "warn"
    assert resolve_deterministic({"training": {"deterministic": False}}) == "off"
    assert resolve_deterministic({"training": {"deterministic": True}}) == "strict"
    assert resolve_deterministic({"training": {"deterministic": True}},
                                 override="off") == "off"


@pytest.mark.parametrize("given,expected", [
    (False, "off"), ("false", "off"), ("off", "off"), ("None", "off"),
    (True, "strict"), ("true", "strict"), ("STRICT", "strict"),
    ("warn", "warn"), ("warn_only", "warn"), (" Warn ", "warn"),
])
def test_every_spelling_of_the_three_modes_normalises(given, expected):
    assert normalise_deterministic(given) == expected


@pytest.mark.parametrize("bad", ["yes please", 2, 1.0, None, ["warn"]])
def test_an_unrecognised_determinism_setting_raises_rather_than_defaulting(bad):
    """Falling back to "off" here would silently drop a determinism
    request - the run would report itself seeded and reproduce nothing."""
    with pytest.raises(SeedingError):
        normalise_deterministic(bad)


# ---------------------------------------------------- seed_everything ----

def test_seeding_is_a_pure_function_of_the_seed():
    pytest.importorskip("torch")
    a = seed_everything(4321)["fingerprint"]
    b = seed_everything(4321)["fingerprint"]
    c = seed_everything(1234)["fingerprint"]
    assert a == b
    assert a != c
    # All three RNGs move, not just torch's - a fingerprint that agreed
    # on numpy while differing on python would mean one of them was
    # never seeded.
    assert all(a[k] != c[k] for k in ("python", "numpy", "torch"))


def test_the_fingerprint_probe_does_not_shift_the_stream():
    """The diagnostic must not change what training then draws."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(2024)
    expected = torch.rand(4)
    seed_everything(2024)
    assert torch.equal(expected, torch.rand(4))


def test_the_record_says_what_it_did():
    pytest.importorskip("torch")
    rec = seed_everything(11, deterministic=False)
    assert rec["seed"] == 11
    assert rec["deterministic"] == "off"
    assert rec["cudnn_benchmark"] is False  # autotuning pinned off
    assert "torch_version" in rec


def test_the_record_survives_a_weights_only_checkpoint_round_trip(tmp_path):
    """The record goes into best.pt, and BaySegmenter loads best.pt with
    weights_only=True. `torch.__version__` unstringified is a
    TorchVersion object and would make every new checkpoint unloadable
    by the only inference path in the project - caught exactly once, by
    this."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {}, "seeding": seed_everything(11)}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["seeding"]["seed"] == 11


# ------------------------------------------------------- the DataLoader --

def _loader(seed, **overrides):
    import torch

    kwargs = dict(dataloader_kwargs(seed))
    kwargs.update(overrides)
    return torch.utils.data.DataLoader(
        list(range(24)), batch_size=4, shuffle=True, **kwargs)


def test_same_seed_gives_the_same_epoch_order_and_a_different_seed_does_not():
    pytest.importorskip("torch")
    order = lambda ldr: [int(i) for b in ldr for i in b]  # noqa: E731
    assert order(_loader(5)) == order(_loader(5))
    assert order(_loader(5)) != order(_loader(6))


def test_a_loader_built_without_the_generator_is_refused():
    torch = pytest.importorskip("torch")
    bare = torch.utils.data.DataLoader(list(range(8)), shuffle=True)
    with pytest.raises(SeedingError, match="without generator"):
        assert_loader_seeded(bare, "train loader")


def test_a_loader_that_lost_the_worker_init_fn_is_refused():
    """num_workers=0 today does not excuse it: raising the worker count
    later must not silently unseed `random` and `numpy` per worker."""
    pytest.importorskip("torch")
    ldr = _loader(5, worker_init_fn=None)
    with pytest.raises(SeedingError, match="worker_init_fn"):
        assert_loader_seeded(ldr, "train loader")


def test_a_loader_seeded_with_the_wrong_seed_is_refused():
    pytest.importorskip("torch")
    ldr = _loader(5)
    with pytest.raises(SeedingError, match="seeded with 5"):
        assert_loader_seeded(ldr, "train loader", expected_seed=6)


def test_a_correctly_seeded_loader_passes():
    pytest.importorskip("torch")
    assert_loader_seeded(_loader(5), "train loader", expected_seed=5) is None


def test_seed_worker_seeds_python_and_numpy_from_the_torch_seed():
    torch = pytest.importorskip("torch")
    import random

    torch.manual_seed(777)
    seed_worker(0)
    first = (random.random(), float(np.random.random()))
    torch.manual_seed(777)
    seed_worker(0)
    assert first == (random.random(), float(np.random.random()))


# -------------------------------------------------------- the augmenter --

class _Unseedable:
    """Neither albumentations' API nor recog.augmentation's fallback."""


class _LyingTransform:
    """set_random_seed() that accepts the call and does nothing.

    The exact shape of the defect: a seeding step that returns cleanly
    and leaves the pipeline unseeded.
    """

    seed = None

    def set_random_seed(self, seed):
        return None


def test_an_unseedable_pipeline_raises_rather_than_train_unseeded():
    with pytest.raises(SeedingError, match="do not know how to seed"):
        seed_transform(_Unseedable(), 3, "train augmentation")


def test_a_seed_call_that_does_not_take_is_caught():
    with pytest.raises(SeedingError, match="not seeded"):
        seed_transform(_LyingTransform(), 3, "train augmentation")


def test_the_numpy_fallback_augmenter_is_seeded_through_its_rng():
    from recog.augmentation import _FallbackSegTransform

    tf = _FallbackSegTransform({}, train=True)
    how = seed_transform(tf, 12, "train augmentation")
    assert "default_rng" in how
    assert tf.rng.bit_generator.state == np.random.default_rng(12).\
        bit_generator.state


@requires_alb
def test_seeding_albumentations_makes_the_pipeline_repeat():
    """The load-bearing one: same seed, identical augmented pixels."""
    img = np.random.default_rng(0).integers(
        0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)

    def run(seed):
        tf = build_seg_train_transform({})
        seed_transform(tf, seed, "train augmentation")
        return tf(image=img.copy(), mask=mask.copy())["image"]

    assert np.array_equal(run(31), run(31))
    assert not np.array_equal(run(31), run(32))


@requires_alb
def test_every_child_transform_gets_its_own_stream_not_a_shared_one():
    """Reproducible is only half the contract; independent is the other half.

    albumentations 2.0.8's ``Compose.set_random_seed`` propagates ONE seed
    to every child, so a single call leaves all 17 transforms in the
    detector pipeline with identical RNG states. Two transforms that draw
    the same way then fire in lockstep forever - measured as the dihedral
    group collapsing from 8 orientations to 4 (audit G, finding 1).
    """
    from recog.augmentation import build_train_transform

    tf = build_train_transform({})
    seed_transform(tf, 4242, "train augmentation")

    nodes = _walk_pipeline(tf)
    assert len(nodes) > 10, f"expected the whole pipeline, walked {len(nodes)}"
    states = [_rng_fingerprint(n) for n in nodes]
    carried = [s for s in states if s is not None]
    assert len(set(carried)) == len(carried), (
        f"{len(carried)} transforms share only {len(set(carried))} distinct "
        "RNG state(s)")

    # And the pair that actually caused it, by behaviour rather than state.
    h = _find(tf, "HorizontalFlip")
    v = _find(tf, "VerticalFlip")
    h_draws = [h.py_random.random() for _ in range(8)]
    v_draws = [v.py_random.random() for _ in range(8)]
    assert h_draws != v_draws, (
        "HorizontalFlip and VerticalFlip draw the same stream, so they fire "
        "together or not at all and the horizontal-only and vertical-only "
        "mirrors never occur")


@requires_alb
def test_the_same_seed_still_gives_the_same_child_streams():
    """Independent, and a pure function of the run seed - not of entropy."""
    from recog.augmentation import build_train_transform

    def states(seed):
        tf = build_train_transform({})
        seed_transform(tf, seed, "train augmentation")
        return [_rng_fingerprint(n) for n in _walk_pipeline(tf)]

    assert states(4242) == states(4242)
    assert states(4242) != states(4243)


class _CollidingCompose:
    """A Compose whose children all end up with the same RNG, as 2.0.8 did.

    ``set_random_seed`` here ignores the seed it is given for the children,
    which is the shape of the regression: the call succeeds, ``.seed`` is
    right on the root, and every child still draws the same stream.
    """

    class _Child:
        def __init__(self):
            self.py_random = random.Random(0)
            self.random_generator = np.random.default_rng(0)
            self.seed = None

        def set_random_seed(self, seed):
            self.seed = seed
            self.py_random = random.Random(0)
            self.random_generator = np.random.default_rng(0)

    def __init__(self):
        self.seed = None
        self.py_random = random.Random(1)
        self.random_generator = np.random.default_rng(1)
        self.transforms = [self._Child(), self._Child()]

    def set_random_seed(self, seed):
        self.seed = seed
        for t in self.transforms:
            t.set_random_seed(seed)


def test_a_pipeline_whose_children_share_a_stream_is_refused():
    """The loud failure the old code did not have.

    This is the exact defect that shipped: seeding that returns success
    having done half its job. If a future albumentations changes the
    per-child seeding contract again, this raises instead of quietly
    halving the augmentation group.
    """
    with pytest.raises(SeedingError, match="distinct RNG state"):
        seed_transform(_CollidingCompose(), 7, "train augmentation")


def test_a_nested_transform_that_cannot_be_seeded_is_refused():
    class _Parent:
        seed = None
        py_random = None
        random_generator = None

        def __init__(self):
            self.transforms = [_Unseedable()]

        def set_random_seed(self, seed):
            self.seed = seed

    with pytest.raises(SeedingError, match="no set_random_seed"):
        seed_transform(_Parent(), 7, "train augmentation")


@requires_alb
def test_an_unseeded_albumentations_pipeline_would_not_repeat():
    """Guards the premise: if this ever passed, the seeding above would
    be measuring nothing."""
    img = np.random.default_rng(1).integers(
        0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    outs = [build_seg_train_transform({})(image=img.copy(), mask=mask.copy())
            ["image"] for _ in range(6)]
    assert any(not np.array_equal(outs[0], o) for o in outs[1:])


# ------------------------------------------------------ resume: streams --

def test_rng_state_round_trips_so_resume_continues_the_stream():
    torch = pytest.importorskip("torch")
    import random

    seed_everything(8)
    gen = torch.Generator()
    gen.manual_seed(8)
    state = capture_rng_state(gen)
    expected = (random.random(), float(np.random.random()),
                torch.rand(1).item(), torch.rand(1, generator=gen).item())

    # Move every stream on, then wind them back.
    random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=gen)
    assert restore_rng_state(state, gen) is True
    assert expected == (random.random(), float(np.random.random()),
                        torch.rand(1).item(),
                        torch.rand(1, generator=gen).item())


def test_a_checkpoint_from_the_unseeded_era_says_so_instead_of_pretending():
    pytest.importorskip("torch")
    assert restore_rng_state(None) is False


# ------------------------------------------------------ the shipped configs

def _training_configs():
    return sorted(list((ROOT / "configs").glob("segmentation*.yaml"))
                  + [ROOT / "configs" / "recognition.yaml"])


@pytest.mark.parametrize("path", _training_configs(),
                         ids=lambda p: p.name)
def test_every_training_config_names_its_seed_and_a_valid_mode(path):
    """Named, not inherited. A config that relies on DEFAULT_SEED is
    reproducible but does not SAY what it will do, and these files are
    what a reader reaches for first."""
    from common.config import load_yaml

    cfg = load_yaml(str(path))
    assert "seed" in cfg["training"], f"{path.name} names no training.seed"
    assert isinstance(resolve_seed(cfg), int)
    assert resolve_deterministic(cfg) in {"off", "warn", "strict"}


# ------------------------------------------------- the wiring, structurally

def _train_fn(module_path: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / module_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError(f"no train() in {module_path}")


def _called_names(node) -> list:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            out.append(fn.id if isinstance(fn, ast.Name)
                       else getattr(fn, "attr", ""))
    return out


@pytest.mark.parametrize("module_path",
                         ["recog/seg_training.py", "recog/training.py"])
def test_both_trainers_call_every_seeding_step(module_path):
    called = _called_names(_train_fn(module_path))
    for step in ("resolve_seed", "seed_everything", "seed_transform",
                 "dataloader_kwargs", "assert_loader_seeded"):
        assert step in called, f"{module_path}: train() never calls {step}()"


@pytest.mark.parametrize("module_path",
                         ["recog/seg_training.py", "recog/training.py"])
def test_the_shuffling_loader_receives_the_seeding_kwargs(module_path):
    """`**dataloader_kwargs(seed)` must land on the DataLoader that
    shuffles - built next to it and not passed through is the whole
    failure mode."""
    fn = _train_fn(module_path)
    seeded_shuffling_loaders = 0
    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
        if getattr(call.func, "attr", "") != "DataLoader":
            continue
        shuffles = any(kw.arg == "shuffle" and getattr(kw.value, "value", None)
                       is True for kw in call.keywords)
        splatted = any(
            kw.arg is None and isinstance(kw.value, ast.Call)
            and getattr(kw.value.func, "id", "") == "dataloader_kwargs"
            for kw in call.keywords)
        if shuffles:
            assert splatted, (f"{module_path}: a shuffling DataLoader is "
                              "built without **dataloader_kwargs(seed)")
            seeded_shuffling_loaders += 1
    assert seeded_shuffling_loaders == 1


@pytest.mark.parametrize("module_path",
                         ["recog/seg_training.py", "recog/training.py"])
def test_every_checkpoint_written_records_the_seed(module_path):
    """A checkpoint whose seed is not recorded cannot be re-derived, and
    this project's checkpoints are the expensive artefact."""
    fn = _train_fn(module_path)
    payloads = 0
    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
        name = (getattr(call.func, "id", "")
                or getattr(call.func, "attr", ""))
        if name not in ("_atomic_save", "save"):
            continue
        payload = call.args[0] if call.args else None
        if not isinstance(payload, ast.Dict):
            continue
        keys = [k.value for k in payload.keys if isinstance(k, ast.Constant)]
        assert "seed" in keys, (f"{module_path}: a checkpoint payload with "
                                f"keys {keys} does not record the seed")
        payloads += 1
    assert payloads >= 2  # best.pt and last.pt at minimum
