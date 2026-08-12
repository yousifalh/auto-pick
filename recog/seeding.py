"""recog.seeding — the one place a training run's randomness is fixed.

Before this module existed, training was *genuinely* unseeded: no
``torch.manual_seed``, no ``use_deterministic_algorithms``,
``DataLoader(shuffle=True)`` with no ``generator=``, and ``A.Compose``
with no ``seed=``. Two one-epoch runs of the same command on the same
data gave loss 1.7839 vs 1.7608 and selected mean IoU **0.4111 vs
0.3957** (audit D, 2026-08-12). The ~8 GPU-hours it costs to reproduce
the generalisation result therefore returned a sample from the same
distribution rather than the published figures, and nothing said so.

Three design rules, all of them reactions to how this specific defect
class behaves in this repository.

**One entry point, called before anything stochastic is built.**
``seed_everything`` seeds Python's ``random``, NumPy's legacy global
RNG, torch's CPU generator and every CUDA device. It must run before
the model is constructed (weight init draws from the CPU generator),
before the transforms are built and before the loaders are built.

**Nothing here is allowed to no-op quietly.** A seed that is set but not
used, or a generator that is created and then not passed to the
DataLoader, would leave the run exactly as unreproducible as it was
while every log line claimed otherwise — this project's characteristic
defect in its purest form. So: ``seed_everything`` re-reads
``torch.initial_seed()`` and raises if it did not take;
``assert_loader_seeded`` raises unless the loader actually carries the
generator and ``worker_init_fn`` this module handed out;
``seed_transform`` raises rather than return quietly when it is given an
augmentation pipeline it does not know how to seed. Every one of those
raises :class:`SeedingError`.

**Seeded is not the same claim as bitwise-deterministic, and on this
model seeding alone was not enough.** Seeding fixes every draw the
process makes — measured: identical initial weights and identical input
batches across two processes. It does not fix the arithmetic. With no
determinism flags, cuDNN picks convolution algorithms by heuristic and
several backward kernels accumulate with atomics, so the loss differs at
~1e-7 relative on step 0 and an epoch of SGD amplifies that into the
reported metrics. ``training.deterministic`` therefore defaults to
``"warn"`` rather than off — see :data:`_MODES` for the three modes and
what each was measured to buy, and
``docs/receipts/seed_reproducibility.txt`` for the receipt. What can be
claimed even then is *reproducible on the same machine and toolchain*;
a different GPU, driver, cuDNN or torch build is outside it.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

from common.logging import get_logger

log = get_logger("recog.seeding")

# The default seed, stated explicitly rather than left to chance. It is
# the date this seeding landed, matching scripts/forbidden_bench.py's
# habit of a dated seed. A config that names no seed gets THIS one and
# says so in the log; it never gets one drawn from OS entropy.
DEFAULT_SEED = 20260812

# NumPy's legacy global generator (`np.random.seed`) takes a 32-bit seed.
# torch and `default_rng` accept far more, but a seed that works for two
# of the three and silently wraps for the third is worse than a refusal.
_MAX_SEED = 2 ** 32


class SeedingError(RuntimeError):
    """A seeding step could not be carried out, or did not take effect.

    Deliberately not a warning. Every site that raises this is a site
    where continuing would produce a run that is unseeded in one
    component while reporting itself as seeded.
    """


# ------------------------------------------------------------- config ----

def resolve_seed(cfg: Dict[str, Any], override: Optional[int] = None,
                 section: str = "training") -> int:
    """The seed this run will use: ``--seed`` > config > :data:`DEFAULT_SEED`.

    ``cfg`` is a whole training YAML; the key read is
    ``<section>.seed``. The resolution order is the usual one, and the
    caller is expected to LOG the result — a run that does not record
    which seed it used is only marginally more reproducible than one
    that had none.
    """
    raw: Any
    if override is not None:
        raw = override
    else:
        raw = (cfg.get(section) or {}).get("seed", DEFAULT_SEED)

    if isinstance(raw, bool):
        raise SeedingError(
            f"{section}.seed must be an integer, got the boolean {raw!r}")
    if isinstance(raw, str):
        try:
            raw = int(raw.strip())
        except ValueError:
            raise SeedingError(
                f"{section}.seed must be an integer, got {raw!r}") from None
    if not isinstance(raw, int):
        raise SeedingError(
            f"{section}.seed must be an integer, got {type(raw).__name__} "
            f"({raw!r})")
    if not 0 <= raw < _MAX_SEED:
        raise SeedingError(
            f"{section}.seed must be in [0, {_MAX_SEED}) so that torch, "
            f"NumPy's legacy global RNG and the per-worker seeds all take "
            f"the SAME value; got {raw}")
    return int(raw)


# The three determinism modes, and what each one buys. Measured on this
# repository's segmenter (RTX 3060, torch 2.13+cu126, cuDNN 9.10.2), six
# steps, same seed, two processes — docs/receipts/seed_reproducibility.txt.
#
#   "off"    seeded only. Model init, shuffle order, crop jitter and
#            augmentation are bitwise identical between runs (verified by
#            hashing the weights at init and every input batch), but the
#            arithmetic is not: cuDNN picks convolution algorithms by
#            heuristic and several backward kernels accumulate with
#            atomics, so step 0's loss already differs at ~1e-7 relative
#            and an epoch of SGD amplifies that to ~0.02 in mean loss.
#   "warn"   + cudnn.deterministic and use_deterministic_algorithms(
#            warn_only=True). Every op that HAS a deterministic
#            implementation uses it; the ones that do not warn, loudly,
#            once each. Measured: the weights after six steps are
#            bitwise identical, and the only residue is the loss SCALAR
#            wobbling by 1-2 float32 ULP (the NLL reduction's summation
#            order, which its gradient does not depend on).
#   "strict" the same with warn_only=False. It does not train this
#            model at all: `nll_loss2d_forward_out_cuda_template does not
#            have a deterministic implementation` on the first batch.
#            Kept because it is the correct setting for a model whose ops
#            all have deterministic implementations, and because it is
#            the honest way to ASK the question rather than assume.
_MODES = {
    False: "off", "false": "off", "off": "off", "none": "off", "no": "off",
    True: "strict", "true": "strict", "strict": "strict", "yes": "strict",
    "warn": "warn", "warn_only": "warn",
}


def normalise_deterministic(value: Any) -> str:
    """``False`` / ``"warn"`` / ``True`` -> ``"off"`` / ``"warn"`` /
    ``"strict"``. Anything else raises rather than fall back to "off",
    which would silently drop a determinism request."""
    # bool and str only. A number would be matched by the dict lookup
    # anyway (1.0 == True hashes equal in Python), and `deterministic:
    # 1.0` silently meaning "strict" is the kind of coincidence that
    # makes a config unreadable.
    if not isinstance(value, (bool, str)):
        raise SeedingError(
            f"deterministic must be one of false/warn/true (aliases: "
            f"off/warn_only/strict), got {type(value).__name__} {value!r}")
    key = value.strip().lower() if isinstance(value, str) else value
    try:
        return _MODES[key]
    except KeyError:
        raise SeedingError(
            f"deterministic must be one of false/warn/true (aliases: "
            f"off/warn_only/strict), got {value!r}") from None


def resolve_deterministic(cfg: Dict[str, Any],
                          override: Any = None,
                          section: str = "training") -> str:
    """The determinism mode this run will use. Default ``"warn"``.

    ``"warn"`` and not ``"off"`` because it is what makes the headline
    claim true on the hardware this was measured on — two runs at one
    seed produce the same weights — and not ``"strict"`` because strict
    does not run this model at all (see :data:`_MODES`). It is a
    deliberate middle, and the receipt states what it does and does not
    cover rather than letting "deterministic" stand unqualified.
    """
    if override is not None:
        return normalise_deterministic(override)
    return normalise_deterministic(
        (cfg.get(section) or {}).get("deterministic", "warn"))


# -------------------------------------------------------- the seeding ----

def seed_everything(seed: int, deterministic: Any = "off") -> Dict[str, Any]:
    """Seed every RNG this process draws from, and prove that it took.

    Returns a record of what was set, suitable for logging and for
    writing into a checkpoint. The record's ``fingerprint`` is three
    numbers drawn from the three global RNGs immediately after seeding
    and then *rewound* — they are a pure function of ``seed``, so two
    runs that log different fingerprints did not seed the same way, and
    a run whose seeding silently failed cannot produce the right ones.
    Rewinding matters: the probe must not shift the stream that model
    initialisation then draws from, or this diagnostic would change what
    training does.

    ``deterministic`` selects one of the three modes documented at
    :data:`_MODES`, and every word of that table was measured rather
    than assumed — including the one that says ``"strict"`` cannot train
    this model, which is why it is not the default.

    ``cudnn.benchmark`` is pinned to ``False`` in every mode. That is
    torch's own default, so it is not a behaviour change — it is written
    down so that autotuning cannot become a source of run-to-run kernel
    selection if a future config or library turns it on.
    """
    import torch

    mode = normalise_deterministic(deterministic)

    if mode != "off":
        # Must be set before the first cuBLAS handle is created, which is
        # why seeding runs before the model is built. Without it, cuBLAS
        # GEMMs on CUDA >= 10.2 raise under use_deterministic_algorithms.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    # Set in every branch, including "off": these are process-global and
    # a mode that only ever turns them ON would let one call leak into
    # the next run in the same process (the test suite, a sweep driver)
    # and quietly change what a later run measured.
    torch.backends.cudnn.deterministic = mode != "off"
    torch.use_deterministic_algorithms(mode != "off",
                                       warn_only=(mode == "warn"))

    # ---- proof, not hope ----
    actual = int(torch.initial_seed())
    if actual != seed:
        raise SeedingError(
            f"torch.manual_seed({seed}) did not take: torch.initial_seed() "
            f"reports {actual}. Refusing to continue and call this run "
            "seeded.")

    fingerprint = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": int(torch.randint(0, 2 ** 31 - 1, (1,)).item()),
    }
    # Rewind all three so the probe costs the run nothing.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    record = {
        "seed": int(seed),
        "deterministic": mode,
        "fingerprint": fingerprint,
        # str(), not torch.__version__ itself: that is a TorchVersion
        # object, and a checkpoint carrying one cannot be loaded with
        # weights_only=True - which is exactly how BaySegmenter and the
        # detector loader read best.pt since the 2026-08-12 security
        # pass. A provenance field that breaks the inference path is a
        # bad trade; the string carries the same information.
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_device": (torch.cuda.get_device_name(0)
                        if cuda_available else None),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    log.info(
        "seed=%d deterministic=%s (fingerprint py=%.12f np=%.12f "
        "torch=%d; torch=%s cuda=%s)",
        record["seed"], record["deterministic"], fingerprint["python"],
        fingerprint["numpy"], fingerprint["torch"], record["torch_version"],
        record["cuda_device"] or "cpu")
    if mode == "off":
        log.warning(
            "training.deterministic=off: every RNG is seeded, but the "
            "KERNELS are not constrained - two runs at this seed diverge "
            "from step 1 and an epoch amplifies it to ~0.02 in mean loss on "
            "the reference GPU. Use 'warn' (the default) if you want the "
            "same seed to return the same weights.")
    else:
        log.info(
            "deterministic=%s: reproducible on THIS machine and toolchain. "
            "Not a claim about other GPUs, drivers or library versions. "
            "See recog/seeding.py and docs/receipts/seed_reproducibility.txt.",
            mode)
    return record


# ------------------------------------------------------ the DataLoader ----

def seed_worker(worker_id: int) -> None:  # pragma: no cover - subprocess
    """``worker_init_fn``: seed the libraries torch does not seed for us.

    torch derives each worker's own torch seed from the loader's
    ``generator``, so ``torch.initial_seed()`` inside a worker is already
    unique per worker AND per epoch. What torch does not do is propagate
    that to ``random`` and ``numpy.random``, which every worker process
    otherwise inherits identically from the parent — so with
    ``num_workers > 1`` each worker would draw the SAME augmentation
    stream. Deriving both from the torch seed fixes the duplication and
    the unreproducibility in one step.
    """
    import torch

    worker_seed = int(torch.initial_seed()) % _MAX_SEED
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_kwargs(seed: int) -> Dict[str, Any]:
    """The kwargs a shuffling ``DataLoader`` needs to be reproducible.

    Splat this into the constructor. ``generator=`` is what fixes the
    epoch's shuffle order (without it the sampler builds its own
    generator from OS entropy, which is precisely the defect this
    replaces), and ``worker_init_fn=`` is what makes ``num_workers > 0``
    behave the same way as ``num_workers = 0``.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"generator": generator, "worker_init_fn": seed_worker}


def assert_loader_seeded(loader, what: str,
                         expected_seed: Optional[int] = None) -> None:
    """Raise unless ``loader`` actually received :func:`dataloader_kwargs`.

    The failure this guards against is not hypothetical: a generator
    that is constructed, logged and then not passed to the DataLoader
    leaves the shuffle order exactly as unseeded as before, and every
    other signal in the run still says "seeded". Call this on every
    loader whose order matters, immediately after constructing it.
    """
    generator = getattr(loader, "generator", None)
    if generator is None:
        raise SeedingError(
            f"{what}: DataLoader was built without generator=... — its "
            "shuffle order would come from OS entropy and the run would be "
            "unreproducible while claiming to be seeded. Pass "
            "**recog.seeding.dataloader_kwargs(seed).")
    if getattr(loader, "worker_init_fn", None) is not seed_worker:
        raise SeedingError(
            f"{what}: DataLoader has no recog.seeding.seed_worker as its "
            "worker_init_fn. With num_workers > 0 every worker would then "
            "seed `random` and `numpy` from the parent's inherited state — "
            "identical across workers and different across runs.")
    if expected_seed is not None:
        got = int(generator.initial_seed())
        if got != int(expected_seed):
            raise SeedingError(
                f"{what}: DataLoader generator was seeded with {got}, not "
                f"the run's resolved seed {expected_seed}.")


# ----------------------------------------------------- the augmenter ----

def seed_transform(transform, seed: int, what: str) -> str:
    """Seed an augmentation pipeline. Returns how it was seeded.

    Two pipelines exist in this repository and both are handled:
    albumentations' ``Compose`` (2.x keeps per-instance ``py_random`` /
    ``random_generator``, so ``set_random_seed`` reaches every child
    transform), and ``recog.augmentation``'s numpy-only fallback used
    when albumentations is absent, which carries a plain ``rng``.

    Anything else raises. A photometric pipeline that silently kept its
    entropy-seeded generator would make the run unreproducible in the
    one component that touches every single training sample, and the
    only visible symptom would be a loss curve that does not repeat.
    """
    setter = getattr(transform, "set_random_seed", None)
    if callable(setter):
        setter(int(seed))
        got = getattr(transform, "seed", None)
        if got != int(seed):
            raise SeedingError(
                f"{what}: {type(transform).__name__}.set_random_seed({seed}) "
                f"left .seed = {got!r}; the pipeline is not seeded.")
        return f"{type(transform).__name__}.set_random_seed"

    rng = getattr(transform, "rng", None)
    if isinstance(rng, np.random.Generator):
        transform.rng = np.random.default_rng(int(seed))
        return f"{type(transform).__name__}.rng = default_rng"

    raise SeedingError(
        f"{what}: do not know how to seed {type(transform).__name__} — it "
        "exposes neither set_random_seed() (albumentations) nor an .rng "
        "numpy Generator (recog.augmentation's fallback). Refusing to run a "
        "training job whose augmentation is unseeded.")


# ---------------------------------------------- resume: RNG continuity ----

def capture_rng_state(generator=None) -> Dict[str, Any]:
    """Every RNG's state, for a checkpoint that ``--resume`` will reload.

    Without this, resuming re-seeds from the top of the stream: the
    resumed epochs would replay epoch 0's shuffle order and epoch 0's
    augmentation draws, and a run that was interrupted would not match
    the same run uninterrupted — same seed, same command, different
    result, silently. That is the defect in a new place.
    """
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        "loader": None if generator is None else generator.get_state(),
    }


def restore_rng_state(state: Optional[Dict[str, Any]], generator=None) -> bool:
    """Restore what :func:`capture_rng_state` saved. ``True`` if it did.

    Returns ``False``, loudly, for a ``train_state.pt`` written before
    this existed — those runs were unseeded, so there is no stream to
    continue and the resumed epochs simply start from the seed. A CUDA
    device-count mismatch (resuming on a different machine) is also a
    warning rather than a raise: the run is still seeded, it is just no
    longer the *same* stream, and that is a residual this project
    documents rather than pretends away.
    """
    import torch

    if not state:
        log.warning(
            "train_state.pt carries no RNG state — it was written by an "
            "UNSEEDED run (before 2026-08-12). Resuming re-seeds from the "
            "top of the stream, so the resumed epochs will not be the ones "
            "an uninterrupted run of this config would have produced.")
        return False

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            log.warning(
                "checkpoint carries CUDA RNG state for %d device(s) but this "
                "machine has %d — CUDA streams restart from the seed. The "
                "run is still seeded; it is no longer the same stream as the "
                "interrupted one.", len(cuda_states), torch.cuda.device_count())
    elif cuda_states is not None:
        log.warning("checkpoint carries CUDA RNG state but this run has no "
                    "CUDA; CUDA streams are not restored.")

    if generator is not None and state.get("loader") is not None:
        generator.set_state(state["loader"])
    elif generator is not None:
        log.warning("checkpoint carries no DataLoader generator state; the "
                    "resumed epochs replay the first epoch's shuffle order.")
    return True


__all__ = [
    "DEFAULT_SEED",
    "SeedingError",
    "resolve_seed",
    "resolve_deterministic",
    "seed_everything",
    "seed_worker",
    "dataloader_kwargs",
    "assert_loader_seeded",
    "seed_transform",
    "capture_rng_state",
    "restore_rng_state",
]
