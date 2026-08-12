"""Prove that segmenter training is seeded — and say what it is not.

Runs `recog.seg_training` three times on `configs/segmentation_seedcheck
.yaml`: twice at one seed and once at another, one epoch each. Two runs
at the same seed must agree; the run at a different seed must not. The
result is written to `docs/receipts/seed_reproducibility.txt`, which is
the receipt behind every "reproducible" claim in the documentation.

Usage::

    python scripts/seed_check.py                  # writes the receipt
    python scripts/seed_check.py --no-write       # prints only
    python scripts/seed_check.py --seed A --other B

Three quantities per run, deliberately, because they fail in that
order as reproducibility weakens:

* **mean training loss** and **selected mean IoU** — the two numbers
  audit D measured moving (1.7839 vs 1.7608, 0.4111 vs 0.3957) when
  training was unseeded. These are what a reader compares against a
  published figure.
* **a SHA-256 over the saved weights** — bitwise equality of the model
  itself, which is the strict claim and the one that fails first. At
  `training.deterministic: off` it fails while the run is fully seeded,
  because seeding fixes the draws and not the arithmetic; that is the
  measurement that made `warn` the default. Whatever this script sees on
  the machine it is run on is what the receipt says.

The exit status is 0 when the same-seed pair agrees on the metrics and
the different-seed run differs; 1 otherwise. A weight-hash mismatch is
reported but does not by itself fail the run — it is a statement about
kernels, and the receipt names which mode produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # same as scripts/forbidden_bench.py

CONFIG = "configs/segmentation_seedcheck.yaml"
RECEIPT = ROOT / "docs" / "receipts" / "seed_reproducibility.txt"

# `epoch=0 loss=1.7839 val_iou={...} selected([...] over counts={...})=0.4111`
_EPOCH_RE = re.compile(r"epoch=(\d+) loss=([0-9.]+).*?\)=([0-9.]+)\s*$")
_SEED_RE = re.compile(r"seed=(\d+) deterministic=(\w+)")
_FINGERPRINT_RE = re.compile(r"fingerprint py=([0-9.]+) np=([0-9.]+) "
                             r"torch=(\d+)")


def _weights_sha256(path: Path) -> str:
    """SHA-256 over every saved tensor, in sorted key order."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_once(seed: int, ckpt_dir: Path, config: str = CONFIG,
             deterministic: Optional[str] = None) -> Dict:
    """One 1-epoch training run at ``seed``; returns what it produced."""
    cmd = [sys.executable, "-m", "recog.seg_training",
           "--config", config, "--seed", str(seed)]
    if deterministic is not None:
        cmd += ["--deterministic", deterministic]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"error: training at seed {seed} failed "
                         f"(exit {proc.returncode})")

    out: Dict = {"seed": seed, "loss": None, "iou": None, "wall_s": wall,
                 "logged_seed": None, "fingerprint": None, "sha256": None}
    for line in proc.stdout.splitlines():
        m = _EPOCH_RE.search(line.strip())
        if m:
            out["loss"], out["iou"] = float(m.group(2)), float(m.group(3))
        m = _SEED_RE.search(line)
        if m:
            out["logged_seed"], out["deterministic"] = int(m.group(1)), m.group(2)
        m = _FINGERPRINT_RE.search(line)
        if m:
            out["fingerprint"] = (m.group(1), m.group(2), m.group(3))
    if out["loss"] is None:
        raise SystemExit("error: no epoch line in the training output; the "
                         "log format this script parses has changed")
    if out["logged_seed"] != seed:
        raise SystemExit(f"error: asked for seed {seed}, the run logged "
                         f"{out['logged_seed']}")
    out["sha256"] = _weights_sha256(ckpt_dir / "last.pt")
    return out


def _env_lines() -> Dict[str, str]:
    import torch

    return {
        "torch": torch.__version__,
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                   else "cpu"),
        "cuda": (torch.version.cuda or "-") if torch.cuda.is_available() else "-",
        "cudnn": str(torch.backends.cudnn.version()
                     if torch.cuda.is_available() else "-"),
    }


def render(a: Dict, b: Dict, c: Dict, env: Dict[str, str],
           config: str = CONFIG) -> str:
    same_metrics = (a["loss"] == b["loss"] and a["iou"] == b["iou"])
    same_weights = a["sha256"] == b["sha256"]
    differs = (a["loss"] != c["loss"] or a["iou"] != c["iou"])

    lines = [
        "Training reproducibility: same seed twice, then a different seed",
        "=" * 66,
        f"  command       : python scripts/seed_check.py",
        f"  config        : {config}  (1 epoch; NOT the 40-epoch schedule)",
        f"  torch         : {env['torch']}  cuda {env['cuda']}  "
        f"cudnn {env['cudnn']}",
        f"  device        : {env['device']}",
        f"  deterministic : {a.get('deterministic')}  "
        "(training.deterministic; see recog/seeding.py)",
        "",
        "Before this existed, training was genuinely unseeded and two runs of",
        "the same command gave loss 1.7839 vs 1.7608 and selected mean IoU",
        "0.4111 vs 0.3957 (docs/superpowers/audit/2026-08-12-D-reproducibility",
        ".md). The published checkpoints came from that era and cannot be",
        "recovered; seeding applies from now on.",
        "",
        "  run   seed        loss      selected IoU   wall s   weights sha256",
    ]
    for label, r in (("A", a), ("B", b), ("C", c)):
        lines.append(f"  {label}     {r['seed']:<11d} {r['loss']:.4f}    "
                     f"{r['iou']:.4f}         {r['wall_s']:6.1f}   "
                     f"{r['sha256'][:16]}")
    lines += [
        "  (wall s is the whole process, ~15 s of which is interpreter and",
        "   torch import; it is a cost signal, not a benchmark.)",
        "",
        f"  A vs B (same seed {a['seed']}):",
        f"    loss delta           : {abs(a['loss'] - b['loss']):.6f}",
        f"    selected IoU delta   : {abs(a['iou'] - b['iou']):.6f}",
        f"    metrics identical    : {'yes' if same_metrics else 'NO'}",
        f"    weights bit-identical: {'yes' if same_weights else 'NO'}",
        "",
        f"  A vs C (seed {a['seed']} vs {c['seed']}):",
        f"    loss delta           : {abs(a['loss'] - c['loss']):.6f}",
        f"    selected IoU delta   : {abs(a['iou'] - c['iou']):.6f}",
        f"    differs              : {'yes' if differs else 'NO'}",
        "",
        "WHAT THIS CLAIMS, EXACTLY. Same seed, same machine, same toolchain ->",
        "the same weights. It is not a claim about a different GPU, driver,",
        "cuDNN or torch build, and it is not a claim about the published",
        "checkpoints, which were trained before any of this existed.",
        "",
        "The three determinism modes, all measured on the hardware named above",
        "(six steps, two processes, docs/superpowers/specs/2026-08-12-seeded-",
        "training.md has the tables):",
        "  off    seeded only. Model init and every input batch are bitwise",
        "         identical, but the ARITHMETIC is not: cuDNN chooses",
        "         convolution algorithms by heuristic and several backward",
        "         kernels accumulate with atomics. Step 0's loss already",
        "         differs at ~1e-7 relative, and one epoch amplifies it into",
        "         the reported metrics: two same-seed pairs measured at 0.0197",
        "         and 0.0076 apart in mean loss, 0.0101 and 0.0409 in selected",
        "         IoU - the size of the UNSEEDED spread. Seeding alone is not",
        "         enough on this hardware, which is why it is not the default.",
        "  warn   the default, and what the table above was produced under.",
        "         cudnn.deterministic plus use_deterministic_algorithms(",
        "         warn_only=True): every op that has a deterministic",
        "         implementation uses it, and the ones that do not say so on",
        "         stderr rather than silently.",
        "  strict warn_only=False. Does not train this model at all:",
        "         `nll_loss2d_forward_out_cuda_template does not have a",
        "         deterministic implementation` on the first batch. That op's",
        "         nondeterminism is in the forward REDUCTION (its gradient does",
        "         not depend on the summation order), which is why `warn` still",
        "         reproduces the weights bit for bit.",
        "",
        "The train/val SPLIT is seeded separately (dataset.split_seed) and does",
        "not follow the run seed: it defines what the metric is measured on, so",
        "changing the seed must not change the yardstick.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--other", type=int, default=20260813)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--deterministic", choices=["off", "warn", "strict"],
                    default=None,
                    help="override the config's determinism mode for all "
                         "three runs (the receipt records which was used)")
    ap.add_argument("--no-write", action="store_true",
                    help="print the receipt instead of writing it")
    args = ap.parse_args(argv)

    from common.config import load_yaml

    cfg = load_yaml(str(ROOT / args.config))
    ckpt_dir = ROOT / cfg["training"]["checkpoint_dir"]
    if ckpt_dir.resolve() == (ROOT / "recog/checkpoints/seg").resolve():
        raise SystemExit(
            "error: this probe would overwrite the published checkpoints in "
            "recog/checkpoints/seg. They were trained unseeded and cannot be "
            "recovered. Point training.checkpoint_dir somewhere else.")

    if args.seed == args.other:
        raise SystemExit("error: --seed and --other must differ; the whole "
                         "point is the comparison between them")

    a = run_once(args.seed, ckpt_dir, args.config, args.deterministic)
    b = run_once(args.seed, ckpt_dir, args.config, args.deterministic)
    c = run_once(args.other, ckpt_dir, args.config, args.deterministic)

    text = render(a, b, c, _env_lines(), args.config)
    print(text)
    if not args.no_write:
        RECEIPT.write_text(text, encoding="utf-8")
        print(f"[written] {RECEIPT}")

    ok = (a["loss"] == b["loss"] and a["iou"] == b["iou"]
          and (a["loss"] != c["loss"] or a["iou"] != c["iou"]))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
