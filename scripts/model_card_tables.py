"""Generate the tables in `docs/MODEL_CARD.md` and `docs/datasets/README.md`.

Every table this script emits is derived from a committed artefact:

  * `docs/receipts/seg_eval*.txt`  - written by `recog.seg_evaluate`
  * `docs/receipts/frcnn_*.txt`    - the detector's evaluation receipts
  * `configs/segmentation*.yaml`   - the training configs, one per model
  * `docs/datasets/*.manifest.json` - the generator manifests, copied here by
    `--sync-datasets` from the gitignored dataset directories

Nothing is typed by hand, so the card cannot drift from the receipts without
`--check` saying so. That is this repository's own standard applied to its
most-read page: a number in the card is a number in a receipt, or it is a bug.

Usage
-----
    python scripts/model_card_tables.py                 # rewrite the tables
    python scripts/model_card_tables.py --check         # fail on drift
    python scripts/model_card_tables.py --sync-datasets # refresh docs/datasets

`--sync-datasets` is the only mode that reads outside the tracked tree: it
copies each `recog/dataset3d*/manifest.json` byte-for-byte into
`docs/datasets/` and records the SHA-256 of both the manifest and the
`instances_seg.json` beside it in `docs/datasets/checksums.json`. It needs the
datasets present and is therefore the author's command, not a cloner's; the
other two modes run on a bare clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = ROOT / "docs" / "receipts"
CONFIGS = ROOT / "configs"
DATASETS_DOC = ROOT / "docs" / "datasets"
MODEL_CARD = ROOT / "docs" / "MODEL_CARD.md"
DATASET_INDEX = DATASETS_DOC / "README.md"

# The order models are presented in, and the display name for each checkpoint
# directory. Ordering is editorial (shipping first, then the two procedural
# arms, then the two diagnostics, then the four controls); every value in the
# rows is not.
MODEL_ORDER = [
    ("recog/checkpoints/seg", "seg (shipping)"),
    ("recog/checkpoints/seg_anchored", "anchored"),
    ("recog/checkpoints/seg_wide", "wide"),
    ("recog/checkpoints/seg_anchored_18650", "anchored_18650"),
    ("recog/checkpoints/seg_anchored_crown", "anchored_crown"),
    ("recog/checkpoints/seg_cad_control_AnkerPowerCore10000", "cad_control_10000"),
    ("recog/checkpoints/seg_cad_control_AnkerPowerCore13000", "cad_control_13000"),
    ("recog/checkpoints/seg_cad_control_AnkerPowerCore20100", "cad_control_20100"),
    ("recog/checkpoints/seg_cad_control_AnkerPowerCore26800", "cad_control_26800"),
]

SKUS = [
    "AnkerPowerCore10000",
    "AnkerPowerCore13000",
    "AnkerPowerCore20100",
    "AnkerPowerCore26800",
]

# Which SKU each leave-one-SKU-out control never saw. Read off the checkpoint
# directory name rather than asserted.
HELD_OUT_RE = re.compile(r"seg_cad_control_(AnkerPowerCore\d+)$")


# --------------------------------------------------------------------------
# receipt parsing
# --------------------------------------------------------------------------

class Receipt:
    """One `recog.seg_evaluate` receipt, parsed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        text = path.read_text(encoding="utf-8")
        self.text = text

        self.checkpoint = self._field(r"^\s*checkpoint\s*:\s*(\S+)")
        self.config = self._field(r"^\s*config\s*:\s*(\S+)")
        self.synth_config = self._field(r"^\s*synth config\s*:\s*(\S+)")
        self.crops = int(self._field(r"^\s*validation crops\s*:\s*(\d+)"))

        # Per-class pooled IoU and the instance count behind each one.
        self.ious: dict[str, float] = {}
        self.instances: dict[str, int] = {}
        for cls, iou, n in re.findall(
            r"^  (background|cartridge|bay|electronics|obstruction|battery)"
            r"\s+([\d.]+|nan)\s+(\d+)\s*$",
            text,
            re.MULTILINE,
        ):
            self.ious[cls] = float(iou)
            self.instances[cls] = int(n)

        m = re.search(
            r"selected mean IoU over (\[[^\]]*\]) \(instances=(\{[^}]*\})\): ([\d.]+)",
            text,
        )
        if m is None:
            raise ValueError(f"{path.name}: no selected mean IoU line")
        self.select_on = [c.strip("' ") for c in m.group(1).strip("[]").split(",")]
        self.selected_instances = {
            k.strip("' "): int(v)
            for k, v in (p.split(":") for p in m.group(2).strip("{}").split(","))
        }
        self.selected_mean = float(m.group(3))

        # The checkpoint-selection note carries the training-time own-val
        # figure for best.pt and last.pt, and the val instance counts the
        # selection was made on - the only place those travel in a committed
        # artefact, since the checkpoints themselves are gitignored.
        m = re.search(
            r"best\.pt ([\d.]+) and last\.pt ([\d.]+) differ by ([\d.]+) "
            r"on (\d+)/(\d+)/(\d+) \(bay/electronics/obstruction\)",
            text,
        )
        if m is None:
            raise ValueError(f"{path.name}: no checkpoint-selection note")
        self.sel_best = float(m.group(1))
        self.sel_last = float(m.group(2))
        self.sel_delta = float(m.group(3))
        self.sel_val_instances = tuple(int(m.group(i)) for i in (4, 5, 6))

        # Per-SKU IoU, present only on the cad_test receipts.
        self.per_sku: dict[str, dict[str, float]] = {}
        for row in re.findall(
            r"^  (AnkerPowerCore\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)"
            r"\s+([\d.]+)\s+([\d.]+)\s*$",
            text,
            re.MULTILINE,
        ):
            self.per_sku[row[0]] = {
                "n_crops": int(row[1]),
                "bay": float(row[2]),
                "electronics": float(row[3]),
                "obstruction": float(row[4]),
                "battery": float(row[5]),
            }

    def _field(self, pattern: str) -> str:
        m = re.search(pattern, self.text, re.MULTILINE)
        if m is None:
            raise ValueError(f"{self.path.name}: no match for {pattern!r}")
        return m.group(1)

    @property
    def is_cad_test(self) -> bool:
        return self.config.endswith("segmentation_cad_test.yaml")


def load_receipts() -> list[Receipt]:
    return [Receipt(p) for p in sorted(RECEIPTS.glob("seg_eval*.txt"))]


# --------------------------------------------------------------------------
# config parsing
# --------------------------------------------------------------------------

def load_training_configs() -> dict[str, dict]:
    """Map checkpoint_dir -> the training config that writes it.

    A config counts as a *training* config only if it actually trains: an
    eval-only config (`train_val_split: 0.0`, i.e. segmentation_cad_test.yaml)
    and the one-epoch seeding harness both name a checkpoint_dir they do not
    own. Excluding them mechanically rather than by name keeps the mapping
    injective, which is asserted below rather than assumed.
    """
    out: dict[str, dict] = {}
    for path in sorted(CONFIGS.glob("segmentation*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        ds, tr = cfg.get("dataset", {}), cfg.get("training", {})
        ckpt = tr.get("checkpoint_dir")
        if ckpt is None:
            continue
        if float(ds.get("train_val_split", 0.0)) == 0.0:
            continue                     # eval-only config
        if ckpt.endswith("seedcheck"):
            continue                     # scratch dir, one epoch
        if ckpt in out:
            raise ValueError(
                f"two training configs claim {ckpt}: "
                f"{out[ckpt]['config']} and {path.name}"
            )
        out[ckpt] = {
            "config": path.name,
            "coco_path": ds.get("coco_path"),
            "train_val_split": ds.get("train_val_split"),
            "split_seed": ds.get("split_seed"),
            "jitter_frac": ds.get("jitter_frac"),
            "crop_size": cfg.get("model", {}).get("crop_size"),
            "half": cfg.get("model", {}).get("half"),
            "epochs": tr.get("epochs"),
            "batch_size": tr.get("batch_size"),
            "learning_rate": tr.get("learning_rate"),
            "lr_scheduler": tr.get("lr_scheduler"),
            "dice_weight": tr.get("dice_weight"),
            "seed": tr.get("seed"),
            "deterministic": tr.get("deterministic"),
            "select_on": tr.get("select_on"),
        }
    return out


def dataset_name(coco_path: str | None) -> str | None:
    if not coco_path:
        return None
    return Path(coco_path).parent.name


def load_manifests() -> dict[str, dict]:
    out = {}
    for p in sorted(DATASETS_DOC.glob("*.manifest.json")):
        out[p.name[: -len(".manifest.json")]] = json.loads(
            p.read_text(encoding="utf-8")
        )
    return out


def load_checksums() -> dict:
    p = DATASETS_DOC / "checksums.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------
# table builders
# --------------------------------------------------------------------------

def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def table_comparison(by_ckpt_cad, by_ckpt_own, manifests, configs) -> str:
    """The consolidated held-out comparison - one row per trained segmenter."""
    lines = [
        "| model | trained on | scenes | `bay` | `electronics` | `obstruction` "
        "| **selected mean IoU** |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for ckpt, label in MODEL_ORDER:
        cfg = configs.get(ckpt, {})
        ds = dataset_name(cfg.get("coco_path"))
        scenes = manifests.get(ds, {}).get("stats", {}).get("n_images")
        r = by_ckpt_cad.get(ckpt)
        if r is None:
            cells = ["—", "—", "—", "**—**"]
        else:
            cells = [
                fmt(r.ious.get("bay")),
                fmt(r.ious.get("electronics")),
                fmt(r.ious.get("obstruction")),
                f"**{r.selected_mean:.4f}**",
            ]
        lines.append(
            f"| `{label}` | `{ds or '—'}` | {scenes if scenes else '—'} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def table_per_sku(by_ckpt_cad) -> str:
    """Per-SKU `bay` IoU on the CAD test set, held-out SKU marked."""
    lines = [
        "| model | " + " | ".join(
            f"`{s.replace('AnkerPowerCore', 'PC')}`" for s in SKUS) + " |",
        "|---|" + "---:|" * len(SKUS),
    ]
    for ckpt, label in MODEL_ORDER:
        r = by_ckpt_cad.get(ckpt)
        if r is None or not r.per_sku:
            lines.append(f"| `{label}` |" + " — |" * len(SKUS))
            continue
        m = HELD_OUT_RE.search(ckpt)
        held = m.group(1) if m else None
        cells = []
        for sku in SKUS:
            v = r.per_sku.get(sku, {}).get("bay")
            cells.append(f"**{v:.4f}** †" if sku == held else fmt(v))
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")

    # The leave-one-out diagonal, computed here rather than quoted: each
    # control's score on the one SKU it never trained on. In how many folds
    # the held-out SKU is the worst in its own row is the interesting part,
    # and it is counted rather than asserted.
    diag, worst = [], 0
    for ckpt, _ in MODEL_ORDER:
        m = HELD_OUT_RE.search(ckpt)
        r = by_ckpt_cad.get(ckpt)
        if not m or r is None or not r.per_sku:
            continue
        row = {s: r.per_sku[s]["bay"] for s in SKUS if s in r.per_sku}
        v = row[m.group(1)]
        diag.append(v)
        if v == min(row.values()):
            worst += 1
    if diag:
        lines.append(
            f"\n† the SKU that fold never trained on. The four held-out values "
            f"are {', '.join(f'{v:.4f}' for v in diag)} — mean "
            f"**{sum(diag) / len(diag):.4f}** — and the held-out SKU is the "
            f"worst cell in its own row in **{worst} of {len(diag)}** folds."
        )
    return "\n".join(lines)


def table_selection(by_ckpt_cad, by_ckpt_own) -> str:
    """Checkpoint selection: the training-time metric, its noise floor, and
    `seg_evaluate`'s re-score of the same checkpoint on the same split."""
    lines = [
        "| model | own-val n (bay/elec/obs) | `best.pt` | `last.pt` | Δ | "
        "re-scored | gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ckpt, label in MODEL_ORDER:
        r = by_ckpt_cad.get(ckpt) or by_ckpt_own.get(ckpt)
        if r is None:
            continue
        own = by_ckpt_own.get(ckpt)
        rescore = f"{own.selected_mean:.4f}" if own else "—"
        gap = f"{r.sel_best - own.selected_mean:+.4f}" if own else "—"
        n = "/".join(str(x) for x in r.sel_val_instances)
        lines.append(
            f"| `{label}` | {n} | {r.sel_best:.4f} | {r.sel_last:.4f} | "
            f"{r.sel_delta:.4f} | {rescore} | {gap} |"
        )
    return "\n".join(lines)


def table_training(configs, manifests) -> str:
    """Training data and hyperparameters, by reference to the committed config."""
    lines = [
        "| model | config | dataset | scenes | crops | generator seed | "
        "`split_seed` | `seed` |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for ckpt, label in MODEL_ORDER:
        cfg = configs.get(ckpt)
        if cfg is None:
            continue
        ds = dataset_name(cfg["coco_path"])
        man = manifests.get(ds, {})
        scenes = man.get("stats", {}).get("n_images")
        kept = man.get("stats", {}).get("per_seg_class_kept", {})
        crops = kept.get("cartridge")
        gseed = man.get("seed")
        lines.append(
            f"| `{label}` | [`{cfg['config']}`](../configs/{cfg['config']}) | "
            f"[`{ds}`](datasets/{ds}.manifest.json) | "
            f"{scenes if scenes is not None else '—'} | "
            f"{crops if crops is not None else '—'} | "
            f"{gseed if gseed is not None else '—'} | "
            f"{cfg['split_seed']} | {cfg['seed']} |"
        )
    return "\n".join(lines)


def table_shared_hyperparams(configs) -> str:
    """The hyperparameters that are identical across every model, listed once.

    Emitted as a table only for the fields that are in fact shared; any field
    that differs between models is reported as differing rather than averaged
    away, because "the configs are byte-identical apart from paths" is a claim
    this table exists to keep honest.
    """
    fields = [
        ("architecture", None),
        ("crop_size", "model.crop_size"),
        ("half", "model.half"),
        ("train_val_split", "dataset.train_val_split"),
        ("jitter_frac", "dataset.jitter_frac"),
        ("epochs", "training.epochs"),
        ("batch_size", "training.batch_size"),
        ("learning_rate", "training.learning_rate"),
        ("lr_scheduler", "training.lr_scheduler"),
        ("dice_weight", "training.dice_weight"),
        ("deterministic", "training.deterministic"),
        ("select_on", "training.select_on"),
    ]
    rows = ["| setting | value | key |", "|---|---|---|"]
    tracked = [configs[c] for c, _ in MODEL_ORDER if c in configs]
    for name, key in fields:
        if key is None:
            continue
        vals = {json.dumps(c[name]) for c in tracked}
        if len(vals) == 1:
            val = json.loads(vals.pop())
            shown = ", ".join(val) if isinstance(val, list) else str(val)
            rows.append(f"| {name} | `{shown}` | `{key}` |")
        else:
            rows.append(f"| {name} | **differs between models** | `{key}` |")
    return "\n".join(rows)


def table_detector() -> str:
    """The detector's two receipts. Two anchor sets, one val split of 15."""
    text = (RECEIPTS / "frcnn_map.txt").read_text(encoding="utf-8")
    default = (RECEIPTS / "frcnn_map_default.txt").read_text(encoding="utf-8")

    def grab(t, pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else None

    custom = {
        "mAP@0.50": grab(text, r"mAP@0\.50: ([\d.]+)"),
        "AP_battery@0.50": grab(text, r"AP_battery@0\.50: ([\d.]+)"),
        "AP_cartridge@0.50": grab(text, r"AP_cartridge@0\.50: ([\d.]+)"),
        "mAP@0.75": grab(text, r"mAP@0\.75: ([\d.]+)"),
    }
    dflt = {
        "mAP@0.50": grab(default, r"final mAP@0\.5: ([\d.]+)"),
        "AP_battery@0.50": grab(default, r"final AP_battery@0\.5: ([\d.]+)"),
        "AP_cartridge@0.50": grab(default, r"final AP_cartridge@0\.5: ([\d.]+)"),
        "mAP@0.75": grab(default, r"final mAP@0\.75: ([\d.]+)"),
    }
    val_size = re.search(r"val size: (\d+)", text)
    gt = re.search(r"gt per class: (\{[^}]*\})", text)
    rows = [
        "| metric | k-means anchors (`best.pt`) | torchvision-default anchors |",
        "|---|---:|---:|",
    ]
    for k in custom:
        rows.append(f"| {k} | {fmt(custom[k])} | {fmt(dflt[k])} |")
    rows.append(
        f"\nBoth columns are the **same {val_size.group(1) if val_size else '?'}-frame "
        f"validation split**, ground truth {gt.group(1) if gt else '?'} "
        "(`2` = cartridge, `1` = battery). Fifteen frames is a very small n and "
        "these are the widest confidence intervals in this card."
    )
    return "\n".join(rows)


def table_datasets(manifests, checksums) -> str:
    lines = [
        "| manifest | scenes | boxes | generator seed | `instances_seg.json` "
        "SHA-256 (first 16) | bytes |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name in sorted(manifests):
        man = manifests[name]
        stats = man.get("stats", {})
        ck = checksums.get(name, {})
        ann = ck.get("annotations", {})
        digest = ann.get("sha256", "")
        lines.append(
            f"| [`{name}`]({name}.manifest.json) | "
            f"{stats.get('n_images', '—')} | {stats.get('n_boxes', '—')} | "
            f"{man.get('seed', '—')} | "
            f"`{digest[:16] if digest else '—'}` | "
            f"{ann.get('bytes', '—')} |"
        )
    return "\n".join(lines)


def table_dataset_classes(manifests) -> str:
    lines = [
        "| manifest | `cartridge` | `battery` | `placement_area` (bay) | "
        "`electronics_module` | `obstruction` | dropped |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(manifests):
        stats = manifests[name].get("stats", {})
        kept = stats.get("per_seg_class_kept")
        if not kept:
            lines.append(f"| `{name}` | " + " | ".join(["—"] * 6) + " |")
            continue
        lines.append(
            f"| `{name}` | {kept.get('cartridge', '—')} | "
            f"{kept.get('battery', '—')} | {kept.get('placement_area', '—')} | "
            f"{kept.get('electronics_module', '—')} | "
            f"{kept.get('obstruction', '—')} | "
            f"{stats.get('dropped_instances', '—')} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# block substitution
# --------------------------------------------------------------------------

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def substitute(text: str, name: str, body: str) -> str:
    begin, end = BEGIN.format(name), END.format(name)
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < 0:
        raise ValueError(f"missing markers for block {name!r}")
    return text[: i + len(begin)] + "\n" + body + "\n" + text[j:]


def build_blocks() -> dict[Path, dict[str, str]]:
    receipts = load_receipts()
    configs = load_training_configs()
    manifests = load_manifests()
    checksums = load_checksums()

    by_ckpt_cad, by_ckpt_own = {}, {}
    for r in receipts:
        (by_ckpt_cad if r.is_cad_test else by_ckpt_own)[
            str(Path(r.checkpoint).parent).replace("\\", "/")
        ] = r

    return {
        MODEL_CARD: {
            "segmenter-comparison": table_comparison(
                by_ckpt_cad, by_ckpt_own, manifests, configs
            ),
            "segmenter-per-sku": table_per_sku(by_ckpt_cad),
            "checkpoint-selection": table_selection(by_ckpt_cad, by_ckpt_own),
            "training-data": table_training(configs, manifests),
            "shared-hyperparameters": table_shared_hyperparams(configs),
            "detector": table_detector(),
        },
        DATASET_INDEX: {
            "dataset-index": table_datasets(manifests, checksums),
            "dataset-classes": table_dataset_classes(manifests),
        },
    }


def sync_datasets() -> int:
    """Copy each generator manifest into docs/datasets/ and record hashes."""
    DATASETS_DOC.mkdir(parents=True, exist_ok=True)
    checksums = {}
    n = 0
    for src in sorted((ROOT / "recog").glob("dataset3d*/manifest.json")):
        name = src.parent.name
        dst = DATASETS_DOC / f"{name}.manifest.json"
        shutil.copyfile(src, dst)
        entry = {
            "manifest": {
                "sha256": _sha256(src),
                "bytes": src.stat().st_size,
            }
        }
        for ann_name in ("instances_seg.json", "instances.json"):
            ann = src.parent / ann_name
            if ann.exists():
                entry["annotations"] = {
                    "file": ann_name,
                    "sha256": _sha256(ann),
                    "bytes": ann.stat().st_size,
                }
                break
        checksums[name] = entry
        n += 1
    (DATASETS_DOC / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"synced {n} manifests into {DATASETS_DOC.relative_to(ROOT)}")
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed tables differ from the receipts")
    ap.add_argument("--sync-datasets", action="store_true",
                    help="copy dataset manifests into docs/datasets/ and hash them")
    args = ap.parse_args(argv)

    if args.sync_datasets:
        return sync_datasets()

    drift = []
    for path, blocks in build_blocks().items():
        if not path.exists():
            print(f"missing: {path.relative_to(ROOT)}", file=sys.stderr)
            return 2
        original = path.read_text(encoding="utf-8")
        updated = original
        for name, body in blocks.items():
            updated = substitute(updated, name, body)
        if updated != original:
            if args.check:
                drift.append(path.relative_to(ROOT))
            else:
                path.write_text(updated, encoding="utf-8")
                print(f"updated {path.relative_to(ROOT)}")
        elif not args.check:
            print(f"unchanged {path.relative_to(ROOT)}")

    if args.check:
        if drift:
            for p in drift:
                print(f"DRIFT: {p} does not match the receipts", file=sys.stderr)
            return 1
        print("no drift: every generated table matches its receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
