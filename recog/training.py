"""Training loop for the Faster R-CNN detector.

Kept separate from :mod:`recog.model` so the heavy torch imports are
only paid for when actually running a training job. The entry point is::

    python -m recog.training --config configs/recognition.yaml

The design follows the PPR:

* COCO-pretrained ResNet-34 + FPN backbone.
* BatchNorm frozen for ``training.frozen_bn_epochs`` (default 20).
* Cosine learning-rate schedule.
* Smooth-L1 box regression (torchvision's default).
* Per-epoch checkpointing plus a best-mAP tracker.

All torch-dependent code is guarded so the module is importable on a
CPU-only CI container, where it will simply refuse to train.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common.config import load_yaml
from common.logging import get_logger

log = get_logger("recog.training")


# ---------------------------------------------------------- torch guard --

def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover - CI path
        raise ImportError(
            "Training requires torch + torchvision. "
            "Install with: pip install torch torchvision"
        ) from exc


# ----------------------------------------------------- build helpers ----

def _build_optimiser(params, cfg: Dict[str, Any]):
    import torch.optim as optim

    name = cfg.get("optimiser", "sgd").lower()
    lr = float(cfg.get("learning_rate", 5e-3))
    wd = float(cfg.get("weight_decay", 5e-4))

    if name == "sgd":
        return optim.SGD(
            params, lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimiser: {name}")


def _build_scheduler(opt, cfg: Dict[str, Any], steps_per_epoch: int):
    import torch.optim as optim

    name = (cfg.get("lr_scheduler") or "cosine").lower()
    epochs = int(cfg.get("epochs", 60))
    total_steps = max(1, epochs * steps_per_epoch)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if name == "step":
        return optim.lr_scheduler.StepLR(
            opt, step_size=max(1, epochs // 3), gamma=0.1,
        )
    return None


def _split_dataset(dataset, train_val_split: float, seed: int = 0):
    import torch

    n = len(dataset)
    n_train = int(round(train_val_split * n))
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=gen,
    )


# ----------------------------------------------- per-epoch routines ----

def train_one_epoch(
    model,
    loader,
    optimiser,
    scheduler,
    device,
    epoch: int,
    frozen_bn: bool,
) -> float:
    """Run one training epoch, returning the mean loss."""
    import torch

    model.train()
    if frozen_bn:
        from recog.model import freeze_batchnorm

        freeze_batchnorm(model)

    total = 0.0
    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


def evaluate_model(model, loader, device) -> Dict[str, float]:
    """Run validation and return a dict of mAP metrics at 0.5 and 0.75."""
    import torch
    from recog.evaluate import mean_ap

    model.eval()
    gts: Dict[int, List] = {}
    preds: Dict[int, List] = {}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, (out, t) in enumerate(zip(outputs, targets)):
                img_id = batch_idx * 1000 + i  # unique per batch
                gts[img_id] = [
                    (tuple(b.cpu().numpy().tolist()), int(c.item()))
                    for b, c in zip(t["boxes"], t["labels"])
                ]
                preds[img_id] = [
                    (
                        tuple(b.cpu().numpy().tolist()),
                        int(c.item()),
                        float(s.item()),
                    )
                    for b, c, s in zip(
                        out["boxes"], out["labels"], out["scores"],
                    )
                ]

    class_ids = [1, 2]
    results: Dict[str, float] = {}
    results.update(mean_ap(gts, preds, class_ids, 0.5))
    results.update(mean_ap(gts, preds, class_ids, 0.75))
    return results


# ----------------------------------------------- top-level entrypoint --

def train(cfg: Dict[str, Any]) -> None:
    """Run the full training schedule using ``cfg`` (a recognition YAML)."""
    _require_torch()
    import torch

    from recog.augmentation import build_train_transform, build_val_transform
    from recog.dataset import BatteryCartridgeDataset, collate_fn
    from recog.model import build_fasterrcnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # ---- Data ----
    ds_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentation", {})
    train_tf = build_train_transform(aug_cfg)
    val_tf = build_val_transform(aug_cfg)

    full_dataset = BatteryCartridgeDataset(
        img_dir=ds_cfg["img_dir"],
        ann_dir=ds_cfg["ann_dir"],
        transforms=train_tf,
    )
    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No images found in {ds_cfg['img_dir']}. "
            "Run recog.synth_dataset to generate a test set first."
        )

    train_set, val_set = _split_dataset(
        full_dataset, float(ds_cfg.get("train_val_split", 0.85)),
    )
    # Swap transforms on the val subset so augmentation is not applied.
    val_set.dataset = BatteryCartridgeDataset(
        img_dir=ds_cfg["img_dir"],
        ann_dir=ds_cfg["ann_dir"],
        transforms=val_tf,
    )

    train_cfg = cfg["training"]
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- Model / optimiser ----
    model = build_fasterrcnn(cfg).to(device)
    optimiser = _build_optimiser(
        [p for p in model.parameters() if p.requires_grad],
        train_cfg,
    )
    scheduler = _build_scheduler(optimiser, train_cfg, len(train_loader))

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "recog/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Main loop ----
    best_map = -1.0
    for epoch in range(int(train_cfg.get("epochs", 60))):
        frozen_bn = epoch < int(train_cfg.get("frozen_bn_epochs", 20))
        mean_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler,
            device, epoch, frozen_bn,
        )
        log.info(
            "epoch=%d loss=%.4f bn_frozen=%s",
            epoch, mean_loss, frozen_bn,
        )

        metrics = evaluate_model(model, val_loader, device)
        log.info("epoch=%d metrics=%s", epoch, metrics)

        map50 = metrics.get("mAP@0.50", 0.0)
        if map50 > best_map:
            best_map = map50
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_dir / "best.pt",
            )
            log.info("New best mAP@0.5=%.4f saved", best_map)

        # Always keep the latest epoch as well. On the synthetic dataset the
        # validation mAP saturates at 1.0 within two epochs, after which
        # ``map50 > best_map`` is never true again — so without this, every
        # subsequent epoch trains and is then discarded, and the model you
        # actually end up with is whichever early epoch happened to peak.
        # Held-out real-photo performance keeps changing long after the
        # synthetic metric stops moving, so the last epoch must be recoverable.
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            },
            ckpt_dir / "last.pt",
        )


# ----------------------------------------------------------- CLI ----

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Faster R-CNN battery/cartridge detector.")
    parser.add_argument("--config", default="configs/recognition.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    _cli()
