"""
recog.seg_training - training loop for the bay segmenter.

Loss is cross-entropy plus Dice. `obstruction` covers a small area and is
absent from roughly 40% of crops, so plain cross-entropy can score well
while never predicting it: getting a rare small class wrong costs almost
nothing per pixel. Dice is computed per class over the batch, which
makes a class's contribution independent of how many pixels it occupies.

Entry point::

    python -m recog.seg_training --config configs/segmentation.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.config import load_yaml
from common.logging import get_logger
from recog.seg_dataset import SEG_CHANNELS

log = get_logger("recog.seg_training")

# Ordered class names, index == the channel a pixel is painted with. Single
# source of truth is recog.seg_dataset.SEG_CHANNELS; do not re-list here.
CLASS_NAMES: List[str] = [None] * len(SEG_CHANNELS)  # type: ignore[list-item]
for _name, _idx in SEG_CHANNELS.items():
    CLASS_NAMES[_idx] = _name


# --------------------------------------------------------------- losses --

def dice_loss(logits, target, num_classes: int, eps: float = 1.0):
    """Mean soft Dice over the classes PRESENT in `target`.

    Classes absent from the batch are skipped rather than scored as a
    perfect miss. Without that, ~40% of batches would carry a constant
    gradient pushing toward an obstruction that is not in the image.
    """
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)

    present = onehot.sum(dims) > 0
    if not bool(present.any()):
        return torch.zeros((), device=logits.device)
    return 1.0 - dice[present].mean()


def combined_loss(logits, target, num_classes: int,
                  dice_weight: float = 0.5):
    import torch.nn.functional as F
    return (F.cross_entropy(logits, target)
            + dice_weight * dice_loss(logits, target, num_classes))


def checkpoint_state_dict(model) -> Dict[str, Any]:
    """The subset of `model.state_dict()` a checkpoint should carry.

    Training builds the model with `pretrained=True` because the aux
    classifier regularises during training (see
    recog.bay_segmenter.build_segmenter). BaySegmenter, given a
    checkpoint, builds with `pretrained=False` - the aux head has no
    inference role and costs latency in a 50ms budget - so its model
    instance has no `aux_classifier` submodule at all. Saving the aux
    weights anyway makes the checkpoint unloadable by the only inference
    path in the project: `load_state_dict` is strict, and there is
    nowhere for those keys to go. Stripping them here, once, at the
    point a checkpoint is written, keeps every consumer of the file
    (BaySegmenter today, whatever reads it next) simply strict-loadable
    with no special-casing on their end.
    """
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("aux_classifier.")}


def _atomic_save(obj: Dict[str, Any], path: Path) -> None:
    """`torch.save` via a temp file + rename, so a run killed mid-write
    (the 10-minute per-command cap this project runs under, or any other
    hard interrupt) cannot leave a half-written checkpoint that a later
    `--resume` would load as if it were valid. Same-directory tmp file so
    the rename is same-filesystem and therefore atomic on both POSIX and
    Windows.
    """
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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
    lr = float(cfg.get("learning_rate", 1e-2))
    wd = float(cfg.get("weight_decay", 1e-4))

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
    epochs = int(cfg.get("epochs", 40))
    total_steps = max(1, epochs * steps_per_epoch)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if name == "step":
        return optim.lr_scheduler.StepLR(
            opt, step_size=max(1, epochs // 3), gamma=0.1,
        )
    return None


def drop_last_batch(n_train: int, batch_size: int) -> bool:
    """Whether the train loader must drop its final, short batch.

    DeepLabV3's ASPP image-pooling branch collapses every feature map to
    1x1, so a batch of ONE reaches BatchNorm as `[1, C, 1, 1]` - one
    value per channel - and `F.batch_norm` raises in training mode
    (`Expected more than 1 value per channel when training`). It is a
    hard crash at the end of epoch 1, not a degradation.

    Kept deliberately narrow: only the singleton remainder is dropped.
    An unconditional `drop_last=True` would silently shorten every
    epoch of every run, including the existing baseline whose published
    IoUs (`docs/receipts/seg_eval.txt`) are the floor this work is
    measured against - a second, gratuitous difference between the
    baseline and the runs being compared to it. batch_size 1 is left
    alone on purpose: it cannot be rescued by dropping one batch, and
    emptying the loader would hide a broken config rather than surface
    it.
    """
    return batch_size > 1 and n_train % batch_size == 1


def _split_dataset(dataset, train_val_split: float, seed: int = 0):
    import torch

    n = len(dataset)
    n_train = int(round(train_val_split * n))
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=gen,
    )


# ------------------------------------------------------------- metrics --

def instance_counts(loader, num_classes: int) -> List[int]:
    """Number of CROPS in `loader` containing at least one pixel of each
    class - not pixel count.

    This is the number that says how much a per-class IoU is actually
    worth. `bay` is 23% of the 361-crop corpus (84 crops); at
    train_val_split 0.85 the validation split is ~54 crops, so its bay
    count is a low double digit at best and its obstruction count can be
    single digits. An IoU computed over that many instances is real
    signal, but it is NOISY signal, and reporting it without the count
    beside it lets a reader mistake noise for a trend.
    """
    import torch

    counts = [0] * num_classes
    for _images, targets in loader:
        for b in range(targets.shape[0]):
            for c in torch.unique(targets[b]).tolist():
                if 0 <= c < num_classes:
                    counts[c] += 1
    return counts


def _update_confusion(inter: List[int], union: List[int],
                      pred, target, num_classes: int) -> None:
    """Accumulate per-class intersection/union pixel counts in place."""
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter[c] += int((p & t).sum().item())
        union[c] += int((p | t).sum().item())


def _per_class_iou(inter: Sequence[int], union: Sequence[int]
                   ) -> Dict[str, Optional[float]]:
    """IoU per class name, or None for a class with zero union.

    A class absent from BOTH the predictions and the ground truth over the
    whole set being scored has nothing to divide - None rather than 1.0 or
    0.0, so callers can tell "perfect" from "never seen" and exclude the
    latter from a mean instead of quietly rewarding or punishing it.
    """
    out: Dict[str, Optional[float]] = {}
    for c, name in enumerate(CLASS_NAMES):
        out[name] = (inter[c] / union[c]) if union[c] > 0 else None
    return out


def selected_mean_iou(ious: Dict[str, Optional[float]],
                      select_on: Sequence[str]) -> float:
    """Mean IoU over `select_on`, skipping classes with no pixels either
    way in the set being scored (see `_per_class_iou`)."""
    present = [ious[c] for c in select_on if ious.get(c) is not None]
    if not present:
        log.warning("none of the select_on classes %s appeared in this "
                    "validation pass; reporting 0.0", list(select_on))
        return 0.0
    return sum(present) / len(present)


# ----------------------------------------------- per-epoch routines ----

def train_one_epoch(model, loader, optimiser, scheduler, device,
                    num_classes: int, dice_weight: float) -> float:
    """Run one training epoch, returning the mean loss.

    The aux classifier torchvision's DeepLabV3 carries (see
    recog.bay_segmenter.build_segmenter) is folded in at the standard 0.5
    weight when present - it is why the head is kept re-sized instead of
    dropped, and an unused head that never receives gradient would make
    that comment in build_segmenter's docstring false.
    """
    import torch

    model.train()
    total = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        out = model(images)
        loss = combined_loss(out["out"], targets, num_classes, dice_weight)
        if "aux" in out:
            loss = loss + 0.5 * combined_loss(
                out["aux"], targets, num_classes, dice_weight)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


def evaluate_model(model, loader, device, num_classes: int
                   ) -> Dict[str, Optional[float]]:
    """Per-class IoU accumulated over the whole loader (not per-batch -
    a class rare enough to miss single batches would otherwise flicker
    between None and a noisy 0/1)."""
    import torch

    model.eval()
    inter = [0] * num_classes
    union = [0] * num_classes
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)["out"]
            pred = logits.argmax(dim=1)
            _update_confusion(inter, union, pred, targets, num_classes)
    return _per_class_iou(inter, union)


# ----------------------------------------------- top-level entrypoint --

def train(cfg: Dict[str, Any], resume: bool = False,
          seed: Optional[int] = None,
          deterministic: Optional[str] = None) -> None:
    """Run the full training schedule using ``cfg`` (a segmentation YAML).

    ``seed`` and ``deterministic`` override ``training.seed`` and
    ``training.deterministic``; both are resolved and LOGGED before
    anything stochastic is constructed, and the resolved seed is written
    into every checkpoint this run produces. See recog/seeding.py for
    what is seeded and for the difference between "seeded" and
    "bitwise-deterministic" — they are not the same claim and this
    project does not blur them.

    `resume=True` picks up from `<checkpoint_dir>/train_state.pt`, written
    unconditionally at the end of every epoch. That file is distinct from
    `best.pt`/`last.pt`: those two stay inference-format (stripped aux
    head, `weights_only=True`-loadable) for every existing consumer
    (bay_segmenter.BaySegmenter, seg_evaluate, calibrate_tau, seg_ablation)
    to keep loading unchanged; `train_state.pt` carries the full model
    (aux head included, so a resumed run's aux loss stays wired exactly
    as a fresh run's does), optimiser and scheduler state, and the epoch
    and best-so-far IoU needed to continue the schedule rather than
    restart it. Without this, any run cut off before its last epoch loses
    every epoch after the last save - there was no way back into a
    40-epoch schedule interrupted at epoch 24.
    """
    _require_torch()
    import torch

    from recog.augmentation import (build_seg_train_transform,
                                    build_seg_val_transform)
    from recog.bay_segmenter import build_segmenter
    from recog.seeding import (assert_loader_seeded, capture_rng_state,
                               dataloader_kwargs, resolve_deterministic,
                               resolve_seed, restore_rng_state, seed_everything,
                               seed_transform)
    from recog.seg_dataset import BaySegDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # ---- Seeding ----
    # FIRST, before the transforms, the loaders or the model: weight
    # initialisation draws from torch's global CPU generator the moment
    # build_segmenter runs, so seeding after that point would seed
    # everything except the initialisation - which is the component the
    # 1-epoch probe showed moving the selection metric by 1.5 points.
    run_seed = resolve_seed(cfg, seed)
    run_deterministic = resolve_deterministic(cfg, deterministic)
    seed_record = seed_everything(run_seed, deterministic=run_deterministic)

    # ---- Data ----
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentation", {})
    num_classes = int(model_cfg.get("num_classes", 6))
    crop_size = int(model_cfg.get("crop_size", 256))

    train_tf = build_seg_train_transform(aug_cfg)
    val_tf = build_seg_val_transform(aug_cfg)
    # albumentations 2.x holds a per-instance RNG that the global seed
    # above does NOT reach, so the pipeline that touches every training
    # sample has to be seeded explicitly. seed_transform raises rather
    # than return quietly if it is handed something it cannot seed.
    log.info("augmentation seeded: train via %s, val via %s",
             seed_transform(train_tf, run_seed, "train augmentation"),
             seed_transform(val_tf, run_seed, "val augmentation"))

    # `dataset.whole_frame` selects the architecture-comparison arm:
    # one sample per IMAGE instead of one per cartridge ROI. Absent or
    # false everywhere in configs/segmentation*.yaml, so every member of
    # the byte-identical generalisation family takes the ROI path
    # unchanged. See recog/seg_wholeframe.py for what the comparison is
    # and why the split has to be derived rather than drawn.
    if bool(ds_cfg.get("whole_frame", False)):
        from recog.seg_wholeframe import WholeFrameSegDataset

        full_dataset = WholeFrameSegDataset(
            coco_path=ds_cfg["coco_path"],
            img_dir=ds_cfg["img_dir"],
            out_size=crop_size,
            transform=train_tf,
            seed=run_seed,
        )
        log.info("whole-frame arm: %d frames at %d px (ROI path disabled)",
                 len(full_dataset), crop_size)
    else:
        full_dataset = BaySegDataset(
            coco_path=ds_cfg["coco_path"],
            img_dir=ds_cfg["img_dir"],
            out_size=crop_size,
            jitter_frac=float(ds_cfg.get("jitter_frac", 0.06)),
            train=True,
            transform=train_tf,
            # The crop-jitter RNG follows the RUN seed, so two runs at
            # different seeds are independent samples in every stochastic
            # component rather than sharing their box jitter. (The
            # train/val SPLIT does not - see split_seed below.)
            seed=run_seed,
        )
    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No crops found for {ds_cfg['coco_path']}. "
            "Run recog.generate3d to build a segmentation dataset first.")

    # Seeded and logged, not just seeded: the split determines how many of
    # each rare class land in validation at all, so which crops fall where
    # has to be reproducible and visible, not merely deterministic.
    #
    # DELIBERATELY NOT the run seed. The split defines what the reported
    # IoU is measured on; tying it to the run seed would mean every
    # change of seed also changed the yardstick, and two seeds would no
    # longer be comparable. It is its own knob and stays at its own
    # default.
    split_seed = int(ds_cfg.get("split_seed", 0))
    if bool(ds_cfg.get("whole_frame", False)):
        # The whole-frame arm must NOT draw its own split, and this is the
        # one place that can go wrong silently.
        #
        # It trains on frames; the architecture comparison scores CROPS,
        # on the crop split. An independent random_split over the 436
        # frames puts a frame in training whenever the draw says so -
        # including frames that contribute crops to the crop split's
        # VALIDATION half. The model would then be scored on pixels it
        # trained on, it would win, and nothing in the output would say
        # so. Measured on this corpus: an independent draw gives 371/65,
        # against 324/112 for the derived split - 47 frames' difference,
        # every one of them a potential leak.
        #
        # So the frame split is DERIVED: a frame trains only if every
        # crop it carries is in the crop split's training half. See
        # recog/seg_wholeframe.frame_split_from_crop_split. The asymmetry
        # handicaps this arm and that is deliberate - a comparison unfair
        # towards the hypothesis under test is the one worth having.
        import torch

        from recog.seg_wholeframe import frame_split_from_crop_split

        crop_ds = full_dataset._crops
        crop_train, _crop_val = _split_dataset(
            crop_ds, float(ds_cfg.get("train_val_split", 0.85)),
            seed=split_seed)
        f_train, f_val = frame_split_from_crop_split(
            full_dataset, [int(i) for i in crop_train.indices])
        train_set = torch.utils.data.Subset(full_dataset, f_train)
        val_set = torch.utils.data.Subset(full_dataset, f_val)
        log.info(
            "whole-frame split DERIVED from the crop split seed=%d "
            "(%d train / %d held out of %d frames; an independent draw "
            "would have been %d/%d and could leak)",
            split_seed, len(f_train), len(f_val), len(full_dataset),
            int(round(float(ds_cfg.get("train_val_split", 0.85))
                      * len(full_dataset))),
            len(full_dataset) - int(round(
                float(ds_cfg.get("train_val_split", 0.85))
                * len(full_dataset))))
    else:
        train_set, val_set = _split_dataset(
            full_dataset, float(ds_cfg.get("train_val_split", 0.85)),
            seed=split_seed)
        log.info("train/val split seed=%d (%d train / %d val of %d crops)",
                 split_seed, len(train_set), len(val_set), len(full_dataset))
    # Swap in the un-augmented, un-jittered dataset for the val subset -
    # jitter models detector box error, which belongs in training only;
    # any geometry at validation time would make the reported IoU a
    # property of the augmenter, not of the model. Same pattern as
    # recog/training.py.
    #
    # The replacement has to be the SAME KIND of dataset the subset's
    # indices were drawn against. For the whole-frame arm those indices
    # are FRAME indices, and substituting a BaySegDataset here would index
    # frame 300 into crop 300 - a different image, silently, with the
    # validation IoU quietly describing the wrong pixels.
    if bool(ds_cfg.get("whole_frame", False)):
        from recog.seg_wholeframe import WholeFrameSegDataset

        val_set.dataset = WholeFrameSegDataset(
            coco_path=ds_cfg["coco_path"],
            img_dir=ds_cfg["img_dir"],
            out_size=crop_size,
            transform=val_tf,
        )
    else:
        val_set.dataset = BaySegDataset(
            coco_path=ds_cfg["coco_path"],
            img_dir=ds_cfg["img_dir"],
            out_size=crop_size,
            jitter_frac=float(ds_cfg.get("jitter_frac", 0.06)),
            train=False,
            transform=val_tf,
        )

    train_cfg = cfg["training"]
    batch_size = int(train_cfg.get("batch_size", 8))
    drop_last = drop_last_batch(len(train_set), batch_size)
    if drop_last:
        log.info(
            "train split of %d crops leaves a final batch of 1 at "
            "batch_size %d; dropping it (BatchNorm cannot train on a "
            "singleton batch - see drop_last_batch)",
            len(train_set), batch_size)
    # shuffle=True builds its own sampler generator from OS entropy
    # unless it is handed one, which is why the epoch order used to
    # differ between two runs of the same command.
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=drop_last,
        **dataloader_kwargs(run_seed),
    )
    # A generator that is built and then not passed through would leave
    # the shuffle unseeded with every other signal still saying "seeded".
    assert_loader_seeded(train_loader, "train loader", expected_seed=run_seed)
    loader_generator = train_loader.generator
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
    )
    log.info("dataset: %d crops (%d train / %d val)",
             len(full_dataset), len(train_set), len(val_set))

    # How many val crops actually carry each class. Computed once, up
    # front - the val transform/box are fixed (no jitter, no geometric
    # augmentation), so this count does not change epoch to epoch, and
    # every IoU logged below should be read next to it.
    select_on = list(train_cfg.get("select_on",
                                   ["bay", "electronics", "obstruction"]))
    val_counts = instance_counts(val_loader, num_classes)
    val_counts_by_name = dict(zip(CLASS_NAMES, val_counts))
    log.info("validation instance counts (crops containing the class, "
             "of %d val crops): %s", len(val_set), val_counts_by_name)
    for cls in select_on:
        n = val_counts_by_name.get(cls, 0)
        if n == 0:
            log.warning(
                "select_on class %r has ZERO instances in the validation "
                "split (seed=%d) - its IoU is undefined (None) all run and "
                "a selection metric averaged over the remaining classes is "
                "quietly missing a third of what it claims to measure. "
                "Consider a different dataset.split_seed.", cls, split_seed)
        elif n < 5:
            log.warning(
                "select_on class %r has only %d instances in the "
                "validation split (seed=%d) - its IoU this run is a "
                "small-sample estimate, not a stable trend.",
                cls, n, split_seed)

    # ---- Model / optimiser ----
    model = build_segmenter(num_classes=num_classes,
                            pretrained=bool(model_cfg.get("pretrained", True))
                            ).to(device)
    optimiser = _build_optimiser(
        [p for p in model.parameters() if p.requires_grad], train_cfg)
    scheduler = _build_scheduler(optimiser, train_cfg, len(train_loader))

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "recog/checkpoints/seg"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "train_state.pt"

    dice_weight = float(train_cfg.get("dice_weight", 0.5))

    # ---- Resume (optional) ----
    start_epoch = 0
    best_iou = -1.0
    if resume:
        if not state_path.exists():
            raise SystemExit(
                f"error: --resume was given but {state_path} does not "
                "exist. There is no in-progress run to continue - drop "
                "--resume to start a fresh schedule.")
        # weights_only=False: unlike best.pt/last.pt (loaded by inference
        # code that may see an untrusted checkpoint), train_state.pt is
        # written and read by this same script, on this same machine, and
        # carries optimiser/scheduler state that is not tensors-only.
        state = torch.load(state_path, map_location="cpu",
                           weights_only=False)
        saved_counts = state.get("val_instance_counts")
        if saved_counts is not None and dict(saved_counts) != val_counts_by_name:
            raise SystemExit(
                "error: --resume's recomputed validation split does not "
                "match train_state.pt's own record - continuing would "
                "silently train/validate against a different split than "
                "the interrupted run used.\n"
                f"  train_state.pt recorded: {dict(saved_counts)}\n"
                f"  this run recomputes:     {val_counts_by_name}\n"
                "The dataset behind --config's dataset.coco_path most "
                "likely changed since this run started. Point --config "
                "at the exact dataset the interrupted run used, or start "
                "a fresh run (drop --resume).")
        saved_seed = state.get("seed")
        if saved_seed is not None and int(saved_seed) != run_seed:
            raise SystemExit(
                "error: --resume was given a different seed than the run "
                "it is continuing.\n"
                f"  train_state.pt recorded seed: {int(saved_seed)}\n"
                f"  this run resolved seed:       {run_seed}\n"
                "Continuing would splice two RNG streams into one run and "
                "call the result reproducible. Drop --seed (or set "
                "training.seed to the recorded value) to continue that "
                "run, or start a fresh one.")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        # Continue the RNG stream rather than restart it: without this a
        # resumed run replays epoch 0's shuffle order and augmentation
        # draws, so the same seed and the same command would produce a
        # different model depending on whether the run was interrupted.
        restore_rng_state(state.get("rng_state"), loader_generator)
        start_epoch = int(state["epoch"]) + 1
        best_iou = float(state["best_iou"])
        log.info("resuming from train_state.pt: completed epoch=%d, "
                 "best_iou=%.4f so far - continuing at epoch=%d",
                 state["epoch"], best_iou, start_epoch)

    # ---- Main loop ----
    total_epochs = int(train_cfg.get("epochs", 40))
    if start_epoch >= total_epochs:
        log.info("start_epoch=%d >= configured epochs=%d - nothing to do",
                 start_epoch, total_epochs)
    for epoch in range(start_epoch, total_epochs):
        mean_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, device,
            num_classes, dice_weight)

        ious = evaluate_model(model, val_loader, device, num_classes)
        selection = selected_mean_iou(ious, select_on)
        log.info("epoch=%d loss=%.4f val_iou=%s selected(%s over "
                 "counts=%s)=%.4f", epoch, mean_loss, ious, select_on,
                 {c: val_counts_by_name.get(c) for c in select_on},
                 selection)

        # Selection is on bay/electronics/obstruction IoU only. Including
        # background and cartridge would let a model that gets the big
        # easy regions right mask a failure on the three classes the
        # placement mask is actually built from.
        if selection > best_iou:
            best_iou = selection
            _atomic_save(
                {
                    "model": checkpoint_state_dict(model),
                    "epoch": epoch,
                    "ious": ious,
                    "selected_mean_iou": selection,
                    "select_on": select_on,
                    "val_instance_counts": val_counts_by_name,
                    "coco_path": ds_cfg["coco_path"],
                    "split_seed": split_seed,
                    # What it would take to get this file back. A
                    # checkpoint that does not record its seed is a
                    # checkpoint nobody can re-derive.
                    "seed": run_seed,
                    "seeding": seed_record,
                },
                ckpt_dir / "best.pt",
            )
            log.info("New best selected mean IoU=%.4f saved", best_iou)

        # Always keep the latest epoch too, UNCONDITIONALLY - commit
        # 0ac6c5b fixed exactly this bug (best-only saving silently
        # discarding every later epoch) in the detector's loop; the same
        # bug is just as easy to reintroduce here.
        _atomic_save(
            {
                "model": checkpoint_state_dict(model),
                "epoch": epoch,
                "ious": ious,
                "selected_mean_iou": selection,
                "select_on": select_on,
                "val_instance_counts": val_counts_by_name,
                "coco_path": ds_cfg["coco_path"],
                "split_seed": split_seed,
                "seed": run_seed,
                "seeding": seed_record,
            },
            ckpt_dir / "last.pt",
        )

        # Full (unstripped) state for --resume: model incl. aux head,
        # optimiser and scheduler, and the epoch/best_iou needed to
        # continue the schedule rather than restart it. Written every
        # epoch, unconditionally, so a run cut off by this project's
        # 10-minute-per-command cap loses at most the epoch in progress.
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": (scheduler.state_dict()
                             if scheduler is not None else None),
                "epoch": epoch,
                "best_iou": best_iou,
                "val_instance_counts": val_counts_by_name,
                "coco_path": ds_cfg["coco_path"],
                "split_seed": split_seed,
                "seed": run_seed,
                "seeding": seed_record,
                # Every RNG's position in its stream, so --resume
                # continues the run instead of replaying epoch 0's
                # shuffle and augmentation draws.
                "rng_state": capture_rng_state(loader_generator),
            },
            state_path,
        )


# ----------------------------------------------------------- CLI ----

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train the bay segmenter.")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from <checkpoint_dir>/train_state.pt instead of "
             "starting a fresh 0-epoch schedule.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override training.seed. The resolved seed is logged and "
             "written into every checkpoint this run produces.")
    parser.add_argument(
        "--deterministic", choices=["off", "warn", "strict"], default=None,
        help="Override training.deterministic. off = seeded only (two runs "
             "at one seed still diverge on CUDA); warn = the default, "
             "deterministic kernels where torch has them and a warning "
             "where it does not; strict = raise instead of warning, which "
             "this model's loss does not survive. See recog/seeding.py.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train(cfg, resume=args.resume, seed=args.seed,
          deterministic=args.deterministic)


if __name__ == "__main__":  # pragma: no cover
    _cli()


__all__ = [
    "dice_loss",
    "combined_loss",
    "checkpoint_state_dict",
    "train",
    "evaluate_model",
    "selected_mean_iou",
    "CLASS_NAMES",
]
