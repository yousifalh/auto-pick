"""Faster R-CNN model factory.

Builds the detector described in the PPR: a ResNet-34 backbone wrapped
in a Feature Pyramid Network (FPN), custom anchor ratios tuned to the
measured battery / cartridge aspect ratios, and a torchvision
FasterRCNN head.

ResNet-34 is not a native FPN backbone in torchvision, so the FPN is
constructed with the generic :class:`BackboneWithFPN` helper over the
standard ``layer1``..``layer4`` features.

Heavy torch imports are deferred behind a try/except guard so this
module still imports (and raises a clear error) in environments where
torch and torchvision are not installed.
"""
from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - import guard
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch-free CI path
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch + torchvision are required to build the detector. "
            "Install with: pip install torch torchvision"
        )


def _build_resnet34_fpn(pretrained: bool):
    """Wrap a ResNet-34 in an FPN, returning the backbone object."""
    _require_torch()
    weights = (
        torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
    )
    backbone = torchvision.models.resnet34(weights=weights)

    # Feature taps → FPN levels (P2..P5 after the FPN projection head).
    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )


def build_fasterrcnn(cfg: Dict[str, Any]) -> "FasterRCNN":
    """Build the detector from a recognition-config dict."""
    _require_torch()

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    num_classes = int(model_cfg.get("num_classes", 3))
    ratios = tuple(
        float(r) for r in model_cfg.get(
            "anchor_ratios", [0.33, 0.5, 1.0, 1.5, 2.0],
        )
    )
    base_scales = tuple(
        int(s) for s in model_cfg.get("anchor_scales", [4, 8, 16, 32])
    )

    backbone = _build_resnet34_fpn(train_cfg.get("pretrained", True))

    # FPN produces 5 levels (P2..P6); add a trailing doubled scale for P6.
    per_level_scales = tuple(
        (s,) for s in (base_scales + (base_scales[-1] * 2,))
    )
    anchor_generator = AnchorGenerator(
        sizes=per_level_scales,
        aspect_ratios=(ratios,) * len(per_level_scales),
    )

    roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=int(model_cfg.get("roi_output_size", 7)),
        sampling_ratio=2,
    )

    # NOTE: min_size/max_size are deliberately NOT set here. They feed
    # GeneralizedRCNNTransform, which runs during TRAINING as well as
    # inference, so changing them here would rescale the training renders and
    # invalidate the anchor_scales tuned against them. Inference resolution is
    # a genuine accuracy knob on real photos and is applied per-detector in
    # recog/inference.py instead.
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        box_nms_thresh=float(model_cfg.get("nms_iou", 0.4)),
        box_score_thresh=float(model_cfg.get("confidence_threshold", 0.70)),
    )


def freeze_batchnorm(model: "nn.Module") -> None:
    """Freeze all BatchNorm layers — used for the first 20 epochs per PPR."""
    _require_torch()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


__all__ = ["build_fasterrcnn", "freeze_batchnorm"]
