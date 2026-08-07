"""
recog.bay_segmenter - per-ROI semantic segmentation of a cartridge crop.

Six channels, fixed order:
    0 background   1 cartridge   2 bay
    3 electronics  4 obstruction 5 battery

`bay` is SEG_CLASSES' `placement_area`. A later plan's segmentation-based
placement-area extractor is expected to index these numbers directly, so
the order is a contract, not just an implementation detail - no such
consumer exists in this repo yet.

Inference is BATCHED and fp16 by default because the latency budget has
no slack. Measured on an RTX 3060 at 384x384: one crop 12.6 ms, eight
crops looped 101 ms, eight batched fp32 59.6 ms, eight batched fp16 at
256x256 18.5 ms - against FDR 10.4's 50 ms end-to-end budget. Only the
last configuration fits.

256 is not a compromise. A PowerCore26800 at the generator's framing is
about 131x288 px, so 256x256 is at or above native crop resolution; 384
was upsampling and paid 3x for it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def build_segmenter(num_classes: int = 6, pretrained: bool = True):
    """DeepLabv3 + MobileNetV3-Large with `num_classes` outputs."""
    import torch.nn as nn
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)

    weights = (DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
               if pretrained else None)
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    # Re-head for our class count. The aux classifier is retained during
    # training (it regularises) and ignored at inference.
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(10, num_classes, kernel_size=1)
    return model


class BaySegmenter:
    """Inference wrapper. `segment_batch` is the primary entry point."""

    def __init__(self, checkpoint: Optional[str] = None,
                 device: str = "cuda", crop_size: int = 256,
                 half: bool = True, num_classes: int = 6,
                 pretrained: Optional[bool] = None) -> None:
        import torch

        self.crop_size = int(crop_size)
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available())
            else "cpu")
        # fp16 on CPU is slower than fp32, and silently so. Refuse rather
        # than let a CPU fallback quietly halve throughput.
        self.half = bool(half) and self.device.type == "cuda"

        # Default: pretrained iff no checkpoint is being loaded (COCO
        # weights give the randomly re-headed classifier something
        # reasonable to start from). `pretrained` lets a caller override
        # that - e.g. a test that passes checkpoint=None to exercise the
        # untrained-weights code path without downloading ~42 MB of COCO
        # weights it never asserts anything about.
        want_pretrained = (checkpoint is None) if pretrained is None \
            else bool(pretrained)
        self.model = build_segmenter(num_classes=num_classes,
                                     pretrained=want_pretrained)
        if checkpoint:
            # weights_only=True is not optional. The default unpickles
            # arbitrary Python objects, so loading a checkpoint becomes
            # arbitrary code execution by whoever produced the file. A
            # segmenter checkpoint holds tensors and nothing else.
            state = torch.load(checkpoint, map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model = self.model.to(self.device).eval()
        if self.half:
            self.model = self.model.half()

    def segment_batch(self, crops: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Label maps for `crops`, each resized back to its own size."""
        import torch

        if not len(crops):
            return []

        n = self.crop_size
        batch = np.empty((len(crops), 3, n, n), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i] = _resize_rgb(c, n).transpose(2, 0, 1) / 255.0

        x = torch.from_numpy(batch).to(self.device)
        if self.half:
            x = x.half()

        with torch.no_grad():
            logits = self.model(x)["out"]
            pred = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()

        out: List[np.ndarray] = []
        for i, c in enumerate(crops):
            h, w = c.shape[:2]
            out.append(_resize_nearest_2d(pred[i], h, w).astype(np.int8))
        return out

    def segment(self, crop: np.ndarray) -> np.ndarray:
        """Single crop. Prefer `segment_batch` - see the module docstring."""
        return self.segment_batch([crop])[0]


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:                     # pragma: no cover - optional
        ys = (np.arange(size) * img.shape[0] / size).astype(int)
        xs = (np.arange(size) * img.shape[1] / size).astype(int)
        return img[ys.clip(0, img.shape[0] - 1)][:, xs.clip(0, img.shape[1] - 1)]


def _resize_nearest_2d(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest only. A label map must never be interpolated."""
    ys = (np.arange(h) * a.shape[0] / h).astype(int).clip(0, a.shape[0] - 1)
    xs = (np.arange(w) * a.shape[1] / w).astype(int).clip(0, a.shape[1] - 1)
    return a[ys][:, xs]


__all__ = ["build_segmenter", "BaySegmenter"]
