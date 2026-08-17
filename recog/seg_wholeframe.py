"""
recog.seg_wholeframe - the whole-frame arm of the segmentation
architecture comparison.

The shipping segmenter runs PER ROI: the detector proposes a cartridge
box, the crop is resized to 256x256, and the model labels that crop.
This module trains and scores the alternative - one forward pass over
the entire camera frame - so the choice between them stops being an
argument and becomes a measurement.

`docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md`
Sec3.1 rejected whole-frame segmentation, but it rejected a DIFFERENT
design: whole-frame semantic segmentation plus connected-component
instance recovery, on the ground that connected components reintroduce
the occlusion failure mode FDR Sec10.6 attributes 22 % of heuristic
detector misses to. That objection does not apply here. This arm keeps
the detector for instance identity and uses the whole-frame model only
for PIXEL LABELS; a cartridge's mask is the frame prediction intersected
with that cartridge's own box. No connected components anywhere.

What the comparison is actually about, therefore, is resolution and
cost, not instance recovery:

  * per-ROI pays `n_cartridges x 2.3 ms` and sees each cartridge at
    256x256 regardless of how small it is in frame;
  * whole-frame pays a FIXED cost and sees each cartridge at whatever
    fraction of the frame it occupies.

The two cross over at roughly 14 cartridges per frame on the measured
numbers (18.5 ms for 8 crops against ~33 ms for one 1280x720 pass), and
this corpus carries 1-8. So the expected result is that per-ROI wins on
THIS corpus and the honest form of Sec3.1's claim is scale-dependent.
The prediction, the resolvable effect and the falsifier are pre-
registered in
`docs/superpowers/specs/2026-08-16-wholeframe-preregistration.md`,
written before the first run.

THE SPLIT IS THE PART TO GET RIGHT, and it is why this module exists
rather than a `whole_frame: true` flag on BaySegDataset. The comparison
scores both arms on the SAME validation CROPS. If the whole-frame arm
split by IMAGE independently, a frame in its training set could contain
a crop in the shared validation set - the whole-frame model would have
trained on the pixels it is scored against, and it would win for that
reason. `frame_split_from_crop_split` below derives the frame split FROM
the crop split and puts a frame in training only when EVERY crop it
carries is in the crop split's training half. Any frame contributing
even one validation crop is held out entirely.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recog.seg_dataset import (SEG_CHANNELS, BaySegDataset, extract_crop,
                               rasterise_crop)

__all__ = [
    "WholeFrameSegDataset",
    "frame_split_from_crop_split",
    "predict_val_crops_from_frames",
]


class WholeFrameSegDataset:
    """One sample per IMAGE: the whole frame and its whole label map.

    Deliberately a thin wrapper over :class:`BaySegDataset` rather than a
    parallel reader of the COCO sidecar. The sidecar parsing, the
    `unit_id` validation and the paint order are subtle and already
    correct in one place; a second copy would drift.

    There is no crop jitter here and that is not an oversight. Jitter
    exists in the ROI path because at inference the crop comes from the
    detector, whose boxes are not ground truth, so training on perfect
    boxes would bake in a distribution shift. A whole-frame model is
    handed the frame, and the frame has no box error to simulate.
    """

    def __init__(self, coco_path: str, img_dir: str, out_size: int = 512,
                 transform=None, seed: int = 0) -> None:
        self._crops = BaySegDataset(
            coco_path=coco_path, img_dir=img_dir, out_size=out_size,
            jitter_frac=0.0, train=False, transform=None, seed=seed)
        self.img_dir = img_dir
        self.out_size = int(out_size)
        self.transform = transform

        # One entry per distinct source image, in first-appearance order
        # so the sequence is deterministic without sorting cost.
        seen: Dict[str, int] = {}
        self.frames: List[Tuple[dict, List[dict]]] = []
        # frame index -> the crop indices that came from it. This is what
        # makes the leak-free split derivable.
        self.crops_of_frame: List[List[int]] = []
        for crop_idx, (img_meta, anns, _box) in enumerate(self._crops.samples):
            name = img_meta["file_name"]
            if name not in seen:
                seen[name] = len(self.frames)
                self.frames.append((img_meta, anns))
                self.crops_of_frame.append([])
            self.crops_of_frame[seen[name]].append(crop_idx)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        img_meta, anns = self.frames[idx]
        path = os.path.join(self.img_dir, img_meta["file_name"])
        image = np.asarray(Image.open(path).convert("RGB"))

        h, w = image.shape[:2]
        box = (0, 0, w, h)
        label = rasterise_crop(anns, box, self.out_size)
        frame = extract_crop(image, box, self.out_size)

        if self.transform is not None:
            from recog.augmentation import apply_with_mask
            out = apply_with_mask(self.transform, frame, label)
            frame, label = out["image"], out["mask"]

        chw = np.ascontiguousarray(frame.transpose(2, 0, 1)).astype(np.float32)
        return (torch.from_numpy(chw / 255.0),
                torch.from_numpy(label.astype(np.int64)))


def frame_split_from_crop_split(frames: WholeFrameSegDataset,
                                crop_train_indices: Sequence[int],
                                ) -> Tuple[List[int], List[int]]:
    """Frame-level train/val indices that cannot leak into the crop split.

    A frame is a training frame only when EVERY crop it contributes is in
    ``crop_train_indices``. A frame carrying even one validation crop is
    held out, because training on it would mean the whole-frame model had
    already seen the exact pixels the comparison scores it on.

    This is asymmetric on purpose - it can only ever move frames OUT of
    the whole-frame training set, so the whole-frame arm is if anything
    handicapped. A comparison that is unfair in the direction of the
    hypothesis being tested is the one worth having; the reverse would be
    unpublishable.
    """
    train_set = set(int(i) for i in crop_train_indices)
    train, val = [], []
    for f_idx, crop_indices in enumerate(frames.crops_of_frame):
        if all(c in train_set for c in crop_indices):
            train.append(f_idx)
        else:
            val.append(f_idx)
    return train, val


def predict_val_crops_from_frames(segmenter, full_dataset,
                                  val_indices: Sequence[int],
                                  batch_size: int = 1) -> List[np.ndarray]:
    """``seg_evaluate.predict_val_crops``' signature, whole-frame inside.

    Drop-in replacement passed to ``seg_evaluate.evaluate(..., predict=)``.
    Same contract: one NATIVE-resolution label map per entry of
    ``val_indices``, in ``val_indices``' own order. Every metric
    downstream is then the identical code path the ROI arm uses, which is
    the entire point - the two arms differ in exactly one step.

    Per crop: run the model once over the whole frame, upsample the
    label map back to the frame's native size with NEAREST (labels must
    never be interpolated - the mean of class 2 and class 4 is class 3, a
    different object), then cut out this crop's own box. The frame
    prediction is cached, so a frame feeding several crops is inferred
    once; ``predict_val_crops`` caches its decode for the same reason and
    measured 14 redundant decodes out of 126 crops on this split.

    ``batch_size`` is accepted and ignored. It exists so the two
    predictors are substitutable without the caller special-casing;
    frames are large enough that batching them is a memory question
    rather than a throughput one.
    """
    from PIL import Image

    from recog.seg_evaluate import crop_image_path

    order = sorted(range(len(val_indices)),
                   key=lambda p: full_dataset.samples[
                       val_indices[p]][0]["file_name"])

    preds: List[Optional[np.ndarray]] = [None] * len(val_indices)
    cached_name: Optional[str] = None
    cached_pred: Optional[np.ndarray] = None

    for pos in order:
        img_meta, _anns, unit_box = full_dataset.samples[val_indices[pos]]
        if img_meta["file_name"] != cached_name:
            image = np.asarray(
                Image.open(crop_image_path(full_dataset, img_meta))
                .convert("RGB"))
            h, w = image.shape[:2]
            small = segmenter.segment(image)          # model's own input size
            ys = (np.arange(h) * small.shape[0] / h).astype(int) \
                .clip(0, small.shape[0] - 1)
            xs = (np.arange(w) * small.shape[1] / w).astype(int) \
                .clip(0, small.shape[1] - 1)
            cached_pred = small[ys][:, xs]
            cached_name = img_meta["file_name"]

        x0, y0, x1, y1 = (int(v) for v in unit_box)
        fh, fw = cached_pred.shape
        out = np.zeros((y1 - y0, x1 - x0), dtype=cached_pred.dtype)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(fw, x1), min(fh, y1)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                cached_pred[sy0:sy1, sx0:sx1]
        preds[pos] = out

    assert all(p is not None for p in preds)
    return [p for p in preds if p is not None]
