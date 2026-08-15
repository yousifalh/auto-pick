"""Detector runtime for the recognition module.

Two concrete detectors are provided:

* :class:`FasterRCNNDetector` — loads a trained torchvision Faster
  R-CNN checkpoint and runs the forward pass on CPU or GPU. Used in
  production once a checkpoint exists.
* :class:`HeuristicDetector` — pure-OpenCV colour / intensity heuristics
  that are tuned for the *synthetic* generator. It is fast,
  deterministic, and importantly has no torch dependency, so the rest
  of the pipeline remains exercisable in CPU-only CI.

:func:`load_detector` is the public factory the main loop calls. It
returns the Faster R-CNN detector when a usable checkpoint is present,
otherwise emits a warning and returns the heuristic fallback.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from common.types import BBox, ClassLabel, Detection, Snapshot

try:  # pragma: no cover - import guard
    import cv2
except Exception as exc:  # pragma: no cover - hard dep
    raise ImportError("opencv-python is required for recog.inference") from exc

try:  # pragma: no cover - import guard
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - executed only in torch-free env
    _TORCH_AVAILABLE = False


# Torchvision's output labels use 1-indexed class ids. We map them to
# the project-wide :class:`ClassLabel` enum here, treating any id we
# don't recognise as background (and skipping it downstream).
_ID_TO_LABEL = {
    1: ClassLabel.BATTERY,
    2: ClassLabel.CARTRIDGE,
}


# ---------------------------------------------------- abstract base ----

class Detector:
    """Abstract detector — ``detect`` returns a :class:`Snapshot`.

    ``__call__`` is a thin alias for ``detect`` so every existing call
    site (``detector(image_rgb)`` in ``main.py``, ``eval_real.py``, and
    the older tests) keeps working unchanged. Subclasses only need to
    implement ``detect``.
    """

    def detect(self, image_rgb: np.ndarray) -> Snapshot:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, image_rgb: np.ndarray) -> Snapshot:
        return self.detect(image_rgb)


# ------------------------------------------------- learned detector ----

class FasterRCNNDetector(Detector):
    """Thin wrapper around a torchvision Faster R-CNN checkpoint."""

    def __init__(self, checkpoint_path: str, cfg: dict, segmenter=None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for FasterRCNNDetector")

        # Optional second-stage per-cartridge segmenter (recog.bay_segmenter
        # .BaySegmenter). None by default so every existing construction
        # site is unaffected and cartridge_masks stays empty.
        self.segmenter = segmenter

        from recog.model import build_fasterrcnn

        self.model = build_fasterrcnn(cfg)
        # weights_only=True is not optional, for the same reason it is not
        # optional in bay_segmenter.py: the default unpickles arbitrary
        # Python objects, so loading a checkpoint is arbitrary code
        # execution by whoever produced the file. No .pt is committed here,
        # so every user following the README's --checkpoint commands has
        # obtained this file from somewhere else - that is the normal path,
        # not an edge case. A detector checkpoint holds tensors and nothing
        # else, so the restriction costs nothing.
        state = torch.load(checkpoint_path, map_location="cpu",
                           weights_only=True)
        self.model.load_state_dict(
            state["model"] if "model" in state else state,
        )
        self.model.eval()

        # Inference resolution is a real accuracy knob, not a detail, and it
        # is set here rather than in build_fasterrcnn because the same
        # transform runs during training — changing it there would rescale the
        # training renders and invalidate the tuned anchor_scales.
        #
        # The detector learns on 1280x720 renders where a cell is ~28px wide.
        # A 3024x4032 phone photo at torchvision's default min_size=800 puts a
        # real cell at ~52px, nearly 2x the trained scale, and accuracy falls
        # off a cliff above that. Measured on the 6-image real-photo set:
        #
        #   min_size   mAP@0.50   mAP@0.75
        #        500     0.6136     0.4040
        #        640     0.6613     0.1912
        #        800     0.4571     0.0227   <- torchvision default
        #       1000     0.0770     0.0030
        #       1600     0.0000     0.0000
        #
        # 500 puts a real cell at ~33px, closest to the trained scale, and
        # gives by far the best localisation. Retune if the deployed camera's
        # working distance puts parts at a different pixel size.
        model_cfg = (cfg or {}).get("model", {})
        self.model.transform.min_size = (
            int(model_cfg.get("inference_min_size", 500)),
        )
        self.model.transform.max_size = int(
            model_cfg.get("inference_max_size", 900),
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.to(self.device)

        # The 255.0 that `detect` divides the frame by, as a 0-dim tensor
        # on the device, built once.
        #
        # It is a TENSOR and not the Python scalar 255 on purpose, and
        # this is measured, not stylistic. `float_tensor.div_(255)` takes
        # torch's Scalar overload, which on CUDA is implemented as a
        # multiply by the reciprocal: over the 256 possible byte values
        # it disagrees with a true divide on 126 of them, by 1 ulp.
        # numpy's `uint8_array.astype(np.float32) / 255.0` - the form
        # this replaced - is a true divide. Feeding the 1-ulp-different
        # frame through this detector on an RTX 3060 did NOT just move
        # the last bits of the scores (max delta 6.9e-5): it returned a
        # DIFFERENT detection set, because NMS and top-k reorder on those
        # ties. Dividing by a 0-dim tensor takes the tensor-tensor path,
        # which is correctly rounded and reproduces the old frame, and
        # the old detections, bit for bit.
        self._u8_scale = torch.tensor(255.0, device=self.device)

    @torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
    def detect(self, image_rgb: np.ndarray) -> Snapshot:
        import torch  # type: ignore

        # Transfer the frame as uint8 and do the widening and the
        # transpose ON the device. The previous form built
        # `image_rgb.astype(np.float32) / 255.0` and then
        # `np.transpose(...)`: two 11 MB host arrays for a 1280x720
        # frame, of which the second is a stride-(1, W*3, 3) view, so
        # `.to(device)` had to materialise a THIRD contiguous buffer by a
        # cache-hostile gather and then push 11 MB across PCIe where
        # 2.76 MB of uint8 carries the identical information.
        #
        # ascontiguousarray is a no-op for every caller in this repo
        # (main.py's cv2.cvtColor and eval_real.py's np.array(im) both
        # hand over a fresh contiguous array) and costs one 2.76 MB uint8
        # copy for one that does not - `astype` used to launder a
        # negative-stride view such as `bgr[:, :, ::-1]`, which
        # `torch.from_numpy` refuses outright, and losing that would be a
        # behaviour change rather than a speed-up.
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .float()
            .div_(self._u8_scale)
        )
        out = self.model([tensor])[0]

        # THREE device-to-host transfers per frame, not 3N. Zipping the
        # output tensors themselves made each `lbl.item()`,
        # `score.item()` and `box.cpu()` its own copy and its own stream
        # synchronisation - 3N of them, so roughly 30-90 per frame at the
        # shipped 0.70 confidence threshold, each one draining the queue
        # to read a single number. The loop below touches the device
        # exactly zero times; it walks numpy rows.
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        detections: List[Detection] = []
        for box, lbl, score in zip(boxes, labels, scores):
            label = _ID_TO_LABEL.get(int(lbl), ClassLabel.BACKGROUND)
            if label == ClassLabel.BACKGROUND:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(BBox(x1, y1, x2, y2), label, float(score)),
            )

        snap = Snapshot(
            detections=detections,
            image_shape=image_rgb.shape[:2],
            timestamp_ns=time.time_ns(),
        )

        # Second stage: one batched segment_batch call for every cartridge
        # in this frame. attach_cartridge_masks is itself a no-op when
        # there are no cartridges, so a segmenter that never sees a
        # cartridge never gets called.
        if self.segmenter is not None:
            attach_cartridge_masks(snap, image_rgb, self.segmenter)

        return snap


# ----------------------------------------------- heuristic fallback ----

class HeuristicDetector(Detector):
    """Deterministic OpenCV detector tuned for the synthetic generator.

    The generator paints cartridges in a narrow green hue band and
    batteries as metallic, bright, low-saturation cylinders. That is
    enough structure for a cascade of HSV thresholding + morphology +
    connected-components to pull out valid boxes.

    This detector is not expected to generalise to real factory-floor
    images — its job is to keep the end-to-end pipeline exercisable
    until a trained model is available.
    """

    # Empirically tuned for the default (480, 640) synthetic scene.
    def __init__(
        self,
        min_battery_area: int = 800,
        min_cartridge_area: int = 25_000,
    ) -> None:
        self.min_battery_area = min_battery_area
        self.min_cartridge_area = min_cartridge_area

    def detect(self, image_rgb: np.ndarray) -> Snapshot:
        h, w = image_rgb.shape[:2]
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        dets: List[Detection] = []
        cartridge_rects: List[BBox] = self._detect_cartridges(hsv, dets)
        self._detect_batteries(image_rgb, hsv, cartridge_rects, dets)

        return Snapshot(
            detections=dets,
            image_shape=(h, w),
            timestamp_ns=time.time_ns(),
        )

    # ---- stage 1: cartridges --------------------------------------------

    def _detect_cartridges(
        self, hsv: np.ndarray, dets: List[Detection],
    ) -> List[BBox]:
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 200))
        green_mask = cv2.morphologyEx(
            green_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        )

        rects: List[BBox] = []
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for c in contours:
            if cv2.contourArea(c) < self.min_cartridge_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            bb = BBox(x, y, x + cw, y + ch)
            rects.append(bb)
            dets.append(Detection(bb, ClassLabel.CARTRIDGE, 0.91))

        # Stash the cartridge mask on the instance so the battery pass
        # can subtract it without recomputing.
        self._green_mask = green_mask
        return rects

    # ---- stage 2: batteries ---------------------------------------------

    def _detect_batteries(
        self,
        image_rgb: np.ndarray,
        hsv: np.ndarray,
        cartridge_rects: List[BBox],
        dets: List[Detection],
    ) -> None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Upper ~12 % of grey values as a "metallic-bright" proxy, with a
        # sanity floor so very dark scenes don't trigger on noise.
        thr = max(130, int(np.percentile(gray, 88)))
        _, bright = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        sat = hsv[:, :, 1]
        metallic = cv2.bitwise_and(bright, cv2.inRange(sat, 0, 60))
        metallic = cv2.bitwise_and(
            metallic, cv2.bitwise_not(self._green_mask),
        )
        metallic = cv2.morphologyEx(
            metallic, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        )

        contours, _ = cv2.findContours(
            metallic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for c in contours:
            if cv2.contourArea(c) < self.min_battery_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            bb = BBox(x, y, x + cw, y + ch)

            # Drop anything clearly inside a cartridge.
            if any(_contains(o, bb) and bb.iou(o) > 0.05
                   for o in cartridge_rects):
                continue

            # 18650/21700 cells are tall cylinders. Reject contours whose
            # aspect ratio isn't roughly in that band.
            aspect = bb.height / max(1.0, bb.width)
            if not 2.0 <= aspect <= 5.0:
                continue

            dets.append(Detection(bb, ClassLabel.BATTERY, 0.82))


def _contains(outer: BBox, inner: BBox) -> bool:
    """``True`` if ``inner`` is fully contained within ``outer``."""
    return (
        inner.xmin >= outer.xmin
        and inner.ymin >= outer.ymin
        and inner.xmax <= outer.xmax
        and inner.ymax <= outer.ymax
    )


# --------------------------------------------------- batched masks -----

def attach_cartridge_masks(snapshot: Snapshot, image_rgb: np.ndarray,
                           segmenter) -> None:
    """Segment every cartridge in ``snapshot``, in ONE batched call.

    Keys are indices into ``snapshot.detections``, not positions within
    the cartridge subset: the planner looks a cartridge up by the
    detection it came from (``Cartridge.detection_index``), and keying
    by subset position misaligns every mask the moment a battery appears
    before a cartridge in the list.

    One call, not a loop. Measured on an RTX 3060 at fp16/256 (the
    deployed config): eight cartridges cost 61.8 ms looped and 18.1 ms
    batched, against FDR 10.4's 50 ms end-to-end budget. One receipt,
    `docs/receipts/seg_eval.txt` (regenerated by `recog.seg_evaluate`),
    is the source for both figures - quote it, not a new measurement.
    """
    idx, crops = [], []
    h, w = image_rgb.shape[:2]
    for i, det in enumerate(snapshot.detections):
        if det.label is not ClassLabel.CARTRIDGE:
            continue
        x0 = max(0, int(det.bbox.xmin))
        y0 = max(0, int(det.bbox.ymin))
        x1 = min(w, int(det.bbox.xmax))
        y1 = min(h, int(det.bbox.ymax))
        if x1 <= x0 or y1 <= y0:
            continue
        idx.append(i)
        crops.append(image_rgb[y0:y1, x0:x1])

    if not crops:
        return

    for i, mask in zip(idx, segmenter.segment_batch(crops)):
        snapshot.cartridge_masks[i] = mask


# ---------------------------------------------------------- factory ----

def load_detector(checkpoint: Optional[str], cfg: Optional[dict],
                  segmenter=None) -> Detector:
    """Return the Faster R-CNN detector if usable, else the heuristic.

    ``segmenter`` is the optional second stage
    (:class:`recog.bay_segmenter.BaySegmenter`, or anything with the same
    ``segment_batch(crops)`` contract). It rides through to
    :class:`FasterRCNNDetector`, which calls it once per frame and fills
    ``Snapshot.cartridge_masks``.

    Passing one is INCOMPATIBLE with the heuristic fallback, and that is
    raised rather than warned. :class:`HeuristicDetector` has no second
    stage: it would return a snapshot with empty ``cartridge_masks``,
    ``SegmentationPlacementAreaExtractor`` would then raise ``ValueError``
    for every cartridge, and ``plan/planner.py``'s blanket
    ``except Exception`` would absorb all of them - a pipeline that
    reports a clean run of zero placements while the model that was
    supposed to be under test never ran at all. The whole point of
    wiring the segmenter in is that its absence is visible.
    """
    from common.logging import get_logger

    log = get_logger("recog.inference")

    if checkpoint and Path(checkpoint).exists() and _TORCH_AVAILABLE:
        log.info("Loading FasterRCNNDetector from %s", checkpoint)
        return FasterRCNNDetector(checkpoint, cfg or {}, segmenter=segmenter)

    if segmenter is not None:
        raise RuntimeError(
            f"a segmenter was supplied but the detector checkpoint "
            f"{checkpoint!r} is missing or torch is unavailable "
            f"(torch available: {_TORCH_AVAILABLE}). HeuristicDetector "
            "cannot run a second stage, so it would silently produce no "
            "cartridge masks and every cartridge would be skipped. "
            "Refusing to run a segmentation config that cannot segment.")

    log.warning(
        "No checkpoint at %s (or torch unavailable). Using HeuristicDetector "
        "fallback — suitable only for synthetic-image pipeline testing.",
        checkpoint,
    )
    return HeuristicDetector()


__all__ = [
    "Detector",
    "FasterRCNNDetector",
    "HeuristicDetector",
    "attach_cartridge_masks",
    "load_detector",
]
