"""Recognition module: Faster R-CNN detector for batteries and cartridges.

The submodules split cleanly by concern:

* :mod:`recog.dataset` — Pascal-VOC parsing and a torch-ready dataset.
* :mod:`recog.augmentation` — Albumentations pipelines, with a
  numpy-only fallback.
* :mod:`recog.model` — the Faster R-CNN + ResNet-34 FPN factory.
* :mod:`recog.training` — the training loop.
* :mod:`recog.inference` — runtime detectors (learned + heuristic).
* :mod:`recog.evaluate` — VOC-style mAP and pose-error metrics.
* :mod:`recog.synth_dataset` — a procedural dataset generator used in
  tests and the software-only demo.
"""
