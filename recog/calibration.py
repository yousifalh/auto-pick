"""Ground sample distance — how many millimetres one pixel spans.

ONE definition of that arithmetic, in one module, because the project has
already paid twice for having two. ``configs/planning.yaml`` carries a
placeholder ``camera.mm_per_px_x`` ("Replace with real intrinsics when
available"); ``configs/demo_seg.yaml`` carried a second, hand-derived
0.625; and ``recog.seg_evaluate`` computed a third from the generator
config. All three agreed on paper and only one of them was ever right for
a given frame.

Two different questions live here and must not be conflated:

``resolve_mm_per_px(synth_cfg)``
    The generator's NOMINAL framing, ``layout.area[0] * 1000 /
    render.res[0]``. This is the framing at ``margin = 1.0, zoom = 1.0``.
    It is the right number for a corpus rendered at a fixed framing and
    the wrong number for this one. Kept, with its name and behaviour
    unchanged, because ``recog.calibrate_tau`` and ``recog.seg_ablation``
    quote it in shipped receipts.

``frame_mm_per_px(meta)``
    The TRUE ground sample distance of ONE rendered frame,
    ``camera.ortho_scale * 1000 / width``. ``recog/synth3d/world.py``'s
    ``setup_camera`` sets ``ortho_scale = need * margin * zoom`` with
    ``margin`` drawn from ``[1.02, 1.10]`` and ``zoom`` from
    ``param_space.zoom``, so on a scale-randomised corpus every frame has
    its own GSD and NO frame is at the nominal framing. Over
    ``recog/dataset3d_seg`` the true GSD runs 0.490-1.045 mm/px against a
    nominal 0.625: the planner under-read 24 of 30 cartridges by 27 % at
    the median, and over-read the rest, which is the unsafe direction
    (docs/superpowers/specs/2026-08-11-placement-feasibility.md sections
    1.1 and 6).

The derived GSD is not taken on trust. Cross-checked against the median
long side of each frame's unoccluded flat 18650 annotations, which are
65.0 mm by CAD, it agrees to better than 1 % on every frame tested
(same spec, section 1.1).

A REAL camera has one fixed GSD and would supply it as a calibration
constant. That case is served by the configured fallback in
``plan.planner.PlannerConfig.mm_per_px``; this module serves the case
where the frame itself carries its own calibration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from common.config import load_yaml


def mm_per_px_from_extent(extent_m: float, res_w_px: float) -> float:
    """Millimetres per pixel for a frame ``extent_m`` metres across.

    The one place metres-across-the-long-axis becomes millimetres per
    pixel. Both callers below go through it so a change to the unit
    convention cannot land in one and miss the other.
    """
    if res_w_px <= 0:
        raise ValueError(
            f"render width must be positive, got {res_w_px!r}: mm_per_px "
            "is undefined for a zero-width frame")
    return float(extent_m) * 1000.0 / float(res_w_px)


# ------------------------------------------------- nominal framing ------

def resolve_mm_per_px(synth_cfg: Dict[str, Any]) -> float:
    """``layout.area[0] * 1000 / render.res[0]`` - the generator's
    NOMINAL framing. Read from config rather than hardcoded, or it
    silently goes wrong the first time anyone changes the framing.

    This is the framing at ``margin = 1.0, zoom = 1.0``. If the corpus
    randomises either (``camera.margin_range``, ``param_space.zoom``),
    this is the framing of NO ACTUAL FRAME and
    :func:`frame_mm_per_px` is the number you want.
    """
    layout = synth_cfg.get("layout") or {}
    render = synth_cfg.get("render") or {}
    if "area" not in layout or "res" not in render:
        raise SystemExit(
            "error: synth config is missing layout.area / render.res; "
            "cannot compute mm_per_px. Pass --synth-config explicitly.")
    return mm_per_px_from_extent(float(layout["area"][0]),
                                 float(render["res"][0]))


def load_synth_config(coco_path: str, override: Optional[str]
                      ) -> Tuple[Dict[str, Any], str]:
    """The render/layout config the dataset's images were actually
    generated under.

    Prefers the manifest.json sidecar next to ``coco_path`` - it is a
    frozen copy of the config at generation time, so it stays correct
    even if configs/synth3d.yaml is edited afterwards. Falls back to
    configs/synth3d.yaml (the authored default) only if no manifest is
    found, and to an explicit --synth-config if one is given.
    """
    if override:
        return load_yaml(override), str(override)

    manifest_path = Path(coco_path).parent / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        cfg = manifest.get("config")
        if cfg:
            return cfg, str(manifest_path)

    default = Path("configs/synth3d.yaml")
    if default.is_file():
        return load_yaml(default), str(default)

    raise SystemExit(
        "error: could not find a render/layout config to compute "
        f"mm_per_px from ({manifest_path} missing 'config', and "
        f"{default} not found). Pass --synth-config explicitly.")


# --------------------------------------------------- per-frame GSD ------

def frame_mm_per_px(meta: Mapping[str, Any]) -> float:
    """The TRUE ground sample distance of the frame ``meta`` describes.

    ``meta`` is one ``recog/dataset3d_seg/meta/<frame>.json`` document,
    as ``recog.generate3d`` writes it: ``camera.ortho_scale`` is the
    metres the LONGER render axis spans (``recog/synth3d/world.py``'s
    ``setup_camera`` docstring records the marker-cube measurement that
    confirms it is the longer axis), and ``width`` is that axis in
    pixels.

    Raises rather than guessing. A sidecar that exists but cannot answer
    the question is a broken sidecar, and a planner that quietly
    substituted a config constant for it is the defect this module was
    written to remove. A frame with NO sidecar at all is a different
    case - see :func:`frame_mm_per_px_for_image`, which returns ``None``
    so the caller can apply an explicitly configured fallback.
    """
    camera = meta.get("camera") or {}
    ortho = camera.get("ortho_scale")
    width = meta.get("width")
    if ortho is None:
        raise ValueError(
            "render metadata carries no camera.ortho_scale, so this "
            "frame's ground sample distance is unknown. A PERSPECTIVE "
            "camera (recog/synth3d/world.py setup_camera, cfg.ortho "
            "false) has no single scalar mm_per_px at all - the scale "
            "varies with depth - and substituting one would be a "
            "fabricated calibration, not a fallback.")
    if not width:
        raise ValueError(
            "render metadata carries no image `width`, so "
            "camera.ortho_scale cannot be turned into mm_per_px.")
    return mm_per_px_from_extent(float(ortho), float(width))


def frame_mm_per_px_for_image(image_path: Union[str, Path]
                              ) -> Optional[float]:
    """The GSD of the rendered frame at ``image_path``, or ``None``.

    ``None`` means "this frame carries no calibration metadata" - a
    photograph, or ``recog/synth_dataset.py``'s cv2-drawn rectangles,
    neither of which has a render sidecar. It does NOT mean "use 0.625";
    the caller decides what an unknown scale is worth, and
    ``plan.planner.Planner`` raises unless a fallback was configured
    deliberately.

    ``recog.generate3d`` writes ``<dataset>/images/<stem>.png`` beside
    ``<dataset>/meta/<stem>.json``; a sibling ``meta/`` directory next to
    the image itself is also accepted so a flat export still resolves.
    """
    img = Path(image_path)
    candidates = (img.parent.parent / "meta" / f"{img.stem}.json",
                  img.parent / "meta" / f"{img.stem}.json")
    for sidecar in candidates:
        if sidecar.is_file():
            with sidecar.open("r", encoding="utf-8") as fh:
                return frame_mm_per_px(json.load(fh))
    return None


__all__ = [
    "frame_mm_per_px",
    "frame_mm_per_px_for_image",
    "load_synth_config",
    "mm_per_px_from_extent",
    "resolve_mm_per_px",
]
