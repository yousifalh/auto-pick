"""End-to-end autonomous recognition → planning → execution loop.

This is the concrete realisation of the sequential flow in PPR §5.4::

    perception → twin update → queue rebuild → command → status → repeat

Everything is driven from a single top-level YAML (typically
``configs/demo.yaml``) which inlines the three module configs and the
``mode`` block. Three axes are configurable:

* image source — ``synthetic`` (cycle through a folder of pre-rendered
  PNGs) or ``camera`` (USB webcam via cv2);
* robot backend — ``mock`` (spawn a local mock KUKA simulator) or
  ``real`` (connect to a physical KUKA controller);
* placement-area path — heuristic (the default, torch-free) or the
  trained bay segmenter, selected by a ``mode.segmentation`` block.

Run from the project root::

    python main.py --config configs/demo.yaml         # torch-free
    python main.py --config configs/demo_seg.yaml \\
        --receipt docs/receipts/main_seg_run.txt      # segmenter in loop

The two configs are not interchangeable in their frames. ``demo.yaml``
reads ``recog/synth_dataset.py``'s flat green rectangles, which the
heuristic extractor was tuned for and on which the segmenter predicts
no ``bay`` at all; ``demo_seg.yaml`` reads Blender renders, which the
segmenter was trained on and where the heuristic returns nothing.
Pointing either one at the other's corpus produces a run that completes
and demonstrates nothing, which is why the segmentation path asserts it
actually segmented something rather than reporting a clean zero.

The entry point :func:`run` returns a dict of run statistics — the
integration tests rely on that return value to assert the pipeline
made forward progress.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    import cv2
except Exception as exc:  # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc

from common.config import load_demo_config
from common.logging import get_logger
from common.types import ClassLabel, PickPlacePose, RobotStatusCode
from execution.execution import ExecutionConfig, KukaClient
from plan.placement_area import (HeuristicPlacementAreaExtractor,
                                 SegmentationPlacementAreaExtractor)
from plan.planner import Planner, PlannerConfig
from plan.scene import WorkspaceBounds
from recog.inference import load_detector

log = get_logger("autopick.main")


# ------------------------------------------------------- image sources ---

Frame = Tuple[np.ndarray, Optional[float]]


def _synthetic_source(directory: str) -> Iterable[Frame]:
    """Cycle through every ``*.png`` in ``directory``.

    Yields ``(rgb, mm_per_px)``. The second element is the frame's OWN
    ground sample distance, read from the render sidecar
    ``<dataset>/meta/<stem>.json`` that ``recog.generate3d`` writes, or
    ``None`` for a corpus that has no sidecar (``recog/synth_dataset.py``
    draws its rectangles with cv2 and records no camera).

    The image source is where the calibration belongs. A frame source
    stands in for a camera, and a camera is the thing that knows how big
    a pixel is - not the planner's config file, which cannot know that
    ``recog/synth3d/world.py`` randomises the framing per scene and was
    consequently wrong on 24 of 30 cartridges
    (docs/superpowers/specs/2026-08-11-scale-calibration.md).
    """
    from recog.calibration import frame_mm_per_px_for_image

    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise RuntimeError(
            f"No synthetic images in {directory} — run "
            "`python -m recog.synth_dataset` first"
        )
    idx = 0
    while True:
        path = files[idx % len(files)]
        bgr = cv2.imread(str(path))
        yield (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
               frame_mm_per_px_for_image(path))
        idx += 1


def _camera_source(device: int = 0) -> Iterable[Frame]:  # pragma: no cover
    """Read frames from a USB camera.

    ``None`` for the scale: a webcam frame carries no calibration, so the
    configured fallback (``planning.camera.mm_per_px_x`` / ``mode
    .mm_per_px``) is what applies — and if none was configured the
    planner raises rather than inventing one.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
    finally:
        cap.release()


def _resolve_image_dir(
    mode_cfg: Dict[str, Any], recog_cfg: Dict[str, Any],
) -> str:
    """Directory the ``synthetic`` source cycles through.

    ``mode.img_dir`` wins, because the demo's frames and the detector's
    training set are not the same corpus: ``recognition.dataset.img_dir``
    points at the Blender-rendered dataset (``recog/dataset3d``), which is
    generated on demand and is not in the repo. Falling back to it keeps
    older configs that carry no ``mode.img_dir`` working unchanged.
    """
    explicit = mode_cfg.get("img_dir")
    if explicit:
        return str(explicit)
    return recog_cfg.get("dataset", {}).get("img_dir", "recog/dataset/images")


def _image_source(
    mode_cfg: Dict[str, Any], image_dir: str,
) -> Iterable[Frame]:
    src = mode_cfg.get("source", "synthetic")
    if src == "synthetic":
        return _synthetic_source(image_dir)
    if src == "camera":  # pragma: no cover
        return _camera_source(int(mode_cfg.get("camera_device", 0)))
    raise ValueError(f"Unsupported image source: {src}")


# ----------------------------------------------------------- wiring ----

def _build_workspace(plan_cfg: Dict[str, Any]) -> WorkspaceBounds:
    bounds = plan_cfg.get("camera", {}).get("workspace_bounds_mm", {}) or {}
    return WorkspaceBounds(
        x_min_mm=float(bounds.get("x_min", -350)),
        x_max_mm=float(bounds.get("x_max",  350)),
        y_min_mm=float(bounds.get("y_min", -350)),
        y_max_mm=float(bounds.get("y_max",  350)),
    )


def _build_planner(
    plan_cfg: Dict[str, Any],
    segmentation: bool = False,
    mm_per_px: Any = None,
) -> Planner:
    """Planner with whichever placement-area extractor ``mode`` selects.

    ``segmentation`` is not a preference, it is a statement about what
    the detector will hand over. ``SegmentationPlacementAreaExtractor``
    needs a ``label_map`` per cartridge and raises if it does not get
    one; ``HeuristicPlacementAreaExtractor`` cannot accept one at all
    (``plan.planner._accepts_label_map`` checks its signature and never
    passes the kwarg). Selecting the wrong one for the detector that was
    built is therefore a hard failure in one direction and a silently
    ignored model in the other, so ``run()`` derives both from the same
    ``mode.segmentation`` block rather than letting them be configured
    apart.

    ``mm_per_px`` overrides ``planning.camera.mm_per_px_x`` as the
    FALLBACK calibration for BOTH the extractor and the planner's
    pixel->workspace mapping. It is what applies to a frame that carries
    no scale of its own; a frame that does carry one
    (``Snapshot.mm_per_px``, filled from the render sidecar by
    ``_run_one_cycle``) overrides it per frame, and ``Planner.cycle``
    resolves the two in one place so the extractor and the planner cannot
    drift apart. Passing ``None`` here AND having no
    ``planning.camera.mm_per_px_x`` means uncalibrated frames raise
    ``UnknownScale`` instead of being measured against a guess.
    """
    planner_cfg = PlannerConfig.from_dict(plan_cfg)
    if mm_per_px is not None:
        planner_cfg.mm_per_px = float(mm_per_px)

    mm_per_cell = float(
        plan_cfg.get("occupancy_grid", {}).get("resolution_mm_per_cell", 1.5),
    )

    if segmentation:
        cartridge_cfg = plan_cfg.get("cartridge", {}) or {}
        kwargs: Dict[str, Any] = {}
        if "wall_inset_mm" in cartridge_cfg:
            kwargs["wall_inset_mm"] = float(cartridge_cfg["wall_inset_mm"])
        extractor: Any = SegmentationPlacementAreaExtractor(
            mm_per_cell=mm_per_cell,
            mm_per_px=planner_cfg.mm_per_px,
            **kwargs,
        )
    else:
        extractor = HeuristicPlacementAreaExtractor(
            safety_margin_px=int(
                plan_cfg.get("cartridge", {}).get("safety_margin_px", 5),
            ),
            mm_per_cell=mm_per_cell,
            mm_per_px=planner_cfg.mm_per_px,
        )
    return Planner(planner_cfg, extractor, _build_workspace(plan_cfg))


def _build_segmenter(seg_cfg: Dict[str, Any]):
    """The second-stage segmenter named by ``mode.segmentation``, if any.

    Returns ``None`` when the block is absent — that is the torch-free
    default path and it must stay reachable on a machine with neither
    torch nor a checkpoint.

    Every failure below raises. A ``mode.segmentation`` block that names
    a checkpoint which is not there is a typo, not a reason to quietly
    run the heuristic pipeline instead and report a normal-looking set
    of statistics.
    """
    if not seg_cfg:
        return None

    checkpoint = seg_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "mode.segmentation is present but names no `checkpoint`. "
            "Remove the block to run the heuristic path deliberately.")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"mode.segmentation.checkpoint does not exist: {checkpoint} — "
            "train one with `python -m recog.seg_training --config "
            "configs/segmentation.yaml`")

    model_cfg: Dict[str, Any] = {}
    cfg_path = seg_cfg.get("config")
    if cfg_path:
        from common.config import load_yaml
        model_cfg = load_yaml(cfg_path).get("model", {}) or {}

    from recog.bay_segmenter import BaySegmenter

    log.info("Loading BaySegmenter from %s", checkpoint)
    return BaySegmenter(
        checkpoint=str(checkpoint),
        crop_size=int(model_cfg.get("crop_size", 256)),
        half=bool(model_cfg.get("half", True)),
        num_classes=int(model_cfg.get("num_classes", 6)),
    )


def _start_robot(exec_cfg: Dict[str, Any], mode_cfg: Dict[str, Any]):
    """Return ``(host, port, server_or_None)`` for the chosen backend."""
    if mode_cfg.get("robot", "mock") != "mock":  # pragma: no cover
        return (
            exec_cfg["kuka"]["host"],
            int(exec_cfg["kuka"]["port"]),
            None,
        )

    from execution.mock_kuka_server import run_in_thread

    sim = exec_cfg.get("simulation", {}) or {}
    host = sim.get("listen_host", "127.0.0.1")
    port = int(sim.get("listen_port", 54600))
    srv, _t = run_in_thread(
        host=host,
        port=port,
        drop_prob=float(sim.get("drop_probability", 0.02)),
        ms_per_100mm=int(
            sim.get("simulated_move_time_ms_per_100mm", 180),
        ),
    )
    log.info("Mock robot listening on %s:%d", host, port)
    return host, port, srv


def _planned_cartridge_ids(planner: Planner) -> Set[int]:
    """Twin IDs that currently hold a placement area.

    Cartridge IDs are allocated monotonically and never reused
    (``plan.scene.EnvironmentModel._next_cartridge_id``), so differencing
    this set across a cycle counts cartridges the extractor newly
    succeeded on, not cartridges that merely persisted.
    """
    return {cid for cid, c in planner.env.cartridges.items()
            if c.placeable_rectangle is not None}


# --------------------------------------------------------- main loop ---

def run(config_path: str, receipt_path: Optional[str] = None) -> Dict[str, int]:
    """Run the full pipeline once. Returns a dict of summary statistics."""
    cfg = load_demo_config(config_path)
    recog_cfg = cfg["recognition"]
    plan_cfg = cfg["planning"]
    exec_cfg = cfg["execution"]
    mode_cfg = cfg.get("mode", {})

    # --- wire up robot, planner, detector --------------------------------
    host, port, srv = _start_robot(exec_cfg, mode_cfg)
    exec_conf = ExecutionConfig.from_dict(exec_cfg)
    exec_conf.host = host
    exec_conf.port = port

    seg_mode_cfg = mode_cfg.get("segmentation") or {}
    segmenter = _build_segmenter(seg_mode_cfg)

    planner = _build_planner(
        plan_cfg,
        segmentation=segmenter is not None,
        mm_per_px=mode_cfg.get("mm_per_px"),
    )

    ckpt_dir = recog_cfg.get("training", {}).get(
        "checkpoint_dir", "recog/checkpoints",
    )
    detector = load_detector(
        checkpoint=f"{ckpt_dir}/best.pt",
        cfg=recog_cfg,
        segmenter=segmenter,
    )

    image_dir = _resolve_image_dir(mode_cfg, recog_cfg)
    images = _image_source(mode_cfg, image_dir)
    img_iter = iter(images)

    max_cycles = int(mode_cfg.get("max_cycles", 10))
    stats: Dict[str, int] = {
        "cycles": 0,
        "placed": 0,
        "pick_failed": 0,
        "place_failed": 0,
        "empty_queue": 0,
        # Perception -> planning accounting, summed over cycles. Zeroes
        # here are what distinguish "the loop ran" from "the loop did
        # something"; see the check after the loop.
        "cartridges_detected": 0,
        "batteries_detected": 0,
        "cartridge_masks": 0,
        "placement_areas": 0,
        "queue_poses": 0,
        # Frames that carried their own ground sample distance. Zero
        # against a non-zero cycle count means every frame was planned
        # against the configured FALLBACK - which is correct for a fixed
        # camera and was the silent defect on this scale-randomised
        # corpus, so it is counted rather than assumed either way.
        "frames_with_scale": 0,
    }

    # --- cycle -----------------------------------------------------------
    # PPR 5.4's "nothing left to place -> stop" is right for a real cell
    # working one physical scene: an empty queue means the job is done.
    # It is wrong for a frame source that CYCLES THROUGH UNRELATED
    # SCENES, which is what `source: synthetic` actually is - there, an
    # empty queue on one render says nothing about the next. The
    # heuristic demo never noticed, because it finds a placement area in
    # every one of synth_dataset.py's green rectangles; the segmenter
    # predicts no `bay` on most crops, so the first such frame would end
    # the whole run. Defaults to True so configs/demo.yaml is unchanged.
    stop_on_empty = bool(mode_cfg.get("stop_on_empty_queue", True))

    try:
        with KukaClient(exec_conf) as kuka:
            for cycle_idx in range(max_cycles):
                if not _run_one_cycle(
                    cycle_idx, img_iter, detector, planner, kuka, stats,
                    stop_on_empty,
                ):
                    break
    finally:
        if srv is not None:
            srv.shutdown()

    stats["placement_disagreements"] = planner.placement_disagreement_count
    stats["bad_detector_boxes"] = planner.bad_detector_box_count
    stats["rescaled_area_drops"] = planner.rescaled_area_drop_count

    # A segmentation run that planned nothing is a FAILED run, not a
    # quiet one. The two ways this happens are both configuration
    # errors that otherwise look like a clean exit: frames the segmenter
    # was not trained on (recog/synth_dataset.py's flat green rectangles
    # produce no `bay` at all, hence zero placement areas), or a
    # detector that finds no cartridges to segment in the first place.
    if segmenter is not None and stats["placement_areas"] == 0:
        raise RuntimeError(
            f"segmentation was configured but produced no placement area "
            f"in {stats['cycles'] + stats['empty_queue']} cycles over "
            f"{image_dir}: {stats['cartridges_detected']} cartridges "
            f"detected, {stats['cartridge_masks']} segmented, "
            f"{stats['placement_disagreements']} disagreements, "
            f"{stats['bad_detector_boxes']} bad boxes. The segmenter is "
            "trained on Blender renders (recog/dataset3d_seg); on other "
            "corpora it predicts no `bay` and this pipeline plans "
            "nothing while still exiting cleanly.")

    log.info("Run summary: %s", stats)

    if receipt_path:
        _write_receipt(receipt_path, config_path, image_dir, seg_mode_cfg,
                       detector, planner, stats)
        log.info("Receipt written to %s", receipt_path)

    return stats


def _run_one_cycle(
    cycle_idx: int,
    img_iter,
    detector,
    planner: Planner,
    kuka: KukaClient,
    stats: Dict[str, int],
    stop_on_empty: bool = True,
) -> bool:
    """Run a single perception → plan → execute step.

    Returns ``True`` to keep looping, ``False`` to stop. An empty queue
    stops the run when ``stop_on_empty`` (the default, and what
    ``configs/demo.yaml`` gets); otherwise it advances to the next frame.
    """
    img_rgb, frame_mm_per_px = next(img_iter)

    # 1. Perception.
    t0 = time.perf_counter()
    snap = detector(img_rgb)
    dt_perc = (time.perf_counter() - t0) * 1000

    # The frame's own calibration, attached here rather than inside the
    # detector: the detector is handed an array and has no idea where it
    # came from, while this loop chose the source. None is a legitimate
    # value meaning "this frame carries no calibration" - Planner.cycle
    # then applies the configured fallback, or raises if there is none.
    snap.mm_per_px = frame_mm_per_px
    if frame_mm_per_px is not None:
        stats["frames_with_scale"] += 1

    n_cartridges = sum(
        1 for d in snap.detections if d.label is ClassLabel.CARTRIDGE)
    n_masks = len(snap.cartridge_masks)
    stats["cartridges_detected"] += n_cartridges
    stats["cartridge_masks"] += n_masks
    stats["batteries_detected"] += sum(
        1 for d in snap.detections if d.label is ClassLabel.BATTERY)

    # A detector carrying a segmenter that returns nothing for a frame
    # full of cartridges is a broken second stage, not an empty scene.
    # attach_cartridge_masks fills one entry per in-bounds cartridge box,
    # so this can only fire if the segmenter itself failed - and it would
    # otherwise be invisible, because every downstream consumer treats a
    # missing mask as "skip this cartridge".
    if getattr(detector, "segmenter", None) is not None \
            and n_cartridges and not n_masks:
        raise RuntimeError(
            f"cycle {cycle_idx}: {n_cartridges} cartridges detected but "
            "the segmenter attached no masks — the second stage is not "
            "running")

    # 2. Planning.
    t0 = time.perf_counter()
    before = _planned_cartridge_ids(planner)
    queue: List[PickPlacePose] = planner.cycle(snap, img_rgb)
    stats["placement_areas"] += len(_planned_cartridge_ids(planner) - before)
    stats["queue_poses"] += len(queue)
    dt_plan = (time.perf_counter() - t0) * 1000

    log.info(
        "cycle=%d perc=%.1fms plan=%.1fms cartridges=%d masks=%d queue=%d",
        cycle_idx, dt_perc, dt_plan, n_cartridges, n_masks, len(queue),
    )

    if not queue:
        stats["empty_queue"] += 1
        return not stop_on_empty

    # 3. Execute one pose, then re-plan next cycle.
    #    PPR §5.4: a failure triggers a fresh queue rebuild; a success
    #    marks the cell PLACED and pops it off.
    pose = queue[0]
    status = kuka.pick_and_place(pose)

    if status.code == RobotStatusCode.SUCCESS:
        planner.confirm_placement(
            pose.cartridge_id, pose.grid_row, pose.grid_col, True,
        )
        stats["placed"] += 1
    elif status.code == RobotStatusCode.PICK_FAILED:
        planner.confirm_placement(
            pose.cartridge_id, pose.grid_row, pose.grid_col, False,
        )
        stats["pick_failed"] += 1
    else:
        planner.confirm_placement(
            pose.cartridge_id, pose.grid_row, pose.grid_col, False,
        )
        stats["place_failed"] += 1

    stats["cycles"] += 1
    return True


# ---------------------------------------------------------- receipt ----

def _write_receipt(
    path: str,
    config_path: str,
    image_dir: str,
    seg_cfg: Dict[str, Any],
    detector,
    planner: Planner,
    stats: Dict[str, int],
) -> None:
    """Write a run receipt in docs/receipts' key-then-numbers format.

    Generated, never hand-written: the point of a receipt is that the
    numbers in it came out of a run rather than out of a paragraph, and
    every claim about this pipeline that was hand-carried into prose is
    the reason this file exists (see FDR v3 13.2.1 and the tau gate that
    outlived its own retirement by a commit).
    """
    import datetime

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = stats["cartridges_detected"]
    masked = stats["cartridge_masks"]
    planned = stats["placement_areas"]

    lines = [
        "",
        "End-to-end run: recognition -> planning -> execution",
        "=" * 66,
        f"  generated     : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"  command       : python main.py --config {config_path} "
        f"--receipt {path}",
        f"  config        : {config_path}",
        f"  frames        : {image_dir}",
        f"  detector      : {type(detector).__name__}",
        f"  segmenter     : {type(getattr(detector, 'segmenter', None)).__name__}",
        f"  seg checkpoint: {seg_cfg.get('checkpoint', '-')}",
        f"  extractor     : {type(planner.extractor).__name__}",
        f"  mm_per_px     : per-frame from the render sidecar "
        f"(camera.ortho_scale); fallback {planner.cfg.mm_per_px}",
        f"  frames w/ own scale: {stats['frames_with_scale']} of "
        f"{stats['cycles'] + stats['empty_queue']}",
        "",
        "Perception -> planning. `cartridges` counts cartridge detections",
        "summed over cycles (the same physical cartridge in two frames",
        "counts twice); `placement areas` counts twin cartridges that newly",
        "obtained one, and IDs are never reused, so it does not double-count",
        "a cartridge that simply persisted.",
        "",
        f"  cycles executed        : {stats['cycles']}",
        f"  cartridges detected    : {seen}",
        f"  cartridges segmented   : {masked}",
        f"  placement areas        : {planned}",
        f"  placement disagreements: {stats['placement_disagreements']}",
        f"  bad detector boxes     : {stats['bad_detector_boxes']}",
        f"  areas dropped (rescale): {stats['rescaled_area_drops']}",
        "",
        "Execution (mock KUKA unless mode.robot is `real`). A pose needs a",
        "placement area AND a free battery in the SAME frame, and the loop",
        "executes at most one pose per cycle before re-planning (PPR 5.4),",
        "so `placed` is bounded by cycles, not by placement areas.",
        "",
        f"  loose batteries detected: {stats['batteries_detected']}",
        f"  poses queued           : {stats['queue_poses']}",
        f"  placed                 : {stats['placed']}",
        f"  pick failed            : {stats['pick_failed']}",
        f"  place failed           : {stats['place_failed']}",
        f"  empty queue (run ended): {stats['empty_queue']}",
        "",
    ]

    if getattr(detector, "segmenter", None) is not None:
        lines += [
            "The remaining cartridges are not errors. The segmenter",
            "predicts no `bay` channel on most crops - those raise a plain",
            "RuntimeError ('no placeable area'), which is the ordinary",
            "'nothing to place here' state and is deliberately not counted",
            "as a disagreement. The `placement areas` figure is what the",
            "retired tau gate used to reduce further; it no longer does.",
            "",
        ]

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------- CLI --------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end pick-and-place pipeline.",
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--receipt", default=None,
        help="write a generated run receipt to this path "
             "(e.g. docs/receipts/main_seg_run.txt)",
    )
    args = parser.parse_args()
    run(args.config, receipt_path=args.receipt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
