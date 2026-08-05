"""End-to-end autonomous recognition → planning → execution loop.

This is the concrete realisation of the sequential flow in PPR §5.4::

    perception → twin update → queue rebuild → command → status → repeat

Everything is driven from a single top-level YAML (typically
``configs/demo.yaml``) which inlines the three module configs and the
``mode`` block. Two axes are configurable:

* image source — ``synthetic`` (cycle through a folder of pre-rendered
  PNGs) or ``camera`` (USB webcam via cv2);
* robot backend — ``mock`` (spawn a local mock KUKA simulator) or
  ``real`` (connect to a physical KUKA controller).

Run from the project root::

    python main.py --config configs/demo.yaml

The entry point :func:`run` returns a dict of run statistics — the
integration tests rely on that return value to assert the pipeline
made forward progress.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:  # pragma: no cover - import guard
    import cv2
except Exception as exc:  # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc

from common.config import load_demo_config
from common.logging import get_logger
from common.types import PickPlacePose, RobotStatusCode
from execution.execution import ExecutionConfig, KukaClient
from plan.placement_area import PlacementAreaExtractor
from plan.planner import Planner, PlannerConfig
from plan.scene import WorkspaceBounds
from recog.inference import load_detector

log = get_logger("autopick.main")


# ------------------------------------------------------- image sources ---

def _synthetic_source(directory: str) -> Iterable[np.ndarray]:
    """Cycle through every ``*.png`` in ``directory``, returning RGB frames."""
    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise RuntimeError(
            f"No synthetic images in {directory} — run "
            "`python -m recog.synth_dataset` first"
        )
    idx = 0
    while True:
        bgr = cv2.imread(str(files[idx % len(files)]))
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        idx += 1


def _camera_source(device: int = 0) -> Iterable[np.ndarray]:  # pragma: no cover
    """Read frames from a USB camera."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
) -> Iterable[np.ndarray]:
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


def _build_planner(plan_cfg: Dict[str, Any]) -> Planner:
    planner_cfg = PlannerConfig.from_dict(plan_cfg)
    extractor = PlacementAreaExtractor(
        safety_margin_px=int(
            plan_cfg.get("cartridge", {}).get("safety_margin_px", 5),
        ),
        mm_per_cell=float(
            plan_cfg.get("occupancy_grid", {}).get(
                "resolution_mm_per_cell", 1.5,
            ),
        ),
        mm_per_px=planner_cfg.mm_per_px,
    )
    return Planner(planner_cfg, extractor, _build_workspace(plan_cfg))


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


# --------------------------------------------------------- main loop ---

def run(config_path: str) -> Dict[str, int]:
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

    planner = _build_planner(plan_cfg)

    ckpt_dir = recog_cfg.get("training", {}).get(
        "checkpoint_dir", "recog/checkpoints",
    )
    detector = load_detector(
        checkpoint=f"{ckpt_dir}/best.pt",
        cfg=recog_cfg,
    )

    images = _image_source(mode_cfg, _resolve_image_dir(mode_cfg, recog_cfg))
    img_iter = iter(images)

    max_cycles = int(mode_cfg.get("max_cycles", 10))
    stats: Dict[str, int] = {
        "cycles": 0,
        "placed": 0,
        "pick_failed": 0,
        "place_failed": 0,
        "empty_queue": 0,
    }

    # --- cycle -----------------------------------------------------------
    try:
        with KukaClient(exec_conf) as kuka:
            for cycle_idx in range(max_cycles):
                if not _run_one_cycle(
                    cycle_idx, img_iter, detector, planner, kuka, stats,
                ):
                    break
    finally:
        if srv is not None:
            srv.shutdown()

    log.info("Run summary: %s", stats)
    return stats


def _run_one_cycle(
    cycle_idx: int,
    img_iter,
    detector,
    planner: Planner,
    kuka: KukaClient,
    stats: Dict[str, int],
) -> bool:
    """Run a single perception → plan → execute step.

    Returns ``True`` to keep looping, ``False`` to stop (empty queue).
    """
    img_rgb = next(img_iter)

    # 1. Perception.
    t0 = time.perf_counter()
    snap = detector(img_rgb)
    dt_perc = (time.perf_counter() - t0) * 1000

    # 2. Planning.
    t0 = time.perf_counter()
    queue: List[PickPlacePose] = planner.cycle(snap, img_rgb)
    dt_plan = (time.perf_counter() - t0) * 1000

    log.info(
        "cycle=%d perc=%.1fms plan=%.1fms queue=%d",
        cycle_idx, dt_perc, dt_plan, len(queue),
    )

    if not queue:
        stats["empty_queue"] += 1
        return False

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


# ---------------------------------------------------------- CLI --------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end pick-and-place pipeline.",
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":  # pragma: no cover
    _cli()
