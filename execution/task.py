"""The application layer: what a pick-and-place *is*, for this cell.

Everything here is a property of the task and the tooling, not of the
robot. It was in the driver until 2026-08-14 and it does not belong
there:

* ``vacuum(on: bool)`` — a suction cup with a percentage level. A
  two-finger gripper has neither. A driver interface carrying it would
  impose this cell's end effector on every future arm.
* ``pick_and_place(pose)`` — approach, descend, suck, lift, traverse,
  insert, blow off, retract. That is *this cell's cartridge-insertion
  choreography*, mirroring ``krl_prog/routines.src``. A different arm
  doing this task would run the same seven steps; a KUKA doing a
  different task would not.
* ``transport_height_mm`` and the insert depth — numbers about a
  cartridge tray.

Separating them is what lets the conformance suite in
``tests/conformance.py`` be about robots rather than about cartridges.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.logging import get_logger
from common.types import PickPlacePose, RobotStatus, RobotStatusCode
from .driver import RobotDriver, RobotFault

log = get_logger("execution.task")

# Codes an intermediate step may return and still leave the state the
# next step assumes.
_CONTINUE_CODES = (RobotStatusCode.OK, RobotStatusCode.SUCCESS)


@dataclass(frozen=True)
class TaskConfig:
    """The application's numbers, split out of ``ExecutionConfig``."""

    transport_height_mm: float = 80.0
    vacuum_level_percent: int = 80

    @classmethod
    def from_execution_config(cls, cfg) -> "TaskConfig":
        return cls(
            transport_height_mm=float(cfg.transport_height_mm),
            vacuum_level_percent=int(cfg.vacuum_level_percent),
        )


class PickPlaceTask:
    """One cartridge cell, picked from the tray and inserted into a bay.

    Holds a driver; does not subclass one. The driver's job is to get a
    request to a controller and a status back; deciding that a cycle is
    *a place-XY latch followed by a pick* is this class's job, and on a
    different protocol it would be a different sequence of requests.

    **The two-frame sequence is deliberately visible here.** The KUKA
    ``PICK_AND_PLACE`` opcode carries one coordinate triple — the pick —
    and the place XY is whatever the arm was at when the subroutine
    began. So a cycle is two commands whose order is load-bearing and
    which the frame does not describe. Writing that as
    ``driver.pick_and_place(pose)`` hid it behind a signature that
    implied one message; writing it as two named calls does not.
    """

    def __init__(self, driver: RobotDriver, cfg: TaskConfig) -> None:
        self.driver = driver
        self.cfg = cfg
        self._insert_depth_warned = False

    def run(self, pose: PickPlacePose) -> RobotStatus:
        """Run one full pick-and-place cycle. Returns the final status.

        The driver supplies the *sequence*; this method only sequences
        it, and stops early if a step that is not the last one comes
        back as anything other than OK or SUCCESS. On the KUKA backend
        step one latches the place XY, so running the pick after it
        failed would insert wherever the arm happened to be.
        """
        self._check_insert_depth(pose)

        requests = self.driver.pick_place_requests(
            pick=pose.pick,
            place=pose.place,
            transport_height_mm=self.cfg.transport_height_mm,
            vacuum_level_percent=self.cfg.vacuum_level_percent,
        )
        if not requests:
            raise RobotFault(
                f"{type(self.driver).__name__}.pick_place_requests returned "
                f"no requests; a cycle needs at least one")

        status = None
        for index, request in enumerate(requests):
            status = self.driver.execute(request)
            is_last = index == len(requests) - 1
            if not is_last and status.code not in _CONTINUE_CODES:
                log.warning(
                    "step %d/%d (%s) returned %s; abandoning the cycle "
                    "rather than running the remaining steps against a "
                    "state it did not establish",
                    index + 1, len(requests), request.name, status.code.name,
                )
                return status
        return status

    def _check_insert_depth(self, pose: PickPlacePose) -> None:
        """Say out loud that ``pose.place.z_mm`` is not going to the robot.

        The planner computes a place Z (``PlannerConfig.
        place_insert_height_mm``), ``PickPlacePose.place`` carries it,
        ``plan.scene.WorkspaceBounds`` validates it — and the 16-byte
        frame has one Z field, which the pick needs. So the value is
        computed, checked, and dropped.

        It cannot be transmitted without a frame-layout change and a
        protocol version bump, which is deliberately not done. What *can*
        be done is to stop the two numbers drifting apart in silence:
        compare the planner's intent against the depth the controller
        will actually use and log once per task when they disagree. A
        second driver picking a fourth insert depth is then a line in the
        log rather than a discovery at the tray.
        """
        declared = getattr(self.driver, "declared_insert_depth_mm", None)
        if declared is None or self._insert_depth_warned:
            return
        if abs(float(pose.place.z_mm) - float(declared)) <= 0.5:
            return
        self._insert_depth_warned = True
        log.warning(
            "the planner asked to insert at z = %.2f mm; this driver's "
            "controller descends to %.2f mm and the wire has no place-Z "
            "field to carry the difference. The planner's value is not "
            "reaching the robot. (Logged once per task.)",
            float(pose.place.z_mm), float(declared),
        )


__all__ = ["PickPlaceTask", "TaskConfig"]
