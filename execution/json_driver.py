"""A second driver, over a second encoding — to prove where the seam is.

**This driver does not talk to a real robot, and is not a step toward
one.** No hardware of any kind speaks this protocol. It exists for one
reason: a claim that "encode / decode / send / recv is the whole vendor
boundary" is worth nothing until a second encoding has been put through
the same :class:`execution.driver.RobotDriver` and made to pass the same
conformance suite, byte-for-byte identically. That is what this file and
``execution/mock_json_server.py`` are for. If the suite passes for both
drivers, the seam is where the design says it is. If it had not, the
design would have been wrong and the report would have said so.

Everything about the encoding is chosen to *differ* from the KUKA one,
because a second driver that framed the same way would prove nothing:

===================  =========================  ==========================
                     KUKA (``execution``)       here
===================  =========================  ==========================
framing              fixed 16 bytes             4-byte big-endian length
                                                prefix, then that many
                                                bytes of UTF-8 JSON
integrity            CRC-16/MODBUS trailer      none; TCP's checksum, and
                                                a JSON parse error
position             integer millimetres        float **metres**, no
                                                quantiser at all
orientation          **not carried**            unit quaternion, carried
redundancy token     **not carried**            carried, opaque
place Z              **not carried**            carried
cycle time           ``uint16`` ms, saturates   float seconds, unbounded
                     at 65 535
gripper              ``VACUUM_ON`` opcode on    a distinct message type
                     the motion channel         with width and force
halt                 opcode 0x06                ``{"type": "halt"}``
pick-and-place       **two** stateful frames    **one** frame; both poses
                     with the place XY latched  travel in it
``validate()``       ``UNKNOWN`` (no model)     real, against a declared
                                                envelope
===================  =========================  ==========================

Not one line of the retry loop, the latch, the deadline or the
escalation is repeated below. Everything in that column is inherited,
which is the point.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from common.types import RobotStatus, RobotStatusCode, WorkspacePoint
from .driver import (
    Capabilities,
    DriverPolicy,
    Gripper,
    Pose,
    Reachability,
    Request,
    RequestKind,
    TcpFrameDriver,
)

#: Length-prefix width, and the ceiling the prefix is allowed to name. A
#: length-prefixed protocol that trusts the prefix lets a peer ask for a
#: 4 GiB allocation with four bytes.
LENGTH_PREFIX_BYTES = 4
MAX_FRAME_BYTES = 64 * 1024

#: The envelope this backend's (simulated) controller enforces. Unlike
#: the KUKA driver, this one can answer ``validate`` for real, which is
#: what makes ``Reachability`` a three-valued answer rather than a
#: two-valued one with a spare label.
ENVELOPE_RADIUS_M = 0.9
ENVELOPE_Z_MIN_M = -0.05
ENVELOPE_Z_MAX_M = 0.9


class NonFiniteCoordinate(ArithmeticError):
    """A coordinate is NaN or infinite, so no frame can carry it.

    Subclasses ``ArithmeticError``, not ``ValueError``, deliberately —
    the same reasoning as
    :class:`execution.protocol.CoordinateOutOfRange`. The base driver's
    transient tuple catches ``ValueError``; a request that cannot be
    encoded will not encode on the second attempt either, so it must
    fall through to the fatal path and fire a halt. This is the second
    driver's version of the exact defect class that once let an
    out-of-int32 coordinate escape without a stop, and it is here so the
    conformance suite can exercise the catch-all on a backend that has
    no ``struct`` in it.
    """


class JsonProtocolError(ValueError):
    """A frame arrived that this end cannot parse. Transient by design."""


def encode_frame(obj: dict) -> bytes:
    """One dict to one length-prefixed JSON frame."""
    body = json.dumps(obj, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_FRAME_BYTES:
        raise JsonProtocolError(
            f"frame body of {len(body)} bytes exceeds the {MAX_FRAME_BYTES} "
            f"byte ceiling")
    return struct.pack(">I", len(body)) + body


def decode_frame(frame: bytes) -> dict:
    """One length-prefixed JSON frame to one dict."""
    if len(frame) < LENGTH_PREFIX_BYTES:
        raise JsonProtocolError(
            f"frame shorter than its {LENGTH_PREFIX_BYTES}-byte length prefix")
    (declared,) = struct.unpack(">I", frame[:LENGTH_PREFIX_BYTES])
    body = frame[LENGTH_PREFIX_BYTES:]
    if declared != len(body):
        raise JsonProtocolError(
            f"length prefix says {declared} bytes, body is {len(body)}")
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JsonProtocolError(f"unparseable frame body: {exc}") from exc
    if not isinstance(obj, dict):
        raise JsonProtocolError(f"frame body is a {type(obj).__name__}, "
                                f"not an object")
    return obj


def _finite(*values: float) -> None:
    for v in values:
        if not math.isfinite(v):
            raise NonFiniteCoordinate(
                f"coordinate {v!r} is not finite; no frame can carry it")


def pose_to_json(pose: Pose) -> dict:
    _finite(pose.x_m, pose.y_m, pose.z_m, *pose.orientation)
    return {
        "position_m": [pose.x_m, pose.y_m, pose.z_m],
        "orientation_wxyz": list(pose.orientation),
        "frame": pose.frame,
        "tool": pose.tool,
        # Opaque. This end does not interpret it, does not validate it,
        # and passes it through unchanged — which is the only correct
        # handling for a token whose meaning belongs to a controller.
        "redundancy": pose.redundancy,
    }


@dataclass(frozen=True)
class JsonDriverConfig:
    host: str = "127.0.0.1"
    port: int = 55600
    handshake_timeout_ms: int = 2000
    command_timeout_ms: int = 5000
    retry_pause_ms: int = 50
    max_retries: int = 3

    def driver_policy(self) -> DriverPolicy:
        return DriverPolicy(
            handshake_timeout_ms=self.handshake_timeout_ms,
            command_timeout_ms=self.command_timeout_ms,
            retry_pause_ms=self.retry_pause_ms,
            max_retries=self.max_retries,
        )


class JawGripper(Gripper):
    """A two-finger gripper with width, force, and a *holding* answer.

    Present to show what the neutral :class:`~execution.driver.Gripper`
    capability is for. This project's vacuum degenerates to one bit and
    would have been perfectly happy as an opcode; a gripper that can
    close on nothing and say so is what forced the capability to have
    its own lifecycle in the first place.
    """

    def __init__(self, driver: "JsonRobotDriver") -> None:
        self._driver = driver
        self._holding: Optional[bool] = None

    @property
    def holding(self) -> Optional[bool]:
        return self._holding

    def grasp(self, width_m: Optional[float] = None,
              force_n: Optional[float] = None) -> RobotStatus:
        status = self._driver.execute(Request(
            RequestKind.GRIPPER, "grasp",
            payload={"close": True,
                     "width_m": 0.0 if width_m is None else float(width_m),
                     "force_n": 20.0 if force_n is None else float(force_n)}))
        self._holding = status.code is RobotStatusCode.SUCCESS
        return status

    def release(self) -> RobotStatus:
        status = self._driver.execute(Request(
            RequestKind.GRIPPER, "release", payload={"close": False}))
        self._holding = False
        return status


class JsonRobotDriver(TcpFrameDriver):
    """Length-prefixed JSON over TCP. A test backend, not a robot."""

    #: This protocol carries the place Z, so there is no controller-side
    #: constant for the task to reconcile against. ``None`` means "the
    #: planner's number reaches the robot".
    declared_insert_depth_mm = None

    def __init__(self, cfg: JsonDriverConfig) -> None:
        super().__init__(cfg.host, cfg.port, cfg.driver_policy(),
                         logger_name="execution.json_driver")
        self.cfg = cfg
        self._gripper = JawGripper(self)

    # ---- capability ------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="json-lengthprefixed-testbackend",
            accepts_cartesian=True,
            accepts_joint=False,
            carries_orientation=True,
            carries_redundancy=True,
            position_resolution_m=0.0,
            validate_is_real=True,
            gripper_shares_motion_channel=True,
            frames=("base", "table"),
            tools=("jaw",),
            lossy_notes=(
                "the gripper shares the motion socket, so a grasp cannot "
                "overlap an approach",
            ),
        )

    @property
    def gripper(self) -> JawGripper:
        return self._gripper

    def validate(self, target: Pose) -> Reachability:
        """A real answer, because this backend has a model to answer from.

        Contrast :meth:`execution.execution.KukaClient.validate`, which
        answers ``UNKNOWN`` for everything the frame can carry because
        that client holds no envelope at all. Both are correct; the
        point of :class:`~execution.driver.Reachability` having three
        values is that they can both be expressed.
        """
        if not all(math.isfinite(v)
                   for v in (target.x_m, target.y_m, target.z_m)):
            return Reachability.UNREACHABLE
        if not ENVELOPE_Z_MIN_M <= target.z_m <= ENVELOPE_Z_MAX_M:
            return Reachability.UNREACHABLE
        radius = math.sqrt(target.x_m ** 2 + target.y_m ** 2
                           + target.z_m ** 2)
        return (Reachability.REACHABLE if radius <= ENVELOPE_RADIUS_M
                else Reachability.UNREACHABLE)

    # ---- the four hooks --------------------------------------------------

    def encode(self, request: Request) -> bytes:
        if request.kind is RequestKind.HANDSHAKE:
            obj = {"type": "handshake", "protocol": 1}
        elif request.kind is RequestKind.HALT:
            obj = {"type": "halt"}
        elif request.kind is RequestKind.MOVE:
            obj = {"type": "move",
                   "target": pose_to_json(request.target),
                   "motion": request.motion.value}
        elif request.kind is RequestKind.GRIPPER:
            obj = {"type": "gripper", **dict(request.payload)}
        else:
            obj = {"type": request.payload.get("type", request.name)}
            if request.target is not None:
                obj["target"] = pose_to_json(request.target)
            for key, value in request.payload.items():
                if key != "type":
                    obj[key] = (pose_to_json(value)
                                if isinstance(value, Pose) else value)
        return encode_frame(obj)

    def decode(self, frame: bytes) -> RobotStatus:
        obj = decode_frame(frame)
        name = obj.get("status")
        try:
            code = RobotStatusCode[str(name)]
        except KeyError as exc:
            # Same rule as the KUKA driver, for the same reason: a status
            # this build does not know must not be quietly rewritten into
            # one it does. Raising a ValueError makes it transient, so it
            # retries and then escalates.
            raise JsonProtocolError(
                f"unknown status code {name!r}") from exc

        pos = obj.get("pose_m") or [0.0, 0.0, 0.0]
        if not isinstance(pos, list) or len(pos) != 3:
            raise JsonProtocolError(f"malformed pose_m: {pos!r}")
        cycle_s = obj.get("cycle_s", 0.0)
        if not isinstance(cycle_s, (int, float)) or cycle_s < 0:
            raise JsonProtocolError(f"malformed cycle_s: {cycle_s!r}")

        return RobotStatus(
            code=code,
            current_pose=WorkspacePoint(pos[0] * 1000.0, pos[1] * 1000.0,
                                        pos[2] * 1000.0),
            cycle_time_ms=float(cycle_s) * 1000.0,
            message=str(obj.get("message", "")),
        )

    def recv(self, deadline: float) -> bytes:
        """Read a length prefix, then exactly that many bytes.

        This is the method that could not have been inherited from the
        KUKA driver: its read loop is ``while len(buf) < STATUS_LEN``,
        which is a fixed-length fact wearing the clothes of a general
        one. The *deadline* is general and comes from the base; the
        framing is not and lives here.
        """
        header = self.read_exactly(LENGTH_PREFIX_BYTES, deadline)
        (declared,) = struct.unpack(">I", header)
        if declared > MAX_FRAME_BYTES:
            raise JsonProtocolError(
                f"peer declared a {declared}-byte frame, over the "
                f"{MAX_FRAME_BYTES}-byte ceiling")
        return header + self.read_exactly(declared, deadline)

    # ---- application shape ----------------------------------------------

    def move_to(self, target: WorkspacePoint) -> RobotStatus:
        return self.move(Pose.from_point(target))

    def pick_place_requests(
        self,
        pick: WorkspacePoint,
        place: WorkspacePoint,
        transport_height_mm: float,
        vacuum_level_percent: int,
    ) -> Tuple[Request, ...]:
        """**One** request. Both poses travel in it, including the place Z.

        The contrast with
        :meth:`execution.execution.KukaClient.pick_place_requests` is the
        argument for returning a sequence rather than executing one: on
        that backend a cycle is two frames whose order latches state, and
        on this one it is a single self-describing message. A shared
        ``pick_and_place(pose) -> RobotStatus`` signature would have made
        those look identical from the outside, and one of them would have
        been wrong.
        """
        return (
            Request(
                RequestKind.VENDOR, "pick_place",
                target=Pose.from_point(pick),
                payload={"type": "pick_place",
                         "place": Pose.from_point(place),
                         "transport_height_m": transport_height_mm / 1000.0,
                         "grip_force_n": float(vacuum_level_percent)},
            ),
        )


__all__ = [
    "ENVELOPE_RADIUS_M",
    "JawGripper",
    "JsonDriverConfig",
    "JsonProtocolError",
    "JsonRobotDriver",
    "MAX_FRAME_BYTES",
    "NonFiniteCoordinate",
    "decode_frame",
    "encode_frame",
    "pose_to_json",
]
