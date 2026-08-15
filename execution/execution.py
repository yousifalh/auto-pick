"""KUKA Ethernet KRL 3.1 driver.

Encodes the vendor-neutral requests defined in :mod:`execution.driver`
into the 16-byte binary frames of :mod:`execution.protocol` and drives
the KUKA controller over a blocking TCP socket.

**Where the safety contract now lives.** It is not in this file. Every
route out of a command — including the handshake, including an exception
the author did not anticipate — either returns a status the caller can
act on or attempts a halt and raises :class:`RobotFault`, and that is
owned by :class:`execution.driver.RobotDriver`, which seals the methods
that implement it. This class cannot weaken it, because the base class
refuses to define a subclass that rebinds any of those names. What this
class supplies is four things:

* ``encode`` — a neutral :class:`~execution.driver.Request` to 16 bytes;
* ``decode`` — 16 bytes to a :class:`~common.types.RobotStatus`;
* ``send`` — inherited from :class:`~execution.driver.TcpFrameDriver`;
* ``recv`` — read exactly ``STATUS_LEN`` bytes inside the deadline.

**What is genuinely KUKA here** is smaller than it looks: a default IP,
the opcode table, the CRC, and the fact that frames are 16 bytes long.
The retry policy, the latch, the escalation and the deadline are not
KUKA and never were.

**What this driver does not do**, so nothing downstream assumes it:
there is no heartbeat and no watchdog (nothing ever sends
``OpCode.HEARTBEAT``), and there is **no sequence number**, so a reply
that arrives after its deadline is mis-paired with whatever goes out
next. That hazard is not closed and cannot be closed from this end.
What *is* closed is this client's contribution to it: the base's retry
loop no longer re-sends a request outside
:attr:`~execution.driver.RobotDriver.IDEMPOTENT_KINDS`, so a
``PICK_AND_PLACE`` that times out halts the arm instead of being
commanded a second time. It performs
**no workspace check** beyond asking whether a coordinate fits in the
wire's fields — reachability is enforced by the planner and by the
controller's software limits, neither of which this class can see, so
:meth:`KukaClient.validate` answers ``UNKNOWN`` for everything the frame
can carry.

**The application is not here either.** The pick-and-place choreography,
the transport height, the vacuum level and the insert depth belong to
:class:`execution.task.PickPlaceTask`. A suction cup with a percentage
level is this cell's tooling, not a property of KUKA controllers, and a
driver that carried it would be exporting one gripper to every future
arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from common.types import RobotStatus, RobotStatusCode, WorkspacePoint
from .driver import (
    Capabilities,
    DriverPolicy,
    Gripper,
    MotionKind,
    Pose,
    Reachability,
    Request,
    RequestKind,
    RobotEstop,
    RobotFault,
    TcpFrameDriver,
)
from .protocol import (
    STATUS_LEN,
    X_MM_MAX,
    X_MM_MIN,
    Y_MM_MAX,
    Y_MM_MIN,
    Z_MM_MAX,
    Z_MM_MIN,
    OpCode,
    pack_command,
    unpack_status,
)

# The CRC-16/MODBUS polynomial `execution.protocol.crc16_modbus`
# implements. `configs/execution.yaml` states it too; the two are
# cross-checked in ExecutionConfig.from_dict rather than left to agree
# by luck.
_CRC_POLYNOMIAL = 0xA001

#: The depth the controller descends to when inserting a cell, in
#: millimetres.
#:
#: This constant exists because the insert depth used to live in three
#: places that could not be checked against each other: ``routines.src``
#: descending ``LIN place_pos``, ``mock_kuka_server._INSERT_Z_MM``, and
#: the planner's ``PlannerConfig.place_insert_height_mm``. The 16-byte
#: frame has one Z field and the pick needs it, so ``pose.place.z_mm`` is
#: computed by the planner, validated, and then **not transmitted** —
#: there is nowhere on the wire to put it.
#:
#: Two of the three are now one: the simulator imports this value. The
#: third is the planner's, and it cannot be merged because it is a
#: different thing (what the planner *wants*) from this one (what the
#: controller *does*). :class:`execution.task.PickPlaceTask` compares
#: them on every cycle and says so when they disagree, which is the most
#: a host with no place-Z field on the wire can honestly do.
KUKA_CONTROLLER_INSERT_Z_MM = 2.0


def wire_mm(v: float) -> int:
    """Quantise a millimetre float to the signed integer the wire carries.

    This is the ONE lossy step between the planner's float millimetres
    and the controller, and it used to be ``int()``. ``int()`` truncates
    toward zero, so every commanded coordinate lost up to 0.999 mm,
    always toward the workspace origin, and — because the truncation is
    toward *zero*, not toward negative infinity — the bias **reverses
    sign at 0**: ``int(12.9) == 12`` but ``int(-12.9) == -12``, so two
    cartridges on opposite sides of the origin were both pulled inward,
    toward each other. On a 4.25 mm shipping-wall inset that is 23 % of
    the margin, the same order as the 8.3 % / 5.2 % unsafe placements
    the README documents.

    ``round`` instead: the error is bounded by 0.5 mm, it is symmetric
    about zero (``round(-x) == -round(x)`` for every x), and Python's
    round-half-to-even makes it unbiased over a population of poses
    rather than systematically outward. Half-integers are the only
    values where the two differ from round-half-up, and no planner
    output is a half-integer by construction.

    The nanometre snap first is not decoration. The interface pose is in
    metres (SI is the only honest choice when one backend counts
    millimetres and another counts metres), so a millimetre value makes a
    round trip through ``/1000`` and ``*1000`` before it arrives here and
    can land one ulp off an exact half. Snapping at 1e-9 m — a thousand
    times finer than the best repeatability in the KR 6 R700 datasheet —
    puts the tie back where the planner put it, and changes nothing else.

    Nothing clamps here on purpose. A coordinate outside the field the
    wire has for that axis raises
    :class:`execution.protocol.CoordinateOutOfRange` in
    :func:`execution.protocol.pack_command`, which the driver turns into
    a halt; silently saturating a pose to 2**31-1 would command a motion
    nobody asked for. **The three axes do not have the same field**: x
    and y are int32 and z is int16, so z is rejected at +/-32 767 mm and
    x and y are not. This docstring used to say "outside int32" for all
    three, which was wrong for the one axis that points at the table.
    """
    return int(round(round(v, 6)))


# ---------------------------------------------------- configuration ---

@dataclass
class ExecutionConfig:
    """Execution-layer knobs loaded from ``configs/execution.yaml``.

    Three different things live here and they are separated by
    :meth:`driver_policy` and by :class:`execution.task.TaskConfig`:
    the transport (``host``, ``port``), the driver's policy (the three
    timeouts and ``max_retries``), and the *application's* choreography
    (``transport_height_mm``, ``vacuum_level_percent``). Only the middle
    group is vendor-neutral; the last group is this cell's tooling and
    this cell's task.

    ``approach_height_mm`` and ``insert_height_mm`` used to be parsed,
    stored, unit-tested and read by nothing — editing them changed no
    behaviour anywhere in the system. Both are gone; the approach and
    insert depths are controller-side, hardcoded in
    ``krl_prog/routines.src`` (+60 mm approach, ``LIN place_pos``
    insert), because the 16-byte frame has no field that could carry
    them. See :data:`KUKA_CONTROLLER_INSERT_Z_MM`.
    """

    host: str = "172.31.1.147"
    port: int = 54600
    handshake_timeout_ms: int = 2000
    command_timeout_ms: int = 5000
    heartbeat_interval_ms: int = 50
    max_retries: int = 3
    transport_height_mm: float = 80.0
    vacuum_level_percent: int = 80

    def driver_policy(self) -> DriverPolicy:
        """The vendor-neutral half of this config."""
        return DriverPolicy(
            handshake_timeout_ms=self.handshake_timeout_ms,
            command_timeout_ms=self.command_timeout_ms,
            retry_pause_ms=self.heartbeat_interval_ms,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExecutionConfig":
        kuka = cfg.get("kuka", {}) or {}
        motion = cfg.get("motion", {}) or {}

        # The descriptive `kuka:` keys are checked rather than ignored.
        # A config file that states a 16-byte frame or polynomial 0xA001
        # is making a claim about this module, and a claim nothing reads
        # is a claim nothing keeps true.
        from .protocol import COMMAND_LEN

        declared_len = kuka.get("command_length_bytes")
        if declared_len is not None and int(declared_len) != COMMAND_LEN:
            raise ValueError(
                f"kuka.command_length_bytes = {declared_len} but "
                f"execution.protocol packs {COMMAND_LEN}-byte frames")
        declared_poly = kuka.get("crc_polynomial")
        if declared_poly is not None and int(declared_poly) != _CRC_POLYNOMIAL:
            raise ValueError(
                f"kuka.crc_polynomial = 0x{int(declared_poly):04X} but "
                f"execution.protocol implements 0x{_CRC_POLYNOMIAL:04X}; "
                "editing this key does not change the CRC")
        declared_cat = kuka.get("stop_category")
        if declared_cat is not None and int(declared_cat) != 0:
            raise ValueError(
                f"kuka.stop_category = {declared_cat}; this client only "
                "implements the IEC 60204 Category-0 immediate stop")

        return cls(
            host=kuka.get("host", cls.host),
            port=int(kuka.get("port", cls.port)),
            handshake_timeout_ms=int(
                kuka.get("handshake_timeout_ms", cls.handshake_timeout_ms),
            ),
            command_timeout_ms=int(
                kuka.get("command_timeout_ms", cls.command_timeout_ms),
            ),
            heartbeat_interval_ms=int(
                kuka.get("heartbeat_interval_ms", cls.heartbeat_interval_ms),
            ),
            max_retries=int(kuka.get("max_retries", cls.max_retries)),
            transport_height_mm=float(
                motion.get("transport_height_mm", cls.transport_height_mm),
            ),
            vacuum_level_percent=int(
                motion.get("vacuum_level_percent", cls.vacuum_level_percent),
            ),
        )


# -------------------------------------------------------- the tool ---

class VacuumGripper(Gripper):
    """This cell's suction cup, behind the neutral gripper capability.

    It is in band — ``VACUUM_ON`` and ``VACUUM_OFF`` are opcodes on the
    same channel as motion, with the level in ``aux_u16`` — and it is
    the only surveyed system that does that. It works here only because
    vacuum is one bit; :attr:`Gripper.holding` answers ``None`` because
    this tooling has no grasp sensor the host can read (on real
    hardware ``$IN[10]`` tells the *controller*, and the controller
    folds the answer into ``PICK_FAILED``).

    ``width_m`` and ``force_n`` are accepted and ignored, loudly: a
    caller asking a suction cup for 30 newtons at 40 mm has the wrong
    tool, and silence would let that reach the arm.
    """

    def __init__(self, driver: "KukaClient", level_percent: int) -> None:
        self._driver = driver
        self.level_percent = int(level_percent)

    def grasp(self, width_m: Optional[float] = None,
              force_n: Optional[float] = None) -> RobotStatus:
        if width_m is not None or force_n is not None:
            self._driver.log.warning(
                "VacuumGripper.grasp ignores width_m=%r and force_n=%r: a "
                "suction cup has neither. The vacuum level is a percentage "
                "and comes from the task's config.", width_m, force_n)
        return self._driver.execute(Request(
            RequestKind.GRIPPER, "VACUUM_ON",
            payload={"on": True, "level_percent": self.level_percent}))

    def release(self) -> RobotStatus:
        return self._driver.execute(Request(
            RequestKind.GRIPPER, "VACUUM_OFF", payload={"on": False}))


# ------------------------------------------------------------ driver ---

class KukaClient(TcpFrameDriver):
    """Blocking TCP driver speaking the 16-byte command/status protocol."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        super().__init__(cfg.host, cfg.port, cfg.driver_policy(),
                         logger_name="execution.kuka")
        self.cfg = cfg

    # ---- capability ------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="kuka-ethernetkrl-16b",
            accepts_cartesian=True,
            accepts_joint=False,
            carries_orientation=False,
            carries_redundancy=False,
            position_resolution_m=0.001,
            validate_is_real=False,
            gripper_shares_motion_channel=True,
            frames=("base",),
            tools=("vacuum",),
            lossy_notes=(
                "orientation is dropped: the frame carries x/y/z only, and "
                "krl_prog/routines.src fills A/B/C from $POS_ACT",
                "the redundancy token is dropped: KUKA's S/T status and turn "
                "bits have no field, so the controller resolves the "
                "configuration itself",
                "position quantises to whole millimetres, +/-0.5 mm",
                "z is int16 while x and y are int32, so z is rejected at "
                "+/-32 767 mm and x/y are not",
                "no place-Z field: the insert depth is controller-side, "
                "KUKA_CONTROLLER_INSERT_Z_MM",
            ),
        )

    @property
    def gripper(self) -> VacuumGripper:
        return VacuumGripper(self, self.cfg.vacuum_level_percent)

    def validate(self, target: Pose) -> Reachability:
        """Only what this driver can honestly answer.

        ``UNREACHABLE`` when a coordinate does not fit the field the wire
        has for it — a real check nothing else in the system performs,
        and the only one that belongs here. ``UNKNOWN`` otherwise: this
        client holds no kinematic model, no envelope and no software
        limits. The +/-350 mm square lives in ``configs/planning.yaml``
        and is enforced by ``plan.scene.WorkspaceBounds``; the KR 6
        R700's 706 mm reach is modelled only by the simulator; on real
        hardware the KRC's software limits decide. A driver claiming
        ``REACHABLE`` here would be reporting a guess as a fact.
        """
        x_mm, y_mm, z_mm = target.xyz_mm
        for value, lo, hi in ((wire_mm(x_mm), X_MM_MIN, X_MM_MAX),
                              (wire_mm(y_mm), Y_MM_MIN, Y_MM_MAX),
                              (wire_mm(z_mm), Z_MM_MIN, Z_MM_MAX)):
            if not lo <= value <= hi:
                return Reachability.UNREACHABLE
        return Reachability.UNKNOWN

    # ---- the four hooks --------------------------------------------------

    def encode(self, request: Request) -> bytes:
        op, x, y, z, aux = self._frame_fields(request)
        return pack_command(op, x, y, z, aux)

    def decode(self, frame: bytes) -> RobotStatus:
        s = unpack_status(frame)
        try:
            code = RobotStatusCode(s["code"])
        except ValueError as exc:
            # Substituting TIMEOUT here made an unknown code from a
            # future firmware indistinguishable from an ordinary
            # placement failure. Raise instead: it is a ValueError, so
            # the base retries and then escalates.
            raise ValueError(f"unknown status code {s['code']}") from exc

        message = ""
        if s["cycle_ms_saturated"]:
            # aux_u16 inbound is the cycle time and it stops counting at
            # 65 535 ms. Saying so beats feeding a censored sample into a
            # mean as though it were measured.
            message = ("cycle time saturated at the wire's 65535 ms "
                       "ceiling; the real cycle was at least that long")
            self.log.warning("mock/controller reported %s", message)

        return RobotStatus(
            code=code,
            # Integral by construction: the wire carries integer
            # millimetres, so a reported pose has whole-millimetre
            # resolution even though the field is typed float. Nothing
            # here can distinguish "the arm is at exactly 10 mm" from
            # "it is at 10.4 mm and the controller reported 10".
            current_pose=WorkspacePoint(s["x_mm"], s["y_mm"], s["z_mm"]),
            cycle_time_ms=float(s["cycle_ms"]),
            message=message,
        )

    def recv(self, deadline: float) -> bytes:
        """One fixed-length status frame, inside the whole-frame deadline."""
        return self.read_exactly(STATUS_LEN, deadline)

    def _frame_fields(self, request: Request):
        """Map a neutral request onto (opcode, x, y, z, aux).

        This method is the whole of the KUKA command model. Note what it
        cannot express: a motion type (the frame has no field, so
        ``MotionKind`` is dropped and ``routines.src`` decides PTP versus
        LIN), a velocity (likewise; the cap is ``$VEL.CP`` controller-
        side), and an orientation.
        """
        target = request.target
        if target is None:
            x = y = z = 0
        else:
            x_mm, y_mm, z_mm = target.xyz_mm
            x, y, z = wire_mm(x_mm), wire_mm(y_mm), wire_mm(z_mm)
            if target.has_orientation:
                self.log.warning(
                    "%s carries an orientation this wire cannot send; the "
                    "controller will use $POS_ACT's A/B/C. See "
                    "KukaClient.capabilities.lossy_notes.", request.name)
            if target.redundancy is not None:
                self.log.warning(
                    "%s carries a redundancy token this wire cannot send; "
                    "the controller resolves the configuration itself.",
                    request.name)

        if request.kind is RequestKind.HANDSHAKE:
            return OpCode.HANDSHAKE, 0, 0, 0, 0
        if request.kind is RequestKind.HALT:
            return OpCode.HALT, 0, 0, 0, 0
        if request.kind is RequestKind.MOVE:
            return OpCode.MOVE_TO, x, y, z, 0
        if request.kind is RequestKind.GRIPPER:
            on = bool(request.payload.get("on"))
            return (
                OpCode.VACUUM_ON if on else OpCode.VACUUM_OFF,
                0, 0, 0,
                int(request.payload.get("level_percent", 0)) if on else 0,
            )
        # VENDOR: the request names its own opcode.
        op = request.payload.get("op")
        if not isinstance(op, OpCode):
            raise ValueError(
                f"vendor request {request.name!r} must carry an OpCode in "
                f"payload['op'], got {op!r}")
        return op, x, y, z, int(request.payload.get("aux", 0))

    # ---- KUKA-shaped conveniences ---------------------------------------
    #
    # Not part of the driver interface. They exist because this
    # protocol's command model is not the interface's: PICK_AND_PLACE is
    # a two-frame stateful exchange and nothing about that generalises.

    #: The depth this backend's controller actually inserts at. The task
    #: compares the planner's intent against it; see
    #: :data:`KUKA_CONTROLLER_INSERT_Z_MM`.
    declared_insert_depth_mm = KUKA_CONTROLLER_INSERT_Z_MM

    def move_to(self, target: WorkspacePoint) -> RobotStatus:
        return self.move(Pose.from_point(target), MotionKind.UNSPECIFIED)

    def pick_place_requests(
        self,
        pick: WorkspacePoint,
        place: WorkspacePoint,
        transport_height_mm: float,
        vacuum_level_percent: int,
    ) -> tuple:
        """The request sequence one pick-and-place cycle takes here.

        **Two requests, and the order is load-bearing.** The
        ``PICK_AND_PLACE`` opcode carries one coordinate triple, the
        pick; the place XY is whatever (x, y) the arm was in when the
        subroutine begins (``routines.src``; ``mock_kuka_server`` reads
        ``self.x, self.y`` at entry). Nothing in the frame says so.

        Returning the sequence rather than executing it is the fix for
        the leak that most threatened this abstraction. The old
        ``pick_and_place(pose) -> RobotStatus`` signature said *one pose
        in, one status out*, which is precisely what the protocol is
        not; a second driver implementing that signature as one message
        would have placed correctly by accident or not at all, silently.
        A driver whose protocol carries both poses in one frame returns
        one request here, and the difference is visible to the caller
        instead of hidden behind a shared signature.

        ``place.z_mm`` does not appear below, because there is nowhere to
        put it. See :data:`KUKA_CONTROLLER_INSERT_Z_MM`.
        """
        return (
            Request(
                RequestKind.MOVE, "MOVE_TO (latches the place XY)",
                target=Pose.from_mm(place.x_mm, place.y_mm,
                                    transport_height_mm),
            ),
            Request(
                RequestKind.VENDOR, "PICK_AND_PLACE",
                target=Pose.from_point(pick),
                payload={"op": OpCode.PICK_AND_PLACE,
                         "aux": int(vacuum_level_percent)},
            ),
        )

    def vacuum(self, on: bool) -> RobotStatus:
        """Shim onto :attr:`gripper`, kept for the tests that predate it."""
        g = self.gripper
        return g.grasp() if on else g.release()


__all__ = ["KUKA_CONTROLLER_INSERT_Z_MM", "ExecutionConfig", "KukaClient",
           "RobotEstop", "RobotFault", "VacuumGripper", "wire_mm"]
