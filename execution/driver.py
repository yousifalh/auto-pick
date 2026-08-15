"""The vendor-neutral half of the execution layer.

This module holds the part of :mod:`execution.execution` that was never
KUKA: the failure taxonomy, the retry policy, the whole-command
deadline, the latching stop, and the escalation that every route out of
a command has to pass through. A driver supplies **four** things — an
encoder, a decoder, a byte-sink and a byte-source — and inherits
everything else.

Why a template method and not an abstract base class
----------------------------------------------------
This project has had a stop with three separate bypass routes: a
``struct.error`` from an out-of-range coordinate, a mid-frame close, and
a ``ConnectionResetError``. All three were closed by one mechanism —
**catch-all ordering** in the command loop: catch the transient tuple
first, then ``except Exception`` and escalate. An ABC cannot enforce
that. If ``execute()`` is the thing a driver implements, the second
driver's author writes ``except socket.timeout`` around their own socket
and opens bypass route four, in exactly the way routes one to three
opened.

So the escalation is not an interface a driver implements; it is code a
driver cannot reach. :class:`RobotDriver` owns

* ``connect`` / ``close`` / ``__enter__`` / ``__exit__`` and the
  refusal to reconnect while latched,
* ``_handshake`` and its retry policy,
* ``execute`` — the retry loop, the catch-all, the ESTOP latch, the
  fatal-status path, the retry-exhaustion path,
* ``halt`` and ``_request_halt`` — the best-effort stop, and the rule
  that a stop which could not be transmitted is logged CRITICAL and does
  **not** replace the error that prompted it,
* the closed-set guarantee: ``execute`` returns only codes a caller can
  act on.

and it **seals** those names: :meth:`RobotDriver.__init_subclass__`
raises ``TypeError`` at class-definition time if a subclass rebinds any
of them. That is the structural difference between an interface that
closes the defect class and one that repaints it.

``HALT`` is not an emergency stop
---------------------------------
The opcode this client used to call ``ESTOP`` is a *request to stop
moving*, sent in band over the application protocol. No robot surveyed
for this design — Universal Robots, ROS 2, Franka FCI, ABB EGM —
accepts a safety-rated stop that way: on UR, Franka and ABB it is a
hardware safety chain, and the ROS 2 action layer does not model one at
all. Clearing a latched stop is out of band everywhere, and on ABB near
a singularity it needs a human at the pendant.

So :meth:`RobotDriver.halt` is documented as *an in-band request to stop
moving, idempotent, explicitly not safety-rated, and — on this
backend — actionable only between commands*. That last clause is not a
hedge: this driver has one socket and one thread, so the only place a
halt can be issued is after a command has returned, and
``krl_prog/laptop_comm.src`` does not read another record until the one
it is executing has finished, so a halt sent mid-cycle would sit in the
EKI buffer until the cycle ended anyway. An in-cycle abort would need a
second channel and a KRL ``INTERRUPT``; neither exists here.
:meth:`RobotDriver.halt` states this in full.

It is still the right thing to send, and this driver still latches on
it — a latched client refusing to command further motion is a real
protection — but neither the name nor the docstring now claims a
function the wire cannot deliver.

:class:`RobotEstop` keeps its name deliberately. It is raised when the
**controller reports** ``RobotStatusCode.ESTOP``, i.e. when the machine
says it is in a Category-0 stop. That is a report about a real safety
state and is not the same thing as the request we send.

What this module deliberately does not model
--------------------------------------------
There is no streaming interface here. The design survey found that the
split between "the controller interpolates" and "the client
interpolates at a fixed rate" is not a parameter of one interface: a
streaming session has no completion event by construction, and silence
on it means *fault* rather than *idle*. Franka FCI and ABB EGM need
that second shape; this project's pick-and-place never will, and
nothing is built for it. Adding a rate argument here would not produce
it.

There is no velocity or duration on a command, because this project's
wire carries none and the cap is controller-side. There is no
guaranteed reachability check: :meth:`RobotDriver.validate` may
legitimately answer ``UNKNOWN``, and on this project's KUKA backend it
mostly does — the envelope is enforced by the planner and by the
controller's software limits, not here.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple

from common.logging import get_logger
from common.types import RobotStatus, RobotStatusCode, WorkspacePoint


# ------------------------------------------------------- exceptions ---

class RobotFault(RuntimeError):
    """A command ended with the arm in an unknown state.

    Subclasses :class:`RuntimeError` so callers written against the
    original ``RuntimeError`` contract keep working.
    """


class RobotEstop(RobotFault):
    """The controller reports a Category-0 stop, or this client is latched.

    Raised when the controller answers ``RobotStatusCode.ESTOP`` —
    including when it refuses the handshake because it is *already*
    stopped — and when a latched driver is asked to command motion
    again. The driver is single-use after this: clearing a Category-0
    stop is a deliberate act at the controller, out of band, and on some
    machines it needs an operator.
    """


# ------------------------------------------------------ value types ---

class Reachability(Enum):
    """The three honest answers to "can you reach that?".

    ``UNKNOWN`` is first-class, not a failure. Of the systems surveyed
    for this design, UR answers the question (``isPoseWithinSafetyLimits``),
    ROS 2 answers it one layer up (MoveIt's ``ComputeIK``), and Franka
    and ABB EGM do not answer it at all — EGM's manual says outright
    that "the robot will react quickly to all position references sent
    to the controller, also faulty ones". An interface that promised
    pre-flight rejection would be lying on two of the four.
    """

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class MotionKind(Enum):
    """What path the arm should take. Not every backend can honour it."""

    LINEAR = "linear"                  # KRL LIN, UR moveL
    JOINT_INTERPOLATED = "joint"       # KRL PTP, UR moveJ
    UNSPECIFIED = "unspecified"        # let the controller choose


@dataclass(frozen=True)
class Pose:
    """A pose in SI units, with everything a real arm needs to use it.

    Position in **metres**, orientation as a **unit quaternion**
    ``(w, x, y, z)``, plus the named frame and tool it is expressed in,
    plus an **opaque redundancy token**.

    Every field here earns its place from a specific backend:

    * *metres and a quaternion* — quaternion converts losslessly to and
      from every orientation form the survey found (KUKA A/B/C Euler,
      UR's rotation vector, Franka's 4x4 transform, ABB's quaternion).
      This project's wire carries integer millimetres and **no**
      orientation, which makes its driver a lossy backend; the loss is
      declared in :class:`Capabilities`, not hidden.
    * *frame and tool* — a pose is meaningless without them. UR poses
      are TCP-in-base, Franka is ``O_T_EE`` with ``NE_T_EE`` configured
      separately, ABB poses live in a work object, KUKA in
      ``$BASE``/``$TOOL``.
    * *redundancy* — KUKA's S/T status-and-turn bits, Franka's ``elbow``
      and UR's ``qnear`` seed all answer the same question (*which of
      the joint configurations that reach this pose?*) in three
      incompatible encodings. It cannot be abstracted, only carried. A
      driver that needs one and did not get one must fail loudly rather
      than pick a configuration for you.
    """

    x_m: float
    y_m: float
    z_m: float
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: str = "base"
    tool: str = "tool0"
    redundancy: Any = None

    @classmethod
    def from_mm(cls, x_mm: float, y_mm: float, z_mm: float, **kw) -> "Pose":
        """Build from millimetres, which is what this project's planner speaks."""
        return cls(x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, **kw)

    @classmethod
    def from_point(cls, point: WorkspacePoint, **kw) -> "Pose":
        return cls.from_mm(point.x_mm, point.y_mm, point.z_mm, **kw)

    @property
    def xyz_mm(self) -> Tuple[float, float, float]:
        return (self.x_m * 1000.0, self.y_m * 1000.0, self.z_m * 1000.0)

    @property
    def has_orientation(self) -> bool:
        """False for the identity quaternion, i.e. "nobody set one"."""
        return self.orientation != (1.0, 0.0, 0.0, 0.0)


class RequestKind(Enum):
    """What a request *is*, independent of how any driver encodes it."""

    HANDSHAKE = "handshake"
    MOVE = "move"
    HALT = "halt"
    GRIPPER = "gripper"
    VENDOR = "vendor"   # a backend-specific operation the base only sequences


@dataclass(frozen=True)
class Request:
    """One thing to ask the robot for, before any encoding.

    ``name`` is what appears in logs and in fault messages, so a driver
    should use its own vocabulary there (``"PICK_AND_PLACE"``,
    ``"moveL"``). ``payload`` carries whatever the backend needs that
    this type does not model; the base class never reads it.
    """

    kind: RequestKind
    name: str
    target: Optional[Pose] = None
    motion: MotionKind = MotionKind.UNSPECIFIED
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriverPolicy:
    """Timing and retry knobs. Vendor-neutral by construction."""

    handshake_timeout_ms: int = 2000
    command_timeout_ms: int = 5000
    retry_pause_ms: int = 50
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError(
                f"max_retries must be at least 1, got {self.max_retries}")
        for name in ("handshake_timeout_ms", "command_timeout_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)}")


@dataclass(frozen=True)
class Capabilities:
    """What a backend can actually do, queried after connect.

    The point of this type is that a *lossy* backend has to say so. This
    project's KUKA driver drops orientation and quantises position to
    whole millimetres; a caller that hands it a tool attitude gets
    silence unless something declares the loss. ``lossy_notes`` is where
    that declaration lives, and the conformance suite asserts it is
    non-empty whenever a driver claims not to carry orientation.
    """

    name: str
    accepts_cartesian: bool = True
    accepts_joint: bool = False
    carries_orientation: bool = False
    carries_redundancy: bool = False
    position_resolution_m: float = 0.0     # 0.0 == continuous
    validate_is_real: bool = False
    gripper_shares_motion_channel: bool = True
    frames: Tuple[str, ...] = ("base",)
    tools: Tuple[str, ...] = ("tool0",)
    lossy_notes: Tuple[str, ...] = ()


# --------------------------------------------------------- gripper ---

class Gripper:
    """End-effector control, as a capability rather than an opcode.

    This project puts the vacuum in band, as ``VACUUM_ON`` /
    ``VACUUM_OFF`` opcodes on the motion channel, and it is the only
    system surveyed that does: UR recommends a separate port (the
    Robotiq UR Cap on 63352) and warns that the in-band alternative
    blocks concurrent motion; ROS 2 uses a separate action on a separate
    controller; Franka opens a second network connection for
    ``franka::Gripper``. It gets away with it here only because vacuum
    is one bit.

    The moment a gripper has width, force, and a *holding* versus
    *closed on nothing* distinction — which is exactly what ROS 2's
    ``stalled`` flag and Franka's ``grasp(..., epsilon_inner,
    epsilon_outer)`` exist to express — it needs its own lifecycle. So
    it gets one here, even though this cell's tooling degenerates to a
    bit. Generalising the other way, from vacuum to everything, is the
    move that does not work.
    """

    def grasp(self, width_m: Optional[float] = None,
              force_n: Optional[float] = None) -> RobotStatus:
        raise NotImplementedError

    def release(self) -> RobotStatus:
        raise NotImplementedError

    @property
    def holding(self) -> Optional[bool]:
        """True / False, or None when the tooling cannot tell."""
        return None


# ---------------------------------------------------------- driver ---

#: Names :class:`RobotDriver` owns. A subclass that rebinds any of them
#: is rejected at class-definition time. This is the whole mechanism by
#: which "every route out of a command either returns an actionable
#: status or attempts a stop" survives a second author.
_SEALED = frozenset({
    "connect", "close", "execute", "halt", "move",
    "__enter__", "__exit__",
    "halted", "estopped", "halt_attempts", "halt_delivered",
    "add_frame_observer", "add_teardown", "teardown_count",
    "_handshake", "_read_status", "_request_halt", "_fatal", "_observe",
    "_run_teardowns",
})


class RobotDriver:
    """Template method. Policy lives here; encoding lives in the subclass.

    A driver supplies six methods, four of which are the seam proper:

    ===================  ===================================================
    ``encode(request)``  the neutral request to bytes
    ``decode(frame)``    bytes to a :class:`common.types.RobotStatus`
    ``send(frame)``      hand bytes to the transport
    ``recv(deadline)``   return exactly one frame's bytes, by ``deadline``
    ``open_channel()``   establish the transport
    ``close_channel()``  tear it down; idempotent
    ===================  ===================================================

    plus two optional ones — :meth:`validate` and :attr:`gripper` —
    whose default answers (``UNKNOWN``, ``None``) are the honest ones
    for a backend that cannot do better.

    Everything else in this class is sealed. See the module docstring
    for why.
    """

    # ---- the taxonomy ---------------------------------------------------
    #
    # Vendor-neutral in shape; the membership is protocol-derived, which
    # is why it is a class attribute a driver may extend rather than a
    # module constant. A driver that has no CRC still has *some*
    # line-fault code, and it belongs in RETRYABLE_STATUS.

    #: Controller-reported codes meaning "try that again".
    RETRYABLE_STATUS: Tuple[RobotStatusCode, ...] = (
        RobotStatusCode.CRC_ERROR, RobotStatusCode.TIMEOUT,
    )
    #: Controller-reported codes meaning "this build cannot talk to that
    #: build". Retrying re-sends the same unparseable frame.
    FATAL_STATUS: Tuple[RobotStatusCode, ...] = (
        RobotStatusCode.VERSION_MISMATCH, RobotStatusCode.UNSUPPORTED_COMMAND,
    )
    #: The closed set :meth:`execute` may return. ``main.py`` branches on
    #: exactly these and raises on anything else; that guarantee is made
    #: here rather than left to the enum's membership happening to be
    #: exhaustive.
    ACTIONABLE_STATUS = frozenset({
        RobotStatusCode.OK, RobotStatusCode.SUCCESS,
        RobotStatusCode.PICK_FAILED, RobotStatusCode.PLACE_FAILED,
    })
    #: Locally-detected failures worth another attempt. ``socket.timeout``
    #: is ``TimeoutError``; every protocol parse error is a
    #: ``ValueError``. The ordering in :meth:`execute` matters: this
    #: tuple is caught FIRST and everything else — ``struct.error``,
    #: ``ConnectionError``, ``ConnectionResetError`` — falls through to
    #: the fatal path. ``struct.error`` is not a ``ValueError`` subclass,
    #: which is how an out-of-range coordinate once escaped without a
    #: stop.
    #:
    #: ``socket.timeout`` is in here for the **read** side only. A write
    #: that times out is a different event and does not reach this
    #: tuple: :meth:`TcpFrameDriver.send` converts it to a
    #: :class:`RobotFault`, because a ``sendall`` that timed out may have
    #: written part of a frame and does not say how much.
    TRANSIENT_EXCEPTIONS: Tuple[type, ...] = (socket.timeout, ValueError)

    #: The request kinds a transient failure may be **retried** on.
    #:
    #: A read timeout does not say the controller failed to act; it says
    #: nothing has come back yet. Neither wire in this repository carries
    #: a sequence number, so a re-send is indistinguishable from the
    #: original at the far end and a late reply is mis-paired with the
    #: retry. Retrying is therefore only safe where running the request
    #: twice reaches the same state as running it once, and the
    #: classification below is by what each kind *does at the
    #: controller*, not by how it is framed:
    #:
    #: * ``HANDSHAKE`` — a query and an ack. ``laptop_comm.src`` CASE 7
    #:   answers ``OK`` and touches nothing. Repeatable.
    #: * ``HALT`` — asks the arm to stop and hold. Stopping twice is
    #:   stopping; a stop that could not be repeated would be a worse
    #:   defect than the one this attribute fixes.
    #: * ``MOVE`` — an **absolute** goal pose, on both drivers. The arm
    #:   converges to the same place whether it is commanded once or
    #:   three times. (It would not be repeatable on a backend whose
    #:   motion command were relative; such a driver must remove ``MOVE``
    #:   from this set, which is why this is a class attribute.)
    #: * ``GRIPPER`` — sets a state, not a delta: vacuum on at 80 %,
    #:   vacuum off, jaws to a width. Applying it twice leaves the same
    #:   state.
    #:
    #: ``VENDOR`` is deliberately **absent**, and that is the whole point
    #: of the attribute. A vendor request is by construction an operation
    #: the base class cannot reason about, and on this project it is
    #: ``PICK_AND_PLACE``: seven controller-side steps at 150 mm/s
    #: against a 5000 ms default deadline, which descends, applies
    #: vacuum, transports and inserts. A second run grasps nothing and
    #: inserts a cell the first run already placed. The base class must
    #: not repeat what it cannot describe.
    IDEMPOTENT_KINDS: frozenset = frozenset({
        RequestKind.HANDSHAKE, RequestKind.HALT,
        RequestKind.MOVE, RequestKind.GRIPPER,
    })

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        clashes = sorted(_SEALED.intersection(cls.__dict__))
        if clashes:
            raise TypeError(
                f"{cls.__name__} overrides sealed RobotDriver member(s) "
                f"{clashes}. These carry the stop escalation: every route "
                f"out of a command must either return an actionable status "
                f"or attempt a halt, and a driver that reimplements them "
                f"re-opens the bypass class they were written to close. "
                f"Supply encode / decode / send / recv instead.")

    def __init__(self, policy: DriverPolicy,
                 logger_name: str = "execution.driver") -> None:
        self.policy = policy
        self.log = get_logger(logger_name)
        self._halted = False
        self._halt_attempts = 0
        self._halt_delivered = False
        self._observers: list = []
        # Callables the *builder* of this driver attached to its
        # lifetime; see add_teardown. Empty for every driver constructed
        # directly, which is every driver in the conformance suite.
        self._teardowns: list = []
        self._teardown_count = 0

    # ---- hooks a driver must supply -------------------------------------

    def open_channel(self) -> None:
        """Establish the transport. Raise :class:`RobotFault` on failure.

        Must leave nothing open on the failure path — a refused
        connection that left a descriptor installed is how the canonical
        ``with`` form used to leak one.
        """
        raise NotImplementedError

    def close_channel(self) -> None:
        """Tear the transport down. Must be idempotent and must not raise."""
        raise NotImplementedError

    def encode(self, request: Request) -> bytes:
        """Serialise one neutral request into one frame."""
        raise NotImplementedError

    def decode(self, frame: bytes) -> RobotStatus:
        """Parse one frame into a status.

        Raise a ``ValueError`` (which includes every protocol parse
        error) for anything unparseable — a bad checksum, a short frame,
        a status code this build does not know. Those are *transient* and
        will be retried and then escalated. Do not substitute a plausible
        status: an unknown code from a future firmware must not be
        indistinguishable from an ordinary placement failure.
        """
        raise NotImplementedError

    def send(self, frame: bytes) -> None:
        """Hand one frame to the transport."""
        raise NotImplementedError

    def recv(self, deadline: float) -> bytes:
        """Return exactly one frame's bytes, or raise by ``deadline``.

        ``deadline`` is a ``time.monotonic()`` value and is a deadline
        for the **whole frame**, not for one read. A controller that
        trickles one byte at a time must not be accepted after three
        seconds at a 250 ms setting; that is the failure this signature
        exists to make impossible to write by accident.
        """
        raise NotImplementedError

    # ---- hooks with an honest default -----------------------------------

    @property
    def capabilities(self) -> Capabilities:
        """What this backend can do. Override; the default claims nothing."""
        return Capabilities(name=type(self).__name__)

    @property
    def channel_open(self) -> bool:
        """Is the transport currently established?"""
        raise NotImplementedError

    @property
    def gripper(self) -> Optional[Gripper]:
        """The end-effector, or ``None`` if this backend has none."""
        return None

    def validate(self, target: Pose) -> Reachability:
        """Can the arm reach ``target``? ``UNKNOWN`` is a real answer.

        The default is ``UNKNOWN`` because that is what a client with no
        kinematic model honestly knows. Do not return ``REACHABLE`` to
        be helpful: a caller that trusts a guess is worse off than one
        that knows it has no answer.
        """
        return Reachability.UNKNOWN

    # ---- sealed: observation --------------------------------------------

    def add_frame_observer(
        self, fn: Callable[[Request, bytes], None],
    ) -> None:
        """Register ``fn(request, frame)``, called just before every send.

        A supported hook, because the alternative was a test
        monkeypatching the private ``_send`` and parsing the bytes it
        caught. That test then pinned a private method and a wire format
        at once: a driver that satisfied every public contract broke it.
        Observers see the neutral request *and* the encoded frame, so a
        caller can assert on either without reaching inside.
        """
        self._observers.append(fn)

    def _observe(self, request: Request, frame: bytes) -> None:
        for fn in self._observers:
            fn(request, frame)

    # ---- sealed: builder-attached teardown -------------------------------

    def add_teardown(self, fn: Callable[[], None]) -> None:
        """Register ``fn()`` to run once, when this driver is closed.

        For whoever BUILT the driver, not for the driver itself. The one
        caller is :func:`execution.build_driver`, which spawns an
        in-process simulator for the ``mock`` and ``json`` backends and
        has to stop that thread again.

        It exists because the alternative had already been written and
        was worse: ``main.py`` held the server handle beside the driver
        and shut it down in its own ``finally``, so every future caller
        of the factory inherited responsibility for a thread it never
        asked for, and one that forgot leaked it. ``close()`` is the one
        event that happens on *every* route out of a run — normal exit,
        a refused handshake, ``RobotEstop`` and ``_request_halt`` all
        reach it — which is exactly the property the escalation was
        built around.

        Teardowns run in reverse registration order, exactly once, and a
        raising teardown does not stop the others or reach the caller:
        ``close()`` is on the fault paths, and a simulator that would
        not stop must not replace the fault that prompted the close. It
        is logged instead.
        """
        self._teardowns.append(fn)
        self._teardown_count += 1

    @property
    def teardown_count(self) -> int:
        """How many teardowns were registered on this driver, ever.

        Does not decrease when they run: it answers "was this driver
        built with a simulator attached?", which is a fact about the
        driver and not about whether it is still open.
        """
        return self._teardown_count

    def _run_teardowns(self) -> None:
        while self._teardowns:
            fn = self._teardowns.pop()
            try:
                fn()
            except Exception as exc:       # pragma: no cover - defensive
                self.log.error(
                    "a teardown registered on this driver raised (%r); the "
                    "resource it owns may still be running", exc)

    # ---- sealed: lifecycle ----------------------------------------------

    @property
    def halted(self) -> bool:
        """True once this driver has requested a stop, or been told of one.

        One-way. A latched driver refuses to command motion and refuses
        to reconnect: clearing a stop is a deliberate act at the
        controller, not something a host reconnect may do.
        """
        return self._halted

    @property
    def estopped(self) -> bool:
        """Deprecated alias for :attr:`halted`, kept for existing callers.

        The name is retained rather than removed because it is what the
        latch has always been called here, but it overstates what the
        client did: this driver requests an in-band halt, it does not
        actuate a safety-rated stop.
        """
        return self._halted

    @property
    def halt_attempts(self) -> int:
        """How many times a stop has been *attempted* on this driver.

        The observation the conformance suite is written against. It
        counts attempts, not deliveries, because "the link died and the
        stop could not go out" is a distinct and much worse outcome that
        must still be visible.
        """
        return self._halt_attempts

    @property
    def halt_delivered(self) -> bool:
        """True once a halt frame has left the transport without raising."""
        return self._halt_delivered

    def connect(self) -> None:
        """Open the channel and complete the handshake.

        Every failure route closes the channel before raising, so the
        canonical ``with Driver(cfg) as d:`` form cannot leak it —
        ``__exit__`` never runs when ``__enter__`` raises.
        """
        if self._halted:
            raise RobotEstop(
                "this driver is latched and will not reconnect; clear the "
                "stop at the controller and build a new driver")

        self.open_channel()
        try:
            self._handshake()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the transport, then run any builder-attached teardowns.

        The transport goes first: a teardown stops the thing at the
        other end of it, and stopping a controller while a socket to it
        is still open is how a test starts reading a reset instead of a
        clean shutdown. Both halves are idempotent, which matters
        because ``close`` is reached repeatedly on the fault paths
        (``halt``'s ``finally``, ``connect``'s handshake failure,
        ``__exit__``).
        """
        try:
            self.close_channel()
        finally:
            self._run_teardowns()

    def __enter__(self) -> "RobotDriver":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- sealed: the handshake ------------------------------------------

    def _handshake(self) -> None:
        """Send a handshake and read the ack, under the same retry policy
        as every other command.

        ``handshake_timeout_ms`` bounds the wait for the ack, which is
        what its name says and what it did not do: the read used to
        re-arm ``command_timeout_ms`` before looking at a byte, so the
        knob bounded only the transport connect.

        Only ``OK`` is accepted. It used to accept ``OK`` *or*
        ``SUCCESS`` while no implementation on either side of the wire
        ever emitted ``SUCCESS`` here — a tolerance wider than anything
        that exists, which a conformance suite cannot pin and a second
        controller would read as permission.

        This loop needs no idempotency gate of the kind
        :meth:`execute` grew: it sends exactly one kind of request, and
        ``HANDSHAKE`` is in :attr:`IDEMPOTENT_KINDS` because it asks the
        controller for an ack and changes nothing (``laptop_comm.src``
        CASE 7).
        """
        timeout_s = self.policy.handshake_timeout_ms / 1000
        last_err: Optional[Exception] = None
        request = Request(RequestKind.HANDSHAKE, "HANDSHAKE")

        for attempt in range(self.policy.max_retries):
            try:
                frame = self.encode(request)
                self._observe(request, frame)
                self.send(frame)
                ack = self._read_status(timeout_s)
            except self.TRANSIENT_EXCEPTIONS as exc:
                last_err = exc
                self.log.warning(
                    "handshake error (attempt %d/%d): %s",
                    attempt + 1, self.policy.max_retries, exc,
                )
                time.sleep(self.policy.retry_pause_ms / 1000)
                continue
            except Exception as exc:
                self._fatal(f"handshake failed: {exc!r}", exc)

            if ack.code is RobotStatusCode.OK:
                self.log.info(
                    "handshake OK; at %.1f,%.1f,%.1f",
                    ack.current_pose.x_mm, ack.current_pose.y_mm,
                    ack.current_pose.z_mm,
                )
                return

            if ack.code is RobotStatusCode.ESTOP:
                # The commonest refusal, and the one that used to raise a
                # bare string with the channel left open: the controller
                # is telling us it is ALREADY in a Category-0 stop.
                self._halted = True
                raise RobotEstop(
                    "the controller refused the handshake because it is in "
                    "a Category-0 stop; clear it at the controller before "
                    "reconnecting")

            # Any other refusal: we do not have a channel we trust, and we
            # do not know what state the arm is in. The transport is
            # alive, so the halt can actually be transmitted.
            self._fatal(f"handshake refused (status={ack.code.name})")

        self._fatal(
            f"handshake did not complete in {self.policy.max_retries} "
            f"attempts: {last_err}", last_err)

    # ---- sealed: the command loop ---------------------------------------

    def execute(self, request: Request) -> RobotStatus:
        """Send one request and wait for the status that answers it.

        Retries transient failures up to ``policy.max_retries``, then
        requests a halt and raises. Fatal failures skip the retries.
        Returns only codes in :attr:`ACTIONABLE_STATUS`.

        **What "retry" is now bounded by.** Neither wire here carries a
        sequence number, so a reply that arrives after its deadline is
        mis-paired with whatever went out next. This loop used to answer
        that by re-sending everything, which made the mis-pairing worse
        for exactly the request it hurt most: a ``PICK_AND_PLACE`` that
        timed out was re-commanded, so a cell that had already been
        picked and inserted was picked and inserted again, and every
        reply after it answered the previous request.

        A transient failure is now retried only for a kind in
        :attr:`IDEMPOTENT_KINDS`; for anything else the first transient
        failure goes straight to :meth:`_fatal`, which requests a halt
        and raises. That is strictly less capable and strictly more
        honest: the driver still cannot tell whether the cycle ran, but
        it no longer resolves that ignorance by running it again.

        The mis-pairing hazard itself is **not** closed — it cannot be,
        without a sequence number on the wire. What is closed is this
        client's contribution to it.
        """
        if self._halted:
            raise RobotEstop(
                f"refusing to send {request.name}: this driver is latched")

        last_err: Optional[Exception] = None
        repeatable = request.kind in self.IDEMPOTENT_KINDS
        for attempt in range(self.policy.max_retries):
            try:
                frame = self.encode(request)
                self._observe(request, frame)
                self.send(frame)
                status = self._read_status(
                    self.policy.command_timeout_ms / 1000)
            except self.TRANSIENT_EXCEPTIONS as exc:
                if not repeatable:
                    # The failure is transient; the request is not. None
                    # of the three things caught here licenses a second
                    # grasp-and-insert: a read timeout says the reply has
                    # not arrived, which is not the same as the command
                    # not having run; an unparseable reply says it ran
                    # and the answer was corrupted; and a ValueError out
                    # of encode() will raise identically on the second
                    # attempt. Stop the arm and say so instead.
                    self._fatal(
                        f"{request.name} failed transiently ({exc!r}) and is "
                        f"a {request.kind.name} request, which this driver "
                        f"does not classify as safe to repeat; re-sending it "
                        f"could run the operation a second time",
                        exc)
                last_err = exc
                self.log.warning(
                    "channel error (attempt %d/%d): %s",
                    attempt + 1, self.policy.max_retries, exc,
                )
                time.sleep(self.policy.retry_pause_ms / 1000)
                continue
            except Exception as exc:
                # An encoder that could not represent the request, a
                # transport that reset, and anything else unforeseen.
                # Retrying cannot help; the transport may still be alive,
                # in which case the halt DOES go out.
                self._fatal(f"{request.name} failed: {exc!r}", exc)

            if status.code is RobotStatusCode.ESTOP:
                # The controller is stopped. So is this driver, now.
                self._halted = True
                self.close()
                raise RobotEstop(
                    f"the controller reported a Category-0 stop in reply to "
                    f"{request.name}; no further motion will be commanded")

            if status.code in self.FATAL_STATUS:
                self._fatal(
                    f"the controller rejected {request.name} as "
                    f"{status.code.name}")

            if status.code in self.RETRYABLE_STATUS:
                # Re-sent even for a kind outside IDEMPOTENT_KINDS, and
                # the asymmetry with the transient-exception path above
                # is deliberate. A controller-reported status is a
                # complete, well-formed reply *to this request*: the
                # pairing is intact, so a re-send cannot skew it, and a
                # line fault reported back means the far end could not
                # read the frame. On this project's controller that is
                # provable — krl_prog/laptop_comm.src compares the CRC
                # before it enters the opcode SWITCH, so a CRC_ERROR
                # reply is evidence that nothing was dispatched.
                #
                # Residual, stated rather than papered over: the other
                # member of the default tuple, a controller-reported
                # TIMEOUT, carries no such evidence. Nothing on either
                # side of this wire emits it — the KRL loop has no branch
                # that can, and neither simulator sends it — so the
                # ambiguity is unreachable here rather than resolved. A
                # driver whose controller does emit it, and whose
                # non-idempotent operations can be half-done when it
                # does, must drop it from RETRYABLE_STATUS.
                last_err = RobotFault(
                    f"controller reported {status.code.name}")
                self.log.warning(
                    "channel error (attempt %d/%d): %s",
                    attempt + 1, self.policy.max_retries, last_err,
                )
                time.sleep(self.policy.retry_pause_ms / 1000)
                continue

            if status.code not in self.ACTIONABLE_STATUS:
                # Unreachable while the taxonomies above stay exhaustive
                # over RobotStatusCode — and that is the point. A code
                # added to the enum without being classified escalates
                # here instead of reaching a caller that branches on four
                # values and treats the fifth as a failed placement.
                self._fatal(
                    f"{request.name} returned {status.code.name}, which this "
                    f"driver classifies as neither actionable, retryable "
                    f"nor fatal")

            return status

        self.log.error(
            "giving up after %d retries", self.policy.max_retries)
        self._fatal(f"{request.name}: {last_err}", last_err)

    def move(self, target: Pose,
             motion: MotionKind = MotionKind.UNSPECIFIED) -> RobotStatus:
        """Move to ``target``. The one motion primitive on this interface."""
        return self.execute(
            Request(RequestKind.MOVE, "MOVE", target=target, motion=motion))

    def _read_status(self, timeout_s: float) -> RobotStatus:
        """One frame in, one status out, inside a whole-frame deadline."""
        deadline = time.monotonic() + timeout_s
        frame = self.recv(deadline)
        return self.decode(frame)

    # ---- sealed: escalation ---------------------------------------------

    def halt(self) -> None:
        """Request an in-band stop, then disconnect. Latches.

        **Not a safety-rated stop.** This asks the controller to stop
        moving over the same application channel it takes motion on; a
        real emergency stop is a hardware chain this host cannot reach,
        and clearing one is out of band. What this method does guarantee
        is the client-side half: after it, the driver refuses to command
        motion and refuses to reconnect.

        **Actionable only between commands.** This is a correction to
        what the module docstring used to claim. "Decelerate and hold, in
        band" describes what the *opcode* means, not what this backend
        can deliver, and on this backend it cannot be delivered mid-cycle
        for two independent reasons — either alone is sufficient:

        * *Host side.* There is one socket and one thread. The only
          caller inside this class is :meth:`_request_halt`, reached
          after :meth:`execute` has already returned or raised, and the
          thread that would call it is the thread blocked in ``recv``.
          A halt cannot be *sent* while a command is in flight.
        * *Controller side.* ``krl_prog/laptop_comm.src`` reads one
          record, dispatches it, replies, and only then calls
          ``EKI_CheckBuffer`` for the next one. CASE 4 dispatches to
          ``PickAndPlace``, which does not return until the whole cycle
          has run. A halt frame arriving mid-cycle sits in the EKI FIFO
          until it ends. A halt cannot be *serviced* while a command is
          in flight either.

        So during the one operation where an abort would matter most —
        the seven-step pick-and-place — this method's effect is confined
        to the client-side latch: no *further* motion is commanded, and
        the vacuum is dropped by the controller's own CASE 6 when the
        frame is finally read.

        A real in-cycle abort is not a change to this method. It needs a
        second, asynchronous channel into the controller and a KRL
        ``INTERRUPT`` declared against it — the shape
        ``krl_prog/routines.src`` already uses for ``$STOPMESS``
        (``GLOBAL INTERRUPT DECL 3 WHEN $STOPMESS==TRUE DO IR_STOPM()``)
        — so that the interpreter services it without the receive loop
        having to come back round. Neither exists here, and nothing in
        this class simulates one: an abort path that returned promptly
        while the arm kept moving would be worse than the honest
        limitation, because a caller would stop watching.

        Idempotent in effect: the latch is one-way, so a second call
        cannot un-set it, and a call with no channel left to send over
        does nothing rather than raising — by then the request has
        either already gone out or become impossible, and neither is a
        new failure for the caller to handle. It is logged CRITICAL
        instead, because "we intended to stop the robot and could not
        reach it" must never be inaudible.

        Best effort otherwise: it *does* raise if there is a channel and
        the frame could not be transmitted over it. Callers inside this
        class go through :meth:`_request_halt`, which swallows that and
        says so loudly.
        """
        self._halted = True
        if not self.channel_open:
            self.log.critical(
                "halt requested with no open channel: the latch is set, but "
                "nothing was transmitted and this driver cannot know "
                "whether the arm is moving")
            return

        self._halt_attempts += 1
        request = Request(RequestKind.HALT, "HALT")
        try:
            frame = self.encode(request)
            self._observe(request, frame)
            self.send(frame)
            self._halt_delivered = True
        finally:
            self.close()

    def _request_halt(self, reason: str) -> None:
        """Best-effort halt. Never raises.

        A stop that cannot be transmitted must not replace the error
        that prompted it — that swap is how the original cause used to
        get lost — but it must not be silent either, because "the link
        died mid-cycle" means the arm may be holding a part with the
        gripper closed and nothing can tell it to let go.
        """
        self.log.critical("HALT REQUESTED: %s", reason)
        self._halted = True
        try:
            self.halt()
        except Exception as exc:
            self.log.critical(
                "the halt request could NOT be transmitted (%s). The link "
                "is down and the arm's state is UNKNOWN — it may be "
                "mid-motion holding a part.", exc,
            )
            self.close()

    def _fatal(self, reason: str,
               cause: Optional[BaseException] = None) -> None:
        """Stop the robot and raise. Never returns."""
        self._request_halt(reason)
        raise RobotFault(f"{type(self).__name__}: {reason}") from cause


# ------------------------------------------------- TCP frame driver ---

class TcpFrameDriver(RobotDriver):
    """A :class:`RobotDriver` whose transport is a blocking TCP socket.

    Supplies ``open_channel`` / ``close_channel`` / ``send`` and a
    ``read_exactly`` helper that enforces the whole-frame deadline. It
    deliberately does **not** supply ``recv``: framing is the driver's
    business, and the two drivers in this repository frame differently
    on purpose — one reads a fixed 16 bytes, the other reads a 4-byte
    length prefix and then that many bytes. A base that owned the read
    loop would have quietly made fixed-length framing part of the
    interface.

    Not every backend dials out. ABB's Externally Guided Motion has the
    *controller* initiate the UDP connection to the client, so a
    listen-mode driver would need a sibling of this class rather than a
    subclass of it. Nothing here is built for that, and nothing here
    pretends to be.
    """

    def __init__(self, host: str, port: int, policy: DriverPolicy,
                 logger_name: str = "execution.driver") -> None:
        super().__init__(policy, logger_name=logger_name)
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None

    @property
    def channel_open(self) -> bool:
        return self._sock is not None

    def open_channel(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.policy.handshake_timeout_ms / 1000)
        try:
            sock.connect((self.host, self.port))
        except OSError as exc:
            sock.close()
            # Nothing has been commanded, so there is nothing to stop —
            # and no channel to stop it over.
            raise RobotFault(
                f"could not open a channel to {self.host}:{self.port}: "
                f"{exc}") from exc
        self._sock = sock

    def close_channel(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def send(self, frame: bytes) -> None:
        """Write one whole frame, under the command timeout, or fail hard.

        **Two things happen here that used to not.**

        *The deadline is re-armed.* :meth:`read_exactly` calls
        ``settimeout(remaining)`` before every chunk and does not restore
        it, so after a reply that arrived near its deadline the socket
        was left carrying whatever was left over — measured at ~0.15 s of
        a 0.4 s command timeout in the conformance suite, and arbitrarily
        close to zero on a reply that only just made it. The next
        ``sendall`` then ran under the *previous read's* remainder, a
        number that has nothing to do with how long a write may take.

        *A write timeout is fatal, not transient.* ``sendall`` raises
        ``socket.timeout`` without reporting how many bytes it managed to
        write, so a timed-out write may have left a partial frame on the
        stream. This protocol cannot resynchronise — no length prefix, no
        sync word, readers take a fixed count — so a retry writing a
        second frame behind a partial one corrupts every frame after it,
        which is the CRC-error-forever behaviour recorded in audit F
        §1.6. Raising :class:`RobotFault` puts it past
        ``TRANSIENT_EXCEPTIONS`` and onto the catch-all, which halts.
        """
        if self._sock is None:
            raise RobotFault("socket not connected")
        self._sock.settimeout(self.policy.command_timeout_ms / 1000)
        try:
            self._sock.sendall(frame)
        except socket.timeout as exc:
            raise RobotFault(
                f"the write of a {len(frame)}-byte frame timed out after "
                f"{self.policy.command_timeout_ms} ms. An unknown prefix of "
                f"it may be on the wire and this stream cannot "
                f"resynchronise, so nothing further may be written over it."
            ) from exc

    def read_exactly(self, n: int, deadline: float) -> bytes:
        """Read exactly ``n`` bytes, or raise before ``deadline``.

        The deadline covers the whole read, not each ``recv``. A
        controller that trickles one byte at a time used to be accepted
        after 3 s at a 250 ms setting, which left the command deadline —
        the thing the bounded-response argument rests on — unenforced.

        This method leaves the transport armed with the remainder of the
        deadline it was given. That is correct for the read it is in the
        middle of and wrong for anything afterwards, which is why
        :meth:`send` arms its own.
        """
        if self._sock is None:
            raise RobotFault("socket not connected")

        buf = b""
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout(
                    f"frame incomplete at the deadline ({len(buf)}/{n} bytes)")
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(
                    f"robot closed connection ({len(buf)}/{n} bytes of a "
                    "frame received)")
            buf += chunk
        return buf


__all__ = [
    "Capabilities",
    "DriverPolicy",
    "Gripper",
    "MotionKind",
    "Pose",
    "Reachability",
    "Request",
    "RequestKind",
    "RobotDriver",
    "RobotEstop",
    "RobotFault",
    "TcpFrameDriver",
]
