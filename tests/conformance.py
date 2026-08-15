"""The robot-driver conformance suite.

One set of tests that **any** driver must pass. Not a description of a
driver, an executable requirement on one.

This exists because the interface is the weaker artefact. An interface
is a suggestion: a second driver's author can satisfy every method
signature and still lose a safety behaviour, and the tests will get
quieter rather than louder when they do. This project has already had a
stop with three separate bypass routes — a ``struct.error`` from a
coordinate the frame could not carry, a mid-frame close, and a
``ConnectionResetError`` — and each one was a route out of a command
that returned neither an actionable status nor an attempted stop. The
suite below is the fourth route's tripwire.

**The property it pins, in one sentence:** *every route out of a command
either returns a status the caller can act on, or attempts a halt and
raises.*

How it stays vendor-neutral
---------------------------
The old versions of these tests asserted by inspecting ``OpCode.ESTOP``
on the wire. That is an assertion about sixteen bytes, and it cannot be
run against any other encoding. They now assert against two neutral
observations instead:

* :attr:`execution.driver.RobotDriver.halt_attempts` — the driver's own
  count of stops it tried to send, which every driver inherits and none
  implements;
* ``endpoint.wait_for_halt()`` — whether the *far end* saw a halt,
  decoded by a harness the driver's author supplies.

Both matter. A driver that latches internally but never puts a halt on a
live wire has done half the job, and only the second observation catches
it; a driver whose link is already dead cannot put anything on the wire
at all, and only the first catches that it tried.

What a driver author supplies
-----------------------------
One :class:`ConformanceHarness` — roughly a hundred lines — and a
three-line subclass of :class:`RobotDriverConformance`. The harness
knows the driver's framing and its status encoding; nothing else in this
file does. See ``tests/test_kuka_conformance.py`` and
``tests/test_json_conformance.py``, which are that, twice.

Scope, stated plainly
---------------------
This suite assumes a **delegated** control shape: the caller hands the
controller a goal and receives a terminal outcome, and a slow client is
harmless. It is not applicable to a streaming backend — one where the
client interpolates at a fixed rate, silence means *fault* rather than
*idle*, and there is no completion event at all (Franka FCI, ABB EGM).
Such a backend needs a different interface and a different suite, and
this project needs neither.

It also assumes a socket-like transport with a scriptable far end. That
is a property of the two mocks here, not of robots in general; a driver
that dials in over a vendor SDK would have to fake the endpoint
differently.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import List

import pytest

from common.types import RobotStatusCode
from execution.driver import (
    Pose,
    Reachability,
    Request,
    RequestKind,
    RobotDriver,
    RobotEstop,
    RobotFault,
)

#: Reply sentinel: drop the connection with an RST rather than a FIN.
RESET = "__reset__"


@dataclass(frozen=True)
class Hangup:
    """Reply sentinel: send ``payload`` (possibly partial), then close."""

    payload: bytes


@dataclass(frozen=True)
class Trickle:
    """Reply sentinel: send ``payload[:split]``, wait, send the rest.

    A whole reply, delivered in two TCP segments. It exists because the
    read path's per-chunk deadline is only observable when there is more
    than one chunk: the driver arms what is *left* of the command
    deadline before each ``recv``, so the value the transport is left
    carrying after a two-segment reply is the remainder, not the whole.
    A one-segment reply leaves the full timeout armed and hides the
    defect on any fixed-length framing.
    """

    payload: bytes
    split: int
    gap_s: float


@dataclass(frozen=True)
class Observed:
    """One request as the far end saw it, in neutral terms."""

    kind: RequestKind
    name: str
    raw: bytes


# ------------------------------------------------- scripted endpoint ---

class ScriptedEndpoint:
    """A TCP server that records requests and replies however you say.

    Generic over framing: the harness supplies ``read_frame`` and
    ``describe``, so the same adversarial machinery drives a fixed
    16-byte protocol and a length-prefixed JSON one without change.

    ``reply(observed, n)`` is called with the decoded request and its
    0-based index and returns the bytes to send back, or ``None`` for
    deliberate silence, or :data:`RESET`, or a :class:`Hangup`.
    """

    def __init__(self, read_frame, describe, reply) -> None:
        self._read_frame = read_frame
        self._describe = describe
        self._reply = reply
        self.requests: List[Observed] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn) -> None:
        n = 0
        with conn:
            conn.settimeout(5.0)
            while True:
                try:
                    frame = self._read_frame(conn)
                except (OSError, ValueError):
                    return
                if not frame:
                    return
                try:
                    observed = self._describe(frame)
                except Exception:
                    return
                self.requests.append(observed)
                out = self._reply(observed, n)
                n += 1
                if out is None:
                    continue
                if out == RESET:
                    # SO_LINGER{on, 0} makes close() send an RST.
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                    struct.pack("ii", 1, 0))
                    return
                if isinstance(out, Trickle):
                    try:
                        conn.sendall(out.payload[:out.split])
                        time.sleep(out.gap_s)
                        conn.sendall(out.payload[out.split:])
                    except OSError:
                        return
                    continue
                hangup = isinstance(out, Hangup)
                payload = out.payload if hangup else out
                try:
                    conn.sendall(payload)
                except OSError:
                    return
                if hangup:
                    return

    # ---- neutral observations -------------------------------------------

    def kinds(self) -> List[RequestKind]:
        return [r.kind for r in self.requests]

    def count(self, kind: RequestKind) -> int:
        return sum(1 for r in self.requests if r.kind is kind)

    def saw_halt(self) -> bool:
        return self.count(RequestKind.HALT) > 0

    def wait_for_halt(self, timeout: float = 3.0) -> bool:
        """Block until a halt has been recorded, or ``timeout`` elapses.

        The driver sends its halt and closes the transport in a
        ``finally``, so ``__exit__`` can return before this server's
        reader thread has appended the request. Asserting the instant the
        driver returns is therefore a race — measured on the KUKA driver,
        roughly one run in fifteen. Waiting weakens nothing: the halt
        still has to arrive, it is just given a bounded chance to.
        """
        deadline = time.monotonic() + timeout
        while not self.saw_halt() and time.monotonic() < deadline:
            time.sleep(0.005)
        return self.saw_halt()

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


class CapturedCritical(logging.Handler):
    """``common.logging`` sets ``propagate = False``, so pytest's caplog
    never sees these records; attach to the driver's own logger."""

    def __init__(self, logger_name: str) -> None:
        super().__init__(level=logging.CRITICAL)
        self.logger_name = logger_name
        self.messages: List[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "CapturedCritical":
        logging.getLogger(self.logger_name).addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logging.getLogger(self.logger_name).removeHandler(self)

    def text(self) -> str:
        return "\n".join(self.messages)


# ------------------------------------------------------- the harness ---

class ConformanceHarness:
    """What a driver author supplies so the suite can run.

    Every method here is about *this driver's* encoding. Nothing in the
    suite below knows any of it.
    """

    #: Logger the driver writes its CRITICAL escalations to.
    logger_name: str = "execution.driver"

    def make_driver(self, port: int, **policy) -> RobotDriver:
        """Build an unconnected driver aimed at ``127.0.0.1:port``.

        ``policy`` may carry ``max_retries``, ``handshake_timeout_ms``,
        ``command_timeout_ms``, ``retry_pause_ms``; the harness maps them
        onto whatever config type the driver takes. Timeouts should
        default short — the suite is full of deliberate silences.
        """
        raise NotImplementedError

    def read_frame(self, conn) -> bytes:
        """Read exactly one whole request frame from ``conn``, or ``b""``."""
        raise NotImplementedError

    def describe(self, frame: bytes) -> Observed:
        """Decode one outbound frame into a neutral :class:`Observed`."""
        raise NotImplementedError

    # ---- reply builders --------------------------------------------------

    def status_frame(self, code: RobotStatusCode) -> bytes:
        """A well-formed status reply carrying ``code``."""
        raise NotImplementedError

    def unknown_status_frame(self) -> bytes:
        """A well-formed frame whose status code this build cannot name."""
        raise NotImplementedError

    def corrupt(self, frame: bytes) -> bytes:
        """``frame`` damaged so that ``decode`` raises a ``ValueError``.

        A line fault, in whatever form this encoding has one: a broken
        checksum, an unparseable body. It must be *transient* — the
        driver should retry it and then escalate.
        """
        raise NotImplementedError

    def truncate(self, frame: bytes) -> bytes:
        """A prefix of ``frame`` that cannot complete. Sent, then hung up."""
        raise NotImplementedError

    def break_channel(self, driver: RobotDriver) -> None:
        """Kill the transport under ``driver`` without telling it.

        A dead descriptor still installed: the driver believes it is
        connected and its next write fails. This is how the suite
        reaches the "the halt itself could not be transmitted" path,
        which is the one a second driver most often gets wrong.
        """
        raise NotImplementedError

    # ---- transport -------------------------------------------------------

    def transport_deadline_s(self, driver: RobotDriver):
        """The write/read deadline currently armed on the transport.

        Returns seconds, or ``None`` for a blocking transport with no
        deadline. This is the one observation in the suite that is about
        the transport rather than the protocol, and it is here because
        the defect it pins is not visible any other way: a reply that
        arrives near the command deadline leaves the socket armed with
        whatever was left of it, and the *next* write inherits that. A
        16-byte frame never blocks long enough for a behavioural test to
        see the difference, so the suite asks what deadline is armed
        instead of timing a write that will not stall.
        """
        raise NotImplementedError

    # ---- targets ---------------------------------------------------------

    def reachable_target(self) -> Pose:
        """A pose this driver can encode and the mock will accept."""
        raise NotImplementedError

    def non_repeatable_request(self, driver: RobotDriver) -> Request:
        """A request this backend must not send twice.

        Every backend surveyed has one: an operation whose effect is not
        a function of the arm's state but of how many times it ran. On
        this project's KUKA wire it is ``PICK_AND_PLACE`` — a
        grasp-and-insert that on a second run picks nothing and inserts
        the cell the first run already placed. The driver must never
        re-send it on a read timeout, because a timeout does not say the
        cycle did not happen; it says nothing came back yet.
        """
        raise NotImplementedError

    def unencodable_target(self) -> Pose:
        """A pose ``encode`` must refuse with a **non-**``ValueError``.

        This is the shape of the defect that opened bypass route one: an
        out-of-range coordinate raised ``struct.error``, which is not a
        ``ValueError``, so it fell past the transient handler and out of
        the command loop entirely — no retry, no stop, and a perfectly
        healthy socket over which the stop could have been sent. Every
        encoding has such a value. Name it here.
        """
        raise NotImplementedError

    # ---- a well-behaved controller ---------------------------------------

    def start_mock(self):
        """Start this driver's normal simulator. Returns ``(port, close)``."""
        raise NotImplementedError

    def smoke_requests(self, driver: RobotDriver):
        """A short, representative sequence of requests for that mock."""
        raise NotImplementedError


# --------------------------------------------------------- the suite ---

class RobotDriverConformance:
    """Subclass this and override the ``harness`` fixture. That is all."""

    @pytest.fixture
    def harness(self) -> ConformanceHarness:
        raise NotImplementedError(
            "a conformance subclass must override the `harness` fixture")

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _endpoint(harness: ConformanceHarness, reply) -> ScriptedEndpoint:
        return ScriptedEndpoint(harness.read_frame, harness.describe, reply)

    @staticmethod
    def _after_handshake(harness: ConformanceHarness, then):
        """Answer the handshake OK, then do ``then`` for everything else."""
        ok = harness.status_frame(RobotStatusCode.OK)

        def reply(observed: Observed, n: int):
            if observed.kind is RequestKind.HANDSHAKE:
                return ok
            return then(observed, n) if callable(then) else then

        return reply

    @staticmethod
    def _move(driver: RobotDriver, target: Pose):
        return driver.execute(
            Request(RequestKind.MOVE, "MOVE", target=target))

    # ---- 1. lifecycle ----------------------------------------------------

    def test_a_context_manager_connects_and_closes(self, harness):
        """The canonical form must work and must not leave the link open."""
        port, close = harness.start_mock()
        try:
            driver = harness.make_driver(port)
            with driver:
                assert driver.channel_open
            assert not driver.channel_open
            assert not driver.halted
        finally:
            close()

    def test_connecting_to_a_closed_port_raises_the_drivers_own_fault(
            self, harness):
        """Nothing has been commanded and there is no channel to stop
        over — but the failure must still arrive as :class:`RobotFault`
        rather than a raw socket error, because that is what the caller
        catches."""
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with pytest.raises(RobotFault):
            harness.make_driver(port).connect()

    def test_a_handshake_refused_because_the_controller_is_stopped(
            self, harness):
        """The commonest refusal: the controller is ALREADY stopped.

        It must latch, raise :class:`RobotEstop`, and close the channel —
        including through the ``with`` form, where ``__exit__`` never
        runs because ``__enter__`` raised. That is how the descriptor
        used to leak on every refusal.
        """
        ep = self._endpoint(
            harness,
            lambda o, n: harness.status_frame(RobotStatusCode.ESTOP))
        try:
            driver = harness.make_driver(ep.port)
            with pytest.raises(RobotEstop):
                driver.connect()
            assert not driver.channel_open, "the descriptor must not leak"
            assert driver.halted

            second = harness.make_driver(ep.port)
            with pytest.raises(RobotEstop):
                with second:
                    pass
            assert not second.channel_open
        finally:
            ep.close()

    def test_a_handshake_refused_with_any_other_status_escalates(
            self, harness):
        """Only one code may complete a handshake.

        The KUKA client used to accept ``OK`` *or* ``SUCCESS`` while
        nothing on either side of the wire emitted ``SUCCESS`` for a
        handshake. A tolerance wider than every implementation is not
        harmless: it is a promise a second controller author will read as
        permission, and no test can pin an accepted set that nothing
        emits.
        """
        ep = self._endpoint(
            harness,
            lambda o, n: harness.status_frame(RobotStatusCode.SUCCESS))
        try:
            driver = harness.make_driver(ep.port)
            with pytest.raises(RobotFault):
                driver.connect()
            assert driver.halt_attempts == 1
            assert ep.wait_for_halt(), (
                "the transport was alive; the halt had to go out")
        finally:
            ep.close()

    def test_a_handshake_that_is_never_answered_retries_then_halts(
            self, harness):
        """``connect()`` was outside the retry machinery entirely: an ack
        that never arrived propagated a bare ``TimeoutError`` with no
        retry and no stop."""
        ep = self._endpoint(harness, lambda o, n: None)
        try:
            driver = harness.make_driver(ep.port, max_retries=3)
            with pytest.raises(RobotFault):
                driver.connect()
            assert ep.count(RequestKind.HANDSHAKE) == 3
            assert ep.wait_for_halt()
            assert not driver.channel_open
        finally:
            ep.close()

    def test_the_handshake_timeout_bounds_the_handshake(self, harness):
        """It must bound the wait for the *ack*, not merely the connect.

        With handshake=200 ms and command=4000 ms, one attempt takes
        ~200 ms. The KUKA client's read used to re-arm the command
        timeout before looking at a byte, so the knob it advertised
        bounded only the TCP connect.
        """
        ep = self._endpoint(harness, lambda o, n: None)
        try:
            driver = harness.make_driver(
                ep.port, max_retries=1, handshake_timeout_ms=200,
                command_timeout_ms=4000, retry_pause_ms=1)
            t0 = time.perf_counter()
            with pytest.raises(RobotFault):
                driver.connect()
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.5, (
                f"handshake took {elapsed:.2f}s at a 200 ms setting")
        finally:
            ep.close()

    # ---- 2. the escape routes -------------------------------------------

    def test_a_request_that_cannot_be_encoded_is_fatal_and_halts(
            self, harness):
        """Bypass route one, in whatever form this encoding has it.

        The failure happens before a byte is sent, on a healthy
        transport, which is why the halt must go out and why this is the
        cheapest of the three routes to get wrong: the natural
        ``except ValueError`` does not catch it.
        """
        ep = self._endpoint(harness, self._after_handshake(
            harness, harness.status_frame(RobotStatusCode.SUCCESS)))
        try:
            with harness.make_driver(ep.port) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.unencodable_target())
                assert driver.halt_attempts == 1
            assert ep.wait_for_halt(), (
                "the transport was alive; the halt had to go out")
            # Not retried: a request that cannot be built will not build
            # on the second attempt either.
            assert ep.count(RequestKind.MOVE) == 0
        finally:
            ep.close()

    def test_a_mid_frame_close_is_fatal_and_the_halt_is_attempted(
            self, harness):
        """Bypass route two. A reply that stops half way used to raise a
        bare ``ConnectionError`` past every handler: no retry, no stop,
        no close, and a socket traceback rather than a statement about a
        robot that may be mid-cycle holding a part."""
        full = harness.status_frame(RobotStatusCode.SUCCESS)
        ep = self._endpoint(harness, self._after_handshake(
            harness, Hangup(harness.truncate(full))))
        try:
            with CapturedCritical(harness.logger_name) as logged:
                with harness.make_driver(ep.port) as driver:
                    with pytest.raises(RobotFault):
                        self._move(driver, harness.reachable_target())
                    assert driver.halted
                    assert not driver.channel_open
                    assert driver.halt_attempts == 1
            assert "HALT REQUESTED" in logged.text()
        finally:
            ep.close()

    def test_a_transport_reset_is_fatal_not_a_bare_traceback(self, harness):
        """Bypass route three. ``ConnectionResetError`` is an
        ``OSError``, not a ``ValueError``, so it escaped the retry loop
        entirely."""
        ep = self._endpoint(harness, self._after_handshake(harness, RESET))
        try:
            with harness.make_driver(ep.port) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.reachable_target())
                assert driver.halted
                assert not driver.channel_open
                assert driver.halt_attempts == 1
        finally:
            ep.close()

    def test_an_undeliverable_halt_is_loud_and_preserves_the_cause(
            self, harness):
        """When the link is already gone the driver cannot do what its
        contract promises. It must say so at CRITICAL — and the failed
        stop must not replace the error that prompted it, which is how
        the original cause used to get lost.

        This is the behaviour a second driver omits without noticing,
        because omitting it makes the tests pass more *quietly*.
        """
        ep = self._endpoint(
            harness, lambda o, n: harness.status_frame(RobotStatusCode.OK))
        try:
            driver = harness.make_driver(ep.port)
            driver.connect()
            harness.break_channel(driver)
            with CapturedCritical(harness.logger_name) as logged:
                with pytest.raises(RobotFault) as err:
                    self._move(driver, harness.reachable_target())
            assert "could NOT be transmitted" in logged.text()
            assert "UNKNOWN" in logged.text()
            assert "MOVE" in str(err.value), (
                "the original cause must survive the failed halt")
            assert driver.halted
            assert driver.halt_attempts == 1
            assert not driver.halt_delivered
        finally:
            ep.close()

    # ---- 3. retry and escalation ----------------------------------------

    def test_retry_exhaustion_sends_the_halt(self, harness):
        """The headline promise, asserted against the far end: exactly
        ``max_retries`` attempts, then one halt."""
        ep = self._endpoint(harness, self._after_handshake(harness, None))
        try:
            with harness.make_driver(ep.port, max_retries=3) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.reachable_target())
            assert ep.count(RequestKind.MOVE) == 3
            assert ep.wait_for_halt()
            assert ep.kinds()[-1] is RequestKind.HALT
        finally:
            ep.close()

    @pytest.mark.parametrize("max_retries", [1, 2, 4])
    def test_max_retries_is_honoured_exactly(self, harness, max_retries):
        """Not "about that many". A driver that retried once more than
        asked would re-send a non-idempotent motion one more time."""
        ep = self._endpoint(harness, self._after_handshake(harness, None))
        try:
            with harness.make_driver(
                    ep.port, max_retries=max_retries) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.reachable_target())
            assert ep.count(RequestKind.MOVE) == max_retries
        finally:
            ep.close()

    def test_an_unparseable_reply_is_retried_then_escalates(self, harness):
        """A locally-detected integrity failure is transient."""
        broken = harness.corrupt(
            harness.status_frame(RobotStatusCode.SUCCESS))
        ep = self._endpoint(harness, self._after_handshake(harness, broken))
        try:
            with harness.make_driver(ep.port, max_retries=3) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.reachable_target())
            assert ep.count(RequestKind.MOVE) == 3
            assert ep.wait_for_halt()
        finally:
            ep.close()

    def test_a_retryable_status_is_retried_then_escalates(self, harness):
        """A controller-reported line fault means IT could not parse US.
        Same class of fault as a locally-detected one, and it was the one
        case never retried."""
        ep = self._endpoint(harness, self._after_handshake(
            harness, harness.status_frame(RobotStatusCode.CRC_ERROR)))
        try:
            with harness.make_driver(ep.port, max_retries=3) as driver:
                with pytest.raises(RobotFault) as err:
                    self._move(driver, harness.reachable_target())
            assert "CRC_ERROR" in str(err.value)
            assert ep.count(RequestKind.MOVE) == 3
            assert ep.wait_for_halt()
        finally:
            ep.close()

    @pytest.mark.parametrize("code", [RobotStatusCode.VERSION_MISMATCH,
                                      RobotStatusCode.UNSUPPORTED_COMMAND])
    def test_a_fatal_status_is_not_retried(self, harness, code):
        """Re-sending a request the far end cannot parse cannot help."""
        ep = self._endpoint(
            harness, self._after_handshake(harness, harness.status_frame(code)))
        try:
            with harness.make_driver(ep.port, max_retries=3) as driver:
                with pytest.raises(RobotFault) as err:
                    self._move(driver, harness.reachable_target())
            assert code.name in str(err.value)
            assert ep.count(RequestKind.MOVE) == 1, "must not be retried"
            assert ep.wait_for_halt()
        finally:
            ep.close()

    def test_an_unknown_status_code_is_not_silently_a_timeout(self, harness):
        """An unrecognised code used to be rewritten as ``TIMEOUT``,
        making a future firmware's status indistinguishable from an
        ordinary placement failure. It must escalate instead."""
        ep = self._endpoint(harness, self._after_handshake(
            harness, harness.unknown_status_frame()))
        try:
            with harness.make_driver(ep.port, max_retries=2) as driver:
                with pytest.raises(RobotFault):
                    self._move(driver, harness.reachable_target())
            assert ep.wait_for_halt()
        finally:
            ep.close()

    def test_the_command_deadline_covers_the_whole_frame(self, harness):
        """A controller that trickles bytes used to be accepted after 3 s
        at a 250 ms setting, because every read got a fresh timeout."""
        ep = self._endpoint(harness, self._after_handshake(harness, None))
        try:
            driver = harness.make_driver(
                ep.port, max_retries=1, command_timeout_ms=200,
                retry_pause_ms=1)
            driver.connect()
            t0 = time.perf_counter()
            with pytest.raises(RobotFault):
                self._move(driver, harness.reachable_target())
            assert time.perf_counter() - t0 < 1.5
        finally:
            ep.close()

    def test_a_request_that_cannot_be_repeated_is_never_re_sent(self, harness):
        """A read timeout is not evidence that the command did not run.

        This is the gap the retry loop had, and it was the loudest one in
        the system: the loop retried *every* transient failure with a
        bare re-send, and neither wire here carries a sequence number, so
        a ``PICK_AND_PLACE`` that timed out at the default 5000 ms
        against a seven-step cycle was re-commanded — a second
        grasp-and-insert on a cell the first attempt may already have
        placed — and every reply after it answered the previous request.

        The driver may still escalate. It may not repeat. Exactly one of
        these frames goes on the wire, and then the halt.
        """
        ep = self._endpoint(harness, self._after_handshake(harness, None))
        try:
            with harness.make_driver(ep.port, max_retries=3) as driver:
                request = harness.non_repeatable_request(driver)
                with pytest.raises(RobotFault):
                    driver.execute(request)
                assert driver.halted
                assert driver.halt_attempts == 1
            assert ep.count(request.kind) == 1, (
                f"{request.name} went out {ep.count(request.kind)} times; a "
                f"request that is not safe to repeat must be sent once and "
                f"then escalated")
            assert ep.wait_for_halt()
        finally:
            ep.close()

    def test_a_reply_after_the_deadline_is_not_answered_by_a_second_send(
            self, harness):
        """The mis-pairing itself, with a reply that really is late.

        The previous test uses silence, which cannot distinguish "the
        controller never got it" from "the controller is still working".
        Here the controller *does* answer, after the deadline, which is
        the case the docstring at ``execute`` used to name as a hazard
        and leave open: the late ``SUCCESS`` must not be paired with a
        re-sent request, and the only way to guarantee that on a wire
        with no sequence number is to not re-send.
        """
        ok = harness.status_frame(RobotStatusCode.OK)
        late = harness.status_frame(RobotStatusCode.SUCCESS)

        def reply(observed: Observed, n: int):
            if observed.kind is RequestKind.HANDSHAKE:
                return ok
            time.sleep(0.4)               # twice the 200 ms command deadline
            return late

        ep = self._endpoint(harness, reply)
        try:
            driver = harness.make_driver(
                ep.port, max_retries=3, command_timeout_ms=200,
                retry_pause_ms=1)
            driver.connect()
            request = harness.non_repeatable_request(driver)
            with pytest.raises(RobotFault):
                driver.execute(request)
            assert ep.count(request.kind) == 1, (
                "the late reply must not be given a second request to be "
                "mis-paired with")
            # The halt still had a live socket, so it went out; this is
            # asserted on the driver rather than the far end because the
            # far end is still asleep on its own reply.
            assert driver.halt_attempts == 1
            assert driver.halt_delivered
        finally:
            ep.close()

    def test_the_send_path_arms_the_full_command_timeout(self, harness):
        """The write side must be bounded, and bounded by its own knob.

        The read path arms ``remaining`` on the transport before each
        chunk and never restores it, so after a reply that lands near the
        deadline the socket carries a near-zero timeout — and the next
        ``sendall`` ran under it. A ``sendall`` that times out may have
        written *part* of a frame, and Python does not report how much:
        on a stream this protocol cannot resynchronise (``protocol.py``
        docstring, "framing is positional"), a second frame written
        behind a partial one corrupts every frame after it.

        Measured here rather than timed, because a 16-byte frame never
        stalls long enough to time a write that fails.
        """
        ok = harness.status_frame(RobotStatusCode.OK)
        done = harness.status_frame(RobotStatusCode.SUCCESS)

        def reply(observed: Observed, n: int):
            if observed.kind is RequestKind.HANDSHAKE:
                return ok
            # Late, and in two segments. Both are needed: the wait spends
            # the deadline, and the split forces a second `recv` that
            # arms what is left of it. A single-segment reply leaves the
            # full timeout armed on a fixed-length framing and the defect
            # is invisible.
            time.sleep(0.25)              # 250 ms of a 400 ms deadline
            return Trickle(done, split=6, gap_s=0.02)

        ep = self._endpoint(harness, reply)
        driver = harness.make_driver(
            ep.port, max_retries=1, command_timeout_ms=400, retry_pause_ms=1)
        try:
            driver.connect()
            status = self._move(driver, harness.reachable_target())
            assert status.code is RobotStatusCode.SUCCESS

            left = harness.transport_deadline_s(driver)
            assert left is not None and left < 0.25, (
                f"precondition: the read path should have left {left!r}s of "
                f"the 400 ms deadline armed on the transport")

            driver.send(driver.encode(Request(RequestKind.HALT, "HALT")))
            assert harness.transport_deadline_s(driver) == pytest.approx(
                0.4, abs=1e-3), (
                "send() must arm the whole command timeout, not inherit what "
                "the last read left behind")
        finally:
            driver.close()
            ep.close()

    # ---- 4. the latch ----------------------------------------------------

    def test_a_controller_reporting_a_stop_latches_the_driver(self, harness):
        """The highest-consequence item.

        A status of ESTOP used to parse cleanly, return as an ordinary
        result, and be counted by the caller as a failed placement before
        it commanded the next motion. It must instead close the channel,
        latch, and raise — and the latch must survive a reconnect
        attempt, because clearing a stop is a deliberate act at the
        controller, out of band, and on some machines it needs an
        operator.
        """
        ep = self._endpoint(harness, self._after_handshake(
            harness, harness.status_frame(RobotStatusCode.ESTOP)))
        try:
            driver = harness.make_driver(ep.port)
            driver.connect()
            with pytest.raises(RobotEstop):
                self._move(driver, harness.reachable_target())
            assert driver.halted
            assert not driver.channel_open

            # Latched. No further motion is sent — not even onto the wire.
            before = list(ep.requests)
            with pytest.raises(RobotEstop):
                self._move(driver, harness.reachable_target())
            assert ep.requests == before

            # And it will not reconnect its way out of the stop.
            with pytest.raises(RobotEstop):
                driver.connect()
        finally:
            ep.close()

    def test_halt_latches_is_idempotent_and_is_counted(self, harness):
        """``halt()`` is the in-band request, and it is one-way.

        Idempotent in *effect*: a second call cannot un-latch, cannot
        raise, and does not manufacture a second attempt it did not
        make. That last part matters — :attr:`halt_attempts` is what the
        rest of this suite asserts against, so it has to count stops
        actually put on a wire rather than calls to a method.
        """
        ep = self._endpoint(
            harness, lambda o, n: harness.status_frame(RobotStatusCode.OK))
        try:
            driver = harness.make_driver(ep.port)
            driver.connect()
            driver.halt()
            assert driver.halted and driver.halt_attempts == 1
            assert driver.halt_delivered
            assert not driver.channel_open
            assert ep.wait_for_halt()

            driver.halt()                       # must not raise
            assert driver.halt_attempts == 1, (
                "there was no channel left; nothing was attempted")
            assert driver.halted
            with pytest.raises(RobotEstop):
                self._move(driver, harness.reachable_target())
        finally:
            ep.close()

    # ---- 5. what the caller is allowed to assume ------------------------

    def test_every_returned_status_is_one_the_caller_can_act_on(
            self, harness):
        """The closed-set guarantee.

        The caller branches on four codes and raises on anything else. It
        may only do that if the driver has already turned every other
        code into an exception — which is a base-class property here, not
        an accident of the status enum happening to be exhaustive.

        **The command timeout is set here rather than left at the
        harness default, and the reason is a measurement.** The harnesses
        default to 300 ms because most of this suite is deliberate
        silence and a short deadline keeps it fast. This test is the one
        that drives a controller which actually moves: the KUKA
        simulator's ``PICK_AND_PLACE`` is seven motions and two dwells
        and takes ~340 ms at ``ms_per_100mm=2``. Under a 300 ms deadline
        it timed out — and this test passed anyway, because the retry
        loop re-sent the cycle and read the first cycle's reply as the
        answer to the second. That is the mis-pairing this suite now pins
        two tests against, and it had been running inside the suite's own
        happy path. A deadline that does not fit the operation is a test
        artefact, not a property of the driver.
        """
        port, close = harness.start_mock()
        try:
            with harness.make_driver(port, command_timeout_ms=5000) as driver:
                for request in harness.smoke_requests(driver):
                    status = driver.execute(request)
                    assert status.code in RobotDriver.ACTIONABLE_STATUS, (
                        f"{request.name} returned {status.code.name}")
                    assert status.cycle_time_ms >= 0.0
        finally:
            close()

    def test_validate_answers_the_three_valued_question_and_never_raises(
            self, harness):
        """``UNKNOWN`` is a first-class answer, not a failure.

        Of the systems this interface was designed against, UR answers
        reachability, ROS 2 answers it one layer up, and Franka and ABB
        EGM do not answer it at all — EGM's manual says the robot "will
        react quickly to all position references sent to the controller,
        also faulty ones". A driver must be able to say it does not know,
        and must never raise instead of saying so.
        """
        driver = harness.make_driver(1)          # never connected
        for target in (harness.reachable_target(),
                       harness.unencodable_target(),
                       Pose(0.0, 0.0, 0.0),
                       Pose(1e6, -1e6, 1e6)):
            assert isinstance(driver.validate(target), Reachability)

    def test_a_lossy_backend_declares_its_losses(self, harness):
        """A capability descriptor that claims nothing is worse than none.

        This project's KUKA wire carries x/y/z integers and no
        orientation, which makes it a strictly lossy backend for any
        pose type richer than that — and every other robot's is richer.
        A driver may be lossy; it may not be lossy in silence.
        """
        driver = harness.make_driver(1)
        caps = driver.capabilities
        assert caps.name
        if not caps.carries_orientation:
            assert any("orientation" in note for note in caps.lossy_notes), (
                "a driver that drops orientation must say so in lossy_notes")
        if caps.position_resolution_m > 0:
            assert caps.lossy_notes, (
                "a driver that quantises position must say so")
        assert caps.accepts_cartesian or caps.accepts_joint

    def test_the_escalation_cannot_be_overridden_by_a_subclass(self, harness):
        """The structural claim, asserted rather than asserted-to.

        An abstract base class cannot stop a driver author from
        reimplementing the command loop and losing the catch-all — which
        is exactly how the three historical bypass routes opened. This
        base rejects such a subclass at class-definition time, before any
        instance exists and before any test runs.
        """
        driver_cls = type(harness.make_driver(1))
        assert issubclass(driver_cls, RobotDriver)

        with pytest.raises(TypeError, match="sealed"):
            type("BypassDriver", (driver_cls,),
                 {"execute": lambda self, request: None})

        with pytest.raises(TypeError, match="sealed"):
            type("BypassDriver2", (driver_cls,),
                 {"_request_halt": lambda self, reason: None})
