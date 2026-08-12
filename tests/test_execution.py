"""Round-trip execution tests using the mock KUKA simulator.

Asserts that the full command pipeline (pack → TCP → mock robot →
status → unpack) works for every high-level command the planner emits,
and — the half that did not exist before 2026-08-12 — that every failure
route out of a command either returns an actionable status or attempts a
Category-0 stop and raises.

The escalation tests use purpose-built adversarial servers rather than
the mock, because the mock is (correctly) well-behaved: to observe what
the client does when a reply never arrives, or arrives half-written, you
need a server that does those things on purpose. Each one records the
opcodes it received, so "the E-stop was sent" is asserted against the
bytes on the wire rather than against a log line.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import struct
import threading
import time

import pytest

from common.types import PickPlacePose, RobotStatusCode, WorkspacePoint
from execution.execution import (ExecutionConfig, KukaClient, RobotEstop,
                                 RobotFault)
from execution.mock_kuka_server import run_in_thread
from execution.protocol import (COMMAND_LEN, STATUS_LEN, OpCode, pack_status,
                                unpack_command)


@pytest.fixture
def mock_server():
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=0.0, ms_per_100mm=5)
    port = srv.server_address[1]
    yield port
    srv.shutdown()


# ------------------------------------------------ adversarial servers ----

class _ScriptedServer:
    """A TCP server that records opcodes and replies however you say.

    ``reply(op, n)`` is called with the opcode and its 0-based index and
    returns the bytes to send back — or ``None`` to send nothing, or
    ``b""`` after setting ``close_after`` to hang up.
    """

    def __init__(self, reply):
        self._reply = reply
        self.ops: list = []
        # The whole parsed command, not just the opcode: the coordinate
        # quantisation tests below assert what actually reached the wire.
        self.cmds: list = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        n = 0
        with conn:
            conn.settimeout(5.0)
            while True:
                buf = b""
                while len(buf) < COMMAND_LEN:
                    try:
                        chunk = conn.recv(COMMAND_LEN - len(buf))
                    except OSError:
                        return
                    if not chunk:
                        return
                    buf += chunk
                try:
                    cmd = unpack_command(buf)
                except ValueError:
                    return
                op = cmd.op
                self.cmds.append(cmd)
                self.ops.append(op)
                out = self._reply(op, n)
                n += 1
                if out is None:
                    continue          # deliberate silence
                if out == b"__reset__":
                    # SO_LINGER{on, 0} makes close() send an RST.
                    conn.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))
                    return
                try:
                    conn.sendall(out)
                except OSError:
                    return
                if len(out) < STATUS_LEN:
                    return            # truncated frame, then hang up

    def wait_for_op(self, op, timeout: float = 3.0):
        """Block until ``op`` has been recorded, or ``timeout`` elapses.

        The client sends its E-stop and closes the socket in a
        ``finally``, so ``KukaClient.__exit__`` can return before this
        server's reader thread has appended the opcode. Asserting on
        ``ops`` the instant the client returns is therefore a race:
        measured at HEAD, ``test_out_of_range_coordinate_fires_the_estop``
        failed roughly 1 run in 15 for that reason and for no other —
        the E-stop had gone out, and the CRITICAL log line proved it.
        Waiting does not weaken any assertion below; the opcode still has
        to arrive, it is just given a bounded chance to.
        """
        deadline = time.monotonic() + timeout
        while op not in self.ops and time.monotonic() < deadline:
            time.sleep(0.005)
        return self.ops

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


def _ok(code=RobotStatusCode.OK):
    return pack_status(code=int(code), x_mm=0, y_mm=0, z_mm=0, cycle_ms=0)


def _fast_cfg(port, **kw):
    kw.setdefault("handshake_timeout_ms", 300)
    kw.setdefault("command_timeout_ms", 300)
    kw.setdefault("heartbeat_interval_ms", 1)
    return ExecutionConfig(host="127.0.0.1", port=port, **kw)


class _CapturedLog(logging.Handler):
    """`common.logging` sets ``propagate = False``, so pytest's caplog
    never sees these records; attach to the client's own logger."""

    def __init__(self):
        super().__init__(level=logging.CRITICAL)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("execution.kuka").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("execution.kuka").removeHandler(self)

    def text(self) -> str:
        return "\n".join(self.messages)


# ----------------------------------------------------- happy path -------

def test_handshake_via_context_manager(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server,
                          handshake_timeout_ms=1000)
    with KukaClient(cfg):
        pass  # enter + exit must not raise


def test_move_to_returns_success(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        s = k.move_to(WorkspacePoint(100, 50, 80))
        assert s.code in (RobotStatusCode.SUCCESS, RobotStatusCode.OK)


def test_vacuum_on_off(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        s = k.vacuum(True)
        assert s.code == RobotStatusCode.SUCCESS
        s = k.vacuum(False)
        assert s.code == RobotStatusCode.SUCCESS


def test_pick_and_place_succeeds(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    pose = PickPlacePose(
        pick=WorkspacePoint(100.0, 50.0, 30.0),
        place=WorkspacePoint(-50.0, 100.0, 5.0),
        cartridge_id=0, grid_row=1, grid_col=2,
    )
    with KukaClient(cfg) as k:
        status = k.pick_and_place(pose)
        assert status.code == RobotStatusCode.SUCCESS
        assert status.cycle_time_ms >= 0
        # Robot should have ended near the place target
        assert abs(status.current_pose.x_mm - (-50)) <= 1
        assert abs(status.current_pose.y_mm - 100) <= 1


def test_pick_failure_reported():
    """drop_prob=1.0 forces every pick to fail."""
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=1.0, ms_per_100mm=2)
    port = srv.server_address[1]
    try:
        cfg = ExecutionConfig(host="127.0.0.1", port=port)
        pose = PickPlacePose(
            pick=WorkspacePoint(10, 10, 5),
            place=WorkspacePoint(20, 20, 5),
            cartridge_id=0, grid_row=0, grid_col=0,
        )
        with KukaClient(cfg) as k:
            status = k.pick_and_place(pose)
            assert status.code == RobotStatusCode.PICK_FAILED
    finally:
        srv.shutdown()


# --------------------------------------- the three E-stop escape routes --

def test_out_of_range_coordinate_fires_the_estop():
    """struct.error is NOT a ValueError subclass, so an out-of-int32
    coordinate used to leave `_cmd_and_wait` uncaught: no retry, no
    E-stop, and — uniquely among the escapes — a perfectly healthy
    socket over which the stop could have been sent."""
    srv = _ScriptedServer(lambda op, n: _ok())
    try:
        with KukaClient(_fast_cfg(srv.port)) as k:
            with pytest.raises(RobotFault):
                k.move_to(WorkspacePoint(2 ** 31, 0, 0))
        assert OpCode.ESTOP in srv.wait_for_op(OpCode.ESTOP), (
            "the socket was alive; the E-stop had to go out")
        # Not retried: a coordinate that does not fit in the frame will
        # not fit on the second attempt either.
        assert srv.ops.count(OpCode.MOVE_TO) == 0
    finally:
        srv.close()


def test_mid_frame_close_is_fatal_and_the_stop_is_attempted():
    """A controller that closes after 7 of 16 status bytes raised a bare
    ConnectionError past every handler: no retry, no E-stop, no close,
    and a socket traceback rather than a statement about a robot that
    may be mid-PICK_AND_PLACE with the vacuum on."""
    def reply(op, n):
        if op is OpCode.HANDSHAKE:
            return _ok()
        return _ok(RobotStatusCode.SUCCESS)[:7]   # half a frame, then close

    srv = _ScriptedServer(reply)
    try:
        with _CapturedLog() as logged:
            with KukaClient(_fast_cfg(srv.port)) as k:
                with pytest.raises(RobotFault) as err:
                    k.move_to(WorkspacePoint(10, 10, 10))
                assert k.estopped
                assert k._sock is None, "the socket must be closed"
        assert "7/16" in str(err.value), (
            "the fault must say the frame was truncated, not just 'closed'")
        assert "E-STOP" in logged.text()
    finally:
        srv.close()


def test_connection_reset_is_fatal_not_a_bare_traceback():
    """RST mid-command. ConnectionResetError is an OSError, not a
    ValueError, so it escaped the retry loop entirely."""
    def reply(op, n):
        if op is OpCode.HANDSHAKE:
            return _ok()
        return b"__reset__"

    srv = _ScriptedServer(reply)
    try:
        with KukaClient(_fast_cfg(srv.port)) as k:
            with pytest.raises(RobotFault):
                k.move_to(WorkspacePoint(10, 10, 10))
            assert k.estopped
            assert k._sock is None
    finally:
        srv.close()


def test_an_undeliverable_estop_is_logged_critical_not_swallowed():
    """When the link is already gone the client cannot do what the
    docstring promises. It must say so — and the failed stop must not
    replace the error that prompted it, which is how the original cause
    used to get lost."""
    srv = _ScriptedServer(lambda op, n: _ok())
    try:
        k = KukaClient(_fast_cfg(srv.port))
        k.connect()
        k._sock.close()          # dead descriptor, still installed
        with _CapturedLog() as logged:
            with pytest.raises(RobotFault) as err:
                k.move_to(WorkspacePoint(10, 10, 10))
        assert "could NOT be transmitted" in logged.text()
        assert "state is UNKNOWN" in logged.text()
        assert "MOVE_TO failed" in str(err.value), (
            "the original cause must survive the failed E-stop")
        assert k.estopped
    finally:
        srv.close()


def test_retry_exhaustion_sends_the_estop():
    """The headline safety promise, asserted against the wire for the
    first time: three MOVE_TOs then one ESTOP. FDR_v2's traceability
    matrix cited `drop_probability=1.0` as evidence for this; that test
    asserts PICK_FAILED and never reaches the escalation at all."""
    srv = _ScriptedServer(
        lambda op, n: _ok() if op is OpCode.HANDSHAKE else None)
    try:
        with KukaClient(_fast_cfg(srv.port, max_retries=3)) as k:
            with pytest.raises(RobotFault):
                k.move_to(WorkspacePoint(10, 10, 10))
        assert srv.ops == [OpCode.HANDSHAKE, OpCode.MOVE_TO, OpCode.MOVE_TO,
                           OpCode.MOVE_TO, OpCode.ESTOP]
    finally:
        srv.close()


def test_corrupt_status_frames_retry_then_escalate():
    """A locally-detected CRC failure is transient: retried, then
    escalated."""
    def reply(op, n):
        if op is OpCode.HANDSHAKE:
            return _ok()
        pkt = bytearray(_ok(RobotStatusCode.SUCCESS))
        pkt[14] ^= 0xFF
        return bytes(pkt)

    srv = _ScriptedServer(reply)
    try:
        with KukaClient(_fast_cfg(srv.port, max_retries=3)) as k:
            with pytest.raises(RobotFault):
                k.move_to(WorkspacePoint(10, 10, 10))
        assert srv.ops.count(OpCode.MOVE_TO) == 3
        assert srv.wait_for_op(OpCode.ESTOP)[-1] is OpCode.ESTOP
    finally:
        srv.close()


# ------------------------------------ a controller that says it stopped --

def test_controller_estop_stops_the_client():
    """The highest-consequence item: a status of ESTOP used to parse
    cleanly, return as an ordinary result, and be counted by main.py as
    a failed place before it commanded the next motion."""
    srv = _ScriptedServer(
        lambda op, n: _ok() if op is OpCode.HANDSHAKE
        else _ok(RobotStatusCode.ESTOP))
    try:
        k = KukaClient(_fast_cfg(srv.port))
        k.connect()
        with pytest.raises(RobotEstop):
            k.move_to(WorkspacePoint(10, 10, 10))

        assert k.estopped
        assert k._sock is None, "a stopped controller closes the channel"

        # Latched. No further motion is sent — not even onto the wire.
        before = list(srv.ops)
        with pytest.raises(RobotEstop):
            k.move_to(WorkspacePoint(20, 20, 20))
        assert srv.ops == before

        # And it will not reconnect its way out of the stop.
        with pytest.raises(RobotEstop):
            k.connect()
    finally:
        srv.close()


def test_controller_crc_error_is_retried_then_escalates():
    """A controller-reported CRC_ERROR means IT could not parse US. That
    is the same class of fault as a locally-detected one and was the one
    case never retried."""
    srv = _ScriptedServer(
        lambda op, n: _ok() if op is OpCode.HANDSHAKE
        else _ok(RobotStatusCode.CRC_ERROR))
    try:
        with KukaClient(_fast_cfg(srv.port, max_retries=3)) as k:
            with pytest.raises(RobotFault) as err:
                k.move_to(WorkspacePoint(10, 10, 10))
        assert "CRC_ERROR" in str(err.value)
        assert srv.ops.count(OpCode.MOVE_TO) == 3
        assert srv.wait_for_op(OpCode.ESTOP)[-1] is OpCode.ESTOP
    finally:
        srv.close()


@pytest.mark.parametrize("code", [RobotStatusCode.VERSION_MISMATCH,
                                  RobotStatusCode.UNSUPPORTED_COMMAND])
def test_protocol_mismatch_is_fatal_without_retrying(code):
    """Re-sending a frame the far end cannot parse cannot help. Stop."""
    srv = _ScriptedServer(
        lambda op, n: _ok() if op is OpCode.HANDSHAKE else _ok(code))
    try:
        with KukaClient(_fast_cfg(srv.port, max_retries=3)) as k:
            with pytest.raises(RobotFault) as err:
                k.move_to(WorkspacePoint(10, 10, 10))
        assert code.name in str(err.value)
        assert srv.ops.count(OpCode.MOVE_TO) == 1, "must not be retried"
        assert srv.wait_for_op(OpCode.ESTOP)[-1] is OpCode.ESTOP
    finally:
        srv.close()


def test_unknown_status_code_is_not_silently_a_timeout():
    """An unrecognised code used to be rewritten as TIMEOUT, making a
    future firmware's status indistinguishable from a placement
    failure."""
    srv = _ScriptedServer(
        lambda op, n: _ok() if op is OpCode.HANDSHAKE
        else pack_status(code=99, x_mm=0, y_mm=0, z_mm=0, cycle_ms=0))
    try:
        with KukaClient(_fast_cfg(srv.port, max_retries=2)) as k:
            with pytest.raises(RobotFault) as err:
                k.move_to(WorkspacePoint(10, 10, 10))
        assert "unknown status code 99" in str(err.value)
        assert srv.wait_for_op(OpCode.ESTOP)[-1] is OpCode.ESTOP
    finally:
        srv.close()


# ------------------------------------------------- the handshake path ----

def test_handshake_refused_with_estop_raises_and_closes_the_socket():
    """The refusal is very often the controller reporting it is ALREADY
    stopped. It used to raise a bare string with the socket still open,
    so `with KukaClient(...)` leaked the descriptor on every refusal —
    __exit__ never runs when __enter__ raises."""
    srv = _ScriptedServer(lambda op, n: _ok(RobotStatusCode.ESTOP))
    try:
        k = KukaClient(_fast_cfg(srv.port))
        with pytest.raises(RobotEstop):
            k.connect()
        assert k._sock is None, "the descriptor must not leak"
        assert k.estopped

        # The context-manager form must not leak either.
        k2 = KukaClient(_fast_cfg(srv.port))
        with pytest.raises(RobotEstop):
            with k2:
                pass
        assert k2._sock is None
    finally:
        srv.close()


def test_handshake_timeout_retries_then_escalates():
    """`connect()` was outside the retry machinery entirely: an ack that
    never arrived propagated a bare TimeoutError with no retry and no
    stop."""
    srv = _ScriptedServer(lambda op, n: None)
    try:
        k = KukaClient(_fast_cfg(srv.port, max_retries=3))
        with pytest.raises(RobotFault):
            k.connect()
        assert srv.ops.count(OpCode.HANDSHAKE) == 3, "no retry at all before"
        assert srv.wait_for_op(OpCode.ESTOP)[-1] is OpCode.ESTOP, (
            "the socket is alive at handshake time, so the stop goes out")
        assert k._sock is None
    finally:
        srv.close()


def test_handshake_timeout_ms_actually_times_the_handshake():
    """`_recv_status` re-armed command_timeout_ms before reading a byte,
    so the knob bounded only the TCP connect. With handshake=200 ms and
    command=4000 ms, one attempt must take ~200 ms, not ~4 s."""
    srv = _ScriptedServer(lambda op, n: None)
    try:
        k = KukaClient(ExecutionConfig(
            host="127.0.0.1", port=srv.port, max_retries=1,
            handshake_timeout_ms=200, command_timeout_ms=4000,
            heartbeat_interval_ms=1))
        t0 = time.perf_counter()
        with pytest.raises(RobotFault):
            k.connect()
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5, (
            f"handshake took {elapsed:.2f}s; handshake_timeout_ms=200 must "
            "bound the wait for the ack, not just connect()")
    finally:
        srv.close()


def test_command_timeout_is_a_whole_frame_deadline():
    """A controller that trickles bytes used to be accepted after 3 s at
    a 250 ms setting, because every recv got a fresh timeout."""
    def reply(op, n):
        if op is OpCode.HANDSHAKE:
            return _ok()
        return None

    srv = _ScriptedServer(reply)
    try:
        k = KukaClient(_fast_cfg(srv.port, max_retries=1,
                                 command_timeout_ms=200))
        k.connect()
        t0 = time.perf_counter()
        with pytest.raises(RobotFault):
            k.move_to(WorkspacePoint(1, 1, 1))
        assert time.perf_counter() - t0 < 1.5
    finally:
        srv.close()


# -------------------- RobotStatusCode <-> the KRL subroutine -------------
#
# `common.types.RobotStatusCode` says "Values must not be renumbered
# without matching updates in execution.protocol and the KRL
# subroutine". `execution/protocol.py` names the enum only in prose and
# packs a raw int; the KRL side is a bare `RETURN 2 ; PICK_FAILED`
# literal in a file no test read and no import touches. Renumbering the
# enum therefore silently redefined what the real controller means by
# `2` - on hardware, and with nothing to notice until a cell was
# dropped. These tests are the missing coupling.

_KRL_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "execution", "krl_prog", "routines.src")

# `      RETURN 2                 ; PICK_FAILED`
_KRL_RETURN = re.compile(
    r"^\s*RETURN\s+(-?\d+)\s*;\s*([A-Z_][A-Z0-9_]*)\s*$", re.MULTILINE)


def _krl_returns() -> list:
    with open(_KRL_SRC, encoding="utf-8", errors="replace") as fh:
        return [(int(v), name) for v, name in _KRL_RETURN.findall(fh.read())]


def test_krl_subroutine_returns_the_numbers_this_enum_declares():
    """Every `RETURN n ; NAME` in routines.src must equal
    RobotStatusCode[NAME].value."""
    returns = _krl_returns()
    for value, name in returns:
        assert name in RobotStatusCode.__members__, (
            f"routines.src returns {value} labelled {name}, which is not a "
            f"RobotStatusCode. The controller and the host disagree about "
            f"what the wire means.")
        assert RobotStatusCode[name].value == value, (
            f"routines.src returns {value} for {name} but "
            f"RobotStatusCode.{name} is {RobotStatusCode[name].value}. "
            f"Renumbering the enum without editing the KRL redefines what "
            f"the real controller is saying.")


def test_the_krl_coupling_test_is_not_vacuous():
    """A reformat of routines.src must not silently disarm the test above.

    The regex is the whole mechanism; if it stops matching, the previous
    test passes over an empty list and the coupling is gone again — the
    exact failure mode this pair exists to prevent.
    """
    returns = _krl_returns()
    assert len(returns) >= 2, (
        f"parsed {len(returns)} `RETURN n ; NAME` lines out of "
        f"{_KRL_SRC}; the coupling test needs the comment labels to "
        f"survive. Keep the `; NAME` suffix, or update the regex here.")
    names = {name for _, name in returns}
    assert {"SUCCESS", "PICK_FAILED"} <= names, (
        f"routines.src no longer returns both SUCCESS and PICK_FAILED "
        f"(found {sorted(names)})")


def test_codes_the_real_controller_never_emits_are_named_as_such():
    """7 and 8 are simulator-only, and that is worth pinning.

    `UNSUPPORTED_COMMAND` and `VERSION_MISMATCH` exist so a controller
    that cannot parse a frame can say WHY, and `KukaClient` treats them
    as fatal rather than retryable. The KRL subroutine does not
    participate: it returns neither. The reasoning is sound and the gap
    is real, so it is asserted rather than left to be rediscovered.
    """
    emitted = {name for _, name in _krl_returns()}
    assert "UNSUPPORTED_COMMAND" not in emitted
    assert "VERSION_MISMATCH" not in emitted


# ------------------------------- millimetre -> wire quantisation ---------
#
# `WorkspacePoint` carries floats; the 16-byte frame carries signed
# int32 millimetres. That conversion used to be `int()`, which truncates
# TOWARD ZERO: every commanded coordinate lost up to 0.999 mm, always
# toward the workspace origin, and the bias REVERSED SIGN at 0 - two
# cartridges either side of the origin were both pulled inward, toward
# each other. On a 4.25 mm shipping-wall inset that is 23 % of the
# margin. It is also applied AFTER `WorkspaceBounds.require` has
# validated the float, so the value that was checked was not the value
# that was commanded.

def _record(fn) -> list:
    """Run ``fn(client)`` against a server that records every command."""
    srv = _ScriptedServer(lambda op, n: _ok(RobotStatusCode.SUCCESS))
    try:
        with KukaClient(_fast_cfg(srv.port)) as k:
            fn(k)
        return [c for c in srv.cmds if c.op is not OpCode.HANDSHAKE]
    finally:
        srv.close()


def test_wire_mm_rounds_and_is_symmetric_about_zero():
    """The unit test of the quantiser itself, both signs and the tie."""
    from execution.execution import wire_mm

    # The measured defect: int() gave 12 and -12 for these two.
    assert wire_mm(12.9) == 13
    assert wire_mm(-12.9) == -13
    assert wire_mm(0.6) == 1
    assert wire_mm(-0.6) == -1
    assert wire_mm(349.9) == 350
    assert wire_mm(-349.9) == -350

    # Symmetric about zero: no coordinate may be treated differently for
    # being on the far side of the origin. This is the property `int()`
    # broke, and it is the one that matters for a workspace that
    # straddles zero.
    for v in (0.1, 0.5, 0.9, 1.4, 1.5, 2.5, 12.3, 12.5, 12.9, 349.4999):
        assert wire_mm(-v) == -wire_mm(v), v

    # Bounded by half a millimetre everywhere, which `int()`'s 0.999 mm
    # was not. Exact halves are included deliberately: round-half-to-even
    # is what makes the error unbiased over a population of poses.
    for i in range(-4000, 4001):
        v = i / 8.0                      # 0.125 mm steps, both signs
        assert abs(wire_mm(v) - v) <= 0.5, v

    assert wire_mm(0.5) == 0 and wire_mm(1.5) == 2 and wire_mm(2.5) == 2, (
        "banker's rounding: ties go to even, so a population of poses "
        "gains no net outward or inward drift")

    # Integers must survive untouched - a pose already on a millimetre
    # is not a rounding question.
    for v in (-350.0, -1.0, 0.0, 1.0, 350.0):
        assert wire_mm(v) == int(v)


def test_move_to_rounds_the_commanded_coordinate():
    cmds = _record(lambda k: k.move_to(WorkspacePoint(12.9, -12.9, 0.6)))
    assert len(cmds) == 1
    c = cmds[0]
    assert c.op is OpCode.MOVE_TO
    # int() would have sent (12, -12, 0): 0.9 mm lost on x, 0.9 mm lost
    # on y in the OPPOSITE direction, and z collapsed to the origin.
    assert (c.x_mm, c.y_mm, c.z_mm) == (13, -13, 1)


def test_pick_and_place_rounds_every_coordinate_it_sends():
    pose = PickPlacePose(
        pick=WorkspacePoint(-100.5, 100.4, 30.7),
        place=WorkspacePoint(-49.6, 100.5, 5.0),
        cartridge_id=0, grid_row=1, grid_col=2,
    )
    cmds = _record(lambda k: k.pick_and_place(pose))
    assert [c.op for c in cmds] == [OpCode.MOVE_TO, OpCode.PICK_AND_PLACE]

    transport, pick = cmds
    # The transport MOVE_TO latches the place XY. int() sent (-49, 100).
    assert (transport.x_mm, transport.y_mm) == (-50, 100)
    assert transport.z_mm == round(80.0)          # cfg.transport_height_mm
    # int() sent (-100, 100, 30) for the pick.
    assert (pick.x_mm, pick.y_mm, pick.z_mm) == (-100, 100, 31)


def test_no_commanded_coordinate_is_displaced_by_more_than_half_a_mm():
    """The property `WorkspaceBounds.require` is entitled to assume.

    `require` validates the float pose in the planner; this quantiser is
    what turns it into the integers the controller receives. The two are
    only reconcilable if the displacement between them is bounded, and
    bounded symmetrically - which is asserted here against the bytes on
    the wire rather than against the helper in isolation.
    """
    from execution.execution import wire_mm

    poses = [(x / 7.0, -x / 7.0, x / 11.0) for x in range(-40, 41)]
    cmds = _record(
        lambda k: [k.move_to(WorkspacePoint(*p)) for p in poses])
    assert len(cmds) == len(poses)
    for (x, y, z), c in zip(poses, cmds):
        assert abs(c.x_mm - x) <= 0.5 and abs(c.y_mm - y) <= 0.5
        assert abs(c.z_mm - z) <= 0.5
        assert (c.x_mm, c.y_mm, c.z_mm) == (
            wire_mm(x), wire_mm(y), wire_mm(z))


def test_connect_to_a_closed_port_raises_robotfault():
    """Nothing has been commanded and there is no channel to stop over —
    but the failure must still arrive as this module's own exception
    type rather than a raw socket error."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(RobotFault, match="could not open"):
        KukaClient(_fast_cfg(port)).connect()


# ------------------------------------------------------- configuration ---

def test_execution_config_from_dict():
    cfg = ExecutionConfig.from_dict({
        "kuka": {"host": "10.0.0.1", "port": 1234, "max_retries": 7},
        "motion": {"transport_height_mm": 55.0, "vacuum_level_percent": 95},
    })
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 1234
    assert cfg.max_retries == 7
    assert cfg.transport_height_mm == 55.0
    assert cfg.vacuum_level_percent == 95


def test_execution_config_defaults():
    cfg = ExecutionConfig.from_dict({})
    assert cfg.host == "172.31.1.147"
    assert cfg.port == 54600
    assert cfg.transport_height_mm == 80.0


def test_inert_motion_keys_are_gone():
    """approach_height_mm / insert_height_mm were parsed, stored, and
    unit-tested, and no KukaClient method read either. A key that a test
    asserts and nothing uses is worse than a missing one."""
    for dead in ("approach_height_mm", "insert_height_mm",
                 "grasp_height_mm", "default_velocity_mm_s",
                 "safety_max_velocity_mm_s"):
        assert not hasattr(ExecutionConfig(), dead), dead

    # ...and unknown keys in the file are still tolerated, so older
    # configs carrying them keep loading.
    ExecutionConfig.from_dict({"motion": {"insert_height_mm": 2.0}})


def test_shipped_execution_yaml_declares_no_dead_safety_key():
    """`safety_max_velocity_mm_s: 250` sat in this file, was cited in the
    FDR's safety discussion, and was enforced by nothing. The frame has
    no velocity field, so the host cannot cap a speed; the cap is
    controller-side in krl_prog/routines.src."""
    from common.config import load_yaml

    raw = load_yaml("configs/execution.yaml")
    motion = raw.get("motion", {})
    for dead in ("safety_max_velocity_mm_s", "default_velocity_mm_s",
                 "insert_height_mm", "approach_height_mm",
                 "grasp_height_mm"):
        assert dead not in motion, f"{dead} is read by nothing"

    # The shipped file must still load, and its descriptive keys must
    # still describe the code.
    cfg = ExecutionConfig.from_dict(raw)
    assert cfg.transport_height_mm == 80.0


@pytest.mark.parametrize("bad, msg", [
    ({"command_length_bytes": 20}, "command_length_bytes"),
    ({"crc_polynomial": 0x8005}, "crc_polynomial"),
    ({"stop_category": 1}, "stop_category"),
])
def test_descriptive_kuka_keys_are_checked_not_ignored(bad, msg):
    """These three keys are claims about execution/protocol.py. They used
    to be read by nothing, so editing crc_polynomial changed no CRC."""
    with pytest.raises(ValueError, match=msg):
        ExecutionConfig.from_dict({"kuka": bad})


def test_status_frame_length_is_what_the_client_reads():
    assert STATUS_LEN == COMMAND_LEN == 16
