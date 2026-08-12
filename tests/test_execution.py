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
                    op = unpack_command(buf).op
                except ValueError:
                    return
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
        assert OpCode.ESTOP in srv.ops, (
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
        assert srv.ops[-1] is OpCode.ESTOP
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
        assert srv.ops[-1] is OpCode.ESTOP
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
        assert srv.ops[-1] is OpCode.ESTOP
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
        assert srv.ops[-1] is OpCode.ESTOP
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
        assert srv.ops[-1] is OpCode.ESTOP, (
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
