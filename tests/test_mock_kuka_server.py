"""Tests for the simulated KUKA controller itself.

`execution/mock_kuka_server.py` is the only controller this project has,
and until 2026-08-12 it had no test file at all — which is why every
optimism in it went unnoticed: it accepted MOVE_TO(5000, 5000, 5000),
did not latch its E-stop, blocked a handler thread forever on a half
frame, and reported CRC_ERROR for an unknown opcode.

These tests talk to it over a real socket with hand-built frames rather
than through KukaClient, so they assert what the *controller* does, not
what the client makes of it.
"""
from __future__ import annotations

import socket
import struct
import time

import pytest

from execution.mock_kuka_server import (REACH_MM, Z_MIN_MM, OpCode,
                                        _INSERT_Z_MM, run_in_thread)
from execution.protocol import (COMMAND_LEN, STATUS_LEN, PROTOCOL_VERSION,
                                crc16_modbus, pack_command, unpack_status)


# --------------------------------------------------------- helpers ------

class _Wire:
    """A raw connection to the simulator."""

    def __init__(self, port: int, timeout: float = 5.0):
        self.sock = socket.create_connection(("127.0.0.1", port),
                                             timeout=timeout)

    def send(self, packet: bytes) -> None:
        self.sock.sendall(packet)

    def command(self, op, x=0, y=0, z=0, aux=0) -> dict:
        self.send(pack_command(op, x, y, z, aux))
        return self.status()

    def status(self) -> dict:
        buf = b""
        while len(buf) < STATUS_LEN:
            chunk = self.sock.recv(STATUS_LEN - len(buf))
            if not chunk:
                raise ConnectionError("server closed")
            buf += chunk
        return unpack_status(buf)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def server():
    srv, _t = run_in_thread(host="127.0.0.1", port=0, drop_prob=0.0,
                            ms_per_100mm=1, frame_timeout_s=0.4,
                            idle_timeout_s=5.0)
    yield srv
    srv.shutdown()


def _port(srv) -> int:
    return srv.server_address[1]


# From common.types.RobotStatusCode; the simulator speaks bytes.
OK, SUCCESS, PICK_FAILED, PLACE_FAILED = 0, 1, 2, 3
CRC_ERROR, ESTOP, UNSUPPORTED, VERSION_MISMATCH = 5, 6, 7, 8


# ---------------------------------------------------- happy baseline ----

def test_handshake_and_move(server):
    with _Wire(_port(server)) as w:
        assert w.command(OpCode.HANDSHAKE)["code"] == OK
        s = w.command(OpCode.MOVE_TO, 100, 50, 80)
        assert s["code"] == SUCCESS
        assert (s["x_mm"], s["y_mm"], s["z_mm"]) == (100, 50, 80)


# ------------------------------------------------ workspace envelope ----

@pytest.mark.parametrize("target", [
    (5000, 5000, 5000),        # far outside the 706 mm reach
    (-100000, 0, 0),
    (0, 0, -500),              # half a metre below the table
    (2 ** 31 - 1, 2 ** 31 - 1, 32767),   # the 63-day time.sleep
])
def test_out_of_envelope_targets_are_refused_and_latch_the_stop(
        server, target):
    """Every one of these returned SUCCESS. Nothing in the pipeline was
    ever told a pose was unreachable, so planning.yaml's ±350 mm
    workspace bound was enforced by no one at the robot end. The last
    row is also security finding 3: one valid 16-byte packet made the
    handler sleep for 63 days."""
    with _Wire(_port(server)) as w:
        assert w.command(OpCode.HANDSHAKE)["code"] == OK
        t0 = time.perf_counter()
        assert w.command(OpCode.MOVE_TO, *target)["code"] == ESTOP
        assert time.perf_counter() - t0 < 2.0, "must refuse before moving"
    assert server.robot.estopped


def test_the_envelope_admits_the_whole_planned_workspace(server):
    """planning.yaml plans inside ±350 mm; the envelope must not refuse
    the poses the rest of the system is designed to produce, or this
    guard silently becomes a no-op the other way."""
    with _Wire(_port(server)) as w:
        assert w.command(OpCode.HANDSHAKE)["code"] == OK
        for x, y, z in [(350, 350, 80), (-350, -350, 2), (0, 0, 0),
                        (350, -350, 60)]:
            assert w.command(OpCode.MOVE_TO, x, y, z)["code"] == SUCCESS
    assert not server.robot.estopped
    # And the constants are the modelled machine, not arbitrary.
    assert REACH_MM == 706.0 and Z_MIN_MM < 0


# ------------------------------------------------- latching E-stop ------

def test_estop_latches_across_reconnects(server):
    """After an ESTOP the simulator used to drop the connection and then
    accept a fresh one, answering MOVE_TO(300,300,300) with SUCCESS.
    The safety mechanism the FDR leads with was a no-op in the only
    implementation that exists."""
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        w.command(OpCode.VACUUM_ON, aux=80)
        assert server.robot.vacuum_on
        assert w.command(OpCode.ESTOP)["code"] == ESTOP

    assert server.robot.estopped
    assert not server.robot.vacuum_on, "a Cat-0 stop removes power"

    with _Wire(_port(server)) as w2:
        # Even the handshake is refused now — this is the case
        # KukaClient.connect()'s ESTOP branch exists for.
        assert w2.command(OpCode.HANDSHAKE)["code"] == ESTOP
        before = (server.robot.x, server.robot.y, server.robot.z)
        assert w2.command(OpCode.MOVE_TO, 300, 300, 300)["code"] == ESTOP
        assert (server.robot.x, server.robot.y, server.robot.z) == before, (
            "a latched controller must not move")


def test_estop_has_no_reset_path(server):
    """Clearing a Category-0 stop is a deliberate act at the controller.
    This simulator deliberately offers no way to do it; if one is ever
    added, this test should be replaced by one that exercises it."""
    server.robot.latch_estop("test")
    assert not hasattr(server.robot, "reset")
    assert not hasattr(server.robot, "clear_estop")


# --------------------------------------------------- socket timeouts ----

def test_a_half_frame_does_not_pin_the_handler_forever(server):
    """8 bytes then silence used to block `_recv_n` indefinitely: no
    reply, no teardown, one thread held per stuck client."""
    w = _Wire(_port(server))
    try:
        w.command(OpCode.HANDSHAKE)
        w.send(pack_command(OpCode.MOVE_TO, 10, 10, 10)[:8])
        w.sock.settimeout(3.0)
        # The server closes the channel once the frame times out.
        assert w.sock.recv(16) == b"", "the channel must be dropped"
    finally:
        w.close()


def test_an_idle_connection_between_commands_is_not_dropped(server):
    """The host is legitimately idle while it runs perception, so the
    frame timeout must not double as an idle timeout."""
    with _Wire(_port(server)) as w:
        assert w.command(OpCode.HANDSHAKE)["code"] == OK
        time.sleep(0.8)   # twice the 0.4 s frame timeout
        assert w.command(OpCode.MOVE_TO, 10, 10, 10)["code"] == SUCCESS


# ------------------------------------------------------- fault codes ----

def test_bad_crc_is_a_crc_error(server):
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        pkt = bytearray(pack_command(OpCode.MOVE_TO, 1, 2, 3))
        pkt[5] ^= 0xFF
        w.send(bytes(pkt))
        assert w.status()["code"] == CRC_ERROR


def test_unknown_opcode_is_not_reported_as_a_crc_error(server):
    """A valid-CRC frame carrying opcode 0x42 is not a line fault. The
    client cannot tell corruption from a protocol mismatch if both come
    back as CRC_ERROR, and the two want opposite responses."""
    body = struct.pack(">BBiihH", PROTOCOL_VERSION, 0x42, 0, 0, 0, 0)
    pkt = body + struct.pack("<H", crc16_modbus(body))
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        w.send(pkt)
        assert w.status()["code"] == UNSUPPORTED


def test_bad_version_is_not_reported_as_a_crc_error(server):
    body = struct.pack(">BBiihH", 0x99, int(OpCode.MOVE_TO), 0, 0, 0, 0)
    pkt = body + struct.pack("<H", crc16_modbus(body))
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        w.send(pkt)
        assert w.status()["code"] == VERSION_MISMATCH


# ------------------------------------------------- pick-and-place Z -----

def test_the_wire_z_drives_the_pick_descent_not_the_insert(server):
    """The client sends `pose.pick.z_mm` as the third coordinate. The
    simulator's parameter was named `place_z`, and it inserted THERE
    while hardcoding the pick descent to 5 — so the commanded pick
    height was silently reinterpreted as a place height and the cycle
    still reported SUCCESS.

    There is no place-Z field in the frame; the insert depth is a
    controller-side constant, as in krl_prog/routines.src.
    """
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        # Latch the place XY at transport height, as KukaClient does.
        assert w.command(OpCode.MOVE_TO, -50, 100, 80)["code"] == SUCCESS
        s = w.command(OpCode.PICK_AND_PLACE, 120, 60, 7, aux=80)
        assert s["code"] == SUCCESS
        # Ends retracted over the PLACE target, having inserted at the
        # controller-side depth.
        assert (s["x_mm"], s["y_mm"]) == (-50, 100)
    assert _INSERT_Z_MM == 2


def test_pick_and_place_refuses_an_unreachable_pick(server):
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        w.command(OpCode.MOVE_TO, 0, 0, 80)
        assert w.command(
            OpCode.PICK_AND_PLACE, 9000, 9000, 5, aux=80)["code"] == ESTOP
    assert server.robot.estopped


# ---------------------------------------------------- frame handling ----

def test_frames_split_across_packets_still_reassemble(server):
    """The frame timeout must bound a half frame without breaking a
    legitimately fragmented one."""
    with _Wire(_port(server)) as w:
        w.command(OpCode.HANDSHAKE)
        pkt = pack_command(OpCode.MOVE_TO, 10, 20, 30)
        w.send(pkt[:5])
        time.sleep(0.05)
        w.send(pkt[5:])
        s = w.status()
        assert s["code"] == SUCCESS
        assert (s["x_mm"], s["y_mm"], s["z_mm"]) == (10, 20, 30)
    assert COMMAND_LEN == 16
