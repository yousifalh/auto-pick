"""The KUKA driver, against the shared conformance suite.

Everything vendor-neutral that ``tests/test_execution.py`` used to
assert lives in ``tests/conformance.py`` now and runs from here. What
stayed behind in that file is what is genuinely about *this* encoding:
the millimetre quantiser, the 16-byte frame, the descriptive config
keys, and the KRL coupling.

The harness below is the whole of what a driver author has to write to
opt into the suite. It is ~110 lines, and none of it is safety logic.
"""
from __future__ import annotations

from common.types import RobotStatusCode, WorkspacePoint
from execution.driver import Pose, Request, RequestKind, RobotDriver
from execution.execution import ExecutionConfig, KukaClient
from execution.mock_kuka_server import run_in_thread
from execution.protocol import (COMMAND_LEN, STATUS_LEN, OpCode, pack_status,
                                unpack_command)
from tests.conformance import (ConformanceHarness, Observed,
                               RobotDriverConformance)

import pytest


_KIND_OF_OPCODE = {
    OpCode.HANDSHAKE: RequestKind.HANDSHAKE,
    OpCode.HALT: RequestKind.HALT,
    OpCode.MOVE_TO: RequestKind.MOVE,
    OpCode.VACUUM_ON: RequestKind.GRIPPER,
    OpCode.VACUUM_OFF: RequestKind.GRIPPER,
}


class KukaHarness(ConformanceHarness):
    """16-byte fixed frames, CRC-16/MODBUS, integer millimetres."""

    logger_name = "execution.kuka"

    def make_driver(self, port: int, **policy) -> RobotDriver:
        return KukaClient(ExecutionConfig(
            host="127.0.0.1",
            port=port,
            handshake_timeout_ms=policy.get("handshake_timeout_ms", 300),
            command_timeout_ms=policy.get("command_timeout_ms", 300),
            heartbeat_interval_ms=policy.get("retry_pause_ms", 1),
            max_retries=policy.get("max_retries", 3),
        ))

    # ---- framing ---------------------------------------------------------

    def read_frame(self, conn) -> bytes:
        buf = b""
        while len(buf) < COMMAND_LEN:
            chunk = conn.recv(COMMAND_LEN - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def describe(self, frame: bytes) -> Observed:
        cmd = unpack_command(frame)
        return Observed(
            kind=_KIND_OF_OPCODE.get(cmd.op, RequestKind.VENDOR),
            name=cmd.op.name,
            raw=frame,
        )

    # ---- replies ---------------------------------------------------------

    def status_frame(self, code: RobotStatusCode) -> bytes:
        return pack_status(code=int(code), x_mm=0, y_mm=0, z_mm=0, cycle_ms=0)

    def unknown_status_frame(self) -> bytes:
        return pack_status(code=99, x_mm=0, y_mm=0, z_mm=0, cycle_ms=0)

    def corrupt(self, frame: bytes) -> bytes:
        """Flip a CRC byte: a line fault, and the only honest CRC_ERROR."""
        pkt = bytearray(frame)
        pkt[14] ^= 0xFF
        return bytes(pkt)

    def truncate(self, frame: bytes) -> bytes:
        return frame[:7]           # 7 of 16 bytes, then the far end hangs up

    def break_channel(self, driver: RobotDriver) -> None:
        driver._sock.close()       # dead descriptor, still installed

    def transport_deadline_s(self, driver: RobotDriver):
        return driver._sock.gettimeout()

    # ---- targets ---------------------------------------------------------

    def reachable_target(self) -> Pose:
        return Pose.from_mm(10.0, 10.0, 10.0)

    def non_repeatable_request(self, driver: RobotDriver) -> Request:
        """``PICK_AND_PLACE``: seven controller-side steps, one frame.

        The second element of ``pick_place_requests`` — the one that
        descends, applies vacuum, transports and inserts. Re-sending it
        after a read timeout runs the cycle again, on a wire with no
        sequence number and against a controller
        (``krl_prog/laptop_comm.src`` CASE 4) that does not read another
        frame until ``PickAndPlace`` has returned.
        """
        return driver.pick_place_requests(
            pick=WorkspacePoint(100.0, 50.0, 5.0),
            place=WorkspacePoint(-50.0, 100.0, 2.0),
            transport_height_mm=80.0,
            vacuum_level_percent=80,
        )[1]

    def unencodable_target(self) -> Pose:
        """Beyond int32 on x. Raises ``CoordinateOutOfRange``, a
        ``struct.error`` — not a ``ValueError``, which is the whole
        point: this is bypass route one's exact shape."""
        return Pose.from_mm(2.0 ** 31, 0.0, 0.0)

    # ---- the well-behaved simulator --------------------------------------

    def start_mock(self):
        srv, _t = run_in_thread(host="127.0.0.1", port=0, drop_prob=0.0,
                                ms_per_100mm=2)
        return srv.server_address[1], srv.shutdown

    def smoke_requests(self, driver: RobotDriver):
        return (
            Request(RequestKind.MOVE, "MOVE_TO",
                    target=Pose.from_mm(100.0, 50.0, 80.0)),
            Request(RequestKind.GRIPPER, "VACUUM_ON",
                    payload={"on": True, "level_percent": 80}),
            Request(RequestKind.GRIPPER, "VACUUM_OFF", payload={"on": False}),
        ) + driver.pick_place_requests(
            pick=WorkspacePoint(100.0, 50.0, 5.0),
            place=WorkspacePoint(-50.0, 100.0, 2.0),
            transport_height_mm=80.0,
            vacuum_level_percent=80,
        )


class TestKukaDriverConformance(RobotDriverConformance):

    @pytest.fixture
    def harness(self) -> ConformanceHarness:
        return KukaHarness()


# ------------------------------------------------ encoding-specific ------

def test_the_halt_opcode_is_the_one_the_suite_observes():
    """The suite asserts "a halt was requested" without naming a byte.

    This is the one place the two are tied together, and it is here
    rather than in the suite on purpose: `0x06` is a KUKA fact.
    """
    assert int(OpCode.HALT) == 0x06
    assert "ESTOP" not in OpCode.__members__, (
        "0x06 was renamed from ESTOP to HALT because no surveyed robot "
        "accepts a safety-rated stop in band; re-adding the old name as an "
        "alias would make the renaming silent, which is the thing it was "
        "supposed not to be")


def test_the_status_enum_still_has_estop_and_that_is_not_a_mistake():
    """``RobotStatusCode.ESTOP`` is the *controller* reporting a real
    Category-0 stop. That claim is true and keeps its name; only the
    outbound opcode's claim was false."""
    assert RobotStatusCode.ESTOP.value == 6
    assert STATUS_LEN == COMMAND_LEN == 16
