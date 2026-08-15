"""The second driver, against the **identical** conformance suite.

The point of this file is that it is short and that it changes nothing.
:class:`tests.conformance.RobotDriverConformance` is imported unmodified
and subclassed; every assertion in it runs against a backend that shares
no byte, no framing rule, no integrity check and no unit with the KUKA
one. If a safety behaviour had leaked into the KUKA driver instead of
into the base class, this file would fail, and the design would have
been wrong.

``execution/json_driver.py`` says it in its own docstring and it is
worth repeating: that driver talks to no hardware and is not a step
toward any. It exists to hold the seam still while it is measured.
"""
from __future__ import annotations

import json
import struct

import pytest

from common.types import RobotStatusCode, WorkspacePoint
from execution.driver import Pose, Request, RequestKind, RobotDriver
from execution.json_driver import (LENGTH_PREFIX_BYTES, MAX_FRAME_BYTES,
                                   JsonDriverConfig, JsonRobotDriver,
                                   decode_frame, encode_frame)
from execution.mock_json_server import run_in_thread
from tests.conformance import (ConformanceHarness, Observed,
                               RobotDriverConformance)


_KIND_OF_TYPE = {
    "handshake": RequestKind.HANDSHAKE,
    "halt": RequestKind.HALT,
    "move": RequestKind.MOVE,
    "gripper": RequestKind.GRIPPER,
}


class JsonHarness(ConformanceHarness):
    """4-byte length prefix, UTF-8 JSON body, float metres, no checksum."""

    logger_name = "execution.json_driver"

    def make_driver(self, port: int, **policy) -> RobotDriver:
        return JsonRobotDriver(JsonDriverConfig(
            host="127.0.0.1",
            port=port,
            handshake_timeout_ms=policy.get("handshake_timeout_ms", 300),
            command_timeout_ms=policy.get("command_timeout_ms", 300),
            retry_pause_ms=policy.get("retry_pause_ms", 1),
            max_retries=policy.get("max_retries", 3),
        ))

    # ---- framing ---------------------------------------------------------

    def read_frame(self, conn) -> bytes:
        header = self._read_n(conn, LENGTH_PREFIX_BYTES)
        if not header:
            return b""
        (declared,) = struct.unpack(">I", header)
        body = self._read_n(conn, declared)
        if len(body) != declared:
            return b""
        return header + body

    @staticmethod
    def _read_n(conn, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return buf
            buf += chunk
        return buf

    def describe(self, frame: bytes) -> Observed:
        obj = decode_frame(frame)
        kind = _KIND_OF_TYPE.get(obj.get("type"), RequestKind.VENDOR)
        return Observed(kind=kind, name=str(obj.get("type")), raw=frame)

    # ---- replies ---------------------------------------------------------

    def status_frame(self, code: RobotStatusCode) -> bytes:
        return encode_frame({"status": code.name,
                             "pose_m": [0.0, 0.0, 0.0],
                             "cycle_s": 0.0})

    def unknown_status_frame(self) -> bytes:
        return encode_frame({"status": "TELEPORTED",
                             "pose_m": [0.0, 0.0, 0.0],
                             "cycle_s": 0.0})

    def corrupt(self, frame: bytes) -> bytes:
        """A body that is not JSON, with a truthful length prefix.

        The line-fault analogue for an encoding with no checksum: the
        frame arrives whole and cannot be understood. The driver must
        treat it as transient, exactly as the KUKA driver treats a CRC
        mismatch.
        """
        body = b"{not json at all" + b"." * 4
        return struct.pack(">I", len(body)) + body

    def truncate(self, frame: bytes) -> bytes:
        """A length prefix promising more than follows."""
        return frame[:LENGTH_PREFIX_BYTES + 3]

    def break_channel(self, driver: RobotDriver) -> None:
        driver._sock.close()

    def transport_deadline_s(self, driver: RobotDriver):
        return driver._sock.gettimeout()

    # ---- targets ---------------------------------------------------------

    def reachable_target(self) -> Pose:
        return Pose(0.10, 0.10, 0.10)

    def non_repeatable_request(self, driver: RobotDriver) -> Request:
        """``pick_place``: one frame, but still one grasp-and-insert.

        This backend carries both poses in a single self-describing
        message, which changes the framing and changes nothing about
        repeatability — running it twice still tries to pick a cell that
        is no longer there. That is the point of classifying by what the
        request *does* rather than by how many frames it takes.
        """
        return driver.pick_place_requests(
            pick=WorkspacePoint(100.0, 50.0, 5.0),
            place=WorkspacePoint(-50.0, 100.0, 2.0),
            transport_height_mm=80.0,
            vacuum_level_percent=80,
        )[0]

    def unencodable_target(self) -> Pose:
        """Infinite. Raises ``NonFiniteCoordinate``, an
        ``ArithmeticError`` — deliberately not a ``ValueError``, for the
        same reason ``struct.error`` is not one on the other driver."""
        return Pose(float("inf"), 0.0, 0.0)

    # ---- the well-behaved simulator --------------------------------------

    def start_mock(self):
        srv, _t = run_in_thread(host="127.0.0.1", port=0, drop_prob=0.0,
                                s_per_m=0.001)
        return srv.server_address[1], srv.shutdown

    def smoke_requests(self, driver: RobotDriver):
        return (
            Request(RequestKind.MOVE, "move", target=Pose(0.1, 0.05, 0.08)),
            Request(RequestKind.GRIPPER, "grasp",
                    payload={"close": True, "width_m": 0.02,
                             "force_n": 15.0}),
            Request(RequestKind.GRIPPER, "release", payload={"close": False}),
        ) + driver.pick_place_requests(
            pick=WorkspacePoint(100.0, 50.0, 5.0),
            place=WorkspacePoint(-50.0, 100.0, 2.0),
            transport_height_mm=80.0,
            vacuum_level_percent=80,
        )


class TestJsonDriverConformance(RobotDriverConformance):

    @pytest.fixture
    def harness(self) -> ConformanceHarness:
        return JsonHarness()


# ------------------- what this driver proves about the seam --------------

def test_the_second_driver_shares_no_encoding_with_the_first():
    """If it did, passing the same suite would prove nothing.

    Framing, integrity, units and command shape all differ. What is
    shared is the base class, and that is the claim under test.
    """
    from execution import execution as kuka_mod
    from execution import json_driver as json_mod

    assert not set(dir(json_mod)) & {"pack_command", "crc16_modbus",
                                     "unpack_status", "wire_mm"}
    assert kuka_mod.KukaClient.recv is not json_mod.JsonRobotDriver.recv
    assert kuka_mod.KukaClient.encode is not json_mod.JsonRobotDriver.encode
    # ...and the whole escalation is literally the same function object.
    assert (kuka_mod.KukaClient.execute
            is json_mod.JsonRobotDriver.execute)
    assert (kuka_mod.KukaClient._request_halt
            is json_mod.JsonRobotDriver._request_halt)


def test_this_backend_carries_what_the_kuka_wire_drops():
    """Orientation, the redundancy token, and the place Z.

    Asserting it on the *bytes* rather than on the capability
    descriptor, because a descriptor can claim anything.
    """
    pose = Pose(0.1, 0.2, 0.3, orientation=(0.0, 1.0, 0.0, 0.0),
                frame="table", tool="jaw", redundancy={"elbow": 0.4})
    driver = JsonRobotDriver(JsonDriverConfig(port=1))
    frame = driver.encode(Request(RequestKind.MOVE, "move", target=pose))
    obj = json.loads(frame[LENGTH_PREFIX_BYTES:].decode())
    assert obj["target"]["orientation_wxyz"] == [0.0, 1.0, 0.0, 0.0]
    assert obj["target"]["redundancy"] == {"elbow": 0.4}
    assert obj["target"]["frame"] == "table"
    assert obj["target"]["position_m"] == [0.1, 0.2, 0.3]

    # And the place Z, which the KUKA frame has nowhere to put.
    requests = driver.pick_place_requests(
        pick=WorkspacePoint(100.0, 50.0, 5.0),
        place=WorkspacePoint(-50.0, 100.0, 2.0),
        transport_height_mm=80.0, vacuum_level_percent=80)
    assert len(requests) == 1, (
        "this protocol is not stateful; one message carries the cycle")
    body = json.loads(
        driver.encode(requests[0])[LENGTH_PREFIX_BYTES:].decode())
    assert body["place"]["position_m"] == [-0.05, 0.1, 0.002]


def test_the_cycle_time_field_has_no_65_second_ceiling():
    """The KUKA wire's ``aux_u16`` stops counting at 65 535 ms. That is a
    property of a 16-bit field, not of robots, and it must not have
    leaked upward into the status type."""
    driver = JsonRobotDriver(JsonDriverConfig(port=1))
    status = driver.decode(encode_frame({
        "status": "SUCCESS", "pose_m": [0.0, 0.0, 0.0], "cycle_s": 300.0}))
    assert status.cycle_time_ms == 300_000.0


def test_the_length_prefix_is_not_trusted():
    """A length-prefixed reader that trusts the prefix lets a peer ask
    for a 4 GiB allocation with four bytes, and lets a short body pass
    as a whole frame."""
    with pytest.raises(ValueError):
        decode_frame(struct.pack(">I", 10) + b"short")
    with pytest.raises(ValueError):
        decode_frame(struct.pack(">I", MAX_FRAME_BYTES + 1) + b"x")
