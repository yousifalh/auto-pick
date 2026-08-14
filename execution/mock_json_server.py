"""The second mock: a controller speaking length-prefixed JSON.

Companion to :mod:`execution.json_driver`, and like it, **not a model of
any real machine**. It exists so the second driver has something to talk
to and so the conformance suite in ``tests/conformance.py`` can be run
against a backend that shares no bytes with the KUKA one.

What it does model, because the conformance suite needs a controller
that does these things:

* a **latching stop** — once stopped, every message including the
  handshake answers ``ESTOP``, the gripper opens, and no reconnect
  clears it;
* a **workspace envelope** — a sphere plus a table, matching
  :func:`execution.json_driver.JsonRobotDriver.validate`, so that
  ``validate`` saying ``UNREACHABLE`` and the controller refusing the
  same target are the same statement;
* **distinct fault codes** — an unknown message type is
  ``UNSUPPORTED_COMMAND``, a bad protocol number is
  ``VERSION_MISMATCH``, an unparseable frame is ``CRC_ERROR`` (the
  nearest thing this enum has to "line fault"; there is no CRC here);
* **a frame deadline and an idle timeout**, bounded separately, for the
  same reason the KUKA simulator does it.

It does **not** model motion time beyond a crude distance rule, drive
enable, operating modes, exclusivity, or anything else, and no claim in
this repository rests on it.
"""
from __future__ import annotations

import argparse
import math
import socket
import socketserver
import struct
import threading
import time
from typing import Optional, Tuple

from common.logging import get_logger
from .json_driver import (
    ENVELOPE_RADIUS_M,
    ENVELOPE_Z_MAX_M,
    ENVELOPE_Z_MIN_M,
    LENGTH_PREFIX_BYTES,
    MAX_FRAME_BYTES,
    JsonProtocolError,
    decode_frame,
    encode_frame,
)

log = get_logger("execution.mock_json")

PROTOCOL = 1
DEFAULT_FRAME_TIMEOUT_S = 2.0
DEFAULT_IDLE_TIMEOUT_S = 120.0


class _RobotState:
    """Pose, gripper and latch for one server."""

    def __init__(self, drop_prob: float, s_per_m: float) -> None:
        self.x = self.y = 0.0
        self.z = 0.2
        self.gripping = False
        self.estopped = False
        self.drop_prob = drop_prob
        self.s_per_m = s_per_m
        self.lock = threading.Lock()

    @staticmethod
    def in_envelope(x: float, y: float, z: float) -> bool:
        if not all(math.isfinite(v) for v in (x, y, z)):
            return False
        if not ENVELOPE_Z_MIN_M <= z <= ENVELOPE_Z_MAX_M:
            return False
        return math.sqrt(x * x + y * y + z * z) <= ENVELOPE_RADIUS_M

    def latch_estop(self, reason: str) -> None:
        """Enter a latching stop. Idempotent, never cleared."""
        with self.lock:
            if self.estopped:
                return
            self.estopped = True
            self.gripping = False
        log.critical("mock-json: LATCHED STOP — %s", reason)

    def move_to(self, x: float, y: float, z: float) -> float:
        """Returns simulated cycle seconds. Raises if out of envelope."""
        if not self.in_envelope(x, y, z):
            raise _OutOfEnvelope(f"({x:.3f}, {y:.3f}, {z:.3f}) m is outside "
                                 f"the modelled envelope")
        dx, dy, dz = x - self.x, y - self.y, z - self.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        with self.lock:
            self.x, self.y, self.z = x, y, z
        t = dist * self.s_per_m + 0.005
        time.sleep(min(t, 0.5))
        return t


class _OutOfEnvelope(Exception):
    pass


def _position(obj: dict, key: str = "target") -> Tuple[float, float, float]:
    target = obj.get(key)
    if not isinstance(target, dict):
        raise JsonProtocolError(f"{key!r} missing or not an object")
    pos = target.get("position_m")
    if not isinstance(pos, list) or len(pos) != 3:
        raise JsonProtocolError(f"{key}.position_m malformed: {pos!r}")
    try:
        return float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, ValueError) as exc:
        raise JsonProtocolError(f"{key}.position_m not numeric") from exc


class _Handler(socketserver.BaseRequestHandler):

    def handle(self) -> None:
        robot: _RobotState = self.server.robot           # type: ignore
        log.info("mock-json: client connected from %s", self.client_address)
        try:
            while True:
                frame = self._recv_frame()
                if frame is None:
                    return
                try:
                    obj = decode_frame(frame)
                except JsonProtocolError as exc:
                    log.warning("mock-json: %s — CRC_ERROR", exc)
                    self._reply("CRC_ERROR", 0.0, robot)
                    continue

                kind = obj.get("type")
                code, cycle_s = self._dispatch(kind, obj, robot)
                self._reply(code, cycle_s, robot)
                if kind == "halt":
                    return
        except (ConnectionError, OSError) as exc:
            log.info("mock-json: client disconnected (%s)", exc)

    # ---- dispatch --------------------------------------------------------

    def _dispatch(self, kind, obj, robot: _RobotState) -> Tuple[str, float]:
        if robot.estopped:
            return "ESTOP", 0.0
        if kind == "halt":
            robot.latch_estop("halt requested by the host")
            return "ESTOP", 0.0
        if kind == "handshake":
            if obj.get("protocol") != PROTOCOL:
                return "VERSION_MISMATCH", 0.0
            return "OK", 0.0
        if kind == "move":
            try:
                return "SUCCESS", robot.move_to(*_position(obj))
            except _OutOfEnvelope as exc:
                robot.latch_estop(f"software limit: {exc}")
                return "ESTOP", 0.0
            except JsonProtocolError:
                return "CRC_ERROR", 0.0
        if kind == "gripper":
            robot.gripping = bool(obj.get("close"))
            return "SUCCESS", 0.0
        if kind == "pick_place":
            return self._pick_place(obj, robot)
        return "UNSUPPORTED_COMMAND", 0.0

    def _pick_place(self, obj, robot: _RobotState) -> Tuple[str, float]:
        """One message, both poses — including the place Z.

        Written out step by step rather than delegated, so that the
        difference from the KUKA simulator is legible: nothing here is
        latched by a previous message, and the insert depth is the one
        the host sent rather than a constant this file chose.
        """
        import random

        try:
            px, py, pz = _position(obj, "target")
            qx, qy, qz = _position(obj, "place")
        except JsonProtocolError:
            return "CRC_ERROR", 0.0
        transport = float(obj.get("transport_height_m", 0.08))

        total = 0.0
        try:
            total += robot.move_to(px, py, max(robot.z, transport))
            total += robot.move_to(px, py, pz)
            robot.gripping = True
            if random.random() < robot.drop_prob:
                robot.gripping = False
                return "PICK_FAILED", total
            total += robot.move_to(px, py, transport)
            total += robot.move_to(qx, qy, transport)
            total += robot.move_to(qx, qy, qz)      # the place Z, on the wire
            robot.gripping = False
            total += robot.move_to(qx, qy, transport)
        except _OutOfEnvelope as exc:
            robot.latch_estop(f"software limit: {exc}")
            return "ESTOP", 0.0
        if random.random() < robot.drop_prob / 2:
            return "PLACE_FAILED", total
        return "SUCCESS", total

    # ---- I/O -------------------------------------------------------------

    def _recv_frame(self) -> Optional[bytes]:
        header = self._recv_n(LENGTH_PREFIX_BYTES, idle=True)
        if header is None:
            return None
        (declared,) = struct.unpack(">I", header)
        if declared > MAX_FRAME_BYTES:
            log.warning("mock-json: peer declared %d bytes; closing", declared)
            return None
        body = self._recv_n(declared, idle=False)
        if body is None:
            return None
        return header + body

    def _recv_n(self, n: int, idle: bool) -> Optional[bytes]:
        server = self.server                              # type: ignore
        buf = b""
        while len(buf) < n:
            timeout = (server.idle_timeout_s if idle and not buf
                       else server.frame_timeout_s)
            try:
                self.request.settimeout(timeout)
                chunk = self.request.recv(n - len(buf))
            except socket.timeout:
                if buf:
                    log.warning("mock-json: partial frame (%d/%d) timed out",
                                len(buf), n)
                return None
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _reply(self, code: str, cycle_s: float, robot: _RobotState) -> None:
        self.request.sendall(encode_frame({
            "status": code,
            "pose_m": [robot.x, robot.y, robot.z],
            # Seconds, as a float, with no ceiling — the field this
            # protocol has instead of a uint16 millisecond counter that
            # wraps at 65 535.
            "cycle_s": round(float(cycle_s), 6),
            "gripping": robot.gripping,
        }))


class _MockJsonServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(
    host: str = "127.0.0.1",
    port: int = 55600,
    drop_prob: float = 0.0,
    s_per_m: float = 0.05,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    idle_timeout_s: Optional[float] = DEFAULT_IDLE_TIMEOUT_S,
) -> _MockJsonServer:
    server = _MockJsonServer((host, port), _Handler)
    server.robot = _RobotState(drop_prob, s_per_m)
    server.frame_timeout_s = frame_timeout_s
    server.idle_timeout_s = idle_timeout_s
    return server


def run_in_thread(
    host: str = "127.0.0.1",
    port: int = 55600,
    drop_prob: float = 0.0,
    s_per_m: float = 0.02,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    idle_timeout_s: Optional[float] = DEFAULT_IDLE_TIMEOUT_S,
) -> Tuple[_MockJsonServer, threading.Thread]:
    server = start_server(host, port, drop_prob, s_per_m,
                          frame_timeout_s, idle_timeout_s)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info("mock-json: listening on %s:%d", host, port)
    return server, t


def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Length-prefixed JSON test controller. Not a robot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55600)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("mock-json: binding %s exposes an unauthenticated "
                    "controller; loopback only", args.host)
    srv = start_server(args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("mock-json: shutting down")


if __name__ == "__main__":  # pragma: no cover
    _cli()
