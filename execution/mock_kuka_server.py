"""Software-only KUKA KR 6 R700 simulator.

Speaks the same 16-byte binary protocol as the real controller so the
full pipeline (perception → twin update → queue rebuild → command →
status) can be exercised without hardware. Used both by the unit tests
(spawned via :func:`run_in_thread`) and as a standalone process for
the software-only demo::

    python -m execution.mock_kuka_server

Random failures are configurable via ``drop_prob`` so tests can
exercise the planner's retry / rollback paths.
"""
from __future__ import annotations

import argparse
import math
import random
import socketserver
import threading
import time
from typing import Tuple

from common.logging import get_logger
from .protocol import (
    COMMAND_LEN,
    OpCode,
    pack_status,
    unpack_command,
)

log = get_logger("execution.mock_kuka")


# Status codes (mirror common.types.RobotStatusCode).
_OK = 0
_SUCCESS = 1
_PICK_FAILED = 2
_PLACE_FAILED = 3
_CRC_ERROR = 5
_ESTOP = 6


# ---------------------------------------------------- robot state ---

class _RobotState:
    """Mutable robot pose / vacuum state shared by a single server."""

    def __init__(self, drop_prob: float, ms_per_100mm: int) -> None:
        self.x = 0
        self.y = 0
        self.z = 200
        self.vacuum_on = False
        self.drop_prob = drop_prob
        self.ms_per_100mm = ms_per_100mm
        self.lock = threading.Lock()

    # ---- motions -------------------------------------------------------

    def move_to(self, x: int, y: int, z: int) -> int:
        """Simulate a PTP move. Returns the simulated cycle time in ms."""
        dx, dy, dz = x - self.x, y - self.y, z - self.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        with self.lock:
            self.x, self.y, self.z = x, y, z

        t_ms = int(dist / 100.0 * self.ms_per_100mm + 50)
        time.sleep(t_ms / 1000)
        return t_ms

    def pick_and_place(
        self,
        pick_x: int,
        pick_y: int,
        place_z: int,
        vacuum_pct: int,
    ) -> Tuple[int, int]:
        """Execute the PICK_AND_PLACE subroutine (mirrors routines.src).

        Returns ``(cycle_ms, status_code)``. The caller sets the place
        target via a preceding ``MOVE_TO`` at transport height, so at
        the moment this method fires ``self.x`` / ``self.y`` already
        hold the place coordinates.
        """
        place_x, place_y = self.x, self.y
        total = 0

        # 1. Approach at transport height.
        total += self.move_to(pick_x, pick_y, max(self.z, 80))
        # 2. Descend onto the pick target.
        total += self.move_to(pick_x, pick_y, 5)
        # 3. Vacuum on.
        self.vacuum_on = True
        time.sleep(0.05)
        total += 50
        if random.random() < self.drop_prob:
            self.vacuum_on = False
            return total, _PICK_FAILED
        # 4. Lift.
        total += self.move_to(pick_x, pick_y, 80)
        # 5. Insert into the cartridge cell.
        total += self.move_to(place_x, place_y, place_z)
        # 6. Vacuum off.
        self.vacuum_on = False
        time.sleep(0.03)
        total += 30
        # 7. Retract.
        total += self.move_to(place_x, place_y, 80)

        if random.random() < self.drop_prob / 2:
            return total, _PLACE_FAILED
        return total, _SUCCESS


# --------------------------------------------------- TCP handler ---

class _Handler(socketserver.BaseRequestHandler):
    """One TCP handler per connected client."""

    def handle(self) -> None:
        robot: _RobotState = self.server.robot  # type: ignore[attr-defined]
        log.info("mock-kuka: client connected from %s", self.client_address)

        try:
            while True:
                buf = self._recv_n(COMMAND_LEN)
                if not buf:
                    return

                try:
                    cmd = unpack_command(buf)
                except ValueError as exc:
                    log.warning("mock-kuka: bad packet (%s) — CRC_ERROR", exc)
                    self._send_status(_CRC_ERROR, cycle_ms=0, robot=robot)
                    continue

                code, cycle_ms = self._dispatch(cmd, robot)

                # ESTOP short-circuits: send the status, then disconnect.
                if cmd.op == OpCode.ESTOP:
                    self._send_status(code, cycle_ms, robot)
                    return

                self._send_status(code, cycle_ms, robot)
        except (ConnectionError, OSError) as exc:
            log.info("mock-kuka: client disconnected (%s)", exc)

    # ---- dispatch ------------------------------------------------------

    def _dispatch(
        self, cmd, robot: _RobotState,
    ) -> Tuple[int, int]:
        """Execute ``cmd`` against the robot. Returns ``(code, cycle_ms)``."""
        if cmd.op == OpCode.HANDSHAKE:
            return _OK, 0
        if cmd.op == OpCode.MOVE_TO:
            return _SUCCESS, robot.move_to(cmd.x_mm, cmd.y_mm, cmd.z_mm)
        if cmd.op == OpCode.VACUUM_ON:
            robot.vacuum_on = True
            return _SUCCESS, 0
        if cmd.op == OpCode.VACUUM_OFF:
            robot.vacuum_on = False
            return _SUCCESS, 0
        if cmd.op == OpCode.PICK_AND_PLACE:
            cycle_ms, code = robot.pick_and_place(
                cmd.x_mm, cmd.y_mm, cmd.z_mm, cmd.aux,
            )
            return code, cycle_ms
        if cmd.op == OpCode.ESTOP:
            return _ESTOP, 0
        if cmd.op == OpCode.HEARTBEAT:
            return _OK, 0
        return _OK, 0  # NOOP / unknown

    # ---- I/O -----------------------------------------------------------

    def _recv_n(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.request.recv(n - len(buf))
            except OSError:
                return b""
            if not chunk:
                return b""
            buf += chunk
        return buf

    def _send_status(
        self, code: int, cycle_ms: int, robot: _RobotState,
    ) -> None:
        self.request.sendall(pack_status(
            code=code,
            x_mm=robot.x, y_mm=robot.y, z_mm=robot.z,
            cycle_ms=cycle_ms,
        ))


# ----------------------------------------------------- server shell ---

class _MockServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(
    host: str = "127.0.0.1",
    port: int = 54600,
    drop_prob: float = 0.02,
    ms_per_100mm: int = 180,
) -> _MockServer:
    """Build (but don't start) a new mock server bound to ``(host, port)``."""
    server = _MockServer((host, port), _Handler)
    server.robot = _RobotState(drop_prob, ms_per_100mm)
    return server


def run_in_thread(
    host: str = "127.0.0.1",
    port: int = 54600,
    drop_prob: float = 0.02,
    ms_per_100mm: int = 50,
) -> Tuple[_MockServer, threading.Thread]:
    """Build the server, start it in a daemon thread, return both."""
    server = start_server(host, port, drop_prob, ms_per_100mm)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info("mock-kuka: listening on %s:%d", host, port)
    return server, t


# ----------------------------------------------------------- CLI ---

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Software-only KUKA simulator speaking the binary protocol.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54600)
    parser.add_argument("--drop-prob", type=float, default=0.02)
    args = parser.parse_args()

    srv = start_server(args.host, args.port, drop_prob=args.drop_prob)
    log.info(
        "mock-kuka: listening on %s:%d — Ctrl-C to stop",
        args.host, args.port,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("mock-kuka: shutting down")


if __name__ == "__main__":  # pragma: no cover
    _cli()
