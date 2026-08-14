"""Software-only KUKA KR 6 R700 simulator.

Speaks the same 16-byte binary protocol as the real controller so the
full pipeline (perception → twin update → queue rebuild → command →
status) can be exercised without hardware. Used both by the unit tests
(spawned via :func:`run_in_thread`) and as a standalone process for
the software-only demo::

    python -m execution.mock_kuka_server

Random failures are configurable via ``drop_prob`` so tests can
exercise the planner's retry / rollback paths.

**What this simulator does and does not model.** It is a faithful model
of the *wire format* and a deliberately partial model of the
*controller*. It enforces four things a real KRC enforces, because
without them the client's error handling is untestable and every test
that exercises escalation ends with a simulated robot still free to
move:

* a **workspace envelope** — the KR 6 R700's 706 mm reach and the table
  it stands on. An out-of-envelope target is a software-limit fault and
  latches the stop, as it would on the real machine;
* a **latching stop** — once stopped, every command including
  ``HANDSHAKE`` answers ``ESTOP``, the vacuum is dropped, and no
  reconnect clears it. Clearing a Category-0 stop is a deliberate act at
  the controller, and this simulator offers no way to do it. Note the
  asymmetry the names now carry: the host sends ``OpCode.HALT``, an
  in-band request to stop moving, and what latches here and comes back
  as ``RobotStatusCode.ESTOP`` is the *controller's* safety state;
* **socket timeouts** — a half frame no longer pins a handler thread
  forever;
* **distinct fault codes** — an unknown opcode and a bad protocol
  version are no longer both reported as ``CRC_ERROR``.

It still does **not** model: drive-enable / operating-mode gates, a
velocity cap (the frame carries no velocity — ``routines.src`` sets
``$VEL.CP`` controller-side), analog-output range checks on the vacuum
level, single-channel exclusivity (two clients share one robot), or a
heartbeat watchdog. Nothing in this repository sends
``OpCode.HEARTBEAT``.
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import socketserver
import threading
import time
from typing import Optional, Tuple

from common.logging import get_logger
from .execution import KUKA_CONTROLLER_INSERT_Z_MM
from .protocol import (
    COMMAND_LEN,
    OpCode,
    UnsupportedOpCode,
    VersionMismatch,
    pack_status,
    unpack_command,
)

log = get_logger("execution.mock_kuka")


# Status codes (mirror common.types.RobotStatusCode).
_OK = 0
_SUCCESS = 1
_PICK_FAILED = 2
_PLACE_FAILED = 3
_TIMEOUT = 4
_CRC_ERROR = 5
_ESTOP = 6
_UNSUPPORTED_COMMAND = 7
_VERSION_MISMATCH = 8

# ------------------------------------------------- modelled machine ---
#
# KR 6 R700 sixx: 706 mm maximum reach. These are properties of the
# robot being simulated, not tuning knobs, which is why they are module
# constants rather than config keys — `configs/execution.yaml` cannot
# make the arm longer.
REACH_MM = 706.0
Z_MIN_MM = -20.0          # the table the cartridges sit on
Z_MAX_MM = 706.0

# The place descent depth. The 16-byte frame has no place-Z field, so
# the insert depth is controller-side — exactly as krl_prog/routines.src
# treats it, descending LIN onto place_pos and deriving its transport
# height from it.
#
# It is imported rather than repeated. The depth used to be written out
# in three places that could not be checked against each other: this
# constant, routines.src's `LIN place_pos`, and the planner's
# `place_insert_height_mm`. Two of the three are now one value, and
# execution.task.PickPlaceTask compares the third against it out loud.
_INSERT_Z_MM = int(KUKA_CONTROLLER_INSERT_Z_MM)

# A grasp happens near the table. `routines.src` derives its approach as
# pick_pos.Z + 60, so the Z on the wire is the GRASP pose, not the
# approach. Anything much above this band means the tool stops short of
# the part and the vacuum sensor would report no grasp — which this
# simulator cannot detect (it models no parts), so it says so once
# instead of returning a confident SUCCESS.
_GRASP_BAND_MM = 25.0

# Half a frame must not pin a handler thread forever, but an IDLE
# connection between commands is normal (the host is busy running
# perception), so the two are bounded separately.
DEFAULT_FRAME_TIMEOUT_S = 2.0
DEFAULT_IDLE_TIMEOUT_S = 120.0


class WorkspaceViolation(Exception):
    """A commanded pose lies outside the modelled robot's envelope."""


# ---------------------------------------------------- robot state ---

class _RobotState:
    """Mutable robot pose / vacuum state shared by a single server."""

    def __init__(self, drop_prob: float, ms_per_100mm: int) -> None:
        self.x = 0
        self.y = 0
        self.z = 200
        self.vacuum_on = False
        self.estopped = False
        self.estop_reason = ""
        self._grasp_warned = False
        self.drop_prob = drop_prob
        self.ms_per_100mm = ms_per_100mm
        self.lock = threading.Lock()

    # ---- safety --------------------------------------------------------

    @staticmethod
    def in_envelope(x: float, y: float, z: float) -> bool:
        """Is ``(x, y, z)`` reachable, and above the table?"""
        if not (Z_MIN_MM <= z <= Z_MAX_MM):
            return False
        return math.sqrt(x * x + y * y + z * z) <= REACH_MM

    def latch_estop(self, reason: str) -> None:
        """Enter a Category-0 stop. Idempotent, and never cleared.

        Drives off means vacuum off: a real Cat-0 stop removes power,
        and a simulator that kept holding the cell would make the
        client's post-stop reasoning wrong in the safe-looking
        direction.
        """
        with self.lock:
            if self.estopped:
                return
            self.estopped = True
            self.estop_reason = reason
            self.vacuum_on = False
        log.critical("mock-kuka: LATCHED Category-0 STOP — %s", reason)

    # ---- motions -------------------------------------------------------

    def move_to(self, x: int, y: int, z: int) -> int:
        """Simulate a PTP move. Returns the simulated cycle time in ms.

        Raises :class:`WorkspaceViolation` before moving or sleeping if
        the target is unreachable — which is also what stops a single
        well-formed ``MOVE_TO(2**31-1, ...)`` from wedging the handler
        thread in a 63-day ``time.sleep``.
        """
        if not self.in_envelope(x, y, z):
            raise WorkspaceViolation(
                f"target ({x}, {y}, {z}) is outside the KR 6 R700 envelope "
                f"(reach {REACH_MM:.0f} mm, z in "
                f"[{Z_MIN_MM:.0f}, {Z_MAX_MM:.0f}])")

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
        pick_z: int,
        vacuum_pct: int,
    ) -> Tuple[int, int]:
        """Execute the PICK_AND_PLACE subroutine (mirrors routines.src).

        Returns ``(cycle_ms, status_code)``. The caller sets the place
        target via a preceding ``MOVE_TO`` at transport height, so at
        the moment this method fires ``self.x`` / ``self.y`` already
        hold the place coordinates.

        ``pick_z`` is the third coordinate off the wire and it is the
        **pick descent height**, which is what the client sends. This
        parameter used to be named ``place_z`` and was used as the
        insert depth, so the commanded pick height was silently
        reinterpreted as a place height and the cycle still reported
        SUCCESS. The insert depth is ``_INSERT_Z_MM``, controller-side,
        because the frame cannot carry one.
        """
        place_x, place_y = self.x, self.y
        total = 0

        if pick_z > _GRASP_BAND_MM and not self._grasp_warned:
            self._grasp_warned = True
            log.warning(
                "mock-kuka: commanded pick z = %d mm is above the %.0f mm "
                "grasp band. routines.src treats the wire Z as the GRASP "
                "pose and derives the approach as Z+60; a host sending its "
                "APPROACH height here stops the tool short of the part. "
                "This simulator models no parts, so it cannot fail the "
                "grasp — on real hardware $IN[10] would and the cycle "
                "would return PICK_FAILED.",
                pick_z, _GRASP_BAND_MM,
            )

        # 1. Approach at transport height.
        total += self.move_to(pick_x, pick_y, max(self.z, 80))
        # 2. Descend onto the pick target.
        total += self.move_to(pick_x, pick_y, pick_z)
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
        total += self.move_to(place_x, place_y, _INSERT_Z_MM)
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
                except VersionMismatch as exc:
                    log.warning("mock-kuka: %s — VERSION_MISMATCH", exc)
                    self._send_status(_VERSION_MISMATCH, 0, robot)
                    continue
                except UnsupportedOpCode as exc:
                    log.warning("mock-kuka: %s — UNSUPPORTED_COMMAND", exc)
                    self._send_status(_UNSUPPORTED_COMMAND, 0, robot)
                    continue
                except ValueError as exc:
                    # Length or CRC: a line fault, and the only one that
                    # is honestly a CRC_ERROR.
                    log.warning("mock-kuka: bad packet (%s) — CRC_ERROR", exc)
                    self._send_status(_CRC_ERROR, cycle_ms=0, robot=robot)
                    continue

                code, cycle_ms = self._dispatch(cmd, robot)

                # HALT short-circuits: send the status, then disconnect.
                if cmd.op == OpCode.HALT:
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
        if robot.estopped:
            # Latched. Answering HANDSHAKE with ESTOP is the case the
            # client's connect() path exists for: the controller telling
            # a fresh connection that it is already stopped. Before this
            # latch existed, a client could reconnect after an
            # escalation and command motion immediately.
            return _ESTOP, 0

        if cmd.op == OpCode.HALT:
            robot.latch_estop("halt requested by the host")
            return _ESTOP, 0

        if cmd.op == OpCode.HANDSHAKE:
            return _OK, 0
        if cmd.op == OpCode.MOVE_TO:
            try:
                return _SUCCESS, robot.move_to(cmd.x_mm, cmd.y_mm, cmd.z_mm)
            except WorkspaceViolation as exc:
                robot.latch_estop(f"software limit: {exc}")
                return _ESTOP, 0
        if cmd.op == OpCode.VACUUM_ON:
            robot.vacuum_on = True
            return _SUCCESS, 0
        if cmd.op == OpCode.VACUUM_OFF:
            robot.vacuum_on = False
            return _SUCCESS, 0
        if cmd.op == OpCode.PICK_AND_PLACE:
            try:
                cycle_ms, code = robot.pick_and_place(
                    cmd.x_mm, cmd.y_mm, cmd.z_mm, cmd.aux,
                )
            except WorkspaceViolation as exc:
                robot.latch_estop(f"software limit: {exc}")
                return _ESTOP, 0
            return code, cycle_ms
        if cmd.op == OpCode.HEARTBEAT:
            return _OK, 0
        return _OK, 0  # NOOP

    # ---- I/O -----------------------------------------------------------

    def _recv_n(self, n: int) -> bytes:
        """Read exactly ``n`` bytes, or return ``b""`` and drop the link.

        Two different timeouts, because a connection sitting idle between
        commands is normal and a connection stuck half-way through a
        frame is not: the host may take seconds to run perception, but a
        controller that has seen 8 bytes of a 16-byte frame is looking at
        a desynchronised stream and should close the channel rather than
        block a thread on it.
        """
        server = self.server  # type: ignore[attr-defined]
        buf = b""
        while len(buf) < n:
            timeout = (server.frame_timeout_s if buf
                       else server.idle_timeout_s)
            try:
                self.request.settimeout(timeout)
                chunk = self.request.recv(n - len(buf))
            except socket.timeout:
                if buf:
                    log.warning(
                        "mock-kuka: partial frame (%d/%d bytes) timed out "
                        "after %.1fs — closing the channel",
                        len(buf), n, timeout,
                    )
                else:
                    log.info(
                        "mock-kuka: idle for %.0fs — closing the channel",
                        timeout,
                    )
                return b""
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
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    idle_timeout_s: Optional[float] = DEFAULT_IDLE_TIMEOUT_S,
) -> _MockServer:
    """Build (but don't start) a new mock server bound to ``(host, port)``."""
    server = _MockServer((host, port), _Handler)
    server.robot = _RobotState(drop_prob, ms_per_100mm)
    server.frame_timeout_s = frame_timeout_s
    server.idle_timeout_s = idle_timeout_s
    return server


def run_in_thread(
    host: str = "127.0.0.1",
    port: int = 54600,
    drop_prob: float = 0.02,
    ms_per_100mm: int = 50,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    idle_timeout_s: Optional[float] = DEFAULT_IDLE_TIMEOUT_S,
) -> Tuple[_MockServer, threading.Thread]:
    """Build the server, start it in a daemon thread, return both."""
    server = start_server(host, port, drop_prob, ms_per_100mm,
                          frame_timeout_s, idle_timeout_s)
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

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "mock-kuka: binding %s exposes an unauthenticated robot "
            "controller to the network; it has no access control of any "
            "kind and is intended for loopback only", args.host,
        )

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
