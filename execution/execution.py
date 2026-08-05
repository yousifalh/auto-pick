"""KUKA Ethernet KRL 3.1 client.

Consumes the planner's :class:`PickPlacePose` queue, serialises each
step into binary packets (see :mod:`execution.protocol`), and drives
the KUKA controller over a blocking TCP socket. CRC failures and
socket timeouts trigger a retry policy; once retries are exhausted,
the client sends a single E-stop packet and raises — this matches the
Category-0 immediate-stop requirement in PPR §7.3 (R4).

The high-level surface is three methods — :meth:`KukaClient.move_to`,
:meth:`KukaClient.vacuum`, and :meth:`KukaClient.pick_and_place` —
which return the :class:`RobotStatus` reported by the controller.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Optional

from common.logging import get_logger
from common.types import (
    PickPlacePose,
    RobotStatus,
    RobotStatusCode,
    WorkspacePoint,
)
from .protocol import (
    STATUS_LEN,
    OpCode,
    pack_command,
    unpack_status,
)

log = get_logger("execution.kuka")


# ---------------------------------------------------- configuration ---

@dataclass
class ExecutionConfig:
    """Execution-layer knobs loaded from ``configs/execution.yaml``."""

    host: str = "172.31.1.147"
    port: int = 54600
    handshake_timeout_ms: int = 2000
    command_timeout_ms: int = 5000
    heartbeat_interval_ms: int = 50
    max_retries: int = 3
    approach_height_mm: float = 60.0
    transport_height_mm: float = 80.0
    insert_height_mm: float = 2.0
    vacuum_level_percent: int = 80

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExecutionConfig":
        kuka = cfg.get("kuka", {}) or {}
        motion = cfg.get("motion", {}) or {}
        return cls(
            host=kuka.get("host", cls.host),
            port=int(kuka.get("port", cls.port)),
            handshake_timeout_ms=int(
                kuka.get("handshake_timeout_ms", cls.handshake_timeout_ms),
            ),
            command_timeout_ms=int(
                kuka.get("command_timeout_ms", cls.command_timeout_ms),
            ),
            heartbeat_interval_ms=int(
                kuka.get("heartbeat_interval_ms", cls.heartbeat_interval_ms),
            ),
            max_retries=int(kuka.get("max_retries", cls.max_retries)),
            approach_height_mm=float(
                motion.get("approach_height_mm", cls.approach_height_mm),
            ),
            transport_height_mm=float(
                motion.get("transport_height_mm", cls.transport_height_mm),
            ),
            insert_height_mm=float(
                motion.get("insert_height_mm", cls.insert_height_mm),
            ),
            vacuum_level_percent=int(
                motion.get("vacuum_level_percent", cls.vacuum_level_percent),
            ),
        )


# ------------------------------------------------------------ client ---

class KukaClient:
    """Blocking TCP client speaking the 16-byte command/status protocol."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None

    # ---- connection lifecycle -------------------------------------------

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.cfg.handshake_timeout_ms / 1000)
        sock.connect((self.cfg.host, self.cfg.port))
        self._sock = sock

        self._send(pack_command(OpCode.HANDSHAKE))
        ack = self._recv_status()
        if ack.code not in (RobotStatusCode.OK, RobotStatusCode.SUCCESS):
            raise RuntimeError(
                f"Handshake refused (status={ack.code.name})",
            )
        log.info(
            "KUKA handshake OK; at %.1f,%.1f,%.1f",
            ack.current_pose.x_mm,
            ack.current_pose.y_mm,
            ack.current_pose.z_mm,
        )

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def __enter__(self) -> "KukaClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- high-level commands --------------------------------------------

    def move_to(self, target: WorkspacePoint) -> RobotStatus:
        return self._cmd_and_wait(
            OpCode.MOVE_TO,
            int(target.x_mm), int(target.y_mm), int(target.z_mm),
        )

    def vacuum(self, on: bool) -> RobotStatus:
        op = OpCode.VACUUM_ON if on else OpCode.VACUUM_OFF
        aux = self.cfg.vacuum_level_percent if on else 0
        return self._cmd_and_wait(op, aux=aux)

    def pick_and_place(self, pose: PickPlacePose) -> RobotStatus:
        """Run one full pick-and-place cycle on the robot.

        The PICK_AND_PLACE opcode only carries the pick coordinates.
        The place target is latched by a preceding MOVE_TO at
        transport height — the controller uses whatever (x, y) position
        it was in at the moment the subroutine begins.
        """
        self._cmd_and_wait(
            OpCode.MOVE_TO,
            int(pose.place.x_mm),
            int(pose.place.y_mm),
            int(self.cfg.transport_height_mm),
        )
        return self._cmd_and_wait(
            OpCode.PICK_AND_PLACE,
            int(pose.pick.x_mm),
            int(pose.pick.y_mm),
            int(pose.pick.z_mm),
            aux=self.cfg.vacuum_level_percent,
        )

    def estop(self) -> None:
        """IEC 60204 Category-0 stop — fire and forget, then disconnect."""
        try:
            self._send(pack_command(OpCode.ESTOP))
        finally:
            self.close()

    # ---- low-level plumbing ---------------------------------------------

    def _cmd_and_wait(
        self,
        op: OpCode,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        aux: int = 0,
    ) -> RobotStatus:
        """Send one command and wait for the corresponding status packet.

        Retries on CRC errors and socket timeouts up to
        ``cfg.max_retries``. If retries are exhausted, fires the E-stop
        and raises.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                self._send(pack_command(op, x, y, z, aux))
                return self._recv_status()
            except (socket.timeout, ValueError) as exc:
                last_err = exc
                log.warning(
                    "EthernetKRL error (attempt %d/%d): %s",
                    attempt + 1, self.cfg.max_retries, exc,
                )
                time.sleep(self.cfg.heartbeat_interval_ms / 1000)

        log.error(
            "EthernetKRL giving up after %d retries", self.cfg.max_retries,
        )
        self.estop()
        raise RuntimeError(f"EthernetKRL failure: {last_err}")

    def _send(self, packet: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("Socket not connected")
        self._sock.sendall(packet)

    def _recv_status(self) -> RobotStatus:
        if self._sock is None:
            raise RuntimeError("Socket not connected")

        self._sock.settimeout(self.cfg.command_timeout_ms / 1000)
        buf = b""
        while len(buf) < STATUS_LEN:
            chunk = self._sock.recv(STATUS_LEN - len(buf))
            if not chunk:
                raise ConnectionError("Robot closed connection")
            buf += chunk

        s = unpack_status(buf)
        try:
            code = RobotStatusCode(s["code"])
        except ValueError:
            code = RobotStatusCode.TIMEOUT
        return RobotStatus(
            code=code,
            current_pose=WorkspacePoint(s["x_mm"], s["y_mm"], s["z_mm"]),
            cycle_time_ms=float(s["cycle_ms"]),
        )


__all__ = ["ExecutionConfig", "KukaClient"]
