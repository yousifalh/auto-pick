"""KUKA Ethernet KRL 3.1 client.

Consumes the planner's :class:`PickPlacePose` queue, serialises each
step into binary packets (see :mod:`execution.protocol`), and drives
the KUKA controller over a blocking TCP socket.

**The safety contract, stated exactly.** Every route out of a command —
including the handshake, including an exception the author did not
anticipate — either returns a status the caller can act on, or attempts
a Category-0 stop and raises :class:`RobotFault`. There are three kinds
of failure and they are handled differently on purpose:

* *transient* — a socket timeout, a CRC failure, a controller-reported
  ``CRC_ERROR`` or ``TIMEOUT``. Retried up to ``cfg.max_retries``; when
  retries are exhausted the client sends one E-stop packet and raises.
* *fatal* — anything else the socket or the packer throws
  (``struct.error`` from an out-of-range coordinate, ``ConnectionError``
  from a mid-frame close, ``ConnectionResetError``), plus a controller
  reporting ``VERSION_MISMATCH`` or ``UNSUPPORTED_COMMAND``. Retrying
  cannot help, so the E-stop is attempted immediately and the client
  raises. The E-stop is best-effort: on a dead socket it cannot be
  transmitted, and that is logged as CRITICAL rather than silently
  replacing the original error.
* *the controller says it is stopped* — a status of ``ESTOP``. The
  client latches: it closes the socket, refuses to reconnect, and
  raises :class:`RobotEstop`. A controller reporting a Category-0 stop
  must stop the host, not be counted as a failed placement.

What this module does **not** do, so nothing downstream assumes it:
there is no heartbeat and no watchdog (nothing ever sends
``OpCode.HEARTBEAT``), there is no sequence number, so a *late* reply is
mis-paired with the next command, and the retry loop re-sends motion
opcodes, which is not idempotent for ``PICK_AND_PLACE``. The retry
pause is a constant ``heartbeat_interval_ms``, not exponential backoff.

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
    COMMAND_LEN,
    STATUS_LEN,
    OpCode,
    pack_command,
    unpack_status,
)

log = get_logger("execution.kuka")

# The CRC-16/MODBUS polynomial this module's `crc16_modbus` implements.
# `configs/execution.yaml` states it too; the two are cross-checked in
# ExecutionConfig.from_dict rather than left to agree by luck.
_CRC_POLYNOMIAL = 0xA001


class RobotFault(RuntimeError):
    """A command ended with the arm in an unknown state.

    Subclasses :class:`RuntimeError` so callers written against the
    original ``RuntimeError`` contract keep working.
    """


class RobotEstop(RobotFault):
    """The robot is in a Category-0 stop.

    Raised when the controller reports ``ESTOP`` — including when it
    refuses the handshake because it is *already* stopped — and when a
    latched client is asked to command motion again. The client is
    single-use after this: clearing a Category-0 stop is a deliberate
    act at the controller, not something a host reconnect may do.
    """


# Controller-reported codes that mean "try that again": an integrity
# failure at the far end, or a controller-side timeout. Symmetric with
# the locally-detected CRC failures the client has always retried.
_RETRYABLE_STATUS = (RobotStatusCode.CRC_ERROR, RobotStatusCode.TIMEOUT)

# Controller-reported codes that mean "this build cannot talk to that
# build". Retrying re-sends the same unparseable frame.
_FATAL_STATUS = (
    RobotStatusCode.VERSION_MISMATCH,
    RobotStatusCode.UNSUPPORTED_COMMAND,
)

# Locally-detected failures worth another attempt. `socket.timeout` is
# `TimeoutError`; every ProtocolError is a ValueError. Note the ordering
# in the handlers below: this tuple is caught FIRST, and everything else
# — struct.error, ConnectionError, ConnectionResetError — falls through
# to the fatal path. struct.error is not a ValueError subclass, which is
# how an out-of-range coordinate used to escape without an E-stop.
_TRANSIENT = (socket.timeout, ValueError)


# ---------------------------------------------------- configuration ---

@dataclass
class ExecutionConfig:
    """Execution-layer knobs loaded from ``configs/execution.yaml``.

    Only fields something actually reads live here. ``approach_height_mm``
    and ``insert_height_mm`` used to be parsed, stored, unit-tested and
    read by nothing — editing them changed no behaviour anywhere in the
    system. Both are gone; the approach and insert depths are
    controller-side, hardcoded in ``krl_prog/routines.src`` (+60 mm
    approach, ``LIN place_pos`` insert), because the 16-byte frame has no
    field that could carry them. See :meth:`KukaClient.pick_and_place`.
    """

    host: str = "172.31.1.147"
    port: int = 54600
    handshake_timeout_ms: int = 2000
    command_timeout_ms: int = 5000
    heartbeat_interval_ms: int = 50
    max_retries: int = 3
    transport_height_mm: float = 80.0
    vacuum_level_percent: int = 80

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExecutionConfig":
        kuka = cfg.get("kuka", {}) or {}
        motion = cfg.get("motion", {}) or {}

        # The descriptive `kuka:` keys are checked rather than ignored.
        # A config file that states a 16-byte frame or polynomial 0xA001
        # is making a claim about this module, and a claim nothing reads
        # is a claim nothing keeps true.
        declared_len = kuka.get("command_length_bytes")
        if declared_len is not None and int(declared_len) != COMMAND_LEN:
            raise ValueError(
                f"kuka.command_length_bytes = {declared_len} but "
                f"execution.protocol packs {COMMAND_LEN}-byte frames")
        declared_poly = kuka.get("crc_polynomial")
        if declared_poly is not None and int(declared_poly) != _CRC_POLYNOMIAL:
            raise ValueError(
                f"kuka.crc_polynomial = 0x{int(declared_poly):04X} but "
                f"execution.protocol implements 0x{_CRC_POLYNOMIAL:04X}; "
                "editing this key does not change the CRC")
        declared_cat = kuka.get("stop_category")
        if declared_cat is not None and int(declared_cat) != 0:
            raise ValueError(
                f"kuka.stop_category = {declared_cat}; this client only "
                "implements the IEC 60204 Category-0 immediate stop")

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
            transport_height_mm=float(
                motion.get("transport_height_mm", cls.transport_height_mm),
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
        self._estopped = False

    @property
    def estopped(self) -> bool:
        """True once this client has stopped, or been told the robot is."""
        return self._estopped

    # ---- connection lifecycle -------------------------------------------

    def connect(self) -> None:
        """Open the channel and complete the handshake.

        Every failure route closes the socket before raising, so the
        canonical ``with KukaClient(cfg) as k:`` form cannot leak the
        descriptor — a refused handshake used to raise with the socket
        still open and ``__exit__`` never reached.
        """
        if self._estopped:
            raise RobotEstop(
                "this client is latched in E-stop and will not reconnect; "
                "clear the stop at the controller and build a new client")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.cfg.handshake_timeout_ms / 1000)
        try:
            sock.connect((self.cfg.host, self.cfg.port))
        except OSError as exc:
            sock.close()
            # Nothing has been commanded, so there is nothing to stop —
            # and no channel to stop it over.
            raise RobotFault(
                f"could not open the EthernetKRL channel to "
                f"{self.cfg.host}:{self.cfg.port}: {exc}") from exc
        self._sock = sock

        try:
            self._handshake()
        except BaseException:
            self.close()
            raise

    def _handshake(self) -> None:
        """Send HANDSHAKE and read the ack, with the same retry policy
        as every other command.

        ``handshake_timeout_ms`` bounds the wait for the ack here, which
        is what its name says and what it did not do: ``_recv_status``
        used to re-arm ``command_timeout_ms`` before reading a byte, so
        the knob bounded only the TCP connect.
        """
        timeout_s = self.cfg.handshake_timeout_ms / 1000
        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries):
            try:
                self._send(pack_command(OpCode.HANDSHAKE))
                ack = self._recv_status(timeout_s=timeout_s)
            except _TRANSIENT as exc:
                last_err = exc
                log.warning(
                    "EthernetKRL handshake error (attempt %d/%d): %s",
                    attempt + 1, self.cfg.max_retries, exc,
                )
                time.sleep(self.cfg.heartbeat_interval_ms / 1000)
                continue
            except Exception as exc:
                self._fatal(f"handshake failed: {exc!r}", exc)

            if ack.code in (RobotStatusCode.OK, RobotStatusCode.SUCCESS):
                log.info(
                    "KUKA handshake OK; at %.1f,%.1f,%.1f",
                    ack.current_pose.x_mm,
                    ack.current_pose.y_mm,
                    ack.current_pose.z_mm,
                )
                return

            if ack.code is RobotStatusCode.ESTOP:
                # The commonest refusal, and the one that used to raise a
                # bare string with the socket left open: the controller is
                # telling us it is ALREADY in a Category-0 stop.
                self._estopped = True
                raise RobotEstop(
                    "the controller refused the handshake because it is in "
                    "a Category-0 stop; clear it at the controller before "
                    "reconnecting")

            # Any other refusal: we do not have a channel we trust, and
            # we do not know what state the arm is in. The socket is
            # alive, so the stop can actually be transmitted.
            self._fatal(f"handshake refused (status={ack.code.name})")

        self._fatal(
            f"handshake did not complete in {self.cfg.max_retries} "
            f"attempts: {last_err}", last_err)

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

        The PICK_AND_PLACE opcode carries **one** coordinate triple, the
        pick target. The place *XY* is latched by the preceding MOVE_TO
        at transport height — the controller uses whatever (x, y) it was
        in when the subroutine begins.

        ``pose.place.z_mm`` is **not transmitted, and cannot be**: the
        16-byte frame has one Z field and the pick needs it. The insert
        depth is therefore a controller-side constant
        (``krl_prog/routines.src`` descends ``LIN place_pos`` and derives
        its transport height from it; the simulator uses
        ``mock_kuka_server._INSERT_Z_MM``). This is stated rather than
        papered over because the planner *does* compute a place Z
        (``PlannerConfig.place_insert_height_mm``) and a reader is
        entitled to know it stops here. Making it reach the wire needs a
        second Z field, i.e. a frame-layout change and a protocol
        version bump — deliberately not done, see
        docs/superpowers/specs/2026-08-12-fix-execution-safety.md.
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
        """IEC 60204 Category-0 stop — fire and forget, then disconnect.

        Latches: the client will not reconnect afterwards. Raises if the
        packet could not be transmitted; callers inside this module go
        through :meth:`_emergency_stop`, which swallows that.
        """
        self._estopped = True
        try:
            self._send(pack_command(OpCode.ESTOP))
        finally:
            self.close()

    # ---- low-level plumbing ---------------------------------------------

    def _emergency_stop(self, reason: str) -> None:
        """Best-effort Category-0 stop. Never raises.

        An E-stop that cannot be transmitted must not replace the error
        that prompted it — that swap is how the original cause used to
        get lost — but it must not be silent either, because "the link
        died mid-``PICK_AND_PLACE``" means the arm may be holding a cell
        with the vacuum on and nothing can tell it to let go.
        """
        log.critical("E-STOP: %s", reason)
        self._estopped = True
        try:
            self.estop()
        except Exception as exc:
            log.critical(
                "E-stop could NOT be transmitted (%s). The link is down "
                "and the arm's state is UNKNOWN — it may be mid-motion "
                "with the vacuum on.", exc,
            )
            self.close()

    def _fatal(self, reason: str, cause: Optional[BaseException] = None):
        """Stop the robot and raise. Never returns."""
        self._emergency_stop(reason)
        raise RobotFault(f"EthernetKRL failure: {reason}") from cause

    def _cmd_and_wait(
        self,
        op: OpCode,
        x: int = 0,
        y: int = 0,
        z: int = 0,
        aux: int = 0,
    ) -> RobotStatus:
        """Send one command and wait for the corresponding status packet.

        Retries transient failures up to ``cfg.max_retries``, then fires
        the E-stop and raises. Fatal failures skip the retries — see the
        module docstring for which is which. Returns only codes a caller
        can act on: OK, SUCCESS, PICK_FAILED, PLACE_FAILED.
        """
        if self._estopped:
            raise RobotEstop(
                f"refusing to send {op.name}: this client is latched in "
                "E-stop")

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                self._send(pack_command(op, x, y, z, aux))
                status = self._recv_status()
            except _TRANSIENT as exc:
                last_err = exc
                log.warning(
                    "EthernetKRL error (attempt %d/%d): %s",
                    attempt + 1, self.cfg.max_retries, exc,
                )
                time.sleep(self.cfg.heartbeat_interval_ms / 1000)
                continue
            except Exception as exc:
                # struct.error (coordinate out of int32/int16 range),
                # ConnectionError, ConnectionResetError, and anything
                # else unforeseen. Retrying cannot help; the socket may
                # still be alive, in which case the E-stop DOES go out.
                self._fatal(f"{op.name} failed: {exc!r}", exc)

            if status.code is RobotStatusCode.ESTOP:
                # The controller is stopped. So is this client, now.
                self._estopped = True
                self.close()
                raise RobotEstop(
                    f"the controller reported a Category-0 stop in reply to "
                    f"{op.name}; no further motion will be commanded")

            if status.code in _FATAL_STATUS:
                self._fatal(
                    f"the controller rejected {op.name} as "
                    f"{status.code.name}")

            if status.code in _RETRYABLE_STATUS:
                last_err = RobotFault(
                    f"controller reported {status.code.name}")
                log.warning(
                    "EthernetKRL error (attempt %d/%d): %s",
                    attempt + 1, self.cfg.max_retries, last_err,
                )
                time.sleep(self.cfg.heartbeat_interval_ms / 1000)
                continue

            return status

        log.error(
            "EthernetKRL giving up after %d retries", self.cfg.max_retries,
        )
        self._fatal(f"{op.name}: {last_err}", last_err)

    def _send(self, packet: bytes) -> None:
        if self._sock is None:
            raise RobotFault("Socket not connected")
        self._sock.sendall(packet)

    def _recv_status(self, timeout_s: Optional[float] = None) -> RobotStatus:
        """Read exactly one 16-byte status frame.

        ``timeout_s`` is a **deadline for the whole frame**, not a
        per-``recv`` timeout: a controller that trickles one byte at a
        time used to be accepted after 3 s at a 250 ms setting, which
        left the command deadline — the thing the whole bounded-response
        argument rests on — unenforced.
        """
        if self._sock is None:
            raise RobotFault("Socket not connected")

        if timeout_s is None:
            timeout_s = self.cfg.command_timeout_ms / 1000
        deadline = time.monotonic() + timeout_s

        buf = b""
        while len(buf) < STATUS_LEN:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout(
                    f"status frame incomplete after {timeout_s * 1000:.0f} "
                    f"ms ({len(buf)}/{STATUS_LEN} bytes)")
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(STATUS_LEN - len(buf))
            if not chunk:
                raise ConnectionError(
                    f"Robot closed connection ({len(buf)}/{STATUS_LEN} "
                    "bytes of a status frame received)")
            buf += chunk

        s = unpack_status(buf)
        try:
            code = RobotStatusCode(s["code"])
        except ValueError as exc:
            # Substituting TIMEOUT here made an unknown code from a
            # future firmware indistinguishable from an ordinary
            # placement failure. Raise instead: it is a transient
            # ValueError, so it retries and then escalates.
            raise ValueError(
                f"unknown status code {s['code']}") from exc
        return RobotStatus(
            code=code,
            current_pose=WorkspacePoint(s["x_mm"], s["y_mm"], s["z_mm"]),
            cycle_time_ms=float(s["cycle_ms"]),
        )


__all__ = ["ExecutionConfig", "KukaClient", "RobotEstop", "RobotFault"]
