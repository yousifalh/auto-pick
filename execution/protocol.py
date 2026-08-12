"""Binary wire format for the KUKA Ethernet KRL 3.1 channel.

Every command is a fixed-size 16-byte packet laid out as follows::

    offset  bytes  field       notes
    0       1      version     protocol version (= 0x01)
    1       1      op          opcode, see OpCode enum below
    2       4      x_mm_i32    signed int, big-endian, millimetres
    6       4      y_mm_i32    signed int, big-endian, millimetres
    10      2      z_mm_i16    signed int, big-endian, millimetres
    12      2      aux_u16     opcode-specific (e.g. vacuum percentage)
    14      2      crc16       CRC-16/MODBUS of bytes 0..13 (little-endian)
    ---
    total:  16 bytes

Status replies reuse the same layout: byte 1 is the status code
(see :class:`common.types.RobotStatusCode`), and ``aux_u16`` holds
the cycle time in milliseconds so the host can log control-loop
latency.

This framing is **modelled on** the KUKA EthernetKRL 3.1 specification
and **augmented with** a CRC-16/MODBUS trailer (polynomial 0xA001,
initial 0xFFFF, no output XOR) — the phrasing FDR §7.1 uses, and the
accurate one. EthernetKRL's own transport is XML, and this project's own
``krl_prog/laptop-comm.xml`` declares an XML payload schema with **no
checksum element**, so the CRC here is not "the standard integrity check
on the XML transport": it is a binary framing this repository defines,
and the CRC is part of that definition rather than of anything KUKA
publishes.

What the CRC does and does not protect, since a checksum invites
assumptions:

* it covers **bytes 0..13 of one frame** — every field including the
  version byte and the opcode — so a bit flip anywhere in a packet is
  caught before any field is interpreted (see :func:`unpack_command`,
  which verifies the CRC first and only then reads version and opcode);
* it does **not** delimit or resynchronise the stream. There is no
  length prefix and no sync word, so framing is positional: readers take
  exactly 16 bytes. A stream that loses or gains a byte does not
  resynchronise, it fails CRC on every subsequent frame — a real and
  observed behaviour (audit F §1.6), not a hypothetical;
* it is an **integrity** check, not an authentication or authorisation
  one. Anyone who can reach the socket can emit a correctly-CRC'd
  command; the channel is unauthenticated and plaintext by design, and
  is meant for a point-to-point robot-cell link.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum


# ------------------------------------------------------ parse errors ---

class ProtocolError(ValueError):
    """Base class for every integrity failure ``unpack_*`` can report.

    It subclasses :class:`ValueError` so that callers written against
    the original contract — ``except ValueError`` — keep working
    unchanged. The subclasses exist because the three failures are not
    the same fault and a controller must not answer them identically: a
    CRC mismatch is a line fault, an unsupported version is a firmware
    mismatch, and an unknown opcode is a command this end cannot
    execute. Collapsing all three into ``CRC_ERROR`` (which
    :mod:`execution.mock_kuka_server` used to do) leaves the host unable
    to tell corruption from a protocol mismatch, and the two want
    opposite responses — retry the line, versus stop and fix the build.
    """


class FrameLengthError(ProtocolError):
    """The buffer handed to ``unpack_*`` was not exactly 16 bytes."""


class CrcError(ProtocolError):
    """The CRC-16 trailer does not match the body."""


class VersionMismatch(ProtocolError):
    """The version byte names a protocol revision this end cannot parse."""


class UnsupportedOpCode(ProtocolError):
    """A well-formed frame carrying an opcode this end does not implement."""


PROTOCOL_VERSION = 0x01
COMMAND_LEN = 16
STATUS_LEN = 16

# The header / payload layout as a single struct format string: it's
# used for both commands and status packets because they share shape.
_BODY_FMT = ">BBiihH"
_BODY_LEN = 14
_CRC_FMT = "<H"


class OpCode(IntEnum):
    """Every opcode understood by the KRL receive loop."""

    NOOP = 0x00
    MOVE_TO = 0x01
    VACUUM_ON = 0x02
    VACUUM_OFF = 0x03
    PICK_AND_PLACE = 0x04
    HEARTBEAT = 0x05
    ESTOP = 0x06
    HANDSHAKE = 0x07


# ------------------------------------------------------------- CRC ---

def crc16_modbus(data: bytes) -> int:
    """Compute CRC-16/MODBUS of ``data``.

    The standard MODBUS variant — poly 0xA001 (reflected 0x8005), init
    0xFFFF, reflected in and out, no output XOR — chosen for this
    project's binary framing because it is ubiquitous and independently
    checkable, NOT because EthernetKRL's XML transport uses it (it
    carries no checksum; see the module docstring).
    ``tests/test_protocol.py`` pins it to the published check value.
    """
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# ---------------------------------------------------------- command --

@dataclass(frozen=True)
class Command:
    """Parsed command packet."""

    op: OpCode
    x_mm: int = 0
    y_mm: int = 0
    z_mm: int = 0
    aux: int = 0


def pack_command(
    op: OpCode,
    x_mm: int = 0,
    y_mm: int = 0,
    z_mm: int = 0,
    aux: int = 0,
) -> bytes:
    """Pack a 16-byte command packet including trailing CRC-16."""
    body = struct.pack(
        _BODY_FMT,
        PROTOCOL_VERSION,
        int(op),
        int(x_mm), int(y_mm), int(z_mm),
        int(aux) & 0xFFFF,
    )
    assert len(body) == _BODY_LEN, f"body len = {len(body)}"
    return body + struct.pack(_CRC_FMT, crc16_modbus(body))


def unpack_command(buf: bytes) -> Command:
    """Parse a command packet. Raises ValueError on any integrity failure."""
    if len(buf) != COMMAND_LEN:
        raise FrameLengthError(
            f"command must be {COMMAND_LEN} bytes, got {len(buf)}"
        )

    body = buf[:_BODY_LEN]
    (crc_rx,) = struct.unpack(_CRC_FMT, buf[_BODY_LEN:])
    crc_calc = crc16_modbus(body)
    if crc_rx != crc_calc:
        raise CrcError(
            f"CRC mismatch: rx=0x{crc_rx:04X} calc=0x{crc_calc:04X}"
        )

    version, op_byte, x, y, z, aux = struct.unpack(_BODY_FMT, body)
    if version != PROTOCOL_VERSION:
        raise VersionMismatch(
            f"unsupported protocol version 0x{version:02X}")
    try:
        op = OpCode(op_byte)
    except ValueError as exc:
        raise UnsupportedOpCode(f"unknown opcode 0x{op_byte:02X}") from exc
    return Command(op, x, y, z, aux)


# ---------------------------------------------------------- status --

def pack_status(
    code: int,
    x_mm: int,
    y_mm: int,
    z_mm: int,
    cycle_ms: int,
) -> bytes:
    """Pack a 16-byte status packet with CRC-16 trailer."""
    body = struct.pack(
        _BODY_FMT,
        PROTOCOL_VERSION,
        int(code) & 0xFF,
        int(x_mm), int(y_mm), int(z_mm),
        int(cycle_ms) & 0xFFFF,
    )
    return body + struct.pack(_CRC_FMT, crc16_modbus(body))


def unpack_status(buf: bytes) -> dict:
    """Parse a status packet. Raises ValueError on length or CRC failure."""
    if len(buf) != STATUS_LEN:
        raise FrameLengthError(
            f"status must be {STATUS_LEN} bytes, got {len(buf)}")

    body = buf[:_BODY_LEN]
    (crc_rx,) = struct.unpack(_CRC_FMT, buf[_BODY_LEN:])
    if crc_rx != crc16_modbus(body):
        raise CrcError("CRC mismatch")

    _version, code, x, y, z, cycle_ms = struct.unpack(_BODY_FMT, body)
    return {
        "code": code,
        "x_mm": x,
        "y_mm": y,
        "z_mm": z,
        "cycle_ms": cycle_ms,
    }


__all__ = [
    "COMMAND_LEN",
    "Command",
    "CrcError",
    "FrameLengthError",
    "OpCode",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "STATUS_LEN",
    "UnsupportedOpCode",
    "VersionMismatch",
    "crc16_modbus",
    "pack_command",
    "pack_status",
    "unpack_command",
    "unpack_status",
]
