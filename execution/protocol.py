"""Binary wire format for the KUKA Ethernet KRL 3.1 channel.

Every command is a fixed-size 16-byte packet laid out as follows::

    offset  bytes  field       notes
    0       1      version     protocol version (= 0x01)
    1       1      op          opcode, see OpCode enum below
    2       4      x_mm_i32    signed int32, big-endian, millimetres
    6       4      y_mm_i32    signed int32, big-endian, millimetres
    10      2      z_mm_i16    signed **int16**, big-endian, millimetres
    12      2      aux_u16     DIRECTIONAL, see below
    14      2      crc16       CRC-16/MODBUS of bytes 0..13 (little-endian)
    ---
    total:  16 bytes

Status replies reuse the same layout: byte 1 is the status code
(see :class:`common.types.RobotStatusCode`).

**Two asymmetries in this layout that are easy to read past.**

*z is int16 while x and y are int32.* The struct format is
``">BBiihH"`` — ``i``, ``i``, ``h``. So x and y accept +/-2**31 and z
accepts only +/-32 767 mm, and a coordinate between those two limits is
rejected on one axis and accepted on the other two. ``pack_command``
raises :class:`CoordinateOutOfRange` naming the axis and its actual
limit rather than letting ``struct`` report ``'h' format requires
-32768 <= number <= 32767`` with no indication of which field it meant.
:class:`CoordinateOutOfRange` subclasses ``struct.error`` **on purpose**:
``struct.error`` is not a ``ValueError``, so it falls through the
driver's transient tuple to the fatal path and fires a halt, which is
exactly the handling an unrepresentable coordinate needs. Making it a
``ValueError`` would silently demote it to "retry the same impossible
frame three times".

*``aux_u16`` carries two different fields, one per direction.* Outbound
it is the vacuum percentage; inbound it is the cycle time in
milliseconds. One field, two meanings, and no bit on the wire says
which — the direction is the only discriminator. The inbound meaning
also does not fit: a cycle longer than 65 535 ms (65.5 s) has no
representation, so ``pack_status`` **saturates** at 0xFFFF rather than
wrapping. It used to mask with ``& 0xFFFF``, which turned a 70-second
cycle into 4.5 seconds and fed that into the latency statistics
``main.py`` prints. Saturation is still lossy, but it is lossy in a
direction a reader can detect: ``unpack_status`` reports
``cycle_ms_saturated``.

This framing targets EthernetKRL's **fixed-length binary mode**
(``BinaryFixed``) and **augments it with** a CRC-16/MODBUS trailer
(polynomial 0xA001, initial 0xFFFF, no output XOR).

**Correction, 2026-08-14.** This docstring used to assert that
"EthernetKRL's own transport is XML". That is false, and it undersold
the design. KUKA's EthernetKRL manual documents three connection data
types — XML, binary of fixed length, and a variable binary stream — and
says of the second that "Binary data records are not interpreted by the
EKI and stored together in a memory", read back with ``EKI_GetString``
and split into KRL variables with ``CAST_FROM``. A 16-byte fixed-length
binary record is therefore a **native EKI capability**, configured with
a ``<RAW><ELEMENT Tag="Buffer" Type="BYTE" Size="16" …/></RAW>`` block,
not a departure from EKI. ``krl_prog/laptop-comm.xml`` now declares
exactly that, and ``krl_prog/laptop_comm.src`` is the controller-side
loop that reads it — neither has ever been loaded on a controller, and
both say so in their own headers.

What remains true is the part that matters here: **EKI supplies no
checksum in any of its three modes.** The CRC-16 trailer in bytes 14–15
is this repository's addition, not "the standard integrity check on the
EthernetKRL transport". The conclusion was always right; the reason
given for it was not.

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


class CoordinateOutOfRange(struct.error):
    """A coordinate does not fit the field the wire has for it.

    Subclasses ``struct.error``, not ``ValueError``, so that
    :class:`execution.driver.RobotDriver` treats it as fatal — retrying
    a frame that cannot be built is three identical failures and a
    delay. The message names the axis and the actual limit, which
    ``struct``'s own does not.
    """


PROTOCOL_VERSION = 0x01
COMMAND_LEN = 16
STATUS_LEN = 16

# The header / payload layout as a single struct format string: it's
# used for both commands and status packets because they share shape.
_BODY_FMT = ">BBiihH"
_BODY_LEN = 14
_CRC_FMT = "<H"

# The per-axis limits the format string above actually imposes. They are
# NOT the same on all three axes and nothing but this table says so.
X_MM_MIN, X_MM_MAX = -(2 ** 31), 2 ** 31 - 1      # int32
Y_MM_MIN, Y_MM_MAX = -(2 ** 31), 2 ** 31 - 1      # int32
Z_MM_MIN, Z_MM_MAX = -(2 ** 15), 2 ** 15 - 1      # int16 — the odd one out

#: The largest cycle time ``aux_u16`` can carry inbound. Longer cycles
#: saturate here rather than wrapping.
CYCLE_MS_MAX = 0xFFFF

_AXIS_LIMITS = (
    ("x", X_MM_MIN, X_MM_MAX, "int32"),
    ("y", Y_MM_MIN, Y_MM_MAX, "int32"),
    ("z", Z_MM_MIN, Z_MM_MAX, "int16"),
)


class OpCode(IntEnum):
    """Every opcode understood by the KRL receive loop.

    **0x06 was called ``ESTOP`` until 2026-08-14 and is now ``HALT``.**
    The value is unchanged, so no controller or KRL dispatch table moves;
    what changes is the claim the name makes. This opcode travels in band
    on the same application channel as motion, and no robot surveyed for
    the driver design — Universal Robots, ROS 2, Franka FCI, ABB EGM —
    accepts a safety-rated stop that way. On UR, Franka and ABB an
    emergency stop is a hardware safety chain; the ROS 2 action layer
    does not model one at all. Four different things wear the word
    "stop" (decelerate-and-hold, exit control mode, protective stop,
    emergency stop) and this protocol has always had exactly the first.

    Calling it ``ESTOP`` invited a reader to believe the host could
    actuate a Category-0 stop by writing two bytes to a socket. It
    cannot. ``RobotStatusCode.ESTOP`` keeps its name, because that is the
    *controller* reporting a real safety state back — which is a
    different claim and a true one.
    """

    NOOP = 0x00
    MOVE_TO = 0x01
    VACUUM_ON = 0x02
    VACUUM_OFF = 0x03
    PICK_AND_PLACE = 0x04
    HEARTBEAT = 0x05
    HALT = 0x06
    HANDSHAKE = 0x07


# ------------------------------------------------------------- CRC ---

def crc16_modbus(data: bytes) -> int:
    """Compute CRC-16/MODBUS of ``data``.

    The standard MODBUS variant — poly 0xA001 (reflected 0x8005), init
    0xFFFF, reflected in and out, no output XOR — chosen for this
    project's binary framing because it is ubiquitous and independently
    checkable, NOT because EthernetKRL prescribes it: EKI carries no
    checksum in any of its three modes (see the module docstring).
    ``tests/test_protocol.py`` pins it to the published check value.
    ``krl_prog/laptop_comm.src`` mirrors this loop in KRL, bit-serial
    and table-free; that mirror has never been compiled or run.
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
    """Pack a 16-byte command packet including trailing CRC-16.

    Raises :class:`CoordinateOutOfRange` — a ``struct.error``, therefore
    fatal to the driver rather than retryable — naming the axis whose
    field the value does not fit. Nothing clamps: silently saturating a
    pose to 2**31-1 would command a motion nobody asked for.
    """
    for value, (axis, lo, hi, width) in zip(
            (int(x_mm), int(y_mm), int(z_mm)), _AXIS_LIMITS):
        if not lo <= value <= hi:
            raise CoordinateOutOfRange(
                f"{axis} = {value} mm does not fit the wire's {width} "
                f"field for that axis ({lo}..{hi} mm). Note the asymmetry: "
                f"x and y are int32, z is int16.")
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
    """Pack a 16-byte status packet with CRC-16 trailer.

    ``cycle_ms`` **saturates** at :data:`CYCLE_MS_MAX` rather than
    wrapping. Masking with ``& 0xFFFF`` reported a 70-second cycle as
    4.5 seconds — a small, plausible, wrong number arriving in the
    latency statistics with nothing to distinguish it from a real one.
    """
    body = struct.pack(
        _BODY_FMT,
        PROTOCOL_VERSION,
        int(code) & 0xFF,
        int(x_mm), int(y_mm), int(z_mm),
        max(0, min(int(cycle_ms), CYCLE_MS_MAX)),
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
        # A cycle at the ceiling is a cycle of AT LEAST 65 535 ms. The
        # field cannot say how much more, and a consumer averaging these
        # into a mean latency is entitled to know which samples are
        # censored rather than measured.
        "cycle_ms_saturated": cycle_ms >= CYCLE_MS_MAX,
    }


__all__ = [
    "COMMAND_LEN",
    "CYCLE_MS_MAX",
    "Command",
    "CoordinateOutOfRange",
    "CrcError",
    "X_MM_MAX",
    "X_MM_MIN",
    "Y_MM_MAX",
    "Y_MM_MIN",
    "Z_MM_MAX",
    "Z_MM_MIN",
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
