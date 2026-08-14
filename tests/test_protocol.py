"""Wire-protocol tests.

These correspond to the integrity requirements R4 in PPR §7.3: bad
packets must be rejected, and good packets must round-trip.
"""
from __future__ import annotations

import pytest

from execution.protocol import (
    COMMAND_LEN, STATUS_LEN, OpCode, PROTOCOL_VERSION,
    crc16_modbus, pack_command, unpack_command,
    pack_status, unpack_status,
)


# --------------------------- CRC vectors ----------------------------------

def test_crc16_modbus_known_vectors():
    # Reference vectors for CRC-16/MODBUS (polynomial 0xA001, init 0xFFFF).
    # Cross-checked against an independent reference implementation.
    #
    # The first is the CRC catalogue's CHECK VALUE - the published
    # constant for CRC-16/MODBUS over b"123456789", the one number that
    # identifies this variant among the ~20 CRC-16s that share a
    # polynomial and differ in init / reflection / output XOR. It is
    # what lets a reader confirm `execution.protocol`'s framing claim
    # against a spec rather than against this repository.
    assert crc16_modbus(b"123456789") == 0x4B37
    assert crc16_modbus(b"\x01\x03\x00\x00\x00\x01") == 0x0A84
    # Empty input → initial value
    assert crc16_modbus(b"") == 0xFFFF
    # Single zero byte
    assert crc16_modbus(b"\x00") == 0x40BF


def test_crc16_deterministic():
    msg = b"hello world"
    assert crc16_modbus(msg) == crc16_modbus(msg)


def test_crc16_byte_change_flips_crc():
    a = crc16_modbus(b"\x00\x01\x02\x03")
    b = crc16_modbus(b"\x00\x01\x02\x04")
    assert a != b


# --------------------------- packing --------------------------------------

def test_pack_length():
    pkt = pack_command(OpCode.HANDSHAKE)
    assert len(pkt) == COMMAND_LEN


def test_pack_unpack_roundtrip():
    pkt = pack_command(OpCode.MOVE_TO, x_mm=100, y_mm=-50, z_mm=200, aux=80)
    cmd = unpack_command(pkt)
    assert cmd.op == OpCode.MOVE_TO
    assert cmd.x_mm == 100
    assert cmd.y_mm == -50
    assert cmd.z_mm == 200
    assert cmd.aux == 80


def test_pack_unpack_all_opcodes():
    for op in OpCode:
        pkt = pack_command(op, 1, 2, 3, 4)
        cmd = unpack_command(pkt)
        assert cmd.op == op


def test_unpack_wrong_length():
    with pytest.raises(ValueError, match="must be 16 bytes"):
        unpack_command(b"\x00" * 15)


def test_unpack_crc_corruption_rejected():
    pkt = bytearray(pack_command(OpCode.MOVE_TO, 1, 2, 3, 4))
    pkt[5] ^= 0xFF  # flip a body byte but leave the CRC intact
    with pytest.raises(ValueError, match="CRC"):
        unpack_command(bytes(pkt))


def test_unpack_bad_version_rejected():
    # Replay a valid CRC over a packet with a different version byte.
    import struct
    body = struct.pack(">BBiihH", 0x02, int(OpCode.MOVE_TO), 0, 0, 0, 0)
    crc = crc16_modbus(body)
    pkt = body + struct.pack("<H", crc)
    with pytest.raises(ValueError, match="version"):
        unpack_command(pkt)


def test_negative_coordinates_roundtrip():
    pkt = pack_command(OpCode.MOVE_TO, -350, -350, -10, 0)
    cmd = unpack_command(pkt)
    assert cmd.x_mm == -350
    assert cmd.y_mm == -350
    assert cmd.z_mm == -10


# ----------------------------- status -------------------------------------

def test_status_roundtrip():
    pkt = pack_status(code=1, x_mm=12, y_mm=-5, z_mm=80, cycle_ms=123)
    assert len(pkt) == STATUS_LEN
    s = unpack_status(pkt)
    assert s == {"code": 1, "x_mm": 12, "y_mm": -5, "z_mm": 80,
                 "cycle_ms": 123, "cycle_ms_saturated": False}


def test_status_bad_crc_rejected():
    pkt = bytearray(pack_status(1, 0, 0, 0, 0))
    pkt[14] ^= 0xAA
    with pytest.raises(ValueError):
        unpack_status(bytes(pkt))


def test_status_wrong_length():
    with pytest.raises(ValueError):
        unpack_status(b"\x00" * 10)
