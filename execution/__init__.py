"""Execution module.

Groups the three pieces needed to talk to a KUKA controller (real or
simulated):

* :mod:`execution.protocol` — the 16-byte binary command / status
  packet layout with CRC-16/MODBUS.
* :mod:`execution.execution` — a blocking TCP client that turns
  :class:`common.types.PickPlacePose` items into packets.
* :mod:`execution.mock_kuka_server` — a software-only KUKA simulator
  used by the tests and the software-only demo.
"""
