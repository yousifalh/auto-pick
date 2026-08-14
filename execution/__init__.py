"""Execution module.

Split, since 2026-08-14, along the line the vendor boundary actually
falls on rather than along the one the file names suggested. See
``docs/superpowers/specs/2026-08-14-robot-driver-abstraction.md``.

**Vendor-neutral — no driver may reimplement these:**

* :mod:`execution.driver` — :class:`~execution.driver.RobotDriver`, a
  template-method base owning the connection lifecycle, the retry
  policy, the whole-command deadline, the latching stop and the
  escalation, plus the neutral value types (:class:`Pose`,
  :class:`Request`, :class:`Reachability`, :class:`Capabilities`,
  :class:`Gripper`). It *seals* the safety methods: a subclass that
  rebinds one is rejected at class-definition time.

**This cell's application — not a property of any robot:**

* :mod:`execution.task` — :class:`~execution.task.PickPlaceTask`, the
  cartridge-insertion choreography, the transport height and the
  vacuum level.

**Backends. Each supplies encode / decode / send / recv and nothing
else:**

* :mod:`execution.protocol` — the 16-byte binary command / status
  packet layout with CRC-16/MODBUS.
* :mod:`execution.execution` — the KUKA EthernetKRL driver over that
  format, plus ``ExecutionConfig``.
* :mod:`execution.mock_kuka_server` — a software-only KUKA simulator
  used by the tests and the software-only demo.
* :mod:`execution.json_driver` and :mod:`execution.mock_json_server` —
  a second driver over a length-prefixed JSON encoding, and its
  controller. **Neither talks to any hardware, and neither is a step
  toward doing so.** They exist so that the claim "encode / decode /
  send / recv is the whole vendor boundary" is measured rather than
  asserted: both drivers pass ``tests/conformance.py`` unmodified.
"""
