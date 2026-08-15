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

**The selection, and why it lives here:**

* :func:`build_driver` — the one place a *backend* is chosen. Until
  2026-08-15 there was no such place: ``main.py`` imported
  ``ExecutionConfig`` and ``KukaClient`` from
  :mod:`execution.execution` and spawned
  :mod:`execution.mock_kuka_server` by name, so the abstraction above
  had two conforming implementations and exactly one consumer, which
  named one of them. A seam nothing chooses across is a seam nobody has
  checked — and the second backend existing only inside tests is how
  "the vendor boundary is encode/decode/send/recv" stays a claim about
  the test suite rather than about the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from common.logging import get_logger

from .driver import RobotDriver, RobotEstop, RobotFault
from .task import TaskConfig

log = get_logger("execution.factory")

__all__ = [
    "RobotDriver",
    "RobotEstop",
    "RobotFault",
    "TaskConfig",
    "build_driver",
    "build_task_config",
]

#: What ``mode.robot`` may say. Listed once, so the error message and
#: the dispatch below cannot disagree about what is supported.
ROBOT_BACKENDS = ("mock", "real", "json")


def _stop(server) -> None:
    """Stop a simulator this module started, and give its port back.

    ``shutdown()`` alone ends ``serve_forever`` but leaves the listening
    socket bound, so the port keeps accepting connections into a backlog
    nobody will ever read — which is what ``main.py``'s ``finally``
    did, and why a second run on the same fixed 54600 could connect to a
    dead listener. ``server_close()`` is the half that releases it.
    """
    server.shutdown()
    server.server_close()


def build_task_config(exec_cfg: Mapping[str, Any]) -> TaskConfig:
    """The application's numbers out of the ``execution:`` config block.

    :class:`execution.task.TaskConfig` is vendor-neutral — a transport
    height and a vacuum level are facts about a cartridge tray — but the
    YAML they arrive in is the same file the transport's host, port and
    timeouts arrive in, and ``ExecutionConfig.from_dict`` is the only
    thing that parses that file (including the checks that reject a
    ``kuka.command_length_bytes`` disagreeing with the packer). Going
    through it here means the two halves of one config file cannot be
    parsed two ways, and it is the reason ``main.py`` no longer has to
    name ``ExecutionConfig`` to obtain two floats.
    """
    from .execution import ExecutionConfig

    return TaskConfig.from_execution_config(ExecutionConfig.from_dict(exec_cfg))


def build_driver(mode_cfg: Mapping[str, Any],
                 exec_cfg: Mapping[str, Any]) -> RobotDriver:
    """The :class:`~execution.driver.RobotDriver` ``mode.robot`` asks for.

    ``mock`` (the default) and ``real`` both give the KUKA backend — the
    difference is only whether this function starts a local simulator
    and points the driver at it, or points it at the ``kuka:`` host in
    the config. ``json`` gives :class:`~execution.json_driver
    .JsonRobotDriver` against its own local controller; there is no
    hardware behind that encoding and there is not meant to be. It is
    selectable because the second conforming backend being reachable
    only from ``tests/`` is what made the seam ornamental.

    **One policy, both backends.** The timeouts and the retry count come
    from ``ExecutionConfig.driver_policy()`` whichever backend is built,
    so the two cannot drift apart on the numbers the conformance suite
    exercises them under.

    **A spawned simulator dies with the driver.** Both mock backends
    start a thread, and its shutdown is registered on the driver via
    ``add_teardown`` rather than handed back for the caller to remember.
    See that method for why: ``close()`` is the one event on every route
    out of a run, and the previous arrangement made a leaked thread the
    default outcome of forgetting a ``finally``.

    Returns an *unconnected* driver. Connecting is ``with driver as
    robot:``, which is where the handshake and its failure paths live.
    """
    from .execution import ExecutionConfig, KukaClient

    backend = str(mode_cfg.get("robot", "mock"))
    if backend not in ROBOT_BACKENDS:
        raise ValueError(
            f"mode.robot = {backend!r} names no backend this cell has. "
            f"Supported: {', '.join(ROBOT_BACKENDS)}. `mock` runs the KUKA "
            f"driver against a local simulator, `real` runs it against "
            f"execution.kuka.host, and `json` runs the second conforming "
            f"driver against its own local controller (a test backend — no "
            f"hardware speaks that encoding).")

    exec_conf = ExecutionConfig.from_dict(exec_cfg)
    sim_cfg: Dict[str, Any] = dict(exec_cfg.get("simulation") or {})

    if backend == "real":  # pragma: no cover - needs a controller
        # host and port are already ExecutionConfig.from_dict's, i.e.
        # the `kuka:` block's. Nothing is spawned.
        return KukaClient(exec_conf)

    if backend == "mock":
        from .mock_kuka_server import run_in_thread

        server, _thread = run_in_thread(
            host=str(sim_cfg.get("listen_host", "127.0.0.1")),
            port=int(sim_cfg.get("listen_port", 54600)),
            drop_prob=float(sim_cfg.get("drop_probability", 0.02)),
            ms_per_100mm=int(
                sim_cfg.get("simulated_move_time_ms_per_100mm", 180)),
        )
        # The port the simulator ACTUALLY bound to, not the one it was
        # asked for. They differ whenever the config says 0, which is
        # how a caller asks the OS for a free port instead of colliding
        # on a fixed 54600 - and reading it back is the only way the
        # driver can then find it.
        exec_conf.host, exec_conf.port = server.server_address[:2]
        # The BOUND address, which is why this line is here and not in
        # the simulator: `run_in_thread` logs the port it was asked for,
        # and a config that asked for 0 makes that number a lie.
        log.info("mock KUKA controller listening on %s:%d",
                 exec_conf.host, exec_conf.port)
        driver: RobotDriver = KukaClient(exec_conf)
        driver.add_teardown(lambda: _stop(server))
        return driver

    from .json_driver import JsonDriverConfig, JsonRobotDriver
    from .mock_json_server import run_in_thread as run_json_in_thread

    json_cfg: Dict[str, Any] = dict(exec_cfg.get("json") or {})
    server, _thread = run_json_in_thread(
        host=str(json_cfg.get("listen_host", "127.0.0.1")),
        port=int(json_cfg.get("listen_port", 55600)),
        drop_prob=float(json_cfg.get("drop_probability", 0.0)),
    )
    host, port = server.server_address[:2]
    log.info("mock JSON controller listening on %s:%d (a test backend - "
             "no hardware speaks this encoding)", host, port)
    policy = exec_conf.driver_policy()
    json_driver = JsonRobotDriver(JsonDriverConfig(
        host=host,
        port=port,
        handshake_timeout_ms=policy.handshake_timeout_ms,
        command_timeout_ms=policy.command_timeout_ms,
        retry_pause_ms=policy.retry_pause_ms,
        max_retries=policy.max_retries,
    ))
    json_driver.add_teardown(lambda: _stop(server))
    return json_driver
