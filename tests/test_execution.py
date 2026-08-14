"""What is left of the KUKA execution tests once the neutral half moved.

Every assertion here is about **this encoding** or **this cell's task**:
the millimetre quantiser, the 16-byte frame's two asymmetric fields, the
descriptive config keys, the KRL coupling, and the two-frame
pick-and-place sequence.

The vendor-neutral half — the three E-stop escape routes, the retry and
escalation policy, the latch, the deadlines, the closed status set — is
in ``tests/conformance.py`` and runs from ``tests/test_kuka_conformance.py``
and ``tests/test_json_conformance.py``. It moved because those
assertions were about a robot driver and were written against sixteen
bytes: roughly half of them checked ``OpCode.ESTOP`` on the wire, which
no second encoding can satisfy however correct its behaviour.
"""
from __future__ import annotations

import os
import re

import pytest

from common.types import (PickPlacePose, RobotStatus, RobotStatusCode,
                          WorkspacePoint)
from execution.driver import Pose, Reachability, RequestKind
from execution.execution import (KUKA_CONTROLLER_INSERT_Z_MM, ExecutionConfig,
                                 KukaClient, wire_mm)
from execution.mock_kuka_server import run_in_thread
from execution.protocol import (COMMAND_LEN, CYCLE_MS_MAX, STATUS_LEN,
                                CoordinateOutOfRange, OpCode, Z_MM_MAX,
                                pack_command, pack_status, unpack_command,
                                unpack_status)
from execution.task import PickPlaceTask, TaskConfig
from tests.conformance import ScriptedEndpoint
from tests.test_kuka_conformance import KukaHarness


@pytest.fixture
def mock_server():
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=0.0, ms_per_100mm=5)
    port = srv.server_address[1]
    yield port
    srv.shutdown()


def _task(driver: KukaClient, cfg: ExecutionConfig) -> PickPlaceTask:
    return PickPlaceTask(driver, TaskConfig.from_execution_config(cfg))


# ----------------------------------------------------- happy path -------

def test_handshake_via_context_manager(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server,
                          handshake_timeout_ms=1000)
    with KukaClient(cfg):
        pass  # enter + exit must not raise


def test_move_to_returns_success(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        s = k.move_to(WorkspacePoint(100, 50, 80))
        assert s.code in (RobotStatusCode.SUCCESS, RobotStatusCode.OK)


def test_vacuum_on_off(mock_server):
    """The vacuum is the gripper capability now, not an interface method."""
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        assert k.gripper.grasp().code == RobotStatusCode.SUCCESS
        assert k.gripper.release().code == RobotStatusCode.SUCCESS
        # This tooling has no grasp sensor the host can read, and says so
        # rather than guessing.
        assert k.gripper.holding is None


def test_pick_and_place_succeeds(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    pose = PickPlacePose(
        pick=WorkspacePoint(100.0, 50.0, 30.0),
        place=WorkspacePoint(-50.0, 100.0, 5.0),
        cartridge_id=0, grid_row=1, grid_col=2,
    )
    with KukaClient(cfg) as k:
        status = _task(k, cfg).run(pose)
        assert status.code == RobotStatusCode.SUCCESS
        assert status.cycle_time_ms >= 0
        # Robot should have ended near the place target
        assert abs(status.current_pose.x_mm - (-50)) <= 1
        assert abs(status.current_pose.y_mm - 100) <= 1


def test_pick_failure_reported():
    """drop_prob=1.0 forces every pick to fail."""
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=1.0, ms_per_100mm=2)
    port = srv.server_address[1]
    try:
        cfg = ExecutionConfig(host="127.0.0.1", port=port)
        pose = PickPlacePose(
            pick=WorkspacePoint(10, 10, 5),
            place=WorkspacePoint(20, 20, 5),
            cartridge_id=0, grid_row=0, grid_col=0,
        )
        with KukaClient(cfg) as k:
            status = _task(k, cfg).run(pose)
            assert status.code == RobotStatusCode.PICK_FAILED
    finally:
        srv.shutdown()


# -------------------------- the two-frame pick-and-place, made honest ----

def test_a_cycle_is_two_frames_and_the_first_one_latches():
    """``PICK_AND_PLACE`` carries ONE coordinate triple.

    The place XY is whatever the arm was at when the subroutine began,
    and nothing in the frame says so. The old ``pick_and_place(pose) ->
    RobotStatus`` signature said *one pose in, one status out*, which is
    precisely what this protocol is not — and it was the leak that most
    threatened the abstraction, because the signature an interface would
    naturally adopt is the one that conceals it.

    The sequence is now returned, named, and countable.
    """
    driver = KukaClient(ExecutionConfig(port=1))
    requests = driver.pick_place_requests(
        pick=WorkspacePoint(100.0, 50.0, 5.0),
        place=WorkspacePoint(-50.0, 100.0, 2.0),
        transport_height_mm=80.0,
        vacuum_level_percent=80,
    )
    assert len(requests) == 2, "this protocol needs two frames per cycle"
    latch, cycle = requests
    assert "latch" in latch.name.lower(), (
        "the first frame's name must say what it does, or the next reader "
        "reads it as an ordinary approach move")
    assert latch.kind is RequestKind.MOVE
    assert latch.target.xyz_mm == pytest.approx((-50.0, 100.0, 80.0))
    assert cycle.payload["op"] is OpCode.PICK_AND_PLACE
    assert cycle.target.xyz_mm == pytest.approx((100.0, 50.0, 5.0))


def test_the_place_z_is_not_on_the_wire_and_the_task_says_so(caplog):
    """``pose.place.z_mm`` is computed by the planner, validated by
    ``WorkspaceBounds``, and then dropped: the frame has one Z field and
    the pick needs it.

    That cannot be fixed without a frame-layout change and a version
    bump, which is deliberately not done. What is fixed is the silence:
    two of the three unreconcilable insert depths are now one constant,
    and the third — the planner's — is compared against it out loud.
    """
    import logging

    driver = KukaClient(ExecutionConfig(port=1))
    assert driver.declared_insert_depth_mm == KUKA_CONTROLLER_INSERT_Z_MM

    requests = driver.pick_place_requests(
        pick=WorkspacePoint(1.0, 2.0, 3.0),
        place=WorkspacePoint(4.0, 5.0, 60.0),      # a place Z of 60 mm
        transport_height_mm=80.0, vacuum_level_percent=80)
    z_values = [r.target.xyz_mm[2] for r in requests if r.target]
    assert 60.0 not in z_values, "the place Z has nowhere to go on this wire"

    handler_records = []

    class _Grab(logging.Handler):
        def emit(self, record):
            handler_records.append(record.getMessage())

    grab = _Grab(level=logging.WARNING)
    logging.getLogger("execution.task").addHandler(grab)
    try:
        task = PickPlaceTask(driver, TaskConfig())
        task._check_insert_depth(PickPlacePose(
            pick=WorkspacePoint(1, 2, 3), place=WorkspacePoint(4, 5, 60.0),
            cartridge_id=0, grid_row=0, grid_col=0))
    finally:
        logging.getLogger("execution.task").removeHandler(grab)

    assert any("not reaching the robot" in m for m in handler_records), (
        "a planner value that stops at the client must be announced, not "
        "discovered at the tray")


def test_the_simulator_and_the_client_agree_on_the_insert_depth():
    """One value, imported, not two literals that happen to match."""
    from execution import mock_kuka_server

    assert mock_kuka_server._INSERT_Z_MM == int(KUKA_CONTROLLER_INSERT_Z_MM)


# ------------------------------------------- the two asymmetric fields ---

def test_z_is_int16_while_x_and_y_are_int32():
    """The frame's least visible hazard, and it points at the table.

    ``_BODY_FMT`` is ``">BBiihH"``: ``i``, ``i``, ``h``. A coordinate
    between 32 767 and 2**31 is accepted on two axes and rejected on the
    third, and ``wire_mm``'s docstring used to say "outside int32" for
    all three — a documentation lie about a safety-relevant range.
    """
    assert Z_MM_MAX == 2 ** 15 - 1

    # x and y take it; z does not.
    pack_command(OpCode.MOVE_TO, 100_000, 100_000, 0)
    with pytest.raises(CoordinateOutOfRange) as err:
        pack_command(OpCode.MOVE_TO, 0, 0, 100_000)
    assert "z" in str(err.value) and "int16" in str(err.value)
    assert "int32" in str(err.value), (
        "the message must name the asymmetry, because that is the part "
        "nobody expects")


def test_an_out_of_range_z_is_fatal_not_retryable():
    """``CoordinateOutOfRange`` subclasses ``struct.error``, so it falls
    past the driver's transient tuple to the fatal path. Making it a
    ``ValueError`` would demote it to "retry the same impossible frame
    three times and then stop", which is slower and no safer."""
    import struct as _struct

    assert issubclass(CoordinateOutOfRange, _struct.error)
    assert not issubclass(CoordinateOutOfRange, ValueError)


def test_the_cycle_time_field_saturates_rather_than_wrapping():
    """``aux_u16`` is vacuum percent outbound and cycle milliseconds
    inbound — one field, two meanings, one direction each. The inbound
    meaning does not fit: it used to mask with ``& 0xFFFF``, so a
    70-second cycle was reported as 4.5 seconds and went straight into
    the latency statistics ``main.py`` prints."""
    wrapped_before = 70_000 & 0xFFFF
    assert wrapped_before == 4464, "the defect, for the record"

    frame = pack_status(code=1, x_mm=0, y_mm=0, z_mm=0, cycle_ms=70_000)
    s = unpack_status(frame)
    assert s["cycle_ms"] == CYCLE_MS_MAX
    assert s["cycle_ms_saturated"] is True

    ordinary = unpack_status(
        pack_status(code=1, x_mm=0, y_mm=0, z_mm=0, cycle_ms=1234))
    assert ordinary["cycle_ms"] == 1234
    assert ordinary["cycle_ms_saturated"] is False


def test_a_saturated_cycle_time_is_flagged_on_the_status():
    """A censored sample must not enter a mean as though measured."""
    driver = KukaClient(ExecutionConfig(port=1))
    status = driver.decode(
        pack_status(code=1, x_mm=0, y_mm=0, z_mm=0, cycle_ms=70_000))
    assert isinstance(status, RobotStatus)
    assert "saturated" in status.message
    assert driver.decode(
        pack_status(code=1, x_mm=0, y_mm=0, z_mm=0, cycle_ms=10)).message == ""


# ------------------------------------------------- what validate knows ---

def test_validate_answers_unknown_for_anything_the_frame_can_carry():
    """This client holds no envelope, and must not pretend otherwise.

    The +/-350 mm square is in ``configs/planning.yaml`` and enforced by
    ``plan.scene.WorkspaceBounds``; the 706 mm reach is modelled only by
    the simulator; on hardware the KRC's software limits decide. Three
    components guarantee reachability and none of them is this one.
    """
    driver = KukaClient(ExecutionConfig(port=1))
    assert driver.validate(Pose.from_mm(100, 100, 50)) is Reachability.UNKNOWN
    assert driver.validate(
        Pose.from_mm(2.0 ** 31, 0, 0)) is Reachability.UNREACHABLE
    # ...and the int16 axis is caught where the int32 ones are not.
    assert driver.validate(Pose.from_mm(0, 0, 100_000)) is (
        Reachability.UNREACHABLE)
    assert driver.validate(
        Pose.from_mm(100_000, 0, 0)) is Reachability.UNKNOWN
    assert driver.capabilities.validate_is_real is False


def test_the_capability_descriptor_admits_what_this_wire_drops():
    caps = KukaClient(ExecutionConfig(port=1)).capabilities
    assert caps.carries_orientation is False
    assert caps.carries_redundancy is False
    assert caps.position_resolution_m == 0.001
    joined = " ".join(caps.lossy_notes)
    for expected in ("orientation", "redundancy", "int16", "place-Z"):
        assert expected in joined, expected


# ------------------------------------------------- the handshake ack -----

def test_the_simulator_acks_the_handshake_with_the_one_accepted_code(
        mock_server):
    """The client used to accept ``OK`` *or* ``SUCCESS`` while nothing on
    either side of the wire emitted ``SUCCESS`` for a handshake — an
    accepted set wider than every implementation, which a conformance
    suite cannot pin and a second controller author would read as
    permission. It accepts only ``OK`` now, so what the simulator sends
    has to be pinned somewhere."""
    import socket as _socket

    sock = _socket.create_connection(("127.0.0.1", mock_server), timeout=5)
    try:
        sock.sendall(pack_command(OpCode.HANDSHAKE))
        buf = b""
        while len(buf) < STATUS_LEN:
            buf += sock.recv(STATUS_LEN - len(buf))
        assert unpack_status(buf)["code"] == RobotStatusCode.OK.value
    finally:
        sock.close()


# -------------------- RobotStatusCode <-> the KRL subroutine -------------
#
# `common.types.RobotStatusCode` says "Values must not be renumbered
# without matching updates in execution.protocol and the KRL
# subroutine". `execution/protocol.py` names the enum only in prose and
# packs a raw int; the KRL side is a bare `RETURN 2 ; PICK_FAILED`
# literal in a file no test read and no import touches. Renumbering the
# enum therefore silently redefined what the real controller means by
# `2` - on hardware, and with nothing to notice until a cell was
# dropped. These tests are the missing coupling.

_KRL_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "execution", "krl_prog", "routines.src")

# `      RETURN 2                 ; PICK_FAILED`
_KRL_RETURN = re.compile(
    r"^\s*RETURN\s+(-?\d+)\s*;\s*([A-Z_][A-Z0-9_]*)\s*$", re.MULTILINE)


def _krl_returns() -> list:
    with open(_KRL_SRC, encoding="utf-8", errors="replace") as fh:
        return [(int(v), name) for v, name in _KRL_RETURN.findall(fh.read())]


# `GLOBAL DEFFCT INT PickAndPlace(...)` / `DEF routines()`
_KRL_DECL = re.compile(
    r"^\s*(GLOBAL\s+)?(DEFFCT|DEF)\s+(?:(\w+)\s+)?(\w+)\s*\(")
_KRL_END = re.compile(r"^\s*(ENDFCT|END)\s*$")


def _krl_routines() -> list:
    """Parse routines.src into ``(name, keyword, ret_type, is_global,
    terminator, has_valued_return)`` tuples, in file order.

    A hand-rolled two-token scan, not a KRL parser: it reads the
    declaration line, the matching terminator, and whether any
    ``RETURN <int>`` appears between them. That is exactly as much
    structure as the assertions below need and no more.
    """
    routines = []
    cur = None
    with open(_KRL_SRC, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            decl = _KRL_DECL.match(line)
            if decl and cur is None:
                is_global, kw, ret_type, name = decl.groups()
                cur = [name, kw, ret_type, bool(is_global), None, False]
                continue
            if cur is None:
                continue
            if re.match(r"^\s*RETURN\s+-?\d+", line):
                cur[5] = True
            end = _KRL_END.match(line)
            if end:
                cur[4] = end.group(1)
                routines.append(tuple(cur))
                cur = None
    return routines


def test_krl_valued_returns_are_inside_a_deffct_not_a_def():
    """`RETURN <value>` is only admissible in a KRL *function*.

    KUKA's Expert Programming manual: "Unlike a subprogram, a function
    sends back a return value. A function begins with the keyword
    DEFFCT. ... The return value itself is transferred via RETURN. The
    function is terminated using the keyword ENDFCT." In a subprogram
    RETURN takes no operand — it only exits — so `DEF PickAndPlace(...)
    ... RETURN 1 ... END`, which is what this file held until
    2026-08-14, is an inadmissible instruction and would not compile.

    The test below pins the *numbers*; it was perfectly happy with them
    inside a construct that could never run, which is how a defect
    carried a green tick for months (audit T §3). This one pins the
    *construct*: any routine that returns a value must be declared
    DEFFCT with a return type and terminated ENDFCT.
    """
    routines = _krl_routines()
    assert routines, f"parsed no routines out of {_KRL_SRC}"

    valued = [r for r in routines if r[5]]
    assert valued, (
        f"no `RETURN <int>` found in {_KRL_SRC}; if the status codes "
        f"moved, move this assertion with them")

    for name, kw, ret_type, _is_global, terminator, _ in valued:
        assert kw == "DEFFCT", (
            f"{name} returns a value but is declared {kw}. RETURN takes "
            f"no operand in a KRL subprogram; this does not compile.")
        assert ret_type, (
            f"DEFFCT {name} declares no return type. The type goes "
            f"directly after the keyword.")
        assert terminator == "ENDFCT", (
            f"DEFFCT {name} is terminated with {terminator}, not ENDFCT.")

    by_name = {r[0]: r for r in routines}
    assert "PickAndPlace" in by_name, (
        "PickAndPlace is gone from routines.src; execution/protocol.py's "
        "PICK_AND_PLACE opcode has nothing to dispatch to.")
    assert by_name["PickAndPlace"][1:3] == ("DEFFCT", "INT")


def test_krl_first_def_matches_the_file_name():
    """"The object name without an extension is also the name of the
    file and is therefore prefixed by DEF" — Expert Programming.

    routines.src used to begin `DEF MoveBetweenPositions(...)`, which
    declares an object named MoveBetweenPositions inside a file named
    routines: the first thing a KRL editor rejects on load.
    """
    first = _krl_routines()[0]
    stem = os.path.splitext(os.path.basename(_KRL_SRC))[0]
    assert first[0] == stem, (
        f"{_KRL_SRC} leads with `{first[1]} {first[0]}` but the file is "
        f"named {stem}.src. The leading declaration must match the file "
        f"name.")


def test_krl_subroutines_reachable_from_another_src_are_global():
    """Only the leading declaration may be local.

    "Local subprograms/functions can only be called from within the SRC
    file in which they were programmed. ... Alternatively, a local
    subprogram can be preceded by the keyword GLOBAL" — Expert
    Programming. The receive loop lives in `laptop_comm.src`, a
    different SRC, so every routine it calls needs GLOBAL. None of them
    carried it, so none was reachable and the KRL half of the system
    could not have been wired together at all.
    """
    routines = _krl_routines()
    not_global = [r[0] for r in routines[1:] if not r[3]]
    assert not not_global, (
        f"{not_global} in {_KRL_SRC} are local subprograms. Nothing "
        f"outside routines.src can call them.")


def test_krl_subroutine_returns_the_numbers_this_enum_declares():
    """Every `RETURN n ; NAME` in routines.src must equal
    RobotStatusCode[NAME].value.

    Scope, because a green tick here has been misread before: this
    couples the enum to the KRL *labels*. It does not compile, parse or
    execute KRL, and no test in this repository does — there is no
    controller. `test_krl_valued_returns_are_inside_a_deffct_not_a_def`
    checks the one construct that had actually broken.
    """
    returns = _krl_returns()
    for value, name in returns:
        assert name in RobotStatusCode.__members__, (
            f"routines.src returns {value} labelled {name}, which is not a "
            f"RobotStatusCode. The controller and the host disagree about "
            f"what the wire means.")
        assert RobotStatusCode[name].value == value, (
            f"routines.src returns {value} for {name} but "
            f"RobotStatusCode.{name} is {RobotStatusCode[name].value}. "
            f"Renumbering the enum without editing the KRL redefines what "
            f"the real controller is saying.")


def test_the_krl_coupling_test_is_not_vacuous():
    """A reformat of routines.src must not silently disarm the test above.

    The regex is the whole mechanism; if it stops matching, the previous
    test passes over an empty list and the coupling is gone again — the
    exact failure mode this pair exists to prevent.
    """
    returns = _krl_returns()
    assert len(returns) >= 2, (
        f"parsed {len(returns)} `RETURN n ; NAME` lines out of "
        f"{_KRL_SRC}; the coupling test needs the comment labels to "
        f"survive. Keep the `; NAME` suffix, or update the regex here.")
    names = {name for _, name in returns}
    assert {"SUCCESS", "PICK_FAILED"} <= names, (
        f"routines.src no longer returns both SUCCESS and PICK_FAILED "
        f"(found {sorted(names)})")


def test_codes_the_real_controller_never_emits_are_named_as_such():
    """7 and 8 are simulator-only, and that is worth pinning.

    `UNSUPPORTED_COMMAND` and `VERSION_MISMATCH` exist so a controller
    that cannot parse a frame can say WHY, and the driver treats them
    as fatal rather than retryable. The KRL subroutine does not
    participate: it returns neither. The reasoning is sound and the gap
    is real, so it is asserted rather than left to be rediscovered.
    """
    emitted = {name for _, name in _krl_returns()}
    assert "UNSUPPORTED_COMMAND" not in emitted
    assert "VERSION_MISMATCH" not in emitted


# ------------------------------- millimetre -> wire quantisation ---------
#
# `WorkspacePoint` carries floats; the 16-byte frame carries signed
# integer millimetres. That conversion used to be `int()`, which
# truncates TOWARD ZERO: every commanded coordinate lost up to 0.999 mm,
# always toward the workspace origin, and the bias REVERSED SIGN at 0 -
# two cartridges either side of the origin were both pulled inward,
# toward each other. On a 4.25 mm shipping-wall inset that is 23 % of
# the margin. It is also applied AFTER `WorkspaceBounds.require` has
# validated the float, so the value that was checked was not the value
# that was commanded.

def _record(fn) -> list:
    """Run ``fn(driver)`` against a server that records every command."""
    harness = KukaHarness()

    def reply(observed, n):
        return harness.status_frame(
            RobotStatusCode.OK if observed.kind is RequestKind.HANDSHAKE
            else RobotStatusCode.SUCCESS)

    endpoint = ScriptedEndpoint(harness.read_frame, harness.describe, reply)
    try:
        with harness.make_driver(endpoint.port) as k:
            fn(k)
        cmds = [unpack_command(r.raw) for r in endpoint.requests]
        return [c for c in cmds if c.op is not OpCode.HANDSHAKE]
    finally:
        endpoint.close()


def test_wire_mm_rounds_and_is_symmetric_about_zero():
    """The unit test of the quantiser itself, both signs and the tie."""
    # The measured defect: int() gave 12 and -12 for these two.
    assert wire_mm(12.9) == 13
    assert wire_mm(-12.9) == -13
    assert wire_mm(0.6) == 1
    assert wire_mm(-0.6) == -1
    assert wire_mm(349.9) == 350
    assert wire_mm(-349.9) == -350

    # Symmetric about zero: no coordinate may be treated differently for
    # being on the far side of the origin. This is the property `int()`
    # broke, and it is the one that matters for a workspace that
    # straddles zero.
    for v in (0.1, 0.5, 0.9, 1.4, 1.5, 2.5, 12.3, 12.5, 12.9, 349.4999):
        assert wire_mm(-v) == -wire_mm(v), v

    # Bounded by half a millimetre everywhere, which `int()`'s 0.999 mm
    # was not. Exact halves are included deliberately: round-half-to-even
    # is what makes the error unbiased over a population of poses.
    for i in range(-4000, 4001):
        v = i / 8.0                      # 0.125 mm steps, both signs
        assert abs(wire_mm(v) - v) <= 0.5, v

    assert wire_mm(0.5) == 0 and wire_mm(1.5) == 2 and wire_mm(2.5) == 2, (
        "banker's rounding: ties go to even, so a population of poses "
        "gains no net outward or inward drift")

    # Integers must survive untouched - a pose already on a millimetre
    # is not a rounding question.
    for v in (-350.0, -1.0, 0.0, 1.0, 350.0):
        assert wire_mm(v) == int(v)


def test_wire_mm_survives_the_metre_round_trip():
    """The interface pose is in SI metres; the wire is in millimetres.

    So every coordinate now makes a ``/1000`` and ``*1000`` round trip
    before it is quantised, and a value that was an exact half in
    millimetres can arrive one ulp above or below it. Without the
    nanometre snap in ``wire_mm`` a tie would then break in whichever
    direction the floating-point noise happened to fall, which is a
    silent, magnitude-dependent reintroduction of the bias the round
    was chosen to remove.
    """
    for i in range(-8000, 8001):
        mm = i / 8.0
        assert wire_mm(Pose.from_mm(mm, 0, 0).xyz_mm[0]) == wire_mm(mm), mm


def test_move_to_rounds_the_commanded_coordinate():
    cmds = _record(lambda k: k.move_to(WorkspacePoint(12.9, -12.9, 0.6)))
    assert len(cmds) == 1
    c = cmds[0]
    assert c.op is OpCode.MOVE_TO
    # int() would have sent (12, -12, 0): 0.9 mm lost on x, 0.9 mm lost
    # on y in the OPPOSITE direction, and z collapsed to the origin.
    assert (c.x_mm, c.y_mm, c.z_mm) == (13, -13, 1)


def test_pick_and_place_rounds_every_coordinate_it_sends():
    pose = PickPlacePose(
        pick=WorkspacePoint(-100.5, 100.4, 30.7),
        place=WorkspacePoint(-49.6, 100.5, 5.0),
        cartridge_id=0, grid_row=1, grid_col=2,
    )
    cmds = _record(lambda k: _task(k, ExecutionConfig()).run(pose))
    assert [c.op for c in cmds] == [OpCode.MOVE_TO, OpCode.PICK_AND_PLACE]

    transport, pick = cmds
    # The transport MOVE_TO latches the place XY. int() sent (-49, 100).
    assert (transport.x_mm, transport.y_mm) == (-50, 100)
    assert transport.z_mm == round(80.0)          # cfg.transport_height_mm
    # int() sent (-100, 100, 30) for the pick.
    assert (pick.x_mm, pick.y_mm, pick.z_mm) == (-100, 100, 31)


def test_no_commanded_coordinate_is_displaced_by_more_than_half_a_mm():
    """The property `WorkspaceBounds.require` is entitled to assume.

    `require` validates the float pose in the planner; this quantiser is
    what turns it into the integers the controller receives. The two are
    only reconcilable if the displacement between them is bounded, and
    bounded symmetrically - which is asserted here against the bytes on
    the wire rather than against the helper in isolation.
    """
    poses = [(x / 7.0, -x / 7.0, x / 11.0) for x in range(-40, 41)]
    cmds = _record(
        lambda k: [k.move_to(WorkspacePoint(*p)) for p in poses])
    assert len(cmds) == len(poses)
    for (x, y, z), c in zip(poses, cmds):
        assert abs(c.x_mm - x) <= 0.5 and abs(c.y_mm - y) <= 0.5
        assert abs(c.z_mm - z) <= 0.5
        assert (c.x_mm, c.y_mm, c.z_mm) == (
            wire_mm(x), wire_mm(y), wire_mm(z))


# ------------------------------------------------------- configuration ---

def test_execution_config_from_dict():
    cfg = ExecutionConfig.from_dict({
        "kuka": {"host": "10.0.0.1", "port": 1234, "max_retries": 7},
        "motion": {"transport_height_mm": 55.0, "vacuum_level_percent": 95},
    })
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 1234
    assert cfg.max_retries == 7
    assert cfg.transport_height_mm == 55.0
    assert cfg.vacuum_level_percent == 95


def test_execution_config_defaults():
    cfg = ExecutionConfig.from_dict({})
    assert cfg.host == "172.31.1.147"
    assert cfg.port == 54600
    assert cfg.transport_height_mm == 80.0


def test_the_config_splits_into_a_driver_policy_and_a_task_config():
    """Three different things used to share one dataclass: the transport,
    the driver's policy, and this cell's choreography. Only the middle
    one is vendor-neutral, and the split is what let the task move out of
    the driver."""
    cfg = ExecutionConfig(handshake_timeout_ms=11, command_timeout_ms=22,
                          heartbeat_interval_ms=33, max_retries=4,
                          transport_height_mm=44.0, vacuum_level_percent=55)
    policy = cfg.driver_policy()
    assert (policy.handshake_timeout_ms, policy.command_timeout_ms,
            policy.retry_pause_ms, policy.max_retries) == (11, 22, 33, 4)
    task = TaskConfig.from_execution_config(cfg)
    assert (task.transport_height_mm, task.vacuum_level_percent) == (44.0, 55)
    for tooling in ("transport_height_mm", "vacuum_level_percent"):
        assert not hasattr(policy, tooling), (
            f"{tooling} is this cell's tooling, not a robot policy")


def test_inert_motion_keys_are_gone():
    """approach_height_mm / insert_height_mm were parsed, stored, and
    unit-tested, and no KukaClient method read either. A key that a test
    asserts and nothing uses is worse than a missing one."""
    for dead in ("approach_height_mm", "insert_height_mm",
                 "grasp_height_mm", "default_velocity_mm_s",
                 "safety_max_velocity_mm_s"):
        assert not hasattr(ExecutionConfig(), dead), dead

    # ...and unknown keys in the file are still tolerated, so older
    # configs carrying them keep loading.
    ExecutionConfig.from_dict({"motion": {"insert_height_mm": 2.0}})


def test_shipped_execution_yaml_declares_no_dead_safety_key():
    """`safety_max_velocity_mm_s: 250` sat in this file, was cited in the
    FDR's safety discussion, and was enforced by nothing. The frame has
    no velocity field, so the host cannot cap a speed; the cap is
    controller-side in krl_prog/routines.src."""
    from common.config import load_yaml

    raw = load_yaml("configs/execution.yaml")
    motion = raw.get("motion", {})
    for dead in ("safety_max_velocity_mm_s", "default_velocity_mm_s",
                 "insert_height_mm", "approach_height_mm",
                 "grasp_height_mm"):
        assert dead not in motion, f"{dead} is read by nothing"

    # The shipped file must still load, and its descriptive keys must
    # still describe the code.
    cfg = ExecutionConfig.from_dict(raw)
    assert cfg.transport_height_mm == 80.0


@pytest.mark.parametrize("bad, msg", [
    ({"command_length_bytes": 20}, "command_length_bytes"),
    ({"crc_polynomial": 0x8005}, "crc_polynomial"),
    ({"stop_category": 1}, "stop_category"),
])
def test_descriptive_kuka_keys_are_checked_not_ignored(bad, msg):
    """These three keys are claims about execution/protocol.py. They used
    to be read by nothing, so editing crc_polynomial changed no CRC."""
    with pytest.raises(ValueError, match=msg):
        ExecutionConfig.from_dict({"kuka": bad})


def test_status_frame_length_is_what_the_client_reads():
    assert STATUS_LEN == COMMAND_LEN == 16


def test_the_driver_exposes_a_supported_observation_hook():
    """A test that wants to see the bytes must not monkeypatch a private.

    ``tests/test_main_integration.py`` used to patch ``KukaClient._send``
    and parse what it caught, which pinned a private method and a wire
    format at once: a driver satisfying every public contract broke it.
    """
    seen = []

    def watch(driver):
        driver.add_frame_observer(lambda request, frame: seen.append(
            (request.kind, unpack_command(frame).op)))
        driver.move_to(WorkspacePoint(1, 2, 3))
        driver.halt()

    _record(watch)

    # Both halves of every send are visible: the neutral request the
    # caller made, and the sixteen bytes it became.
    assert (RequestKind.MOVE, OpCode.MOVE_TO) in seen
    assert (RequestKind.HALT, OpCode.HALT) in seen
