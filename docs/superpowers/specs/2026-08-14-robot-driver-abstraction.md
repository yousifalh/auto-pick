# Generalising the execution layer

**Date:** 2026-08-14 · **Base:** `19d8468` (1221 passing, 1 skipped)
**Inputs:** audit V (`2026-08-14-V-execution-seam.md`), audit U
(`2026-08-14-U-robot-interface-survey.md`), audit T
(`2026-08-14-T-kuka-conformance.md`).
**Result:** 1276 passing, 1 skipped (+55). Torch-free: 1236 (+55; nothing
added here imports torch).

---

## 0. What was built, and in what order

The two audits agree on one thing more strongly than on anything else:
the interface is the *weaker* artefact. Audit V's §6 says the
conformance suite "is worth more than the driver interface itself", and
its §4 says an ABC cannot enforce the one mechanism that closed this
project's three E-stop bypasses. So the order was suite, then base
class, then second driver — and the suite was allowed to dictate the
base class's shape rather than the other way round.

| # | Artefact | File | Lines |
|---|---|---|---:|
| 1 | the conformance suite | `tests/conformance.py` | 811 |
| 1a | KUKA harness + subclass | `tests/test_kuka_conformance.py` | 150 |
| 1b | JSON harness + subclass | `tests/test_json_conformance.py` | 219 |
| 2 | template-method base | `execution/driver.py` | 885 |
| 3 | second driver | `execution/json_driver.py` | 375 |
| 3a | second mock | `execution/mock_json_server.py` | 309 |
| — | the application, moved out of the driver | `execution/task.py` | 141 |

Deliberately **not** built, per the brief and per audit U §4: any
streaming (Shape B) interface, any real vendor driver, any ROS 2 or
RTDE integration. `execution/driver.py`'s module docstring states the
absence and the reason — the split is interpolator ownership, adding a
rate parameter to a delegated interface does not produce a streaming
one, and removing completion from it to accommodate streaming destroys
the only thing this project's callers need.

---

## 1. The conformance suite

**23 test functions, 26 tests after parametrisation, run twice — once
per driver. 58 collected in total**: 26 + 26 from the suite, plus 2
encoding-specific tests in the KUKA file and 4 in the JSON file.

### What it covers

| Group | Functions | Property |
|---|---:|---|
| lifecycle | 6 | context manager connects and closes; a closed port raises the driver's own fault type, not a socket error; a handshake refused because the controller is already stopped latches, raises `RobotEstop` and does not leak the descriptor — including through the `with` form, where `__exit__` never runs; a handshake refused with *any* other status escalates; silence retries exactly `max_retries` then halts; `handshake_timeout_ms` bounds the ack, not just the connect |
| the three bypass routes | 4 | a request that cannot be **encoded** (non-`ValueError` by construction on both drivers) is fatal, is not retried, and the halt goes out over the still-healthy transport; a mid-frame close is fatal and logged CRITICAL; a transport reset is fatal; an **undeliverable** halt is logged CRITICAL, does not replace the error that prompted it, and leaves `halt_delivered` false |
| retry and escalation | 7 | retry exhaustion sends the halt as the last thing on the wire; `max_retries` is honoured **exactly** (parametrised 1, 2, 4); an unparseable reply is transient; a controller-reported line fault is transient; the two protocol-mismatch codes are fatal without retry (parametrised); an unknown status code is not silently a timeout; the command deadline covers the whole frame |
| the latch | 2 | a controller reporting a stop closes, latches, raises, sends nothing further **onto the wire**, and refuses to reconnect; `halt()` is one-way, idempotent in effect, and counts attempts actually made |
| what the caller may assume | 4 | every returned code is in the actionable set, against the real mock; `validate` returns a `Reachability` and never raises; a lossy backend declares its losses; **the escalation cannot be overridden by a subclass** |

(6 + 4 + 7 + 2 + 4 = 23 functions; the three parametrised ones
bring it to 26 tests per driver.)

### The rework audit V asked for

The old assertions inspected `OpCode.ESTOP` on the wire — sixteen bytes,
unsatisfiable by any other encoding however correct the behaviour. They
now assert against two neutral observations:

* `RobotDriver.halt_attempts` — the driver's own count, inherited by
  every driver and implemented by none;
* `endpoint.wait_for_halt()` — whether the *far end* saw a halt,
  decoded by a harness the driver author supplies.

Both are needed. A driver that latches internally but never puts a halt
on a live wire has done half the job, and only the second observation
catches it; a driver whose link is already dead cannot put anything on
the wire, and only the first catches that it tried.

The adversarial server was generalised rather than duplicated:
`ScriptedEndpoint` takes `read_frame` and `describe` from the harness,
so the same machinery drives fixed 16-byte framing and length-prefixed
JSON. Reply sentinels (`None` for silence, `RESET`, `Hangup(payload)`)
replaced the KUKA-specific "a short reply means hang up" rule.

**What a driver author writes to opt in:** one `ConformanceHarness`
(~110 lines: framing, four reply builders, two targets, a mock) and a
three-line subclass. None of it is safety logic.

**Stated scope.** The suite assumes a *delegated* control shape and a
socket-like transport with a scriptable far end. It is not applicable to
a streaming backend, and says so in its own docstring.

---

## 2. The template-method base

`execution/driver.py`. A driver supplies six methods, four of which are
the seam proper:

| Hook | KUKA | JSON |
|---|---|---|
| `encode(request) -> bytes` | `pack_command`, integer mm | `json.dumps`, float metres |
| `decode(frame) -> RobotStatus` | `unpack_status` + CRC | length prefix + JSON |
| `send(frame)` | inherited from `TcpFrameDriver` | inherited |
| `recv(deadline) -> bytes` | `read_exactly(16, deadline)` | prefix, then N |
| `open_channel` / `close_channel` | inherited | inherited |

plus two optional ones whose defaults are the honest answers for a
backend that cannot do better: `validate() -> UNKNOWN` and
`gripper -> None`.

Everything else is owned by the base: `connect`, `close`, `__enter__`,
`__exit__`, the refusal to reconnect while latched, `_handshake` and its
retry policy, `execute` (the retry loop, the catch-all, the ESTOP latch,
the fatal-status path, the exhaustion path, the closed-set guarantee),
`halt`, `_request_halt`, `_fatal`, `move`, and `add_frame_observer`.

### Why not an ABC — and how it is enforced

Audit V §4 is the governing fact: all three historical bypasses were
closed by **catch-all ordering**, and an ABC cannot make a subclass
write one. So the base does not merely *provide* the escalation, it
**seals** it. `__init_subclass__` raises `TypeError` at class-definition
time if a subclass rebinds any of seventeen names:

```
connect close execute halt move __enter__ __exit__
halted estopped halt_attempts halt_delivered add_frame_observer
_handshake _read_status _request_halt _fatal _observe
```

This is asserted, not asserted-to: the conformance suite constructs a
subclass overriding `execute`, and another overriding `_request_halt`,
and requires both to be rejected — once per driver.

A second structural check ran alongside it, in
`test_the_second_driver_shares_no_encoding_with_the_first`:
`KukaClient.execute is JsonRobotDriver.execute` and
`KukaClient._request_halt is JsonRobotDriver._request_halt` — the same
function object, not two copies that happen to agree today.

### One behaviour change inside the sealed core

`execute` now escalates a status code that is neither actionable,
retryable nor fatal. It is unreachable while the three taxonomies stay
exhaustive over `RobotStatusCode` — and that is the point: a code added
to the enum without being classified now fires a halt here, instead of
reaching `main.py`'s four-way branch and being counted as a placement
failure.

---

## 3. The second driver

`execution/json_driver.py` + `execution/mock_json_server.py`. Both
docstrings open by saying they talk to no hardware and are not a step
toward doing so; they exist so the seam is measured rather than
asserted.

Every encoding decision was chosen to *differ*, because a second driver
that framed the same way would prove nothing:

| | KUKA | JSON |
|---|---|---|
| framing | fixed 16 bytes | 4-byte big-endian length prefix |
| integrity | CRC-16/MODBUS trailer | none; a JSON parse error |
| position | integer millimetres | float **metres**, no quantiser |
| orientation | **dropped** | unit quaternion, carried |
| redundancy token | **dropped** | carried, opaque |
| place Z | **dropped** | carried |
| cycle time | `uint16` ms, saturating at 65 535 | float seconds, unbounded |
| gripper | `VACUUM_ON` opcode, one bit | width, force, and a `holding` answer |
| halt | opcode `0x06` | `{"type": "halt"}` |
| a pick-and-place cycle | **two** stateful frames | **one** frame |
| `validate()` | `UNKNOWN` | real, against a declared envelope |
| unencodable coordinate | `CoordinateOutOfRange(struct.error)` | `NonFiniteCoordinate(ArithmeticError)` |

**It passes the identical suite.** `tests/test_json_conformance.py`
imports `RobotDriverConformance` unmodified and subclasses it; all 24
inherited test functions pass, first run, with no change to the suite
and no change to the base. The `validate()` row is the one that made
`Reachability`'s third value earn its place: one driver answers
`UNKNOWN` honestly and the other answers for real, and both are correct.

---

## 4. Defects fixed

**`z` is `int16` while `x`/`y` are `int32`, and the docstring lied.**
`protocol.py` now publishes per-axis limits (`X_MM_MIN`… `Z_MM_MAX`),
and `pack_command` raises `CoordinateOutOfRange` naming the axis, its
actual limit, *and the asymmetry* — `struct`'s own message
(`'h' format requires -32768 <= number <= 32767`) never said which field
it meant. It subclasses `struct.error`, **not** `ValueError`, on
purpose: a `ValueError` would be caught by the driver's transient tuple
and demoted to "retry the same impossible frame three times". Pinned by
`test_z_is_int16_while_x_and_y_are_int32` and
`test_an_out_of_range_z_is_fatal_not_retryable`. `wire_mm`'s docstring
now states the asymmetry instead of claiming int32 for all three.

**`aux_u16` wrapping at 65 535 ms.** `pack_status` saturates instead of
masking; `unpack_status` returns `cycle_ms_saturated`; the KUKA driver
puts a sentence in `RobotStatus.message` and logs a warning. The old
behaviour reported a 70-second cycle as 4 464 ms — a small, plausible,
wrong number entering the latency statistics `main.py` prints with
nothing to mark it. The field's directional double meaning (vacuum
percent outbound, cycle milliseconds inbound) is now stated in the
module docstring rather than left to be inferred.

**`PICK_AND_PLACE` is two stateful frames.** The concealing signature is
gone. Drivers now expose `pick_place_requests(...) -> tuple[Request,
...]` and the task *sequences* what it is given: KUKA returns two, named
`"MOVE_TO (latches the place XY)"` and `"PICK_AND_PLACE"`; the JSON
driver returns one. A shared `pick_and_place(pose) -> RobotStatus` would
have made those look identical from outside, and one of them would have
been wrong. `PickPlaceTask.run` additionally abandons a cycle if a
non-final step returns anything but OK/SUCCESS — running the pick after
the latch move failed would insert wherever the arm happened to be.

**`place.z_mm` computed then discarded; insert depth in three places.**
Two of the three are now one: `KUKA_CONTROLLER_INSERT_Z_MM` in
`execution/execution.py`, imported by `mock_kuka_server._INSERT_Z_MM`.
The third is the planner's and *cannot* be merged, because it is a
different thing (what the planner wants) from the other (what the
controller does) — so `PickPlaceTask._check_insert_depth` compares them
and logs once per task when they disagree, naming both numbers. The
value still does not reach the wire; there is nowhere to put it without
a frame-layout change and a version bump, which remains deliberately not
done. It is now announced instead of silent.

**`tests/test_main_integration.py` monkeypatching the private `_send`.**
Replaced by `RobotDriver.add_frame_observer(fn)`, a supported hook
called with the *neutral request* and the *encoded frame* just before
the transport. The test still parses with `unpack_command`, which is
correct — `main.py` runs the KUKA driver and the assertion is about what
those bytes carry — but it no longer pins a private method.

**The handshake accepting `OK` or `SUCCESS`.** Only `OK` now. Pinned
from both sides: a conformance test requires any other status to
escalate, and a KUKA test pins that the simulator answers `OK`.

**`OpCode.ESTOP` renamed to `OpCode.HALT`.** Value unchanged (`0x06`),
so no KRL dispatch table moves. The rename is legible, not silent: it is
documented at the enum, in the protocol module docstring, in the driver
module docstring, in FDR v3 §7.2, and pinned by a test that asserts
`"ESTOP" not in OpCode.__members__` — re-adding the old name as an alias
would make the renaming exactly the silent redefinition it was supposed
not to be. `RobotStatusCode.ESTOP` **keeps** its name: that is the
controller reporting a real Category-0 state, which is a different claim
and a true one. `_emergency_stop` became `_request_halt`; `estopped` is
retained as a documented alias of `halted`.

### From the survey, also done

* **Pose carries orientation.** `driver.Pose` is metres + unit
  quaternion + frame + tool + an **opaque** redundancy token, with each
  field's justification named in the docstring. The KUKA driver declares
  the loss in `Capabilities.lossy_notes` and warns when it is handed an
  orientation or a redundancy token it must drop.
* **Gripper is a capability, not an opcode.** `driver.Gripper`;
  `VacuumGripper` for this cell (which loudly ignores `width_m` and
  `force_n`, because a caller asking a suction cup for 30 N has the
  wrong tool), `JawGripper` on the second driver with a `holding`
  answer.
* **`validate() -> REACHABLE | UNREACHABLE | UNKNOWN`** on the
  interface. The KUKA driver answers `UNREACHABLE` only for coordinates
  the *wire* cannot carry — a real check nothing else in the system
  performs — and `UNKNOWN` otherwise, because it holds no envelope at
  all. `capabilities.validate_is_real` is `False` for it and `True` for
  the second driver.
* **Application concerns left the driver:** `vacuum`, the choreography,
  `transport_height_mm` and the insert depth are in `execution/task.py`.
  `ExecutionConfig` now splits into `driver_policy()` (vendor-neutral)
  and `TaskConfig` (this cell's tooling).

---

## 5. Change to `main.py`'s coupling surface

Audit V §2 lists six names and one method. Five of the six are
unchanged: `ExecutionConfig.from_dict`, the post-construction mutation of
`.host`/`.port`, `with KukaClient(exec_conf) as kuka:`,
`except RobotEstop:` before `except RobotFault:`, the closed status set,
and `run_in_thread`. **One changed:**

```python
-        status = kuka.pick_and_place(pose)
+        status = task.run(pose)
```

with two supporting edits: an import of `PickPlaceTask, TaskConfig`, and
one construction inside the existing `with` block:

```python
task = PickPlaceTask(kuka, TaskConfig.from_execution_config(exec_conf))
```

`_run_one_cycle`'s `kuka: KukaClient` parameter became
`task: PickPlaceTask`. Nothing else in `main.py` moved; the error message
in the closed-set `else` branch now names `execution.driver.RobotDriver`
rather than `execution.KukaClient`, because that is where the guarantee
now lives.

This change is the point rather than a cost. `pick_and_place(pose)` was
the signature audit V called "a trap with a green tick" — it is the one
an interface would naturally adopt and the one this protocol cannot
honour. `main.py` now asks the *task* to run a cycle and the task asks
the *driver* what a cycle costs in requests.

---

## 6. Test accounting

| File | Before | After |
|---|---:|---:|
| `tests/test_execution.py` | 39 | 36 |
| `tests/test_kuka_conformance.py` | — | 28 |
| `tests/test_json_conformance.py` | — | 30 |
| `tests/test_protocol.py` | 13 | 13 |
| `tests/test_mock_kuka_server.py` | 16 | 16 |
| **whole suite** | **1222** | **1277** |

`test_execution.py` kept what is genuinely KUKA — the millimetre
quantiser, the KRL structural coupling, the descriptive config keys —
and gained tests for the defects above; it lost the ~20 vendor-neutral
tests that moved into the suite. Its `_ScriptedServer`, `_ok`,
`_fast_cfg` and `_CapturedLog` helpers (~100 lines) are gone, replaced
by reuse of the conformance harness.

One new quantiser test earns a mention.
`test_wire_mm_survives_the_metre_round_trip` exists because the
interface pose is in SI metres, so every millimetre value now makes a
`/1000` and `*1000` round trip before quantisation and can land one ulp
off an exact half. Without the nanometre snap added to `wire_mm`, a tie
would break in whichever direction the floating-point noise fell — a
silent, magnitude-dependent reintroduction of exactly the bias
round-half-to-even was chosen to remove. Twelve values on a 0.125 mm
grid were measurably affected before the snap.

---

## 7. What was judged not worth doing, and why

**A streaming (Shape B) interface.** Audit U §4 is decisive and this
project needs only Shape A. Not built; the absence and its reasoning are
in `execution/driver.py`'s docstring so the next reader does not
rediscover the question.

**A waypoint-sequence goal type (audit U's R1).** The survey is right
that ROS 2's native unit is a trajectory and that a single-pose
interface makes a ROS 2 backend dishonest. But nothing in this
repository has more than one waypoint to give: the planner emits a
`PickPlacePose`, and a `waypoints: tuple[Waypoint, ...]` that is always
length one is an untested generalisation carried by every driver.
`Request` carries a single `target`, and the *sequence* lives one level
up, in `pick_place_requests` — which is where this protocol's actual
multi-step structure was hiding anyway. If a trajectory-native backend
is ever added, that is the seam to widen.

**A separate state channel (audit U's R5).** Real, and a genuine
deficiency: this project's single blocking socket cannot be read while a
motion is in flight. But fixing it means a second socket in every driver
and a threading model in the base class, and nothing in this repository
reads robot state during a motion. Recorded as a known limitation, not
built.

**A listen-mode driver (audit U's R11).** ABB EGM has the controller
dial out, so `TcpFrameDriver` is wrong for it. Its docstring says so and
says a sibling class rather than a subclass would be needed. Building
one with no ABB to talk to would be speculation.

**Merging the third insert depth.** The planner's
`place_insert_height_mm` and the controller's descent are different
quantities and merging them would assert an agreement the wire cannot
carry. Compared and announced instead.

**A `RobotDriver` ABC with a rich method set.** Audit V's closing
paragraph, followed as written.

### The 29 % figure, revisited

Audit V measured ~368 of 1 265 Python lines as reusable and warned that
"29 % reusable is not a large fraction". That was the right number and
the wrong unit. The 368 lines are where every execution-layer defect in
this project's history was found and fixed, and the second driver —
375 lines of encoding plus a 309-line mock — reimplemented **none** of
them: the two drivers share `execute` and `_request_halt` as the same
function objects, verified by identity. The reuse that mattered was not
29 % of the lines, it was 100 % of the escalation.

The suite is the artefact that would survive this refactor being
reverted, and it is the one that stops the fourth bypass route.
