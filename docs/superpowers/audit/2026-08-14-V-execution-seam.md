# Audit V — where the vendor boundary actually falls in `execution/`

**Scope.** `execution/execution.py` (548), `execution/protocol.py` (267),
`execution/mock_kuka_server.py` (450), `execution/krl_prog/` (1014:
`routines.src` 333, `laptop_comm.src` 583, `laptop-comm.xml` 98).
HEAD `19d8468`. Read-only; nothing staged or committed.

**Question.** Before designing a `RobotDriver`, map the seam as written.

**Headline.** The commissioning brief's two guesses are half right and the
half that is wrong is the expensive half. The lifecycle *is* vendor-neutral —
more so than expected: `execution.py` contains roughly **eight lines that are
genuinely KUKA**, all of them strings and a default IP. But the encoding is
not merely "the wire format"; the encoding has **eaten the command model**.
`pick_and_place` is a two-frame stateful protocol, the place Z does not exist
on the wire, and z is int16 while x/y are int32. A `RobotDriver` whose method
signature is `pick_and_place(pose) -> RobotStatus` is a signature this driver
cannot honestly implement and does not today: it drops a field the planner
computed. The seam is real and it is *not* between "KUKA" and "not KUKA"; it
is between **policy** (retry, escalate, latch, deadline) and **command model**
(what a pick even is), and only the first is portable.

---

## 1. Classification, with line ranges

### 1.1 `execution/execution.py` — 548 lines

| Category | Lines | Ranges |
|---|---:|---|
| vendor-neutral | ~247 | 1–39 (contract docstring), 70–107, 213–224, 228–258, 260–313 (policy half), 315–328, 378–417, 419–487 |
| encoding-specific | ~106 | 54–60, 64–67, 110–136, 168–187, 489–492, 494–524 |
| KUKA-specific | **~8** | 154 (`host = "172.31.1.147"`), 249–251 ("EthernetKRL channel"), 289 ("KUKA handshake OK"), 417 ("EthernetKRL failure"), 213 (class name) |
| application-specific | ~45 | 160–161, 202–207, 338–341 (`vacuum`), 343–376 (`pick_and_place`) |
| scaffolding (imports, `__all__`, dataclass boilerplate) | ~142 | 40–53, 141–167, 189–208, 525–548 |

Function by function:

* **`RobotFault` / `RobotEstop` 70–86** — vendor-neutral. Two exception types
  and a documented subclass relationship (`RobotEstop <: RobotFault <:
  RuntimeError`). Nothing here mentions a wire or a controller.
* **`_RETRYABLE_STATUS` 89–92, `_FATAL_STATUS` 96–99, `_TRANSIENT` 101–107** —
  the *taxonomy* is vendor-neutral (transient / fatal / stopped) and is the
  single most valuable thing in the file. Its *membership* is encoding-derived:
  `VERSION_MISMATCH` and `UNSUPPORTED_COMMAND` are protocol concepts, and 107's
  `(socket.timeout, ValueError)` is a TCP-and-`ProtocolError` fact.
* **`wire_mm` 110–136** — encoding-specific despite appearances. The
  round-half-to-even reasoning (117–133) is a general numerical argument, but
  the *existence* of a quantiser is a wire fact: this wire carries integer
  millimetres. A driver over a float-valued RPC has no `wire_mm`.
* **`ExecutionConfig` 141–208** — three-way split. `host`/`port`/three
  timeouts/`max_retries` (154–159) vendor-neutral; `transport_height_mm` and
  `vacuum_level_percent` (160–161, 202–207) **application-specific**;
  the cross-checks at 168–187 (`command_length_bytes`, `crc_polynomial`,
  `stop_category`) encoding-specific.
* **`estopped` 221–224, `connect` 228–258, `close` 315–321, `__enter__` /
  `__exit__` 323–328** — vendor-neutral. The brief's guess holds. Note 236–239:
  the latch is checked *in `connect`*, so the refusal to reconnect is a
  lifecycle property, not a command property.
* **`_handshake` 260–313** — vendor-neutral policy (retry, sleep, escalate,
  the ESTOP-refusal branch 296–304), encoding-specific body (274's
  `pack_command(OpCode.HANDSHAKE)`).
* **`move_to` 332–336** — the signature is vendor-neutral; the body is three
  `wire_mm` calls and an opcode.
* **`vacuum` 338–341** — **application-specific**, not vendor. A suction
  end-effector with a percentage level. A two-finger gripper has no `vacuum`
  and no percent; a driver interface carrying `vacuum(on: bool)` would be
  imposing this cell's tooling on every future arm.
* **`pick_and_place` 343–376** — **application-specific**, and the densest
  concentration of leaks in the repository (see §3.1–§3.3).
* **`estop` 378–389, `_emergency_stop` 393–412, `_fatal` 414–417** —
  vendor-neutral. 402–412's "log CRITICAL rather than swallow, and do not let
  the failed stop replace the original error" is a *property*, not plumbing.
* **`_cmd_and_wait` 419–487** — vendor-neutral, and the heart of the matter.
  69 lines: latch check (434–437), retry loop (440–451), the **catch-all**
  (452–457), ESTOP latch (459–465), fatal-status (467–470), retryable-status
  (472–480), exhaustion (484–487).
* **`_send` 489–492** — TCP.
* **`_recv_status` 494–544** — mixed, and mis-splittable. The *whole-frame
  deadline* (508–517) is a vendor-neutral property. The loop that implements it
  only works because frames are **fixed-length** (511's `while len(buf) <
  STATUS_LEN`); a length-prefixed or RPC transport cannot inherit this code,
  only the property. 526–534 (unknown code → `ValueError`, not `TIMEOUT`) is
  vendor-neutral and safety-relevant.

**The surprise.** Strip five strings and one IP address and `execution.py`
never mentions KUKA. The brief expected the vendor boundary to run through
this file. It does not — the file is `TcpFixedFrameClient` with KUKA in the
comments. The real boundary is one level down, in what the frames *mean*.

### 1.2 `execution/protocol.py` — 267 lines

**100 % encoding-specific**, with one embedded neutral idea.

* 1–61 docstring — encoding, plus 21–43 which is *KUKA justification*
  (EKI `BinaryFixed`, `<RAW><ELEMENT …Size="16"/>`), the only KUKA content.
* 71–101 `ProtocolError` / `FrameLengthError` / `CrcError` /
  `VersionMismatch` / `UnsupportedOpCode` — **31 lines whose taxonomy is
  vendor-neutral**: line fault vs firmware mismatch vs unimplementable command
  is the distinction `_RETRYABLE_STATUS` / `_FATAL_STATUS` keys on. Any second
  driver needs the same three-way split; only the names of the wire failures
  change.
* 103–124 constants and `OpCode`, 129–149 `crc16_modbus`, 154–248 pack/unpack —
  encoding, all of it.

Reusable to a second driver: the 31-line taxonomy. Nothing else. There is no
vendor-neutral line in the codec, which is correct — a codec that were
vendor-neutral would be doing nothing.

### 1.3 `execution/mock_kuka_server.py` — 450 lines

| Category | Lines | Ranges |
|---|---:|---|
| vendor-neutral | ~105 | 106–108, 112–151, 382–417, 422–450 |
| encoding-specific | ~153 | 52–73, 244–328, 332–377 |
| KUKA / robot-model-specific | ~34 | 76–103 (`REACH_MM = 706.0`, `Z_MIN_MM`, `Z_MAX_MM`), 129–134 (`in_envelope`), 162–166 |
| application-specific | ~65 | 89 (`_INSERT_Z_MM = 2`), 97 (`_GRASP_BAND_MM`), 177–239 (`pick_and_place`) |
| scaffolding | ~93 | docstring 1–39, imports, blanks |

Two classifications worth defending. `latch_estop` **136–151** is
vendor-neutral: "a Category-0 stop is idempotent, never cleared, and drops the
vacuum" is IEC 60204, not KUKA. And `_RobotState.pick_and_place` **177–239** is
application-specific in a way easy to mis-file as vendor: steps 1–7 (216–235)
are *this cell's cartridge-insertion choreography* — approach at 80, descend,
suck, lift to 80, traverse, insert at `_INSERT_Z_MM`, blow off, retract to 80 —
mirroring `routines.src`. A different arm doing this same task would run the
same seven steps. A KUKA doing a different task would not.

### 1.4 `execution/krl_prog/` — 1014 lines

**100 % KUKA-specific, 0 % portable**, and that is a property of the *language*
rather than of the logic. `routines.src` 80–86 (`BAS(#INITMOV,0)`, `$TOOL =
TOOL_DATA[1]`, `$BASE = BASE_DATA[1]`), 129–139 (`$POS_ACT`), 196–200
(`$VEL_AXIS[]`), 232–249 (`$ANOUT`, `$OUT`), 278–333 (`PTP`/`LIN`/`$IN[10]`/
`$VEL.CP = 0.150`); `laptop_comm.src` 160–324 (`EKI_Init` / `EKI_Open` /
`EKI_CheckBuffer` / `EKI_GetString` / `EKI_Send` / `EKI_Close` / `EKI_Clear`);
`laptop-comm.xml` all 98 lines (the `<RAW><ELEMENT Tag="Buffer" …>` channel
declaration).

Note that 360–532 of `laptop_comm.src` (`ByteAt` / `PutByte` / `Int32At` /
`Int16At` / `Uint16At` / `PutInt32` / `PutInt16` / `Crc16Modbus`) are a
*re-implementation of `protocol.py` in KRL* — ~170 lines that are conceptually
encoding-specific but practically unusable anywhere else. This is the third
independent implementation of the same frame layout in the repository (Python
codec, KRL codec, and the simulator's dispatch table), none of which is
generated from the other two.

---

## 2. The true coupling surface

Everything outside `execution/` that touches it, exhaustively:

**Production (`main.py`) — six names:**

| Touch | Site |
|---|---|
| `ExecutionConfig.from_dict(exec_cfg)` | `main.py:326` |
| `exec_conf.host = host` / `.port = port` (**mutation after construction**) | `main.py:327–328` |
| `with KukaClient(exec_conf) as kuka:` | `main.py:388` |
| `kuka.pick_and_place(pose) -> RobotStatus` | `main.py:600` |
| `except RobotEstop:` before `except RobotFault:` | `main.py:395`, `408` |
| `status.code in {SUCCESS, PICK_FAILED, PLACE_FAILED}` else raise | `main.py:602–627` |
| `run_in_thread(host, port, drop_prob, ms_per_100mm)` | `main.py:274–286` |

**Not touched anywhere outside `execution/`:** `move_to`, `vacuum`, `estop`,
`estopped`, `connect`, `close`, `wire_mm`, `OpCode`, `pack_command`,
`unpack_status`, `crc16_modbus`, `RobotStatus.current_pose`,
`RobotStatus.cycle_time_ms`, `RobotStatus.message`.

**So the interface an abstraction must cover is far smaller than the class:**
a constructor, a context manager, **one** method, two exception types, and a
closed status set. `move_to` and `vacuum` are dead weight from the outside
world's point of view — used only by tests. That is a strong argument for a
*narrow* `RobotDriver`: `__enter__` / `__exit__` / `execute(pose) -> Status`,
and nothing else. Any richer interface is speculation.

**Two coupling defects on that surface:**

1. `main.py:327–328` mutates `ExecutionConfig` after `from_dict`, so config is
   not a value. A driver *factory* taking a frozen config breaks `main`.
2. `tests/test_main_integration.py:325–331` monkeypatches **`KukaClient._send`**
   — a private method — and imports `unpack_command` / `OpCode` to assert what
   reached the wire. The integration test is bound to the 16-byte format
   through a private hook. A second driver breaks this test even if it
   satisfies every public contract. This is the surface's only genuine
   entanglement with `main`, and it is in the test, not the code.

---

## 3. Where the abstraction already leaks

Twelve. The known one is #1.

**3.1 Three-DoF wire, six-DoF arm.** `protocol.py:5–14` carries x, y, z only.
`routines.src:129–139` fills the rest from `$POS_ACT` — the whole E6POS copied,
X/Y/Z overwritten (135–138). The host cannot express orientation and the
planner cannot either: `common/types.py:299–307` `PickPlacePose` has no
orientation field. So the *task model*, not just the wire, is 3-DoF. A second
arm whose driver wants a tool attitude has nowhere to get one, and 95–128 of
`routines.src` says plainly that the fixed tool-down constant "would have to be
invented; this repository holds no taught cell data".

**3.2 `place.z_mm` is computed, validated, and discarded.**
`execution.py:343–376` states it. The planner *does* compute it
(`PlannerConfig.place_insert_height_mm`), `PickPlacePose.place` carries it, and
`pick_and_place` never packs it — 364–369 sends `place.x`, `place.y`,
`cfg.transport_height_mm`. The real insert depth therefore lives in **three
places that cannot be checked against each other**: `routines.src:319`
(`LIN place_pos`), `mock_kuka_server.py:89` (`_INSERT_Z_MM = 2`), and the
planner's config. A second driver picks a fourth and nothing notices.

**3.3 `PICK_AND_PLACE` is a two-frame stateful command.** The place XY is
latched by the preceding `MOVE_TO` (`execution.py:364–369`), and the controller
uses "whatever (x, y) it was in when the subroutine begins"
(`mock_kuka_server.py:199`, `place_x, place_y = self.x, self.y`). Nothing in
the frame says so. A driver that implements `execute(pose)` as one message —
the natural thing — places correctly by accident or not at all, silently. This
is the leak that most threatens an interface, because the interface's signature
(`pose in, status out`) actively conceals it.

**3.4 No sequence number.** `execution.py:29–33`: a late reply is mis-paired
with the *next* command. Combined with 440–451, the retry loop re-sends motion
opcodes, and `PICK_AND_PLACE` is **not idempotent**. This is a hazard the
interface must *state*, or a second driver's author assumes the retry is safe.

**3.5 No velocity field.** `configs/execution.yaml` records that
`safety_max_velocity_mm_s: 250` was deleted because nothing could enforce it;
the cap is `routines.src:293` `$VEL.CP = 0.150`. Any `RobotDriver` method
taking a speed argument would be a lie for this driver.

**3.6 The workspace check is on the wrong side of the seam.** `KukaClient`
performs **no envelope check at all**. The simulator does
(`mock_kuka_server.py:129–134`, raising at 162–166 and latching at 309/323) and
the planner does (`configs/planning.yaml:93`, `±350 mm`, enforced in
`plan/scene.py:620–635`). On real hardware the KRC's software limits do. So
"the commanded pose is reachable" is guaranteed by *three different components,
none of them the client* — and if a second controller lacks software limits,
the driver inherits no protection whatsoever. Do not let a `RobotDriver` doc
claim this.

**3.7 Quantisation happens after validation.** `tests/test_execution.py:727–729`
records it: `WorkspaceBounds.require` validates the float, `wire_mm` then
displaces it by up to 0.5 mm. The value checked is not the value commanded. The
bound is a *wire* property (§1.1), so a driver with coarser resolution silently
widens the planner's margin violation.

**3.8 z is int16, x/y are int32.** `protocol.py:109`, `_BODY_FMT = ">BBiihH"`.
An out-of-range z raises `struct.error` at ±32 768; x and y at ±2³¹. The
asymmetry is invisible to the host, and `wire_mm`'s docstring
(`execution.py:131–134`) says only "outside int32" — it is wrong for z.

**3.9 `aux_u16` is two different fields.** Outbound it is vacuum percent;
inbound it is cycle time (`protocol.py:11`, `16–19`, masked `& 0xFFFF` at 225).
A cycle over **65 535 ms wraps** and reports a small wrong number into the
statistics `main.py` prints. One field, two meanings, one direction each — a
driver mapping it to a typed struct will get this wrong.

**3.10 Two of the four fatal status codes are emitted by no real controller.**
`tests/test_execution.py:705–716` pins that `routines.src` returns neither
`UNSUPPORTED_COMMAND` nor `VERSION_MISMATCH`. The `_FATAL_STATUS` branch
(`execution.py:467–470`) is exercised only against the simulator. Its
conformance value is real but it is testing an agreement that only the
simulator has signed.

**3.11 The handshake accepts `OK` *or* `SUCCESS`** (`execution.py:287`) while
the simulator only ever answers `OK` (`mock_kuka_server.py:304`) and the KRL
answers neither. The accepted set is wider than any implementation emits — a
harmless-looking tolerance that a conformance suite cannot pin.

**3.12 `RobotStatus.cycle_time_ms` is typed float, carries integers**, and
`current_pose` is float-typed but whole-millimetre by construction
(`execution.py:536–542` says so). Type says one thing, wire says another.

---

## 4. Safety behaviours: interface property, or driver detail?

The project has already had an E-stop with three bypass routes — `struct.error`,
mid-frame close, `ConnectionResetError` (pinned by
`tests/test_execution.py:238`, `257`, `282`). **The mechanism that closed all
three was a catch-all ordering**, `execution.py:444–457`: catch `_TRANSIENT`
first, then `except Exception: self._fatal(...)`. This is the governing fact for
the whole design question.

| Behaviour | Site | Verdict |
|---|---|---|
| ESTOP status → latch, close, raise `RobotEstop` | 459–465 | **Interface.** `main.py:395` aborts the run and *withholds statistics* on it. A driver that returned ESTOP as a status would silently restore the pre-fix behaviour. |
| Latch survives reconnect | 236–239, 385, 403 | **Interface.** One-way door; a driver cannot be trusted to re-derive it. |
| Handshake refused with ESTOP → latch + raise | 296–304 | **Interface.** The controller-already-stopped case. |
| Retries exhausted → E-stop, then raise | 484–487 | **Interface**, as the property "no command returns without either an actionable status or an attempted stop". |
| Any unforeseen exception → E-stop, then raise | 452–457 | **Interface, and structurally.** See below. |
| Undeliverable E-stop → CRITICAL, original error preserved | 393–412 | **Interface.** Exactly the property a second driver omits without noticing, because omitting it makes the tests *pass more quietly*. |
| Unknown status code → `ValueError` (retry, then escalate) | 526–534 | **Interface.** `main.py:617–627`'s closed-set guarantee rests on it. |
| Whole-frame deadline | 508–517 | **Interface as a property** ("a command completes or faults within `command_timeout_ms`"); **driver detail as code** (the byte loop is fixed-frame-specific). |
| Workspace envelope | *not in the client* | **Neither.** It is the planner's and the controller's. An interface must not claim it — see §3.6. |
| Retry re-sends non-idempotent motion | 440–451 | **Interface, as a stated hazard**, not a guarantee. |
| Best-effort semantics of `estop()` | 378–389 | Driver detail (the frame), interface property (it is attempted). |

**The structural conclusion.** A `RobotDriver` **ABC cannot enforce a
catch-all in a subclass.** If `execute()` is driver-implemented, driver #2's
author writes `try/except socket.timeout` and reopens bypass route #4 —
identically to how routes #1–#3 opened. The only design that closes this by
construction is a **base class owning `_cmd_and_wait` as a template method**,
with drivers supplying only `send_frame(bytes)` / `recv_frame() -> bytes` (or
`encode(cmd)` / `decode(frame)`) and *never* the escalation. Every safety
behaviour in the table above then lives in code no driver can edit.

That is not a stylistic preference; it is the difference between an interface
that closes the defect class and one that re-opens it with a fresh coat of
paint.

---

## 5. What a second driver reimplements, what it inherits

Of the **1,265 Python lines** (548 + 267 + 450):

| Inheritable as-is | Lines | What |
|---|---:|---|
| `RobotFault` / `RobotEstop` | 17 | `execution.py:70–86` |
| failure taxonomy + `_TRANSIENT` ordering | 19 | 89–107 |
| lifecycle (`connect` shape, `close`, `__enter__`/`__exit__`, `estopped`) | 57 | 213–258, 315–328 |
| escalation (`estop`, `_emergency_stop`, `_fatal`) | 40 | 378–417 |
| `_cmd_and_wait` template | 69 | 419–487 |
| handshake retry policy | ~30 | 260–313 (policy half) |
| `ProtocolError` taxonomy | 31 | `protocol.py:71–101` |
| simulator shell, latch, CLI | ~105 | `mock_kuka_server.py:106–151`, 382–450 |
| **Total** | **~368** | **29 % of the Python** |

Must be rewritten: the codec (267), the framing I/O (`_send`, `_recv_status`,
~55), `wire_mm` (27), the config cross-checks (20), the command model
(`vacuum`, `pick_and_place`, 38), the simulator's dispatch and motion model
(~250), and **all 1,014 lines of KRL/XML**.

**Honest figure: ~368 of 1,265 Python lines are genuinely reusable (29 %); ~16 %
of the 2,279-line total.** But the 368 is not evenly valuable — the 69-line
`_cmd_and_wait` plus the 40-line escalation is where every one of this
project's execution-layer defects was found and fixed, and it is exactly the
part that a from-scratch second driver would get wrong. Extracting 109 lines to
guarantee that is worth more than the line count suggests.

---

## 6. Test coupling and the conformance suite

`tests/test_execution.py` 907 lines / 36 tests; `test_mock_kuka_server.py` 269 /
13; `test_protocol.py` 121 / 13.

| Group | Tests | Lines | Portable? |
|---|---:|---:|---|
| lifecycle + happy path (178–233) | 5 | 56 | yes, except `vacuum` (§1.1) |
| the three E-stop escape routes (238–320) | 4 | 83 | **yes — the core of the suite** |
| retry / escalation / latch (323–505) | 9 | 183 | yes |
| whole-frame deadline (507–526) | 1 | 20 | property yes, mechanism no |
| KRL structural coupling (588–716) | 6 | 129 | **no — pure KUKA** |
| `wire_mm` quantisation (719–825) | 4 | 107 | property yes, ±0.5 mm bound no |
| connect failure (827–836) | 1 | 10 | yes |
| config (841–907) | 6 | 67 | 3 portable, 3 KUKA (`crc_polynomial` etc.) |

**~20 of 36 tests (~450 lines) assert vendor-neutral behaviour.** In
`test_mock_kuka_server.py`, the latch tests (132–166) are neutral; the envelope
tests (101–131) are robot-model; framing and code-distinction (167–223,
257–268) are encoding. `test_protocol.py` is 100 % encoding.

**A conformance suite is extractable — with one caveat that must be designed
for.** The assertions are neutral; the *harness* is not. `_ScriptedServer`
(47–141), `_ok()` (143–144) and `_fast_cfg` (147–151) all speak the 16-byte
format, and roughly ten of the twenty portable tests assert by **inspecting
`OpCode.ESTOP` on the wire** (e.g. 248, 252, 332–338). So the suite cannot be
lifted; it must be re-expressed against a driver-supplied observation hook —
"was a stop attempted?" and "what did you emit?" — rather than opcode
inspection. That is ~100 lines of harness rework plus a small driver-side test
contract.

**Verdict: yes, and it is the most valuable artefact of this refactor.** Twenty
tests that pin *"every route out of a command either returns an actionable
status or attempts a stop"* is precisely the thing that stops driver #2 from
shipping the fourth bypass route. It is worth more than the driver interface
itself.

---

## 7. Recommendation

Not "too entangled to split" — but **not the split the brief imagined**, and
worth doing only in this shape:

1. **Extract the policy core, not a `RobotDriver` interface.** A base class
   owning `connect` / `close` / `__enter__` / `__exit__` / `_cmd_and_wait` /
   `_emergency_stop` / `_fatal` / the taxonomy — ~250 lines — with drivers
   supplying `encode` / `decode` / `send_frame` / `recv_frame` **only**.
   Anything that lets a driver implement the escalation re-opens the defect
   class (§4).
2. **Keep the interface at one method.** `main.py` uses `pick_and_place` and
   nothing else (§2). `move_to` and `vacuum` are test-only and
   application-specific; promoting them to an interface exports this cell's
   tooling to every future arm.
3. **Fix §3.3 before abstracting, not after.** The two-frame stateful place is
   concealed by exactly the signature an interface would adopt. Either the
   frame gains a second coordinate triple (protocol version bump) or the
   interface documents the sequence as mandatory — but shipping
   `execute(pose)` over a latched-XY protocol is a trap with a green tick.
4. **Do not put the workspace check in the interface** (§3.6) and **do not put
   a velocity parameter in it** (§3.5). Both are enforced elsewhere or nowhere;
   an interface that names them makes a promise no driver keeps.
5. **Build the conformance suite first.** It is independently valuable, it
   survives if the refactor is abandoned, and writing it will surface whether
   the `_ScriptedServer` harness can be parameterised at all.

The one thing not worth doing is a `RobotDriver` ABC with a rich method set,
because the vendor boundary is not where the class boundary is: `execution.py`
is already vendor-neutral to within eight lines, and what is *not* portable is
the command model that a rich interface would enshrine.
