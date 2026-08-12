# Audit F — the robot-facing execution path, and configuration coherence

**Date:** 2026-08-12 · **Tree:** `d:\dev\auto-pick` @ `fa7a4f0` (`feat/blender-synth-dataset`)
**Scope:** Part 1 — `execution/protocol.py`, `execution/execution.py`, `execution/mock_kuka_server.py`,
`execution/krl_prog/`, and the FDR sections that describe them.
Part 2 — all 21 files in `configs/`, `recog/sync_config.py`, `recog/synth3d/config.py`.
**Mode:** read-only. Nothing in the repo was modified, staged or committed. Every script written
for this audit lives in the session scratchpad and imports the repo without touching it.

Throughout, **[EXEC]** marks a claim established by running code and reading its output;
**[READ]** marks a claim inferred from reading source. Where a claim concerns how a *real* KUKA
controller behaves — which I obviously could not run — it is marked **[SPEC]** and is inference
from the KR 6 R700's published envelope and ordinary industrial-controller practice, not
measurement.

**Severity is rated by realistic consequence.** The execution path is the only code in this
repository that would move a physical arm, so it is held to a higher standard than the research
scripts. Where a defect is currently harmless *because* the robot is mocked, that is stated.

---

## Part 1 — the execution path

### 1.0 What is correct, established by execution

Three things the brief asked me to check are **right**, and I want that on the record before the
findings, because the failures below are failures of the surrounding machinery, not of the wire
format.

**The CRC is correct.** [EXEC] I wrote three implementations that share no code with
`protocol.crc16_modbus` and no code with each other:

1. a byte-at-a-time table-driven form, with the 256-entry table generated independently;
2. an MSB-first shift register over the **non-reflected** polynomial `0x8005` with explicit
   input-byte and output-word reflection — i.e. the algebraically-equivalent mirror construction;
3. a bit-serial LSB-first register written the naive way.

All three agree with `crc16_modbus` on the published check value for CRC-16/MODBUS —
`crc("123456789") == 0x4B37` — and on **20 000 random byte strings of length 0–39, with zero
mismatches**. Polynomial `0xA001`, init `0xFFFF`, no output XOR, reflected in and out: exactly as
`protocol.py`'s docstring and FDR §7.1 claim. The existing `test_crc16_modbus_known_vectors`
vectors (`0x0A84`, `0xFFFF`, `0x40BF`) are also correct. This was checked against independent
implementations, not against the mock — both could have been wrong in the same way, and they are
not.

**The frame layout matches its specification.** [EXEC] Packing a distinctive payload and dumping
the bytes gives:

```
01 | 01 | 01 02 03 04 | ff ff ff fe | 05 06 | 07 08 | 03 1e
ver  op   x = BE i32    y = BE i32    z=BEi16 aux=BEu16  crc = LE u16
```

16 bytes; body 14; version at 0, opcode at 1; X and Y signed 32-bit big-endian; Z signed 16-bit
big-endian; aux unsigned 16-bit big-endian; CRC over bytes 0..13 in **little-endian** order — the
MODBUS convention of a low-byte-first CRC trailer on an otherwise big-endian frame. This is
unusual-looking but it is what both the module docstring and FDR §7.1 specify, and it is what the
code does. Commands and status packets share the layout, as documented.

**Retry exhaustion does send the E-stop.** [EXEC] Against an adversarial server, both documented
exhaustion routes behave as the docstring claims — the client sends `HANDSHAKE, MOVE_TO, MOVE_TO,
MOVE_TO, ESTOP` and then raises `RuntimeError`:

| provocation | opcodes the server observed | E-stop sent |
|---|---|---|
| every status frame has a corrupt CRC | `HANDSHAKE, MOVE_TO ×3, ESTOP` | yes |
| server silent after handshake (timeouts) | `HANDSHAKE, MOVE_TO ×3, ESTOP` | yes |
| stream desynchronised by 3 stray bytes | `HANDSHAKE, MOVE_TO ×4, ESTOP` | yes |

So the headline safety promise holds on the two paths it was written for. The problem is the
paths it was **not** written for, which is §1.2.

### 1.1 FINDING (high) — there is no sequence number, and the timeout retry therefore duplicates motion commands and mis-pairs replies

The brief asks about sequence numbering. **There is none.** [EXEC] The 14-byte body is
`version, op, x, y, z, aux` and nothing else; there is no sequence field, no transaction id, and
no other means of correlating a status frame with the command that provoked it. The client's only
correlation rule is positional: the next 16 bytes to arrive are assumed to answer the last command
sent.

That assumption breaks the moment a reply is merely **late** rather than lost, which is the
ordinary failure mode of a busy motion controller. Verified [EXEC]: with
`command_timeout_ms = 300`, a server that answered command #1 after 450 ms produced this —

- the client sent `MOVE_TO(222,222,222)`, timed out, and **re-sent `MOVE_TO(222,222,222)`**;
- it then read the *stale* first reply and returned it as the result of the second send:
  `SUCCESS, pose (111,111,111), cycle_time 999 ms`;
- no exception, no warning beyond the routine "attempt 1/3: timed out", and the caller received a
  pose the robot was never commanded to.

Two consequences, both of which matter on real hardware:

1. **A motion command is executed twice.** `MOVE_TO` re-sent is idempotent in position but not in
   time; `PICK_AND_PLACE` re-sent is *not* idempotent at all — it is a second grasp attempt at the
   same coordinates, with the gripper possibly already holding a cell. `_cmd_and_wait` retries
   every opcode identically, including `PICK_AND_PLACE`. [READ]
2. **The client's model of where the arm is diverges from reality, silently, and stays diverged.**
   Every subsequent reply is off by one, because the extra frame sits in the socket buffer forever.

This is not fixable by tuning the timeout. It needs either a sequence/echo field in the frame
(there is no spare room in 16 bytes without a version bump) or a rule that a timed-out *motion*
command is never retried — only re-synchronised or escalated. The current design chose retry
without correlation, which is the one combination that can duplicate physical motion.

**Currently harmless** because the mock is local and never late. On a real controller under load
this is the first thing that would bite.

### 1.2 FINDING (high) — three failure routes bypass the E-stop entirely

`_cmd_and_wait` catches `(socket.timeout, ValueError)`. Everything else escapes the loop
uncaught: no retry, no E-stop, no `close()`, straight out through `main.py`, which has no handler
for it. Verified by execution, one adversarial server per row:

| provocation | what escapes | retried | E-stop |
|---|---|---|---|
| server closes after sending 7 of 16 status bytes | `ConnectionError: Robot closed connection` | no | **no** |
| connection reset (RST) after a bad-CRC reply | `ConnectionResetError` (WinError 10054) | no | **no** |
| coordinate outside int32 / z outside int16 | `struct.error` | no | **no** |

The third is the sharpest. `issubclass(struct.error, ValueError)` is **`False`** [EXEC] — I
verified this rather than assuming it, because it looks like it ought to be true. So
`pack_command` raising on an out-of-range coordinate is not a "CRC or parse error" as far as the
except clause is concerned. A planner that ever emits `z > 32767 mm`, or an x beyond ±2.1 km,
raises out of the client mid-cycle **while the socket is still perfectly healthy and an E-stop
could have been sent**. That is precisely the "documented but not sent" pattern the brief names,
and it is the one instance of it where the socket is alive and the omission is a pure oversight
rather than a physical impossibility.

The two `ConnectionError` rows are different in character: once the socket is gone, no E-stop can
be transmitted, so the client cannot do what the docstring promises. But it also does not
*acknowledge* that. It raises a bare exception describing a socket, not a robot. The honest
behaviour when the link dies mid-command is to treat the arm as being in an unknown state and say
so — the arm may be mid-`PICK_AND_PLACE` with the vacuum on. `main.py` has no `except` around
`kuka.pick_and_place(pose)` [READ, `main.py:485`], so the run aborts with a socket traceback and
no record that a cell may be in the gripper.

Note also that `estop()` itself can raise: `_send` on a dead socket raises inside the
`self.estop()` call in `_cmd_and_wait`, which **replaces** the intended
`RuntimeError(f"EthernetKRL failure: {last_err}")` and loses the original cause. [READ]

### 1.3 FINDING (high) — the handshake path has no retry, no E-stop, and leaks the socket

`connect()` is outside the retry machinery entirely. [EXEC]

- **Handshake ack never arrives** → `TimeoutError` propagates. No retry, no E-stop.
- **Handshake refused** → `RuntimeError("Handshake refused (status=ESTOP)")` propagates **and
  `self._sock` is still an open, connected socket** — I printed the live socket object after the
  raise to confirm it. Because `connect()` is called from `__enter__`, `__exit__` never runs, so
  the canonical `with KukaClient(cfg) as k:` form leaks the file descriptor on every refused
  handshake.

The refusal case deserves emphasis: the status code that triggers it is very often
`RobotStatusCode.ESTOP` — the controller is telling you it is *already* in a Category-0 stop. The
client's response is to raise a string and hold the connection open. It neither closes, nor
escalates, nor records the controller's state anywhere a caller can act on.

The brief specifically asks whether the E-stop fires "including timeouts during the handshake". It
does not. The mitigating fact is that at handshake time the arm has not been commanded to move, so
the physical consequence is small — but the module docstring says "CRC failures and socket
timeouts trigger a retry policy", unqualified, and a reader has no way to know the handshake is
exempt.

### 1.4 FINDING (medium) — `handshake_timeout_ms` does not time the handshake

`connect()` sets `sock.settimeout(handshake_timeout_ms / 1000)`, then calls `_recv_status()`,
which **unconditionally re-arms** `settimeout(command_timeout_ms / 1000)` before reading a single
byte. [READ, `execution.py:100` vs `:218`] So the handshake timeout bounds only the TCP `connect()`
itself; the wait for the ack is governed by the command timeout.

Verified [EXEC]: with `handshake_timeout_ms = 300` and `command_timeout_ms = 700`, a server that
accepted the connection and never replied caused the client to raise after **0.73 s**, not 0.3 s.
The knob does not do what its name and its config comment say.

### 1.5 FINDING (medium) — `command_timeout_ms` is a per-`recv` timeout, not a per-command deadline

`_recv_status` loops `while len(buf) < STATUS_LEN`, and each `recv` gets a fresh timeout. A
controller that trickles bytes never trips it. Verified [EXEC]: with `command_timeout_ms = 250`, a
server sending one byte every 200 ms had its status frame **accepted after 3.02 s**, reported as a
clean `SUCCESS`. There is no bound on how long a single command may take. For a machine whose
whole safety argument rests on bounded response times, the command deadline is unenforced.

### 1.6 FINDING (medium) — a controller-reported `CRC_ERROR` or `ESTOP` is neither retried nor fatal

The retry policy only covers integrity failures the client detects **locally**. A well-formed
status frame carrying code `5` (`CRC_ERROR`) or `6` (`ESTOP`) parses without error, so
`_cmd_and_wait` returns it as an ordinary result. [READ] Downstream, `main._run_one_cycle`
buckets everything that is not `SUCCESS` or `PICK_FAILED` into an `else` branch: mark the cell
failed, increment `place_failed`, **continue to the next cycle** [READ, `main.py:494-506`].

So "the controller reports it is in a Category-0 stop" is recorded as a failed place, and the loop
immediately plans and commands the next motion. The same branch also swallows `TIMEOUT` — including
the synthetic `TIMEOUT` that `_recv_status` substitutes for any status code it does not recognise,
which means an unknown code from a future controller firmware is indistinguishable from a normal
placement failure.

The module docstring's claim that "CRC failures ... trigger a retry policy" is true only of
locally-detected ones. The remote-reported case, which is the one that indicates a problem on the
robot's side of the wire, is not retried and not escalated.

### 1.7 FINDING (medium) — no resynchronisation after a stream desync

The framing is fixed-length with no delimiter, no length field, and (per §1.1) no sequence number.
The brief asks what happens on "a frame claiming a length longer than the payload" — not
applicable, there is no length field, which is a genuine strength. But the corresponding weakness
is that **once the byte stream is off by even one byte, it can never recover.**

Verified [EXEC]: a server that appended three stray bytes after one good status frame caused every
subsequent frame to fail CRC. The client behaved *correctly at the policy level* — retried, then
E-stopped and raised — but it sent `MOVE_TO` four times in the process, and diagnosed the fault as
"timed out" rather than "the stream is desynchronised". On a real controller those are four
duplicate motion commands (see §1.1) and a misleading log.

A resync rule as simple as "on CRC failure, discard bytes until a plausible version byte, or drop
the connection and reconnect" would convert this from unrecoverable to recoverable. Reconnecting is
the safer of the two.

### 1.8 FINDING (high, documentation) — the heartbeat does not exist

FDR.md §7.5 states:

> "The client-side implementation sends heartbeats inside the `_cmd_and_wait` retry loops and on
> connection hand-off to the mock, **using a dedicated daemon thread** so that heartbeat delivery
> is decoupled from application-level blocking."

None of that is implemented. [EXEC] `OpCode.HEARTBEAT` appears in exactly two places in the entire
repository: its own enum definition in `protocol.py:51`, and the mock's dispatch arm at
`mock_kuka_server.py:171`. **Nothing ever sends one.** `execution/execution.py` contains no
`Thread`, no `daemon`, no timer of any kind. The `heartbeat_interval_ms` config key is used for
exactly one thing: `time.sleep(cfg.heartbeat_interval_ms / 1000)` as the fixed pause between
retries.

Both FDRs further state that "missing three consecutive heartbeats triggers an automatic Category-0
stop at the controller end". The mock has no watchdog either — it blocks on `recv` forever and is
perfectly content to hear nothing for an hour.

This is the most consequential documentation defect in the module, and it is worse than the
E-stop gaps above, because the heartbeat is the mechanism that would stop the robot **if the host
died** — the failure mode no client-side error handler can ever cover. The FDR presents a
dead-man's switch that exists at neither end. FDR_v2 quietly dropped the false sentence about the
daemon thread but kept the claim about the controller-side watchdog, so the corrected version still
describes a liveness guarantee nothing provides.

### 1.9 FINDING (medium, documentation) — "exponential backoff" is a constant sleep, and the test cited as evidence tests something else

- FDR.md:1136 and FDR_v2:1005 / :1107 / :1426 all describe "three-attempt retry with **exponential
  backoff**". The backoff is `time.sleep(cfg.heartbeat_interval_ms / 1000)` — a constant 50 ms,
  identical on every attempt. [READ]
- FDR_v2's traceability matrix, row **O4.b**, claims: *"Three-attempt retry with exponential
  backoff | `execution/execution.py` | `tests/test_execution.py` (`drop_probability=1.0`) | Pass —
  escalates ESTOP"*. The cited test is `test_pick_failure_reported`, which asserts
  `status.code == RobotStatusCode.PICK_FAILED`. [READ] It never exhausts retries, never triggers
  the E-stop path, and never observes an `ESTOP` packet. The escalation is real — §1.0 shows I
  verified it — but **no test in this repository checks it**, and the matrix cites a test that
  cannot fail if the escalation is deleted.

For context on coverage: `tests/test_execution.py` is 99 lines covering the happy path plus config
parsing; `tests/test_protocol.py` is 113 lines and is genuinely good (CRC vectors, corruption,
version, length, negatives). `mock_kuka_server.py` — 255 lines, the entire simulated controller —
**has no test file**. Nothing anywhere tests a malformed status frame, a mid-frame close, retry
exhaustion, or the E-stop packet.

### 1.10 FINDING (low) — the KRL side and the Python side describe different protocols

`execution/krl_prog/laptop-comm.xml` configures an EthernetKRL channel with an **XML** payload
schema — `<Element Tag="Command"><Element Tag="Target" Type="INT"/></Element>` — a single integer
field, no binary mode, and no checksum element. `execution/protocol.py` implements a 16-byte binary
framing with a CRC trailer. These cannot both be the interface. [READ]

Similarly, `routines.src` defines `PickAndPlace(pick_pos, place_pos, vacuum_pct)` taking **two**
Cartesian poses, while the `PICK_AND_PLACE` opcode carries **one** coordinate triple; the client's
own docstring acknowledges the gap and papers over it with "the place target is latched by a
preceding MOVE_TO".

Relatedly, `protocol.py`'s docstring asserts that CRC-16/MODBUS "is the standard integrity check on
the KUKA EthernetKRL XML transport (PPR §7.3, R4)". The project's own `laptop-comm.xml` contains no
checksum element, which is internally sufficient to show the claim is wrong without appealing to
anything outside the repo. FDR §7.1 is more careful — "modelled on the KUKA EthernetKRL 3.1
specification and **augmented with** a CRC-16/MODBUS trailer" — which is accurate and is the phrasing
`protocol.py` should adopt.

Low severity because nothing executes the KRL side. It matters because these three files are the
artefact a reader would use to judge whether the author understands the real interface.

### 1.11 Where the mock is more permissive than a real controller

All rows verified by executing probes against `execution.mock_kuka_server`. **[EXEC]** for what the
mock accepts; **[SPEC]** for the claim about real-controller behaviour.

| # | The mock accepts | A real KRC would | Effect on test optimism |
|---|---|---|---|
| 1 | `MOVE_TO` as the **very first packet**, no handshake — returns `SUCCESS` | require the channel to be opened and configured first [SPEC] | The handshake is never load-bearing, so §1.3's total lack of handshake error handling is invisible to the suite |
| 2 | `MOVE_TO(5000,5000,5000)`, `(-100000,0,0)`, `(0,0,-500)` — **all `SUCCESS`** | reject as a workspace/software-limit fault; the KR 6 R700's reach is 706 mm and z=-500 is half a metre below the table [SPEC] | **The single most optimistic thing about the mock.** Nothing in the pipeline is ever told a pose is unreachable. `planning.yaml`'s `workspace_bounds_mm: ±350` is enforced by no one at the robot end |
| 3 | any velocity implicitly; no operating-mode or drive-enable gate | refuse motion without drives enabled, and cap speed in T1 [SPEC] | `execution.yaml`'s `safety_max_velocity_mm_s: 250` is never tested against anything (it is also a dead key — §2.1) |
| 4 | `VACUUM_ON` with `aux = 60000` → `SUCCESS` | reject; `routines.src` documents `vacuum_level` as 0..100 | Out-of-range analog output never surfaces |
| 5 | `PICK_AND_PLACE` with the vacuum **already on** and **no preceding `MOVE_TO`** — silently uses whatever pose it is at | enforce a state machine / interlock [SPEC] | The client's "place target is latched by a preceding MOVE_TO" convention is unverifiable; violating it is not an error, it is a wrong pick |
| 6 | **E-stop does not latch.** After `ESTOP` the mock drops that connection, but a fresh connection is accepted at once and `MOVE_TO(300,300,300)` returns `SUCCESS`. `_dispatch` returns `(_ESTOP, 0)` and mutates no state — vacuum is not dropped, in-flight motion is not halted | latch the stop and require an acknowledged reset before re-enabling motion [SPEC] | **The safety mechanism the FDR leads with is a no-op in the only implementation that exists.** After escalation the client believes the robot is stopped; the mock is fully operational |
| 7 | two concurrent clients mutating one shared `_RobotState` — both commanding different poses received the **same** final pose. `lock` guards the coordinate assignment but not the `time.sleep` modelling the motion | expose one EthernetKRL channel [SPEC] | Interleaved motion is representable and untested |
| 8 | a **half frame** (8 bytes then silence) blocks the handler thread forever — no reply, no teardown, `self.request.recv` has no timeout | close the channel on a receive timeout [SPEC] | The controller-side half of §1.2's truncation story is never exercised |
| 9 | reports `CRC_ERROR (5)` for a **valid-CRC** frame with an unknown opcode (`0x42`) **and** for a bad protocol version (`0x99`) | distinguish a line fault from an unsupported command [SPEC] | The client cannot tell corruption from a protocol mismatch — and per §1.6 treats neither as fatal |
| 10 | the third coordinate of `PICK_AND_PLACE`: the client sends `pose.pick.z_mm`; the mock's parameter is named **`place_z`** and is used as the *insert* height at the **place** location, while the pick descent is hardcoded to `z = 5` | — (this is an internal contract mismatch, not leniency) | The commanded pick height is silently ignored and reinterpreted as a place height. `execution.yaml: motion.grasp_height_mm: 5.0` is exactly the number the mock hardcodes and never reads |

Summary judgement: the mock is a faithful model of the **wire format** and an optimistic model of
the **controller**. It validates framing and CRC honestly, and it validates nothing about
reachability, interlocks, latched safety state, or channel discipline. The most important
consequence is row 6 — every test that exercises escalation ends with a robot that is, in the
simulation, still free to move.

---

## Part 2 — configuration coherence

### 2.1 Dead keys, enumerated

Method [EXEC]: extract every leaf key from every `configs/*.yaml`, then test each key name against
all repository `.py` and `.src` sources for a string lookup (`.get("k")`, `["k"]`), a dataclass
field, or a keyword argument of that exact name. The sweep is deliberately generous — a key counts
as live if its bare name appears *anywhere* — so anything it lists is dead with high confidence.
Each hit below was then confirmed by hand against the relevant loader.

`arbitration.tau` — the one already found — is genuinely gone: `configs/planning.yaml` now carries
a 10-line comment in its place recording that it was read by nothing, that three documents quoted
three different values, and that the gate it fed was retired on measurement. That is the right
disposal and the right amount of prose. The remaining dead keys:

**`configs/execution.yaml`** — 7 of 24 keys are read by nothing:

| key | value | note |
|---|---|---|
| `kuka.protocol` | `ethernet_krl_3_1` | descriptive only |
| `kuka.command_length_bytes` | `16` | the operative constant is `protocol.COMMAND_LEN` |
| `kuka.crc_polynomial` | `0xA001` | hardcoded in `crc16_modbus`; editing this key changes nothing |
| `kuka.stop_category` | `0` | never read |
| `motion.grasp_height_mm` | `5.0` | the mock hardcodes `5` (see §1.11 row 10) |
| `motion.default_velocity_mm_s` | `150.0` | `routines.src` hardcodes `$VEL.CP = 0.150` |
| `motion.safety_max_velocity_mm_s` | `250.0` | **a safety limit that no code reads or enforces** |

The last one is worth naming plainly: a key called `safety_max_velocity_mm_s` sitting in a config
file, cited in the FDR's safety discussion, enforced by nothing.

Plus two keys that are **worse than dead, because they look live**: `motion.approach_height_mm`
and `motion.insert_height_mm` are parsed into `ExecutionConfig` fields — and
`tests/test_execution.py:91` asserts the parse works — but **no `KukaClient` method reads either
field**. [READ] Only `transport_height_mm` and `vacuum_level_percent` are used. Meanwhile the
planner *does* have `pick_approach_height_mm` / `place_insert_height_mm`, and
`PlannerConfig.from_dict` reads them from a `motion:` block of the dict it is given — which is
`configs/planning.yaml`, and **`planning.yaml` has no `motion:` block at all** [EXEC: its
top-level keys are `battery, cartridge, occupancy_grid, packing, queue, camera`]. So the planner
always takes its hardcoded 60.0 / 2.0.

Net effect, and this is the trap: **editing `execution.yaml: motion.approach_height_mm` changes
nothing anywhere in the system.** It is parsed, it is stored, it is tested, and it is inert. The
key the planner would read lives in a different file's non-existent section.

**`configs/planning.yaml`** — 9 dead keys:

`cartridge.green_channel_thresh` (`otsu`), `cartridge.pcb_exclusion_required` (`true`),
`packing.rotation_allowed` (`[0, 90]`), `packing.deterministic` (`true`),
`packing.worst_case_bound` (`1.7`), `packing.max_ms_budget` (`50`), `queue.fill_order`
(`row_major`), `queue.assignment` (`nearest_available`), `camera.mm_per_px_y` (`0.38`).

Two of these deserve a note. `packing.max_ms_budget: 50` is the 50 ms budget the FDR discusses at
length; nothing reads it, and the budget is asserted elsewhere as a literal. And
`camera.mm_per_px_y` is never read — `PlannerConfig.from_dict` takes only `mm_per_px_x` [READ,
`planner.py:77`]. The pipeline is isotropic-only by construction (Audit-C's own notes say the
scalar `mm_per_px` "is only physically valid in an isotropic pixel", and call that the correct
choice), so the `_y` key is a promise of anisotropic support that does not exist. It should either
go, or become an assertion that `x == y`.

**`configs/recognition.yaml`** — 3 dead: `training.log_dir` (`recog/runs`),
`evaluation.centroid_error_target_px` (`2.0`), `evaluation.edge_error_target_px` (`4.0`). The two
`*_target_px` keys are acceptance thresholds that no evaluator compares anything against.

**`configs/demo.yaml`, `configs/demo_seg.yaml`** — 1 each: `mode.log_level: INFO`, read by nothing;
the level is set inside `common.logging`.

**Clean:** all ten `configs/segmentation*.yaml` have **zero** dead keys, and `execution.yaml`'s
`simulation:` block is fully live (`main._start_robot` reads all four keys). The three
`synth3d*.yaml` files are validated structurally on load — `config.load_config` raises
`ValueError` on any unknown top-level or section key — which is why they cannot accumulate dead
keys in the first place. **That validator is the fix for everything in this section**, and it
already exists in this repository; it is simply not applied to `execution.yaml`, `planning.yaml`
or `recognition.yaml`.

### 2.2 Contradictions

**`mm_per_px` 0.625 vs 0.38 is deliberate, correctly scoped, and adequately documented — not a
contradiction.** [READ + EXEC] They are different physical quantities:

- `planning.yaml: camera.mm_per_px_x: 0.38` — the placeholder pinhole scale for the real
  fixed-mount camera, flagged in-file as "Replace with real intrinsics when available".
- `demo_seg.yaml: mode.mm_per_px: 0.625` — the Blender generator's *nominal* framing,
  `layout.area[0] * 1000 / render.res[0]` = 800/1280.

Both are **fallbacks only**. A frame carrying its own calibration (from the render sidecar)
overrides both, `Planner.cycle` resolves the two in one place so the extractor and the planner
cannot drift apart [READ, `planner.py:172`], and `plan/placement_area.py` raises `UnknownScale`
rather than guessing when neither exists. The scoping is not merely correct, it is defended in
comments at all three sites, including an explicit note that a fixed 0.625 "cost weeks". And the
receipts have already caught up: `seg_eval_wide_on_cad_test.txt` records `median 0.7815, range
0.4903–1.0915 mm/px` measured per frame and states outright that the nominal 0.625 "describes NO
frame in this corpus" and that pre-2026-08-11 receipts understated millimetre figures by 1.25×.
This is the healthiest thing I examined in either half of the audit.

**The one real duplicate of that constant:** `plan/bin_packing.py:56` —
`def pack_cartridge(..., mm_per_px: float = 0.38)` — hardcodes `planning.yaml`'s value as a
default. It is currently harmless only because **`pack_cartridge` is dead in production**:
`Planner._pack_cartridge` calls `common.packing.pack_best_effort` directly, and the only reference
to `plan.bin_packing.pack_cartridge` anywhere in the repo is `tests/test_packing_move.py` asserting
that the symbol lives in the right module. [EXEC] Recommendation: delete it, or wire it and take
the scale from the caller — but do not leave a second packing entry point carrying a stale scale
constant that would silently disagree with the resolved per-frame scale if anyone ever called it.

**The genuine value contradictions are config-vs-code, not config-vs-config.** The four motion
heights in `execution.yaml` (`grasp 5.0`, `approach 60.0`, `insert 2.0`, `transport 80.0`) are
authored in YAML while the two implementations hardcode their own: the mock uses `5` for the pick
descent and `80` for both lift and retract; `routines.src` hardcodes `+60` for approach and `+80`
for transport. Three of the four numbers happen to agree today by coincidence of authorship.
`insert_height_mm: 2.0` corresponds to nothing either implementation does — the mock inserts at
whatever z arrived on the wire (§1.11 row 10). Nothing keeps any of them in step, and per §2.1 the
config values are unread, so the agreement is decorative.

### 2.3 Sidecar drift

The mechanism: `recog/sync_config.py` loads `configs/synth3d*.yaml` and dumps it to a same-stem
`.json`. `recog/synth3d/config._read_raw` prefers YAML when `import yaml` succeeds and falls back
to the sidecar otherwise — the fallback exists because Blender's bundled Python has numpy but no
PyYAML.

**Can it go stale? Yes. Is it detected? Yes, loudly — but only on the path where it is already too
late to be convenient, and by a proxy rather than by content.**

1. **The check exists and is loud.** [READ, `config.py:359-363`] If the sidecar's `st_mtime` is
   older than the YAML's, `_read_raw` raises
   `RuntimeError(f"{sidecar} is older than {path.name}; the config is stale. Run: python -m
   recog.sync_config")`. That is a hard failure with the exact remedy in the message. Good.
2. **It is unreachable on the dev machine.** The comparison sits in the *else* branch — it runs
   only when PyYAML is unavailable. With PyYAML present the YAML is read directly and the sidecar
   is never opened. So every test, every CLI tool and every review on a dev machine passes with an
   arbitrarily stale sidecar, and the staleness surfaces for the first time **inside Blender, at
   render time**, which is the longest and most expensive job in the project. There is no cheap
   pre-flight that catches it.
3. **It compares mtimes, not content**, which is wrong in both directions. `touch
   configs/synth3d.json` makes a genuinely stale sidecar pass silently. Conversely a fresh `git
   clone` stamps every file at checkout time in arbitrary order, so a perfectly in-sync tree can
   raise a spurious "config is stale" on the first Blender run. An mtime is a proxy for content;
   embedding a hash of the source YAML in the JSON and comparing that would be both stricter and
   clone-safe, at about four lines.
4. **Current state is clean.** [EXEC] All three pairs (`synth3d`, `synth3d_18650`, `synth3d_crown`)
   are **content-identical** after normalising list/tuple representation, and every sidecar's mtime
   is greater than or equal to its YAML's. Nothing is drifting right now.
5. **The unknown-key validation is path-independent** — `load_config` runs the same
   `_SECTIONS`/`_PASSTHROUGH` check on whichever dict `_read_raw` returned, so a typo is caught
   identically in Blender and out. That part is solid, and it is the reason §2.1 found no dead keys
   in these files.

Recommendation, in priority order: (a) replace the mtime comparison with a content hash recorded in
the sidecar; (b) run that comparison on **both** paths, so a dev-machine test run fails fast
instead of a Blender job failing slow.

### 2.4 Experiment-config sprawl — recommendation: keep them, add a one-page index, and enforce the invariant with a test

The nine `segmentation_*.yaml` experiment configs are **reproducible artefacts, not clutter.** The
evidence:

- All 15 config files are committed and the tree is clean [EXEC: `git status --porcelain configs/`
  returns nothing]. (The stale git snapshot in my brief showed seven as untracked; they have since
  been committed.)
- Every one is cited **by name** in a `docs/receipts/*.txt` header — e.g.
  `config : configs/segmentation_cad_test.yaml` — so each published number traces to a file in the
  tree.
- Each header carries the exact `blender -b --python recog/generate3d.py -- --n 502 --out ...
  --tray-set anchored --resume` invocation that produced its dataset. Since `recog/dataset3d_seg_*/`
  and `recog/checkpoints/` are both gitignored [EXEC], **these configs are the only committed
  record of how each run was produced.** Consolidating them away would destroy reproducibility,
  not tidy it.
- The comparability claim in their headers — "Same architecture/training/augmentation as
  `configs/segmentation.yaml`, only `dataset.coco_path`/`img_dir` and `training.checkpoint_dir`
  differ, so the two procedural runs and the four leave-one-SKU-out CAD-control runs are comparable
  on training conditions alone" — **is true, and I verified it mechanically** [EXEC]. Across all
  ten `segmentation*.yaml`, comparing every leaf key, the *only* divergence outside the three
  permitted keys is `dataset.train_val_split`, which is `0.0` in `segmentation_cad_test.yaml` (a
  pure held-out test set, so the exception is correct) and `0.85` in the other nine. Epochs,
  learning rate, momentum, weight decay, scheduler, dice weight, `select_on`, `split_seed`,
  `jitter_frac`, `crop_size`, `half`, and all twelve augmentation knobs are byte-identical
  everywhere. Cross-run comparisons in the receipts really are confounded by the dataset alone.

What is genuinely missing is discoverability and enforcement:

1. **Add `configs/README.md`, about twenty lines**: a table mapping each experiment config →
   dataset directory → checkpoint directory → the receipt it backs, plus one sentence stating the
   invariant. A reader currently has to open nine near-identical files and diff them to learn that
   they are near-identical.
2. **Add a five-line test** asserting the invariant: load all `segmentation*.yaml`, and assert
   that every leaf key except `dataset.coco_path`, `dataset.img_dir`, `training.checkpoint_dir` and
   `dataset.train_val_split` is equal across all of them. This converts a convention that currently
   holds by authorial care into a guarantee. Without it, one future edit to one file's `epochs`
   silently invalidates every cross-run comparison in `docs/receipts/`, and nothing would catch it.

That is the whole recommendation. Do not consolidate, do not delete, do not parameterise them into
one file with overrides — the flat files are what makes each receipt independently reproducible.

---

## Summary of findings by severity

| # | Finding | Severity | Basis |
|---|---|---|---|
| 1.1 | No sequence number; timeout retry duplicates motion commands and returns a stale, mis-paired status | High | EXEC |
| 1.2 | `ConnectionError`, `ConnectionResetError` and `struct.error` all bypass retry **and** E-stop | High | EXEC |
| 1.3 | Handshake path: no retry, no E-stop, socket leaked on refusal | High | EXEC |
| 1.8 | The heartbeat / controller watchdog described in both FDRs exists nowhere | High | EXEC |
| 1.4 | `handshake_timeout_ms` does not time the handshake | Medium | EXEC |
| 1.5 | `command_timeout_ms` is per-`recv`; a trickled reply was accepted after 3.02 s at a 250 ms setting | Medium | EXEC |
| 1.6 | Controller-reported `CRC_ERROR` / `ESTOP` is neither retried nor fatal; `main.py` counts it as a failed place and continues | Medium | READ |
| 1.7 | No resync after a stream desync; one stray byte is unrecoverable | Medium | EXEC |
| 1.9 | "Exponential backoff" is a constant sleep; the traceability row citing an E-stop test cites a test that never sees one | Medium | READ |
| 1.11 | Mock accepts unreachable poses, requires no handshake, and **does not latch the E-stop** | Medium (High if hardware is ever attached) | EXEC / SPEC |
| 1.10 | `laptop-comm.xml` declares an XML channel; `routines.src` takes two poses the opcode cannot carry | Low | READ |
| 2.1 | 22 dead config keys, incl. `safety_max_velocity_mm_s`; plus `approach_height_mm`/`insert_height_mm` parsed, tested, and inert | Medium | EXEC |
| 2.3 | Sidecar staleness check is mtime-based and only reachable inside Blender | Low–Medium | READ / EXEC |
| 2.2 | `mm_per_px` 0.625 vs 0.38 — **not** a defect; deliberate, scoped, documented, and reflected in the receipts | None | READ / EXEC |
| 2.4 | Experiment configs — reproducible artefacts; invariant verified across all ten files | None | EXEC |

### The single cheapest high-value change

Widen `_cmd_and_wait`'s `except` clause from `(socket.timeout, ValueError)` to `Exception`, and
make the E-stop attempt best-effort (`try: self.estop() except Exception: log.critical(...)`). That
one edit closes §1.2 entirely, and converts the three bare escapes into the documented behaviour.
It does not address §1.1, which needs a protocol change, or §1.8, which needs a heartbeat thread on
one side and a watchdog on the other.
