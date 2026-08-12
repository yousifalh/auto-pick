# Fixing the execution layer's error handling

**Date:** 2026-08-12 · **Branch:** `feat/blender-synth-dataset` · **Base:** `fa7a4f0`
**Prompted by:** audit F (`2026-08-12-F-execution-and-config.md`) §§1.2, 1.3, 1.4, 1.5, 1.6, 1.11, 2.1
and audit B (`2026-08-12-B-security.md`) findings 2, 3, 4.

The CRC and the frame layout were independently verified correct (20 000 vectors against three
implementations) and **are not touched here**. Every defect fixed below is in the machinery
*around* the wire format: which failures reach the E-stop, what the client does when the
controller says it is already stopped, and what the simulator lets through.

The organising rule for the whole change: **anything the FDR claims exists must either exist or
stop being claimed.** Where a claim could not be made true without changing the protocol, the
claim was deleted rather than faked, and §6 lists every one of those for the documentation pass.

---

## 1. The E-stop had three escape routes. It now has none.

`_cmd_and_wait` caught `(socket.timeout, ValueError)`. Three failures escaped with no retry, no
E-stop, no `close()`, and no handler anywhere upstream.

`struct.error` is the sharp one: `issubclass(struct.error, ValueError)` is **`False`**, so an
out-of-range coordinate — the failure reachable from a bad perception result — tore down the call
stack **with the socket still perfectly healthy and the E-stop packet perfectly sendable.**

Failures are now sorted into three kinds, and the sort is the whole fix:

| kind | what | policy |
|---|---|---|
| transient | `socket.timeout`, any `ValueError` (CRC, length, unknown status code), controller-reported `CRC_ERROR` / `TIMEOUT` | retry ×`max_retries`, then E-stop and raise |
| fatal | **everything else** — `struct.error`, `ConnectionError`, `ConnectionResetError`, and anything unforeseen — plus controller-reported `VERSION_MISMATCH` / `UNSUPPORTED_COMMAND` | E-stop **immediately**, no retry, raise |
| the controller says it is stopped | status `ESTOP` | latch, close, raise `RobotEstop` (§3) |

Fatal failures skip the retries deliberately. A coordinate that does not fit in an `int32` will not
fit on the second attempt, and a frame the far end cannot parse will not become parseable by being
re-sent; retrying is pure delay before a stop that should already have happened.

Two supporting changes:

* **The E-stop is best-effort.** `_emergency_stop()` never raises. `estop()` calling `_send` on a
  dead socket used to raise *inside* `_cmd_and_wait`, **replacing** the `RuntimeError` that carried
  the original cause. It is now caught and logged `CRITICAL` — *"E-stop could NOT be transmitted …
  the arm's state is UNKNOWN — it may be mid-motion with the vacuum on."* An E-stop that cannot be
  sent must not be silent, and it must not eat the error that prompted it.
* **New exception types.** `RobotFault(RuntimeError)` and `RobotEstop(RobotFault)`. Both subclass
  `RuntimeError`, so any caller written against the old contract is unaffected. A socket
  traceback escaping a robot client is not an acceptable way to say "the arm may be holding a
  cell".

Tested: `test_out_of_range_coordinate_fires_the_estop` (asserts `ESTOP` reached the wire and
`MOVE_TO` never did), `test_mid_frame_close_is_fatal_and_the_stop_is_attempted`,
`test_connection_reset_is_fatal_not_a_bare_traceback`,
`test_an_undeliverable_estop_is_logged_critical_not_swallowed`.

**`test_retry_exhaustion_sends_the_estop` is new and matters on its own.** FDR_v2's traceability
row O4.b cited `drop_probability=1.0` as evidence that retry exhaustion escalates; that test
asserts `PICK_FAILED` and never reaches the escalation. Nothing in the repository observed an
E-stop packet until now. It asserts the exact sequence
`HANDSHAKE, MOVE_TO, MOVE_TO, MOVE_TO, ESTOP`.

## 2. The handshake now has the same safety path as everything else

`connect()` was outside the retry machinery entirely. It had no retry, no E-stop, and on refusal it
raised **with the socket still open** — and because `connect()` is called from `__enter__`,
`__exit__` never ran, so the canonical `with KukaClient(cfg) as k:` leaked a descriptor on every
refused handshake.

* The handshake retries like any other command, and on exhaustion fires the E-stop. The socket is
  alive at handshake time, so this one genuinely goes out.
* **Every** failure route closes the socket. `connect()` wraps `_handshake()` in
  `except BaseException: self.close(); raise`.
* A refusal carrying `ESTOP` — very often the controller saying it is *already* in a Category-0
  stop — latches the client and raises `RobotEstop`. Any other refusal fires the E-stop and raises
  `RobotFault`, because we have neither a channel we trust nor knowledge of where the arm is.
* **`handshake_timeout_ms` now times the handshake.** `_recv_status` unconditionally re-armed
  `command_timeout_ms` before reading a byte, so the knob bounded only the TCP connect. It now
  takes a `timeout_s` argument, asserted by `test_handshake_timeout_ms_actually_times_the_handshake`
  (200 ms handshake, 4000 ms command → must return in well under 1.5 s).
* **`command_timeout_ms` is now a whole-frame deadline** rather than a per-`recv` timeout. A
  controller trickling one byte at a time was accepted after 3.02 s at a 250 ms setting; for a
  machine whose safety argument rests on bounded response times, the command deadline was
  unenforced. Four lines, and it was unavoidable while fixing the handshake timeout.

## 3. A controller that says it is stopped now stops the client

This was the highest-consequence item. A well-formed status carrying code 6 (`ESTOP`) parsed
cleanly and returned as an ordinary result; `main._run_one_cycle` bucketed everything that was not
`SUCCESS` or `PICK_FAILED` into an `else`, counted it as `place_failed`, and **planned and
commanded the next motion.** The controller reporting a Category-0 stop was recorded as a failed
placement.

Now:

* `_cmd_and_wait` sees `ESTOP` → sets `_estopped`, closes the socket, raises `RobotEstop`.
* The client **latches**. A latched client refuses to send anything (`RobotEstop` before any bytes
  reach the wire) and refuses to `connect()` again. Clearing a Category-0 stop is a deliberate act
  at the controller, not something a host reconnect may do.
* `main.run` catches `RobotEstop` and `RobotFault`, logs `CRITICAL` with the partial statistics,
  and **re-raises**. A stopped run does not return a statistics dict; there is no code path on
  which a Category-0 stop produces a normal-looking summary.
* `main._run_one_cycle`'s `else` branch is gone. `PLACE_FAILED` is now handled explicitly, and
  anything else **raises** — if the client's classification ever lets a new code through, that is a
  bug in the mapping and it stops the run rather than being counted as a placement failure.
* Related: `_recv_status` substituted `TIMEOUT` for any status code it did not recognise, making an
  unknown code from a future firmware indistinguishable from an ordinary placement failure. It now
  raises `ValueError("unknown status code N")`, which is transient — retried, then escalated.

## 4. `place.z_mm`: **deleted, not added to the frame**

**The decision, and why.** The 16-byte frame carries **one** Z per command and the pick needs it.
Making `pose.place.z_mm` reach the wire requires a second Z field — a frame-layout change and a
protocol version bump. That would mean:

* changing the framing the audit verified correct against three independent implementations;
* invalidating the "modelled on the KUKA EthernetKRL 3.1 specification" framing in FDR §7.1, since
  the layout would no longer be the one described;
* inventing a wire format for a robot nobody has, to carry a number whose only current consumer is
  a simulator.

The brief's instruction was to prefer the honest smaller change when unsure. So: **`place.z_mm` is
not transmitted, and that is now stated at every site that could mislead a reader** rather than
papered over by a mock parameter *named* `place_z` that quietly received `pick.z`, "inserted"
there, and returned `SUCCESS`.

Concretely:

* `KukaClient.pick_and_place`'s docstring says outright that `place.z_mm` is not transmitted and
  cannot be, names the controller-side constant that governs the insert depth
  (`routines.src` descends `LIN place_pos`; the simulator uses `_INSERT_Z_MM = 2`), and points here.
* `ExecutionConfig.insert_height_mm` and `configs/execution.yaml: motion.insert_height_mm` are
  **deleted**. `approach_height_mm` and `grasp_height_mm` go with them — same defect, same reason:
  parsed, stored, unit-tested, and read by no `KukaClient` method. Editing them changed nothing
  anywhere in the system, which is worse than a missing key because it looks live.
* The simulator's parameter is renamed `pick_z` and now drives the **pick descent**, which is what
  the client actually sends; the insert descends to `_INSERT_Z_MM`.
* `test_inert_motion_keys_are_gone` and `test_shipped_execution_yaml_declares_no_dead_safety_key`
  keep them gone.

**One thing this surfaced that I did not fix, because it is `plan/`'s.** `plan/planner.py:464` sets
`pick.z_mm = cfg.pick_approach_height_mm` (60 mm), so the Z the client puts on the wire as the pick
target is an **approach** height, not a grasp. `routines.src` treats the wire Z as the grasp pose
and *derives* the approach as `Z + 60`. The simulator now logs a `WARNING` once when a commanded
pick Z is above a 25 mm grasp band, saying that on real hardware `$IN[10]` would report no grasp and
the cycle would return `PICK_FAILED`. It cannot fail the grasp itself — it models no parts. This is
a live cross-module defect and it belongs to whoever owns `plan/`.

## 5. `safety_max_velocity_mm_s`: **deleted**

`motion.safety_max_velocity_mm_s: 250.0` was a safety-named key, cited in the FDR's safety
discussion, read and enforced by nothing.

It cannot be enforced host-side: **the frame carries no velocity field.** The host can neither
command a speed nor cap one. The only real cap is controller-side — `$VEL.CP = 0.150` in
`krl_prog/routines.src`, plus the KRC's own T1/T2 limits. `default_velocity_mm_s: 150.0` goes with
it for the same reason. Both are replaced by a comment in `configs/execution.yaml` recording where
the cap actually lives and warning against re-adding a host-side speed cap without a velocity field
to enforce it with.

**The three descriptive `kuka:` keys went the other way and are now enforced.**
`command_length_bytes`, `crc_polynomial` and `stop_category` are claims *about* `execution/protocol.py`;
editing `crc_polynomial` changed no CRC. `ExecutionConfig.from_dict` now cross-checks all three
against the code and raises `ValueError` naming the key. Unknown keys are still tolerated, so older
config files keep loading.

## 6. Simulator hardening

`execution/mock_kuka_server.py` had **no test file**, and that correlation is not a coincidence —
it is where every unnoticed optimism lived. `tests/test_mock_kuka_server.py` is new (17 tests, raw
sockets and hand-built frames, so it asserts what the *controller* does rather than what the client
makes of it).

| what it now enforces | what it did before |
|---|---|
| **Workspace envelope** — KR 6 R700's 706 mm reach, `z ∈ [-20, 706]`. Violation is a software-limit fault: it latches the stop and answers `ESTOP` **before moving or sleeping** | `MOVE_TO(5000,5000,5000)`, `(-100000,0,0)`, `(0,0,-500)` all returned `SUCCESS`. Nothing in the pipeline was ever told a pose was unreachable, so `planning.yaml`'s ±350 mm bound was enforced by no one at the robot end. Also closes security finding 3: `MOVE_TO(2³¹-1, 2³¹-1, 32767)` made one handler `time.sleep` for **63 days** |
| **Latching E-stop** — every command *including `HANDSHAKE`* answers `ESTOP`, the vacuum drops (a Cat-0 stop removes power), no reconnect clears it, and there is deliberately no reset method | The stop dropped the connection and mutated no state. A fresh connection was accepted at once and `MOVE_TO(300,300,300)` returned `SUCCESS`. **The safety mechanism the FDR leads with was a no-op in the only implementation that exists** |
| **Socket timeouts** — a *frame* timeout (2 s) and an *idle* timeout (120 s), separately, because a host idling between commands while it runs perception is normal and a controller holding half a frame is not | No `settimeout` anywhere. 8 bytes then silence pinned a handler thread forever; 200 half-open connections held 200 threads |
| **Distinct fault codes** — `UNSUPPORTED_COMMAND` (7) for an unknown opcode, `VERSION_MISMATCH` (8) for a bad version, `CRC_ERROR` (5) only for a genuine line fault | All three reported `CRC_ERROR`. The host could not tell corruption from a protocol mismatch, and the two want opposite responses: retry the line, versus stop and fix the build |
| A `--host` other than loopback logs a warning that this is an unauthenticated controller | Silent |

Supporting changes: `execution/protocol.py` gains `ProtocolError(ValueError)` and four subclasses
(`FrameLengthError`, `CrcError`, `VersionMismatch`, `UnsupportedOpCode`) so the three faults can be
told apart. **All subclass `ValueError`**, so every existing `except ValueError` and every
`pytest.raises(ValueError)` in `tests/test_protocol.py` is unaffected. No arithmetic changed.

`common/types.RobotStatusCode` gains `UNSUPPORTED_COMMAND = 7` and `VERSION_MISMATCH = 8`. This is
the one file touched outside the assigned set; the enum is the canonical status set and there is no
other correct home for two codes that cross the wire. Additive, values 7 and 8, nothing renumbered.

Not modelled, still, and deliberately: drive-enable / operating-mode gates, analog-output range
checks on the vacuum level (`VACUUM_ON` with `aux = 60000` is still accepted), single-channel
exclusivity (two clients still share one robot), and any heartbeat watchdog. The module docstring
lists these so a reader is not left to infer them from silence.

---

## Deliberately not implemented — for the documentation pass

1. **The heartbeat and the controller watchdog (FDR §7.5).** Not built, on instruction. `OpCode.HEARTBEAT`
   appears in exactly two places — its enum definition and the simulator's dispatch arm — and
   **nothing ever sends one**. There is no thread, no timer, no watchdog at either end.
   `heartbeat_interval_ms` is the constant pause between retries and nothing else; `configs/execution.yaml`
   now says so in a comment, and `execution/execution.py`'s module docstring says so too. The FDR
   describes a dead-man's switch that exists at neither end, and it is the mechanism that would stop
   the robot **if the host died** — the one failure no client-side handler can cover. **This is a
   documentation correction, not a feature.**
2. **No sequence number (audit §1.1).** Unfixed and unfixable inside 16 bytes. A *late* reply is
   still mis-paired with the next command, and the retry loop still re-sends motion opcodes —
   `PICK_AND_PLACE` re-sent is a second grasp attempt, not an idempotent operation. The module
   docstring now states both plainly instead of leaving a reader to discover them. Fixing it needs a
   frame-layout change (a sequence/echo field) or a rule that a timed-out *motion* command is never
   retried, only resynchronised. Both are protocol decisions, not error-handling bugs.
3. **"Exponential backoff" (FDR.md:1136, FDR_v2:1005/1107/1426).** It is `time.sleep(heartbeat_interval_ms/1000)`
   — a constant 50 ms, identical on every attempt. Left as a constant sleep; the docs should stop
   calling it exponential. FDR_v2's traceability row **O4.b** should also stop citing
   `test_pick_failure_reported` as evidence for escalation; cite
   `tests/test_execution.py::test_retry_exhaustion_sends_the_estop`, which actually observes the
   `ESTOP` packet.
4. **No stream resynchronisation (audit §1.7).** One stray byte is still unrecoverable — the client
   retries, escalates and stops, which is *safe* but diagnoses the fault as "timed out" rather than
   "the stream is desynchronised".
5. **`krl_prog/laptop-comm.xml` still declares an XML channel with no checksum element** while
   `protocol.py` implements a 16-byte binary framing with a CRC trailer (audit §1.10). Also
   `protocol.py`'s docstring still asserts CRC-16/MODBUS "is the standard integrity check on the
   KUKA EthernetKRL XML transport"; FDR §7.1's "augmented with a CRC-16/MODBUS trailer" is the
   accurate phrasing and the docstring should adopt it. Not touched — the KRL side executes nothing
   and rewriting it is a documentation job.
6. **`.github/workflows/ci.yml` says "719 of the 752 tests".** Both numbers move. This change adds
   **37**, none of which need torch, so both counts rise by 37 — and the concurrent audit fixes add
   more on top. Not my file; flagged so the number is recomputed rather than patched.
7. **`plan/planner.py` sends an approach height in the pick-Z field** (§4). Cross-module, owned by
   `plan/`.

## Test results

**+37 tests from this change** — `tests/test_execution.py` 8 → 29, and `tests/test_mock_kuka_server.py`
(16) where there was no test file at all. Full suite green at **814** at the time of writing
(752 at `fa7a4f0`; the balance is concurrent audit work in `plan/` and `recog/`). `ruff` clean.

**One test asserted a defect and was changed:** `tests/test_execution.py::test_execution_config_from_dict`
asserted `cfg.approach_height_mm == 55.0`, and `test_execution_config_defaults` asserted
`cfg.approach_height_mm == 60.0` — a field no `KukaClient` method read. Both now assert
`transport_height_mm`, which is live (it is the Z of the latching `MOVE_TO`).

Files changed: `execution/execution.py`, `execution/protocol.py`, `execution/mock_kuka_server.py`,
`main.py`, `configs/execution.yaml`, `common/types.py`, `tests/test_execution.py`, and new
`tests/test_mock_kuka_server.py`.
