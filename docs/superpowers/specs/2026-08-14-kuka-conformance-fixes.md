# KUKA conformance fixes — 2026-08-14

**Base:** `docs/superpowers/audit/2026-08-14-T-kuka-conformance.md`, researched
against KUKA's *Expert Programming* (`ProgExperteBHR5.2`), *System Variables*
(`SysVar 08.02.03`), *EthernetKRL 2.2* and the KR 6 R700 sixx datasheet.
**HEAD when this started:** `54f6790` — 1 218 passing, 1 skipped.
**Suite when it finished:** 1 221 passing, 1 skipped (three tests added).
**Files changed:** `execution/krl_prog/routines.src`,
`execution/krl_prog/laptop-comm.xml`, `execution/protocol.py`,
`tests/test_execution.py`, `configs/planning.yaml`, `plan/scene.py`,
`main.py`, `docs/FDR_v3.md`, plus one new file,
`execution/krl_prog/laptop_comm.src`.

**Nothing here was executed.** There is no KUKA controller, no KRL compiler and
no WorkVisual or OrangeEdit instance in this project or in this session. Every
KRL and EKI claim below is a documentation-based inference. The KRL files and
the XML each open with an unmissable statement to that effect, in their own
headers, so that no reader can take their existence for evidence that they run.
The Python suite is the only thing that was run, and it does not touch the KRL.

Source labels used throughout, following the audit's convention: **[KUKA]** for
a KUKA-authored manual, **[VENDOR]** for a third-party integrator manual that
reproduces KUKA's interface, **[FORUM]** for a forum post used where nothing
better was found.

---

## 1. `routines.src`

### 1.1 `DEF` + `RETURN <value>` → `DEFFCT INT` … `ENDFCT`

`PickAndPlace` was `DEF PickAndPlace(...) … RETURN 2 … RETURN 1 … END`.
[KUKA, Expert Programming §6]: "Unlike a subprogram, a function sends back a
return value. A function begins with the keyword `DEFFCT`. The data type of the
return value is specified directly after the keyword `DEFFCT`. The return value
itself is transferred via `RETURN`. The function is terminated using the keyword
`ENDFCT`." In a subprogram `RETURN` takes no operand — it only exits — so
supplying one is an inadmissible instruction.

It is now `GLOBAL DEFFCT INT PickAndPlace(...)` … `ENDFCT`, and the caller
consumes the value as `status = PickAndPlace(pick_pos, place_pos, aux)`.

**Every subroutine was checked for the same pattern.** `PickAndPlace` was the
only one with a valued `RETURN`; `MoveBetweenPositions`, `SimpleMove` and
`VacuumControl` return nothing and stay `DEF` … `END`, which is correct for
them. The new routines in `laptop_comm.src` that do return values
(`ByteAt`, `Int32At`, `Int16At`, `Uint16At`, `Crc16Modbus`) are all `DEFFCT INT`
… `ENDFCT`; the ones that do not (`PutByte`, `PutInt32`, `PutInt16`,
`BuildStatus`) are `DEF` … `END`.

### 1.2 `$ANOUT` scaling

[KUKA, System Variables]: `$ANOUT[n]` — data type **Real**, minimum **−1.0**,
maximum **+1.0**, unit **V**, `[n] = [1] … [16]`, `−1.0 ⟶ −10 V`,
`+1.0 ⟶ +10 V`.

`$ANOUT[analog_out] = vacuum_level` with `vacuum_level` documented 0–100 was
therefore out of range for every non-zero value, clamped to full scale, and
annunciated. The vacuum worked, at full power, always, and the percentage the
Python side threads through `aux_u16` did nothing. It is now

```krl
$ANOUT[analog_out] = pct / 100.0
```

with `pct` first clamped to 0…100, so a host-side bug cannot reach the
annunciated out-of-range path at all. Saturating rather than raising is the
conservative reading in both directions: it cannot command more suction than
full scale, and it cannot command negative voltage on a channel where voltage
means suction.

**Not verified:** that 0–10 V maps linearly to 0–100 % vacuum is a property of
the ejector, not of the controller. No gripper datasheet exists in this
repository, so the mapping is asserted by the comment and by nothing else.

### 1.3 `GLOBAL`

[KUKA, Expert Programming §6]: "Local subprograms/functions can only be called
from within the SRC file in which they were programmed. … Alternatively, a local
subprogram can be preceded by the keyword `GLOBAL`." None of the four carried
it, so a receive loop in another SRC could not have called any of them. All five
subprograms after the leading declaration are now `GLOBAL`.

`laptop_comm.src` additionally declares their signatures with `EXT` / `EXTFCT`,
in the form KUKA's own program template uses for the basis package
(`EXT BAS(BAS_COMMAND :IN, REAL :IN)` — data *types* in the parameter list, not
names). Whether the declarations are strictly required in KSS 8.x was not
established; they are correct KRL either way and cost nothing.

### 1.4 The filename rule

[KUKA, Expert Programming]: "The object name without an extension is also the
name of the file and is therefore prefixed by `DEF`." The file led with
`DEF MoveBetweenPositions(pos1 :IN, pos2 :IN)`, declaring an object named
`MoveBetweenPositions` in a file named `routines` — the first thing a KRL editor
rejects on load. It now leads with `DEF routines()`, an entry point whose whole
body is `InitMotion()`, so that selecting the file by accident initialises
motion and returns rather than moving the arm.

The new receive loop is `laptop_comm.src` with `DEF laptop_comm()` for the same
reason. It underscores rather than hyphenates because a KRL identifier cannot
contain a hyphen, while the EKI configuration keeps the hyphenated name
`laptop-comm.xml` that `EKI_Init` / `EKI_Open` take without its extension.

### 1.5 Motion initialisation, and what actually governs `PTP`

[KUKA, Expert Programming, PTP section]: "`$VEL_AXIS[axis number]`: to program
maximum axis-specific velocities, and `$ACC_AXIS[axis number]`: to program
maximum axis-specific acceleration rates. All entries are given as percentages
of the maximum value defined in the machine data. **If these two system
variables have not been programmed for all axes, execution of the program will
cause a corresponding error message to be generated.**"

Nothing programmed them, so the first `PTP` was a documented runtime fault.
A new `GLOBAL DEF InitMotion()` calls `BAS(#INITMOV, 0)` — the basis-package
call that programs both arrays from the machine data, the one the controller's
own template puts in its BASISTECH INI fold and the one that appears in KUKA's
EthernetKRL `BinaryFixed` sample program — and then sets `$TOOL`, `$BASE` and
`$IPO_MODE`, which Cartesian motion to an `E6POS` requires and which nothing
set before. `laptop_comm.src` calls it once, before the channel opens.

`SimpleMove` used to write its `vel` / `acc` parameters into `$VEL.CP` /
`$ACC.CP` and then issue a `PTP`, where they govern nothing — so the two
parameters that are the entire purpose of the subroutine were inert. They are
now `ptp_vel_pct` / `ptp_acc_pct`, written into `$VEL_AXIS[1..6]` and
`$ACC_AXIS[1..6]`, which is what a `PTP` obeys and which take percentages. **The
unit of those parameters changed with their meaning**, and the comment says so.
Nothing on the wire sets them; `laptop_comm.src` passes a deliberately low cell
default of 30 %.

**`$VEL.CP = 0.150` and `$ACC.CP = 0.500` are untouched.** [KUKA, System
Variables] gives `$VEL` as "Velocities in the advance run. **CP = m/s**" and
`$ACC` as "**CP = m/s²**", so 0.150 genuinely is 150 mm/s and the comment was
right. They govern the three `LIN` segments of `PickAndPlace` and no `PTP`
anywhere; `docs/FDR_v3.md` Appendix D's "the only real cap is `$VEL.CP`" is
narrowed accordingly.

### 1.6 The orientation decision

**Decided: derive from the current pose.** The 16-byte frame carries X, Y and Z
and nothing else. A, B, C, the configuration bits S and T, and the external-axis
components E1…E6 are not commanded and cannot be. `SimpleMove`'s comment claimed
"we copy the current orientation" and no line of code copied anything, so every
`PTP` and `LIN` in the file was issued to an `E6POS` whose orientation and
configuration were never assigned.

A new `GLOBAL DEF TargetFromXYZ(target :OUT, x, y, z)` assigns
`target = $POS_ACT` — "current robot position", type `E6POS` [KUKA, System
Variables] — and then overwrites X, Y and Z only. Copying the whole structure
rather than five named components is deliberate: it leaves no component of the
`E6POS` unassigned, E1…E6 included. Every commanded pose in `laptop_comm.src`
is built through it, so the code and the comment now say the same thing.

What that buys and costs, all stated in the file:

* the wire commands translation only, and the cell's start-up procedure — not
  this program — establishes the tool attitude. There is no fixed tool-down
  A/B/C constant because it would have to be invented; this repository holds no
  taught cell data, and inventing one is exactly the kind of unlabelled
  plausible number this audit series exists to remove;
* carrying S and T forward holds the arm configuration constant across a `PTP`,
  which is what a fixed-tool pick-and-place wants. A target reachable only in a
  different configuration is then refused rather than silently reconfigured;
* reading a `$`-variable stops the advance run [KUKA, Expert Programming §5.1:
  "all input/output instructions and access to inputs/outputs via system
  variables trigger an advance run stop"], so this costs continuous-path
  smoothness at every target built. For a point-to-point cycle that is an
  acceptable price and for a contouring application it would not be.

The alternative — a fixed tool-down orientation — is the better choice on a
commissioned cell and needs exactly one measurement this project does not have.
That sentence is in the file too.

### 1.7 Also done

`MoveBetweenPositions` is dead — no opcode maps to it and the receive loop never
calls it. It is kept, corrected alongside the rest, and **labelled DEAD CODE at
its declaration** rather than quietly retained.

---

## 2. `laptop-comm.xml` — rewritten as fixed-length binary

The old file configured **XML** payloads while the Python side sends **binary**,
and was not a valid EKI configuration in either mode: `<Config>` for
`<CONFIGURATION>`, no `<EXTERNAL>` / `<INTERNAL>`, lower-case element names
where EKI is upper-case (and case-sensitive), `<Element>` nested inside
`<Element>` where EKI takes a flat list, and no `Set_Flag`, so the KRL side had
nothing to trigger on.

It is now a `BinaryFixed` configuration, 16 bytes each way:

```xml
<RECEIVE><RAW><ELEMENT Tag="Buffer" Type="BYTE" Size="16" Set_Flag="1"/></RAW></RECEIVE>
<SEND><RAW><ELEMENT Tag="Buffer" Type="BYTE" Size="16"/></RAW></SEND>
```

Relied on, from [KUKA, EthernetKRL 2.2]: `Type="BYTE"` selects a fixed-length
binary record and `Size` gives its length (1…3 600 bytes); `Type="STREAM"` with
an `EOS` delimiter is the variable-length variant and is not what this frame is;
`Set_Flag` (1…1 025) raises `$FLAG[n]` when the element arrives;
`EXTERNAL`/`TYPE` says whether the *external* system is server or client, and
`EXTERNAL`/`IP`/`PORT` are documented as ignored when it is the client — which
it is here, since the laptop connects to the controller.

**Flagged in the file rather than asserted:** the exact placement of the
controller's own IP and PORT in server mode (given under `<INTERNAL>` because
`<EXTERNAL>`'s are documented as the *external* system's; the manual page
listing `<INTERNAL>`'s children was not obtained); whether `BUFFERING` and
`BUFFSIZE` are attributes of `<INTERNAL>` or of a child element; and the exact
shape of `<ALIVE>`. The KRL loop's logic does not depend on any of the three.

---

## 3. The receive loop — `execution/krl_prog/laptop_comm.src` (new)

Nothing opened a channel, polled it, verified a CRC or dispatched an opcode, so
nothing would ever have called `PickAndPlace`. `routines.src` contained zero
`EKI_` calls.

**Complete, not stubbed, with four gaps named at their line.** The loop does:
`EKI_Init` → `EKI_Open` → `LOOP` → `WAIT FOR $FLAG[1]` → `EKI_CheckBuffer` →
`WHILE ret.Buff > 0` → zero `rx[]` → `EKI_GetString` → CRC-16/MODBUS check →
version check → `SWITCH` on all eight opcodes → `BuildStatus` → `EKI_Send` →
`EKI_Close` / `EKI_Clear` / `HALT`. Call shapes follow [KUKA, EthernetKRL 2.2]'s
own `BinaryFixed` sample program (`RET=EKI_Init("BinaryFixed")`,
`RET=EKI_Open(...)`, `WAIT FOR $FLAG[1]`, `RET=EKI_GetString(ch,"Buffer",
Bytes[])`, `RET=EKI_Send(ch, Bytes[])`, `EKI_Close` then `EKI_Clear`).

The four named gaps, each marked at the point of use:

1. **No `EKI_CHECK` error handling.** A production program checks the status
   after `EKI_Init` / `EKI_Open`; its exact argument list could not be confirmed
   from the pages obtained, and a guessed error handler is worse than a named
   gap. **This file does not check that the channel opened**, and says so in
   capitals.
2. **`EKI_CheckBuffer`'s parameter list is UNCONFIRMED.** Written as
   `(channel, element tag)` with the count read from `ret.Buff`; the
   `EKI_STATUS` components (`Buff`, `Read`, `Msg_no`, `Connected`) are
   [KUKA]-sourced, the signature is [VENDOR]-usage-sourced.
3. **`CHAR` → `INT` conversion.** The one construct the whole file rests on,
   isolated in a single function (`ByteAt`) precisely so that it is one line to
   check and one line to change. The `INT` → `CHAR` direction is [KUKA] (the
   `BinaryFixed` sample writes `Bytes[i]=0`); the `CHAR` → `INT` direction was
   found only on [FORUM] and is labelled as such.
4. **`ESTOP` is not a Category-0 stop and the file says so.** No KRL instruction
   can command one — IEC 60204 Category 0 is the safety controller's, wired to
   the KRC safety interface. The branch drops the vacuum so a held cell is not
   carried through an unattended stop, answers the host, and `HALT`s, which is
   the strongest refusal-to-continue KRL offers. A real integration **must**
   route the host E-stop into the safety circuit as well.

### 3.1 A finding the audit missed: `CAST_FROM` cannot decode this frame

The audit (§2, and the brief following it) expects `CAST_FROM` to split the
binary record into fields, which is the documented route [KUKA, EthernetKRL 2.2:
"Binary data records of fixed length must be divided into individual variables
again in the KRL program with `CAST_FROM()`"]. It cannot carry *this* frame, for
two reasons:

1. **KRL has no 16-bit numeric type.** `CAST_FROM` decodes type-specifically and
   the types available to it are `REAL` (4 bytes), `INT` (4 bytes), `CHAR`
   (1 byte) and `BOOL` (1 byte). The frame's `z` (int16), `aux` (uint16) and
   `crc16` (uint16) fields are 16 bits each. There is no variable `CAST_FROM`
   could decode them into.
2. **Byte order.** The frame is big-endian by definition (`struct` format
   `">BBiihH"`). No KUKA document obtained here states the byte order
   `CAST_FROM` uses, and the only claims found were [FORUM] and mutually
   inconsistent.

So the fields are decoded byte by byte in explicit big-endian order, which is
correct whatever the controller's native order is, and the question never has to
be answered. The same walk is needed for the CRC in any case. The file explains
this where the decoders are, not only here.

### 3.2 The KRL-side CRC-16/MODBUS

KRL has no CRC primitive and nothing in this repository had ever written one.
`Crc16Modbus(buf[], n)` is bit-serial and table-free, the same shape as
`execution.protocol.crc16_modbus`. KRL has no shift operator, so `>> 1` is
integer division by 2 — `crc` is held in 0…65 535 throughout, so it is never
negative and KRL's truncating `INT` division is the arithmetic that is wanted.
Bit operators are infix and hexadecimal literals are written `'HA001'`
[KUKA, Expert Programming, "Bit operators": `A = 10 B_EXOR 9` gives `A=3`;
`A = B_NOT 'HC5'`], with `B_AND` binding tighter than `B_EXOR`, which binds
tighter than the comparisons.

Status assembly writes big-endian integers from the low end, masking the low
byte off and *subtracting* it before dividing, so the division is exact and
truncation-toward-zero cannot corrupt a negative value. That matters: the
workspace is centred on the robot, so roughly half the coordinates the reply
carries are negative.

### 3.3 The audit's `0x00` risk, carried forward and made loud

Audit §1.4 raises, on [FORUM] evidence only, that KRL string handling may
terminate at `0x00` and that `EKI_GetString` therefore might deliver two usable
bytes out of a frame dense with zeros. That is unresolved and unresolvable here.
It is mitigated only in the sense that the failure is loud rather than silent:
`rx[]` is zeroed before every read, so a truncated record fails the CRC check and
answers `CRC_ERROR` instead of executing a corrupt motion. The file says that if
every frame on real hardware answers `CRC_ERROR`, this is the first thing to
test.

---

## 4. `protocol.py` and the test

### 4.1 The false KUKA fact

The module docstring asserted that "EthernetKRL's own transport is XML" and used
it to argue that the CRC is this repository's addition. **The conclusion was
right and the reason was wrong**, in the project's own disfavour: KUKA's manual
documents three connection data types — XML, fixed-length binary and a variable
binary stream — so a 16-byte fixed-length record is a *native* EKI capability
configured with `<RAW>`, not a departure from EKI. The docstring now says that,
cites the manual's "Binary data records are not interpreted by the EKI and
stored together in a memory", points at the rewritten XML and the new loop, and
keeps the part that is true: **EKI carries no checksum in any of its three
modes**, so the CRC-16 trailer is this repository's. `crc16_modbus`'s own
docstring carried the same error in miniature and is corrected the same way.

### 4.2 `tests/test_execution.py`

`test_krl_subroutine_returns_the_numbers_this_enum_declares` regex-pinned
`RETURN n ; NAME` against `RobotStatusCode` — sound intent, and the effect was
to put a green tick next to a construct that does not compile. It is kept
(the enum coupling is real and worth pinning) and its docstring now **states its
scope**: it couples the enum to the KRL *labels*, it does not compile, parse or
execute KRL, and no test in this repository does.

Three tests are added, over a small two-token scan of `routines.src`:

* **`test_krl_valued_returns_are_inside_a_deffct_not_a_def`** — every routine
  containing a `RETURN <int>` must be declared `DEFFCT`, must declare a return
  type, and must be terminated `ENDFCT`; and `PickAndPlace` specifically must be
  `DEFFCT INT`. This is the assertion that would have caught §1.1.
* **`test_krl_first_def_matches_the_file_name`** — the leading declaration must
  be `routines`. Catches §1.4.
* **`test_krl_subroutines_reachable_from_another_src_are_global`** — every
  declaration after the first must carry `GLOBAL`. Catches §1.3.

None of them validates KRL beyond these three structural rules, and none of them
compiles anything. Suite: **1 221 passed, 1 skipped** (from 1 218 + 1).

---

## 5. The ±350 mm square, labelled

`REACH_MM = 706.0` in `mock_kuka_server.py` is a correct, very slightly
conservative transcription of the datasheet's 706.7 mm and is untouched.

The ±350 mm square is kept — as an *outer* bound it is conservative and safe,
its worst case being the corner at 350√2 ≈ 495 mm — and is now labelled a
**placeholder** with its three limitations named, in three places a reader will
actually hit:

* **`configs/planning.yaml`**, at the value itself: it contains the base and a
  6-axis arm cannot reach its own base column, so the true envelope is an
  annulus with an inner dead zone; A1 travels ±170°, not ±180°, leaving a ~20°
  unreachable wedge directly behind the robot that a square centred on the base
  claims; and reach is a shell, not a prism, so a square valid at Z = 300 mm is
  not valid at Z = 0. The dead-zone radius is deliberately not quoted — the
  datasheet's working-range diagram is a raster image that could not be
  measured, so the claim is qualitative on purpose.
* **`plan/scene.py`**, in `WorkspaceBounds`' docstring, where the enforcement
  lives: everything inside the square is within reach *on radius*; not
  everything inside it is reachable.
* **`main.py`**, where "a 700 mm envelope" now reads as the placeholder it is.

---

## 6. `docs/FDR_v3.md`

Three passages were falsified or narrowed by the work above and are corrected in
the report's own voice:

* **§7.1** — the paragraph recording the XML-vs-binary contradiction now records
  its resolution instead, states plainly that the report previously implied EKI
  had no binary mode and that it has one, and gives the honest description of
  the execution layer: *a binary command protocol designed for EthernetKRL's
  fixed-length binary mode, with a controller-side program written from the
  manuals and unverified against hardware* — **not** "implemented to the
  EthernetKRL 3.1 specification". The two-pose / one-coordinate mismatch stays,
  unfixed by design, with a pointer to how `laptop_comm.src` implements the
  §7.3 convention.
* **§7.3** — names the `DEFFCT`, names `laptop_comm.src`, splits the velocity
  claim into the `LIN` segments that `$VEL.CP` governs and the `PTP` segments it
  does not, and records that **`PLACE_FAILED` (3) is a simulator behaviour with
  no controller-side counterpart**: nothing on this arm reports that a cell
  seated.
* **Appendix D** — "the only real cap is `$VEL.CP`" narrowed to the `LIN`
  segments, with `$VEL_AXIS[]`/`$ACC_AXIS[]` named for the rest.

**Not done, and named as not done:** the phrase "implemented to the EthernetKRL
3.1 specification" also appears in the abstract, §1, §10.3 and ADR-003. §7.1 now
contradicts it in the report's own voice, which is better than leaving it
unremarked but is not the same as rewording all four. That is the audit's
recommendation §9 bullet 1 and it remains open.

---

## 7. Everything that could not be verified without hardware

Grouped so it can be worked through with a controller in one sitting.

**Would-not-compile / would-not-load verdicts.** Every one of them —
`DEF`+`RETURN`, the filename rule, `GLOBAL`, the `$ANOUT` range, the missing
`BAS(#INITMOV,0)` — is a documentation-based inference. The first two rest on
explicit statements in KUKA's own manual and are as strong as a non-compiler
verdict gets; none is a compiler result. The same is true in reverse of every
fix: **nothing in `routines.src`, `laptop_comm.src` or `laptop-comm.xml` has
been compiled, parsed by EKI, loaded, or run.**

**Constructs used whose source is weaker than [KUKA]:**

| Construct | Where | Status |
|---|---|---|
| `CHAR` → `INT` by assignment | `ByteAt`, `laptop_comm.src` | [FORUM] only. The whole decode and the whole CRC pass through it. |
| `EKI_CheckBuffer(channel, tag)` signature | receive loop | [VENDOR] usage; KUKA's function-reference page not obtained. |
| `EKI_STATUS` components `Buff`/`Read`/`Msg_no`/`Connected` | receive loop | [KUKA] page extract, single source. |
| `REAL` → `INT` assignment rounds | `BuildStatus` | [KUKA] type-conversion section, corroborated [FORUM]. |
| `EXT` / `EXTFCT` being required at all for `GLOBAL` cross-SRC calls | `laptop_comm.src` | Unestablished. Declared anyway; correct KRL either way. |
| `<INTERNAL>` holding the controller's own IP/PORT in server mode | `laptop-comm.xml` | Inferred from `<EXTERNAL>`'s documented meaning. |
| `BUFFERING` / `BUFFSIZE` as attributes of `<INTERNAL>` | `laptop-comm.xml` | One extracted page calls them attributes without naming the parent. |
| `<ALIVE Set_Flag="2"/>` shape | `laptop-comm.xml` | Not confirmed. Nothing depends on it. |

**Behaviours that only hardware can settle:**

* whether `EKI_GetString` truncates a binary record at the first `0x00` (§3.3).
  If it does, every frame answers `CRC_ERROR` and the binary design needs a
  different carrier. This is the single highest-value thing to test first.
* whether `TOOL_DATA[1]` / `BASE_DATA[1]` are the right indices, and what the
  taught values are. There is no cell.
* whether 0–10 V on the analog channel means 0–100 % vacuum. There is no
  gripper datasheet.
* whether 2 mm is the right insert depth. It matches
  `mock_kuka_server._INSERT_Z_MM` so that the simulator and the KRL describe the
  same cycle; both are placeholders for a taught value.
* round-trip timing. Real EKI deposits data in a buffer the KRL interpreter
  polls on its own schedule, bound to interpolation cycle and advance-run state;
  the mock answers from its socket handler. No timing measured against the mock
  predicts the controller, and the FDR does not claim it does.

**Out of scope and untouched, per the brief:** the wire format, `KukaClient`'s
behaviour, every metric definition, every dataset and every model.
