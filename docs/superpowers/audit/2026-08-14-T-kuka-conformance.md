# Audit T — KUKA conformance of the robot-side artefacts

**Date:** 2026-08-14 · **Tree:** `d:\dev\auto-pick` @ `54f6790` (`feat/blender-synth-dataset`)
**Scope:** `execution/krl_prog/laptop-comm.xml`, `execution/krl_prog/routines.src`, and the
FDR claims that rest on them (`docs/FDR_v3.md` §§7.1, 7.3, Appendix D, ADR-003), checked against
published KUKA documentation. Secondary: `execution/protocol.py`,
`execution/mock_kuka_server.py`, `configs/planning.yaml`, `tests/test_execution.py`.
**Mode:** read-only. Nothing modified, staged or committed.

**Source weighting.** **[KUKA]** marks a claim taken from a KUKA-authored document (the
Expert Programming manual, the System Variables manual, the EthernetKRL manual, the KR 6 R700
datasheet). **[VENDOR]** marks a third-party integration manual from a robotics vendor
documenting its own KUKA interface (Roboception, Balluff) — these reproduce EKI usage but are
not KUKA. **[FORUM]** marks a forum or blog, used only where nothing better exists and
weighted accordingly. **[READ]** marks a fact about this repository established by reading it.

**Standard of judgement.** The project is explicitly software-only with a mock robot, and that
is legitimate. The question this audit answers is narrower: would a reader with KUKA experience
conclude that the FDR's claim to target *EthernetKRL 3.1 on a KR 6 R700* is honest, and is each
deviation labelled as the simplification it is. Findings are graded **[NO-COMPILE]** (the KRL
compiler or the EKI parser rejects it), **[MISBEHAVES]** (it is accepted but does not do what
the comments say), and **[SIMPLIFICATION]** (fine for a software-only project, but it must be
labelled).

---

## 0. Sources

| # | Document | Class |
|---|---|---|
| S1 | KUKA, *Expert Programming — KUKA System Software (KSS) Release 5.2*, `ProgExperteBHR5.2 09.03.00 en`, 178 pp. Retrieved as PDF and text-extracted locally. | [KUKA] |
| S2 | KUKA, *System Variables — KUKA System Software*, `SysVar 08.02.03 en`, 170 pp. Retrieved as PDF and text-extracted locally. | [KUKA] |
| S3 | KUKA, *KUKA.EthernetKRL 2.2*, System Technology manual (via ManualsLib page 33, "Reading Out Data"). | [KUKA] |
| S4 | KUKA, *KR 6 R700 sixx* datasheet, doc. `0000-210-361 / V23.1 / 06.05.2019 / en`, KUKA Deutschland GmbH, Augsburg. | [KUKA] |
| S5 | Roboception, *KUKA Ethernet KRL Interface*, `rc_visard` documentation. | [VENDOR] |
| S6 | Balluff, *KUKA Ethernet KRL Interface*, `BVS 3D-RV0` documentation `DRF_957554_AA_000`. | [VENDOR] |
| S7 | Robot-Forum threads on `$ANOUT` scaling, EKI binary reads, and `RETURN` usage. | [FORUM] |

S1 and S2 are older KSS releases than KRC4/KSS 8.x. Every construct they are cited for here
(`DEF`/`DEFFCT`/`RETURN`, `$ANOUT`, `$VEL.CP`, `$ACC.CP`, `$VEL_AXIS`, the file-naming rule,
the I/O advance-run stop) is core KRL that is unchanged in 8.x, and S5/S6 confirm the EKI
surface on current controllers. Where a claim might be version-sensitive it is flagged.

---

## 1. Can the XML config and the binary protocol interoperate?

**Verdict: as committed, no — but the premise in the brief is wrong in the project's favour, and
that correction matters.**

### 1.1 EthernetKRL *does* support raw binary. The FDR and `protocol.py` both say otherwise.

The brief asked whether EKI can carry raw binary at all. It can. KUKA's own EthernetKRL 2.2
manual documents three connection data types — XML, binary of fixed length, and a variable
binary stream — and distinguishes their handling explicitly [KUKA, S3]:

> "XML data are extracted by the EKI and stored type-specifically in different memories. It is
> possible to access each saved value individually." … "Binary data records are not interpreted
> by the EKI and stored together in a memory."

Binary is configured with a `<RAW>` block in place of `<XML>`, e.g. [KUKA, S3]:

```xml
<RAW><ELEMENT Tag="Buffer" Type="STREAM" EOS="65,66" /></RAW>
```

and read back with [KUKA, S3]:

> "The `EKI_GetString()` access function must be used to read a binary data record out of a
> memory. Binary data records are read out of the memory as strings." … "Binary data records of
> fixed length must be divided into individual variables again in the KRL program with
> `CAST_FROM()`."

This feature is present from EthernetKRL 2.2 onward, and the version the FDR targets — 3.1 —
is inside the supported band (S5 and S6 both state the add-on must be "version 2.2 up to
version 5.x" / "2.2 or newer") [VENDOR].

**Consequence for the repository.** `execution/protocol.py`'s module docstring asserts
(lines 25–26) that "EthernetKRL's own transport is XML". That is **factually wrong**, and it is
load-bearing: the docstring uses it to argue that the CRC "is a binary framing this repository
defines". The conclusion is right, the reason given for it is not. A 16-byte fixed-length binary
frame is a *native* EKI capability (`BinaryFixed`), not a departure from EKI. The honest
statement is: *EKI supports fixed-length binary, this repository uses it, and the CRC trailer is
this repository's addition because EKI supplies no checksum.* FDR §7.1 (line ~1385) inherits the
same error more mildly — it says the XML file has "no binary mode", which is accurate about the
file, but the surrounding argument reads as though binary mode did not exist.

This is the one place where the project is **harder on itself than the facts warrant**. Worth
correcting, in the project's favour.

### 1.2 The committed XML nevertheless cannot carry the binary frame

`laptop-comm.xml` [READ] configures `<Send>` as three REAL elements and `<Receive>` as one INT
element, both inside `<XML>` blocks. A client that writes 16 raw bytes at this channel hands
them to EKI's XML parser, which will not parse them. The two committed halves do not
interoperate. Reconciling them requires **replacing** `<XML>` with `<RAW>` on both directions
and a fixed 16-byte record length — a different file, not a different reading of this one.

### 1.3 The XML is also not a valid EKI configuration file, in either mode

Independently of XML-vs-binary, this file would not be accepted. Real EKI configuration files,
per two vendor manuals that reproduce KUKA's schema [VENDOR, S5 and S6]:

```xml
<ETHERNETKRL>
  <CONFIGURATION>
    <EXTERNAL><IP>…</IP><PORT>…</PORT></EXTERNAL>
    <INTERNAL><BUFFERING>500</BUFFERING></INTERNAL>
  </CONFIGURATION>
  <SEND><XML><ELEMENT Tag="req/parameters/quality/@value" Type="STRING"/></XML></SEND>
  <RECEIVE><XML>
    <ELEMENT Tag="res/return_code/@value" Type="INT"/>
    <ELEMENT Tag="res" Set_Flag="998"/>
  </XML></RECEIVE>
</ETHERNETKRL>
```

Against that, `laptop-comm.xml` diverges on four counts [READ]:

1. **Section names.** It has `<Config>`; EKI expects `<CONFIGURATION>`, and expects `<EXTERNAL>`
   and `<INTERNAL>` children inside it. There is no `<EXTERNAL>` wrapper around `<IP>`/`<Port>`.
2. **Case.** `<EthernetKRL>`, `<Send>`, `<Receive>`, `<Port>`, `<Element>` against KUKA's
   `<ETHERNETKRL>`, `<SEND>`, `<RECEIVE>`, `<PORT>`, `<ELEMENT>`. XML is case-sensitive by
   specification, and every KUKA and vendor example uses the upper-case forms while keeping the
   *attributes* mixed-case (`Tag`, `Type`, `Set_Flag`) — a specific convention the file gets
   backwards on elements and right on attributes.
3. **Nesting.** The file nests `<Element Tag="Command">` around `<Element Tag="Target">`. EKI
   uses a **flat** list of `<ELEMENT>` entries whose `Tag` carries the XPath-like path. The
   intended declaration is `<ELEMENT Tag="Command/Target" Type="INT"/>`, not a nest.
4. **No arrival flag.** There is no `Set_Flag` on the receive root. That is the mechanism by
   which KRL learns a telegram arrived [VENDOR, S5/S6]; without it the KRL side has nothing to
   trigger on.

The IP `172.31.1.147` and port `54600` are plausible — that address is the conventional KRC4 KLI
default — so the file looks right at a glance and is wrong in every structural particular.

### 1.4 A caution if the RAW route were taken

Even a correct `<RAW>` config would meet a real obstacle with *this* frame. EKI hands binary
records to KRL **as strings** [KUKA, S3], and KRL string handling terminates at `0x00`
[FORUM, S7 — flagged as forum-sourced, and I found no KUKA statement either way]. The 16-byte
frame is dense with zero bytes: `PROTOCOL_VERSION = 0x01` is followed by an opcode, then two
big-endian `int32`s whose upper bytes are `0x00` for every coordinate under ±8 388 608 mm —
i.e. always. A `MOVE_TO(200, 150, 80)` frame contains at least seven `0x00` bytes, the first at
offset 2. If the forum claim holds, `EKI_GetString` would surface a two-byte string. This is not
established from KUKA documentation and should be treated as a risk to verify on hardware, not
a finding — but it is the kind of thing that decides whether a binary EKI design survives
contact, and no artefact in this repository shows awareness of it.

---

## 2. The receive loop: what a real EKI program needs, and what is here

**Verdict: there is no robot-side program at all — only four leaf subroutines with no caller.**

A minimal working EKI channel requires, at least [VENDOR, S5 and S6; function set corroborated
by KUKA S3]:

| Step | Call | Present in `routines.src`? |
|---|---|---|
| Declare status | `DECL EKI_STATUS RET` | no |
| Initialise | `RET = EKI_INIT("laptop-comm")` — argument is the XML filename without extension | no |
| Open | `RET = EKI_Open("laptop-comm")` | no |
| Poll arrival | `RET = EKI_CheckBuffer(…)` — "used to request the number of instances that can then be read" | no |
| Read fields | `EKI_GetInt` / `EKI_GetReal` / `EKI_GetString` / `EKI_GetFrame` | no |
| Reply | `EKI_Send(…)` after `EKI_SetInt`/`EKI_SetReal`/`EKI_SetString` | no |
| Clear | `EKI_ClearBuffer(…)` | no |
| Close | `EKI_Close("laptop-comm")` | no |

`routines.src` contains **zero** `EKI_` calls [READ]. Beyond the calls, it is also missing every
surrounding element a dispatcher needs: no loop, no `SWITCH`/`CASE` over the opcode, no CRC-16
routine (KRL has no CRC primitive — the 0xA001 table-free loop in `protocol.py` would have to be
hand-written in KRL, and no such code exists anywhere in the repo), no status-frame assembly, and
no `INTERRUPT` armed on the `Set_Flag` that §1.3 notes is absent from the config.

Two further structural blockers mean the four subroutines could not be reached even if a loop
were written elsewhere:

**(a) They are local, and nothing declares them `GLOBAL`.** KUKA is explicit [KUKA, S1]:

> "In the case of local subprograms or functions, the main program and the subprograms/functions
> are found in the same SRC file. … Local subprograms/functions can only be called from within
> the SRC file in which they were programmed." … "If it is necessary to be able to call
> subprograms/functions from other programs they must be global, i.e. saved in a separate SRC
> file. Alternatively, a local subprogram can be preceded by the keyword `GLOBAL`."

None of the four carries `GLOBAL` [READ]. A receive loop in `laptop_comm.src` could not call
`PickAndPlace`.

**(b) The file's first `DEF` does not match the filename.** [KUKA, S1]:

> "The object name without an extension is also the name of the file and is therefore prefixed
> by `DEF`. … Every file begins with the declaration `DEF` and ends with `END`."

`routines.src` begins `DEF MoveBetweenPositions(pos1 :IN, pos2 :IN)` [READ]. The leading `DEF`
must be `routines()`. As written, the file declares an object named `MoveBetweenPositions` in a
file named `routines` — the first thing the KRL editor would reject on load.

`MoveBetweenPositions` is additionally dead: no opcode in `OpCode` maps to it [READ].

**Grade: [NO-COMPILE] for (b); the absence of a receive loop is not a defect in KRL syntax but a
missing half of the system.** FDR §7.1 does say "Nothing executes the KRL side in this project",
which is true and correctly stated. What is *not* stated anywhere is that no KRL side exists to
execute — a reader is invited to believe the loop is present but unrun.

---

## 3. `DEF` … `RETURN 1` — verdict

**Verdict: [NO-COMPILE]. The brief's understanding is correct.**

KUKA is unambiguous [KUKA, S1, §6 Subprograms and functions]:

> "Unlike a subprogram, a function sends back a return value. A function begins with the keyword
> `DEFFCT`. The data type of the return value is specified directly after the keyword `DEFFCT`.
> The return value itself is transferred via `RETURN`. The function is terminated using the
> keyword `ENDFCT`."

with the canonical example:

```krl
DEFFCT INT Function()
  DECL INT Sample
  Sample = 11
  RETURN Sample
ENDFCT
```

and the file-concept section states the same distinction at the top of the manual: "There are
two variants: `DEF` and `DEFFCT` (with return value)" [KUKA, S1].

`PickAndPlace` is declared `DEF PickAndPlace(...)` … `END` and executes `RETURN 2` and
`RETURN 1` [READ]. In a subprogram `RETURN` takes no argument — it simply exits the routine
[KUKA, S1 by construction; corroborated FORUM, S7: "since subprograms do not return a value, no
parameter is used with `RETURN`"]. Supplying an operand is an inadmissible instruction.

The correct form is `DEFFCT INT PickAndPlace(...)` … `ENDFCT`, and the caller must then consume
the value (`status = PickAndPlace(...)`) rather than invoking it as a statement.

**Aggravating detail.** `tests/test_execution.py:552` (`test_krl_subroutine_returns_the_numbers_this_enum_declares`)
regex-scans `routines.src` for `RETURN n ; NAME` and asserts each matches `RobotStatusCode`
[READ]. The test's intent is sound — it couples the enum to the KRL so renumbering cannot
silently redefine the wire. But its effect is to **pin a construct that does not compile**, and
to put a green tick next to the KRL file in a suite that never compiles or runs it. A reader
scanning the test names would reasonably infer the KRL had been validated. It has not; only its
comment labels have.

---

## 4. `$ANOUT` — verdict

**Verdict: [MISBEHAVES]. The brief's understanding is correct.**

KUKA's System Variables manual gives the entry verbatim [KUKA, S2]:

> `$ANOUT[n]` **Analog outputs** — Data type **Real** · Value min **−1.0** · Value max **+1.0** ·
> Unit **V** · Original line `REAL $ANOUT[n]` · Comments `[n] = [1] ... [16]`,
> `−1.0 ⟶ −10 V`, `+1.0 ⟶ +10 V`

`routines.src:49` writes `$ANOUT[analog_out] = vacuum_level`, where `vacuum_level` is
`DECL INT` documented at line 41 as "0..100 percentage" [READ].

- It very likely **compiles**: KRL widens `INT` to `REAL` implicitly, and the index (`1`, from
  the `VacuumControl(1, 1, …)` call sites) is inside `[1..16]`.
- It **misbehaves at runtime**. Every value from 1 to 100 is out of range by a factor of 1 to
  100. Out-of-range writes are clamped and annunciated — the manual states that attempting to
  set a value outside the range displays a `Limit {Signal name}` message [KUKA, S2, corroborated
  FORUM, S7]. The documented 0–100 % vacuum scale therefore collapses to a two-state output:
  `0` → 0 V, and *every* value 1–100 → +10 V full scale, each accompanied by a controller
  message. A `vacuum_pct` of 40 and of 95 are indistinguishable at the gripper.
- The correct expression is `$ANOUT[analog_out] = vacuum_level / 100.0`.

This is the most consequential single-line defect in the file, because it is silent in exactly
the direction that looks fine: the vacuum works, at full power, always, and the percentage
parameter that the Python side carefully threads through `aux_u16` does nothing. Note that
`mock_kuka_server.py`'s docstring already lists "analog-output range checks on the vacuum level"
among the things it does not model [READ] — the *gap* is labelled, but the KRL bug it conceals
is not identified.

---

## 5. `$VEL.CP` and `$ACC.CP` — verdict

**Verdict: the units are correct and the comment is right. This one is fine. The defect is
elsewhere in the same two lines.**

KUKA's System Variables manual [KUKA, S2]:

> `$VEL` — Data type Structure · `DECL CP $VEL` · Comments: "Velocities in the advance run.
> **CP = m/s**, ORI1 = °/s, ORI2 = °/s"
> `$VEL.CP` — **CP velocity in the advance run** · Data type Real · Value min **>0** · **Unit m/s**

> `$ACC` — Data type Structure · `DECL CP $ACC` · Comments: "Accelerations in the advance run.
> **CP = m/s²**, ORI1 = °/s², ORI2 = °/s²"

So `$VEL.CP = 0.150` **is** 150 mm/s, exactly as `routines.src:68` claims. **The comment is
correct and should not be "fixed".** `$ACC.CP = 0.500` is 0.5 m/s² — legal (`>0`), conservative
(roughly g/20), and entirely sane: it reaches 150 mm/s in 0.3 s over 22 mm of path. For a KR 6
R700 doing cell insertion both numbers are defensible and unremarkable. 150 mm/s also sits well
under the 250 mm/s reduced-velocity limit that applies in T1.

**But the values govern less than the file implies.** KUKA's PTP section [KUKA, S1]:

> "The following system variables are used: `$VEL_AXIS[axis number]`: to program maximum
> axis-specific velocities, and `$ACC_AXIS[axis number]`: to program maximum axis-specific
> acceleration rates. All entries are given as percentages of the maximum value defined in the
> machine data. **If these two system variables have not been programmed for all axes, execution
> of the program will cause a corresponding error message to be generated.**"

Two consequences [READ]:

1. **`$VEL.CP` does not govern `PTP`.** `SimpleMove` sets `$VEL.CP = vel` and then issues
   `PTP pos` — the assignment has no effect on that motion. In `PickAndPlace`, only the three
   `LIN` moves are governed by `$VEL.CP`; the two `PTP` moves (the approach and the transport)
   are not. **[MISBEHAVES]** — mild, but it means `SimpleMove`'s `vel` and `acc` parameters are
   inert, which is the whole purpose of that subroutine.
2. **No motion initialisation exists.** `routines.src` has no `INI` fold and no
   `BAS(#INITMOV,0)` call, so `$VEL_AXIS[]` and `$ACC_AXIS[]` are never programmed. Per the
   quotation above, that is a documented runtime error on the first `PTP`. **[NO-COMPILE-adjacent
   — it compiles and faults at execution.]** Nor is `$BASE`, `$TOOL` or `$IPO_MODE` ever set,
   which Cartesian motion requires before any `LIN` to an `E6POS` is meaningful.

**Bearing on the FDR.** Appendix D (line ~4362) states "The only real cap is controller-side —
`$VEL.CP = 0.150` in `krl_prog/routines.src`, plus the KRC's T1/T2 limits". That is half true:
it caps the LIN segments and not the PTP segments. The claim should be narrowed.

---

## 6. Other KRL findings

**Correct, and worth recording as correct** [READ, checked against KUKA S1/S2]:

- Parameter style. `DEF Name(p :IN)` followed by `DECL <TYPE> p` on the next lines matches the
  manual's rule that parameter variables are declared immediately after the `DEF` line. The
  space before `:IN` is cosmetic. `DECL` is legal for structure types. **Fine.**
- `E6POS` component access. `approach_pos.Z = pick_pos.Z + 60` uses the correct component name;
  whole-struct assignment `approach_pos = pick_pos` is valid KRL. **Fine.**
- `IF $IN[10] == FALSE THEN` … `ENDIF`. Correct comparison operator, correct block terminator,
  and `$IN[]` is `BOOL` [KUKA, S1 §5.2: "Binary outputs … are therefore treated as BOOL-type
  variables. The state of an `$IN[No]` input can be read into a Boolean variable"]. **Fine.**
- **The advance-run trap does *not* apply here**, which is worth saying because it is the
  obvious objection. KUKA [KUKA, S1 §5.1]: "For safety reasons, all input/output instructions
  and access to inputs/outputs via system variables trigger an advance run stop." So the
  `$IN[10]` read after `WAIT SEC 0.05` is evaluated in the main run, not the advance run, with
  no `CONTINUE` needed. **Correct as written.**
- `$OUT[digital_out] = vacuum_on` with a `BOOL`. **Fine.**
- `WAIT SEC 1` / `WAIT SEC wait_time`. **Fine.**

**Defects not covered above:**

- **`SimpleMove`'s comment contradicts its code.** Line 21–22 says "`pos` is the commanded
  Cartesian pose (we copy the current orientation)". No such copy exists in the body [READ].
  This matters because the 16-byte frame carries only X/Y/Z — A, B, C and the status/turn bits
  S and T must come from somewhere, and nothing in `routines.src` supplies them. `PTP`/`LIN` to
  an `E6POS` with unset orientation and unset S/T is not a well-formed motion. **[MISBEHAVES]**,
  and it is the deepest unaddressed gap between the wire format and the arm: a 3-DoF protocol
  driving a 6-DoF machine with no stated convention for the other three.
- **`PickAndPlace` takes two poses; the opcode carries one.** FDR §7.1 already names this
  honestly (line ~1389) and §7.3 documents the two-packet workaround. **Correctly labelled —
  no action.**

---

## 7. The KR 6 R700 envelope, and the ±350 mm square

**Datasheet, official** [KUKA, S4 — `0000-210-361 / V23.1 / 06.05.2019`]:

| | |
|---|---|
| Maximum reach | **706.7 mm** |
| Maximum payload | **6 kg** (rated 3 kg "in order to optimize the dynamic performance") |
| Axes | 6 · Repeatability ±0.03 mm · Weight ~50 kg · Footprint 320 × 320 mm |
| A1 motion range | **±170°** |
| A2 / A3 | −190°/45° · −120°/156° |
| Controller | **KR C4 smallsize-2; KR C4 compact** |

The brief's recollection of ~706 mm is right, and `mock_kuka_server.py`'s `REACH_MM = 706.0` is
a correct (very slightly conservative) transcription [READ]. The controller family is genuinely
KRC4, so "KRC4-class" in the FDR is accurate.

**The ±350 mm square** (`configs/planning.yaml:66`, `camera.workspace_bounds_mm`) is enforced —
`plan.scene.WorkspaceBounds.require` is called from `Planner._build_pose` on both pick and place
poses, and `main.py` raises if a whole run is declined as unreachable [READ]. That enforcement
is good, and is itself the fix for an earlier audit finding.

**Is it a defensible approximation?** Partly. As an *outer* bound it is conservative and safe:
the worst case is the corner at 350√2 ≈ **495 mm**, comfortably inside 706.7 mm. Nothing inside
the square is beyond the arm's reach on radius alone.

**But it is not the reachable workspace, on four counts** [KUKA S4 for the axis data; geometry
is arithmetic]:

1. **It contains the base.** The square includes (0, 0) and its neighbourhood. A 6-axis arm
   cannot reach its own base column; the true envelope is an annulus with an inner unreachable
   cylinder. Every point within roughly a couple of hundred millimetres of the origin is claimed
   reachable and is not.
2. **A1 is ±170°, not ±180°.** There is a ~20° unreachable wedge directly behind the robot. A
   square centred on the base spans all azimuths and so claims reach into that wedge.
3. **Reach is a shell, not a prism.** The 706.7 mm figure is a radius; usable horizontal radius
   shrinks as Z approaches the table. A square valid at Z = 300 mm is not valid at Z = 0. The
   mock's spherical `sqrt(x²+y²+z²) ≤ 706` [READ] models this better than the planner's square
   does, so the two layers disagree about the same robot.
4. **Reach is quoted to the wrist reference point**, not to a TCP with a vacuum tool fitted. A
   tool changes the envelope in both directions.

**Verdict: a defensible conservative placeholder, and legitimately so for a software-only
project — but a placeholder, and not labelled as one.** `main.py:434` describes it as "a 700 mm
envelope" and the FDR treats it as *the* robot workspace envelope. Neither mentions the inner
dead zone, the A1 notch, or the Z-dependence. One sentence in the FDR saying "a square inscribed
in the reachable annulus, deliberately conservative; the real envelope is an annulus with a 20°
notch and would be modelled from the datasheet's working-range diagram at integration" would
close this entirely. As it stands, it is the kind of number a KUKA-experienced reader spots in
about ten seconds and it costs more credibility than it needs to.

---

## 8. Does the mock faithfully represent a real controller?

**For the claims the FDR actually makes: partly, and its self-labelling is unusually good.**

**Credit where due** [READ]. `mock_kuka_server.py`'s docstring enumerates what it does *not*
model — drive-enable and operating-mode gates, a velocity cap, analog-output range checks,
single-channel exclusivity, heartbeat watchdog — and it models four things a real KRC does
enforce (the 706 mm envelope, a latching Category-0 stop that no reconnect clears, socket
timeouts, distinct fault codes). Naming the omissions in the artefact itself is better practice
than most of this class of project manages, and it should be said plainly.

**Where it diverges in ways that bear on the FDR's claims:**

1. **It validates the Python against itself, not against KUKA.** The mock speaks the 16-byte
   binary protocol; no correctly-configured EKI channel in this repository speaks it (§1.2). The
   prior audit's CRC verification against three independent implementations establishes that the
   CRC is correct **MODBUS** — a MODBUS fact, not a KUKA fact. No test in the repository
   exercises a KUKA behaviour.
2. **The reply model is wrong in kind.** The mock answers every frame with a status frame from
   the socket handler. Real EKI deposits received data in a buffer that the KRL program must
   poll with `EKI_CheckBuffer` and answer explicitly with `EKI_Send`, on the KRL interpreter's
   schedule [VENDOR, S5/S6]. Reply latency on a real KRC is bound to interpreter cycle and
   advance-run state, not to socket handling. **Any round-trip timing measured against the mock
   therefore has no predictive relationship to the real controller** — and, to the FDR's credit,
   §7 does not appear to claim otherwise; the cited 150–350 ms robot round-trip is presented as
   a simulated figure.
3. **No advance-run model.** `$ADVANCE` is the single largest behavioural difference between "a
   TCP server that updates a coordinate" and a KRC. The mock has no analogue, so any
   host-side reasoning about when a command has *actually* executed is untested.
4. **Nothing exercises `routines.src`.** The only contact any test makes with it is a regex over
   its `RETURN` lines (§3).

---

## 9. Overall: is the KUKA targeting claim honest as written?

**Mixed, and the mix is specific — the disclosure is good, the artefacts are not, and the gap
between them is the problem.**

**Honest, and unusually so:** FDR §7.1 (lines 1382–1404) states outright that
`laptop-comm.xml` and `protocol.py` "cannot both be the interface", names the two-pose /
one-coordinate mismatch in `PickAndPlace`, says "Nothing executes the KRL side in this project",
and explicitly retracts the earlier claim that CRC-16 is "the standard integrity check on the
KUKA EthernetKRL XML transport". The abstract and §1 both state the execution layer "is
implemented to specification but is not validated against the real robot". The withdrawn
vacuum-dwell claim (§11.x) is retracted in the report's own voice. That is a report that has
been through an honesty pass, and it shows.

**Not honest as written, in three respects:**

1. **"Implemented to the EthernetKRL 3.1 specification" overstates what exists.** The phrase
   appears in the abstract, §1, §10.3 and ADR-003. A reader takes it to mean a KRL program that
   conforms to EKI. What exists is a Python binary framing plus four KRL leaf subroutines with
   no channel, no loop, no dispatcher, and no `EKI_` call anywhere (§2). The Python half is
   *compatible in principle* with EKI's `BinaryFixed` mode — a real point in its favour — but
   nothing in the repository configures or implements that. The defensible phrasing is
   "a binary command protocol designed for EthernetKRL's fixed-length binary mode, with the
   controller-side program not implemented".
2. **The KRL artefacts contain hard errors that no disclosure covers.** `RETURN <int>` in a
   `DEF` (§3) does not compile. The file's leading `DEF` does not match its filename (§2b). No
   subroutine is `GLOBAL` (§2a). `$ANOUT = 0..100` silently saturates (§4). `$VEL.CP` does not
   govern the `PTP` moves, and no `BAS(#INITMOV,0)` initialises the ones that do (§5). §7.1
   discloses an *interface* contradiction; it does not disclose that the KRL would not load.
   A reader with KUKA experience opening `routines.src` finds three or four of these inside a
   minute, and the contrast with the FDR's otherwise careful tone is what will cost credibility.
3. **`protocol.py` states a false fact about KUKA** — "EthernetKRL's own transport is XML"
   (§1.1). EKI has supported fixed-length and streaming binary since 2.2. Correcting this
   *strengthens* the design rationale rather than weakening it.

**The fair summary.** The FDR is honest about *not having tested against hardware* and honest
about *the interface contradiction it noticed*. It is not honest — more likely not aware — that
the robot-side artefacts are not valid KUKA at all. The distinction a reader will draw is between
"software-only, controller side not implemented, here is the protocol design" (entirely
respectable, and close to what this project actually is) and "implemented to the EthernetKRL 3.1
specification" (not supported by the artefacts). Moving to the first framing costs the report
nothing it has earned and removes every finding in §9.

**Smallest set of changes that would make the claim honest**, none requiring a robot:

- Re-word "implemented to the EthernetKRL 3.1 specification" to name the controller-side program
  as not implemented (abstract, §1, §10.3, ADR-003).
- Correct `protocol.py`'s "EthernetKRL's own transport is XML" and note that EKI's `BinaryFixed`
  mode is the intended carrier — the CRC remains this repo's addition.
- Add a short "known non-conformances in `krl_prog/`" block listing §§2–5: `DEF`→`DEFFCT`, the
  filename rule, `GLOBAL`, `$ANOUT` scaling, `$VEL.CP`-vs-`PTP`, and the missing `INI`. Either
  fix them or label them; both are defensible, labelling is cheaper and equally honest.
- Label the ±350 mm square as a conservative inscribed placeholder and name the annulus, the
  ±170° A1 notch, and the Z-dependence it does not model (§7).
- Add a comment at `tests/test_execution.py:552` noting that the test pins the enum coupling and
  does **not** validate KRL syntax (§3).

---

## Appendix — verification method

KUKA PDFs were fetched and text-extracted locally with `pypdf` into the session scratchpad;
quotations in §§3–5 are from those extractions and preserve the manuals' own wording (the
extractor renders the degree sign as `/g176` and hyphenates as `-​-`; these are artefacts of
extraction, not of the source). The KR 6 R700 datasheet text in §7 is the complete extractable
body of a single-page KUKA datasheet; its working-range *diagram* is a raster image and could
not be measured, so the inner-dead-zone claim in §7.1 is stated qualitatively from the axis data
and 6-axis kinematics rather than as a dimension. No KUKA controller, KRL compiler, or
OrangeEdit/WorkVisual instance was available, so every "would not compile" verdict is a
documentation-based inference — strongly supported for §3 and §2b, which rest on explicit
statements in KUKA's own manual, but not a compiler result.
