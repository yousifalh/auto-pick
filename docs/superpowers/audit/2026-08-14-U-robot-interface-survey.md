# Audit U — What a second and third robot actually require

**Date:** 2026-08-14 · **Tree:** `d:\dev\auto-pick` @ `54f6790` (`feat/blender-synth-dataset`)
**Scope:** an external survey of Universal Robots (RTDE / `ur_rtde`), ROS 2
(`FollowJointTrajectory`, `ros2_control`, MoveIt), Franka (FCI / `libfranka`) and — as a
corroborating fourth — ABB Externally Guided Motion, against this project's existing
KUKA execution layer (`execution/protocol.py`, `execution/execution.py`,
`configs/planning.yaml`).
**Mode:** read-only. Nothing modified, staged or committed. No code designed — this
establishes requirements only.

**Source weighting.** **[VENDOR]** marks a document authored by the robot manufacturer
(Universal Robots, ABB, Franka Robotics). **[OSS]** marks the canonical open-source
project's own documentation for an interface it defines (`ur_rtde` from SDU Robotics,
`ros-controls`, MoveIt) — authoritative for that library's contract, third-party with
respect to the robot vendor. **[THIRD-PARTY]** marks anything else. **[READ]** marks a
fact about this repository established by reading it.

**A caution on the KUKA column.** Everything in the KUKA/baseline column of the tables
below describes *this project's protocol*, not KUKA's capabilities. KUKA controllers
support far more than a 16-byte X/Y/Z frame. The column is here because it is the one
instance the interface would otherwise be designed from.

---

## 0. Sources

| # | Document | Class |
|---|---|---|
| S1 | Universal Robots, *Real-Time Data Exchange (RTDE) Guide*, `docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html` | [VENDOR] |
| S2 | SDU Robotics, *ur_rtde 1.6.3 — API Reference*, `sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html` | [OSS] |
| S3 | SDU Robotics, *ur_rtde — FAQ*, *servoJ example*, *async move example*, *Use with a Robotiq gripper* | [OSS] |
| S4 | Universal Robots, *Dashboard Server e-Series, port 29999*, article 42728 | [VENDOR] |
| S5 | Universal Robots, *ROS 2 Driver / `ur_client_library` architecture* — Reverse Interface, Trajectory Point Interface | [VENDOR] |
| S6 | `ros-controls/control_msgs`, `action/FollowJointTrajectory.action` and `action/GripperCommand.action`, `master` | [OSS] |
| S7 | `ros-controls`, *joint_trajectory_controller* user documentation, `control.ros.org/rolling` | [OSS] |
| S8 | MoveIt, *MoveIt Task Constructor* concepts, `moveit.picknik.ai/main` | [OSS] |
| S9 | Franka Robotics, *FCI Documentation* — Overview, Troubleshooting | [VENDOR] |
| S10 | Franka Robotics, *libfranka 0.15.0 API* — `franka::Robot`, `franka::Gripper` | [VENDOR] |
| S11 | ABB, *Application manual — Externally Guided Motion (OmniCore)*, `3HAC073318-001` Rev. P, © 2019–2026. Retrieved as PDF, text-extracted locally; line references are to that extraction. | [VENDOR] |
| S12 | `abb_robot_client` documentation, EGM API | [THIRD-PARTY] |

---

## 1. Why Franka FCI is the third robot

The brief offered ABB EGM, Fanuc or a Franka. I chose **Franka FCI**, and included **ABB
EGM** as a fourth column because I had the primary manual in hand and it independently
corroborates the same finding from a different vendor.

The reason for Franka: UR and ROS 2, for all their surface differences, share an
assumption with this project that is easy to miss precisely because all three share it —
**you may hand the controller a goal and then go away.** `moveL` blocks or you poll;
`FollowJointTrajectory` gives you a terminal result; this project's `_cmd_and_wait` blocks
on a status frame. In all three, a slow or absent client is *harmless*.

FCI breaks that assumption completely, and breaks it in the most demanding available form:
the client is the interpolator, it runs at 1 kHz, and a late client is a **fault**, not a
pause. If a `RobotDriver` interface survives contact with FCI it will survive anything
softer. ABB EGM then shows that this is not a Franka eccentricity — a second vendor,
independently, built the same shape at 250 Hz.

Fanuc was rejected as the third choice for the opposite reason: its common integration
paths (Karel/socket messaging, the `fanuc_driver` ROS bridge) are structurally close to
what this project already does, so it would have confirmed a KUKA-shaped interface rather
than stressed it.

---

## 2. Comparison table

### 2.1 Command unit — the deepest question

| | **This project (KUKA/EKI)** | **UR via `ur_rtde`** | **ROS 2** | **Franka FCI** | **ABB EGM** |
|---|---|---|---|---|---|
| Unit | One Cartesian **position** (X/Y/Z int mm) per 16-byte frame [READ] | Several: `moveL(pose)`, `moveJ(q)`, `movePath(waypoints, blend)`, `servoL/servoJ(setpoint)`, `speedL(twist)` [S2] | One **joint trajectory** — `trajectory_msgs/JointTrajectory` with time-parameterised points [S6] | A **callback** returning the next setpoint, invoked every 1 ms [S10] | A **UDP setpoint** every 4 ms (joint array or pose) [S11] |
| Cartesian accepted? | Yes, position only | Yes (`moveL`, `servoL`, pose is `[x,y,z,Rx,Ry,Rz]`) [S2] | **No.** "This controller operates exclusively in joint space and does not accept Cartesian targets" [S7]. Cartesian lives at the MoveIt layer, converted by `ComputeIK` [S8] | Yes (`CartesianPose`) and joint, as separate motion-generator types [S9] | Yes (pose mode) and joint mode [S11] |
| Who interpolates | Controller (KRL `LIN`/`PTP`) | Controller for `move*`; **client** for `servo*`/`speed*` [S2][S3] | Controller (JTC interpolates between points) [S7] | **Client.** FCI has no planner | **Client.** "EGM goes directly into the motor reference generation, i.e. it does not provide any path planning" [S11 §1.3] |
| Speed/time in the command? | No (fixed in KRL) | Yes — `speed`, `acceleration` args [S2] | Yes — `time_from_start` per point [S6] | Implicit in the setpoint sequence | **No.** "It is not possible either to order a movement with a specified speed or order a movement that is supposed to take a specified time" [S11 §1.3] |
| Rate contract | None; request/response | 500 Hz e-Series / UR-Series, 125 Hz CB-series [S2][S1] | None on the action; controller update rate internal | **1 kHz**, read→write within **500 µs** [S9] | 250 Hz; 4.032 ms real, ~4 ms virtual [S11 §1.3] |

### 2.2 Motion completion

| | Reported how | Client posture |
|---|---|---|
| **This project** | A 16-byte status frame with a `RobotStatusCode` and cycle time [READ] | **Blocks** on `recv`, `command_timeout_ms` default 5000 [READ, `execution.py:156`] |
| **UR** | `moveL` **blocks by default**; with `asynchronous=true` poll `getAsyncOperationProgress()` — a return `< 0` means no async operation running *or* one has finished [S2]. `isSteady()` for at-rest [S2] | Blocks **or** polls (client's choice, set per call) |
| **ROS 2** | Action terminal `Result.error_code` ∈ {`SUCCESSFUL`=0, `INVALID_GOAL`=-1, `INVALID_JOINTS`=-2, `OLD_HEADER_TIMESTAMP`=-3, `PATH_TOLERANCE_VIOLATED`=-4, `GOAL_TOLERANCE_VIOLATED`=-5} plus `error_string` [S6] | **Subscribes** — continuous `Feedback` with `desired`/`actual`/`error`/`index`, then one terminal result |
| **Franka** | **There is none.** The *client* declares completion by returning a value marked `MotionFinished()` — "Helper method to indicate that a motion should stop after processing the given command" [S10] | Owns the loop; completion is its own judgement |
| **ABB EGM** | Convergence criteria (`egm_minmax`, defaults ±1.0 mm / ±0.5°) held for `\CondTime` (default **1 s**) before the target "is considered to be reached" and `EGMRunJoint` releases the RAPID pointer [S11 §5.1.8]. The external client sees `mciConvergenceMet` in the `EgmRobot` feedback protobuf [S11 §3.1.3] | Subscribes to a per-cycle boolean; RAPID side blocks unless `\NoWaitCond` |

**The finding.** Completion is an *event* in the delegated systems and a *judgement* in the
streaming ones. Franka has no completion concept at all; ABB manufactures one out of a
tolerance plus a dwell time, on the controller, because there is nothing else to evaluate.
A `RobotDriver.move()` that returns a completion status is therefore not implementable on
FCI without the driver inventing the semantics, and any such invention is a policy
decision (what tolerance? what dwell?) that belongs to the caller, not the driver.

### 2.3 Stop

| | Mechanism | In-band? | Latches? |
|---|---|---|---|
| **This project** | `OpCode.ESTOP = 0x06` [READ] | **Yes — an application opcode** | Yes, client-side: `self._estopped = True`, and `connect()` refuses while set [READ, `execution.py:236`] |
| **UR** | `stopL(a)` / `stopJ(a)` / `servoStop(a)` (decelerate); `stopScript()` (exit mode); `triggerProtectiveStop()` (protective, "for testing/debugging") [S2] | Decelerate + protective stop: yes. **Emergency stop: no** — hardware safety chain | Protective stop latches; **clearing needs the Dashboard Server on port 29999** (`Unlock Protective Stop`, which "fails if less than 5 seconds has passed since the protective stop occurred") [S4] |
| **ROS 2** | Cancel the goal. "Only one action goal can be active at any moment"; a new goal aborts the active one; an empty trajectory on `~/joint_trajectory` overrides without aborting [S7] | Yes, at the action layer | No. On cancel or completion the controller **holds position** (optionally decelerating smoothly) [S7]. **There is no e-stop in the ROS 2 message model at all.** |
| **Franka** | `stop()` — "If a control or motion generator loop is running in another thread, it will be preempted with a `franka::ControlException`" [S10]; reflexes and limit violations throw the same | Yes for `stop()`; safety stops are hardware | **Yes** — errors latch until `automaticErrorRecovery()` [S10] |
| **ABB EGM** | `EGMStop`; `EGM_STOP_HOLD` vs `EGM_STOP_RAMP_DOWN` mode on `EGMRunX`; `\CommTimeout` on `EGMSetupUC`; `ERR_UDPUC_COMM` is recoverable in sync mode but "always raised as fatal error" under `\NoWait` [S11 §5.1.14] | Yes, from RAPID | Yes — and near a singularity "the robot movement will be stopped with an error message. In that situation **the only way is to jog the robot out of the singularity**" [S11 §1.1] |

**The finding.** Four distinct things wear the word "stop": *decelerate-and-hold*, *exit
control mode*, *protective stop*, and *emergency stop*. This project collapses all of them
into one opcode named `ESTOP`. **No surveyed robot accepts a safety-rated stop over its
application protocol** — on UR, Franka and ABB it is a hardware safety chain, and ROS 2's
action layer does not model it at all. An in-band `ESTOP` opcode is a *request to stop
moving*; calling it an emergency stop in a generalised interface is a safety-relevant
misnomer that would propagate to every backend.

Equally: on every surveyed robot **clearing** a latched stop is out-of-band or manual —
a second socket (UR 29999), a distinct API call (`automaticErrorRecovery`), or a human at
the pendant (ABB singularity). "Recover" must be allowed to answer *needs an operator*.

### 2.4 Pose representation

| | Position | Orientation | Redundancy / configuration |
|---|---|---|---|
| **This project** | **int millimetres**, X/Y/Z [READ] | **None on the wire** | **None on the wire** |
| **KUKA `E6POS`** | X/Y/Z mm | A/B/C Euler (Z-Y-X intrinsic) | **S/T status & turn bits** |
| **UR** | metres | **Rotation vector** `Rx,Ry,Rz` (axis-angle), radians [S2] | Implicit; `getInverseKinematics(x, qnear, …)` takes a **seed configuration** [S2] |
| **ROS 2 (JTC)** | — | — | **Joint angles are the representation.** Cartesian poses exist only above, as `geometry_msgs/Pose` = point + **quaternion** [S6][S8] |
| **Franka** | `CartesianPose` = `O_T_EE`, a **4×4 homogeneous transform**, column-major [S10] | (same matrix) | **`elbow`** — an explicit redundancy parameter [S10] |
| **ABB EGM** | mm | **Quaternion** `[w,x,y,z]` over UdpUc [S12]; Euler `rx,ry,rz` when signals are the data source [S11 §2.3] | RAPID `confdata` outside EGM |

**The honest lowest common denominator: position as three SI floats (metres) plus
orientation as a unit quaternion, plus an *opaque vendor redundancy token*, plus a named
frame and tool.**

Quaternion converts losslessly to and from every orientation form above — A/B/C Euler,
rotation vector, rotation matrix, quaternion. That part is genuinely common. The two parts
that are *not*:

- **Redundancy is not orientation and has no common form.** KUKA's S/T bits, Franka's
  `elbow`, and UR's `qnear` seed all answer the same question — *which of the several joint
  configurations that reach this pose?* — with three incompatible encodings. A 6-DOF pose
  simply does not determine a joint configuration. This must be an opaque per-vendor blob
  attached to the pose (or a seed joint vector, which is the closest thing to a portable
  form), and code that constructs a pose without it must be able to say so.
- **A pose is meaningless without its frame and tool.** UR poses are TCP-in-base; Franka is
  `O_T_EE` with `NE_T_EE` configured separately; ABB poses are defined in a *work object*;
  KUKA in `$BASE`/`$TOOL`. Tool and payload are configured through vendor-specific side
  channels (`setPayload` on UR [S2], `setLoad` on Franka).

Note also that **this project's integer-millimetre wire format is below every other robot's
resolution** and carries no orientation at all — audit T already found that "an `E6POS`
with unset orientation and unset S/T is not a well-formed motion" [READ]. Any generalised
pose type will be strictly richer than this wire format, which makes the KUKA driver a
*lossy* backend. That loss must be declared in a capability descriptor, not hidden.

### 2.5 Gripper

| | Channel |
|---|---|
| **This project** | **In-band**: `VACUUM_ON` / `VACUUM_OFF` opcodes, `aux_u16` carries vacuum percentage [READ] |
| **UR** | **Out-of-band, recommended.** Robotiq via "a port (63352) that is opened by the Robotiq_grippers UR Cap". The in-band alternative (`sendCustomScriptFunction`) is discouraged because "Simultaneous robot movements is not possible (since the `rtde_control` script is interrupted)" [S3]. Generic tooling uses `setToolDigitalOut(pin, value)` on the separate `RTDEIOInterface` [S2] |
| **ROS 2** | **Separate action on a separate controller.** `control_msgs/action/GripperCommand`: Goal `{position, max_effort}`; Result/Feedback `{position, effort, stalled, reached_goal}` [S6] |
| **Franka** | **Separate object and separate network connection.** `franka::Gripper` — "Establishes a connection with a gripper connected to a robot" — with its own `homing/move/grasp/stop/readOnce` [S10]. Not on the 1 kHz channel |
| **ABB** | RAPID I/O signals, or `RAPIDtoRobot`/`RAPIDfromRobot` fields inside the EGM protobuf [S11 §3.1.3] |

**The finding.** This project is the only surveyed system with the gripper in-band, and it
gets away with it *only because vacuum is a single bit*. The moment a gripper has width,
force, and a "gripped vs. closed on nothing" distinction — which is exactly what
`GripperCommand`'s `stalled` flag and Franka's `grasp(width, speed, force, epsilon_inner,
epsilon_outer)` exist to express — it needs its own command lifecycle with its own
completion. And note the UR case specifically: doing it in-band there *blocks concurrent
motion*, which forecloses approach-while-preshaping.

### 2.6 Workspace and reachability

| | Who checks |
|---|---|
| **This project** | **Client, against a hand-set square.** `workspace_bounds_mm: {x_min: -350, x_max: 350, y_min: -350, y_max: 350}` — and the config itself calls it a "PLACEHOLDER, and deliberately conservative" [READ, `configs/planning.yaml:66,93`], enforced by `Planner._reaches` [READ, `plan/planner.py:443`] |
| **UR** | **Controller answers a query.** `isPoseWithinSafetyLimits(pose)` and `isJointsWithinSafetyLimits(q)` verify the target "is reachable and complies with safety constraints"; `getInverseKinematics(x, qnear, …)` resolves it [S2] |
| **ROS 2** | **Split.** MoveIt checks: `ComputeIK` converts Cartesian poses to joint configurations and planning fails if it cannot [S8]. The JTC does **not** — it commands the joints it is handed and aborts only *after the fact* on `PATH_TOLERANCE_VIOLATED` / `GOAL_TOLERANCE_VIOLATED` [S6][S7] |
| **Franka** | **Runtime enforcement only.** Limit violations and reflexes throw `ControlException`; rate limiters clamp commanded rates of change. No pre-flight validity query |
| **ABB EGM** | **None, and the manual says so bluntly.** "The robot will react quickly to all position references sent to the controller, also faulty ones" [S11 §1.3]. Singularity is discovered by hitting it, and recovery is manual [S11 §1.1] |

**The finding.** Three regimes — *client guesses*, *controller answers*, *nobody checks
until it hurts* — and they are not reconcilable into a promise. An interface **cannot**
guarantee "unreachable targets are rejected before motion." It can only expose a
`validate(goal)` whose legitimate answers include **UNKNOWN**, plus a capability flag
saying whether validation is real.

### 2.7 What each requires that the others do not

| Robot | Requirement unique to it |
|---|---|
| **This project** | A CRC-16 on every frame, and positional framing with no resynchronisation [READ]. Integer-millimetre quantisation. |
| **UR** | **Liveness watchdog** — `setWatchdog(min_frequency)` [S2]. **Script re-upload as the recovery path** — `reuploadScript()`: "In the event of an error, this function can be used to resume operation" [S2]. Motion and data are **two different protocols on two different sockets**, and stop-clearing is a **third** (Dashboard 29999) [S1][S4]. `stopL`/`stopJ` only work if the move was launched with `asynchronous=true` [S3] — a blocking `moveL` is *not* interruptible through the same client. |
| **ROS 2** | **Named joints** — the goal must specify all controller joints unless `allow_partial_joints_goal` [S7]. **Header timestamps are semantically load-bearing** (`OLD_HEADER_TIMESTAMP` is a distinct failure). **Per-joint path *and* goal tolerances** as part of the goal, not driver config [S6]. Preemption semantics ("only one goal active"). |
| **Franka** | **Exclusivity** — "While the FCI is active, you have full, exclusive control of the Arm and Hand, which means you cannot use Desk or Apps simultaneously" [S9]. A **real-time kernel** and a 500 µs read→write window; "Packet loss can also produce `communication_constraints_violation`" [S9]. **Latching faults requiring explicit recovery.** |
| **ABB EGM** | **A RAPID program must be running** to host the session (`EGMSetupUC` + `EGMActPose` + `EGMRunPose`). **The controller initiates the UDP connection**, not the client [S11 §1.2]. Motion "has to start and to end in a fine point", and "the first movement that is performed after a controller restart cannot be an EGM movement" [S11 §1.1]. |

That last row is the one most likely to break a naive interface: on ABB, **the client
cannot initiate a session at all.** Any `connect()` that assumes the client dials out is
wrong for EGM.

---

## 3. Proposed common surface

Stated as requirements, not code. Each item is here because **at least three** of the
surveyed systems support it natively.

**R1 — The goal is a sequence of waypoints of length ≥ 1, not a single pose.**
This is the single largest correction the survey forces on a KUKA-shaped interface. ROS 2's
native unit *is* a trajectory [S6]; UR has `movePath` with per-waypoint blend radii [S2];
UR's own ROS 2 driver has a `FORWARD` mode for "trajectory interpolation on the robot"
[S5]. Designing around a single pose makes the ROS 2 backend a lie — you would emit N
one-point trajectories and lose blending, timing, and preemption. A single pose is the
degenerate one-waypoint case, not the primitive.

**R2 — A waypoint is `(target, motion_type, limits)` where `target` is a closed sum type:
Cartesian pose or joint vector.** Both are irreducibly necessary: ROS 2's JTC accepts only
joint targets [S7], while this project and UR's `moveL` are Cartesian-native. A driver
declares which it accepts; a Cartesian goal on a joint-only backend requires an IK step
that the driver must either provide or refuse — it must not silently invent one.

**R3 — Pose type = position (3 × float, metres) + unit quaternion + frame name + tool name
+ opaque redundancy token.** Per §2.4. The redundancy token may be empty, and a driver that
needs one and did not get one must fail loudly rather than pick a configuration.

**R4 — Motion completion is a typed terminal outcome, not a bool.** ROS 2's six error codes
plus `error_string` [S6] are the right shape and the right granularity: the caller needs to
distinguish *goal was invalid* from *goal was rejected by the controller* from *motion
started and deviated* from *motion finished out of tolerance* from *comms failed*. This
project's status-code enum is already close; the requirement is that the enum be part of the
interface, not per-vendor.

**R5 — State is readable on a channel separate from the command channel.** Every surveyed
system has this and in every one it is a *different* socket, topic, or struct: RTDE (30004)
versus URScript (30001–30003) [S1]; `/joint_states` versus the action [S7]; `EgmRobot`
feedback versus `EgmSensor` [S11]; `RobotState` passed into the callback [S10]. Minimum
content: joint positions, TCP pose, moving/idle, fault code. **This project is the outlier**
— its single blocking socket cannot be read while a motion is in flight, which is a
concrete deficiency the redesign must fix.

**R6 — `halt()`: decelerate and hold, in-band, idempotent, explicitly not safety-rated.**
Supported everywhere (`stopL`/`stopJ`, goal cancel, `Robot::stop()`, `EGMStop`). It must
*not* be called `estop`.

**R7 — `fault_state()` and `recover()`, where `recover()` may legitimately return
NEEDS_OPERATOR.** Per §2.3.

**R8 — Gripper as a separate capability object with its own lifecycle.** Operations:
`open()`, `close()`, `grasp(width, force)`, `halt()`, and a state including a
*holding-versus-closed-on-nothing* distinction. `GripperCommand`'s `stalled`/`reached_goal`
pair [S6] and Franka's `grasp(..., epsilon_inner, epsilon_outer)` [S10] are the same idea.
Vacuum degenerates to a bit, and that degeneration is fine — the reverse (generalising from
vacuum) is not.

**R9 — `validate(goal) -> REACHABLE | UNREACHABLE | UNKNOWN`.** UNKNOWN is a first-class
answer (Franka, EGM). This project's ±350 mm square becomes one implementation of
`validate`, correctly labelled as a client-side approximation.

**R10 — A capability descriptor, queried at connect.** It must at minimum answer: which
control shape(s) (§4); Cartesian and/or joint targets; whether `validate` is real; whether
the gripper shares the motion channel; available frames and tools; control rate if
streaming; whether the pose encoding is lossy (the KUKA backend must declare that it drops
orientation).

**R11 — `connect()` must permit a listen-mode driver.** ABB EGM's controller dials *out*
[S11 §1.2]. A connect API that assumes the client is the initiator excludes it.

---

## 4. One interface shape, or two?

**Two. And the split is not "Cartesian versus joint" — it is *who owns the interpolator*.**

- **Shape A — delegated (send-and-wait / send-and-poll).** The controller interpolates. The
  client supplies a goal (one pose, or a whole trajectory) and receives a terminal outcome.
  Timing is best-effort; a slow client is harmless. *Instances:* this project's EKI frame,
  `moveL`/`moveJ`/`movePath`, `FollowJointTrajectory`, UR's `FORWARD` trajectory-point mode.
- **Shape B — streaming (stream-and-monitor).** The client interpolates. It emits one
  setpoint per control period at a fixed rate; the controller has no notion of "arrived" and
  never reports completion; a late client is a **fault**. *Instances:* `servoL`/`servoJ`/
  `speedL`, EGM Position Guidance, FCI motion generators.

Three independent pieces of evidence that these cannot be one interface:

1. **B has no completion event, by construction.** Franka's "done" is the *client* marking
   `MotionFinished()` [S10]; ABB's is a tolerance held for a dwell (`egm_minmax` +
   `\CondTime`) evaluated on the controller [S11 §5.1.8] precisely because nothing else can
   evaluate it. In A, completion is an event the controller sends.
2. **B has a liveness contract A does not have.** FCI's 500 µs read→write window and
   `communication_constraints_violation` on packet loss [S9]; EGM's `\CommTimeout`, fatal
   under `\NoWait` [S11 §5.1.14]; UR's `setWatchdog(min_frequency)` [S2]. In A, silence
   means idle. In B, silence means fault. That is not a parameter — it inverts the meaning
   of the same observation.
3. **A can express things B cannot express at all.** "Move linearly to here at this speed."
   ABB states it outright: EGM cannot order a linear move, nor one with a specified speed or
   duration, "since EGM Position Guidance does not contain interpolator functionality"
   [S11 §1.1, §1.3]. Every A-shape command carries `speed`/`acceleration`/`time_from_start`;
   no B-shape command can.

And critically, **the split is not a per-vendor property.** UR's own ROS 2 client library
implements both side by side — `SERVOJ`/`POSE` streaming through the Reverse Interface *and*
`FORWARD` mode for on-robot trajectory interpolation [S5]. So shape is a per-session
capability, queried per robot, and the caller must be able to *request* one.

**The practical recommendation.** This project's application — pick and place with dwell —
needs only Shape A. Shape A should be *the* interface. Shape B should be a **separate,
optional interface** that a driver may or may not implement, and that a pick-and-place
application should not use. The specific mistake to avoid is generalising A's
`move(goal) -> outcome` into something that "also supports streaming" by adding a rate
parameter. It does not: adding a rate to A does not give you B's liveness contract, and
removing completion from A to accommodate B destroys the only thing A's callers need.

---

## 5. What cannot be abstracted

Listed in descending order of how badly a common interface would lie about it.

1. **Emergency stop.** No surveyed robot accepts a safety-rated stop over its application
   protocol. `OpCode.ESTOP = 0x06` [READ] must be renamed to something honest (`HALT`), and
   the interface must not offer an operation whose name implies a safety function.
2. **Interpolator ownership.** §4. Two shapes.
3. **Redundancy resolution.** KUKA S/T, Franka `elbow`, UR `qnear`. Opaque vendor token or
   seed joint vector; never inferred.
4. **Stop clearing and fault recovery.** Out-of-band on every surveyed robot, and sometimes
   requires a human. `recover()` must be able to answer NEEDS_OPERATOR.
5. **Reachability guarantees.** Three regimes (§2.6). Expose as a capability with an
   UNKNOWN result; never promise pre-flight rejection.
6. **Motion type and path shape.** LIN vs PTP vs joint-interpolated vs no-interpolator-at-all.
   A pose does not say what path is taken, and pick-and-place *does* care — a straight-line
   approach along the tool axis is a functional requirement, not a nicety. Carry a
   motion-type enum and accept that some backends cannot honour it (EGM cannot honour any of
   them).
7. **Frames, tool and payload.** Vendor-specific configuration side channels; a pose without
   a named frame and tool is not a pose.
8. **Session establishment and exclusivity.** FCI takes exclusive control of arm and hand
   [S9]; EGM needs a RAPID program running, must start and end in a fine point, and dials
   out [S11]; `ur_rtde` uploads a control script and may need `reuploadScript()` [S2].
   No common shape — vendor config plus a `ready` predicate.
9. **Units and resolution.** SI floats (m, rad) is the only honest interface choice. This
   project's integer-millimetre encoding is a lossy wire detail that must live *below* the
   interface, and the driver must declare the loss.
10. **Blending and preemption semantics.** UR blend radii, ROS 2's "one goal active, new
    goal aborts the old", EGM's ramp-in/ramp-out. Related but not identical; expose as
    capability flags rather than a common parameter.

---

## 6. Requirements this survey places on the existing protocol

Not a design — a checklist the redesign must satisfy, all grounded above.

| # | Requirement | Because |
|---|---|---|
| U-1 | The wire format must carry orientation, or the driver must declare itself position-only in its capability descriptor | §2.4; corroborates audit T's `[MISBEHAVES]` on unset A/B/C and S/T |
| U-2 | State must be readable while a motion is in flight | R5 — the single blocking socket cannot do this |
| U-3 | The goal type must be a waypoint sequence, not a single point | R1 — otherwise the ROS 2 backend is dishonest |
| U-4 | `ESTOP` must be renamed and demoted to a non-safety halt | §5.1 |
| U-5 | Vacuum control must move behind a gripper capability with its own completion | R8 — in-band works only for a one-bit end effector |
| U-6 | The ±350 mm square must be reframed as one implementation of `validate`, returning a client-side approximation, not as *the* reachability contract | §2.6, `configs/planning.yaml:66` already calls itself a placeholder |
| U-7 | Pose values must be SI floats at the interface; integer millimetres stays below it | §5.9 |
| U-8 | Completion must be a typed outcome shared across drivers | R4 |

---

## 7. Verdict

A single `RobotDriver` interface designed from the KUKA implementation alone would be wrong
in five specific ways that the survey identifies concretely: it would take a single pose
(ROS 2 takes trajectories), it would carry position without orientation or redundancy (every
other robot needs both), it would put the gripper in-band (nobody else does, and on UR it
blocks motion), it would promise completion status (Franka has none to give), and it would
call an application opcode an emergency stop (no vendor allows that).

A common surface *does* exist, and §3 states it: connect-with-capabilities, a waypoint-
sequence goal with a closed target type, a typed terminal outcome, a separate state channel,
a non-safety `halt`, a fault/recover pair that may demand an operator, a separate gripper
capability, and a `validate` that may answer UNKNOWN. That surface covers this project,
UR's `move*` family, ROS 2's `FollowJointTrajectory`, and UR's on-robot trajectory mode.

It does not cover FCI or EGM, and it should not pretend to. Those need a second interface
with a rate contract, a liveness fault, and no completion event — which this project does
not need and should not build until something requires it.
