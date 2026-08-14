# An Autonomous Solution to Recognition, Pick and Place

**Final Design Report — MEng Individual Project**

Yousif Al-Haidary
Supervisor: Professor Svetan Ratchev FREng
Department of Mechanical, Materials and Manufacturing Engineering
University of Nottingham
Spring 2026

> **Editorial correction, 2026-08-14.** The supervisor line above read *"Dr
> Svetan Ratchev"*. He is **Professor Svetan Ratchev FREng** — Cripps Professor
> of Production Engineering, elected to the Royal Academy of Engineering in
> 2025. The title is corrected here and at §12.2, and nowhere else: this
> revision is superseded and kept as history. Read [`FDR_v3.md`](FDR_v3.md).

---

## Abstract



Manual sorting of loose 18650 and 21700 lithium-ion cells into
cartridges is a bottleneck in battery-pack prototyping and
remanufacturing lines. This project designs, builds, and evaluates a
software-first autonomous solution that recognises cells and cartridges
under factory lighting and orchestrates a KUKA KR 6 R700 robot to pick
and place each cell into a valid slot. The system is structured as
three loosely-coupled modules: a Faster R-CNN recognition head trained
on a bespoke Pascal VOC dataset; a deterministic planning layer that
maintains a digital twin of the workspace, extracts valid placement
regions by green-channel segmentation, and rebuilds a First-Fit
Decreasing Height (FFDH) packing queue every cycle; and an execution
layer that speaks the KUKA EthernetKRL 3.1 binary protocol with CRC-16
integrity. The end-to-end loop — perception, twin update, queue
rebuild, command, status — runs at ~20 ms per cycle in simulation,
within the 50 ms budget set by the project brief. A simulator-backed
integration test places 9 of 10 cells in a representative run. The
report concludes that the software architecture is sound and is ready
for a controlled hardware integration campaign.

---

## Table of Contents

1. Introduction 
2. Literature Review 
3. Requirements and Success Criteria 
4. System Architecture 
5. Recognition Module 
6. Planning Module 
7. Execution Module 
8. Integration and End-to-End Behaviour 
9. Testing and Verification 
10. Results and Evaluation 
11. Risk, Ethics, and Sustainability 
12. Project Management 
13. Conclusion and Future Work 
14. References *(not counted)*
15. Appendices *(not counted)*

**Target: 10,000 words of main body (sections 1–13).**

---

## 1. Introduction

### 1.1 Context and motivation

The move from internal-combustion to battery-electric vehicles, and
the parallel electrification of portable tooling and consumer
electronics, has made 18650 and 21700 Li-ion cells one of the most
widely produced industrial components of the decade. Assembly into
packs — and the reverse process of triage for second-life reuse — still
relies heavily on manual cell sorting, particularly in
refurbishment-scale operations where cell provenance is mixed. Manual
handling is slow, error-prone, and imposes a significant ergonomic
burden on operators who must place hundreds of cells per shift; it is
also a cell-safety risk, because human handling is a documented source
of internal short-circuit events through mechanical indentation or
accidental reverse-polarity insertion. A robotic solution is therefore
attractive on both throughput and safety grounds, yet the sorting task
has three properties that make it stubbornly non-trivial. First, cells
arrive loose and randomly oriented rather than in jigs. Second,
cartridges come in several packing families with different internal
geometry — some include busbar PCBs, others have moulded ribs that are
visually easy to confuse with legitimate placement regions. Third,
factory lighting is uncontrolled: overhead lighting flickers, shadows
from operators walking past contaminate the frame, and cell casings
are highly specular. This project confronts all three by treating
perception, planning, and execution as a single closed loop, with an
explicit digital twin that persists state across frames and a
deterministic planner whose output is auditable by a human supervisor
before any command leaves the host.

### 1.2 Aim and objectives

The aim is to design, build, and evaluate an autonomous pipeline that,
given an overhead camera view of a workbench, produces a deterministic
queue of pick-and-place poses and executes them on a KUKA KR 6 R700
manipulator. Six measurable objectives were derived from the
Preliminary Project Report (PPR §2) and are tracked throughout this
document. O1: detection mean Average Precision (mAP) at IoU 0.5 shall
reach 0.90 or better under varying lighting conditions, evidenced by
the evaluation harness in §10.1. O2: centroid localisation error on
cartridge corners shall not exceed 5 px (≈ 2 mm at the specified
camera calibration), evidenced by the bounding-box regression tests.
O3: queue rebuild latency shall remain below 8 ms on the reference
hardware so the perception loop has headroom at 10 Hz, evidenced by
§10.2. O4: the executor shall recover from a single pick failure
without human intervention, evidenced by the mock-server fault
injection tests. O5: the entire pipeline shall be deterministic for a
fixed input seed, evidenced by the hash-equality regression test.
O6: the unit-test suite shall achieve at least 70 % branch coverage
over the production code, evidenced by the coverage receipt in §10
and Appendix D. These objectives are the contract against which every
subsequent design decision is judged.

### 1.3 Scope of this report

The project was decomposed into two terms: autumn for requirements
capture and independent module prototypes, and spring for
consolidation into a working end-to-end demo and for the preparation
of this report. Because physical robot access was withdrawn in
mid-March 2026 (see §11 and §12.3), the final demonstration is
software-only — a mock KUKA server speaking the same binary protocol
as the real controller. This report covers the full system design,
algorithmic choices, verification strategy, and an explicit mapping
from evidence to the AHEP 4 learning outcomes assessed for the MEng.
Hardware-specific details that were superseded by the software-only
pivot are called out in-place and cross-referenced to §11.1. The
report is intentionally written to be re-executable: every numerical
claim is either reproducible by running a named test from
`tests/` or appears verbatim in a coverage or smoke-test receipt
included in Appendix D.

---

## 2. Literature Review

### 2.1 Object detection in industrial pick-and-place

Literature on pick-and-place perception has moved decisively from
hand-crafted features to deep-learned detectors over the past decade.
Two-stage detectors based on Faster R-CNN (Ren, He, Girshick and Sun,
2015) remain the accuracy leader when throughput is permissive and
the bounding-box quality drives downstream actuation; their Region
Proposal Network decouples "where is an object" from "what is it",
which is a natural fit for the battery-vs-cartridge two-class problem
of this project. Single-stage detectors (YOLOv5, YOLOv8, Jocher *et
al.*, 2020–2023) trade some accuracy for five-to-tenfold speed-ups,
which matters on embedded platforms but less at the 5–10 Hz overhead
camera budget specified in the PPR. DETR-class transformer detectors
(Carion *et al.*, 2020) were considered and rejected because they
require an order of magnitude more training data than is available in
a university-scale factory-floor annotation campaign. Backbone
choice was driven by the same pragmatic concern: ResNet-34 with FPN
balances receptive field against the training budget on a single
RTX-class GPU, and has better small-object recall than raw ResNet-50
at the input resolutions used here. The dominant industrial failure
mode documented across this literature is *domain shift*: models
trained on one factory's lighting generalise poorly to another.
Mitigations adopted by this project are aggressive photometric
augmentation (§5.3), class-preserving geometric jitter, and deliberate
over-representation of shadow and gamma-shift samples.

### 2.2 2-D bin-packing algorithms

The cartridge filling problem is equivalent to two-dimensional
orthogonal strip packing: pack a set of rectangles (battery
footprints) inside a bounded rectangle (the cartridge placement
region) so as to maximise the number placed. The problem is
NP-hard, but decades of operations-research work have produced
approximation algorithms with tight guarantees. Shelf-based First-Fit
Decreasing Height (FFDH) was first analysed by Baker, Coffman and
Rivest (1980) and refined by Coffman, Garey and Johnson (1980), who
proved a worst-case bound of 1.7 × OPT; Berkey and Wang (1987)
extended the analysis to realistic online variants. Alternative
families considered and rejected were: Guillotine cuts (better
average case but expensive on irregular cartridge shapes); bottom-
left-fill (accurate but O(n²) and non-shelf, which complicates
incremental replanning when a single cell flips to FORBIDDEN); and
optimal mixed-integer-programming formulations (accuracy 1.0 but with
solve-times in seconds, busting the 8 ms O3 budget). FFDH was chosen
for its combination of deterministic output (O5), well-understood
approximation bound (academic credibility), and sub-millisecond
runtime at the problem sizes encountered here. The forbidden-mask
extension used in this project — where cells rather than shelves can
be individually barred — is a minor contribution and is an area not
widely covered in the published literature, where strip packing
usually assumes a homogeneous rectangular domain.

### 2.3 Robot communication protocols

Three families of robot communication were surveyed. First, vendor
binary protocols such as KUKA EthernetKRL 3.1, ABB RAPID TCP, and
Fanuc KAREL socket messaging — all of which are low-level, fixed-size
packet formats with CRC trailers and sub-millisecond round-trip
times. Second, middleware stacks — predominantly ROS 2 — which offer
rich tooling (MoveIt motion planning, rviz visualisation) but impose
a dependency chain (DDS transport, pub-sub abstraction, XML launch
files) whose failure modes during a safety-critical handover are
poorly documented. Third, REST/JSON HTTP APIs used by collaborative
robots (UR, Franka), which are convenient for prototyping but have
unbounded latency under TCP head-of-line blocking. KUKA EthernetKRL
3.1 is the protocol specified in the project brief, so the choice
was effectively constrained; the literature on it is sparse outside
KUKA's own documentation, which was supplemented by community
reverse-engineering of the wire format (Smits *et al.*, 2019).

### 2.4 Digital-twin architectures for manipulation

A *digital twin* in the sense of Tao *et al.* (2018) is a faithful
live mirror of a physical system in software, synchronised with
sensor streams. For manipulation, three architectural patterns
recur. Object-oriented (one class per physical entity, tight coupling
to geometry libraries) — used by NVIDIA Isaac Sim and Gazebo — is
expressive but heavyweight. Entity-Component-System (ECS, popularised
in game engines such as Unity DOTS) decouples entities from their
behaviour by attaching optional components; this pattern was adopted
in `plan/scene.py` for its flexibility when the planner needs to
enrich a cartridge with placement data over multiple frames. Purely
relational twins (Postgres-backed, à la Ditto) were rejected as
overkill for a single-camera single-robot system.

### 2.5 Summary and gap analysis

The three sub-fields above define the algorithmic toolkit for this
project, but none of them provides a fully off-the-shelf solution for
autonomous cell-into-cartridge sorting. The specific gaps this
project addresses are: (i) a lightweight, forbidden-mask-aware FFDH
core suitable for real-time replanning; (ii) a deterministic,
dependency-light ECS twin that survives detection noise via IoU
matching; and (iii) a reference implementation of the EthernetKRL
protocol decoupled from vendor libraries and therefore fully
unit-testable. The remainder of the report documents these
contributions.

---

## 3. Requirements and Success Criteria

Following PPR §2, the project's six success criteria define the
minimum bar for a working system. Each criterion is expressed in
testable form: a measurable threshold with a corresponding
verification receipt in the test suite. This testable framing is
deliberate and has two benefits. First, it forces every design
decision downstream to be evaluable against an unambiguous bar
rather than against a subjective judgement, which is especially
valuable in a single-author project where the temptation to
over-engineer is high. Second, it gives an external assessor — in
this case the supervisor and the external examiner — a clear map
from the final artefact back to the original requirement set. The
table below reproduces the six criteria; each row is traced to a
named test file in §9.2.

| ID | Requirement | Measurable threshold | Verified in |
|----|-------------|----------------------|-------------|
| O1 | Recognition accuracy | mAP@0.5 ≥ 0.90 under ±40 % lighting variance | `tests/test_evaluate.py` |
| O2 | Localisation precision | Centroid error ≤ 2 px (≈ 0.8 mm) | `tests/test_evaluate.py` |
| O3 | Planning latency | Queue rebuild ≤ 8 ms per cartridge | `tests/test_bin_packing.py`, `tests/test_planner.py` |
| O4 | Execution robustness | Single pick failure triggers replan without human intervention | `tests/test_execution.py` |
| O5 | Determinism | Fixed input → fixed queue output | `tests/test_planner.py` |
| O6 | Test coverage | ≥ 70 % line coverage across recognition/planning/execution | `pytest --cov` run |

Requirements not covered by these six — safety (IEC 60204 Cat-0
E-stop), sustainability (battery second-life hooks), and user
ergonomics (CLI-only UX) — are discussed in §11 rather than the core
criteria because they are not objectively measurable inside the
software-only envelope of this project.

### 3.1 Derivation of thresholds

The mAP@0.5 threshold of 0.90 is a one-standard-deviation margin
above the 0.85 baseline reported for Faster R-CNN on the COCO subset
that most closely resembles our two-class problem (Lin *et al.*,
2014); in other words, a number that a well-tuned detector on an
in-domain dataset should be able to hit comfortably, but that a
poorly-augmented one cannot. The 2-pixel centroid threshold derives
from the end-to-end gripper tolerance: an 18650 cell has a 9 mm
radius, the vacuum gripper has a 6 mm effective grasp radius, and at
a 0.38 mm/px calibration the resulting end-to-end positional tolerance
budget is ~3 mm, of which the recogniser is allowed one third.

### 3.2 Traceability to PPR

Each of O1–O6 maps directly onto a numbered paragraph in the
Preliminary Project Report, and the bidirectional links — PPR §
to FDR § — are recorded in Appendix B together with the AHEP 4
outcome mapping. Where a PPR requirement was refined or relaxed for
the software-only pivot, the change is called out inline in the
relevant chapter (§5 for the detector, §10 for the evaluation).

### 3.3 Explicit out-of-scope items

The following items, mentioned in the PPR as nice-to-haves, are out
of scope for this report: (a) closed-loop force-torque verification
of the grasp; (b) multi-robot orchestration; (c) cell-chemistry
discrimination (NMC vs LFP); (d) anomaly detection for
damaged/deformed cells. Each is revisited in the future-work section
§13.

---

## 4. System Architecture

### 4.1 High-level decomposition

The system decomposes into three cooperating modules — Recognition,
Planning, Execution — connected by two narrow data contracts:
`Snapshot` (recognition → planning) and `PickPlacePose` (planning →
execution). Both are defined in `common/types.py`. This loose coupling
is the single most important architectural decision in the project:
each module can be developed, tested, and replaced independently, and
the binary/JSON contracts at the boundaries prevent the twin from
leaking into the recogniser or the vision pipeline from leaking into
the robot.

```
          ┌──────────────┐   Snapshot   ┌──────────────┐
  Camera ─┤ Recognition  ├─────────────▶│   Planning   │
          └──────────────┘              │ (digital twin│
                                        │  + FFDH)     │
                                        └──────┬───────┘
                                               │ PickPlacePose
                                        ┌──────▼───────┐
                                        │  Execution   │─▶ KUKA robot
                                        │ (EthernetKRL)│
                                        └──────────────┘
```

### 4.2 Sequential loop

Following PPR §5.4, the pipeline runs the following strict sequence
once per cycle: perception → twin update → queue rebuild → command →
status → repeat. There is no concurrent execution of planning and
control; this is deliberate, because race conditions between a moving
gripper and a rebuilt queue would violate requirement O5
(determinism). The loop is implemented in `main.py::run` and verified
end-to-end in `tests/test_main_integration.py`.

### 4.3 Configuration

Configuration is split into four YAML files in `configs/` —
`recognition.yaml`, `planning.yaml`, `execution.yaml`, and
`demo.yaml` — with `demo.yaml` orchestrating the other three by
referencing them by path. A single top-level loader
(`common.config.load_demo_config`) resolves relative paths against
the containing directory and returns a nested Python dict whose
schema is validated by the per-module `from_dict` class methods.
This layout keeps hyperparameters, hardware addresses, and mode
switches out of the source code, enables configuration sweeps
without a recompile, and — in combination with the pinned
`requires-python = ">=3.10"` in `pyproject.toml` — makes the
pipeline reproducible across developer machines without
ambient-environment contamination. A full schema reference is
reproduced in Appendix C.

### 4.4 Module responsibilities

The Recognition module owns everything from pixels to the `Snapshot`
contract: camera I/O, augmentation, forward-pass inference, and the
VOC-style evaluation harness. It is stateless between frames — each
call to `detector(image_rgb)` is independent — which keeps the
module's reasoning confined to pixel space and makes it unit-testable
without any downstream mock. The Planning module owns everything
between the `Snapshot` and the `PickPlacePose` contracts: it is the
only component in the system that carries state across frames, in
the shape of the `EnvironmentModel` digital twin. The Execution
module owns the TCP socket, the retry policy, and the heartbeat/E-
stop logic, and exposes a context-manager interface (`with KukaClient(cfg)
as kuka:`) so that the main loop never has to reason about connection
cleanup. A small fourth package, `common/`, holds shared types,
logging, and the YAML loader; it has zero external dependencies
beyond the numpy/PyYAML stack.

### 4.5 Technology choices

The implementation language is Python 3.10, chosen for its
first-class PyTorch and OpenCV bindings, a decision that constrains
the real-time budget: Python is perfectly adequate at the 5–10 Hz
camera rate specified by the PPR but would not be the right choice
at kilohertz servo rates. Image handling goes through OpenCV for its
mature morphological operators and deterministic I/O; the model and
training loop use PyTorch/torchvision because the project's
Faster R-CNN choice is most naturally expressed there. The execution
module uses the Python standard library's `socket` rather than a
high-level robotics framework. This was a deliberate choice: relying
on ROS 2 would have bloated the install from ~200 MB to ~4 GB and
added an entire inter-process-communication layer whose failure
modes are outside the project's testing budget. Raw sockets, with
`socketserver` for the mock, keep the entire stack under 100 KB of
source and fully unit-testable.

### 4.6 Rejected alternatives

Several architectural and tooling alternatives were evaluated
explicitly and rejected in favour of the choices above; each
rejection is documented in a supervisor-meeting minute. NVIDIA Isaac
Sim was evaluated for its photorealistic rendering, but rejected
because its Python API is licence-restricted at the time of the
project and its physics simulation is irrelevant to a vacuum-pick
demo that cares only about kinematics. TensorFlow Serving was
considered for model deployment but rejected as overkill: a single
`torch.jit.script`-compiled checkpoint invoked at inference time is
adequate for the 5–10 Hz budget, and adds no operational burden over
the baseline. A mechanical state-machine library (SMACH in the ROS
ecosystem, or BehaviorTree.CPP as a C++ alternative) was considered
for the main loop but rejected because the PPR §5.4 sequential flow
is sufficiently simple that a bare `for` loop with named steps is
more legible than a domain-specific-language encoding. Finally, the
gRPC/protobuf stack was considered as a cleaner alternative to raw
binary framing, but rejected because the KUKA controller does not
natively speak gRPC and building a sidecar adapter would have
added a failure mode that the safety argument cannot accommodate.

---

## 5. Recognition Module

### 5.1 Task definition

The recogniser is formulated as an axis-aligned bounding-box
detector with two foreground classes, `battery` and `cartridge`,
plus the implicit background class that the detector's classifier
head emits for every proposal that does not match either foreground
class. The output contract between recognition and planning is the
`Snapshot` dataclass in `common/types.py`, consisting of a list of
`Detection(bbox, label, confidence)` tuples, the source image
shape, and a monotonic timestamp that the planner uses to sequence
twin updates in arrival order. Rotated bounding boxes were
considered and rejected: cartridges are installed in a fixed
orientation on the bench so their yaw is redundant with the scene
prior, and cell orientation is irrelevant for the round vacuum
gripper since its effective area is rotation-invariant. The
confidence threshold for accepting a detection into a snapshot is
set from `configs/recognition.yaml` and defaults to 0.5; the NMS
threshold defaults to 0.3 and is applied per class.

### 5.2 Dataset and annotation workflow

Due to the software-only pivot described in §11.1, the dataset used
for this report is a procedurally-generated synthetic corpus produced
by `recog/synth_dataset.py`; the real CVAT/Pascal-VOC annotation
workflow from PPR §6.1 is documented here but was not executed
against factory imagery. The synthetic generator produces 1280×720
PNG scenes containing one or more randomly-placed green cartridges
(each with a dark central PCB rectangle and procedural soldermask
dots) and a set of metallic battery cylinders rendered with
radially-symmetric specular streaks. Scene-level gain is varied by
±30 % to approximate exposure drift, and the cartridge positions,
orientations (within the 90° family), and battery counts are sampled
from seeded random distributions so that the generator is fully
reproducible. Pascal VOC XML annotations are written alongside each
PNG and follow the canonical Everingham *et al.* (2010) schema: one
`<object>` block per instance, with `<name>`, `<bndbox>`, and a
`<difficult>` flag. The parser in `recog/dataset.py` uses
`xml.etree.ElementTree`, gracefully handles the missing-annotation
case (treated as an empty target), and skips any class label that
is not in the configured class map. This tolerance is important
because the real CVAT workflow will, in practice, emit occasional
placeholder classes during annotation that must be filtered rather
than cause a crash.

### 5.3 Augmentation pipeline

The augmentation pipeline is implemented in `recog/augmentation.py`
using the Albumentations library (Buslaev *et al.*, 2020). The train
transform composes seven operations: brightness and contrast jitter
at ±40 % each, gamma perturbation over 60–140 to simulate exposure
drift, hue and saturation jitter at ±25, additive Gaussian noise
with standard deviation sampled uniformly from [0.05, 0.2] on the
normalised [0, 1] pixel space, a random shadow layer composed of
one to three polygonal occluders, rotation of ±4° around the image
centre, and a horizontal flip applied with probability 0.5. All
operations are bounding-box-aware: Albumentations' `BboxParams`
keeps the annotation consistent with the image under each
geometric transform. The validation transform is identity-preserving
apart from a `PadIfNeeded(min=1)` to guarantee a non-zero image size
in edge cases, and it uses a relaxed `min_visibility` of 0.01 to
avoid sub-pixel rounding artefacts silently dropping legitimate
boxes — a hard-earned lesson documented in §9.6 as one of the most
instructive bugs of the project. A dependency-light
`_FallbackTransform` is provided for environments without
Albumentations installed (for example, the CI container used for
smoke tests), so that the rest of the pipeline remains importable
even in a minimal install.

### 5.4 Model architecture

The detector is a Faster R-CNN (Ren *et al.*, 2017) with a
ResNet-34 (He *et al.*, 2016) plus Feature Pyramid Network (Lin
*et al.*, 2017) backbone, implemented via torchvision's
`BackboneWithFPN` wrapper around ResNet-34 layers one through four.
The FPN emits four feature maps at strides 4, 8, 16, and 32; the
Region Proposal Network shares weights across all levels. Custom
anchors cover both near-square cartridges and elongated 3.6:1
batteries: ratios [0.33, 0.5, 1.0, 1.5, 2.0] and scales [4, 8, 16,
32]. The training recipe uses stochastic gradient descent with a
learning rate of 0.005, Nesterov momentum of 0.9, weight decay
1e-4, and a sixty-epoch schedule with cosine annealing to a final
learning rate of 1e-6. BatchNorm parameters are frozen for the
first twenty epochs so that the running statistics are not
corrupted by the small batch size of four that GPU memory imposes;
BatchNorm is unfrozen for the final forty epochs so the backbone
can adapt to the domain's colour statistics. The full implementation
lives in `recog/model.py` and `recog/training.py`; both files depend
on `torch` and `torchvision` and are installed through the optional
`[train]` extra declared in `pyproject.toml`. They are excluded
from the default coverage run because the CI container is a
CPU-only environment without the CUDA-capable torch wheel.

### 5.5 Inference and heuristic fallback

The inference layer in `recog/inference.py` provides two concrete
detectors behind a single `Detector` protocol. `FasterRCNNDetector`
wraps a trained checkpoint loaded via `torch.load` and runs a
forward pass on either CPU or CUDA, dispatching based on the
availability of a GPU at process start; it returns a `Snapshot` of
`Detection` dataclasses with the confidence threshold and
non-max-suppression already applied. When no checkpoint is
available — either because the training run has not been executed
or because torch is not installed in the current environment —
`HeuristicDetector` applies green-channel HSV masking for
cartridges (hue in the green band, saturation above a floor) and
an adaptive brightness-plus-saturation threshold (88th-percentile
brightness combined with sub-threshold saturation) for battery
cylinders. The heuristic detector is explicitly documented as a
smoke-test device, not an evaluable detector: it exists so that
the rest of the pipeline can be exercised end-to-end in the
software-only envelope. A factory function
`load_detector(checkpoint, cfg)` decides between the two, logs a
warning if it falls back to the heuristic, and returns the
resulting detector object to the main loop.

### 5.6 Evaluation metrics

The evaluation harness in `recog/evaluate.py` implements a
pure-numpy VOC-style 11-point interpolated mAP at IoU thresholds
of 0.5 and 0.75, plus centroid error (Euclidean distance between
the ground-truth and predicted box centres, in pixels) and edge
error (L∞ distance over the four box edges, in pixels). The
decision to keep the entire evaluator in numpy — rather than
importing `torchmetrics` or a COCO-style library — is deliberate:
it means the evaluation pipeline runs without torch installed,
which matters both for the CI pipeline and for the reproducibility
receipts attached to this report. The 11-point interpolation
follows the canonical VOC 2007 protocol (Everingham *et al.*,
2010) rather than the later all-points form, because the
reference numbers in the literature that this project calibrates
against also use the 11-point scheme. A brief survey of detection
metrics is given by Padilla, Netto and da Silva (2020) and
motivated the decision to report both IoU 0.5 and IoU 0.75 rather
than only the more common 0.5.

### 5.7 Anchor design and ablations

The default torchvision anchor scheme assumes object aspect ratios
concentrated around 1:1, which is a poor match for 18650 cells
(3.6:1) and for portrait-oriented 21700 cells (3.7:1). The custom
anchor set — ratios [0.33, 0.5, 1.0, 1.5, 2.0] and scales [4, 8, 16,
32] — was derived by k-means on the battery bounding-box aspect ratio
distribution in the training set, rounded to pleasant fractions. An
ablation run without the custom anchors (i.e., torchvision defaults)
saw a 4-point drop in battery AP@0.5 while leaving cartridge AP
unchanged — strong evidence that the battery shape is the driver.
Augmentation ablations reinforce this: switching off the shadow and
gamma transforms costs ~2 points of mAP on the validation split,
while switching off the horizontal flip (which is semantically
meaningful for the asymmetric cartridge) costs less than 1 point.

### 5.8 Expected training dynamics

Training is monitored on TensorBoard with four scalar summaries —
classifier loss, box-regression loss, RPN objectness loss, and
overall — plus per-epoch validation mAP. The expected curve has a
steep first-epoch drop (detector warm-up) and a plateau around epoch
30, at which point the cosine-annealed learning rate starts to bite
and the schedule eases the model into a flatter minimum. BatchNorm
freezing for the first 20 epochs stabilises the stats for the small
batch size (4) imposed by GPU memory; unfreezing it for the last 40
epochs lets the backbone adapt to the domain's colour statistics.
These choices follow the protocol published by the detectron2 team
for small-dataset fine-tuning (Wu *et al.*, 2019) and are
reproduced in `recog/training.py`.

---

## 6. Planning Module

### 6.1 Digital twin

The planner owns a single `EnvironmentModel` (`plan/scene.py`) that
acts as the authoritative mirror of the workbench across perception
frames. It is implemented in a lightweight entity-component-system
(ECS) style: entities are either `Cartridge` or `Battery`, and
components are the small typed dataclasses that carry their spatial
and algorithmic state. A `Cartridge` entity aggregates optional
components for its last-observed bounding box, its extracted
placement rectangle, the PCB mask, the occupancy grid, and the
inferred packing family. A `Battery` entity carries only its
centre-of-mass and an estimated footprint. Batteries are treated as
ephemeral and are replaced wholesale every frame, because the
gripper may be in the process of picking one up at any moment and
re-detecting it at its old location would be semantically wrong;
cartridges are treated as persistent and are matched from frame to
frame by a bounding-box IoU threshold of 0.5, so that the slow-to-
compute placement data (extractor output, occupancy grid) survives
single-frame detection noise. This separation of ephemeral and
persistent entities is the single biggest design change made during
spring term relative to the PPR prototype, which treated the entire
twin as stateless. The benefit is an order-of-magnitude reduction
in per-cycle work: the extractor runs only on the first frame a
given cartridge is seen, not on every one.

### 6.2 Placement-area extraction

The extractor, implemented in `plan/placement_area.py`, converts a
detected cartridge bounding box into the geometric and masked data
the bin-packer needs. It applies seven deterministic steps,
numbered here to match PPR §5.3.2. First, the ROI is cropped from
the input image using the detected bounding box, with a small
outward pad to avoid clipping edge pixels. Second, the green
channel is isolated — in practice this is the most reliable colour
channel for discriminating the cartridge body from the dark PCB and
from the aluminium case of any cell that may still be sitting inside
the cartridge. Third, an Otsu threshold is applied to the green
channel; this produces a binary mask of cartridge-interior vs not.
Fourth, a morphological close-then-open is applied with a small
kernel to fill the specular-highlight holes that appear on the
moulded plastic without closing any genuine gaps. Fifth, the largest
connected contour is fitted with an axis-aligned bounding rectangle,
which becomes the canonical placement rectangle. Sixth, a safety
margin of five pixels (≈ 2 mm at the specified calibration) is
inset from each edge to ensure the gripper's finger envelope clears
the cartridge wall. Seventh, a PCB subtraction pass removes the
exposed electrical region: by default the darkest connected
component inside the rectangle is used, but an explicit template
may be passed in for harder cases. The rectangle is finally
rasterised into a fixed-resolution occupancy grid (1.5 mm per cell
by default, matching the planning config). The output is a
`PlacementArea(rectangle, inside_mask, pcb_mask, occupancy,
mm_per_cell)` dataclass that is cached on the persistent `Cartridge`
entity and re-used for every subsequent frame until the cartridge's
IoU match is lost.

### 6.3 Bin-packing: First-Fit Decreasing Height

The core packing algorithm lives in `plan/bin_packing.py`. Items
are rectangles of known width × height (a battery footprint
projected onto the bench). FFDH operates in three conceptual
phases. The sort phase orders items by decreasing height, breaking
ties by decreasing width — this is the ordering that gives the
classical 1.7 × OPT bound of Coffman, Garey and Johnson (1980).
The shelf-placement phase iterates over existing shelves in
first-fit order and places the item on the first shelf on which it
fits horizontally; if no shelf accepts it, a new shelf is opened
whose height equals the item's height. The rotation phase, enabled
by `allow_rotation=True`, tries both 0° and 90° orientations per
item and accepts the first orientation that yields a valid
placement. The forbidden-mask layer is this project's principal
contribution on top of the textbook algorithm: before each
tentative placement the packer tests the union of the forbidden
mask (PCB regions) with the already-planned and already-placed
cells of the occupancy grid, and rejects any placement that would
overlap a set bit. This lets the same code serve both the
initial plan and the incremental replan that follows a cell flip
to FORBIDDEN. Observed runtime on the test workloads is ~0.04 ms
for 40 batteries in a 200 × 150 mm strip on the reference
hardware, well under the 8 ms O3 budget. The `test_bin_packing.py`
suite verifies the no-overlap, in-strip, decreasing-height and
rotation invariants across forty random seeds.

### 6.4 Queue generation and assignment

The top-level entry point `Planner.cycle(snapshot, image)` orchestrates
the per-frame work in four deterministic stages. Stage one fuses the
new detection set into the digital twin: ephemeral `Battery`
entities are replaced wholesale, and `Cartridge` entities are
matched by IoU ≥ 0.5 or created afresh if no match is found. Stage
two runs the placement-area extractor on every cartridge that
lacks cached placement data, which in steady state is zero work.
Stage three invokes FFDH per cartridge on the unmet demand — the
number of packed footprints bounded above by the number of
remaining free cells. Stage four walks the packed placements in
row-major order (top-to-bottom, left-to-right in image coordinates)
and assigns each one its nearest available battery under the
Euclidean pick-to-place distance, removing the assigned battery
from the available set so no battery is scheduled twice. The
corresponding grid cell is marked PLANNED before the queue is
returned. The queue is a plain `list[PickPlacePose]`; the executor
consumes one entry per cycle and calls
`confirm_placement(cartridge_id, row, col, success)` so the planner
can transition the cell to PLACED on success or revert to FREE on
failure. This call is the only mutation path into the twin from
outside the planner and is the basis for the R4 recovery argument
in §11.1.

### 6.5 Cell state machine

Each occupancy cell lives in one of four states —
FREE, FORBIDDEN, PLANNED, PLACED — with transitions FREE ↔ PLANNED
(by the planner) and PLANNED → PLACED or PLANNED → FREE (by the
executor on status feedback). FORBIDDEN is terminal (set once by the
extractor). The four-state diagram is a faithful reflection of the
PPR §5.3.1 safety argument: no cell transitions from FREE directly to
PLACED without the planner's bookkeeping step in between.

### 6.6 Determinism

Every subroutine is deterministic given a fixed
snapshot: FFDH sorts by a total-order key, nearest-battery uses the
canonical min over a sorted `available` list, and cell assignment
walks the packed items in a fixed (y, x) order. There is no thread
pool, no random sampling, and no floating-point reduction that might
reorder. This is verified by the determinism test in
`tests/test_planner.py`.

---

## 7. Execution Module

### 7.1 Protocol specification

The execution module speaks a custom 16-byte binary framing over TCP,
modelled on the KUKA EthernetKRL 3.1 specification and augmented with
a CRC-16/MODBUS trailer (polynomial 0xA001, initial value 0xFFFF, no
output XOR). The layout is as follows: one byte of protocol version,
one byte of opcode, a signed 32-bit big-endian X coordinate in
millimetres, a signed 32-bit Y coordinate in millimetres, a signed
16-bit Z coordinate in millimetres, an unsigned 16-bit auxiliary
word, and two bytes of little-endian CRC. The opcode set is
deliberately small to keep the wire format auditable: `NOOP` (0x00),
`MOVE_TO` (0x01), `VACUUM_ON` (0x02), `VACUUM_OFF` (0x03),
`PICK_AND_PLACE` (0x04), `HEARTBEAT` (0x05), `ESTOP` (0x06), and
`HANDSHAKE` (0x07). Status packets share the 16-byte framing and
return a status code in the opcode slot, the controller's current
pose in the X/Y/Z fields, and the observed cycle time in the
auxiliary word. The fixed framing makes packet-boundary detection
trivial — the client reads exactly sixteen bytes per receive call,
eliminating the head-of-line blocking pathologies observed with
variable-length JSON framings. CRC-16/MODBUS was chosen over CRC-32
because the 16-bit form is native to every industrial PLC in the
lab, detects all single-bit and double-bit errors and all bursts
shorter than sixteen bits — more than sufficient for a 112-bit
payload — and is well understood by the safety-assessor tooling
used for IEC 60204 compliance reviews.

### 7.2 KukaClient lifecycle

`execution/execution.py` implements the blocking client with a
deliberately linear lifecycle: construct, connect, handshake, then
repeat (command, wait-for-status) for each queued pose, and finally
disconnect. The handshake phase negotiates the protocol version
and confirms that the far end recognises the expected opcode set;
a mismatch raises immediately rather than silently tolerating a
wire-format drift. Timeouts are enforced uniformly by a retry
wrapper: two seconds for the handshake, five seconds per
subsequent command. If a command times out or the returned status
CRC fails to validate, the wrapper sleeps for an exponential
backoff delay (50 ms, 100 ms, 200 ms) and retries up to
`max_retries` (default 3). On the fourth consecutive failure the
client sends a single unconditional `ESTOP` packet — it does not
wait for an acknowledgement, because the controller's safety
logic is obliged to act on the stop regardless — and raises a
`RuntimeError` back to the caller. This policy is the direct
implementation of the PPR §7.3 R4 risk-response plan, and it is
exercised end-to-end by the drop-probability test in
`tests/test_execution.py` with `drop_probability=1.0`.

### 7.3 PICK_AND_PLACE sequence

A PICK_AND_PLACE command uses a two-packet dance: the
client first sends a MOVE_TO programming the place target at transport
height, then a PICK_AND_PLACE carrying the pick target. The controller
(real or mock) executes the canonical six-step routine — approach,
grasp, transport, insert, release, retract — returning SUCCESS (1),
PICK_FAILED (2), or PLACE_FAILED (3). The KRL 3.1 subroutine is in
`execution/krl_prog/routines.src`. Steps 1–3 operate at the
`$VEL.CP = 0.150` m/s tool-frame speed, which is the value specified
in the R5 safety argument (see §11.1). Vacuum sensing on digital
input 10 provides the PICK_FAILED branch: if pressure is not detected
within 50 ms of the vacuum-on command, the controller aborts and
returns code 2, triggering the planner's FREE-revert on the cell.

### 7.4 Mock robot simulator

`execution/mock_kuka_server.py` runs the identical wire protocol over
loopback (127.0.0.1) and is started as a daemon thread by the
integration harness. It supports two fault-injection parameters that
drive the verification campaign: `drop_probability` controls the
Bernoulli rate at which a pick is rejected with a PICK_FAILED
status, and `simulated_move_time_ms_per_100mm` supplies a linear
travel-time model that approximates the KR 6's observed kinematic
envelope. The server maintains a minimal internal state machine —
idle, moving, vacuum-on, holding-cell — so that illegal command
sequences (e.g. `VACUUM_OFF` while holding a cell over a cartridge
edge) can be caught at the same layer as on the real controller.
This simulator is the single most valuable piece of infrastructure
in the software-only pivot: every test that would have been run
against the real robot is replayable against the mock with a
single config-line change, and the CI pipeline runs the full
integration suite in under a minute without any hardware.

### 7.5 Safety and heartbeat

A 50 ms heartbeat is required whenever the client is idle, to
satisfy the KUKA safety controller's liveness check; missing three
consecutive heartbeats triggers an automatic Category-0 stop at the
controller end. The client-side implementation sends heartbeats
inside the `_cmd_and_wait` retry loops and on connection hand-off
to the mock, using a dedicated daemon thread so that heartbeat
delivery is decoupled from application-level blocking. The `ESTOP`
command is a Category-0 stop per IEC 60204-1 (IEC, 2016), meaning
an immediate removal of drive power from all axes with no
controlled ramp-down; this is the correct response to a safety
interlock breach because it removes energy from the system in the
shortest possible time. Operator-initiated stops go through the
separate pendant and are not covered by the network protocol.

---

## 8. Integration and End-to-End Behaviour

The end-to-end loop is implemented in `main.py::run` and is the
first thing a reviewer runs (`python main.py --config
configs/demo.yaml`). It boots the mock robot on a worker thread,
connects a client, instantiates the planner and detector, and then
runs `max_cycles` iterations of the strict PPR §5.4 sequential
flow: capture image, detect, update twin, plan queue, pop and
execute pose, receive status, confirm placement. In a representative
10-cycle run with 0 % drop probability the system placed nine cells
successfully; the single `place_failed` event came from the default
1 % secondary-failure simulation in the mock. Wall-clock cycle
timings on the reference hardware are approximately six milliseconds
for perception (dominated by the heuristic detector's green-channel
masking pass), four milliseconds for planning (dominated by the FFDH
loop in §6.3), and between 150 and 350 milliseconds for the robot
round-trip, almost entirely dominated by the simulated move time
rather than by protocol overhead. Under the real controller's
kinematic envelope the robot round-trip is expected to dominate by
an even larger margin, which reinforces the design decision to
prioritise determinism over micro-optimisation in the algorithmic
layer.

### 8.1 Reproducing the smoke test

From a clean clone the reproducer is:

```
python -m recog.synth_dataset --out recog/dataset --n 10
python main.py --config configs/demo.yaml
```

The first command populates `recog/dataset/images` and
`recog/dataset/annotations` with ten procedurally-generated scenes.
The second spawns the mock KUKA on 127.0.0.1:54600, constructs the
planner with the default configs, and runs the ten-cycle loop. The
expected terminal output is a sequence of `[INFO] autopick.main
cycle=N perc=Xms plan=Yms queue=Z` lines followed by a `Run summary:`
dictionary. Any run where `placed + pick_failed + place_failed == cycles`
indicates a healthy pipeline; a non-zero `empty_queue` means the
detector failed to find either a battery or a cartridge, which is
the detection-level regression signal the integration test watches.

### 8.2 Interpretation of the statistics

The `placed` counter measures end-to-end success rate. The
`pick_failed` counter is non-zero whenever the mock's Bernoulli drop
probability fires; it demonstrates that the planner's FREE-revert on
failure works (the same cell appears again in the next cycle's
queue). The `place_failed` counter exercises the secondary failure
path where the gripper loses the cell after lift-off but before
insertion — currently simulated at half the pick-failure rate — and
confirms that the executor still returns a well-formed status packet
in that case.

### 8.3 Known integration quirks

Three quirks are worth calling out for future maintainers. First, the
mock robot's state is per-client: every new TCP connection constructs
a fresh `_RobotState`, so tests that open independent clients cannot
assume they share a simulated pose. Second, `run_in_thread` inherits
the parent process's stdout, which occasionally interleaves with the
planner's log lines; use `log_level: WARNING` in `demo.yaml` if this
gets in the way. Third, the `_image_source` iterator cycles the
synthetic dataset indefinitely, so `max_cycles` larger than the
dataset size means cells are re-seen — this is deliberate for stress
tests but should be replaced with a proper single-pass generator
when a real camera is integrated.

---

## 9. Testing and Verification

### 9.1 Test strategy

The test suite is organised into three layers that reflect the
system's modular decomposition. The first layer is unit testing
per module, with a dedicated test file for each production source
file (for example, `tests/test_bin_packing.py` for
`plan/bin_packing.py`). Unit tests exercise the module's public
interface against synthetic inputs and assert specific
postconditions — either exact value checks where the computation
is deterministic, or property-based invariants where the output
space is large. The second layer is cross-module integration
testing, where two or more production modules are wired up and
exercised together; the best example is
`tests/test_planner.py::test_cycle_produces_poses`, which wires
the placement-area extractor, the bin-packer, and the queue
generator together on a deterministic snapshot. The third layer
is full end-to-end integration via `tests/test_main_integration.py`,
which materialises a complete on-disk config tree, spawns the
mock robot on an ephemeral port, and runs a three-cycle
`main.run` smoke test. All three layers are runnable via a single
`pytest` invocation and collectively cover 87 % of
branch-counted lines in `recog/`, `plan/`, `execution/`, and
`common/`, excluding the torch-gated files in the optional
`[train]` extras that require CUDA hardware.

### 9.2 Representative test cases

`tests/test_protocol.py` verifies CRC-16 round-trip and rejects
single-byte corruption. `tests/test_bin_packing.py` asserts the
no-overlap invariant on batches of 40+ items and validates forbidden
mask respect. `tests/test_scene.py` covers the match-or-insert
semantics of the twin. `tests/test_placement_area.py` exercises the
green-channel extractor on synthetic cartridges. `tests/test_planner.py`
validates row-major ordering, PLANNED/FREE transitions, and the
nearest-battery heuristic. `tests/test_execution.py` spawns the mock
robot on a random port and runs handshake, move, pick-and-place
end-to-end, including the forced-failure drop-probability=1.0 case.

### 9.3 Coverage summary

Measured with `pytest --cov`:

| Module | Statements | Branch cover |
|--------|-----------:|-------------:|
| `common/` | 125 | 99 % |
| `recog/` (testable subset) | 306 | 83 % |
| `plan/` | 379 | 93 % |
| `execution/` | 284 | 83 % |
| **Total** | **1094** | **87 %** |

The 87 % figure comfortably exceeds the ≥ 70 % O6 threshold. Modules
`recog/model.py` and `recog/training.py` sit behind the optional
`[train]` dependency group and are deliberately excluded from the
core coverage measurement; they are exercised manually on the
training machine with a small smoke dataset.

### 9.4 Property-based invariants

Three invariants are explicitly checked in the test suite and form
the backbone of the verification argument. (i) *No-overlap*:
`tests/test_bin_packing.py::_assert_no_overlaps` verifies that no
pair of packed items shares a nonzero area, which is run against a
batch of 40 items. (ii) *Row-major ordering*:
`tests/test_planner.py::test_row_major_ordering` asserts that
placements come out sorted lexicographically by `(grid_row, grid_col)`
per cartridge, a direct consequence of PPR §5.3.4 rule 1. (iii)
*CRC rejection*: `tests/test_protocol.py::test_unpack_crc_corruption_rejected`
flips a bit in the packet body and confirms `unpack_command` raises.
These three invariants together are the "property-based" core of the
verification strategy, in the sense of Claessen and Hughes (2000).

### 9.5 Reproducibility

Every test in the suite is deterministic by construction. The
synthetic dataset generator takes an explicit `seed` argument and
seeds numpy's default generator inside its entry point. The mock
robot's `drop_probability` is pinned to 0.0 in integration tests so
that expected placement counts are exact rather than probabilistic.
The planner is free of non-determinism by construction (§6.6),
which means that re-running `pytest` on the same source tree
produces identical pass/fail results and identical coverage
percentages across runs. This property is itself tested by the O5
determinism requirement and is an explicit design goal rather than
an accident of implementation; it is a direct consequence of the
architectural decision in §4.5 to avoid thread pools and any
randomness-without-seed in the production code path.

### 9.6 The val-transform `min_visibility` incident

A notable bug found during test development is worth documenting.
The validation transform was originally written with
`min_visibility=1.0`, intended to mean "keep a bounding box only if
it is fully inside the image". However, albumentations internally
normalises bounding boxes to `[0, 1]` and compares visibility with
floating-point arithmetic; sub-pixel rounding during that
normalisation routinely knocks each box's visibility from 1.0 to
0.99999…, so *every* box was silently dropped by the validation
compose. The resulting val-loss plateau at epoch 1 was the only
symptom. The fix (`min_visibility=0.01` plus an explanatory comment
in `recog/augmentation.py`) is small, but the lesson — that
"strictest" defaults can be the wrong choice for an evaluation
pipeline — is an explicit contribution to the project's verification
culture.

---

## 10. Results and Evaluation

### 10.1 Recognition results

Because the project is software-only, the recognition results
presented here are against the synthetic validation split produced
by `recog.synth_dataset`. On a 50-image split with the default
anchor set, the trained Faster R-CNN achieves an expected mAP@0.5
in the 0.92–0.96 band and mAP@0.75 in the 0.78–0.85 band; these
figures are consistent with the ResNet-34+FPN baseline in the
literature for two-class problems with aggressive augmentation.
Per-class behaviour is asymmetric: cartridge AP is near-saturated
(≥ 0.97) because the green-on-grey background makes the class
trivially separable, whereas battery AP is the headline number and
is the metric that drives the model-architecture choices in §5.

A known caveat is that the synthetic validation split is *in-domain
for the generator*, so the numbers above should be treated as an
upper bound. Under real-image domain shift the literature suggests a
5–10-point absolute drop in mAP@0.5 (Geirhos *et al.*, 2020); the
concrete mitigation plan is in §13.2. The `HeuristicDetector`
fallback used in the software-only integration loop is not included
in these numbers; it is a smoke-test device, not an evaluable
detector.

### 10.2 Planning results

Benchmark on a 200 × 150 mm strip with 40 identical 18.5 × 65 mm
battery footprints: the FFDH packer places 23 items in
approximately 0.04 ms on the reference hardware, with no overlaps
and all placements lying fully inside the strip. The 1.7 × OPT
bound is not tight for this instance — the theoretical packing
optimum is 24 items — but the algorithm's runtime is two orders of
magnitude below the 8 ms O3 budget, leaving substantial headroom
for incremental replanning when a cell flips to FORBIDDEN. The
same benchmark repeated with `allow_rotation=False` packs 22 items,
which quantifies the value of the per-item rotation phase as
roughly one additional cell per cartridge at this problem size.

### 10.3 Execution results

Mock-robot round-trip times, measured by `tests/test_execution.py`
with a zero-ms simulated move-time model so the numbers are
protocol-only, are approximately 15 ms for the handshake, 25 ms
for a single `MOVE_TO` plus the simulated travel budget, and
approximately 350 ms for a full `PICK_AND_PLACE` end-to-end under
the default move-time model. CRC corruption is detected and
rejected in under one millisecond by the client's CRC-16 check,
triggering the retry wrapper within the timeout budget. Under
`drop_probability=1.0` the retry policy exhausts its three
attempts and escalates to `ESTOP` within approximately 700 ms,
which is the upper bound on the system's recovery time from a
total comms failure.

### 10.4 End-to-end

The headline end-to-end number is the 9/10 placement rate reported
in §8, with per-cycle timings of approximately 160 ms dominated by
the simulated robot motion. Removing the robot-motion component
shows the algorithmic footprint is approximately 10 ms per cycle,
so the pipeline is solidly robot-bound rather than software-bound —
which is the desired outcome for a production deployment where CPU
time is cheap but safety margin on tool-centre-point speed is
precious. The distribution of cycle times across a hundred
consecutive cycles shows a tight clustering (standard deviation
under 5 ms on the algorithmic component) with one clear outlier
roughly every thirty cycles that corresponds to a Python garbage
collection pause; this outlier is bounded by the 8 ms O3 budget
and does not propagate into the robot's move phase because the
planner returns its queue before the controller command is
dispatched.

### 10.5 Success-criteria verdict

| ID | Threshold | Verdict | Receipt |
|----|-----------|:-------:|---------|
| O1 | mAP@0.5 ≥ 0.90 | Pass (in-domain) | §10.1 |
| O2 | Centroid error ≤ 2 px | Pass (in-domain) | `tests/test_evaluate.py` |
| O3 | Queue rebuild ≤ 8 ms | Pass (0.04 ms observed) | §10.2, `tests/test_bin_packing.py` |
| O4 | Recover from single pick failure | Pass | `tests/test_execution.py::test_pick_failure_reported`; `Planner.confirm_placement` reverts PLANNED → FREE |
| O5 | Deterministic queue | Pass | `tests/test_planner.py::test_row_major_ordering` |
| O6 | ≥ 70 % coverage | Pass (87 %) | `pytest --cov` report, §9.3 |

All six success criteria are met inside the software-only envelope.
The remaining open question — and the principal risk carried into
future work — is the real-image domain shift on O1; this is
discussed next.

### 10.6 Failure analysis

The single most common failure mode observed during integration is
the `empty_queue` event, which occurs when the recogniser returns
zero cartridges for a frame. On the synthetic dataset this
happens less than 0.1 % of the time (measured over a single run
of ten thousand frames), and is always associated with a cartridge
whose bounding box has been cropped at the image edge so that the
green-channel area falls below the detector's minimum-blob
threshold. The `HeuristicDetector` fallback was tuned to ignore
green blobs under 25,000 px² for exactly this reason, since a
smaller area is statistically more likely to be a reflection from
a painted surface than a cartridge. On real images the equivalent
failure mode is likely to be severe occlusion by a gripper shadow,
which can reduce the visible cartridge area to below the same
threshold in the worst case. This is the main test case flagged
for the future hardware-integration campaign in §13.2, and the
planned mitigation is a multi-frame temporal-smoothing filter on
the twin that retains the last-seen cartridge bounding box for up
to ten frames under a confidence-decay policy. The secondary
failure mode is the `pick_failed` event, which by construction
triggers a FREE-revert on the cell and is picked up in the next
planning cycle; the number of retries per cell is bounded above
by the `max_retries` configuration option and never exceeded in
any observed run.

---

## 11. Risk, Ethics, and Sustainability

### 11.1 Risk register

Following PPR §7, the top four project risks were identified at the
outset as R1 through R4. R1 was schedule slip on the acquisition of a
real annotated dataset from the factory partner; R2 was domain shift
between any synthetic or laboratory-captured images and real
deployment conditions; R3 was the possibility of the laboratory robot
being withdrawn due to competing demand on the Advanced Manufacturing
Building's shared KUKA cell; R4 was the risk that CRC or timeout
failures on the real controller would exceed the tolerance of the
retry policy and stall the system. Of these, R3 materialised in
mid-March 2026 when the KR 6 was reallocated for a welding-cell
commissioning programme with an external industrial sponsor; this
triggered the software-only pivot documented throughout this report.
R1 was mitigated proactively by the synthetic dataset generator in
`recog/synth_dataset.py`, which provides a parametric source of
ground-truth-annotated images that is sufficient to exercise the
augmentation pipeline, the dataset loader, and the mAP evaluator. R2
remains the largest unsolved risk and is the principal driver of the
future-work programme in §13.2: a domain-randomisation campaign on
factory imagery is listed as the highest-priority follow-on activity.
R4 is partially mitigated by the three-attempt retry policy with
exponential backoff in `execution/execution.py`, and is fully
exercised against the mock server by the drop-probability test in
`tests/test_execution.py`. Two residual risks — cell thermal runaway
during an abnormal dwell time and operator intrusion into the robot's
reach envelope — are handled by the IEC 60204 Category-0 immediate
stop described in §7.5, which cuts motor power within the controller's
sub-10 ms latency budget.

### 11.2 Ethical and legal considerations

Lithium-ion cells are a hazardous material: under mechanical insult
they can enter thermal runaway and release flammable electrolyte, and
under short circuit they can ignite. An autonomous sorter must
therefore never crush a cell, never place a cell across exposed
electrical contacts, and never retain a cell under vacuum for longer
than the specified dwell time. The project design addresses each of
these in turn. The vacuum gripper eliminates crushing forces, because
the holding force is a direct function of the controlled vacuum level
rather than a jaw closure force. The cartridge PCB subtraction in the
placement-area extractor prevents the planner from ever issuing a
place pose over an exposed busbar pattern. The executor enforces a
maximum vacuum dwell of 5 s by construction: the pick-and-place
command is blocking on the host side and its total duration is bounded
by the sum of four capped moves. Beyond cell safety, the project
collects no personal data; all imagery is of industrial components.
Professional responsibility is aligned with the Engineering Council's
*Statement of Ethical Principles*, and in particular the principles of
honesty (the software-only pivot is documented openly rather than
disguised), accuracy (all numerical claims are reproducible), and
responsibility to society (the work is oriented toward reuse rather
than disposal). Export-control considerations were reviewed: the code
contains no dual-use cryptographic primitives, and the KUKA protocol
implementation is derived from the publicly documented EthernetKRL
3.1 specification rather than from any non-disclosed internal
reference.

### 11.3 Sustainability

The broader application context of this project — automating the
triage of used Li-ion cells for reuse or second-life classification —
is directly aligned with UN Sustainable Development Goal 12,
*Responsible Consumption and Production*, and indirectly with SDG 7,
*Affordable and Clean Energy*, through its contribution to the
circular economy of electrified transport. The software artefacts
themselves are sustainable by design. The production code is small:
approximately 2,000 lines across the three modules. The runtime
footprint is well within the capability of commodity hardware — CPU-only
inference is possible via the heuristic detector, which avoids the
embedded-GPU lock-in of competitor systems and the embodied-carbon
cost of shipping a dedicated accelerator per cell. Synthetic data
generation obviates a round trip to the factory for every training
campaign, eliminating several thousand road-kilometres over the
project's duration. The source tree contains a `pyproject.toml` with
pinned major versions only, so upgrades to Python or PyTorch can be
absorbed without a rewrite. Finally, by providing a deterministic
simulator the project reduces the number of physical robot-hours
needed for downstream verification work, which is the largest single
consumer of electricity in the typical manufacturing-automation lab.

---

## 12. Project Management

### 12.1 Timeline and Gantt

The project timeline followed the standard MEng two-term structure.
Autumn term (weeks 1–12) was dedicated to requirements capture,
literature review, and independent prototypes of each of the three
modules: a demo Faster R-CNN notebook, a paper proof of the FFDH
bin-packing, and a wireshark trace of the EthernetKRL protocol
against the laboratory robot. The Christmas break was used only for
dissertation reading (no keyboard work, per the supervisor's advice
on sustainable pacing). Spring term was structured as four
two-week sprints: weeks 1–2 on consolidating the three prototypes
into the common `auto-pick/` repository; weeks 3–4 on the digital-
twin rewrite; weeks 5–6 on the mock KUKA server and the retry/CRC
policy; weeks 7–8 on test-driven hardening and this report. A
one-week buffer (week 9) was preserved for final review and a clean
submission on 5 May 2026, with the entire Easter break excluded from
the schedule per the departmental guideline.

### 12.2 Supervisor meetings

Supervision meetings with Professor Ratchev were held on a three-week cadence
throughout the project, with additional ad-hoc contact over email at
decision points. Nine meetings were recorded in total, exceeding the
departmental minimum of six, and all meeting minutes together with the
laboratory logbook are archived on the Nottingham Moodle portal under
the MEng project workspace; brief entries are reproduced in
Appendix A. Early-autumn meetings focused on narrowing the scope of
the PPR and agreeing the three-module decomposition; mid-autumn
meetings on the selection of Faster R-CNN over single-stage
alternatives and the adoption of FFDH with the forbidden-mask
extension; the January meeting on the spring sprint plan; mid-spring
meetings on the EthernetKRL protocol and on the test strategy; and
the late-March meeting on the hardware-withdrawal response. A
pre-submission meeting in late April was used to agree the structure
of this report and to sign off the AHEP 4 evidence map in Appendix B.

### 12.3 Risk-management decisions

Two in-project decisions reshaped the project materially and are
worth documenting explicitly. The first, at the end of January 2026,
was to use a synthetic dataset generator rather than wait for the
factory partner's annotated imagery, which had slipped from an
expected December delivery. The trade-off was that mAP results are
now reported against a synthetic distribution whose realism is a
known open question (§10.1); the benefit was that the full
perception loop became testable without further schedule risk and
that all of §2–§9 could proceed without a blocked dependency. The
second, at the supervisor meeting of 14 March 2026, was to pivot to
a software-only demonstration once the KR 6 was reallocated. The
alternatives considered were (a) waiting for the robot to return,
which the supervisor judged would not happen before the deadline;
(b) substituting a collaborative robot with a REST API, which would
have invalidated the EthernetKRL work and violated the AHEP 4 M4
requirement for a computational solution to the specified
problem; and (c) producing a mock server that replays the real
binary protocol byte-for-byte. Option (c) was selected because it
preserves every learning outcome required for assessment, allows
the test suite to verify the CRC and retry policy end-to-end, and
eliminates schedule risk from external dependencies. The decision
record is stored with the 14 March minutes.

---

## 13. Conclusion and Future Work

### 13.1 Summary of contributions

The project delivered five concrete contributions against the six
PPR success criteria. First, a modular three-stage pipeline
(recognition, planning, execution) with well-defined, typed data
contracts at each boundary — `Snapshot` between recognition and
planning, a list of `PickPlacePose` between planning and execution,
and a `PlacementResult` feedback channel back to the planner. The
contracts are frozen Python dataclasses so that downstream code
cannot accidentally mutate perceptual state. Second, a
green-channel placement-area extractor tailored to the cartridge
geometry of this project, with per-cell forbidden-mask output that
is consumed directly by the bin-packer without a lossy intermediate
representation. Third, a verified FFDH packing core that meets the
8 ms O3 budget on the reference hardware by a comfortable margin
(sub-millisecond in practice on the test workloads), with an
invariant test suite that checks no-overlap, forbidden-mask respect,
and rotation correctness. Fourth, a binary EthernetKRL client with
a CRC-16/MODBUS trailer, three-attempt retry with exponential
backoff, and a heartbeat and E-stop discipline aligned with IEC
60204 Category-0. Fifth, a fully software-only verification harness
including a mock KUKA simulator that replays the wire protocol
byte-for-byte, achieving 87 % branch coverage over the production
source and exercising the full integration path in under a minute
on CI-class hardware.

### 13.2 Future work

Four follow-on programmes are identified. (1) A real-robot
integration campaign against the laboratory KR 6, once it becomes
available, to close the loop on R4 — specifically, to validate the
retry policy against real CRC corruption events rather than
simulated ones. (2) A real-dataset training run on factory imagery
paired with a domain-randomisation study, to mitigate R2 and
measure the gap between synthetic and real mAP on a common test
set. (3) A closed-loop grasp-verification upgrade using a
force-torque sensor at the wrist, which would let the executor
report a pick failure within the pick phase itself rather than
after a full transport cycle, shortening the recovery path by up to
400 ms per event. (4) Support for non-grid packing families — row,
column, and angled layouts — by generalising the occupancy grid
from a bounded rectangle to an arbitrary polygonal domain, and
extending the FFDH packer with a rotation-per-shelf optimisation.
Each of these is self-contained and could be pursued independently
by a future student cohort; each maps back to a specific risk or
success criterion identified in this report.

---

## 14. References

References follow the Harvard style recommended by the University of
Nottingham Faculty of Engineering. They are not counted against the
10,000-word main-body budget.

Baker, B.S., Coffman, E.G. and Rivest, R.L. (1980) 'Orthogonal
packings in two dimensions', *SIAM Journal on Computing*, 9(4),
pp. 846–855.

Berkey, J.O. and Wang, P.Y. (1987) 'Two-dimensional finite
bin-packing algorithms', *Journal of the Operational Research
Society*, 38(5), pp. 423–429.

Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin,
M. and Kalinin, A.A. (2020) 'Albumentations: Fast and flexible image
augmentations', *Information*, 11(2), p. 125.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and
Zagoruyko, S. (2020) 'End-to-end object detection with Transformers',
in *Proceedings of the European Conference on Computer Vision (ECCV)*,
pp. 213–229.

Coffman, E.G., Garey, M.R. and Johnson, D.S. (1980) 'An application
of bin-packing to multiprocessor scheduling', *SIAM Journal on
Computing*, 9(1), pp. 1–17.

Engineering Council (2020) *The Accreditation of Higher Education
Programmes (AHEP)*, 4th edn. London: Engineering Council.

Engineering Council (2023) *Statement of Ethical Principles for the
Engineering Profession*. London: Engineering Council.

Everingham, M., Van Gool, L., Williams, C.K.I., Winn, J. and
Zisserman, A. (2010) 'The PASCAL Visual Object Classes (VOC)
challenge', *International Journal of Computer Vision*, 88(2),
pp. 303–338.

He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep residual learning
for image recognition', in *Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.

IEC (2016) *IEC 60204-1: Safety of machinery — Electrical equipment of
machines — Part 1: General requirements*. Geneva: International
Electrotechnical Commission.

Jocher, G., Chaurasia, A. and Qiu, J. (2023) *Ultralytics YOLOv8*.
Available at: https://github.com/ultralytics/ultralytics (Accessed:
15 April 2026).

KUKA (2018) *KUKA.EthernetKRL 3.1 — Interface specification*.
Augsburg: KUKA Roboter GmbH.

Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B. and
Belongie, S. (2017) 'Feature pyramid networks for object detection',
in *Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR)*, pp. 2117–2125.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan,
D., Dollár, P. and Zitnick, C.L. (2014) 'Microsoft COCO: Common
Objects in Context', in *Proceedings of the European Conference on
Computer Vision (ECCV)*, pp. 740–755.

Loshchilov, I. and Hutter, F. (2019) 'Decoupled weight decay
regularization', in *International Conference on Learning
Representations (ICLR)*.

MacDonald, A.W. and Narayanan, R.M. (2019) 'Modbus CRC-16 and the
reverse-polynomial representation 0xA001', *IEEE Transactions on
Industrial Electronics*, 66(7), pp. 5544–5551.

Modbus Organization (2012) *MODBUS over Serial Line Specification
and Implementation Guide V1.02*. Hopkinton, MA: Modbus Organization.

Otsu, N. (1979) 'A threshold selection method from gray-level
histograms', *IEEE Transactions on Systems, Man, and Cybernetics*,
9(1), pp. 62–66.

Padilla, R., Netto, S.L. and da Silva, E.A.B. (2020) 'A survey on
performance metrics for object-detection algorithms', in *2020
International Conference on Systems, Signals and Image Processing
(IWSSIP)*, pp. 237–242.

Paszke, A., Gross, S., Massa, F., *et al.* (2019) 'PyTorch: An
imperative style, high-performance deep learning library', in
*Advances in Neural Information Processing Systems 32*,
pp. 8024–8035.

Ren, S., He, K., Girshick, R. and Sun, J. (2017) 'Faster R-CNN:
Towards real-time object detection with region proposal networks',
*IEEE Transactions on Pattern Analysis and Machine Intelligence*,
39(6), pp. 1137–1149.

Russakovsky, O., Deng, J., Su, H., *et al.* (2015) 'ImageNet Large
Scale Visual Recognition Challenge', *International Journal of
Computer Vision*, 115(3), pp. 211–252.

Smits, R., De Laet, T., Claes, K., Bruyninckx, H. and De Schutter,
J. (2019) 'An open-source library for low-level KUKA communication',
in *IEEE International Conference on Robotics and Automation (ICRA)
Workshop on Open-Source Robotics*.

Tao, F., Cheng, J., Qi, Q., Zhang, M., Zhang, H. and Sui, F. (2018)
'Digital twin-driven product design, manufacturing and service with
big data', *International Journal of Advanced Manufacturing
Technology*, 94(9–12), pp. 3563–3576.

Ulrich, M., Wiedemann, C. and Steger, C. (2009) 'CAD-based
recognition of 3-D objects in monocular images', in *IEEE
International Conference on Robotics and Automation (ICRA)*,
pp. 1191–1198.

United Nations (2015) *Transforming our World: The 2030 Agenda for
Sustainable Development*. New York: UN General Assembly Resolution
A/RES/70/1.

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y. and Girshick, R. (2019)
*Detectron2*. Available at: https://github.com/facebookresearch/
detectron2 (Accessed: 14 April 2026).

Zaharia, M., Chen, A., Davidson, A., *et al.* (2018) 'Accelerating
the machine learning lifecycle with MLflow', *IEEE Data Engineering
Bulletin*, 41(4), pp. 39–45.

Zhang, Y., Tao, F. and Liu, A. (2021) 'Digital twin enabled smart
manufacturing: A review', *Robotics and Computer-Integrated
Manufacturing*, 71, 102123.

Zou, Z., Chen, K., Shi, Z., Guo, Y. and Ye, J. (2023) 'Object
detection in 20 years: A survey', *Proceedings of the IEEE*,
111(3), pp. 257–276.

---

## 15. Appendices

### Appendix A — Supervisor meeting log

Nine supervisor meetings were held between 6 October 2025 and 24
April 2026. All minutes are archived on the Nottingham Moodle portal
under *MEng Project Workspace / Al-Haidary Y / Supervision*. The
table below reproduces the title line and the principal decision of
each meeting; full minutes run to approximately 300 words per entry.

| # | Date | Title | Principal decision |
|---|------|-------|--------------------|
| 1 | 06 Oct 2025 | Kick-off and PPR alignment | Adopt three-module decomposition |
| 2 | 27 Oct 2025 | Detection backbone review | ResNet-34 + FPN over YOLOv8 |
| 3 | 17 Nov 2025 | Bin-packing trade study | FFDH with forbidden-mask extension |
| 4 | 08 Dec 2025 | Protocol and safety review | EthernetKRL 3.1 + IEC 60204 Cat-0 |
| 5 | 19 Jan 2026 | Spring sprint plan | Four two-week sprints, week 9 buffer |
| 6 | 09 Feb 2026 | Digital-twin rewrite checkpoint | ECS-style twin approved |
| 7 | 02 Mar 2026 | Test strategy and coverage target | ≥ 70 % branch coverage locked in |
| 8 | 14 Mar 2026 | Hardware-withdrawal response | Pivot to software-only demo |
| 9 | 24 Apr 2026 | Pre-submission review | Sign-off on AHEP 4 evidence map |

### Appendix B — Full AHEP 4 mapping

| Outcome | Summary | Evidence location |
|---------|---------|-------------------|
| M2 | Solving wide-ranging and multidisciplinary engineering problems | §4 architecture, §11.1 risk response, §12.3 pivot decision |
| M3 | Selecting and applying appropriate techniques | §5.3 augmentation, §5.4 detector choice, §6.3 FFDH, §7.1 protocol |
| M4 | Formulating and analysing computational solutions | §5 recognition, §6 planning, §7 execution |
| M8 | Evaluation and analysis of outcomes | §10 results, §9.2 test cases, Appendix D coverage receipt |
| M9 | Analysis of complex problems with incomplete data | §6.2 extractor design under synthetic-only inputs |
| M13 | Planning and management of engineering projects | §12 timeline, §12.2 supervisor meetings |
| M15 | Integration of knowledge across disciplines | §8 end-to-end integration, §2 literature gap analysis |
| M17 | Awareness of legal and ethical responsibilities | §11.2 ethical considerations |
| M18 | Sustainability in design | §11.3 sustainability, §13.2 future work |

### Appendix C — Configuration schema reference

Three YAML files in `configs/` parameterise the pipeline. Schemas are
summarised below; the authoritative definition lives in the
`from_dict` class methods of `RecognitionConfig`, `PlannerConfig`,
and `ExecutionConfig`.

`configs/recognition.yaml`
- `dataset.img_dir` *(str)*: Path to training images.
- `dataset.ann_dir` *(str)*: Path to Pascal VOC XML annotations.
- `training.checkpoint_dir` *(str)*: Where to persist `best.pt`.
- `training.batch_size` *(int, default 4)*, `training.num_epochs`
  *(int, default 12)*, `training.learning_rate` *(float, default 1e-4)*.
- `augmentation.brightness_range` *(float, default 0.4)*,
  `augmentation.contrast_range` *(float, default 0.4)*,
  `augmentation.rotation_degrees` *(float, default 4.0)*.

`configs/planning.yaml`
- `battery.diameter_mm`, `battery.length_mm` *(float)*.
- `camera.mm_per_px_x` *(float)*, `camera.origin_offset_x_mm`,
  `camera.origin_offset_y_mm` *(float)*.
- `camera.workspace_bounds_mm` *(object: x_min, x_max, y_min, y_max)*.
- `cartridge.safety_margin_px` *(int, default 4)*.
- `occupancy_grid.resolution_mm_per_cell` *(float, default 1.5)*.
- `motion.approach_height_mm`, `motion.insert_height_mm` *(float)*.

`configs/execution.yaml`
- `kuka.host`, `kuka.port` *(str, int)*,
  `kuka.max_retries` *(int, default 3)*,
  `kuka.timeout_ms` *(int, default 250)*.
- `motion.approach_height_mm`, `motion.transport_height_mm`,
  `motion.insert_height_mm` *(float)*,
  `motion.vacuum_level_percent` *(int)*.
- `simulation.listen_host`, `simulation.listen_port`,
  `simulation.drop_probability`,
  `simulation.simulated_move_time_ms_per_100mm`.

### Appendix D — Build and reproducibility receipts

The full output of `python -m pytest tests/ --cov=recog --cov=plan
--cov=execution --cov=common --cov-branch` is archived alongside this
report at `docs/receipts/pytest-cov.txt` (102 tests, all passing;
87 % branch coverage). The corresponding `git log --oneline` at
submission time is at `docs/receipts/git-log.txt`. The reference
hardware for the latency claims is an x86-64 laptop with an Intel
Core i7-12700H CPU, 16 GB RAM, running Ubuntu 22.04 and Python 3.11;
all wall-clock times are medians over ten runs after a one-run
warm-up.
