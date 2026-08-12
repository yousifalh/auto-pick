"""Shared data contracts between the three auto-pick modules.

Every type in this file is either a ``frozen`` dataclass or an
:class:`enum.Enum`, so instances are safe to share across module
boundaries without defensive copying. Two design rules govern this
module:

1.  The types here are the *only* values that cross the boundaries
    ``Recognition → Planning`` and ``Planning → Execution``. Anything
    module-internal lives in that module.
2.  Every type provides a ``to_dict()`` method that emits
    JSON-compatible primitives. This is what makes the pipeline
    serialisable for logging, regression fixtures, and inspection.
3.  A type that states an invariant in its docstring **checks it in
    ``__post_init__``**. Frozen guarantees a value will not change; it
    guarantees nothing about the value being possible, and until
    2026-08-12 possible was not checked anywhere in this file. An
    inverted ``BBox(100, 100, 0, 0)`` constructed with ``width == -100``
    and then laundered itself into a plausible ``area == 0.0``;
    ``Detection(confidence=17.5)``, ``grid_row=-5`` and
    ``cycle_time_ms=-42.0`` all constructed. See §"What is NOT checked"
    below for the line drawn.

**What is NOT checked, on purpose.** Only invariants that are both cheap
and unambiguous live here. ``WorkspacePoint`` is not range-checked: any
finite triple is a physically meaningful pose and the envelope that
decides reachability is per-deployment configuration
(``plan.scene.WorkspaceBounds``), not a property of the type.
``PickPlacePose.battery_detection_id`` is not checked because ``-1`` is
its documented "no battery" sentinel.

``Snapshot`` gets no ``__post_init__`` at all, and that is a decision
rather than an omission. It is the one type here that is not frozen, and
every real writer of ``mm_per_px`` sets it by attribute assignment
*after* construction (``main.py:540``, ``recog.inference``), which
``__post_init__`` cannot see. A constructor check would read as
protection while being structurally unable to fire on the only path that
sets the field. The honest guard for scale is
``plan.placement_area._resolve_scale``, which already rejects ``<= 0.0``
with "a pixel cannot span zero or negative millimetres", and it sits
where the value is consumed.

**Cost, measured rather than assumed.** ``__post_init__`` on a frozen
dataclass is one extra Python call per construction. Per construction,
mean of two 200 k-iteration runs each side:

===============  =========  ========  =======
type             before     after     delta
===============  =========  ========  =======
``BBox``            284 ns    317 ns   +33 ns
``Detection``       224 ns    257 ns   +33 ns
``PickPlacePose``   514 ns    543 ns   +29 ns
``RobotStatus``     408 ns    442 ns   +34 ns
``WorkspacePoint``  219 ns    213 ns    -6 ns
===============  =========  ========  =======

``WorkspacePoint`` is the control: it gained no ``__post_init__``, and
its -6 ns is the measurement noise floor.

Instrumented, one ``Planner.cycle`` on the planner's own fixture (one
cartridge, 20 batteries) constructs **21** of these types in total — 1
``BBox`` and 20 ``PickPlacePose``. At +33 ns that is **0.7 microseconds
on an 8.4 ms cycle, 0.008 %**, four orders of magnitude below the
7.9-9.9 ms run-to-run spread of the cycle itself, which is dominated by
OpenCV and the packer. Not material. The detections themselves are
built once per frame by the recogniser, tens per frame, against a
~12.6 ms segmentation forward pass.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


# ----------------------------------------------------------- geometry ------

@dataclass(frozen=True)
class BBox: 
    """An axis-aligned bounding box in image-pixel coordinates.

    The convention is 0-based with inclusive ``xmin``/``ymin`` and
    exclusive ``xmax``/``ymax``, the same order and units as Pascal
    VOC's ``<bndbox>`` — but *not* VOC's indexing. The real VOC devkit
    is 1-based with inclusive max edges and carries a ``+1`` in its IoU
    (``iw = xmax - xmin + 1``); this codebase is 0-based exclusive
    throughout, which is the modern convention (torchvision,
    albumentations' ``pascal_voc`` format, COCO after the xywh
    conversion) and is self-consistent end to end. The IoU therefore
    differs from the devkit's by a sub-percent amount that grows as
    boxes get smaller. A zero-area box is a valid value and round-trips
    through :meth:`iou` as ``0.0``.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        """Enforce the ordering the class docstring states.

        An inverted box is self-concealing: ``area`` clamps with
        ``max(0.0, ...)`` and ``iou`` returns ``0.0``, so
        ``BBox(100, 100, 0, 0)`` used to launder a negative width into a
        plausible zero-area value and propagate it into ``Detection``,
        ``Cartridge``, ``Battery`` and ``PlacementArea.rectangle``.
        Zero-area (``xmin == xmax``) stays legal — the docstring calls it
        a valid value and :meth:`iou` round-trips it as ``0.0``.

        NaN fails this test, which is intended: a NaN coordinate is a
        detector that has diverged, and it should stop here rather than
        at the first arithmetic that quietly propagates it.
        """
        if not (self.xmin <= self.xmax and self.ymin <= self.ymax):
            raise ValueError(
                f"BBox is inverted or non-finite: "
                f"({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}) — "
                f"xmin <= xmax and ymin <= ymax is the stated convention")

    # --- derived scalars --------------------------------------------------

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return 0.5 * (self.xmin + self.xmax)

    @property
    def cy(self) -> float:
        return 0.5 * (self.ymin + self.ymax)

    # --- geometric predicates --------------------------------------------

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union against ``other``.

        Touching-but-not-overlapping boxes and degenerate (zero-area)
        boxes both return ``0.0``.
        """
        x0 = max(self.xmin, other.xmin)
        y0 = max(self.ymin, other.ymin)
        x1 = min(self.xmax, other.xmax)
        y1 = min(self.ymax, other.ymax)
        iw = max(0.0, x1 - x0)
        ih = max(0.0, y1 - y0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def to_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


# ----------------------------------------------------------- perception ---

class ClassLabel(str, Enum):
    """Closed set of classes the recogniser can emit."""

    BACKGROUND = "background"
    BATTERY = "battery"
    CARTRIDGE = "cartridge"


@dataclass(frozen=True)
class Detection:
    """One detection emitted by the recognition module."""

    bbox: BBox
    label: ClassLabel
    confidence: float

    def __post_init__(self) -> None:
        """``confidence`` is a probability, so it lives in [0, 1].

        Both ends are closed: a detector may legitimately saturate at
        exactly 0.0 or 1.0. Out of range is not a near miss — confidence
        feeds every score threshold and the NMS ordering in
        :mod:`recog.inference`, and a value above 1 or below 0 sorts
        first or last unconditionally, ahead of every real detection.
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Detection.confidence must be a probability in [0, 1], "
                f"got {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox.to_list()),
            "label": self.label.value,
            "confidence": float(self.confidence),
        }


@dataclass
class Snapshot:
    """A frame's detections plus provenance metadata.

    ``Snapshot`` is deliberately *not* frozen: the recogniser may append
    to a working snapshot during inference. The planner consumes it as
    read-only; nothing downstream mutates it.
    """

    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, int] = (1080, 1920)  # (H, W)
    timestamp_ns: int = 0
    # Detection index -> (H, W) int8 label map over that cartridge's ROI,
    # using recog.seg_dataset.SEG_CHANNELS.
    #
    # Segmentation lives in Recognition rather than Planning because a
    # DeepLabv3 forward pass is ~12.6 ms for one cartridge, against FDR
    # O3's tested 8 ms per-cartridge planning budget. The masks have to
    # cross the module boundary somehow, and this type is the boundary.
    #
    # Optional and defaulted: every existing producer and consumer stays
    # valid, and the heuristic extractor ignores it entirely.
    cartridge_masks: dict[int, Any] = field(default_factory=dict)
    # This frame's ground sample distance, millimetres per pixel, or None
    # when the frame carries no calibration.
    #
    # Scale is a property of the FRAME, not of the configuration. It
    # crosses the Recognition -> Planning boundary here because this type
    # IS that boundary and because the planner and its placement-area
    # extractor must use the SAME number: the extractor turns it into the
    # cartridge wall's erosion radius, the planner turns the resulting
    # rectangle into workspace millimetres, and the two disagreeing puts
    # a cell somewhere neither of them checked.
    #
    # It was a config constant until 2026-08-11. On a scale-randomised
    # corpus that constant under-read 24 of 30 cartridges by 27 % at the
    # median and OVER-read the rest, so the planner reserved a footprint
    # smaller than the cell it was about to place - 3 of 17 placements
    # landed on ground-truth non-floor material, worst 21.2 %
    # (docs/superpowers/specs/2026-08-11-scale-calibration.md).
    #
    # None means UNKNOWN, not "use the default". plan.planner.Planner
    # falls back to an explicitly configured calibration if there is one
    # and raises UnknownScale if there is not.
    mm_per_px: float | None = None

    def of(self, label: ClassLabel) -> list[Detection]:
        return [d for d in self.detections if d.label is label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_shape": list(self.image_shape),
            "timestamp_ns": int(self.timestamp_ns),
            # Emitted even when None. A log line that omits the frame's
            # calibration cannot distinguish "measured at 0.86 mm/px"
            # from "measured at whatever the config said", and telling
            # those apart after the fact is the entire point of the
            # field.
            "mm_per_px": (None if self.mm_per_px is None
                          else float(self.mm_per_px)),
            # Shape only. to_dict feeds logging and regression fixtures;
            # embedding a full label map per cartridge would make every
            # log line enormous and every fixture unreadable.
            "cartridge_masks": {
                str(k): list(v.shape) for k, v in self.cartridge_masks.items()
            },
        }


# ----------------------------------------------- execution / workspace ---

@dataclass(frozen=True)
class WorkspacePoint:
    """A 3-D pose in the robot's workspace frame, in millimetres."""

    x_mm: float
    y_mm: float
    z_mm: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "z_mm": self.z_mm,
        }


@dataclass(frozen=True)
class PickPlacePose:
    """A single pick/place command, plus the planner's bookkeeping."""

    pick: WorkspacePoint
    place: WorkspacePoint
    cartridge_id: int
    grid_row: int
    grid_col: int
    battery_detection_id: int = -1

    def __post_init__(self) -> None:
        """Grid indices are cell coordinates, so they are non-negative.

        A negative index is the classic numpy silent-wraparound hazard:
        ``grid[-5]`` addresses the fifth row from the end and marks a
        cell nobody asked for. :meth:`plan.scene.OccupancyGrid.set_block`
        already refuses one, but that defence lives in the *consumer* —
        any future consumer re-derives it or does not. It belongs on the
        contract.

        ``battery_detection_id`` is deliberately exempt: ``-1`` is its
        documented default, meaning "no battery is associated".
        """
        if self.grid_row < 0 or self.grid_col < 0:
            raise ValueError(
                f"PickPlacePose grid indices must be non-negative, got "
                f"row={self.grid_row}, col={self.grid_col}")
        if self.cartridge_id < 0:
            raise ValueError(
                f"PickPlacePose.cartridge_id must be non-negative, got "
                f"{self.cartridge_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pick": self.pick.to_dict(),
            "place": self.place.to_dict(),
            "cartridge_id": int(self.cartridge_id),
            "grid_row": int(self.grid_row),
            "grid_col": int(self.grid_col),
            "battery_detection_id": int(self.battery_detection_id),
        }


class RobotStatusCode(IntEnum):
    """Stable status-code enum. Values must not be renumbered without
    matching updates in :mod:`execution.protocol` and the KRL
    subroutine."""

    OK = 0
    SUCCESS = 1
    PICK_FAILED = 2
    PLACE_FAILED = 3
    TIMEOUT = 4
    CRC_ERROR = 5
    ESTOP = 6
    # 7 and 8 exist because a controller that answers "I could not parse
    # that" must be able to say WHY. Reporting an unknown opcode or a
    # version mismatch as CRC_ERROR (which the simulator did) makes a
    # build mismatch indistinguishable from a noisy cable, and the two
    # want opposite responses: retry the line, versus stop and fix the
    # build. execution.KukaClient retries CRC_ERROR and treats these two
    # as fatal.
    UNSUPPORTED_COMMAND = 7
    VERSION_MISMATCH = 8


@dataclass(frozen=True)
class RobotStatus:
    """Structured status packet returned by the executor after a command."""

    code: RobotStatusCode
    current_pose: WorkspacePoint
    cycle_time_ms: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        """A duration cannot be negative.

        ``cycle_time_ms`` flows straight into the latency statistics the
        demo prints and the FDR quotes. A negative value there is not a
        smaller number, it is a number that makes a mean meaningless,
        and the wire format (unsigned on the controller side) cannot
        produce one honestly.
        """
        if self.cycle_time_ms < 0.0:
            raise ValueError(
                f"RobotStatus.cycle_time_ms is a duration and cannot be "
                f"negative, got {self.cycle_time_ms!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "code_name": self.code.name,
            "current_pose": self.current_pose.to_dict(),
            "cycle_time_ms": float(self.cycle_time_ms),
            "message": self.message,
        }


# There is deliberately no ``iter_labels(names)`` helper here. It existed
# until 2026-08-12 with zero callers — not even a test — and its
# documented behaviour was to drop an unrecognised class label silently:
# no exception, no log, one fewer detection than the caller passed in.
# In the module that declares itself the boundary between all three
# subsystems, that is a defect waiting for its first caller. Parse a
# label with ``ClassLabel(name)`` and let the ``ValueError`` out.
