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
"""
from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Iterable


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "code_name": self.code.name,
            "current_pose": self.current_pose.to_dict(),
            "cycle_time_ms": float(self.cycle_time_ms),
            "message": self.message,
        }


# ---------------------------------------------------------- small helpers --

def iter_labels(names: Iterable[str]) -> list[ClassLabel]:
    """Parse a sequence of label strings, silently dropping unknown ones."""
    out: list[ClassLabel] = []
    for n in names:
        try:
            out.append(ClassLabel(n))
        except ValueError:
            continue
    return out
