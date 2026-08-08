"""Digital-twin / environment model.

The planner and executor both read from a single authoritative
:class:`EnvironmentModel`. Keeping one source of truth is what makes
the pipeline debuggable: every action the robot takes maps back to a
specific twin state, and the state transitions are explicit.

The design takes inspiration from entity-component-system architectures.
Each tracked entity (a battery, a cartridge) is a thin container with
an AABB and a handful of optional *components* (``placeable_rectangle``,
``occupancy``, ``pcb_mask``). Planning passes attach those components
lazily as they need them, which keeps the hot recognition path free
of planning-specific state.

Cartridges are **persistent** across frames: the model matches each
new cartridge detection to an existing entity by IoU, preserving
component data like the rasterised occupancy grid. Batteries are
**ephemeral**: replaced wholesale every frame because the gripper
picks them up between snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.types import BBox, ClassLabel, Snapshot


# ---------------------------------------------------------- enumerations --

class CellState(Enum):
    """States an occupancy-grid cell can take."""

    FREE = 0
    FORBIDDEN = 1   # PCB region or mechanical fixture
    PLANNED = 2     # Assigned to the queue but not yet placed
    PLACED = 3      # Physically placed and confirmed by the executor


class PackingFamily(Enum):
    """Which packing strategy applies to a given cartridge."""

    GRID = "grid"
    ROW = "row"
    COLUMN = "column"


# -------------------------------------------------------------- entities --

@dataclass
class Battery:
    """A loose battery observed by the camera."""

    id: int
    bbox: BBox
    confidence: float
    assigned_to_pose: bool = False

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.bbox.cx, self.bbox.cy)


@dataclass
class Cartridge:
    """A cartridge tracked in the digital twin.

    All planning-specific components (``pcb_mask``,
    ``placeable_rectangle``, ``occupancy``) are optional — they are
    populated by the placement-area extractor on first use and then
    carried across frames.
    """

    id: int
    bbox: BBox
    confidence: float
    # Index into the originating Snapshot.detections this cartridge came
    # from. Links back to Snapshot.cartridge_masks, which is keyed the
    # same way (recog.inference.attach_cartridge_masks) - NOT by position
    # within the cartridge-only subset, which would misalign every mask
    # the moment a battery detection precedes a cartridge in the list.
    # -1 means "no detection this frame" (matches nothing in
    # cartridge_masks, which is exactly the safe default).
    detection_index: int = -1
    # PCB-region binary mask in full-image coordinates.
    pcb_mask: Optional[np.ndarray] = None
    # Axis-aligned placeable rectangle after insetting + PCB subtraction.
    placeable_rectangle: Optional[BBox] = None
    # Rasterised occupancy over the placeable rectangle.
    occupancy: Optional["OccupancyGrid"] = None
    packing_family: PackingFamily = PackingFamily.GRID

    def mark_cell(self, row: int, col: int, state: CellState) -> None:
        if self.occupancy is None:
            raise RuntimeError("occupancy not initialised")
        self.occupancy.set(row, col, state)


# ------------------------------------------------------- occupancy grid --

@dataclass
class OccupancyGrid:
    """Rasterised occupancy grid for a single cartridge.

    Cells are ``resolution_mm`` square. Cell ``(0, 0)`` is the top-left
    of the cartridge bounding box in image coordinates, aligned to the
    image axes.
    """

    rows: int
    cols: int
    resolution_mm: float
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = np.full(
            (self.rows, self.cols), CellState.FREE.value, dtype=np.uint8,
        )

    # ---- mutation --------------------------------------------------------

    def set(self, row: int, col: int, state: CellState) -> None:
        self.grid[row, col] = state.value

    def get(self, row: int, col: int) -> CellState:
        return CellState(int(self.grid[row, col]))

    # ---- aggregates ------------------------------------------------------

    def free_count(self) -> int:
        return int((self.grid == CellState.FREE.value).sum())

    def placed_count(self) -> int:
        return int((self.grid == CellState.PLACED.value).sum())

    def planned_count(self) -> int:
        return int((self.grid == CellState.PLANNED.value).sum())

    def mask_of(self, *states: CellState) -> np.ndarray:
        """Return a boolean mask of cells in any of ``states``."""
        out = np.zeros_like(self.grid, dtype=bool)
        for s in states:
            out |= (self.grid == s.value)
        return out


# --------------------------------------------------------- environment --

@dataclass
class WorkspaceBounds:
    """Axis-aligned robot-workspace volume in millimetres."""

    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float


@dataclass
class EnvironmentModel:
    """Top-level scene state consumed by both the planner and executor."""

    workspace: WorkspaceBounds
    batteries: List[Battery] = field(default_factory=list)
    cartridges: Dict[int, Cartridge] = field(default_factory=dict)
    _next_battery_id: int = field(default=0, init=False, repr=False)
    _next_cartridge_id: int = field(default=0, init=False, repr=False)

    # ---- update semantics -----------------------------------------------

    def update_from_snapshot(
        self,
        snapshot: Snapshot,
        iou_match_threshold: float = 0.5,
    ) -> None:
        """Fuse ``snapshot`` into the twin.

        * Batteries are replaced wholesale (ephemeral entities).
        * Cartridges are matched to existing IDs by IoU so persistent
          components (occupancy, PCB mask) survive across frames.
          Cartridges absent from the new snapshot are dropped.
        """
        self._replace_batteries(snapshot)
        self._match_or_insert_cartridges(snapshot, iou_match_threshold)

    def _replace_batteries(self, snapshot: Snapshot) -> None:
        self.batteries = []
        for det in snapshot.of(ClassLabel.BATTERY):
            self.batteries.append(Battery(
                id=self._next_battery_id,
                bbox=det.bbox,
                confidence=det.confidence,
            ))
            self._next_battery_id += 1

    def _match_or_insert_cartridges(
        self, snapshot: Snapshot, iou_match_threshold: float,
    ) -> None:
        # Track each detection's index into the FULL snapshot.detections
        # list (not its position within this cartridge-only subset): that
        # is the convention recog.inference.attach_cartridge_masks uses to
        # key Snapshot.cartridge_masks, and the two must agree or every
        # mask lookup in _ensure_placement_areas misaligns.
        new_dets = [
            (i, det) for i, det in enumerate(snapshot.detections)
            if det.label is ClassLabel.CARTRIDGE
        ]
        matched: set[int] = set()

        for det_idx, det in new_dets:
            best_id, best_iou = -1, 0.0
            for cid, ctg in self.cartridges.items():
                if cid in matched:
                    continue
                iou = det.bbox.iou(ctg.bbox)
                if iou > best_iou:
                    best_id, best_iou = cid, iou

            if best_iou >= iou_match_threshold:
                # Existing cartridge — update geometry, keep components.
                self.cartridges[best_id].bbox = det.bbox
                self.cartridges[best_id].confidence = det.confidence
                self.cartridges[best_id].detection_index = det_idx
                matched.add(best_id)
            else:
                # New cartridge — assign a fresh ID.
                new_id = self._next_cartridge_id
                self.cartridges[new_id] = Cartridge(
                    id=new_id, bbox=det.bbox, confidence=det.confidence,
                    detection_index=det_idx,
                )
                matched.add(new_id)
                self._next_cartridge_id += 1

        # Drop cartridges the snapshot didn't observe.
        for cid in list(self.cartridges.keys()):
            if cid not in matched:
                del self.cartridges[cid]

    # ---- queries --------------------------------------------------------

    def available_batteries(self) -> List[Battery]:
        """Batteries that have not yet been assigned to a pose."""
        return [b for b in self.batteries if not b.assigned_to_pose]

    def cartridge(self, cid: int) -> Cartridge:
        return self.cartridges[cid]

    def summary(self) -> Dict[str, int]:
        """A small dict of counts for logging / dashboards."""
        placed = sum(
            c.occupancy.placed_count() if c.occupancy is not None else 0
            for c in self.cartridges.values()
        )
        return {
            "batteries": len(self.batteries),
            "cartridges": len(self.cartridges),
            "placed_cells": placed,
        }


__all__ = [
    "Battery",
    "Cartridge",
    "CellState",
    "EnvironmentModel",
    "OccupancyGrid",
    "PackingFamily",
    "WorkspaceBounds",
]
