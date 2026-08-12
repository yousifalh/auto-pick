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


# -------------------------------------------------------- exceptions --

class PlacementCollision(RuntimeError):
    """A battery was about to be reserved into space already spoken for.

    Raised, never absorbed. The occupancy grid exists so that a second
    planning cycle cannot seat a cell inside one already in the tray; if
    the packer ever proposes such a position anyway, that is a bug in the
    packer or in the marking, and a bug that continues quietly here puts
    an 18650 on top of another 18650 (or on the PCB) while the queue, the
    counters and the robot's status all read normal.
    """


class OutOfWorkspace(RuntimeError):
    """A pose fell outside the robot's declared workspace envelope."""


# ---------------------------------------------------------- reservation --

@dataclass
class Reservation:
    """One queued battery: its real footprint, and the cells it covers.

    ``x_mm``/``y_mm``/``w_mm``/``h_mm`` are the packer's placement in
    strip millimetres **in the orientation it was placed at** — a
    rotated 18.5 x 65 mm cell is 65 x 18.5 mm here
    (:class:`common.packing.PackedItem` already swaps them when
    ``rotated``). The cell block is that footprint quantised OUTWARD:
    floor on the near edge, ceil on the far one. That is the same
    convention :func:`common.packing._overlaps_forbidden` uses to test a
    candidate against the mask, and the two have to agree — a footprint
    marked smaller than it is tested against is exactly how a battery
    gets packed into space another one already occupies.
    """

    row: int
    col: int
    row0: int
    row1: int
    col0: int
    col1: int
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    state: CellState = CellState.PLANNED

    @property
    def cell_count(self) -> int:
        return (self.row1 - self.row0) * (self.col1 - self.col0)

    def overlaps_mm(self, other: "Reservation", tol: float = 1e-6) -> bool:
        """Whether two footprints share physical area, in MILLIMETRES.

        Millimetres, not cells. Cells round outward, so two batteries
        that merely abut share a boundary cell without sharing any
        physical space — which is normal and happens between neighbours
        in a single pack. Only this test says whether two cells would
        actually collide.
        """
        return not (
            self.x_mm + self.w_mm <= other.x_mm + tol
            or other.x_mm + other.w_mm <= self.x_mm + tol
            or self.y_mm + self.h_mm <= other.y_mm + tol
            or other.y_mm + other.h_mm <= self.y_mm + tol
        )


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
    # Millimetres per pixel of the FRAME that produced
    # `placeable_rectangle` and `occupancy`. Both are pixel quantities,
    # so both are meaningless without it - and because cartridges are
    # persistent across frames while the scale is not, the twin must
    # remember which scale its components were measured at rather than
    # letting the planner re-apply whatever the current frame happens to
    # be. Set by plan.planner._ensure_placement_areas at the same moment
    # as the rectangle; None exactly while the rectangle is None.
    mm_per_px: Optional[float] = None
    packing_family: PackingFamily = PackingFamily.GRID
    # Live reservations, keyed by the anchor (row, col) that travels on
    # the PickPlacePose. One entry per queued or placed battery; entries
    # in state PLANNED belong to the CURRENT queue and are released when
    # the next cycle rebuilds it (Planner._release_stale_reservations),
    # entries in state PLACED are physical batteries and persist.
    reservations: Dict[Tuple[int, int], Reservation] = field(
        default_factory=dict)

    def mark_cell(self, row: int, col: int, state: CellState) -> None:
        if self.occupancy is None:
            raise RuntimeError("occupancy not initialised")
        self.occupancy.set(row, col, state)

    # ---- reservations ----------------------------------------------------

    def reserve(self, res: Reservation) -> None:
        """Mark ``res``'s WHOLE cell block PLANNED and remember it.

        The whole block, not the corner cell. An 18.5 x 65 mm battery on
        a 1.5 mm grid covers 13 x 44 cells; marking one of them left the
        other 571 reading FREE, and ``_pack_cartridge`` re-packs from
        this grid every frame, so the next cycle could legally seat a
        cell 1.5 mm from one already in the tray — ~17 mm of physical
        overlap (audit E, finding 1).

        Two interlocks, at two different resolutions, because they are
        two different questions:

        * **FORBIDDEN, in cells.** ``_overlaps_forbidden`` rejects a
          candidate whose outward-rounded block touches any forbidden
          cell, so a placement that reaches here cannot contain one. If
          it does, a battery is about to go on the PCB.
        * **PLANNED / PLACED, in millimetres.** A cell check would be
          wrong here: neighbours in one pack legitimately share a
          boundary cell without sharing any physical space. Only the mm
          footprints answer "would these two collide?".

        Both raise. Neither can fire while the packer is correct, which
        is the point — they are the alarm for the day it is not.
        """
        if self.occupancy is None:
            raise RuntimeError("occupancy not initialised")
        if res.row1 <= res.row0 or res.col1 <= res.col0:
            raise ValueError(
                f"cartridge {self.id}: degenerate reservation {res} — a "
                "footprint that quantises to no cells reserves nothing")

        block = self.occupancy.grid[res.row0:res.row1, res.col0:res.col1]
        if bool((block == CellState.FORBIDDEN.value).any()):
            raise PlacementCollision(
                f"cartridge {self.id}: placement at "
                f"({res.x_mm:.1f}, {res.y_mm:.1f}) mm covers FORBIDDEN "
                "cells — the packer proposed a battery on the PCB or "
                "outside the placeable floor")
        for other in self.reservations.values():
            if res.overlaps_mm(other):
                raise PlacementCollision(
                    f"cartridge {self.id}: placement "
                    f"({res.x_mm:.1f}, {res.y_mm:.1f}, {res.w_mm:.1f} x "
                    f"{res.h_mm:.1f}) mm overlaps a {other.state.name} "
                    f"battery at ({other.x_mm:.1f}, {other.y_mm:.1f}, "
                    f"{other.w_mm:.1f} x {other.h_mm:.1f}) mm")

        self.occupancy.set_block(
            res.row0, res.row1, res.col0, res.col1, CellState.PLANNED)
        self.reservations[(res.row, res.col)] = res
        # The marking must have taken. A guard that silently no-ops is
        # the failure mode this whole change exists to remove.
        if not bool((block == CellState.PLANNED.value).all()):
            raise RuntimeError(
                f"cartridge {self.id}: reserving {res.cell_count} cells "
                "left some of them unmarked")

    def confirm(self, row: int, col: int, success: bool) -> None:
        """Resolve the reservation anchored at ``(row, col)``."""
        res = self.reservations.get((row, col))
        if res is None:
            raise KeyError(
                f"cartridge {self.id}: no reservation at ({row}, {col}) — "
                "the executor reported a cell nobody planned")
        if self.occupancy is None:
            raise RuntimeError("occupancy not initialised")
        if success:
            self.occupancy.set_block(
                res.row0, res.row1, res.col0, res.col1, CellState.PLACED)
            res.state = CellState.PLACED
            return
        # Revert PLANNED -> FREE so a future cycle can retry this cell.
        # Only the cells still reading PLANNED: a boundary cell shared
        # with a neighbour that is already PLACED must not be freed.
        self.occupancy.set_block(
            res.row0, res.row1, res.col0, res.col1, CellState.FREE,
            only_from=CellState.PLANNED)
        del self.reservations[(row, col)]

    def release_planned(self) -> int:
        """Drop every PLANNED reservation; return how many were dropped.

        The queue is rebuilt from scratch every cycle, so a PLANNED
        reservation belongs to the queue that is about to be discarded.
        Left in place they accumulate: ``main`` executes one pose per
        cycle, so after a few frames a cartridge would read full of
        batteries that were never picked, the packer would place nothing
        and the run would report "queue empty, job done" over a nearly
        empty tray. PLACED reservations are physical and survive.
        """
        stale = [k for k, r in self.reservations.items()
                 if r.state is CellState.PLANNED]
        for key in stale:
            res = self.reservations.pop(key)
            if self.occupancy is not None:
                self.occupancy.set_block(
                    res.row0, res.row1, res.col0, res.col1, CellState.FREE,
                    only_from=CellState.PLANNED)
        return len(stale)


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

    def set_block(
        self,
        row0: int,
        row1: int,
        col0: int,
        col1: int,
        state: CellState,
        only_from: Optional[CellState] = None,
    ) -> int:
        """Set the half-open block ``[row0, row1) x [col0, col1)``.

        ``only_from`` restricts the write to cells currently in that
        state, which is how a reservation is reverted without freeing a
        boundary cell that a PLACED neighbour also covers.

        Returns the number of cells written. Slices are taken by
        explicit bounds rather than by numpy's forgiving negative /
        past-the-end indexing: a block that ran off the grid would
        silently write a smaller region than the caller asked for.
        """
        if not (0 <= row0 < row1 <= self.rows
                and 0 <= col0 < col1 <= self.cols):
            raise IndexError(
                f"block [{row0}:{row1}, {col0}:{col1}] is not inside a "
                f"{self.rows} x {self.cols} grid")
        block = self.grid[row0:row1, col0:col1]
        if only_from is None:
            block[:] = state.value
            return int(block.size)
        sel = block == only_from.value
        block[sel] = state.value
        return int(sel.sum())

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
    """Axis-aligned robot-workspace volume in millimetres.

    Declared in ``configs/planning.yaml`` as
    ``camera.workspace_bounds_mm`` and enforced by
    :meth:`plan.planner.Planner._build_pose` on every pose it emits —
    both the pick point and the place target, because both are
    commanded to the arm.

    It used to be parsed, stored and compared against nothing (audit E,
    finding 5): a declared safety envelope that enforced nothing, which
    is worse than no envelope at all because it reads like an interlock.
    """

    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    def __post_init__(self) -> None:
        # An inverted or degenerate envelope rejects every pose, which
        # would read as "the planner is broken" rather than "the bounds
        # are". Say so at construction, where the numbers came from.
        if self.x_min_mm >= self.x_max_mm or self.y_min_mm >= self.y_max_mm:
            raise ValueError(
                f"empty workspace envelope: x [{self.x_min_mm}, "
                f"{self.x_max_mm}] y [{self.y_min_mm}, {self.y_max_mm}]")

    def contains(self, x_mm: float, y_mm: float) -> bool:
        return (self.x_min_mm <= x_mm <= self.x_max_mm
                and self.y_min_mm <= y_mm <= self.y_max_mm)

    def require(self, x_mm: float, y_mm: float, what: str) -> None:
        """Raise :class:`OutOfWorkspace` unless ``(x, y)`` is inside.

        Raise, not clamp. A place target is where a cartridge slot
        physically is; moving it to the envelope's edge does not make it
        reachable, it makes it wrong — the cell gets inserted into a
        wall and the twin records the slot as PLACED. A pick point is
        where a battery physically lies; clamping it grasps empty table.
        In both directions clamping turns a planning or calibration bug
        into a slightly-wrong motion that nothing downstream can tell
        from a correct one, which is this project's characteristic
        failure. An unreachable pose is a configuration error about the
        whole run (mm_per_px, origin_offset, or these bounds), so it
        reads as one — the same call this module already makes for
        :class:`plan.placement_area.UnknownScale`.
        """
        if self.contains(x_mm, y_mm):
            return
        raise OutOfWorkspace(
            f"{what} at ({x_mm:.1f}, {y_mm:.1f}) mm is outside the robot "
            f"workspace x [{self.x_min_mm}, {self.x_max_mm}] "
            f"y [{self.y_min_mm}, {self.y_max_mm}] mm")


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
        """A small dict of counts for logging / dashboards.

        ``placed_cells`` counts GRID cells, not batteries: one 18650
        covers 13 x 44 of them at 1.5 mm. It read like a battery count
        back when a placed battery marked exactly one cell, which it no
        longer does. Divide by the footprint, or count
        ``Cartridge.reservations`` in state PLACED, to get batteries.
        """
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
    "OutOfWorkspace",
    "PackingFamily",
    "PlacementCollision",
    "Reservation",
    "WorkspaceBounds",
]
