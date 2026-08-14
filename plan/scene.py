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

Persistent means TRACKED, not "present in the last frame". A cartridge
the camera does not see this frame becomes *not currently visible* and
keeps its occupancy grid and its reservations; it is re-matched when it
comes back. It used to be deleted outright, and the delete took the
whole physical record with it: a cartridge holding 572 PLACED cells
that missed ONE frame returned as a fresh empty ID whose first place
target was byte-identical to one already executed, with the millimetre
interlock comparing against reservations that had been deleted
alongside it (audit H, finding 1). One low-confidence frame, one
gripper arm across the box, and the arm drives an 18650 into a slot
that already holds one while every counter reads normal.

Three rules follow from that, and they are what this module now
enforces:

* **A track is memory; a detection is evidence.** Only a cartridge seen
  THIS frame may be planned into (``Cartridge.visible``) — the planner
  skips the rest. Memory is enough to refuse a slot; it is not enough
  to command the arm into one.
* **Memory is bounded and its expiry is counted.** A track missing for
  more than ``TrackingConfig.max_missing_frames`` is forgotten, and the
  PLACED cells forgotten with it are counted
  (``expired_placed_cell_count``). Retaining tracks for ever would grow
  without bound on a moving corpus; forgetting them silently is the
  defect this module exists to close.
* **Nothing is discarded silently.** Every drop, expiry, duplicate
  suppression and geometry invalidation increments a counter that
  reaches ``main``'s run statistics and the receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.logging import get_logger
from common.types import BBox, ClassLabel, Snapshot

log = get_logger("autopick.scene")


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
    # The SAME footprint's near corner in ROBOT-WORKSPACE millimetres.
    # `x_mm`/`y_mm` are cartridge-local (strip) coordinates, so two
    # reservations belonging to two different twin entries are not
    # comparable in them - which is exactly how a segmenter that split
    # one physical cartridge into two boxes produced eight pairs of
    # physically overlapping place targets with no interlock able to see
    # across them (audit H, finding 3). These two are the frame every
    # cartridge in the scene shares, and they are REQUIRED, not
    # defaulted: a cross-cartridge check that silently skips a
    # reservation carrying None is not a check.
    wx_mm: float
    wy_mm: float
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

    def overlaps_workspace_mm(
        self, other: "Reservation", tol: float = 1e-6,
    ) -> bool:
        """:meth:`overlaps_mm`, asked in the frame the ARM works in.

        The only question that means anything BETWEEN two cartridges:
        strip millimetres are measured from each cartridge's own
        placeable rectangle, so comparing them across entries compares
        two different origins and answers nothing.
        """
        return not (
            self.wx_mm + self.w_mm <= other.wx_mm + tol
            or other.wx_mm + other.w_mm <= self.wx_mm + tol
            or self.wy_mm + self.h_mm <= other.wy_mm + tol
            or other.wy_mm + other.h_mm <= self.wy_mm + tol
        )


# ------------------------------------------------------ tracking config --

@dataclass
class TrackingConfig:
    """How the twin decides that two frames show the same cartridge.

    Every number here used to be a hardcoded default that no caller and
    no config ever set (audit H, finding 4): ``update_from_snapshot``
    took ``iou_match_threshold: float = 0.5`` and ``Planner.cycle``
    called it bare. A threshold that governs whether the robot's memory
    of what is already in a tray survives the next frame is not a
    signature default; it is a machine parameter, and it is declared in
    ``configs/planning.yaml`` under ``tracking:``.
    """

    # IoU at or above which a detection IS an existing track. Matching
    # is global-best rather than first-come (see
    # EnvironmentModel._match_or_insert_cartridges), so this is a floor
    # on acceptance, not a tie-break.
    iou_match_threshold: float = 0.5
    # How many consecutive frames a track may go unseen before the twin
    # forgets it. Memory has to be bounded - a run over a moving corpus
    # would otherwise accumulate one entry per cartridge ever seen, each
    # holding a grid - but the bound is what turns "the camera blinked"
    # into "the tray is empty", so expiry is counted and logged.
    # 5 frames at ~2 Hz is a couple of seconds of occlusion; a cartridge
    # genuinely removed from the cell stays gone for far longer.
    max_missing_frames: int = 5
    # A detection that matched nothing but overlaps a cartridge ALREADY
    # OBSERVED this frame by at least this much is a second view of that
    # cartridge - a split mask, or a duplicate that survived NMS - not a
    # second cartridge. Two twin entries over one physical object each
    # get their own grid and their own reservations, and the millimetre
    # interlock is per-entry, so they double-book the same space with
    # nothing able to see it (audit H, finding 3). Deliberately BELOW
    # `iou_match_threshold`: the band between the two is precisely where
    # "not confident enough to be the same track" and "far too close to
    # be a different object" overlap, and the safe answer there is to
    # trust the entry that already holds the physical record.
    duplicate_iou_threshold: float = 0.3
    # How far a matched cartridge's box may move before its measured
    # geometry is thrown away and re-extracted. The placement rectangle
    # and occupancy grid are in PIXELS of the frame that measured them;
    # a matched cartridge used to keep them for ever, so a box that
    # moved 38 mm kept commanding the millimetres it had before it moved
    # (audit H, finding 2). One occupancy cell (1.5 mm at the default
    # resolution) is the finest distinction the grid can represent, so
    # it is the point past which stale geometry is a lie the grid could
    # have told the truth about.
    geometry_refresh_mm: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_match_threshold <= 1.0:
            raise ValueError(
                "tracking.iou_match_threshold must be in (0, 1]; got "
                f"{self.iou_match_threshold}")
        if not 0.0 < self.duplicate_iou_threshold <= self.iou_match_threshold:
            raise ValueError(
                "tracking.duplicate_iou_threshold must be in (0, "
                f"iou_match_threshold]; got {self.duplicate_iou_threshold} "
                f"against {self.iou_match_threshold}. Above the match "
                "threshold it would suppress detections that should have "
                "been matched, which loses the observation instead of "
                "merging it.")
        if self.max_missing_frames < 0:
            raise ValueError(
                "tracking.max_missing_frames must be >= 0; got "
                f"{self.max_missing_frames} (0 means forget immediately, "
                "which is the pre-fix behaviour and drops the occupancy "
                "grid on a single missed frame)")
        if self.geometry_refresh_mm <= 0.0:
            raise ValueError(
                "tracking.geometry_refresh_mm must be > 0; got "
                f"{self.geometry_refresh_mm}")

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "TrackingConfig":
        """Build from ``planning.yaml``'s ``tracking:`` block.

        Unknown keys are refused by name rather than ignored. A
        misspelled ``max_missing_frame`` that silently kept the default
        is the same trap as a threshold nobody sets.
        """
        cfg = cfg or {}
        known = {f for f in (
            "iou_match_threshold", "max_missing_frames",
            "duplicate_iou_threshold", "geometry_refresh_mm")}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"planning.tracking has unknown keys {sorted(unknown)}; "
                f"known keys are {sorted(known)}")
        defaults = cls()
        return cls(
            iou_match_threshold=float(cfg.get(
                "iou_match_threshold", defaults.iou_match_threshold)),
            max_missing_frames=int(cfg.get(
                "max_missing_frames", defaults.max_missing_frames)),
            duplicate_iou_threshold=float(cfg.get(
                "duplicate_iou_threshold", defaults.duplicate_iou_threshold)),
            geometry_refresh_mm=float(cfg.get(
                "geometry_refresh_mm", defaults.geometry_refresh_mm)),
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
    # Consecutive frames this cartridge has NOT been detected in. 0 means
    # "seen in the snapshot currently being planned against"; anything
    # above 0 means the twin is running on memory. The planner refuses to
    # plan into a cartridge it cannot see this frame, so this is not a
    # cosmetic field: it is the difference between remembering that a
    # slot is full and commanding the arm into a box nobody is looking
    # at. EnvironmentModel forgets a track once this passes
    # TrackingConfig.max_missing_frames.
    frames_since_seen: int = 0

    @property
    def visible(self) -> bool:
        """Whether THIS frame's snapshot contained this cartridge."""
        return self.frames_since_seen == 0

    def invalidate_geometry(self) -> None:
        """Forget the measured geometry; KEEP the physical record.

        Called when the cartridge's box has moved far enough that the
        rectangle and grid measured at its old position no longer
        describe it. The two halves are deliberately different:

        * ``placeable_rectangle`` / ``occupancy`` / ``pcb_mask`` /
          ``mm_per_px`` are MEASUREMENTS of a frame, and a measurement of
          where the box used to be is worse than none - it produces
          place targets frozen at last frame's millimetres, which is
          audit H finding 2 exactly.
        * ``reservations`` are BATTERIES. A cell that was physically
          placed is still physically placed after the box slides 40 mm;
          it slides with the box. They are re-projected onto the fresh
          grid by ``Planner._reproject_placed_reservations`` as soon as
          one exists, and until then they still answer the millimetre
          interlock.
        """
        self.placeable_rectangle = None
        self.occupancy = None
        self.pcb_mask = None
        self.mm_per_px = None

    def placed_reservations(self) -> List[Reservation]:
        """The reservations that correspond to a physical battery."""
        return [r for r in self.reservations.values()
                if r.state is CellState.PLACED]

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

        # `only_from=FREE`, because PLACED is not writable. Without it a
        # block abutting a placed battery took its shared boundary cells
        # PLACED -> PLANNED, and the next `release_planned` or
        # `confirm(False)` then took them PLANNED -> FREE: 44 cells of a
        # battery that is physically in the tray reading FREE, and the
        # packer free to seat another one into them (audit H, finding
        # 5). It was latent - `_pack_cartridge` masks PLACED and
        # `_overlaps_forbidden` rounds outward, so today's packer cannot
        # propose such a block - but it is the invariant this method's
        # whole docstring is about, and the guard costs one keyword.
        #
        # Cells left PLANNED (a neighbour in the same pack) or PLACED (a
        # legitimately abutting battery, already proved non-overlapping
        # in millimetres above) stay as they are. Only FREE becomes
        # PLANNED.
        self.occupancy.set_block(
            res.row0, res.row1, res.col0, res.col1, CellState.PLANNED,
            only_from=CellState.FREE)
        self.reservations[(res.row, res.col)] = res
        # The marking must have taken. A guard that silently no-ops is
        # the failure mode this whole change exists to remove. The
        # assertion is "no cell of this footprint is still FREE" rather
        # than "every cell is PLANNED", because under `only_from` an
        # abutting neighbour's PLACED/PLANNED boundary cell legitimately
        # stays as it was - while a cell left FREE means part of a
        # battery is unmarked and the next cycle may pack into it.
        if bool((block == CellState.FREE.value).any()):
            raise RuntimeError(
                f"cartridge {self.id}: reserving {res.cell_count} cells "
                f"left {int((block == CellState.FREE.value).sum())} of them "
                "reading FREE")

    def confirm(self, row: int, col: int, success: bool) -> None:
        """Resolve the reservation anchored at ``(row, col)``."""
        res = self.reservations.get((row, col))
        if res is None:
            raise KeyError(
                f"cartridge {self.id}: no reservation at ({row}, {col}) — "
                "the executor reported a cell nobody planned")
        if self.occupancy is None:
            raise RuntimeError("occupancy not initialised")
        if res.state is CellState.PLACED:
            # A second report about a battery already recorded as
            # physical. `confirm(..., False)` would delete the PLACED
            # Reservation while leaving its cells PLACED, which loses
            # that battery's footprint from the millimetre interlock
            # while the grid still shows it (audit H, finding 5) - a
            # half-forgotten battery is worse than either state.
            raise PlacementCollision(
                f"cartridge {self.id}: reservation at ({row}, {col}) is "
                "already PLACED; the executor reported twice on one "
                "battery")
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

    **This is an axis-aligned rectangle, and the arm's envelope is not.**
    The shipped ±350 mm square is a conservative placeholder inscribed in
    the KR 6 R700 sixx's 706.7 mm reach; it does not model the inner dead
    zone around the base, the ~20° notch behind the robot that A1's ±170°
    travel leaves, or the fact that usable radius shrinks with Z. The
    three limitations are spelled out at the value itself, in
    ``configs/planning.yaml`` under ``camera.workspace_bounds_mm``.
    Everything inside the square is within reach *on radius*; not
    everything inside it is reachable.
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
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    batteries: List[Battery] = field(default_factory=list)
    cartridges: Dict[int, Cartridge] = field(default_factory=dict)
    _next_battery_id: int = field(default=0, init=False, repr=False)
    _next_cartridge_id: int = field(default=0, init=False, repr=False)

    # ---- observability ---------------------------------------------------
    #
    # Every one of these counts something the twin used to do silently.
    # They are summed over the whole run and reach `main`'s statistics
    # and the receipt; a dropout that quietly lost 572 PLACED cells is
    # the defect this module was rebuilt around, so "retained", "expired"
    # and "refused" are all numbers rather than absences.

    # Tracks that stopped being detected (one per dropout event, not per
    # missing frame). Non-zero is ORDINARY on a moving corpus.
    track_dropout_count: int = field(default=0, init=False)
    # Missing tracks that were matched again before they expired. This is
    # the number that says the memory did its job.
    track_reacquired_count: int = field(default=0, init=False)
    # Tracks forgotten after `max_missing_frames`, and the PLACED cells /
    # placed batteries forgotten with them. The last two are the audit's
    # own currency: "572 placed cells" going missing must be a printed
    # number, never an absence.
    track_expired_count: int = field(default=0, init=False)
    expired_placed_cell_count: int = field(default=0, init=False)
    expired_placed_battery_count: int = field(default=0, init=False)
    # Detections refused as a second view of a cartridge already observed
    # this frame (split mask, duplicate box). Each one is a twin entry
    # that would otherwise have double-booked physical space.
    duplicate_detection_count: int = field(default=0, init=False)
    # Matched tracks whose box moved further than
    # `tracking.geometry_refresh_mm`, so their measured rectangle and
    # grid were thrown away to be re-extracted at the new position.
    geometry_refresh_count: int = field(default=0, init=False)

    # ---- update semantics -----------------------------------------------

    def update_from_snapshot(self, snapshot: Snapshot) -> None:
        """Fuse ``snapshot`` into the twin.

        * Batteries are replaced wholesale (ephemeral entities).
        * Cartridges are TRACKED: matched to existing IDs by IoU so
          persistent components (occupancy, PCB mask, reservations)
          survive across frames, including across frames the cartridge
          was not detected in.

        The IoU threshold is no longer an argument with a default nobody
        set; it lives in :class:`TrackingConfig` on this model, which
        ``configs/planning.yaml`` configures.

        **Cost is O(D·C) — quadratic in cartridges**, because
        `_match_or_insert_cartridges` scores every (detection, track)
        pair with no spatial index and no early exit. Nothing above said
        so, and cartridge persistence is advertised as a feature two
        docstrings up. Measured (audit K §3, min of 5, n cartridges *and*
        n batteries): 8 → 0.018 ms, 32 → 0.172, 128 → 2.814, 256 →
        10.387 — a clean 4×-per-doubling. This is a WHOLE-FRAME cost
        charged against the PPR's 50 ms budget, which it crosses at
        roughly **500–560 cartridges per frame**; the real corpus maximum
        is **4** (`recog/dataset3d_seg`, 502 frames), so the margin is
        ~500× and the shape is deliberately left alone. The global
        best-first ordering below is what makes it quadratic and is also
        the whole point of it (audit H, finding 4) — do not trade that
        away for a spatial index unless a scene ever gets big enough to
        need one.
        """
        self._replace_batteries(snapshot)
        self._match_or_insert_cartridges(snapshot)

    def _replace_batteries(self, snapshot: Snapshot) -> None:
        self.batteries = []
        for det in snapshot.of(ClassLabel.BATTERY):
            self.batteries.append(Battery(
                id=self._next_battery_id,
                bbox=det.bbox,
                confidence=det.confidence,
            ))
            self._next_battery_id += 1

    def _match_or_insert_cartridges(self, snapshot: Snapshot) -> None:
        """Re-identify this frame's cartridge detections against the twin.

        Four steps, in this order and for these reasons:

        1. **Score every (track, detection) pair, then take them
           best-first.** The old loop walked detections in SNAPSHOT ORDER
           and gave each one the best still-unmatched track — so a
           detection scoring 0.667 against two tracks took the first by
           dict insertion order, and the detection that scored **1.000**
           against that same track arrived to find it taken, fell below
           threshold against the other, and was inserted as a new empty
           ID (audit H, finding 4). One frame, and the twin's record of
           which cartridge holds what is transposed.

           Greedy over globally sorted pairs, NOT a Hungarian assignment,
           and the difference is deliberate: Hungarian maximises the SUM
           of the accepted IoUs and will trade the best individual pair
           away to do it, which is the exact failure being fixed. This
           order guarantees the highest-scoring pair in the frame is
           always honoured, then the next, and so on. Ties break on
           ``(track id, detection index)`` so the same frame always
           produces the same assignment.

        2. **Apply the matches**, updating the box and — if it moved far
           enough — dropping the geometry measured at the old one.

        3. **Insert the leftovers**, unless a leftover overlaps a
           cartridge already observed THIS frame, in which case it is a
           second view of that cartridge rather than a second cartridge.

        4. **Age everything unseen.** Not delete: age. This is the
           finding-1 fix, and the whole reason the method is shaped this
           way.
        """
        cfg = self.tracking
        # Track each detection's index into the FULL snapshot.detections
        # list (not its position within this cartridge-only subset): that
        # is the convention recog.inference.attach_cartridge_masks uses to
        # key Snapshot.cartridge_masks, and the two must agree or every
        # mask lookup in _ensure_placement_areas misaligns.
        new_dets = [
            (i, det) for i, det in enumerate(snapshot.detections)
            if det.label is ClassLabel.CARTRIDGE
        ]

        # 1. Global best-first assignment.
        pairs = [
            (det.bbox.iou(ctg.bbox), cid, det_idx)
            for cid, ctg in self.cartridges.items()
            for det_idx, det in new_dets
        ]
        pairs = [p for p in pairs if p[0] >= cfg.iou_match_threshold]
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        matched: Dict[int, int] = {}          # track id -> detection index
        claimed: Dict[int, int] = {}          # detection index -> track id
        for iou, cid, det_idx in pairs:
            if cid in matched or det_idx in claimed:
                continue
            matched[cid] = det_idx
            claimed[det_idx] = cid

        # 2. Apply the matches.
        det_by_index = {i: d for i, d in new_dets}
        for cid, det_idx in matched.items():
            self._observe(self.cartridges[cid], det_by_index[det_idx], det_idx)

        # 3. Insert what is left, minus duplicate views.
        observed: List[int] = list(matched.keys())
        for det_idx, det in new_dets:
            if det_idx in claimed:
                continue
            twin = self._already_observed_as(det.bbox, observed)
            if twin is not None:
                self.duplicate_detection_count += 1
                log.warning(
                    "cartridge detection %d overlaps twin cartridge %d at "
                    "IoU %.3f but did not match it: treating it as a second "
                    "view of one physical cartridge, not a second cartridge "
                    "(a second twin entry would get its own occupancy grid "
                    "and its own reservations, and the millimetre interlock "
                    "cannot see across two entries)",
                    det_idx, twin, det.bbox.iou(self.cartridges[twin].bbox))
                continue
            new_id = self._next_cartridge_id
            self.cartridges[new_id] = Cartridge(
                id=new_id, bbox=det.bbox, confidence=det.confidence,
                detection_index=det_idx,
            )
            observed.append(new_id)
            self._next_cartridge_id += 1

        # 4. Age — never delete on sight — the cartridges this frame did
        #    not contain.
        self._age_unobserved(set(observed))

    def _observe(self, ctg: Cartridge, det, det_idx: int) -> None:
        """Fold one matched detection into an existing track."""
        moved_mm = self._box_shift_mm(ctg, det.bbox)
        if ctg.frames_since_seen > 0:
            self.track_reacquired_count += 1
            log.info(
                "cartridge %d re-acquired after %d frame(s) unseen, holding "
                "%d placed cells", ctg.id, ctg.frames_since_seen,
                ctg.occupancy.placed_count() if ctg.occupancy else 0)
        ctg.bbox = det.bbox
        ctg.confidence = det.confidence
        ctg.detection_index = det_idx
        ctg.frames_since_seen = 0

        refresh_mm = self.tracking.geometry_refresh_mm
        if moved_mm is not None and moved_mm > refresh_mm:
            # The box moved under its own geometry. Keep the identity and
            # the batteries; throw away the measurement. Re-measuring is
            # the SAFE direction of the two available: if this really is
            # the same cartridge, its grid slides with it and the record
            # stays true; if a different cartridge has been swapped into
            # the fixture, the twin reads it as full and places nothing,
            # which stalls rather than collides. Rejecting the match
            # instead (fresh ID, empty grid) is the unsafe direction - a
            # cartridge that merely slid would be planned into over
            # batteries it already holds.
            self.geometry_refresh_count += 1
            log.info(
                "cartridge %d moved %.1f mm (> %.1f mm): re-measuring its "
                "placement geometry, keeping %d placed batteries",
                ctg.id, moved_mm, refresh_mm,
                len(ctg.placed_reservations()))
            ctg.invalidate_geometry()

    def _box_shift_mm(self, ctg: Cartridge, new: BBox) -> Optional[float]:
        """How far ``ctg``'s box has moved, in millimetres, or ``None``.

        ``None`` when the cartridge carries no measured geometry — there
        is then nothing that could be stale, and no scale to convert
        pixels with. Corner displacement rather than centre
        displacement: a box that grows on one side only has not moved its
        centre far, but the rectangle extracted inside it has moved.
        """
        if ctg.placeable_rectangle is None or ctg.mm_per_px is None:
            return None
        old = ctg.bbox
        return ctg.mm_per_px * max(
            abs(new.xmin - old.xmin), abs(new.ymin - old.ymin),
            abs(new.xmax - old.xmax), abs(new.ymax - old.ymax),
        )

    def _already_observed_as(
        self, box: BBox, observed: List[int],
    ) -> Optional[int]:
        """The twin ID ``box`` is a duplicate view of, if any.

        Only cartridges observed in THIS frame are candidates. A stale
        remembered box is not evidence that a new detection is a
        duplicate of it — suppressing against memory would let a track
        that has expired-but-not-yet-been-forgotten swallow a real new
        cartridge.
        """
        best_id, best_iou = None, self.tracking.duplicate_iou_threshold
        for cid in observed:
            iou = box.iou(self.cartridges[cid].bbox)
            if iou >= best_iou:
                best_id, best_iou = cid, iou
        return best_id

    def _age_unobserved(self, observed: set) -> None:
        """Carry unseen tracks forward; forget the ones past their bound.

        The line this replaced was ``del self.cartridges[cid]`` for
        anything the frame did not contain. The delete took the whole
        ``Cartridge``, so it took the occupancy grid, every PLACED
        reservation (the only input to the millimetre interlock), the
        placement rectangle, the scale and the ID: a cartridge five
        batteries deep lost all five to a single low-confidence frame,
        came back as a new empty ID, and the next queue aimed at slots
        that were already full while every counter read normal.
        """
        for cid in list(self.cartridges.keys()):
            if cid in observed:
                continue
            ctg = self.cartridges[cid]
            if ctg.frames_since_seen == 0:
                self.track_dropout_count += 1
            ctg.frames_since_seen += 1
            # It is not in this frame, so it indexes nothing in this
            # frame's cartridge_masks. -1 is the documented "no detection
            # this frame" value and matches nothing.
            ctg.detection_index = -1
            if ctg.frames_since_seen <= self.tracking.max_missing_frames:
                continue
            self._expire(cid)

    def _expire(self, cid: int) -> None:
        """Forget a track, counting exactly what is being forgotten."""
        ctg = self.cartridges.pop(cid)
        cells = ctg.occupancy.placed_count() if ctg.occupancy else 0
        batteries = len(ctg.placed_reservations())
        self.track_expired_count += 1
        self.expired_placed_cell_count += cells
        self.expired_placed_battery_count += batteries
        # WARNING, not debug: this is the twin forgetting physical state.
        # It is a legitimate and necessary bound on memory, and it is
        # also the exact moment at which a returning cartridge stops
        # being protected from being planned into twice.
        log.warning(
            "cartridge %d expired after %d frames unseen: forgetting %d "
            "placed cells (%d batteries). A cartridge that returns after "
            "this is a NEW, EMPTY entry to the twin",
            cid, ctg.frames_since_seen, cells, batteries)

    # ---- cross-cartridge interlock --------------------------------------

    def conflicting_reservation(
        self, res: Reservation, exclude_cartridge_id: int,
    ) -> Optional[Tuple[int, Reservation]]:
        """A footprint in ANOTHER visible cartridge that ``res`` overlaps.

        ``Cartridge.reserve``'s millimetre interlock is per-cartridge and
        cannot see across two twin entries, which is why one physical
        cartridge detected twice produced eight pairs of overlapping
        place targets with nothing raising (audit H, finding 3). This is
        the same question asked in the workspace frame, where every
        cartridge's footprints are comparable.

        Only cartridges VISIBLE this frame are compared. A remembered
        cartridge is memory, and a live detection standing where memory
        says a battery is means the memory is stale, not that the space
        is occupied — declining the slot on that basis would let one
        expired scene block the next one.
        """
        for cid, ctg in self.cartridges.items():
            if cid == exclude_cartridge_id or not ctg.visible:
                continue
            for other in ctg.reservations.values():
                if res.overlaps_workspace_mm(other):
                    return cid, other
        return None

    # ---- queries --------------------------------------------------------

    def available_batteries(self) -> List[Battery]:
        """Batteries that have not yet been assigned to a pose."""
        return [b for b in self.batteries if not b.assigned_to_pose]

    def cartridge(self, cid: int) -> Cartridge:
        return self.cartridges[cid]

    def visible_cartridges(self) -> List[Cartridge]:
        """The cartridges the CURRENT frame actually contained.

        The only ones anything may plan into. A remembered cartridge is
        enough to refuse a slot; commanding the arm into a box that is
        not in this frame's snapshot is planning against a photograph.
        """
        return [c for c in self.cartridges.values() if c.visible]

    def tracking_stats(self) -> Dict[str, int]:
        """Everything the identity layer retained, refused or forgot."""
        return {
            "track_dropouts": self.track_dropout_count,
            "tracks_reacquired": self.track_reacquired_count,
            "tracks_expired": self.track_expired_count,
            "expired_placed_cells": self.expired_placed_cell_count,
            "expired_placed_batteries": self.expired_placed_battery_count,
            "duplicate_detections": self.duplicate_detection_count,
            "geometry_refreshes": self.geometry_refresh_count,
        }

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
            # Tracked, which is not the same as seen: a cartridge missing
            # from this frame is still tracked, still holds its placed
            # cells, and is still counted here. `visible_cartridges` is
            # the subset this frame observed and the only subset the
            # planner will plan into.
            "cartridges": len(self.cartridges),
            "visible_cartridges": len(self.visible_cartridges()),
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
    "TrackingConfig",
    "WorkspaceBounds",
]
