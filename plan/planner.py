"""Top-level planning orchestrator.

Takes a fresh :class:`common.types.Snapshot`, updates the digital
twin, extracts the valid placement area for any cartridge that
doesn't yet have one, runs FFDH on each cartridge, and emits a
deterministic FIFO queue of :class:`common.types.PickPlacePose`
items for the executor.

Cycle-time rules (PPR §5.3):

* Row-major fill order (top-left → bottom-right) inside each cartridge.
* Nearest-available-battery assignment to minimise transport distance.
* Cells marked ``FORBIDDEN``, ``PLACED`` or ``PLANNED`` are skipped.
* The queue never contains more picks than available batteries.

After the executor reports back, :meth:`Planner.confirm_placement`
flips each PLANNED cell to PLACED on success, or reverts it to FREE
on failure so a future cycle can retry.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.types import PickPlacePose, Snapshot, WorkspacePoint
from plan.bin_packing import Item, PackResult, pack_best_effort
from plan.placement_area import (
    BadDetectorBox,
    PlacementAreaExtractor,
    PlacementDisagreement,
)
from plan.scene import (
    Battery,
    Cartridge,
    CellState,
    EnvironmentModel,
    WorkspaceBounds,
)


# ------------------------------------------------------ configuration --

@dataclass
class PlannerConfig:
    """Knobs loaded from ``configs/planning.yaml``."""

    battery_width_mm: float = 18.5
    battery_length_mm: float = 65.0
    mm_per_px: float = 0.38
    origin_offset_x_mm: float = 0.0
    origin_offset_y_mm: float = 0.0
    pick_approach_height_mm: float = 60.0
    place_insert_height_mm: float = 2.0
    allow_rotation: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict) -> "PlannerConfig":
        battery = cfg.get("battery", {}) or {}
        camera = cfg.get("camera", {}) or {}
        motion = cfg.get("motion", {}) or {}
        return cls(
            battery_width_mm=float(battery.get("diameter_mm", 18.5)),
            battery_length_mm=float(battery.get("length_mm", 65.0)),
            mm_per_px=float(camera.get("mm_per_px_x", 0.38)),
            origin_offset_x_mm=float(camera.get("origin_offset_x_mm", 0.0)),
            origin_offset_y_mm=float(camera.get("origin_offset_y_mm", 0.0)),
            pick_approach_height_mm=float(
                motion.get("approach_height_mm", 60.0),
            ),
            place_insert_height_mm=float(motion.get("insert_height_mm", 2.0)),
            allow_rotation=True,
        )


def _accepts_label_map(extractor) -> bool:
    """Whether ``extractor.extract`` will accept a ``label_map`` kwarg.

    Checked ONCE, at construction time, against the extractor's actual
    declared signature (an explicit ``label_map`` parameter, or
    ``**kwargs``) - not inferred from whether a snapshot happens to carry
    ``cartridge_masks`` that cycle. The two are different questions:
    ``HeuristicPlacementAreaExtractor.extract`` takes no ``**kwargs``, so
    if ``cartridge_masks`` is ever non-empty while it is the selected
    extractor, passing ``label_map`` through raises ``TypeError`` -
    which ``_ensure_placement_areas``'s blanket ``except Exception``
    swallows silently, unplanning every cartridge forever with no
    counter incrementing. Gating on the signature, rather than on the
    dict being empty, is what prevents that (final whole-branch review,
    D-integration-arbitration).
    """
    try:
        params = inspect.signature(extractor.extract).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == "label_map" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in params)


# ---------------------------------------------------------- planner ---

class Planner:
    """Stateful planner that owns the digital twin."""

    def __init__(
        self,
        planner_cfg: PlannerConfig,
        placement_extractor: PlacementAreaExtractor,
        workspace: WorkspaceBounds,
    ) -> None:
        self.cfg = planner_cfg
        self.extractor = placement_extractor
        self.env = EnvironmentModel(workspace=workspace)
        # Computed once here rather than per-cartridge-per-cycle in
        # _ensure_placement_areas - the extractor's signature can't
        # change mid-run. See _accepts_label_map's docstring.
        self._extractor_accepts_label_map = _accepts_label_map(placement_extractor)
        # Observability counters for _ensure_placement_areas. A blanket
        # `except Exception: continue` already existed and already meant
        # "skip this cartridge, retry next frame" - that behaviour is
        # unchanged. What changes is that PlacementDisagreement (the
        # estimates disagree about a real cartridge) and its more severe
        # subclass BadDetectorBox (a misaligned detector box - a
        # perception failure) no longer fire invisibly: a safety
        # interlock that fires invisibly is not a safety interlock, and
        # if BadDetectorBox climbs relative to PlacementDisagreement it
        # means the detector is drifting, not that cartridges are
        # suddenly all full.
        self.placement_disagreement_count = 0
        self.bad_detector_box_count = 0

    # ---- main cycle -----------------------------------------------------

    def cycle(
        self,
        snapshot: Snapshot,
        image_rgb: Optional[np.ndarray],
    ) -> List[PickPlacePose]:
        """Run one planning cycle and return a freshly built queue."""
        self.env.update_from_snapshot(snapshot)

        if image_rgb is not None:
            self._ensure_placement_areas(image_rgb, snapshot)

        queue: List[PickPlacePose] = []
        available: List[Battery] = list(self.env.available_batteries())

        for ctg in sorted(self.env.cartridges.values(), key=lambda c: c.id):
            if ctg.placeable_rectangle is None or ctg.occupancy is None:
                continue

            pack = self._pack_cartridge(ctg)

            # Row-major fill: sort packed placements by (y, x).
            for p in sorted(pack.placements, key=lambda x: (x.y, x.x)):
                if not available:
                    return queue

                queue.append(self._build_pose(ctg, p, available))

        return queue

    # ---- placement-area extraction --------------------------------------

    def _ensure_placement_areas(
        self, image_rgb: np.ndarray, snapshot: Optional[Snapshot] = None,
    ) -> None:
        """Fill in the placement rectangle / occupancy for new cartridges.

        When ``snapshot`` carries a label map for a cartridge
        (``Snapshot.cartridge_masks``, keyed by ``detection_index``) AND
        ``self.extractor`` declares a ``label_map`` parameter (checked
        once at construction time, see ``_accepts_label_map``), it is
        passed through. An extractor that doesn't declare one - the
        heuristic path - never receives the kwarg, REGARDLESS of whether
        ``cartridge_masks`` is populated: passing it anyway would raise
        ``TypeError`` (the heuristic's ``extract`` takes no ``**kwargs``),
        which the blanket ``except Exception`` below would swallow
        silently, leaving every cartridge permanently unplanned and
        uncounted. This is a capability check on the extractor, not a
        statement that the dict happens to be empty.
        """
        for ctg in self.env.cartridges.values():
            if ctg.placeable_rectangle is not None:
                continue
            try:
                kwargs = {}
                if (self._extractor_accepts_label_map and snapshot is not None
                        and ctg.detection_index in snapshot.cartridge_masks):
                    kwargs["label_map"] = \
                        snapshot.cartridge_masks[ctg.detection_index]
                pa = self.extractor.extract(image_rgb, ctg.bbox, **kwargs)
            except BadDetectorBox:
                # A misaligned detector box - a PERCEPTION failure, not
                # an empty cartridge. Counted separately from ordinary
                # disagreements so a systematically drifting detector
                # is visible as itself rather than reading as "every
                # cartridge is full". Still skip-and-retry-next-frame:
                # the same exception may well be transient.
                self.bad_detector_box_count += 1
                continue
            except PlacementDisagreement:
                # The extractor judged this a real cartridge it is not
                # safe to plan on. Rate is observable via the counter -
                # a safety interlock that fires invisibly is not a
                # safety interlock.
                #
                # In-tree this branch is currently unreachable: the tau
                # gate that used to raise the bare class is retired
                # (FDR v3 section 13.2.1) and BadDetectorBox, its only
                # remaining subclass, is caught above. Kept because the
                # base class is the documented extension point for an
                # extractor that has a real reason to refuse - deleting
                # it would move that decision into the blanket
                # `except Exception` below, where it stops being
                # counted at all. The counter reading 0 forever is
                # therefore the honest number, not a broken one.
                self.placement_disagreement_count += 1
                continue
            except Exception:
                # Leave unplanned this cycle — it'll retry next frame.
                # This also catches the ordinary "no placeable area"
                # RuntimeError a genuinely full cartridge raises - that
                # is a normal, expected state and is deliberately NOT
                # counted alongside the two cases above.
                continue
            ctg.placeable_rectangle = pa.rectangle
            ctg.occupancy = pa.occupancy
            ctg.pcb_mask = pa.pcb_mask

    # ---- FFDH wrapper ---------------------------------------------------

    def _pack_cartridge(self, ctg: Cartridge) -> PackResult:
        pr = ctg.placeable_rectangle
        strip_w_mm = pr.width * self.cfg.mm_per_px
        strip_h_mm = pr.height * self.cfg.mm_per_px

        # Estimate an upper bound on how many items might fit, with 2x
        # slack so FFDH has enough candidates to saturate the strip.
        n_est = max(
            4,
            int((strip_w_mm * strip_h_mm)
                / (self.cfg.battery_width_mm * self.cfg.battery_length_mm))
            * 2,
        )
        items = [
            Item(
                id=i,
                width=self.cfg.battery_width_mm,
                height=self.cfg.battery_length_mm,
            )
            for i in range(n_est)
        ]

        forbidden = ctg.occupancy.mask_of(
            CellState.FORBIDDEN, CellState.PLANNED, CellState.PLACED,
        )
        # pack_best_effort, not first_fit_decreasing: FFDH alone placed
        # zero cells on scene_00005's 93%-free grid, because its shelves
        # span the strip width and their origins never scan in y. It is
        # still one of the arms competed here and still wins on lightly
        # obstructed cartridges, so this can only place MORE than before.
        # docs/superpowers/specs/2026-08-11-packing-ceiling.md.
        return pack_best_effort(
            items, strip_w_mm, strip_h_mm,
            allow_rotation=self.cfg.allow_rotation,
            forbidden_mask=forbidden,
            mm_per_cell=ctg.occupancy.resolution_mm,
        )

    # ---- pose construction ---------------------------------------------

    def _build_pose(
        self,
        ctg: Cartridge,
        placement,
        available: List[Battery],
    ) -> PickPlacePose:
        """Turn one FFDH placement into a :class:`PickPlacePose`."""
        # Target place point (workspace mm).
        cx_mm = placement.x + placement.width / 2
        cy_mm = placement.y + placement.height / 2
        target_x, target_y = self._cell_to_workspace(ctg, cx_mm, cy_mm)

        # Choose the nearest battery and consume it.
        bat = self._nearest_battery(available, ctg, placement)
        available.remove(bat)
        bat.assigned_to_pose = True

        pick_x, pick_y = self._image_to_workspace(bat.bbox.cx, bat.bbox.cy)
        row, col = self._xy_mm_to_cell(ctg, placement.x, placement.y)
        ctg.mark_cell(row, col, CellState.PLANNED)

        return PickPlacePose(
            pick=WorkspacePoint(
                x_mm=pick_x,
                y_mm=pick_y,
                z_mm=self.cfg.pick_approach_height_mm,
            ),
            place=WorkspacePoint(
                x_mm=target_x,
                y_mm=target_y,
                z_mm=self.cfg.place_insert_height_mm,
            ),
            cartridge_id=ctg.id,
            grid_row=row,
            grid_col=col,
            battery_detection_id=bat.id,
        )

    # ---- geometry helpers ----------------------------------------------

    def _nearest_battery(
        self,
        available: List[Battery],
        ctg: Cartridge,
        placement,
    ) -> Battery:
        """Pick the battery nearest to the placement target (squared px)."""
        tx = (
            ctg.placeable_rectangle.xmin
            + (placement.x + placement.width / 2) / self.cfg.mm_per_px
        )
        ty = (
            ctg.placeable_rectangle.ymin
            + (placement.y + placement.height / 2) / self.cfg.mm_per_px
        )
        return min(
            available,
            key=lambda b: (b.bbox.cx - tx) ** 2 + (b.bbox.cy - ty) ** 2,
        )

    def _image_to_workspace(
        self, px: float, py: float,
    ) -> Tuple[float, float]:
        """Map camera pixel → workspace mm using the calibration."""
        return (
            px * self.cfg.mm_per_px + self.cfg.origin_offset_x_mm,
            py * self.cfg.mm_per_px + self.cfg.origin_offset_y_mm,
        )

    def _cell_to_workspace(
        self, ctg: Cartridge, x_mm: float, y_mm: float,
    ) -> Tuple[float, float]:
        """Cartridge-local mm → workspace mm."""
        pr = ctg.placeable_rectangle
        return (
            pr.xmin * self.cfg.mm_per_px + x_mm + self.cfg.origin_offset_x_mm,
            pr.ymin * self.cfg.mm_per_px + y_mm + self.cfg.origin_offset_y_mm,
        )

    def _xy_mm_to_cell(
        self, ctg: Cartridge, x_mm: float, y_mm: float,
    ) -> Tuple[int, int]:
        """Strip mm → grid (row, col), clipped to grid bounds."""
        res = ctg.occupancy.resolution_mm
        row = max(0, min(ctg.occupancy.rows - 1, int(y_mm / res)))
        col = max(0, min(ctg.occupancy.cols - 1, int(x_mm / res)))
        return row, col

    # ---- execution feedback --------------------------------------------

    def confirm_placement(
        self,
        cartridge_id: int,
        row: int,
        col: int,
        success: bool,
    ) -> None:
        """Update the occupancy grid in response to an execution result."""
        if cartridge_id not in self.env.cartridges:
            return
        ctg = self.env.cartridge(cartridge_id)
        if success:
            ctg.mark_cell(row, col, CellState.PLACED)
            return
        # Revert PLANNED → FREE so the next cycle can retry this cell.
        if ctg.occupancy.get(row, col) == CellState.PLANNED:
            ctg.mark_cell(row, col, CellState.FREE)


__all__ = ["Planner", "PlannerConfig"]
