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

Every queued battery reserves the WHOLE cell block its footprint
covers (:class:`plan.scene.Reservation`), in the orientation it was
placed at. Marking only the corner cell — which is what this did — let
a later cycle seat a cell 1.5 mm from one already in the tray, ~17 mm
of physical overlap on an 18.5 mm battery, with the queue, the
counters and the robot's status all reading normal (audit E, finding
1). Reservations from a queue that was never executed are released at
the top of the next cycle, because the queue is rebuilt from scratch
every frame.

Every pose is checked against the robot's declared
:class:`plan.scene.WorkspaceBounds` before it is emitted — pick point
and place target both, because both are commanded to the arm. That
envelope was parsed and compared against nothing (audit E, finding 5).

The envelope is applied at **two altitudes**, because two different
things were being conflated:

* **A cartridge slot or a loose battery that lies outside the arm's
  reach is a scene condition, not a bug.** A fixed-mount camera
  legitimately images more table than a 706 mm-reach arm can serve, and
  ``configs/demo_seg.yaml``'s frames span up to 1338 x 752 mm against a
  700 mm envelope — no origin offset makes that fit. Those candidates
  are **filtered out before a pose is built**, and counted
  (``unreachable_place_target_count``, ``unreachable_battery_count``,
  ``unreachable_cartridge_count``). Declining to serve part of the scene
  is normal; declining *silently* is the failure this project keeps
  finding, so it is a reported number rather than a shorter queue.
* **A commanded pose outside the envelope is a planning bug**, and
  :meth:`Planner._build_pose` still raises
  :class:`plan.scene.OutOfWorkspace` on one. After the filter that
  branch is unreachable in normal operation — which is the point. It is
  the invariant on the only thing in this pipeline that constructs a
  ``PickPlacePose``, and every byte the executor puts on the wire comes
  from one. It is never a clamp.

Cross-frame identity is the twin's (:mod:`plan.scene`), but two of its
consequences are this module's:

* **Only cartridges VISIBLE in this frame are planned into.** A tracked
  cartridge the snapshot did not contain keeps its grid and its
  reservations — that memory is what stops a returning cartridge from
  being planned into twice — but it is not evidence about the scene in
  front of the camera, and neither ``_pack_cartridge`` nor
  ``_ensure_placement_areas`` will touch it.
* **A cartridge that moved is re-measured, and its batteries move with
  it.** ``_ensure_placement_areas`` re-extracts whenever the twin has
  invalidated the geometry, and ``_reproject_placed_reservations`` then
  re-quantises every physically placed footprint onto the new grid. The
  rectangle used to be extracted exactly once, ever, so a cartridge that
  moved 38 mm kept commanding the millimetres it had before it moved
  (audit H, finding 2).

The millimetre interlock in :meth:`plan.scene.Cartridge.reserve` is
per-cartridge and cannot see across two twin entries, so
:meth:`Planner._conflicts_with_another_cartridge` asks the same question
in the workspace frame, where every cartridge's footprints are
comparable. Same filter/invariant pairing as the workspace envelope: the
filter declines and counts, ``_build_pose`` raises.

After the executor reports back, :meth:`Planner.confirm_placement`
flips each PLANNED cell to PLACED on success, or reverts it to FREE
on failure so a future cycle can retry.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.logging import get_logger
from common.types import PickPlacePose, Snapshot, WorkspacePoint
from plan.bin_packing import Item, PackResult, pack_best_effort
from plan.placement_area import (
    BadDetectorBox,
    PlacementAreaExtractor,
    PlacementDisagreement,
    UnknownScale,
    _resolve_scale,
)
from plan.scene import (
    Battery,
    Cartridge,
    CellState,
    EnvironmentModel,
    PlacementCollision,
    Reservation,
    TrackingConfig,
    WorkspaceBounds,
)

log = get_logger("autopick.planner")


# ------------------------------------------------------ configuration --

@dataclass
class PlannerConfig:
    """Knobs loaded from ``configs/planning.yaml``."""

    battery_width_mm: float = 18.5
    battery_length_mm: float = 65.0
    # FALLBACK calibration, millimetres per pixel, for frames that carry
    # none of their own - which is what a real fixed-mount camera is:
    # one lens, one working distance, one GSD, measured once. It is NOT
    # the scale to use when the frame states its own; see
    # `Planner.frame_scale`.
    #
    # None means no fallback was configured, and an uncalibrated frame
    # then raises UnknownScale rather than silently adopting a number
    # nobody chose. The old default was 0.38 - `planning.yaml`'s
    # placeholder, marked "Replace with real intrinsics when available" -
    # which meant a caller who never set it got millimetres anyway.
    mm_per_px: Optional[float] = None
    origin_offset_x_mm: float = 0.0
    origin_offset_y_mm: float = 0.0
    # The Z that goes on the wire as the PICK target, and it is a GRASP
    # height, not an approach. `krl_prog/routines.src` takes the
    # commanded pick pose as the point to close the gripper at and
    # DERIVES its own approach from it (`approach_pos.Z = pick_pos.Z +
    # 60`, then `LIN pick_pos`). Putting an approach height here made
    # the controller approach at +120 mm and "grasp" at 60 mm - clear
    # of the part - after which `$IN[10]` reports no vacuum and the
    # cycle returns PICK_FAILED. `mock_kuka_server` cannot fail that
    # grasp (it models no parts) so it warns once above
    # `_GRASP_BAND_MM`; `tests/test_planner.py` asserts the queue stays
    # inside that band.
    #
    # There is deliberately no approach height in this config: the
    # approach is controller-side and the frame has no field to carry a
    # second Z. See docs/superpowers/specs/2026-08-12-fix-execution-safety.md.
    pick_grasp_height_mm: float = 5.0
    place_insert_height_mm: float = 2.0
    allow_rotation: bool = True
    # Cross-frame identity: how the twin decides two frames show the same
    # cartridge, how long it remembers one it cannot see, and how far a
    # box may move before its measured geometry is thrown away. Declared
    # in `planning.yaml` under `tracking:`; every one of these numbers
    # used to be a hardcoded default no config could reach (audit H,
    # finding 4).
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_dict(cls, cfg: Dict) -> "PlannerConfig":
        battery = cfg.get("battery", {}) or {}
        camera = cfg.get("camera", {}) or {}
        motion = cfg.get("motion", {}) or {}
        # `approach_height_mm` used to feed the pick Z. Silently ignoring
        # it now would leave a key that looks like it still sets the
        # pick height and does not - the exact failure mode
        # `configs/execution.yaml`'s five deleted motion keys were, so
        # it is refused by name instead.
        if "approach_height_mm" in motion:
            raise ValueError(
                "motion.approach_height_mm no longer configures anything: "
                "the wire Z is the GRASP pose and krl_prog/routines.src "
                "derives the approach from it (+60 mm). Use "
                "motion.grasp_height_mm if you mean the grasp height, or "
                "delete the key - the approach is controller-side and the "
                "16-byte frame has no field for it.")
        raw_scale = camera.get("mm_per_px_x")
        return cls(
            battery_width_mm=float(battery.get("diameter_mm", 18.5)),
            battery_length_mm=float(battery.get("length_mm", 65.0)),
            # Absent key -> no fallback, not a default. A config that
            # says nothing about the camera scale has not calibrated the
            # camera.
            mm_per_px=None if raw_scale is None else float(raw_scale),
            origin_offset_x_mm=float(camera.get("origin_offset_x_mm", 0.0)),
            origin_offset_y_mm=float(camera.get("origin_offset_y_mm", 0.0)),
            pick_grasp_height_mm=float(
                motion.get("grasp_height_mm", 5.0),
            ),
            place_insert_height_mm=float(motion.get("insert_height_mm", 2.0)),
            allow_rotation=True,
            tracking=TrackingConfig.from_dict(cfg.get("tracking")),
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
        self.env = EnvironmentModel(workspace=workspace,
                                    tracking=planner_cfg.tracking)
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
        # Placement areas discarded because the frame's scale changed
        # under them. Expected to stay 0 on a fixed-mount camera; a
        # non-zero count on real hardware means the calibration is
        # moving, which is worth seeing rather than absorbing.
        self.rescaled_area_drop_count = 0
        # Reservations released because the queue that owned them was
        # rebuilt before they were executed. Expected to be non-zero on
        # a normal run - `main` executes one pose per cycle and re-plans
        # - so this is a rate to watch, not an alarm. It exists because
        # the alternative to releasing them is a cartridge that fills
        # with batteries nobody picked, and the alternative to counting
        # them is doing that silently.
        self.released_reservation_count = 0
        # Candidates declined because the arm cannot reach them. A camera
        # legitimately sees more table than the arm can serve, so these
        # are NORMAL scene conditions rather than errors - but a run that
        # serves 3 of 14 cartridges while reporting success is the
        # silent-degradation failure this project keeps closing, so
        # nothing is dropped without incrementing one of these.
        #
        # All three are summed over cycles, on the same convention as
        # `main`'s `cartridges_detected`: the same physical slot seen in
        # two frames counts twice.
        #
        # * place targets: packed placements whose centre falls outside
        #   the envelope. One cartridge standing off the edge of the
        #   reachable region contributes one per slot the packer proposed
        #   in it.
        # * batteries: loose cells whose pick point falls outside it.
        #   Filtered from `available_batteries` BEFORE anything can
        #   consume one, so an unreachable battery does not also spend a
        #   slot.
        # * cartridges: cartridges the packer found room in where NOT ONE
        #   of those slots was reachable. This is the number that answers
        #   "how much of the scene did the arm decline to serve", which
        #   the per-slot counts do not.
        self.unreachable_place_target_count = 0
        self.unreachable_battery_count = 0
        self.unreachable_cartridge_count = 0
        # Physical batteries carried onto a freshly measured occupancy
        # grid after their cartridge moved, and the ones that could not
        # be: a footprint whose millimetres fall outside the new
        # rectangle has nowhere to go in the new grid. The reservation is
        # KEPT either way (it still answers the millimetre interlock) but
        # a battery the grid cannot represent is a battery the packer
        # cannot see, so it is counted rather than absorbed.
        self.placed_reservation_reprojected_count = 0
        self.placed_reservation_lost_count = 0
        # Candidate placements declined because they overlap a footprint
        # in a DIFFERENT visible cartridge. Expected to be 0: it means
        # two twin entries claim the same physical space, which is a
        # perception fault (a split or duplicated cartridge box), not a
        # scene condition.
        self.cross_cartridge_conflict_count = 0
        # Executor results the twin could not record because their
        # cartridge is no longer tracked. `main` counts these as placed -
        # a battery WAS put down - so a non-zero count here is the twin
        # and the world disagreeing about what is in the tray.
        self.untracked_confirmation_count = 0
        # PLACED reservations discarded because the frame's scale moved
        # under their cartridge. `rescaled_area_drop_count` counts the
        # cartridges; this counts the physical batteries that went with
        # them.
        self.rescaled_placed_reservation_drop_count = 0

    # ---- main cycle -----------------------------------------------------

    def frame_scale(self, snapshot: Snapshot) -> float:
        """Millimetres per pixel for ``snapshot``'s frame.

        The frame's own calibration wins; the configured fallback stands
        in for frames that carry none; an unknown scale raises
        :class:`UnknownScale` rather than defaulting.

        Resolved ONCE per cycle and handed to everything that needs it,
        rather than looked up independently by the extractor and the
        planner. Those two used to hold separate copies of the same
        constant and ``main._build_planner`` had to keep them in step by
        hand; one resolution point makes them agree by construction.
        """
        return _resolve_scale(snapshot.mm_per_px, self.cfg.mm_per_px,
                              type(self).__name__)

    def cycle(
        self,
        snapshot: Snapshot,
        image_rgb: Optional[np.ndarray],
    ) -> List[PickPlacePose]:
        """Run one planning cycle and return a freshly built queue."""
        scale = self.frame_scale(snapshot)
        self.env.update_from_snapshot(snapshot)
        self._drop_areas_measured_at_another_scale(scale)
        self._release_stale_reservations()

        if image_rgb is not None:
            self._ensure_placement_areas(image_rgb, snapshot, scale)

        queue: List[PickPlacePose] = []
        available: List[Battery] = self._reachable_batteries(
            self.env.available_batteries(), scale)

        # VISIBLE cartridges only. A tracked-but-unseen cartridge keeps
        # its grid and its reservations - that memory is what stops a
        # returning cartridge from being planned into twice - but it is
        # not evidence about the scene in front of the camera right now,
        # and planning into it would command the arm into a box that is
        # not in this frame's snapshot.
        for ctg in sorted(self.env.visible_cartridges(), key=lambda c: c.id):
            if ctg.placeable_rectangle is None or ctg.occupancy is None:
                continue

            pack = self._pack_cartridge(ctg)
            # Row-major fill: sort packed placements by (y, x).
            placements = sorted(pack.placements, key=lambda x: (x.y, x.x))

            # Reachability is decided for the WHOLE cartridge before any
            # pose is built, deliberately. It is a property of the
            # cartridge's geometry against the envelope and has nothing
            # to do with how many loose batteries happen to be left, so
            # deciding it inside the pose loop would let the
            # ran-out-of-batteries early return below report a cartridge
            # as unreachable when it was merely unvisited.
            reachable: List[Tuple[object, Reservation]] = []
            for p in placements:
                tx, ty = self._place_target(ctg, p)
                if not self._reaches(tx, ty):
                    self.unreachable_place_target_count += 1
                    continue
                # The reservation is built HERE, before the pose, because
                # the cross-cartridge interlock needs its workspace
                # footprint and because building it twice would let the
                # filter and the guard disagree about the same placement.
                res = self._reservation_for(ctg, p)
                clash = self._conflicts_with_another_cartridge(ctg, res)
                if clash is not None:
                    self._note_cross_cartridge_conflict(ctg, res, clash)
                    continue
                reachable.append((p, res))
            if placements and not reachable:
                self.unreachable_cartridge_count += 1

            for p, res in reachable:
                if not available:
                    return queue

                queue.append(self._build_pose(ctg, p, res, available, scale))

        return queue

    def _conflicts_with_another_cartridge(
        self, ctg: Cartridge, res: Reservation,
    ) -> Optional[Tuple[int, Reservation]]:
        """The FILTER half of the cross-cartridge interlock.

        A named method for the same reason :meth:`_reaches` is one: it is
        the seam a test disables in order to prove that the INVARIANT in
        ``_build_pose`` is still live. They ask the same question of the
        same reservations and they must — but a guard whose only proof is
        that nothing has tripped it is not a proven guard.
        """
        return self.env.conflicting_reservation(
            res, exclude_cartridge_id=ctg.id)

    def _note_cross_cartridge_conflict(
        self, ctg: Cartridge, res: Reservation,
        clash: Tuple[int, Reservation],
    ) -> None:
        """Record a slot declined because another cartridge holds it.

        Skipped and counted rather than raised, on the same reasoning as
        an unreachable candidate: the usual cause is a segmenter that
        described one physical cartridge with two boxes, and aborting the
        run over an ordinary perception artefact would trade a silent
        collision for a fragile pipeline. What must not happen is that
        two overlapping place targets are both QUEUED, and after this
        neither can be — the interlock in ``_build_pose`` re-asks the
        same question and raises if one ever gets that far.
        """
        other_id, other = clash
        self.cross_cartridge_conflict_count += 1
        log.warning(
            "cartridge %d: declining a placement at (%.1f, %.1f) mm - it "
            "overlaps a %s battery of cartridge %d at (%.1f, %.1f) mm in "
            "the workspace frame. Two twin entries claiming one piece of "
            "physical space is a split or duplicated cartridge box, not "
            "two cartridges",
            ctg.id, res.wx_mm, res.wy_mm, other.state.name, other_id,
            other.wx_mm, other.wy_mm)

    def _reaches(self, x_mm: float, y_mm: float) -> bool:
        """Whether the arm can serve a candidate at this workspace point.

        A named method rather than an inline ``workspace.contains`` call,
        deliberately. This is the **filter**: it decides which candidates
        become poses, and answering "no" is an ordinary outcome.
        :meth:`plan.scene.WorkspaceBounds.require`, in ``_build_pose``, is
        the **invariant**: it decides whether a pose that already exists
        may be commanded, and answering "no" is fatal.

        They ask the same question of the same envelope, and they must —
        but they are separate code paths on purpose, so a test can
        disable this one and prove the other still fires. A guard whose
        only proof is that nothing has tripped it is not a proven guard
        (``tests/test_planner.py::
        test_the_envelope_guard_fires_when_the_reachability_filter_is_bypassed``).
        """
        return self.env.workspace.contains(x_mm, y_mm)

    def _reachable_batteries(
        self, batteries: List[Battery], frame_mm_per_px: float,
    ) -> List[Battery]:
        """The loose batteries the arm can actually reach, counting the rest.

        A battery lying outside the envelope is a **scene condition**: a
        fixed-mount camera images more table than the arm can serve, and
        on ``configs/demo_seg.yaml``'s renders it images nearly twice the
        envelope in each axis. Raising here would make an ordinary frame
        fatal; clamping would grasp empty table or the neighbour.

        So it is skipped — and *counted*, because the third option, a
        silently shorter queue, makes "the arm cannot reach those" and
        "there are no batteries" the same observation. The filter runs
        before ``_nearest_battery`` can consume anything, so an
        unreachable cell does not also spend a cartridge slot.
        """
        out: List[Battery] = []
        for bat in batteries:
            x, y = self._image_to_workspace(
                bat.bbox.cx, bat.bbox.cy, frame_mm_per_px)
            if self._reaches(x, y):
                out.append(bat)
                continue
            self.unreachable_battery_count += 1
        return out

    def _drop_areas_measured_at_another_scale(self, scale: float) -> None:
        """Forget any placement area measured at a different scale.

        Cartridges are persistent across frames (``plan.scene``'s module
        docstring) and their ``placeable_rectangle`` / ``occupancy`` are
        in PIXELS. If the frame's scale changes, those pixels no longer
        mean the millimetres they were measured to mean, and packing them
        against the new scale would silently mis-size every cell - the
        same class of error as the constant this replaced, arriving by a
        different route.

        On a real fixed-mount camera the scale never changes and this
        never fires. It fires on a scale-randomised corpus, where a
        persisting match across two unrelated renders was already
        carrying one scene's rectangle into another.
        """
        for ctg in self.env.cartridges.values():
            if ctg.placeable_rectangle is None:
                continue
            # A cartridge this frame did not contain is not re-measured
            # against this frame's scale, because it is not being
            # measured at all: it is memory, the planner will not plan
            # into it (`cycle` iterates `visible_cartridges`), and
            # dropping its grid here would destroy the physical record
            # the tracking layer exists to keep. It is re-measured when
            # it comes back, by the geometry-refresh path.
            if not ctg.visible:
                continue
            if ctg.mm_per_px is not None and ctg.mm_per_px == scale:
                continue
            self.rescaled_area_drop_count += 1
            ctg.placeable_rectangle = None
            ctg.occupancy = None
            ctg.pcb_mask = None
            ctg.mm_per_px = None
            # The reservations went with the grid they indexed: their
            # rows, columns AND millimetres were all measured against a
            # rectangle that no longer applies. Keeping them would let
            # the collision interlock compare footprints from two
            # different scales, which is worse than not comparing.
            #
            # Counted, though. `rescaled_area_drop_count` says a
            # cartridge lost its geometry; this says how many PHYSICAL
            # batteries the twin forgot in the process, which is the
            # number that matters and was previously invisible.
            self.rescaled_placed_reservation_drop_count += len(
                ctg.placed_reservations())
            ctg.reservations.clear()

    def _release_stale_reservations(self) -> None:
        """Release PLANNED reservations left over from the last queue.

        ``cycle`` rebuilds the queue from scratch every frame, so a
        reservation still PLANNED when the next cycle starts belongs to
        a queue that is being discarded. ``main`` executes exactly one
        pose per cycle and re-plans, so on a queue of N this leaks N-1
        reservations per frame if nothing releases them - and with the
        whole footprint now marked (not one cell) that fills a cartridge
        with batteries nobody picked within a few frames. The packer
        then places nothing, the run reports "queue empty, job done",
        and the tray is nearly empty. PLACED reservations are physical
        and are not touched.
        """
        for ctg in self.env.cartridges.values():
            self.released_reservation_count += ctg.release_planned()

    # ---- placement-area extraction --------------------------------------

    def _ensure_placement_areas(
        self, image_rgb: np.ndarray, snapshot: Optional[Snapshot] = None,
        mm_per_px: Optional[float] = None,
    ) -> None:
        """Fill in the placement rectangle / occupancy for new cartridges.

        ``mm_per_px`` is the scale ``cycle`` resolved for this frame and
        is passed to every extractor call. It is also recorded on the
        cartridge, because the rectangle it produces is in pixels and the
        twin outlives the frame.

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
            # Extraction reads THIS frame's image inside the cartridge's
            # box. For a cartridge this frame did not contain, that box
            # is a memory and the pixels under it belong to whatever is
            # there now, so there is nothing here to measure.
            if not ctg.visible:
                continue
            try:
                kwargs = {}
                if (self._extractor_accepts_label_map and snapshot is not None
                        and ctg.detection_index in snapshot.cartridge_masks):
                    kwargs["label_map"] = \
                        snapshot.cartridge_masks[ctg.detection_index]
                pa = self.extractor.extract(
                    image_rgb, ctg.bbox, mm_per_px=mm_per_px, **kwargs)
            except UnknownScale:
                # NOT swallowed by the blanket handler below. An
                # uncalibrated frame is a configuration error affecting
                # every cartridge in the run, and "skip and retry next
                # frame" would retry it forever while reporting a clean
                # zero - which is the exact shape of the failure this
                # whole change is repairing.
                raise
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
            # From the PlacementArea, not from the local variable: the
            # extractor is the authority on what it actually measured at,
            # and reading it back means an extractor that resolved the
            # scale differently cannot go unnoticed.
            ctg.mm_per_px = pa.mm_per_px
            # A fresh grid reads all-FREE, and this cartridge may be
            # holding batteries that were placed against the grid this
            # one replaces. Put them back before anything packs against
            # it.
            self._reproject_placed_reservations(ctg)

    def _reproject_placed_reservations(self, ctg: Cartridge) -> None:
        """Re-mark physical batteries onto a newly measured grid.

        A cartridge that moves keeps its batteries and loses its
        measurement (:meth:`plan.scene.Cartridge.invalidate_geometry`).
        Those batteries are recorded as ``Reservation``s in strip
        millimetres relative to the placeable rectangle's corner — which
        is the frame that MOVES WITH THE CARTRIDGE — so re-quantising the
        same millimetres against the new grid puts each battery back
        where it physically is.

        The grid is a raster of the reservations; the reservations are
        the record. That is why this direction is possible at all, and
        why the reservation is kept even when the new grid cannot hold
        it: an unrepresentable battery still answers the millimetre
        interlock, and it is counted so that "the packer cannot see it"
        is a number rather than a silence.
        """
        placed = ctg.placed_reservations()
        if not placed:
            return
        occ = ctg.occupancy
        restored = 0
        rekeyed: Dict[Tuple[int, int], Reservation] = {
            k: r for k, r in ctg.reservations.items()
            if r.state is not CellState.PLACED
        }
        for res in placed:
            row0, row1, col0, col1 = self._cell_block_for(
                ctg, res.x_mm, res.y_mm, res.w_mm, res.h_mm, clip=True)
            if row1 <= row0 or col1 <= col0:
                self.placed_reservation_lost_count += 1
                log.warning(
                    "cartridge %d: a placed battery at (%.1f, %.1f) mm falls "
                    "outside the re-measured placeable rectangle - the twin "
                    "still refuses to place over it, but the packer can no "
                    "longer see it", ctg.id, res.x_mm, res.y_mm)
                rekeyed[(res.row, res.col)] = res
                continue
            res.row0, res.row1, res.col0, res.col1 = row0, row1, col0, col1
            res.row, res.col = row0, col0
            occ.set_block(row0, row1, col0, col1, CellState.PLACED)
            if (row0, col0) in rekeyed:
                # Two physical batteries quantising to one anchor cell
                # cannot happen - a 13 x 44 cell footprint and the
                # millimetre interlock together forbid it - so it means
                # the reservations have been corrupted, and silently
                # dropping one of them is how a battery stops existing.
                raise RuntimeError(
                    f"cartridge {ctg.id}: two reservations re-project onto "
                    f"anchor ({row0}, {col0}) — the placed footprints no "
                    "longer describe distinct batteries")
            rekeyed[(row0, col0)] = res
            restored += 1
            self.placed_reservation_reprojected_count += 1
        ctg.reservations = rekeyed
        # The whole point of the exercise: a fresh grid that still reads
        # empty after re-projecting batteries onto it would let the
        # packer fill slots that are physically full.
        if restored and occ.placed_count() == 0:
            raise RuntimeError(
                f"cartridge {ctg.id}: re-projected {len(placed)} placed "
                "batteries onto the new occupancy grid and it still reads "
                "as holding none")

    # ---- FFDH wrapper ---------------------------------------------------

    def _pack_cartridge(self, ctg: Cartridge) -> PackResult:
        pr = ctg.placeable_rectangle
        # The cartridge's OWN scale, not the current frame's: this
        # rectangle is in the pixels of the frame that measured it.
        # _drop_areas_measured_at_another_scale guarantees they are the
        # same number whenever both exist, and this reads the one that
        # is true by construction rather than the one that is true by
        # convention.
        scale = _resolve_scale(ctg.mm_per_px, self.cfg.mm_per_px,
                               f"cartridge {ctg.id}")
        strip_w_mm = pr.width * scale
        strip_h_mm = pr.height * scale

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
        res: Reservation,
        available: List[Battery],
        frame_mm_per_px: float,
    ) -> PickPlacePose:
        """Turn one FFDH placement into a :class:`PickPlacePose`.

        Two scales, deliberately. The PLACE point comes off the
        cartridge's rectangle and uses the scale that rectangle was
        measured at (``ctg.mm_per_px``); the PICK point comes off a loose
        battery detected in THIS frame and uses this frame's scale. They
        are the same number on a fixed-mount camera and are only
        different while a cached area outlives a re-calibration - at
        which point using one for the other would put the gripper in the
        wrong place.

        **The two ``require`` calls below are invariants, not filters.**
        ``cycle`` has already dropped every unreachable place target and
        every unreachable battery, so neither can fire while the filter
        is correct — which is exactly why they stay. This method is the
        only thing in the pipeline that constructs a ``PickPlacePose``,
        and ``main`` hands ``queue[0]`` to the executor unmodified, so
        every coordinate that reaches the wire came through here. If a
        pose ever gets this far out of envelope, the filter is broken and
        the arm must not be commanded; that is a hard stop, never a
        clamp. ``tests/test_planner.py`` proves the guard is still live
        by bypassing the filter and watching it fire.
        """
        # Checked BEFORE a battery is consumed, so an out-of-envelope
        # target does not also silently spend one.
        target_x, target_y = self._place_target(ctg, placement)
        self.env.workspace.require(
            target_x, target_y, f"place target for cartridge {ctg.id}")

        # The cross-cartridge interlock, in its invariant form. `cycle`
        # has already declined every placement that overlaps another
        # visible cartridge's footprint, so this cannot fire while that
        # filter is correct — which is why it stays. Two queued poses
        # aimed at one piece of physical space is the failure it exists
        # for, and it is a hard stop, never a skip at this altitude.
        clash = self.env.conflicting_reservation(
            res, exclude_cartridge_id=ctg.id)
        if clash is not None:
            other_id, other = clash
            raise PlacementCollision(
                f"cartridge {ctg.id}: placement at ({res.wx_mm:.1f}, "
                f"{res.wy_mm:.1f}) mm overlaps a {other.state.name} battery "
                f"of cartridge {other_id} at ({other.wx_mm:.1f}, "
                f"{other.wy_mm:.1f}) mm — two twin entries are claiming the "
                "same physical space")

        # Choose the nearest battery and consume it.
        bat = self._nearest_battery(available, ctg, placement)
        available.remove(bat)
        bat.assigned_to_pose = True

        pick_x, pick_y = self._image_to_workspace(
            bat.bbox.cx, bat.bbox.cy, frame_mm_per_px)
        self.env.workspace.require(
            pick_x, pick_y, f"pick point for battery {bat.id}")

        row, col = res.row, res.col
        ctg.reserve(res)

        # `pick.z_mm` is the GRASP height and reaches the wire;
        # `place.z_mm` is the insert depth and does NOT (the 16-byte
        # frame has one Z field, and the pick needs it - see
        # KukaClient.pick_and_place). Both are documented on
        # PlannerConfig.
        return PickPlacePose(
            pick=WorkspacePoint(
                x_mm=pick_x,
                y_mm=pick_y,
                z_mm=self.cfg.pick_grasp_height_mm,
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

    def _place_target(self, ctg: Cartridge, placement) -> Tuple[float, float]:
        """Workspace mm of the centre of one packed placement.

        One function, two callers: ``cycle``'s reachability filter and
        ``_build_pose``'s envelope invariant. They MUST agree — a filter
        that computes the target one way and a guard that computes it
        another is how a pose gets past the first and trips the second on
        a scene that is perfectly ordinary.
        """
        cx_mm = placement.x + placement.width / 2
        cy_mm = placement.y + placement.height / 2
        return self._cell_to_workspace(ctg, cx_mm, cy_mm)

    def _nearest_battery(
        self,
        available: List[Battery],
        ctg: Cartridge,
        placement,
    ) -> Battery:
        """Pick the battery nearest to the placement target (squared px)."""
        scale = _resolve_scale(ctg.mm_per_px, self.cfg.mm_per_px,
                               f"cartridge {ctg.id}")
        tx = (
            ctg.placeable_rectangle.xmin
            + (placement.x + placement.width / 2) / scale
        )
        ty = (
            ctg.placeable_rectangle.ymin
            + (placement.y + placement.height / 2) / scale
        )
        return min(
            available,
            key=lambda b: (b.bbox.cx - tx) ** 2 + (b.bbox.cy - ty) ** 2,
        )

    def _image_to_workspace(
        self, px: float, py: float, mm_per_px: float,
    ) -> Tuple[float, float]:
        """Map camera pixel → workspace mm at THIS frame's scale."""
        return (
            px * mm_per_px + self.cfg.origin_offset_x_mm,
            py * mm_per_px + self.cfg.origin_offset_y_mm,
        )

    def _cell_to_workspace(
        self, ctg: Cartridge, x_mm: float, y_mm: float,
    ) -> Tuple[float, float]:
        """Cartridge-local mm → workspace mm, at the cartridge's scale."""
        pr = ctg.placeable_rectangle
        scale = _resolve_scale(ctg.mm_per_px, self.cfg.mm_per_px,
                               f"cartridge {ctg.id}")
        return (
            pr.xmin * scale + x_mm + self.cfg.origin_offset_x_mm,
            pr.ymin * scale + y_mm + self.cfg.origin_offset_y_mm,
        )

    def _xy_mm_to_cell(
        self, ctg: Cartridge, x_mm: float, y_mm: float,
    ) -> Tuple[int, int]:
        """Strip mm → grid (row, col), clipped to grid bounds."""
        res = ctg.occupancy.resolution_mm
        row = max(0, min(ctg.occupancy.rows - 1, int(y_mm / res)))
        col = max(0, min(ctg.occupancy.cols - 1, int(x_mm / res)))
        return row, col

    def _reservation_for(self, ctg: Cartridge, placement) -> Reservation:
        """The cell block one placement's real footprint covers.

        ``placement.width`` / ``.height`` are the dimensions of the
        orientation the packer actually CHOSE — :class:`PackedItem`
        swaps the item's nominal width and height when ``rotated`` — so
        an 18650 placed at 90° reserves 13 rows x 44 columns, where
        upright it reserves 44 rows x 13 columns.
        Reading ``self.cfg.battery_width_mm`` here instead would mark
        the nominal footprint and leave the long axis of every rotated
        placement unmarked, which is the same defect in a rarer costume
        (``pack_best_effort`` rotates on real cartridges: 36 placements
        on the test fixture, both orientations present).

        The near edge floors and the far edge ceils, matching
        :func:`common.packing._overlaps_forbidden` exactly. Any other
        rounding lets the mask be tested at one size and marked at
        another.
        """
        row0, row1, col0, col1 = self._cell_block_for(
            ctg, placement.x, placement.y, placement.width, placement.height)
        if ctg.placeable_rectangle is None:
            # The workspace millimetres below come off the rectangle, and
            # a Reservation carrying no workspace frame cannot be checked
            # against any other cartridge's. A cartridge with a grid but
            # no rectangle is not a state the planner produces (they are
            # set together in _ensure_placement_areas), so this is a
            # caller error rather than a case to paper over with a
            # fallback that would silently disable the interlock.
            raise RuntimeError(
                f"cartridge {ctg.id}: cannot build a reservation without a "
                "placeable_rectangle — there is no workspace frame to place "
                "the footprint in")
        wx_mm, wy_mm = self._cell_to_workspace(ctg, placement.x, placement.y)

        return Reservation(
            row=row0, col=col0, row0=row0, row1=row1, col0=col0, col1=col1,
            x_mm=float(placement.x), y_mm=float(placement.y),
            w_mm=float(placement.width), h_mm=float(placement.height),
            wx_mm=float(wx_mm), wy_mm=float(wy_mm),
        )

    def _cell_block_for(
        self, ctg: Cartridge, x_mm: float, y_mm: float,
        w_mm: float, h_mm: float, clip: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Quantise one strip-millimetre footprint to a half-open block.

        One quantiser, two callers — ``_reservation_for`` for a new
        placement and ``_reproject_placed_reservations`` for a battery
        being carried onto a re-measured grid. They MUST round
        identically: a battery marked at one size and tested at another
        is how the packer seats a cell into space that is already full.

        ``clip`` is the difference between the two. For a new placement,
        a footprint reaching more than one cell past the grid means the
        packing strip and the grid disagree about the rectangle and
        every reservation in the cartridge is off by an unknown amount,
        so it raises. For a battery being re-projected after the
        cartridge moved, falling outside the new rectangle is an
        ordinary consequence of re-measuring, and the caller counts it.
        """
        occ = ctg.occupancy
        if occ is None:
            raise RuntimeError(
                f"cartridge {ctg.id}: no occupancy grid to quantise against")
        res_mm = occ.resolution_mm
        row0, col0 = self._xy_mm_to_cell(ctg, x_mm, y_mm)
        row1 = int(np.ceil((y_mm + h_mm) / res_mm))
        col1 = int(np.ceil((x_mm + w_mm) / res_mm))

        # The strip is `pr.height * scale` mm while the grid is
        # `int(height_px / px_per_cell)` cells, so the far edge can sit
        # up to one cell past the last row: that is quantisation, and it
        # clips. More than one cell means the strip and the grid are
        # describing different rectangles, and every reservation in this
        # cartridge is then off by an unknown amount.
        if not clip and (row1 > occ.rows + 1 or col1 > occ.cols + 1):
            raise RuntimeError(
                f"cartridge {ctg.id}: placement at ({x_mm:.1f}, "
                f"{y_mm:.1f}) + ({w_mm:.1f} x {h_mm:.1f}) mm needs cells up "
                f"to ({row1}, {col1}) in a {occ.rows} x {occ.cols} grid at "
                f"{res_mm} mm — the packing strip and the occupancy grid "
                "disagree about the placeable rectangle")
        # `_xy_mm_to_cell` clips the near corner INTO the grid, which
        # would turn a footprint lying entirely past the far edge into a
        # one-cell block on the boundary — a battery marked somewhere it
        # is not. Ask the millimetres instead; an empty block is the
        # honest answer and the caller counts it.
        if (x_mm >= occ.cols * res_mm or y_mm >= occ.rows * res_mm
                or x_mm + w_mm <= 0 or y_mm + h_mm <= 0):
            return (0, 0, 0, 0)
        return (row0, max(row0 + 1, min(occ.rows, row1)),
                col0, max(col0 + 1, min(occ.cols, col1)))

    # ---- execution feedback --------------------------------------------

    def confirm_placement(
        self,
        cartridge_id: int,
        row: int,
        col: int,
        success: bool,
    ) -> None:
        """Update the occupancy grid in response to an execution result.

        ``(row, col)`` is the anchor the pose carried, which
        :meth:`plan.scene.Cartridge.confirm` resolves back to the whole
        block that battery reserved. Flipping the anchor cell alone
        would leave the other ~571 cells of a placed battery reading
        PLANNED forever on success, and would free none of them on
        failure.
        """
        if cartridge_id not in self.env.cartridges:
            # The twin no longer tracks this cartridge, so there is no
            # grid to update. A cartridge merely MISSING from the last
            # snapshot still lands above - that is the point of tracking
            # it - so reaching here means the track expired between the
            # pose being built and the executor reporting on it, i.e.
            # `tracking.max_missing_frames` frames went by mid-motion.
            #
            # `main` counts this as `placed`, because a battery really
            # was put down; the twin has no record of where. That
            # disagreement used to be a bare `return` (audit H, finding
            # 1) and is now a counted, logged event.
            self.untracked_confirmation_count += 1
            log.error(
                "confirm_placement for cartridge %d (%d, %d) success=%s: the "
                "twin no longer tracks that cartridge, so this placement is "
                "recorded NOWHERE. The next queue built for that space will "
                "not know a battery is in it",
                cartridge_id, row, col, success)
            return
        self.env.cartridge(cartridge_id).confirm(row, col, success)


__all__ = ["Planner", "PlannerConfig"]
