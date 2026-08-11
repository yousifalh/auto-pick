"""Placement-area extractor tests.

Builds synthetic cartridges from scratch (no dependency on the OpenCV
dataset generator) so these run in a fraction of a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.types import BBox
from plan.placement_area import PlacementAreaExtractor
from plan.scene import CellState

# HeuristicPlacementAreaExtractor (== PlacementAreaExtractor) now warns
# about its scope limit at construction time (spec 1.1: zero placeable
# area on 7/20 real cartridges). Every test in this file that constructs
# it directly for pipeline behaviour, not for the warning itself, opts
# out explicitly rather than letting the warning pass silently.
pytestmark = pytest.mark.filterwarnings(
    "ignore:HeuristicPlacementAreaExtractor assumes:RuntimeWarning")


def _synthetic_cartridge_image(H=160, W=200) -> np.ndarray:
    """A black background with a green rectangle and a dark central PCB."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Green tray in the middle
    img[20:140, 30:170, 1] = 200  # green channel
    img[20:140, 30:170, 0] = 40   # muted red
    img[20:140, 30:170, 2] = 40   # muted blue
    # Dark PCB in the centre
    img[60:100, 80:120] = (20, 20, 20)
    return img


def test_extract_returns_valid_occupancy_grid():
    img = _synthetic_cartridge_image()
    ext = PlacementAreaExtractor(safety_margin_px=3,
                                 mm_per_cell=1.5, mm_per_px=0.38)
    cartridge_bbox = BBox(20, 15, 180, 145)
    pa = ext.extract(img, cartridge_bbox)

    assert pa.occupancy.rows > 0
    assert pa.occupancy.cols > 0
    # Placement rectangle must be inside the cartridge bbox
    assert pa.rectangle.xmin >= cartridge_bbox.xmin
    assert pa.rectangle.xmax <= cartridge_bbox.xmax


def test_forbidden_cells_created_for_pcb():
    img = _synthetic_cartridge_image()
    ext = PlacementAreaExtractor(safety_margin_px=3,
                                 mm_per_cell=1.5, mm_per_px=0.38)
    pa = ext.extract(img, BBox(20, 15, 180, 145))

    forbidden = pa.occupancy.mask_of(CellState.FORBIDDEN)
    # Central PCB must yield at least some forbidden cells
    assert int(forbidden.sum()) > 0


def test_extract_rejects_empty_bbox():
    img = _synthetic_cartridge_image()
    ext = PlacementAreaExtractor()
    with pytest.raises(ValueError):
        ext.extract(img, BBox(10, 10, 10, 10))


def test_extract_raises_on_no_green():
    # Entirely black image
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    ext = PlacementAreaExtractor()
    with pytest.raises(RuntimeError):
        ext.extract(img, BBox(10, 10, 70, 70))


def test_extract_with_explicit_pcb_template():
    img = _synthetic_cartridge_image()
    ext = PlacementAreaExtractor(safety_margin_px=2,
                                 mm_per_cell=1.5, mm_per_px=0.38)
    # Make a PCB template that covers roughly the centre
    pcb_template = np.zeros((130, 160), dtype=np.uint8)
    pcb_template[40:80, 50:110] = 1
    pa = ext.extract(img, BBox(20, 15, 180, 145),
                     pcb_template_mask=pcb_template)
    assert pa.occupancy.rows > 0


# ------------------------------------- HeuristicPlacementAreaExtractor ----

def test_heuristic_extractor_warns_about_its_scope_limit():
    """It returns zero placeable area on 7 of 20 real cartridges
    (spec 1.1). It must not be selectable for real imagery silently."""
    import warnings

    from plan.placement_area import HeuristicPlacementAreaExtractor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HeuristicPlacementAreaExtractor()
    assert any("green" in str(w.message).lower() for w in caught)


# ---------------------------------- SegmentationPlacementAreaExtractor ----

def test_segmentation_extractor_produces_the_same_contract():
    """Both implementations must satisfy one contract, or the planner
    cannot swap them."""
    import numpy as np

    from common.types import BBox
    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from plan.placement_area import SegmentationPlacementAreaExtractor

    label = np.zeros((80, 60), np.int8)
    label[5:75, 5:55] = CH_CARTRIDGE
    label[12:68, 12:48] = CH_BAY

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0)
    pa = ex.extract(np.zeros((200, 200, 3), np.uint8),
                    BBox(20, 30, 80, 110), label_map=label)

    assert pa.rectangle.width > 0 and pa.rectangle.height > 0
    assert pa.occupancy is not None
    assert pa.inside_mask.shape == label.shape
    assert pa.mm_per_cell == 1.5
    assert 0.0 <= pa.consistency_iou <= 1.0
    # Backward compatibility: pcb_mask is still populated.
    assert pa.pcb_mask.shape == label.shape


def test_rectangle_is_in_full_image_coordinates():
    """The heuristic returns image-space coordinates and the planner
    stores them on the Cartridge. The segmentation path must match."""
    import numpy as np

    from common.types import BBox
    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from plan.placement_area import SegmentationPlacementAreaExtractor

    label = np.zeros((80, 60), np.int8)
    label[5:75, 5:55] = CH_CARTRIDGE
    label[12:68, 12:48] = CH_BAY

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0)
    pa = ex.extract(np.zeros((200, 200, 3), np.uint8),
                    BBox(20, 30, 80, 110), label_map=label)

    assert pa.rectangle.xmin >= 20 and pa.rectangle.ymin >= 30
    assert pa.rectangle.xmax <= 80 and pa.rectangle.ymax <= 110


def test_low_agreement_is_planned_anyway_the_tau_gate_is_retired():
    """The inverse of the test this replaces.

    This label map is the old gate's own fixture: most of the predicted
    `bay` is really electronics, so the two estimates agree over well
    under half their union (measured IoU ~0.43 here) - comfortably
    below every tau this repo ever quoted (0.85 / 0.7492 / 0.5715), and
    it used to raise PlacementDisagreement.

    It must now plan. The IoU is measured to correlate with placement
    error in the WRONG direction (FDR v3 section 13.2.1), so refusing
    on it discarded good cartridges for no safety return - 5 of 8 on
    real frames. What still protects the cell is P_safe: the electronics
    are excluded from the returned area by the INTERSECTION, which this
    test also pins, so "the gate is gone" cannot quietly become
    "the exclusion is gone".
    """
    import numpy as np

    from common.types import BBox
    from plan.arbitration import (CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS,
                                  arbitrate)
    from plan.placement_area import SegmentationPlacementAreaExtractor

    label = np.zeros((80, 60), np.int8)
    label[5:75, 5:55] = CH_CARTRIDGE
    label[12:68, 12:48] = CH_BAY
    label[12:50, 12:48] = CH_ELECTRONICS      # most of "bay" is really PCB

    _safe, iou = arbitrate(label, 0)
    assert iou < 0.5715, (
        "fixture no longer exercises a low-agreement cartridge - it has "
        "to sit below every tau this repo ever quoted for the test to "
        "mean anything")

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0)
    pa = ex.extract(np.zeros((200, 200, 3), np.uint8),
                    BBox(20, 30, 80, 110), label_map=label)

    assert pa.inside_mask.any(), "a low-IoU cartridge must still plan"
    assert pa.consistency_iou == iou, (
        "the IoU must still be REPORTED - it is retired as a gate, not "
        "as observability")
    # P_safe is load-bearing and stays: no pixel the segmenter called
    # electronics may end up inside the placeable area.
    assert not (pa.inside_mask.astype(bool)
                & (label == CH_ELECTRONICS)).any()


def test_tau_is_gone_from_the_constructor_not_silently_ignored():
    """Passing the retired argument must fail loudly.

    A constructor that accepted `tau=` and did nothing with it is the
    exact failure this whole change is repairing: a decision that lives
    in prose while the behaviour continues underneath.
    """
    import pytest

    from plan.placement_area import SegmentationPlacementAreaExtractor

    with pytest.raises(TypeError, match="tau"):
        SegmentationPlacementAreaExtractor(mm_per_px=0.625, tau=0.85)


def test_missing_label_map_raises_rather_than_silently_degrading():
    import numpy as np
    import pytest

    from common.types import BBox
    from plan.placement_area import SegmentationPlacementAreaExtractor

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0)
    with pytest.raises(ValueError, match="label_map"):
        ex.extract(np.zeros((200, 200, 3), np.uint8), BBox(0, 0, 40, 40))


# ---------------------------- bad-box vs. full-cartridge distinction ----
#
# plan.arbitration.arbitrate() alone cannot tell a badly-placed detector
# box apart from a genuinely full cartridge: both reduce to an empty
# P_safe and IoU 0.0 (mask_iou's union == 0 rule treats "both estimates
# empty" as maximally disagreeing, not as the trivial agreement it
# actually is). These tests pin the extra disambiguation in `extract()`
# so the two do not silently collapse into the same signal - a full
# cartridge is a normal state, a systematically bad box means the
# detector is drifting, and conflating them would hide that.

def test_bad_detector_box_is_distinguished_from_a_disagreement():
    """The crop centre lands on background, but real cartridge material
    (a neighbour's bay, caught at the crop's edge) is visible elsewhere
    in the frame - exactly the condition
    plan.arbitration.centre_component logs a warning for. Must raise
    BadDetectorBox, not the generic PlacementDisagreement a low-IoU
    reading over a REAL, centred cartridge would raise."""
    import numpy as np
    import pytest

    from common.types import BBox
    from plan.arbitration import CH_BAY
    from plan.placement_area import (BadDetectorBox, PlacementDisagreement,
                                     SegmentationPlacementAreaExtractor)

    label = np.zeros((60, 60), np.int8)
    label[0:20, 0:20] = CH_BAY  # a neighbour's bay, in frame but off-centre

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0)
    with pytest.raises(BadDetectorBox):
        ex.extract(np.zeros((200, 200, 3), np.uint8),
                   BBox(20, 30, 80, 90), label_map=label)
    # BadDetectorBox IS-A PlacementDisagreement (skip-and-retry is still
    # correct), so a caller that only knows the base type still catches
    # it - this is what makes it a safe, additive change.
    with pytest.raises(PlacementDisagreement):
        ex.extract(np.zeros((200, 200, 3), np.uint8),
                   BBox(20, 30, 80, 90), label_map=label)


def test_a_full_cartridge_does_not_raise_placement_disagreement():
    """A well-centred cartridge (not a bad box) whose bay is entirely
    occupied by existing cells: both estimates legitimately agree there
    is no room. This must raise a plain RuntimeError - not
    PlacementDisagreement, and not BadDetectorBox - so a fleet-wide full
    rate is never misread as the detector drifting."""
    import numpy as np
    import pytest

    from common.types import BBox
    from plan.arbitration import CH_BATTERY
    from plan.placement_area import (BadDetectorBox, PlacementDisagreement,
                                     SegmentationPlacementAreaExtractor)

    label = np.zeros((80, 60), np.int8)
    label[12:68, 12:48] = CH_BATTERY   # every cell position already full

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0)
    with pytest.raises(RuntimeError) as excinfo:
        ex.extract(np.zeros((200, 200, 3), np.uint8),
                   BBox(20, 30, 80, 110), label_map=label)
    assert not isinstance(excinfo.value, PlacementDisagreement)
    assert not isinstance(excinfo.value, BadDetectorBox)
