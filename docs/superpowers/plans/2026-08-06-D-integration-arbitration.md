# Integration and Arbitration Implementation Plan

*What this is: the task-by-task implementation plan a design spec was executed from, kept as part of this project's working record rather than written as documentation for a reader. The note that follows is tooling direction for the coding agent that executed the plan. For what these documents are, how they were used and what came of them, see [`../specs/README.md`](../specs/README.md).*

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the trained segmenter into the pipeline behind the existing `PlacementAreaExtractor` contract, reconcile the two independent placement-area estimates with an explicit arbitration rule, and calibrate the disagreement threshold against a measured safety criterion rather than a guessed number.

**Architecture:** Segmentation runs in **Recognition**, batched over every cartridge in a frame, and rides to Planning as one new optional field on `Snapshot`. `SegmentationPlacementAreaExtractor` consumes that field and performs only mask arithmetic, keeping Planning inside the tested 8 ms O3 budget. The two estimates — the network's direct `bay` channel and a derived `interior ∖ electronics ∖ obstruction ∖ battery` — are intersected, and their IoU gates the cartridge.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyTorch (Recognition side only), pytest.

## Global Constraints

- `SEG_CHANNELS` from Plan C is a contract: `0 background, 1 cartridge, 2 bay, 3 electronics, 4 obstruction, 5 battery`.
- FDR **O3 is a tested requirement**: queue rebuild ≤ 8 ms per cartridge, verified in `tests/test_planner.py`. Planning must do mask arithmetic only — no model forward pass.
- FDR **§10.4's 50 ms end-to-end budget** now absorbs segmentation. It must be re-measured and reported, not assumed.
- `PlacementAreaExtractor.extract(image_rgb, cartridge_bbox, ...) -> PlacementArea` is preserved. `plan/planner.py` and `main.py` keep working with either implementation.
- `Snapshot` is a mutable dataclass; the new field must be **optional with a default**, so every existing producer and consumer stays valid.
- The torch-free demo path must keep running. `python main.py --config configs/demo.yaml` needs no torch.
- τ is **calibrated, never asserted**. A hardcoded 0.85 that was never measured is a placeholder wearing a number's clothes.

---

## File Structure

| File | Responsibility |
|---|---|
| `plan/arbitration.py` | **New, dependency-light.** `P_direct` / `P_derived` / `P_safe`, the IoU gate, the erosion safety test. Pure NumPy + cv2, no torch. |
| `plan/placement_area.py` | Gains `SegmentationPlacementAreaExtractor`; the heuristic one gains a scope limit and a warning. |
| `common/types.py` | `Snapshot` gains `cartridge_masks`. |
| `recog/inference.py` | Gains an optional `BaySegmenter`, batched over a frame's cartridges. |
| `recog/calibrate_tau.py` | **New.** Calibrates τ against the erosion criterion and writes a receipt. |
| `tests/test_arbitration.py` | **New.** |
| `tests/test_placement_area.py` | Existing. Gains the shared contract test. |

---

### Task 1: Arbitration arithmetic

**Files:**
- Create: `plan/arbitration.py`
- Test: `tests/test_arbitration.py`

**Interfaces:**
- Consumes: a `(H, W)` int8 label map using `SEG_CHANNELS`
- Produces:
  - `CH_BACKGROUND, CH_CARTRIDGE, CH_BAY, CH_ELECTRONICS, CH_OBSTRUCTION, CH_BATTERY` — ints 0–5
  - `centre_component(label_map) -> np.ndarray` — bool mask of the connected component of all non-background pixels containing the crop centre
  - `derived_placement(label_map, wall_inset_px) -> np.ndarray`
  - `direct_placement(label_map) -> np.ndarray`
  - `arbitrate(label_map, wall_inset_px) -> Tuple[np.ndarray, float]` — `(P_safe, iou)`

Connectivity is computed over **all five** non-background channels, `battery` included. Excluding it lets a cell lying across a bay sever the region, leaving the centre component covering half the cartridge. Cells are part of the cartridge footprint for deciding what is connected, and are subtracted only afterwards.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arbitration.py`:

```python
"""Placement-area arbitration: two estimates, one conservative answer."""
from __future__ import annotations

import numpy as np
import pytest

from plan.arbitration import (CH_BATTERY, CH_BAY, CH_CARTRIDGE,
                              CH_ELECTRONICS, CH_OBSTRUCTION, arbitrate,
                              centre_component, derived_placement,
                              direct_placement)


def _cartridge(h=60, w=60, wall=4):
    """A cartridge occupying the middle of the crop: wall ring of
    `cartridge`, bay inside."""
    m = np.zeros((h, w), np.int8)
    m[10:h - 10, 10:w - 10] = CH_CARTRIDGE
    m[10 + wall:h - 10 - wall, 10 + wall:w - 10 - wall] = CH_BAY
    return m


def test_centre_component_keeps_only_the_centre_cartridge():
    """An over-large or jittered crop catches a neighbour's edge. The
    neighbour must not contribute to the derived estimate."""
    m = np.zeros((60, 60), np.int8)
    m[20:40, 20:40] = CH_BAY            # the one we want, at the centre
    m[0:6, 0:6] = CH_CARTRIDGE          # a neighbour's corner
    keep = centre_component(m)
    assert keep[30, 30]
    assert not keep[2, 2]


def test_centre_component_spans_battery_so_a_cell_cannot_split_it():
    """A cell lying right across a bay would sever the region into two
    if battery were excluded, and the centre component would then cover
    half the cartridge."""
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:50, 28:32] = CH_BATTERY        # a full-height bar across it
    keep = centre_component(m)
    assert keep[15, 15] and keep[15, 45], "region was severed by the cell"


def test_direct_placement_is_exactly_the_bay_channel():
    m = _cartridge()
    assert np.array_equal(direct_placement(m), m == CH_BAY)


def test_derived_placement_subtracts_electronics_obstruction_and_battery():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:20, 10:50] = CH_ELECTRONICS
    m[30:34, 30:34] = CH_OBSTRUCTION
    m[40:46, 12:18] = CH_BATTERY

    d = derived_placement(m, wall_inset_px=0)
    assert not d[15, 30], "electronics not subtracted"
    assert not d[32, 32], "obstruction not subtracted"
    assert not d[43, 15], "battery not subtracted"
    assert d[25, 25], "clear floor was removed"


def test_derived_placement_erodes_by_the_wall_inset():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    d0 = derived_placement(m, wall_inset_px=0)
    d5 = derived_placement(m, wall_inset_px=5)
    assert d5.sum() < d0.sum()
    assert not d5[10, 30], "the wall band survived the inset"
    assert d5[30, 30]


def test_p_safe_is_the_intersection_not_the_union():
    """Conservative on purpose: siting a cell on a PCB is a damage
    event, skipping a cartridge costs one cycle."""
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:20, 10:50] = CH_ELECTRONICS    # direct says bay, derived says no

    safe, iou = arbitrate(m, wall_inset_px=0)
    assert not safe[15, 30], "P_safe took the union"
    assert safe[30, 30]
    assert 0.0 <= iou <= 1.0


def test_iou_is_one_when_the_estimates_agree():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    _, iou = arbitrate(m, wall_inset_px=0)
    assert iou == pytest.approx(1.0)


def test_iou_falls_when_the_estimates_disagree():
    m = np.zeros((60, 60), np.int8)
    m[10:50, 10:50] = CH_BAY
    m[10:30, 10:50] = CH_ELECTRONICS    # half the bay is really PCB
    _, iou = arbitrate(m, wall_inset_px=0)
    assert iou < 0.6


def test_empty_estimates_give_zero_iou_not_a_crash():
    m = np.zeros((20, 20), np.int8)
    safe, iou = arbitrate(m, wall_inset_px=0)
    assert not safe.any()
    assert iou == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_arbitration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plan.arbitration'`

- [ ] **Step 3: Implement**

Create `plan/arbitration.py`:

```python
"""
plan.arbitration - reconciling two placement-area estimates.

The segmenter produces one estimate directly (its `bay` channel) and a
second derivable from the other channels. Both are modal - what the
camera can see - so they answer the same question two ways and can
disagree.

The planner consumes their INTERSECTION. The asymmetry is deliberate:
siting a cell on a PCB is a damage event, whereas skipping a cartridge
costs one cycle. Their IoU is the confidence signal that says how much
to trust the frame at all.

Pure NumPy and cv2. No torch: this runs inside the planning cycle, which
FDR O3 caps at 8 ms per cartridge.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except Exception as exc:                # pragma: no cover - hard dep
    raise ImportError("opencv-python is required") from exc


# Fixed order, matching recog.seg_dataset.SEG_CHANNELS. A test in
# tests/test_arbitration.py pins the two together.
CH_BACKGROUND = 0
CH_CARTRIDGE = 1
CH_BAY = 2
CH_ELECTRONICS = 3
CH_OBSTRUCTION = 4
CH_BATTERY = 5


def centre_component(label_map: np.ndarray) -> np.ndarray:
    """The connected component of non-background containing the centre.

    Cartridges sit adjacent in the jig, so a jittered or over-large crop
    catches a neighbour's edge; without this the derived estimate would
    erode the union of two cartridges.

    Connectivity spans ALL non-background channels, battery included. A
    cell lying across a bay would otherwise sever the region and leave
    the centre component covering half the cartridge.
    """
    fg = (label_map != CH_BACKGROUND).astype(np.uint8)
    if not fg.any():
        return fg.astype(bool)

    n, comps = cv2.connectedComponents(fg, connectivity=8)
    h, w = label_map.shape
    centre = comps[h // 2, w // 2]
    if centre == 0:
        # The crop centre landed on background - a badly-placed box.
        # Fall back to the largest component rather than returning
        # nothing, so one bad box does not silently void the cartridge.
        counts = np.bincount(comps.ravel())
        counts[0] = 0
        if counts.max() == 0:
            return np.zeros_like(fg, dtype=bool)
        centre = int(counts.argmax())
    return comps == centre


def direct_placement(label_map: np.ndarray) -> np.ndarray:
    """The network's own answer: the `bay` channel."""
    return label_map == CH_BAY


def derived_placement(label_map: np.ndarray,
                      wall_inset_px: int) -> np.ndarray:
    """Interior floor minus everything occupying it.

    Built from the channels the direct estimate does NOT use, so the two
    are genuinely independent and their disagreement carries
    information.
    """
    keep = centre_component(label_map)
    interior = keep.astype(np.uint8)

    if wall_inset_px > 0:
        k = 2 * int(wall_inset_px) + 1
        interior = cv2.erode(
            interior, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    out = interior.astype(bool)
    for ch in (CH_ELECTRONICS, CH_OBSTRUCTION, CH_BATTERY):
        out &= label_map != ch
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def arbitrate(label_map: np.ndarray,
              wall_inset_px: int) -> Tuple[np.ndarray, float]:
    """``(P_safe, iou)``.

    ``P_safe`` is the intersection: a pixel is placeable only where both
    estimates agree. ``iou`` is the per-cartridge confidence the caller
    gates on - see plan.placement_area and the calibrated threshold in
    docs/receipts/tau_calibration.txt.
    """
    direct = direct_placement(label_map)
    derived = derived_placement(label_map, wall_inset_px)
    return direct & derived, mask_iou(direct, derived)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_arbitration.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Pin the channel constants to Plan C's contract**

Append to `tests/test_arbitration.py`:

```python
def test_channel_constants_match_the_segmenter_contract():
    """plan.arbitration and recog.seg_dataset must agree, or the
    arbitration subtracts the wrong masks and nothing raises."""
    pytest.importorskip("torch")
    from recog.seg_dataset import SEG_CHANNELS

    assert SEG_CHANNELS == {
        "background": CH_BACKGROUND, "cartridge": CH_CARTRIDGE,
        "bay": CH_BAY, "electronics": CH_ELECTRONICS,
        "obstruction": CH_OBSTRUCTION, "battery": CH_BATTERY,
    }
```

Run: `pytest tests/test_arbitration.py -v`
Expected: PASS, 10 tests

- [ ] **Step 6: Commit**

```bash
git add plan/arbitration.py tests/test_arbitration.py
git commit -m "feat(plan): placement-area arbitration

Two modal estimates of the same region: the segmenter's bay channel, and
a derived interior minus electronics, obstruction and battery. The
planner gets their INTERSECTION - siting a cell on a PCB is a damage
event, skipping a cartridge costs a cycle.

Connectivity spans the battery channel. Excluding it lets a cell lying
across a bay sever the region, after which the centre component covers
half the cartridge.

Pure numpy and cv2, no torch: this runs inside the planning cycle, which
FDR O3 caps at 8 ms per cartridge."
```

---

### Task 2: The τ safety criterion

**Files:**
- Modify: `plan/arbitration.py`
- Test: `tests/test_arbitration.py` (append)

**Interfaces:**
- Produces: `admits_a_cell(error_mask, cell_w_px, cell_h_px) -> bool` — `True` if the region can contain a cell footprint at any packer orientation

The criterion is **morphological, not areal**. An earlier draft of the spec required the optimistic error area to fall below one cell footprint (1190 mm²); that is the wrong test, because 1190 mm² spread along a boundary as a one-pixel rim is harmless while the same area in one blob is a misplacement. What matters is whether the error region can *contain* a cell.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arbitration.py`:

```python
def test_a_cell_sized_blob_admits_a_cell():
    from plan.arbitration import admits_a_cell

    m = np.zeros((80, 80), bool)
    m[10:40, 10:24] = True              # 30 x 14, a cell is 29 x 13
    assert admits_a_cell(m, cell_w_px=13, cell_h_px=29)


def test_a_boundary_rim_of_equal_area_does_not():
    """The whole point of a morphological criterion: same area, one is
    a misplacement and the other is noise."""
    from plan.arbitration import admits_a_cell

    blob = np.zeros((80, 80), bool)
    blob[10:40, 10:24] = True
    rim = np.zeros((80, 80), bool)
    rim[5, 5:5 + blob.sum()] = True      # identical area, one pixel tall

    assert blob.sum() == rim.sum()
    assert admits_a_cell(blob, 13, 29)
    assert not admits_a_cell(rim, 13, 29)


def test_admits_a_cell_tries_both_orientations():
    from plan.arbitration import admits_a_cell

    m = np.zeros((80, 80), bool)
    m[10:24, 10:40] = True              # 14 tall, 30 wide - rotated
    assert admits_a_cell(m, cell_w_px=13, cell_h_px=29)


def test_empty_region_admits_nothing():
    from plan.arbitration import admits_a_cell

    assert not admits_a_cell(np.zeros((40, 40), bool), 13, 29)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_arbitration.py -v -k admits`
Expected: FAIL — `ImportError: cannot import name 'admits_a_cell'`

- [ ] **Step 3: Implement**

Append to `plan/arbitration.py`:

```python
def admits_a_cell(region: np.ndarray, cell_w_px: int,
                  cell_h_px: int) -> bool:
    """Can ``region`` contain a cell footprint at any packer orientation?

    This is the criterion τ is calibrated against, and it is
    MORPHOLOGICAL rather than areal on purpose. An earlier draft of the
    design spec tested whether the optimistic-error AREA exceeded one
    cell footprint. That is the wrong question: 1190 mm2 spread along a
    boundary as a one-pixel rim cannot hold a cell, while the same area
    in one blob is a misplacement waiting to happen.

    Eroding by the cell's own footprint answers the right question
    directly - a non-empty result means some position exists where the
    whole cell fits inside the error.
    """
    if not region.any():
        return False

    src = region.astype(np.uint8)
    for w, h in ((int(cell_w_px), int(cell_h_px)),
                 (int(cell_h_px), int(cell_w_px))):
        if w < 1 or h < 1:
            continue
        if w > region.shape[1] or h > region.shape[0]:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
        if cv2.erode(src, kernel).any():
            return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_arbitration.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add plan/arbitration.py tests/test_arbitration.py
git commit -m "feat(plan): morphological cell-admission test for tau

An earlier spec draft calibrated tau on the AREA of the optimistic error
against one cell footprint. Wrong question: 1190 mm2 spread along a
boundary as a one-pixel rim is harmless, the same area in one blob is a
misplacement.

Eroding the error region by the cell's own footprint at both packer
orientations answers it directly - a non-empty result means a position
exists where the whole cell fits inside the error."
```

---

### Task 3: `Snapshot` carries the masks

**Files:**
- Modify: `common/types.py:114-135`
- Test: `tests/test_common.py` (append)

**Interfaces:**
- Produces: `Snapshot.cartridge_masks: dict[int, Any]` — detection index → `(H, W)` int8 label map, default empty

Segmentation runs in Recognition (spec §3.2) because a DeepLabv3 forward is 12.6 ms unbatched against O3's 8 ms allowance. The masks then have to reach Planning, and `Snapshot` is the only value that crosses that boundary.

This modifies a module contract, which the README names as a design principle. The field is optional and defaulted, so every existing producer and consumer stays valid — and the alternative is a design that knowingly breaks a tested requirement.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common.py`:

```python
def test_snapshot_carries_optional_cartridge_masks():
    import numpy as np

    from common.types import Snapshot

    s = Snapshot()
    assert s.cartridge_masks == {}, "must default to empty, not None"

    s.cartridge_masks[0] = np.zeros((8, 8), np.int8)
    assert s.cartridge_masks[0].shape == (8, 8)


def test_snapshot_to_dict_summarises_masks_rather_than_embedding_them():
    """to_dict feeds logging and regression fixtures. Embedding a
    label map per cartridge would make every log line enormous."""
    import numpy as np

    from common.types import Snapshot

    s = Snapshot()
    s.cartridge_masks[3] = np.zeros((16, 32), np.int8)
    d = s.to_dict()
    assert d["cartridge_masks"] == {"3": [16, 32]}


def test_existing_snapshot_construction_is_unaffected():
    from common.types import BBox, ClassLabel, Detection, Snapshot

    s = Snapshot(detections=[
        Detection(BBox(0, 0, 4, 4), ClassLabel.BATTERY, 0.9)])
    assert len(s.of(ClassLabel.BATTERY)) == 1
    assert s.to_dict()["detections"][0]["label"] == "battery"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_common.py -v -k snapshot`
Expected: FAIL — `AttributeError: 'Snapshot' object has no attribute 'cartridge_masks'`

- [ ] **Step 3: Add the field**

In `common/types.py`, extend `Snapshot`:

```python
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

    def of(self, label: ClassLabel) -> list[Detection]:
        return [d for d in self.detections if d.label is label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_shape": list(self.image_shape),
            "timestamp_ns": int(self.timestamp_ns),
            # Shape only. to_dict feeds logging and regression fixtures;
            # embedding a full label map per cartridge would make every
            # log line enormous and every fixture unreadable.
            "cartridge_masks": {
                str(k): list(v.shape) for k, v in self.cartridge_masks.items()
            },
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_common.py -v`
Expected: PASS, including every existing test.

- [ ] **Step 5: Commit**

```bash
git add common/types.py tests/test_common.py
git commit -m "feat(common): Snapshot carries per-cartridge label maps

Segmentation runs in Recognition, not Planning: a DeepLabv3 forward is
~12.6 ms for one cartridge against FDR O3's tested 8 ms per-cartridge
planning budget, so running it in the planner would break a requirement
the suite checks.

The masks then have to cross the module boundary, and Snapshot is the
boundary. Optional and defaulted, so every existing producer and
consumer stays valid.

to_dict records shapes only - it feeds logging and fixtures, and
embedding a label map per cartridge would make both unreadable."
```

---

### Task 4: `SegmentationPlacementAreaExtractor`

**Files:**
- Modify: `plan/placement_area.py`
- Test: `tests/test_placement_area.py` (append)

**Interfaces:**
- Consumes: `plan.arbitration.arbitrate`, `Snapshot.cartridge_masks`
- Produces:
  - `PlacementArea` gains `bay_mask`, `electronics_mask`, `obstruction_mask`, `consistency_iou`
  - `SegmentationPlacementAreaExtractor(mm_per_cell, mm_per_px, wall_inset_mm, tau)` with the same `extract(...)` signature plus `label_map=`
  - `HeuristicPlacementAreaExtractor` — the existing class, renamed, with a scope limit and a construction warning

The heuristic implementation is **not** an equal alternative. Spec §1.1 measured it returning zero placeable area on 7 of 20 real cartridges. Its role narrows to keeping the torch-free demo runnable on `synth_dataset.py`'s flat green rectangles — the hardware its PPR §5.3.2 assumption actually describes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_placement_area.py`:

```python
def test_heuristic_extractor_warns_about_its_scope_limit():
    """It returns zero placeable area on 7 of 20 real cartridges
    (spec 1.1). It must not be selectable for real imagery silently."""
    import warnings

    from plan.placement_area import HeuristicPlacementAreaExtractor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HeuristicPlacementAreaExtractor()
    assert any("green" in str(w.message).lower() for w in caught)


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
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0, tau=0.0)
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
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0, tau=0.0)
    pa = ex.extract(np.zeros((200, 200, 3), np.uint8),
                    BBox(20, 30, 80, 110), label_map=label)

    assert pa.rectangle.xmin >= 20 and pa.rectangle.ymin >= 30
    assert pa.rectangle.xmax <= 80 and pa.rectangle.ymax <= 110


def test_low_agreement_raises_so_the_planner_skips_the_cartridge():
    import numpy as np
    import pytest

    from common.types import BBox
    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS
    from plan.placement_area import (PlacementDisagreement,
                                     SegmentationPlacementAreaExtractor)

    label = np.zeros((80, 60), np.int8)
    label[5:75, 5:55] = CH_CARTRIDGE
    label[12:68, 12:48] = CH_BAY
    label[12:50, 12:48] = CH_ELECTRONICS      # most of "bay" is really PCB

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=0.0, tau=0.95)
    with pytest.raises(PlacementDisagreement):
        ex.extract(np.zeros((200, 200, 3), np.uint8),
                   BBox(20, 30, 80, 110), label_map=label)


def test_missing_label_map_raises_rather_than_silently_degrading():
    import numpy as np
    import pytest

    from common.types import BBox
    from plan.placement_area import SegmentationPlacementAreaExtractor

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0, tau=0.0)
    with pytest.raises(ValueError, match="label_map"):
        ex.extract(np.zeros((200, 200, 3), np.uint8), BBox(0, 0, 40, 40))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_placement_area.py -v -k "heuristic_extractor_warns or segmentation_extractor or disagreement or label_map"`
Expected: FAIL — `ImportError: cannot import name 'HeuristicPlacementAreaExtractor'`

- [ ] **Step 3: Extend `PlacementArea` and rename the heuristic class**

In `plan/placement_area.py`, extend the dataclass with defaulted fields so existing constructions keep working:

```python
@dataclass
class PlacementArea:
    """Output of :meth:`PlacementAreaExtractor.extract`."""

    rectangle: BBox           # image-space axis-aligned inset bbox
    inside_mask: np.ndarray   # uint8 0/1 mask inside cartridge ROI
    pcb_mask: np.ndarray      # uint8 0/1 mask of excluded PCB region
    occupancy: OccupancyGrid  # rasterised placement grid
    mm_per_cell: float
    # Segmentation path only; the heuristic path leaves these empty.
    # pcb_mask above is retained and populated with electronics_mask, so
    # plan/planner.py's existing assignment keeps working unchanged.
    bay_mask: Optional[np.ndarray] = None
    electronics_mask: Optional[np.ndarray] = None
    obstruction_mask: Optional[np.ndarray] = None
    consistency_iou: float = 1.0
```

Rename `PlacementAreaExtractor` to `HeuristicPlacementAreaExtractor`, keep `PlacementAreaExtractor = HeuristicPlacementAreaExtractor` as an alias so `main.py:124` and `plan/planner.py:29` keep importing successfully, and add the scope limit to its docstring plus a warning in `__init__`:

```python
    def __init__(self, ...) -> None:
        warnings.warn(
            "HeuristicPlacementAreaExtractor assumes a LIGHT tray with a "
            "DARK interior module (PPR 5.3.2). On the black cartridges in "
            "recog/realtest/ it returns zero placeable area for 7 of 20 "
            "cartridges, because Otsu on the green channel selects "
            "whichever foreign matter is brightest and the gray<80 "
            "threshold classifies the black bay as PCB. Use "
            "SegmentationPlacementAreaExtractor for real imagery; this "
            "path exists for the torch-free demo on synth_dataset.py's "
            "green rectangles.",
            RuntimeWarning, stacklevel=2,
        )
        ...
```

Add `import warnings` and `from typing import Optional`.

- [ ] **Step 4: Implement the segmentation extractor**

Append to `plan/placement_area.py`:

```python
class PlacementDisagreement(RuntimeError):
    """The two placement estimates disagreed beyond tau.

    plan/planner.py's _ensure_placement_areas already catches every
    exception and leaves the cartridge unplanned for the cycle, so
    raising is exactly the behaviour wanted: the cartridge is skipped
    and retried next frame.
    """


class SegmentationPlacementAreaExtractor:
    """Placement area from a segmenter's label map.

    Does mask arithmetic only. The model runs in Recognition and its
    output arrives on Snapshot.cartridge_masks - FDR O3 caps planning at
    8 ms per cartridge and a DeepLabv3 forward is ~12.6 ms.
    """

    def __init__(self, mm_per_cell: float = 1.5, mm_per_px: float = 0.625,
                 wall_inset_mm: float = 4.0, tau: float = 0.85) -> None:
        self.mm_per_cell = float(mm_per_cell)
        self.mm_per_px = float(mm_per_px)
        self.wall_inset_mm = float(wall_inset_mm)
        self.tau = float(tau)

    @property
    def wall_inset_px(self) -> int:
        """Wall thickness in pixels at the CURRENT framing.

        Not the old hardcoded safety_margin_px = 5: a fixed pixel inset
        is wrong at two different camera framings. mm_per_px must be the
        calibrated scale for the deployed mount, not a config default.
        """
        return max(0, int(round(self.wall_inset_mm / self.mm_per_px)))

    def extract(self, image_rgb: np.ndarray, cartridge_bbox: BBox,
                pcb_template_mask: Optional[np.ndarray] = None,
                label_map: Optional[np.ndarray] = None) -> PlacementArea:
        from plan.arbitration import (CH_ELECTRONICS, CH_OBSTRUCTION,
                                      arbitrate, direct_placement)

        if label_map is None:
            raise ValueError(
                "SegmentationPlacementAreaExtractor needs a label_map "
                "from Snapshot.cartridge_masks. Falling back to the "
                "heuristic silently would hide a perception failure "
                "behind a plausible-looking rectangle.")

        safe, iou = arbitrate(label_map, self.wall_inset_px)
        if iou < self.tau:
            raise PlacementDisagreement(
                f"placement estimates disagree: IoU {iou:.3f} < "
                f"tau {self.tau:.3f}")

        ys, xs = np.nonzero(safe)
        if xs.size == 0:
            raise RuntimeError("no placeable area in this cartridge")

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

        inside = safe.astype(np.uint8)
        electronics = (label_map == CH_ELECTRONICS).astype(np.uint8)
        obstruction = (label_map == CH_OBSTRUCTION).astype(np.uint8)

        occupancy = _rasterise_mask(
            inside, (x0, y0, x1, y1), self.mm_per_cell, self.mm_per_px)

        ox, oy = int(cartridge_bbox.xmin), int(cartridge_bbox.ymin)
        return PlacementArea(
            rectangle=BBox(ox + x0, oy + y0, ox + x1, oy + y1),
            inside_mask=inside,
            pcb_mask=electronics,          # backward compatibility
            occupancy=occupancy,
            mm_per_cell=self.mm_per_cell,
            bay_mask=direct_placement(label_map).astype(np.uint8),
            electronics_mask=electronics,
            obstruction_mask=obstruction,
            consistency_iou=iou,
        )
```

Extract the existing `_rasterise` body into a module-level `_rasterise_mask(inside_mask, rect, mm_per_cell, mm_per_px)` and have both classes call it, so the two implementations produce identically-shaped grids by construction rather than by coincidence.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_placement_area.py -v`
Expected: PASS, including every existing test. If the existing tests now emit `RuntimeWarning`, that is the scope-limit warning firing correctly — add `filterwarnings` to those tests rather than removing the warning.

- [ ] **Step 6: Wire the label map through the planner**

In `plan/planner.py`'s `_ensure_placement_areas`, pass the mask when the extractor accepts one:

```python
    def _ensure_placement_areas(self, image_rgb: np.ndarray,
                                snapshot=None) -> None:
        """Fill in the placement rectangle / occupancy for new cartridges."""
        for ctg in self.env.cartridges.values():
            if ctg.placeable_rectangle is not None:
                continue
            try:
                kwargs = {}
                if snapshot is not None and ctg.detection_index in \
                        snapshot.cartridge_masks:
                    kwargs["label_map"] = \
                        snapshot.cartridge_masks[ctg.detection_index]
                pa = self.extractor.extract(image_rgb, ctg.bbox, **kwargs)
            except Exception:
                # Leave unplanned this cycle - it'll retry next frame.
                # PlacementDisagreement lands here too, which is the
                # intended behaviour for a low-confidence cartridge.
                continue
            ctg.placeable_rectangle = pa.rectangle
            ctg.occupancy = pa.occupancy
            ctg.pcb_mask = pa.pcb_mask
```

The blanket `except Exception` was already there and already means "skip this cartridge, retry next frame" — but it will now also swallow a `PlacementDisagreement` without recording it. Add a counter so the rate is observable; a safety interlock that fires invisibly is not a safety interlock. `Cartridge` needs a `detection_index` field linking it back to the `Snapshot` detection it came from.

- [ ] **Step 7: Commit**

```bash
git add plan/placement_area.py plan/planner.py plan/scene.py tests/test_placement_area.py
git commit -m "feat(plan): SegmentationPlacementAreaExtractor

Same extract() contract as the heuristic, so planner.py and main.py are
unchanged and either implementation can be selected. Does mask
arithmetic only - the model runs in Recognition, because FDR O3 caps
planning at 8 ms per cartridge and a DeepLabv3 forward is ~12.6 ms.

Disagreement beyond tau raises PlacementDisagreement, which
_ensure_placement_areas already handles as skip-and-retry-next-frame.
The rate is counted rather than silently swallowed: an interlock that
fires invisibly is not an interlock.

The heuristic extractor gains a construction warning naming its scope
limit. It returns zero placeable area on 7 of 20 real cartridges and
must not be selected for real imagery by accident."
```

---

### Task 5: Recognition runs the segmenter, batched

**Files:**
- Modify: `recog/inference.py`
- Test: `tests/test_inference.py` (append)

**Interfaces:**
- Consumes: `BaySegmenter.segment_batch` (Plan C Task 3)
- Produces: `FasterRCNNDetector(..., segmenter=None)` — when a segmenter is supplied, `detect` fills `Snapshot.cartridge_masks`

All cartridge crops in a frame go through **one** `segment_batch` call. Looping cost 101 ms for eight cartridges against 18.5 ms batched (spec §3.2); the 50 ms budget does not survive the loop.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inference.py`:

```python
def test_detector_without_a_segmenter_leaves_masks_empty():
    """The existing single-stage path must be completely unaffected."""
    import numpy as np

    from recog.inference import HeuristicDetector

    det = HeuristicDetector()
    snap = det.detect(np.zeros((240, 320, 3), np.uint8))
    assert snap.cartridge_masks == {}


def test_segmenter_is_called_once_per_frame_not_once_per_cartridge():
    """Batching is load-bearing: 8 cartridges cost 101 ms looped and
    18.5 ms batched, against a 50 ms end-to-end budget."""
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _SpySegmenter:
        def __init__(self):
            self.calls = 0
            self.batch_sizes = []

        def segment_batch(self, crops):
            self.calls += 1
            self.batch_sizes.append(len(crops))
            return [np.zeros(c.shape[:2], np.int8) for c in crops]

    spy = _SpySegmenter()
    snap = Snapshot(detections=[
        Detection(BBox(10, 10, 50, 60), ClassLabel.CARTRIDGE, 0.9),
        Detection(BBox(70, 10, 110, 60), ClassLabel.CARTRIDGE, 0.9),
        Detection(BBox(10, 80, 30, 100), ClassLabel.BATTERY, 0.9),
    ])
    attach_cartridge_masks(snap, np.zeros((200, 200, 3), np.uint8), spy)

    assert spy.calls == 1, f"segmenter called {spy.calls} times, not once"
    assert spy.batch_sizes == [2], "batteries must not be segmented"
    assert set(snap.cartridge_masks) == {0, 1}


def test_masks_are_keyed_by_detection_index():
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _Seg:
        def segment_batch(self, crops):
            return [np.full(c.shape[:2], 2, np.int8) for c in crops]

    snap = Snapshot(detections=[
        Detection(BBox(10, 80, 30, 100), ClassLabel.BATTERY, 0.9),
        Detection(BBox(10, 10, 50, 60), ClassLabel.CARTRIDGE, 0.9),
    ])
    attach_cartridge_masks(snap, np.zeros((200, 200, 3), np.uint8), _Seg())

    assert set(snap.cartridge_masks) == {1}, (
        "index 1 is the cartridge; keying by position within the "
        "cartridge subset would misalign every mask")
    assert snap.cartridge_masks[1].shape == (50, 40)


def test_no_cartridges_means_no_segmenter_call():
    import numpy as np

    from common.types import BBox, ClassLabel, Detection, Snapshot
    from recog.inference import attach_cartridge_masks

    class _Seg:
        def __init__(self):
            self.calls = 0

        def segment_batch(self, crops):
            self.calls += 1
            return []

    seg = _Seg()
    snap = Snapshot(detections=[
        Detection(BBox(0, 0, 4, 4), ClassLabel.BATTERY, 0.9)])
    attach_cartridge_masks(snap, np.zeros((50, 50, 3), np.uint8), seg)
    assert seg.calls == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_inference.py -v -k "segmenter or cartridge_masks or detection_index"`
Expected: FAIL — `ImportError: cannot import name 'attach_cartridge_masks'`

- [ ] **Step 3: Implement**

Append to `recog/inference.py`:

```python
def attach_cartridge_masks(snapshot, image_rgb, segmenter) -> None:
    """Segment every cartridge in `snapshot`, in ONE batched call.

    Keys are indices into `snapshot.detections`, not positions within the
    cartridge subset: the planner looks a cartridge up by the detection
    it came from, and keying by subset position misaligns every mask the
    moment a battery appears before a cartridge in the list.

    One call, not a loop. Measured on an RTX 3060: eight cartridges cost
    101 ms looped and 18.5 ms batched at fp16/256, against FDR 10.4's
    50 ms end-to-end budget.
    """
    from common.types import ClassLabel

    idx, crops = [], []
    h, w = image_rgb.shape[:2]
    for i, det in enumerate(snapshot.detections):
        if det.label is not ClassLabel.CARTRIDGE:
            continue
        x0 = max(0, int(det.bbox.xmin))
        y0 = max(0, int(det.bbox.ymin))
        x1 = min(w, int(det.bbox.xmax))
        y1 = min(h, int(det.bbox.ymax))
        if x1 <= x0 or y1 <= y0:
            continue
        idx.append(i)
        crops.append(image_rgb[y0:y1, x0:x1])

    if not crops:
        return

    for i, mask in zip(idx, segmenter.segment_batch(crops)):
        snapshot.cartridge_masks[i] = mask
```

Then give `FasterRCNNDetector.__init__` an optional `segmenter=None`, and call `attach_cartridge_masks(snap, image_rgb, self.segmenter)` at the end of `detect` when it is set. `HeuristicDetector` is left alone — it has no segmenter and the torch-free path must stay torch-free.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_inference.py -v`
Expected: PASS, including every existing test.

- [ ] **Step 5: Commit**

```bash
git add recog/inference.py tests/test_inference.py
git commit -m "feat(recog): attach batched cartridge masks to the Snapshot

One segment_batch call per frame, not one per cartridge. Measured on an
RTX 3060, eight cartridges cost 101 ms looped and 18.5 ms batched, and
the 50 ms end-to-end budget does not survive the loop.

Masks are keyed by index into snapshot.detections, not by position
within the cartridge subset - the latter misaligns every mask as soon as
a battery appears before a cartridge in the list.

HeuristicDetector is untouched; the torch-free demo path stays
torch-free."
```

---

### Task 6: Calibrate τ

**Files:**
- Create: `recog/calibrate_tau.py`
- Test: `tests/test_arbitration.py` (append)

**Interfaces:**
- Consumes: `arbitrate`, `admits_a_cell`, the validation split, a trained checkpoint
- Produces:
  - `calibrate(records, cell_w_px, cell_h_px, fail_budget=0.05) -> dict`
  - CLI writing `docs/receipts/tau_calibration.txt`

τ is the **smallest** threshold at which at most `fail_budget` of accepted cartridges admit a cell inside their optimistic error. Smallest, because a larger τ rejects more cartridges — so this maximises throughput subject to the safety bound.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arbitration.py`:

```python
def test_calibrate_picks_the_smallest_tau_meeting_the_budget():
    from recog.calibrate_tau import calibrate

    # (iou, admits_a_cell_in_the_optimistic_error)
    records = [(0.99, False)] * 90 + [(0.50, True)] * 10
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert 0.50 < out["tau"] <= 0.99
    assert out["fail_rate"] <= 0.05
    assert out["rejected_fraction"] == pytest.approx(0.10, abs=0.02)


def test_calibrate_reports_failure_when_no_tau_works():
    """A negative result about the design, reported as one rather than
    tuned around."""
    records = [(0.99, True)] * 100
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert out["tau"] is None
    assert "no threshold" in out["note"].lower()


def test_calibrate_maximises_throughput_subject_to_safety():
    """Two thresholds both meet the budget; the smaller must win
    because it accepts more cartridges."""
    records = [(0.70, False)] * 50 + [(0.95, False)] * 50
    out = calibrate_from_pairs(records, fail_budget=0.05)
    assert out["tau"] <= 0.70
    assert out["rejected_fraction"] == pytest.approx(0.0)


def calibrate_from_pairs(pairs, fail_budget):
    """Adapter so the tests read as data rather than as fixtures."""
    from recog.calibrate_tau import calibrate
    return calibrate([{"iou": i, "admits_cell": a} for i, a in pairs],
                     fail_budget=fail_budget)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_arbitration.py -v -k calibrate`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.calibrate_tau'`

- [ ] **Step 3: Implement**

Create `recog/calibrate_tau.py`:

```python
"""
recog.calibrate_tau - choose the placement-disagreement threshold.

tau is calibrated, never asserted. A hardcoded 0.85 that was never
measured is a placeholder wearing a number's clothes.

For each validation cartridge, record the arbitration IoU and whether
the OPTIMISTIC error - area P_safe claims is placeable where the ground
truth says it is not - can contain a whole cell. The second is the
morphological test in plan.arbitration.admits_a_cell, not an area
threshold: error spread along a boundary cannot hold a cell, the same
area in one blob can.

tau is then the SMALLEST threshold at which at most `fail_budget` of
accepted cartridges admit a cell. Smallest, because a larger tau rejects
more cartridges - so this maximises throughput subject to the safety
bound rather than simply being safe by refusing everything.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def calibrate(records: Sequence[dict], fail_budget: float = 0.05) -> Dict:
    """Pick tau from `records` of {"iou": float, "admits_cell": bool}."""
    if not records:
        return {"tau": None, "note": "no records", "n": 0}

    candidates = sorted({round(float(r["iou"]), 4) for r in records})
    n = len(records)

    for tau in candidates:
        accepted = [r for r in records if float(r["iou"]) >= tau]
        if not accepted:
            continue
        fails = sum(1 for r in accepted if r["admits_cell"])
        rate = fails / len(accepted)
        if rate <= fail_budget:
            return {
                "tau": float(tau),
                "fail_rate": rate,
                "n": n,
                "n_accepted": len(accepted),
                "rejected_fraction": 1.0 - len(accepted) / n,
                "fail_budget": fail_budget,
                "note": (f"smallest tau meeting the budget; "
                         f"{len(accepted)}/{n} cartridges accepted"),
            }

    return {
        "tau": None,
        "n": n,
        "fail_budget": fail_budget,
        "note": ("No threshold meets the budget: even the most confident "
                 "cartridges admit a cell inside their optimistic error. "
                 "This is a negative result about the segmenter, and is "
                 "reported rather than tuned around."),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_arbitration.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Add the CLI and produce the receipt**

Add a CLI that loads the trained checkpoint and the validation split, runs `arbitrate` per cartridge against the ground-truth label map, computes the optimistic error `P_safe & ~gt_bay`, tests it with `admits_a_cell` at the cell footprint in pixels (`18.3 / mm_per_px` by `65.0 / mm_per_px`), and writes `docs/receipts/tau_calibration.txt`.

Run:
```bash
python -m recog.calibrate_tau --checkpoint recog/checkpoints/seg/best.pt --config configs/segmentation.yaml
```

Put the resulting τ into `configs/planning.yaml`. **If `tau` comes back `None`, stop and report it** — that is the calibration telling you the segmenter is not safe at any confidence level, and the correct response is to improve the segmenter, not to lower the budget until a number appears.

- [ ] **Step 6: Commit**

```bash
git add recog/calibrate_tau.py configs/planning.yaml docs/receipts/tau_calibration.txt tests/test_arbitration.py
git commit -m "feat(recog): calibrate the placement-disagreement threshold

tau is measured, not asserted. For each validation cartridge, record the
arbitration IoU and whether the optimistic error can contain a whole
cell, then take the SMALLEST tau at which at most 5% of accepted
cartridges fail - smallest because a larger tau rejects more cartridges,
so this maximises throughput subject to the safety bound.

Reports tau=None rather than a tuned number when no threshold works.
That is a negative result about the segmenter and belongs in the report."
```

---

### Task 7: Budget verification and end-to-end run

**Files:**
- Test: `tests/test_planner.py` (append), `tests/test_main_integration.py` (append)
- Modify: `configs/demo.yaml`, `README.md`

- [ ] **Step 1: Write the budget tests**

Append to `tests/test_planner.py`:

```python
def test_planning_stays_under_the_o3_budget_with_masks_supplied():
    """FDR O3: queue rebuild <= 8 ms per cartridge. The segmenter runs
    in Recognition precisely so this holds - Planning does mask
    arithmetic only."""
    import time

    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from plan.placement_area import SegmentationPlacementAreaExtractor

    label = np.zeros((288, 131), np.int8)
    label[5:283, 5:126] = CH_CARTRIDGE
    label[12:276, 12:119] = CH_BAY

    ex = SegmentationPlacementAreaExtractor(
        mm_per_cell=1.5, mm_per_px=0.625, wall_inset_mm=4.0, tau=0.0)
    img = np.zeros((720, 1280, 3), np.uint8)
    from common.types import BBox
    box = BBox(100, 100, 231, 388)

    ex.extract(img, box, label_map=label)          # warm caches
    t0 = time.perf_counter()
    for _ in range(20):
        ex.extract(img, box, label_map=label)
    per_call_ms = (time.perf_counter() - t0) / 20 * 1000

    assert per_call_ms < 8.0, (
        f"{per_call_ms:.1f} ms per cartridge breaks the O3 budget")
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_planner.py -v -k o3_budget`
Expected: PASS. If it fails, the arithmetic in `arbitrate` or `_rasterise_mask` is too slow — `_rasterise` is a nested Python loop over grid cells (`plan/placement_area.py:220-226`) and is the first place to look. Vectorising it with NumPy fancy indexing is the fix, not raising the threshold.

- [ ] **Step 3: Verify the torch-free demo still runs**

Run: `python main.py --config configs/demo.yaml`
Expected: the loop runs to completion with per-cycle latencies logged, exactly as before. This is the regression that matters most — the demo is what the FDR's reproducibility claim rests on.

- [ ] **Step 4: Add an end-to-end integration test**

Append to `tests/test_main_integration.py` a test that runs one full cycle with a stub segmenter attached, asserting that `Snapshot.cartridge_masks` is populated, that the planner produces a queue, and that a cartridge whose masks disagree below τ is skipped rather than crashing the cycle.

- [ ] **Step 5: Update the README**

Add `recog/bay_segmenter.py`, `recog/seg_dataset.py`, `recog/seg_training.py`, `recog/seg_evaluate.py`, `plan/arbitration.py` and `recog/calibrate_tau.py` to the repository-layout tree, and add their entry points to the "Other entry points" table. State plainly that the heuristic extractor is the demo-only path and why.

- [ ] **Step 6: Commit**

```bash
git add tests/test_planner.py tests/test_main_integration.py README.md configs/demo.yaml
git commit -m "test: verify the O3 budget holds with segmentation masks

Planning does mask arithmetic only - the model runs in Recognition - and
this pins that: 8 ms per cartridge, the same bound FDR O3 states and
test_bin_packing already checks for the packer.

Also verifies the torch-free demo path is unaffected, which is what the
FDR's reproducibility claim rests on."
```

---

---

### Task 8: Δcells and the real-photo ablation

**Files:**
- Create: `recog/seg_ablation.py`
- Test: `tests/test_arbitration.py` (append)

**Interfaces:**
- Consumes: `common.packing.first_fit_decreasing` (needs **Plan A**), `plan.arbitration.arbitrate`, both extractors
- Produces:
  - `delta_cells(gt_label_map, pred_label_map, mm_per_px, wall_inset_mm=4.0) -> int`
  - `heuristic_vs_segmenter(real_dir, segmenter, cfg) -> dict`
  - CLI writing `docs/receipts/seg_ablation.txt`

Spec §9 calls Δcells "the end-to-end number", and it is the only metric stated in the unit the project actually cares about: cells in cartridges. Mask IoU can improve while Δcells gets worse, if the error simply moves to where a cell would have gone.

**This task depends on Plan A.** Measured on the unfixed packer, Δcells is dominated by the shelf-cursor defect rather than by mask error — FDR §6.3.1 records 23.0 cells at 0 % forbidden coverage against 3.2 at 2.5 % — so the number would say nothing about the segmenter. Every other metric in Plan C Task 5 is measurable without Plan A; only this one is gated.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arbitration.py`:

```python
def test_delta_cells_is_zero_for_a_perfect_prediction():
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE
    from recog.seg_ablation import delta_cells

    m = np.zeros((288, 131), np.int8)
    m[5:283, 5:126] = CH_CARTRIDGE
    m[12:276, 12:119] = CH_BAY
    assert delta_cells(m, m.copy(), mm_per_px=0.625) == 0


def test_delta_cells_is_positive_when_the_prediction_loses_room():
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_OBSTRUCTION
    from recog.seg_ablation import delta_cells

    gt = np.zeros((288, 131), np.int8)
    gt[5:283, 5:126] = CH_CARTRIDGE
    gt[12:276, 12:119] = CH_BAY

    pred = gt.copy()
    pred[12:150, 12:119] = CH_OBSTRUCTION      # half the bay hallucinated

    assert delta_cells(gt, pred, mm_per_px=0.625) > 0


def test_delta_cells_is_negative_when_the_prediction_claims_too_much():
    """An optimistic prediction packs MORE cells than truth allows. A
    negative delta is the damage case, not a good score."""
    import numpy as np

    from plan.arbitration import CH_BAY, CH_CARTRIDGE, CH_ELECTRONICS
    from recog.seg_ablation import delta_cells

    gt = np.zeros((288, 131), np.int8)
    gt[5:283, 5:126] = CH_CARTRIDGE
    gt[12:276, 12:119] = CH_BAY
    gt[12:100, 12:119] = CH_ELECTRONICS

    pred = gt.copy()
    pred[12:100, 12:119] = CH_BAY              # PCB predicted as placeable

    assert delta_cells(gt, pred, mm_per_px=0.625) < 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_arbitration.py -v -k delta_cells`
Expected: FAIL — `ModuleNotFoundError: No module named 'recog.seg_ablation'`

- [ ] **Step 3: Implement `delta_cells`**

Create `recog/seg_ablation.py`:

```python
"""
recog.seg_ablation - the end-to-end numbers.

delta_cells is the metric stated in the unit the project cares about:
cells in cartridges. Mask IoU can improve while delta_cells gets worse,
if the error simply moves to where a cell would have gone.

Requires the packer fix (FDR 6.3.1, Plan A). On the unfixed shelf-cursor
logic the figure is dominated by the packer abandoning shelves - 23.0
cells at 0% forbidden coverage against 3.2 at 2.5% - and would say
nothing about the segmenter at all.

Sign convention: POSITIVE means the prediction lost cells the ground
truth would have placed (conservative, a throughput cost). NEGATIVE
means the prediction packed cells the ground truth forbids (optimistic,
a damage risk). Negative is the serious direction.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

CELL_W_MM = 18.3
CELL_H_MM = 65.0


def _pack_count(label_map: np.ndarray, mm_per_px: float,
                wall_inset_mm: float = 4.0) -> int:
    from common.packing import Item, first_fit_decreasing
    from plan.arbitration import arbitrate

    inset_px = max(0, int(round(wall_inset_mm / mm_per_px)))
    safe, _ = arbitrate(label_map, inset_px)
    if not safe.any():
        return 0

    ys, xs = np.nonzero(safe)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    strip_w = (x1 - x0) * mm_per_px
    strip_h = (y1 - y0) * mm_per_px

    mm_per_cell = 1.5
    px_per_cell = max(1, int(round(mm_per_cell / mm_per_px)))
    sub = safe[y0:y1, x0:x1][::px_per_cell, ::px_per_cell]
    forbidden = (~sub).astype(np.uint8)

    n_max = max(4, int(strip_w * strip_h / (CELL_W_MM * CELL_H_MM)) * 2)
    items = [Item(id=i, width=CELL_W_MM, height=CELL_H_MM)
             for i in range(n_max)]
    return first_fit_decreasing(
        items, strip_w, strip_h, allow_rotation=True,
        forbidden_mask=forbidden, mm_per_cell=mm_per_cell).count


def delta_cells(gt_label_map: np.ndarray, pred_label_map: np.ndarray,
                mm_per_px: float, wall_inset_mm: float = 4.0) -> int:
    """Cells the packer places on truth, minus cells on the prediction.

    See the module docstring for the sign convention.
    """
    return (_pack_count(gt_label_map, mm_per_px, wall_inset_mm)
            - _pack_count(pred_label_map, mm_per_px, wall_inset_mm))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_arbitration.py -v -k delta_cells`
Expected: PASS, 3 tests

- [ ] **Step 5: Add the real-photo ablation**

Spec §9's headline comparison is heuristic versus segmenter on the real photographs, scored as placeable-area IoU against human polygons. The heuristic's baseline is already measured and is the number to beat: **mean placeable fraction 0.218, with 7 of 20 cartridges at exactly zero**.

Append `heuristic_vs_segmenter` to `recog/seg_ablation.py`. For each cartridge in the re-annotated `recog/realtest/` set (spec §9.1 tier 1):

- run `HeuristicPlacementAreaExtractor` on the ROI, suppressing its `RuntimeWarning` since firing it twenty times is noise here — but record in the report that it fired;
- run the segmenter and `SegmentationPlacementAreaExtractor` with τ set to 0, so nothing is skipped and the comparison covers every cartridge;
- score both against the human `placement_area` polygon as IoU, and report signed area error for each.

Report per-cartridge rows plus the summary, and **carry the instance count beside every number**. With 20 cartridges this is a smoke test, not a claim — spec §9.1 says so, and the receipt must repeat it, because a table of numbers detached from its sample size gets quoted as though it were a result.

Write `docs/receipts/seg_ablation.txt`.

- [ ] **Step 6: Run it**

Run: `python -m recog.seg_ablation --checkpoint recog/checkpoints/seg/best.pt --real recog/realtest --config configs/segmentation.yaml`

Expected: the segmenter beats mean IoU 0.218 and has no zero-area cartridges.

**If it does not, that is the result.** Report it rather than adjusting the comparison until it looks better. A segmenter that cannot beat a measured baseline is not worth shipping, and the report is better for saying so.

- [ ] **Step 7: Commit**

```bash
git add recog/seg_ablation.py docs/receipts/seg_ablation.txt tests/test_arbitration.py
git commit -m "feat(recog): delta-cells and the heuristic-vs-segmenter ablation

delta_cells is the end-to-end number, stated in the unit the project
cares about: cells in cartridges. Mask IoU can improve while this gets
worse, if the error moves to where a cell would have gone. Positive
means cells lost (throughput); negative means cells packed where truth
forbids (damage), which is the serious direction.

Requires Plan A: on the unfixed shelf-cursor logic the figure is
dominated by the packer abandoning shelves, not by mask error.

The real-photo ablation scores both extractors against human polygons.
The heuristic's baseline is already measured - mean placeable fraction
0.218, 7 of 20 cartridges at zero - so the comparison is against a
number rather than an expectation."
```

---

### Task 9: Update the FDR

**Files:**
- Modify: `docs/FDR_v3.md` §6.2, §10.6, §13.2(5)
- Modify: `docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md` (status header)

Spec §12 lists four FDR edits. Plan A Task 4 made the §13.2(2) one. These are the other three, and they come last deliberately: each states a measured result, and none can be written honestly before the measurement exists.

- [ ] **Step 1: Rewrite §13.2(5)**

It currently proposes "an instance-segmentation extension … a pixel-precise placement mask predicted by a Mask R-CNN head", motivated by "removing the 2-D rectangular approximation".

Replace both halves. The motivation becomes the measured result of spec §1.1 — across 20 held-out cartridges the extractor returns **zero** placeable area for 7, and for the rest a region determined by incidental foreign matter. The architecture becomes a detector plus per-ROI segmenter, with the boundary-displacement figure from Plan C Task 5 as evidence, and the 28×28 mask head recorded as torchvision's configurable *default* rather than a fixed ceiling.

If this work is complete when the FDR is revised, move the item out of §13.2 entirely and into the results sections. Leaving finished work in a future-work list misrepresents the project.

- [ ] **Step 2: Add the scope limit to §6.2**

§6.2 describes the green-channel extractor as though it were general. Add: the pipeline assumes a light-coloured tray with a dark interior module, per PPR §5.3.2, and does not hold on the black cartridges in `recog/realtest/`. Include spec §1.1's measured table — 20 cartridges, mean placeable fraction 0.218, 7 at exactly zero, maximum `pcb_mask` fraction 0.718 — and the two-stage mechanism: Otsu on the green channel selects whichever foreign matter is brightest, and the fixed `gray < 80` threshold classifies the black bay as PCB.

State it as a **scope limit rather than a defect**. The assumption is self-consistent and was inherited from the PPR; the hardware simply is not the hardware it describes. That distinction is worth drawing precisely — it is the difference between a coding error and an unstated precondition, and only the second is what happened here.

- [ ] **Step 3: Extend §10.6's failure taxonomy**

Add `placement_disagreement` alongside `empty_queue` and `pick_failed`, with the observed rate from Plan D Task 6's calibration receipt and the calibrated τ.

Note that it is a *deliberate skip* rather than a failure: the cartridge is retried next frame, and the alternative to skipping is placing a cell somewhere two independent estimates could not agree was safe.

- [ ] **Step 4: Update the spec's status header**

Mark the design spec as implemented, with pointers to the four plans and to the receipts each produced — `forbidden_bench.csv`, `seg_eval.txt`, `tau_calibration.txt`, `seg_ablation.txt`.

Where a measurement contradicted the design, say so in the header rather than silently editing the body. The spec's value to a future reader is partly the record of what turned out to be wrong.

- [ ] **Step 5: Commit**

```bash
git add docs/FDR_v3.md docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md
git commit -m "docs: fold the segmentation results into the FDR

13.2(5) is rewritten: the motivation becomes the measured result - 7 of
20 held-out cartridges returning zero placeable area - rather than 'the
rectangle is a coarse approximation', and the architecture becomes a
detector plus per-ROI segmenter on the boundary-displacement evidence.

6.2 gains the scope limit it always had implicitly: the green-channel
pipeline assumes a light tray with a dark interior module per PPR 5.3.2,
which the black cartridges in realtest/ are not. Recorded as a
precondition rather than a defect, because that is what it is.

10.6's taxonomy gains placement_disagreement, a deliberate skip rather
than a failure."
```

## Acceptance

- [ ] `pytest -q` passes with no regressions against Plan C's baseline.
- [ ] `plan.arbitration`'s channel constants match `recog.seg_dataset.SEG_CHANNELS`, pinned by test.
- [ ] `python main.py --config configs/demo.yaml` runs with no torch installed.
- [ ] Planning stays under 8 ms per cartridge with masks supplied (FDR O3).
- [ ] End-to-end cycle latency is measured and reported against the 50 ms PPR budget for 1, 2, 4 and 8 cartridges.
- [ ] `docs/receipts/tau_calibration.txt` exists and `configs/planning.yaml` carries the calibrated τ — **or** τ came back `None` and that is written up as a negative result.
- [ ] `PlacementDisagreement` events are counted and reported, not silently swallowed.
- [ ] Constructing `HeuristicPlacementAreaExtractor` emits its scope-limit warning.
- [ ] `docs/receipts/seg_ablation.txt` exists, reporting delta-cells and the
      heuristic-vs-segmenter comparison against the measured 0.218 baseline,
      with instance counts beside every number.
- [ ] FDR 6.2, 10.6 and 13.2(5) reflect the measured results (spec 12).
