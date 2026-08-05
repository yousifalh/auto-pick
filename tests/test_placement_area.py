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
