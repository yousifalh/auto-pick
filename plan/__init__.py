"""Planning module.

Contains the pieces that turn a detection :class:`common.types.Snapshot`
into a FIFO queue of pick-place poses:

* :mod:`plan.scene` — the digital twin (entities + occupancy grids).
* :mod:`plan.placement_area` — green-channel / PCB-aware extraction of
  the valid placement polygon inside a cartridge.
* :mod:`plan.bin_packing` — shelf-based FFDH algorithm.
* :mod:`plan.planner` — the orchestrator that ties them together.
"""
