"""The FFDH algorithm lives in common/ so both plan/ and recog/ can use it."""
import numpy as np


def test_algorithm_importable_from_common():
    from common.packing import Item, PackedItem, PackResult, first_fit_decreasing
    res = first_fit_decreasing([Item(0, 10, 10)], 100, 100)
    assert res.count == 1
    assert isinstance(res.placements[0], PackedItem)
    assert isinstance(res, PackResult)


def test_plan_reexports_the_same_objects():
    """plan.bin_packing must re-export, not redefine — identity, not equality."""
    from common import packing
    from plan import bin_packing
    for name in ("Item", "PackedItem", "PackResult", "first_fit_decreasing"):
        assert getattr(bin_packing, name) is getattr(packing, name), name


def test_common_packing_does_not_import_plan():
    import common.packing
    src = open(common.packing.__file__, encoding="utf-8").read()
    assert "import plan" not in src
    assert "from plan" not in src


def test_pack_cartridge_is_gone_from_both_modules():
    """The dead ``pack_cartridge`` adapter must stay deleted.

    This test used to assert ``hasattr(bin_packing, "pack_cartridge")``,
    which kept alive a function nothing called and which re-armed
    ``mm_per_px: float = 0.38`` — ``planning.yaml``'s placeholder, the
    constant whose removal is recorded in
    docs/superpowers/specs/2026-08-11-scale-calibration.md as having
    under-read 24 of 30 cartridges by 27 % at the median and produced
    3 unsafe placements. The live path
    (``Planner._pack_cartridge``) resolves scale through
    ``_resolve_scale``, which raises ``UnknownScale`` rather than
    guessing. A green test guarding the guessing version is worse than
    no test.
    """
    import common.packing
    from plan import bin_packing
    assert not hasattr(bin_packing, "pack_cartridge")
    assert not hasattr(common.packing, "pack_cartridge")
    assert "pack_cartridge" not in bin_packing.__all__


def test_no_module_re_arms_the_placeholder_scale():
    """No packing entry point may default mm_per_px at all.

    Scale is a property of the frame. Anything here that carries a
    default for it is guessing, and the guess is invisible at the call
    site — which is exactly how 0.38 survived.
    """
    import inspect

    import common.packing
    from plan import bin_packing

    for mod in (bin_packing, common.packing):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if not inspect.isfunction(obj):
                continue
            sig = inspect.signature(obj)
            assert "mm_per_px" not in sig.parameters, (
                f"{mod.__name__}.{name} takes mm_per_px; scale belongs to "
                f"the frame, not to a packing default")


def test_forbidden_mask_still_honoured():
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :] = True
    from common.packing import Item, first_fit_decreasing
    res = first_fit_decreasing([Item(0, 5, 5)], 30, 30,
                               forbidden_mask=mask, mm_per_cell=1.5)
    assert res.count == 0
    assert res.unplaced_ids == [0]
