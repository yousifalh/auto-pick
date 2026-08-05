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


def test_pack_cartridge_stays_in_plan():
    from plan import bin_packing
    assert hasattr(bin_packing, "pack_cartridge")
    import common.packing
    assert not hasattr(common.packing, "pack_cartridge")


def test_forbidden_mask_still_honoured():
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :] = True
    from common.packing import Item, first_fit_decreasing
    res = first_fit_decreasing([Item(0, 5, 5)], 30, 30,
                               forbidden_mask=mask, mm_per_cell=1.5)
    assert res.count == 0
    assert res.unplaced_ids == [0]
