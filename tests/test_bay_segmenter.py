"""Bay segmenter model factory and inference wrapper."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_model_has_six_output_channels():
    from recog.bay_segmenter import build_segmenter

    m = build_segmenter(num_classes=6, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 64, 64))["out"]
    assert out.shape[1] == 6


def test_segment_batch_returns_one_map_per_crop_at_native_size():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    crops = [np.zeros((40, 90, 3), np.uint8),
             np.zeros((120, 55, 3), np.uint8)]
    out = seg.segment_batch(crops)

    assert len(out) == 2
    assert out[0].shape == (40, 90)
    assert out[1].shape == (120, 55)
    for m in out:
        assert m.dtype == np.int8
        assert int(m.max()) < 6


def test_segment_matches_segment_batch_of_one():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    crop = (np.arange(48 * 64 * 3, dtype=np.uint8) % 255).reshape(48, 64, 3)
    assert np.array_equal(seg.segment(crop), seg.segment_batch([crop])[0])


def test_empty_batch_returns_empty_list():
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=False, pretrained=False)
    assert seg.segment_batch([]) == []


def test_half_precision_is_refused_on_cpu():
    """fp16 on CPU is slower than fp32 and silently so. Better to say."""
    from recog.bay_segmenter import BaySegmenter

    seg = BaySegmenter(checkpoint=None, device="cpu",
                       crop_size=64, half=True)
    assert seg.half is False


def test_baysegmenter_loads_a_checkpoint_the_training_loop_wrote(tmp_path):
    """The train -> save -> load -> infer round trip Plan D depends on.

    Training builds the model with `pretrained=True` (the aux classifier
    regularises), so its state_dict carries aux_classifier.* keys.
    BaySegmenter, given a checkpoint, builds with `pretrained=False` (no
    inference-time role for aux - see the module docstring), so those
    keys have nowhere to land and load_state_dict's strict check raises.
    None of the other tests in this file catch it: they all construct
    BaySegmenter(checkpoint=None), which never exercises loading a real
    checkpoint at all.
    """
    from recog.bay_segmenter import BaySegmenter, build_segmenter
    from recog.seg_training import checkpoint_state_dict

    # Mirrors exactly how recog.seg_training.train() builds and saves a
    # checkpoint - same constructor call, same save helper - so this test
    # exercises the production code path rather than a re-implementation
    # of it that could silently drift out of sync.
    model = build_segmenter(num_classes=6, pretrained=True)
    ckpt_path = tmp_path / "roundtrip.pt"
    torch.save({"model": checkpoint_state_dict(model)}, ckpt_path)

    seg = BaySegmenter(checkpoint=str(ckpt_path), device="cpu",
                       crop_size=64, half=False)
    crop = np.zeros((48, 64, 3), np.uint8)
    out = seg.segment_batch([crop])

    assert out[0].shape == (48, 64)
    assert out[0].dtype == np.int8
    assert 0 <= int(out[0].min()) and int(out[0].max()) < 6


def test_dice_loss_is_zero_for_a_perfect_prediction():
    from recog.seg_training import dice_loss

    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :4] = 2
    logits = torch.full((2, 6, 8, 8), -20.0)
    for b in range(2):
        for y in range(8):
            for x in range(8):
                logits[b, int(target[b, y, x]), y, x] = 20.0
    assert float(dice_loss(logits, target, 6)) < 0.02


def test_dice_loss_is_large_for_an_inverted_prediction():
    from recog.seg_training import dice_loss

    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :4] = 2
    logits = torch.full((2, 6, 8, 8), -20.0)
    logits[:, 5] = 20.0                    # predict class 5 everywhere
    assert float(dice_loss(logits, target, 6)) > 0.8


def test_dice_loss_ignores_classes_absent_from_the_batch():
    """A batch with no obstruction pixels must not be penalised for
    failing to predict obstruction - otherwise 40% of batches carry a
    constant gradient toward a class that is not there."""
    from recog.seg_training import dice_loss

    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 6, 4, 4), -20.0)
    logits[:, 0] = 20.0
    assert float(dice_loss(logits, target, 6)) < 0.02


def test_signed_area_error_separates_optimistic_from_conservative():
    """Optimistic error puts a cell where one cannot go - a damage
    event. Conservative error refuses where one can - a lost cell. Only
    the first is a safety issue and a single unsigned number hides it."""
    import numpy as np

    from recog.seg_evaluate import signed_area_error_mm2

    target = np.zeros((20, 20), np.int8)
    target[5:15, 5:15] = 2                 # 100 px of true bay
    pred = np.zeros((20, 20), np.int8)
    pred[5:15, 5:17] = 2                   # 20 px too many, all optimistic

    opt, cons = signed_area_error_mm2(pred, target, cls=2, mm_per_px=0.5)
    assert opt == pytest.approx(20 * 0.25)
    assert cons == pytest.approx(0.0)


def test_signed_area_error_reports_conservative_when_under_predicting():
    import numpy as np

    from recog.seg_evaluate import signed_area_error_mm2

    target = np.zeros((20, 20), np.int8)
    target[5:15, 5:15] = 2
    pred = np.zeros((20, 20), np.int8)
    pred[5:15, 5:13] = 2                   # 20 px too few

    opt, cons = signed_area_error_mm2(pred, target, cls=2, mm_per_px=0.5)
    assert opt == pytest.approx(0.0)
    assert cons == pytest.approx(20 * 0.25)


def test_boundary_displacement_is_zero_for_an_exact_match():
    import numpy as np

    from recog.seg_evaluate import boundary_displacement_mm

    m = np.zeros((30, 30), np.int8)
    m[8:22, 8:22] = 2
    assert boundary_displacement_mm(m, m, cls=2, mm_per_px=0.63) == \
        pytest.approx(0.0)


def test_boundary_displacement_scales_with_mm_per_px():
    import numpy as np

    from recog.seg_evaluate import boundary_displacement_mm

    target = np.zeros((30, 30), np.int8); target[8:22, 8:22] = 2
    pred = np.zeros((30, 30), np.int8); pred[8:22, 8:24] = 2

    a = boundary_displacement_mm(pred, target, cls=2, mm_per_px=1.0)
    b = boundary_displacement_mm(pred, target, cls=2, mm_per_px=2.0)
    assert b == pytest.approx(2.0 * a)
    assert a > 0.0


def test_per_class_iou_handles_a_class_absent_from_both():
    import numpy as np

    from recog.seg_evaluate import per_class_iou

    m = np.zeros((10, 10), np.int8)
    m[2:8, 2:8] = 2
    iou = per_class_iou(m, m, num_classes=6)
    assert iou["bay"] == pytest.approx(1.0)
    assert np.isnan(iou["obstruction"]), (
        "a class in neither prediction nor truth has no IoU; reporting "
        "0.0 would drag the mean down for a class that was never tested")


# ------------------------------------------------------ split guard --
#
# recog/dataset3d_seg is gitignored and came from a resumable generation
# run stopped part-way through. Resuming or regenerating it changes
# len(full_dataset), so random_split silently returns a DIFFERENT
# partition for the same seed and a checkpoint could be scored on crops
# it trained on - with nothing to signal it. seg_training already writes
# val_instance_counts into every checkpoint for exactly this check.

def test_check_split_matches_checkpoint_passes_when_counts_agree(tmp_path):
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    counts = {"background": 53, "cartridge": 37, "bay": 19,
             "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": counts,
               "coco_path": "recog/dataset3d_seg/instances_seg.json"},
              ckpt_path)

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
        counts)  # must not raise


def test_check_split_matches_checkpoint_raises_loudly_on_mismatch(tmp_path):
    """A resumed/regenerated dataset changes len(full_dataset), which
    changes what random_split returns for the same seed - this is the
    live failure mode the guard exists to catch, not a hypothetical."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg/instances_seg.json"},
              ckpt_path)

    recomputed_after_resume = dict(trained_on, bay=24)  # dataset grew

    with pytest.raises(SystemExit) as exc:
        check_split_matches_checkpoint(
            str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
            recomputed_after_resume)

    msg = str(exc.value)
    assert "19" in msg and "24" in msg, (
        "the error must name BOTH the checkpoint's recorded counts and "
        "the recomputed ones, not just say 'mismatch'")


def test_check_split_matches_checkpoint_warns_but_does_not_raise_on_old_checkpoint(tmp_path):
    """A checkpoint written before this field existed has nothing to
    check against - warn, do not fail a run that has no way to comply."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": {}}, ckpt_path)

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg/instances_seg.json",
        {"bay": 1})  # must not raise


def test_check_split_matches_checkpoint_skips_count_check_across_datasets(tmp_path):
    """Spec #2 (generalisation): a checkpoint trained on one dataset (e.g.
    the anchored procedural set) is deliberately evaluated against a
    DIFFERENT held-out dataset (the CAD test set) - that is the plan's
    entire Task 19, not a drifted split. The two datasets' val instance
    counts are expected to differ (different scenes entirely); this must
    not raise just because the checkpoint's own coco_path differs from
    the eval config's."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg_anchored/instances_seg.json"},
              ckpt_path)

    wildly_different = {"background": 400, "cartridge": 300, "bay": 0,
                        "electronics": 0, "obstruction": 90, "battery": 250}

    check_split_matches_checkpoint(
        str(ckpt_path), "recog/dataset3d_seg_cad_test/instances_seg.json",
        wildly_different)  # must not raise - different dataset, by design


def test_check_split_matches_checkpoint_still_raises_when_coco_path_matches(tmp_path):
    """The cross-dataset skip must not become a blanket bypass: if the
    eval's coco_path is the SAME one the checkpoint recorded, a count
    mismatch is still the drifted-split bug and must still raise."""
    from recog.seg_evaluate import check_split_matches_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    trained_on = {"background": 53, "cartridge": 37, "bay": 19,
                 "electronics": 19, "obstruction": 11, "battery": 13}
    torch.save({"val_instance_counts": trained_on,
               "coco_path": "recog/dataset3d_seg_anchored/instances_seg.json"},
              ckpt_path)

    with pytest.raises(SystemExit):
        check_split_matches_checkpoint(
            str(ckpt_path), "recog/dataset3d_seg_anchored/instances_seg.json",
            dict(trained_on, bay=24))


# ------------------------------------------------- singleton last batch --

def test_drop_last_batch_true_when_the_final_batch_would_be_a_singleton():
    """A train split of 721 crops at batch_size 8 leaves a final batch of
    exactly 1. DeepLabV3's ASPP pooling branch reduces that to a
    [1, 256, 1, 1] tensor, and BatchNorm in TRAINING mode raises
    `ValueError: Expected more than 1 value per channel when training`.
    This is not hypothetical: it is what
    `configs/segmentation_anchored.yaml` (848 crops, 0.85 split -> 721
    train) did on its first epoch, and it kills the run outright.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(721, 8) is True


def test_drop_last_batch_false_when_the_final_batch_is_safe():
    """Every other remainder trains fine, so nothing is dropped - the
    existing baseline (361 crops -> 307 train, 307 % 8 == 3) must keep
    seeing every crop in every epoch, exactly as it did when
    `docs/receipts/seg_eval.txt` was produced.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(307, 8) is False
    assert drop_last_batch(720, 8) is False   # remainder 0
    assert drop_last_batch(722, 8) is False   # remainder 2


def test_drop_last_batch_false_when_batch_size_is_one():
    """batch_size 1 makes EVERY batch a singleton; dropping the last one
    would not help and dropping them all would train on nothing. Such a
    config is broken for BatchNorm regardless, and silently emptying the
    loader would hide that instead of letting it surface.
    """
    from recog.seg_training import drop_last_batch

    assert drop_last_batch(9, 1) is False


# --------------------------------------------------------- latency exit --

def test_latency_within_budget_true_when_the_batch8_row_passes():
    from recog.seg_evaluate import latency_within_budget

    latency = [{"cartridges": 1, "within_50ms_budget": True},
              {"cartridges": 8, "within_50ms_budget": True}]
    assert latency_within_budget(latency) is True


def test_latency_within_budget_false_when_the_batch8_row_fails():
    """This is what main()'s exit code is derived from - previously only
    a log.warning fired here, so a CI job could not gate on the plan's
    latency acceptance criterion at all."""
    from recog.seg_evaluate import latency_within_budget

    latency = [{"cartridges": 1, "within_50ms_budget": True},
              {"cartridges": 8, "within_50ms_budget": False}]
    assert latency_within_budget(latency) is False


# --------------------------------------------------- checkpoint note --

def test_sibling_checkpoint_note_reports_both_ious_and_their_delta(tmp_path):
    from recog.seg_evaluate import _sibling_checkpoint_note

    counts = {"bay": 19, "electronics": 19, "obstruction": 11}
    torch.save({"selected_mean_iou": 0.8158, "val_instance_counts": counts},
              tmp_path / "best.pt")
    torch.save({"selected_mean_iou": 0.8140, "val_instance_counts": counts},
              tmp_path / "last.pt")

    note = _sibling_checkpoint_note(str(tmp_path / "best.pt"))
    assert note is not None
    assert "0.8158" in note and "0.8140" in note


def test_sibling_checkpoint_note_is_none_without_both_checkpoints(tmp_path):
    from recog.seg_evaluate import _sibling_checkpoint_note

    torch.save({"selected_mean_iou": 0.8}, tmp_path / "best.pt")
    assert _sibling_checkpoint_note(str(tmp_path / "best.pt")) is None


# ------------------------------------------------------------- per-SKU --

def test_group_indices_by_asset_partitions_by_sku():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A", None]

    out = group_indices_by_asset(_FakeDataset(), [0, 1, 2, 3])
    assert out == {"A": [0, 2], "B": [1], None: [3]}


def test_group_indices_by_asset_only_includes_requested_indices():
    from recog.seg_evaluate import group_indices_by_asset

    class _FakeDataset:
        sample_assets = ["A", "B", "A"]

    out = group_indices_by_asset(_FakeDataset(), [0, 1])
    assert out == {"A": [0], "B": [1]}


def test_format_per_sku_table_lists_every_sku_with_its_crop_count():
    from recog.seg_evaluate import format_per_sku_table

    results = {
        "AnkerPowerCore10000": {"n_val_crops": 12,
                                "ious": {"bay": 0.80, "obstruction": 0.60}},
        "AnkerPowerCore13000": {"n_val_crops": 9,
                                "ious": {"bay": 0.75, "obstruction": 0.55}},
    }
    table = format_per_sku_table(results)
    assert "AnkerPowerCore10000" in table and "12" in table
    assert "AnkerPowerCore13000" in table and "9" in table
