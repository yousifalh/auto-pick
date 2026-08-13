"""The zero-ground-truth exclusion in ``recog.eval_real`` must stay loud.

``recog/realtest/annotations/instances_default.json`` carries 80 boxes over
seven photographs, and ``IMG_4428.jpg`` gets none of them while the photograph
itself is full of cells and shells. It is a gap in the CVAT export, not an
empty scene. Scoring it turns every prediction on it into a false positive,
which lowers precision — and therefore AP — for the whole set.

``eval_real`` excludes zero-ground-truth images by default. The exclusion is
*correct* and it is *load-bearing*, which is exactly why it must never be
quiet: an eval set that silently drops its inconvenient member is a worse
failure than the depressed number it avoids. These tests pin the three places
the fact is stated (stderr, report, receipt) and the property that decides
membership, so that

* an unannotated image cannot silently rejoin the scored set,
* an unannotated image cannot silently *leave* it either, and
* the ``--include-empty`` figure cannot be mistaken for the real-photo result.

``tests/test_dataset.py`` covers ``partition_records`` and the exclusion block
itself. This file covers the loudness added on top of them and the shipped
corpus that motivated it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recog.dataset import CLASS_MAP, parse_coco_json
from recog.eval_real import (
    DEFAULT_ANN_PATH,
    build_image_rows,
    format_report,
    partition_records,
    receipt_text,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = REPO_ROOT / "docs" / "receipts" / "real_photo_eval.txt"
RECEIPT_INCLUDE_EMPTY = (
    REPO_ROOT / "docs" / "receipts" / "real_photo_eval_include_empty.txt"
)


# ---------------------------------------------------------- the corpus ----

def _shipped_records():
    path = REPO_ROOT / DEFAULT_ANN_PATH
    if not path.is_file():  # pragma: no cover - the file is committed
        pytest.skip(f"{DEFAULT_ANN_PATH} is not present")
    return parse_coco_json(path)


def test_shipped_corpus_agrees_with_its_receipt_about_what_is_scored():
    """The scored set is a property of the data, and the receipt says so.

    This is the anti-drift test. If someone annotates ``IMG_4428.jpg`` the
    scored count goes to 7 and this fails until the receipt is regenerated —
    which is the point: a corpus change that moves a published number must
    move the receipt with it. It fails equally if an image quietly *loses*
    its annotations. Neither direction can happen in silence.
    """
    records = _shipped_records()
    scored, excluded = partition_records(records)

    assert len(records) == 7, "the real-photo corpus is seven photographs"
    assert scored, "every image is unlabelled - that is a corpus failure"
    assert len(scored) + len(excluded) == len(records)

    if not RECEIPT.is_file():  # pragma: no cover - the receipt is committed
        pytest.skip("docs/receipts/real_photo_eval.txt is not present")
    receipt = RECEIPT.read_text(encoding="utf-8")

    assert f"{len(scored)} of {len(records)} scored" in receipt, (
        "docs/receipts/real_photo_eval.txt disagrees with the annotations "
        "about how many images are scorable. Regenerate it:\n"
        "  python -m recog.eval_real --checkpoint recog/checkpoints/best.pt "
        "--out docs/receipts/real_photo_eval.txt"
    )
    # Every excluded image is named in the receipt, so the exclusion can be
    # audited without rerunning anything.
    for rec in excluded:
        assert rec.file_name in receipt, (
            f"{rec.file_name} has zero ground-truth boxes and is excluded "
            f"from the published figure, but the receipt does not name it"
        )


def test_the_gap_that_motivated_the_exclusion_is_recorded_not_assumed():
    """Whatever the exclusion costs, the receipts measure it.

    The default receipt and the ``--include-empty`` one are the same command
    over the same weights, differing only in whether the unlabelled image is
    scored. Keeping both committed is what stops the exclusion from being a
    convenience: the number it avoids is on the record beside the number it
    publishes.
    """
    for path in (RECEIPT, RECEIPT_INCLUDE_EMPTY):
        if not path.is_file():  # pragma: no cover - both are committed
            pytest.skip(f"{path.name} is not present")

    default = RECEIPT.read_text(encoding="utf-8")
    counterfactual = RECEIPT_INCLUDE_EMPTY.read_text(encoding="utf-8")

    # The counterfactual carries the banner and disclaims itself.
    assert "DEPRESSED BY AN ANNOTATION GAP" in counterfactual
    assert "Do not quote it as the real-photo result" in counterfactual
    assert "--include-empty" in counterfactual
    # The publishable one does not pretend the excluded image never existed.
    assert "EXCLUDED FROM THE SCORE" in default
    # Both pin the weights they scored: no .pt is tracked in this repository,
    # so the path alone identifies nothing.
    for text in (default, counterfactual):
        assert "sha256:" in text
        assert "python -m recog.eval_real" in text


# ------------------------------------------------- the loudness itself ----

def _two_image_doc():
    """A COCO doc with one labelled image and one unlabelled one."""
    return {
        "categories": [
            {"id": CLASS_MAP["battery"], "name": "Battery"},
            {"id": CLASS_MAP["cartridge"], "name": "Cartridge"},
        ],
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "full.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "empty.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": CLASS_MAP["battery"],
             "bbox": [10.0, 10.0, 20.0, 20.0], "iscrowd": 0},
        ],
    }


def _report_for(tmp_path: Path, include_empty: bool) -> str:
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(_two_image_doc()), encoding="utf-8")
    records = parse_coco_json(path)
    scored, excluded = partition_records(records, include_empty=include_empty)
    preds = {
        1: [((10.0, 10.0, 30.0, 30.0), CLASS_MAP["battery"], 0.9)],
        2: [((0.0, 0.0, 40.0, 40.0), CLASS_MAP["battery"], 0.95),
            ((50.0, 50.0, 80.0, 80.0), CLASS_MAP["battery"], 0.8)],
    }
    gts = {r.image_id: list(zip(r.boxes, r.labels)) for r in scored}
    scored_preds = {r.image_id: preds[r.image_id] for r in scored}
    return format_report(
        summarise(gts, scored_preds),
        {CLASS_MAP["battery"]: sum(len(v) for v in gts.values()),
         CLASS_MAP["cartridge"]: 0},
        {CLASS_MAP["battery"]: sum(len(v) for v in scored_preds.values()),
         CLASS_MAP["cartridge"]: 0},
        n_images=len(scored),
        confidence=0.7,
        detector_name="Stub",
        checkpoint="recog/checkpoints/best.pt",
        config_path=None,
        elapsed_s=1.0,
        rows=build_image_rows(scored, excluded, preds),
        n_found=len(records),
        checkpoint_digest="sha256:deadbeef (1 bytes)",
    )


def test_include_empty_report_disclaims_its_own_headline(tmp_path: Path):
    """The number a reader copies is the mAP line, so warn above it.

    Before this banner, an ``--include-empty`` report differed from the
    default one by a per-image note and a count on line five. The headline
    looked identical and was 4.6 points lower on the shipped corpus.
    """
    text = _report_for(tmp_path, include_empty=True)

    assert "DEPRESSED BY AN ANNOTATION GAP" in text
    assert "empty.jpg" in text
    assert "2 false" in text, "the false-positive count must be surfaced"
    assert "NOT comparable with the default" in text
    assert "Do not quote it as the real-photo result" in text
    # The scope line says it too, before the table.
    assert "WITH ZERO GROUND TRUTH, scored anyway" in text
    # ... and the mAP is not left to stand alone at the bottom.
    assert "scored with ZERO ground truth" in text
    # The banner is a property of the data, not of the flag: it names the
    # image because the image has no boxes.
    assert "empty.jpg  0 GT, 2 prediction(s)" in text


def test_default_report_carries_no_contamination_banner(tmp_path: Path):
    """The banner must not cry wolf on a clean run."""
    text = _report_for(tmp_path, include_empty=False)

    assert "DEPRESSED BY AN ANNOTATION GAP" not in text
    assert "WITH ZERO GROUND TRUTH, scored anyway" not in text
    # The exclusion is still stated, with the count.
    assert "EXCLUDED FROM THE SCORE - 1 image(s)" in text
    assert "1 of 2 scored" in text
    assert "empty.jpg  0 GT, 2 prediction(s)" in text


def test_report_carries_the_weights_digest(tmp_path: Path):
    """A receipt naming only a gitignored path identifies no network."""
    text = _report_for(tmp_path, include_empty=False)
    assert "weights     : sha256:deadbeef (1 bytes)" in text


def test_receipt_text_states_how_to_regenerate_itself():
    report = "\n\nReal-photo held-out evaluation\n  detector    : Stub\n"
    text = receipt_text(report, ["--checkpoint", "recog/checkpoints/best.pt",
                                 "--out", "docs/receipts/real_photo_eval.txt"])
    assert text.endswith(report)
    assert ("python -m recog.eval_real --checkpoint "
            "recog/checkpoints/best.pt") in text
    assert "NO .pt IS TRACKED IN THIS REPOSITORY" in text


def test_zero_gt_images_are_selected_by_their_boxes_not_their_name(
    tmp_path: Path,
):
    """Nothing in the exclusion path knows the string ``IMG_4428``.

    A filename-keyed skip list would be the silent-drop failure mode this
    whole mechanism exists to avoid, and it would not catch the next
    unlabelled image. Renaming the file must not change what is scored.
    """
    doc = _two_image_doc()
    doc["images"][1]["file_name"] = "IMG_4428.jpg"
    doc["images"][0]["file_name"] = "IMG_9999.jpg"
    path = tmp_path / "renamed.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    scored, excluded = partition_records(parse_coco_json(path))
    assert [r.file_name for r in scored] == ["IMG_9999.jpg"]
    assert [r.file_name for r in excluded] == ["IMG_4428.jpg"]

    # And with the boxes moved onto IMG_4428, IMG_4428 is what gets scored.
    doc["annotations"][0]["image_id"] = 2
    path.write_text(json.dumps(doc), encoding="utf-8")
    scored, excluded = partition_records(parse_coco_json(path))
    assert [r.file_name for r in scored] == ["IMG_4428.jpg"]
    assert [r.file_name for r in excluded] == ["IMG_9999.jpg"]
