"""``recog.verify_dataset`` — the gate that makes checksums.json falsifiable.

`docs/datasets/checksums.json` has been written by
`scripts/model_card_tables.py --sync-datasets` since it was introduced and
read back by nothing: no module imported it, no test asserted on it, no CI
step ran it. A hash that is never compared is not provenance, it is a
decorative string, and every dataset figure in `docs/MODEL_CARD.md` rested
on it. These tests exist so that the comparison happens and so that it
*fails* — a verifier that cannot be made to go red is the same decoration
one layer up, which is why most of what follows constructs a corruption and
demands a non-zero exit rather than checking the happy path again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recog import verify_dataset as VD

ROOT = Path(__file__).resolve().parent.parent
DOCS_DATASETS = ROOT / "docs" / "datasets"


# --------------------------------------------------------------- helpers ----

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_dataset(tmp_path: Path, name: str = "dataset3d_seg_probe",
                  manifest: bytes = b'{"n": 1}\n',
                  annotations: bytes | None = b'{"images": []}\n'):
    """A dataset directory plus the checksums dict that describes it.

    Mirrors what `sync_datasets()` records: the manifest always, and the
    first of instances_seg.json / instances.json that exists.
    """
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_bytes(manifest)
    entry = {"manifest": {"sha256": _sha256(manifest), "bytes": len(manifest)}}
    if annotations is not None:
        (d / "instances_seg.json").write_bytes(annotations)
        entry["annotations"] = {
            "file": "instances_seg.json",
            "sha256": _sha256(annotations),
            "bytes": len(annotations),
        }
    return d, {name: entry}


# ------------------------------------------------- a dataset that matches ----

def test_matching_dataset_reports_no_failures(tmp_path):
    d, checksums = _make_dataset(tmp_path)
    assert VD.verify_dataset_dir(d, checksums) == []


def test_matching_dataset_exits_zero(tmp_path, capsys):
    d, checksums = _make_dataset(tmp_path)
    ck = tmp_path / "checksums.json"
    ck.write_text(json.dumps(checksums), encoding="utf-8")
    assert VD.main([str(d), "--checksums", str(ck)]) == 0


# ------------------------------------------------------ corruption is red ----

def test_one_flipped_byte_in_the_manifest_fails(tmp_path):
    """The minimum detectable change. Same length, so a size-only check
    would pass it; this is the whole reason the file records a SHA-256."""
    d, checksums = _make_dataset(tmp_path, manifest=b'{"n": 1}\n')
    (d / "manifest.json").write_bytes(b'{"n": 2}\n')
    failures = VD.verify_dataset_dir(d, checksums)
    assert len(failures) == 1
    assert failures[0].kind == "sha256"
    assert failures[0].path.name == "manifest.json"


def test_changed_annotations_file_fails(tmp_path):
    """The annotations, not just the manifest: a manifest can be identical
    across two renders whose COCO polygons are not. Edited to exactly the
    same 15 bytes so only the digest can catch it."""
    d, checksums = _make_dataset(tmp_path)
    assert len(b'{"images": []}\n') == len(b'{"images": [0]}')
    (d / "instances_seg.json").write_bytes(b'{"images": [0]}')
    failures = VD.verify_dataset_dir(d, checksums)
    assert [(f.path.name, f.kind) for f in failures] == [
        ("instances_seg.json", "sha256")]


def test_truncated_file_reports_the_size_too(tmp_path):
    d, checksums = _make_dataset(tmp_path)
    (d / "manifest.json").write_bytes(b"")
    kinds = {f.kind for f in VD.verify_dataset_dir(d, checksums)}
    assert kinds == {"bytes", "sha256"}


def test_missing_recorded_file_fails(tmp_path):
    """checksums.json names instances_seg.json; deleting it must not read
    as 'nothing to compare, therefore fine'."""
    d, checksums = _make_dataset(tmp_path)
    (d / "instances_seg.json").unlink()
    failures = VD.verify_dataset_dir(d, checksums)
    assert [f.kind for f in failures] == ["missing"]


def test_corrupt_dataset_exits_non_zero(tmp_path):
    """The exit code is the entire point: a verifier that prints a
    complaint and returns 0 is not a gate."""
    d, checksums = _make_dataset(tmp_path)
    (d / "manifest.json").write_bytes(b'{"n": 99}\n')
    ck = tmp_path / "checksums.json"
    ck.write_text(json.dumps(checksums), encoding="utf-8")
    assert VD.main([str(d), "--checksums", str(ck)]) != 0


# ----------------------------------------- nothing-to-check is a failure ----

def test_unknown_dataset_name_fails(tmp_path):
    """A dataset with no entry cannot be *verified*, and reporting success
    for it would be exactly the unfalsifiable claim this module removes."""
    d, checksums = _make_dataset(tmp_path)
    with pytest.raises(VD.VerificationError, match="no entry"):
        VD.verify_dataset_dir(d, {"some_other_dataset": checksums.popitem()[1]})


def test_empty_checksums_file_fails(tmp_path):
    ck = tmp_path / "checksums.json"
    ck.write_text("{}", encoding="utf-8")
    assert VD.main(["--docs", str(DOCS_DATASETS), "--checksums", str(ck)]) != 0


def test_missing_checksums_file_fails(tmp_path):
    assert VD.main([str(tmp_path), "--checksums", str(tmp_path / "nope.json")]) != 0


# ------------------------------------------------ the committed doc copies ----

def test_committed_docs_copies_match_their_recorded_hashes():
    """`--sync-datasets` copies each generator manifest into docs/datasets/
    byte-for-byte (shutil.copyfile) and hashes the source, so the committed
    copy must hash to the committed digest. This is the check that runs on a
    bare clone with no datasets present, and it is what CI gates on."""
    checksums = VD.load_checksums(VD.DEFAULT_CHECKSUMS)
    assert checksums, "docs/datasets/checksums.json is empty"
    assert VD.verify_docs_copies(DOCS_DATASETS, checksums) == []


def test_docs_mode_flags_a_manifest_copy_with_no_entry(tmp_path):
    """A manifest copied into docs/ without a recorded hash is provenance
    with nothing behind it — the exact state this module exists to end."""
    docs = tmp_path / "datasets"
    docs.mkdir()
    body = b'{"n": 1}\n'
    (docs / "dataset3d_known.manifest.json").write_bytes(body)
    (docs / "dataset3d_orphan.manifest.json").write_bytes(body)
    checksums = {"dataset3d_known": {
        "manifest": {"sha256": _sha256(body), "bytes": len(body)}}}
    failures = VD.verify_docs_copies(docs, checksums)
    assert [f.kind for f in failures] == ["unrecorded"]
    assert "orphan" in failures[0].path.name


def test_docs_mode_flags_a_recorded_dataset_with_no_copy(tmp_path):
    docs = tmp_path / "datasets"
    docs.mkdir()
    body = b'{"n": 1}\n'
    checksums = {"dataset3d_known": {
        "manifest": {"sha256": _sha256(body), "bytes": len(body)}}}
    failures = VD.verify_docs_copies(docs, checksums)
    assert [f.kind for f in failures] == ["missing"]


def test_docs_mode_on_the_real_tree_exits_zero():
    assert VD.main(["--docs", str(DOCS_DATASETS),
                    "--checksums", str(VD.DEFAULT_CHECKSUMS)]) == 0
