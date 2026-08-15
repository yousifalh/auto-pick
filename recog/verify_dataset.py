"""Check a dataset against the hashes recorded in ``docs/datasets/checksums.json``.

``scripts/model_card_tables.py --sync-datasets`` has written that file since
it was introduced: for every ``recog/dataset3d*/`` it records the SHA-256 and
byte count of ``manifest.json`` and of the COCO annotations beside it, and
copies the manifest into ``docs/datasets/``. Every dataset row in
``docs/MODEL_CARD.md`` and ``docs/datasets/README.md`` cites those digests.

Nothing read them back. A grep over the tracked tree for ``checksums`` found
the writer and the table formatter and no consumer at all: no module, no
test, no CI step. So the digests were provenance in the decorative sense —
they could not disagree with anything, and a dataset that had been
regenerated, truncated or silently re-rendered under the same name would
have produced the same green suite and the same model card. This module is
the missing half. It hashes files and compares, and it exits non-zero.

Two modes, because there are two things worth checking and only one of them
is possible on a bare clone:

``--docs`` (the default)
    Verify the committed ``docs/datasets/*.manifest.json`` copies against
    their recorded digests. ``--sync-datasets`` copies each manifest
    byte-for-byte (``shutil.copyfile``) and hashes the *source*, so a
    committed copy must hash to the committed digest; if it does not, either
    the copy or the checksum file was edited by hand. This needs no
    datasets, runs in a second, and is what CI gates on. It also fails on a
    manifest copy with no recorded entry and on a recorded entry with no
    copy — "there was nothing to compare" is a failure here, not a pass.

``<dataset dir>``
    Verify a live dataset: ``manifest.json`` and the annotations file named
    in the entry. This is the author's check, before training on a dataset
    or publishing a number derived from one, and it is the one that would
    catch a regenerated dataset masquerading under an old name.

Usage
-----
    python -m recog.verify_dataset                       # the committed copies
    python -m recog.verify_dataset recog/dataset3d_seg   # a live dataset
    python -m recog.verify_dataset recog/dataset3d_seg --name dataset3d_seg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DOC = ROOT / "docs" / "datasets"
DEFAULT_CHECKSUMS = DATASETS_DOC / "checksums.json"

MANIFEST_NAME = "manifest.json"
DOC_SUFFIX = ".manifest.json"


class VerificationError(Exception):
    """The comparison could not be made at all.

    Distinct from a mismatch on purpose: a mismatch is a Failure in the
    returned list, whereas this is "I was asked to verify something I have
    no recorded hash for". Both end in a non-zero exit, but only one of
    them is a corrupt file.
    """


class Failure(NamedTuple):
    """One thing that did not check out.

    ``kind`` is the machine-readable reason — ``sha256``, ``bytes``,
    ``missing`` (recorded but not on disk) or ``unrecorded`` (on disk but
    not recorded). ``expected`` and ``actual`` are printed as-is.
    """

    path: Path
    kind: str
    expected: Any
    actual: Any

    def render(self) -> str:
        return (f"  {self.path}: {self.kind} "
                f"expected {self.expected!r}, got {self.actual!r}")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB blocks.

    Same chunk size and same digest as ``model_card_tables._sha256``; the
    two must agree or this module compares against nothing.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(path: Path | str = DEFAULT_CHECKSUMS) -> Dict[str, dict]:
    """Read checksums.json. A missing or malformed file is an error, not
    an empty dict — silently verifying nothing is the failure mode this
    module exists to remove."""
    p = Path(path)
    if not p.is_file():
        raise VerificationError(f"checksums file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{p}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise VerificationError(f"{p}: expected a JSON object of dataset entries")
    return data


def _check_one(path: Path, recorded: dict) -> List[Failure]:
    """Compare one file against one ``{"sha256": ..., "bytes": ...}`` record.

    Both fields are checked. The byte count alone would miss a same-length
    edit, and the digest alone would report a truncation as a bare hash
    mismatch; reporting both says which kind of damage it is.
    """
    if not path.is_file():
        return [Failure(path, "missing", recorded.get("sha256"), None)]

    out: List[Failure] = []
    expected_bytes = recorded.get("bytes")
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        out.append(Failure(path, "bytes", expected_bytes, actual_bytes))

    expected_sha = recorded.get("sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            out.append(Failure(path, "sha256", expected_sha, actual_sha))
    return out


def verify_dataset_dir(dataset_dir: Path | str,
                       checksums: Dict[str, dict],
                       name: Optional[str] = None) -> List[Failure]:
    """Verify a live dataset directory. Returns every mismatch found.

    ``name`` defaults to the directory's own name, which is the key
    ``sync_datasets()`` files it under. An unrecognised name raises rather
    than returning ``[]``: an unrecorded dataset is unverifiable, and
    returning "no failures" for it would restate the problem.
    """
    d = Path(dataset_dir)
    key = name or d.name
    entry = checksums.get(key)
    if entry is None:
        raise VerificationError(
            f"{d}: no entry named {key!r} in the checksums file, so there is "
            f"nothing to verify it against. Known: {sorted(checksums)}. "
            "Re-run `python scripts/model_card_tables.py --sync-datasets` if "
            "this dataset is meant to be recorded.")

    failures = _check_one(d / MANIFEST_NAME, entry.get("manifest", {}))
    ann = entry.get("annotations")
    if ann:
        failures += _check_one(d / ann["file"], ann)
    return failures


def verify_docs_copies(docs_dir: Path | str,
                       checksums: Dict[str, dict]) -> List[Failure]:
    """Verify the committed ``docs/datasets/*.manifest.json`` copies.

    Runs on a bare clone. Covers both directions — a copy with no record
    and a record with no copy — because either one means the committed
    provenance and the committed files have drifted apart.
    """
    docs = Path(docs_dir)
    failures: List[Failure] = []

    for name in sorted(checksums):
        copy = docs / f"{name}{DOC_SUFFIX}"
        failures += _check_one(copy, checksums[name].get("manifest", {}))

    for copy in sorted(docs.glob(f"*{DOC_SUFFIX}")):
        name = copy.name[: -len(DOC_SUFFIX)]
        if name not in checksums:
            failures.append(Failure(copy, "unrecorded", "a checksums entry", None))

    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a dataset against docs/datasets/checksums.json.")
    ap.add_argument("dataset", nargs="?",
                    help="a dataset directory (default: verify the committed "
                         "docs/datasets manifest copies instead)")
    ap.add_argument("--docs", metavar="DIR", nargs="?", const=str(DATASETS_DOC),
                    help="verify the manifest copies in DIR "
                         f"(default {DATASETS_DOC})")
    ap.add_argument("--checksums", default=str(DEFAULT_CHECKSUMS),
                    help=f"checksums file (default {DEFAULT_CHECKSUMS})")
    ap.add_argument("--name",
                    help="dataset key to look up (default: the directory name)")
    args = ap.parse_args(argv)

    try:
        checksums = load_checksums(args.checksums)
    except VerificationError as exc:
        print(f"verify_dataset: {exc}", file=sys.stderr)
        return 2

    if not checksums:
        # An empty file passes every comparison there is to make, which is
        # precisely the state that made these digests worthless.
        print(f"verify_dataset: {args.checksums} records no datasets; there is "
              "nothing to verify. Run `python scripts/model_card_tables.py "
              "--sync-datasets`.", file=sys.stderr)
        return 2

    try:
        if args.dataset:
            target = Path(args.dataset)
            failures = verify_dataset_dir(target, checksums, name=args.name)
            checked = f"{target} against {len(checksums)} recorded dataset(s)"
        else:
            docs = Path(args.docs or DATASETS_DOC)
            failures = verify_docs_copies(docs, checksums)
            checked = f"{len(checksums)} manifest copies in {docs}"
    except VerificationError as exc:
        print(f"verify_dataset: {exc}", file=sys.stderr)
        return 2

    if failures:
        print(f"verify_dataset: {len(failures)} mismatch(es):", file=sys.stderr)
        for f in failures:
            print(f.render(), file=sys.stderr)
        return 1

    # ASCII only: this prints to a Windows console under cp1252 as often as
    # to a UTF-8 CI log, and a gate must not fall over on its success message.
    print(f"verify_dataset: OK - {checked}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
