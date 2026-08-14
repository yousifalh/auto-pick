"""``docs/receipts/placement_feasibility.txt`` must stay a measurement.

``docs/RESULTS_SUMMARY.md`` §2 leads with the project's strongest claim -
that a bay packed to exact tolerance cannot be certified by a camera -
and quotes five numbers for it. Until ``scripts/placement_feasibility.py``
existed those five were the only figures in that document whose citation
was a spec file rather than a ``docs/receipts/`` artefact, which is
exactly where a careful reader pushes.

These tests pin the two things a receipt can quietly stop being:

* **regenerable** - the committed file must equal what the generator
  produces today, so a change to ``catalog.json``, ``planning.yaml``, the
  extractor, the packer or the occupancy rasteriser moves the receipt or
  fails here. The geometry half needs only committed files and is checked
  everywhere; the census half needs the gitignored dataset and is checked
  wherever it is present.
* **cited** - the summary's receipt index must point at the artefact.
  A receipt nobody references is not a receipt.

They deliberately do NOT assert the CONTENT of the finding. The
generator measures; if the measurement changes, the receipt changes with
it and these tests pass. What they refuse is a receipt that disagrees
with the code that claims to have produced it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "placement_feasibility.py"
RECEIPT = REPO_ROOT / "docs" / "receipts" / "placement_feasibility.txt"
SUMMARY = REPO_ROOT / "docs" / "RESULTS_SUMMARY.md"


def _load_generator():
    """Import ``scripts/placement_feasibility.py`` by path.

    ``scripts/`` is not a package (it is not in ``pyproject.toml``'s
    package list, on purpose - these are author tools, not shipped
    modules), so there is no import path to it. Loading by file location
    keeps that true instead of quietly adding one.
    """
    if not GENERATOR.is_file():  # pragma: no cover - it is committed
        pytest.skip("scripts/placement_feasibility.py is not present")
    spec = importlib.util.spec_from_file_location(
        "_placement_feasibility_under_test", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def receipt() -> str:
    if not RECEIPT.is_file():  # pragma: no cover - it is committed
        pytest.skip("docs/receipts/placement_feasibility.txt is not present")
    return RECEIPT.read_text(encoding="utf-8")


# ------------------------------------------------------ regenerable ----

def test_the_geometry_in_the_receipt_is_what_the_catalog_says_today(
    generator, receipt,
):
    """The half that needs no dataset, checked on every machine.

    This is the anti-drift test proper. Change the 10000's
    ``interior_mm``, change ``battery.diameter_mm``, change the rotation
    model - and the committed receipt no longer contains the section the
    generator now produces. It fails equally if someone edits the receipt
    by hand, which is the failure mode a receipt exists to prevent.
    """
    skus = generator.load_skus()
    width_mm, length_mm, _ = generator.load_cell_nominal()
    section = generator._norm(
        generator.section_geometry(skus, width_mm, length_mm))

    assert section in receipt, (
        "docs/receipts/placement_feasibility.txt no longer matches the "
        "geometry its generator computes from recog/synth3d/assets/"
        "catalog.json and configs/planning.yaml. Regenerate it:\n"
        "  python scripts/placement_feasibility.py")


def test_a_fresh_run_agrees_with_the_committed_receipt(generator):
    """``--check`` is the generator's own drift mode; run it.

    Every arm whose inputs are present is recomputed and required to
    appear verbatim in the committed file. On a tree without
    ``recog/dataset3d_seg`` (gitignored) that is the geometry arm alone,
    which the script reports as a skip rather than as a pass - the arms
    that did not run are counted in its output, so a green check on a
    bare clone cannot be mistaken for a green check on the full
    measurement.
    """
    assert generator.main(["--check"]) == 0, (
        "a fresh run of scripts/placement_feasibility.py disagrees with "
        "docs/receipts/placement_feasibility.txt. Regenerate it:\n"
        "  python scripts/placement_feasibility.py")


def test_the_receipt_names_its_inputs_and_how_to_rebuild_itself(receipt):
    """A figure with no reproduction line is not receipted, it is asserted."""
    assert "python scripts/placement_feasibility.py" in receipt
    # The two committed inputs are hashed: the geometry claim is exactly
    # as good as the files it was read from, and both are in the tree.
    assert "recog/synth3d/assets/catalog.json" in receipt
    assert "configs/planning.yaml" in receipt
    assert receipt.count("sha256:") >= 2
    # The dataset is gitignored, so its digest is the only thing that
    # pins the census arm to a specific corpus - the same reason
    # real_photo_eval.txt carries the weights digest.
    assert "instances_seg.json sha256:" in receipt


# ----------------------------------------------------------- cited ----

def test_the_summary_cites_the_receipt_and_not_only_the_spec():
    """The gap this receipt closes must not reopen silently."""
    if not SUMMARY.is_file():  # pragma: no cover - it is committed
        pytest.skip("docs/RESULTS_SUMMARY.md is not present")
    text = SUMMARY.read_text(encoding="utf-8")

    index = text.split("### Receipt index", 1)
    assert len(index) == 2, "docs/RESULTS_SUMMARY.md has no receipt index"
    rows = [r for r in index[1].splitlines() if "zero-placement" in r]
    assert rows, "the 10-of-10 claim has no row in the receipt index"
    assert all("docs/receipts/placement_feasibility.txt" in r for r in rows), (
        "docs/RESULTS_SUMMARY.md's receipt index cites something other than "
        "docs/receipts/placement_feasibility.txt for the ground-truth "
        "zero-placement claim. That row cited a spec file until this "
        "receipt existed; it must not go back to citing one.")
    assert RECEIPT.is_file(), (
        "docs/RESULTS_SUMMARY.md cites "
        "docs/receipts/placement_feasibility.txt and it does not exist")


def test_the_receipt_and_the_summary_agree_about_ten_of_ten(receipt):
    """The document's headline count and the receipt's are one number.

    Coupled deliberately. If the corpus, the extractor or the packer ever
    moves that count, this fails until BOTH the receipt and the sentence
    that quotes it have been brought to the new number - which is the
    whole point of publishing a receipt for a sentence.
    """
    if not SUMMARY.is_file():  # pragma: no cover - it is committed
        pytest.skip("docs/RESULTS_SUMMARY.md is not present")
    import re

    m = re.search(r"PLACES ZERO CELLS IN (\d+) OF (\d+) INSTANCES", receipt)
    if m is None:
        pytest.skip("the census arm did not run when this receipt was written")
    zero, n = m.group(1), m.group(2)

    summary = SUMMARY.read_text(encoding="utf-8")
    assert f"**{zero} of {n}**" in summary or f"{zero} of {n}" in summary, (
        f"the receipt measures {zero} of {n} zero-placement instances and "
        f"docs/RESULTS_SUMMARY.md §2 does not quote that figure")


def test_every_quoted_figure_carries_a_verdict(receipt):
    """The receipt grades the sentence it backs, including where it fails.

    A receipt that only records the figures that agreed would be a
    citation, not a check. The verdict block is the part a reader should
    look at first, so it must exist and must be per-figure.
    """
    if "4. The five figures" not in receipt:
        pytest.skip("the census arm did not run when this receipt was written")
    block = receipt.split("4. The five figures", 1)[1].split("\n5. ", 1)[0]
    lines = [line.strip() for line in block.splitlines()]
    verdicts = [line for line in lines if line.startswith("[")]
    assert len(verdicts) == 5, (
        f"the receipt grades {len(verdicts)} of the five figures "
        f"docs/RESULTS_SUMMARY.md §2 quotes")
    quoted = [line for line in lines if line.startswith("quoted")]
    measured = [line for line in lines if line.startswith("measured")]
    assert len(quoted) == 5 and len(measured) == 5, (
        "every graded figure must show both the quoted value and the "
        "measured one; a verdict without both is unauditable")
