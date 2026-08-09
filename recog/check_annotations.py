"""
recog.check_annotations - validate a converted real-photo segmentation
sidecar before it is used for evaluation or fine-tuning.

    python -m recog.check_annotations recog/realtest_seg
    python -m recog.check_annotations recog/realtest_seg --max-overlap-px 5

Expects the directory layout `recog.labelme_to_seg` writes and
`recog/realtest` already uses: ``<dir>/annotations/*.json`` (one or more
COCO-RLE sidecars) and ``<dir>/images/`` (the photographs). ``images_dir``
can be overridden with ``--images-dir``; pass ``--no-images`` to skip the
on-disk image checks entirely (annotation-only checks still run).

Checks, per image and for the whole set:

* **Pairwise pixel overlap between the five SEG_CLASSES.** This
  project's central invariant on synthetic data is 0 overlapping pixels
  across every mask pair (docs/NEXT_STEPS.md; measured across 3280 pairs
  of the 502-scene dataset) - because a single Blender object-index pass
  cannot paint two ids onto the same pixel. A human tracing five classes
  independently has no such guarantee, so this re-measures the same
  invariant on converted real annotations rather than assuming it holds.
  ``placement_area``/``battery`` is singled out and always reported by
  name, clean or not: under the modal definition (ruling 4,
  docs/superpowers/specs/2026-08-06-segmentation-placement-area-design.md)
  it is the single most likely mistake - drawing the free floor as if a
  seated cell were not occluding it.
* The same check WITHIN one class (two overlapping instances of the same
  class - a duplicated or re-traced polygon). Not one of the five
  classes pairs, but it corrupts per-instance area/count statistics the
  same way, and is essentially free to compute alongside the cross-class
  check.
* Degenerate or corrupted annotations: an RLE that decodes to zero
  pixels, or whose recorded ``area``/``bbox``/RLE ``size`` disagrees with
  what the segmentation actually decodes to.
* An image with zero annotations, or an image FILE with no annotation
  RECORD at all - the two shapes of the gap `recog/realtest`'s
  IMG_4428 already burned this project once (recog/eval_real.py's module
  docstring: "carries zero boxes ... while the photograph itself is full
  of cells").
* Recorded image width/height disagreeing with the real file on disk, or
  with the RLE's own recorded ``size``.
* A class with zero instances anywhere in the scanned set - worth
  knowing before calling a collection complete, not itself a corruption.

Pure NumPy + Pillow, no cv2, no torch - like plan/arbitration.py, this
must run in the same torch-free environment. Reuses
`recog.synth3d.annotate.rle_decode`: a hand-rolled RLE codec exists in
exactly one place in this project, not two.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recog.synth3d.annotate import rle_decode
from recog.synth3d.config import SEG_CLASSES, seg_class_ids

# One source of truth for the five-class vocabulary and their ids.
CANONICAL_CATEGORY_IDS: Dict[str, int] = seg_class_ids()

# The pair ruling 4 of the design spec calls out by name. Always reported,
# clean or not, so a reader never has to wonder whether it was checked.
# Sorted to match pairwise_class_overlaps' key convention.
_HIGH_RISK_PAIR: Tuple[str, str] = tuple(sorted(("placement_area", "battery")))


@dataclass
class Issue:
    """One finding. ``image`` is None for a whole-file or whole-set issue."""

    severity: str            # "ERROR" or "WARNING"
    check: str
    detail: str
    source: str = ""         # the json file this was found in (or "" for a
                              # dataset-wide finding spanning several files)
    image: Optional[str] = None


# ------------------------------------------------------------- loading ----

def load_coco_doc(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _category_names(doc: dict) -> Dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in doc.get("categories", [])}


def check_category_schema(doc: dict, source: str) -> List[Issue]:
    """The categories declared in the file must be exactly the five-class
    vocabulary at the ids `recog.synth3d.config.seg_class_ids` assigns.
    A mismatch means either a hand-edited file or a converter that has
    drifted from the class the rest of the pipeline reads - either way,
    every category_id read after this point would mean the wrong thing.
    """
    found = {name: cid for cid, name in _category_names(doc).items()}
    if found == CANONICAL_CATEGORY_IDS:
        return []
    return [Issue(
        "ERROR", "category_schema",
        f"categories {found} do not match the expected "
        f"{CANONICAL_CATEGORY_IDS}.",
        source=source,
    )]


# --------------------------------------------------------- per-image ----

def _decode_instances(
    anns: Sequence[dict], img_w: int, img_h: int, source: str, image: str,
) -> Tuple[List[Tuple[str, np.ndarray, dict]], List[Issue]]:
    """Decode every annotation's RLE, checking it against the recorded
    image size and its own bookkeeping fields on the way.

    Returns the decoded instances that are safe to use in the overlap
    checks (RLE size matches the image), and the issues found - a size
    mismatch drops that one instance from the overlap checks (it cannot
    be ANDed against masks of a different shape) but is itself reported,
    so a corrupt instance is never silently absent from the report.
    """
    issues: List[Issue] = []
    usable: List[Tuple[str, np.ndarray, dict]] = []
    cat_names = {v: k for k, v in CANONICAL_CATEGORY_IDS.items()}

    for a in anns:
        cls = cat_names.get(int(a.get("category_id", -1)),
                            str(a.get("category_id")))
        seg = a.get("segmentation")
        if not seg or "size" not in seg or "counts" not in seg:
            issues.append(Issue(
                "ERROR", "degenerate_polygon",
                f"annotation id={a.get('id')} class={cls} has no usable "
                f"RLE segmentation.",
                source=source, image=image))
            continue

        rh, rw = int(seg["size"][0]), int(seg["size"][1])
        if (rh, rw) != (img_h, img_w):
            issues.append(Issue(
                "ERROR", "size_mismatch",
                f"annotation id={a.get('id')} class={cls} RLE size "
                f"{rw}x{rh} does not match the image's recorded "
                f"{img_w}x{img_h}; excluded from the overlap checks.",
                source=source, image=image))
            continue

        mask = rle_decode(seg)
        px = int(mask.sum())
        if px == 0:
            issues.append(Issue(
                "ERROR", "degenerate_polygon",
                f"annotation id={a.get('id')} class={cls} decodes to zero "
                f"pixels.",
                source=source, image=image))
            continue

        area = a.get("area")
        if area is not None and int(area) != px:
            issues.append(Issue(
                "ERROR", "degenerate_polygon",
                f"annotation id={a.get('id')} class={cls} recorded "
                f"area={area} but the RLE decodes to {px} px.",
                source=source, image=image))

        bbox = a.get("bbox")
        if bbox is not None:
            ys, xs = np.nonzero(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0
            if list(bbox) != [x0, y0, w, h]:
                issues.append(Issue(
                    "ERROR", "degenerate_polygon",
                    f"annotation id={a.get('id')} class={cls} recorded "
                    f"bbox={list(bbox)} but the RLE's own extent is "
                    f"{[x0, y0, w, h]}.",
                    source=source, image=image))

        usable.append((cls, mask, a))

    return usable, issues


def pairwise_class_overlaps(
    instances: Sequence[Tuple[str, np.ndarray, dict]],
) -> Dict[Tuple[str, str], int]:
    """Overlap pixel count for every unordered pair of the five classes,
    unioning instances of the same class first. Always returns all
    10 pairs (0 for a pair absent from the image), so a caller can print
    the high-risk pair even when there is nothing to report."""
    by_class: Dict[str, np.ndarray] = {}
    for cls, mask, _a in instances:
        if cls not in CANONICAL_CATEGORY_IDS:
            continue
        if cls in by_class:
            by_class[cls] = by_class[cls] | mask
        else:
            by_class[cls] = mask.copy()

    # Keys are alphabetically sorted pairs, independent of SEG_CLASSES'
    # own (unrelated) ordering, so a caller never has to guess or try
    # both orders to look one up (see _HIGH_RISK_PAIR below).
    out: Dict[Tuple[str, str], int] = {}
    for a, b in itertools.combinations(sorted(SEG_CLASSES), 2):
        ma, mb = by_class.get(a), by_class.get(b)
        out[(a, b)] = int(np.count_nonzero(ma & mb)) if ma is not None and mb is not None else 0
    return out


def same_class_instance_overlaps(
    instances: Sequence[Tuple[str, np.ndarray, dict]],
) -> Dict[str, int]:
    """Overlap pixel count within each class, summed over every pair of
    that class's own instances (a duplicated or re-traced polygon)."""
    by_class: Dict[str, List[np.ndarray]] = {}
    for cls, mask, _a in instances:
        by_class.setdefault(cls, []).append(mask)

    out: Dict[str, int] = {}
    for cls, masks in by_class.items():
        if len(masks) < 2:
            continue
        total = 0
        for m1, m2 in itertools.combinations(masks, 2):
            total += int(np.count_nonzero(m1 & m2))
        if total:
            out[cls] = total
    return out


# --------------------------------------------------------------- files ----

def check_image_file(
    image_record: dict, images_dir: Optional[Path], source: str,
) -> List[Issue]:
    """The recorded width/height must match the actual file, when one of
    the annotator's original photographs is available to check against."""
    if images_dir is None:
        return []
    file_name = image_record.get("file_name", "")
    path = images_dir / file_name
    if not path.is_file():
        return [Issue(
            "ERROR", "missing_image_file",
            f"{file_name!r} has an annotation record but no file at "
            f"{path}.",
            source=source, image=file_name)]

    from PIL import Image

    with Image.open(path) as im:
        actual = (im.width, im.height)
    recorded = (int(image_record.get("width", -1)),
               int(image_record.get("height", -1)))
    if actual != recorded:
        return [Issue(
            "ERROR", "dimension_mismatch",
            f"recorded size {recorded[0]}x{recorded[1]} does not match "
            f"the file on disk {actual[0]}x{actual[1]}.",
            source=source, image=file_name)]
    return []


def check_orphan_image_files(
    doc_file_names: Sequence[str], images_dir: Optional[Path], source: str,
) -> List[Issue]:
    """A photograph that was never annotated at all - not even a
    zero-annotation record - is the earliest form of the IMG_4428 gap:
    the file exists, `git status` shows it, but nothing downstream will
    ever know it was supposed to carry ground truth."""
    if images_dir is None:
        return []
    known = set(doc_file_names)
    exts = {".jpg", ".jpeg", ".png"}
    issues: List[Issue] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and p.name not in known:
            issues.append(Issue(
                "ERROR", "unannotated_photo",
                f"{p.name!r} exists under {images_dir} but has no entry "
                f"in {source} at all (never saved in the annotation tool, "
                f"or saved under a different file name).",
                source=source, image=p.name))
    return issues


# ------------------------------------------------------------ per-file ----

def validate_file(
    json_path: Path, images_dir: Optional[Path], max_overlap_px: int,
) -> Tuple[List[Issue], Dict[str, int]]:
    """Validate one COCO-RLE sidecar. Returns (issues, class totals)."""
    source = str(json_path)
    doc = load_coco_doc(json_path)
    issues: List[Issue] = list(check_category_schema(doc, source))

    anns_by_image: Dict[int, List[dict]] = {}
    for a in doc.get("annotations", []):
        anns_by_image.setdefault(int(a["image_id"]), []).append(a)

    class_totals: Dict[str, int] = {c: 0 for c in SEG_CLASSES}
    file_names: List[str] = []

    for image_record in doc.get("images", []):
        image = str(image_record.get("file_name", f"image_{image_record.get('id')}"))
        file_names.append(image)
        img_w = int(image_record.get("width", 0))
        img_h = int(image_record.get("height", 0))
        anns = anns_by_image.get(int(image_record["id"]), [])

        if not anns:
            issues.append(Issue(
                "ERROR", "no_annotations",
                f"{image!r} has an image record but zero annotations - "
                f"the IMG_4428 gap: label it, or if it is truly empty "
                f"(no cartridge, no cell, nothing) confirm that by eye, "
                f"since an empty photo is a legitimate but rare case.",
                source=source, image=image))

        instances, decode_issues = _decode_instances(
            anns, img_w, img_h, source, image)
        issues.extend(decode_issues)

        for cls, _mask, _a in instances:
            if cls in class_totals:
                class_totals[cls] += 1

        overlaps = pairwise_class_overlaps(instances)
        for (a, b), n in overlaps.items():
            if n > max_overlap_px:
                issues.append(Issue(
                    "ERROR", "cross_class_overlap",
                    f"{a} / {b} overlap by {n} px (limit {max_overlap_px}).",
                    source=source, image=image))
        hi = overlaps[_HIGH_RISK_PAIR]
        issues.append(Issue(
            "OK" if hi <= max_overlap_px else "ERROR",
            "placement_area_battery_overlap",
            f"{_HIGH_RISK_PAIR[0]} / {_HIGH_RISK_PAIR[1]}: {hi} px "
            f"(the modal-definition check, ruling 4).",
            source=source, image=image))

        same_class = same_class_instance_overlaps(instances)
        for cls, n in same_class.items():
            if n > max_overlap_px:
                issues.append(Issue(
                    "ERROR", "same_class_overlap",
                    f"{cls}: two or more instances overlap by {n} px "
                    f"(limit {max_overlap_px}).",
                    source=source, image=image))

        issues.extend(check_image_file(image_record, images_dir, source))

    issues.extend(check_orphan_image_files(file_names, images_dir, source))
    return issues, class_totals


# -------------------------------------------------------------- report ----

def _find_annotation_files(dir_path: Path) -> List[Path]:
    ann_dir = dir_path / "annotations"
    search_dir = ann_dir if ann_dir.is_dir() else dir_path
    return sorted(search_dir.glob("*.json"))


def default_images_dir(dir_path: Path) -> Optional[Path]:
    """``dir_path/images`` if it exists, else ``None``. The CLI's default
    when neither ``--images-dir`` nor ``--no-images`` was given."""
    candidate = dir_path / "images"
    return candidate if candidate.is_dir() else None


def validate_dir(
    dir_path: Path, images_dir: Optional[Path] = None,
    max_overlap_px: int = 0,
) -> Tuple[List[Issue], Dict[str, int], List[Path]]:
    """Validate every ``*.json`` under ``dir_path`` (or ``dir_path/annotations``).

    Returns ``(all_issues, class_totals, files_scanned)``. ``images_dir``
    is used exactly as given: ``None`` means image-file checks (file
    existence, dimension match) are skipped and only the annotation-only
    checks run. Callers that want the ``dir_path/images`` convention
    (the CLI does, by default) resolve it themselves via
    :func:`default_images_dir` and pass the result in explicitly - this
    function does not guess.
    """
    files = _find_annotation_files(dir_path)
    if not files:
        return ([Issue("ERROR", "no_annotation_files",
                       f"no *.json found under {dir_path} or "
                       f"{dir_path / 'annotations'}.")],
               {c: 0 for c in SEG_CLASSES}, [])

    all_issues: List[Issue] = []
    class_totals: Dict[str, int] = {c: 0 for c in SEG_CLASSES}
    for jp in files:
        issues, totals = validate_file(jp, images_dir, max_overlap_px)
        all_issues.extend(issues)
        for c, n in totals.items():
            class_totals[c] += n

    for c in SEG_CLASSES:
        if class_totals[c] == 0:
            all_issues.append(Issue(
                "WARNING", "empty_class",
                f"{c!r} has zero instances anywhere in the scanned set.",
            ))

    return all_issues, class_totals, files


def format_report(
    issues: Sequence[Issue], class_totals: Dict[str, int],
    files: Sequence[Path], dir_path: Path, max_overlap_px: int,
) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("Real-photo annotation validation")
    lines.append(f"  scanned        : {dir_path}")
    lines.append(f"  json file(s)   : {len(files)}")
    lines.append(f"  max_overlap_px : {max_overlap_px}")
    lines.append("")

    by_image: Dict[Tuple[str, Optional[str]], List[Issue]] = {}
    dataset_level: List[Issue] = []
    for it in issues:
        if it.image is None:
            dataset_level.append(it)
        else:
            by_image.setdefault((it.source, it.image), []).append(it)

    for (source, image), items in sorted(by_image.items()):
        lines.append(f"{image}  ({source})")
        for it in items:
            tag = {"ERROR": "[ERROR]", "WARNING": "[WARN] ",
                  "OK": "[OK]   "}.get(it.severity, it.severity)
            lines.append(f"  {tag} {it.check}: {it.detail}")
        lines.append("")

    if dataset_level:
        lines.append("Dataset-wide")
        for it in dataset_level:
            tag = {"ERROR": "[ERROR]", "WARNING": "[WARN] "}.get(
                it.severity, it.severity)
            lines.append(f"  {tag} {it.check}: {it.detail}")
        lines.append("")

    lines.append("Class totals (whole scanned set)")
    for c in SEG_CLASSES:
        lines.append(f"  {c:<20}{class_totals.get(c, 0)}")
    lines.append("")

    n_error = sum(1 for it in issues if it.severity == "ERROR")
    n_warn = sum(1 for it in issues if it.severity == "WARNING")
    n_images = len(by_image)
    lines.append(f"images scanned : {n_images}")
    lines.append(f"errors         : {n_error}")
    lines.append(f"warnings       : {n_warn}")
    lines.append("")
    lines.append("RESULT: " + ("FAIL" if n_error else "PASS"))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------- CLI ----

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m recog.check_annotations",
        description=(
            "Validate a directory of converted real-photo segmentation "
            "annotations before they are used for evaluation or "
            "fine-tuning. Exits non-zero if any ERROR is found."
        ),
    )
    ap.add_argument("dir", help="directory holding annotations/*.json "
                                "(and images/, by default)")
    ap.add_argument("--images-dir", default=None,
                    help="override the images directory (default: "
                         "<dir>/images if it exists)")
    ap.add_argument("--no-images", action="store_true",
                    help="skip on-disk image checks even if <dir>/images "
                         "exists (annotation-only checks still run)")
    ap.add_argument("--max-overlap-px", type=int, default=0,
                    help="pixel overlap allowed before a class pair (or "
                         "same-class instance pair) is flagged as an "
                         "ERROR (default 0, matching the project's "
                         "0-overlapping-pixels invariant on synthetic "
                         "data)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        raise SystemExit(f"error: not a directory: {dir_path}")

    # --no-images wins outright (explicit opt-out); an explicit
    # --images-dir is used as given; passing neither falls back to
    # <dir>/images if that exists, else image checks are skipped.
    if args.no_images:
        images_dir: Optional[Path] = None
    elif args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise SystemExit(f"error: not a directory: {images_dir}")
    else:
        images_dir = default_images_dir(dir_path)

    issues, class_totals, files = validate_dir(
        dir_path, images_dir=images_dir, max_overlap_px=args.max_overlap_px)

    print(format_report(issues, class_totals, files, dir_path,
                        args.max_overlap_px))
    return 1 if any(it.severity == "ERROR" for it in issues) else 0


__all__ = [
    "CANONICAL_CATEGORY_IDS",
    "Issue",
    "build_arg_parser",
    "check_category_schema",
    "check_image_file",
    "check_orphan_image_files",
    "default_images_dir",
    "format_report",
    "load_coco_doc",
    "main",
    "pairwise_class_overlaps",
    "same_class_instance_overlaps",
    "validate_dir",
    "validate_file",
]


if __name__ == "__main__":
    sys.exit(main())
