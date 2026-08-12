# `unit_id`, malformed RLE, and three docstrings that lied — 2026-08-12

**Base:** `docs/superpowers/audit/2026-08-12-I-data-pipeline.md` (findings 6, 7a,
7b, 8) and `docs/superpowers/audit/2026-08-12-K-complexity.md` (§1.5, §1.6, §3).
**HEAD when this started:** `b8653bd`, 1 128 tests passing.
**Suite after:** **1 166 passed, 1 skipped** (the skip is the pre-existing
`tests/test_orientation_gate.py` Blender skip). +38 tests, 30 of them mine.

**No algorithm changed.** Nothing here alters a metric definition, a dataset, a
model, a packing strategy, a paint order, or the RLE *encoder* — which audit I
verified correct against an independent implementation of the COCO spec and
which is untouched. Every behavioural change is a refusal: something that used
to proceed silently on malformed input now raises. Everything else is a comment.

**Files changed**

| File | Kind |
|---|---|
| `recog/seg_dataset.py` | refuses a sidecar with no `unit_id`; comments |
| `recog/check_annotations.py` | new `unit_id_issues` check; catches malformed RLE |
| `recog/synth3d/annotate.py` | `rle_decode` validates `counts`; `unit_id` comment |
| `recog/labelme_to_seg.py` | carries the SKU into `asset`; `_unit_id` comment |
| `recog/synth3d/scene.py` | comment only (`unit_id` scope) |
| `common/packing.py` | docstrings only (`_drop_unsafe`, `pack_best_effort`) |
| `plan/scene.py` | docstring only (`update_from_snapshot` cost) |
| `plan/placement_area.py` | comment only (the O3 coupling) — **see the note at the end** |
| `docs/ANNOTATION_PROTOCOL.md` | §5, §6.1, §8 |
| `tests/test_seg_dataset.py`, `tests/test_annotate_masks.py`, `tests/test_labelme_to_seg.py`, `tests/test_check_annotations.py` | +30 tests |

---

## 1 · A missing `unit_id` no longer collapses an image

Audit I finding 7b: three annotations in one image, two physical units. With
`unit_id` absent, `None`, or duplicated, `BaySegDataset` returned **one** crop
whose union box `(10,20,260,80)` spanned both units, where the correct answer is
two crops at `(10,20,140,80)` and `(180,20,260,80)`. No exception, no warning, no
log line. Training proceeds on a crop at the wrong scale over the wrong content
and nothing downstream can detect it.

The mechanism is a one-liner: grouping is `by_image[image_id] -> by_unit[unit_id]`
and a Python dict buckets on `None` as cheerfully as on any other key.

**`BaySegDataset.__init__` now raises `ValueError`** if any annotation has a
`unit_id` that is missing, `None`, or blank — before any grouping happens, naming
the file, the count, and the first ten offending annotation ids, and saying what
would have happened. Blank strings are included because `""` is a perfectly good
dict key and collapses identically.

**Duplication is not detectable here and is not treated as an error.** Two units
sharing one id produce a crop indistinguishable from one genuinely large unit.
That case is handled in the validator (§2) as a warning, not here as a refusal.

**This changes nothing for any dataset that exists.** Measured over all 10
sidecars on disk — 5 018 images, 43 449 annotations — **zero** annotations lack a
`unit_id`. Neither live producer can emit one: `masks_from_index` always writes
the key and `labelme_to_seg._unit_id` never returns `None`. The reachable path is
a hand-edited sidecar or a third-party converter, which is precisely the path
real-photograph ground truth would arrive on, and precisely the population this
project's annotation tooling exists to serve.

Two fixtures in `tests/test_seg_dataset.py` that omitted `unit_id` were given
one. They were testing the `asset` field, not this.

## 2 · `check_annotations` now has a `unit_id` code path

It had none at all — audit I read every check to confirm it. New
`unit_id_issues(anns, source, image)`, called per image from `validate_file`:

- **`missing_unit_id` — ERROR.** The §1 case, reported here *as well* so it lands
  in the report next to everything else wrong with the file rather than as a
  traceback at the first training step. Exits the CLI non-zero.
- **`single_unit_id` — WARNING.** Every annotation in one image sharing one id.
  This is the merge case, and no checker can prove it wrong: a duplicated Group
  ID is a valid id, and "one photo, one cartridge" is a legitimate photograph. So
  it asks rather than asserts, and only when the image has ≥2 annotations.

**Measured false-positive rate on the corpus: zero.** Across all 5 018 synthetic
images, every image whose annotations all share one id has exactly **one**
annotation, so the warning never fires. It *will* fire on a legitimate close-up
of a single grouped cartridge on the real path — which is exactly why it is a
warning.

**A rule I built and then deleted.** My first version flagged a unit carrying two
instances of `cartridge`, `placement_area` or `electronics_module` — clean on the
corpus (0 hits in 43 449 annotations) and a precise detector of the merge case.
It is wrong on the path that matters: `docs/ANNOTATION_PROTOCOL.md` §1.2 asks for
tiled `cartridge` wall pieces and §1.1 for split `placement_area` floors, so
"two of one class in one unit" is the documented norm for a hand annotation, not
a signal. It would have fired on most real open cartridges. Recorded here because
the synthetic corpus endorsed a rule the real protocol forbids, which is the
generic trap of validating a real path against synthetic evidence.

## 3 · `rle_decode` raises on a malformed `counts` array

Audit I finding 8, measured on a 4×4 mask — every one of these decoded silently:

| `counts` | was | now |
|---|---|---|
| `[0, 999]` | 16 px set (overrun clipped by numpy slicing) | `ValueError` |
| `[0, 3]` | 3 px set (underrun zero-filled) | `ValueError` |
| `[0, -2, 5]` | 14 px, every run after the negative displaced | `ValueError` |

The check is `sum(counts) == h*w` and no negative run. Both are C-level passes
over the list; the existing decode already walks it in Python. **Measured cost on
2 000 real annotations from `recog/dataset3d_seg` (mean 134 runs each): 99 ms
before, 102 ms after — 3 %, about 1.5 µs per decode.** Not a codec change: a
well-formed RLE decodes byte-for-byte as it always did, pinned by a test over
five shapes including the non-square mirrored pair.

Float counts still raise, as they always did — now usually as the `ValueError`
rather than a `TypeError` from numpy, which is an improvement in message quality
and nothing else.

**`check_annotations` catches the new exception** and reports it as a
`degenerate_polygon` ERROR rather than dying. A validator that stops at the first
bad annotation hides every later one; a test asserts a good annotation *after* a
malformed one is still counted.

`rle_encode` was **not** touched, including audit I finding 9 (a genuinely
multi-valued array breaks the alternation). It was verified correct against an
independent COCO implementation, both live callers pass boolean arrays, and the
brief was explicit.

## 4 · `unit_id` is scene-local, and three places now say so

Audit I finding 7a: `recog/dataset3d_seg` has **69 distinct `unit_id` values
across 502 images**; `item0` appears in **252 of them**. `scene.build` derives the
id from a per-scene group index or pass index, so collision across images is by
construction. The docstring called it "unique by construction".

It is harmless **only** because `BaySegDataset` buckets by `image_id` first and
groups by `unit_id` within that bucket. Anyone flattening that to one dict over
the sidecar — for a per-unit split, a per-unit dedup, a per-unit metric — would
merge 252 unrelated units and raise nothing.

Corrected in `recog/synth3d/scene.py` (both the `build` return docstring and the
comment above the `units` dict), and in `recog/synth3d/annotate.py`'s `unit_id`
comment. Each states the scope, gives the measurement, and says the consumer must
bucket by `image_id` first.

The two producers genuinely **disagree** on the field's scope —
`labelme_to_seg._unit_id` keys on the image stem (`photo1#g1`) and *is* globally
unique — so each docstring now points at the other rather than letting a reader
generalise from one.

**`tests/test_seg_dataset.py::test_the_same_unit_id_in_two_images_stays_two_crops`**
pins the bucketing: the same `"item0"` in two images must stay two crops at their
own boxes. Two more tests pin the refusal (parametrised over absent / `None` /
blank) and its control case.

## 5 · The `asset` field: carried through, not documented away

`labelme_to_seg` emitted no `asset`, so `ds.sample_assets` came back `[None]` and
`seg_evaluate --per-sku` would have put every crop from every real photograph in
one bucket (audit I finding 6).

**Decision: carry it, via LabelMe's own flags.** The alternative the brief
offered — write the limitation into the protocol — was rejected because the gap
is not inherent. LabelMe already has a field for exactly this kind of attribute,
per-shape and per-image `flags`, driven by `--labelflags` and `--flags`. Using it
costs a checkbox per photo and makes a published per-SKU breakdown possible on
real hardware, which is the whole reason the CAD-holdout SKU work exists.

`asset_of(shape, image_flags, where)`: per-shape flags win, image-level flags are
the fallback (the ergonomic case — one photo of one product, one checkbox).
Exactly one flag may be true; zero means "not declared" and is fine; **two is an
error, not a guess**, because there is no vocabulary here to arbitrate with.
Deliberately no vocabulary: real photographs may show a SKU absent from
`catalog.json`, and a converter that refused those would be worse than one that
cannot choose between two.

Granularity is the **unit**, matching the synthetic sidecar — the annotator ticks
one shape and the whole unit inherits it. Two shapes of the *same* unit declaring
different SKUs is an error (either the flags disagree or two units share a Group
ID — both worth stopping for). The map is built from **raw** shapes, not surviving
ones: a shape can lose every pixel to paint order and still be the shape whose
flag named the SKU, and losing the declaration with the pixels would be a fresh
silent gap.

**A directory-wide `--asset` CLI flag was considered and rejected.** It cannot
express a photo holding two SKUs, and it would mislabel one silently — the exact
failure mode this whole task exists to remove.

`asset` is written explicitly as `None` when nothing was declared, and the CLI
prints how many landed there and what `--per-sku` will do with them. Twelve tests
cover it, ending with a full round trip into `BaySegDataset.sample_assets`.
`docs/ANNOTATION_PROTOCOL.md` §6.1 gains the `labelme --flags` invocation.

## 6 · `_drop_unsafe` is O(p²), not O(n)

Audit K §1.6. Each kept placement is tested against every earlier kept placement,
and `pack_best_effort` runs it **three times per pack**:

| placements | one call | ×3 |
|---:|---:|---:|
| 25 | 0.042 ms | 0.126 ms |
| 100 | 0.558 ms | 1.675 ms |
| 200 | 2.266 ms | 6.799 ms |
| 400 | 8.586 ms | 25.757 ms |
| 800 | 37.968 ms | 113.904 ms |

Docstring corrected, with the table and the reason the wrong claim survived:
today p ≤ 24, so it costs ~0.1 ms and no test could contradict it. **The
algorithm is unchanged** — the brief was explicit, the packer's real-world margin
is 3.9×, and the quadratic term is not binding. The docstring says what to do if
it ever is (sort by x and sweep, or bucket into grid cells).

`plan/scene.py::update_from_snapshot` said nothing about cost at all, while the
module advertises cartridge persistence as a feature. It is **O(D·C)** — measured
0.018 / 0.172 / 2.814 / 10.387 ms at 8 / 32 / 128 / 256 cartridges, a clean
4×-per-doubling, crossing the 50 ms frame budget at ~500–560 cartridges against a
real corpus maximum of **4**. Docstring now says so, and says the global
best-first ordering that makes it quadratic is also the audit-H finding-4 fix and
must not be traded for a spatial index without cause. **Docstring only** — that
file's logic was rewritten by another agent this session.

## 7 · The accidental protection, recorded at both ends

Audit K §1.5. `pack_best_effort` breaches the 8 ms O3 budget at a **158 × 314 mm**
floor (8.16 ms). It never sees one, because
`SegmentationPlacementAreaExtractor.reject_if_not_one_cartridge_floor` refuses any
placeable floor larger than `_MAX_CARTRIDGE_EXTENT_MM = (81.7, 180.0)` mm **before
the occupancy grid is built** — capping the packer at **2.04 ms, a 3.9× margin**.

That interlock was written to catch a detector box spanning a cartridge and three
loose cells. It has nothing to do with latency, and nothing anywhere said it was
holding up a timing budget. The breach point is only ~1.94× its long axis and
~1.93× its short axis, so raising it for a larger SKU is a realistic thing to
want to do and would silently cost the margin.

Comments now record the coupling at **both** ends, each pointing at the other:
beside `_MAX_CARTRIDGE_EXTENT_MM` in `plan/placement_area.py` (with the bisection
table and "re-measure `pack_best_effort` before widening this") and in
`pack_best_effort`'s own docstring in `common/packing.py`. Documentation of a
real dependency; no behaviour changed at either end.

---

## Verification

- **Full suite: 1 166 passed, 1 skipped** (`python -m pytest`), against 1 128 at
  `b8653bd`. The pre-existing skip needs Blender.
- `tests/test_seg_dataset.py::test_dataset_yields_crops_with_every_channel_present_somewhere`
  builds a `BaySegDataset` over the **real** training sidecar and passes, so the
  new refusal is confirmed not to fire on production data.
- Every count in this document was measured here, over the sidecars on disk, not
  copied from the audit.

## One thing to know before committing

`plan/placement_area.py` carries **another agent's uncommitted work** in the same
session (removal of `attach_placement_area`). My change to it is the
comment-only §7 note; it applied cleanly on top and conflicts with nothing, but I
have **not staged that file**, because staging it would commit their changes too.
The comment sits in the working tree and should ride along with their commit of
that file. Same reasoning would apply to any other shared file.
