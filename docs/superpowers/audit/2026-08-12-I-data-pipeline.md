# Audit I — Data pipeline and label encoding

**Scope.** The full ground-truth chain: Blender object-index pass → per-class masks →
hand-rolled COCO RLE → JSON on disk → loader → torch tensor → metric.
**HEAD** `39429a4`. **Read-only**: nothing staged, committed, regenerated or modified.
Everything below marked *[executed]* was verified by running code; *[read]* means
inferred from source without execution.

---

## Verdict, first

**No systematic encoding error exists. Ground truth is trustworthy.**

The RLE encoder is correct against an independently written implementation of the COCO
specification. Every annotation in every dataset on disk has a bounding box that exactly
bounds its own decoded mask, an `area` that exactly equals its own decoded pixel count,
and an RLE `size` that exactly matches its image. Five-class disjointness holds at
**0 overlapping pixels across all 43 449 annotations in all 5 018 images of all 10
datasets** — measured exhaustively, by me, not sampled. The published segmentation
numbers are measured against ground truth that is, at every hop I could execute, exactly
what the generator intended.

Findings below are ranked by consequence. The top item is a real defect in a real
consumer path; everything after it is a hardening note or a latent trap, not a live
corruption.

---

## 1. Is the RLE correct? — **Yes.** *[executed]*

`recog/synth3d/annotate.py:385` `rle_encode` / `:411` `rle_decode`.

`pycocotools` is not installed in this environment, so I wrote an independent encoder and
decoder from the COCO `MaskApi.c` specification (column-major scan, alternating runs
beginning with a run of **zeros**, `size = [h, w]`, `sum(counts) == h*w`). The independent
decoder deliberately does **not** use `numpy.reshape(order="F")` — it walks columns
explicitly with `divmod(idx, h)` — so a transposition in the project's code cannot be
mirrored by a transposition in mine.

Script: `scratchpad/rle_audit.py`. **15 mask cases, 0 failures.**

Every case was subjected to all seven of these checks:

| Check | Result |
|---|---|
| project `counts` list == independent `counts` list, exactly | pass |
| project encode → **independent** decode == original | pass |
| independent encode → **project** decode == original | pass |
| `sum(counts) == h*w` | pass |
| `size == [rows, cols]` (not `[cols, rows]`) | pass |
| mask with first pixel set begins with a literal `0` | pass |
| **no double leading zero** (the historical defect) | pass |
| decoded `.shape` equals original `.shape` | pass |

Cases covered exactly what was asked: first-pixel-set (4×6), all-zero (4×6), all-one
(4×6), single pixel (5×3), 1×N (1×7), N×1 (7×1), last-pixel-set (6×4), top-row-only
(2×8), left-col-only (8×2), five random masks including the mirrored non-square pair
13×29 / 29×13, and a purpose-built **transposition trap**: a 3×5 mask with four set
pixels placed so that a transposed decode preserves the popcount but relocates every
pixel.

The trap is discriminating, not decorative — I confirmed *[executed]* that a row-major
encoder produces a **different** `counts` list for it than the project's encoder does.
A transposition bug would therefore have failed this suite rather than round-tripping
through it.

Concretely, for the transposition trap the project emits `size: [3, 5]` and a counts
list whose first element is `0` (first pixel set), and my independent column-walking
decoder reconstructs the original array bit-for-bit. The column-major convention at
`annotate.py:396` (`flatten(order="F")`) and `:421` (`reshape(order="F")`) is COCO's,
correctly, and the docstring's claim is accurate.

## 2. Round-trip fidelity across the whole chain — **clean** *[executed]*

encode → `json.dumps` → `json.loads` → decode, for all 15 cases: **exact**.

- Every element of `counts` and `size` is a Python `int` on the way out and an `int` on
  the way back. `run` accumulates as a plain Python `int` (never a NumPy scalar), and
  `size` is cast through `int()` at `annotate.py:408`.
- I string-searched the serialised JSON for `.` and `e` **inside the counts array**: no
  float or exponent notation anywhere.
- Large-count precision: a 4000×4000 all-zero mask encodes to `[16000000]`, survives
  `json` round-trip exactly, stays `int`, and still sums to `h*w`. Well inside the
  2^53 float boundary; there is no realistic mask that could approach it.
- **On disk** *[executed]*: I re-checked types against the real files rather than trusting
  the in-memory path — all 43 449 annotations across all 10 sidecars carry integer
  `counts` and `size` (`scratchpad/scan_datasets.py`, `nonint=0` on every file).

## 3. Mask / box agreement — **exact, everywhere** *[executed]*

For every annotation in every sidecar I decoded the RLE and recomputed the tight extent
from the decoded pixels, then compared to the recorded `bbox` and `area`:

```
dataset3d_seg                              502 imgs  5179 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_anchored                     502 imgs  2802 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_anchored_18650               502 imgs  2820 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_anchored_crown               502 imgs  2802 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_cad_control_holdout_10000    502 imgs  5957 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_cad_control_holdout_13000    502 imgs  5675 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_cad_control_holdout_20100    502 imgs  5222 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_cad_control_holdout_26800    502 imgs  4738 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_cad_test                     500 imgs  5156 anns  bad_box=0 bad_area=0 bad_size=0
dataset3d_seg_wide                         502 imgs  3098 anns  bad_box=0 bad_area=0 bad_size=0
                                          ----------------------------------------------------
GRAND                                     5018 imgs 43449 anns  bad_box=0 bad_area=0 bad_size=0
```

`bad_size` covers both `sum(counts) == h*w` and RLE `size == image (height, width)`.
`schema_ok=True` on every file (categories exactly `seg_class_ids()`).

**Detector-vs-segmenter divergence: none observed.** `masks_from_index` deliberately does
*not* apply the `merge_group_boxes` / `extend_group_boxes` step that the VOC detector path
applies, and its docstring (`annotate.py:494-506`) warns that parity is "coincidental, not
structural" and would break if `min_px` were lowered or `--res` raised. I tested the claim
*[executed]*: on `dataset3d_seg`, the VOC XML files and the COCO sidecar agree **exactly**
— 4278 objects each, split 840 `cartridge` / 3438 `battery` in both. The warned-about
divergence has not materialised at current settings. It remains a live risk if those knobs
move, exactly as documented.

## 4. Class-index consistency end to end — **sound, and structurally so** *[executed]*

Traced `placement_area` by name from generator to metric:

```
config.SEG_CLASSES index 3  ->  seg_class_ids()  ->  category_id 4
  -> written to categories[] as {"id": 4, "name": "placement_area"}
  -> BaySegDataset: names = {c["id"]: c["name"]}; rec["class"] = names[category_id]
  -> _PAINT_ORDER maps "placement_area" -> channel name "bay"
  -> SEG_CHANNELS["bay"] = 2
  -> seg_evaluate.CHANNEL_NAMES[2] == "bay"   (derived from SEG_CHANNELS, not restated)
```

I confirmed each hop by execution: pixel (25, 40) inside a `placement_area` polygon
rasterises to `2`; `electronics_module` → `3`; `battery` → `5`; `cartridge` → `1`. The
native label map for a 200×120 image came back shaped `(120, 200)`, i.e. `(H, W)` — a
transposition at the loader would have shown here and did not.

The reason an off-by-one is hard to introduce is architectural, and worth stating: **the
chain is keyed by class *name*, never by integer position.** `BaySegDataset` builds
`{id: name}` from the file's own `categories` block and looks up by name;
`seg_evaluate.CHANNEL_NAMES` is derived from `SEG_CHANNELS` by sorting on the value rather
than being a second hand-written list; `check_annotations.check_category_schema` pins the
ids to `seg_class_ids()` and errors on any drift. I grepped for hard-coded numeric
`category_id`s across `recog/`, `plan/` and `common/`: **none** *[executed]*. `SEG_CLASSES`
starting with `CLASSES` (so ids 1/2 mean the same in both the VOC and COCO files) is
enforced by `tests/test_synth3d.py:1572`.

## 5. The disjointness sweep's real coverage — **the synthetic invariant is a tautology; the sweep's teeth are on the real path** *[executed + read]*

**What I measured.** I ran the equivalent sweep myself, exhaustively — all 10 pairs of the
five classes, unioning same-class instances first, on every image of every dataset:
**0 overlapping pixels, 43 449 annotations, 5 018 images.** The invariant holds. The
documented "0 overlapping pixels across 3280 mask pairs" figure is corroborated and then
some.

**Where the sweep lives.** `recog/check_annotations.py:192` `pairwise_class_overlaps`, run
via `validate_dir`. Two things about its coverage matter:

- It is **exhaustive, not sampled**, over whatever directory it is pointed at: every
  image, every annotation, all 10 pairs, default threshold `--max-overlap-px 0`. It would
  catch a systematic overlap and a sporadic one equally — a systematic one would simply
  appear on every image rather than one.
- It is a **manual CLI**. *[read]* It is not invoked from `generate3d.py`, `verify3d.py`,
  any test fixture over the real datasets, or `.github/workflows/ci.yml` (CI runs
  `pytest -q` only). `docs/ANNOTATION_PROTOCOL.md:434` is the only place that tells a human
  to run it, and it points at `recog/realtest_rig` — a real-photo directory that does not
  yet exist.

**The important structural point.** On synthetic data the invariant cannot fail. Each mask
is built as `inst = (ids == pid)` from a single `int32` index map (`annotate.py:525`), and
each `pid` maps to exactly one class. Two masks from distinct pids are disjoint by set
theory; two class unions are disjoint because no pixel carries two pids. A single Blender
object-index pass physically cannot paint two ids onto one pixel. So the synthetic sweep is
**verifying something the encoder makes unfalsifiable** — reassuring, and my 0-overlap
measurement was guaranteed before I ran it. It is not wasted (it would catch a future
change that composed masks some other way), but it should not be read as independent
evidence.

The sweep's real value is on the **hand-annotated** path, where two polygons sharing a
boundary genuinely do share pixels after `cv2.fillPoly`. There, `resolve_paint_order`
(`labelme_to_seg.py:110`) enforces disjointness at conversion time and
`check_annotations` re-measures it. I verified this works *[executed]*: I fed the converter
a LabelMe export with a `placement_area` and an `electronics_module` deliberately sharing
an edge, plus an `obstruction` and a `battery` drawn inside the `placement_area`, and
`validate_dir` returned **0 errors** with the high-risk `battery`/`placement_area` pair at
0 px. That guarantee is real and it works — but it has never run on an actual photograph,
because none exist.

## 6. The LabelMe path — **produces loader-acceptable output; verified against the real loader** *[executed]*

`scratchpad/labelme_e2e.py`. Nothing mocked. A hand-written LabelMe export (six polygons,
five classes, one grouped unit plus one loose cell, on a deliberately **non-square**
200×120 image) went through the whole chain:

```
[convert]  images=1 annotations=6 dropped=0
           categories: battery:1 cartridge:2 electronics_module:3 placement_area:4 obstruction:5
           unit_ids: photo1#g1 x5, photo1#u5
           all required COCO keys present: True
[validate] check_annotations: 0 ERRORS, all five classes present
[loader]   BaySegDataset accepted it: len=1
           tensors: image=(3,64,64) float32   label=(64,64) int64
           label values all within SEG_CHANNELS: True
```

The loose ungrouped cell was correctly excluded (no cartridge-related class in its unit),
and the grouped unit produced one crop containing all five classes. The converter's output
is genuinely consumable by the production training loader — this is the first time that has
been demonstrated end to end rather than asserted.

One gap worth knowing before real photos arrive: the converter emits no **`asset`** field
(`labelme_to_seg.py:277-285`), so `ds.sample_assets` comes back `[None]`. `seg_evaluate.py
--per-sku` groups on `asset`; on converted real data every crop would land in a single
`None` bucket. Not a corruption — `.get("asset")` handles it — but per-SKU breakdowns will
be unavailable on real photographs until the converter learns to carry a SKU.

## 7. `unit_id` grouping — **survives every hop; degrades gracefully on the LabelMe path, silently on a hand-edited one** *[executed]*

Survival, verified: `scene.build` stamps it (`scene.py:505`) → `masks_from_index` carries
it via `.get()` (`annotate.py:578`) → `write_coco_json` writes it (`:604`) → `BaySegDataset`
groups on it (`seg_dataset.py:296`). On disk, `unit_id_missing=0` across all 43 449
annotations. `BaySegDataset` is the **only** consumer — I grepped `seg_evaluate.py`,
`calibrate_tau.py`, `calibration.py`, `seg_training.py`, `seg_ablation.py`, `verify3d.py`,
`generate3d.py` and `plan/`: no other reader *[executed]*.

### 7a. `unit_id` is scene-local, not globally unique — contained, but a trap for the next consumer

*[executed]* `dataset3d_seg` has **69 distinct `unit_id` values across 502 images**.
`item0` appears in **252 different images**; `solo1` in 244. `scene.py:505` derives the id
from a per-scene group id or pass index (`item{n}` / `solo{pid}`), so collision across
images is by construction.

This is harmless **today** only because `BaySegDataset` buckets by `image_id` first and
groups by `unit_id` *within* each bucket (`seg_dataset.py:293-296`). Any future consumer
that groups by `unit_id` globally — a per-unit train/val split, a per-unit dedup, a
per-unit metric — would silently merge 252 unrelated physical units. The field's own
docstring calls the id "unique by construction", which is true within a scene and false
across the dataset; that wording is the trap. The LabelMe converter, by contrast, keys on
the image stem (`photo1#g1`) and *is* globally unique — so the two producers disagree on
the field's scope.

### 7b. A missing or duplicated `unit_id` collapses a whole image into one crop, with no error

*[executed]* Three annotations in one image (two units: a `cartridge`+`placement_area`
pair, and a second `cartridge`):

```
normal                   crops=2  union_boxes=[(10,20,140,80), (180,20,260,80)]
unit_id key absent       crops=1  union_boxes=[(10,20,260,80)]
unit_id all duplicated   crops=1  union_boxes=[(10,20,260,80)]
unit_id all None         crops=1  union_boxes=[(10,20,260,80)]
```

All three degenerate cases silently produce a **single crop spanning both units** — no
exception, no warning, no log line. Training would proceed on a crop whose scale and
content are wrong, and nothing downstream could detect it. `check_annotations.py` does
**not** validate `unit_id` at all (I read every check; there is no `unit_id` code path).

The mitigating fact: this is unreachable from either live producer. `masks_from_index`
always writes the key, and `_unit_id` never returns `None`. It is reachable from a
hand-edited sidecar, a third-party converter, or a future producer — which is precisely
the population `check_annotations` exists to police, and the one check it is missing.

*[executed]* Reassuringly, the specific human error one would expect on the real path —
**forgetting to set Group IDs in LabelMe** — does *not* collapse. `_unit_id` gives each
ungrouped shape its own `#u{i}`, so the failure mode is **fragmentation**, not merging:

```
group_ids set              unit_ids=[p#g1, p#g1, p#g1]  -> 1 crop
group_id forgotten (None)  unit_ids=[p#u0, p#u1, p#u2]  -> 3 crops
```

Three tight crops instead of one unit crop is a degraded but non-corrupt training signal,
and it is visible in the crop count. This is good design and it holds up under test.

## 8. `rle_decode` accepts a malformed counts array without complaint *[executed]*

```
counts=[0,999],  size=[4,4]  ->  16 px set   (overrun silently clipped by numpy slicing)
counts=[0,3],    size=[4,4]  ->   3 px set   (underrun silently zero-filled)
counts=[0,-2,5], size=[4,4]  ->  14 px set   (negative run scrambles the output)
counts=[0.0,4.0]             ->  TypeError   (floats do at least raise)
```

No `sum(counts) == h*w` assertion anywhere. The encoder cannot produce such a thing, and
`check_annotations` catches it *indirectly* — it recomputes `area` and `bbox` from the
decoded mask and compares to the recorded values, so a truncated counts array would
mismatch both. So this is a hardening note, not a live defect. A one-line length check in
`rle_decode` would turn a silent corruption into an exception for any hand-edited or
third-party sidecar.

## 9. `rle_encode` on a genuinely multi-valued array corrupts silently *[executed]*

`rle_encode(np.array([[0,2],[3,0]]))` returns `counts=[1,1,1,1]`, which decodes to
`[[0,0],[1,1]]` rather than the binarised `[[0,1],[1,0]]`. The encoder compares raw values
(`v == last`) rather than truthiness, so an array with two *different* nonzero values
breaks the alternation.

I initially suspected the common 0/255 mask convention would trigger this — **it does
not**. `[[0,255],[255,0]]` and `[[0,1],[1,0]]` both encode to `counts=[1,2,1]` and decode
identically; any two-valued array with 0 as background is handled correctly. Only a
genuinely multi-valued array (an *ids map* passed where a mask was expected) misbehaves.
Both live callers pass boolean arrays — `masks_from_index` passes `(ids == pid)`,
`labelme_to_seg` passes a boolean AND cast to uint8 — so this is a misuse trap, not a live
bug. `m = np.asarray(mask, dtype=np.uint8) > 0` would close it.

---

## Test suite

`pytest tests/test_annotate_masks.py tests/test_seg_dataset.py tests/test_labelme_to_seg.py
tests/test_check_annotations.py -q` → **86 passed** *[executed]*.

## What I did not verify

- No real photographs exist, so §6's evidence is a synthetic LabelMe export I wrote, not a
  human annotation of real hardware. The converter is proven mechanically correct and
  loader-compatible; it is not proven ergonomic.
- I did not re-render anything. The chain was audited from the Blender index pass
  *onward*; whether the index pass itself faithfully represents the CAD is Audit H's
  ground.
- `recog/realtest/annotations/instances_default.json` is a VOC-derived detector file with
  polygon-style `segmentation`, not a COCO-RLE sidecar; it is outside this chain and was
  not audited here.

## Recommended fixes, in priority order

1. **Add a `unit_id` check to `check_annotations.py`** — flag missing/`None` ids, and flag
   an image where every annotation shares one id. This is the one gap that lets a silent
   crop collapse (§7b) through the validator that exists to prevent exactly that.
2. **Correct the `unit_id` uniqueness docstring** in `scene.py:493-505` — say
   "unique within a scene", and note that consumers must bucket by `image_id` first (§7a).
3. **Assert `sum(counts) == h*w` in `rle_decode`** (§8) — one line, turns a silent
   corruption into an exception.
4. **Binarise in `rle_encode`** (§9) — `np.asarray(mask, dtype=np.uint8) > 0`.
5. **Carry `asset` through `labelme_to_seg`** (§6) so `--per-sku` works on real photos.
6. **Wire `check_annotations` into CI** over one synthetic dataset — cheap, and it would
   catch a future regression in the mask-building code that the current structural
   guarantee (§5) is silently carrying.

None of these change any published number. The ground truth is sound.
