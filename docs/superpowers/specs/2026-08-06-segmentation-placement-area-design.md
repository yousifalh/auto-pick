# Segmentation-driven placement-area extraction

Design spec — 2026-08-06

Supersedes FDR §13.2(5) and reclassifies §13.2(2). Design only; no
implementation is proposed here. The implementation plan is a separate
document.

§9 and §10 specify what an implementation **must produce** in order to be
considered complete. They are acceptance criteria for a future cohort, not work
in progress. The measurements in §1.1 and §3.2 were run against the current
repository and are results, not projections.

## 1. Motivation

The planner decides where a cell may be placed inside a cartridge by calling
[`PlacementAreaExtractor.extract`](../../../plan/placement_area.py), which runs
a fixed classical pipeline: Otsu-threshold the green channel of the cartridge
ROI, take the largest bright contour, inset it by 5 px, then subtract the
*dark* region inside as the PCB.

### 1.1 Measured behaviour on the held-out photographs

The extractor was run on all 20 annotated cartridges in `recog/realtest/`, using
the ground-truth cartridge boxes so that detector error plays no part.

| Quantity | Result |
|---|---|
| Cartridges processed | 20 (no exceptions raised) |
| Placeable fraction of ROI — mean / median | 0.218 / 0.224 |
| Returned **exactly zero** placeable area | **7 / 20** |
| Below 0.10 placeable | 8 / 20 |
| `pcb_mask` fraction — mean / max | 0.231 / 0.718 |

Inspecting the two stages separately on IMG_4426's five cartridges shows the
mechanism, and it is not the clean inversion an earlier draft of this spec
claimed. The two stages fail for different reasons and the result is unstable
rather than consistently complementary:

| Cartridge | Placeable | What the stages did |
|---|---|---|
| 1 | 0.342 | `_green_mask` selected the **white tape cross** as the placement area |
| 2 | **0.000** | Selected a small printed label; PCB subtraction was correct; area still nil |
| 3 | 0.360 | Selected the bay *and* a red auxiliary board; subtracted only the green one |
| 4 | 0.097 | Selected the **adhesive fingers**; subtracted glue blobs and the PCB |
| 5 | **0.000** | Classified the **entire battery bay as PCB** and subtracted it |

The accurate characterisation is therefore in two parts:

- **`_green_mask`** ([L118-133](../../../plan/placement_area.py#L118-L133))
  Otsu-thresholds for brightness, so on a black cartridge it selects whichever
  foreign matter happens to be brightest — tape, a label, adhesive, or a
  differently-coloured board. Which one wins varies per cartridge, so the output
  is not a stable function of the geometry it is supposed to measure.
- **`_infer_pcb_mask`** ([L169-201](../../../plan/placement_area.py#L169-L201))
  applies a fixed `gray < 80` threshold. The hardware's battery bay is black, so
  on cartridges 4 and 5 the bay itself is classified as PCB and subtracted —
  those two *are* genuine inversions, and they drive the 0.718 maximum
  `pcb_mask` fraction.

So the failure is not that the axis-aligned rectangle is a coarse approximation
of a curved interior, which is how FDR §13.2(5) frames it. **On 7 of 20
cartridges the extractor returns no placeable area at all**, and on the rest it
returns a region determined by incidental foreign matter. That is the motivation
for replacing it, and it is measured rather than argued.

The `PlacementAreaExtractor` docstring attributes the design to PPR §5.3.2 and
assumes a green tray with a dark interior module. That assumption is
self-consistent; the hardware in `recog/realtest/` simply is not that hardware.
The defect is an unstated scope limit, not a coding error.

### 1.2 The CAD already contains the electronics module, as negative space

Walking the world-space transforms of the four converted assemblies gives the
free interior strip on each side:

| Assembly | Case interior (mm) | Cell union (mm) | −x | +x | −y | **+y** |
|---|---|---|---|---|---|---|
| PowerCore10000 | 62.9 × 90.9 | 54.9 × 65.0 | 4.0 | 4.0 | 2.4 | **23.5** |
| PowerCore13000 | 80.7 × 97.0 | 73.2 × 65.0 | 3.8 | 3.8 | 5.5 | **26.5** |
| PowerCore20100 | 62.3 × 167.8 | 54.9 × 133.0 | 3.7 | 3.7 | 5.9 | **28.9** |
| PowerCore26800 | 81.7 × 180.0 | 73.2 × 140.0 | 4.2 | 4.2 | 5.0 | **35.0** |

The ±x and −y gaps are wall thickness. The +y gap is 23–35 mm, scales with pack
size, and sits on one short side in every assembly. That is the module bay. The
battery placement area is therefore derivable from CAD already in the
repository: it is the cells' union footprint in assembled pose. No new geometry
needs authoring to produce the *label*.

### 1.3 The real bays are not empty

IMG_4426 shows thermal adhesive blobs, white foam pads, tape crosses and printed
labels sitting in the battery bays. None of it is in the CAD. A pixel-precise
placement mask derived from clean CAD would site a cell on a glue blob — and
§1.1 shows this foreign matter is already what the current extractor latches
onto.

## 2. Goals and non-goals

**Goals**

- Five-class segmentation covering `battery`, `cartridge`, `electronics_module`,
  `placement_area` and `obstruction`.
- A placement mask that is physically valid on the real hardware — the visible
  free floor, with the module, any foreign matter and any cell already resting
  in the bay all excluded.
- Two independent estimates of the placement area, reconciled by an explicit,
  documented arbitration rule rather than left to disagree silently.
- Ground truth generated by extending the existing Blender pipeline, reusing the
  object-index pass that already produces pixel-exact labels.
- **Training data containing partly-filled cartridges** (§5.3), because that is
  what the deployed system spends most of its time looking at.
- No regression to the existing detector: its Pascal-VOC training path, its
  class set, its checkpoints and its published numbers are untouched (§4.1).

**Non-goals**

- Replacing the Faster R-CNN detector. It keeps finding `battery` and
  `cartridge`; this extension consumes its cartridge boxes.
- Generalising the packer to non-rectangular domains. That is FDR §13.2(7) and
  stays separate.
- Real-robot validation. Software-only, as with the rest of the project.
- Any claim of synthetic-to-real transfer. §9.1 states what the available real
  data can and cannot support.

This spec is **not** strictly additive at the module boundary: §3.2 moves
segmentation into Recognition, which adds one optional field to the `Snapshot`
contract. That is a deliberate, argued exception, not an oversight.

## 3. Architecture

```
Camera ─▶ Recognition ─────────────▶ Planning ─▶ Execution ─▶ KUKA
          ├ Faster R-CNN (unchanged)  ├ digital twin
          └ BaySegmenter (new)        ├ SegmentationPlacementAreaExtractor
                                      └ FFDH packer
```

Module placement follows a pattern the codebase already established: just as
[`inference.py`](../../../recog/inference.py) pairs `FasterRCNNDetector` with
`HeuristicDetector` behind one interface, placement-area extraction becomes a
protocol with two implementations.

```
recog/bay_segmenter.py    BaySegmenter.segment(roi_rgb) -> (H, W) int8 label map

plan/placement_area.py    PlacementAreaExtractor            (protocol)
                          ├─ HeuristicPlacementAreaExtractor  existing code + scope limit
                          └─ SegmentationPlacementAreaExtractor
```

The heuristic implementation is **not** an equal alternative. §1.1 measured it
returning zero placeable area on 7 of 20 real cartridges. Its role narrows to one
thing: keeping the torch-free demo runnable on
[`synth_dataset.py`](../../../recog/synth_dataset.py)'s flat green rectangles,
which are the hardware its PPR §5.3.2 assumption actually describes. Its
docstring gains that scope limit, and constructing it logs a warning naming the
assumption, so it cannot be selected for real imagery by accident.

The model lives in `recog/` because it is perception and shares the training and
evaluation infrastructure. The adapter lives in `plan/` because that is where
the packer's contract is owned. The
`extract(image_rgb, cartridge_bbox, ...) -> PlacementArea` signature is
preserved, so the torch-free demo path keeps working by selecting the heuristic
implementation.

### 3.1 Why a per-ROI segmenter rather than a Mask R-CNN head

FDR §13.2(5) currently proposes a Mask R-CNN head on the existing backbone. This
spec revises that, on a resolution argument.

torchvision's `MaskRCNNPredictor` emits **28 × 28 per instance by default**,
upsampled to the instance box. At the generator's framing — 1280 px across an
800 mm layout, ≈ 0.63 mm/px — a PowerCore26800 cartridge occupies roughly
131 × 288 px, so one mask cell covers **2.9 × 6.4 mm**. Against an 18.3 mm cell
diameter, a 6.4 mm quantisation along the long axis is a third of a cell. That
is the precision on which "does the last cell fit" turns.

The resolution is configurable — raising the mask ROI-pool output trades memory
and training cost for boundary precision, and the default is not a hard ceiling.
The argument for the ROI segmenter is therefore not "Mask R-CNN cannot do this"
but that operating on the crop sidesteps the trade entirely: masks come out at
crop resolution, which is exactly where the placement decision is made. The
costs are honest: two models to train and document, latency of detect +
N × segment (§3.2), and a missed cartridge box means no segmentation for that
unit at all.

Whole-frame semantic segmentation with connected-component instance recovery was
considered and rejected. FDR §10.6 attributes 22 % of heuristic detector misses
to OCCLUSION — "the green-mask + bounding-rect contour extraction collapses any
cluster of touching cells into a single blob". Connected-component instance
recovery reintroduces that exact failure mode on the same hardware, after the
project already paid to diagnose it.

**Amended 2026-08-16, on measurement.** That paragraph rejected the wrong thing.
It disposes of whole-frame *plus connected components*, but a whole-frame model
can keep the detector for instance identity and supply pixel labels only — a
cartridge's mask being the frame prediction intersected with its own box — and
no connected components appear anywhere. Against that design this section had
**no evidence at all**, only an objection that did not apply to it.

It does now. A whole-frame arm was pre-registered and trained
(`2026-08-16-wholeframe-preregistration.md`, `configs/wholeframe_segmentation.yaml`)
and scored on the same 126 validation crops through the same metric code, differing
only in the prediction step. Bay boundary displacement **1.226 → 6.852 mm**, bay IoU
**0.8903 → 0.6161**, and `obstruction` **0.6579 → 0.0000** — not learned at any epoch,
because obstructions are the smallest objects in the corpus and 512/1280 leaves them a
few pixels wide. Receipt: `docs/receipts/wholeframe_comparison.txt`.

So the conclusion survives and the *reason* is replaced: this section rests on
resolution per object at this scale, not on connected components. Two limits on
that, both stated in the pre-registration and neither resolved here: class
imbalance is not excluded as a contributor to the `obstruction` collapse, so
"the default whole-frame configuration is unusable here" is the claim, not "no
whole-frame configuration could work"; and the cost argument is scale-dependent —
per-ROI pays 2.3 ms per cartridge against a fixed whole-frame pass, so the two
cross at roughly **14 cartridges per frame**, and this corpus carries 1–8.

### 3.2 Where the segmenter runs, and the O3 budget

FDR O3 is a **tested requirement**: queue rebuild ≤ 8 ms per cartridge, verified
in `tests/test_bin_packing.py` and `tests/test_planner.py`, currently met with
two orders of magnitude of headroom (median 6 ms per cycle).

[`_ensure_placement_areas`](../../../plan/planner.py#L121-L133) currently caches
— it skips any cartridge that already has a `placeable_rectangle`. Ruling 4
removes that option: a modal placement area is only valid for the frame it was
computed from, so the mask must be recomputed every cycle.

DeepLabv3 + MobileNetV3-Large was benchmarked on the project's own RTX 3060:

| Configuration | 1 cartridge | 8 cartridges | per cartridge |
|---|---|---|---|
| fp32, 384², unbatched | 12.6 ms | 101 ms | 12.6 ms |
| fp32, 384², batched | — | 59.6 ms | 7.5 ms |
| fp16, 256², batched | — | **18.5 ms** | **2.3 ms** |
| fp16, 224², batched | — | 11.4 ms | 1.4 ms |

The naive configuration costs 12.6 ms for a single cartridge — **1.5× the entire
8 ms O3 allowance** — and 101 ms for eight, which alone exceeds the 50 ms
end-to-end PPR budget. O3 would be violated on every cycle, not just the first.

Three options were considered:

| Option | Effect |
|---|---|
| Keep in Planning, re-specify O3 as amortised | Rejected — changes a requirement to fit the design |
| Defer to a background worker | Rejected — introduces concurrency into a pipeline the FDR describes as strictly sequential, for a benefit (b) already delivers |
| **Move segmentation into Recognition** | **Adopted** |

Segmentation is perception, and it belongs in the perception budget alongside
the detector that feeds it. `Snapshot` gains one optional field carrying a
per-cartridge label map; `SegmentationPlacementAreaExtractor` consumes it
instead of re-running the model. Planning then performs only mask arithmetic
(§7), which is O(pixels) and comfortably inside 8 ms.

Moving the model is necessary but not sufficient — it changes which budget is
overrun, not whether one is. Two further requirements follow from the table, and
both are binding rather than optional:

- **All cartridge crops in a frame are segmented as one batch**, not in a loop.
  Batching alone takes eight cartridges from 101 ms to 59.6 ms.
- **Inference runs at fp16 on 256 × 256 crops**, giving 18.5 ms for eight
  cartridges — inside the 50 ms budget with room for the detector.

256² is not a resolution compromise. At the generator's framing a PowerCore26800
cartridge occupies roughly 131 × 288 px, so a 256² input is at or above the
crop's native resolution; the §3.1 argument (256 versus 28, a factor of nine
linearly) survives intact. Resizing to 384² was upsampling, and paid 3× for it.

This modifies a module contract, which the README calls out as a design
principle. The field is optional and defaulted, so existing producers and
consumers remain valid, and the alternative is a design that knowingly breaks a
tested requirement.

The **50 ms PPR end-to-end budget** (FDR §10.4) is the one that now absorbs the
cost, and it must be re-measured rather than assumed. §9 makes that a reported
deliverable.

## 4. Class definitions

These definitions must be applied identically by a Blender script and by a human
in CVAT, so they are stated to be unambiguous rather than merely descriptive.

| Class | Definition | Instance rule |
|---|---|---|
| `battery` | One 18650 cell, visible silhouette | One per cell |
| `cartridge` | Power-bank case — whole silhouette when assembled, shell when open | One per unit |
| `electronics_module` | PCB assembly including connectors, inductor and potting | One per module, in-cartridge or loose |
| `placement_area` | Visible interior floor where a cell may legitimately sit — the floor, inset from the wall, that no module, obstruction or already-placed cell covers | One per **open** cartridge |
| `obstruction` | Foreign matter in the bay — adhesive, foam, tape, printed labels | One per blob, pad or strip |

`battery` is defined as 18650 only. The four converted assemblies contain
18.3 × 65 mm cells exclusively; there is no 21700 geometry in `cad/`. Extending
the class to 21700 requires new CAD and is out of scope here.

Five rulings close the ambiguities:

1. **`placement_area` is defined on the interior floor plane**, not the projected
   interior volume. A cell overhanging a wall projects inside the cartridge
   silhouette but does not sit on the floor.
2. **A sealed cartridge yields no `placement_area` and no `electronics_module`**,
   by construction rather than by occlusion: both are built only for the
   `open_case` variant ([scene.py:213-218](../../../recog/synth3d/scene.py#L213-L218)),
   so an assembled unit never has them to begin with.
3. **A loose module is labelled but subtracted from nothing.** IMG_4426 shows one
   in its own jig pocket. Subtraction requires containment in a specific
   cartridge.
4. **`placement_area` is the *currently free* bay**, not the nominal empty one.
   A cell resting in the bay carves itself out of it, as do the module and any
   obstruction.

   An earlier draft ruled the opposite — nominal bay, with occupancy tracked
   separately in `OccupancyGrid`. That forced *amodal* labelling: the network
   would have had to report the floor surface underneath an occluding cell.
   Amodal segmentation is a materially harder problem than modal, it would have
   required a second index pass to generate, and the resulting mask would have
   been cached and progressively stale. The modal definition deletes all three
   problems and answers the planner's actual question — where can a cell go
   *now* — including for cells the planner did not place itself.

   `OccupancyGrid` narrows to within-cycle reservation: the visually-free area
   is the starting state each frame, and the grid marks cells as the queue is
   built. State is re-derived from vision every frame rather than accumulated,
   so a shifted cell or a human intervention self-corrects.
5. **Every class is labelled *modally*** — what the camera can see, nothing
   inferred behind an occluder. This follows from ruling 4 and applies equally
   to the Blender ground truth and to the CVAT annotation guide.

### 4.1 Two class sets, not one

An earlier draft grew `CLASSES` in
[`config.py:35`](../../../recog/synth3d/config.py#L35) from 2 to 5. That
contradicts the no-regression goal: `CLASSES` is pinned to
`recog.dataset.CLASS_MAP` by test, `CLASS_MAP` sizes the detector's
classification head, and changing it invalidates every existing checkpoint and
published number.

The generator therefore carries **two class sets**:

- `CLASSES` — `battery`, `cartridge`. Unchanged. Written to Pascal-VOC XML, read
  by the detector's training path, pinned to `CLASS_MAP` by the existing test.
- `SEG_CLASSES` — all five. Written only to the COCO sidecar (§5.5), read only
  by the segmenter.

The VOC writer filters to `CLASSES`, so its output for an existing scene is
unchanged except where §5.1 moves the module.

## 5. Synthetic ground truth

### 5.1 The module exists, in the wrong place

[`world.build_pcb`](../../../recog/synth3d/world.py#L569) already constructs a
procedural green board with 3–7 extruded components for the `open_case` variant,
deliberately unlabelled at `pass_index = 0`. Three changes:

- **Position.** The board is currently centred: `cx = (x0+x1)/2`,
  `cy = (y0+y1)/2 ± 10 mm`
  ([L583-584](../../../recog/synth3d/world.py#L583-L584)). Both the CAD negative
  space and IMG_4426 place it hard against one short side. Its *size* is already
  approximately right — `h = (y1-y0) × U(0.20, 0.38)` yields 18–35 mm on the
  10000 against a measured 23.5 mm bay — so only the anchor is wrong.

  This is a **realism** defect, not a labelling one. An earlier draft claimed it
  mis-shapes the cartridge silhouette for the current detector; that is wrong. A
  centred board punches a hole in the middle of the case's visible pixels, and
  `boxes_from_mask` takes the min/max of non-zero pixels, so the bounding box is
  unaffected. Moving the board to the short side leaves the surrounding wall
  visible, so the box is unaffected there too. The detector's *boxes* do not
  change; what changes is the appearance the detector learns from, which no
  longer contradicts every real photograph.
- **Labelling.** The board and its components gain a `pass_index`, becoming
  `electronics_module` in `SEG_CLASSES`. `layout.plan` already rotates cartridges
  and `allow_90s` flips them, so the module's orientation randomises in image
  space for free. It must stay fixed on the short side in *cartridge* frame,
  because that is what the hardware does; randomising the side would be
  unfaithful to the design.
- **Detail.** Gold USB shells and a copper inductor. In IMG_4426 those are the
  module's most distinctive features; the current board has only dark cuboids.

### 5.2 Measurements live in the catalog

[`catalog.inspect_glb`](../../../recog/synth3d/catalog.py#L58-L84) already walks
sub-parts and records extents. It is extended to record, per assembly, the
world-space **cell-union AABB** and the derived **module bay** — the
23.5 / 26.5 / 28.9 / 35.0 mm strips of §1.2.

`build_pcb`, the placement-area proxy and the `wall_inset` of §7 then all read
one source of truth computed from the CAD, rather than several hardcoded
constants that can drift apart. Adding a new assembly to `cad/` propagates
automatically.

### 5.3 The generator has no open cartridges, and must

This is the largest gap in the current pipeline.

`open_case` keeps `("cell", "case")`, and
[`assets.instantiate`](../../../recog/synth3d/assets.py#L200-L201) clones every
role in `keep_roles`, merging the shell halves into one object. So the "open
case" is a **closed shell** with cells scattered *beside* it, plus a fake board
laid on its top surface — as `build_pcb`'s own docstring states: *"the board is
laid on top of the shell rather than modelled inside it: from a bird's-eye
camera the two read the same, and this needs no interior geometry."*

Two consequences:

**The bay proxy is a plane on the shell top, not on an interior floor.** There is
no interior floor. The proxy is coplanar with the fake board, occupying the
complement of the module footprint within the shell's top face — the same trick
`build_pcb` already uses, and equally valid under a bird's-eye camera.

**A populated-bay layout mode is required — but the machinery already exists.**
The deployed system fills cartridges one cell at a time, so the camera sees
partly-filled bays for most of every run, and a segmenter trained only on empty
bays would first meet a partly-filled one during the operation it exists to
support.

An earlier draft claimed `layout.plan` guarantees exact AABB non-overlap so a
cell can never sit on a cartridge. That was read from
[`isolated_areas`'s docstring](../../../recog/synth3d/render.py#L352-L365),
which is stale. `LayoutCfg.max_overlap_iou` is live, `configs/synth3d.yaml` sets
it to **0.20**, and [`scene.py:198-207`](../../../recog/synth3d/scene.py#L198-L207)
passes it to `plan()` together with per-item `heights`, so overlapping parts are
lifted onto one another via `Placement.z`. Cells already land on cartridges
today, and partly-occluded bays already occur — incidentally.

What is missing is not the capability but the *control*. Random scatter lands a
cell anywhere on a shell at an arbitrary angle; the deployed system produces
cells seated in the bay, axis-aligned, at the pitch the packer chose. The new
mode is therefore a targeted placement built on the existing lift machinery —
sample 0–N of the packer's own placements for that cartridge and seat cells
there — not new overlap infrastructure.

### 5.4 One index pass, unchanged

Under ruling 4 the existing single index pass produces every label directly. The
bay proxy carries its own `pass_index` and is rendered alongside everything else;
cells, module and obstructions resting on it occlude it, and
[`boxes_from_mask`](../../../recog/synth3d/annotate.py#L33-L36) already reports
only visible pixels. `placement_area` therefore emerges as *exactly* the free
floor, with no set arithmetic and no second render.

This is the main dividend of the modal definition. An earlier draft specified a
second index pass with cells hidden, to recover the nominal bay; it is deleted
along with the amodal requirement that motivated it.

One consequence must be recorded rather than inherited.
[`isolated_areas`'s docstring](../../../recog/synth3d/render.py#L352-L365)
still states that `--visibility` yields no signal because nothing labelled can
occlude anything labelled. That has not been true since `max_overlap_iou` went
live at 0.20 (§5.3); the docstring is stale and should be corrected as part of
this work, since a future reader will otherwise re-derive a conclusion from it
as this spec's earlier draft did.

**The existing filters must not apply to `placement_area`.** This follows
directly from ruling 4 and is easy to miss.
[`FilterCfg`](../../../recog/synth3d/config.py#L136-L141) drops any instance
below `min_px = 80`, `min_side = 6` or `min_visibility = 0.25`. Under the modal
definition a nearly-full cartridge has a *small, thin, mostly-occluded* strip of
free floor — which is precisely the configuration all three filters discard. The
generator would then silently emit no `placement_area` for exactly the cartridges
where knowing the remaining room matters most, and the segmenter would learn that
a nearly-full bay has no placement area at all rather than a small one.

The ruling: `placement_area` is exempt from `min_px`, `min_side` and
`min_visibility`. A bay with one visible pixel of free floor is a one-pixel
`placement_area`, and it is the arbitration and packing stages' job to decide
that no cell fits there — not the annotation filter's. A bay with *zero* visible
free floor correctly yields no instance, since it never appears in `np.unique`.

### 5.5 Obstructions

Procedural, each with its own `pass_index` and therefore its own instance:

| Kind | Geometry | Randomisation |
|---|---|---|
| Adhesive blob | Displaced sphere, translucent white | 0–6 per bay, position, scale |
| Foam pad | Rounded box, white or grey | 0–1 per bay, position |
| Tape strip | Thin plane, matte white | 0–2 per bay, orientation |
| Printed label | Textured plane | 0–1 per bay, position |

Roughly 40 % of open cartridges receive none, so the network also sees clean
bays. Obstruction pose is sampled within the bay footprint from the catalog.

### 5.6 Output format

Pascal-VOC XML has no mask field. Generation writes a **parallel COCO JSON with
RLE segmentation** alongside the existing VOC output, carrying `SEG_CLASSES`.
[`dataset.py`](../../../recog/dataset.py) already holds a COCO reader for
`realtest/`. The detector's training path keeps reading the VOC output at
`CLASSES` and therefore cannot regress.

## 6. The segmenter

This is **per-ROI semantic segmentation**, not instance segmentation. Instances
come from the detector: one crop contains one cartridge, so a semantic label map
over that crop yields one `placement_area` instance by construction.

**Input.** The cartridge ROI cropped from the full frame and resized to a fixed
input resolution.

**Output.** Six channels: `background`, `cartridge`, `bay`, `electronics`,
`obstruction`, `battery`.

`battery` needs a channel of its own. Under ruling 4 a cell resting in the bay
carves itself out of the placement area, so the derived estimate of §7 must
subtract cells — and it cannot subtract what it cannot see. The detector's
battery boxes are not a substitute: they are axis-aligned rectangles around
cylinders lying at arbitrary angles, so a box over-subtracts by roughly the
corner area a rotated cylinder does not fill.

The shell needs its own channel. With only four — folding the shell into
background — the derived estimate of §7 collapses to `erode(bay)`, becoming
near-identical to the direct estimate, and the redundancy check turns vacuous.
Giving the shell a channel makes the derived estimate a function of the *other
three* channels, so disagreement carries information.

**Multiple cartridges in one crop.** Cartridges are adjacent in the jig, and a
jittered or over-large box will include a neighbour's edge. The `cartridge`
channel then contains two blobs and a naive `P_derived` would erode their union.
The adapter therefore keeps only the connected component containing the crop
centre, which is the detected cartridge by construction.

Connectivity is computed over the union of **all five** non-background
channels — `battery` included. Excluding it would let a cell lying across a bay
sever the region into two components, leaving the centre component covering
half the cartridge. Cells are part of the cartridge's footprint for the purpose
of deciding what is connected, and are subtracted only afterwards.

**Backbone.** DeepLabv3 with MobileNetV3-Large. Small, available in torchvision,
and sized appropriately for a single-ROI crop.

**Training crops are harvested from jittered boxes, not ground-truth boxes.** At
inference the crop comes from the detector, so training on perfect boxes would
bake in a distribution shift the model then meets for the first time in
deployment. Jitter magnitude is sampled from the detector's measured box error on
the validation set, not chosen arbitrarily.

**Loss.** Cross-entropy plus Dice. `obstruction` is small in area and absent from
roughly 40 % of samples, so an unweighted cross-entropy would let the model
ignore it at low cost.

**Augmentation.** [`augmentation.py`](../../../recog/augmentation.py)'s existing
photometric pipeline is reused, with albumentations `mask` targets wired through
so masks ride with the geometric transforms. The existing box-only path stays
available for the detector.

## 7. Arbitration

Two estimates of the placement area are produced per cartridge:

- `P_direct` = the `bay` channel.
- `P_derived` = `erode(C, wall_inset) ∖ electronics ∖ obstruction ∖ battery`,
  where `C` is the centre-containing connected component of the union of all
  five non-background channels (§6). Eroding `C` yields the interior floor;
  the three subtractions remove everything occupying it.

`wall_inset` is **not** the existing hardcoded `safety_margin_px = 5`. It is
derived per assembly from the catalog's measured wall thickness of §5.2 — the
2.4–5.9 mm ±x and −y strips — and converted to pixels using the **calibrated
scale of the deployed framing**, not `planning.yaml`'s default `mm_per_px:
0.38`. That default is itself a single-framing constant, so converting through
it would reproduce one level up the very error this paragraph rejects. The
scale is a property of the camera mount and belongs in the calibration the cell
already requires.

**The planner consumes `P_safe = P_direct ∩ P_derived`.** A pixel is placeable
only where both estimates agree. The asymmetry is deliberate: siting a cell on a
PCB is a damage event, whereas skipping a cartridge costs one cycle.

**`IoU(P_direct, P_derived)` is a per-cartridge confidence gate.** Below τ the
cartridge is skipped for that cycle and logged as `placement_disagreement` — a
third planner failure mode beside the `empty_queue` and `pick_failed` already
reported in FDR §10.6.

### 7.1 Calibrating τ

τ is chosen, not assumed. On the synthetic validation set, for each cartridge
compute `IoU(P_direct, P_derived)` and the **signed optimistic area error** —
the area of `P_safe` that falls outside ground-truth placeable region, which is
the quantity that causes a cell to be sited on a PCB.

The criterion on that error is **morphological, not areal**. An earlier draft
required the error area to fall below one cell footprint (1190 mm²); that is the
wrong test, because 1190 mm² spread along a boundary as a one-pixel rim is
harmless, while the same area in a single blob is a misplacement. What matters is
whether the optimistic-error region can *contain* a cell.

The test is therefore: erode the optimistic-error region by an 18.3 × 65 mm
structuring element, at every orientation the packer may use. If the result is
empty, no cell can be sited inside the error and the cartridge is safe.

τ is the smallest threshold for which at most 5 % of accepted cartridges fail
that erosion test — smallest, because larger τ rejects more cartridges, so this
maximises throughput subject to the safety bound. Reported alongside τ: the
fraction of cartridges rejected at that threshold, so the accuracy/throughput
trade is visible rather than buried. If no τ satisfies the criterion, that is a
negative result about the design and is reported as one.

This turns four-way class redundancy into a safety interlock rather than an
unresolved contradiction between two outputs.

## 8. Integration with the packer

`PlacementArea` gains `bay_mask`, `electronics_mask`, `obstruction_mask` and
`consistency_iou`. `pcb_mask` is retained and populated with `electronics_mask`,
so existing consumers are unaffected. `rectangle` becomes the axis-aligned
bounding box of `P_safe` and continues to serve as the occupancy grid's origin.
`_rasterise` and the FFDH packer need no change.

**They do, however, need fixing first.** FDR §6.3.1 benchmarks the
forbidden-mask FFDH variant against a rejection-sampling baseline and reports
mean cells placed collapsing from **23.0 at 0 % forbidden coverage to 3.2 at
2.5 %**, with the root cause identified: when a placement overlaps a forbidden
cell the implementation abandons the entire shelf rather than advancing the
cursor past the obstacle. The masks in that benchmark were drawn as "small 2–6
cell rectangular blobs to mimic PCB obstructions" — which is precisely what this
extension will now deliver, except measured rather than simulated.

Feeding pixel-precise obstruction masks into the current shelf-cursor logic would
therefore make packing *worse* than the rectangle it replaces.

**FDR §13.2(2) is reclassified from priority-2 future work to a blocking
prerequisite of §13.2(5).** The FDR currently presents the two items as
independent programmes pursuable in any order. They are not, and the ordering is
load-bearing.

## 9. Evaluation

Per-class mask IoU and mask AP@[.5:.95] are reported as table stakes. Four
metrics carry the argument:

- **Boundary displacement in millimetres.** This is the quantity that justified
  the architecture choice in §3.1 over a 2.9 × 6.4 mm mask-head quantisation.
  Reporting only IoU would hide it.
- **Δcells**, measured **on the fixed packer of §8**. Run the packer on the
  ground-truth mask and on the predicted mask; report cells lost per cartridge.
  Measured on the current packer the figure would be dominated by the
  shelf-cursor defect rather than by mask error, and would say nothing about the
  segmenter.

  This is the only metric gated on §8. Mask IoU, boundary displacement, signed
  area error and latency are all measurable against a segmenter alone, so the
  two work programmes can proceed in parallel and need only converge before
  Δcells is reported.
- **Signed placeable-area error in mm².** Signed, so that optimistic error
  (placing where a cell cannot go) is distinguishable from conservative error
  (refusing where it can). Only the first is a damage risk, and it is the
  quantity §7.1 calibrates against.
- **End-to-end cycle latency against the 50 ms PPR budget** (§3.2), reported for
  1, 2, 4 and 8 cartridges in frame, since the cost is linear in cartridge count.

**The headline ablation is heuristic versus segmenter on the real photographs**,
scored as placeable-area IoU against human polygons. The heuristic's baseline is
already measured: mean placeable fraction 0.218 with 7 of 20 cartridges at zero
(§1.1). Any segmenter that cannot beat that is not worth shipping, and the
comparison is against a number rather than an expectation.

### 9.1 Real-image validation

`recog/realtest/annotations/instances_default.json` currently holds 7 images,
2 categories, 80 boxes and **zero segmentation polygons**. There is no existing
real basis on which a mask model can be validated.

A two-tier approach:

1. **Now.** Re-annotate the existing 7 photographs with 5-class polygons in CVAT,
   preserving the held-out provenance. Every reported number carries its
   per-class instance count, and the set is framed explicitly as a smoke test.
   With 20 cartridges in total, per-class AP for `placement_area`,
   `electronics_module` and `obstruction` rests on double-digit instance counts
   at best and cannot support a transfer claim.
2. **Prerequisite for any transfer claim.** A 50–100 image polygon-annotated
   collection, folding into the 200–500 image programme already proposed in FDR
   §13.2(4). No synthetic-to-real transfer figure is published before this
   exists.

Annotating `placement_area` by hand means tracing only the floor the camera can
see, stopping at the edge of any cell, module or obstruction resting on it
(ruling 5). This is easier and more repeatable than the amodal convention an
earlier draft required, but the boundary between "bay floor" and "shadowed bay
floor" is still a judgement call under oblique lighting. The annotation guide
must fix a convention, and a second annotator scores agreement on a subset.

## 10. Testing

| Area | Test |
|---|---|
| `annotate` | RLE round-trips through write and read without loss |
| `annotate` | A sealed cartridge yields zero `placement_area` instances |
| `annotate` | A cell placed on the bay reduces `placement_area` by its silhouette |
| Layout | The populated-bay mode places cells within the bay footprint only |
| Adapter | Connectivity spans `battery`, so a cell across a bay does not split it |
| Budget | 8 cartridges segment within the 50 ms PPR budget at fp16 / 256² / batched |
| Containment | A loose module is labelled and subtracted from nothing |
| `catalog` | Bay measurements reproduce the four known strips of §1.2 |
| Adapter | A crop containing two cartridges keeps only the centre component |
| Arbitration | Constructed mask pairs exercise `P_safe` and the τ gate |
| Arbitration | The τ erosion test rejects a cell-sized blob and accepts a boundary rim of equal area |
| Contract | One shared test both extractor implementations must satisfy |
| Regression | `CLASSES` still equals `CLASS_MAP`; VOC output still 2-class |
| Regression | The no-torch demo path still runs with the heuristic extractor |
| Budget | Planning stays under 8 ms per cartridge with masks supplied |

## 11. Limitations

- **The latency budget has no headroom left.** §3.2 fits eight cartridges into
  18.5 ms only at fp16 on 256² crops with full batching. All three are load-
  bearing; losing any one puts the design back over budget. A frame with more
  cartridges than the batch was sized for, or a fallback to fp32 on hardware
  without fast half-precision, breaks the 50 ms PPR budget rather than degrading
  gracefully. The cartridge count per frame must be bounded explicitly, and the
  bound stated in the configuration rather than discovered in deployment.
- **Modal `placement_area` couples the mask to occupancy.** A cell that the
  detector sees but the segmenter labels `bay` — or the reverse — now changes the
  placeable region directly, where under the nominal definition it would only
  have changed the occupancy grid. The §7 intersection is the mitigation, but the
  failure is quieter than a missed detection and the `placement_disagreement`
  rate is the only signal that it is happening.
- **Procedural adhesive is not real adhesive.** `obstruction` will have both the
  weakest synthetic fidelity and the fewest real instances — the worst
  combination in the class set.
- **The CAD carries no colour**, while the hardware is black on a blue jig.
  Material randomisation must span that range or the domain gap is structural.
- **Seven photographs are not a validation**, and no number derived from them is
  presented as one.
- **The bay proxy assumes the CAD cell layout equals the true placement area.**
  This holds for the four Anker assemblies, where cells fill the bay. It is false
  for any cartridge design where they do not, and would under-report the
  placement area there.
- **Detector recall becomes a hard ceiling.** A missed cartridge box means no
  segmentation for that unit at all — the cascade cannot recover what the first
  stage never proposed.
- **The bay proxy is a flat plane on the shell top.** It reads correctly under
  the near-orthographic bird's-eye camera the generator uses
  (`camera.ortho: true`). It would not survive a change to a strongly perspective
  or oblique viewpoint, which would expose that no interior geometry exists.

## 12. Changes required to the FDR

1. **§13.2(5)** is rewritten. The motivation changes from "the rectangle is a
   coarse approximation" to the measured result of §1.1 — 7 of 20 held-out
   cartridges returning zero placeable area. The proposed architecture changes
   from a Mask R-CNN head to a detector plus per-ROI segmenter (§3.1).
2. **§13.2(2)** is reclassified from independent priority-2 work to a blocking
   prerequisite of §13.2(5), on the §6.3.1 benchmark evidence (§8).
3. **§6.2** gains a stated scope limit: the green-channel pipeline assumes a
   light tray with a dark interior module, per PPR §5.3.2, and does not hold on
   the black cartridges in `recog/realtest/`. §1.1's table is the evidence.
4. **§10.6's failure taxonomy** gains `placement_disagreement` (§7) as a planner
   failure mode alongside `empty_queue` and `pick_failed`.
