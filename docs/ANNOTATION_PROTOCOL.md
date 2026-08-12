# Annotation protocol: real-photo polygon ground truth

This is for the person holding a camera and a mouse, not a developer. If you
follow it exactly, the same photo labelled twice — by you a second time, or by
someone else — should come out pixel-for-pixel the same. That repeatability is
the entire point: three synthetic checkpoints have now scored 0.211, 0.232 and
0.318 against a fixed real-photo comparison, and the run-to-run noise is as
large as the effect we're trying to measure. Sloppy or inconsistent labels
would make that worse, not better.

You are producing **50-100 polygon-annotated photographs** across five
classes: `battery`, `cartridge`, `electronics_module`, `placement_area`,
`obstruction`. Read this whole document before annotating your first image —
most of it is rules for cases that look obvious until you hit them.

---

## 0. The target domain, in one sentence

**The deployed camera is a fixed overhead machine-vision camera: near
top-down, 0-10° off vertical, no roll, about 400 mm working distance** —
not the handheld phone that shot the photos already in `recog/realtest/`.
Every photograph you take for this set must approximate that geometry
(§7). This is not fussiness: a photo shot at a jaunty handheld angle
measures transfer to a domain the robot will never actually operate in,
and the whole reason this document exists is to make the real-photo
measurement trustworthy. See
`docs/superpowers/specs/2026-08-09-spec3-realism-decisions.md` for the
decision record.

---

## 1. The five classes

| Class | What it is |
|---|---|
| `battery` | One visible 18650 cell — loose on the bench, seated in a bay, or resting in a jig storage hole |
| `cartridge` | The power-bank case: its outer shell, and — for an open case — the interior tray walls and rim around the bay |
| `electronics_module` | The PCB assembly: the board itself, its connectors (USB shells), inductor, and any potting/glue that is part of the board's own construction |
| `placement_area` | The floor inside an open cartridge that is **currently free** — not covered by a module, an obstruction, or a cell, and not occluded from the camera by anything |
| `obstruction` | Foreign matter resting in the bay that is not part of the module or a cell: adhesive blobs, foam pads, tape strips, printed labels |

Everything not in one of these five classes — the blue jig plate, the metal
bench, cables, screws, tools, a shadow — is left unlabelled. There is no
"background" class to paint; you simply don't draw a polygon there.

### 1.1 `placement_area` is modal — the most important rule in this document

`placement_area` is **the floor the camera can currently see as free**, not
the floor that would be free if the cartridge were empty. If a cell is
sitting in the bay, the floor **under and behind** it is not
`placement_area` — it is occluded. You do not draw floor there at all. The
cell itself is `battery`.

This is easy to get backwards because your brain knows the floor is *there*
underneath the cell. The rule is: **draw only what the camera would need to
tell you is free, not what you know is physically underneath.** A robot
deciding where to place the next cell only has the picture, and if the
picture shows a cell, that spot is not a candidate — regardless of what's
under it.

**Worked example (IMG_4426.jpg, second row, right cartridge):** the PCB
module sits across the left portion of the bay, and a small white label sits
on the floor to its right. `placement_area` is the ribbed floor visible to
the right of the module, **with a hole cut out under the label** — not the
module's footprint, and not the label's footprint. If a cell were resting in
that same bay, `placement_area` would additionally stop at the cell's
visible edge, leaving no floor drawn under or behind it.

**If the free floor is visually split into two or more separate pieces**
(for example, a cell sitting in the middle of a bay leaves free floor on
both sides of it), draw **one `placement_area` polygon per piece**, all
carrying the same class label. Do not try to draw one polygon that dips
through the occupied area to connect the pieces — that would incorrectly
claim the occupied floor as free. See §5 for how these pieces are linked
together.

### 1.2 `cartridge` includes the tray walls and rim

This changed recently in the synthetic generator and the same convention
applies here: for an **open** cartridge, `cartridge` is not just the outer
case outline — it is the outer shell **and** the interior tray walls and rim
you can see standing around the bay. Under the near-top-down camera these
walls have real, visible depth (they are not a flat decal).

For a **sealed/closed** cartridge (no visible interior at all — just a solid
lid), `cartridge` is the entire visible outer silhouette, and nothing else.
A sealed cartridge never gets a `placement_area` or `electronics_module`
label, because there is no interior to see.

**Worked example (IMG_4426.jpg):** the second-row-left cartridge is open —
its four walls (visible as a dark rim around the lighter floor) are all
`cartridge`. The bottom-row-right cartridge is a smooth, featureless black
block with no ribbing or interior visible at all — that one is sealed:
`cartridge` only, drawn as one outline, nothing inside it.

**Practical note for LabelMe:** an open cartridge's wall/rim region is
shaped like a picture frame (a hole in the middle), and a LabelMe polygon
cannot have a hole. Draw the walls as **several separate rectangle-ish
pieces** (e.g. one per side, or however the visible geometry breaks up)
rather than trying to trace one shape with a hole in it. All the pieces get
the label `cartridge` and the same group id (§5). You do not need to be
careful about the pieces overlapping each other at the corners — see §4.

### 1.3 Truncated cartridges at the frame edge

Label **exactly what is visible**, stopping cleanly at the image edge — this
is the same modal principle as any other occlusion, just occluded by the
frame boundary instead of an object. Do not guess or extend a polygon past
the edge to complete a shape you can't see.

Do not avoid or discard a photo just because a neighbouring cartridge clips
the edge of the frame — that happens in real deployment too, and labelling
it (modally, only the visible part) is useful signal, not noise. What you
*should* avoid is framing your **primary** subject cartridge so badly that
most of it is cropped away (§7) — a neighbour clipping the edge is fine; the
cartridge you're centring the shot on should not be.

**Worked example (IMG_4426.jpg, fourth row, far left):** a cartridge is cut
off by the left edge of the frame, showing only a sliver of its case. Label
that sliver `cartridge` and stop at the frame edge. Do not attempt to guess
the shape of the rest of the unit.

### 1.4 `electronics_module` versus `obstruction`

Both can look like similarly-coloured blobs of white material sitting near a
PCB, and this is the second most common mislabel after the modal rule
above. Use this mechanical test:

> **If you lifted the module out of the bay, would the item come with it?**
> If yes (it's soldered, glued, or otherwise mounted to the board itself —
> connectors, an inductor, potting compound securing a component to the
> board) — it's `electronics_module`. If no (it would stay behind on the
> bay floor — a separate glue blob, a foam pad, a tape strip, a printed
> label) — it's `obstruction`, even if the material looks identical to
> potting compound.

**Worked example (IMG_4426.jpg):** the third-row-right cartridge has a
module on one side and a large smear of white adhesive across most of the
remaining floor, clearly not attached to the board — that smear is
`obstruction`. Contrast with the module in the second-row-right cartridge,
where a similar-looking white patch sits *on the board itself*, securing a
component — that patch is part of `electronics_module`, not a separate
`obstruction` polygon.

**Worked example, loose module (IMG_4426.jpg, third row, left):** a small
PCB sits by itself in its own jig pocket, with no cartridge shell around it.
Label the board `electronics_module`, on its own — do not invent a
`cartridge` polygon for the pocket, and do not subtract it from anything.
This mirrors design-spec ruling 3 exactly, which cites this same photograph.

### 1.5 Loose cells (not in a cartridge)

Cells resting directly in a jig's own storage holes, or lying loose on the
bench, are still `battery` — one polygon per visible cell. They are not
linked to any cartridge (§5): they get a group id of their own, or none at
all.

**Worked example (IMG_4426.jpg, top row):** seven cells (two lilac, one
green, four grey) sit in round holes drilled directly into the blue jig —
these are not inside any power-bank case. Label each one `battery`; do not
label the jig itself.

---

## 2. Ambiguous regions: a fixed rule, not your judgement

Under oblique or uneven lighting, a shadow can look like it might be free
floor or might be occluding something; a blurry boundary can leave real
doubt about exactly where a cell ends and free floor begins. **Do not decide
this case by case.** Use the same asymmetry the rest of this project uses
(design spec §7, `plan/arbitration.py`): siting a cell on a PCB or on
foreign matter is a damage event, while skipping a patch of genuinely free
floor only costs one throughput cycle. The safe direction and the unsafe
direction are not symmetric, so the default should not be either.

**The rule:** when you cannot confidently tell whether a pixel region is
free floor or something else, do **not** label it `placement_area`. Either
assign it to whichever more-specific class is a plausible reading
(`obstruction` is the usual fallback for "something is probably there"), or
leave it unlabelled entirely if no class fits with reasonable confidence.
Losing a sliver of ambiguous floor from the ground truth costs nothing but a
few pixels of recall; wrongly calling occupied or shadowed space "free"
teaches exactly the mistake this whole class exists to prevent.

**If ambiguity affects a large fraction of one photo** (heavy shadow
obscuring most of a bay, for instance), that photo is a poor candidate for
this set. Reshoot it under more even lighting (§7) rather than annotating
through the ambiguity — a cleanly-lit image you label confidently is worth
more than an ambiguous one you don't.

---

## 3. What NOT to draw

- The blue jig plate itself, the bench, screws, cables, tools — unlabelled,
  even where they fill most of the frame.
- A shadow on its own, with nothing physically resting there — unlabelled
  (it is not `obstruction`; nothing foreign is actually in the bay).
- Reflections and glare — unlabelled; do not let them split a
  `placement_area` polygon that is otherwise one continuous piece of visible
  floor.

---

## 4. You do not need pixel-perfect boundaries

This is the part that makes the job much less tedious than it sounds, and
it is a direct consequence of how the converter works (`recog/labelme_to_seg.py`),
not a suggestion.

The five classes have a fixed precedence, identical to the one
`recog.seg_dataset.rasterise_crop` already applies at training time:

```
cartridge  →  placement_area  →  electronics_module  →  obstruction  →  battery
(earliest)                                                            (wins ties)
```

Wherever two polygons of different classes overlap, **the later class in
this list wins the contested pixels automatically.** This means:

- It is *safe* to draw an earlier class generously into a later class's
  territory. If you're not sure exactly where the `cartridge` rim stops and
  the `placement_area` floor begins, draw the rim a little long — the
  `placement_area` polygon will correctly reclaim its own pixels regardless.
- It is *not* safe to draw a later class generously into an earlier class's
  territory — the reverse direction is not corrected. **Trace `battery` and
  `obstruction` boundaries carefully**: nothing rescues an over-generous
  battery polygon, and an oversized one will silently eat real
  `placement_area`, `electronics_module`, or `obstruction` pixels out from
  under it. When genuinely unsure, err toward drawing the *later*-in-the-list
  class slightly larger rather than the earlier one — the tie resolves
  correctly either way, but only in that direction.
- Two polygons of the **same** class that overlap (e.g. two `cartridge` wall
  pieces sharing a corner) resolve by whichever was drawn first — harmless,
  since both carry the same label anyway.
- If one polygon ends up **entirely** inside a higher-priority class's
  territory once this resolves, the converter drops it and tells you so.
  That's not an error — it usually means you drew something redundant — but
  read the note it prints, since it can also mean you drew the wrong shape
  in the wrong place.

Concretely: don't spend ten minutes tracing the exact hole in a
`placement_area` polygon around a module — draw the floor generously across
the module's area too, and the `electronics_module` polygon (drawn to its
own, more careful outline) will correctly cut the hole for you.

---

## 5. Linking a unit's shapes together

One physical cartridge's shell, its floor, its module, and anything sitting
in it are one **unit**. LabelMe lets you assign a numeric **Group ID** to a
shape (right-click a shape → "Edit Group ID", or select it and use the
Edit menu). Give every shape belonging to one physical cartridge the **same**
group id — including a cell that is seated in that cartridge's bay.

- A **loose** cell or a **loose** module (not inside any cartridge) gets no
  group id, or a group id used by nothing else — either way, leave it
  disconnected from every other shape's group.
- A cartridge with two disconnected `placement_area` pieces (§1.1) still
  uses just **one** group id shared across the cartridge, both floor pieces,
  the module, and anything else that belongs to that same unit.
- Different cartridges in the same photo get different group ids.

This maps directly onto the `unit_id` field the synthetic pipeline already
uses (`recog/synth3d/annotate.py`, read by
`recog.seg_dataset.BaySegDataset`) to build one training crop per physical
cartridge.

**Getting this wrong in the two possible directions is not symmetric.**

- **Forgetting** a group id **fragments**: each ungrouped shape becomes its
  own unit, so one cartridge comes back as three tight crops instead of one.
  Degraded, visible in the crop count, and not corrupting.
- **Sharing** one group id between two different cartridges **merges**:
  their shapes become one unit and you get a single crop spanning both, at
  the wrong scale, over the wrong content. Nothing downstream can tell.
  `recog.check_annotations` warns (`single_unit_id`) whenever every
  annotation in a photo shares one id — which is also what a legitimate
  single-cartridge close-up looks like, so the warning is a question, not a
  verdict: confirm the photo really shows one unit.

An annotation with **no** unit id at all is refused outright — both by the
validator (`missing_unit_id`, an ERROR) and by `BaySegDataset` itself,
which raises rather than load a sidecar carrying one. That case is not
reachable from LabelMe (§8's converter always assigns an id) but is very
reachable from a hand-edited JSON.

Ids need only be unique **within one photo**. The converter keys them on
the image file's stem (`photo1#g1`), so they happen to be unique across the
whole converted set as well — but do not rely on that: the synthetic
producer's ids are per-scene counters (`item0`) that repeat in every image
of the dataset, and every consumer must group by image first regardless.

---

## 6. The tool: LabelMe

Use **[LabelMe](https://github.com/wkentaro/labelme)**
(`pip install labelme`, then run `labelme`).

**Why LabelMe and not something else:** it's pip-installable and runs fully
offline — no server, no account, no upload of photographs of real hardware
anywhere. Its polygon tool is simple, and critically, its per-shape **Group
ID** field maps directly onto this project's existing `unit_id` concept
(§5) with no extra bookkeeping. CVAT (already used for the box-only labels
in `recog/realtest/`) can also do polygons, but needs a running server
(Docker or a hosted account) for what is, here, a single annotator working
locally — heavier setup for no benefit at this scale.

### 6.1 Setup

1. `pip install labelme`
2. Put your photographs in one folder, e.g. `recog/realtest_rig/images/`.
3. Run:
   ```
   labelme recog/realtest_rig/images \
       --labels battery,cartridge,electronics_module,placement_area,obstruction \
       --validate-label exact
   ```
   - `--labels` pre-populates exactly the five class names below — use them
     **exactly as spelled**, lower case with underscores. The converter
     matches case-insensitively as a safety net, but don't rely on that.
   - `--validate-label exact` refuses a label that isn't one of the five —
     it catches a typo at the moment you make it instead of at conversion
     time.
   - LabelMe does **not** embed a copy of the image inside the JSON by
     default (that only happens if you pass `--with-image-data`, which you
     should not) — the converter reads the image file directly, so leave
     the default alone.
4. Save each image's annotation as `<image-stem>.json` next to the images
   (LabelMe's default — pass `--output DIR` to save elsewhere) — the
   converter in §8 accepts either layout.

#### Recording which product it is (optional, but do it)

`recog.seg_evaluate --per-sku` breaks every segmentation number down by
product, using an `asset` field the synthetic pipeline reads from the CAD
catalog. A photograph has no such field, so you have to say. Add the SKUs
you are shooting to the `labelme` command as **flags**:

```
labelme recog/realtest_rig/images \
    --labels battery,cartridge,electronics_module,placement_area,obstruction \
    --validate-label exact \
    --flags AnkerPowerCore10000,AnkerPowerCore13000,AnkerPowerCore20100,AnkerPowerCore26800
```

`--flags` puts a checkbox row at the top of the window. Tick the one that
names the product in the photo; the converter copies it onto every
annotation in that image. **Tick exactly one** — two set flags is an error,
not a coin toss, because nothing here can arbitrate between them.

If a single photo shows **two different products**, image-level flags cannot
express that. Use per-shape flags instead — `--labelflags` with a JSON map
of label pattern to flag list, e.g. `'{".*": ["AnkerPowerCore10000",
"AnkerPowerCore26800"]}'` — and tick the SKU on any one shape of each unit.
The whole unit inherits it; two shapes of the *same* unit declaring
different SKUs is an error.

Declaring nothing is fine — the converter prints how many annotations came
out without a SKU, and `--per-sku` will simply report them all in one
`None` bucket. It is only per-SKU breakdowns you lose; nothing else in the
pipeline reads this field.

### 6.2 Drawing a shape

- Use the **polygon** tool (not rectangle, not circle) for every class.
  A rectangle *can* still describe a shape whose true geometry is a
  rectangle (most `cartridge` wall segments, most `electronics_module`
  boards) — draw those as 4-point polygons, not the separate rectangle
  tool, since the converter only accepts `shape_type: polygon`.
- Zoom in (scroll wheel) before clicking vertices near a boundary you care
  about — LabelMe's precision is only as good as how far you've zoomed.
- Set the **Group ID** (§5) as you go, or afterward via "Edit Group ID".
- Save after every image. LabelMe's default is one `.json` per image, named
  after the image.

---

## 7. Capturing the photographs

**Target: 50-100 usable images.** Variety matters more than volume — twenty
carefully varied photographs teach more than eighty near-duplicates.

### 7.1 Geometry: match the deployed rig, not a phone in your hand

Per §0, the deployed camera is near top-down at a fixed ~400 mm working
distance. Your capture rig must approximate that:

- **Shoot near-overhead**, camera roughly parallel to the bench (0-10° off
  vertical), no roll (don't tilt the phone sideways). Centre your subject
  cartridge in the frame.
- **Keep the working distance roughly consistent, around 400 mm**, so scale
  is stable across the set. A phone's native photo already records enough
  detail at that distance; you don't need a macro lens.
- **Hold that steady with a fixture, not your hand.** A phone tripod arm, a
  copy stand, or simply resting the phone flat on a shelf edge or a box of
  fixed height all work. This is the single most important piece of capture
  advice in this document: **an inconsistent rig is the main thing that
  would waste the labelling effort** — a wobbly, varying-angle, varying-
  distance set defeats the entire purpose of separating this collection from
  the existing handheld obliques (§7.3).
- A phone is a completely fine capture *device* — it's the geometry that
  has to match the deployed camera, not the sensor.

Do **not** deliberately vary tilt and roll widely across the set. That
variation is out of scope for the target domain (§0) and would spend
annotation budget on cases the robot will never actually see.

### 7.2 What to vary

Spend the variation budget here instead:

- **Fill level** — empty bay, partially filled, nearly full. This is the
  single most important axis: `placement_area`'s whole reason for existing
  is the modal, currently-free floor, and that only exercises the label
  fully across a range of occupancy.
- **Cartridge type** — every case type/colour you have access to, especially
  the black cartridges the CAD carries no colour information for.
- **Lighting** — overhead room light, a directional lamp, mixed sources.
  Real deployment lighting won't be perfectly even.
- **Bench clutter.** Include, spread across the set (not necessarily every
  photo):
  - the blue jig plate itself, with a realistic surface (screws, wear,
    reflections);
  - **loose 18650 cells sitting outside any cartridge** — call this out
    specifically: it's the highest-value clutter item, because it forces
    the model to distinguish an in-bay cell from an out-of-bay one, which
    is exactly the discrimination the placement logic depends on and which
    the synthetic generator does not yet exercise;
  - tools and cables in frame;
  - other cartridges partly visible at the frame edge (§1.3);
  - general incidental background clutter you'd actually see on the bench.

### 7.3 Keep this set separate from `recog/realtest/`

The seven photographs already in `recog/realtest/images/` were shot
handheld, at oblique angles, and stay useful — as a **stress set**, not as
part of this collection. Put your new rig-realistic photographs in a
**different** directory (e.g. `recog/realtest_rig/`), and do not mix the
two.

**Why this matters:** mixing an oblique stress set into a rig-realistic
measurement set would blur the one number this entire exercise exists to
produce — how the segmenter transfers to the camera geometry the robot will
actually use. A photo shot at a jaunty angle and a photo shot at 0-10° off
vertical are measuring two different things; averaging them together hides
both answers instead of giving you either one.

### 7.4 What to avoid

- Handheld shots at an arbitrary angle (§7.1) — that's what `recog/realtest/`
  already is.
- Motion blur — hold still, or use the fixture from §7.1.
- Wildly inconsistent working distance between shots — pick a rig height and
  keep it (§7.1).
- Framing where your primary subject cartridge is mostly cropped out of
  frame (§1.3 — a neighbour clipping the edge is fine; your subject
  shouldn't be).
- A single dominant lighting/fill-level/cartridge-type combination repeated
  across most of the set — see §7.2 for what to vary instead.

---

## 8. From LabelMe to this project's mask format

Once a batch of `.json` files exists, convert them. If you followed §6.1's
default (LabelMe saves each `<image-stem>.json` next to its image, so
`recog/realtest_rig/images/` holds both the photos and the JSON files
together), point the converter at that one folder twice — once as the
source of `.json` files, once as the source of images to check dimensions
against:

```
python -m recog.labelme_to_seg recog/realtest_rig/images \
    recog/realtest_rig/annotations/instances_seg.json \
    --images-dir recog/realtest_rig/images
```

(If you instead used `labelme --output DIR` to keep the JSON files in a
separate folder from the images, pass that folder as the first argument
instead and `--images-dir` still points at the photos.)

Then validate the result before trusting it for anything:

```
python -m recog.check_annotations recog/realtest_rig
```

Read any `[ERROR]` lines the validator prints — see
`recog/check_annotations.py`'s module docstring for exactly what it checks
and why. Fix the annotation in LabelMe, re-run the converter, and
re-validate. `[WARN]` lines (e.g. a class with zero instances so far) are
worth knowing but do not need to be fixed before moving to the next photo —
they matter once the whole 50-100 image set is done, not per image.

Two of those lines are about unit grouping specifically and are worth
knowing before you see them:

- `[ERROR] missing_unit_id` — an annotation with no unit id. Not reachable
  from LabelMe; if you see it, the JSON has been hand-edited. `BaySegDataset`
  refuses the file outright, so this must be fixed.
- `[WARN] single_unit_id` — every annotation in one photo shares one id, so
  the photo yields exactly one crop. Right for a close-up of a single
  cartridge; wrong, and silently so, if the photo shows two units whose
  Group IDs got merged (§5). Check it by eye once and move on.

---

## 9. A worked photo, start to finish: `IMG_4426.jpg`

This photograph (already in `recog/realtest/images/`, part of the *existing*
oblique set, not something to re-annotate into the new collection — used
here only because its clutter is representative) shows a blue jig with
several units. Walking it row by row:

- **Top row** — seven loose cells in round jig holes: seven `battery`
  polygons, no group ids linking them, no `cartridge` anywhere (the jig
  holes are not power-bank cases).
- **Second row, left** — an open, empty cartridge with a white tape cross on
  the floor: `cartridge` for the walls (several tiled pieces, §1.2),
  `obstruction` for the tape, `placement_area` for the remaining floor with
  a hole where the tape sits. One group id ties all of these together.
- **Second row, right** — an open cartridge with a module on one side and a
  small separate label on the floor: `cartridge` (walls),
  `electronics_module` (the board, its USB shells, its inductor),
  `obstruction` (the separate label — §1.4's test says it would stay behind
  if you lifted the board), `placement_area` (everything else, with holes
  under both the module and the label). One group id.
- **Third row, left** — a loose module in its own jig pocket, no cartridge
  shell: `electronics_module` alone, no group, subtracted from nothing
  (§1.4, ruling 3).
- **Third row, right** — an open cartridge with a module and a large
  adhesive smear covering most of the remaining floor: `cartridge`,
  `electronics_module`, `obstruction` (the smear), and whatever sliver of
  `placement_area` is genuinely still free floor — likely very little, and
  that's fine (a bay can legitimately have almost no free floor; that is
  the correct label, not a sign you did something wrong).
- **Fourth row, far left** — a cartridge sliced off by the frame edge: label
  only the visible sliver as `cartridge` (§1.3), stop at the frame boundary.
- **Fourth row, second cartridge** — a foam pad sitting in one pocket:
  `obstruction` (§1.4's test: lift the module out, the foam stays behind).
- **Fourth row, right, upper** — an open, apparently empty cartridge with
  its ribbed floor entirely visible: `cartridge` (walls) and
  `placement_area` (the whole visible floor) — the simple case, no
  subtractions needed.
- **Fourth row, right, lower** — a smooth, featureless black block with no
  interior visible: a sealed `cartridge`, one outline, nothing else.
- **The cables and screws visible throughout the frame** are left
  unlabelled (§3).

---

## 10. Second-opinion check (recommended, not required)

If a second person is available, have them independently label a handful of
your images (5-10 is enough for a sanity check) and diff the results by eye
or with the validator run twice with `--max-overlap-px` set high enough to
compare gross shape rather than boundary pixels. Large disagreement on
*which pixels belong to which class* (not just boundary imprecision) means
this document under-specified something — file it as a note, don't just
average the two annotations together.
