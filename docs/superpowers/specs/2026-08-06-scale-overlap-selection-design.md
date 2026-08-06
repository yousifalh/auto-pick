# Scale variety, overlapping parts, and honest model selection

Design spec — 2026-08-06. Follows
`2026-08-05-blender-synthetic-dataset-design.md`.

Every number here was measured on the current pipeline, not estimated.

## Motivation

The detector scores **mAP@0.50 = 0.7296 / @0.75 = 0.6851** on 6 real photos
from synthetic-only training. Per-image AP ranges from 1.000 to 0.380. A
diagnosis of the worst image found a single dominant cause, and it comes with
its own control.

### Finding 1 — the training set is effectively single-scale

| | p05 | p50 | p95 | p95/p05 |
| --- | --- | --- | --- | --- |
| battery `sqrt(area)`, training | 50.9 | 53.3 | 56.2 | **1.10** |
| cartridge `sqrt(area)`, training | 113.1 | 140.9 | 191.1 | 1.69 |

Batteries render at essentially one size because every 18650 is identical
geometry and the orthographic camera frames a fixed `0.80 x 0.45 m` area with
only `margin_range` 1.02–1.10 of jitter.

Real batteries, at the `min_size=500` inference scale, span **43 → 65 px, a
ratio of 1.51**. Per-image AP tracks it monotonically:

| image | AP@0.50 | battery `sqrt(area)` at eval |
| --- | --- | --- |
| IMG_4433 | 1.000 | 65 px |
| IMG_4427 | 1.000 | 54 px |
| IMG_4426 | 0.857 | 46 px |
| IMG_4429 | **0.380** | **43 px** |

**The control is the cartridge class.** It has a 1.69 scale ratio in training —
because there are four different power-bank models — and its AP is uniformly
high on every image, including IMG_4429, where all five cartridges are detected
at 0.96–1.00 while the cells around them are missed. The class with natural
scale variety is robust; the class without it collapses when scale shifts.

This also retro-explains the earlier inference-resolution cliff (mAP@0.75 0.023
at `min_size=800` versus 0.404 at 500). A single global inference scale cannot
fix a training set with no scale spread: tuning `min_size` down for IMG_4429
pushes IMG_4433 out of band the other way.

### Finding 2 — nothing in training is truncated

**0 of 8542** training annotations touch the frame edge, because the camera
always frames the whole layout area plus margin. Real photos have 2 of 80. Minor
in isolation, but free to fix alongside Finding 1 — widening the camera framing
downward naturally crops parts at the edge.

### Finding 3 — the layout solver guarantees non-overlap

Parts never touch or occlude. Real trays hold cells shoulder to shoulder. Two
consequences beyond the obvious domain gap:

- The synthetic validation metric saturates at mAP 1.0 within 0–5 epochs,
  making it useless for model selection. It has now chosen a **worse** checkpoint
  three times (measured: `best.pt` 0.6858 versus `last.pt` 0.7296).
- `--visibility` is inert. `min_visibility: 0.25` can never fire, because no
  labelled object can occlude another.

## Goals and non-goals

**Goals**

- Battery scale spread in training covering the real 1.51 ratio with margin.
- Some training parts truncated by the frame edge.
- Parts allowed to touch and occlude, with labels still correct.
- A validation signal that keeps discriminating after the easy scenes are solved.

**Non-goals**

- New CAD. Blocked on the user's STEP export.
- Closing the perception→planning loop (`plan/placement_area.py`'s green-channel
  heuristic). Separately valuable, separately specced.
- Fine-tuning on real photos. Only 6 annotated images exist and no more are
  obtainable, so they must stay a test set.

## 1. Scale variety

Add a per-scene `zoom` multiplier applied to the camera's `ortho_scale` in
`world.setup_camera`, sampled from `param_space`. Larger `ortho_scale` means a
wider view and smaller parts.

Proposed range **`[0.75, 1.60]`**, giving a battery `sqrt(area)` span of roughly
33–75 px against the real 43–65 px — covering it with headroom at both ends.

The layout area stays fixed at `0.80 x 0.45 m`. At `zoom < 1.0` the frame is
narrower than the layout area, so parts near the boundary are cropped — which
is exactly what Finding 2 wants, at no extra cost. At `zoom > 1.0` there is
extra backdrop around the layout, which is harmless.

### Anchors must move with it — these are coupled

This is the part most likely to be got wrong. Anchors are currently
`[56, 80, 112, 160]`, which `recog/model.py` extends to
`[56, 80, 112, 160, 320]`. At `zoom = 1.6` a battery lands at ~33 px, well below
the 56 px smallest anchor — recreating the very mismatch that cost 20% of boxes
before.

Retune to approximately **`[40, 64, 96, 144]`** → covering 40–288 px, against a
post-change distribution of roughly 33–75 px for batteries and 71–306 px for
cartridges. Recompute against the actual generated data rather than trusting
these figures, using the same best-centred-IoU method as before, and record the
measured table in the config comment.

Also widen albumentations `scale_limit` from `0.10`. At ±10% it spans a 1.22
ratio end to end and cannot cover 1.51 on its own — but it is free, applies every
epoch, and stacks with the render-time change.

## 2. Overlapping parts

Add `max_overlap_iou` to `LayoutCfg`, defaulting to `0.0` to preserve current
behaviour, and a per-scene probability of allowing overlap. In `layout.plan`,
accept a candidate placement when its IoU with every already-placed footprint is
below the threshold, rather than requiring strict disjointness.

Labelling needs no special handling: boxes come from a rendered object-index
pass, so an occluded part simply contributes fewer pixels, and a fully hidden one
never appears in `np.unique(mask)`. This is the property the original design was
built around and it has been verified working.

Two consequences to verify rather than assume:

- `annotate`'s `min_visibility` filter becomes live for the first time. Check its
  `0.25` value against real overlapping renders; it was never exercised.
- `merge_group_boxes` hard-codes `visible_fraction: None`, so `min_visibility`
  still cannot apply to a merged `cartridge`. Decide whether to fix or document.

For jig mode, cells within a pocket row should be allowed to sit touching, since
that is what the reference photos show.

## 3. Model selection

The synthetic metric stops discriminating almost immediately. Rather than
delete it, add a **hard validation subset** and select on that.

Build the subset from the existing per-scene `meta/*.json`, which already records
every drawn parameter. Select scenes in the intersection of the hardest
conditions: lowest-exposure quartile, highest-`zoom` quartile (smallest parts),
and — once §2 lands — scenes containing overlap. Report both metrics each epoch:
the easy one as a sanity check, the hard one for checkpoint selection.

Keep saving `last.pt` unconditionally regardless. That fix is what made the
+39% localisation gain recoverable, and a saturating metric must never again be
able to discard 24 epochs of training.

## Verification

The dataset must be regenerated and the model retrained; the real-photo score is
the acceptance test.

- Battery `sqrt(area)` p95/p05 in the regenerated set is **≥ 1.5**, up from 1.10.
- Truncated annotations are **> 0**, up from 0 of 8542.
- Scenes containing overlapping labelled parts are **> 0**.
- Best-centred-IoU against the retuned anchors has **no boxes below 0.5**, the
  bar met by the current `[56, 80, 112, 160]`.
- Clipping does not regress: labelled-object pixels above 0.98 stay near the
  current 0.074%, and boxes under 3/255 contrast stay near the current 3.6%.
- **IMG_4429's AP improves from 0.380.** It is the reason this work exists.
- Overall mAP@0.50 ≥ 0.7296 and mAP@0.75 ≥ 0.6851 — the current numbers must not
  regress in exchange for the tail.

With 6 images and 80 boxes the headline number is noisy; IMG_4429's per-image AP
is the more direct signal that the scale fix worked.

## Risks

- **Anchor/zoom coupling.** Changing zoom without retuning anchors would make
  things worse. They must be measured together, on generated data.
- **Small parts may fall under the `min_px` / `min_side` filters** at high zoom
  and be silently dropped, shrinking the dataset. Watch `dropped_instances`,
  currently 398 of ~8900 placed.
- **Overlap plus a live `min_visibility`** could start discarding legitimate
  boxes. Compare annotation counts before and after.
- Three coupled changes again resist attribution. Regenerating once with all
  three and comparing to the current numbers is the pragmatic call, but the
  result will be a combined effect, as before.
