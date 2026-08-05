# Blender synthetic dataset — known issues and next steps

Companion to `specs/2026-08-05-blender-synthetic-dataset-design.md` and
`plans/2026-08-05-blender-synthetic-dataset.md`. Everything here was measured,
not estimated. Nothing here blocks generating a dataset; items 1 and 2 change
how good that dataset is.

## Before you generate 2000 images

### 1. Exposure is still tuned for object pixels, not whole frames — PARKED, your call

The shipped default (`exposure: 0.0`) clipped **40–49% of labelled-object
pixels** above 0.98; the worst scenes were blank white frames. That is fixed:
`exposure: -1.5` plus `harsh_inspection.energy: [80, 150]` and a
`brushed_metal` roughness floor of 0.35 took it to **8.2%**.

What was *not* re-tuned is the rest of the frame. Backdrop albedos span
0.035–0.90 linear — a 25× range — but render into a 0.25-wide band:

| backdrop | rendered corner luminance |
| --- | --- |
| conveyor_belt | 0.60 (authored as near-black rubber) |
| fabric | 0.74 |
| concrete | 0.82 |
| brushed_metal | 0.83 |
| paper | 0.85 |

Measured effect of further exposure reduction, same 12 scenes:

| exposure | backdrop spread | whole-frame clip >0.98 |
| --- | --- | --- |
| −1.5 (shipped) | 0.25 | 2.6% |
| −2.5 | 0.32 | 0.6% |
| −3.5 | 0.36 | 0.0% |

Roughly 1.5–2 more stops of headroom would restore the backdrop axis as a real
source of variation. Use the sweep to decide:

```bash
blender -b --python recog/generate3d.py -- --sweep backdrop --seed 7 --out recog/sweeps
python -m recog.verify3d --sweep recog/sweeps --out recog/sweeps/backdrop_sheet.png
```

### 2. White shells vanish on pale backdrops — PARKED, your call

**35% of boxes** have under 0.05 luminance difference from their surround; p05
is 0.004. These carry correct labels while being invisible, which teaches the
detector to hallucinate and will cost precision on real photos. Lowering
exposure does not help (37% → 34% at −3.5) because the cause is albedo
collision: `shell_white` (0.68–0.86) and `shell_alu` (0.55–0.77) against
`paper` (0.76–0.90).

Fix by constraining the joint material/backdrop draw to require a minimum
albedo separation, or by darkening `paper`. Both live in `configs/synth3d.yaml`.

### 3. Anchors were 8× too small — FIXED, but verify against your training run

`anchor_scales: [4, 8, 16, 32]` gave a largest anchor of 64 px against boxes
measuring 50–57 px (batteries) and 112–185 px (cartridges) at 1280×720. Best
centred-IoU was min 0.09 / median 0.62, with **20% of boxes below 0.5** and a
cartridge median of 0.21.

Now `[56, 80, 112, 160]` with ratios `[0.28, 0.5, 1.0, 2.0, 3.5]`: min 0.72,
zero boxes below 0.5, cartridge median 0.74. The ratios mattered as much as the
scales — cells are 3.55:1 and the old maximum ratio was 2.0, so no anchor could
reach a cell better than ~0.63 regardless of scale.

`recog/checkpoints/best.pt` was trained under the old anchors on the old cv2
dataset. The head shape is unchanged so it still loads, but it is stale for any
evaluation against this config.

## Parked with rulings

| Item | Ruling |
| --- | --- |
| The anchor check's tail warnings cry wolf — it uses a hard floor, so `p05 = 51 px` against a 56 px anchor warns even though that box matches at IoU 0.84, and every `--res 640 360` dev run trips it. | Real but cosmetic; it is a printed warning, not behaviour. Fix by giving the tails a matching-aware band (`p05 < lo·√0.7`) and normalising percentiles for resolution. Do **not** lower `anchor_scales[0]` to 48 — that would drop the median battery IoU from 0.93 to 0.79 to rescue a 5th-percentile box already at 0.84. |
| Jig mode still packs into a single FFDH shelf; a plate can land in a frame corner at ~6% frame coverage. | Acceptable. Part pixel sizes are unaffected (`ortho_scale` derives from `layout.area`, not the plate) and the plate is unlabelled, so no annotation is harmed. Tighten later by packing into a narrower strip so multiple shelves form. |
| `--visibility` is inert: `merge_group_boxes` hard-codes `visible_fraction: None` so `min_visibility` never applies to a cartridge, and both layout solvers guarantee non-overlap so no labelled object can occlude another. | Documented at four sites plus a runtime warning rather than redesigned. It costs one extra index render per instance (up to ~32/sample) for zero signal. Leave off. |
| `Variant.explode` and `cluster_offsets` were deleted as dead code. | The underlying feature — clustering loose cells near their opened case, as in IMG_4426 — was never wired. Spec §5.1 describes it; it is not implemented. Worth doing if you want that scene composition. |
| `lay_flat`'s argmin tie makes it non-idempotent. | Inert for the real geometry (an 18650's tie swaps y/z, extents unchanged). One-line fix if an asymmetric part is ever added: `if ext.z <= min(ext.x, ext.y) + tol: return`. |
| `render_index_map`'s directory-scan fallback prefix-matches, so `_iso1` also matches `_iso10`. | Safe only while `--visibility` is off. Must be fixed before turning it on. |

## Deliberately out of scope

Reading the seven real photos in `d:/dev/rb/recognition/ann_btt_ctrdge/` as a
held-out real-image test set. This needs a COCO reader in `recog/dataset.py`
(~60 lines) and is the single most valuable follow-up: **mAP on real photos from
a model trained only on synthetic renders** is the result that justifies this
whole exercise. It is cleanly separable and blocks nothing here.

## What five silent defects were found in the source pipeline

Recorded because each would have produced a plausible-looking, wrong dataset,
and because they are the strongest evidence that the object-index approach needs
its gates:

1. **`rotation_euler` was a no-op.** The glTF importer sets
   `rotation_mode = 'QUATERNION'`, so `lay_flat` and `place_item` silently did
   nothing. Every cell would have stood on its end and `place_item` would have
   discarded `Placement.rot_deg` entirely — a dataset with zero rotation
   diversity while `layout.py` computed rotations correctly.
2. **The jig plate rendered black.** Its top face sat coplanar with the backdrop,
   so every shadow ray self-occluded. The index pass is byte-identical either
   way, so no mask-based gate could detect it.
3. **`build_pcb`'s parenting flung components 339 mm off the board** (missing
   `matrix_parent_inverse`). They carry `pass_index 0`, so strays would occlude
   labelled parts and shrink boxes with no audit trail.
4. **Jig pockets abutted with zero wall material** on 400/400 seeds, so the
   boolean difference would carve one merged trough rather than separate
   recesses.
5. **`CompositorNodeOutputFile` was rewritten in Blender 5.0.** A non-empty
   output-item name becomes an EXR layer prefix and the mask reads back as a
   **0×0 image** — a valid 68 KB file on disk, an empty array in memory, no
   error raised.

Blender 5.0 also removed `scene.node_tree` while leaving `scene.use_nodes = True`
succeeding silently, so the compositor path fails late rather than at the point
of the mistake. See spec §11.1 for the full API delta table.
