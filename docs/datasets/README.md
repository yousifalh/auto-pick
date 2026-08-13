# Data card — the eleven generated datasets

The datasets themselves are **not in this repository**. They are ~8 GPU-hours
of Blender/Cycles renders and `.gitignore` excludes `recog/dataset3d*/`
wholesale. What is committed here is the thing that makes them describable: a
byte-for-byte copy of each dataset's **`manifest.json`**, which
[`recog/generate3d.py`](../../recog/generate3d.py) writes beside every dataset
and which carries its **entire generator config, its seed, its class list, its
class→id map and its measured per-class statistics**.

That is not a summary of the generator config. It *is* the generator config,
as resolved at render time — every lighting rig, backdrop, material, tray
parameter range, camera setting and filter threshold that produced the images.
It is also what [`recog/calibration.py`](../../recog/calibration.py) *prefers*
over the authored default when resolving scale, raising rather than falling
back silently if the two disagree.

Copied and hashed by:

```bash
python scripts/model_card_tables.py --sync-datasets
```

## The datasets

<!-- BEGIN GENERATED: dataset-index -->
| manifest | scenes | boxes | generator seed | `instances_seg.json` SHA-256 (first 16) | bytes |
|---|---:|---:|---:|---|---:|
| [`dataset3d`](dataset3d.manifest.json) | 1000 | 8092 | 23 | `—` | — |
| [`dataset3d_seg`](dataset3d_seg.manifest.json) | 502 | 4278 | 0 | `8d18f8f46cb81f8b` | 4092743 |
| [`dataset3d_seg_anchored`](dataset3d_seg_anchored.manifest.json) | 502 | 1744 | 0 | `fc24619b753aca5b` | 3408089 |
| [`dataset3d_seg_anchored_18650`](dataset3d_seg_anchored_18650.manifest.json) | 502 | 1771 | 0 | `d5e0b31ef2320d43` | 3219835 |
| [`dataset3d_seg_anchored_crown`](dataset3d_seg_anchored_crown.manifest.json) | 502 | 1744 | 0 | `561c0781dba3ca83` | 3408089 |
| [`dataset3d_seg_cad_control_holdout_AnkerPowerCore10000`](dataset3d_seg_cad_control_holdout_AnkerPowerCore10000.manifest.json) | 502 | 4906 | 0 | `ac81fe7422f598a2` | 4993005 |
| [`dataset3d_seg_cad_control_holdout_AnkerPowerCore13000`](dataset3d_seg_cad_control_holdout_AnkerPowerCore13000.manifest.json) | 502 | 4663 | 0 | `e74d41e7599e58bd` | 4715250 |
| [`dataset3d_seg_cad_control_holdout_AnkerPowerCore20100`](dataset3d_seg_cad_control_holdout_AnkerPowerCore20100.manifest.json) | 502 | 4197 | 0 | `9746aa20fd0e385f` | 4320468 |
| [`dataset3d_seg_cad_control_holdout_AnkerPowerCore26800`](dataset3d_seg_cad_control_holdout_AnkerPowerCore26800.manifest.json) | 502 | 3738 | 0 | `532bbfa2af067bc6` | 3895303 |
| [`dataset3d_seg_cad_test`](dataset3d_seg_cad_test.manifest.json) | 500 | 4262 | 0 | `acfd09a1771e556d` | 4236510 |
| [`dataset3d_seg_wide`](dataset3d_seg_wide.manifest.json) | 502 | 2116 | 0 | `0cefc20f60a1beeb` | 4454941 |
<!-- END GENERATED: dataset-index -->

`dataset3d` is the **detector's** corpus and has no `instances_seg.json`; its
annotations are per-scene VOC XML under `annotations/`. The other ten are
segmenter corpora.

What each is for:

| dataset | role |
|---|---|
| `dataset3d` | Detector training corpus — 1,000 scenes, the only one at generator seed 23. |
| `dataset3d_seg` | The **shipping** segmenter's corpus, rendered from the four measured Anker CAD assemblies. `configs/demo_seg.yaml` reads its frames. |
| `dataset3d_seg_cad_test` | The **held-out test set**. Rendered separately, `train_val_split: 0.0`, 836 crops over 434 frames. Every comparable figure in the model card is scored here. |
| `dataset3d_seg_anchored` | Procedural trays sampled from a band that brackets the four CAD assemblies. The generalisation question. |
| `dataset3d_seg_wide` | Procedural trays well outside that band — the extrapolation arm. Came out null. |
| `dataset3d_seg_anchored_18650` | `anchored` with cell formats restricted to 18650. One line of generator config apart. Came out null. |
| `dataset3d_seg_anchored_crown` | `anchored` plus a lid crown. **Exactly one parsed field** differs from `anchored`'s config, asserted field-by-field before rendering rather than eyeballed. |
| `dataset3d_seg_cad_control_holdout_*` | Four leave-one-SKU-out controls, each rendered without one of the four Anker assemblies. |

## Per-class instance counts

Kept instances by segmentation class, from each manifest's own `stats` block.
`placement_area` is the class the model calls `bay`; `dropped` counts
instances the generator's visibility filter removed (too small, too occluded,
too elongated) before they reached the annotation file.

<!-- BEGIN GENERATED: dataset-classes -->
| manifest | `cartridge` | `battery` | `placement_area` (bay) | `electronics_module` | `obstruction` | dropped |
|---|---:|---:|---:|---:|---:|---:|
| `dataset3d` | — | — | — | — | — | — |
| `dataset3d_seg` | 840 | 3438 | 211 | 211 | 479 | 1068 |
| `dataset3d_seg_anchored` | 848 | 896 | 234 | 184 | 640 | 1189 |
| `dataset3d_seg_anchored_18650` | 848 | 923 | 234 | 200 | 615 | 1173 |
| `dataset3d_seg_anchored_crown` | 848 | 896 | 234 | 184 | 640 | 1190 |
| `dataset3d_seg_cad_control_holdout_AnkerPowerCore10000` | 852 | 4054 | 237 | 234 | 580 | 1218 |
| `dataset3d_seg_cad_control_holdout_AnkerPowerCore13000` | 852 | 3811 | 237 | 234 | 541 | 1205 |
| `dataset3d_seg_cad_control_holdout_AnkerPowerCore20100` | 852 | 3345 | 236 | 233 | 556 | 1200 |
| `dataset3d_seg_cad_control_holdout_AnkerPowerCore26800` | 852 | 2886 | 236 | 235 | 529 | 1184 |
| `dataset3d_seg_cad_test` | 835 | 3427 | 209 | 209 | 476 | 1060 |
| `dataset3d_seg_wide` | 814 | 1300 | 233 | 119 | 632 | 1201 |
<!-- END GENERATED: dataset-classes -->

The class balance is worth reading before any IoU is. `placement_area` runs
209–237 instances against 814–852 `cartridge` and 896–4,054 `battery`: the
class the whole pipeline exists to find is the **rarest** one in every corpus.
The four CAD-control sets carry 3–4× the batteries of the procedural sets
because the Anker assemblies are rendered with their cells in view.

## Disjointness

FDR §13.1.1 records an MD5 check over **4,536 renders across nine datasets**
finding **zero shared images in 36 pairings**. The held-out figures are
therefore not measured on training frames. Within a single dataset the
train/val split is a different matter and is *not* clean — 73.2 % of
`anchored`'s validation crops come from a frame that also contributed to
training, which is why the own-val column in the model card is labelled
optimistic.

## Checksums, and what they buy

[`checksums.json`](checksums.json) records, per dataset, the SHA-256 and byte
length of both the manifest and its annotation file.

**What that buys.** A checkpoint records `coco_path` — a *path*, not an
identity. Two different renders can occupy the same path, and nothing in the
checkpoint pins the content it trained on. The hash converts the path into an
identity: given a dataset, you can establish whether it is the one a figure
was measured against. It is not idle bookkeeping —
`dataset3d_seg_anchored/instances_seg.json` and
`dataset3d_seg_anchored_crown/instances_seg.json` are **the same size to the
byte** (3,408,089) and are distinguishable only by hash.

**What it does not buy.** These hashes were computed from the author's working
tree, and the files they cover are not in this repository. A cloner cannot
verify them against anything, and they do not make the datasets recoverable.
They are a forward guarantee — if the datasets are ever shared or
regenerated, a mismatch is detectable — not a reproducibility claim. The
manifests beside them are different: those *are* committed, so the copy here
can be checked against the original byte-for-byte.

**They do not cover the images.** Only the annotation file and the manifest
are hashed. Two renders with identical annotations and different pixels would
not be distinguished by anything recorded here.

## Known issues

[`../superpowers/blender-dataset-known-issues.md`](../superpowers/blender-dataset-known-issues.md)
records the measured defects in the generator. The one worth knowing before
reading any figure: Blender's glTF importer maps `(x,y,z) → (x,−z,y)`, and
because this CAD's up-axis is Y with the cavity opening toward −Y, every
`open_case` cartridge once rendered **upside down and closed** — the
electronics module and the bay plane painted on the outside of a lid. It was
caught by measuring the rendered shell against the source CAD (z ∈ [11.1,
22.2] mm against the CAD's [0, 11.1] mm, a mirror about the lid's own
mid-plane), not by any test. The datasets above are all post-fix
(`a31ac28`..`043e92d`).
