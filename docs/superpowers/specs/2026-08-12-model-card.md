# Model card, data card, and the consolidated results table

**Date** 2026-08-12 · **Baseline** `885a044` · **Scope** documentation and one
new script. No code behaviour, metric definition, dataset, checkpoint or
receipt was changed. Nothing was retrained or regenerated.

Closes gaps **1 (model card)**, **2 (consolidated results)**, **3 (data
provenance)** and **5 (failure modes)** of
`docs/superpowers/audit/2026-08-12-O-ml-maturity.md`, whose finding was that
all four artefacts were "absent as artefacts and present as material".

## What was written

| file | what it is |
|---|---|
| `docs/MODEL_CARD.md` | New. Nine sections; six tables, all generated. |
| `docs/datasets/README.md` | New. The data card. |
| `docs/datasets/*.manifest.json` | New. Eleven byte-for-byte copies of the gitignored generator manifests, 392 KB total. |
| `docs/datasets/checksums.json` | New. SHA-256 + byte length of each manifest and each annotation file. |
| `scripts/model_card_tables.py` | New. Generates every table in both pages; `--check` fails on drift; `--sync-datasets` refreshes the manifests. |
| `README.md` | One item added to the existing "Start here" block. |
| `docs/README.md` | Two rows added to the "Written for a reader" table. |

## Generated, not typed

The audit's own recommendation was to generate the table "so it is a receipt
and not a paragraph". Every table in both pages sits between
`<!-- BEGIN GENERATED: name -->` markers and is emitted by
`scripts/model_card_tables.py` from:

* the eleven `docs/receipts/seg_eval*.txt` receipts (parsed: checkpoint,
  config, crop count, per-class IoU and instance counts, selected mean IoU and
  its `select_on` set, the checkpoint-selection note, the per-SKU table)
* `docs/receipts/frcnn_map.txt` and `frcnn_map_default.txt`
* the ten `configs/segmentation*.yaml` training configs
* `docs/datasets/*.manifest.json`

`python scripts/model_card_tables.py --check` re-derives all six tables and
exits 1 if the committed text differs. It reads only committed artefacts and
runs on a bare clone without torch, so drift is detectable by anyone.

Two mechanical guards are worth naming, because both would otherwise be
silent:

* **The config→checkpoint map is asserted injective.** `segmentation.yaml` and
  `segmentation_cad_test.yaml` both name `checkpoint_dir:
  recog/checkpoints/seg`. Eval-only configs are excluded by
  `train_val_split == 0.0` rather than by filename, and the script raises if
  two *training* configs claim one directory.
* **Shared hyperparameters are checked, not assumed.** The premise of the
  whole comparison is that the ten configs differ only in paths. The generator
  emits any field that differs across models as **"differs between models"**
  instead of collapsing it to one value. All eleven checked fields are in fact
  shared.

## The consolidated table

Nine rows (one per segmenter checkpoint) × `bay` / `electronics` /
`obstruction` / selected mean IoU on the common 836-crop CAD test set, with
the training dataset and scene count beside each. Two supporting tables: a
per-SKU `bay` matrix with the leave-one-SKU-out diagonal marked and
summarised, and a checkpoint-selection table carrying `best.pt`, `last.pt`,
their Δ and the own-val instance counts.

**`anchored_18650`'s 0.6677 now appears in a table.** It previously existed
only in `docs/receipts/seg_eval_anchored_18650_on_cad_test.txt` and as a
relative phrase in the null-result prose.

## Findings surfaced by the consolidation

Assembling the figures in one place produced five statements no single
document held. All are derived from committed artefacts unless marked.

1. **The detector that ships has no published held-out mAP.** Both committed
   mAP receipts (0.7643 k-means, 0.8736 torchvision-default) are the
   **2026-04-20** anchor ablation recorded in `docs/receipts/train_eval.txt`:
   100 `synth_dataset.py` cv2 frames, 15-frame val, ResNet-34 + FPN from
   scratch, 15 epochs. `configs/recognition.yaml` ships a **third** anchor set
   re-tuned against `recog/dataset3d` with COCO pretraining for 35 epochs
   (FDR §5.7 / ADR-005, both corrected 2026-08-12), and no receipt scores it.
   Its in-training metric is uninformative by the config's own admission —
   the full-set synthetic metric "saturates at mAP 1.0 within 0–5 epochs".
   The card states this rather than presenting the ablation as the shipping
   model's accuracy.
2. **The shipping segmenter is the one checkpoint with no held-out row.**
   `recog/checkpoints/seg` has only an own-val figure (0.8032 on 126 crops).
   No receipt scores it on `cad_test` and no document records why not. The
   card also notes such a score would differ in kind: `seg` trained on the
   same four CAD assemblies `cad_test` is built from, so it would be held out
   on images but not on assets.
3. **The training-time selection metric reads systematically high.** Where a
   `seg_evaluate` re-score of the same checkpoint on the same split exists
   (three of nine models), it lands **0.0094 / 0.0162 / 0.0219 below** the
   figure training recorded. Same direction all three times. The card quotes
   the re-score and names fp16-evaluation-against-fp32-training as the
   untested candidate explanation.
4. **`anchored` and `anchored_crown`'s annotation files are the same size to
   the byte** (3,408,089) and differ only by hash — a concrete demonstration
   of why `coco_path` is a path and not an identity.
5. **There are eleven manifests, not nine.** The audit counted nine; the
   detector's `dataset3d` and the held-out `dataset3d_seg_cad_test` are the
   two it did not include. All eleven are copied.

## Dataset provenance

All eleven `manifest.json` files are copied byte-for-byte into
`docs/datasets/`, so the copies can be checked against the originals.
`checksums.json` records the SHA-256 and byte length of each manifest **and**
of each dataset's `instances_seg.json`.

**What the annotation hash buys, stated in the data card rather than implied:**
it converts `coco_path` from a path into an identity, so a dataset can be
established as the one a figure was measured against. **What it does not buy:**
the hashed files are not in the repository, so a cloner cannot verify them —
it is a forward guarantee, not a reproducibility claim. It also does not cover
the **images**; two renders with identical annotations and different pixels
would be indistinguishable from anything recorded.

## Figures that could not be traced to a committed artefact

Named in `docs/MODEL_CARD.md` §8 rather than presented alongside the rest.

1. **21.8 % → 2.6 %** sealed-crop rates (136/623 → 16/623). From a scratch
   diagnostic that was never committed and emitted no receipt — README "Where
   to look" already says so. Its anchor, pooled `bay` 0.6555, *is* a receipt
   figure, and the 0.6555 → 0.8755 movement is fully receipt-backed; the
   per-crop rates are not.
2. **The 2-of-25 placement overlaps at 8.3 % / 5.2 %**, and the oracle
   comparison. These trace to a results table in
   `specs/2026-08-11-placement-safety.md` — a committed document, but not a
   receipt with a committed generator.
3. **"No checkpoint carries a `seed`"** (§7). Measured by loading all nine
   `best.pt` files in the working tree with `weights_only=True` and listing
   their non-weight keys; the command is recorded in §8. Every checkpoint
   carries `epoch`, `ious`, `selected_mean_iou`, `select_on`,
   `val_instance_counts` and `split_seed`; all but `recog/checkpoints/seg`
   also carry `coco_path`; **none carries `seed` or `seeding`**, although
   `recog/seg_training.py` L579–586 writes both. This is the mechanical
   confirmation of the "published checkpoints predate seeding" claim, and it
   is labelled a working-tree observation because `recog/checkpoints/` is
   gitignored.
4. **The dataset SHA-256 checksums**, for the same reason — computed from
   gitignored files.

One figure was **corrected against its receipt rather than copied from the
README**: failure mode #1 quotes `bay` boundary displacement **1.226 mm** and
optimistic placeable area **+79.2 mm²/crop** from `docs/receipts/seg_eval.txt`
at HEAD. The README's headline section quotes 0.949 mm and 51.5 mm²/crop,
which are the same rows converted at the generator's superseded nominal
0.625 mm/px constant — the README says so itself ("≈1.3 mm at this corpus's
true scale"), and the receipt's own `mm_per_px` note explains why that
constant describes no frame in the corpus.

## Failure modes

Ten rows: what fails, how it presents, how you would detect it, what it costs,
and where the evidence lives. Includes the two the audit expected a reader to
miss — checkpoint selection being noise-limited (≤ 0.0036 across nine models)
and the detector's `inference_min_size` cliff (mAP@0.75 0.404 → 0.023) — plus
the operating-envelope statement: a bay packed to exact tolerance cannot be
certified by any vision system with non-zero measurement error.

## Where it is linked

* `README.md` "Start here" — inserted as **item 2**, directly after
  `PORTFOLIO.md`, extending the block another agent added rather than
  duplicating it. The remaining items were renumbered 3–5. The block already
  linked `docs/README.md` at its last item.
* `docs/README.md` "Written for a reader" — two rows added above
  `PORTFOLIO.md`, for the model card and the data card.

`docs/superpowers/specs/README.md` indexes the specs and was **not** touched;
adding this document's entry is unclaimed.

## Verification

* `python scripts/model_card_tables.py --check` → "no drift: every generated
  table matches its receipt", exit 0.
* All 100 internal markdown links and all 9 intra-page anchors in the four
  edited/created pages resolve.
* `python -m pytest -q` → exit 0 at the time of writing. Two transient
  failures in `tests/test_main_integration.py` were observed during an earlier
  run and did not reproduce: they passed in isolation, in pairs, and on
  re-run, and the working tree carried two other agents' in-flight edits to
  `recog/eval_real.py`, `tests/test_planner.py` and a new
  `tests/test_eval_real_exclusion.py` at the time. Nothing in this change is
  imported by any test — the new script is not collected and no test reads
  `docs/` other than in a comment.

## What was deliberately not done

* **No ONNX / TorchScript export**, per the audit's §4 and "what I would
  explicitly not add". The card states the absence and the reasoning instead.
* **No retraining**, including the missing `seg`-on-`cad_test` score. The card
  names the gap rather than filling it.
* **No edit to `docs/FDR_v3.md`, `docs/NEXT_STEPS.md`, `tests/`,
  `recog/realtest/` or `recog/eval_real.py`** — owned by other agents in this
  tree.
* **No new claim.** Every statement in the card is a consolidation of an
  existing measurement, a figure read from a receipt, or an explicitly
  labelled working-tree observation.
