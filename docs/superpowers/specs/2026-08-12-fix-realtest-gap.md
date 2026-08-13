# Fix: the unannotated real photograph in `recog/realtest/`

Branch `feat/blender-synth-dataset`, from HEAD `885a044`.
Files touched: `recog/eval_real.py`, `docs/receipts/real_photo_eval.txt`,
`docs/receipts/real_photo_eval_include_empty.txt` (both new, both generated),
`tests/test_eval_real_exclusion.py` (new), and this file. Nothing else.

**Headline: no published figure moves, and the brief's premise is half true.**
`IMG_4428.jpg` really does carry zero annotations, and scoring it really does
cost 0.046–0.075 mAP@0.50 depending on the checkpoint. But the exclusion the
brief proposes as option (1) **already landed**, at commit `3b2d2cf`, and the
0.8647 quoted in audits G and N is already the excluded figure. What was
missing was not the fix; it was the *evidence*: the number had no committed
generator, the exclusion was loud only in a block of stdout prose, and the
`--include-empty` counterfactual could be quoted with no warning attached. All
three are now closed.

The one thing a documentation agent must act on is not the exclusion at all —
it is a **checkpoint mismatch** (§5).

---

## 1. The facts, established rather than assumed

`recog/realtest/annotations/instances_default.json`, 80 annotations, 7 image
records, 2 categories, **zero `segmentation` entries** (boxes only):

| image | anns |
|---|---|
| `IMG_4426.jpg` | 11 |
| `IMG_4427.jpg` | 23 |
| **`IMG_4428.jpg`** | **0** |
| `IMG_4429.jpg` | 17 |
| `IMG_4433.jpg` | 23 |
| `IMG_4434.jpg` | 5 |
| `IMG_4435.jpg` | 1 |

`IMG_4428.jpg` has an `images` record — width, height, filename, id 3 — and
**not one annotation refers to it**. It is not excluded by a filter, not
missing from the file, not dropped by `parse_coco_json`'s degenerate-box or
unknown-category guards (there is nothing to drop). It is a gap in the CVAT
export. `parse_coco_json` returns it, correctly, as a record with empty
`boxes` and `labels`; its docstring says so explicitly.

**The mechanism, confirmed.** `recog.evaluate.mean_ap` scores a pooled,
score-sorted detection list per class. An image with no ground truth
contributes zero true positives and zero to the recall denominator, so every
prediction on it lands in `cum_fp`. Under the shipped `best.pt` the detector
returns **16 predictions** on `IMG_4428` (7 battery, 9 cartridge) at the
production confidence of 0.70. All 16 become false positives, they interleave
into the pooled ranking by score, and they depress the precision at every
recall level at or above where they land. That is the whole effect: it is a
precision penalty applied to the *set*, and it is invisible in a headline mAP.

---

## 2. Quantified, both ways, on both checkpoints

Same weights, same config, same confidence, same predictions — scored twice,
differing only in whether `IMG_4428` is in the scored set.

**`recog/checkpoints/best.pt`** — the checkpoint README's commands name and
`docs/receipts/detector_bench.txt` arm 3 calls "shipped". Receipted at
`docs/receipts/real_photo_eval.txt` and `..._include_empty.txt`.

| | 6 scored (excluded) | 7 scored (`--include-empty`) | cost of the gap |
|---|---|---|---|
| `AP_battery@0.50` | 0.9655 | 0.9429 | −0.0226 |
| `AP_cartridge@0.50` | 0.7314 | 0.6627 | −0.0687 |
| **`mAP@0.50`** | **0.8484** | **0.8028** | **−0.0457** |
| `AP_battery@0.75` | 0.8812 | 0.8638 | −0.0174 |
| `AP_cartridge@0.75` | 0.7275 | 0.6597 | −0.0678 |
| **`mAP@0.75`** | **0.8044** | **0.7617** | **−0.0426** |

**`recog/checkpoints/last.pt`** — the checkpoint the published 0.8647 came
from. Not receipted (§5 explains why one receipt, not two checkpoints), but
reproduced here to four decimals, which is the check that this measurement is
sound: **0.8647 / 0.7878 exactly matches** `docs/superpowers/specs/2026-08-12-fix-detector.md`.

| | 6 scored | 7 scored | cost of the gap |
|---|---|---|---|
| `AP_battery@0.50` | 0.9675 | 0.9502 | −0.0173 |
| `AP_cartridge@0.50` | 0.7619 | 0.6292 | −0.1327 |
| **`mAP@0.50`** | **0.8647** | **0.7897** | **−0.0750** |
| **`mAP@0.75`** | **0.7878** | **0.7211** | **−0.0667** |

**Not negligible.** 4.6 to 7.5 points of mAP@0.50, or 5.4 % to 8.7 % relative,
from one image out of seven. The cartridge class takes the brunt (−0.069 to
−0.133) because it has 20 ground-truth boxes to battery's 60, so each false
positive is worth three times as much to it. The effect is also **checkpoint-
dependent by a factor of 1.6**, which is worth stating: how much an annotation
gap costs is a property of how eagerly the network fires on the unlabelled
image, not a fixed tax.

---

## 3. The choice: exclude (option 1), which is what the code already does

**Chosen: exclude, made loud.** Three reasons, in order of weight.

1. **An unannotated image cannot serve as ground truth.** This is the ground,
   and it is not "the number is nicer without it". The distinction is the
   whole thing: if the ground were the number, this would be an eval set
   dropping its inconvenient member, which is the failure mode two audits have
   been eliminating. `partition_records`' docstring now states the ground in
   those words, and states that it is a property of the *data* — nothing in the
   exclusion path knows the string `IMG_4428`, and a test pins that (§4).
2. **It was already the decision.** Commit `3b2d2cf`, "fix(recog): exclude
   under-annotated real photos from the eval score", predates this brief. The
   published 0.8647 is already computed over six images; audit N says "over
   **six** scorable photographs" in the same sentence it quotes the number.
   Re-deciding it was not the task; evidencing it was.
3. **Annotating is better and remains open** — see §6. It is human work, it is
   cheap, and I did not do it. **No annotation was generated, by any means.**
   The two receipts are pure evaluation output over the 80 boxes that were
   already committed on 2026-08-05.

**What was actually wrong, and is now fixed:**

| defect | before | after |
|---|---|---|
| The published real-photo AP had **no committed generator** — it lived in audit prose (`audit/…-G-detector.md`, `…-N-objective-closure.md`, `specs/…-fix-detector.md`) and in no receipt. Audit N §3 names this exact pattern as the defect the audit campaign exists to correct, three paragraphs above quoting the figure. | prose | `docs/receipts/real_photo_eval.txt`, written by `--out` |
| The exclusion was loud **only in stdout**. A receipt run redirects stdout; `--quiet` exists; `\| tail -3` exists. Audit E found this class of silence in five places. | report block only | report block **plus a stderr warning** naming the images and the count |
| **`--include-empty` had no headline warning.** Its report differed from the default by one per-image note and a count on line five. The mAP line — the line a reader copies — looked identical and was 4.6 points lower. | silent | a banner above the table: image names, false-positive count, "NOT comparable with the default figure", "Do not quote it as the real-photo result" |
| Neither figure identified the **weights**. `recog/checkpoints/` is gitignored and no `.pt` is tracked anywhere, so `--checkpoint recog/checkpoints/best.pt` pins nothing — and `best.pt` and `last.pt` differ by 0.016 mAP on this corpus (§5). | path only | `sha256:…  (N bytes)`, in the report and both receipts |

`recog/eval_real.py` gained one flag (`--out`), two helpers
(`checkpoint_fingerprint`, `receipt_text`), the banner, and the stderr
warnings. **No metric definition, model, config, threshold or dataset
changed**, and the default `mAP@0.50` over six images is bit-identical before
and after — the receipt is the proof.

---

## 4. The test that stops it recurring

`tests/test_eval_real_exclusion.py`, seven tests. New file, deliberately: the
`tests/` tree is another agent's, and `tests/test_dataset.py` already covers
`partition_records` and the exclusion block. This file covers only what was
added and the shipped corpus.

The load-bearing one is `test_shipped_corpus_agrees_with_its_receipt_about_what_is_scored`.
It reads the real annotations, partitions them, and asserts the receipt says
the same thing — "6 of 7 scored" — and names every excluded image. It fails in
**both** directions:

- annotate `IMG_4428` → 7 of 7, receipt stale, test fails with the regenerating
  command in the assertion message. A corpus change that moves a published
  number cannot land without moving the receipt.
- an image quietly loses its boxes → the count drops, same failure.

So an unannotated image cannot silently rejoin the scored set, and it cannot
silently leave it either. The other six pin the banner and its false-positive
count, the absence of the banner on a clean run, the weights digest, the
regenerating command in `receipt_text`, and —
`test_zero_gt_images_are_selected_by_their_boxes_not_their_name` — that
membership follows the boxes: renaming the labelled image to `IMG_4428.jpg`
gets it scored, and moving the boxes onto `IMG_4428` scores `IMG_4428`. A
filename skip list would pass every other test in this file and is the exact
thing the brief warns against.

---

## 5. For the documentation agents: what moves

**Nothing in `docs/FDR_v3.md` or `README.md` needs a numeric edit because of
this change.** I grepped both, plus `docs/README.md`, `docs/NEXT_STEPS.md` and
`docs/MODEL_CARD.md`: **no real-photo detector AP appears in any shipped
document.** 0.8647 occurs only in `docs/superpowers/audit/2026-08-12-G-detector.md`,
`…-N-objective-closure.md`, `docs/superpowers/specs/2026-08-12-fix-detector.md`
and `…-fdr-claim-corrections.md`. Those are audit records of past runs and
should stay as they are.

**One thing does need acting on, and it is not the exclusion.**

> ### The real-photo 0.8647 and the in-domain 0.9053 are two different networks.
>
> `docs/receipts/detector_bench.txt` arm 3 is **`best.pt`** → in-domain
> `mAP@0.50 = 0.9053`. The real-photo `0.8647` in audits G and N is
> **`last.pt`**. `docs/superpowers/audit/2026-08-12-N-objective-closure.md`
> §3 recommends restating objective O1 as "Pass in-domain (0.9053) with the
> real-photo 0.8647 (n = 6) reported alongside". **Adopting that wording as
> written would put two checkpoints in one row**, and README's own quick-start
> command (`--checkpoint recog/checkpoints/best.pt`) points at the other one.
>
> The shipped `best.pt` real-photo figure is **`mAP@0.50 = 0.8484`,
> `mAP@0.75 = 0.8044`, over 6 of 7 photographs**, receipted at
> `docs/receipts/real_photo_eval.txt` with the weights sha256 recorded.
>
> **If FDR §10.5 adopts audit N's recommendation, quote 0.8484 beside 0.9053 —
> both `best.pt`, both receipted — not 0.8647.** If 0.8647 is preferred for
> continuity with audit G, the row must name `last.pt` and say that the
> in-domain figure beside it is a different checkpoint.

Two smaller notes, both optional:

- Any text describing the real-photo set should say **6 scorable of 7
  photographs**, with `IMG_4428.jpg` unannotated. `n = 6`, and it is 5
  physical cartridges (audit M §2) — the small-*n* framing in FDR §13.2.1/§13.2.2
  is unaffected and needs no softening.
- `docs/receipts/real_photo_eval.txt` is now citable wherever the real-photo
  detector figure is mentioned. It did not exist before.

---

## 6. Owner action, not agent action: annotate `IMG_4428.jpg`

Excluding is honest; annotating is better, and it is the smallest remaining
item in audit M.

- **Estimate: 0.4 h** (audit M §4, which counts resolving the white-cylinder
  question, the clipped cases at the right edge, and the cell strip).
- **What it buys:** *n* goes from 6 to 7 photographs, and the published
  `mAP@0.50` moves by an amount nobody can predict — the 16 predictions
  currently discarded would be matched against real boxes, some as true
  positives. It will not simply be 0.8028; that figure is the floor where all
  16 are wrong.
- **Boxes only, in CVAT, into `instances_default.json`** — matching the
  existing 80. Audit M §1.4 flags two genuine ambiguities in this photograph
  (the white plastic cylinders, and whether the frame is legitimately empty)
  that `docs/ANNOTATION_PROTOCOL.md` §2 does not resolve, so the annotator
  should record a ruling.
- **Then regenerate**, and the test in §4 will tell you if you forget:
  ```
  python -m recog.eval_real --checkpoint recog/checkpoints/best.pt --quiet \
      --out docs/receipts/real_photo_eval.txt
  python -m recog.eval_real --checkpoint recog/checkpoints/best.pt --quiet \
      --include-empty --out docs/receipts/real_photo_eval_include_empty.txt
  ```
  No code change is needed when the annotation arrives: a labelled image
  simply stops being empty, and `partition_records` stops excluding it.

**Explicitly not done, and not to be done:** generating boxes from model
predictions. The detector's 16 predictions on `IMG_4428` are sitting right
there in the receipt and would have taken a minute to convert. That is
circular — it would score the model against its own output and permanently
corrupt the only real-image evaluation set this project has.

A second, smaller gap is recorded in audit M §6(a) and not addressed here:
`IMG_4426.jpg` carries 11 boxes where `ANNOTATION_PROTOCOL.md` §1.5 counts a
7th cell the COCO file does not. That is inside a *scored* image, so it costs
recall rather than precision, and it is the same 0.5 h annotator sitting.

---

## 7. Incidental finding: the over-prediction heuristic is coarser than its comment claimed

`OVER_PREDICTION_FACTOR = 3.0`'s comment said `IMG_4435` "is what this exists
to surface". Under `last.pt` it fires (1 GT, 4 predictions, 4.0×). Under the
shipped `best.pt` it does **not**: 1 GT, 3 predictions, exactly 3.0×, and the
test is strict `>`. The comment was true when written and is not true of the
shipped checkpoint.

**I did not move the threshold** — that would change reported behaviour for a
cosmetic reason, and the heuristic's own comment already calls it "deliberately
loose". I corrected the comment to say what it now does: it catches a gross
labelling gap, not a marginal one, and the gap that actually mattered on this
corpus is caught by `partition_records`, which needs no threshold at all.

---

## 8. Verification

- `python -m pytest -q` — full suite green, including the 7 new tests.
- `python -m pyflakes recog/eval_real.py tests/test_eval_real_exclusion.py` — clean.
- Both receipts regenerated from `recog/eval_real.py --out` after the final
  edit. Neither was written or touched by hand.
- `last.pt` reproduces the published 0.8647 / 0.7878 to four decimals, which is
  the control on the whole measurement.
- `git status --porcelain` checked before staging; only the five paths listed
  at the top were staged, by explicit path.
