# Spec #2 (generalisation) — design decisions taken 2026-08-10

Recorded before the spec is written so the reasoning is not re-derived. The
groundwork investigation runs in parallel and may contradict some of this; where
it does, the measurement wins and this document should be corrected, not
defended.

## Why this spec carries more weight than it originally did

Real photographs are unobtainable, so sim-to-real transfer cannot be measured
(see `docs/NEXT_STEPS.md`, "The constraint this plan works around"). Training on
one synthetic distribution and testing on a **disjoint** one is a legitimate,
answerable question with no photograph in it, and it is now the project's
primary robustness evidence.

It is **not** a sim-to-real measurement and must never be reported as one.

## Decision 1 — split: train procedural, test on real CAD

Train only on procedurally generated trays. Hold out **all four** Anker CAD
assemblies as the test set; the model never sees real measured geometry during
training.

This is the strongest claim the available data supports, and the closest
analogue to sim-to-real this project can construct: parametric synthetic
geometry on one side, real measured CAD on the other.

### The risk this carries, which the spec must address

A poor test score would be **ambiguous**: it could mean the model fails to
generalise, or it could mean the procedural trays are unrealistic enough that
the CAD assemblies fall outside their distribution. Those are different
failures with different fixes.

The spec needs a way to tell them apart. The obvious instrument is a control:
also train a model on the CAD assemblies directly and compare per-class scores.
If the CAD-trained model does well where the procedural-trained one fails, the
procedural distribution is the problem, not the model's capacity to generalise.
Decide this in the spec rather than discovering it after a bad number.

## Decision 2 — two procedural sets: anchored and wide, kept separable

- **Anchored**: sampled within and slightly beyond the range the four Anker
  assemblies span. Every tray stays plausible as real hardware.
- **Wide**: sampled well outside that range — unusual aspect ratios, thicker and
  thinner walls, denser packing. Stronger domain randomisation; some trays will
  not resemble any real product.

Kept as **separate, separately-scored sets**, not blended.

This composes well with decision 1 and turns it into a 2x2 that answers a
question neither set could alone:

| trained on | tested on | tells you |
|---|---|---|
| anchored | CAD | does plausible synthetic geometry transfer? |
| wide | CAD | does extra variation help or hurt? |

If wide beats anchored on the CAD test, domain randomisation is earning its
keep. If it loses, the variation is noise and the spec should say so.

## Decision 3 — three cell formats: 18650, 21700, 26650

18650 (18.3 x 65 mm, the existing CAD) stays as the anchor. 21700 (21 x 70 mm)
and 26650 (26 x 65 mm) are added.

*Interpretation note:* the owner's answer selected 21700, 26650 and "keep 18650
only" together. Read as "all three formats" — 18650 is retained rather than
replaced. Flagged for correction if wrong.

Neither new format has CAD, so both need a parametric representation. The
groundwork investigation is enumerating every hardcoded 18650 dimension in the
repo; that list bounds the work. Known sites to check at minimum: the packer,
`admits_a_cell`'s 18.3 x 65 mm structuring element, `seg_dataset`, and the class
definitions.

## Open, for the spec to settle

- **Whether the procedural trays are built as Blender primitives at scene-build
  time or as glTF assets generated offline.** The groundwork investigation
  recommends one; the deciding factor should be reliability, not elegance. The
  existing glTF import path has silently surprised this project three times — an
  inverted up-axis, a fused two-material object, and an inner liner sharing a
  role — and each cost a render cycle to find.
- **Whether the electronics module position is sampled or always on a short
  side.** Real cartridges vary; the current CAD does not.
- **How the two procedural sets are sized relative to the 502-scene baseline**,
  and whether the CAD test set is large enough to separate the four SKUs.
- **What "does not regress" means here**: five-class disjointness at 0
  overlapping pixels, the `assembled` variant, `unit_id` grouping and the
  torch-free demo all still apply.
