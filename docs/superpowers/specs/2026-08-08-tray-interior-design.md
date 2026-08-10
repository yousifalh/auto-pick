# Open-cartridge tray interior

Design spec — 2026-08-08

First of four specs on generator variety. Design only; the implementation plan
is a separate document.

## 1. The defect

`recog/synth3d/config.py`'s `open_case` variant declares
`keep_roles=("cell", "case")`, and `assets.instantiate` merges an assembly's
shell halves into a single object. Both halves are therefore linked into every
"open" scene, and the cartridge renders **closed**.

`world.build_pcb` and `world.build_bay_proxy` are then handed `hi.z` — the top
of that closed group — and draw the electronics module and the
`placement_area` plane there. For `AnkerPowerCore10000` that is **z = 22.2 mm,
the outer surface of the lid**.

The generator has been painting a bay and a PCB onto the outside of a shut box.

`build_pcb`'s own docstring records the simplification without treating it as a
defect: *"the board is laid on top of the shell rather than modelled inside
it: from a bird's-eye camera the two read the same, and this needs no interior
geometry."* Under a fixed orthographic top-down camera the claim is nearly
true, which is why it survived review. It is not true of the hardware.

### The CAD is correctly oriented; only the variant is wrong

Vertex distribution in z for `AnkerPowerCore10000`, counting vertices below and
above each part's own mid-height:

| Part | z span (mm) | verts low / high | Reading |
|---|---|---|---|
| `Case10000_btm_36314b` | −0.0 → 11.1 | **604 / 392** | material at the bottom — floor down, **opening up** |
| `Case10000_top` | 11.1 → 22.2 | 384 / **612** | material at the top — **opening down**, a lid |
| `Cell_18650` ×3 | 1.9 → 20.3 | — | sandwiched between |

So the assembly is modelled correctly: tray below, lid above, cells inside.
Nothing needs flipping. The fault is that the open-case variant never removes
the lid, and the proxy geometry is anchored to the assembly's top rather than
the tray's floor.

## 2. What a real open cartridge looks like

`recog/realtest/images/IMG_4426.jpg` shows opened power banks as **recessed
trays**: a rim of shell wall around a cavity, the PCB seated at one short end
*inside* that cavity, cells lying on the floor below the rim, and visible
self-shadowing where the walls occlude the light.

The current renders show none of that — a flat green rectangle and a flat dark
rectangle, coplanar, on an unbroken surface.

## 3. Required outcome

**`open_case` links only the tray**, and the module and bay proxy sit inside
its actual cavity.

- `open_case` keeps the bottom shell and drops the lid. The `assembled`
  variant is unchanged and must stay so — a sealed cartridge is a real scene
  state and its labels are already correct.
- The bay proxy and electronics module are placed at the **tray's interior
  floor**, not at the assembly's `hi.z`.
- Walls surrounding the cavity render as `cartridge`, by geometry rather than
  by the artificial inset Task 10 introduced.

### Three consequences that fall out rather than needing separate work

1. **Self-shadowing.** A recessed cavity under directional light shadows
   itself. The real photographs always have this; the renders never do.
2. **`cartridge` regains real pixels.** Task 10 inset the proxy by a measured
   wall thickness specifically so a rim would survive. With a genuine tray the
   walls *are* cartridge, and that inset becomes a fallback for the
   procedural-tray work rather than the primary mechanism.
3. **`placement_area` becomes physically meaningful** — the visible floor of a
   cavity, which is what the label has always claimed to be, rather than a
   decal at lid height.

## 4. Measurement changes

`catalog.json`'s `case_interior_mm` is misnamed: it is the AABB of *all*
`case`-role meshes, i.e. the assembly's **outer** extent. Every derived value
inherits that error — `module_bay_mm`, `case_wall_mm`, and through them the
module rect, the placement rect and every label.

With the tray identified, the interior can be measured properly: the cavity of
the `btm` part, its floor height, and its true wall thickness.

**This is the part of the work most likely to surface further defects**, and it
should be treated as measurement, not arithmetic — derive each value from the
mesh and check it against the four assets, exactly as the original bay
measurement was.

Rename `case_interior_mm` to reflect what it holds, or replace it with a
correctly-measured interior. Do not leave a field whose name asserts something
false; it has already caused one bug.

## 5. Blast radius

**This invalidates the current dataset and checkpoint.** Regeneration and a
retrain are part of this work, not a follow-on:

- every label geometry moves
- `recog/dataset3d_seg` (502 scenes / 841 crops) must be regenerated
- `recog/checkpoints/seg/*.pt` must be retrained
- `docs/receipts/seg_eval.txt`, `tau_calibration.txt` and `seg_ablation.txt`
  must be regenerated, and the FDR figures updated with them

Stale numbers have bitten this project three times. The spec states the
sequence so it is not rediscovered.

## 6. What must not regress

- **`assembled` cartridges** — geometry, labels and VOC boxes unchanged.
- **Five-class pixel disjointness** — currently 0 overlapping pixels; the
  design's central invariant.
- **`unit_id` grouping** and unit-scoped VOC boxes from Plan B Task 9.
- **The torch-free demo**, `python main.py --config configs/demo.yaml`.
- **Every geometric decision stays in bpy-free `recog/synth3d/bay.py`.**
  `world.py` and `scene.py` import bpy and cannot be unit-tested. This line has
  held across ten tasks and is what makes the geometry checkable at all.

## 7. Verification

Unit-testable, in `bay.py`:

- interior floor and wall thickness derived per asset match hand-measured
  values for all four assemblies
- the module rect and placement rect tile the *interior*, not the outer extent
- both stay inside the tray cavity under rotation

Requires looking at renders, because `world.py`/`scene.py` cannot be tested:

- an open cartridge reads as a **recessed tray** — visible walls, visible depth
- the module sits **inside** the cavity at one short end, not on a surface
- self-shadowing appears in the cavity under directional lighting
- seated cells lie **below the rim**
- sealed cartridges are visually unchanged

Measured, and compared against current values:

- five-class disjointness stays at 0 overlapping pixels
- `cartridge` pixel count per open unit **rises** (walls are now real geometry)
- validation mean IoU after retrain, against the current 0.8045
- whether the real-photograph comparison moves — currently unresolved, with
  two same-recipe checkpoints scoring 0.211 and 0.232 against the heuristic's
  0.217

That last one is the point of the whole exercise. It should not be expected to
resolve on n = 20, and a shift either way at that sample size is not evidence.

## 8. The other three specs

Recorded here so the ordering is not lost. Each gets its own spec, plan and
implementation cycle.

**2 — Generalisation.** 21700 and 26650 cell formats (the `battery` class
definition already names 21700 and no CAD exists for it), plus a procedural
cartridge-tray family: sampled footprint, wall thickness, bay depth, cell count
and pitch. Needs a tray concept, so it follows this spec.

**Superseded, not merely imprecise: this used to say the four Anker
assemblies "stay in the mix as real-CAD anchors."**
`docs/superpowers/specs/2026-08-10-generalisation-decisions.md` Decision 1
(2026-08-10, settled after this sentence was written) replaces that with a
hard split: the model trains **only** on procedurally generated trays, and
all **four** Anker CAD assemblies are held out entirely as the test set —
never seen during training. If you remember the "anchors" plan, this is
that plan changing, not a second, conflicting description of the same one.
The CAD assemblies are not dropped from the project — they become the
whole test set, plus the basis of a separate CAD-trained control model
used as a ceiling reference — see
`docs/superpowers/specs/2026-08-10-generalisation-design.md` §10 for the
full design.

**3 — Realism.** Perspective camera with tilt and handheld variation, phone-like
aspect ratio (the real photos are 3024×4032 portrait against 1280×720
landscape renders), and bench clutter — cables, tools, the blue jig plate given
a proper material. **Blocked on this spec**: a perspective or oblique camera
would immediately expose the missing interior geometry, which is the reason the
orthographic camera was chosen.

**4 — Difficulty.** Occlusion and clutter, lighting extremes, truncation and
framing, and cluttered bays. Last, so it stresses the generalised and realistic
generator rather than the current one. Aimed at the measured blocker on τ: the
largest optimistic error across the whole validation split is 79.4 % of one
cell's area, so every record fails the admission test on area alone and the
morphological criterion never fires. Harder scenes are the only way to
calibrate τ on synthetic data — more of the same renders will not do it.
