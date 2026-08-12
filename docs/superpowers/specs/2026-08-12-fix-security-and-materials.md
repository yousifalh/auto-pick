# Fix: checkpoint loading, dependency floors, and the synth3d wear mix

**Date** 2026-08-12 · **Base** `fa7a4f0` · **Sources**
`docs/superpowers/audit/2026-08-12-B-security.md` (findings 1 and 5) and
`docs/superpowers/audit/2026-08-12-E-silent-failures.md` (A2, A3, A6, and the
`render.py:204` census entry).

Two independent changes, committed separately:

1. **Security** — `torch.load(weights_only=True)` on the detector path, plus two
   dependency changes in `pyproject.toml`.
2. **Silent corruption** — the swallowed `except` in
   `recog/synth3d/materials.py` that discarded the drawn roughness *and* wear on
   100 % of surfaces, and the two clauses in `recog/synth3d/render.py` that
   could silently render a whole dataset differently from what its manifest
   records.

Nothing was retrained, re-rendered or regenerated. No dataset, checkpoint,
receipt or metric definition was touched.

---

## 1. `recog/inference.py:84` — `weights_only=True`

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

`torch.load`'s default runs the pickle machinery, so a checkpoint file executes
arbitrary code at load time via `__reduce__`. `FasterRCNNDetector` is the
detector on the main end-to-end path (`main.py` → detector → planner → robot),
and **no `.pt` is committed anywhere in the repository** — so every user
following the README's `--checkpoint recog/checkpoints/best.pt` commands has
obtained that file from somewhere else. Following the README was the vulnerable
act; this is the ordinary path, not an edge case.

The surrounding code needed no change: `state["model"] if "model" in state else
state` works unaltered on the plain tensor dict `weights_only=True` returns.

An in-line comment now states the rule and why it is not optional here,
mirroring `recog/bay_segmenter.py:81-84`, which already carried it.

### The call-site table after the change

| File:line | `weights_only` | Verdict |
|---|---|---|
| `recog/inference.py` | **`True`** (this change) | Fixed |
| `recog/bay_segmenter.py:85` | `True` | Already correct |
| `recog/seg_evaluate.py:619` | `True` | Already correct |
| `recog/seg_evaluate.py:670` | `True` | Already correct |
| `recog/seg_training.py:458` | `False`, explicit | **Left alone, deliberately.** `train_state.pt` carries optimiser and scheduler state that is not tensors-only; it is written and read by the same script on the same machine, reached only behind `--resume`, which additionally requires the file to exist locally. Not a distribution vector. The audit judged this acceptable and the in-line justification stands. |

That is every `torch.load` in the tree.

## 2. `pyproject.toml` — two dependency changes

```diff
-    "opencv-python>=4.8",
-    "pillow>=10.0",
+    "opencv-python-headless>=4.8",
+    "pillow>=10.3",
```

These are floors, not pins, and there is no lockfile — a fresh install resolves
far above them (this environment: pillow 12.2.0, opencv 5.0.0.93). The exposure
is a reproducible or constrained resolve landing *at* the floor.

**`pillow>=10.3`.** 10.0 admits **CVE-2023-4863** (libwebp heap buffer overflow,
critical, fixed in 10.0.1) and **CVE-2023-50447** (arbitrary code execution via
`PIL.ImageMath.eval`, fixed in 10.2.0). Pillow decodes attacker-supplied images
throughout `recog/`, so the floor has to sit above both.

**`opencv-python-headless`.** The audit's claim that there are zero GUI calls
was re-verified independently here rather than taken on trust: a grep over every
tracked `.py` for the whole highgui surface — `imshow`, `waitKey`, `waitKeyEx`,
`pollKey`, `namedWindow`, `destroyWindow`, `destroyAllWindows`, `moveWindow`,
`resizeWindow`, `setWindowTitle`, `setWindowProperty`, `getWindowProperty`,
`createTrackbar`, `getTrackbarPos`, `setTrackbarPos`, `setMouseCallback`,
`startWindowThread`, `selectROI`, `selectROIs`, `displayOverlay`,
`displayStatusBar`, `createButton` — returns **zero hits**. The one videoio call,
`main.py:110`'s `cv2.VideoCapture`, is present in the headless build; headless
drops highgui only.

So the GUI build was dragging a Qt/GTK/X11 stack (`libGL`, `libglib`) that is
never entered: install weight, CI weight, and the `ImportError: libGL.so.1`
class of container failure, for nothing.

### Verification

* `pip install --dry-run --no-build-isolation -e .` resolves the new
  requirement set cleanly (`Would install auto-pick-0.2.0`); both
  `opencv-python-headless>=4.8` and `pillow>=10.3` are satisfied.
* Clean import of `cv2`, `PIL`, `numpy`, `yaml`, `albumentations` and of every
  module that touches them (`main`, `plan.arbitration`, `plan.placement_area`,
  `recog.inference`, `recog.dataset`, `recog.seg_dataset`,
  `recog.labelme_to_seg`) succeeds.
* Full suite green before and after (see "Suite" below).

### Two follow-ups deliberately **not** made here, because the files belong to
### other work in flight

* **`.github/workflows/ci.yml:38-43`** installs `libgl1` with the comment
  *"opencv-python (not -headless) is a hard dependency, and its manylinux wheel
  links libGL"*. That step and its comment are now obsolete — dropping them is
  the visible half of this change's benefit. Left for whoever owns
  `docs/superpowers/specs/2026-08-12-ci-and-tone.md`.
* Six `raise ImportError("opencv-python is required")` messages
  (`main.py:49`, `plan/arbitration.py:33`, `plan/placement_area.py:50`,
  `recog/inference.py:30`, `recog/labelme_to_seg.py:57`,
  `recog/synth_dataset.py:33`). Five of the six sit in files owned by other
  agents; changing one of six would be worse than changing none. The message is
  not wrong — headless provides the same `cv2` module — but it should eventually
  name the headless distribution.

---

## 3. `recog/synth3d/materials.py` — the wear mix

### What was wrong

```python
mix.inputs["Factor"].default_value = drawn["wear"]
try:
    mix.inputs[2].default_value = drawn["roughness"]
    nt.links.new(ramp.outputs["Color"], mix.inputs[3])
    nt.links.new(mix.outputs[0], bsdf.inputs["Roughness"])
except Exception:
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
```

`ShaderNodeMix` — the node that replaced `ShaderNodeMixRGB` in Blender 4.0 —
carries one typed A/B/Result triple per `data_type` behind a single interface.
Every triple is *named* `"A"`/`"B"`/`"Result"`, and there are two sockets named
`"Factor"`. Both name lookup and positional index therefore resolve to whichever
typed pair the running build happens to put first.

The fallback links the raw noise ramp — a 0→1 swing with ramp positions
0.35/0.75 — **directly** into Principled's Roughness. Both `drawn["roughness"]`
and `drawn["wear"]` then stop reaching the shader entirely: every surface renders
with extreme, all-or-nothing specularity instead of a wear-weighted mix. Every
preset in `configs/synth3d.yaml` has `wear >= 0.05 > 0.01`, so the branch is
entered for every material on every object.

`meta["materials"]` records the two values that never reached the shader, so the
receipt describes a render that did not happen. `MIN_LUMA_DELTA`
(`materials.py:22-39`) was measured from renders made on the correct path, so the
contrast gate would have been scoring the wrong appearance too.

### The fix

A new `socket_by_identifier(sockets, identifier)` addresses sockets by their
stable `identifier` — `"Factor_Float"`, `"A_Float"`, `"B_Float"`,
`"Result_Float"` — and raises `KeyError` naming what the running build *does*
offer rather than returning `None`. The four assignments now use it, with **no
`try`**: a generator that renders something other than what it recorded is worse
than one that stops.

### How it is proved

`materials.py` imports bpy and pytest cannot reach it, so the primary evidence is
`_assert_wear_mix_took(mix, ramp, bsdf, drawn)`, which runs on **every material
built, not in a test** — the same treatment
`world._assert_procedural_tray_geometry` gives the procedural tray, and for the
same reason. It checks four things:

1. `drawn["wear"]` actually reads back off `Factor_Float`;
2. `drawn["roughness"]` actually reads back off `A_Float`;
3. the ramp drives `B_Float` (not the shader directly — the exact old-fallback
   shape);
4. Principled's Roughness is driven by the mix's `Result_Float`.

An assertion nobody has ever seen fail is not evidence, so
`tests/test_synth3d_materials.py` (new, 11 tests) loads `materials.py` under a
stub `bpy` — the module only touches `bpy.data` inside `build`, which the tests
never call — under a private module name, restoring `sys.modules` afterwards so
nothing leaks. It then drives the real functions against fake nodes and requires
the assertion to fire on each way the graph can silently mis-wire:

* roughness left at the socket default;
* wear left at the socket default;
* the ramp wired straight into Roughness (the old fallback, reproduced exactly);
* Roughness unlinked;
* the ramp not driving `B_Float`.

Plus a demonstration of *why* index and name access were wrong: on a fake node
whose Color pair comes first, `mix.inputs[2]` and `mix.inputs["A"]` both hand
back `A_Color` while `socket_by_identifier` still resolves `A_Float`.

## 4. `recog/synth3d/render.py` — three clauses assessed

### `configure_beauty:219` — view transform. **Genuinely wrong; fixed.**

`try: s.view_settings.view_transform = cfg.view_transform / except TypeError:
pass` left the run on the build's **default** transform while the exposure on the
very next line was applied regardless. `param_space.exposure` samples −5.2…−3.2,
a band tuned specifically against AgX's response, so under any other transform
every image in the run is uniformly mis-tone-mapped — plausible pictures, wrong
distribution — and `manifest.json` records `cfg.to_dict()`, i.e. the *requested*
`AgX`. `--sweep` sheets look internally consistent precisely because every entry
is wrong the same way.

Now an unguarded assignment plus an explicit read-back assertion naming both the
requested and the actual transform. (The audit also suggested recording the
effective value into `meta`; the assertion is strictly stronger — the run cannot
proceed with a transform other than the requested one — and recording it would
mean editing `recog/generate3d.py`, which is outside this change's file set.)

### `read_index_exr:294` — `Non-Color` colorspace. **Genuinely wrong; fixed.**

This is the single line between `np.rint()` and a colour-managed float buffer. If
the object-index pass is read back through a display transform, every instance id
decodes to the wrong value and every mask in the dataset is silently mislabelled
while looking entirely normal. A build or OCIO config with no `"Non-Color"` space
cannot produce correct masks here at all, so it must stop. The `except Exception:
pass` guaranteed nobody would find out. The `try` is gone.

The sibling assignment in **`save_mask_png:422` keeps its `pass`, deliberately**:
that image is a diagnostic PNG written from ids that are already decoded, so a
colour-space slip there costs a slightly wrong debug picture and nothing else.
Not changed.

### `configure_beauty:204` — the denoiser loop. **Partly deliberate; narrowed.**

```python
for attr, val in (("denoiser", "OPENIMAGEDENOISE"), ...):
    try: setattr(c, attr, val)
    except Exception: pass
```

The *intent* — tolerate Cycles' denoiser options moving across Blender versions —
is legitimate, and the module already uses the explicit form of it three lines
earlier (`if hasattr(c, "use_light_tree")`). What was wrong is the blanket
`except`, which also swallows an attribute that **exists and rejects the value**
(a renamed enum item such as `"ACCURATE"`), silently rendering a whole dataset
noisier than `manifest.json`'s `denoise: true` claims.

Now a `hasattr` check that warns and skips — preserving the version tolerance —
with the `setattr` itself unguarded. Note `c.use_denoising = cfg.denoise` above
it is already unguarded, so a build with no denoiser at all raises before
reaching the loop; this loop tolerates attribute-name drift and nothing else.

### Left alone

`enable_gpu:230/237` (`KeyError` on a missing cycles addon, `TypeError` while
probing compute backends) — both are genuine capability probes that print or
continue to the next candidate, and neither can produce a wrong-but-plausible
label. `save_mask_png:422`, as above.

---

## This changes rendering output

Say it plainly: **any new render will differ from the committed ones.** Every
surface in every existing synthetic image was rendered with the raw noise ramp
wired straight into Roughness rather than the wear-weighted mix, because the
fallback branch was reachable on 100 % of surfaces. With the fix, surfaces render
with the drawn roughness that `meta["materials"]` always claimed.

**Existing datasets are not invalidated.** The labels are geometric — boxes and
masks come from the object-index pass, which materials do not touch — so every
committed annotation remains exactly as correct as it was. What changes is
appearance, and therefore what a *newly generated* corpus looks like relative to
the committed one.

Two consequences worth knowing before anyone re-renders:

* `MIN_LUMA_DELTA` and the `luma_ref` table in `configs/synth3d.yaml` were
  measured from renders. They were measured on the broken path, so they describe
  the appearance of surfaces rendered with ramp-driven roughness. They are not
  obviously wrong afterwards — `luma_ref` is dominated by base colour and coat,
  not by the roughness mix — but they are no longer measurements of the path
  that now runs, and should be re-measured if a corpus is ever redrawn.
* Checkpoints trained on the committed corpus remain valid against that corpus.
  A corpus redrawn after this change is a different visual distribution, so
  mixing renders from before and after in one training set would be a silent
  domain split.

**No dataset was regenerated, no render was run, and no checkpoint was
retrained as part of this change.**

## Suite

* Before: 752 passing.
* After: 763 collected — 752 existing plus the 11 new tests in
  `tests/test_synth3d_materials.py`.
* At the time of the final run, `tests/test_planner.py` reported 2 failures
  (`test_cycle_marks_cells_planned`, `test_planner_measures_each_frame_at_that_
  frames_own_scale`). **These are not from this change.** They come from another
  agent's in-flight edit to `plan/planner.py` / `plan/scene.py` (the occupancy
  block-marking fix — `assert 1100 == 2` is a footprint block against the old
  single cell). `tests/test_planner.py` imports only `common.types` and `plan.*`;
  none of the four files touched here are reachable from it, and the suite was
  green with all four changes in place before those edits landed.
  `pytest --ignore=tests/test_planner.py` → **743 passed**.
