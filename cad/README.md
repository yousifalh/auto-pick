# cad/

Drop CAD here, then import it into the asset library:

```bash
python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/
```

That tessellates every `.stp`/`.step` to glTF and **merges** the result into
`recog/synth3d/assets/catalog.json`. Assets already in the catalog survive;
re-importing a file whose asset name is already there replaces that one entry.

Everything in this directory except this README is gitignored — the source CAD
is large, and the converted `.glb` plus `catalog.json` are what the generator
actually reads.

## STEP only

Blender cannot read STEP, and the importer cannot read native CAD formats.
**SolidWorks `.SLDPRT` / `.SLDASM` will not work**, nor will `.f3d`, `.prt`,
`.CATPart` or `.ipt`. Export to STEP first:

* SolidWorks — File > Save As > STEP AP214 (`.step`). For an assembly, save as
  a single file; do not tick "export as separate files" or the sub-part
  structure the role rules depend on is lost.
* Fusion 360 — File > Export > STEP.
* Onshape — right-click the tab > Export > STEP.

`.stl` is not accepted either: it carries no sub-part names, so every triangle
soup would collapse into a single unnamed part with no role.

## Sub-part names decide the labels

Each sub-part's name is matched against `CLASS_RULES` in
`recog/synth3d/config.py`; the first regex that hits assigns the role, and
unmatched sub-parts fall back to `case`. Roles are what `VARIANTS` turns into
the `battery` / `cartridge` classes.

The existing rules were written for NX names such as
`004695_A;1-Cell_18650` and `004697_A;2-Case10000_top`:

```python
CLASS_RULES = [
    (r"Cell[_ ]?\d+", "cell"),
    (r"Case.*_(top|btm)", "case"),
]
```

A different CAD tool will name things differently. If nothing matches, every
sub-part silently becomes `case` and the dataset ends up with nothing labelled
`battery` — so the importer prints a warning listing the names that matched no
rule. When you see it, either rename the sub-parts in CAD or add a pattern to
`CLASS_RULES`. Cells especially must match a `cell` rule.

## Scale

The importer reads each file's declared length unit and refuses to write an
entry whose largest extent falls outside roughly 10-500 mm, printing what it
measured and what it detected. That guard exists because a part imported
1000x too small still places and renders fine, just sub-pixel — every one of
its boxes is then dropped by `filter.min_px`, and you get an empty dataset
after hours of rendering with no error anywhere.

Some exporters declare a unit that disagrees with the geometry they wrote
(`Lug.stp` declares MILLIMETRE but is authored in metres). Override with:

```bash
python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/ --assume-unit m
```

`--dry-run` reports extents without writing anything; `--force` overrides the
guard. Check the printed extents against the real part before trusting either.
