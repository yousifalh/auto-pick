"""
recog.convert_cad - import CAD into the synthetic-dataset asset library.

    python -m recog.convert_cad --src cad/ --out recog/synth3d/assets/

Converts every .stp/.step under --src to glTF and MERGES the result into
--out/catalog.json. Existing entries survive; an asset re-imported under a
name that is already in the catalog replaces that one entry in place.

Runs outside Blender - needs cascadio + trimesh, which Blender's bundled
Python does not have. Blender cannot read STEP at all, so this is a required
preprocessing step.

    pip install cascadio trimesh

WHY THE SCALE GUARD EXISTS
--------------------------
A part that lands in the catalog 1000x too small is placed happily by the
layout solver, renders sub-pixel, and then has every one of its boxes
dropped by filter.min_px. The result is hours of rendering that produce a
dataset nearly empty of that asset, with no error raised anywhere. So this
importer refuses to write an entry whose extents are outside a sane range
for this domain (roughly 10mm to 500mm) - see --min-mm/--max-mm/--force.

MEASURED BEHAVIOUR OF cascadio 0.1.1 (OCCT)
-------------------------------------------
cascadio honours the STEP length-unit declaration correctly. Rewriting only
the unit entities of one real file and re-converting gives:

    SI_UNIT(.MILLI.,.METRE.)                 -> 0.12 x 0.05 x 0.02 mm
    SI_UNIT($,.METRE.)                       ->  125 x   50 x   20 mm
    CONVERSION_BASED_UNIT + factor 1000.0    ->  125 x   50 x   20 mm
    CONVERSION_BASED_UNIT 'INCH' factor 25.4 -> 3.18 x 1.27 x 0.51 mm

i.e. the conversion factor is applied, not just the SI base unit, and
CONVERSION_BASED_UNIT is not treated differently from the SI form.

The real hazard is therefore a file whose declaration DISAGREES with its
geometry. Lug.stp is exactly that: it declares MILLIMETRE but its raw
coordinates span 0.125 units, so it is authored in metres and converts to a
0.12mm part. No parser can detect that from the header alone, which is why
the plausibility guard - not the unit parser - is the safety net. Override a
bad declaration with --assume-unit m (or a raw --scale).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

from recog.synth3d.catalog import convert_step, inspect_glb
from recog.synth3d.config import CLASS_RULES

STEP_SUFFIXES = (".stp", ".step")

# Parts in this domain: an 18650 cell is 18mm, a jig plate is a few hundred.
DEFAULT_MIN_MM = 10.0
DEFAULT_MAX_MM = 500.0

# --assume-unit choices, metres per file unit.
KNOWN_UNITS: Dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
}

_SI_PREFIX = {
    "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6,
    "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1, "DECI": 1e-1, "CENTI": 1e-2,
    "MILLI": 1e-3, "MICRO": 1e-6, "NANO": 1e-9, "PICO": 1e-12,
    "FEMTO": 1e-15, "ATTO": 1e-18,
}

_MASKED_SEMI = "\x00"


# ========================================================== unit parsing ====

class LengthUnit(NamedTuple):
    """The length unit a STEP file declares for its own coordinates."""
    metres: float          # metres per file unit
    label: str             # "MILLIMETRE", "INCH", ...
    encoding: str          # "SI_UNIT" | "CONVERSION_BASED_UNIT"

    def describe(self) -> str:
        return f"{self.label} [{self.encoding}] (1 unit = {self.metres:g} m)"


class UnitError(ValueError):
    """The STEP file's length unit could not be determined unambiguously."""


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", " ", text, flags=re.S)


def _mask_string_semicolons(text: str) -> str:
    """Hide ';' inside STEP string literals so entity splitting is safe."""
    return re.sub(r"'(?:[^']|'')*'",
                  lambda m: m.group(0).replace(";", _MASKED_SEMI), text)


def _entities(text: str) -> Dict[str, str]:
    """Map "#12=FOO(...);" to {"12": "FOO(...)"}, across line breaks."""
    masked = _mask_string_semicolons(text)
    out: Dict[str, str] = {}
    for m in re.finditer(r"#(\d+)\s*=\s*(.*?);", masked, flags=re.S):
        out[m.group(1)] = m.group(2).replace(_MASKED_SEMI, ";")
    return out


def _parse_real(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_length_unit(body: str, ents: Dict[str, str],
                         depth: int = 0) -> Optional[LengthUnit]:
    """Resolve one unit entity body to metres-per-unit, following references."""
    if depth > 8:                       # cyclic or pathological reference chain
        return None

    m = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'\s*,\s*#(\d+)", body)
    if m:
        name, ref = m.group(1), m.group(2)
        target = ents.get(ref, "")
        factor = re.search(r"LENGTH_MEASURE\s*\(\s*([-+0-9.eE]+)\s*\)", target)
        base_ref = re.search(
            r"LENGTH_MEASURE_WITH_UNIT\s*\(.*?,\s*#(\d+)", target, flags=re.S)
        if not (factor and base_ref):
            return None
        value = _parse_real(factor.group(1))
        base = _resolve_length_unit(ents.get(base_ref.group(1), ""),
                                    ents, depth + 1)
        if value is None or base is None:
            return None
        return LengthUnit(value * base.metres, name.upper() or "UNNAMED",
                          "CONVERSION_BASED_UNIT")

    m = re.search(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.(\w+)\.\s*\)", body)
    if m:
        prefix, unit = m.group(1), m.group(2).upper()
        if unit != "METRE":             # a SI_UNIT that is not a length
            return None
        if prefix == "$":
            return LengthUnit(1.0, "METRE", "SI_UNIT")
        scale = _SI_PREFIX.get(prefix.strip(".").upper())
        if scale is None:
            return None
        return LengthUnit(scale, f"{prefix.strip('.').upper()}METRE", "SI_UNIT")

    return None


def parse_step_length_unit(text: str) -> LengthUnit:
    """
    Read the length unit a STEP file declares for its own coordinates.

    Handles the two encodings real exporters emit:

      * SI_UNIT(.MILLI.,.METRE.) / SI_UNIT($,.METRE.)  - NX, and most
        AP242 writers
      * CONVERSION_BASED_UNIT('MILLIMETRE', #n) where #n is a
        LENGTH_MEASURE_WITH_UNIT carrying the conversion factor and a
        reference to an SI length unit - SolidWorks, Fusion, ACIS-based
        translators. The factor is applied, so INCH (25.4 x MILLIMETRE),
        CENTIMETRE and METRE all resolve numerically.

    Prefers the units named by GLOBAL_UNIT_ASSIGNED_CONTEXT; if the file has
    none, every length unit in the file is considered. Raises UnitError when
    nothing resolves or when contexts disagree - the caller must NOT guess.
    """
    ents = _entities(_strip_comments(text))
    if not ents:
        raise UnitError("no STEP entities found (is this a STEP file?)")

    ids: List[str] = []
    for body in ents.values():
        for m in re.finditer(
                r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\((.*?)\)", body,
                flags=re.S):
            ids.extend(re.findall(r"#(\d+)", m.group(1)))
    if not ids:
        ids = list(ents)

    found: Dict[Tuple[float, str, str], None] = {}
    for i in ids:
        body = ents.get(i)
        if body is None or "LENGTH_UNIT" not in body:
            continue
        unit = _resolve_length_unit(body, ents)
        if unit is not None:
            found[tuple(unit)] = None    # dict keeps first-seen order

    if not found:
        raise UnitError(
            "no length unit declared (looked for SI_UNIT(...,.METRE.) and "
            "CONVERSION_BASED_UNIT with a LENGTH_MEASURE_WITH_UNIT factor)")

    units = [LengthUnit(*k) for k in found]
    if len({round(u.metres, 15) for u in units}) > 1:
        raise UnitError(
            "the file declares conflicting length units: "
            + ", ".join(u.describe() for u in units))
    return units[0]


def read_step_length_unit(path: str) -> LengthUnit:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_step_length_unit(fh.read())


# ==================================================== plausibility guard ====

def implausible(extents_mm, min_mm: float = DEFAULT_MIN_MM,
                max_mm: float = DEFAULT_MAX_MM) -> Optional[str]:
    """
    Reason the asset's size is not credible for this domain, or None.

    Guards the LARGEST extent only. A jig plate is legitimately 6mm thick,
    so a floor on the smallest extent would reject real parts; a 1000x scale
    error moves every extent together and so always shows up in the largest.
    """
    values = [float(v) for v in extents_mm]
    if not values or not all(v == v for v in values):      # empty or NaN
        return "extents could not be measured"
    biggest = max(values)
    if biggest <= 0:
        return "zero-sized geometry"
    if biggest < min_mm:
        return (f"largest extent {biggest:g}mm is below {min_mm:g}mm - "
                f"likely declared in the wrong unit")
    if biggest > max_mm:
        return (f"largest extent {biggest:g}mm is above {max_mm:g}mm - "
                f"likely declared in the wrong unit")
    return None


def suggest_units(extents_mm, unit: Optional[LengthUnit],
                  min_mm: float = DEFAULT_MIN_MM,
                  max_mm: float = DEFAULT_MAX_MM) -> List[str]:
    """Which --assume-unit values would bring this part into range."""
    if unit is None:
        return []
    biggest = max(float(v) for v in extents_mm)
    if biggest <= 0:
        return []
    out = []
    for name, metres in KNOWN_UNITS.items():
        if min_mm <= biggest * (metres / unit.metres) <= max_mm:
            out.append(name)
    return out


# ============================================================ role check ====

def unmatched_subparts(subparts) -> List[str]:
    """
    Sub-part names that hit ROLE_FALLBACK because no CLASS_RULES regex fired.

    role_of() cannot report this: its fallback role ("case") is also a role a
    rule can produce, so a genuine match and a fallback are indistinguishable
    from the outside.
    """
    out = []
    for sp in subparts:
        name = sp["name"]
        if not any(re.search(p, name, flags=re.IGNORECASE)
                   for p, _ in CLASS_RULES):
            out.append(name)
    return out


# =============================================================== catalog ====

def asset_name_for(filename: str) -> str:
    """
    "004708_A_2-AnkerPowerCore26800.stp" -> "AnkerPowerCore26800".

    Mirrors catalog.build_catalog so re-importing the original CAD reproduces
    the names already in the committed catalog.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("-", 1)[-1] if "-" in stem else stem


def merge_catalog(existing: Optional[dict], new_assets: List[dict],
                  tol_linear: float, tol_angular: float) -> dict:
    """
    Fold new assets into an existing catalog without disturbing the rest.

    Entries already present keep their position and are replaced only when
    re-imported under the same name; every other entry is passed through
    untouched, as are the catalog's top-level keys.
    """
    if existing is None:
        catalog = {
            "units": "m",
            "note": "glTF geometry is in metres; extents_mm are millimetres",
            "tol_linear_mm": tol_linear,
            "tol_angular": tol_angular,
            "assets": [],
        }
    else:
        catalog = dict(existing)
    assets = list(catalog.get("assets") or [])

    index = {a.get("name"): i for i, a in enumerate(assets)}
    for asset in new_assets:
        pos = index.get(asset["name"])
        if pos is None:
            index[asset["name"]] = len(assets)
            assets.append(asset)
        else:
            assets[pos] = asset
    catalog["assets"] = assets
    return catalog


def load_existing_catalog(out_dir: str) -> Optional[dict]:
    path = os.path.join(out_dir, "catalog.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _rescale_glb(path: str, factor: float) -> None:
    import trimesh
    scene = trimesh.load(path)
    scene.apply_scale(factor)
    scene.export(path)


# ================================================================== main ====

def _step_files(src: str) -> List[str]:
    if os.path.isfile(src):
        return [src]
    if not os.path.isdir(src):
        raise FileNotFoundError(f"--src {src} does not exist")
    names = sorted(f for f in os.listdir(src)
                   if f.lower().endswith(STEP_SUFFIXES))
    return [os.path.join(src, f) for f in names]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m recog.convert_cad",
        description="Convert STEP CAD to glTF and merge it into the "
                    "synth3d asset catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sub-part names drive role assignment through CLASS_RULES in "
               "recog/synth3d/config.py. If a new assembly's sub-parts do "
               "not match any rule they all fall back to 'case' and nothing "
               "is ever labelled 'battery' - this tool warns when that "
               "happens.")
    ap.add_argument("--src", required=True,
                    help="directory of .stp/.step files, or a single file")
    ap.add_argument("--out", default=os.path.join("recog", "synth3d", "assets"),
                    help="asset directory holding catalog.json "
                         "(default: %(default)s)")
    ap.add_argument("--tol-linear", type=float, default=0.05,
                    help="tessellation chord tolerance, mm (default: %(default)s)")
    ap.add_argument("--tol-angular", type=float, default=0.3)
    ap.add_argument("--assume-unit", choices=sorted(KNOWN_UNITS),
                    default=None,
                    help="override a wrong unit declaration in the file")
    ap.add_argument("--scale", type=float, default=None,
                    help="raw multiplier on the converted geometry; use when "
                         "the unit cannot be parsed at all")
    ap.add_argument("--min-mm", type=float, default=DEFAULT_MIN_MM)
    ap.add_argument("--max-mm", type=float, default=DEFAULT_MAX_MM)
    ap.add_argument("--force", action="store_true",
                    help="write entries that fail the plausibility guard or "
                         "whose unit could not be determined")
    ap.add_argument("--dry-run", action="store_true",
                    help="convert and report, but do not write catalog.json")
    args = ap.parse_args(argv)
    if args.assume_unit and args.scale is not None:
        ap.error("--assume-unit and --scale are mutually exclusive")
    return args


def convert_one(path: str, out_dir: str, args) -> Tuple[Optional[dict], bool]:
    """
    Convert one STEP file. Returns (catalog entry or None, ok).

    Tessellates to a temporary file and only moves it into place once the
    entry is accepted: a rejected import must not overwrite the .glb of a
    good asset that the existing catalog still points at.
    """
    name = asset_name_for(path)
    glb = os.path.join(out_dir, name + ".glb")
    # Keeps the .glb suffix: trimesh sniffs the format from the extension.
    # Same directory as the target, so the final os.replace is atomic.
    tmp = os.path.join(out_dir, f".{name}.part.glb")
    replacing = os.path.isfile(glb)

    unit: Optional[LengthUnit] = None
    unit_error: Optional[str] = None
    try:
        unit = read_step_length_unit(path)
    except (UnitError, OSError) as exc:
        unit_error = str(exc)

    print(f"\n{name}  <- {os.path.basename(path)}")
    if unit is not None:
        print(f"  unit      {unit.describe()}")
    else:
        print(f"  unit      UNDETERMINED: {unit_error}")

    if unit is None and args.scale is None and not args.force:
        print("  SKIPPED   cannot determine the length unit, and guessing it "
              "would silently mis-scale the dataset.\n"
              "            Re-run with --scale <factor> if you know the file, "
              "or --force to accept it as converted.")
        return None, False
    if unit is None and args.assume_unit:
        print("  SKIPPED   --assume-unit needs a parsed declaration to "
              "correct; use --scale instead.")
        return None, False

    factor = 1.0
    if args.scale is not None:
        factor = args.scale
    elif args.assume_unit:
        factor = KNOWN_UNITS[args.assume_unit] / unit.metres

    try:
        convert_step(path, tmp, args.tol_linear, args.tol_angular)
        if factor != 1.0:
            _rescale_glb(tmp, factor)
            print(f"  rescale   x{factor:g}"
                  + (f" (declared {unit.label}, treated as "
                     f"{args.assume_unit})" if args.assume_unit else ""))
        info = inspect_glb(tmp)
    except Exception:
        _unlink(tmp)
        raise
    ext = info["extents_mm"]
    print(f"  extents   {ext[0]:g} x {ext[1]:g} x {ext[2]:g} mm")
    print(f"  triangles {info['triangles']}")
    print(f"  roles     {info['role_counts']}")

    stray = unmatched_subparts(info["subparts"])
    if stray:
        print(f"  [warn] {len(stray)} of {len(info['subparts'])} sub-parts "
              f"matched no CLASS_RULES pattern and fell back to role "
              f"'case':")
        for n in stray[:12]:
            print(f"           {n}")
        if len(stray) > 12:
            print(f"           ... and {len(stray) - 12} more")
        print("         Add a pattern to CLASS_RULES in "
              "recog/synth3d/config.py, or nothing in this asset will ever "
              "be labelled 'battery'.")

    reason = implausible(ext, args.min_mm, args.max_mm)
    if reason:
        # Suggestions must be relative to the scale the geometry is ALREADY
        # at, not the declared one - otherwise a run that passed
        # --assume-unit cm gets told to try --assume-unit cm.
        effective = None if unit is None else unit._replace(
            metres=unit.metres * factor)
        print(f"  {'FORCED' if args.force else 'REJECTED'}  {reason}")
        print(f"            measured {ext[0]:g} x {ext[1]:g} x {ext[2]:g} mm, "
              f"unit {unit.describe() if unit else 'UNDETERMINED'}")
        hints = suggest_units(ext, effective, args.min_mm, args.max_mm)
        if hints:
            print("            this would land in range with "
                  + " or ".join(f"--assume-unit {h}" for h in hints))
        if not args.force:
            _unlink(tmp)
            print("            no catalog entry written, "
                  f"{os.path.basename(glb)} left untouched (--force overrides)")
            return None, False

    if args.dry_run:
        _unlink(tmp)
    else:
        os.replace(tmp, glb)
    info.update({
        "name": name,
        "source": os.path.basename(path),
        "file": os.path.basename(glb),
        "tol_linear_mm": args.tol_linear,
        "tol_angular": args.tol_angular,
        "detected_unit": unit.label if unit else None,
        "unit_m": unit.metres if unit else None,
    })
    if factor != 1.0:
        info["scale_applied"] = factor
    if args.assume_unit:
        info["assumed_unit"] = args.assume_unit
    verb = ("would replace" if replacing else "would add") if args.dry_run \
        else ("replaced" if replacing else "added")
    print(f"  {verb}  {info['file']}")
    return info, True


def main(argv=None) -> int:
    args = parse_args(argv)
    files = _step_files(args.src)
    if not files:
        print(f"no {'/'.join(STEP_SUFFIXES)} files in {args.src}",
              file=sys.stderr)
        return 1

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    existing = load_existing_catalog(out_dir)
    print(f"{len(files)} STEP file(s) -> {out_dir}")
    if existing:
        print(f"merging into {len(existing.get('assets') or [])} existing "
              f"asset(s)")
        for key, value in (("tol_linear_mm", args.tol_linear),
                           ("tol_angular", args.tol_angular)):
            if key in existing and existing[key] != value:
                print(f"[warn] catalog {key}={existing[key]} but this run "
                      f"uses {value}; the top-level value is left as-is and "
                      f"the new entries record their own.")

    new_assets, skipped = [], []
    for path in files:
        entry, ok = convert_one(path, out_dir, args)
        if entry is not None:
            new_assets.append(entry)
        if not ok:
            skipped.append(os.path.basename(path))

    if args.dry_run:
        print(f"\n--dry-run: catalog.json not written "
              f"({len(new_assets)} entr(y/ies) would be merged)")
        return 1 if skipped else 0

    if new_assets:
        catalog = merge_catalog(existing, new_assets,
                                args.tol_linear, args.tol_angular)
        path = os.path.join(out_dir, "catalog.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(catalog, fh, indent=2)
        print(f"\nwrote {path}: {len(catalog['assets'])} asset(s) "
              f"({len(new_assets)} imported this run)")
    else:
        print("\nnothing imported; catalog.json left untouched")

    if skipped:
        print(f"SKIPPED {len(skipped)} file(s): {', '.join(skipped)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
