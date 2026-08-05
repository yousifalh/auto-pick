"""Transcribe configs/synth3d.yaml to the JSON sidecar Blender reads.

Blender's bundled Python has numpy but no PyYAML, so the generator cannot
read the authored YAML directly. Run this after every edit to synth3d.yaml:

    python -m recog.sync_config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from recog.synth3d.config import default_config_path


def sync(yaml_path: Path) -> Path:
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out = yaml_path.with_suffix(".json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    src = Path(args.config) if args.config else default_config_path()
    out = sync(src)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
