"""Thin YAML configuration loader.

The project deliberately avoids the pydantic / hydra / dynaconf stack:
configuration is a small number of shallow YAML files, and a function
that returns nested :class:`dict` is the simplest thing that keeps
this surface testable and dependency-light.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_SUB_KEYS: tuple[str, ...] = ("recognition", "planning", "execution")


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Load a single YAML file and return its top-level mapping.

    An empty file is returned as an empty dict (YAML's canonical
    interpretation of *nothing*). A missing file raises
    :class:`FileNotFoundError`, since mis-pointed config paths are
    almost always a user error worth surfacing loudly.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return dict(data) if isinstance(data, dict) else {}


def load_demo_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a top-level ``demo.yaml`` and inline the three sub-configs.

    ``demo.yaml`` names the three module configs by path (either
    absolute, or relative to the *project root* — i.e. the grandparent
    of ``demo.yaml``). Returns a single nested dict with the keys
    ``recognition``, ``planning``, ``execution`` and ``mode``.
    """
    top = load_yaml(path)
    project_root = Path(path).resolve().parent.parent

    resolved: dict[str, Any] = {}
    for key in _SUB_KEYS:
        sub = top.get(key)
        if isinstance(sub, str):
            sub_path = Path(sub)
            if not sub_path.is_absolute():
                sub_path = project_root / sub_path
            resolved[key] = load_yaml(sub_path)
        elif isinstance(sub, dict):
            resolved[key] = sub
        else:
            resolved[key] = {}

    resolved["mode"] = top.get("mode", {}) or {}
    return resolved
