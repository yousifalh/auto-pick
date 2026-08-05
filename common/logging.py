"""Single-source logger setup.

The entire pipeline uses :func:`get_logger` so that log formatting is
uniform across perception, planning, execution, and the main loop. The
function is idempotent: it attaches at most one handler per logger
name so that repeated imports do not produce duplicated lines.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for ``name``.

    Subsequent calls with the same ``name`` return the same logger
    instance without re-attaching handlers. ``level`` is an optional
    case-insensitive string (e.g. ``"DEBUG"``, ``"WARNING"``); when
    omitted the logger defaults to ``INFO``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.setLevel((level or "INFO").upper())
    logger.propagate = False
    return logger
