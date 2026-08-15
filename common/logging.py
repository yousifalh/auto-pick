"""Single-source logger setup.

The entire pipeline uses :func:`get_logger` so that log formatting is
uniform across perception, planning, execution, and the main loop. The
function is idempotent: it attaches at most one handler per logger
name so that repeated imports do not produce duplicated lines.

Levels are set on the loggers themselves, never inherited. Each logger
gets its own handler and ``propagate = False`` (so one line is emitted
once, and pytest's ``caplog`` deliberately does not see these records —
``tests/conformance.CapturedCritical`` attaches to the named logger
instead). That means there is no single ancestor whose level governs
the pipeline, which is why :func:`set_level` exists and why it walks
the loggers this module has issued rather than touching the root.
"""
from __future__ import annotations

import logging
import sys
from typing import List, Optional

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Every logger this module has handed out, in issue order. Needed
# because each one carries its own level (see the module docstring):
# `mode.log_level` has to reach loggers that modules built at IMPORT
# time, long before any config was read.
_ISSUED: List[logging.Logger] = []


# The five levels this project uses, spelled out rather than looked up.
# `logging.getLevelName` is the obvious validator and is the wrong one:
# it consults a PROCESS-GLOBAL registry any installed package may add to,
# and measured in this environment on 2026-08-15 it resolved "VERBOSE" to
# 15 — a level nothing here emits at, contributed by a dependency, and
# accepted silently. A `mode.log_level` typo that happens to collide with
# some library's custom level would then set a level no code in this
# repository logs at, which is indistinguishable from silence.
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _resolve_level(level: str) -> int:
    """``"WARNING"`` -> 30, and anything that is not a level raises.

    A typo in ``mode.log_level`` must not leave the run at whatever
    level it happened to have.
    """
    try:
        return _LEVELS[str(level).upper()]
    except KeyError:
        raise ValueError(
            f"{level!r} is not a log level this project emits at. Use one "
            f"of {', '.join(_LEVELS)}.") from None


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for ``name``.

    Subsequent calls with the same ``name`` return the same logger
    instance without re-attaching handlers. ``level`` is an optional
    case-insensitive string (e.g. ``"DEBUG"``, ``"WARNING"``); when
    omitted the logger defaults to ``INFO`` on first issue and is left
    alone afterwards.

    ``level`` is applied on EVERY call, not only the first. The early
    return that stops duplicated handlers used to take the level with
    it, and since every module in this pipeline builds its logger at
    import time with no level, the argument did nothing by the time any
    caller had a configured value to pass - a parameter that looks like
    it sets the level and does not.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        if level is not None:
            logger.setLevel(_resolve_level(level))
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.setLevel(_resolve_level(level or "INFO"))
    logger.propagate = False
    _ISSUED.append(logger)
    return logger


def set_level(level: str) -> None:
    """Apply ``level`` to every logger :func:`get_logger` has issued.

    This is what ``mode.log_level`` drives. It was a key in
    ``configs/demo.yaml`` and ``configs/demo_seg.yaml`` that no Python
    read (audit 2026-08-15, finding C1) - the same class as the twelve
    dead keys in ``configs/planning.yaml``, and in the one file whose
    first line a new reader opens.
    """
    numeric = _resolve_level(level)
    for logger in _ISSUED:
        logger.setLevel(numeric)
