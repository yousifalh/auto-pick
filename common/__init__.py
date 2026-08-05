"""Cross-module types and utilities.

This package is deliberately small and has no heavy dependencies
(no torch, no OpenCV) so it can be imported freely across the rest
of the codebase — including in unit tests that run in a
CPU-only CI environment without the optional ``[train]`` extras.
"""
from .types import (
    BBox,
    ClassLabel,
    Detection,
    PickPlacePose,
    RobotStatus,
    RobotStatusCode,
    Snapshot,
    WorkspacePoint,
)

__all__ = [
    "BBox",
    "ClassLabel",
    "Detection",
    "PickPlacePose",
    "RobotStatus",
    "RobotStatusCode",
    "Snapshot",
    "WorkspacePoint",
]
