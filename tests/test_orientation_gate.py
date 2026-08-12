"""The orientation gate must actually be wired to something that runs it.

`recog/synth3d/_gate_orientation.py` is the only executable check in this
repository against the `lay_flat` / `place_item` no-op class of failure —
the defect that rendered every cartridge unturned for its entire life
without anyone noticing, because the scene still rendered. It called
itself a CI gate while `.github/workflows/ci.yml` invoked `pytest` and
the demo loop and nothing else.

Two tests here, doing two different jobs:

* :func:`test_ci_workflow_runs_the_orientation_gate` is the *wiring*
  lock. It is deterministic, needs no Blender, and runs in the ordinary
  torch-free suite, so deleting the CI step turns the suite red instead
  of silently disarming the gate. This is the one that would have caught
  the original state.
* :func:`test_orientation_gate_passes` actually *runs* the gate, and
  skips when `bpy` is absent. On a machine or a runner with Blender it
  is the real check; in the torch-free environment it is honest about
  being skipped rather than pretending to have verified anything.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE = os.path.join(_ROOT, "recog", "synth3d", "_gate_orientation.py")
_WORKFLOW = os.path.join(_ROOT, ".github", "workflows", "ci.yml")


def test_the_gate_script_still_exists():
    assert os.path.isfile(_GATE), (
        "the orientation gate is the only check against the lay_flat "
        "no-op; deleting it needs a spec, not a commit")


def test_ci_workflow_runs_the_orientation_gate():
    """CI must name the gate script in a step that runs it.

    Asserted against the workflow text rather than against a job name,
    because the property that matters is "something executes this file",
    not "a job called orientation-gate exists".
    """
    with open(_WORKFLOW, encoding="utf-8") as fh:
        text = fh.read()

    assert "_gate_orientation.py" in text, (
        "ci.yml does not invoke recog/synth3d/_gate_orientation.py. A "
        "file that calls itself a CI gate and that no CI runs is not a "
        "gate — that was the state this test was written to end.")

    # The step must be able to fail the build. `continue-on-error: true`
    # anywhere in the job would turn the gate back into decoration.
    job = text.split("orientation-gate:", 1)[1]
    assert "continue-on-error" not in job, (
        "the orientation gate must be able to fail the build")


def test_the_gate_is_runnable_without_a_blender_binary():
    """The wheel form (`python <gate>`) must stay usable.

    The CI job runs the gate as a plain script against the `bpy` PyPI
    wheel, so the file must keep its own `sys.path` bootstrap and its
    `if __name__ == "__main__"` entry — running it under `blender -b
    --python` alone would need a Blender binary that no hosted runner
    has.
    """
    with open(_GATE, encoding="utf-8") as fh:
        src = fh.read()
    assert 'if __name__ == "__main__":' in src
    assert "sys.path.insert" in src
    assert "sys.exit(code)" in src


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None,
                    reason="needs Blender (pip install bpy, or run under "
                           "`blender -b --python`)")
def test_orientation_gate_passes():
    """Run the gate for real. Exit code 0 means every check passed."""
    proc = subprocess.run(
        [sys.executable, _GATE], cwd=_ROOT,
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"orientation gate failed:\n{proc.stdout[-4000:]}\n"
        f"{proc.stderr[-2000:]}")
