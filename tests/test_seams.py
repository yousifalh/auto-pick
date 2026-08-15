"""The two seams the layering rests on, asserted rather than assumed.

Both of these are architecture facts that no functional test can notice
going wrong. A back-edge from ``recog`` into ``plan`` does not break a
run; a driver abstraction that nothing consumes polymorphically still
passes its own conformance suite. Each defect shows up only as the next
change being harder than it should have been, which is exactly the kind
of thing that needs a test to hold it.

SEAM-1 - the layering. ``common`` is a leaf; ``execution`` and ``plan``
depend on it; ``recog`` depends on it. ``recog`` must not depend on
``plan`` at module load. ``plan/bin_packing.py``'s own docstring already
gives this as the reason the packing algorithms live in
``common.packing`` - "so that recog.synth3d.layout can use them too
without creating a back-edge from recog into plan" - and the back-edge
was nevertheless taken twice, for two shared CONTRACTS (``UnknownScale``
and the wall inset) that had simply been filed in the wrong package.

SEAM-2 - the driver factory. ``execution.driver.RobotDriver`` seals
sixteen members and is implemented by two independent backends that pass
one conformance suite, and until 2026-08-15 nothing selected between
them: ``main.py`` named ``execution.mock_kuka_server``, ``KukaClient``
and ``ExecutionConfig`` directly, and ``JsonRobotDriver`` appeared in no
non-test module. A seam nothing chooses across is a seam nobody has
checked.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Packages whose edges this file governs. ``common`` is the leaf.
PACKAGES = ("common", "recog", "plan", "execution")


def _imports(path: Path):
    """``[(module, is_module_level, lineno)]`` for one file.

    Relative imports are skipped: they cannot cross a package boundary,
    which is the only thing being measured here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    top = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top.add(id(node))
        else:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    top.discard(id(sub))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, id(node) in top, node.lineno))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.append((node.module, id(node) in top, node.lineno))
    return out


def _package_edges(source_package: str, target_package: str):
    """``([module-level], [function-level])`` edges, as printable strings."""
    module_level, function_level = [], []
    for path in sorted((REPO_ROOT / source_package).rglob("*.py")):
        for module, is_top, lineno in _imports(path):
            if module.split(".")[0] != target_package:
                continue
            entry = f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno} {module}"
            (module_level if is_top else function_level).append(entry)
    return module_level, function_level


# ------------------------------------------------- SEAM-1: the layering ---

def test_recog_does_not_import_plan_at_module_level():
    """No back-edge from Recognition into Planning at import time.

    The two that existed were both CONTRACTS, not algorithms:
    ``recog/seg_evaluate.py`` imported ``UnknownScale`` and
    ``recog/calibrate_tau.py`` imported ``_DEFAULT_WALL_INSET_MM``, each
    with a comment explaining - correctly - that importing beats
    restating. Importing was right; the location was wrong. Both now
    live in ``common.types``, which both sides already depend on.
    """
    module_level, _ = _package_edges("recog", "plan")
    assert module_level == [], (
        "recog imports plan at module level:\n  "
        + "\n  ".join(module_level))


def test_the_remaining_recog_to_plan_edges_are_lazy_and_are_algorithms():
    """What is left is deliberate, and this test says what it is.

    ``recog.seg_ablation`` and ``recog.calibrate_tau`` measure what
    PRODUCTION does - they call ``plan.arbitration.arbitrate``,
    ``plan.placement_area._rasterise_mask`` and the two extractors
    themselves, precisely so the ablation quantises the way the planner
    quantises rather than re-deriving it. Those are genuine cross-layer
    uses by offline analysis scripts and no contract move can remove
    them. They are function-level, so importing ``recog`` still does not
    pull in ``plan`` (or cv2, which is why they were written lazily).

    Pinned as an exact set: a NEW lazy edge is a decision somebody should
    have to make on purpose.
    """
    module_level, function_level = _package_edges("recog", "plan")
    assert module_level == []
    modules = sorted({entry.split(" ")[1] for entry in function_level})
    assert modules == ["plan.arbitration", "plan.placement_area", "plan.scene"]


def test_common_is_a_leaf():
    """``common`` may not import any sibling package, at any depth.

    This is what makes it a legal home for a contract that two packages
    share. The moment it imports one of them, moving a contract into it
    relocates a cycle instead of removing one.
    """
    offenders = []
    for target in ("recog", "plan", "execution"):
        module_level, function_level = _package_edges("common", target)
        offenders += module_level + function_level
    assert offenders == [], (
        "common is no longer a leaf:\n  " + "\n  ".join(offenders))


def test_the_moved_contracts_are_one_object_not_two():
    """``plan.placement_area`` re-exports; it does not restate.

    Identity, not equality. Two ``UnknownScale`` classes that are equal
    by name would let ``except UnknownScale`` miss the one raised by the
    other module - and the planner catches it to turn an uncalibrated
    frame into a counted skip.
    """
    import common.types as ct
    import plan.placement_area as pa

    assert pa.UnknownScale is ct.UnknownScale
    assert pa._DEFAULT_WALL_INSET_MM is ct.DEFAULT_WALL_INSET_MM
    assert ct.DEFAULT_WALL_INSET_MM == 4.25


def test_the_wall_inset_still_reaches_the_extractor_that_erodes_by_it():
    """The move must not change what production erodes.

    4.25 mm is the MAX of the four CAD-measured ``case_wall_mm`` in
    ``recog/synth3d/assets/catalog.json``; the extractor's constructor
    default is the thing that carries it into the erosion radius.
    """
    from common.types import DEFAULT_WALL_INSET_MM
    from plan.placement_area import SegmentationPlacementAreaExtractor

    extractor = SegmentationPlacementAreaExtractor(mm_per_px=0.5)
    assert extractor.wall_inset_mm == DEFAULT_WALL_INSET_MM
    # 4.25 mm at 0.5 mm/px is 8.5 px, which rounds to 8 (banker's
    # rounding on the .5); pinned so a change to either the constant or
    # the rounding is visible here.
    assert extractor.wall_inset_px_at(0.5) == 8


def test_the_recog_analysis_scripts_read_the_shared_constant():
    """Not a private copy that agrees today.

    ``recog/seg_ablation.py`` restated ``4.25`` with a comment saying it
    did so to keep cv2 out of its import surface. ``common.types``
    imports only ``dataclasses``, ``enum`` and ``typing``, so that reason
    no longer applies and the duplicate is gone.
    """
    from common.types import DEFAULT_WALL_INSET_MM
    import recog.calibrate_tau as calibrate_tau
    import recog.seg_ablation as seg_ablation
    import recog.seg_evaluate as seg_evaluate

    assert seg_ablation.DEFAULT_WALL_INSET_MM is DEFAULT_WALL_INSET_MM
    assert calibrate_tau.DEFAULT_WALL_INSET_MM is DEFAULT_WALL_INSET_MM
    assert seg_evaluate.UnknownScale is __import__(
        "common.types", fromlist=["UnknownScale"]).UnknownScale


def test_importing_recog_analysis_modules_does_not_import_plan():
    """The observable consequence of the back-edge being gone.

    Import-time cost is the point: ``plan.placement_area`` imports cv2
    and ``plan.scene``, and these three modules are lazy about cv2
    throughout precisely to stay cheap to import.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import recog.seg_evaluate, recog.calibrate_tau, recog.seg_ablation;"
        "leaked=[m for m in sys.modules if m=='plan' or m.startswith('plan.')];"
        "print(sorted(leaked))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


# --------------------------------------------- SEAM-2: the driver factory --

#: Modules that ARE a vendor, or a vendor's simulator. ``main.py`` names
#: none of them: which backend the run gets is a config question, and
#: ``execution.build_driver`` is where it is answered.
_VENDOR_MODULES = {
    "execution.execution",
    "execution.protocol",
    "execution.mock_kuka_server",
    "execution.json_driver",
    "execution.mock_json_server",
}


def test_main_names_no_vendor_module():
    """``main.py`` imports the seam, never a backend behind it.

    It used to import ``ExecutionConfig``/``KukaClient`` from
    ``execution.execution`` and spawn ``execution.mock_kuka_server`` by
    name, which meant the second backend could not be selected without
    editing the pipeline. It also took ``RobotEstop`` and ``RobotFault``
    from ``execution.execution`` although both are DEFINED in
    ``execution.driver`` - a re-export it was reading through.
    """
    named = {module for module, _top, _line
             in _imports(REPO_ROOT / "main.py")}
    assert not (named & _VENDOR_MODULES), sorted(named & _VENDOR_MODULES)


def test_main_takes_the_failure_taxonomy_from_where_it_is_defined():
    from execution.driver import RobotEstop, RobotFault
    import main

    assert main.RobotEstop is RobotEstop
    assert main.RobotFault is RobotFault


def _exec_cfg(**overrides):
    """A minimal ``execution:`` config block, on an ephemeral port.

    Port 0 asks the OS for a free one; ``build_driver`` reads the port
    the simulator actually bound to rather than the one it asked for,
    which is what makes a test able to run two backends at once without
    a hardcoded 54600/55600 collision.
    """
    cfg = {
        "kuka": {"host": "127.0.0.1", "port": 54600},
        "motion": {"transport_height_mm": 75.0, "vacuum_level_percent": 70},
        "simulation": {"listen_host": "127.0.0.1", "listen_port": 0},
        "json": {"listen_host": "127.0.0.1", "listen_port": 0},
    }
    cfg.update(overrides)
    return cfg


def test_build_driver_returns_the_kuka_backend_for_mock():
    from execution import build_driver
    from execution.driver import RobotDriver
    from execution.execution import KukaClient

    driver = build_driver({"robot": "mock"}, _exec_cfg())
    try:
        assert isinstance(driver, RobotDriver)
        assert isinstance(driver, KukaClient)
        # It talks to the simulator this factory spawned, not to the
        # `kuka:` host in the config.
        assert driver.host == "127.0.0.1"
        assert driver.port != 54600 and driver.port > 0
    finally:
        driver.close()


def test_build_driver_returns_the_kuka_backend_pointed_at_real_hardware():
    """`robot: real` builds the same class against the configured host.

    Nothing is connected here - there is no controller at 172.31.1.147 -
    so this asserts the wiring only.
    """
    from execution import build_driver
    from execution.execution import KukaClient

    cfg = _exec_cfg(kuka={"host": "172.31.1.147", "port": 54601})
    driver = build_driver({"robot": "real"}, cfg)
    try:
        assert isinstance(driver, KukaClient)
        assert (driver.host, driver.port) == ("172.31.1.147", 54601)
        # No simulator was spawned for a real controller.
        assert driver.teardown_count == 0
    finally:
        driver.close()


def test_build_driver_returns_the_json_backend_for_json():
    """The second backend is reachable from configuration.

    This is the whole point of the factory. ``JsonRobotDriver`` passes
    the same conformance suite as ``KukaClient`` and appeared in no
    non-test module, so the claim "encode/decode/send/recv is the whole
    vendor boundary" was measured by tests and used by nothing.
    """
    from execution import build_driver
    from execution.driver import RobotDriver
    from execution.json_driver import JsonRobotDriver

    driver = build_driver({"robot": "json"}, _exec_cfg())
    try:
        assert isinstance(driver, RobotDriver)
        assert isinstance(driver, JsonRobotDriver)
        assert driver.port > 0
    finally:
        driver.close()


def test_both_backends_get_the_same_policy_from_the_same_config():
    """One config block, one retry/timeout policy, two vendors.

    The policy is the vendor-neutral half of ``ExecutionConfig`` and
    ``DriverPolicy`` is where it lives; a factory that let the two
    backends drift apart on timeouts would make the conformance suite's
    shared-policy claim false in production.
    """
    from execution import build_driver

    cfg = _exec_cfg()
    cfg["kuka"].update({"handshake_timeout_ms": 1234,
                        "command_timeout_ms": 4321,
                        "max_retries": 2})
    kuka = build_driver({"robot": "mock"}, cfg)
    json_driver = build_driver({"robot": "json"}, cfg)
    try:
        assert kuka.policy == json_driver.policy
        assert kuka.policy.handshake_timeout_ms == 1234
        assert kuka.policy.command_timeout_ms == 4321
        assert kuka.policy.max_retries == 2
    finally:
        kuka.close()
        json_driver.close()


def test_build_driver_refuses_a_backend_it_does_not_have():
    from execution import build_driver

    with pytest.raises(ValueError, match="mode.robot"):
        build_driver({"robot": "universal-robots"}, _exec_cfg())


def test_closing_the_driver_stops_the_simulator_the_factory_spawned():
    """Whoever spawns the thread shuts it down.

    ``main.py`` used to hold the server handle and stop it in a
    ``finally``. That worked, but it made every future caller of the
    factory responsible for a thread it never asked for; the driver's
    own ``close()`` is the one event that happens on every route out of
    a run - normal exit, ``RobotEstop``, ``RobotFault`` and a failed
    handshake all reach it.
    """
    import socket

    from execution import build_driver

    driver = build_driver({"robot": "mock"}, _exec_cfg())
    host, port = driver.host, driver.port
    assert driver.teardown_count == 1

    with socket.create_connection((host, port), timeout=2.0) as probe:
        assert probe is not None          # the simulator is accepting

    driver.close()
    with pytest.raises(OSError):
        socket.create_connection((host, port), timeout=2.0).close()

    # Idempotent: close() is reached more than once on the fault paths.
    driver.close()


def test_build_task_config_reads_the_motion_block():
    """The application's numbers come out of the same parse as the
    driver's, so the two cannot drift.

    ``TaskConfig`` is vendor-neutral (``execution.task``) but its numbers
    live in the ``execution:`` YAML alongside the transport's, and
    ``ExecutionConfig.from_dict`` is the only thing that parses that
    file. ``main.py`` reaching for ``ExecutionConfig`` purely to get two
    floats is what put a vendor name in the pipeline.
    """
    from execution import build_task_config
    from execution.task import TaskConfig

    cfg = build_task_config(_exec_cfg())
    assert isinstance(cfg, TaskConfig)
    assert cfg.transport_height_mm == 75.0
    assert cfg.vacuum_level_percent == 70
