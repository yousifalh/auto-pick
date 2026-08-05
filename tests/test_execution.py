"""Round-trip execution tests using the mock KUKA simulator.

Asserts that the full command pipeline (pack → TCP → mock robot →
status → unpack) works for every high-level command the planner emits.
"""
from __future__ import annotations

import time
import pytest

from common.types import PickPlacePose, RobotStatusCode, WorkspacePoint
from execution.execution import ExecutionConfig, KukaClient
from execution.mock_kuka_server import run_in_thread


@pytest.fixture
def mock_server():
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=0.0, ms_per_100mm=5)
    port = srv.server_address[1]
    yield port
    srv.shutdown()


def test_handshake_via_context_manager(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server,
                          handshake_timeout_ms=1000)
    with KukaClient(cfg):
        pass  # enter + exit must not raise


def test_move_to_returns_success(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        s = k.move_to(WorkspacePoint(100, 50, 80))
        assert s.code in (RobotStatusCode.SUCCESS, RobotStatusCode.OK)


def test_vacuum_on_off(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    with KukaClient(cfg) as k:
        s = k.vacuum(True)
        assert s.code == RobotStatusCode.SUCCESS
        s = k.vacuum(False)
        assert s.code == RobotStatusCode.SUCCESS


def test_pick_and_place_succeeds(mock_server):
    cfg = ExecutionConfig(host="127.0.0.1", port=mock_server)
    pose = PickPlacePose(
        pick=WorkspacePoint(100.0, 50.0, 30.0),
        place=WorkspacePoint(-50.0, 100.0, 5.0),
        cartridge_id=0, grid_row=1, grid_col=2,
    )
    with KukaClient(cfg) as k:
        status = k.pick_and_place(pose)
        assert status.code == RobotStatusCode.SUCCESS
        assert status.cycle_time_ms >= 0
        # Robot should have ended near the place target
        assert abs(status.current_pose.x_mm - (-50)) <= 1
        assert abs(status.current_pose.y_mm - 100) <= 1


def test_pick_failure_reported():
    """drop_prob=1.0 forces every pick to fail."""
    srv, _t = run_in_thread(host="127.0.0.1", port=0,
                            drop_prob=1.0, ms_per_100mm=2)
    port = srv.server_address[1]
    try:
        cfg = ExecutionConfig(host="127.0.0.1", port=port)
        pose = PickPlacePose(
            pick=WorkspacePoint(10, 10, 5),
            place=WorkspacePoint(20, 20, 5),
            cartridge_id=0, grid_row=0, grid_col=0,
        )
        with KukaClient(cfg) as k:
            status = k.pick_and_place(pose)
            assert status.code == RobotStatusCode.PICK_FAILED
    finally:
        srv.shutdown()


def test_execution_config_from_dict():
    cfg = ExecutionConfig.from_dict({
        "kuka": {"host": "10.0.0.1", "port": 1234, "max_retries": 7},
        "motion": {"approach_height_mm": 55.0, "vacuum_level_percent": 95},
    })
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 1234
    assert cfg.max_retries == 7
    assert cfg.approach_height_mm == 55.0
    assert cfg.vacuum_level_percent == 95


def test_execution_config_defaults():
    cfg = ExecutionConfig.from_dict({})
    assert cfg.host == "172.31.1.147"
    assert cfg.port == 54600
    assert cfg.approach_height_mm == 60.0
