import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.check_host_idle import check_host_idle
from scripts.lifecycle_manager import write_state, LifecycleState


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setenv("FAIR_FEEDER_STATE_FILE", str(state_file))
    yield state_file


def test_check_host_idle_returns_true_when_daytime_idle_and_healthy():
    write_state({
        "state": LifecycleState.DAYTIME_IDLE,
        "local_drain_complete": True,
        "services_active": False,
        "active_workers": 0
    })

    with patch("scripts.check_host_idle.is_service_active", return_value=False), \
         patch("scripts.check_host_idle.get_active_workers_count", return_value=0), \
         patch("scripts.check_host_idle.get_mem_available_mb", return_value=1650):

        res = check_host_idle()
        assert res["fair_feeder_idle"] is True
        assert res["ready_for_heavy_workload"] is True
        assert res["mem_available_mb"] == 1650
        assert res["mem_threshold_met"] is True


def test_check_host_idle_returns_true_in_evening_readiness_state():
    write_state({
        "state": LifecycleState.EVENING_READINESS,
        "local_drain_complete": True,
        "services_active": False,
        "active_workers": 0
    })

    with patch("scripts.check_host_idle.is_service_active", return_value=False), \
         patch("scripts.check_host_idle.get_active_workers_count", return_value=0), \
         patch("scripts.check_host_idle.get_mem_available_mb", return_value=1600):

        res = check_host_idle()
        assert res["fair_feeder_idle"] is True
        assert res["ready_for_heavy_workload"] is True


def test_check_host_idle_blocks_when_morning_active_or_services_running():
    # Case A: Morning active
    write_state({
        "state": LifecycleState.MORNING_ACTIVE,
        "local_drain_complete": False,
        "services_active": True,
        "active_workers": 0
    })

    with patch("scripts.check_host_idle.is_service_active", return_value=True), \
         patch("scripts.check_host_idle.get_active_workers_count", return_value=0), \
         patch("scripts.check_host_idle.get_mem_available_mb", return_value=600):

        res = check_host_idle()
        assert res["fair_feeder_idle"] is False
        assert res["ready_for_heavy_workload"] is False

    # Case B: State claims IDLE but worker is active
    write_state({
        "state": LifecycleState.DAYTIME_IDLE,
        "local_drain_complete": True,
        "services_active": False,
        "active_workers": 1
    })

    with patch("scripts.check_host_idle.is_service_active", return_value=False), \
         patch("scripts.check_host_idle.get_active_workers_count", return_value=1), \
         patch("scripts.check_host_idle.get_mem_available_mb", return_value=1400):

        res = check_host_idle()
        assert res["fair_feeder_idle"] is False
        assert res["ready_for_heavy_workload"] is False


def test_check_host_idle_blocks_when_ram_is_constrained():
    write_state({
        "state": LifecycleState.DAYTIME_IDLE,
        "local_drain_complete": True,
        "services_active": False,
        "active_workers": 0
    })

    with patch("scripts.check_host_idle.is_service_active", return_value=False), \
         patch("scripts.check_host_idle.get_active_workers_count", return_value=0), \
         patch("scripts.check_host_idle.get_mem_available_mb", return_value=800):  # Below 1200 MB threshold

        res = check_host_idle()
        assert res["fair_feeder_idle"] is True
        assert res["ready_for_heavy_workload"] is False
        assert res["mem_threshold_met"] is False
