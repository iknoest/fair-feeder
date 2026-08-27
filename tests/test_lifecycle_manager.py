import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.lifecycle_manager import (
    LifecycleState,
    read_state,
    write_state,
    activate_morning,
    drain_and_idle,
    check_drain_prerequisites,
    on_demand_capture,
    evening_readiness,
    get_active_workers_count
)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setenv("FAIR_FEEDER_STATE_FILE", str(state_file))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")
    yield state_file


def test_initial_state_and_state_persistence(tmp_path):
    state = read_state()
    assert state["state"] == LifecycleState.UNKNOWN
    assert state["local_drain_complete"] is False

    state["state"] = LifecycleState.DAYTIME_IDLE
    state["local_drain_complete"] = True
    write_state(state)

    reloaded = read_state()
    assert reloaded["state"] == LifecycleState.DAYTIME_IDLE
    assert reloaded["local_drain_complete"] is True
    assert "updated_at" in reloaded
    assert "mem_available_mb" in reloaded


def test_activate_morning_starts_services_and_updates_state():
    with patch("subprocess.run") as mock_run, \
         patch("scripts.lifecycle_manager.is_service_active", return_value=True), \
         patch("shutil.which", return_value="/bin/systemctl"), \
         patch("time.sleep"):

        mock_run.return_value = MagicMock(returncode=0)
        success = activate_morning()
        assert success is True

        state = read_state()
        assert state["state"] == LifecycleState.MORNING_ACTIVE
        assert state["services_active"] is True
        assert state["local_drain_complete"] is False


def test_activate_morning_fails_loudly_on_service_error():
    alerts = []
    with patch("subprocess.run") as mock_run, \
         patch("scripts.lifecycle_manager.send_telegram_alert", side_effect=alerts.append), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("shutil.which", return_value="/bin/systemctl"), \
         patch("time.sleep"):

        mock_run.return_value = MagicMock(returncode=1, stderr="Failed to connect to bus")
        success = activate_morning()
        assert success is False
        assert len(alerts) == 1
        assert "🚨 Fair Feeder Morning Activation FAILED" in alerts[0]

        state = read_state()
        assert state["services_active"] is False


def test_drain_prerequisites_blocks_when_workers_or_files_active(tmp_path):
    temp1 = tmp_path / "recordings_temp"
    temp2 = tmp_path / "recordings_usb_temp"
    temp1.mkdir()
    temp2.mkdir()

    # Case A: active temp MP4 file
    (temp1 / "motion_recording.mp4").write_bytes(b"data")
    with patch("scripts.lifecycle_manager.get_active_workers_count", return_value=0):
        drained, reason = check_drain_prerequisites([temp1, temp2])
        assert drained is False
        assert "unfinished file(s)" in reason

    # Clean up file
    (temp1 / "motion_recording.mp4").unlink()

    # Case B: active ffmpeg/rclone child worker
    with patch("scripts.lifecycle_manager.get_active_workers_count", return_value=2):
        drained, reason = check_drain_prerequisites([temp1, temp2])
        assert drained is False
        assert "2 background ffmpeg/rclone worker(s)" in reason

    # Case C: clean
    with patch("scripts.lifecycle_manager.get_active_workers_count", return_value=0):
        drained, reason = check_drain_prerequisites([temp1, temp2])
        assert drained is True
        assert "complete" in reason


def test_drain_and_idle_stops_services_and_enters_daytime_idle():
    with patch("scripts.lifecycle_manager.check_drain_prerequisites", return_value=(True, "all clear")), \
         patch("subprocess.run") as mock_run, \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1750), \
         patch("time.sleep"):

        mock_run.return_value = MagicMock(returncode=0)
        success = drain_and_idle(max_wait_sec=10)
        assert success is True

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["local_drain_complete"] is True
        assert state["source_evidence_ready"] is True
        assert state["services_active"] is False
        assert state["active_workers"] == 0
        assert "MemAvailable: 1750 MB" in state["last_event"]


def test_on_demand_capture_executes_and_returns_to_idle(tmp_path):
    # Set initial state to DAYTIME_IDLE
    init_state = read_state()
    init_state["state"] = LifecycleState.DAYTIME_IDLE
    init_state["local_drain_complete"] = True
    write_state(init_state)

    out_file = tmp_path / "test_capture.mp4"

    class MockCap:
        def __init__(self, *args, **kwargs):
            self._frames = 0
        def isOpened(self): return True
        def get(self, prop): return 30.0
        def read(self):
            self._frames += 1
            if self._frames > 5:
                return False, None
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        def release(self): pass

    with patch("cv2.VideoCapture", return_value=MockCap()), \
         patch("subprocess.run") as mock_ffmpeg, \
         patch.dict(os.environ, {"RTSP_URL": "rtsp://mock:554/stream"}), \
         patch("scripts.lifecycle_manager.send_telegram_alert"):

        mock_ffmpeg.return_value = MagicMock(returncode=0)

        # Mock ffmpeg creating final file
        def fake_ffmpeg(*args, **kwargs):
            out_file.write_bytes(b"mp4data")
            return MagicMock(returncode=0)
        mock_ffmpeg.side_effect = fake_ffmpeg

        res = on_demand_capture(camera="tapo", duration_sec=1, out_path=str(out_file), send_telegram=False)
        assert res == str(out_file)
        assert out_file.exists()

        # Verify state returned to DAYTIME_IDLE
        after_state = read_state()
        assert after_state["state"] == LifecycleState.DAYTIME_IDLE


def test_evening_readiness_healthy_returns_ready(tmp_path):
    class MockCap:
        def isOpened(self): return True
        def read(self): return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        def release(self): pass

    with patch("cv2.VideoCapture", return_value=MockCap()), \
         patch("shutil.disk_usage", return_value=MagicMock(free=5 * 1024**3)), \
         patch.dict(os.environ, {"RTSP_URL": "rtsp://fake:554/stream1", "V4L2_DEVICE": "/dev/video0"}):

        is_ready, results = evening_readiness(send_telegram_on_failure=True)
        assert is_ready is True
        assert results["tapo_reachable"] is True
        assert results["storage_ok"] is True
        assert len(results["issues"]) == 0

        state = read_state()
        assert state["state"] == LifecycleState.EVENING_READINESS
        assert state["readiness"]["ready"] is True


def test_evening_readiness_failure_sends_actionable_alert(tmp_path):
    alerts = []
    with patch("cv2.VideoCapture", side_effect=Exception("Connection refused")), \
         patch("shutil.disk_usage", return_value=MagicMock(free=500 * 1024**2)), \
         patch.dict(os.environ, {"RTSP_URL": "rtsp://fake:554/stream1", "V4L2_DEVICE": "/dev/video0"}), \
         patch("scripts.lifecycle_manager.send_telegram_alert", side_effect=alerts.append):

        is_ready, results = evening_readiness(send_telegram_on_failure=True)
        assert is_ready is False
        assert len(results["issues"]) >= 2
        assert len(alerts) == 1
        assert "⚠️ Fair Feeder Evening Readiness Alert" in alerts[0]
        assert "Action needed before tomorrow's breakfast" in alerts[0]
