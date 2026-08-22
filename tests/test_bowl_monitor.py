import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch("pathlib.Path.mkdir"):
    import motion_recorder


class _Box:
    def __init__(self, xyxy, cls=45):
        self.cls = [cls]
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    names = {45: "bowl"}

    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, frame, imgsz=640, conf=0.25, verbose=False):
        return [_Result([_Box(box) for box in self.boxes])]


class _Reader:
    def get_latest_frame(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)


class _MockController:
    def __init__(self, is_recording=False, cat_seen=False, last_cat_time=None):
        self.is_recording = is_recording
        self.cat_seen = cat_seen
        self.last_cat_time = last_cat_time
        self._continuation_requested = False


def test_bowl_is_ok_when_full_view_is_on_right_side():
    model = _Model([[129, 43, 188, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model)

    status = monitor._detect_bowls(np.zeros((100, 200, 3), dtype=np.uint8))

    assert status["ok"] is True
    assert status["visible_count"] == 1


def test_bowl_is_bad_when_bbox_is_clipped_by_frame_edge():
    model = _Model([[170, 43, 200, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model)

    status = monitor._detect_bowls(np.zeros((100, 200, 3), dtype=np.uint8))

    assert status["ok"] is False
    assert status["reason"] == "not fully visible"


def test_pi_notification_first_lines_use_requested_emojis():
    source = Path(motion_recorder.__file__).read_text(encoding="utf-8")

    assert "📷 Fair Feeder Monitor" in source
    assert "✅🥣 Bowl position recovered" in source
    assert "👀? Camera position alert" in source
    assert "🥣? Bowl not reliably visible" in source


def test_sustained_bowl_absence_idle_alerts():
    # Empty model = bowl not detected
    model = _Model([])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model)

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        with patch("time.time", return_value=t0):
            monitor.tick()
        assert len(alerts) == 0
        assert monitor._bad_since == t0
        assert monitor._alert_active is False

        # After 5 minutes (300s < 600s), still no alert
        with patch("time.time", return_value=t0 + 300.0):
            monitor.tick()
        assert len(alerts) == 0
        assert monitor._alert_active is False

        # After 10 minutes (601s >= 600s), alert fires
        with patch("time.time", return_value=t0 + 601.0):
            monitor.tick()
        assert len(alerts) == 1
        assert "🥣? Bowl not reliably visible" in alerts[0]
        assert "not detected for ~10 min" in alerts[0]
        assert monitor._alert_active is True


def test_active_feeding_suppresses_bowl_alert():
    model = _Model([])  # Bowl occluded by eating cat
    controller = _MockController(is_recording=True, cat_seen=True)
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model, controller=controller)

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        # Tick during active feeding
        with patch("time.time", return_value=t0):
            monitor.tick()
        assert monitor._bad_since is None  # Paused/reset
        assert len(alerts) == 0

        # Even after 15 minutes of feeding, no false bowl alert
        with patch("time.time", return_value=t0 + 900.0):
            monitor.tick()
        assert monitor._bad_since is None
        assert len(alerts) == 0
        assert monitor._alert_active is False


def test_one_good_frame_does_not_recover_alert():
    empty_model = _Model([])
    ok_model = _Model([[129, 43, 188, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), empty_model)

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        # Force alert active
        with patch("time.time", return_value=t0):
            monitor.tick()
        with patch("time.time", return_value=t0 + 601.0):
            monitor.tick()
        assert monitor._alert_active is True
        assert len(alerts) == 1

        # Now 1 good frame occurs
        monitor.yolo_model = ok_model
        with patch("time.time", return_value=t0 + 635.0):
            monitor.tick()

        # Must NOT send recovery after only 1 check
        assert len(alerts) == 1
        assert monitor._alert_active is True
        assert monitor._consecutive_good == 1


def test_stable_consecutive_good_checks_recovers_once():
    empty_model = _Model([])
    ok_model = _Model([[129, 43, 188, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), empty_model)

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        with patch("time.time", return_value=t0):
            monitor.tick()
        with patch("time.time", return_value=t0 + 601.0):
            monitor.tick()
        assert monitor._alert_active is True
        assert len(alerts) == 1

        # Switch to good bowl model
        monitor.yolo_model = ok_model

        # Good check 1
        with patch("time.time", return_value=t0 + 635.0):
            monitor.tick()
        assert monitor._consecutive_good == 1
        assert len(alerts) == 1

        # Good check 2
        with patch("time.time", return_value=t0 + 665.0):
            monitor.tick()
        assert monitor._consecutive_good == 2
        assert len(alerts) == 1

        # Good check 3 -> triggers recovery!
        with patch("time.time", return_value=t0 + 695.0):
            monitor.tick()
        assert monitor._consecutive_good == 3
        assert len(alerts) == 2
        assert "✅🥣 Bowl position recovered" in alerts[1]
        assert monitor._alert_active is False
        assert monitor._bad_since is None


def test_recovery_not_repeatedly_resent():
    ok_model = _Model([[129, 43, 188, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), ok_model)
    monitor._alert_active = False

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        # Subsequent good checks when alert was not active
        for i in range(10):
            with patch("time.time", return_value=t0 + i * 35.0):
                monitor.tick()

        assert len(alerts) == 0


def test_alert_cooldown_respected():
    empty_model = _Model([])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), empty_model)

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append), \
         patch("motion_recorder.CAMERA_TYPE", "rtsp"):

        t0 = 1000.0
        # Alert fires at t0 + 601s
        with patch("time.time", return_value=t0):
            monitor.tick()
        with patch("time.time", return_value=t0 + 601.0):
            monitor.tick()
        assert len(alerts) == 1

        # Another tick 1 minute later (bad is still active, but in cooldown)
        with patch("time.time", return_value=t0 + 661.0):
            monitor.tick()
        assert len(alerts) == 1  # No repeated alert
