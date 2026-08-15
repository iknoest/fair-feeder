import os
import sys
import time
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import motion_recorder
from motion_recorder import (
    acquire_process_lock,
    FrameMotionDetector,
    RecordingController,
    MAX_RECORDING_SECS,
    COOLDOWN_SECONDS,
)

# ── Process Lock Tests (Cases 4 & 5) ──────────────────────────────────────────

def test_process_lock_first_attempt_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    lock_file = acquire_process_lock("rtsp")
    assert lock_file is not None
    lock_file.close()

def test_process_lock_same_camera_second_attempt_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    lock1 = acquire_process_lock("rtsp")
    assert lock1 is not None
    
    # Second attempt for same camera ('rtsp')
    lock2 = acquire_process_lock("rtsp")
    assert lock2 is None
    
    lock1.close()

def test_process_lock_different_camera_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    lock_tapo = acquire_process_lock("rtsp")
    assert lock_tapo is not None
    
    # Different camera ('usb')
    lock_usb = acquire_process_lock("usb")
    assert lock_usb is not None
    
    lock_tapo.close()
    lock_usb.close()

def test_process_lock_becomes_available_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    lock1 = acquire_process_lock("rtsp")
    assert lock1 is not None
    lock1.close()  # OS releases lock
    
    # Now second attempt succeeds
    lock2 = acquire_process_lock("rtsp")
    assert lock2 is not None
    lock2.close()


# ── Continuation Fix Tests (Cases 1, 2, 3) ────────────────────────────────────

class MockReader:
    def __init__(self, fps=15):
        self.stream_fps = fps
        self.frame_width = 320
        self.frame_height = 180
    def get_latest_frame(self):
        return np.zeros((180, 320, 3), dtype=np.uint8)
    def get_buffer_snapshot(self):
        return [np.zeros((180, 320, 3), dtype=np.uint8)]

class MockListener:
    def __init__(self, motion_detected=False):
        self.motion_detected = motion_detected
        self.last_motion_time = None
        self.reset_called = False
    def reset_background(self):
        self.reset_called = True


def test_case_1_max_duration_boundary_starts_continuation_when_cat_present(tmp_path, monkeypatch):
    """
    CASE 1: Session active (cat_seen=True). Reaches 150s boundary with low motion.
    Old behavior: went idle.
    New behavior: resets background and immediately starts continuation recording.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)  # MOG2 motion has dropped to low/False!
    controller = RecordingController(reader, listener, yolo_model=None)
    
    # Simulate active recording session where cat was seen
    controller._start_recording()
    controller.cat_seen = True
    controller.recording_start = time.time() - (MAX_RECORDING_SECS + 1)  # past 150s
    
    with patch.object(controller, "_stop_recording", wraps=controller._stop_recording) as mock_stop:
        controller.tick()  # Triggers max-duration check
        assert mock_stop.called

    assert listener.reset_called is True
    assert controller._continuation_requested is True
    
    # Next tick: even though listener.motion_detected is False, continuation starts!
    controller.tick()
    assert controller.is_recording is True
    assert controller._continuation_requested is False


def test_case_2_continuation_eventually_returns_to_idle_after_session_ends(tmp_path, monkeypatch):
    """
    CASE 2: Continuation clip starts, cat leaves and motion stops.
    After COOLDOWN_SECONDS of no motion, recording stops cleanly and returns to idle.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    controller = RecordingController(reader, listener, yolo_model=None)
    
    # Request continuation and start clip
    controller._continuation_requested = True
    controller.tick()
    assert controller.is_recording is True

    # Simulate cat leaving: cat_seen is False, motion is False past cooldown
    controller.cat_seen = False
    controller._last_motion_ts = time.time() - (COOLDOWN_SECONDS + 1)
    
    controller.tick()  # Triggers cooldown stop
    assert controller.is_recording is False
    assert controller._continuation_requested is False  # Does NOT trigger another continuation!


def test_case_3_no_cat_false_motion_does_not_create_endless_continuation(tmp_path, monkeypatch):
    """
    CASE 3: False motion (cat_seen=False) reaches 150s max duration.
    Must NOT set continuation_requested; clip deleted, recorder returns to idle.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    # Model present, but cat_seen remains False
    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}
    
    controller = RecordingController(reader, listener, yolo_model=mock_yolo)
    
    # Start recording for false motion
    controller._start_recording()
    controller.cat_seen = False
    controller.recording_start = time.time() - (MAX_RECORDING_SECS + 1)
    
    controller.tick()  # Max duration reached
    
    assert controller.is_recording is False
    assert listener.reset_called is False
    assert controller._continuation_requested is False
    
    # Next tick: stays idle!
    controller.tick()
    assert controller.is_recording is False
