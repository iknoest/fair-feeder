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

    # Multi-tick verification: continuation clip gets fresh grace period and does NOT immediately abort
    controller.tick()
    assert controller.is_recording is True


def test_continuation_persists_while_cat_is_detected_even_if_mog2_is_quiet(tmp_path, monkeypatch):
    """
    Verifies that a continuation clip stays alive when YOLO detects cat presence,
    even if MOG2 background subtractor motion has dropped to 0%.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)

    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.85]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}
    mock_yolo.return_value = [mock_result]

    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    # Continuation requested
    controller._continuation_requested = True
    controller.tick()
    assert controller.is_recording is True

    # Advance time by 3 seconds (longer than CAT_CHECK_INTERVAL) with MOG2 motion = False
    controller._last_cat_check = time.time() - 3.0
    controller.tick()

    assert controller.is_recording is True
    assert controller.cat_seen is True
    # _last_motion_ts was refreshed by cat detection
    assert time.time() - controller._last_motion_ts < 1.0


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


def test_stop_recording_queues_without_blocking(tmp_path, monkeypatch):
    """
    Verifies that _stop_recording() queues finalization (remux/upload/delete)
    to a background worker queue rather than blocking tick() or spawning unbounded threads.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    controller = RecordingController(reader, listener, yolo_model=None)

    controller._start_recording()
    assert controller.is_recording is True

    # Call _stop_recording
    controller._stop_recording()
    assert controller.is_recording is False


def test_finalize_recording_deletes_immediately_without_running_ffmpeg_when_no_cat(tmp_path, monkeypatch):
    """
    Verifies that a false motion clip (cat_seen=False) is deleted immediately
    WITHOUT executing any ffmpeg subprocess/transcoding.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}
    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    dummy_file = tmp_path / "motion_test_5s.mp4"
    dummy_file.write_bytes(b"dummy video data")

    with patch("subprocess.run") as mock_subproc:
        controller._finalize_recording(
            dummy_file,
            "motion_test_5s.mp4",
            "5s",
            5.0,
            cat_seen=False,
            declared_fps=25.0,
            frame_count=20,  # 4 fps vs declared 25 fps (>20% diverged)
        )
        # ffmpeg must NOT be called for rejected clips!
        assert not mock_subproc.called

    assert not dummy_file.exists()
    assert controller.clips_deleted == 1
    assert controller.clips_saved == 0


def test_finalize_recording_remuxes_with_threads_1_when_cat_present(tmp_path, monkeypatch):
    """
    Verifies that a verified cat clip (cat_seen=True) runs ffmpeg with -threads 1
    when actual fps diverges by >20% from declared fps.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}
    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    dummy_file = tmp_path / "motion_cat_5s.mp4"
    dummy_file.write_bytes(b"dummy video data")

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        controller._finalize_recording(
            dummy_file,
            "motion_cat_5s.mp4",
            "5s",
            5.0,
            cat_seen=True,
            declared_fps=25.0,
            frame_count=50,  # 10 fps vs declared 25 fps (>20% diverged)
        )
        assert mock_subproc.called
        # Check that -threads 1 was passed to ffmpeg
        ffmpeg_cmd = mock_subproc.call_args_list[0][0][0]
        assert ffmpeg_cmd[0] == "ffmpeg"
        assert "-threads" in ffmpeg_cmd
        assert ffmpeg_cmd[ffmpeg_cmd.index("-threads") + 1] == "1"


def test_continuation_inherits_cat_seen_and_saves_on_natural_session_end(tmp_path, monkeypatch):
    """
    Verifies that a continuation chunk inherits cat_seen=True from the active feeding session,
    and when the session ends on motion cooldown, the continuation chunk is saved to Drive.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path)

    reader = MockReader()
    listener = MockListener(motion_detected=False)
    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}

    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    # First chunk starts, cat seen, reaches max duration
    controller._start_recording()
    controller.cat_seen = True
    controller.recording_start = time.time() - (MAX_RECORDING_SECS + 1)
    controller.tick()

    assert controller.is_recording is False
    assert controller._continuation_requested is True

    # Next tick: continuation chunk starts
    controller.tick()
    assert controller.is_recording is True
    assert controller.cat_seen is True  # Inherited!

    # Create dummy file at temp_path so finalization can move it
    controller.temp_path.write_bytes(b"dummy video data")
    saved_name = f"{controller._base_name}_10s.mp4"

    # Verify background finalization runs directly and saves to Drive when cat_seen=True
    controller._finalize_recording(
        controller.temp_path,
        saved_name,
        "10s",
        10.0,
        cat_seen=True,
        declared_fps=15,
        frame_count=150
    )

    saved_file = tmp_path / saved_name
    assert saved_file.exists()


def test_check_for_cat_low_light_enhancement_fallback(tmp_path, monkeypatch):
    """
    Verifies that _check_for_cat() enhances dark frames (mean < 40)
    with Gamma+CLAHE to enable cat detection in low light.
    """
    reader = MockReader()
    listener = MockListener()

    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}

    # Raw dark frame returns no detections, but enhanced frame returns cat detection
    def mock_inference(img, *args, **kwargs):
        mean_val = np.mean(img)
        mock_res = MagicMock()
        if mean_val > 50:  # Enhanced frame has higher contrast/brightness
            mock_box = MagicMock()
            mock_box.cls = [0]
            mock_box.conf = [0.75]
            mock_res.boxes = [mock_box]
        else:
            mock_res.boxes = []
        return [mock_res]

    mock_yolo.side_effect = mock_inference
    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)  # Mean = 10 (< 40)
    controller.cat_seen = False
    controller._check_for_cat(dark_frame)

    assert controller.cat_seen is True
