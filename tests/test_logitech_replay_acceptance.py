import os
import time
import shutil
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import motion_recorder
from motion_recorder import RecordingController
from scripts.logitech_vlm_shadow import group_clips_into_sessions


def test_logitech_low_light_real_frame_retention(tmp_path, monkeypatch):
    """
    Verifies that low-light Logitech frames (mean < 15) trigger cat_seen
    retention on dedicated USB camera, preventing clip deletion.
    """
    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", tmp_path / "drive")
    (tmp_path / "temp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)

    real_video = Path("scratch/replay_acceptance/motion_20260817_061955_1m_58s.mp4")
    if real_video.exists():
        cap = cv2.VideoCapture(str(real_video))
        ret, dark_frame = cap.read()
        cap.release()
        assert ret is True
    else:
        dark_frame = np.full((720, 1280, 3), 3, dtype=np.uint8)

    mean_lum = np.mean(dark_frame)
    assert mean_lum < 15.0

    class MockUSBReader:
        def __init__(self, frame):
            self.frame = frame
            self.stream_fps = 10.0
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]
            self.is_usb = True
            self.source = 0
        def get_latest_frame(self):
            return self.frame
        def get_buffer_snapshot(self):
            return [self.frame]

    class MockListener:
        motion_detected = True
        last_motion_time = None
        def reset_background(self): pass

    mock_yolo = MagicMock()
    mock_yolo.names = {0: "cat"}
    mock_res = MagicMock()
    mock_res.boxes = []
    mock_yolo.return_value = [mock_res]

    reader = MockUSBReader(dark_frame)
    listener = MockListener()
    controller = RecordingController(reader, listener, yolo_model=mock_yolo)

    controller._start_recording()
    assert controller.is_recording is True

    # Check cat on dark frame -> low-light USB fallback activates
    controller._check_for_cat(dark_frame)
    assert controller.cat_seen is True

    # Write dummy content to temp_path
    controller.temp_path.write_bytes(b"test video data")

    # Stop recording -> background worker saves to Drive (does NOT delete)
    saved_name = f"{controller._base_name}_10s.mp4"
    controller._finalize_recording(
        controller.temp_path,
        saved_name,
        "10s",
        10.0,
        cat_seen=controller.cat_seen,
        declared_fps=10.0,
        frame_count=100
    )

    saved_file = (tmp_path / "drive") / saved_name
    assert saved_file.exists()


def test_logitech_downstream_event_segmentation():
    """
    Verifies that the downstream shadow pipeline correctly segments
    retained clips based on the 10-second gap boundary rule.
    """
    clips = [
        {"name": "motion_20260817_061955_1m_58s.mp4", "id": "f1"},
        {"name": "motion_20260817_062528_35s.mp4", "id": "f2"}
    ]

    # 3m35s gap -> 2 distinct feeding sessions
    sessions = group_clips_into_sessions(clips, gap_threshold_sec=10)
    assert len(sessions) == 2
    assert len(sessions[0]) == 1
    assert len(sessions[1]) == 1
    assert sessions[0][0]["name"] == "motion_20260817_061955_1m_58s.mp4"
    assert sessions[1][0]["name"] == "motion_20260817_062528_35s.mp4"
