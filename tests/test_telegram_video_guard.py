import os
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.telegram_video_guard import (
    TELEGRAM_MAX_VIDEO_BYTES,
    calculate_target_bitrate_kbps,
    get_video_duration_sec,
    validate_video_content,
    compress_video_for_telegram
)


def test_telegram_max_video_bytes_is_conservative():
    assert TELEGRAM_MAX_VIDEO_BYTES == 45 * 1024 * 1024
    assert TELEGRAM_MAX_VIDEO_BYTES < 50 * 1024 * 1024


def test_calculate_target_bitrate_kbps():
    br_150 = calculate_target_bitrate_kbps(150.0)
    assert 1800 <= br_150 <= 2500

    br_300 = calculate_target_bitrate_kbps(300.0)
    assert 900 <= br_300 <= 1300

    br_0 = calculate_target_bitrate_kbps(0.0)
    assert br_0 == 1500

    br_long = calculate_target_bitrate_kbps(10000.0)
    assert br_long == 200

    br_short = calculate_target_bitrate_kbps(5.0)
    assert br_short == 4000


def test_compress_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        compress_video_for_telegram(Path("/nonexistent/video.mp4"))


def test_validate_video_content_valid_synthetic_video(tmp_path):
    vid_path = tmp_path / "valid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
    writer.release()

    is_val, msg, details = validate_video_content(vid_path)
    assert is_val is True
    assert details["frame_count"] == 15
    assert details["frames_passed"] == 5
    assert details["width"] == 320
    assert details["height"] == 240


def test_validate_video_content_blank_video_fails(tmp_path):
    vid_path = tmp_path / "blank.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (320, 240))
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))  # Solid black
    writer.release()

    is_val, msg, details = validate_video_content(vid_path)
    assert is_val is False
    assert "black/empty" in msg


def test_validate_video_content_zero_bytes(tmp_path):
    vid_path = tmp_path / "zero.mp4"
    vid_path.write_bytes(b"")
    is_val, msg, details = validate_video_content(vid_path)
    assert is_val is False
    assert "0 bytes" in msg


def test_compress_video_real_ffmpeg_tapo_profile(tmp_path):
    raw_path = tmp_path / "raw.mp4"
    out_target = tmp_path / "tapo_compressed.mp4"

    # Create raw 720p 20-frame video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(raw_path), fourcc, 10.0, (1280, 720))
    for i in range(20):
        # Create non-blank pattern
        frame = np.full((720, 1280, 3), 30 + i * 2, dtype=np.uint8)
        cv2.circle(frame, (640, 360), 50, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    ok, final_p, size_b = compress_video_for_telegram(raw_path, output_path=out_target, crf=28, target_height=480)
    assert ok is True
    assert final_p.exists()
    assert size_b > 0
    assert size_b < 5 * 1024 * 1024  # Highly compact (few KB for 20 frames)

    # Validate output geometry (height is scaled to 480p)
    cap = cv2.VideoCapture(str(final_p))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert h == 480


def test_compress_video_ffmpeg_failure_returns_false(tmp_path):
    raw_path = tmp_path / "raw.mp4"
    raw_path.write_bytes(b"invalid_video_data")
    out_target = tmp_path / "out.mp4"

    ok, final_p, size_b = compress_video_for_telegram(raw_path, output_path=out_target)
    assert ok is False
    assert final_p == raw_path
