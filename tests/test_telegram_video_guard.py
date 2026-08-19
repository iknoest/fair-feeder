import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.telegram_video_guard import (
    TELEGRAM_MAX_VIDEO_BYTES,
    calculate_target_bitrate_kbps,
    get_video_duration_sec,
    compress_video_for_telegram
)


def test_telegram_max_video_bytes_is_conservative():
    assert TELEGRAM_MAX_VIDEO_BYTES == 45 * 1024 * 1024
    assert TELEGRAM_MAX_VIDEO_BYTES < 50 * 1024 * 1024  # Under Telegram 50MB hard limit


def test_calculate_target_bitrate_kbps():
    # 150 seconds video: (45*1024*1024 * 8 * 0.88) / 150 / 1000 ~= 2214 kbps
    br_150 = calculate_target_bitrate_kbps(150.0)
    assert 1800 <= br_150 <= 2500

    # 300 seconds video (5 min):
    br_300 = calculate_target_bitrate_kbps(300.0)
    assert 900 <= br_300 <= 1300

    # Edge cases
    br_0 = calculate_target_bitrate_kbps(0.0)
    assert br_0 == 1500

    # Very long video clamped to minimum 200 kbps
    br_long = calculate_target_bitrate_kbps(10000.0)
    assert br_long == 200

    # Very short video clamped to maximum 4000 kbps
    br_short = calculate_target_bitrate_kbps(5.0)
    assert br_short == 4000


def test_compress_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        compress_video_for_telegram(Path("/nonexistent/video.mp4"))


def test_compress_video_already_under_limit_no_ffmpeg():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"0" * 1024)
        tf_path = Path(tf.name)

    try:
        with patch("shutil.which", return_value=None):
            ok, out_path, size = compress_video_for_telegram(tf_path)
            assert ok is True
            assert out_path == tf_path
            assert size == 1024
    finally:
        if tf_path.exists():
            tf_path.unlink()


def test_compress_video_oversized_no_ffmpeg():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"0" * (46 * 1024 * 1024))
        tf_path = Path(tf.name)

    try:
        with patch("shutil.which", return_value=None):
            ok, out_path, size = compress_video_for_telegram(tf_path)
            assert ok is False
            assert size == 46 * 1024 * 1024
    finally:
        if tf_path.exists():
            tf_path.unlink()


def test_compress_video_ffmpeg_success():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"0" * (55 * 1024 * 1024))  # 55 MB oversized
        tf_path = Path(tf.name)

    out_target = tf_path.with_name("compressed_out.mp4")

    try:
        def mock_subprocess_run(cmd, *args, **kwargs):
            # Create the output file with safe 20 MB size
            out_file = Path(cmd[-1])
            with open(out_file, "wb") as f:
                f.write(b"0" * (20 * 1024 * 1024))
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("scripts.telegram_video_guard.get_video_duration_sec", return_value=170.0), \
             patch("subprocess.run", side_effect=mock_subprocess_run):
            
            ok, out_path, size = compress_video_for_telegram(tf_path, output_path=out_target)
            assert ok is True
            assert out_path == out_target
            assert size == 20 * 1024 * 1024
            assert size < TELEGRAM_MAX_VIDEO_BYTES
    finally:
        if tf_path.exists(): tf_path.unlink()
        if out_target.exists(): out_target.unlink()
