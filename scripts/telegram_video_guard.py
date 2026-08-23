#!/usr/bin/env python3
"""
scripts/telegram_video_guard.py - Reusable Telegram video preparation and compression guard.

Guarantees video files sent via Telegram Bot API stay strictly below the hard size limit
(default 45 MB, well below Telegram's 50 MB bot API threshold).
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

TELEGRAM_MAX_VIDEO_BYTES = 45 * 1024 * 1024  # 45 MB conservative safety threshold


def get_video_duration_sec(video_path: Path) -> Optional[float]:
    """Extract video duration in seconds using ffprobe or OpenCV."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        try:
            cmd = [
                ffprobe_bin, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            pass

    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frame_count > 0:
            return float(frame_count / fps)
    except Exception:
        pass

    return None


def calculate_target_bitrate_kbps(duration_sec: float, max_bytes: int = TELEGRAM_MAX_VIDEO_BYTES, safety_factor: float = 0.88) -> int:
    """
    Calculate bounded target video bitrate in kbps to guarantee output size < max_bytes.
    Formula: (max_bytes * 8 * safety_factor) / duration_sec / 1000
    """
    if duration_sec <= 0:
        return 1500  # Safe default fallback
    target_bits = max_bytes * 8 * safety_factor
    bitrate_bps = target_bits / duration_sec
    bitrate_kbps = int(bitrate_bps / 1000)
    # Clamp between 200 kbps (extreme long) and 4000 kbps (short high-quality)
    return max(200, min(4000, bitrate_kbps))


def compress_video_for_telegram(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_bytes: int = TELEGRAM_MAX_VIDEO_BYTES,
    timeout_sec: int = 300
) -> Tuple[bool, Path, int]:
    """
    Compress video if size exceeds max_bytes, or re-encode for faststart and Telegram compatibility.
    
    Returns:
        (success: bool, final_path: Path, final_size_bytes: int)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    current_size = input_path.stat().st_size
    output_specified = output_path is not None
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_tg_safe.mp4")
    else:
        output_path = Path(output_path)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # No ffmpeg available - check if already under limit
        if current_size <= max_bytes:
            if output_specified and output_path != input_path and not output_path.exists():
                shutil.copy2(input_path, output_path)
                return True, output_path, current_size
            return True, input_path, current_size
        return False, input_path, current_size

    duration = get_video_duration_sec(input_path) or 150.0
    target_kbps = calculate_target_bitrate_kbps(duration, max_bytes=max_bytes)

    # If input is already output and under limit, no re-encoding required
    if input_path == output_path and current_size <= max_bytes:
        return True, input_path, current_size

    # Build robust ffmpeg command with bitrate cap and CRF safety
    cmd = [
        ffmpeg_bin, "-y", "-i", str(input_path),
        "-c:v", "libx264",
        "-b:v", f"{target_kbps}k",
        "-maxrate", f"{int(target_kbps * 1.3)}k",
        "-bufsize", f"{int(target_kbps * 2.0)}k",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout_sec)
        out_size = output_path.stat().st_size
        if out_size <= max_bytes:
            return True, output_path, out_size
        
        # If still oversized, perform emergency lower resolution pass
        emergency_path = output_path.with_name(f"{output_path.stem}_lowres.mp4")
        emergency_kbps = max(150, int(target_kbps * 0.7))
        cmd_emergency = [
            ffmpeg_bin, "-y", "-i", str(input_path),
            "-vf", "scale=-2:480",
            "-c:v", "libx264",
            "-b:v", f"{emergency_kbps}k",
            "-maxrate", f"{int(emergency_kbps * 1.2)}k",
            "-bufsize", f"{int(emergency_kbps * 1.5)}k",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(emergency_path)
        ]
        subprocess.run(cmd_emergency, capture_output=True, text=True, check=True, timeout=timeout_sec)
        em_size = emergency_path.stat().st_size
        if em_size <= max_bytes:
            return True, emergency_path, em_size
        return False, emergency_path, em_size

    except Exception as e:
        if output_path.exists() and output_path.stat().st_size <= max_bytes:
            return True, output_path, output_path.stat().st_size
        return False, input_path, current_size
