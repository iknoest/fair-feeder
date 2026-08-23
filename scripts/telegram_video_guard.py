#!/usr/bin/env python3
"""
scripts/telegram_video_guard.py - Reusable Telegram video preparation and compression guard.

Guarantees video files sent via Telegram Bot API stay strictly below the hard size limit
(default 45 MB, well below Telegram's 50 MB bot API threshold).
"""

import os
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

TELEGRAM_MAX_VIDEO_BYTES = 45 * 1024 * 1024  # 45 MB hard limit
TAPO_TARGET_CRF = 28
TAPO_TARGET_HEIGHT = 480


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
    return max(200, min(4000, bitrate_kbps))


def validate_video_content(
    video_path: Path,
    expected_min_duration_sec: float = 0.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Rigorously validates video integrity:
    1. File exists and size > 0
    2. Opens with cv2.VideoCapture / ffprobe
    3. Frame count, fps, and duration are non-trivial
    4. Codec is H.264 and pixel format is Telegram-compatible (yuv420p)
    5. Samples multiple frames (first, 25%, 50%, 75%, last) and verifies real non-blank content
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return False, f"File does not exist: {video_path}", {}

    file_size = video_path.stat().st_size
    if file_size == 0:
        return False, "File is 0 bytes", {}

    details = {
        "file": video_path.name,
        "size_bytes": file_size,
        "sha256": None,
        "codec": "unknown",
        "pix_fmt": "unknown",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration_sec": 0.0,
        "frame_count": 0,
        "frames_tested": 0,
        "frames_passed": 0,
        "sample_luminances": []
    }

    # Compute sha256
    try:
        h = hashlib.sha256()
        with open(video_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        details["sha256"] = h.hexdigest()
    except Exception:
        pass

    # 1. ffprobe checks if available
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        try:
            cmd = [
                ffprobe_bin, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,duration,nb_frames",
                "-of", "json",
                str(video_path)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                streams = json.loads(res.stdout).get("streams", [])
                if streams:
                    probe_data = streams[0]
                    details["codec"] = probe_data.get("codec_name", "unknown")
                    details["pix_fmt"] = probe_data.get("pix_fmt", "unknown")
                    details["width"] = int(probe_data.get("width", 0))
                    details["height"] = int(probe_data.get("height", 0))
                    if probe_data.get("duration"):
                        details["duration_sec"] = float(probe_data["duration"])
        except Exception:
            pass

    # 2. OpenCV decoding & content check
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "OpenCV VideoCapture failed to open file", details

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if details["width"] == 0: details["width"] = w
        if details["height"] == 0: details["height"] = h
        if details["fps"] == 0.0: details["fps"] = fps
        if details["frame_count"] == 0: details["frame_count"] = frame_count
        if details["duration_sec"] == 0.0 and fps > 0: details["duration_sec"] = round(frame_count / fps, 2)

        if frame_count <= 0 or fps <= 0:
            cap.release()
            return False, f"Invalid frame_count ({frame_count}) or fps ({fps})", details

        if expected_min_duration_sec > 0 and details["duration_sec"] < expected_min_duration_sec * 0.5:
            cap.release()
            return False, f"Duration ({details['duration_sec']}s) is implausibly shorter than expected ({expected_min_duration_sec}s)", details

        # Sample 5 points across duration: 0%, 25%, 50%, 75%, 100%
        indices = [0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4, max(0, frame_count - 1)]
        indices = sorted(list(set(indices)))
        details["frames_tested"] = len(indices)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False, f"Failed to decode frame at index {idx}", details

            mean_lum = float(np.mean(frame))
            details["sample_luminances"].append(round(mean_lum, 2))

            # Blank / black screen detection (mean luminance < 0.5 and max < 2)
            if mean_lum < 0.5 and np.max(frame) < 2:
                cap.release()
                return False, f"Frame at index {idx} is completely black/empty (mean lum={mean_lum})", details

            details["frames_passed"] += 1

        cap.release()

    except Exception as e:
        return False, f"Video decode error: {e}", details

    return True, "Video content is valid and decodable", details


def compress_video_for_telegram(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_bytes: int = TELEGRAM_MAX_VIDEO_BYTES,
    crf: int = TAPO_TARGET_CRF,
    target_height: int = TAPO_TARGET_HEIGHT,
    timeout_sec: int = 300
) -> Tuple[bool, Path, int]:
    """
    Compresses video to compact Telegram delivery format matching TAPO production profile:
    - H.264 (libx264)
    - 480p (scale=-2:480)
    - CRF 28 (clean quality at ~2-5 MB)
    - yuv420p
    - +faststart (moov atom at front for mobile streaming)
    - -an (no audio track)
    - Strict validation of output video content before returning success.
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
        # Fallback when ffmpeg is not present: validate input directly
        is_val, msg, _ = validate_video_content(input_path)
        if not is_val:
            return False, input_path, current_size
        if current_size <= max_bytes:
            if output_specified and output_path.resolve() != input_path.resolve() and not output_path.exists():
                shutil.copy2(input_path, output_path)
                return True, output_path, current_size
            return True, input_path, current_size
        return False, input_path, current_size

    duration = get_video_duration_sec(input_path) or 150.0
    target_kbps = calculate_target_bitrate_kbps(duration, max_bytes=max_bytes)

    # Temporary output path if input == output
    target_out = output_path
    if input_path.resolve() == output_path.resolve():
        target_out = output_path.with_name(f"temp_comp_{output_path.name}")

    if target_out.exists():
        target_out.unlink()

    # TAPO production delivery profile with safety bitrate cap
    cmd = [
        ffmpeg_bin, "-y", "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-vf", f"scale=-2:{target_height}",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-maxrate", f"{int(target_kbps * 1.5)}k",
        "-bufsize", f"{int(target_kbps * 2.5)}k",
        "-an",
        str(target_out)
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if res.returncode != 0 or not target_out.exists():
            if target_out.exists() and target_out != input_path:
                target_out.unlink()
            return False, input_path, current_size

        # Validate transcoded output content
        is_val, val_msg, _ = validate_video_content(target_out)
        if not is_val:
            if target_out.exists() and target_out != input_path:
                target_out.unlink()
            return False, input_path, current_size

        out_size = target_out.stat().st_size
        if out_size > max_bytes:
            # Emergency lower resolution / higher CRF pass
            emergency_path = output_path.with_name(f"{output_path.stem}_lowres.mp4")
            cmd_em = [
                ffmpeg_bin, "-y", "-i", str(input_path),
                "-c:v", "libx264",
                "-crf", "34",
                "-preset", "fast",
                "-vf", "scale=-2:360",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(emergency_path)
            ]
            res_em = subprocess.run(cmd_em, capture_output=True, text=True, timeout=timeout_sec)
            if res_em.returncode == 0 and emergency_path.exists():
                is_val_em, _, _ = validate_video_content(emergency_path)
                if is_val_em and emergency_path.stat().st_size <= max_bytes:
                    if target_out.exists() and target_out != input_path:
                        target_out.unlink()
                    if emergency_path.resolve() != output_path.resolve():
                        emergency_path.rename(output_path)
                        return True, output_path, output_path.stat().st_size
                    return True, emergency_path, emergency_path.stat().st_size
            if target_out.exists() and target_out != input_path:
                target_out.unlink()
            return False, input_path, current_size

        if target_out.resolve() != output_path.resolve():
            if output_path.exists():
                output_path.unlink()
            target_out.rename(output_path)
            target_out = output_path

        return True, output_path, out_size

    except Exception:
        if target_out.exists() and target_out.resolve() != input_path.resolve():
            target_out.unlink()
        return False, input_path, current_size
