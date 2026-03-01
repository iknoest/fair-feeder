"""
test_motion_pi.py — Raspberry Pi motion recorder (no ONVIF dependency).

Replaces ONVIF PullPoint events with frame-based MOG2 background subtraction.
This is a self-contained test script to validate:
  1. RTSP stream connectivity
  2. Frame-based motion detection (MOG2)
  3. Video recording with pre-buffer
  4. rclone bisync to Google Drive

Usage:
  python test_motion_pi.py

  Or with env vars:
  TAPO_IP=192.168.1.246 TAPO_USER=myuser TAPO_PASS=mypass python test_motion_pi.py
"""

import os
import sys
import time
import platform
import subprocess
import shutil
import threading
import logging
import cv2
from pathlib import Path
from datetime import datetime
from collections import deque
from urllib.parse import quote

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('motion-pi')

# ── CONFIGURATION ──────────────────────────────────────────────────

# Try config.py first, then env vars, then fail loudly
CAMERA_IP = None
CAMERA_USER = None
CAMERA_PASS = None

try:
    import config as _cfg
    CAMERA_IP   = _cfg.TAPO_IP
    CAMERA_USER = _cfg.TAPO_USER
    CAMERA_PASS = _cfg.TAPO_PASS
    log.info('Credentials loaded from config.py')
except Exception:
    CAMERA_IP   = os.getenv('TAPO_IP')
    CAMERA_USER = os.getenv('TAPO_USER')
    CAMERA_PASS = os.getenv('TAPO_PASS')
    if CAMERA_IP and CAMERA_USER and CAMERA_PASS:
        log.info('Credentials loaded from environment variables')

# Validate credentials — fail early, not silently
if not CAMERA_IP or not CAMERA_USER or not CAMERA_PASS:
    log.error('='*55)
    log.error('  MISSING CAMERA CREDENTIALS')
    log.error('='*55)
    log.error('Set them via environment variables:')
    log.error('  export TAPO_IP="192.168.1.246"')
    log.error('  export TAPO_USER="your_camera_user"')
    log.error('  export TAPO_PASS="your_camera_password"')
    log.error('')
    log.error('Or create a config.py with TAPO_IP, TAPO_USER, TAPO_PASS')
    sys.exit(1)

RTSP_PORT   = 554
RTSP_STREAM = 'stream1'
# Use password as-is (ffmpeg verified it works). OpenCV may have trouble with special chars,
# but we'll try TCP transport as fallback.
RTSP_URL = f'rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:{RTSP_PORT}/{RTSP_STREAM}'
RTSP_URL_TCP = RTSP_URL + '?rtsp_transport=tcp'

PRE_BUFFER_SECONDS  = 3
COOLDOWN_SECONDS    = 5
MAX_RECORDING_SECS  = 150
VIDEO_FPS           = 15

# ── MOG2 motion detection tuning ──
MOTION_THRESHOLD    = 0.005   # fraction of pixels that must change (0.5%)
MOTION_SCALE_WIDTH  = 320     # downscale frame for faster processing
MOG2_HISTORY        = 500     # number of frames for background model
MOG2_VAR_THRESHOLD  = 40      # variance threshold for background detection
MOTION_BLUR_KSIZE   = 21      # Gaussian blur kernel size (must be odd)

# ── Output paths ──
if platform.system() == 'Windows':
    DRIVE_OUTPUT_DIR = Path(r'H:\My Drive\Fun Project\Cat monitor\TAPO_autoupload')
else:
    DRIVE_OUTPUT_DIR = Path('/home/pi5/Pictures/gdrive-randomdice-sync')

LOCAL_TEMP_DIR = Path('recordings_temp')

LOCAL_TEMP_DIR.mkdir(exist_ok=True)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CLASSES ────────────────────────────────────────────────────────

class FrameMotionDetector:
    """MOG2-based motion detection — no external dependencies."""

    def __init__(self, threshold=MOTION_THRESHOLD):
        self.threshold = threshold
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.motion_detected = False
        self.last_motion_time = None
        self._warmup_frames = 0
        self._warmup_needed = 30  # let MOG2 learn the background first

    def check(self, frame):
        """Check a frame for motion. Returns True if motion detected."""
        # Downscale for speed
        h, w = frame.shape[:2]
        scale = MOTION_SCALE_WIDTH / w
        small = cv2.resize(frame, (MOTION_SCALE_WIDTH, int(h * scale)))

        # Convert to grayscale (IR camera is grayscale anyway)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Blur to suppress noise
        blurred = cv2.GaussianBlur(gray, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)

        # Apply background subtraction
        fg_mask = self.bg_sub.apply(blurred)

        # Count motion pixels
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_ratio = motion_pixels / total_pixels

        # Warmup: let background model stabilize before triggering
        if self._warmup_frames < self._warmup_needed:
            self._warmup_frames += 1
            self.motion_detected = False
            return False

        self.motion_detected = motion_ratio > self.threshold

        if self.motion_detected:
            self.last_motion_time = datetime.now()

        return self.motion_detected


class RTSPFrameReader:
    """Reads RTSP frames in a background thread with a ring buffer."""

    def __init__(self, url, buffer_seconds=3, fps=15):
        self.url, self.fps = url, fps
        self.buffer_size = int(buffer_seconds * fps)
        self.buffer = deque(maxlen=self.buffer_size)
        self.running = False
        self.frame_width = self.frame_height = 0
        self.frames_read = 0
        self.consecutive_failures = 0
        self._lock = threading.Lock()

    def start(self):
        log.info(f'Connecting to RTSP: rtsp://{CAMERA_USER}:****@{CAMERA_IP}:{RTSP_PORT}/{RTSP_STREAM}')
        
        # Try with TCP transport first (more reliable on some networks)
        log.info('Attempting TCP transport...')
        self.cap = cv2.VideoCapture(self.url + '?rtsp_transport=tcp')
        
        # Fallback to UDP if TCP fails
        if not self.cap.isOpened():
            log.warning('TCP failed, trying UDP...')
            self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            log.error('RTSP stream could not be opened. Check credentials and network.')
            log.error(f'URL attempted: {self.url}')
            sys.exit(1)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width == 0 or self.frame_height == 0:
            log.error('RTSP connected but returned 0x0 frame size. Stream may be invalid.')
            sys.exit(1)

        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        log.info(f'📹 Stream connected: {self.frame_width}x{self.frame_height}')

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 30:
                    log.warning(f'Lost RTSP stream ({self.consecutive_failures} failures). Reconnecting...')
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.url)
                    self.consecutive_failures = 0
                else:
                    time.sleep(0.1)
                continue

            self.consecutive_failures = 0
            with self._lock:
                self.buffer.append(frame)
            self.frames_read += 1

    def get_buffer_snapshot(self):
        with self._lock:
            return list(self.buffer)

    def get_latest_frame(self):
        with self._lock:
            return self.buffer[-1].copy() if self.buffer else None


class RecordingController:
    """Manages start/stop of video recording based on motion events."""

    def __init__(self, reader, motion_detector):
        self.reader = reader
        self.motion = motion_detector
        self.is_recording = False
        self.writer = None
        self.clips_saved = 0
        self.clips_deleted = 0
        self._last_motion_ts = 0

    def tick(self):
        now = time.time()

        # Get the latest frame and check for motion
        frame = self.reader.get_latest_frame()
        if frame is None:
            return

        self.motion.check(frame)

        # Track the latest motion event time
        if self.motion.last_motion_time:
            self._last_motion_ts = max(self._last_motion_ts,
                                       self.motion.last_motion_time.timestamp())

        seconds_since_motion = now - self._last_motion_ts if self._last_motion_ts else float('inf')

        if not self.is_recording:
            if self.motion.motion_detected:
                self._start_recording()
        else:
            # Write frame
            if self.writer:
                self.writer.write(frame)
                self.frames_written += 1

            elapsed = now - self.recording_start

            # Stop conditions
            if elapsed >= MAX_RECORDING_SECS:
                log.info(f'⏱️  Max duration reached ({MAX_RECORDING_SECS}s)')
                self._stop_recording()
            elif seconds_since_motion >= COOLDOWN_SECONDS:
                log.info(f'⏸️  No motion for {COOLDOWN_SECONDS}s — stopping')
                self._stop_recording()

    def _duration_str(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f'{m}m_{s}s' if m > 0 else f'{s}s'

    def _start_recording(self):
        self._base_name = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.temp_path = LOCAL_TEMP_DIR / f'{self._base_name}.mp4'
        self.writer = cv2.VideoWriter(
            str(self.temp_path),
            cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS,
            (self.reader.frame_width, self.reader.frame_height),
        )

        if not self.writer.isOpened():
            log.error(f'VideoWriter failed to open: {self.temp_path}')
            log.error('Try: sudo apt install libavcodec-extra')
            self.writer = None
            return

        pre_frames = self.reader.get_buffer_snapshot()
        for f in pre_frames:
            self.writer.write(f)

        self.is_recording = True
        self.recording_start = time.time()
        self.frames_written = len(pre_frames)
        log.info(f'🔴 Recording started: {self._base_name} (pre-buffer: {len(pre_frames)} frames)')

    def _stop_recording(self):
        if self.writer:
            self.writer.release()
        self.is_recording = False

        duration = time.time() - self.recording_start if self.recording_start else 0
        dur_str = self._duration_str(duration)
        final_name = f'{self._base_name}_{dur_str}.mp4'

        if not self.temp_path.exists():
            log.warning('Recording stopped but no file found')
            return

        # Rename with duration and move to Drive folder
        final_temp = LOCAL_TEMP_DIR / final_name
        self.temp_path.rename(final_temp)
        dest = DRIVE_OUTPUT_DIR / final_name
        shutil.move(str(final_temp), str(dest))
        self.clips_saved += 1
        size_mb = dest.stat().st_size / (1024 * 1024)
        log.info(f'✅ Saved: {final_name} ({size_mb:.1f} MB)')

        # Trigger rclone sync on Pi
        if platform.system() != 'Windows':
            log.info('🔄 Triggering rclone bisync to Google Drive...')
            subprocess.Popen(
                ['rclone', 'bisync', str(DRIVE_OUTPUT_DIR), 'gdrive-randomdice:', '--progress'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


# ── MAIN ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    log.info('='*55)
    log.info('  Fair Feeder — Frame-Based Motion Recorder')
    log.info('  (No ONVIF dependency)')
    log.info('='*55)
    log.info(f'Camera IP:    {CAMERA_IP}')
    log.info(f'Output dir:   {DRIVE_OUTPUT_DIR}')
    log.info(f'Temp dir:     {LOCAL_TEMP_DIR}')
    log.info(f'Motion threshold: {MOTION_THRESHOLD*100:.1f}% of pixels')
    log.info(f'Cooldown:     {COOLDOWN_SECONDS}s')
    log.info(f'Max recording: {MAX_RECORDING_SECS}s')
    log.info('')

    reader = RTSPFrameReader(RTSP_URL, buffer_seconds=PRE_BUFFER_SECONDS, fps=VIDEO_FPS)
    motion = FrameMotionDetector(threshold=MOTION_THRESHOLD)
    reader.start()

    # Wait for buffer to fill and MOG2 to warm up
    log.info(f'⏳ Warming up background model ({motion._warmup_needed} frames)...')
    while motion._warmup_frames < motion._warmup_needed:
        frame = reader.get_latest_frame()
        if frame is not None:
            motion.check(frame)
        time.sleep(1.0 / VIDEO_FPS)
    log.info('✅ Background model ready')

    controller = RecordingController(reader, motion)
    log.info('')
    log.info('🚀 Monitoring for motion... Press Ctrl+C to stop.')
    log.info('')

    status_interval = 30
    last_status = time.time()

    try:
        while True:
            controller.tick()

            if time.time() - last_status >= status_interval:
                state = '🔴 REC' if controller.is_recording else '⏸️  Idle'
                log.info(f'{state} | Saved: {controller.clips_saved}'
                         f' | Frames: {reader.frames_read}')
                last_status = time.time()

            time.sleep(1.0 / VIDEO_FPS)

    except KeyboardInterrupt:
        log.info('')
        log.info('👋 Stopping...')
        if controller.is_recording:
            controller._stop_recording()
        log.info(f'Total clips saved: {controller.clips_saved}')
