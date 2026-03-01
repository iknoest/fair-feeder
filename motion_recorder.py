"""
Fair Feeder Motion Recorder with Cat Detection
===============================================

WHAT IT DOES:
  1. Connects to Tapo IP camera via RTSP
  2. Detects motion using MOG2 background subtraction
  3. Identifies cats using EfficientDet TFLite model
  4. Records video ONLY when both motion AND cats are detected
  5. Saves videos to Google Drive (via rclone)

HOW TO RUN FROM TERMINAL:
=========================

1. SIMPLE TEST (foreground - can see output):
   cd /home/pi5/Feeder/fair-feeder
   source .venv/bin/activate
   python motion_recorder.py
   
   Stop with: Ctrl+C

2. 24/7 BACKGROUND MODE (keeps running, even if terminal closes):
   cd /home/pi5/Feeder/fair-feeder
   source .venv/bin/activate
   nohup python motion_recorder.py > motion_recorder.log 2>&1 &
   
   Monitor logs with: tail -f motion_recorder.log
   Stop with: pkill -f motion_recorder.py

3. ONE-LINE VERSION (skip virtual env activation):
   /home/pi5/Feeder/fair-feeder/.venv/bin/python motion_recorder.py

CONFIGURATION:
===============
Edit /home/pi5/Feeder/fair-feeder/config.py to change:
  - TAPO_IP: Camera IP address (default: 192.168.1.246)
  - TAPO_USER: Camera username
  - TAPO_PASS: Camera password
  - DRIVE_OUTPUT_DIR: Where to save videos

TROUBLESHOOTING:
=================
- "No route to host": Camera is offline or IP is wrong
- "No module named 'cv2'": Virtual environment not activated
- "RTSP stream could not be opened": Check credentials

For more detailed guide, see: MOTION_RECORDER_GUIDE.ipynb
"""

import os
import sys
import time
import threading
import logging
import shutil
import cv2
from pathlib import Path
from datetime import datetime
from collections import deque

# Suppress verbose logs
logging.getLogger('zeep').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('motion-recorder')

# ── CONFIGURATION ──────────────────────────────────────────────────
try:
    import config as _cfg
    CAMERA_IP   = _cfg.TAPO_IP
    CAMERA_USER = _cfg.TAPO_USER
    CAMERA_PASS = _cfg.TAPO_PASS
except Exception:
    # Manual Fallback
    # NOTE: Set these environment variables or edit these locally after pulling from git.
    # NEVER commit your real credentials to git!
    CAMERA_IP   = os.getenv('TAPO_IP',   '<YOUR_CAMERA_IP>')
    CAMERA_USER = os.getenv('TAPO_USER', '<YOUR_CAMERA_USER>')
    CAMERA_PASS = os.getenv('TAPO_PASS', '<YOUR_CAMERA_PASSWORD>')

RTSP_PORT  = 554
RTSP_STREAM = 'stream1' 

# Use TCP transport for more reliable RTSP connection (proven to work on Raspberry Pi)
RTSP_URL = f'rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:{RTSP_PORT}/{RTSP_STREAM}'
RTSP_URL_TCP = RTSP_URL + '?rtsp_transport=tcp'

PRE_BUFFER_SECONDS  = 3      
COOLDOWN_SECONDS    = 5      
MAX_RECORDING_SECS  = 150    
VIDEO_FPS           = 15     

import platform

# Google Drive path
if platform.system() == 'Windows':
    DRIVE_OUTPUT_DIR = Path(r'H:\My Drive\Fun Project\Cat monitor\TAPO_autoupload')
else:
    # Raspberry Pi rclone path
    DRIVE_OUTPUT_DIR = Path('/home/pi5/Pictures/gdrive-randomdice-sync')

LOCAL_TEMP_DIR   = Path('recordings_temp')

# Ensure directories exist
LOCAL_TEMP_DIR.mkdir(exist_ok=True)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CLASSES ────────────────────────────────────────────────────────

class FrameMotionDetector:
    """Detects motion from consecutive RTSP frames using background subtraction."""
    def __init__(self, reader, threshold_percent=2.0, sample_interval=0.5):
        self.reader = reader
        self.threshold_percent = threshold_percent  # % of pixels that must change
        self.sample_interval = sample_interval  # seconds between checks
        self.motion_detected = False
        self.last_motion_time = None
        self.running = False
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self._last_check = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Background thread that checks for motion in the frame buffer."""
        while self.running:
            now = time.time()
            if now - self._last_check >= self.sample_interval:
                frame = self.reader.get_latest_frame()
                if frame is not None:
                    self._check_motion(frame)
                self._last_check = now
            time.sleep(0.1)

    def _check_motion(self, frame):
        """Analyze frame for motion and update motion_detected flag."""
        try:
            # Downscale for faster processing
            small = cv2.resize(frame, (320, 180))
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(small)
            
            # Count non-zero pixels (moving pixels)
            motion_pixels = cv2.countNonZero(fg_mask)
            total_pixels = 320 * 180
            motion_percent = (motion_pixels / total_pixels) * 100
            
            # Threshold check
            self.motion_detected = motion_percent > self.threshold_percent
            
            if self.motion_detected:
                self.last_motion_time = datetime.now()
        except Exception as e:
            print(f"⚠️ Motion detection error: {e}")

class RTSPFrameReader:
    def __init__(self, url, buffer_seconds=3, fps=15):
        self.url, self.fps = url, fps
        self.buffer_size = int(buffer_seconds * fps)
        self.buffer = deque(maxlen=self.buffer_size)
        self.running = False
        self.frame_width = self.frame_height = 0
        self.frames_read = 0
        self._lock = threading.Lock()

    def start(self):
        log.info(f'Connecting to RTSP: rtsp://****:****@{CAMERA_IP}:{RTSP_PORT}/{RTSP_STREAM}')
        
        # Try with TCP transport first (more reliable on Raspberry Pi)
        log.info('Attempting TCP transport...')
        self.cap = cv2.VideoCapture(self.url + '?rtsp_transport=tcp')
        
        # Fallback to UDP if TCP fails
        if not self.cap.isOpened():
            log.warning('TCP failed, trying UDP...')
            self.cap = cv2.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            log.error('RTSP stream could not be opened. Check credentials and network.')
            sys.exit(1)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.frame_width == 0 or self.frame_height == 0:
            log.error('RTSP connected but returned 0x0 frame size. Stream may be invalid.')
            sys.exit(1)
        
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        log.info(f"📹 Stream Connected: {self.frame_width}x{self.frame_height}")

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)
                continue
            with self._lock: self.buffer.append(frame)
            self.frames_read += 1

    def get_buffer_snapshot(self):
        with self._lock: return list(self.buffer)

    def get_latest_frame(self):
        with self._lock: return self.buffer[-1].copy() if self.buffer else None

class RecordingController:
    # How often to sample a frame for cat detection (seconds)
    CAT_CHECK_INTERVAL = 2

    def __init__(self, reader, listener, cat_detector=None):
        self.reader, self.listener = reader, listener
        self.cat_detector = cat_detector
        self.is_recording = False
        self.writer = None
        self.clips_saved = 0
        self.clips_deleted = 0
        self._last_motion_ts = 0

    def tick(self):
        now = time.time()

        # Track the latest motion event time (from ONVIF listener)
        if self.listener.last_motion_time:
            self._last_motion_ts = max(self._last_motion_ts,
                                       self.listener.last_motion_time.timestamp())

        seconds_since_motion = now - self._last_motion_ts if self._last_motion_ts else float('inf')

        if not self.is_recording:
            if self.listener.motion_detected:
                self._start_recording()
        else:
            # Write frame regardless of motion state
            frame = self.reader.get_latest_frame()
            if frame is not None and self.writer:
                self.writer.write(frame)
                self.frames_written += 1

                # Periodic cat detection check
                if self.cat_detector and not self.cat_seen:
                    if now - self._last_cat_check >= self.CAT_CHECK_INTERVAL:
                        self._last_cat_check = now
                        self._check_for_cat(frame)

            elapsed = now - self.recording_start

            # Stop conditions
            if elapsed >= MAX_RECORDING_SECS:
                print(f'   ⏱️  Max duration reached ({MAX_RECORDING_SECS}s)')
                self._stop_recording()
            elif seconds_since_motion >= COOLDOWN_SECONDS:
                print(f'   ⏸️  No motion for {COOLDOWN_SECONDS}s — stopping')
                self._stop_recording()

    def _check_for_cat(self, frame):
        """Run cat detection on a downscaled frame."""
        h, w = frame.shape[:2]
        scale = 640 / w
        small = cv2.resize(frame, (640, int(h * scale)))
        detections = self.cat_detector.detect(small, filter_cats=True)
        if detections:
            self.cat_seen = True
            best = max(detections, key=lambda d: d['score'])
            elapsed = time.time() - self.recording_start
            print(f'   🐱 Cat detected! (conf={best["score"]:.0%}, t={elapsed:.0f}s)')

    def _duration_str(self, seconds):
        """Format seconds as e.g. '4m_26s' or '10s'."""
        m, s = divmod(int(seconds), 60)
        if m > 0:
            return f'{m}m_{s}s'
        return f'{s}s'

    def _start_recording(self):
        self._base_name = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Temporary name (duration added on stop)
        self.temp_path = LOCAL_TEMP_DIR / f'{self._base_name}.mp4'
        self.writer = cv2.VideoWriter(
            str(self.temp_path),
            cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS,
            (self.reader.frame_width, self.reader.frame_height),
        )

        pre_frames = self.reader.get_buffer_snapshot()
        for f in pre_frames:
            self.writer.write(f)

        self.is_recording = True
        self.recording_start = time.time()
        self.frames_written = len(pre_frames)
        self.cat_seen = False
        self._last_cat_check = time.time()
        print(f'🔴 Recording started: {self._base_name}')

        # Check pre-buffer frames for cat (catches cats already in frame)
        if self.cat_detector and pre_frames:
            self._check_for_cat(pre_frames[-1])

    def _stop_recording(self):
        if self.writer:
            self.writer.release()
        self.is_recording = False

        duration = time.time() - self.recording_start if self.recording_start else 0
        dur_str = self._duration_str(duration)
        final_name = f'{self._base_name}_{dur_str}.mp4'

        if not self.temp_path.exists():
            print('⬜ Recording stopped (no file)')
            return

        # Cat detection filter: delete if no cat found in any sampled frame
        if self.cat_detector and not self.cat_seen:
            self.temp_path.unlink()
            self.clips_deleted += 1
            print(f'🗑️  Deleted (no cat, {dur_str}): {final_name}')
            return

        # Rename with duration and move to Drive
        final_temp = LOCAL_TEMP_DIR / final_name
        self.temp_path.rename(final_temp)
        dest = DRIVE_OUTPUT_DIR / final_name
        shutil.move(str(final_temp), str(dest))
        self.clips_saved += 1
        size_mb = dest.stat().st_size / (1024 * 1024)
        cat_status = '🐱' if self.cat_seen else '❓ no cat'
        print(f'✅ Saved: {final_name} ({size_mb:.1f} MB) [{cat_status}]')

        # Trigger rclone sync if on Raspberry Pi
        if platform.system() != 'Windows':
            import subprocess
            print('🔄 Triggering rclone to Google Drive...')
            # Run in the background (fire-and-forget) so it doesn't block the next recording
            subprocess.Popen(
                ["rclone", "bisync", str(DRIVE_OUTPUT_DIR), "gdrive-randomdice:", "--progress"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

# ── MAIN ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────
    # HOW TO RUN THIS SCRIPT:
    # ──────────────────────────────────────────────────────────────────────
    # 1. From terminal (with virtual environment activated):
    #    cd /home/pi5/Feeder/fair-feeder
    #    source .venv/bin/activate
    #    python motion_recorder.py
    #
    # 2. Or directly in one command:
    #    /home/pi5/Feeder/fair-feeder/.venv/bin/python motion_recorder.py
    #
    # 3. To run in background (keep running even if terminal closes):
    #    nohup python motion_recorder.py > motion.log 2>&1 &
    #
    # 4. To stop the recorder:
    #    pkill -f motion_recorder.py
    # ──────────────────────────────────────────────────────────────────────
    
    log.info('='*60)
    log.info('  Fair Feeder — Motion Recorder with Cat Detection')
    log.info('='*60)
    
    # Initialize cat detector (MediaPipe EfficientDet)
    cat_detector = None
    try:
        from vision.detector import CatDetector
        cat_detector = CatDetector(model_path='efficientdet_lite2.tflite',
                                   min_detection_confidence=0.35)
        log.info('🐱 Cat detector loaded (EfficientDet Lite2)')
    except ImportError as e:
        log.warning(f'⚠️  ai-edge-litert not found: {e}')
        log.warning('   Try: pip install ai-edge-litert')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')
    except Exception as e:
        log.warning(f'⚠️  Cat detector not available ({e})')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')

    listener = FrameMotionDetector(None, threshold_percent=2.0)
    reader = RTSPFrameReader(RTSP_URL, buffer_seconds=PRE_BUFFER_SECONDS, fps=VIDEO_FPS)
    listener.reader = reader  # Wire the reader into the detector

    try:
        listener.start()
        reader.start()
        controller = RecordingController(reader, listener, cat_detector=cat_detector)
        log.info('')
        log.info('\ud83d\ude80 Monitoring... Press Ctrl+C to stop.')
        log.info('')

        status_interval = 30
        last_status = time.time()

        while True:
            controller.tick()

            if time.time() - last_status >= status_interval:
                state = '\ud83d\udd34 REC' if controller.is_recording else '\u23f8\ufe0f  Idle'
                log.info(f'[{datetime.now().strftime("%H:%M:%S")}] {state}'
                      f' | Saved: {controller.clips_saved}'
                      f' | Deleted: {controller.clips_deleted}'
                      f' | Frames: {reader.frames_read}')
                last_status = time.time()

            time.sleep(1.0 / VIDEO_FPS)
    except KeyboardInterrupt:
        log.info('')
        log.info('\ud83d\udc4b Stopping...')
        if controller.is_recording:
            controller._stop_recording()
        log.info(f'Total saved: {controller.clips_saved}, deleted: {controller.clips_deleted}')

