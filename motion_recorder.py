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
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('motion-recorder')
START_TIME = time.time()

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

# ── TELEGRAM ───────────────────────────────────────────────────────
import requests

def send_telegram_alert(message):
    """Sends a ping to the user's Telegram."""
    try:
        # 1. We try to get from env first
        bot_token = os.getenv('TelegramBotToken')
        chat_id = os.getenv('TelegramChatId')
        
        # 2. If not in env, use Infisical REST API (No SDK required for Pi ARM compatibility)
        if not bot_token or not chat_id:
            client_id = os.getenv('INFISICAL_ID')
            client_secret = os.getenv('INFISICAL_SECRET')
            proj_id = os.getenv('INFISICAL_PROJECT_ID')
            
            if client_id and client_secret and proj_id:
                try:
                    r = requests.post('https://app.infisical.com/api/v1/auth/universal-auth/login', 
                                      json={'clientId': client_id, 'clientSecret': client_secret}, timeout=10)
                    if r.status_code == 200:
                        token = r.json()['accessToken']
                        r2 = requests.get(f'https://app.infisical.com/api/v3/secrets/raw?workspaceId={proj_id}&environment=dev', 
                                          headers={'Authorization': f'Bearer {token}'}, timeout=10)
                        if r2.status_code == 200:
                            secrets = r2.json().get('secrets', [])
                            for s in secrets:
                                if s.get('secretKey') == 'TelegramBotToken':
                                    bot_token = s.get('secretValue')
                                elif s.get('secretKey') == 'TelegramChatId':
                                    chat_id = s.get('secretValue')
                except Exception as e:
                    log.warning(f"⚠️ Failed to fetch Telegram secrets from Infisical API: {e}")
        
        if bot_token and chat_id:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            try:
                requests.post(url, json=payload, timeout=5)
                log.info("📲 Telegram startup message sent.")
            except Exception as e:
                log.warning(f"⚠️ Failed to send Telegram: {e}")
        else:
            log.info("No Telegram credentials found; skipped msg.")
            
    except Exception as e:
        log.warning(f"⚠️ Telegram config error: {e}")

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

        reported = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = reported if reported > 0 else self.fps

        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        log.info(f"📹 Stream Connected: {self.frame_width}x{self.frame_height} @ {self.stream_fps:.1f} fps")

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url + '?rtsp_transport=tcp')
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
    # YOLO confidence threshold (low because ground-level camera sees partial cats)
    YOLO_CONF = 0.10

    def __init__(self, reader, listener, yolo_model=None):
        self.reader, self.listener = reader, listener
        self.yolo_model = yolo_model
        self.is_recording = False
        self.writer = None
        self.clips_saved = 0
        self.clips_deleted = 0
        self._clips_lock = threading.Lock()  # add this line
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
                self._frame_count += 1

                # Periodic cat detection check
                if self.yolo_model and not self.cat_seen:
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
        """Run YOLO cat detection on the frame."""
        try:
            results = self.yolo_model(frame, imgsz=640, conf=self.YOLO_CONF, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.yolo_model.names[cls_id]
                    if name == 'cat':
                        self.cat_seen = True
                        elapsed = time.time() - self.recording_start
                        print(f'   🐱 Cat detected! (conf={conf:.0%}, t={elapsed:.0f}s)')
                        return
        except Exception as e:
            print(f'   ⚠️ YOLO detection error: {e}')

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
        # Use the stream's reported FPS so playback matches real wall-clock time
        self._declared_fps = getattr(self.reader, 'stream_fps', VIDEO_FPS)
        self.writer = cv2.VideoWriter(
            str(self.temp_path),
            cv2.VideoWriter_fourcc(*'mp4v'), self._declared_fps,
            (self.reader.frame_width, self.reader.frame_height),
        )

        pre_frames = self.reader.get_buffer_snapshot()
        for f in pre_frames:
            self.writer.write(f)

        self.is_recording = True
        self.recording_start = time.time()
        self.frames_written = len(pre_frames)
        self._frame_count = len(pre_frames)
        self.cat_seen = False
        self._last_cat_check = time.time()
        print(f'🔴 Recording started: {self._base_name}')

        # Check pre-buffer frames for cat (catches cats already in frame)
        if self.yolo_model and pre_frames:
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

        # Remux if actual frame rate diverged significantly from declared FPS
        # (fixes sped-up playback when RTSP delivers fewer frames than declared)
        if duration > 0:
            import subprocess
            actual_fps = self._frame_count / duration
            if abs(actual_fps - self._declared_fps) / max(self._declared_fps, 1) > 0.20:
                corrected = str(self.temp_path).replace('.mp4', '_fixed.mp4')
                pts_factor = self._declared_fps / actual_fps
                result = subprocess.run(
                    ["ffmpeg", "-i", str(self.temp_path),
                     "-vf", f"setpts={pts_factor:.6f}*PTS",
                     "-r", str(self._declared_fps), "-c:v", "libx264",
                     "-preset", "fast", "-y", corrected],
                    capture_output=True
                )
                if result.returncode == 0 and os.path.exists(corrected):
                    os.replace(corrected, str(self.temp_path))
                    print(f'   🔧 Remuxed: actual {actual_fps:.1f} fps → declared {self._declared_fps:.1f} fps')
                else:
                    print(f'   ⚠️ Remux failed (ffmpeg rc={result.returncode}), keeping original')

        # Cat detection filter: delete if no cat found in any sampled frame
        if self.yolo_model and not self.cat_seen:
            self.temp_path.unlink()
            with self._clips_lock:
                self.clips_deleted += 1
            print(f'🗑️  Deleted (no cat, {dur_str}): {final_name}')
            return

        # Rename with duration and move to Drive
        final_temp = LOCAL_TEMP_DIR / final_name
        self.temp_path.rename(final_temp)
        dest = DRIVE_OUTPUT_DIR / final_name
        shutil.move(str(final_temp), str(dest))
        with self._clips_lock:
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

class TelegramCommandListener:
    """Polls Telegram Bot API for commands and replies with live Pi stats."""
    POLL_INTERVAL = 2  # seconds between polls

    def __init__(self, bot_token, chat_id, controller):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.controller = controller
        self.running = False
        self._last_update_id = 0

    def start(self):
        self.running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        log.info('📲 Telegram command listener started')

    def _poll_loop(self):
        while self.running:
            try:
                self._check_messages()
            except Exception as e:
                log.warning(f'Telegram poll error: {e}')
            time.sleep(self.POLL_INTERVAL)

    def _check_messages(self):
        url = f'https://api.telegram.org/bot{self.bot_token}/getUpdates'
        r = requests.get(url, params={'offset': self._last_update_id + 1, 'timeout': 0}, timeout=5)
        if r.status_code != 200:
            return
        for update in r.json().get('result', []):
            self._last_update_id = update['update_id']
            msg = update.get('message', {})
            text = msg.get('text', '').strip()
            sender_id = str(msg.get('chat', {}).get('id', ''))
            if sender_id == self.chat_id and text.startswith('/'):
                self._handle_command(text.split()[0].lower())

    def _handle_command(self, cmd):
        dispatch = {'/status': self._cmd_status, '/lastclip': self._cmd_lastclip, '/help': self._cmd_help}
        handler = dispatch.get(cmd)
        if handler:
            handler()

    def _cmd_status(self):
        uptime_secs = int(time.time() - START_TIME)
        h, rem = divmod(uptime_secs, 3600)
        m = rem // 60
        try:
            disk = shutil.disk_usage(str(DRIVE_OUTPUT_DIR))
            free_gb = disk.free / (1024 ** 3)
            disk_str = f'{free_gb:.1f} GB free'
        except Exception:
            disk_str = 'unknown'
        last_motion = self.controller.listener.last_motion_time
        motion_str = last_motion.strftime('%H:%M:%S') if last_motion else 'none yet'
        with self.controller._clips_lock:
            saved = self.controller.clips_saved
            deleted = self.controller.clips_deleted
        self._send(
            f'✅ Fair Feeder Status\n'
            f'Uptime: {h}h {m}m\n'
            f'Clips saved: {saved}\n'
            f'Clips deleted: {deleted}\n'
            f'Drive space: {disk_str}\n'
            f'Last motion: {motion_str}'
        )

    def _cmd_lastclip(self):
        clips = sorted(DRIVE_OUTPUT_DIR.glob('*.mp4'), key=lambda p: p.stat().st_mtime)
        if not clips:
            self._send('No clips saved yet.')
            return
        latest = clips[-1]
        size_mb = latest.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            self._send(f'Latest clip too large to send ({size_mb:.0f} MB): {latest.name}')
            return
        url = f'https://api.telegram.org/bot{self.bot_token}/sendVideo'
        try:
            with open(latest, 'rb') as f:
                resp = requests.post(url, data={'chat_id': self.chat_id}, files={'video': f}, timeout=60)
            if resp.status_code != 200:
                self._send(f'Telegram rejected clip (HTTP {resp.status_code}): {latest.name}')
        except Exception as e:
            self._send(f'Failed to send clip: {e}')

    def _cmd_help(self):
        self._send(
            '🐱 Fair Feeder Commands\n'
            '/status — uptime, clips, disk space\n'
            '/lastclip — send most recent cat clip\n'
            '/help — this message'
        )

    def _send(self, text):
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        try:
            requests.post(url, json={'chat_id': self.chat_id, 'text': text}, timeout=5)
        except Exception as e:
            log.warning(f'Telegram reply failed: {e}')

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
    
    # Initialize YOLO cat detector
    yolo_model = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')
        log.info('🐱 Cat detector loaded (YOLOv8n)')
    except ImportError as e:
        log.warning(f'⚠️  ultralytics not found: {e}')
        log.warning('   Try: pip install ultralytics')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')
    except Exception as e:
        log.warning(f'⚠️  YOLO not available ({e})')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')

    listener = FrameMotionDetector(None, threshold_percent=2.0)
    reader = RTSPFrameReader(RTSP_URL, buffer_seconds=PRE_BUFFER_SECONDS, fps=VIDEO_FPS)
    listener.reader = reader  # Wire the reader into the detector

    try:
        listener.start()
        reader.start()
        controller = RecordingController(reader, listener, yolo_model=yolo_model)
        log.info('')
        log.info('\ud83d\ude80 Monitoring... Press Ctrl+C to stop.')
        
        # Ping Telegram so user knows the service is running (e.g. after a reboot)
        send_telegram_alert("✅ Fair Feeder Monitor is LIVE and protecting the bowl! 🐱\nRaspberry Pi 5 is officially monitoring 24/7.")

        # Start Telegram command listener (two-way health check)
        cmd_bot_token = os.getenv('TelegramBotToken')
        cmd_chat_id = os.getenv('TelegramChatId')
        if cmd_bot_token and cmd_chat_id:
            cmd_listener = TelegramCommandListener(cmd_bot_token, cmd_chat_id, controller)
            cmd_listener.start()
        else:
            log.info('No Telegram credentials — command listener disabled')

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

