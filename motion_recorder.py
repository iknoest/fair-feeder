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
  - TAPO_IP: Camera IP address (default: <YOUR_CAMERA_IP>)
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

BOWL_CHECK_INTERVAL_SECONDS = int(os.getenv('BOWL_CHECK_INTERVAL_SECONDS', '30'))
BOWL_BAD_SECONDS = int(os.getenv('BOWL_BAD_SECONDS', '600'))
BOWL_ALERT_COOLDOWN_SECONDS = int(os.getenv('BOWL_ALERT_COOLDOWN_SECONDS', '21600'))
BOWL_CENTER_MIN = float(os.getenv('BOWL_CENTER_MIN', '0.25'))
BOWL_CENTER_MAX = float(os.getenv('BOWL_CENTER_MAX', '0.75'))
BOWL_CONF = float(os.getenv('BOWL_CONF', '0.25'))

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

def get_telegram_credentials():
    """Fetches Telegram credentials from env or Infisical."""
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
    return bot_token, chat_id

def send_telegram_alert(message):
    """Sends a ping to the user's Telegram."""
    try:
        bot_token, chat_id = get_telegram_credentials()
        
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
                ["rclone", "copy", str(DRIVE_OUTPUT_DIR), "gdrive-randomdice:"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

class BowlPositionMonitor:
    """Periodically checks whether the COCO bowl class is framed."""

    def __init__(self, reader, yolo_model=None):
        self.reader = reader
        self.yolo_model = yolo_model
        self._last_check = 0
        self._bad_since = None
        self._last_alert = 0
        self._alert_active = False

    def tick(self):
        if not self.yolo_model:
            return

        now = time.time()
        if now - self._last_check < BOWL_CHECK_INTERVAL_SECONDS:
            return
        self._last_check = now

        frame = self.reader.get_latest_frame()
        if frame is None:
            return

        ok, reason = self._check_bowl(frame)
        if ok:
            if self._alert_active:
                send_telegram_alert('Bowl position recovered. Camera sees the bowl in frame again.')
            self._bad_since = None
            self._alert_active = False
            return

        if self._bad_since is None:
            self._bad_since = now

        bad_seconds = now - self._bad_since
        if bad_seconds < BOWL_BAD_SECONDS:
            return

        if now - self._last_alert < BOWL_ALERT_COOLDOWN_SECONDS:
            return

        minutes = round(bad_seconds / 60)
        send_telegram_alert(
            f'Camera position alert\n'
            f'Bowl has been {reason} for ~{minutes} min.\n'
            f'Please check the Tapo camera position.'
        )
        self._last_alert = now
        self._alert_active = True

    def _check_bowl(self, frame):
        try:
            results = self.yolo_model(frame, imgsz=640, conf=BOWL_CONF, verbose=False)
            height, width = frame.shape[:2]
            best = None
            best_area = 0

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = str(self.yolo_model.names[cls_id])
                    if name.lower() != 'bowl':
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area > best_area:
                        best_area = area
                        best = (x1, y1, x2, y2)

            if best is None:
                return False, 'not detected'

            x1, y1, x2, y2 = best
            cx = ((x1 + x2) / 2) / max(width, 1)
            cy = ((y1 + y2) / 2) / max(height, 1)
            if BOWL_CENTER_MIN <= cx <= BOWL_CENTER_MAX and BOWL_CENTER_MIN <= cy <= BOWL_CENTER_MAX:
                return True, ''
            return False, 'outside the center area'
        except Exception as e:
            log.warning(f'Bowl position check failed: {e}')
            return True, ''


class TelegramCommandListener:
    """Polls Telegram Bot API for commands and replies with live Pi stats."""
    POLL_INTERVAL = 2  # seconds between polls

    def __init__(self, bot_token, chat_id, controller):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.controller = controller
        self.running = False
        self._last_update_id = 0
        self._pending = {}  # sender_id -> {'cmd': str, 'step': str, 'data': dict}

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

            # ── Inline keyboard callback ──────────────────────────
            cq = update.get('callback_query', {})
            if cq:
                cq_id      = cq['id']
                cq_data    = cq.get('data', '')
                cq_chat_id = str(cq.get('message', {}).get('chat', {}).get('id', ''))
                allowed_chats = [self.chat_id]
                group_id = os.getenv("ALLOWED_GROUP_ID")
                if group_id:
                    allowed_chats.append(group_id)
                # Always answer to remove spinner
                try:
                    requests.post(
                        f'https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery',
                        json={'callback_query_id': cq_id}, timeout=5
                    )
                except Exception:
                    pass
                if cq_chat_id in allowed_chats:
                    if cq_data == 'weight_menu_log':
                        self._pending[cq_chat_id] = {'cmd': '/weight', 'step': 'cat', 'data': {}}
                        self._send('Which cat?', sender_id=cq_chat_id, reply_markup={'inline_keyboard': [[
                            {'text': 'Dan', 'callback_data': 'dan'},
                            {'text': 'Sanbo', 'callback_data': 'sanbo'},
                        ]]})
                    elif cq_data == 'weight_menu_history':
                        self._cmd_weight_history(sender_id=cq_chat_id)
                    elif cq_data == 'weight_menu_edit':
                        self._cmd_weight_edit(sender_id=cq_chat_id)
                    elif cq_chat_id in self._pending:
                        self._handle_dialog(cq_data, cq_chat_id)
                continue

            # ── Regular message ───────────────────────────────────
            msg = update.get('message', {})
            text = msg.get('text', '').strip()
            if not text:
                continue

            sender_id = str(msg.get('chat', {}).get('id', ''))

            # 若需授權群組，請在 .env 中加入 ALLOWED_GROUP_ID=-100XXXXXXXXX
            allowed_chats = [self.chat_id]
            group_id = os.getenv("ALLOWED_GROUP_ID")
            if group_id:
                allowed_chats.append(group_id)

            if text.startswith('/'):
                # 處理群組中帶有 @botname 的指令 (例: /status@FeederBot -> /status)
                cmd = text.split()[0].lower().split('@')[0]
                if sender_id in allowed_chats:
                    self._pending.pop(sender_id, None)  # new command resets dialog
                    self._handle_command(cmd, sender_id=sender_id)
                else:
                    log.info(f"⚠️ 收到來自未授權 Chat ID 的指令: {sender_id} (訊息: {text})")
                    try:
                        reply_msg = (
                            f"🛑 存取被拒\n\n"
                            f"要在這裡使用 Bot，請將這個 Chat ID 加入 `motion_recorder.py` 的白名單內：\n\n"
                            f"`{sender_id}`\n\n"
                            f"(您可以複製上面這串數字)"
                        )
                        requests.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                                      json={'chat_id': sender_id, 'text': reply_msg}, timeout=5)
                    except Exception as e:
                        log.warning(f"無法回覆未授權 Chat ID ({sender_id}): {e}")
            elif sender_id in allowed_chats and sender_id in self._pending:
                self._handle_dialog(text, sender_id)

    def _weight_file(self):
        return DRIVE_OUTPUT_DIR / 'weight_log.csv'

    def _load_weights(self):
        import csv as _csv
        wf = self._weight_file()
        if not wf.exists():
            return []
        with open(wf, newline='', encoding='utf-8') as f:
            return list(_csv.DictReader(f))

    def _save_weights(self, rows):
        import csv as _csv
        wf = self._weight_file()
        with open(wf, 'w', newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=['date', 'cat', 'weight_kg'])
            w.writeheader()
            w.writerows(rows)
        # Sync to Drive
        try:
            import subprocess as _sp
            _sp.Popen(
                ['rclone', 'copy', str(wf), 'gdrive-randomdice:'],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
            )
        except Exception as e:
            log.warning(f'rclone sync weight_log.csv failed: {e}')

    def _handle_dialog(self, text, sender_id):
        state = self._pending.get(sender_id)
        if not state:
            return
        cmd  = state['cmd']
        step = state['step']

        if cmd == '/weight':
            if step == 'cat':
                cat = text.strip().lower()
                if cat not in ('dan', 'sanbo'):
                    self._send('Please reply with dan or sanbo.', sender_id=sender_id)
                    return
                state['data']['cat'] = cat
                state['step'] = 'value'
                self._send(f'Enter {cat.capitalize()} weight in kg (e.g. 5.2):', sender_id=sender_id)

            elif step == 'value':
                try:
                    kg = float(text.strip().replace(',', '.'))
                    if not (0.5 <= kg <= 20):
                        raise ValueError('out of range')
                except ValueError:
                    self._send('Invalid number. Enter kg as a decimal (e.g. 5.2):', sender_id=sender_id)
                    return
                cat = state['data']['cat']
                today = __import__('datetime').date.today().isoformat()
                rows = self._load_weights()
                rows.append({'date': today, 'cat': cat, 'weight_kg': f'{kg:.2f}'})
                self._save_weights(rows)
                self._pending.pop(sender_id, None)
                self._send(
                    f'Saved: {cat.capitalize()} = {kg} kg on {today}',
                    sender_id=sender_id
                )

        elif cmd == '/weight_edit':
            if step == 'select':
                try:
                    idx = int(text.strip()) - 1
                    entries = state['data']['entries']
                    if not (0 <= idx < len(entries)):
                        raise ValueError
                except ValueError:
                    self._send('Enter a number from the list:', sender_id=sender_id)
                    return
                state['data']['idx'] = idx
                e = entries[idx]
                state['step'] = 'value'
                self._send(
                    f'Current: {e["cat"].capitalize()} {e["weight_kg"]} kg on {e["date"]}\n'
                    f'Enter new weight in kg (or "delete" to remove):',
                    sender_id=sender_id
                )

            elif step == 'value':
                entries = state['data']['entries']
                idx     = state['data']['idx']
                rows    = self._load_weights()
                entry   = entries[idx]
                if text.strip().lower() == 'delete':
                    rows = [
                        r for r in rows
                        if not (r['date'] == entry['date'] and r['cat'] == entry['cat'])
                    ]
                    self._save_weights(rows)
                    self._pending.pop(sender_id, None)
                    self._send(f'Deleted {entry["cat"].capitalize()} entry for {entry["date"]}.', sender_id=sender_id)
                    return
                try:
                    kg = float(text.strip().replace(',', '.'))
                    if not (0.5 <= kg <= 20):
                        raise ValueError('out of range')
                except ValueError:
                    self._send('Invalid number. Enter kg or "delete":', sender_id=sender_id)
                    return
                for r in rows:
                    if r['date'] == entry['date'] and r['cat'] == entry['cat']:
                        r['weight_kg'] = f'{kg:.2f}'
                        break
                self._save_weights(rows)
                self._pending.pop(sender_id, None)
                self._send(
                    f'Updated {entry["cat"].capitalize()} on {entry["date"]} to {kg} kg.',
                    sender_id=sender_id
                )

    def _handle_command(self, cmd, sender_id=None):
        dispatch = {
            '/status':   self._cmd_status,
            '/lastclip': self._cmd_lastclip,
            '/weight':   self._cmd_weight,
            '/help':     self._cmd_help,
            '/start':    self._cmd_help,
        }
        handler = dispatch.get(cmd)
        if handler:
            handler(sender_id=sender_id)

    def _cmd_status(self, sender_id=None):
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
        # Quick Drive sync check
        try:
            import subprocess as _sp, json as _json
            _res = _sp.run(
                ['rclone', 'size', 'gdrive-randomdice:', '--json'],
                capture_output=True, text=True, timeout=8
            )
            if _res.returncode == 0:
                _rdata = _json.loads(_res.stdout)
                sync_str = f'Drive: {_rdata.get("count", "?")} files ✅'
            else:
                sync_str = f'Drive: unreachable ⚠️'
        except Exception:
            sync_str = 'Drive: check skipped'

        self._send(
            f'✅ Fair Feeder Status\n'
            f'Uptime: {h}h {m}m\n'
            f'Clips saved: {saved}\n'
            f'Clips deleted: {deleted}\n'
            f'Local space: {disk_str}\n'
            f'Last motion: {motion_str}\n'
            f'{sync_str}',
            sender_id=sender_id
        )

    def _cmd_lastclip(self, sender_id=None):
        target_id = sender_id if sender_id else self.chat_id
        clips = sorted(DRIVE_OUTPUT_DIR.glob('*.mp4'), key=lambda p: p.stat().st_mtime)
        if not clips:
            self._send('No clips saved yet.', sender_id=sender_id)
            return
        latest = clips[-1]
        size_mb = latest.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            self._send(f'Latest clip too large to send ({size_mb:.0f} MB): {latest.name}', sender_id=sender_id)
            return
        url = f'https://api.telegram.org/bot{self.bot_token}/sendVideo'
        try:
            with open(latest, 'rb') as f:
                resp = requests.post(url, data={'chat_id': target_id}, files={'video': f}, timeout=60)
            if resp.status_code != 200:
                self._send(f'Telegram rejected clip (HTTP {resp.status_code}): {latest.name}', sender_id=sender_id)
        except Exception as e:
            self._send(f'Failed to send clip: {e}', sender_id=sender_id)

    def _cmd_weight(self, sender_id=None):
        self._send(
            'Weight menu:',
            sender_id=sender_id,
            reply_markup={'inline_keyboard': [[
                {'text': 'Log Weight', 'callback_data': 'weight_menu_log'},
                {'text': 'History',    'callback_data': 'weight_menu_history'},
                {'text': 'Edit',       'callback_data': 'weight_menu_edit'},
            ]]}
        )

    def _cmd_weight_history(self, sender_id=None):
        rows = self._load_weights()
        if not rows:
            self._send('No weight entries yet. Use /weight to log.', sender_id=sender_id)
            return

        # Text summary (most recent 10 per cat)
        from datetime import datetime as _dt
        dan_rows   = sorted([r for r in rows if r['cat'] == 'dan'],   key=lambda r: r['date'])
        sanbo_rows = sorted([r for r in rows if r['cat'] == 'sanbo'], key=lambda r: r['date'])

        lines = ['Weight history:']
        lines.append('Dan:')
        for r in dan_rows[-5:]:
            lines.append(f"  {r['date']}  {r['weight_kg']} kg")
        lines.append('Sanbo:')
        for r in sanbo_rows[-5:]:
            lines.append(f"  {r['date']}  {r['weight_kg']} kg")

        self._send('\n'.join(lines), sender_id=sender_id)

        # Chart
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import tempfile

            # Unified sorted date list → integer X positions (avoids matplotlib.dates Jan-01 bug)
            all_dates = sorted(set(r['date'] for r in rows))
            date_to_x = {d: i for i, d in enumerate(all_dates)}

            def _to_xy(rs):
                xs = [date_to_x[r['date']] for r in rs]
                ys = [float(r['weight_kg']) for r in rs]
                return xs, ys

            fig, ax = plt.subplots(figsize=(8, 4))
            all_y = []

            # Dan = black (#1a1a1a), Sanbo = orange (#f5a623)
            if len(dan_rows) >= 1:
                dx, dy = _to_xy(dan_rows)
                all_y.extend(dy)
                ax.plot(dx, dy, 'o-', color='#1a1a1a', label='Dan', linewidth=2)
                if len(dan_rows) >= 2:
                    xi = np.linspace(dx[0], dx[-1], 200)
                    ax.plot(xi, np.interp(xi, dx, dy), '-', color='#1a1a1a', alpha=0.3, linewidth=1)

            if len(sanbo_rows) >= 1:
                sx, sy = _to_xy(sanbo_rows)
                all_y.extend(sy)
                ax.plot(sx, sy, 'o-', color='#f5a623', label='Sanbo', linewidth=2)
                if len(sanbo_rows) >= 2:
                    xi = np.linspace(sx[0], sx[-1], 200)
                    ax.plot(xi, np.interp(xi, sx, sy), '-', color='#f5a623', alpha=0.3, linewidth=1)

            # Y axis: variable range padded around actual data, not starting from 0
            if all_y:
                y_min, y_max = min(all_y), max(all_y)
                y_pad = max(0.15, (y_max - y_min) * 0.2) if y_max > y_min else 0.15
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            # X axis: integer ticks with MM-DD string labels
            ax.set_xticks(range(len(all_dates)))
            ax.set_xticklabels([d[5:] for d in all_dates], rotation=45, ha='right')
            ax.set_ylabel('kg')
            ax.set_title('Dan & Sanbo Weight')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=120, bbox_inches='tight')
                tmp_path = tmp.name
            plt.close(fig)

            url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
            target = sender_id if sender_id else self.chat_id
            with open(tmp_path, 'rb') as img:
                requests.post(url, data={'chat_id': target}, files={'photo': img}, timeout=30)
            import os as _os
            _os.unlink(tmp_path)

        except ImportError:
            self._send('(Chart requires matplotlib on Pi)', sender_id=sender_id)
        except Exception as e:
            self._send(f'Chart error: {e}', sender_id=sender_id)

    def _cmd_weight_edit(self, sender_id=None):
        rows = self._load_weights()
        if not rows:
            self._send('No weight entries to edit.', sender_id=sender_id)
            return
        recent = sorted(rows, key=lambda r: r['date'])[-10:]
        lines = ['Recent weight entries (reply with number to edit):']
        for i, r in enumerate(recent, 1):
            lines.append(f"{i}. {r['date']}  {r['cat'].capitalize()}  {r['weight_kg']} kg")
        self._send('\n'.join(lines), sender_id=sender_id)
        self._pending[sender_id] = {
            'cmd': '/weight_edit', 'step': 'select', 'data': {'entries': recent}
        }

    def _cmd_help(self, sender_id=None):
        self._send(
            'Fair Feeder Commands\n'
            '/status — uptime, clips, disk, Drive sync\n'
            '/lastclip — send most recent cat clip\n'
            '/weight — log, view history, or edit weights\n'
            '/help — this message',
            sender_id=sender_id
        )

    def _send(self, text, sender_id=None, reply_markup=None):
        target = sender_id if sender_id else self.chat_id
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        payload = {'chat_id': target, 'text': text}
        if reply_markup:
            payload['reply_markup'] = reply_markup
        try:
            requests.post(url, json=payload, timeout=5)
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
    
    # Initialize YOLO detectors
    yolo_model = None
    bowl_monitor_model = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')
        names = getattr(yolo_model, 'names', {})
        has_bowl = any(str(name).lower() == 'bowl' for name in names.values())
        log.info(f'YOLOv8n detector loaded (cat filter, bowl monitor: {"on" if has_bowl else "off"})')
        if has_bowl:
            bowl_monitor_model = yolo_model
        else:
            log.warning('Bowl position monitor disabled: YOLOv8n class list has no bowl class')
    except ImportError as e:
        YOLO = None
        log.warning(f'ultralytics not found: {e}')
        log.warning('   Try: pip install ultralytics')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')
    except Exception as e:
        YOLO = None
        log.warning(f'YOLO not available ({e})')
        log.info('   Falling back to: all clips will be kept (no cat filtering)')

    listener = FrameMotionDetector(None, threshold_percent=2.0)
    reader = RTSPFrameReader(RTSP_URL, buffer_seconds=PRE_BUFFER_SECONDS, fps=VIDEO_FPS)
    listener.reader = reader  # Wire the reader into the detector

    try:
        listener.start()
        reader.start()
        controller = RecordingController(reader, listener, yolo_model=yolo_model)
        bowl_monitor = BowlPositionMonitor(reader, yolo_model=bowl_monitor_model)
        log.info('')
        log.info('\ud83d\ude80 Monitoring... Press Ctrl+C to stop.')
        
        # Ping Telegram so user knows the service is running (e.g. after a reboot)
        send_telegram_alert("✅ Fair Feeder Monitor is LIVE and protecting the bowl! 🐱\nRaspberry Pi 5 is officially monitoring 24/7.")

        # Start Telegram command listener (two-way health check)
        cmd_bot_token, cmd_chat_id = get_telegram_credentials()
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
            bowl_monitor.tick()

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
