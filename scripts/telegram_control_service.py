#!/usr/bin/env python3
"""
scripts/telegram_control_service.py - Fair Feeder Standalone Telegram Control Plane

Runs 24/7 as a lightweight daemon (fair-feeder-telegram.service) on the Raspberry Pi:
1. Responds to /help, /start, /status, /profile, /weight, /lastclip, /streaming_logitech, /streaming_tapo.
2. Continues running uninterrupted during DAYTIME_IDLE when heavy recorder processes are stopped.
3. Enforces single-consumer authority via file lock: exactly ONE poller consumes getUpdates on Pi.
4. Uses minimal RAM (~15-25 MB RSS) by avoiding heavy continuous video capture or YOLO models.
5. Provides rich interactive Cat Care dashboard, treatment actions (Done, Not yet, Skip),
   weight history, and charts backed by canonical data stores.
"""

import os
import sys
import time
import fcntl
import shutil
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.weight_store import (
    load_weights,
    save_weights,
    get_cat_weight_summary,
    get_canonical_weight_path,
)
from scripts.care_store import (
    CareStore,
    CAT_CONFIGS,
    get_amsterdam_today,
    get_amsterdam_now,
    compute_age,
    compute_next_birthday,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('TelegramControl')

LOCK_FILE_PATH = "/tmp/fair_feeder_telegram.lock"
START_TIME = time.time()


def load_env_safe():
    env_path = REPO_ROOT / '.env'
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    k, v = parts
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v


def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    load_env_safe()
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("TelegramBotToken")
    )
    chat_id = (
        os.environ.get("TELEGRAM_CHAT_ID")
        or os.environ.get("TelegramChatId")
    )
    if token and chat_id:
        return token, str(chat_id)

    # Fallback to Infisical REST API on Raspberry Pi (ARM compatibility, no SDK required)
    client_id = os.environ.get("INFISICAL_ID")
    client_secret = os.environ.get("INFISICAL_SECRET")
    proj_id = os.environ.get("INFISICAL_PROJECT_ID")
    if client_id and client_secret and proj_id:
        try:
            import requests
            r = requests.post(
                "https://app.infisical.com/api/v1/auth/universal-auth/login",
                json={"clientId": client_id, "clientSecret": client_secret},
                timeout=10
            )
            if r.status_code == 200:
                access_token = r.json().get("accessToken")
                r2 = requests.get(
                    f"https://app.infisical.com/api/v3/secrets/raw?workspaceId={proj_id}&environment=dev",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10
                )
                if r2.status_code == 200:
                    secrets = r2.json().get("secrets", [])
                    for s in secrets:
                        if s.get("secretKey") in ("TelegramBotToken", "TELEGRAM_BOT_TOKEN"):
                            token = s.get("secretValue")
                        elif s.get("secretKey") in ("TelegramChatId", "TELEGRAM_CHAT_ID"):
                            chat_id = s.get("secretValue")
                    if token and chat_id:
                        return token, str(chat_id)
        except Exception as e:
            log.warning(f"Failed to fetch Telegram secrets from Infisical REST API: {e}")

    return None, None


class HostLockError(RuntimeError):
    """Raised when another Telegram control service is already running on this host."""
    pass


def acquire_host_lock(lock_path: str = LOCK_FILE_PATH) -> Any:
    """
    Acquires an exclusive non-blocking file lock on the host.
    Returns the open file descriptor or raises HostLockError if locked.
    """
    try:
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        return lock_file
    except (BlockingIOError, IOError):
        raise HostLockError(f"Another Telegram control service is already running (locked by {lock_path}).")


class TelegramControlService:
    """
    Lightweight Telegram command poller and control plane.
    Handles user commands without keeping heavy recording pipelines active.
    """
    POLL_INTERVAL = 2.0

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        allowed_group_id: Optional[str] = None,
        weight_file: Optional[Path] = None,
        care_store: Optional[CareStore] = None,
        state_file: Optional[Path] = None,
    ):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.allowed_group_id = allowed_group_id or os.environ.get("ALLOWED_GROUP_ID")
        self.weight_file = weight_file or get_canonical_weight_path()
        self.care_store = care_store or CareStore()
        self.state_file = state_file
        self.running = False
        self._last_update_id = 0
        self._pending: Dict[str, Dict[str, Any]] = {}  # sender_id -> dialog state

    @property
    def allowed_chats(self) -> List[str]:
        chats = [self.chat_id]
        if self.allowed_group_id:
            chats.append(str(self.allowed_group_id))
        return chats

    def start(self):
        self.running = True
        log.info(f"📲 Telegram control service started for chat {self.chat_id[:4]}***")

    def stop(self):
        self.running = False
        log.info("📲 Telegram control service stopped")

    def run_forever(self):
        self.start()
        while self.running:
            try:
                self.poll_updates()
            except Exception as e:
                log.warning(f"Telegram poll error: {e}")
            time.sleep(self.POLL_INTERVAL)

    def poll_updates(self, timeout: int = 10) -> int:
        """Polls Telegram for updates and processes them. Returns count of updates handled."""
        import requests
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": timeout}
        try:
            resp = requests.get(url, params=params, timeout=timeout + 5)
            if resp.status_code != 200:
                log.warning(f"Telegram getUpdates returned HTTP {resp.status_code}: {resp.text[:100]}")
                return 0
            data = resp.json()
            updates = data.get("result", [])
            for u in updates:
                self._last_update_id = u["update_id"]
                self.process_update(u)
            return len(updates)
        except Exception as e:
            log.warning(f"Error checking Telegram messages: {e}")
            return 0

    def process_update(self, update: Dict[str, Any]):
        """Processes a single Telegram update dict."""
        import requests

        # 1. Inline keyboard callback query
        cq = update.get("callback_query")
        if cq:
            cq_id = cq["id"]
            cq_data = cq.get("data", "")
            cq_msg = cq.get("message", {})
            cq_chat_id = str(cq_msg.get("chat", {}).get("id", ""))
            cq_msg_id = cq_msg.get("message_id")

            # Always acknowledge callback query to dismiss spinner
            try:
                requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery",
                    json={"callback_query_id": cq_id},
                    timeout=5
                )
            except Exception:
                pass

            if cq_chat_id in self.allowed_chats:
                self._handle_callback_query(cq_data, cq_chat_id, message_id=cq_msg_id)
            return

        # 2. Regular message
        msg = update.get("message")
        if not msg:
            return
        text = msg.get("text", "").strip()
        if not text:
            return

        sender_id = str(msg.get("chat", {}).get("id", ""))
        if sender_id not in self.allowed_chats:
            log.info(f"⚠️ Unauthorized command attempt from {sender_id}: {text}")
            self._send(
                f"🛑 存取被拒\n\n要在這裡使用 Bot，請將 Chat ID 加入授權白名單：\n`{sender_id}`",
                sender_id=sender_id
            )
            return

        if text.startswith("/"):
            # Strip bot mention if in group e.g. /help@FeederBot -> /help
            cmd = text.split()[0].lower().split("@")[0]
            self._pending.pop(sender_id, None)  # Reset dialog state on new command
            self.dispatch_command(cmd, sender_id=sender_id)
        elif sender_id in self._pending:
            self.handle_dialog_reply(text, sender_id)

    def dispatch_command(self, cmd: str, sender_id: str):
        dispatch = {
            "/help": self._cmd_help,
            "/start": self._cmd_help,
            "/status": self._cmd_status,
            "/profile": self._cmd_profile,
            "/weight": self._cmd_weight,
            "/lastclip": self._cmd_lastclip,
            "/streaming_logitech": self._cmd_streaming_logitech,
            "/streaming_tapo": self._cmd_streaming_tapo,
        }
        handler = dispatch.get(cmd)
        if handler:
            handler(sender_id=sender_id)
        else:
            self._send(f"Unknown command {cmd}. Type /help for available commands.", sender_id=sender_id)

    # ── Command Handlers ──────────────────────────────────────────────

    def _cmd_help(self, sender_id: str):
        self._send(
            "🐾 Fair Feeder Commands\n"
            "/profile — View cat care status, treatments, and profiles\n"
            "/weight — Log, view history, or edit cat weights\n"
            "/status — View system state, uptime, memory, and sync status\n"
            "/lastclip — Send most recent cat video from staging\n"
            "/streaming_logitech — See a 5s live look from Logitech 🎥\n"
            "/streaming_tapo — See a 5s live look from Tapo 🏠\n"
            "/help — This message",
            sender_id=sender_id
        )

    def _cmd_status(self, sender_id: str):
        # 1. Read lifecycle state
        from scripts.lifecycle_manager import read_state, is_service_active, get_mem_available_mb
        state_data = read_state()
        state_name = state_data.get("state", "UNKNOWN")

        # 2. Uptime
        uptime_secs = int(time.time() - START_TIME)
        h, rem = divmod(uptime_secs, 3600)
        m = rem // 60

        # 3. Memory & Disk
        mem_mb = get_mem_available_mb()
        try:
            staging_dir = REPO_ROOT / "recordings_temp"
            disk = shutil.disk_usage(str(staging_dir.parent))
            free_gb = disk.free / (1024 ** 3)
            disk_str = f"{free_gb:.1f} GB free"
        except Exception:
            disk_str = "unknown"

        # 4. Service status
        cat_active = "active ✅" if is_service_active("cat-monitor") else "inactive ⚪"
        usb_active = "active ✅" if is_service_active("usb-monitor") else "inactive ⚪"

        # 5. Staging clips check
        recent_clip_time = "none"
        clip_count = 0
        staging_candidates = [
            Path("/home/pi5/Pictures/gdrive-randomdice-sync"),
            Path("/home/pi5/Pictures/usb-camera-sync"),
            REPO_ROOT / "recordings_temp",
            REPO_ROOT / "recordings_usb_temp",
        ]
        all_clips = []
        for d in staging_candidates:
            if d.exists():
                for p in d.glob("*.mp4"):
                    all_clips.append(p)
        if all_clips:
            all_clips.sort(key=lambda p: p.stat().st_mtime)
            latest = all_clips[-1]
            clip_count = len(all_clips)
            recent_clip_time = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S")

        # 6. Drive sync status
        sync_str = "Drive: not checked"
        try:
            import subprocess
            import json as _json
            remote = os.environ.get("RCLONE_REMOTE", "gdrive-randomdice:")
            res = subprocess.run(
                ["rclone", "size", remote, "--json"],
                capture_output=True,
                text=True,
                timeout=15
            )
            if res.returncode == 0:
                data = _json.loads(res.stdout)
                sync_str = f"Drive: {data.get('count', '?')} files ✅"
            else:
                sync_str = "Drive: unreachable ⚠️"
        except Exception:
            sync_str = "Drive: check error ❌"

        msg = (
            f"✅ Fair Feeder Status\n"
            f"Lifecycle: {state_name}\n"
            f"Bot Uptime: {h}h {m}m\n"
            f"MemAvailable: {mem_mb} MB\n"
            f"Local space: {disk_str}\n"
            f"Cat-monitor (Tapo): {cat_active}\n"
            f"Usb-monitor (Logi): {usb_active}\n"
            f"Staging clips: {clip_count} (latest: {recent_clip_time})\n"
            f"{sync_str}"
        )
        self._send(msg, sender_id=sender_id)

    def _cmd_lastclip(self, sender_id: str):
        candidates = [
            Path("/home/pi5/Pictures/gdrive-randomdice-sync"),
            Path("/home/pi5/Pictures/usb-camera-sync"),
            REPO_ROOT / "recordings_temp",
            REPO_ROOT / "recordings_usb_temp",
        ]
        all_clips = []
        for d in candidates:
            if d.exists():
                all_clips.extend(d.glob("*.mp4"))
        if not all_clips:
            self._send("No clips saved yet in staging.", sender_id=sender_id)
            return

        all_clips.sort(key=lambda p: p.stat().st_mtime)
        latest = all_clips[-1]
        size_mb = latest.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            self._send(f"Latest clip too large for Telegram ({size_mb:.1f} MB): {latest.name}", sender_id=sender_id)
            return

        self._send(f"📹 Sending latest clip ({size_mb:.1f} MB): {latest.name}...", sender_id=sender_id)
        self._send_video_file(latest, sender_id=sender_id)

    def _cmd_streaming_logitech(self, sender_id: str):
        self._send("⏳ Capturing 5s live look from LOGITECH 🎥...", sender_id=sender_id)
        try:
            from scripts.lifecycle_manager import on_demand_capture
            out_file = on_demand_capture(camera="logitech", duration_sec=5)
            if out_file and Path(out_file).exists():
                self._send_video_file(Path(out_file), sender_id=sender_id)
            else:
                self._send("⚠️ On-demand capture returned no video for Logitech.", sender_id=sender_id)
        except Exception as e:
            self._send(f"❌ Logitech streaming error: {e}", sender_id=sender_id)

    def _cmd_streaming_tapo(self, sender_id: str):
        self._send("⏳ Capturing 5s live look from TAPO 🏠...", sender_id=sender_id)
        try:
            from scripts.lifecycle_manager import on_demand_capture
            out_file = on_demand_capture(camera="tapo", duration_sec=5)
            if out_file and Path(out_file).exists():
                self._send_video_file(Path(out_file), sender_id=sender_id)
            else:
                self._send("⚠️ On-demand capture returned no video for Tapo.", sender_id=sender_id)
        except Exception as e:
            self._send(f"❌ Tapo streaming error: {e}", sender_id=sender_id)

    # ── Cat Care Profile & Dashboard ──────────────────────────────────

    def _cmd_profile(self, sender_id: str, message_id: Optional[int] = None):
        """Displays the Cat Care Status Dashboard."""
        dan_p = self.care_store.get_cat_profile("dan")
        sanbo_p = self.care_store.get_cat_profile("sanbo")

        dan_w = get_cat_weight_summary("dan")
        sanbo_w = get_cat_weight_summary("sanbo")

        dan_w_str = f"{dan_w['latest_weight']:.2f} kg · {dan_w['latest_date']}" if dan_w["latest_weight"] else "No weight logged"
        sanbo_w_str = f"{sanbo_w['latest_weight']:.2f} kg · {sanbo_w['latest_date']}" if sanbo_w["latest_weight"] else "No weight logged"

        dashboard_text = (
            "🐾 Cat Care\n\n"
            "Dan\n"
            f"⚖️ {dan_w_str}\n"
            f"🪲 Deflea: {dan_p['treatments']['deflea']['display_str']}\n"
            f"🪱 Deworm: {dan_p['treatments']['deworm']['display_str']}\n"
            f"🎂 Birthday: {dan_p['next_birthday']} ({dan_p['days_to_birthday']} days)\n\n"
            "Sanbo\n"
            f"⚖️ {sanbo_w_str}\n"
            f"🪲 Deflea: {sanbo_p['treatments']['deflea']['display_str']}\n"
            f"🪱 Deworm: {sanbo_p['treatments']['deworm']['display_str']}\n"
            f"🎂 Birthday: {sanbo_p['next_birthday']} ({sanbo_p['days_to_birthday']} days)"
        )

        markup = {
            "inline_keyboard": [
                [
                    {"text": "🐱 Dan", "callback_data": "care_view:dan"},
                    {"text": "🐱 Sanbo", "callback_data": "care_view:sanbo"},
                ]
            ]
        }

        if message_id:
            self._edit_message(dashboard_text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
        else:
            self._send(dashboard_text, sender_id=sender_id, reply_markup=markup)

    def _show_cat_profile(self, cat: str, sender_id: str, message_id: Optional[int] = None):
        """Displays detailed profile for an individual cat."""
        p = self.care_store.get_cat_profile(cat)
        w = get_cat_weight_summary(cat)

        latest_w = f"{w['latest_weight']:.2f} kg ({w['latest_date']})" if w["latest_weight"] else "None"
        prev_w = f"{w['previous_weight']:.2f} kg ({w['previous_date']})" if w["previous_weight"] else "None"
        if w["change"] is not None:
            sign = "+" if w["change"] > 0 else ""
            delta_str = f"{sign}{w['change']:.2f} kg"
        else:
            delta_str = "N/A"

        trend_emoji = {"gaining": "↗️ gaining", "losing": "↘️ losing", "stable": "➡️ stable"}.get(w["trend"], "N/A")

        deflea = p["treatments"]["deflea"]
        deworm = p["treatments"]["deworm"]

        recent_lines = []
        for e in p["recent_events"][:3]:
            recent_lines.append(f"• {e.get('actual_date')} {e.get('type').capitalize()} {e.get('status')}")
        recent_str = "\n".join(recent_lines) if recent_lines else "(none recorded yet)"

        text = (
            f"🐱 {p['name']}\n\n"
            f"🎂 Birthday\n"
            f"DOB: {p['dob']}\n"
            f"Age: {p['age']} years\n"
            f"Next birthday: {p['next_birthday']}\n"
            f"Turns: {p['turns_age']} (in {p['days_to_birthday']} days)\n\n"
            f"⚖️ Weight\n"
            f"Latest: {latest_w}\n"
            f"Previous: {prev_w}\n"
            f"Change: {delta_str}\n"
            f"Trend: {trend_emoji}\n\n"
            f"🪲 Deflea\n"
            f"Last completed: {deflea['last_completed']}\n"
            f"Next due: {deflea['next_due']}\n"
            f"Status: {deflea['status_label']}\n\n"
            f"🪱 Deworm\n"
            f"Last completed: {deworm['last_completed']}\n"
            f"Next due: {deworm['next_due']}\n"
            f"Status: {deworm['status_label']}\n\n"
            f"Recent care history:\n"
            f"{recent_str}"
        )

        markup = {
            "inline_keyboard": [
                [
                    {"text": "⚖️ Weight History", "callback_data": f"care_weighthist:{cat}"},
                    {"text": "📋 Care History", "callback_data": f"care_hist:{cat}"},
                ],
                [
                    {"text": "🪲 Deflea", "callback_data": f"care_treat:{cat}:deflea"},
                    {"text": "🪱 Deworm", "callback_data": f"care_treat:{cat}:deworm"},
                ],
                [
                    {"text": "⬅️ Back to Dashboard", "callback_data": "care_dashboard"},
                ]
            ]
        }

        if message_id:
            self._edit_message(text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
        else:
            self._send(text, sender_id=sender_id, reply_markup=markup)

    def _show_treatment_view(self, cat: str, treatment: str, sender_id: str, message_id: Optional[int] = None):
        """Displays treatment status and action buttons."""
        p = self.care_store.get_cat_profile(cat)
        t_info = p["treatments"].get(treatment)
        if not t_info:
            return

        cat_name = p["name"]
        emoji = "🪲" if treatment == "deflea" else "🪱"
        t_title = "Deflea" if treatment == "deflea" else "Deworm"

        text = (
            f"🐱 {cat_name} · {emoji} {t_title}\n\n"
            f"Cadence: every {t_info['cadence_months']} month(s)\n"
            f"Last completed: {t_info['last_completed']}\n"
            f"Next due: {t_info['next_due']}\n"
            f"Status: {t_info['display_str']}\n"
            f"Reminder attempts: {t_info['reminder_night_count']} of 3"
        )

        due_tag = t_info["next_due"] if t_info["next_due"] != "Unknown" else "baseline"

        if t_info["baseline_status"] == "UNKNOWN":
            markup = {
                "inline_keyboard": [
                    [
                        {"text": "✅ Done today", "callback_data": f"care_act:done:{cat}:{treatment}:baseline"},
                        {"text": "📅 Set last date", "callback_data": f"care_act:setdate:{cat}:{treatment}"},
                    ],
                    [
                        {"text": "🌙 Not yet", "callback_data": f"care_act:notyet:{cat}:{treatment}:baseline"},
                        {"text": "⏭ Skip this cycle", "callback_data": f"care_act:skip:{cat}:{treatment}:baseline"},
                    ],
                    [
                        {"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"},
                    ]
                ]
            }
        else:
            markup = {
                "inline_keyboard": [
                    [
                        {"text": "✅ Done", "callback_data": f"care_act:done:{cat}:{treatment}:{due_tag}"},
                        {"text": "🌙 Not yet", "callback_data": f"care_act:notyet:{cat}:{treatment}:{due_tag}"},
                    ],
                    [
                        {"text": "⏭ Skip this cycle", "callback_data": f"care_act:skip:{cat}:{treatment}:{due_tag}"},
                    ],
                    [
                        {"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"},
                    ]
                ]
            }

        if message_id:
            self._edit_message(text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
        else:
            self._send(text, sender_id=sender_id, reply_markup=markup)

    def _show_care_history(self, cat: str, sender_id: str, message_id: Optional[int] = None):
        """Displays complete care history for a cat."""
        p = self.care_store.get_cat_profile(cat)
        cat_name = p["name"]
        events = p["all_events"]

        lines = [f"📋 Complete Care History — {cat_name}:"]
        if not events:
            lines.append("(no care events recorded yet)")
        else:
            for e in events:
                status_icon = "✅" if e.get("status") == "DONE" else "⏭"
                lines.append(f"{status_icon} {e.get('actual_date')}  {e.get('type').capitalize()}  {e.get('status')}")

        markup = {
            "inline_keyboard": [
                [{"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"}]
            ]
        }
        text = "\n".join(lines)
        if message_id:
            self._edit_message(text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
        else:
            self._send(text, sender_id=sender_id, reply_markup=markup)

    def _show_cat_weight_history(self, cat: str, sender_id: str, message_id: Optional[int] = None):
        """Displays complete recovered weight history for a cat."""
        summary = get_cat_weight_summary(cat)
        cat_name = CAT_CONFIGS[cat]["name"]
        history = summary["history"]

        lines = [f"⚖️ Complete Weight History — {cat_name}:"]
        if not history:
            lines.append("(no weight records found)")
        else:
            for r in sorted(history, key=lambda x: x["date"], reverse=True):
                lines.append(f"• {r['date']}  {r['weight_kg']} kg")

        markup = {
            "inline_keyboard": [
                [{"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"}]
            ]
        }
        text = "\n".join(lines)
        if message_id:
            self._edit_message(text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
        else:
            self._send(text, sender_id=sender_id, reply_markup=markup)

    # ── Weight Command ────────────────────────────────────────────────

    def _cmd_weight(self, sender_id: str):
        self._send(
            "Weight menu:",
            sender_id=sender_id,
            reply_markup={
                "inline_keyboard": [[
                    {"text": "Log Weight", "callback_data": "weight_menu_log"},
                    {"text": "History",    "callback_data": "weight_menu_history"},
                    {"text": "Edit",       "callback_data": "weight_menu_edit"},
                ]]
            }
        )

    def _load_weights(self) -> List[Dict[str, Any]]:
        return load_weights(path=self.weight_file)

    def _save_weights(self, rows: List[Dict[str, Any]]):
        save_weights(rows, path=self.weight_file)

    def _cmd_weight_history(self, sender_id: str):
        rows = load_weights(path=self.weight_file)
        if not rows:
            self._send("No weight entries yet. Use /weight to log.", sender_id=sender_id)
            return

        dan_rows = sorted([r for r in rows if r["cat"] == "dan"], key=lambda r: r["date"])
        sanbo_rows = sorted([r for r in rows if r["cat"] == "sanbo"], key=lambda r: r["date"])

        lines = ["Weight history:"]
        lines.append("Dan:")
        for r in dan_rows[-5:]:
            lines.append(f"  {r['date']}  {r['weight_kg']} kg")
        lines.append("Sanbo:")
        for r in sanbo_rows[-5:]:
            lines.append(f"  {r['date']}  {r['weight_kg']} kg")

        self._send("\n".join(lines), sender_id=sender_id)

        # Matplotlib chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            all_dates = sorted(set(r["date"] for r in rows))
            date_to_x = {d: i for i, d in enumerate(all_dates)}

            def _to_xy(rs):
                xs = [date_to_x[r["date"]] for r in rs]
                ys = [float(r["weight_kg"]) for r in rs]
                return xs, ys

            fig, ax = plt.subplots(figsize=(8, 4))
            all_y = []

            if len(dan_rows) >= 1:
                dx, dy = _to_xy(dan_rows)
                all_y.extend(dy)
                ax.plot(dx, dy, "o-", color="#1a1a1a", label="Dan", linewidth=2)
                if len(dan_rows) >= 2:
                    xi = np.linspace(dx[0], dx[-1], 200)
                    ax.plot(xi, np.interp(xi, dx, dy), "-", color="#1a1a1a", alpha=0.3, linewidth=1)

            if len(sanbo_rows) >= 1:
                sx, sy = _to_xy(sanbo_rows)
                all_y.extend(sy)
                ax.plot(sx, sy, "o-", color="#f5a623", label="Sanbo", linewidth=2)
                if len(sanbo_rows) >= 2:
                    xi = np.linspace(sx[0], sx[-1], 200)
                    ax.plot(xi, np.interp(xi, sx, sy), "-", color="#f5a623", alpha=0.3, linewidth=1)

            if all_y:
                y_min, y_max = min(all_y), max(all_y)
                y_pad = max(0.15, (y_max - y_min) * 0.2) if y_max > y_min else 0.15
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

            ax.set_xticks(range(len(all_dates)))
            ax.set_xticklabels([d[5:] for d in all_dates], rotation=45, ha="right")
            ax.set_ylabel("kg")
            ax.set_title("Dan & Sanbo Weight")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
                tmp_path = Path(tmp.name)
            plt.close(fig)

            self._send_photo_file(tmp_path, sender_id=sender_id)
            tmp_path.unlink(missing_ok=True)
        except ImportError:
            pass
        except Exception as e:
            log.warning(f"Weight chart generation error: {e}")

    def _cmd_weight_edit(self, sender_id: str):
        rows = load_weights(path=self.weight_file)
        if not rows:
            self._send("No weight entries to edit.", sender_id=sender_id)
            return
        recent = sorted(rows, key=lambda r: r["date"])[-10:]
        lines = ["Recent weight entries (reply with number to edit):"]
        for i, r in enumerate(recent, 1):
            lines.append(f"{i}. {r['date']}  {r['cat'].capitalize()}  {r['weight_kg']} kg")
        self._send("\n".join(lines), sender_id=sender_id)
        self._pending[sender_id] = {
            "cmd": "/weight_edit", "step": "select", "data": {"entries": recent}
        }

    # ── Callback Dispatcher ───────────────────────────────────────────

    def _handle_callback_query(self, data: str, sender_id: str, message_id: Optional[int] = None):
        # 1. Weight menu callbacks
        if data == "weight_menu_log":
            self._pending[sender_id] = {"cmd": "/weight", "step": "cat", "data": {}}
            self._send(
                "Which cat?",
                sender_id=sender_id,
                reply_markup={
                    "inline_keyboard": [[
                        {"text": "Dan", "callback_data": "dan"},
                        {"text": "Sanbo", "callback_data": "sanbo"},
                    ]]
                }
            )
            return
        elif data == "weight_menu_history":
            self._cmd_weight_history(sender_id=sender_id)
            return
        elif data == "weight_menu_edit":
            self._cmd_weight_edit(sender_id=sender_id)
            return

        # 2. Cat care navigation callbacks
        if data == "care_dashboard":
            self._cmd_profile(sender_id, message_id=message_id)
            return
        elif data.startswith("care_view:"):
            cat = data.split(":", 1)[1]
            self._show_cat_profile(cat, sender_id, message_id=message_id)
            return
        elif data.startswith("care_treat:"):
            parts = data.split(":")
            if len(parts) >= 3:
                cat, treatment = parts[1], parts[2]
                self._show_treatment_view(cat, treatment, sender_id, message_id=message_id)
            return
        elif data.startswith("care_hist:"):
            cat = data.split(":", 1)[1]
            self._show_care_history(cat, sender_id, message_id=message_id)
            return
        elif data.startswith("care_weighthist:"):
            cat = data.split(":", 1)[1]
            self._show_cat_weight_history(cat, sender_id, message_id=message_id)
            return

        # 3. Treatment action callbacks
        elif data.startswith("care_act:"):
            self._handle_care_action(data, sender_id, message_id=message_id)
            return

        # 4. Fallback to dialog reply
        if sender_id in self._pending:
            self.handle_dialog_reply(data, sender_id)

    def _handle_care_action(self, data: str, sender_id: str, message_id: Optional[int] = None):
        """
        Handles treatment inline buttons:
        care_act:done:<cat>:<treatment>:<due>
        care_act:notyet:<cat>:<treatment>:<due>
        care_act:skip:<cat>:<treatment>:<due>
        care_act:setdate:<cat>:<treatment>
        """
        parts = data.split(":")
        if len(parts) < 4:
            return
        action, cat, treatment = parts[1], parts[2], parts[3]
        due_tag = parts[4] if len(parts) > 4 else None

        cat_name = CAT_CONFIGS.get(cat, {}).get("name", cat.capitalize())
        t_title = "Deflea" if treatment == "deflea" else "Deworm"

        if action == "done":
            res = self.care_store.record_done(
                cat=cat,
                treatment=treatment,
                originating_due=due_tag if due_tag != "baseline" else None,
                source="telegram_button"
            )
            next_due = res.get("next_due", "scheduled")
            today_str = get_amsterdam_today().strftime("%Y-%m-%d")
            resp_text = (
                f"✅ Recorded: {cat_name} {t_title} · {today_str}\n"
                f"Next due: {next_due}"
            )
            markup = {
                "inline_keyboard": [
                    [{"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"}]
                ]
            }
            if message_id:
                self._edit_message(resp_text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
            else:
                self._send(resp_text, sender_id=sender_id, reply_markup=markup)

        elif action == "notyet":
            res = self.care_store.record_not_yet(cat=cat, treatment=treatment)
            night_cnt = res.get("night_count", 1)
            resp_text = (
                f"🌙 Not yet\n"
                f"Next reminder: tomorrow 21:30\n"
                f"Night {night_cnt} of 3"
            )
            markup = {
                "inline_keyboard": [
                    [{"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"}]
                ]
            }
            if message_id:
                self._edit_message(resp_text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
            else:
                self._send(resp_text, sender_id=sender_id, reply_markup=markup)

        elif action == "skip":
            res = self.care_store.record_skip(
                cat=cat,
                treatment=treatment,
                planned_due=due_tag if due_tag != "baseline" else None,
                source="telegram_button"
            )
            next_due = res.get("next_due", "scheduled")
            resp_text = (
                f"⏭ Skipped this cycle\n"
                f"Next due: {next_due}"
            )
            markup = {
                "inline_keyboard": [
                    [{"text": f"⬅️ Back to {cat_name}", "callback_data": f"care_view:{cat}"}]
                ]
            }
            if message_id:
                self._edit_message(resp_text, chat_id=sender_id, message_id=message_id, reply_markup=markup)
            else:
                self._send(resp_text, sender_id=sender_id, reply_markup=markup)

        elif action == "setdate":
            self._pending[sender_id] = {
                "cmd": "/care_setdate",
                "cat": cat,
                "treatment": treatment,
            }
            self._send(
                f"Enter last completed date for {cat_name} {t_title} (YYYY-MM-DD):\n"
                f"Example: 2026-08-15",
                sender_id=sender_id
            )

    # ── Dialog Replies ────────────────────────────────────────────────

    def handle_dialog_reply(self, text: str, sender_id: str):
        state = self._pending.get(sender_id)
        if not state:
            return

        cmd = state.get("cmd")

        # 1. Setting historical care date
        if cmd == "/care_setdate":
            cat = state["cat"]
            treatment = state["treatment"]
            raw_date = text.strip()
            cat_name = CAT_CONFIGS.get(cat, {}).get("name", cat.capitalize())
            t_title = "Deflea" if treatment == "deflea" else "Deworm"
            try:
                res = self.care_store.set_last_date(cat, treatment, raw_date)
                self._pending.pop(sender_id, None)
                self._send(
                    f"✅ Baseline established for {cat_name} {t_title}!\n"
                    f"Last completed: {raw_date}\n"
                    f"Next due: {res.get('next_due')}",
                    sender_id=sender_id,
                    reply_markup={
                        "inline_keyboard": [
                            [{"text": f"🐱 {cat_name} Profile", "callback_data": f"care_view:{cat}"}]
                        ]
                    }
                )
            except Exception as e:
                self._send(f"Invalid date format: {e}. Please enter YYYY-MM-DD (e.g. 2026-08-15):", sender_id=sender_id)
            return

        # 2. Logging weight
        elif cmd == "/weight":
            step = state.get("step")
            if step == "cat":
                cat = text.strip().lower()
                if cat not in ("dan", "sanbo"):
                    self._send("Please choose Dan or Sanbo:", sender_id=sender_id)
                    return
                state["data"]["cat"] = cat
                state["step"] = "weight"
                self._send(f"Enter {cat.capitalize()} weight in kg (e.g. 3.45):", sender_id=sender_id)
            elif step == "weight":
                try:
                    kg = float(text.strip().replace(",", "."))
                    if not (0.5 <= kg <= 20.0):
                        raise ValueError
                except ValueError:
                    self._send("Invalid weight. Enter a number between 0.5 and 20.0:", sender_id=sender_id)
                    return
                cat = state["data"]["cat"]
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                rows = load_weights(path=self.weight_file)
                found = False
                for r in rows:
                    if r["date"] == today and r["cat"] == cat:
                        r["weight_kg"] = f"{kg:.2f}"
                        found = True
                        break
                if not found:
                    rows.append({"date": today, "cat": cat, "weight_kg": f"{kg:.2f}"})
                save_weights(rows, path=self.weight_file)
                self._pending.pop(sender_id, None)
                self._send(
                    f"✅ Saved: {cat.capitalize()} = {kg:.2f} kg on {today}",
                    sender_id=sender_id,
                    reply_markup={
                        "inline_keyboard": [[
                            {"text": "View History", "callback_data": "weight_menu_history"},
                        ]]
                    }
                )

        # 3. Editing weight
        elif cmd == "/weight_edit":
            step = state.get("step")
            if step == "select":
                entries = state["data"]["entries"]
                try:
                    idx = int(text.strip()) - 1
                    if not (0 <= idx < len(entries)):
                        raise ValueError
                except ValueError:
                    self._send("Enter a number from the list:", sender_id=sender_id)
                    return
                state["data"]["idx"] = idx
                e = entries[idx]
                state["step"] = "value"
                self._send(
                    f"Current: {e['cat'].capitalize()} {e['weight_kg']} kg on {e['date']}\n"
                    f"Enter new weight in kg (or 'delete' to remove):",
                    sender_id=sender_id
                )
            elif step == "value":
                entries = state["data"]["entries"]
                idx = state["data"]["idx"]
                entry = entries[idx]
                rows = load_weights(path=self.weight_file)

                if text.strip().lower() == "delete":
                    rows = [
                        r for r in rows
                        if not (r["date"] == entry["date"] and r["cat"] == entry["cat"])
                    ]
                    save_weights(rows, path=self.weight_file)
                    self._pending.pop(sender_id, None)
                    self._send(f"Deleted {entry['cat'].capitalize()} entry for {entry['date']}.", sender_id=sender_id)
                    return

                try:
                    kg = float(text.strip().replace(",", "."))
                    if not (0.5 <= kg <= 20.0):
                        raise ValueError
                except ValueError:
                    self._send("Invalid number. Enter kg or 'delete':", sender_id=sender_id)
                    return

                for r in rows:
                    if r["date"] == entry["date"] and r["cat"] == entry["cat"]:
                        r["weight_kg"] = f"{kg:.2f}"
                        break
                save_weights(rows, path=self.weight_file)
                self._pending.pop(sender_id, None)
                self._send(f"Updated {entry['cat'].capitalize()} on {entry['date']} to {kg:.2f} kg.", sender_id=sender_id)

    # ── Telegram API Helpers ──────────────────────────────────────────

    def _send(self, text: str, sender_id: Optional[str] = None, reply_markup: Optional[Dict[str, Any]] = None):
        import requests
        target = sender_id or self.chat_id
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": target, "text": text}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            log.warning(f"Telegram sendMessage failed: {e}")

    def _edit_message(self, text: str, chat_id: str, message_id: int, reply_markup: Optional[Dict[str, Any]] = None):
        """Updates an existing message in-place to avoid clutter."""
        import requests
        url = f"https://api.telegram.org/bot{self.bot_token}/editMessageText"
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            log.warning(f"Telegram editMessageText failed: {e}")

    def _send_video_file(self, video_path: Path, sender_id: Optional[str] = None):
        import requests
        target = sender_id or self.chat_id
        url = f"https://api.telegram.org/bot{self.bot_token}/sendVideo"
        try:
            with open(video_path, "rb") as f:
                resp = requests.post(url, data={"chat_id": target}, files={"video": f}, timeout=60)
            if resp.status_code != 200:
                self._send(f"Telegram video send failed (HTTP {resp.status_code})", sender_id=target)
        except Exception as e:
            self._send(f"Failed to upload video: {e}", sender_id=target)

    def _send_photo_file(self, photo_path: Path, sender_id: Optional[str] = None):
        import requests
        target = sender_id or self.chat_id
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        try:
            with open(photo_path, "rb") as f:
                requests.post(url, data={"chat_id": target}, files={"photo": f}, timeout=30)
        except Exception as e:
            log.warning(f"Telegram sendPhoto failed: {e}")


def main():
    # 1. Enforce single poller on host
    try:
        _lock = acquire_host_lock()
    except HostLockError as e:
        log.error(f"Singleton gate: {e}")
        sys.exit(0)

    # 2. Load credentials
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        log.error("Missing Telegram credentials. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        sys.exit(1)

    service = TelegramControlService(bot_token=token, chat_id=chat_id)
    log.info("Starting standalone Fair Feeder Telegram Control Service...")
    try:
        service.run_forever()
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting.")
        service.stop()


if __name__ == "__main__":
    main()
