"""
scripts/care_reminder_scheduler.py - Independent Care & Birthday Reminder Evaluator

Evaluates daily at 21:30 Europe/Amsterdam on the Raspberry Pi:
1. Cat birthday reminders (D-2, D-1, D0; no D+1).
2. Unknown treatment baseline reminder (sent tonight 2026-09-06 if not already sent).
3. Recurring treatment due reminders (maximum 3 nightly pushes per cycle).
4. Idempotency against multiple runs on the same calendar night.
5. Strict destination routing to feeder monitor group (ALLOWED_GROUP_ID).

Does NOT depend on cat-monitor, usb-monitor, morning workflow, or GitHub Actions.
"""

import os
import sys
import json
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.care_store import (
    CareStore,
    CAT_CONFIGS,
    get_amsterdam_today,
    get_amsterdam_now,
    get_birthday_reminder,
    AMSTERDAM_TZ,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("CareReminder")


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


def get_telegram_config() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves Bot token and Care Reminder Group Destination.
    Strictly prefers ALLOWED_GROUP_ID (feeder monitor group).
    Never exposes credentials or IDs.
    """
    load_env_safe()
    try:
        from scripts.telegram_control_service import _resolve_credentials
        token, chat_id = _resolve_credentials()
    except Exception as e:
        log.warning(f"Error resolving credentials from telegram_control_service: {e}")
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TelegramBotToken")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TelegramChatId")

    group_id = (
        os.environ.get("ALLOWED_GROUP_ID")
        or chat_id
    )
    return token, group_id


def send_telegram_message(
    token: str,
    target_chat: str,
    text: str,
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    """Sends a Telegram message without logging sensitive IDs or tokens."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": text,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram notification delivered successfully.")
            return True
        else:
            log.warning(f"Telegram sendMessage returned HTTP {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        log.warning(f"Failed to deliver Telegram notification: {e}")
        return False


class CareReminderEvaluator:
    def __init__(self, store: Optional[CareStore] = None, dry_run: bool = False):
        self.store = store or CareStore()
        self.dry_run = dry_run
        self.token, self.target_chat = get_telegram_config()

    def evaluate(self, as_of: Optional[date] = None) -> List[Dict[str, Any]]:
        """
        Main daily evaluation routine:
        1. Birthday reminders (D-2, D-1, D0)
        2. Unknown baseline reminder (if baseline needed and not sent tonight)
        3. Due treatment reminders (if established, due/overdue, night_count < 3)
        """
        ref_date = as_of or get_amsterdam_today()
        ref_date_str = ref_date.strftime("%Y-%m-%d")
        actions_taken = []

        data = self.store.load()
        notifications_sent = data.setdefault("notifications_sent", [])

        # ── 1. Birthday Check ─────────────────────────────────────────
        for cat_id in sorted(CAT_CONFIGS.keys()):
            bday_rem = get_birthday_reminder(cat_id, as_of=ref_date)
            if bday_rem:
                # Idempotency check for this cat, tier, and year
                bday_key = f"bday_{cat_id}_{bday_rem['tier']}_{ref_date_str}"
                already_sent = any(n.get("key") == bday_key for n in notifications_sent)

                if not already_sent:
                    msg_text = bday_rem["text"]
                    log.info(f"Triggering birthday reminder: {msg_text}")
                    sent = True
                    if not self.dry_run and self.token and self.target_chat:
                        sent = send_telegram_message(self.token, self.target_chat, msg_text)

                    if sent or self.dry_run:
                        record = {
                            "key": bday_key,
                            "type": "birthday",
                            "cat": cat_id,
                            "tier": bday_rem["tier"],
                            "date": ref_date_str,
                            "sent_at": get_amsterdam_now().isoformat(),
                        }
                        notifications_sent.append(record)
                        actions_taken.append(record)

        # ── 2. Care Treatments Check (Independent Evaluation) ─────────
        rem_state = data.setdefault("reminder_state", {})

        # A. Baseline Needed Check: evaluate each unknown item with a 3-night cap
        eligible_unknowns = []
        unknown_status = {c: {} for c in sorted(CAT_CONFIGS.keys())}

        for cat_id in sorted(CAT_CONFIGS.keys()):
            for t_name in ["deflea", "deworm"]:
                t_info = rem_state.get(cat_id, {}).get(t_name, {})
                is_unknown = (
                    t_info.get("baseline_status", "UNKNOWN") == "UNKNOWN"
                    or not t_info.get("current_due_date")
                )
                if is_unknown:
                    count = t_info.get("reminder_night_count", 0)
                    if count >= 3 or t_info.get("terminal_push_suppressed"):
                        t_info["terminal_push_suppressed"] = True
                        unknown_status[cat_id][t_name] = "last date unknown"
                    else:
                        eligible_unknowns.append((cat_id, t_name, t_info, count))
                        unknown_status[cat_id][t_name] = "last date unknown"

        if eligible_unknowns:
            baseline_key = f"care_baseline_needed_{ref_date_str}"
            already_sent = any(n.get("key") == baseline_key for n in notifications_sent)

            if not already_sent:
                night_num = max(cnt for _, _, _, cnt in eligible_unknowns) + 1
                baseline_text = (
                    f"🐾 Cat Care · Baseline needed (Night {night_num} of 3)\n\n"
                    "Dan\n"
                    f"🪲 Deflea: {unknown_status.get('dan', {}).get('deflea', 'OK')}\n"
                    f"🪱 Deworm: {unknown_status.get('dan', {}).get('deworm', 'OK')}\n\n"
                    "Sanbo\n"
                    f"🪲 Deflea: {unknown_status.get('sanbo', {}).get('deflea', 'OK')}\n"
                    f"🪱 Deworm: {unknown_status.get('sanbo', {}).get('deworm', 'OK')}\n\n"
                    "Open a cat profile to set or act on each care item."
                )
                markup = {
                    "inline_keyboard": [
                        [
                            {"text": "🐱 Dan", "callback_data": "care_view:dan"},
                            {"text": "🐱 Sanbo", "callback_data": "care_view:sanbo"},
                        ]
                    ]
                }
                log.info(f"Triggering baseline-needed reminder (night {night_num} of 3).")
                sent = True
                if not self.dry_run and self.token and self.target_chat:
                    sent = send_telegram_message(self.token, self.target_chat, baseline_text, reply_markup=markup)

                if sent or self.dry_run:
                    for _, _, t_info, cnt in eligible_unknowns:
                        new_cnt = cnt + 1
                        t_info["reminder_night_count"] = new_cnt
                        t_info["last_reminder_at"] = get_amsterdam_now().isoformat()
                        if new_cnt >= 3:
                            t_info["terminal_push_suppressed"] = True

                    record = {
                        "key": baseline_key,
                        "type": "care_baseline_needed",
                        "night_number": night_num,
                        "date": ref_date_str,
                        "sent_at": get_amsterdam_now().isoformat(),
                    }
                    notifications_sent.append(record)
                    actions_taken.append(record)

        # B. Due Treatments Check: independently evaluate each established item
        for cat_id in sorted(CAT_CONFIGS.keys()):
            cat_name = CAT_CONFIGS[cat_id]["name"]
            for t_name in ["deflea", "deworm"]:
                t_info = rem_state.get(cat_id, {}).get(t_name, {})
                if t_info.get("baseline_status", "UNKNOWN") == "UNKNOWN":
                    continue  # Evaluated independently in baseline check above

                due_date_str = t_info.get("current_due_date")
                if not due_date_str:
                    continue

                try:
                    due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                # Check if due or overdue today
                if due_date <= ref_date:
                    count = t_info.get("reminder_night_count", 0)

                    if count >= 3 or t_info.get("terminal_push_suppressed"):
                        t_info["terminal_push_suppressed"] = True
                        continue

                    # Idempotency for tonight
                    treatment_key = f"care_due_{cat_id}_{t_name}_{due_date_str}_{ref_date_str}"
                    already_sent = any(n.get("key") == treatment_key for n in notifications_sent)

                    if not already_sent:
                        emoji = "🪲" if t_name == "deflea" else "🪱"
                        t_title = "Deflea" if t_name == "deflea" else "Deworm"
                        night_num = count + 1

                        rem_msg = (
                            f"🐾 Cat Care Reminder · Night {night_num} of 3\n\n"
                            f"🐱 {cat_name} is due for {emoji} {t_title}.\n"
                            f"Scheduled due: {due_date_str}"
                        )
                        markup = {
                            "inline_keyboard": [
                                [
                                    {"text": "✅ Done", "callback_data": f"care_act:done:{cat_id}:{t_name}:{due_date_str}"},
                                    {"text": "🌙 Not yet", "callback_data": f"care_act:notyet:{cat_id}:{t_name}:{due_date_str}"},
                                    {"text": "⏭ Skip this cycle", "callback_data": f"care_act:skip:{cat_id}:{t_name}:{due_date_str}"},
                                ]
                            ]
                        }
                        log.info(f"Triggering care reminder: {cat_name} {t_name} (night {night_num})")
                        sent = True
                        if not self.dry_run and self.token and self.target_chat:
                            sent = send_telegram_message(self.token, self.target_chat, rem_msg, reply_markup=markup)

                        if sent or self.dry_run:
                            t_info["reminder_night_count"] = night_num
                            t_info["last_reminder_at"] = get_amsterdam_now().isoformat()
                            t_info["snoozed"] = False
                            if night_num >= 3:
                                t_info["terminal_push_suppressed"] = True

                            record = {
                                "key": treatment_key,
                                "type": "care_due_reminder",
                                "cat": cat_id,
                                "treatment": t_name,
                                "due_date": due_date_str,
                                "night_number": night_num,
                                "date": ref_date_str,
                                "sent_at": get_amsterdam_now().isoformat(),
                            }
                            notifications_sent.append(record)
                            actions_taken.append(record)

        # Save updated care state if any action occurred
        if actions_taken and not self.dry_run:
            self.store.save(data, sync_drive=True)

        return actions_taken


def main():
    parser = argparse.ArgumentParser(description="Fair Feeder Care & Birthday Reminder Evaluator")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation and send pending reminders")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without sending Telegram messages or writing changes")
    parser.add_argument("--date", type=str, default=None, help="Override evaluation date (YYYY-MM-DD)")
    args = parser.parse_args()

    ref_d = None
    if args.date:
        ref_d = datetime.strptime(args.date, "%Y-%m-%d").date()

    evaluator = CareReminderEvaluator(dry_run=args.dry_run)
    actions = evaluator.evaluate(as_of=ref_d)

    log.info(f"Evaluation complete. Actions taken: {len(actions)}")
    for a in actions:
        log.info(f"Action: {a.get('type')} key={a.get('key')}")


if __name__ == "__main__":
    main()
