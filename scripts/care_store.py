"""
scripts/care_store.py - Durable Cat Care Ledger and Profile Engine

Provides an immutable event ledger and state engine for Dan and Sanbo:
- Cat profiles (Dan DOB 2020-12-09, Sanbo DOB 2025-05-25)
- Deflea (monthly) & Deworm (3-monthly) treatment cycles
- Unknown baseline initialization without fabricated historical dates
- Immutable care events ledger (DONE, SKIPPED)
- Calendar-month recurrence arithmetic
- Birthday countdowns and D-2, D-1, D0 reminder triggers
- Max 3 reminder nights per due cycle with terminal push suppression
- Atomic writes, re-entrant advisory locking (fcntl), and verified Google Drive sync
"""

import os
import sys
import json
import uuid
import shutil
import logging
import tempfile
import calendar
import threading
import contextlib
import subprocess
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("CareStore")

AMSTERDAM_TZ = ZoneInfo("Europe/Amsterdam")
DEFAULT_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive-randomdice:")

CAT_CONFIGS = {
    "dan": {
        "name": "Dan",
        "dob": "2020-12-09",
    },
    "sanbo": {
        "name": "Sanbo",
        "dob": "2025-05-25",
    },
}

TREATMENT_CADENCE_MONTHS = {
    "deflea": 1,
    "deworm": 3,
}

MAX_REMINDER_NIGHTS = 3


def get_amsterdam_now() -> datetime:
    return datetime.now(AMSTERDAM_TZ)


def get_amsterdam_today() -> date:
    return get_amsterdam_now().date()


def get_amsterdam_reminder_time(target_date: date) -> datetime:
    """Returns 21:30 local Amsterdam time on target_date as a timezone-aware datetime."""
    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        21, 30, 0,
        tzinfo=AMSTERDAM_TZ
    )


def add_calendar_months(source_date: date, months: int) -> date:
    """
    Computes exact calendar-month addition preserving month-end semantics.
    E.g.:
    2026-09-30 + 1 month -> 2026-10-30
    2026-01-31 + 1 month -> 2026-02-28 (non-leap year)
    2026-08-31 + 3 months -> 2026-11-30
    """
    total_months = source_date.year * 12 + (source_date.month - 1) + months
    new_year = total_months // 12
    new_month = (total_months % 12) + 1
    max_day = calendar.monthrange(new_year, new_month)[1]
    new_day = min(source_date.day, max_day)
    return date(new_year, new_month, new_day)


def compute_age(dob: date, as_of: date) -> int:
    """Computes full completed years of age."""
    age = as_of.year - dob.year
    if (as_of.month, as_of.day) < (dob.month, dob.day):
        age -= 1
    return max(0, age)


def compute_next_birthday(dob: date, as_of: date) -> Tuple[date, int, int]:
    """
    Computes (next_birthday_date, days_remaining, turns_age).
    """
    this_year_bday = date(as_of.year, dob.month, dob.day)
    if this_year_bday >= as_of:
        next_bday = this_year_bday
    else:
        next_bday = date(as_of.year + 1, dob.month, dob.day)

    days_remaining = (next_bday - as_of).days
    turns_age = next_bday.year - dob.year
    return next_bday, days_remaining, turns_age


def get_birthday_reminder(cat_id: str, as_of: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if today is D-2, D-1, or D0 for cat's birthday.
    Returns reminder dict or None. No D+1 reminder.
    """
    cat_cfg = CAT_CONFIGS.get(cat_id.lower())
    if not cat_cfg:
        return None

    ref_date = as_of or get_amsterdam_today()
    dob = datetime.strptime(cat_cfg["dob"], "%Y-%m-%d").date()
    next_bday, days_rem, turns = compute_next_birthday(dob, ref_date)

    if days_rem == 2:
        return {
            "cat": cat_id,
            "type": "birthday",
            "tier": "D-2",
            "text": f"🎂 {cat_cfg['name']}'s birthday is in 2 days",
            "birthday_date": next_bday.isoformat(),
            "turns": turns,
        }
    elif days_rem == 1:
        return {
            "cat": cat_id,
            "type": "birthday",
            "tier": "D-1",
            "text": f"🎂 {cat_cfg['name']}'s birthday is tomorrow",
            "birthday_date": next_bday.isoformat(),
            "turns": turns,
        }
    elif days_rem == 0:
        return {
            "cat": cat_id,
            "type": "birthday",
            "tier": "D0",
            "text": f"🎉 {cat_cfg['name']} turns {turns} today",
            "birthday_date": next_bday.isoformat(),
            "turns": turns,
        }
    return None


def get_canonical_care_path() -> Path:
    """Resolves single canonical path for cat_care.json."""
    env_path = os.environ.get("CAT_CARE_PATH")
    if env_path:
        return Path(env_path)

    pi_staging = Path("/home/pi5/Pictures/gdrive-randomdice-sync")
    if pi_staging.exists():
        return pi_staging / "cat_care.json"

    return REPO_ROOT / "cat_care.json"


def get_initial_care_state() -> Dict[str, Any]:
    """
    Creates initial Cat Care state with UNKNOWN BASELINE for all treatments.
    Never invents historical treatment dates.
    """
    initial_reminder_at = get_amsterdam_reminder_time(date(2026, 9, 6)).isoformat()
    return {
        "version": 1,
        "cats": CAT_CONFIGS,
        "care_events": [],
        "reminder_state": {
            "dan": {
                "deflea": {
                    "baseline_status": "UNKNOWN",
                    "current_due_date": None,
                    "last_completed_date": None,
                    "reminder_night_count": 0,
                    "last_reminder_at": None,
                    "next_reminder_at": initial_reminder_at,
                    "snoozed": False,
                    "terminal_push_suppressed": False,
                },
                "deworm": {
                    "baseline_status": "UNKNOWN",
                    "current_due_date": None,
                    "last_completed_date": None,
                    "reminder_night_count": 0,
                    "last_reminder_at": None,
                    "next_reminder_at": initial_reminder_at,
                    "snoozed": False,
                    "terminal_push_suppressed": False,
                },
            },
            "sanbo": {
                "deflea": {
                    "baseline_status": "UNKNOWN",
                    "current_due_date": None,
                    "last_completed_date": None,
                    "reminder_night_count": 0,
                    "last_reminder_at": None,
                    "next_reminder_at": initial_reminder_at,
                    "snoozed": False,
                    "terminal_push_suppressed": False,
                },
                "deworm": {
                    "baseline_status": "UNKNOWN",
                    "current_due_date": None,
                    "last_completed_date": None,
                    "reminder_night_count": 0,
                    "last_reminder_at": None,
                    "next_reminder_at": initial_reminder_at,
                    "snoozed": False,
                    "terminal_push_suppressed": False,
                },
            },
        },
        "notifications_sent": [],
    }


class CareStore:
    def __init__(self, path: Optional[Path] = None, remote: str = DEFAULT_REMOTE):
        self.path = path or get_canonical_care_path()
        self.remote = remote
        self.lock_path = self.path.with_suffix(".lock")
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_file = None

    @contextlib.contextmanager
    def _lock(self):
        """Re-entrant advisory file lock supporting nested calls safely."""
        with self._thread_lock:
            if self._lock_depth == 0:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._lock_file = open(self.lock_path, "w")
                if fcntl:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    if fcntl and self._lock_file:
                        try:
                            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
                    if self._lock_file:
                        try:
                            self._lock_file.close()
                        except Exception:
                            pass
                        self._lock_file = None

    def load(self) -> Dict[str, Any]:
        """Loads current Cat Care ledger or initializes default state."""
        if not self.path.exists():
            # Check secondary fallback only if using default canonical path
            if self.path == get_canonical_care_path() and not os.environ.get("CAT_CARE_PATH"):
                fallback = REPO_ROOT / "cat_care.json"
                if self.path != fallback and fallback.exists():
                    try:
                        with open(fallback, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
            initial = get_initial_care_state()
            self.save(initial, sync_drive=False)
            return initial

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading care ledger from {self.path}: {e}")
            return get_initial_care_state()

    def save(self, data: Dict[str, Any], sync_drive: bool = True) -> bool:
        """Atomically saves care ledger to disk and verifies Drive sync."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            with tempfile.NamedTemporaryFile("w", dir=str(self.path.parent), delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, indent=2, ensure_ascii=False)
                temp_path = Path(tf.name)

            os.replace(temp_path, self.path)

            # Mirror to secondary repo root if different and not in custom test path
            if not os.environ.get("CAT_CARE_PATH"):
                mirror = REPO_ROOT / "cat_care.json"
                if mirror.resolve() != self.path.resolve() and mirror.parent.exists():
                    try:
                        shutil.copy2(self.path, mirror)
                    except Exception as e:
                        log.warning(f"Failed to copy cat_care.json to mirror {mirror}: {e}")

        if sync_drive:
            return self.sync_to_drive()
        return True

    def sync_to_drive(self, timeout_sec: int = 15) -> bool:
        """Synchronizes cat_care.json to Google Drive with verification."""
        if not self.path.exists():
            return False

        remote_dest = self.remote.rstrip("/") + "/cat_care.json"
        try:
            res = subprocess.run(
                ["rclone", "copyto", str(self.path), remote_dest],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if res.returncode != 0:
                log.error(f"rclone copyto failed for cat_care.json: {res.stderr.strip()}")
                return False

            check_res = subprocess.run(
                ["rclone", "size", remote_dest, "--json"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if check_res.returncode != 0:
                log.error(f"rclone check failed for cat_care.json: {check_res.stderr.strip()}")
                return False

            size_data = json.loads(check_res.stdout)
            if size_data.get("bytes", 0) != self.path.stat().st_size:
                log.error("Drive verification size mismatch for cat_care.json")
                return False

            log.info(f"Verified cat_care.json durable sync to {remote_dest}")
            return True
        except FileNotFoundError:
            log.warning("rclone not found; skipping cat_care.json Drive sync")
            return False
        except Exception as e:
            log.error(f"Error syncing cat_care.json to Drive: {e}")
            return False

    # ── Treatment Actions ─────────────────────────────────────────────

    def record_done(
        self,
        cat: str,
        treatment: str,
        actual_date: Optional[date] = None,
        originating_due: Optional[str] = None,
        source: str = "telegram_button",
        sync_drive: bool = True,
    ) -> Dict[str, Any]:
        """
        Records a treatment as DONE:
        - Creates immutable care_event
        - Computes next due: actual completion date + 1m (deflea) or + 3m (deworm)
        - Idempotent: double tap returns existing event without creating duplicate
        """
        cat = cat.lower()
        treatment = treatment.lower()
        act_date = actual_date or get_amsterdam_today()
        act_date_str = act_date.strftime("%Y-%m-%d")

        with self._lock():
            data = self.load()
            rem_state = data.setdefault("reminder_state", {}).setdefault(cat, {}).setdefault(treatment, {})

            # Double-tap idempotency check:
            # If there is already a DONE event for this cat/treatment on this actual_date
            # or matching the originating_due, return it without duplicate event creation.
            for ev in data.get("care_events", []):
                if (
                    ev.get("cat") == cat
                    and ev.get("type") == treatment
                    and ev.get("status") == "DONE"
                    and ev.get("actual_date") == act_date_str
                ):
                    log.info(f"Duplicate DONE tapped for {cat} {treatment} on {act_date_str}; returning existing.")
                    return {"status": "already_done", "event": ev, "state": rem_state}

            cadence_m = TREATMENT_CADENCE_MONTHS.get(treatment, 1)
            next_due = add_calendar_months(act_date, cadence_m)
            next_due_str = next_due.strftime("%Y-%m-%d")

            event_id = f"care_{cat}_{treatment}_{act_date_str}_{uuid.uuid4().hex[:6]}"
            new_event = {
                "event_id": event_id,
                "cat": cat,
                "type": treatment,
                "status": "DONE",
                "actual_date": act_date_str,
                "scheduled_due_date": originating_due or rem_state.get("current_due_date"),
                "created_at": get_amsterdam_now().isoformat(),
                "action_source": source,
                "next_due_date": next_due_str,
            }
            data.setdefault("care_events", []).append(new_event)

            # Update reminder state
            rem_state["baseline_status"] = "ESTABLISHED"
            rem_state["last_completed_date"] = act_date_str
            rem_state["current_due_date"] = next_due_str
            rem_state["reminder_night_count"] = 0
            rem_state["last_reminder_at"] = None
            rem_state["next_reminder_at"] = get_amsterdam_reminder_time(next_due).isoformat()
            rem_state["snoozed"] = False
            rem_state["terminal_push_suppressed"] = False

            self.save(data, sync_drive=sync_drive)
            return {"status": "success", "event": new_event, "next_due": next_due_str}

    def record_skip(
        self,
        cat: str,
        treatment: str,
        planned_due: Optional[str] = None,
        source: str = "telegram_button",
        sync_drive: bool = True,
    ) -> Dict[str, Any]:
        """
        Records a treatment occurrence as SKIPPED:
        - Advances next due from PLANNED due occurrence date
        - If baseline unknown: anchors on today as skipped planned date
        - Preserves skipped record in care_events
        """
        cat = cat.lower()
        treatment = treatment.lower()
        today = get_amsterdam_today()
        today_str = today.strftime("%Y-%m-%d")

        with self._lock():
            data = self.load()
            rem_state = data.setdefault("reminder_state", {}).setdefault(cat, {}).setdefault(treatment, {})

            # Determine planned anchor
            if planned_due:
                try:
                    anchor_date = datetime.strptime(planned_due, "%Y-%m-%d").date()
                    plan_str = planned_due
                except ValueError:
                    anchor_date = today
                    plan_str = today_str
            elif rem_state.get("current_due_date"):
                try:
                    anchor_date = datetime.strptime(rem_state["current_due_date"], "%Y-%m-%d").date()
                    plan_str = rem_state["current_due_date"]
                except ValueError:
                    anchor_date = today
                    plan_str = today_str
            else:
                anchor_date = today
                plan_str = today_str

            cadence_m = TREATMENT_CADENCE_MONTHS.get(treatment, 1)
            next_due = add_calendar_months(anchor_date, cadence_m)
            next_due_str = next_due.strftime("%Y-%m-%d")

            event_id = f"skip_{cat}_{treatment}_{plan_str}_{uuid.uuid4().hex[:6]}"
            skip_event = {
                "event_id": event_id,
                "cat": cat,
                "type": treatment,
                "status": "SKIPPED",
                "actual_date": today_str,
                "scheduled_due_date": plan_str,
                "created_at": get_amsterdam_now().isoformat(),
                "action_source": source,
                "next_due_date": next_due_str,
            }
            data.setdefault("care_events", []).append(skip_event)

            rem_state["baseline_status"] = "ESTABLISHED"
            rem_state["current_due_date"] = next_due_str
            rem_state["reminder_night_count"] = 0
            rem_state["last_reminder_at"] = None
            rem_state["next_reminder_at"] = get_amsterdam_reminder_time(next_due).isoformat()
            rem_state["snoozed"] = False
            rem_state["terminal_push_suppressed"] = False

            self.save(data, sync_drive=sync_drive)
            return {"status": "success", "event": skip_event, "next_due": next_due_str}

    def record_not_yet(
        self,
        cat: str,
        treatment: str,
        sync_drive: bool = True,
    ) -> Dict[str, Any]:
        """
        Records 'Not yet' for a due treatment:
        - Does NOT create a completion event
        - Does NOT increment reminder_night_count (semantic authority is scheduler send)
        - Schedules next reminder for tomorrow 21:30 local Amsterdam time
        - If reminder_night_count >= MAX_REMINDER_NIGHTS: marks terminal_push_suppressed = True
        """
        cat = cat.lower()
        treatment = treatment.lower()
        tomorrow = get_amsterdam_today() + timedelta(days=1)
        next_push_time = get_amsterdam_reminder_time(tomorrow).isoformat()

        with self._lock():
            data = self.load()
            rem_state = data.setdefault("reminder_state", {}).setdefault(cat, {}).setdefault(treatment, {})

            current_count = rem_state.get("reminder_night_count", 0)
            rem_state["snoozed"] = True
            rem_state["next_reminder_at"] = next_push_time
            rem_state["last_action"] = "not_yet"
            rem_state["last_action_at"] = get_amsterdam_now().isoformat()

            if current_count >= MAX_REMINDER_NIGHTS:
                rem_state["terminal_push_suppressed"] = True

            self.save(data, sync_drive=sync_drive)
            return {
                "status": "snoozed",
                "night_count": current_count,
                "max_nights": MAX_REMINDER_NIGHTS,
                "push_suppressed": rem_state.get("terminal_push_suppressed", False),
                "next_reminder": next_push_time,
            }

    def set_last_date(
        self,
        cat: str,
        treatment: str,
        last_date_str: str,
        sync_drive: bool = True,
    ) -> Dict[str, Any]:
        """
        Establishes historical baseline by entering known last completion date.
        """
        cat = cat.lower()
        treatment = treatment.lower()
        try:
            dt = datetime.strptime(last_date_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format '{last_date_str}'. Expected YYYY-MM-DD.")

        cadence_m = TREATMENT_CADENCE_MONTHS.get(treatment, 1)
        next_due = add_calendar_months(dt, cadence_m)
        next_due_str = next_due.strftime("%Y-%m-%d")

        with self._lock():
            data = self.load()
            rem_state = data.setdefault("reminder_state", {}).setdefault(cat, {}).setdefault(treatment, {})

            event_id = f"baseline_{cat}_{treatment}_{last_date_str}_{uuid.uuid4().hex[:6]}"
            hist_event = {
                "event_id": event_id,
                "cat": cat,
                "type": treatment,
                "status": "DONE",
                "actual_date": last_date_str,
                "scheduled_due_date": None,
                "created_at": get_amsterdam_now().isoformat(),
                "action_source": "manual_baseline_set",
                "next_due_date": next_due_str,
            }
            data.setdefault("care_events", []).append(hist_event)

            rem_state["baseline_status"] = "ESTABLISHED"
            rem_state["last_completed_date"] = last_date_str
            rem_state["current_due_date"] = next_due_str
            rem_state["reminder_night_count"] = 0
            rem_state["last_reminder_at"] = None
            rem_state["next_reminder_at"] = get_amsterdam_reminder_time(next_due).isoformat()
            rem_state["snoozed"] = False
            rem_state["terminal_push_suppressed"] = False

            self.save(data, sync_drive=sync_drive)
            return {"status": "success", "event": hist_event, "next_due": next_due_str}

    # ── Queries & Profiles ────────────────────────────────────────────

    def get_cat_profile(self, cat: str, as_of: Optional[date] = None) -> Dict[str, Any]:
        """
        Builds a comprehensive profile for Dan or Sanbo:
        - Birthday, live age, next birthday, turns
        - Deflea & Deworm statuses (UNKNOWN, OK, OVERDUE, or due in N days)
        - Last completed dates, next due dates
        - Recent care events
        """
        cat = cat.lower()
        ref_date = as_of or get_amsterdam_today()
        cfg = CAT_CONFIGS.get(cat)
        if not cfg:
            raise ValueError(f"Unknown cat: {cat}")

        data = self.load()
        dob = datetime.strptime(cfg["dob"], "%Y-%m-%d").date()
        age = compute_age(dob, ref_date)
        next_bday, bday_days, turns = compute_next_birthday(dob, ref_date)

        cat_state = data.get("reminder_state", {}).get(cat, {})
        treatments = {}

        for t_name in ["deflea", "deworm"]:
            t_state = cat_state.get(t_name, {})
            baseline = t_state.get("baseline_status", "UNKNOWN")
            last_done = t_state.get("last_completed_date")
            next_due_str = t_state.get("current_due_date")

            if baseline == "UNKNOWN" or not next_due_str:
                status_label = "UNKNOWN"
                display_str = "last date unknown"
            else:
                try:
                    due_date = datetime.strptime(next_due_str, "%Y-%m-%d").date()
                    delta = (due_date - ref_date).days
                    if delta < 0:
                        status_label = "OVERDUE"
                        display_str = f"OVERDUE ({abs(delta)}d ago · {next_due_str})"
                    elif delta == 0:
                        status_label = "DUE_TODAY"
                        display_str = f"due today ({next_due_str})"
                    elif delta <= 7:
                        status_label = "DUE_SOON"
                        display_str = f"due in {delta}d ({next_due_str})"
                    else:
                        status_label = "OK"
                        display_str = f"due in {delta}d ({next_due_str})"
                except ValueError:
                    status_label = "UNKNOWN"
                    display_str = "unknown"

            treatments[t_name] = {
                "name": t_name,
                "cadence_months": TREATMENT_CADENCE_MONTHS.get(t_name),
                "baseline_status": baseline,
                "last_completed": last_done or "Unknown",
                "next_due": next_due_str or "Unknown",
                "status_label": status_label,
                "display_str": display_str,
                "reminder_night_count": t_state.get("reminder_night_count", 0),
                "terminal_push_suppressed": t_state.get("terminal_push_suppressed", False),
            }

        # History for this cat
        all_events = data.get("care_events", [])
        cat_events = [e for e in all_events if e.get("cat") == cat]
        cat_events.sort(key=lambda e: e.get("actual_date", ""), reverse=True)

        return {
            "cat": cat,
            "name": cfg["name"],
            "dob": cfg["dob"],
            "age": age,
            "next_birthday": next_bday.strftime("%Y-%m-%d"),
            "days_to_birthday": bday_days,
            "turns_age": turns,
            "treatments": treatments,
            "recent_events": cat_events[:10],
            "all_events": cat_events,
        }

    def get_dashboard(self, as_of: Optional[date] = None) -> Dict[str, Any]:
        """Returns dashboard data for all cats."""
        ref_date = as_of or get_amsterdam_today()
        return {
            cat_id: self.get_cat_profile(cat_id, as_of=ref_date)
            for cat_id in CAT_CONFIGS
        }
