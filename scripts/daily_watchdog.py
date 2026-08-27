#!/usr/bin/env python3
"""
Fair Feeder Daily Breakfast Watchdog

Monitors breakfast report delivery status every morning (e.g. at 07:30 Europe/Amsterdam).
If GitHub Actions scheduled cron is delayed or missed, it triggers workflow_dispatch
to ensure timely report delivery by ~07:35 every day without duplicate runs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pytz

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.lifecycle_manager import load_env_safe, send_telegram_alert

load_env_safe()

REPORT_TRACKER_FILE = repo_root / "scratch" / "delivered_reports.json"


def get_amsterdam_date_str() -> str:
    tz = pytz.timezone("Europe/Amsterdam")
    return datetime.now(tz).strftime("%Y%m%d")


def is_report_delivered(date_str: str) -> bool:
    """Checks if report for date_str is recorded as delivered locally or in tracker."""
    if REPORT_TRACKER_FILE.exists():
        try:
            tracker = json.loads(REPORT_TRACKER_FILE.read_text(encoding="utf-8"))
            if tracker.get(date_str, {}).get("delivered", False):
                return True
        except Exception:
            pass
    return False


def record_report_delivered(date_str: str, details: dict):
    """Records that a report was delivered for date_str to ensure idempotency."""
    REPORT_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tracker = {}
    if REPORT_TRACKER_FILE.exists():
        try:
            tracker = json.loads(REPORT_TRACKER_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    tracker[date_str] = {
        "delivered": True,
        "delivered_at": datetime.now(pytz.timezone("Europe/Amsterdam")).isoformat(),
        "details": details
    }
    REPORT_TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def check_and_recover_report(date_str: Optional[str] = None, force_dispatch: bool = False) -> bool:
    if not date_str:
        date_str = get_amsterdam_date_str()

    print(f"[{datetime.now().isoformat()}] Checking breakfast report status for {date_str}...")

    if is_report_delivered(date_str) and not force_dispatch:
        print(f"✅ Report for {date_str} already delivered. No action needed.")
        return True

    # Check GitHub Actions API for existing runs on target date
    gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "iknoest/fair-feeder")

    if not gh_token:
        print("⚠️ GITHUB_TOKEN not configured. Checking local delivery only.")
        return False

    import requests

    headers = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. Query recent workflow runs
    runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/morning-report.yml/runs?per_page=5"
    try:
        r = requests.get(runs_url, headers=headers, timeout=10)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            for run in runs:
                status = run.get("status")
                conclusion = run.get("conclusion")
                created_at = run.get("created_at", "")
                if status in ["in_progress", "queued"]:
                    print(f"ℹ️ Workflow run {run.get('id')} is currently {status}. Watchdog will not duplicate.")
                    return True
                if conclusion == "success" and created_at.startswith(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"):
                    print(f"✅ Successful workflow run {run.get('id')} found for today.")
                    record_report_delivered(date_str, {"github_run_id": run.get("id")})
                    return True
    except Exception as e:
        print(f"⚠️ Error querying GitHub Actions: {e}")

    # 2. If not delivered and not running, trigger workflow dispatch
    print(f"🚨 Report for {date_str} not delivered or running. Dispatching GitHub Actions workflow...")
    dispatch_url = f"https://api.github.com/repos/{repo}/actions/workflows/morning-report.yml/dispatches"
    payload = {
        "ref": "main",
        "inputs": {
            "date_override": date_str
        }
    }
    try:
        resp = requests.post(dispatch_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in [200, 204]:
            print(f"✅ Successfully dispatched workflow for {date_str}.")
            return True
        else:
            err = f"Failed to dispatch workflow: HTTP {resp.status_code} - {resp.text}"
            print(f"❌ {err}")
            send_telegram_alert(f"⚠️ Fair Feeder Watchdog: Could not trigger morning report for {date_str}.\n{err}")
            return False
    except Exception as e:
        err = f"Exception triggering workflow dispatch: {e}"
        print(f"❌ {err}")
        send_telegram_alert(f"⚠️ Fair Feeder Watchdog: {err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fair Feeder Breakfast Report Watchdog")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYYMMDD (default: today Amsterdam)")
    parser.add_argument("--force-dispatch", action="store_true", help="Force workflow dispatch even if recorded delivered")
    args = parser.parse_args()

    success = check_and_recover_report(date_str=args.date, force_dispatch=args.force_dispatch)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
