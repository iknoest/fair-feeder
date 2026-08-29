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


def parse_github_datetime_amsterdam(dt_str: str) -> Optional[datetime]:
    """Parses GitHub ISO-8601 UTC timestamp and converts to Europe/Amsterdam timezone."""
    if not dt_str:
        return None
    try:
        # e.g. "2026-08-29T07:15:23Z"
        clean_str = dt_str.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_str)
        tz = pytz.timezone("Europe/Amsterdam")
        return dt_utc.astimezone(tz)
    except Exception:
        return None


def get_github_token() -> Optional[str]:
    """Resolves GitHub Token from env, .env file, or Infisical Universal Auth."""
    # 1. Check environment variables
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token

    # 2. Check local .env file
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in ["GITHUB_TOKEN", "GH_TOKEN"]:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val

    # 3. Check Infisical Universal Auth
    client_id = os.environ.get("INFISICAL_ID")
    client_secret = os.environ.get("INFISICAL_SECRET")
    proj_id = os.environ.get("INFISICAL_PROJECT_ID")
    if client_id and client_secret and proj_id:
        try:
            import requests
            r = requests.post("https://app.infisical.com/api/v1/auth/universal-auth/login",
                              json={"clientId": client_id, "clientSecret": client_secret}, timeout=10)
            if r.status_code == 200:
                auth_token = r.json().get("accessToken")
                r2 = requests.get(f"https://app.infisical.com/api/v3/secrets/raw?workspaceId={proj_id}&environment=dev",
                                  headers={"Authorization": f"Bearer {auth_token}"}, timeout=10)
                if r2.status_code == 200:
                    secrets = r2.json().get("secrets", [])
                    for s in secrets:
                        if s.get("secretKey") in ["GITHUB_TOKEN", "GH_TOKEN"]:
                            return s.get("secretValue")
        except Exception:
            pass

    return None


def is_report_delivered(date_str: str) -> bool:
    """Checks if report for date_str is recorded as delivered locally."""
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
        print(f"✅ Report for {date_str} already recorded as delivered. No action needed.")
        return True

    # Resolve GitHub API credentials
    gh_token = get_github_token()
    repo = os.environ.get("GITHUB_REPOSITORY", "iknoest/fair-feeder")

    if not gh_token:
        err = "⚠️ GITHUB_TOKEN not configured. Cannot query GitHub API or dispatch workflow."
        print(err)
        return False

    import requests

    headers = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 1. Query recent workflow runs
    runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/morning-report.yml/runs?per_page=10"
    try:
        r = requests.get(runs_url, headers=headers, timeout=10)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            for run in runs:
                status = run.get("status")
                conclusion = run.get("conclusion")
                created_at = run.get("created_at", "")
                amsterdam_dt = parse_github_datetime_amsterdam(created_at)

                if amsterdam_dt and amsterdam_dt.strftime("%Y%m%d") == date_str:
                    if status in ["in_progress", "queued"]:
                        print(f"ℹ️ Workflow run {run.get('id')} for {date_str} is currently {status}. Watchdog will not duplicate.")
                        return True
                    if conclusion == "success" and not force_dispatch:
                        print(f"✅ Successful workflow run {run.get('id')} found for {date_str}.")
                        record_report_delivered(date_str, {"github_run_id": run.get("id")})
                        return True
    except Exception as e:
        print(f"⚠️ Error querying GitHub Actions API: {e}")

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
