from datetime import datetime, timezone
from zoneinfo import ZoneInfo


AMS_TZ = ZoneInfo("Europe/Amsterdam")


def _parse_utc(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def utc_to_amsterdam_text(value):
    dt = _parse_utc(value)
    if dt is None:
        return ""
    return dt.astimezone(AMS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def schedule_time_for_date(date_text, cron):
    if not date_text or not cron:
        return ""
    try:
        minute, hour, *_ = cron.split()
        local_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        utc_dt = datetime(
            local_date.year, local_date.month, local_date.day,
            int(hour), int(minute), tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return ""
    return utc_dt.astimezone(AMS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def current_schedule_fields(env, date_text):
    cron = env.get("GHA_SCHEDULE_CRON", "")
    return {
        "schedule_time": schedule_time_for_date(date_text, cron),
        "start_time": utc_to_amsterdam_text(env.get("GHA_JOB_STARTED_AT_UTC", "")),
    }


def build_schedule_lookup(runs, cron):
    lookup = {}
    for run in runs:
        if run.get("event") != "schedule":
            continue
        started = run.get("run_started_at") or run.get("created_at")
        start_text = utc_to_amsterdam_text(started)
        if not start_text:
            continue
        date_text = start_text[:10]
        lookup[date_text] = {
            "schedule_time": schedule_time_for_date(date_text, cron),
            "start_time": start_text,
        }
    return lookup


def fetch_schedule_lookup(token, repo, workflow_file, cron):
    if not token or not repo:
        return {}
    import requests

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params = {
        "event": "schedule",
        "per_page": 100,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            print(f"GitHub run lookup skipped: HTTP {r.status_code} {r.text[:120]}")
            return {}
        return build_schedule_lookup(r.json().get("workflow_runs", []), cron)
    except Exception as e:
        print(f"GitHub run lookup skipped: {e}")
        return {}
