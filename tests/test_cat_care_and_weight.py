"""
tests/test_cat_care_and_weight.py - Comprehensive Unit & Integration Tests for Cat Care + Weight Recovery

Tests:
- Weight recovery merge, dedupe, conflict guard, historical preservation, Sep 6 3.50kg preservation,
  canonical source parity, chart historical range, and remote sync durability.
- Profile & Care calculations: Dan/Sanbo DOB/age/birthday, D-2/D-1/D0, no D+1, calendar recurrence,
  Done/Skip/NotYet/SetLastDate semantics, 3-night reminder cap, terminal push suppression, idempotency,
  double-tap safety, stale callback protection, reboot persistence, routing, and singleton polling.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, MagicMock

from scripts.weight_store import (
    validate_weight_record,
    merge_weight_rows,
    load_weights,
    save_weights,
    get_cat_weight_summary,
    sync_weight_to_drive,
    WeightConflictError,
    WeightValidationError,
)
from scripts.care_store import (
    CareStore,
    compute_age,
    compute_next_birthday,
    get_birthday_reminder,
    add_calendar_months,
    CAT_CONFIGS,
    MAX_REMINDER_NIGHTS,
)
from scripts.care_reminder_scheduler import (
    CareReminderEvaluator,
    get_telegram_config,
)
from scripts.telegram_control_service import (
    TelegramControlService,
    acquire_host_lock,
    HostLockError,
)


# ═════════════════════════════════════════════════════════════════════
# 1. Weight Tests
# ═════════════════════════════════════════════════════════════════════

def test_legacy_plus_new_weight_merge():
    """1. Legacy historical CSV + new repo CSV merge without data loss."""
    legacy_rows = [
        {"date": "2026-04-11", "cat": "dan", "weight_kg": "3.40"},
        {"date": "2026-04-11", "cat": "sanbo", "weight_kg": "3.50"},
        {"date": "2026-05-02", "cat": "dan", "weight_kg": "3.60"},
    ]
    new_rows = [
        {"date": "2026-09-06", "cat": "dan", "weight_kg": "3.50"},
        {"date": "2026-09-06", "cat": "sanbo", "weight_kg": "3.50"},
    ]
    merged = merge_weight_rows(legacy_rows, new_rows)
    assert len(merged) == 5
    dates = [r["date"] for r in merged]
    assert dates == ["2026-04-11", "2026-04-11", "2026-05-02", "2026-09-06", "2026-09-06"]


def test_exact_duplicate_deduplication():
    """2. Exact duplicate rows are deduplicated safely."""
    ds1 = [
        {"date": "2026-05-28", "cat": "dan", "weight_kg": "3.70"},
        {"date": "2026-05-28", "cat": "sanbo", "weight_kg": "3.80"},
    ]
    ds2 = [
        {"date": "2026-05-28", "cat": "dan", "weight_kg": "3.70"},
        {"date": "2026-05-28", "cat": "sanbo", "weight_kg": "3.80"},
    ]
    merged = merge_weight_rows(ds1, ds2)
    assert len(merged) == 2


def test_conflicting_same_cat_date_fails_safely():
    """3. Conflicting values for the same (cat, date) fail safely without mutation."""
    ds1 = [{"date": "2026-07-04", "cat": "dan", "weight_kg": "3.50"}]
    ds2 = [{"date": "2026-07-04", "cat": "dan", "weight_kg": "3.90"}]  # Conflict!
    with pytest.raises(WeightConflictError) as exc_info:
        merge_weight_rows(ds1, ds2)
    assert "Weight conflict for Dan on 2026-07-04" in str(exc_info.value)


def test_historical_rows_preserved():
    """4. Historical rows (April through August) are fully preserved in canonical store."""
    rows = load_weights()
    april_rows = [r for r in rows if r["date"].startswith("2026-04")]
    august_rows = [r for r in rows if r["date"].startswith("2026-08")]
    assert len(april_rows) == 2
    assert len(august_rows) == 2


def test_todays_valid_350_entries_preserved():
    """5. Today's valid 3.50 kg entries for Dan and Sanbo on 2026-09-06 are preserved."""
    rows = load_weights()
    sep6_rows = [r for r in rows if r["date"] == "2026-09-06"]
    assert len(sep6_rows) == 2
    dan_sep6 = next(r for r in sep6_rows if r["cat"] == "dan")
    sanbo_sep6 = next(r for r in sep6_rows if r["cat"] == "sanbo")
    assert dan_sep6["weight_kg"] == "3.50"
    assert sanbo_sep6["weight_kg"] == "3.50"


def test_canonical_source_parity(tmp_path):
    """6. All weight functions read and write to the same canonical source."""
    test_csv = tmp_path / "weight_log.csv"
    with patch.dict(os.environ, {"WEIGHT_LOG_PATH": str(test_csv)}):
        save_weights([{"date": "2026-09-06", "cat": "dan", "weight_kg": "3.50"}], sync_drive=False)
        loaded = load_weights()
        assert len(loaded) == 1
        assert loaded[0]["cat"] == "dan"

        summary = get_cat_weight_summary("dan")
        assert summary["latest_weight"] == 3.50
        assert summary["latest_date"] == "2026-09-06"


def test_chart_includes_historical_dates():
    """7. Chart / history includes all historical dates across the full date range."""
    summary_dan = get_cat_weight_summary("dan")
    summary_sanbo = get_cat_weight_summary("sanbo")
    assert summary_dan["count"] >= 7
    assert summary_sanbo["count"] >= 7
    dan_dates = [r["date"] for r in summary_dan["history"]]
    assert "2026-04-11" in dan_dates
    assert "2026-09-06" in dan_dates


def test_remote_sync_failure_does_not_claim_success(tmp_path):
    """8. Remote sync failure does not silently become durable success."""
    test_csv = tmp_path / "weight_log.csv"
    test_csv.write_text("date,cat,weight_kg\n2026-09-06,dan,3.50\n")

    with patch("subprocess.run") as mock_run:
        # Mock rclone returning error code 1
        mock_run.return_value = MagicMock(returncode=1, stderr="network timeout")
        success = sync_weight_to_drive(test_csv, remote="gdrive-randomdice:")
        assert success is False


# ═════════════════════════════════════════════════════════════════════
# 2. Profile & Care Tests
# ═════════════════════════════════════════════════════════════════════

def test_dan_dob_age_and_next_birthday():
    """1. Dan DOB 2020-12-09, age, and next birthday calculations."""
    dob = date(2020, 12, 9)
    ref = date(2026, 9, 6)
    age = compute_age(dob, ref)
    assert age == 5

    next_bday, days_rem, turns = compute_next_birthday(dob, ref)
    assert next_bday == date(2026, 12, 9)
    assert days_rem == 94
    assert turns == 6


def test_sanbo_dob_age_and_next_birthday():
    """2. Sanbo DOB 2025-05-25, age, and next birthday calculations."""
    dob = date(2025, 5, 25)
    ref = date(2026, 9, 6)
    age = compute_age(dob, ref)
    assert age == 1

    next_bday, days_rem, turns = compute_next_birthday(dob, ref)
    assert next_bday == date(2027, 5, 25)
    assert days_rem == 261
    assert turns == 2


def test_birthday_reminder_d_minus_2():
    """3. Birthday D-2 reminder triggers exactly 2 days before birthday."""
    d_minus_2 = date(2026, 12, 7)
    rem = get_birthday_reminder("dan", as_of=d_minus_2)
    assert rem is not None
    assert rem["tier"] == "D-2"
    assert rem["text"] == "🎂 Dan's birthday is in 2 days"


def test_birthday_reminder_d_minus_1():
    """4. Birthday D-1 reminder triggers exactly 1 day before birthday."""
    d_minus_1 = date(2026, 12, 8)
    rem = get_birthday_reminder("dan", as_of=d_minus_1)
    assert rem is not None
    assert rem["tier"] == "D-1"
    assert rem["text"] == "🎂 Dan's birthday is tomorrow"


def test_birthday_reminder_d0():
    """5. Birthday D0 reminder triggers on the birthday."""
    d0 = date(2026, 12, 9)
    rem = get_birthday_reminder("dan", as_of=d0)
    assert rem is not None
    assert rem["tier"] == "D0"
    assert rem["text"] == "🎉 Dan turns 6 today"


def test_no_birthday_reminder_d_plus_1():
    """6. No birthday reminder on D+1 or ordinary days."""
    d_plus_1 = date(2026, 12, 10)
    assert get_birthday_reminder("dan", as_of=d_plus_1) is None
    assert get_birthday_reminder("dan", as_of=date(2026, 9, 6)) is None


def test_calendar_recurrence_deflea_monthly():
    """7. Monthly calendar recurrence for deflea with month-end clamping."""
    # Standard addition
    assert add_calendar_months(date(2026, 9, 6), 1) == date(2026, 10, 6)
    # Month-end clamping: Sep 30 + 1m -> Oct 30
    assert add_calendar_months(date(2026, 9, 30), 1) == date(2026, 10, 30)
    # Leap year / month-end clamping: Jan 31 + 1m -> Feb 28
    assert add_calendar_months(date(2026, 1, 31), 1) == date(2026, 2, 28)


def test_calendar_recurrence_deworm_three_months():
    """8. 3-month calendar recurrence for deworm."""
    assert add_calendar_months(date(2026, 9, 6), 3) == date(2026, 12, 6)
    # Aug 31 + 3m -> Nov 30 (Nov has 30 days)
    assert add_calendar_months(date(2026, 8, 31), 3) == date(2026, 11, 30)


def test_done_uses_actual_completion_date(tmp_path):
    """9. Done uses actual completion date for next due calculation."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    # Done completed on 2026-09-10 (originating due was 2026-09-01)
    act_d = date(2026, 9, 10)
    res = store.record_done("dan", "deflea", actual_date=act_d, originating_due="2026-09-01", sync_drive=False)
    assert res["status"] == "success"
    assert res["next_due"] == "2026-10-10"  # 2026-09-10 + 1 month


def test_skip_uses_planned_due_date(tmp_path):
    """10. Skip advances from planned due occurrence date."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    # Planned due Sep 1, skipped on Sep 3
    res = store.record_skip("dan", "deflea", planned_due="2026-09-01", sync_drive=False)
    assert res["status"] == "success"
    assert res["next_due"] == "2026-10-01"  # 2026-09-01 + 1 month


def test_unknown_baseline_does_not_invent_treatment(tmp_path):
    """11. Unknown baseline does not invent historical treatment."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)
    p = store.get_cat_profile("dan")
    assert p["treatments"]["deflea"]["baseline_status"] == "UNKNOWN"
    assert p["treatments"]["deflea"]["last_completed"] == "Unknown"
    assert p["treatments"]["deflea"]["next_due"] == "Unknown"
    assert p["treatments"]["deflea"]["display_str"] == "last date unknown"


def test_set_last_date_establishes_baseline(tmp_path):
    """12. Set-last-date establishes baseline from user-provided date."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    res = store.set_last_date("dan", "deflea", "2026-08-15", sync_drive=False)
    assert res["status"] == "success"
    assert res["next_due"] == "2026-09-15"

    p = store.get_cat_profile("dan")
    assert p["treatments"]["deflea"]["baseline_status"] == "ESTABLISHED"
    assert p["treatments"]["deflea"]["last_completed"] == "2026-08-15"
    assert p["treatments"]["deflea"]["next_due"] == "2026-09-15"


def test_not_yet_schedules_next_night(tmp_path):
    """13. Not yet increments reminder night count and schedules next night."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    res = store.record_not_yet("dan", "deflea", sync_drive=False)
    assert res["status"] == "snoozed"
    assert res["night_count"] == 1
    assert res["push_suppressed"] is False


def test_maximum_three_reminder_nights(tmp_path):
    """14 & 15. Maximum 3 reminder nights, after which push is suppressed and profile remains OVERDUE."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    # Establish baseline due in the past
    store.set_last_date("dan", "deflea", "2026-07-01", sync_drive=False)  # next due was 2026-08-01

    # Night 1 Not yet
    res1 = store.record_not_yet("dan", "deflea", sync_drive=False)
    assert res1["night_count"] == 1
    assert res1["push_suppressed"] is False

    # Night 2 Not yet
    res2 = store.record_not_yet("dan", "deflea", sync_drive=False)
    assert res2["night_count"] == 2
    assert res2["push_suppressed"] is False

    # Night 3 Not yet
    res3 = store.record_not_yet("dan", "deflea", sync_drive=False)
    assert res3["night_count"] == 3
    assert res3["push_suppressed"] is True

    # Check profile on 2026-09-06: visibly OVERDUE
    p = store.get_cat_profile("dan", as_of=date(2026, 9, 6))
    assert p["treatments"]["deflea"]["status_label"] == "OVERDUE"
    assert p["treatments"]["deflea"]["terminal_push_suppressed"] is True


def test_duplicate_evaluator_invocation_is_idempotent(tmp_path):
    """16. Duplicate timer invocation sends reminder once."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    evaluator = CareReminderEvaluator(store=store, dry_run=False)
    with patch.object(evaluator, "token", "mock_token"), \
         patch.object(evaluator, "target_chat", "mock_chat"), \
         patch("scripts.care_reminder_scheduler.send_telegram_message", return_value=True), \
         patch.object(store, "sync_to_drive", return_value=True):
        actions1 = evaluator.evaluate(as_of=date(2026, 9, 6))
        assert len(actions1) == 1
        assert actions1[0]["type"] == "care_baseline_needed"

        # Immediately evaluate again on same night
        actions2 = evaluator.evaluate(as_of=date(2026, 9, 6))
        assert len(actions2) == 0  # Idempotent!


def test_double_done_callback_creates_one_event(tmp_path):
    """17. Double Done callback creates one care event."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)

    res1 = store.record_done("sanbo", "deworm", actual_date=date(2026, 9, 6), sync_drive=False)
    assert res1["status"] == "success"

    # Double tap
    res2 = store.record_done("sanbo", "deworm", actual_date=date(2026, 9, 6), sync_drive=False)
    assert res2["status"] == "already_done"

    data = store.load()
    deworm_events = [e for e in data["care_events"] if e["cat"] == "sanbo" and e["type"] == "deworm"]
    assert len(deworm_events) == 1


def test_reboot_preserves_pending_reminder(tmp_path):
    """19. Reboot reconstruction preserves pending reminder and count."""
    care_file = tmp_path / "cat_care.json"
    store = CareStore(path=care_file)
    store.record_not_yet("dan", "deworm", sync_drive=False)

    # Simulate fresh process restart reading from disk
    new_store = CareStore(path=care_file)
    data = new_store.load()
    assert data["reminder_state"]["dan"]["deworm"]["reminder_night_count"] == 1
    assert data["reminder_state"]["dan"]["deworm"]["snoozed"] is True


def test_profile_direct_weight_and_care_access():
    """20. Profile includes direct weight + care history access."""
    store = CareStore()
    p = store.get_cat_profile("dan")
    assert "treatments" in p
    assert "all_events" in p
    assert "dob" in p
    assert "age" in p

    w = get_cat_weight_summary("dan")
    assert "history" in w
    assert len(w["history"]) >= 7


def test_destination_routing_feeder_monitor_group():
    """22 & 23. Care reminder goes to configured feeder monitor group; control replies to sender."""
    with patch.dict(os.environ, {"ALLOWED_GROUP_ID": "-1001234567890", "TELEGRAM_CHAT_ID": "987654321"}):
        token, target_group = get_telegram_config()
        assert target_group == "-1001234567890"

        service = TelegramControlService(bot_token="test_token", chat_id="987654321", allowed_group_id="-1001234567890")
        assert "987654321" in service.allowed_chats
        assert "-1001234567890" in service.allowed_chats


def test_host_singleton_lock(tmp_path):
    """24. No duplicate Telegram getUpdates poller is created on host."""
    lock_file = tmp_path / "test_host.lock"
    f1 = acquire_host_lock(str(lock_file))
    assert f1 is not None

    with pytest.raises(HostLockError):
        acquire_host_lock(str(lock_file))

    f1.close()
