import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import schedule_log


def test_schedule_time_uses_amsterdam_dst_for_summer():
    assert schedule_log.schedule_time_for_date("2026-05-02", "0 3 * * *") == "2026-05-02 05:00:00"


def test_schedule_time_uses_amsterdam_dst_for_winter():
    assert schedule_log.schedule_time_for_date("2026-01-15", "0 3 * * *") == "2026-01-15 04:00:00"


def test_start_time_from_utc_is_converted_to_amsterdam():
    assert schedule_log.utc_to_amsterdam_text("2026-05-02T05:26:00Z") == "2026-05-02 07:26:00"


def test_run_lookup_backfills_rows_by_amsterdam_start_date():
    runs = [
        {"event": "schedule", "run_started_at": "2026-05-02T05:26:00Z"},
        {"event": "workflow_dispatch", "run_started_at": "2026-05-03T08:00:00Z"},
    ]

    lookup = schedule_log.build_schedule_lookup(runs, "0 3 * * *")

    assert lookup == {
        "2026-05-02": {
            "schedule_time": "2026-05-02 05:00:00",
            "start_time": "2026-05-02 07:26:00",
        }
    }
