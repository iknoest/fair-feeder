import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.daily_watchdog import (
    is_report_delivered,
    record_report_delivered,
    check_and_recover_report
)


@pytest.fixture(autouse=True)
def isolated_tracker(tmp_path, monkeypatch):
    tracker_file = tmp_path / "delivered_reports.json"
    monkeypatch.setattr("scripts.daily_watchdog.REPORT_TRACKER_FILE", tracker_file)
    monkeypatch.setenv("GITHUB_TOKEN", "fake_gh_token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "iknoest/fair-feeder")
    yield tracker_file


def test_idempotent_report_recording_and_check():
    assert is_report_delivered("20260827") is False

    record_report_delivered("20260827", {"source": "test"})
    assert is_report_delivered("20260827") is True


def test_watchdog_does_not_duplicate_when_already_delivered():
    record_report_delivered("20260827", {"source": "test"})

    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        success = check_and_recover_report("20260827")
        assert success is True
        mock_get.assert_not_called()
        mock_post.assert_not_called()


def test_watchdog_does_not_duplicate_when_run_is_in_progress():
    mock_runs_resp = MagicMock()
    mock_runs_resp.status_code = 200
    mock_runs_resp.json.return_value = {
        "workflow_runs": [
            {"id": 12345, "status": "in_progress", "conclusion": None, "created_at": "2026-08-27T07:15:00Z"}
        ]
    }

    with patch("requests.get", return_value=mock_runs_resp), \
         patch("requests.post") as mock_post:

        success = check_and_recover_report("20260827")
        assert success is True
        # Must not dispatch duplicate run
        mock_post.assert_not_called()


def test_watchdog_dispatches_workflow_when_report_missing():
    mock_runs_resp = MagicMock()
    mock_runs_resp.status_code = 200
    mock_runs_resp.json.return_value = {"workflow_runs": []}

    mock_dispatch_resp = MagicMock()
    mock_dispatch_resp.status_code = 204

    with patch("requests.get", return_value=mock_runs_resp), \
         patch("requests.post", return_value=mock_dispatch_resp) as mock_post:

        success = check_and_recover_report("20260827")
        assert success is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "dispatches" in args[0]
        assert kwargs["json"]["inputs"]["date_override"] == "20260827"
