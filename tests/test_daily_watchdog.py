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


def test_parse_github_datetime_amsterdam():
    from scripts.daily_watchdog import parse_github_datetime_amsterdam
    # 05:00 UTC during CEST is 07:00 Amsterdam (same date)
    dt1 = parse_github_datetime_amsterdam("2026-08-29T05:00:00Z")
    assert dt1 is not None
    assert dt1.strftime("%Y%m%d") == "20260829"
    assert dt1.hour == 7

    # 23:00 UTC Aug 28 during CEST is 01:00 Amsterdam Aug 29!
    dt2 = parse_github_datetime_amsterdam("2026-08-28T23:00:00Z")
    assert dt2 is not None
    assert dt2.strftime("%Y%m%d") == "20260829"
    assert dt2.hour == 1


def test_get_github_token_from_infisical(monkeypatch):
    from scripts.daily_watchdog import get_github_token
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("INFISICAL_ID", "mock_id")
    monkeypatch.setenv("INFISICAL_SECRET", "mock_secret")
    monkeypatch.setenv("INFISICAL_PROJECT_ID", "mock_proj")

    mock_login = MagicMock(status_code=200)
    mock_login.json.return_value = {"accessToken": "fake_access_token"}

    mock_secrets = MagicMock(status_code=200)
    mock_secrets.json.return_value = {
        "secrets": [
            {"secretKey": "GITHUB_TOKEN", "secretValue": "infisical_gh_token_123"}
        ]
    }

    def fake_post(url, **kwargs):
        return mock_login

    def fake_get(url, **kwargs):
        return mock_secrets

    with patch("requests.post", side_effect=fake_post), patch("requests.get", side_effect=fake_get):
        token = get_github_token()
        assert token == "infisical_gh_token_123"

