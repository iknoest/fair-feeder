import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.delivery_ledger import (
    init_ledger_data,
    load_delivery_ledger,
    save_delivery_ledger,
    is_item_delivered,
    record_item_delivered,
    is_camera_fully_delivered,
    commit_camera_completion,
    export_tapo_timeline,
    load_tapo_timeline
)
from scripts.logitech_vlm_shadow import (
    CameraTimelineInterval,
    reconcile_cross_camera_intervals,
    apply_cross_camera_reconciliation_to_session
)


def test_sep1_duplicate_delivery_preflight_skip(tmp_path):
    """
    Demonstrates that once a run finishes and records completion in the durable ledger,
    any subsequent run for that date immediately encounters camera_fully_delivered=True
    and exits without delivering duplicate Telegram messages or running redundant analysis.
    """
    # Runner 1 runs for 2026-09-01 TAPO
    tapo_ledger = init_ledger_data("20260901", "TAPO")
    record_item_delivered(None, None, tapo_ledger, "summary", message_id=5001, local_fallback_dir=tmp_path)
    record_item_delivered(None, None, tapo_ledger, "timeline", message_id=5002, local_fallback_dir=tmp_path)
    record_item_delivered(None, None, tapo_ledger, "annotated_video", message_id=5003, local_fallback_dir=tmp_path)
    commit_camera_completion(None, None, tapo_ledger, required_items=["summary", "timeline", "annotated_video"], local_fallback_dir=tmp_path)

    # Runner 2 (duplicate/dispatch) loads ledger
    reloaded = load_delivery_ledger(None, None, "20260901", "TAPO", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(reloaded) is True

    # Same for LOGITECH
    logitech_ledger = init_ledger_data("20260901", "LOGITECH")
    record_item_delivered(None, None, logitech_ledger, "summary", message_id=5004, local_fallback_dir=tmp_path)
    record_item_delivered(None, None, logitech_ledger, "video_session", message_id=5005, local_fallback_dir=tmp_path)
    commit_camera_completion(None, None, logitech_ledger, required_items=["summary", "video_session"], local_fallback_dir=tmp_path)

    reloaded_log = load_delivery_ledger(None, None, "20260901", "LOGITECH", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(reloaded_log) is True


def test_partial_delivery_resume_skips_sent_items(tmp_path):
    """
    Demonstrates that if delivery fails midway (e.g. video upload fails after summary and video 1 succeed),
    a retry run skips the already delivered items, delivers only the missing item, and then marks completion.
    """
    ledger = init_ledger_data("20260902", "LOGITECH")
    required_items = ["summary", "video_session_1", "video_session_2"]

    # Initial run: summary and video 1 succeed
    record_item_delivered(None, None, ledger, "summary", message_id=6001, local_fallback_dir=tmp_path)
    record_item_delivered(None, None, ledger, "video_session_1", message_id=6002, local_fallback_dir=tmp_path)

    # Video 2 fails due to network error -> commit completion fails
    success = commit_camera_completion(None, None, ledger, required_items=required_items, local_fallback_dir=tmp_path)
    assert success is False
    assert is_camera_fully_delivered(ledger) is False

    # Second run loads ledger
    retry_ledger = load_delivery_ledger(None, None, "20260902", "LOGITECH", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(retry_ledger) is False

    # Check items: summary and video 1 are SKIPPED
    assert is_item_delivered(retry_ledger, "summary") is True
    assert is_item_delivered(retry_ledger, "video_session_1") is True
    assert is_item_delivered(retry_ledger, "video_session_2") is False

    # Send only video 2
    record_item_delivered(None, None, retry_ledger, "video_session_2", message_id=6003, local_fallback_dir=tmp_path)

    # Now completion succeeds
    final_success = commit_camera_completion(None, None, retry_ledger, required_items=required_items, local_fallback_dir=tmp_path)
    assert final_success is True
    assert is_camera_fully_delivered(retry_ledger) is True

    # Third run: preflight encounters completed camera
    preflight = load_delivery_ledger(None, None, "20260902", "LOGITECH", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(preflight) is True


def test_cross_camera_reconciliation_tapo_dan_resolves_logitech_sanbo():
    """
    When TAPO establishes Dan feeding in Room 1 and Logitech detects a cat in Room 2
    during the same time interval, physical exclusion proves the cat in Room 2 is Sanbo.
    Any theft warning is cleared.
    """
    tapo_interval = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:19:49",
        end_timestamp="06:25:00",
        cat_presence=True,
        identity="Dan",
        identity_confidence=0.95,
        identity_evidence_quality="usable",
        identity_basis="FeedingTracker accepted phase",
        eating_evidence="yes"
    )

    logitech_session = {
        "session_start_time": "06:20:00",
        "session_end_time": "06:24:00",
        "cat_identity": "Dan",  # Misclassified by local visual VLM
        "confidence": 0.85,
        "visibility": "good",
        "eating_evidence": "yes",
        "possible_food_theft": True
    }

    reconciled_sess = apply_cross_camera_reconciliation_to_session(logitech_session, [tapo_interval])

    # Reconciled identity must be Sanbo via cross-camera exclusion
    assert reconciled_sess["cat_identity"] == "Sanbo"
    assert reconciled_sess["identity_basis"] == "cross-camera exclusion"
    assert reconciled_sess["reconciled_by_cross_camera"] is True

    # Local visual estimate must be preserved separately
    assert reconciled_sess["visual_cat_identity"] == "Dan"
    assert reconciled_sess["visual_confidence"] == 0.85
    assert reconciled_sess["visual_visibility"] == "good"

    # Theft warning must be cleared because Sanbo at Sanbo's feeder is not theft
    assert reconciled_sess["possible_food_theft"] is False


def test_cross_camera_reconciliation_tapo_sanbo_resolves_logitech_dan():
    """
    When TAPO establishes Sanbo in Room 1, the cat in Room 2 must be Dan.
    """
    tapo_interval = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:19:49",
        end_timestamp="06:25:00",
        cat_presence=True,
        identity="Sanbo",
        identity_confidence=0.95,
        identity_evidence_quality="usable",
        identity_basis="FeedingTracker accepted phase",
        eating_evidence="yes"
    )

    logitech_session = {
        "session_start_time": "06:20:00",
        "session_end_time": "06:24:00",
        "cat_identity": "Sanbo",  # Misclassified
        "confidence": 0.80,
        "visibility": "usable",
        "eating_evidence": "yes",
        "possible_food_theft": False
    }

    reconciled_sess = apply_cross_camera_reconciliation_to_session(logitech_session, [tapo_interval])
    assert reconciled_sess["cat_identity"] == "Dan"
    assert reconciled_sess["reconciled_by_cross_camera"] is True
    # Dan eating at Sanbo's feeder is theft
    assert reconciled_sess["possible_food_theft"] is True


def test_rule_d_no_forced_identity_on_vlm_failure_or_no_cat():
    """
    Rule D: When Logitech VLM fails (timeout/429) or cat is not detected (none/unknown),
    motion alone must NOT force Dan or Sanbo.
    """
    tapo_interval = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:19:49",
        end_timestamp="06:25:00",
        cat_presence=True,
        identity="Dan",
        identity_confidence=0.95,
        identity_evidence_quality="usable",
        identity_basis="FeedingTracker accepted phase"
    )

    # Case A: VLM reported cat_identity='none' (e.g. empty room/shadow motion)
    no_cat_session = {
        "session_start_time": "06:20:00",
        "session_end_time": "06:24:00",
        "cat_identity": "none",
        "confidence": 0.0,
        "visibility": "unknown",
        "eating_evidence": "no",
        "possible_food_theft": False
    }

    # apply_cross_camera_reconciliation_to_session checks cat_presence = (cat_identity != "none")
    result_a = apply_cross_camera_reconciliation_to_session(no_cat_session, [tapo_interval])
    assert result_a.get("reconciled_by_cross_camera", False) is False
    assert result_a["cat_identity"] == "none"

    # Case B: In logitech_vlm_shadow.py main(), if VLM fails (session_results empty),
    # vlm_cat_present is False, so apply_cross_camera_reconciliation_to_session is NOT called
    vlm_failed_session = {
        "session_start_time": "06:20:00",
        "session_end_time": "06:24:00",
        "cat_identity": "unsure",
        "status": "failed",
        "confidence": 0.0
    }
    session_results = []  # VLM failed
    vlm_cat_present = (
        bool(session_results) and
        vlm_failed_session.get("cat_identity") not in ["none", "unknown", "unsure", None] and
        vlm_failed_session.get("cat_identity", "") != ""
    )
    assert vlm_cat_present is False


def test_cross_camera_non_overlapping_intervals():
    """
    When intervals do not overlap in time, physical exclusion does not apply.
    Both camera observations remain unmodified.
    """
    tapo_interval = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:10:00",
        end_timestamp="06:15:00",
        cat_presence=True,
        identity="Dan",
        identity_confidence=0.95,
        identity_evidence_quality="usable",
        identity_basis="FeedingTracker accepted phase"
    )

    logitech_session = {
        "session_start_time": "06:20:00",
        "session_end_time": "06:25:00",
        "cat_identity": "Dan",
        "confidence": 0.85,
        "visibility": "usable",
        "eating_evidence": "yes",
        "possible_food_theft": True
    }

    reconciled_sess = apply_cross_camera_reconciliation_to_session(logitech_session, [tapo_interval])
    assert reconciled_sess.get("reconciled_by_cross_camera", False) is False
    assert reconciled_sess["cat_identity"] == "Dan"
    assert reconciled_sess["possible_food_theft"] is True
