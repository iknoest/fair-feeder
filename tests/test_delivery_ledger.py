import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.delivery_ledger import (
    get_ledger_filename,
    init_ledger_data,
    load_delivery_ledger,
    save_delivery_ledger,
    is_item_delivered,
    record_item_delivered,
    is_camera_fully_delivered,
    commit_camera_completion,
    save_durable_artifact,
    load_durable_artifact
)


def test_ledger_filename_and_init():
    fn = get_ledger_filename("2026-09-01", "logitech")
    assert fn == "delivery_ledger_20260901_LOGITECH.json"

    data = init_ledger_data("20260901", "TAPO")
    assert data["date"] == "20260901"
    assert data["camera"] == "TAPO"
    assert data["analysis_completed"] is False
    assert data["camera_fully_delivered"] is False
    assert data["items"] == {}


def test_local_save_and_load(tmp_path):
    ledger = init_ledger_data("20260901", "LOGITECH")
    save_delivery_ledger(None, None, ledger, local_fallback_dir=tmp_path)

    loaded = load_delivery_ledger(None, None, "20260901", "LOGITECH", local_fallback_dir=tmp_path)
    assert loaded["date"] == "20260901"
    assert loaded["camera"] == "LOGITECH"


def test_partial_delivery_and_resume(tmp_path):
    ledger = init_ledger_data("20260901", "LOGITECH")
    assert is_item_delivered(ledger, "summary") is False
    assert is_item_delivered(ledger, "video_session_1") is False
    assert is_item_delivered(ledger, "video_session_2") is False

    # Step 1: Deliver summary and video 1
    record_item_delivered(None, None, ledger, "summary", message_id=101, local_fallback_dir=tmp_path)
    record_item_delivered(None, None, ledger, "video_session_1", message_id=102, local_fallback_dir=tmp_path)

    # Required items: summary, video_session_1, video_session_2
    required = ["summary", "video_session_1", "video_session_2"]
    fully_delivered = commit_camera_completion(None, None, ledger, required_items=required, local_fallback_dir=tmp_path)
    assert fully_delivered is False
    assert is_camera_fully_delivered(ledger) is False

    # Step 2: Retry runner loads ledger
    reloaded = load_delivery_ledger(None, None, "20260901", "LOGITECH", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(reloaded) is False
    assert is_item_delivered(reloaded, "summary") is True
    assert is_item_delivered(reloaded, "video_session_1") is True
    assert is_item_delivered(reloaded, "video_session_2") is False

    # Summary and video 1 are skipped; only video 2 is sent
    record_item_delivered(None, None, reloaded, "video_session_2", message_id=103, local_fallback_dir=tmp_path)
    fully_delivered = commit_camera_completion(None, None, reloaded, required_items=required, local_fallback_dir=tmp_path)
    assert fully_delivered is True
    assert is_camera_fully_delivered(reloaded) is True

    # Step 3: Third runner checks preflight
    preflight = load_delivery_ledger(None, None, "20260901", "LOGITECH", local_fallback_dir=tmp_path)
    assert is_camera_fully_delivered(preflight) is True


def test_durable_artifact_save_and_load(tmp_path):
    payload = b'{"feeding_phases": [{"start": "06:20", "end": "06:22"}]}'
    save_durable_artifact(None, None, "tapo_timeline_20260901.json", payload, local_fallback_dir=tmp_path)

    loaded = load_durable_artifact(None, None, "tapo_timeline_20260901.json", local_fallback_dir=tmp_path)
    assert loaded == payload
