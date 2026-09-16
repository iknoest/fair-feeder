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


def test_evidence_album_partial_delivery_and_retry(tmp_path):
    """
    Verifies that evidence_album is tracked as a required delivery item:
    1. summary alone -> not fully delivered
    2. summary + evidence_album -> not fully delivered
    3. summary + evidence_album + combined_video -> fully delivered
    4. retry sends nothing twice
    """
    from scripts.delivery_ledger import (
        init_registry_data,
        record_unified_item_delivered,
        is_breakfast_fully_delivered,
        is_unified_item_delivered,
        commit_breakfast_completion
    )
    registry = init_registry_data()
    date = "20260916"

    # Initially not delivered
    assert not is_breakfast_fully_delivered(registry, date)

    # 1. Summary delivered
    record_unified_item_delivered(None, "dummy_folder", registry, date, "summary", local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(registry, date, "summary") is True
    assert is_unified_item_delivered(registry, date, "evidence_album") is False
    assert is_unified_item_delivered(registry, date, "combined_video") is False
    assert not is_breakfast_fully_delivered(registry, date)

    # Committing completion should fail-closed because evidence_album and combined_video are missing
    committed = commit_breakfast_completion(
        None, "dummy_folder", date,
        required_items=["summary", "evidence_album", "combined_video"],
        local_fallback_dir=tmp_path
    )
    assert committed is False
    assert not is_breakfast_fully_delivered(registry, date)

    # 2. Evidence album delivered
    record_unified_item_delivered(
        None, "dummy_folder", registry, date, "evidence_album",
        extra={"evidence_count": 5}, local_fallback_dir=tmp_path
    )
    assert is_unified_item_delivered(registry, date, "evidence_album") is True
    assert is_unified_item_delivered(registry, date, "combined_video") is False
    assert not is_breakfast_fully_delivered(registry, date)

    # Committing completion should still fail-closed because combined_video is missing
    committed = commit_breakfast_completion(
        None, "dummy_folder", date,
        required_items=["summary", "evidence_album", "combined_video"],
        local_fallback_dir=tmp_path
    )
    assert committed is False

    # 3. Combined video delivered
    record_unified_item_delivered(None, "dummy_folder", registry, date, "combined_video", local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(registry, date, "combined_video") is True

    # Now completion succeeds
    committed = commit_breakfast_completion(
        None, "dummy_folder", date,
        required_items=["summary", "evidence_album", "combined_video"],
        local_fallback_dir=tmp_path
    )
    assert committed is True
    assert is_breakfast_fully_delivered(registry, date)


def test_evidence_album_revision_tracking(tmp_path):
    """Verifies that evidence_album is supported under explicit revision/correction identifiers."""
    from scripts.delivery_ledger import (
        init_registry_data,
        record_unified_item_delivered,
        is_breakfast_fully_delivered,
        is_unified_item_delivered,
        commit_breakfast_completion
    )
    registry = init_registry_data()
    date = "20260916"
    rev = "correction-album-1"

    assert not is_breakfast_fully_delivered(registry, date, revision=rev)

    record_unified_item_delivered(None, "dummy_folder", registry, date, "summary", revision=rev, local_fallback_dir=tmp_path)
    record_unified_item_delivered(None, "dummy_folder", registry, date, "evidence_album", revision=rev, local_fallback_dir=tmp_path)
    assert not is_breakfast_fully_delivered(registry, date, revision=rev)

    record_unified_item_delivered(None, "dummy_folder", registry, date, "combined_video", revision=rev, local_fallback_dir=tmp_path)
    committed = commit_breakfast_completion(
        None, "dummy_folder", date,
        required_items=["summary", "evidence_album", "combined_video"],
        revision=rev, local_fallback_dir=tmp_path
    )
    assert committed is True
    assert is_breakfast_fully_delivered(registry, date, revision=rev)
    # Original scheduled date entry remains unaffected
    assert not is_breakfast_fully_delivered(registry, date, revision=None)

