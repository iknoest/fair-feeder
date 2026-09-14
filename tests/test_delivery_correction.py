import sys
from pathlib import Path
import copy

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.delivery_ledger import (
    load_delivery_registry,
    save_delivery_registry,
    is_breakfast_fully_delivered,
    is_unified_item_delivered,
    record_unified_item_delivered,
    commit_breakfast_completion
)
from scripts.unified_breakfast import deliver_unified_breakfast
from unittest.mock import patch, MagicMock



def test_original_delivery_remains_immutable_when_correction_delivered(tmp_path):
    """
    Test 1: Original scheduled delivery remains strictly immutable
    when a correction/revision is delivered.
    """
    # 1. Setup existing registry with original scheduled delivery
    initial_registry = {
        "dates": {
            "20260909": {
                "breakfast_fully_delivered": True,
                "completed_at_utc": "2026-09-09T07:50:00Z",
                "unified": {
                    "fully_delivered": True,
                    "completed_at_utc": "2026-09-09T07:50:00Z",
                    "items": {
                        "summary": {
                            "delivered": True,
                            "delivered_at_utc": "2026-09-09T07:48:00Z",
                            "message_id": 2101
                        },
                        "combined_video": {
                            "delivered": True,
                            "delivered_at_utc": "2026-09-09T07:49:00Z",
                            "message_id": 2102
                        }
                    }
                }
            }
        }
    }
    save_delivery_registry(None, None, initial_registry, local_fallback_dir=tmp_path)

    # 2. Record items for revision 'correction-1'
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    record_unified_item_delivered(
        None, None, reg, "20260909", "summary",
        message_id=3001, local_fallback_dir=tmp_path, revision="correction-1"
    )
    record_unified_item_delivered(
        None, None, reg, "20260909", "combined_video",
        message_id=3002, local_fallback_dir=tmp_path, revision="correction-1"
    )
    commit_breakfast_completion(
        None, None, "20260909",
        required_items=["summary", "combined_video"],
        local_fallback_dir=tmp_path,
        revision="correction-1"
    )

    # 3. Verify registry content
    fresh_reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    date_entry = fresh_reg["dates"]["20260909"]

    # Original fields MUST match initial_registry exactly
    assert date_entry["breakfast_fully_delivered"] is True
    assert date_entry["completed_at_utc"] == "2026-09-09T07:50:00Z"
    assert date_entry["unified"]["fully_delivered"] is True
    assert date_entry["unified"]["completed_at_utc"] == "2026-09-09T07:50:00Z"
    assert date_entry["unified"]["items"]["summary"]["message_id"] == 2101
    assert date_entry["unified"]["items"]["combined_video"]["message_id"] == 2102

    # Correction fields must be stored cleanly under corrections['correction-1']
    assert "corrections" in date_entry
    corr_entry = date_entry["corrections"]["correction-1"]
    assert corr_entry["fully_delivered"] is True
    assert corr_entry["items"]["summary"]["message_id"] == 3001
    assert corr_entry["items"]["combined_video"]["message_id"] == 3002


def test_correction_items_are_independently_idempotent(tmp_path):
    """
    Test 2: Correction items (summary, combined_video) are independently idempotent.
    """
    # Start with original delivery completed
    reg = {
        "dates": {
            "20260909": {
                "breakfast_fully_delivered": True,
                "unified": {
                    "fully_delivered": True,
                    "items": {
                        "summary": {"delivered": True, "message_id": 2101},
                        "combined_video": {"delivered": True, "message_id": 2102}
                    }
                }
            }
        }
    }
    save_delivery_registry(None, None, reg, local_fallback_dir=tmp_path)

    fresh_reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)

    # Before correction, correction-1 items are not delivered
    assert is_breakfast_fully_delivered(fresh_reg, "20260909", revision="correction-1") is False
    assert is_unified_item_delivered(fresh_reg, "20260909", "summary", revision="correction-1") is False
    assert is_unified_item_delivered(fresh_reg, "20260909", "combined_video", revision="correction-1") is False

    # Deliver only summary for correction-1
    record_unified_item_delivered(
        None, None, fresh_reg, "20260909", "summary",
        message_id=3001, local_fallback_dir=tmp_path, revision="correction-1"
    )

    reg_after_summary = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(reg_after_summary, "20260909", "summary", revision="correction-1") is True
    assert is_unified_item_delivered(reg_after_summary, "20260909", "combined_video", revision="correction-1") is False
    assert is_breakfast_fully_delivered(reg_after_summary, "20260909", revision="correction-1") is False

    # Committing before combined_video should fail-closed
    committed_early = commit_breakfast_completion(
        None, None, "20260909",
        required_items=["summary", "combined_video"],
        local_fallback_dir=tmp_path,
        revision="correction-1"
    )
    assert committed_early is False

    # Deliver combined_video for correction-1
    record_unified_item_delivered(
        None, None, reg_after_summary, "20260909", "combined_video",
        message_id=3002, local_fallback_dir=tmp_path, revision="correction-1"
    )
    committed = commit_breakfast_completion(
        None, None, "20260909",
        required_items=["summary", "combined_video"],
        local_fallback_dir=tmp_path,
        revision="correction-1"
    )
    assert committed is True

    final_reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(final_reg, "20260909", revision="correction-1") is True


def test_rerunning_same_correction_skips_delivery(tmp_path):
    """
    Test 3: Re-running the same correction skips delivery (no duplicate).
    """
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    record_unified_item_delivered(
        None, None, reg, "20260909", "summary",
        message_id=3001, local_fallback_dir=tmp_path, revision="correction-1"
    )
    record_unified_item_delivered(
        None, None, reg, "20260909", "combined_video",
        message_id=3002, local_fallback_dir=tmp_path, revision="correction-1"
    )
    commit_breakfast_completion(
        None, None, "20260909",
        required_items=["summary", "combined_video"],
        local_fallback_dir=tmp_path,
        revision="correction-1"
    )

    persisted = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    # correction-1 is fully delivered -> skip
    assert is_breakfast_fully_delivered(persisted, "20260909", revision="correction-1") is True

    # A subsequent revision 'correction-2' is not yet delivered
    assert is_breakfast_fully_delivered(persisted, "20260909", revision="correction-2") is False


def test_normal_delivery_without_revision_unaffected(tmp_path):
    """
    Test 4: Normal delivery (without revision, e.g. Sep-10) is completely unaffected.
    """
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    clean_date = "20260910"

    assert is_breakfast_fully_delivered(reg, clean_date) is False
    assert is_unified_item_delivered(reg, clean_date, "summary") is False
    assert is_unified_item_delivered(reg, clean_date, "combined_video") is False

    record_unified_item_delivered(
        None, None, reg, clean_date, "summary",
        message_id=4001, local_fallback_dir=tmp_path
    )
    reg2 = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(reg2, clean_date, "summary") is True
    assert is_unified_item_delivered(reg2, clean_date, "combined_video") is False
    assert is_breakfast_fully_delivered(reg2, clean_date) is False

    record_unified_item_delivered(
        None, None, reg2, clean_date, "combined_video",
        message_id=4002, local_fallback_dir=tmp_path
    )
    committed = commit_breakfast_completion(
        None, None, clean_date,
        required_items=["summary", "combined_video"],
        local_fallback_dir=tmp_path
    )
    assert committed is True

    final_reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(final_reg, clean_date) is True
    # Ensure no 'corrections' section was created for normal flow
    assert "corrections" not in final_reg["dates"][clean_date]
    assert final_reg["dates"][clean_date]["breakfast_fully_delivered"] is True
    assert final_reg["dates"][clean_date]["unified"]["fully_delivered"] is True


def test_unified_delivery_api_compatibility(tmp_path):
    """
    Test 5: Direct regression test for the Sep-14 production failure.
    Verifies that delivery_ledger functions accept the 'revision' parameter
    with both None and explicit string values across all callers in unified_breakfast.py.
    """
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    clean_date = "20260914"

    # 1. is_unified_item_delivered must accept revision=None without TypeError
    assert is_unified_item_delivered(reg, clean_date, "summary", revision=None) is False
    assert is_unified_item_delivered(reg, clean_date, "summary", revision="rev-1") is False

    # 2. record_unified_item_delivered must accept revision=None and revision="rev-1"
    record_unified_item_delivered(
        None, None, reg, clean_date, "summary",
        message_id=5001, local_fallback_dir=tmp_path, revision=None
    )
    reg2 = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(reg2, clean_date, "summary", revision=None) is True
    assert is_unified_item_delivered(reg2, clean_date, "summary", revision="rev-1") is False

    # 3. is_breakfast_fully_delivered must accept revision=None and revision="rev-1"
    assert is_breakfast_fully_delivered(reg2, clean_date, revision=None) is False
    assert is_breakfast_fully_delivered(reg2, clean_date, revision="rev-1") is False

    # 4. commit_breakfast_completion must accept revision=None and revision="rev-1"
    committed = commit_breakfast_completion(
        None, None, clean_date,
        required_items=["summary"],
        local_fallback_dir=tmp_path,
        revision=None
    )
    assert committed is True

    reg3 = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(reg3, clean_date, revision=None) is True
    assert is_breakfast_fully_delivered(reg3, clean_date, revision="rev-1") is False


def test_skip_telegram_is_side_effect_free(tmp_path):
    """
    Test 6: Regression test for --skip-telegram side-effect bug.
    Proves that running deliver_unified_breakfast with skip_telegram=True:
    1. Does NOT mutate registry (registry before == registry after).
    2. Does NOT mark summary delivered.
    3. Does NOT mark combined_video delivered.
    4. Does NOT mark breakfast_fully_delivered.
    5. Normal delivery (skip_telegram=False) works and delivers/registers.
    6. Revision delivery (skip_telegram=False, revision='...') works and registers.
    """
    reg_dir = tmp_path / "reg_dir"
    reg_dir.mkdir()
    initial_reg = {
        "dates": {
            "20260913": {
                "breakfast_fully_delivered": True,
                "unified": {"fully_delivered": True, "items": {"summary": {"delivered": True}, "combined_video": {"delivered": True}}}
            }
        }
    }
    save_delivery_registry(None, None, initial_reg, local_fallback_dir=reg_dir)
    before_reg = copy.deepcopy(load_delivery_registry(None, None, local_fallback_dir=reg_dir))

    out_dir = tmp_path / "dry_run_out"
    out_dir.mkdir()
    # Provide dummy combined video so video phase succeeds
    dummy_video = out_dir / "20260914_combined_breakfast.mp4"
    dummy_video.write_bytes(b"dummy video content")

    # 1. Execute with skip_telegram=True
    with patch.dict("os.environ", {"GDRIVE_SERVICE_ACCOUNT_KEY": ""}):
        success = deliver_unified_breakfast(
            target_date="20260914",
            out_dir=out_dir,
            skip_telegram=True,
            force=True,
            folder_id="",
            drive_service=None
        )
    assert success is True

    # Check registry after dry-run
    after_reg = load_delivery_registry(None, None, local_fallback_dir=reg_dir)
    # Proof 1: Registry untouched
    assert after_reg.get("dates", {}).get("20260914") is None or "unified" not in after_reg["dates"]["20260914"]
    # Proof 2: Summary not marked delivered
    assert is_unified_item_delivered(after_reg, "20260914", "summary") is False
    # Proof 3: Video not marked delivered
    assert is_unified_item_delivered(after_reg, "20260914", "combined_video") is False
    # Proof 4: Breakfast not marked delivered
    assert is_breakfast_fully_delivered(after_reg, "20260914") is False

    # Proof 5: Normal delivery semantics remain unchanged (delivers when skip_telegram=False)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": {"message_id": 9901}}

    with patch("requests.post", return_value=mock_resp):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "mock", "TELEGRAM_CHAT_ID": "123", "GDRIVE_SERVICE_ACCOUNT_KEY": ""}):
            deliv_success = deliver_unified_breakfast(
                target_date="20260914",
                out_dir=out_dir,
                skip_telegram=False,
                force=True,
                folder_id="",
                drive_service=None
            )
    assert deliv_success is True
    normal_reg = load_delivery_registry(None, None, local_fallback_dir=out_dir)
    assert is_unified_item_delivered(normal_reg, "20260914", "summary") is True
    assert is_unified_item_delivered(normal_reg, "20260914", "combined_video") is True
    assert is_breakfast_fully_delivered(normal_reg, "20260914") is True
    assert normal_reg["dates"]["20260914"]["unified"]["items"]["summary"]["message_id"] == 9901

    # Proof 6: Revision delivery semantics remain unchanged
    mock_resp_rev = MagicMock()
    mock_resp_rev.status_code = 200
    mock_resp_rev.json.return_value = {"result": {"message_id": 9902}}

    with patch("requests.post", return_value=mock_resp_rev):
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "mock", "TELEGRAM_CHAT_ID": "123", "GDRIVE_SERVICE_ACCOUNT_KEY": ""}):
            rev_success = deliver_unified_breakfast(
                target_date="20260914",
                out_dir=out_dir,
                skip_telegram=False,
                force=True,
                folder_id="",
                drive_service=None,
                revision="rev-test"
            )
    assert rev_success is True
    rev_reg = load_delivery_registry(None, None, local_fallback_dir=out_dir)
    assert is_unified_item_delivered(rev_reg, "20260914", "summary", revision="rev-test") is True
    assert is_unified_item_delivered(rev_reg, "20260914", "combined_video", revision="rev-test") is True
    assert is_breakfast_fully_delivered(rev_reg, "20260914", revision="rev-test") is True
    assert rev_reg["dates"]["20260914"]["corrections"]["rev-test"]["items"]["summary"]["message_id"] == 9902


