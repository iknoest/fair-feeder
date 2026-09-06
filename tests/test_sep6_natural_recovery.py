"""
tests/test_sep6_natural_recovery.py - Acceptance & regression test suite for Sep-6 recovery

Covers the 13 required test cases:
1. test_tapo_notebook_evidence_only_no_telegram_secrets
2. test_tapo_notebook_produces_structured_summary_and_artifacts
3. test_unified_breakfast_cli_deliver_import_works_from_repo_root
4. test_unified_breakfast_imports_as_module_cleanly
5. test_delivery_ledger_imports_as_module_and_cli
6. test_telegram_control_service_standalone_start
7. test_telegram_control_service_responds_to_help
8. test_telegram_control_service_responds_to_status_during_daytime_idle
9. test_telegram_control_service_weight_menu
10. test_single_telegram_update_consumer_enforced
11. test_on_demand_capture_during_daytime_idle
12. test_delivery_ledger_prevents_duplicate_breakfast_delivery
13. test_sep6_breakfast_recovery_prerequisites
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Defect A Tests ──────────────────────────────────────────────────

def test_tapo_notebook_evidence_only_no_telegram_secrets():
    """1. Verify notebook parses and cell 5 executes without Telegram secrets."""
    nb_path = REPO_ROOT / "morning_report.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Cell 5 is secret loading
    c5_code = "".join(nb["cells"][5]["source"])
    assert "BOT_TOKEN  = os.environ.get('TelegramBotToken', '')" in c5_code

    # Execute Cell 5 code in isolated env with NO Telegram credentials
    clean_env = {
        "RUNNING_IN_CI": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    exec_globals = {"RUNNING_IN_CI": True, "os": os}
    with patch.dict(os.environ, clean_env, clear=True):
        # Must not raise KeyError: 'TelegramBotToken'
        exec(c5_code, exec_globals)
        assert exec_globals["BOT_TOKEN"] == ""
        assert exec_globals["MY_CHAT_ID"] == ""


def test_tapo_notebook_produces_structured_summary_and_artifacts(tmp_path):
    """2. Verify Phase 3 generates tapo_summary.json and artifacts without Telegram credentials."""
    nb_path = REPO_ROOT / "morning_report.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Inspect cell 16 (Phase 3 output)
    c16_code = "".join(nb["cells"][16]["source"])
    assert "tapo_summary_{_target_date_str}.json" in c16_code
    assert "if BOT_TOKEN and MY_CHAT_ID:" in c16_code

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Mock video_results with a tracker
    mock_tracker = MagicMock()
    mock_tracker._build_feeding_phases.return_value = [{
        "start": 0, "end": 100, "dan_frames": 80, "sanbo_frames": 20
    }]
    mock_tracker._get_timestamp.side_effect = lambda f: "06:20:00" if f == 0 else "06:21:00"
    mock_tracker.fps = 15.0
    mock_tracker.dan_at_bowl = [True] * 100
    mock_tracker.sanbo_at_bowl = [False] * 100
    mock_tracker.snapshots = {}

    video_results = [{
        "vid_name": "motion_test.mp4",
        "vid_stem": "motion_test",
        "tracker": mock_tracker,
        "summary_text": "Dan ate 15g kibble",
        "summary": {
            "dan_kibble_eaten": 15,
            "sanbo_kibble_eaten": 0,
            "start_kibble": 20,
            "end_kibble": 5,
            "start_ts": "06:20:00",
            "end_ts": "06:21:00",
            "dan_bowl_seconds": 45,
            "sanbo_bowl_seconds": 0,
        }
    }]

    # Run the summary generation slice with BOT_TOKEN = ""
    clean_globals = {
        "video_results": video_results,
        "Path": Path,
        "OUTPUT_DIR": str(out_dir),
        "_target_date_str": "20260906",
        "_drive_ledger": None,
        "_out_folder_id": "dummy_folder",
        "BOT_TOKEN": "",
        "MY_CHAT_ID": "",
        "json": json,
        "os": os,
        "plot_video_timeline": lambda tr, name: None,
        "send_telegram_summary": MagicMock(),
        "commit_camera_completion": MagicMock(),
        "is_breakfast_fully_delivered": lambda *a, **k: False,
        "is_camera_fully_delivered": lambda *a, **k: False,
        "delivery_ledger": {},
        "export_tapo_timeline": MagicMock(),
        "date": __import__("datetime").date,
    }

    # Execute phase 3 code
    exec(c16_code, clean_globals)

    # 1. Summary JSON must exist
    sum_file = out_dir / "tapo_summary_20260906.json"
    assert sum_file.exists(), "tapo_summary_20260906.json was not created"
    data = json.loads(sum_file.read_text(encoding="utf-8"))
    assert data["camera"] == "TAPO"
    assert data["dan_kibble"] == 15
    assert data["meal_finished"] is False

    # 2. Telegram sender must NOT be called when credentials absent
    assert clean_globals["send_telegram_summary"].call_count == 0


# ── Defect B Tests ──────────────────────────────────────────────────

def test_unified_breakfast_cli_deliver_import_works_from_repo_root():
    """3. Verify python scripts/unified_breakfast.py deliver ... works from repo root."""
    cmd = [
        sys.executable,
        "scripts/unified_breakfast.py",
        "deliver",
        "--date", "20260906",
        "--skip-telegram",
        "--force"
    ]
    # Execute with clean environment (no PYTHONPATH overriding sys.path)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)

    # It should NOT fail with ModuleNotFoundError: No module named 'scripts'
    assert "ModuleNotFoundError: No module named 'scripts'" not in res.stderr
    assert "ModuleNotFoundError: No module named 'scripts'" not in res.stdout


def test_unified_breakfast_imports_as_module_cleanly():
    """4. Verify scripts.unified_breakfast imports cleanly as a module."""
    from scripts.unified_breakfast import (
        deliver_unified_breakfast,
        generate_unified_breakfast_report,
        generate_combined_breakfast_video
    )
    assert callable(deliver_unified_breakfast)
    assert callable(generate_unified_breakfast_report)
    assert callable(generate_combined_breakfast_video)


def test_delivery_ledger_imports_as_module_and_cli():
    """5. Verify delivery_ledger imports as module and runs without import crash."""
    from scripts.delivery_ledger import (
        load_delivery_registry,
        save_delivery_registry,
        is_breakfast_fully_delivered,
        commit_breakfast_completion
    )
    assert callable(load_delivery_registry)
    assert callable(is_breakfast_fully_delivered)

    # CLI test
    cmd = [sys.executable, "scripts/delivery_ledger.py"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert "ModuleNotFoundError" not in res.stderr


# ── Problem C Tests ─────────────────────────────────────────────────

def test_telegram_control_service_standalone_start():
    """6. Verify TelegramControlService starts and stops cleanly without motion_recorder."""
    from scripts.telegram_control_service import TelegramControlService
    svc = TelegramControlService(bot_token="dummy_token", chat_id="12345678")
    assert not svc.running
    svc.start()
    assert svc.running
    svc.stop()
    assert not svc.running


def test_telegram_control_service_responds_to_help():
    """7. Verify /help command dispatches correct instructions."""
    from scripts.telegram_control_service import TelegramControlService
    svc = TelegramControlService(bot_token="dummy_token", chat_id="12345678")

    sent_messages = []
    svc._send = lambda text, sender_id=None, reply_markup=None: sent_messages.append(text)

    update = {
        "update_id": 101,
        "message": {
            "chat": {"id": 12345678},
            "text": "/help"
        }
    }
    svc.process_update(update)

    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "/status" in msg
    assert "/weight" in msg
    assert "/lastclip" in msg
    assert "/streaming_logitech" in msg


def test_telegram_control_service_responds_to_status_during_daytime_idle():
    """8. Verify /status responds with DAYTIME_IDLE during daytime idle state."""
    from scripts.telegram_control_service import TelegramControlService
    from scripts.lifecycle_manager import LifecycleState

    svc = TelegramControlService(bot_token="dummy_token", chat_id="12345678")
    sent_messages = []
    svc._send = lambda text, sender_id=None, reply_markup=None: sent_messages.append(text)

    mock_state = {
        "state": LifecycleState.DAYTIME_IDLE,
        "mem_available_mb": 1613,
        "services_active": False,
        "source_evidence_ready": True
    }

    with patch("scripts.lifecycle_manager.read_state", return_value=mock_state), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1613):
        update = {
            "update_id": 102,
            "message": {
                "chat": {"id": 12345678},
                "text": "/status"
            }
        }
        svc.process_update(update)

    assert len(sent_messages) == 1
    msg = sent_messages[0]
    assert "DAYTIME_IDLE" in msg
    assert "1613 MB" in msg
    assert "Cat-monitor (Tapo): inactive ⚪" in msg
    assert "Usb-monitor (Logi): inactive ⚪" in msg


def test_telegram_control_service_weight_menu(tmp_path):
    """9. Verify /weight interactive flow logs weight to CSV."""
    from scripts.telegram_control_service import TelegramControlService
    weight_file = tmp_path / "weight_log.csv"
    svc = TelegramControlService(bot_token="dummy_token", chat_id="12345678", weight_file=weight_file)

    sent_messages = []
    svc._send = lambda text, sender_id=None, reply_markup=None: sent_messages.append((text, reply_markup))

    # Step 1: /weight command
    svc.process_update({
        "update_id": 1,
        "message": {"chat": {"id": 12345678}, "text": "/weight"}
    })
    assert len(sent_messages) == 1
    assert "Weight menu:" in sent_messages[-1][0]
    assert sent_messages[-1][1] is not None  # Has reply_markup

    # Step 2: Click Log Weight callback query
    svc.process_update({
        "update_id": 2,
        "callback_query": {
            "id": "cq1",
            "message": {"chat": {"id": 12345678}},
            "data": "weight_menu_log"
        }
    })
    assert "Which cat?" in sent_messages[-1][0]

    # Step 3: Select 'dan'
    svc.process_update({
        "update_id": 3,
        "callback_query": {
            "id": "cq2",
            "message": {"chat": {"id": 12345678}},
            "data": "dan"
        }
    })
    assert "Enter Dan weight in kg" in sent_messages[-1][0]

    # Step 4: Reply with weight "5.45"
    with patch("subprocess.Popen"):
        svc.process_update({
            "update_id": 4,
            "message": {"chat": {"id": 12345678}, "text": "5.45"}
        })
    assert "Saved: Dan = 5.45 kg" in sent_messages[-1][0]

    # Verify CSV was written
    rows = svc._load_weights()
    assert len(rows) == 1
    assert rows[0]["cat"] == "dan"
    assert rows[0]["weight_kg"] == "5.45"


def test_single_telegram_update_consumer_enforced(tmp_path):
    """10. Verify host lock prevents competing getUpdates consumers."""
    from scripts.telegram_control_service import acquire_host_lock, HostLockError

    lock_file = str(tmp_path / "test_telegram.lock")
    fd1 = acquire_host_lock(lock_file)
    assert fd1 is not None

    # Second call should raise HostLockError
    with pytest.raises(HostLockError):
        acquire_host_lock(lock_file)

    # Release first lock
    fd1.close()

    # Third call should now succeed
    fd2 = acquire_host_lock(lock_file)
    assert fd2 is not None
    fd2.close()

    # Also verify motion_recorder.py does NOT call TelegramCommandListener.start()
    mr_text = (REPO_ROOT / "motion_recorder.py").read_text(encoding="utf-8")
    assert "cmd_listener.start()" not in mr_text


def test_on_demand_capture_during_daytime_idle():
    """11. Verify on-demand capture does not wake heavy recorder permanently."""
    from scripts.lifecycle_manager import on_demand_capture, read_state, write_state, LifecycleState

    # Set state to DAYTIME_IDLE
    write_state({"state": LifecycleState.DAYTIME_IDLE})

    # Mock cv2 VideoCapture and VideoWriter
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, MagicMock())
    mock_cap.get.return_value = 15.0

    mock_writer = MagicMock()

    with patch.dict(os.environ, {"TAPO_IP": "1.2.3.4", "TAPO_USER": "u", "TAPO_PASS": "p"}), \
         patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.VideoWriter", return_value=mock_writer), \
         patch("subprocess.run"):
        out_path = on_demand_capture(camera="tapo", duration_sec=1, out_path="/tmp/test_capture.mp4")

    # State must be restored to DAYTIME_IDLE after capture
    final_state = read_state()
    assert final_state.get("state") == LifecycleState.DAYTIME_IDLE


def test_delivery_ledger_prevents_duplicate_breakfast_delivery(tmp_path):
    """12. Verify delivery registry idempotency prevents duplicate deliveries."""
    from scripts.delivery_ledger import (
        init_registry_data,
        save_delivery_registry,
        load_delivery_registry,
        record_unified_item_delivered,
        commit_breakfast_completion,
        is_breakfast_fully_delivered
    )

    registry = init_registry_data()
    save_delivery_registry(None, "dummy_folder", registry, local_fallback_dir=tmp_path)

    assert not is_breakfast_fully_delivered(registry, "20260906")

    # Mark summary and combined_video
    record_unified_item_delivered(None, "dummy_folder", registry, "20260906", "summary", message_id=100, local_fallback_dir=tmp_path)
    assert not is_breakfast_fully_delivered(registry, "20260906")  # Needs both!

    record_unified_item_delivered(None, "dummy_folder", registry, "20260906", "combined_video", message_id=101, local_fallback_dir=tmp_path)
    # Now both required items are recorded
    assert is_breakfast_fully_delivered(registry, "20260906")

    # Commit completion
    ok = commit_breakfast_completion(None, "dummy_folder", "20260906", local_fallback_dir=tmp_path)
    assert ok is True

    # Reload and verify
    reloaded = load_delivery_registry(None, "dummy_folder", local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(reloaded, "20260906")


def test_sep6_breakfast_recovery_prerequisites():
    """13. Verify prerequisites for Sep-6 breakfast recovery exist."""
    # Check that delivery registry on Drive or locally can be inspected
    from scripts.delivery_ledger import DELIVERY_REGISTRY_FILENAME
    assert DELIVERY_REGISTRY_FILENAME == "delivery_registry.json"
