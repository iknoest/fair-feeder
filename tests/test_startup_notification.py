import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch("pathlib.Path.mkdir"):
    import motion_recorder
from motion_recorder import (
    get_system_boot_id,
    send_startup_notification_once_per_boot,
)


def test_first_call_tapo_sends_notification(tmp_path):
    state_dir = tmp_path / "state"
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("boot-uuid-1111\n")

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append):
        sent = send_startup_notification_once_per_boot(
            camera_type="rtsp",
            state_dir=state_dir,
            boot_id_file=boot_file,
        )

    assert sent is True
    assert len(alerts) == 1
    assert "📷 Fair Feeder Monitor [TAPO] is LIVE" in alerts[0]

    # State file should be written
    tapo_state = state_dir / "startup_notified_rtsp.txt"
    assert tapo_state.exists()
    assert tapo_state.read_text().strip() == "boot-uuid-1111"


def test_second_call_same_tapo_same_boot_suppresses(tmp_path):
    state_dir = tmp_path / "state"
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("boot-uuid-1111\n")

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append):
        # First call
        sent1 = send_startup_notification_once_per_boot("rtsp", state_dir=state_dir, boot_id_file=boot_file)
        assert sent1 is True
        assert len(alerts) == 1

        # Second call (service restart on same boot)
        sent2 = send_startup_notification_once_per_boot("rtsp", state_dir=state_dir, boot_id_file=boot_file)
        assert sent2 is False
        assert len(alerts) == 1  # Suppressed!


def test_logitech_same_boot_sends_once_independently(tmp_path):
    state_dir = tmp_path / "state"
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("boot-uuid-1111\n")

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append):
        # TAPO sends
        sent_tapo = send_startup_notification_once_per_boot("rtsp", state_dir=state_dir, boot_id_file=boot_file)
        assert sent_tapo is True
        assert len(alerts) == 1
        assert "[TAPO]" in alerts[0]

        # LOGITECH sends independently on same boot
        sent_usb = send_startup_notification_once_per_boot("usb", state_dir=state_dir, boot_id_file=boot_file)
        assert sent_usb is True
        assert len(alerts) == 2
        assert "[LOGITECH]" in alerts[1]

        # Second call for LOGITECH is suppressed
        sent_usb_2 = send_startup_notification_once_per_boot("usb", state_dir=state_dir, boot_id_file=boot_file)
        assert sent_usb_2 is False
        assert len(alerts) == 2


def test_new_boot_id_resets_and_sends_again(tmp_path):
    state_dir = tmp_path / "state"
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("boot-uuid-1111\n")

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append):
        sent1 = send_startup_notification_once_per_boot("rtsp", state_dir=state_dir, boot_id_file=boot_file)
        assert sent1 is True
        assert len(alerts) == 1

        # Host reboots -> new boot ID
        boot_file.write_text("boot-uuid-2222\n")

        sent2 = send_startup_notification_once_per_boot("rtsp", state_dir=state_dir, boot_id_file=boot_file)
        assert sent2 is True
        assert len(alerts) == 2
        assert (state_dir / "startup_notified_rtsp.txt").read_text().strip() == "boot-uuid-2222"


def test_state_file_write_failure_does_not_crash(tmp_path):
    boot_file = tmp_path / "boot_id"
    boot_file.write_text("boot-uuid-1111\n")

    # Invalid state dir where mkdir will fail
    fake_state_dir = tmp_path / "not_a_dir"
    fake_state_dir.write_text("blocking file")

    alerts = []
    with patch("motion_recorder.send_telegram_alert", side_effect=alerts.append):
        # Should not raise exception
        sent = send_startup_notification_once_per_boot("rtsp", state_dir=fake_state_dir, boot_id_file=boot_file)
        assert sent is True
        assert len(alerts) == 1


def test_env_boot_id_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_FEEDER_BOOT_ID", "custom-env-boot-9999")
    boot_id = get_system_boot_id()
    assert boot_id == "custom-env-boot-9999"
