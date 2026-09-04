import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.upload_queue import UploadQueue, UploadQueueWorker, UploadState
from scripts.lifecycle_manager import check_drain_prerequisites, drain_and_idle, read_state, write_state, LifecycleState


@pytest.fixture
def temp_queue(tmp_path):
    ledger_file = tmp_path / "test_upload_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, upload_timeout_sec=10, max_attempts=3)
    return queue


def test_register_file_durably_creates_pending_state(temp_queue, tmp_path):
    test_video = tmp_path / "motion_20260904_062007_2m_30s.mp4"
    test_video.write_bytes(b"x" * 1024 * 100)

    item = temp_queue.register_file(test_video, camera_type="rtsp", rclone_remote="gdrive-randomdice:")
    assert item["filename"] == "motion_20260904_062007_2m_30s.mp4"
    assert item["state"] == UploadState.PENDING
    assert item["file_size_bytes"] == 1024 * 100
    assert item["remote_verified"] is False

    # Read back from ledger
    saved_item = temp_queue.get_item("motion_20260904_062007_2m_30s.mp4")
    assert saved_item is not None
    assert saved_item["state"] == UploadState.PENDING


def test_missing_local_file_registration_raises_error(temp_queue, tmp_path):
    missing_video = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        temp_queue.register_file(missing_video)


def test_small_upload_succeeds_first_try(temp_queue, tmp_path):
    test_video = tmp_path / "motion_20260904_062239_43s.mp4"
    test_video.write_bytes(b"small_clip_content")
    file_size = test_video.stat().st_size

    temp_queue.register_file(test_video)

    # Mock subprocess.run for rclone lsf and copyto
    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 0
        if "lsf" in cmd:
            # First verification before upload: not found
            # Second verification after upload: found with matching size
            if getattr(mock_run, "uploaded", False):
                mock_res.stdout = f"{file_size};motion_20260904_062239_43s.mp4\n"
            else:
                mock_res.stdout = ""
        elif "copyto" in cmd:
            mock_run.uploaded = True
            mock_res.stdout = "Transferred 1 file"
        return mock_res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):
        ok, msg = temp_queue.upload_file("motion_20260904_062239_43s.mp4")
        assert ok is True
        assert "Upload successful" in msg

        item = temp_queue.get_item("motion_20260904_062239_43s.mp4")
        assert item["state"] == UploadState.UPLOADED
        assert item["remote_verified"] is True
        assert item["attempt_count"] == 1


def test_large_upload_timeout_then_retry_success_sep4_regression(temp_queue, tmp_path):
    """
    Demonstrated Sep 4 failure reproduction:
    Attempt 1: 27 MB file times out in rclone. Local file remains intact, state becomes FAILED_RETRYABLE.
    Drain prerequisites must reject drain while upload is pending.
    Attempt 2: Retry succeeds, remote is verified, state becomes UPLOADED.
    Drain prerequisites can then pass.
    """
    test_video = tmp_path / "motion_20260904_062007_2m_30s.mp4"
    test_video.write_bytes(b"27MB_video_data_payload")
    file_size = test_video.stat().st_size

    temp_queue.register_file(test_video)

    call_count = {"copyto": 0}

    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        if "lsf" in cmd:
            mock_res.returncode = 0
            if call_count["copyto"] >= 2:
                mock_res.stdout = f"{file_size};motion_20260904_062007_2m_30s.mp4\n"
            else:
                mock_res.stdout = ""
            return mock_res
        elif "copyto" in cmd:
            call_count["copyto"] += 1
            if call_count["copyto"] == 1:
                # First attempt times out (demonstrated Sep 4 failure)
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)
            else:
                # Second attempt succeeds
                mock_res.returncode = 0
                mock_res.stdout = "Transferred 1 file"
                return mock_res
        mock_res.returncode = 0
        return mock_res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):

        # Attempt 1: Upload fails with timeout
        ok, msg = temp_queue.upload_file("motion_20260904_062007_2m_30s.mp4")
        assert ok is False
        assert "timed out" in msg

        # Verify local file is intact
        assert test_video.exists()

        # Verify state is FAILED_RETRYABLE, not lost
        item = temp_queue.get_item("motion_20260904_062007_2m_30s.mp4")
        assert item["state"] == UploadState.FAILED_RETRYABLE
        assert item["attempt_count"] == 1
        assert "timed out" in item["last_error"]

        # Drain must REJECT because item is unresolved
        has_unres, count, details = temp_queue.has_unresolved_uploads()
        assert has_unres is True
        assert count == 1

        drained, reason = check_drain_prerequisites(temp_dirs=[], upload_queue=temp_queue)
        assert drained is False
        assert "1 unresolved upload(s) in queue" in reason

        # Attempt 2: Retry succeeds
        ok2, msg2 = temp_queue.upload_file("motion_20260904_062007_2m_30s.mp4")
        assert ok2 is True

        item_after = temp_queue.get_item("motion_20260904_062007_2m_30s.mp4")
        assert item_after["state"] == UploadState.UPLOADED
        assert item_after["remote_verified"] is True
        assert item_after["attempt_count"] == 2

        # Drain can now PASS
        has_unres2, count2, _ = temp_queue.has_unresolved_uploads()
        assert has_unres2 is False
        assert count2 == 0

        drained2, reason2 = check_drain_prerequisites(temp_dirs=[], upload_queue=temp_queue)
        assert drained2 is True


def test_idempotent_acceptance_when_remote_already_matches(temp_queue, tmp_path):
    test_video = tmp_path / "motion_already_synced.mp4"
    test_video.write_bytes(b"identical_remote_content")
    file_size = test_video.stat().st_size

    temp_queue.register_file(test_video)

    # Remote lsf immediately reports matching file
    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 0
        if "lsf" in cmd:
            mock_res.stdout = f"{file_size};motion_already_synced.mp4\n"
        elif "copyto" in cmd:
            pytest.fail("copyto should NOT be called when file is already verified remotely")
        return mock_res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):
        ok, msg = temp_queue.upload_file("motion_already_synced.mp4")
        assert ok is True
        assert "Idempotent match" in msg

        item = temp_queue.get_item("motion_already_synced.mp4")
        assert item["state"] == UploadState.UPLOADED
        assert item["remote_verified"] is True


def test_crash_recovery_resets_uploading_to_pending(temp_queue, tmp_path):
    test_video = tmp_path / "motion_interrupted.mp4"
    test_video.write_bytes(b"content")

    temp_queue.register_file(test_video)
    # Simulate crash mid-upload
    temp_queue._update_item_state("motion_interrupted.mp4", UploadState.UPLOADING)

    item = temp_queue.get_item("motion_interrupted.mp4")
    assert item["state"] == UploadState.UPLOADING

    # Simulate new process startup
    recovered = temp_queue.recover_pending()
    assert recovered == 1

    item_recovered = temp_queue.get_item("motion_interrupted.mp4")
    assert item_recovered["state"] == UploadState.PENDING
    assert "Process restarted" in item_recovered["last_error"]


def test_missing_local_file_during_upload_fails_visibly(temp_queue, tmp_path):
    test_video = tmp_path / "motion_deleted_locally.mp4"
    test_video.write_bytes(b"will_be_deleted")

    temp_queue.register_file(test_video)
    # Delete local file before upload executes
    test_video.unlink()

    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = ""
        return mock_res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):
        ok, msg = temp_queue.upload_file("motion_deleted_locally.mp4")
        assert ok is False
        assert "Local file missing" in msg

        item = temp_queue.get_item("motion_deleted_locally.mp4")
        assert item["state"] == UploadState.FAILED_RETRYABLE
        assert "Local file missing" in item["last_error"]


def test_exact_file_command_does_not_copy_entire_folder(temp_queue, tmp_path):
    test_video = tmp_path / "motion_single.mp4"
    test_video.write_bytes(b"single")
    file_size = test_video.stat().st_size

    temp_queue.register_file(test_video, rclone_remote="gdrive-randomdice:", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")

    recorded_cmds = []

    def mock_run(cmd, *args, **kwargs):
        recorded_cmds.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        if "lsf" in cmd:
            mock_res.stdout = f"{file_size};motion_single.mp4\n" if len(recorded_cmds) > 1 else ""
        return mock_res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):
        ok, msg = temp_queue.upload_file("motion_single.mp4")
        assert ok is True

        copy_cmds = [c for c in recorded_cmds if "copyto" in c]
        assert len(copy_cmds) == 1
        copy_cmd = copy_cmds[0]
        # Must be single-file copyto with local file as source
        assert copy_cmd[1] == "copyto"
        assert copy_cmd[2] == str(test_video.resolve())
        assert "--drive-root-folder-id" in copy_cmd
        assert "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS" in copy_cmd


def test_retained_uploaded_local_file_does_not_block_drain(temp_queue, tmp_path):
    test_video = tmp_path / "motion_retained.mp4"
    test_video.write_bytes(b"retained_file_on_disk")

    temp_queue.register_file(test_video)
    temp_queue._update_item_state("motion_retained.mp4", UploadState.UPLOADED, verified=True)

    # File still exists locally on disk (not deleted)
    assert test_video.exists()

    # Drain must pass because item state is UPLOADED
    drained, reason = check_drain_prerequisites(temp_dirs=[], upload_queue=temp_queue)
    assert drained is True


def test_drain_and_idle_fails_closed_when_upload_unresolved(tmp_path):
    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    test_video = tmp_path / "motion_pending.mp4"
    test_video.write_bytes(b"data")
    queue.register_file(test_video)

    state_file = tmp_path / "state.json"

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("subprocess.run") as mock_run, \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1600), \
         patch("scripts.daily_watchdog.check_and_recover_report") as mock_watchdog, \
         patch("time.sleep"):

        mock_run.return_value = MagicMock(returncode=0)
        success = drain_and_idle(max_wait_sec=1, temp_dirs=[])
        assert success is False
        mock_watchdog.assert_not_called()

        state = read_state()
        assert state["state"] == LifecycleState.LOCAL_MORNING_DRAIN
        assert state["local_drain_complete"] is False
        assert state["source_evidence_ready"] is False
        assert "Evidence not durable" in state["last_event"]


def test_drain_and_idle_succeeds_when_all_uploads_complete(tmp_path):
    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    test_video = tmp_path / "motion_done.mp4"
    test_video.write_bytes(b"data")
    queue.register_file(test_video)
    queue._update_item_state("motion_done.mp4", UploadState.UPLOADED, verified=True)

    state_file = tmp_path / "state.json"

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("subprocess.run") as mock_run, \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1600), \
         patch("scripts.daily_watchdog.check_and_recover_report") as mock_watchdog, \
         patch("time.sleep"):

        mock_run.return_value = MagicMock(returncode=0)
        success = drain_and_idle(max_wait_sec=5, temp_dirs=[])
        assert success is True
        mock_watchdog.assert_called_once()

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["local_drain_complete"] is True
        assert state["source_evidence_ready"] is True
