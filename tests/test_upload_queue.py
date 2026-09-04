import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.upload_queue import UploadQueue, UploadQueueWorker, UploadState, CorruptLedgerError
from scripts.lifecycle_manager import (
    check_drain_prerequisites,
    complete_drain_if_ready,
    drain_and_idle,
    read_state,
    write_state,
    LifecycleState,
)


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


def test_corrupt_ledger_raises_corrupt_ledger_error_and_fails_closed_in_drain(tmp_path):
    """
    Defect 1: Corrupt or unreadable ledger must raise CorruptLedgerError and check_drain_prerequisites
    must fail closed (returning False, not True).
    """
    ledger_file = tmp_path / "corrupt_ledger.json"
    ledger_file.write_text("{corrupt: json syntax error", encoding="utf-8")

    # Direct UploadQueue init must raise CorruptLedgerError
    with pytest.raises(CorruptLedgerError):
        UploadQueue(ledger_path=ledger_file)

    # check_drain_prerequisites must fail closed
    with patch.dict(os.environ, {"UPLOAD_LEDGER_PATH": str(ledger_file)}):
        drained, reason = check_drain_prerequisites(temp_dirs=[], staging_dirs=[], check_uploads=True, upload_queue=None)
        assert drained is False
        assert "Durable upload authority check failed" in reason


def test_max_attempts_exhaustion_transitions_to_failed_exhausted_and_blocks_drain(tmp_path):
    """
    Defect 2: UploadQueue retry attempts must be bounded. When max_attempts is reached,
    item transitions to FAILED_EXHAUSTED and CONTINUES to block drain.
    """
    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, max_attempts=2, backoff_base_sec=0.01)
    test_video = tmp_path / "motion_failing.mp4"
    test_video.write_bytes(b"content")

    queue.register_file(test_video)

    def mock_fail(cmd, *args, **kwargs):
        res = MagicMock()
        if "lsf" in cmd:
            res.returncode = 0
            res.stdout = ""
        else:
            res.returncode = 1
            res.stderr = "network connection dropped"
        return res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_fail):

        # Attempt 1 -> FAILED_RETRYABLE
        ok1, msg1 = queue.upload_file("motion_failing.mp4")
        assert ok1 is False
        item1 = queue.get_item("motion_failing.mp4")
        assert item1["state"] == UploadState.FAILED_RETRYABLE
        assert item1["attempt_count"] == 1
        assert item1["next_attempt_at"] is not None

        # Drain must block
        drained1, _ = check_drain_prerequisites(temp_dirs=[], staging_dirs=[], upload_queue=queue)
        assert drained1 is False

        # Attempt 2 -> FAILED_EXHAUSTED (max_attempts = 2)
        ok2, msg2 = queue.upload_file("motion_failing.mp4")
        assert ok2 is False
        item2 = queue.get_item("motion_failing.mp4")
        assert item2["state"] == UploadState.FAILED_EXHAUSTED
        assert item2["attempt_count"] == 2
        assert item2["next_attempt_at"] is None

        # Drain must STILL fail closed because exhausted item remains unresolved!
        drained2, reason2 = check_drain_prerequisites(temp_dirs=[], staging_dirs=[], upload_queue=queue)
        assert drained2 is False
        assert "unresolved upload(s) in queue" in reason2

        # Subsequent attempts are rejected immediately
        ok3, msg3 = queue.upload_file("motion_failing.mp4")
        assert ok3 is False
        assert "Max upload attempts" in msg3


def test_worker_skips_failed_exhausted_and_respects_backoff(tmp_path):
    """
    Defect 2: UploadQueueWorker must skip FAILED_EXHAUSTED items and not retry them in an infinite loop.
    """
    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, max_attempts=2)
    test_video = tmp_path / "motion_exhausted.mp4"
    test_video.write_bytes(b"content")

    queue.register_file(test_video)
    queue._update_item_state("motion_exhausted.mp4", UploadState.FAILED_EXHAUSTED)

    worker = UploadQueueWorker(queue, poll_interval_sec=0.1)

    with patch.object(queue, "upload_file") as mock_upload:
        worker.start()
        worker.notify()
        time.sleep(0.2)
        worker.stop()
        mock_upload.assert_not_called()


def test_untracked_clip_in_staging_blocks_drain_and_is_auto_registered(tmp_path):
    """
    Defect 4: Local clips present in staging directory that are not registered in upload ledger
    must block drain and be auto-registered into the queue as PENDING.
    """
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    untracked_video = staging_dir / "motion_20260904_untracked.mp4"
    untracked_video.write_bytes(b"untracked_data")

    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)

    drained, reason = check_drain_prerequisites(
        temp_dirs=[],
        staging_dirs=[staging_dir],
        check_uploads=True,
        upload_queue=queue,
    )
    assert drained is False
    assert "Untracked clip found in staging" in reason

    item = queue.get_item("motion_20260904_untracked.mp4")
    assert item is not None
    assert item["state"] == UploadState.PENDING


def test_complete_drain_if_ready_transitions_to_daytime_idle_and_dispatches_report(tmp_path):
    """
    Defect 3: complete_drain_if_ready deterministically transitions to DAYTIME_IDLE,
    sets source_evidence_ready=True, and triggers morning report dispatch once.
    """
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    clip = staging_dir / "motion_ready.mp4"
    clip.write_bytes(b"data")

    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    queue.register_file(clip)
    queue._update_item_state("motion_ready.mp4", UploadState.UPLOADED, verified=True)

    state_file = tmp_path / "state.json"

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1600), \
         patch("scripts.daily_watchdog.check_and_recover_report") as mock_watchdog:

        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": False})

        # First call: completes drain and dispatches report
        ok, reason = complete_drain_if_ready(temp_dirs=[], staging_dirs=[staging_dir])
        assert ok is True
        assert "DAYTIME_IDLE active" in reason
        mock_watchdog.assert_called_once()

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["source_evidence_ready"] is True
        assert state["local_drain_complete"] is True

        # Second call: idempotent, does not re-dispatch report
        mock_watchdog.reset_mock()
        ok2, reason2 = complete_drain_if_ready(temp_dirs=[], staging_dirs=[staging_dir])
        assert ok2 is True
        mock_watchdog.assert_not_called()


def test_run_until_empty_completes_lifecycle(tmp_path):
    """
    Defect 3: Standalone uploader processes queue items until empty, then triggers complete_drain_if_ready.
    """
    ledger_file = tmp_path / "test_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    test_video = tmp_path / "motion_pending.mp4"
    test_video.write_bytes(b"data")
    file_sz = test_video.stat().st_size
    queue.register_file(test_video)

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "lsf" in cmd:
            res.stdout = f"{file_sz};motion_pending.mp4\n"
        return res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain completed")) as mock_complete:

        ok, msg = queue.run_until_empty(max_wait_sec=5, poll_interval_sec=0.05)
        assert ok is True
        assert "Uploads finished and drain completed" in msg
        mock_complete.assert_called_once()


def test_motion_recorder_records_registration_faults_without_untracked_fallback(tmp_path, monkeypatch):
    """
    Defect 4: motion_recorder must record registration faults without untracked direct rclone fallback,
    preserving local file.
    """
    import motion_recorder
    recorder = motion_recorder.RecordingController.__new__(motion_recorder.RecordingController)
    recorder.upload_queue = None
    recorder.upload_worker = None
    recorder.registration_faults = []
    recorder._clips_lock = threading.Lock()
    recorder.clips_saved = 0

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    drive_dir = tmp_path / "drive"
    drive_dir.mkdir()

    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", temp_dir)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", drive_dir)
    monkeypatch.setattr(motion_recorder.platform, "system", lambda: "Linux")

    raw_video = temp_dir / "temp_motion.mp4"
    raw_video.write_bytes(b"test_video_data")

    recorder.yolo_model = None
    recorder._finalize_recording(
        temp_path=raw_video,
        final_name="motion_test_final.mp4",
        dur_str="10s",
        duration=10.0,
        cat_seen=True,
        declared_fps=15.0,
        frame_count=150,
    )

    dest_file = drive_dir / "motion_test_final.mp4"
    assert dest_file.exists()
    assert len(recorder.registration_faults) == 1
    assert "upload_queue is None" in recorder.registration_faults[0]["error"]
    assert recorder.registration_faults[0]["file"] == str(dest_file)


# ==============================================================================
# Final Upload-Authority Correction: 14 Required Regression Tests
# ==============================================================================

def test_identical_filenames_across_cameras_produce_separate_ledger_identities(tmp_path):
    """Test 1: TAPO and USB files with identical filenames produce separate ledger identities."""
    queue = UploadQueue(ledger_path=tmp_path / "ledger.json")
    tapo_file = tmp_path / "tapo" / "motion_20260904_shared.mp4"
    tapo_file.parent.mkdir()
    tapo_file.write_bytes(b"tapo_bytes")

    usb_file = tmp_path / "usb" / "motion_20260904_shared.mp4"
    usb_file.parent.mkdir()
    usb_file.write_bytes(b"usb_bytes")

    tapo_item = queue.register_file(tapo_file, camera_type="rtsp", rclone_dest_path="")
    usb_item = queue.register_file(usb_file, camera_type="usb", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")

    assert tapo_item["key"] == "rtsp:motion_20260904_shared.mp4"
    assert usb_item["key"] == "usb:motion_20260904_shared.mp4"
    assert tapo_item["key"] != usb_item["key"]

    all_items = queue.get_all_items()
    assert "rtsp:motion_20260904_shared.mp4" in all_items
    assert "usb:motion_20260904_shared.mp4" in all_items
    assert len(all_items) == 2


def test_tapo_verification_cannot_satisfy_usb_upload_authority(tmp_path):
    """Test 2: TAPO verification cannot satisfy USB upload authority."""
    queue = UploadQueue(ledger_path=tmp_path / "ledger.json")
    tapo_file = tmp_path / "tapo" / "motion_test.mp4"
    tapo_file.parent.mkdir()
    tapo_file.write_bytes(b"tapo_data")
    usb_file = tmp_path / "usb" / "motion_test.mp4"
    usb_file.parent.mkdir()
    usb_file.write_bytes(b"usb_data")

    queue.register_file(tapo_file, camera_type="rtsp", rclone_dest_path="")
    queue.register_file(usb_file, camera_type="usb", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")

    # Mark TAPO uploaded and verified
    queue._update_item_state("rtsp:motion_test.mp4", UploadState.UPLOADED, verified=True)

    # Verify USB item is STILL unresolved
    usb_item = queue.get_item("motion_test.mp4", camera_type="usb")
    assert usb_item["state"] == UploadState.PENDING
    assert usb_item["remote_verified"] is False

    has_unres, count, details = queue.has_unresolved_uploads()
    assert has_unres is True
    assert count == 1
    assert "usb:motion_test.mp4" in details


def test_usb_staging_auto_registration_receives_usb_contract(tmp_path):
    """Test 3: USB staging auto-registration receives the USB camera/destination contract."""
    staging = tmp_path / "usb-camera-sync"
    staging.mkdir()
    clip = staging / "motion_usb_auto.mp4"
    clip.write_bytes(b"usb_clip_data")

    queue = UploadQueue(ledger_path=tmp_path / "ledger.json")
    registered = queue.scan_and_register_untracked_clips(staging)

    assert len(registered) == 1
    item = registered[0]
    assert item["key"] == "usb:motion_usb_auto.mp4"
    assert item["camera_type"] == "usb"
    assert item["rclone_dest_path"] == "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS"
    assert item["state"] == UploadState.PENDING


def test_tapo_staging_auto_registration_receives_tapo_contract(tmp_path):
    """Test 4: TAPO staging auto-registration receives the TAPO camera/destination contract."""
    staging = tmp_path / "gdrive-randomdice-sync"
    staging.mkdir()
    clip = staging / "motion_tapo_auto.mp4"
    clip.write_bytes(b"tapo_clip_data")

    queue = UploadQueue(ledger_path=tmp_path / "ledger.json")
    registered = queue.scan_and_register_untracked_clips(staging)

    assert len(registered) == 1
    item = registered[0]
    assert item["key"] == "rtsp:motion_tapo_auto.mp4"
    assert item["camera_type"] == "rtsp"
    assert item["rclone_dest_path"] == ""
    assert item["state"] == UploadState.PENDING


def test_contract_mismatch_fails_visibly_rather_than_reusing_record(tmp_path):
    """Test 5: existing UPLOADED item with same filename but different camera/path/destination is not incorrectly reused."""
    queue = UploadQueue(ledger_path=tmp_path / "ledger.json")
    clip = tmp_path / "motion_reuse.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")
    queue._update_item_state("rtsp:motion_reuse.mp4", UploadState.UPLOADED, verified=True)

    # Attempting to re-register under rtsp key with conflicting destination must fail visibly
    with pytest.raises(ValueError) as exc:
        queue.register_file(clip, camera_type="rtsp", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")
    assert "Destination mismatch" in str(exc.value)

    # Moving file to a different path under same key must also fail visibly
    clip2 = tmp_path / "sub" / "motion_reuse.mp4"
    clip2.parent.mkdir()
    clip2.write_bytes(b"data")
    with pytest.raises(ValueError) as exc2:
        queue.register_file(clip2, camera_type="rtsp", rclone_dest_path="")
    assert "Filepath mismatch" in str(exc2.value)


def test_v1_filename_only_ledger_migrates_safely(tmp_path):
    """Test 6: existing v1 filename-only ledger migrates safely."""
    ledger_file = tmp_path / "upload_ledger.json"
    tapo_p = tmp_path / "gdrive-randomdice-sync" / "motion_tapo.mp4"
    tapo_p.parent.mkdir()
    tapo_p.write_bytes(b"tapo")
    usb_p = tmp_path / "usb-camera-sync" / "motion_usb.mp4"
    usb_p.parent.mkdir()
    usb_p.write_bytes(b"usb")

    v1_data = {
        "version": 1,
        "updated_at": "2026-09-04T12:00:00+02:00",
        "items": {
            "motion_tapo.mp4": {
                "filename": "motion_tapo.mp4",
                "filepath": str(tapo_p),
                "camera_type": "rtsp",
                "rclone_remote": "gdrive-randomdice:",
                "rclone_dest_path": "",
                "file_size_bytes": 4,
                "state": "UPLOADED",
                "attempt_count": 1,
                "remote_verified": True,
            },
            "motion_usb.mp4": {
                "filename": "motion_usb.mp4",
                "filepath": str(usb_p),
                "camera_type": "usb",
                "rclone_remote": "gdrive-randomdice:",
                "rclone_dest_path": "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS",
                "file_size_bytes": 3,
                "state": "UPLOADED",
                "attempt_count": 1,
                "remote_verified": True,
            },
        },
    }
    import json
    ledger_file.write_text(json.dumps(v1_data))

    queue = UploadQueue(ledger_path=ledger_file)
    all_items = queue.get_all_items()

    assert "rtsp:motion_tapo.mp4" in all_items
    assert "usb:motion_usb.mp4" in all_items
    assert len(all_items) == 2

    disk_data = json.loads(ledger_file.read_text())
    assert disk_data["version"] == 2


def test_misclassified_historical_usb_entry_is_detected_and_reconciled(tmp_path):
    """Test 7: misclassified historical USB entry is detected rather than trusted."""
    ledger_file = tmp_path / "upload_ledger.json"
    usb_p = tmp_path / "usb-camera-sync" / "motion_misclassified.mp4"
    usb_p.parent.mkdir()
    usb_p.write_bytes(b"usb_content")

    # Simulate historical defect: USB staging file recorded as rtsp with root destination
    v1_data = {
        "version": 1,
        "updated_at": "2026-09-04T12:00:00+02:00",
        "items": {
            "motion_misclassified.mp4": {
                "filename": "motion_misclassified.mp4",
                "filepath": str(usb_p),
                "camera_type": "rtsp",
                "rclone_remote": "gdrive-randomdice:",
                "rclone_dest_path": "",
                "file_size_bytes": 11,
                "state": "UPLOADED",
                "attempt_count": 1,
                "remote_verified": True,
            },
        },
    }
    import json
    ledger_file.write_text(json.dumps(v1_data))

    queue = UploadQueue(ledger_path=ledger_file)
    all_items = queue.get_all_items()

    assert "usb:motion_misclassified.mp4" in all_items
    item = all_items["usb:motion_misclassified.mp4"]
    assert item["camera_type"] == "usb"
    assert item["rclone_dest_path"] == "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS"
    assert item["remote_verified"] is False
    assert item["state"] == UploadState.PENDING


def test_failed_exhausted_bounded_later_recovery_path(tmp_path):
    """Test 8: FAILED_EXHAUSTED has a bounded later recovery path."""
    ledger_file = tmp_path / "upload_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, max_attempts=2, max_recovery_cycles=2, recovery_cooldown_sec=10.0)
    clip = tmp_path / "motion_exhaust.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")
    queue._update_item_state(
        "rtsp:motion_exhaust.mp4",
        UploadState.FAILED_EXHAUSTED,
        error="test error",
        exhausted_at="2026-09-04T12:00:00Z",
        exhausted_at_ts=time.time() - 20.0,
    )

    item_before = queue.get_item("rtsp:motion_exhaust.mp4")
    assert item_before["state"] == UploadState.FAILED_EXHAUSTED

    recovered = queue.reset_exhausted_items()
    assert recovered == 1

    item_after = queue.get_item("rtsp:motion_exhaust.mp4")
    assert item_after["state"] == UploadState.PENDING
    assert item_after["attempt_count"] == 0
    assert item_after["recovery_cycle"] == 1


def test_recovery_does_not_produce_unbounded_hammering(tmp_path):
    """Test 9: recovery does not produce continuous/unbounded retry hammering."""
    ledger_file = tmp_path / "upload_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, max_attempts=2, max_recovery_cycles=2, recovery_cooldown_sec=60.0)
    clip = tmp_path / "motion_hammer.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")

    # 1. During cooldown, recovery must NOT trigger
    queue._update_item_state(
        "rtsp:motion_hammer.mp4",
        UploadState.FAILED_EXHAUSTED,
        error="network down",
        exhausted_at_ts=time.time() - 5.0,
    )
    recovered = queue.reset_exhausted_items()
    assert recovered == 0
    assert queue.get_item("rtsp:motion_hammer.mp4")["state"] == UploadState.FAILED_EXHAUSTED

    # 2. When max_recovery_cycles reached, recovery permanently refuses
    queue._update_item_state(
        "rtsp:motion_hammer.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=time.time() - 100.0,
        recovery_cycle=2,
    )
    recovered2 = queue.reset_exhausted_items()
    assert recovered2 == 0
    assert queue.get_item("rtsp:motion_hammer.mp4")["state"] == UploadState.FAILED_EXHAUSTED


def test_lifecycle_completion_exception_causes_uploader_failure(tmp_path):
    """Test 10: lifecycle-completion exception causes uploader service failure."""
    ledger_file = tmp_path / "upload_ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    clip = tmp_path / "motion_clean.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip)
    queue._update_item_state("rtsp:motion_clean.mp4", UploadState.UPLOADED, verified=True)

    with patch("scripts.lifecycle_manager.complete_drain_if_ready", side_effect=RuntimeError("Disk full or lock timeout")):
        ok, msg = queue.run_until_empty(max_wait_sec=2, poll_interval_sec=0.05, staging_dirs=[])
        assert ok is False
        assert "Lifecycle completion failed with exception" in msg
        assert "Disk full or lock timeout" in msg


def test_complete_drain_if_ready_refuses_invalid_morning_active_transition(tmp_path):
    """Test 11: complete_drain_if_ready refuses an invalid MORNING_ACTIVE -> DAYTIME_IDLE transition."""
    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"
    UploadQueue(ledger_path=ledger_file)

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False):

        write_state({"state": LifecycleState.MORNING_ACTIVE, "services_active": True})

        ok, reason = complete_drain_if_ready(temp_dirs=[], staging_dirs=[])
        assert ok is False
        assert "Invalid lifecycle state transition" in reason
        assert "MORNING_ACTIVE" in reason

        state = read_state()
        assert state["state"] == LifecycleState.MORNING_ACTIVE


def test_heavy_recorder_active_prevents_false_host_idle_publication(tmp_path):
    """Test 12: actual heavy recorder still active prevents false host-idle publication."""
    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"
    UploadQueue(ledger_path=ledger_file)

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}):
        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": True})

        # Case A: cat-monitor systemd service active
        with patch("scripts.lifecycle_manager.is_service_active", side_effect=lambda svc: svc == "cat-monitor"), \
             patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]):
            ok, reason = complete_drain_if_ready(temp_dirs=[], staging_dirs=[])
            assert ok is False
            assert "Heavy recorder service still active" in reason

        # Case B: motion_recorder process running
        with patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
             patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[{"pid": "1234", "cmd": "python motion_recorder.py", "pss_kb": 50000}]):
            ok, reason = complete_drain_if_ready(temp_dirs=[], staging_dirs=[])
            assert ok is False
            assert "Motion recorder process still running" in reason

        state = read_state()
        assert state["state"] == LifecycleState.LOCAL_MORNING_DRAIN
        assert state.get("source_evidence_ready") is not True


def test_valid_local_morning_drain_to_daytime_idle_succeeds(tmp_path):
    """Test 13: valid LOCAL_MORNING_DRAIN -> upload complete -> DAYTIME_IDLE still succeeds."""
    staging_dir = tmp_path / "gdrive-randomdice-sync"
    staging_dir.mkdir()
    clip = staging_dir / "motion_valid.mp4"
    clip.write_bytes(b"clip_content")

    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")
    queue._update_item_state("rtsp:motion_valid.mp4", UploadState.UPLOADED, verified=True)

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1800), \
         patch("scripts.daily_watchdog.check_and_recover_report"):

        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": False})

        ok, reason = complete_drain_if_ready(temp_dirs=[], staging_dirs=[staging_dir])
        assert ok is True
        assert "Drain completed and DAYTIME_IDLE active" in reason

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["source_evidence_ready"] is True
        assert state["local_drain_complete"] is True
        assert state["services_active"] is False


def test_existing_exact_file_upload_and_remote_size_verification_remain_green(tmp_path):
    """Test 14: existing exact-file upload and remote size verification remain green."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    clip = tmp_path / "motion_single_upload.mp4"
    clip.write_bytes(b"exact_bytes_test")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")

    copyto_called = []
    lsf_called = []

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "copyto" in cmd:
            copyto_called.append(cmd)
        elif "lsf" in cmd:
            lsf_called.append(cmd)
            if copyto_called:
                res.stdout = f"{sz};motion_single_upload.mp4\n"
            else:
                res.stdout = ""
        return res

    with patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run):
        ok, msg = queue.upload_file("rtsp:motion_single_upload.mp4")
        assert ok is True
        assert "Upload successful and remote verified" in msg

        assert len(copyto_called) == 1
        assert "copyto" in copyto_called[0]
        assert str(clip) in copyto_called[0]
        assert "gdrive-randomdice:motion_single_upload.mp4" in copyto_called[0]

        item = queue.get_item("rtsp:motion_single_upload.mp4")
        assert item["state"] == UploadState.UPLOADED
        assert item["remote_verified"] is True


