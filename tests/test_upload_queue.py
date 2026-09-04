import json
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

from scripts.upload_queue import (
    UploadQueue,
    UploadQueueWorker,
    UploadState,
    CorruptLedgerError,
    DEFAULT_WAKE_SERVICE_NAME,
    DEFAULT_WAKE_TIMER_UNIT,
)
from scripts.lifecycle_manager import (
    check_drain_prerequisites,
    complete_drain_if_ready,
    drain_and_idle,
    read_state,
    write_state,
    LifecycleState,
)


@pytest.fixture(autouse=True)
def isolate_test_staging(tmp_path, monkeypatch):
    """Isolates tests from host production staging directories."""
    test_staging = tmp_path / "test_staging_empty"
    test_staging.mkdir()
    monkeypatch.setenv("FAIR_FEEDER_STAGING_DIRS", str(test_staging))
    test_temp = tmp_path / "test_temp_empty"
    test_temp.mkdir()
    monkeypatch.setenv("FAIR_FEEDER_TEMP_DIRS", str(test_temp))


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


# ==============================================================================
# Bounded Autonomous FAILED_EXHAUSTED Recovery: 12 Targeted Unit Tests
# ==============================================================================

class FakeTimeContext:
    def __init__(self, initial_time=1000.0):
        self.current_time = initial_time
        self.sleep_calls = []

    def time(self):
        return self.current_time

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)
        self.current_time += seconds


def test_newly_exhausted_item_does_not_recover_before_cooldown(tmp_path):
    """Test 1: newly exhausted item does not recover before cooldown."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0, max_recovery_cycles=3)
    clip = tmp_path / "motion_test1.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test1.mp4",
        UploadState.FAILED_EXHAUSTED,
        error="upload error",
        exhausted_at_ts=1000.0,
    )

    # At 1100.0 (100s elapsed < 300s cooldown)
    item = queue.get_item("rtsp:motion_test1.mp4")
    is_eligible, earliest_ts = queue.get_recovery_eligibility(item, now=1100.0)
    assert is_eligible is False
    assert earliest_ts == 1300.0

    recovered = queue.reset_exhausted_items(now=1100.0)
    assert recovered == 0

    item_after = queue.get_item("rtsp:motion_test1.mp4")
    assert item_after["state"] == UploadState.FAILED_EXHAUSTED
    assert item_after["recovery_cycle"] == 0


def test_run_until_empty_does_not_exit_when_items_are_recoverable(tmp_path):
    """Test 2: run_until_empty does not exit when items are recoverable."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=3,
        max_attempts=2,
    )
    clip = tmp_path / "motion_test2.mp4"
    clip.write_bytes(b"data_test2")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test2.mp4",
        UploadState.FAILED_EXHAUSTED,
        error="network failed",
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    fake_time = FakeTimeContext(1000.0)

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "lsf" in cmd:
            res.stdout = f"{sz};motion_test2.mp4\n"
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")) as mock_drain:

        ok, msg = queue.run_until_empty(max_wait_sec=900, poll_interval_sec=2.0, staging_dirs=[])
        assert ok is True
        assert "Uploads finished and drain completed" in msg
        mock_drain.assert_called_once()

        item = queue.get_item("rtsp:motion_test2.mp4")
        assert item["state"] == UploadState.UPLOADED
        assert item["remote_verified"] is True
        assert item["recovery_cycle"] == 1
        # Proves it did not exit immediately: fake time advanced past cooldown
        assert fake_time.current_time >= 1300.0


def test_uploader_waits_without_busy_looping_until_recovery_eligibility(tmp_path):
    """Test 3: uploader waits without busy-looping until recovery eligibility."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
    )
    clip = tmp_path / "motion_test3.mp4"
    clip.write_bytes(b"data_test3")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test3.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    fake_time = FakeTimeContext(1000.0)

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "lsf" in cmd:
            res.stdout = f"{sz};motion_test3.mp4\n"
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")):

        ok, msg = queue.run_until_empty(max_wait_sec=1000, poll_interval_sec=2.0, staging_dirs=[])
        assert ok is True

        # Non-busy-loop verification:
        # Instead of 150 iterations of 2.0s sleeps (busy loop),
        # there must be a single direct sleep covering the 300s cooldown!
        cooldown_sleeps = [s for s in fake_time.sleep_calls if s >= 300.0]
        assert len(cooldown_sleeps) == 1
        assert 300.0 <= cooldown_sleeps[0] <= 300.2
        # Total sleep calls must be very small (e.g. <= 3), not hundreds
        assert len(fake_time.sleep_calls) <= 5


def test_recovery_cycle_increments_exactly_once_after_cooldown(tmp_path):
    """Test 4: recovery cycle increments exactly once after cooldown."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0, max_recovery_cycles=3)
    clip = tmp_path / "motion_test4.mp4"
    clip.write_bytes(b"data_test4")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test4.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    # First reset after cooldown: recovers item
    recovered1 = queue.reset_exhausted_items(now=1305.0)
    assert recovered1 == 1

    item1 = queue.get_item("rtsp:motion_test4.mp4")
    assert item1["state"] == UploadState.PENDING
    assert item1["recovery_cycle"] == 1

    # Immediate second reset: must NOT increment again
    recovered2 = queue.reset_exhausted_items(now=1310.0)
    assert recovered2 == 0

    item2 = queue.get_item("rtsp:motion_test4.mp4")
    assert item2["state"] == UploadState.PENDING
    assert item2["recovery_cycle"] == 1


def test_attempt_count_resets_to_0_for_new_cycle(tmp_path):
    """Test 5: attempt count resets to 0 for new cycle."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0)
    clip = tmp_path / "motion_test5.mp4"
    clip.write_bytes(b"data_test5")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test5.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )
    # Set attempt_count to 5 (max attempts)
    def _set_att(data):
        data["items"]["rtsp:motion_test5.mp4"]["attempt_count"] = 5
        return None, data
    queue._locked_transaction(_set_att)

    assert queue.get_item("rtsp:motion_test5.mp4")["attempt_count"] == 5

    queue.reset_exhausted_items(now=1350.0)

    item = queue.get_item("rtsp:motion_test5.mp4")
    assert item["state"] == UploadState.PENDING
    assert item["attempt_count"] == 0
    assert item["recovery_cycle"] == 1


def test_multi_cycle_autonomous_execution_without_human_restart(tmp_path):
    """Test 6: multi-cycle (2nd/3rd) autonomous execution without human restart."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=100.0,
        max_recovery_cycles=3,
        max_attempts=2,
    )
    clip = tmp_path / "motion_test6.mp4"
    clip.write_bytes(b"data_test6")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp")

    fake_time = FakeTimeContext(1000.0)
    upload_attempts = []

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        if "copyto" in cmd:
            attempt_num = len(upload_attempts) + 1
            upload_attempts.append((attempt_num, fake_time.time()))
            # Fail cycles 0 and 1 (attempts 1..4), succeed on attempt 5 (cycle 2)
            if attempt_num < 5:
                res.returncode = 1
                res.stderr = "temporary connection failure"
            else:
                res.returncode = 0
            return res
        elif "lsf" in cmd:
            res.returncode = 0
            # Succeeded only after attempt 5
            if len(upload_attempts) >= 5:
                res.stdout = f"{sz};motion_test6.mp4\n"
            else:
                res.stdout = ""
            return res
        res.returncode = 0
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")):

        ok, msg = queue.run_until_empty(max_wait_sec=2000, poll_interval_sec=1.0, staging_dirs=[])
        assert ok is True

        item = queue.get_item("rtsp:motion_test6.mp4")
        assert item["state"] == UploadState.UPLOADED
        assert item["remote_verified"] is True
        # Must have completed on cycle 2
        assert item["recovery_cycle"] == 2
        # Verify 2 cooldown sleeps occurred
        cooldown_sleeps = [s for s in fake_time.sleep_calls if s >= 90.0]
        assert len(cooldown_sleeps) == 2


def test_success_in_later_cycle_completes_remote_verification_and_drain(tmp_path):
    """Test 7: success in later cycle completes remote verification and drain."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    clip = staging_dir / "motion_test7.mp4"
    clip.write_bytes(b"data_test7")
    sz = clip.stat().st_size

    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"

    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")
    queue._update_item_state(
        "rtsp:motion_test7.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    fake_time = FakeTimeContext(1000.0)

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "lsf" in cmd:
            res.stdout = f"{sz};motion_test7.mp4\n"
        return res

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1800), \
         patch("scripts.daily_watchdog.check_and_recover_report"):

        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": False})

        ok, msg = queue.run_until_empty(max_wait_sec=500, poll_interval_sec=1.0, staging_dirs=[staging_dir])
        assert ok is True

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["source_evidence_ready"] is True
        assert state["local_drain_complete"] is True


def test_max_recovery_cycle_exhaustion_exits_failure_and_remains_fail_closed(tmp_path):
    """Test 8: max recovery cycle exhaustion exits failure and remains fail-closed."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=2,
    )
    clip = tmp_path / "motion_test8.mp4"
    clip.write_bytes(b"data_test8")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test8.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=2,  # Already reached max_recovery_cycles
    )

    # Even 100,000s after exhaustion, eligibility is permanently False
    item = queue.get_item("rtsp:motion_test8.mp4")
    is_eligible, earliest_ts = queue.get_recovery_eligibility(item, now=100000.0)
    assert is_eligible is False
    assert earliest_ts is None

    # run_until_empty must exit failure immediately without sleeping
    fake_time = FakeTimeContext(10000.0)
    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep):
        ok, msg = queue.run_until_empty(max_wait_sec=500, poll_interval_sec=1.0, staging_dirs=[])
        assert ok is False
        assert "permanently FAILED_EXHAUSTED" in msg
        assert len(fake_time.sleep_calls) == 0

    # Drain prerequisites must fail closed
    drained, reason = check_drain_prerequisites(temp_dirs=[], staging_dirs=[], upload_queue=queue)
    assert drained is False
    assert "unresolved upload(s) in queue" in reason


def test_register_file_cannot_bypass_cooldown(tmp_path):
    """Test 9: register_file cannot bypass cooldown."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0, max_recovery_cycles=3)
    clip = tmp_path / "motion_test9.mp4"
    clip.write_bytes(b"data_test9")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:motion_test9.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    # Re-registering during cooldown (at 1100s < 1300s) must NOT bypass cooldown
    with patch("time.time", return_value=1100.0):
        item_refused = queue.register_file(clip, camera_type="rtsp")
        assert item_refused["state"] == UploadState.FAILED_EXHAUSTED
        assert item_refused["recovery_cycle"] == 0

    # Re-registering after cooldown has elapsed (at 1350s >= 1300s) recovers item
    with patch("time.time", return_value=1350.0):
        item_recovered = queue.register_file(clip, camera_type="rtsp")
        assert item_recovered["state"] == UploadState.PENDING
        assert item_recovered["recovery_cycle"] == 1
        assert item_recovered["attempt_count"] == 0


def test_no_infinite_retry_or_rapid_hammering(tmp_path):
    """Test 10: no infinite retry or rapid hammering (bounded total attempts & backoffs)."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        max_attempts=3,
        max_recovery_cycles=2,
        recovery_cooldown_sec=50.0,
        backoff_base_sec=1.0,
        backoff_max_sec=10.0,
    )
    clip = tmp_path / "motion_test10.mp4"
    clip.write_bytes(b"data_test10")

    queue.register_file(clip, camera_type="rtsp")

    fake_time = FakeTimeContext(1000.0)
    upload_attempts = []

    def mock_fail(cmd, *args, **kwargs):
        res = MagicMock()
        if "copyto" in cmd:
            upload_attempts.append(fake_time.time())
            res.returncode = 1
            res.stderr = "network down"
            return res
        elif "lsf" in cmd:
            res.returncode = 0
            res.stdout = ""
            return res
        res.returncode = 0
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_fail):

        ok, msg = queue.run_until_empty(max_wait_sec=1000, poll_interval_sec=1.0, staging_dirs=[])
        assert ok is False
        assert "permanently FAILED_EXHAUSTED" in msg

        # Total attempts across ALL cycles must be strictly bounded:
        # (max_recovery_cycles + 1) * max_attempts = (2 + 1) * 3 = 9 attempts
        assert len(upload_attempts) == 9

        item = queue.get_item("rtsp:motion_test10.mp4")
        assert item["state"] == UploadState.FAILED_EXHAUSTED
        assert item["recovery_cycle"] == 2


def test_camera_safe_composite_keys_and_destination_mapping_regression(tmp_path):
    """Test 11: camera-safe composite keys (rtsp:*, usb:*) & destination mapping regression."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=50.0, max_recovery_cycles=2)

    tapo_clip = tmp_path / "tapo" / "motion_shared.mp4"
    tapo_clip.parent.mkdir()
    tapo_clip.write_bytes(b"tapo_bytes")

    usb_clip = tmp_path / "usb" / "motion_shared.mp4"
    usb_clip.parent.mkdir()
    usb_clip.write_bytes(b"usb_bytes")

    tapo_item = queue.register_file(tapo_clip, camera_type="rtsp", rclone_dest_path="")
    usb_item = queue.register_file(usb_clip, camera_type="usb", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")

    assert tapo_item["key"] == "rtsp:motion_shared.mp4"
    assert usb_item["key"] == "usb:motion_shared.mp4"

    # Exhaust both items at different timestamps
    queue._update_item_state("rtsp:motion_shared.mp4", UploadState.FAILED_EXHAUSTED, exhausted_at_ts=1000.0)
    queue._update_item_state("usb:motion_shared.mp4", UploadState.FAILED_EXHAUSTED, exhausted_at_ts=1020.0)

    # At 1055.0: TAPO has elapsed (1000 + 50 = 1050), USB has not (1020 + 50 = 1070)
    recovered = queue.reset_exhausted_items(now=1055.0)
    assert recovered == 1

    tapo_now = queue.get_item("motion_shared.mp4", camera_type="rtsp")
    usb_now = queue.get_item("motion_shared.mp4", camera_type="usb")

    assert tapo_now["state"] == UploadState.PENDING
    assert tapo_now["rclone_dest_path"] == ""

    assert usb_now["state"] == UploadState.FAILED_EXHAUSTED
    assert usb_now["rclone_dest_path"] == "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS"


def test_valid_upload_completion_reaches_daytime_idle_and_dispatch_is_idempotent(tmp_path):
    """Test 12: valid upload completion reaches DAYTIME_IDLE and dispatch is idempotent."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    clip = staging_dir / "motion_test12.mp4"
    clip.write_bytes(b"data_test12")
    sz = clip.stat().st_size

    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"

    queue = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=10.0)
    queue.register_file(clip, camera_type="rtsp")

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "lsf" in cmd:
            res.stdout = f"{sz};motion_test12.mp4\n"
        return res

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1800), \
         patch("scripts.daily_watchdog.check_and_recover_report") as mock_watchdog:

        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": False})

        ok, msg = queue.run_until_empty(max_wait_sec=50, poll_interval_sec=0.05, staging_dirs=[staging_dir])
        assert ok is True
        mock_watchdog.assert_called_once()

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["source_evidence_ready"] is True

        # Second call to complete_drain_if_ready: idempotent, does not re-dispatch watchdog
        mock_watchdog.reset_mock()
        ok2, msg2 = complete_drain_if_ready(temp_dirs=[], staging_dirs=[staging_dir])
        assert ok2 is True
        mock_watchdog.assert_not_called()


# ── Timing Budget & Cross-Invocation Wake Tests (14 Scenarios) ─────────────────

def test_wake_scenario_1_slow_upload_timeout_preserves_future_recovery_eligibility(tmp_path):
    """Scenario 1: Slow upload timeout can consume a large part of one invocation without losing future recovery eligibility."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        upload_timeout_sec=600,
        max_attempts=2,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
    )
    clip = tmp_path / "slow_motion.mp4"
    clip.write_bytes(b"slow_video_bytes")
    queue.register_file(clip, camera_type="rtsp")

    fake_time = FakeTimeContext(1000.0)

    def mock_slow_timeout(cmd, *args, **kwargs):
        res = MagicMock()
        if "copyto" in cmd:
            fake_time.sleep(600.0)
            res.returncode = 1
            res.stderr = "rclone copyto timed out after 600s"
            return res
        res.returncode = 0
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_slow_timeout):

        ok, msg = queue.run_until_empty(max_wait_sec=3600, poll_interval_sec=1.0, staging_dirs=[], schedule_wake_on_cooldown=False)
        item = queue.get_item("rtsp:slow_motion.mp4")
        assert item["recovery_cycle"] >= 1


def test_wake_scenario_2_invocation_timeout_preserves_and_schedules_future_recovery(tmp_path):
    """Scenario 2: Uploader invocation timing out does not eliminate a scheduled future recovery."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        upload_timeout_sec=600,
        max_attempts=5,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
    )
    clip = tmp_path / "timeout_clip.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip, camera_type="rtsp")

    queue._update_item_state(
        "rtsp:timeout_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    wake_calls = []

    def mock_wake(wait_sec, service_name=DEFAULT_WAKE_SERVICE_NAME, timer_unit=DEFAULT_WAKE_TIMER_UNIT):
        wake_calls.append((wait_sec, service_name, timer_unit))
        return True, "Mock wake scheduled"

    fake_time = FakeTimeContext(1050.0)
    with patch("time.time", side_effect=fake_time.time), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=mock_wake):
        ok, msg = queue.run_until_empty(max_wait_sec=10, poll_interval_sec=1.0, staging_dirs=[], schedule_wake_on_cooldown=True)
        assert ok is False
        assert len(wake_calls) == 1
        assert wake_calls[0][0] == pytest.approx(250.0, abs=1.0)


def test_wake_scenario_3_temporarily_failed_exhausted_produces_automatic_future_wake(tmp_path):
    """Scenario 3: Temporarily FAILED_EXHAUSTED item produces an automatic future wake."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=3,
        max_attempts=1,
    )
    clip = tmp_path / "wake_clip.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip, camera_type="rtsp")

    scheduled_wakes = []

    def mock_schedule(wait_sec, service_name=DEFAULT_WAKE_SERVICE_NAME, timer_unit=DEFAULT_WAKE_TIMER_UNIT):
        scheduled_wakes.append(wait_sec)
        return True, "Scheduled"

    fake_time = FakeTimeContext(1000.0)

    def mock_fail(cmd, *args, **kwargs):
        res = MagicMock()
        if "copyto" in cmd:
            res.returncode = 1
            res.stderr = "network error"
            return res
        res.returncode = 0
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_fail), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=mock_schedule):

        ok, msg = queue.run_until_empty(max_wait_sec=600, poll_interval_sec=1.0, staging_dirs=[], schedule_wake_on_cooldown=True)
        assert ok is False
        assert "in recovery cooldown" in msg
        assert len(scheduled_wakes) == 1
        assert scheduled_wakes[0] == pytest.approx(300.0, abs=1.0)

        item = queue.get_item("rtsp:wake_clip.mp4")
        assert item["state"] == UploadState.FAILED_EXHAUSTED
        assert item["earliest_recovery_ts"] == pytest.approx(1300.0, abs=1.0)


def test_wake_scenario_4_wake_after_cooldown_starts_next_cycle_without_human_action(tmp_path):
    """Scenario 4: Wake after cooldown starts the next cycle without human action across process boundary."""
    ledger_file = tmp_path / "ledger.json"
    clip = tmp_path / "proc_boundary.mp4"
    clip.write_bytes(b"data")
    sz = clip.stat().st_size

    # --- INVOCATION A ---
    queue_a = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=100.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    queue_a.register_file(clip, camera_type="rtsp")

    wake_scheduled = []
    with patch("time.time", return_value=1000.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="transient fail")), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=lambda w, **kw: (wake_scheduled.append(w), (True, "ok"))[1]):
        ok_a, msg_a = queue_a.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok_a is False
        assert len(wake_scheduled) == 1
        assert wake_scheduled[0] == pytest.approx(100.0, abs=1.0)

    item_after_a = queue_a.get_item("rtsp:proc_boundary.mp4")
    assert item_after_a["state"] == UploadState.FAILED_EXHAUSTED
    assert item_after_a["recovery_cycle"] == 0

    # --- TIME PASSES: 105s later, systemd timer fires, starting INVOCATION B ---
    queue_b = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=100.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )

    copyto_called = []
    def mock_success(cmd, *args, **kwargs):
        res = MagicMock(returncode=0)
        if "lsf" in cmd:
            res.stdout = f"{sz};proc_boundary.mp4\n" if copyto_called else ""
        elif "copyto" in cmd:
            copyto_called.append(True)
        return res

    with patch("time.time", return_value=1105.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_success), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")):
        ok_b, msg_b = queue_b.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok_b is True

    item_after_b = queue_b.get_item("rtsp:proc_boundary.mp4")
    assert item_after_b["state"] == UploadState.UPLOADED
    assert item_after_b["recovery_cycle"] == 1
    assert item_after_b["attempt_count"] == 1


def test_wake_scenario_5_multiple_recovery_cycles_across_separate_invocations(tmp_path):
    """Scenario 5: Multiple recovery cycles can occur across separate uploader invocations."""
    ledger_file = tmp_path / "ledger.json"
    clip = tmp_path / "multi_invoc.mp4"
    clip.write_bytes(b"data")
    sz = clip.stat().st_size

    # Cycle 0: Invocation 1 fails -> FAILED_EXHAUSTED (cycle 0)
    q1 = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=60.0, max_recovery_cycles=2, max_attempts=1)
    q1.register_file(clip, camera_type="rtsp")

    with patch("time.time", return_value=1000.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="fail1")), \
         patch("scripts.upload_queue.schedule_systemd_wake", return_value=(True, "ok")):
        ok1, _ = q1.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok1 is False

    assert q1.get_item("rtsp:multi_invoc.mp4")["recovery_cycle"] == 0

    # Cycle 1: Invocation 2 wakes at t=1070.0 (cooldown elapsed), fails again -> FAILED_EXHAUSTED (cycle 1)
    q2 = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=60.0, max_recovery_cycles=2, max_attempts=1)
    with patch("time.time", return_value=1070.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="fail2")), \
         patch("scripts.upload_queue.schedule_systemd_wake", return_value=(True, "ok")):
        ok2, _ = q2.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok2 is False

    assert q2.get_item("rtsp:multi_invoc.mp4")["recovery_cycle"] == 1

    # Cycle 2: Invocation 3 wakes at t=1140.0 (cooldown elapsed), succeeds!
    q3 = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=60.0, max_recovery_cycles=2, max_attempts=1)

    def mock_q3_run(cmd, *args, **kwargs):
        res = MagicMock(returncode=0)
        if "lsf" in cmd:
            res.stdout = f"{sz};multi_invoc.mp4\n"
        return res

    with patch("time.time", return_value=1140.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_q3_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")):
        ok3, _ = q3.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok3 is True

    item_final = q3.get_item("rtsp:multi_invoc.mp4")
    assert item_final["state"] == UploadState.UPLOADED
    assert item_final["recovery_cycle"] == 2


def test_wake_scenario_6_max_recovery_cycle_ceiling_stops_future_scheduling(tmp_path):
    """Scenario 6: Max recovery-cycle ceiling stops future scheduling and cancels wake timer."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=1,
        max_attempts=1,
    )
    clip = tmp_path / "ceiling_clip.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip, camera_type="rtsp")

    queue._update_item_state(
        "rtsp:ceiling_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=1,
    )

    wake_calls = []
    cancel_calls = []

    with patch("time.time", return_value=10000.0), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=lambda *a, **k: (wake_calls.append(a), (True, "ok"))[1]), \
         patch("scripts.upload_queue.cancel_systemd_wake", side_effect=lambda *a, **k: (cancel_calls.append(a), (True, "ok"))[1]):
        ok, msg = queue.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is False
        assert "permanently FAILED_EXHAUSTED (max cycles reached)" in msg
        assert len(wake_calls) == 0
        assert len(cancel_calls) == 1


def test_wake_scenario_7_reboot_restart_reconstructs_pending_future_recovery(tmp_path):
    """Scenario 7: Reboot/restart reconstructs pending future recovery from ledger."""
    ledger_file = tmp_path / "ledger.json"
    clip = tmp_path / "reboot_clip.mp4"
    clip.write_bytes(b"data")

    q_init = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0, max_recovery_cycles=2)
    q_init.register_file(clip, camera_type="rtsp")
    q_init._update_item_state(
        "rtsp:reboot_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    raw_data = json.loads(ledger_file.read_text())
    assert raw_data["items"]["rtsp:reboot_clip.mp4"]["earliest_recovery_ts"] == pytest.approx(1300.0)
    assert raw_data["next_recovery_ts"] == pytest.approx(1300.0)

    q_boot = UploadQueue(ledger_path=ledger_file, recovery_cooldown_sec=300.0, max_recovery_cycles=2)
    assert q_boot.get_earliest_recovery_ts(now=1100.0) == pytest.approx(1300.0)

    wake_scheduled = []
    with patch("time.time", return_value=1100.0), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=lambda w, **kw: (wake_scheduled.append(w), (True, "ok"))[1]):
        ok, msg = q_boot.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is False
        assert len(wake_scheduled) == 1
        assert wake_scheduled[0] == pytest.approx(200.0, abs=1.0)


def test_wake_scenario_8_uploading_crash_recovery_resets_to_pending(tmp_path):
    """Scenario 8: UPLOADING crash recovery remains correct across reboot/crash."""
    ledger_file = tmp_path / "ledger.json"
    clip = tmp_path / "crash_clip.mp4"
    clip.write_bytes(b"data")

    q1 = UploadQueue(ledger_path=ledger_file)
    q1.register_file(clip, camera_type="rtsp")
    q1._update_item_state("rtsp:crash_clip.mp4", UploadState.UPLOADING)

    q2 = UploadQueue(ledger_path=ledger_file)
    assert q2.get_item("rtsp:crash_clip.mp4")["state"] == UploadState.UPLOADING

    recovered = q2.recover_pending()
    assert recovered == 1
    assert q2.get_item("rtsp:crash_clip.mp4")["state"] == UploadState.PENDING


def test_wake_scenario_9_no_rapid_timer_or_restart_hammering(tmp_path):
    """Scenario 9: No rapid timer or restart hammering (bounded scheduling)."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    clip = tmp_path / "hammer_clip.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip, camera_type="rtsp")

    wakes = []
    with patch("time.time", return_value=1000.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="fail")), \
         patch("scripts.upload_queue.schedule_systemd_wake", side_effect=lambda w, **kw: (wakes.append(w), (True, "ok"))[1]):
        ok, _ = queue.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is False
        assert len(wakes) == 1
        assert wakes[0] >= 300.0


def test_wake_scenario_10_successful_later_cycle_performs_remote_verification(tmp_path):
    """Scenario 10: Successful later cycle performs remote existence and size verification."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    clip = tmp_path / "verify_clip.mp4"
    clip.write_bytes(b"valid_payload_bytes")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:verify_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    commands_run = []

    def mock_run(cmd, *args, **kwargs):
        commands_run.append(" ".join(cmd) if isinstance(cmd, list) else str(cmd))
        res = MagicMock(returncode=0)
        if "lsf" in cmd:
            res.stdout = f"{sz};verify_clip.mp4\n"
        return res

    with patch("time.time", return_value=1100.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain done")):
        ok, _ = queue.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is True

    item = queue.get_item("rtsp:verify_clip.mp4")
    assert item["state"] == UploadState.UPLOADED
    assert item["remote_verified"] is True
    assert any("lsf" in c for c in commands_run)


def test_wake_scenario_11_late_success_completes_drain_report_dispatch_daytime_idle(tmp_path):
    """Scenario 11: Late success still completes drain, report dispatch, and DAYTIME_IDLE transition."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    clip = staging_dir / "late_success.mp4"
    clip.write_bytes(b"data")
    sz = clip.stat().st_size

    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"

    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    queue.register_file(clip, camera_type="rtsp", rclone_dest_path="")
    queue._update_item_state(
        "rtsp:late_success.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock(returncode=0)
        if "lsf" in cmd:
            res.stdout = f"{sz};late_success.mp4\n"
        return res

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file), "UPLOAD_LEDGER_PATH": str(ledger_file)}), \
         patch("time.time", return_value=1100.0), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.lifecycle_manager.get_fair_feeder_processes", return_value=[]), \
         patch("scripts.lifecycle_manager.is_service_active", return_value=False), \
         patch("scripts.lifecycle_manager.get_mem_available_mb", return_value=1800), \
         patch("scripts.daily_watchdog.check_and_recover_report") as mock_report_dispatch:

        write_state({"state": LifecycleState.LOCAL_MORNING_DRAIN, "services_active": False})

        ok, msg = queue.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[staging_dir])
        assert ok is True
        mock_report_dispatch.assert_called_once()

        state = read_state()
        assert state["state"] == LifecycleState.DAYTIME_IDLE
        assert state["source_evidence_ready"] is True
        assert state["local_drain_complete"] is True


def test_wake_scenario_12_unresolved_state_still_blocks_host_ready(tmp_path):
    """Scenario 12: Unresolved state still blocks host-ready and drain prerequisites."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
    )
    clip = tmp_path / "blocking_clip.mp4"
    clip.write_bytes(b"data")
    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:blocking_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
    )

    drained, reason = check_drain_prerequisites(temp_dirs=[], staging_dirs=[], upload_queue=queue)
    assert drained is False
    assert "unresolved upload(s) in queue" in reason


def test_wake_scenario_13_camera_safe_rtsp_usb_identity_remains_unchanged(tmp_path):
    """Scenario 13: Camera-safe rtsp:* / usb:* ledger separation remains unchanged."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    tapo = tmp_path / "motion_same.mp4"
    tapo.write_bytes(b"tapo")
    usb = tmp_path / "usb_dir" / "motion_same.mp4"
    usb.parent.mkdir()
    usb.write_bytes(b"usb")

    it_tapo = queue.register_file(tapo, camera_type="rtsp")
    it_usb = queue.register_file(usb, camera_type="usb")

    assert it_tapo["key"] == "rtsp:motion_same.mp4"
    assert it_usb["key"] == "usb:motion_same.mp4"
    assert it_tapo["key"] != it_usb["key"]


def test_wake_scenario_14_tapo_logitech_destination_routing_remains_unchanged(tmp_path):
    """Scenario 14: TAPO / Logitech destination routing remains unchanged."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)
    tapo = tmp_path / "motion_tapo.mp4"
    tapo.write_bytes(b"tapo")
    usb = tmp_path / "motion_usb.mp4"
    usb.write_bytes(b"usb")

    it_tapo = queue.register_file(tapo, camera_type="rtsp", rclone_dest_path="")
    it_usb = queue.register_file(usb, camera_type="usb", rclone_dest_path="14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS")

    assert it_tapo["rclone_dest_path"] == ""
    assert it_usb["rclone_dest_path"] == "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS"


def test_defensive_gap_1_wake_failure_uses_in_process_fallback_when_budget_permits(tmp_path):
    """Gap 1: When transient wake scheduling fails but budget permits, in-process fallback is used."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=50.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    clip = tmp_path / "wake_fail_clip.mp4"
    clip.write_bytes(b"data")
    sz = clip.stat().st_size

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:wake_fail_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    fake_time = FakeTimeContext(1000.0)

    def mock_run(cmd, *args, **kwargs):
        res = MagicMock(returncode=0)
        if "lsf" in cmd:
            res.stdout = f"{sz};wake_fail_clip.mp4\n"
        return res

    with patch("time.time", side_effect=fake_time.time), \
         patch("time.sleep", side_effect=fake_time.sleep), \
         patch("shutil.which", return_value="/usr/bin/rclone"), \
         patch("subprocess.run", side_effect=mock_run), \
         patch("scripts.upload_queue.schedule_systemd_wake", return_value=(False, "unit already loaded")), \
         patch("scripts.lifecycle_manager.complete_drain_if_ready", return_value=(True, "Drain completed")):

        ok, msg = queue.run_until_empty(max_wait_sec=200, schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is True
        assert "Uploads finished and drain completed" in msg
        # Assert in-process sleep fallback was used
        assert len(fake_time.sleep_calls) > 0

    item = queue.get_item("rtsp:wake_fail_clip.mp4")
    assert item["state"] == UploadState.UPLOADED
    assert item["remote_verified"] is True


def test_defensive_gap_2_wake_failure_with_insufficient_budget_fails_closed(tmp_path):
    """Gap 2: When transient wake scheduling fails and budget is insufficient, fail closed immediately."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    clip = tmp_path / "insufficient_clip.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:insufficient_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    with patch("time.time", return_value=1000.0), \
         patch("shutil.which", return_value="/usr/bin/systemd-run"), \
         patch("scripts.upload_queue.schedule_systemd_wake", return_value=(False, "permission denied")):

        ok, msg = queue.run_until_empty(max_wait_sec=60, schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is False
        assert "Automatic wake scheduling failed" in msg
        assert "permission denied" in msg
        assert "Scheduled wake at" not in msg

    item = queue.get_item("rtsp:insufficient_clip.mp4")
    assert item["state"] == UploadState.FAILED_EXHAUSTED
    assert item["earliest_recovery_ts"] == pytest.approx(1300.0)


def test_defensive_gap_3_wake_scheduling_success_exits_bounded_invocation(tmp_path):
    """Gap 3: When transient wake scheduling succeeds, cleanly exit bounded invocation."""
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(
        ledger_path=ledger_file,
        recovery_cooldown_sec=300.0,
        max_recovery_cycles=2,
        max_attempts=1,
    )
    clip = tmp_path / "success_wake_clip.mp4"
    clip.write_bytes(b"data")

    queue.register_file(clip, camera_type="rtsp")
    queue._update_item_state(
        "rtsp:success_wake_clip.mp4",
        UploadState.FAILED_EXHAUSTED,
        exhausted_at_ts=1000.0,
        recovery_cycle=0,
    )

    with patch("time.time", return_value=1000.0), \
         patch("shutil.which", return_value="/usr/bin/systemd-run"), \
         patch("scripts.upload_queue.schedule_systemd_wake", return_value=(True, "Scheduled")):

        ok, msg = queue.run_until_empty(schedule_wake_on_cooldown=True, staging_dirs=[])
        assert ok is False
        assert "Scheduled wake at" in msg
        assert "Exiting bounded invocation" in msg


def test_defensive_gap_4_evening_readiness_exits_cleanly_without_rewriting_state(tmp_path):
    """Gap 4: Legitimate EVENING_READINESS with empty queue exits cleanly without altering lifecycle state."""
    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"

    initial_state = {
        "state": LifecycleState.EVENING_READINESS,
        "ready": True,
        "services_active": False,
        "local_drain_complete": True,
        "last_event": "Evening check passed",
    }
    state_file.write_text(json.dumps(initial_state))

    queue = UploadQueue(ledger_path=ledger_file)

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file)}):
        ok, msg = queue.run_until_empty(staging_dirs=[])
        assert ok is True
        assert "EVENING_READINESS" in msg
        assert "Uploads finished and verified" in msg

        # Ensure state.json was NOT modified/rewritten to DAYTIME_IDLE
        saved_state = json.loads(state_file.read_text())
        assert saved_state["state"] == LifecycleState.EVENING_READINESS
        assert saved_state["last_event"] == "Evening check passed"


def test_defensive_gap_5_unknown_or_unexpected_lifecycle_state_fails_closed(tmp_path):
    """Gap 5: UNKNOWN or unexpected lifecycle states fail closed (not converted to success)."""
    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)

    # Sub-case A: UNKNOWN state
    state_file.write_text(json.dumps({"state": LifecycleState.UNKNOWN}))
    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file)}):
        ok, msg = queue.run_until_empty(staging_dirs=[])
        assert ok is False
        assert "Cannot complete drain: invalid or unexpected lifecycle state 'UNKNOWN'" in msg

    # Sub-case B: Arbitrary/corrupt state
    state_file.write_text(json.dumps({"state": "ARBITRARY_CORRUPTED_STATE"}))
    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file)}):
        ok, msg = queue.run_until_empty(staging_dirs=[])
        assert ok is False
        assert "Cannot complete drain: invalid or unexpected lifecycle state 'ARBITRARY_CORRUPTED_STATE'" in msg


def test_defensive_gap_6_morning_active_guard_refuses_daytime_idle_publication(tmp_path):
    """Gap 6: MORNING_ACTIVE guard refuses to publish DAYTIME_IDLE and fails closed."""
    state_file = tmp_path / "state.json"
    ledger_file = tmp_path / "ledger.json"
    queue = UploadQueue(ledger_path=ledger_file)

    state_file.write_text(json.dumps({
        "state": LifecycleState.MORNING_ACTIVE,
        "services_active": True,
        "active_workers": 2,
    }))

    with patch.dict(os.environ, {"FAIR_FEEDER_STATE_FILE": str(state_file)}):
        ok, msg = queue.run_until_empty(staging_dirs=[])
        assert ok is False
        assert "Cannot complete drain: invalid or unexpected lifecycle state 'MORNING_ACTIVE'" in msg

        saved_state = json.loads(state_file.read_text())
        assert saved_state["state"] == LifecycleState.MORNING_ACTIVE
        assert saved_state["services_active"] is True




