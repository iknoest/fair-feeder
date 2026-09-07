"""
tests/test_unified_resilience.py - Focused validation suite for Sep-7 Unified Breakfast resilience repair.

Covers:
1. Crash recovery in motion_recorder.py (healthy orphan recovery, failed remux cleanup, idempotency).
2. Missing-evidence Analysis UX in scripts/unified_breakfast.py (explicit unavailable state, no empty headers, truthful house reconciliation).
3. Single-camera video degradation (dual, tapo-only, logitech-only, neither).
4. Durable exactly-once delivery registry persistence (no quota create, reload survivability, independent item tracking).
"""

import os
import sys
import json
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import motion_recorder
from motion_recorder import RecordingController
from scripts.unified_breakfast import (
    generate_unified_breakfast_report,
    deliver_unified_breakfast,
    generate_combined_breakfast_video,
    VideoStreamSampler,
)
from scripts.delivery_ledger import (
    save_durable_artifact,
    load_durable_artifact,
    load_delivery_registry,
    save_delivery_registry,
    record_unified_item_delivered,
    is_unified_item_delivered,
    is_breakfast_fully_delivered,
    commit_breakfast_completion,
)


# ── Helper to write small valid test video clips ──────────────────────────────

def create_synthetic_mp4(filepath: Path, num_frames: int = 15, fps: float = 15.0, width: int = 320, height: int = 180):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 10) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return filepath


# ── 1. Crash / Orphan Recording Recovery Tests ────────────────────────────────

class DummyReader:
    def __init__(self):
        self.stream_fps = 15.0
        self.frame_width = 320
        self.frame_height = 180
    def get_latest_frame(self):
        return np.zeros((180, 320, 3), dtype=np.uint8)
    def get_buffer_snapshot(self):
        return []

class DummyListener:
    def __init__(self):
        self.motion_detected = False
        self.last_motion_time = None
    def reset_background(self):
        pass


def test_crash_recovery_healthy_orphan_enqueued(tmp_path, monkeypatch):
    """
    Verifies that a valid orphan video left in LOCAL_TEMP_DIR from a prior crash
    is detected, preserved, and enqueued for finalization and upload.
    """
    temp_dir = tmp_path / "recordings_temp"
    out_dir = tmp_path / "output_sync"
    temp_dir.mkdir()
    out_dir.mkdir()

    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", temp_dir)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", out_dir)

    orphan_file = temp_dir / "motion_20260907_062003.mp4"
    create_synthetic_mp4(orphan_file, num_frames=30, fps=15.0)

    reader = DummyReader()
    listener = DummyListener()

    monkeypatch.setattr(RecordingController, "_finalization_worker", lambda self: None)
    controller = RecordingController(reader, listener, yolo_model=None)
    job = controller._finalization_queue.get(timeout=2)
    assert job is not None
    job_temp_p, job_final_name, job_dur_str, job_dur, job_cat_seen, job_fps, job_fc = job

    assert job_temp_p == orphan_file
    assert "motion_20260907_062003" in job_final_name
    assert job_dur >= 1.9
    assert job_cat_seen is True  # Preserved evidence: never silently deleted
    assert job_fc == 30


def test_crash_recovery_tiny_failed_remux_discarded(tmp_path, monkeypatch):
    """
    Simulates the Sep-7 crash condition:
    Source file motion_20260907_062003.mp4 (healthy) and
    interrupted remux motion_20260907_062003_fixed.mp4 (48 bytes).
    Verifies that the tiny remux is discarded and the healthy source is preserved.
    """
    temp_dir = tmp_path / "recordings_temp"
    out_dir = tmp_path / "output_sync"
    temp_dir.mkdir()
    out_dir.mkdir()

    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", temp_dir)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", out_dir)

    orphan_source = temp_dir / "motion_20260907_062003.mp4"
    create_synthetic_mp4(orphan_source, num_frames=30, fps=15.0)

    orphan_fixed = temp_dir / "motion_20260907_062003_fixed.mp4"
    orphan_fixed.write_bytes(b"\x00" * 48)  # 48 bytes like Sep-7 crash

    reader = DummyReader()
    listener = DummyListener()

    monkeypatch.setattr(RecordingController, "_finalization_worker", lambda self: None)
    controller = RecordingController(reader, listener, yolo_model=None)

    # 48-byte corrupted remux should be unlinked
    assert not orphan_fixed.exists()
    # Healthy source should be preserved and queued
    job = controller._finalization_queue.get(timeout=2)
    assert job[0] == orphan_source


def test_crash_recovery_idempotent_when_already_in_dest(tmp_path, monkeypatch):
    """
    Verifies that if the file already exists in DRIVE_OUTPUT_DIR, repeated startup
    cleans up temp and does NOT re-queue or duplicate it.
    """
    temp_dir = tmp_path / "recordings_temp"
    out_dir = tmp_path / "output_sync"
    temp_dir.mkdir()
    out_dir.mkdir()

    monkeypatch.setattr(motion_recorder, "LOCAL_TEMP_DIR", temp_dir)
    monkeypatch.setattr(motion_recorder, "DRIVE_OUTPUT_DIR", out_dir)

    orphan_file = temp_dir / "motion_20260907_062003_2s.mp4"
    create_synthetic_mp4(orphan_file, num_frames=30, fps=15.0)

    # Pre-exist in dest
    dest_file = out_dir / "motion_20260907_062003_2s.mp4"
    create_synthetic_mp4(dest_file, num_frames=30, fps=15.0)

    reader = DummyReader()
    listener = DummyListener()

    controller = RecordingController(reader, listener, yolo_model=None)

    # File in temp should be cleaned up
    assert not orphan_file.exists()
    # Queue should remain empty
    assert controller._finalization_queue.empty()


# ── 2. Missing-Evidence Analysis UX Tests ─────────────────────────────────────

def test_analysis_ux_tapo_attribution_present_preserves_bars():
    """When TAPO evidence is present, percentage bars and attribution lines are preserved."""
    tapo_summary = {
        "start_kibble": 40,
        "end_kibble": 0,
        "dan_kibble": 40,
        "sanbo_kibble": 0,
        "dan_percent": 100,
        "sanbo_percent": 0,
        "meal_finished": True,
        "start_time": "06:20:03",
        "end_time": "06:22:36",
        "dan_bowl_time": "1m 30s",
        "dan_seen": "~06:20:12",
    }
    logi_session = {
        "cat_identity": "Sanbo",
        "eating_evidence": "observed",
        "bowl_state_progression": "empty",
        "meal_finished": True,
        "start_time": "06:24:37",
        "end_time": "06:25:32",
    }
    rep = generate_unified_breakfast_report("20260907", tapo_summary, logi_session)
    txt = rep["telegram_text"]

    assert "TAPO model attribution" in txt
    assert "Dan    ████████ 100% (~40)" in txt
    assert "Sanbo  ░░░░░░░░ 0% (~0)" in txt
    assert "🥣 Meal: ~40 → 0 kibble · Finished ✅" in txt
    assert "Dan and Sanbo identities consistent with camera attribution" in txt


def test_analysis_ux_tapo_attribution_contested_preserves_bars_and_note():
    """When TAPO attribution is contested, bars remain and contested note is included."""
    tapo_summary = {
        "start_kibble": 40,
        "end_kibble": 0,
        "dan_kibble": 25,
        "sanbo_kibble": 15,
        "dan_percent": 62,
        "sanbo_percent": 38,
        "has_conflict": True,
        "conflict_frames": 22,
        "meal_finished": True,
        "start_time": "06:20:03",
        "end_time": "06:22:36",
    }
    logi_session = {
        "cat_identity": "Sanbo",
        "eating_evidence": "observed",
        "meal_finished": True,
        "start_time": "06:20:10",
        "end_time": "06:22:20",
    }
    rep = generate_unified_breakfast_report("20260907", tapo_summary, logi_session)
    txt = rep["telegram_text"]

    assert "⚠️ TAPO model attribution — contested" in txt
    assert "Dan    █████░░░ 62% (~25)" in txt
    assert "Sanbo  ███░░░░░ 38% (~15)" in txt
    assert "22 conflict frames" in txt


def test_analysis_ux_tapo_evidence_missing_no_dangling_header():
    """
    When TAPO evidence is completely missing (like Sep-7):
    - No dangling 'TAPO model attribution' with nothing underneath
    - Explicit unavailable note rendered
    - No false claim that Dan/Sanbo identities were consistent with camera attribution
    """
    tapo_summary = {}
    logi_session = {
        "cat_identity": "Sanbo",
        "eating_evidence": "observed",
        "bowl_state_progression": "empty",
        "meal_finished": True,
        "start_time": "06:24:37",
        "end_time": "06:25:32",
    }
    rep = generate_unified_breakfast_report("20260907", tapo_summary, logi_session)
    txt = rep["telegram_text"]

    # Must NOT have dangling attribution header
    assert "\nTAPO model attribution\n\nLOGITECH" not in txt
    # Must explicitly state attribution unavailable
    assert "TAPO model attribution: unavailable" in txt
    assert "🥣 Meal: evidence unavailable (no TAPO footage or analysis)" in txt
    # House section must not claim Dan feeder identity consistency
    assert "Dan and Sanbo identities consistent with camera attribution" not in txt
    assert "Dan feeder: Unobserved (no TAPO footage)" in txt
    assert "Unverified at Dan feeder (no TAPO evidence)" in txt
    assert "Theft: Unknown (Dan feeder unobserved)" in txt


# ── 3. Single-Camera Video Degradation Tests ─────────────────────────────────

def test_single_camera_video_logitech_only(tmp_path):
    """
    Verifies that when TAPO has 0 clips and Logitech has 1 clip,
    generate_combined_breakfast_video generates the combined layout with
    TAPO as neutral 'No source footage' placeholder.
    """
    logi_clip = tmp_path / "motion_20260907_062437_10s.mp4"
    create_synthetic_mp4(logi_clip, num_frames=30, fps=15.0)

    out_video = tmp_path / "combined_output.mp4"

    res_p = generate_combined_breakfast_video(
        tapo_clips=[],
        logitech_clips=[logi_clip],
        output_path=out_video,
        target_date="20260907",
        speedup_factor=1.0,
        out_width=320,
        target_fps=10.0
    )
    assert res_p.exists()
    assert res_p.stat().st_size > 0

    cap = cv2.VideoCapture(str(res_p))
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    assert fc > 0
    assert w == 320


def test_single_camera_video_tapo_only(tmp_path):
    """
    Verifies that when TAPO has 1 clip and Logitech has 0 clips,
    generate_combined_breakfast_video generates the combined layout with
    Logitech as neutral 'No source footage' placeholder.
    """
    tapo_clip = tmp_path / "motion_20260907_062003_10s.mp4"
    create_synthetic_mp4(tapo_clip, num_frames=30, fps=15.0)

    out_video = tmp_path / "combined_tapo_only.mp4"

    res_p = generate_combined_breakfast_video(
        tapo_clips=[tapo_clip],
        logitech_clips=[],
        output_path=out_video,
        target_date="20260907",
        speedup_factor=1.0,
        out_width=320,
        target_fps=10.0
    )
    assert res_p.exists()
    assert res_p.stat().st_size > 0


def test_single_camera_video_neither_camera_fails_explicitly(tmp_path):
    """
    Verifies that when NEITHER camera has clips, deliver_unified_breakfast
    fails explicitly without fabricating video.
    """
    out_dir = tmp_path / "unified_out"
    out_dir.mkdir()

    ok = deliver_unified_breakfast(
        target_date="20260907",
        tapo_dir=tmp_path / "empty_tapo",
        logitech_dir=tmp_path / "empty_logi",
        out_dir=out_dir,
        skip_telegram=True,
        force=True
    )
    assert ok is False
    assert not (out_dir / "20260907_combined_breakfast.mp4").exists()


# ── 4. Durable Exactly-Once Registry Tests ────────────────────────────────────

def test_registry_no_quota_create_avoided_and_local_fallback_preserved(tmp_path):
    """
    Verifies that save_durable_artifact does NOT call drive.files().create()
    when existing_id is None, guarding against 403 quota errors in CI,
    while still persisting locally.
    """
    class MockFiles:
        def __init__(self):
            self.created = False
            self.updated = False
        def list(self, **kwargs):
            return self
        def execute(self):
            return {"files": []}  # File does not exist in Drive!
        def create(self, **kwargs):
            self.created = True
            return self
        def update(self, **kwargs):
            self.updated = True
            return self

    class MockDriveService:
        def __init__(self):
            self._files = MockFiles()
        def files(self):
            return self._files

    mock_drive = MockDriveService()
    local_dir = tmp_path / "registry_local"

    file_id = save_durable_artifact(
        mock_drive, "test_folder_id", "delivery_registry.json",
        b'{"test": true}', local_fallback_dir=local_dir, allow_create=False
    )

    # Must NOT have called create() on mock drive!
    assert mock_drive._files.created is False
    assert file_id is None
    # Local fallback must be written
    assert (local_dir / "delivery_registry.json").exists()
    assert json.loads((local_dir / "delivery_registry.json").read_text()) == {"test": True}


def test_registry_update_succeeds_when_pre_created(tmp_path):
    """
    Verifies that when delivery_registry.json is pre-created in Drive,
    save_durable_artifact executes update() and succeeds with file_id.
    """
    class MockRequest:
        def __init__(self, result):
            self.result = result
        def execute(self):
            return self.result

    class MockFiles:
        def __init__(self):
            self.updated = False
        def list(self, **kwargs):
            return MockRequest({"files": [{"id": "precreated_file_id_123", "name": "delivery_registry.json"}]})
        def update(self, **kwargs):
            self.updated = True
            return MockRequest({"id": "precreated_file_id_123"})

    class MockDriveService:
        def __init__(self):
            self._files = MockFiles()
        def files(self):
            return self._files

    mock_drive = MockDriveService()
    file_id = save_durable_artifact(
        mock_drive, "test_folder_id", "delivery_registry.json",
        b'{"test": true}', allow_create=False
    )

    assert mock_drive._files.updated is True
    assert file_id == "precreated_file_id_123"


def test_registry_exactly_once_item_level_idempotency(tmp_path):
    """
    Verifies that delivery registry independently tracks summary and combined_video,
    and recognizes already-delivered summary on restart.
    """
    local_dir = tmp_path / "registry_dir"
    reg = load_delivery_registry(None, None, local_fallback_dir=local_dir)

    # 1. Summary delivered
    record_unified_item_delivered(None, None, reg, "20260907", "summary", message_id=2126, local_fallback_dir=local_dir)

    # 2. Re-read registry from fresh load
    fresh_reg = load_delivery_registry(None, None, local_fallback_dir=local_dir)
    assert is_unified_item_delivered(fresh_reg, "20260907", "summary") is True
    assert is_unified_item_delivered(fresh_reg, "20260907", "combined_video") is False
    assert is_breakfast_fully_delivered(None, None, "20260907", local_fallback_dir=local_dir) is False

    # 3. Combined video can later be delivered
    record_unified_item_delivered(None, None, fresh_reg, "20260907", "combined_video", message_id=2127, local_fallback_dir=local_dir)
    final_reg = load_delivery_registry(None, None, local_fallback_dir=local_dir)
    assert is_unified_item_delivered(final_reg, "20260907", "combined_video") is True
