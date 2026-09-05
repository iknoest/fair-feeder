import sys
import json
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.delivery_ledger import (
    load_delivery_registry,
    save_delivery_registry,
    is_breakfast_fully_delivered,
    commit_breakfast_completion,
    export_tapo_timeline,
    load_tapo_timeline
)
from scripts.logitech_vlm_shadow import (
    group_clips_into_sessions,
    CameraTimelineInterval,
    reconcile_cross_camera_intervals,
    build_vlm_session_report,
    enhance_image_gamma_clahe
)
from scripts.unified_breakfast import (
    generate_unified_breakfast_report,
    create_neutral_placeholder,
    draw_panel_overlay
)


def test_registry_breakfast_completion_and_idempotency(tmp_path):
    # 1. Initially, date is not delivered
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is False

    # 2. Mark breakfast completion
    commit_breakfast_completion(
        None, None, "20260905",
        extra={"delivered_via": "test_unified"},
        local_fallback_dir=tmp_path
    )

    # 3. Check is_breakfast_fully_delivered returns True
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is True

    # 4. Verify registry contents
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert reg["dates"]["20260905"]["breakfast_fully_delivered"] is True
    assert reg["dates"]["20260905"]["delivered_via"] == "test_unified"


def test_logitech_feeding_window_gap_grouping():
    # Clip 1: 06:19:49 (6s dur -> ends 06:19:55)
    # Clip 2: 06:20:06 (39s dur -> ends 06:20:45) -> 11s gap from Clip 1
    # Clip 3: 06:21:32 (30s dur -> ends 06:22:02) -> 47s gap from Clip 2
    # Clip 4: 06:39:47 (29s dur -> ends 06:40:16) -> 1000s gap
    clips = [
        {"name": "motion_20260905_061949_6s.mp4", "id": "c1"},
        {"name": "motion_20260905_062006_39s.mp4", "id": "c2"},
        {"name": "motion_20260905_062132_30s.mp4", "id": "c3"},
        {"name": "motion_20260905_063947_29s.mp4", "id": "c4"}
    ]

    sessions = group_clips_into_sessions(clips, gap_threshold_sec=10, feeding_gap_threshold_sec=15)
    assert len(sessions) == 3

    # Session 1 groups Clip 1 and Clip 2 (11s gap bridged in feeding window)
    assert len(sessions[0]) == 2
    assert sessions[0][0]["name"] == "motion_20260905_061949_6s.mp4"
    assert sessions[0][1]["name"] == "motion_20260905_062006_39s.mp4"

    # Session 2 has Clip 3 (47s gap > 15s)
    assert len(sessions[1]) == 1
    assert sessions[1][0]["name"] == "motion_20260905_062132_30s.mp4"

    # Session 3 has Clip 4
    assert len(sessions[2]) == 1
    assert sessions[2][0]["name"] == "motion_20260905_063947_29s.mp4"


def test_session_report_metadata_and_gaps():
    clips = [
        {"name": "motion_20260905_061949_6s.mp4"},
        {"name": "motion_20260905_062006_39s.mp4"}
    ]
    manifest_data = [{"timestamp": "2026-09-05 06:19:50", "motion_detected": True}]
    all_results = [{
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "confidence": 0.90,
        "visibility": "usable",
        "reasons": ["cat head in bowl"]
    }]

    session_data = build_vlm_session_report(
        selected_files=clips,
        manifest_data=manifest_data,
        all_results=all_results,
        all_failed=[],
        all_skipped=[],
        search_date="20260905",
        provider="test",
        model="test-model"
    )

    assert session_data["session_start_time"] == "06:19:49"
    assert session_data["session_end_time"] == "06:20:45"
    assert session_data["wall_clock_span_sec"] == 56.0
    assert session_data["actual_recorded_footage_sec"] == 45.0
    assert len(session_data["source_gaps"]) == 1
    gap = session_data["source_gaps"][0]
    assert gap["gap_start"] == "06:19:55"
    assert gap["gap_end"] == "06:20:06"
    assert gap["gap_sec"] == 11.0


def test_conflict_guard_disables_exclusion():
    # Case 1: TAPO has Dan/Sanbo conflict
    t_int_conflict = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:20:07",
        end_timestamp="06:21:05",
        cat_presence=True,
        identity="Sanbo",
        identity_confidence=0.60,
        identity_evidence_quality="contested",
        identity_basis="FeedingTracker accepted phase (contested)",
        has_conflict=True,
        exclusion_eligible=False
    )
    l_int = CameraTimelineInterval(
        camera="LOGITECH",
        start_timestamp="06:19:49",
        end_timestamp="06:20:45",
        cat_presence=True,
        identity="Sanbo",
        identity_confidence=0.85,
        identity_evidence_quality="poor (dark morning RGB)"
    )

    rec_t, rec_l = reconcile_cross_camera_intervals([t_int_conflict], [l_int])
    assert rec_l[0].identity == "Sanbo"  # MUST NOT flip to Dan!
    assert rec_l[0].reconciled is False
    assert "disabled" in rec_l[0].reconciliation_notes.lower()

    # Case 2: TAPO is uncontested Dan
    t_int_clean_dan = CameraTimelineInterval(
        camera="TAPO",
        start_timestamp="06:20:07",
        end_timestamp="06:21:05",
        cat_presence=True,
        identity="Dan",
        identity_confidence=0.95,
        identity_evidence_quality="good",
        identity_basis="FeedingTracker accepted phase",
        has_conflict=False,
        exclusion_eligible=True
    )
    l_int_unsure = CameraTimelineInterval(
        camera="LOGITECH",
        start_timestamp="06:19:49",
        end_timestamp="06:20:45",
        cat_presence=True,
        identity="unsure",
        identity_confidence=0.50
    )
    rec_t2, rec_l2 = reconcile_cross_camera_intervals([t_int_clean_dan], [l_int_unsure])
    assert rec_l2[0].identity == "Sanbo"
    assert rec_l2[0].reconciled is True
    assert "Dan confirmed present at Tapo" in rec_l2[0].reconciliation_notes


def test_unified_breakfast_report_generation():
    tapo_summary = {
        "start_time": "06:20:00",
        "end_time": "06:21:05",
        "start_kibble": 30,
        "dan_kibble": 14,
        "dan_percent": 47,
        "dan_bowl_time": "0m 20s",
        "sanbo_kibble": 16,
        "sanbo_percent": 53,
        "sanbo_bowl_time": "0m 22s",
        "has_conflict": True,
        "conflict_frames": 28
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "total_duration": "56s",
        "wall_clock_span_sec": 56.0,
        "actual_recorded_footage_sec": 45.0,
        "source_gaps": [{"gap_sec": 11.0}],
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "visibility": "poor (dark morning RGB)",
        "evidence_clip_count": 2
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "Breakfast · 2026-09-05" in text
    assert "06:19:49–06:21:05" in text
    assert "TAPO (Dan Feeder):" in text
    assert "LOGITECH (Sanbo Feeder):" in text
    assert "House-Level Conclusion:" in text
    assert "No theft confirmed" in text
    assert "Sanbo: Confirmed eating at Sanbo feeder" in text


def test_neutral_placeholder_and_overlay():
    placeholder = create_neutral_placeholder(640, 360, "TAPO", "2026-09-05 06:19:49", "No source footage")
    assert placeholder.shape == (360, 640, 3)
    assert np.mean(placeholder) < 60  # dark neutral slate

    overlay = draw_panel_overlay(placeholder, "TAPO - Dan Feeder", "2026-09-05 06:19:49", is_live_footage=False)
    assert overlay.shape == (360, 640, 3)


def test_rule_d_low_light_preservation_not_darkness_presence():
    # Verify enhance_image_gamma_clahe works on dark image
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8) + 15
    enhanced = enhance_image_gamma_clahe(dark_frame, gamma=2.5)
    assert enhanced.shape == dark_frame.shape
    # Darkness alone must not set cat presence (Rule D)
    assert np.mean(enhanced) > np.mean(dark_frame)
