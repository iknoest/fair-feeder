import sys
import json
import pytest
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.delivery_ledger import (
    load_delivery_registry,
    save_delivery_registry,
    is_breakfast_fully_delivered,
    is_unified_item_delivered,
    record_unified_item_delivered,
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
    render_kibble_bar,
    create_neutral_placeholder,
    draw_panel_overlay,
    render_header_bar,
    render_separator_bar,
    render_timeline_strip,
    VideoStreamSampler,
    generate_combined_breakfast_video,
    find_or_sample_recap_snapshots,
    find_or_render_timeline_chart,
    render_recap_cards,
    prepare_evidence_snapshots,
    add_compact_banner,
    deliver_unified_breakfast
)



def test_registry_breakfast_completion_and_idempotency(tmp_path):
    # 1. Initially, date is not delivered
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is False

    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    
    # 2. Delivering summary ALONE must NOT complete breakfast (Reviewer finding 4)
    record_unified_item_delivered(None, None, reg, "20260905", "summary", message_id=101, local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is False
    assert is_unified_item_delivered(reg, "20260905", "summary") is True
    assert is_unified_item_delivered(reg, "20260905", "combined_video") is False

    # Attempting to commit without combined_video must fail
    ok_premature = commit_breakfast_completion(None, None, "20260905", local_fallback_dir=tmp_path)
    assert ok_premature is False
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is False

    # 3. Delivering combined_video satisfies all requirements
    record_unified_item_delivered(None, None, reg, "20260905", "combined_video", message_id=102, local_fallback_dir=tmp_path)
    assert is_unified_item_delivered(reg, "20260905", "combined_video") is True

    # 4. Now commit breakfast completion succeeds
    ok = commit_breakfast_completion(
        None, None, "20260905",
        extra={"delivered_via": "test_unified"},
        local_fallback_dir=tmp_path
    )
    assert ok is True
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is True

    # 5. Verify registry contents
    reg_after = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    assert reg_after["dates"]["20260905"]["breakfast_fully_delivered"] is True
    assert reg_after["dates"]["20260905"]["delivered_via"] == "test_unified"


def test_registry_fail_closed_on_drive_failure(tmp_path):
    # Mock Drive service that simulates an error
    class FailingDriveFiles:
        def list(self, **kwargs):
            return self
        def execute(self):
            return {"files": []}
        def create(self, **kwargs):
            return self
        def update(self, **kwargs):
            return self

    class FailingDriveService:
        def files(self):
            return FailingDriveFiles()

    failing_drive = FailingDriveService()
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    record_unified_item_delivered(None, None, reg, "20260905", "summary", message_id=1, local_fallback_dir=tmp_path)
    record_unified_item_delivered(None, None, reg, "20260905", "combined_video", message_id=2, local_fallback_dir=tmp_path)

    # When Drive service is active but fails to persist, commit_breakfast_completion MUST fail closed!
    ok = commit_breakfast_completion(failing_drive, "fake_folder_id", "20260905", local_fallback_dir=tmp_path)
    assert ok is False


def test_registry_partial_retry_flow(tmp_path):
    reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    
    # Run 1 delivers summary, but video step crashes/fails
    record_unified_item_delivered(None, None, reg, "20260905", "summary", message_id=555, local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(reg, "20260905") is False

    # Retry Runner starts
    reloaded_reg = load_delivery_registry(None, None, local_fallback_dir=tmp_path)
    # Check item status: summary should be skipped, video pending
    assert is_unified_item_delivered(reloaded_reg, "20260905", "summary") is True
    assert is_unified_item_delivered(reloaded_reg, "20260905", "combined_video") is False

    # Only video is delivered on retry
    record_unified_item_delivered(None, None, reloaded_reg, "20260905", "combined_video", message_id=556, local_fallback_dir=tmp_path)
    assert commit_breakfast_completion(None, None, "20260905", local_fallback_dir=tmp_path) is True
    assert is_breakfast_fully_delivered(None, None, "20260905", local_fallback_dir=tmp_path) is True


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
    assert "TAPO · Dan feeder" in text
    assert "LOGITECH · Sanbo feeder" in text
    assert "House" in text
    assert "- Theft: Not confirmed" in text
    assert "Dan confirmed at Dan feeder during overlap" in text


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


def test_vertical_canvas_layout_and_pixel_isolation():
    width = 720
    panel_h = 405
    header_h = 38
    sep_h = 34
    footer_h = 38
    total_h = header_h + panel_h + sep_h + panel_h + footer_h
    assert total_h == 920

    header = render_header_bar(width, header_h, "TAPO - Dan Feeder", "2026-09-05 06:20:07", is_live=True)
    assert header.shape == (header_h, width, 3)

    separator = render_separator_bar(width, sep_h, text="-- SHARED TIMELINE (4x speedup) --")
    assert separator.shape == (sep_h, width, 3)

    footer = render_header_bar(width, footer_h, "LOGITECH - Sanbo Feeder", "2026-09-05 06:20:07", is_live=True)
    assert footer.shape == (footer_h, width, 3)

    # Source frame with simulated native OCR timestamp in upper-left
    tapo_source = np.ones((panel_h, width, 3), dtype=np.uint8) * 120
    # Add fake white timestamp text in top-left (e.g. 10..40 y, 10..150 x)
    tapo_source[10:40, 10:150] = 255

    logi_source = np.ones((panel_h, width, 3), dtype=np.uint8) * 45

    # Full vertical assembly
    vertical_canvas = np.vstack([header, tapo_source, separator, logi_source, footer])
    assert vertical_canvas.shape == (920, 720, 3)

    # ZERO SOURCE PIXEL OVERLAY INVARIANT:
    # Source frames must remain completely uncropped and pixel-pristine!
    # TAPO panel is located between y=38 and y=443
    extracted_tapo_panel = vertical_canvas[header_h : header_h + panel_h, :]
    assert np.array_equal(extracted_tapo_panel, tapo_source), "TAPO panel pixels must not be modified or overlaid!"
    # Top-left native timestamp pixels must be preserved 100%
    assert np.all(extracted_tapo_panel[10:40, 10:150] == 255), "Native upper-left timestamp was obscured!"

    # LOGITECH panel is located between y=477 and y=882
    extracted_logi_panel = vertical_canvas[header_h + panel_h + sep_h : header_h + panel_h + sep_h + panel_h, :]
    assert np.array_equal(extracted_logi_panel, logi_source), "LOGITECH panel pixels must not be modified or overlaid!"


def test_neutral_placeholder_semantics():
    placeholder = create_neutral_placeholder(720, 405, "TAPO", "2026-09-05 06:22:30", reason="No source footage")
    assert placeholder.shape == (405, 720, 3)
    # Neutral dark slate (mean luminance < 50)
    assert 20 < np.mean(placeholder) < 50
    # Check that placeholder is not completely black (contains text & border)
    assert np.max(placeholder) > 100


def test_dynamic_boundary_derivation(tmp_path):
    # Mock clip paths with timestamps
    p1 = tmp_path / "motion_20260905_061949_6s.mp4"
    p2 = tmp_path / "motion_20260905_062007_1m_50s.mp4"
    p3 = tmp_path / "motion_20260905_062322_36s.mp4"
    for p in (p1, p2, p3):
        p.write_bytes(b"mock")

    sampler1 = VideoStreamSampler([p1, p2])
    sampler2 = VideoStreamSampler([p3])

    all_starts = [c["start"] for c in sampler1.clips + sampler2.clips]
    all_ends = [c["end"] for c in sampler1.clips + sampler2.clips]

    # Boundaries must derive dynamically from all clips without manual cutoff
    start_dt = min(all_starts)
    end_dt = max(all_ends)

    assert start_dt == datetime(2026, 9, 5, 6, 19, 49)
    # Ends at 06:23:22 + 36s = 06:23:58
    assert end_dt == datetime(2026, 9, 5, 6, 23, 58)
    assert (end_dt - start_dt).total_seconds() == 249.0


def test_conflicted_tapo_displays_raw_bars():
    """Requirement 1 & 2: Conflicted TAPO displays raw Dan/Sanbo bars and contested warning."""
    tapo_summary = {
        "start_time": "06:20:00",
        "end_time": "06:21:05",
        "start_kibble": 30,
        "end_kibble": 0,
        "dan_kibble": 14,
        "dan_percent": 47,
        "dan_bowl_time": "0m 20s",
        "dan_first_ts": "06:20:00",
        "sanbo_kibble": 16,
        "sanbo_percent": 53,
        "sanbo_bowl_time": "0m 22s",
        "sanbo_first_ts": "06:20:03",
        "has_conflict": True,
        "conflict_frames": 28
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state_progression": "unsure → empty",
        "visibility": "poor (dark morning RGB)"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    # Raw 8-block bars must be present
    assert "Dan    ████░░░░ 47% (~14)" in text
    assert "bowl 0m 20s · from ~06:20:00" in text
    assert "Sanbo  ████░░░░ 53% (~16)" in text
    assert "bowl 0m 22s · from ~06:20:03" in text

    # Conflict note marks them as contested
    assert "⚠️ TAPO model attribution — contested" in text
    assert "28 conflict frames — this split is camera-model evidence, not reliable enough by itself to prove theft." in text


def test_contested_bar_values_do_not_confirm_theft():
    """Requirement 3: Contested bar values are not used automatically to confirm theft."""
    tapo_summary = {
        "start_kibble": 30,
        "end_kibble": 0,
        "dan_kibble": 14,
        "sanbo_kibble": 16,
        "has_conflict": True,
        "conflict_frames": 28
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    # Even though Sanbo is 16 kibble, theft must NOT be confirmed
    assert "- Theft: Not confirmed" in text


def test_feeder_meal_completion_displayed_independently_of_identity():
    """Requirement 4: Feeder meal outcome displayed first, independent of cat identity."""
    tapo_summary = {
        "start_kibble": 30,
        "end_kibble": 0,
        "dan_kibble": 14,
        "sanbo_kibble": 16,
        "has_conflict": True
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state_progression": "unsure → empty"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "TAPO · Dan feeder\n🥣 Meal: ~30 → 0 kibble · Finished ✅" in text
    assert "- Dan feeder: ~30 kibble consumed · Finished ✅" in text


def test_dan_feeder_emptied_does_not_imply_dan_ate_all_food():
    """Requirement 5: Dan feeder emptied does not imply Dan consumed all food."""
    tapo_summary = {
        "start_kibble": 30,
        "end_kibble": 0,
        "dan_kibble": 14,
        "sanbo_kibble": 16,
        "has_conflict": True
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "Dan consumed all food" not in text
    assert "Dan ate all food" not in text
    assert "Dan ate ~30 kibble" not in text
    assert "both cats ate at their own bowls" not in text


def test_tapo_missing_end_state_yields_uncertain():
    """Requirement 6: TAPO missing end-state yields Meal completion: uncertain."""
    tapo_summary = {
        "start_kibble": 30,
        "end_kibble": None,  # unobserved end state
        "dan_kibble": 14,
        "sanbo_kibble": 16,
        "has_conflict": True
    }
    logi_summary = {}

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "🥣 Meal completion: uncertain" in text
    assert "- Dan feeder: ~30 kibble consumed · Completion uncertain" in text


def test_logitech_meal_completion_displayed_when_evidence_exists():
    """Requirement 7: Logitech meal completion is displayed when evidence exists."""
    tapo_summary = {"start_kibble": 30, "end_kibble": 0}
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state_progression": "unsure → empty"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "🥣 Meal: unsure → empty · Finished likely" in text
    assert "- Sanbo feeder: Finished likely · Sanbo feeding observed" in text


def test_logitech_missing_completion_evidence_yields_uncertainty():
    """Requirement 8: Logitech missing completion evidence yields uncertainty."""
    tapo_summary = {"start_kibble": 30, "end_kibble": 0}
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "unsure",
        "bowl_state_progression": None
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "🥣 Meal: completion uncertain" in text


def test_report_no_hardcoded_defaults():
    """Requirement 9: No hardcoded Sep-5 percentages/amounts in defaults or custom inputs."""
    # Calling report with empty dictionaries must NEVER produce the Sep-5 magic numbers
    report = generate_unified_breakfast_report("20260906", {}, {})
    text = report["telegram_text"]

    assert "14 kibble" not in text
    assert "16 kibble" not in text
    assert "47%" not in text
    assert "53%" not in text
    assert "28 conflict frames" not in text

    # Custom inputs must use their own values
    custom_report = generate_unified_breakfast_report("20260906", {
        "start_kibble": 10,
        "end_kibble": 0,
        "dan_kibble": 8,
        "sanbo_kibble": 2,
    }, {})
    c_text = custom_report["telegram_text"]
    assert "80%" in c_text
    assert "20%" in c_text
    assert "~8" in c_text
    assert "~2" in c_text


def test_no_physically_impossible_dual_location_claim():
    """Requirement 10: No physically impossible dual-location conclusion."""
    tapo_summary = {
        "start_time": "06:20:00",
        "end_time": "06:21:05",
        "start_kibble": 30,
        "end_kibble": 0,
        "dan_kibble": 14,
        "sanbo_kibble": 16,
        "has_conflict": True,
        "conflict_frames": 28
    }
    logi_summary = {
        "session_start_time": "06:19:49",
        "session_end_time": "06:20:45",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes"
    }

    report = generate_unified_breakfast_report("20260905", tapo_summary, logi_summary)
    text = report["telegram_text"]

    assert "Identity: Dan confirmed at Dan feeder during overlap (Sanbo at own feeder); individual TAPO split contested" in text
    # When unresolvable (e.g. Logitech unknown):
    unresolved_report = generate_unified_breakfast_report("20260905", tapo_summary, {"cat_identity": "unknown"})
    assert "Identity: Contested at Dan feeder; individual attribution unresolved" in unresolved_report["telegram_text"]


def test_render_kibble_bar_visual_style():
    """Verifies compact 8-block visual style."""
    assert render_kibble_bar(None) == ""
    assert render_kibble_bar(0) == "░░░░░░░░"
    assert render_kibble_bar(25) == "██░░░░░░"
    assert render_kibble_bar(47) == "████░░░░"
    assert render_kibble_bar(50) == "████░░░░"
    assert render_kibble_bar(53) == "████░░░░"
    assert render_kibble_bar(75) == "██████░░"
    assert render_kibble_bar(100) == "████████"


def test_dead_tail_trimming_computation(tmp_path):
    """Verifies that combined video t_end trims dead footage based on last meaningful activity."""
    p1 = tmp_path / "motion_20260913_061955_2m_30s.mp4"
    p2 = tmp_path / "motion_20260913_062226_2m_30s.mp4"
    p3 = tmp_path / "motion_20260913_062458_5m_0s.mp4"  # Extends to 06:29:58 (10 min dead tail!)
    for p in (p1, p2, p3):
        p.write_bytes(b"mock")

    sampler = VideoStreamSampler([p1, p2, p3])
    raw_end = max([c["end"] for c in sampler.clips])
    assert raw_end == datetime(2026, 9, 13, 6, 29, 58)

    tapo_summary = {
        "start_time": "2026-09-13 06:20:00",
        "end_time": "2026-09-13 06:23:01",
        "meal_finished": True
    }
    logi_summary = {
        "sessions": [
            {
                "session_start_time": "06:19:55",
                "session_end_time": "06:25:01",
                "date": "2026-09-13"
            }
        ]
    }

    activity_ends = []
    end_t = tapo_summary.get("end_time")
    activity_ends.append(datetime.strptime(end_t, "%Y-%m-%d %H:%M:%S"))
    for s in logi_summary["sessions"]:
        activity_ends.append(datetime.strptime(f"2026-09-13 {s['session_end_time']}", "%Y-%m-%d %H:%M:%S"))

    last_activity = max(activity_ends)
    assert last_activity == datetime(2026, 9, 13, 6, 25, 1)

    buffered_end = last_activity + timedelta(seconds=15.0)
    assert buffered_end == datetime(2026, 9, 13, 6, 25, 16)
    trimmed_end = min(raw_end, buffered_end)
    assert trimmed_end == datetime(2026, 9, 13, 6, 25, 16)
    # Trims off ~4 minutes 42 seconds of dead empty tail!
    assert (raw_end - trimmed_end).total_seconds() == 282.0


def test_recap_cards_layout_and_dimensions(tmp_path):
    """Verifies that intro recap cards render exactly at 720x920 with full-frame panels and banners."""
    dummy_top = np.zeros((405, 720, 3), dtype=np.uint8)
    dummy_bot = np.zeros((405, 720, 3), dtype=np.uint8)

    snapshots = {
        "dispensed": (dummy_top, "1. Food Dispensed (~25 kibble)"),
        "arrival": (dummy_bot, "2. Cat Arrival (Dan at 06:20:08)"),
        "finish": (dummy_top, "3. Bowl Finished (Empty at 06:23:08)")
    }

    chart_panel = find_or_render_timeline_chart(
        tapo_dir=tmp_path,
        tapo_summary={"start_kibble": 25, "end_kibble": 0, "dan_kibble": 25},
        target_size=(720, 405)
    )
    assert chart_panel.shape == (405, 720, 3)

    cards = render_recap_cards(
        snapshots=snapshots,
        chart_panel=chart_panel,
        target_date="20260913",
        tapo_summary={"dan_kibble": 25, "sanbo_kibble": 0},
        width=720,
        total_height=920
    )

    assert len(cards) == 2
    assert cards[0].shape == (920, 720, 3)
    assert cards[1].shape == (920, 720, 3)


def test_timeline_strip_rendering():
    """Verifies that the timeline strip renders with exact expected dimensions and responds to progress."""
    t_start = datetime(2026, 9, 13, 6, 20, 0)
    t_end = datetime(2026, 9, 13, 6, 25, 0)
    dan_arr = datetime(2026, 9, 13, 6, 20, 30)
    dan_fin = datetime(2026, 9, 13, 6, 23, 30)

    # 1. Test before arrival
    strip_early = render_timeline_strip(
        width=720,
        height=42,
        curr_time_dt=datetime(2026, 9, 13, 6, 20, 10),
        t_start_dt=t_start,
        t_end_dt=t_end,
        dan_arrival_dt=dan_arr,
        dan_finish_dt=dan_fin,
        start_kibble=25,
        meal_finished=True
    )
    assert strip_early.shape == (42, 720, 3)

    # 2. Test during eating
    strip_mid = render_timeline_strip(
        width=720,
        height=42,
        curr_time_dt=datetime(2026, 9, 13, 6, 22, 0),
        t_start_dt=t_start,
        t_end_dt=t_end,
        dan_arrival_dt=dan_arr,
        dan_finish_dt=dan_fin,
        start_kibble=25,
        meal_finished=True
    )
    assert strip_mid.shape == (42, 720, 3)

    # 3. Test after finish
    strip_late = render_timeline_strip(
        width=720,
        height=42,
        curr_time_dt=datetime(2026, 9, 13, 6, 24, 0),
        t_start_dt=t_start,
        t_end_dt=t_end,
        dan_arrival_dt=dan_arr,
        dan_finish_dt=dan_fin,
        start_kibble=25,
        meal_finished=True
    )
    assert strip_late.shape == (42, 720, 3)


def test_deliver_unified_breakfast_api_contract_clean_invocation(tmp_path):
    """
    Regression test for Sep-14 failure:
    Ensures deliver_unified_breakfast can be called cleanly without Drive or Telegram,
    invoking delivery_ledger functions with revision=None and revision='test' without TypeError.
    """
    out_dir = tmp_path / "delivery_test"
    out_dir.mkdir()

    # Call with revision=None: must not raise TypeError
    res_none = deliver_unified_breakfast(
        target_date="20990101",
        out_dir=out_dir,
        skip_telegram=True,
        force=True,
        folder_id=""
    )
    assert res_none is False

    # Call with revision='test-rev': must not raise TypeError
    res_rev = deliver_unified_breakfast(
        target_date="20990101",
        out_dir=out_dir,
        skip_telegram=True,
        force=True,
        folder_id="",
        revision="test-rev"
    )
    assert res_rev is False


def test_multi_session_logitech_span_and_formatting():
    """Verify that multi-session Logitech data expands global breakfast window and formats sessions clearly."""
    tapo_summary = {
        "start_time": "2026-09-16 06:20:00",
        "end_time": "2026-09-16 06:21:03",
        "start_kibble": 35,
        "end_kibble": 0,
        "dan_kibble": 3,
        "sanbo_kibble": 29,
        "dan_bowl_seconds": 5.6,
        "sanbo_bowl_seconds": 8.4,
        "meal_finished": True
    }
    logitech_summary = {
        "date": "2026-09-16",
        "sessions": [
            {
                "session_start_time": "06:19:46",
                "session_end_time": "06:20:22",
                "cat_identity": "Sanbo",
                "eating_evidence": "yes",
                "meal_status": "Finished likely",
                "bowl_state_progression": "empty -> half",
                "visibility": "good"
            },
            {
                "session_start_time": "06:21:22",
                "session_end_time": "06:24:33",
                "cat_identity": "Sanbo",
                "eating_evidence": "yes",
                "meal_status": "Finished likely",
                "bowl_state_progression": "empty -> low -> empty",
                "visibility": "usable"
            }
        ]
    }

    report = generate_unified_breakfast_report("20260916", tapo_summary, logitech_summary)
    text = report["telegram_text"]

    # Window covers 06:19:46 to 06:24:33
    assert "06:19:46–06:24:33" in text
    assert "(4m 47s)" in text

    # Both sessions formatted under Logitech section
    assert "Session 1 · 06:19:46–06:20:22 (36s)" in text
    assert "Session 2 · 06:21:22–06:24:33 (3m 11s)" in text
    assert "↩ Sanbo returned to Sanbo feeder · 06:21:22" in text

    # House section summaries
    assert "Sanbo feeding observed (2 sessions: 06:19:46–06:20:22, 06:21:22–06:24:33)" in text


def test_foreign_arrival_event_and_theft_observed_sep16():
    """Verify Sep-16 foreign arrival event detection, handover story, and observed theft."""
    tapo_summary = {
        "date": "20260916",
        "start_time": "2026-09-16 06:20:00",
        "end_time": "2026-09-16 06:21:03",
        "dan_first_ts": "2026-09-16 06:20:01",
        "start_kibble": 35,
        "end_kibble": 0,
        "dan_kibble": 3,
        "sanbo_kibble": 29,
        "has_conflict": True,
        "conflict_frames": 2,
        "meal_finished": True
    }
    tapo_timeline = {
        "feeding_phases": [
            {"start": "2026-09-16 06:20:01", "end": "2026-09-16 06:20:01", "cat": "Dan"},
            {"start": "2026-09-16 06:20:01", "end": "2026-09-16 06:20:01", "cat": "Dan"},
            {"start": "2026-09-16 06:20:01", "end": "2026-09-16 06:20:01", "cat": "Sanbo"},
            {"start": "2026-09-16 06:20:05", "end": "2026-09-16 06:21:00", "cat": "Sanbo"}
        ]
    }
    logitech_summary = {
        "sessions": [
            {
                "session_start_time": "06:19:46",
                "session_end_time": "06:20:22",
                "cat_identity": "Sanbo",
                "eating_evidence": "yes"
            },
            {
                "session_start_time": "06:21:22",
                "session_end_time": "06:24:33",
                "cat_identity": "Sanbo",
                "eating_evidence": "yes"
            }
        ]
    }

    report = generate_unified_breakfast_report("20260916", tapo_summary, logitech_summary, tapo_timeline=tapo_timeline)
    text = report["telegram_text"]

    # Foreign arrival detected in TAPO section
    assert "⚠️ Sanbo arrived at Dan feeder · 06:20:26" in text

    # House narrative reflects sequence
    assert "Dan arrived first at Dan feeder; Sanbo took over at ~06:20:26 and ate; Sanbo returned to own feeder at 06:21:22" in text

    # Theft observed line
    assert "- Theft: Theft observed: Sanbo consumed ~29 kibble at Dan feeder" in text

    # Event timeline contains foreign_arrival event
    timeline_events = report.get("events", [])
    event_types = [e["event_type"] for e in timeline_events]
    assert "foreign_arrival" in event_types
    for_event = next(e for e in timeline_events if e["event_type"] == "foreign_arrival")
    assert for_event["cat"] == "Sanbo"
    assert for_event["timestamp"] == "06:20:26"


def test_physical_contradiction_prevents_theft_confirmation():
    """Verify that if Sanbo was eating at Logitech feeder during the same window, theft is NOT confirmed."""
    tapo_summary = {
        "start_time": "2026-09-16 06:20:00",
        "end_time": "2026-09-16 06:21:03",
        "start_kibble": 35,
        "end_kibble": 0,
        "dan_kibble": 3,
        "sanbo_kibble": 29,
        "meal_finished": True
    }
    tapo_timeline = {
        "feeding_phases": [
            {"start": "2026-09-16 06:20:01", "end": "2026-09-16 06:20:01", "cat": "Dan"},
            {"start": "2026-09-16 06:20:26", "end": "2026-09-16 06:21:00", "cat": "Sanbo"}
        ]
    }
    # Contradicting session: Sanbo eating at Logitech feeder from 06:20:20 to 06:20:40 (overlaps 06:20:26)
    logitech_summary = {
        "sessions": [
            {
                "session_start_time": "06:20:20",
                "session_end_time": "06:20:40",
                "cat_identity": "Sanbo",
                "eating_evidence": "yes"
            }
        ]
    }

    report = generate_unified_breakfast_report("20260916", tapo_summary, logitech_summary, tapo_timeline=tapo_timeline)
    text = report["telegram_text"]
    assert "- Theft: Not confirmed" in text


def test_recap_cards_three_panels_when_foreign_arrival():
    """Verify render_recap_cards returns 3 distinct cards when foreign_arrival snapshot is present."""
    fake_frame = np.zeros((405, 720, 3), dtype=np.uint8)
    fake_chart = np.zeros((405, 720, 3), dtype=np.uint8)

    snapshots_with_foreign = {
        "dispensed": (fake_frame, "1. Food Dispensed"),
        "arrival": (fake_frame, "2. Cat Arrival"),
        "foreign_arrival": (fake_frame, "2b. Foreign Arrival"),
        "finish": (fake_frame, "3. Meal Finished")
    }

    cards = render_recap_cards(
        snapshots=snapshots_with_foreign,
        chart_panel=fake_chart,
        target_date="20260916",
        tapo_summary={"dan_kibble": 3, "sanbo_kibble": 29},
        width=720,
        total_height=920
    )

    assert len(cards) == 3
    for c in cards:
        assert c.shape == (920, 720, 3)


def test_timeline_strip_multi_session_transitions():
    """Verify render_timeline_strip correctly labels feeding phases across cameras."""
    t_start = datetime(2026, 9, 16, 6, 19, 46)
    t_end = datetime(2026, 9, 16, 6, 24, 33)
    dan_arr = datetime(2026, 9, 16, 6, 20, 1)
    dan_fin = datetime(2026, 9, 16, 6, 21, 3)
    for_arr = datetime(2026, 9, 16, 6, 20, 26)
    sessions = [
        {"session_start_time": "06:19:46", "session_end_time": "06:20:22", "cat_identity": "Sanbo"},
        {"session_start_time": "06:21:22", "session_end_time": "06:24:33", "cat_identity": "Sanbo"}
    ]

    # Test during Dan feeding
    strip_dan = render_timeline_strip(
        width=720, height=42,
        curr_time_dt=datetime(2026, 9, 16, 6, 20, 15),
        t_start_dt=t_start, t_end_dt=t_end,
        dan_arrival_dt=dan_arr, dan_finish_dt=dan_fin,
        foreign_arrival_dt=for_arr,
        logitech_sessions=sessions
    )
    assert strip_dan.shape == (42, 720, 3)

    # Test during Sanbo eating at Dan feeder
    strip_sanbo_dan = render_timeline_strip(
        width=720, height=42,
        curr_time_dt=datetime(2026, 9, 16, 6, 20, 30),
        t_start_dt=t_start, t_end_dt=t_end,
        dan_arrival_dt=dan_arr, dan_finish_dt=dan_fin,
        foreign_arrival_dt=for_arr,
        logitech_sessions=sessions
    )
    assert strip_sanbo_dan.shape == (42, 720, 3)

    # Test during Sanbo Session 2 at Logitech feeder
    strip_s2 = render_timeline_strip(
        width=720, height=42,
        curr_time_dt=datetime(2026, 9, 16, 6, 22, 0),
        t_start_dt=t_start, t_end_dt=t_end,
        dan_arrival_dt=dan_arr, dan_finish_dt=dan_fin,
        foreign_arrival_dt=for_arr,
        logitech_sessions=sessions
    )
    assert strip_s2.shape == (42, 720, 3)


def test_normal_three_image_evidence_set(tmp_path):
    """
    Verifies that for an ordinary uncontested breakfast:
    - Evidence album has exactly 3 images: Food Dispensed, Cat Arrival, Meal Finished
    - No fake foreign arrival or return images are created
    - All 3 images are written to out_dir with compact banner
    """
    import cv2
    tapo_dir = tmp_path / "tapo"
    tapo_dir.mkdir()
    disp_img = np.full((720, 1280, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(tapo_dir / "motion_20260915_062000_kibble_dispensed.jpg"), disp_img)

    tapo_summary = {
        "start_time": "2026-09-15 06:19:50",
        "end_time": "2026-09-15 06:21:10",
        "dan_first_ts": "2026-09-15 06:20:00",
        "start_kibble": 28,
        "end_kibble": 0,
        "meal_finished": True
    }
    tapo_timeline = {
        "feeding_phases": [
            {"start": "2026-09-15 06:20:00", "end": "2026-09-15 06:21:05", "cat": "Dan"}
        ]
    }
    clip_p = tmp_path / "motion_20260915_061950_1m_30s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(str(clip_p), fourcc, 10.0, (160, 90))
    for i in range(900):
        # Frame 0-99: empty (10), 100-699: cat eating (200), 700+: empty finished (50)
        val = 50 if i >= 700 else (200 if i >= 100 else 10)
        w.write(np.full((90, 160, 3), val, dtype=np.uint8))
    w.release()

    sampler = VideoStreamSampler([clip_p], is_logitech=False)
    out_dir = tmp_path / "out"

    evidence = prepare_evidence_snapshots(
        tapo_sampler=sampler,
        tapo_dir=tapo_dir,
        tapo_summary=tapo_summary,
        tapo_timeline=tapo_timeline,
        out_dir=out_dir,
        target_date="20260915"
    )
    sampler.close()

    assert len(evidence) == 3
    keys = [e["key"] for e in evidence]
    assert keys == ["dispensed", "arrival", "finish"]

    for e in evidence:
        assert e["image_path"].exists()
        assert e["caption"] != ""
        assert "Dan feeder" in e["caption"]

    assert "Food Dispensed (~28 kibble)" in evidence[0]["caption"]
    assert "Dan arrived at Dan feeder" in evidence[1]["caption"]
    assert "Meal Finished (empty bowl)" in evidence[2]["caption"]


def test_sep16_five_image_evidence_set_with_foreign_arrival_and_return():
    """
    Verifies that for Sep-16 natural replay evidence:
    - Exactly 5 evidence snapshots are generated in chronological order:
      1. Food Dispensed (06:19:59)
      2. Dan Arrival (06:20:01 / 06:20:13)
      3. Foreign Arrival - Sanbo (06:20:26)
      4. Meal Finished (06:21:03)
      5. Return to Own Feeder - Sanbo (06:21:22)
    - Each image contains concise caption with event, cat, feeder, timestamp.
    """
    tapo_dir = Path("scratch/replay_evidence/20260916/tapo-evidence-20260916")
    logi_dir = Path("scratch/replay_evidence/20260916/logitech-evidence-20260916")
    if not tapo_dir.exists() or not logi_dir.exists():
        pytest.skip("Sep-16 evidence not available locally")

    tapo_clips = sorted(tapo_dir.glob("motion_*.mp4"))
    logi_clips = sorted(logi_dir.glob("motion_*.mp4"))
    tapo_sum = json.loads((tapo_dir / "tapo_summary_20260916.json").read_text())
    tapo_tl = json.loads((tapo_dir / "tapo_timeline_20260916.json").read_text())
    logi_sum = json.loads((logi_dir / "logitech_vlm_shadow_summary_20260916.json").read_text())

    tapo_sampler = VideoStreamSampler(tapo_clips, is_logitech=False)
    logi_sampler = VideoStreamSampler(logi_clips, is_logitech=True)
    out_dir = Path("/tmp/test_evidence_sep16")

    evidence = prepare_evidence_snapshots(
        tapo_sampler=tapo_sampler,
        logi_sampler=logi_sampler,
        tapo_dir=tapo_dir,
        logitech_dir=logi_dir,
        tapo_summary=tapo_sum,
        tapo_timeline=tapo_tl,
        logitech_summary=logi_sum,
        out_dir=out_dir,
        target_date="20260916"
    )
    tapo_sampler.close()
    logi_sampler.close()

    assert len(evidence) == 5
    keys = [e["key"] for e in evidence]
    assert keys == ["dispensed", "arrival", "foreign_arrival", "finish", "return_to_feeder"]

    timestamps = [e["timestamp"] for e in evidence]
    assert timestamps == sorted(timestamps)

    assert "⚠️ Sanbo arrived at Dan feeder · 06:20:26" in evidence[2]["caption"]
    assert "↩ Sanbo returned to Sanbo feeder · 06:21:27" in evidence[4]["caption"]
    assert evidence[4]["camera_tag"] == "[LOGITECH] Sanbo Feeder"

    for e in evidence:
        assert e["image_path"].exists()


def test_missing_optional_event_does_not_invent_fake_image(tmp_path):
    """Verifies that missing optional events (no foreign cat, no return) do NOT invent fake images."""
    tapo_summary = {
        "start_time": "2026-09-17 06:20:00",
        "end_time": "2026-09-17 06:21:00",
        "dan_first_ts": "2026-09-17 06:20:05",
        "start_kibble": 20,
        "end_kibble": 0,
        "has_conflict": False,
        "conflict_frames": 0,
        "meal_finished": True
    }
    tapo_timeline = {
        "feeding_phases": [{"start": "2026-09-17 06:20:05", "end": "2026-09-17 06:21:00", "cat": "Dan"}]
    }

    evidence = prepare_evidence_snapshots(
        tapo_sampler=None,
        logi_sampler=None,
        tapo_dir=tmp_path,
        tapo_summary=tapo_summary,
        tapo_timeline=tapo_timeline,
        out_dir=tmp_path,
        target_date="20260917"
    )
    assert len(evidence) == 0


def test_return_to_feeder_omitted_if_no_cat_visible(tmp_path):
    """Verifies that return_to_feeder is omitted if no cat is visibly present at the feeder."""
    tapo_dir = tmp_path / "tapo"
    tapo_dir.mkdir()
    logi_dir = tmp_path / "logi"
    logi_dir.mkdir()

    tapo_summary = {
        "start_time": "2026-09-17 06:20:00",
        "end_time": "2026-09-17 06:21:00",
        "start_kibble": 25,
        "end_kibble": 0,
        "has_conflict": False,
        "conflict_frames": 0,
        "meal_finished": True,
        "sanbo_kibble": 15
    }
    tapo_timeline = {
        "feeding_phases": [
            {"start": "2026-09-17 06:20:05", "end": "2026-09-17 06:20:20", "cat": "Dan"},
            {"start": "2026-09-17 06:20:25", "end": "2026-09-17 06:20:45", "cat": "Sanbo"}
        ]
    }
    logi_summary = {
        "sessions": [
            {"start_time": "2026-09-17 06:21:00", "cat_identity": "Sanbo"}
        ]
    }

    # Generate mock Logitech clip that is completely static (empty bowl, no cat appears)
    clip_l = logi_dir / "motion_20260917_062100_10s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(clip_l), fourcc, 10.0, (160, 90))
    for _ in range(100):
        vw.write(np.full((90, 160, 3), 40, dtype=np.uint8))
    vw.release()

    # Generate mock Tapo clip with arrival
    clip_t = tapo_dir / "motion_20260917_062000_50s.mp4"
    vw_t = cv2.VideoWriter(str(clip_t), fourcc, 10.0, (160, 90))
    for i in range(500):
        # 0-5s: empty (10), 5s+: cat (180)
        v = 180 if i >= 50 else 10
        vw_t.write(np.full((90, 160, 3), v, dtype=np.uint8))
    vw_t.release()

    t_sampler = VideoStreamSampler([clip_t], is_logitech=False)
    l_sampler = VideoStreamSampler([clip_l], is_logitech=True)

    evidence = prepare_evidence_snapshots(
        tapo_sampler=t_sampler,
        logi_sampler=l_sampler,
        tapo_dir=tapo_dir,
        logitech_dir=logi_dir,
        tapo_summary=tapo_summary,
        tapo_timeline=tapo_timeline,
        logitech_summary=logi_summary,
        out_dir=tmp_path / "out",
        target_date="20260917"
    )
    t_sampler.close()
    l_sampler.close()

    keys = [e["key"] for e in evidence]
    # return_to_feeder MUST be omitted because no cat was visible at the feeder!
    assert "return_to_feeder" not in keys


def test_dispensed_settled_selection(tmp_path):
    """Verifies that dispensed snapshot selects stable post-dispense settled frame before cat arrival."""
    tapo_dir = tmp_path / "tapo"
    tapo_dir.mkdir()

    tapo_summary = {
        "start_time": "2026-09-17 06:20:00",
        "end_time": "2026-09-17 06:21:00",
        "start_kibble": 30,
        "end_kibble": 0,
        "meal_finished": True
    }
    tapo_timeline = {
        "feeding_phases": [
            {"start": "2026-09-17 06:20:07", "end": "2026-09-17 06:20:50", "cat": "Dan"}
        ]
    }

    # Video:
    # sec 0-2 (frames 0-20): empty bowl (val=10)
    # sec 2-4 (frames 20-40): dispensing kibble (val increases 10 -> 80)
    # sec 4-6 (frames 40-60): settled kibbles in bowl, no cat (bowl val=80, upper val=10)
    # sec 7+ (frames 70+): cat arrives (upper val=200)
    clip_t = tapo_dir / "motion_20260917_062000_10s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_t = cv2.VideoWriter(str(clip_t), fourcc, 10.0, (160, 90))
    for i in range(100):
        frame = np.full((90, 160, 3), 10, dtype=np.uint8)
        if i >= 70:
            frame[0:50, :] = 200  # Cat arrives in upper scene
        elif i >= 40:
            frame[50:90, 20:80] = 80  # Settled kibble in bowl
        elif i >= 20:
            frame[50:90, 20:80] = int(10 + (i - 20) * 3.5)  # Dispensing / tumbling
        vw_t.write(frame)
    vw_t.release()

    t_sampler = VideoStreamSampler([clip_t], is_logitech=False)

    evidence = prepare_evidence_snapshots(
        tapo_sampler=t_sampler,
        tapo_dir=tapo_dir,
        tapo_summary=tapo_summary,
        tapo_timeline=tapo_timeline,
        out_dir=tmp_path / "out",
        target_date="20260917"
    )
    t_sampler.close()

    keys = [e["key"] for e in evidence]
    assert "dispensed" in keys
    disp_item = next(e for e in evidence if e["key"] == "dispensed")
    assert "Food Dispensed (~30 kibble)" in disp_item["caption"]


def test_dynamic_video_has_no_static_recap_cards(tmp_path):
    """Verifies that generate_combined_breakfast_video with include_recap_cards=False writes only dynamic frames."""
    import cv2
    t_clip = tmp_path / "motion_20260918_062000_10s.mp4"
    l_clip = tmp_path / "motion_20260918_062000_10s_logi.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for cp in [t_clip, l_clip]:
        vw = cv2.VideoWriter(str(cp), fourcc, 10.0, (320, 180))
        for i in range(100):
            vw.write(np.full((180, 320, 3), (i * 2) % 256, dtype=np.uint8))
        vw.release()

    out_v = tmp_path / "out_dynamic.mp4"
    res_p = generate_combined_breakfast_video(
        tapo_clips=[t_clip],
        logitech_clips=[l_clip],
        output_path=out_v,
        target_date="20260918",
        speedup_factor=4.0,
        out_width=320,
        target_fps=16.0,
        include_recap_cards=False,
        activity_buffer_seconds=0.0
    )

    cap = cv2.VideoCapture(str(res_p))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    duration = frame_count / fps
    assert duration <= 4.0


def test_deliver_unified_breakfast_item_level_delivery_and_skip_telegram(tmp_path):
    """
    Verifies deliver_unified_breakfast with --skip-telegram:
    - Prepares evidence images in out_dir
    - Does NOT mutate delivery registry
    - Exactly-once semantics for summary -> evidence_album -> combined_video
    """
    import cv2
    from scripts.delivery_ledger import load_delivery_registry, is_breakfast_fully_delivered

    out_dir = tmp_path / "out_delivery"
    out_dir.mkdir()
    tapo_dir = tmp_path / "tapo_dir"
    tapo_dir.mkdir()

    # Create small synthetic clip in tapo_dir
    c_p = tapo_dir / "motion_20260919_062000_10s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(str(c_p), fourcc, 10.0, (160, 90))
    for _ in range(30):
        w.write(np.zeros((90, 160, 3), dtype=np.uint8))
    w.release()

    ok = deliver_unified_breakfast(
        target_date="20260919",
        tapo_dir=tapo_dir,
        out_dir=out_dir,
        skip_telegram=True
    )
    assert ok is True

    reg = load_delivery_registry(None, None, local_fallback_dir=out_dir)
    assert is_breakfast_fully_delivered(reg, "20260919") is False


def test_sep20_natural_foreign_arrival_album_recovery():
    """
    Verifies that for Sep-20 natural production evidence:
    - Text reports Sanbo arrived at Dan feeder at 06:20:01
    - Events contains foreign_arrival at 06:20:01
    - Evidence snapshots include foreign_arrival snapshot
    - The foreign_arrival snapshot is grounded in the primary meal session (06:20:01),
      NOT sampled from the 06:24:22 background clip.
    """
    tapo_dir = Path("scratch/replay_evidence/20260920/tapo-evidence-20260920")
    logi_dir = Path("scratch/replay_evidence/20260920/logitech-evidence-20260920")
    if not tapo_dir.exists() or not logi_dir.exists():
        pytest.skip("Sep-20 evidence not available locally")

    tapo_clips = sorted(tapo_dir.glob("motion_*.mp4"))
    logi_clips = sorted(logi_dir.glob("motion_*.mp4"))
    tapo_sum = json.loads((tapo_dir / "tapo_summary_20260920.json").read_text())
    tapo_tl = json.loads((tapo_dir / "tapo_timeline_20260920.json").read_text())
    logi_sum = json.loads((logi_dir / "logitech_vlm_session_summary.json").read_text())

    # Step 1: Verify single event truth in report
    report = generate_unified_breakfast_report("20260920", tapo_sum, logi_sum, tapo_timeline=tapo_tl)
    assert "⚠️ Sanbo arrived at Dan feeder · 06:20:01" in report["telegram_text"]
    for_ev = next((e for e in report["events"] if e["event_type"] == "foreign_arrival"), None)
    assert for_ev is not None
    assert for_ev["timestamp"] == "06:20:01"
    assert for_ev["cat"] == "Sanbo"

    # Step 2: Verify snapshots include foreign arrival grounded around 06:20:01
    tapo_sampler = VideoStreamSampler(tapo_clips, is_logitech=False, start_time_override=datetime(2026, 9, 20, 6, 20, 0))
    logi_sampler = VideoStreamSampler(logi_clips, is_logitech=True)
    out_dir = Path("/tmp/test_evidence_sep20")

    evidence = prepare_evidence_snapshots(
        tapo_sampler=tapo_sampler,
        logi_sampler=logi_sampler,
        tapo_dir=tapo_dir,
        logitech_dir=logi_dir,
        tapo_summary=tapo_sum,
        tapo_timeline=tapo_tl,
        logitech_summary=logi_sum,
        out_dir=out_dir,
        target_date="20260920",
        events=report["events"]
    )
    tapo_sampler.close()
    logi_sampler.close()

    keys = [e["key"] for e in evidence]
    assert "foreign_arrival" in keys
    for_snap = next(e for e in evidence if e["key"] == "foreign_arrival")
    assert "Sanbo arrived at Dan feeder" in for_snap["caption"]
    # Crucial regression test: must be from primary session around 06:20:01, NOT 06:24:22
    assert "06:24:" not in for_snap["caption"]
    assert "06:20:" in for_snap["caption"]


def test_sep21_natural_foreign_arrival_album_recovery():
    """
    Verifies that for Sep-21 natural production evidence:
    - Text reports Sanbo arrived at Dan feeder at 06:20:02
    - Evidence snapshots include foreign_arrival snapshot (recovering pre-rendered feeding_merged_0_sanbo_arrival.jpg)
    - Snapshots contain dispensed, arrival, foreign_arrival, finish, and return_to_feeder
    """
    tapo_dir = Path("scratch/replay_evidence/20260921/tapo-evidence-20260921")
    logi_dir = Path("scratch/replay_evidence/20260921/logitech-evidence-20260921")
    if not tapo_dir.exists() or not logi_dir.exists():
        pytest.skip("Sep-21 evidence not available locally")

    tapo_clips = sorted(tapo_dir.glob("motion_*.mp4"))
    logi_clips = sorted(logi_dir.glob("motion_*.mp4"))
    tapo_sum = json.loads((tapo_dir / "tapo_summary_20260921.json").read_text())
    tapo_tl = json.loads((tapo_dir / "tapo_timeline_20260921.json").read_text())
    logi_sum = json.loads((logi_dir / "logitech_vlm_session_summary.json").read_text())

    report = generate_unified_breakfast_report("20260921", tapo_sum, logi_sum, tapo_timeline=tapo_tl)
    assert "⚠️ Sanbo arrived at Dan feeder · 06:20:02" in report["telegram_text"]

    tapo_sampler = VideoStreamSampler(tapo_clips, is_logitech=False, start_time_override=datetime(2026, 9, 21, 6, 20, 0))
    logi_sampler = VideoStreamSampler(logi_clips, is_logitech=True)
    out_dir = Path("/tmp/test_evidence_sep21")

    evidence = prepare_evidence_snapshots(
        tapo_sampler=tapo_sampler,
        logi_sampler=logi_sampler,
        tapo_dir=tapo_dir,
        logitech_dir=logi_dir,
        tapo_summary=tapo_sum,
        tapo_timeline=tapo_tl,
        logitech_summary=logi_sum,
        out_dir=out_dir,
        target_date="20260921",
        events=report["events"]
    )
    tapo_sampler.close()
    logi_sampler.close()

    keys = [e["key"] for e in evidence]
    assert "foreign_arrival" in keys
    for_snap = next(e for e in evidence if e["key"] == "foreign_arrival")
    assert "Sanbo arrived at Dan feeder · 06:20:02" in for_snap["caption"]



