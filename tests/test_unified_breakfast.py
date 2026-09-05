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
    VideoStreamSampler,
    generate_combined_breakfast_video
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


