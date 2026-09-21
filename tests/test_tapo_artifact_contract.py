"""
tests/test_tapo_artifact_contract.py

Validation suite for TAPO CI evidence artifact contract repair.
Ensures:
1. Fail-closed producer validation (scripts/validate_tapo_artifact.py).
2. Fail-safe consumer integrity distinction (scripts/unified_breakfast.py).
3. End-to-end evidence artifact bundle packaging and consumption across the CI job boundary.
4. Sep-8 historical model evidence transport regression proving lossless preservation of recorded model output (~24 Dan / ~10 Sanbo / 71% / 29% + conflict metadata) and video recovery across the CI boundary.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_tapo_artifact import validate_tapo_evidence_bundle
from scripts.unified_breakfast import (
    generate_unified_breakfast_report,
    deliver_unified_breakfast,
    generate_combined_breakfast_video,
)


def create_synthetic_mp4(
    filepath: Path,
    num_frames: int = 15,
    fps: float = 15.0,
    width: int = 320,
    height: int = 180
) -> Path:
    """Creates a small valid synthetic mp4 file for testing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return filepath


@pytest.fixture(autouse=True)
def isolate_drive_and_telegram(monkeypatch):
    """Ensure tests in this module never touch live Google Drive or Telegram."""
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("GDRIVE_OUTPUT_FOLDER_ID", raising=False)
    monkeypatch.delenv("GDRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GDRIVE_UPLOAD_FOLDER_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


# ── Test 1: Pre-fix Artifact Fails Assertions (Sep-8 Failure Model) ───────────

def test_pre_fix_artifact_fails_assertions(tmp_path, capsys):
    """
    Models the exact Sep-8 defect: tapo-report exported only tapo_timeline into /tmp/output/,
    omitting tapo_summary and source video clips from the artifact bundle.
    Verifies that:
    1. The producer validator rejects this incomplete bundle with ValueError.
    2. The consumer (deliver_unified_breakfast) detects pipeline integrity failure,
       logs [Pipeline Integrity Error], and reports 'artifact incomplete' rather than masking it.
    """
    date = "20260908"
    evidence_dir = tmp_path / "tapo_evidence_sep8_broken"
    evidence_dir.mkdir()

    # Only timeline is present, recording that Dan fed
    timeline_data = {
        "date": date,
        "camera": "TAPO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feeding_phases": [
            {
                "cat": "Dan",
                "start_time": "06:20:03",
                "end_time": "06:21:40",
                "duration_seconds": 97,
                "confidence": 0.88
            }
        ]
    }
    (evidence_dir / f"tapo_timeline_{date}.json").write_text(json.dumps(timeline_data), encoding="utf-8")

    # 1. Producer validator must fail-closed
    with pytest.raises(ValueError) as exc_info:
        validate_tapo_evidence_bundle(evidence_dir, date)
    assert "Missing required tapo_summary_20260908.json" in str(exc_info.value)
    assert "Missing TAPO source feeding video(s)" in str(exc_info.value)

    # 2. Consumer delivery run
    out_dir = tmp_path / "unified_out"
    out_dir.mkdir()

    # Provide minimal Logitech artifact so the job proceeds
    logi_dir = tmp_path / "logi_dir"
    logi_dir.mkdir()
    (logi_dir / "summary.json").write_text(json.dumps({
        "session_start_time": "06:20:10",
        "session_end_time": "06:21:30",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "visibility": "normal"
    }), encoding="utf-8")
    create_synthetic_mp4(logi_dir / f"motion_{date}_062010_1m_20s.mp4", num_frames=30)

    deliver_unified_breakfast(
        target_date=date,
        tapo_dir=evidence_dir,
        logitech_dir=logi_dir,
        out_dir=out_dir,
        skip_telegram=True,
        force=True
    )

    captured = capsys.readouterr()
    assert "[Pipeline Integrity Error] Upstream TAPO evidence was detected in timeline but omitted from delivery artifact bundle" in captured.out

    # Verify consumer output report truthfully states artifact incomplete
    report_p = out_dir / f"unified_report_{date}.json"
    assert report_p.exists()
    report = json.loads(report_p.read_text(encoding="utf-8"))
    assert "artifact incomplete" in report["telegram_text"]
    assert "Unobserved (artifact incomplete)" in report["telegram_text"]
    assert "⚠️ TAPO model attribution: pipeline artifact incomplete" in report["telegram_text"]


# ── Test 2: Synthetic Artifact Bundle Boundary Transfer ────────────────────────

def test_synthetic_artifact_bundle_boundary_transfer(tmp_path):
    """
    Synthetic contract test: verifies that when summary, timeline, and source feeding
    clips are staged into the artifact bundle, validation passes and consumer fully
    recovers attribution and dual-panel combined video across the GHA boundary.

    NOTE: This test uses synthetic fixture values solely to verify pipeline transport
    mechanics and does not claim to represent historical or physical ground truth.
    """
    date = "20260908"
    producer_stage_dir = tmp_path / "producer_output"
    producer_stage_dir.mkdir()

    # 1. Producer stages complete bundle
    summary_data = {
        "date": date,
        "camera": "TAPO",
        "start_kibble": 30,
        "end_kibble": 2,
        "dan_kibble": 28,
        "sanbo_kibble": 0,
        "dan_percent": 100,
        "sanbo_percent": 0,
        "dan_bowl_seconds": 97,
        "dan_first_arrival": "06:20:03",
        "meal_finished": True,
        "start_time": "06:20:03",
        "end_time": "06:21:40"
    }
    (producer_stage_dir / f"tapo_summary_{date}.json").write_text(json.dumps(summary_data), encoding="utf-8")

    timeline_data = {
        "date": date,
        "camera": "TAPO",
        "feeding_phases": [
            {"cat": "Dan", "start_time": "06:20:03", "end_time": "06:21:40"}
        ]
    }
    (producer_stage_dir / f"tapo_timeline_{date}.json").write_text(json.dumps(timeline_data), encoding="utf-8")

    create_synthetic_mp4(producer_stage_dir / f"motion_{date}_062003_2m_30s.mp4", num_frames=30)

    # Validate producer bundle
    res = validate_tapo_evidence_bundle(producer_stage_dir, date)
    assert res["feeding_detected"] is True
    assert res["has_summary"] is True
    assert res["has_timeline"] is True
    assert res["source_clips_count"] == 1

    # 2. Simulate GitHub Actions artifact upload -> download boundary to isolated consumer dir
    consumer_tapo_dir = tmp_path / "consumer_tapo_evidence"
    shutil.copytree(producer_stage_dir, consumer_tapo_dir)

    # Logitech evidence
    consumer_logi_dir = tmp_path / "consumer_logitech_evidence"
    consumer_logi_dir.mkdir()
    (consumer_logi_dir / "summary.json").write_text(json.dumps({
        "session_start_time": "06:20:10",
        "session_end_time": "06:21:30",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "visibility": "normal"
    }), encoding="utf-8")
    create_synthetic_mp4(consumer_logi_dir / f"motion_{date}_062010_1m_20s.mp4", num_frames=30)

    out_dir = tmp_path / "unified_delivery_out"
    out_dir.mkdir()

    ok = deliver_unified_breakfast(
        target_date=date,
        tapo_dir=consumer_tapo_dir,
        logitech_dir=consumer_logi_dir,
        out_dir=out_dir,
        skip_telegram=True,
        force=True
    )
    assert ok is True

    # Check report
    report_p = out_dir / f"unified_report_{date}.json"
    assert report_p.exists()
    report = json.loads(report_p.read_text(encoding="utf-8"))

    # Assert Dan attribution recovered
    assert "Dan" in report["telegram_text"]
    assert "100%" in report["telegram_text"]
    assert "(~28)" in report["telegram_text"]
    assert "Finished ✅" in report["telegram_text"]
    assert "unavailable (no source footage/inference)" not in report["telegram_text"]
    assert "artifact incomplete" not in report["telegram_text"]

    # Assert combined video generated
    video_p = out_dir / f"{date}_combined_breakfast.mp4"
    assert video_p.exists()
    assert video_p.stat().st_size > 0


# ── Test 3: Producer Validator Fails on Incomplete Evidence ───────────────────

def test_producer_validator_fails_on_incomplete_evidence(tmp_path):
    """Tests all failure modes of validate_tapo_evidence_bundle when feeding occurred."""
    date = "20260908"

    # Case A: Missing summary
    d1 = tmp_path / "case_a"
    d1.mkdir()
    (d1 / f"tapo_timeline_{date}.json").write_text(json.dumps({"feeding_phases": [{"cat": "Dan"}]}))
    create_synthetic_mp4(d1 / f"motion_{date}_062000.mp4")
    with pytest.raises(ValueError, match="Missing required tapo_summary_20260908.json"):
        validate_tapo_evidence_bundle(d1, date)

    # Case B: Missing timeline
    d2 = tmp_path / "case_b"
    d2.mkdir()
    (d2 / f"tapo_summary_{date}.json").write_text(json.dumps({"dan_kibble": 20}))
    create_synthetic_mp4(d2 / f"motion_{date}_062000.mp4")
    with pytest.raises(ValueError, match="Missing required tapo_timeline_20260908.json"):
        validate_tapo_evidence_bundle(d2, date)

    # Case C: Missing source clips
    d3 = tmp_path / "case_c"
    d3.mkdir()
    (d3 / f"tapo_summary_{date}.json").write_text(json.dumps({"dan_kibble": 20}))
    (d3 / f"tapo_timeline_{date}.json").write_text(json.dumps({"feeding_phases": [{"cat": "Dan"}]}))
    with pytest.raises(ValueError, match="Missing TAPO source feeding video"):
        validate_tapo_evidence_bundle(d3, date)


# ── Test 4: Producer Validator Passes on Complete Bundle ──────────────────────

def test_producer_validator_passes_on_valid_bundle(tmp_path):
    date = "20260908"
    d = tmp_path / "valid_bundle"
    d.mkdir()
    (d / f"tapo_summary_{date}.json").write_text(json.dumps({"dan_kibble": 25, "meal_finished": True}))
    (d / f"tapo_timeline_{date}.json").write_text(json.dumps({"feeding_phases": [{"cat": "Dan"}]}))
    create_synthetic_mp4(d / f"motion_{date}_062000_2m_00s.mp4")

    res = validate_tapo_evidence_bundle(d, date)
    assert res["feeding_detected"] is True
    assert res["has_summary"] is True
    assert res["has_timeline"] is True
    assert res["source_clips_count"] == 1


# ── Test 5: Producer Validator Passes on Genuine No-Footage ───────────────────

def test_producer_validator_passes_on_genuine_no_footage(tmp_path):
    """Verifies that on genuine no-footage days, validation passes without requiring videos."""
    date = "20260908"
    d = tmp_path / "no_footage"
    d.mkdir()
    (d / f"tapo_timeline_{date}.json").write_text(json.dumps({"feeding_phases": []}))

    res = validate_tapo_evidence_bundle(d, date)
    assert res["feeding_detected"] is False
    assert res["source_clips_count"] == 0


# ── Test 6: Consumer Distinguishes Pipeline Failure from No-Footage ───────────

def test_consumer_distinguishes_pipeline_failure_from_no_footage():
    date = "20260908"
    logi_session = {
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "session_start_time": "06:20:00",
        "session_end_time": "06:21:00"
    }

    # Case 1: Genuine absence (no footage, no upstream feeding)
    report_genuine = generate_unified_breakfast_report(
        target_date=date,
        tapo_summary={},
        logitech_session=logi_session,
        pipeline_artifact_incomplete=False
    )
    assert "evidence unavailable (no TAPO footage or analysis)" in report_genuine["telegram_text"]
    assert "⚠️ TAPO model attribution: unavailable (no source footage/inference)" in report_genuine["telegram_text"]
    assert "Dan feeder: Unobserved (no TAPO footage)" in report_genuine["telegram_text"]

    # Case 2: Pipeline failure (upstream analysis succeeded, but artifact dropped)
    report_failure = generate_unified_breakfast_report(
        target_date=date,
        tapo_summary={},
        logitech_session=logi_session,
        pipeline_artifact_incomplete=True
    )
    assert "evidence unavailable (pipeline artifact incomplete)" in report_failure["telegram_text"]
    assert "⚠️ TAPO model attribution: pipeline artifact incomplete (upstream analysis succeeded but clips/summary missing from artifact)" in report_failure["telegram_text"]
    assert "Dan feeder: Unobserved (artifact incomplete)" in report_failure["telegram_text"]


# ── Test 7: CLI Invocation of validate_tapo_artifact.py ───────────────────────

def test_validate_tapo_artifact_cli(tmp_path):
    """Verifies CLI exit codes of validate_tapo_artifact.py."""
    date = "20260908"
    d = tmp_path / "cli_test"
    d.mkdir()

    script_path = REPO_ROOT / "scripts" / "validate_tapo_artifact.py"

    # Incomplete -> Exit code 1
    (d / f"tapo_timeline_{date}.json").write_text(json.dumps({"feeding_phases": [{"cat": "Dan"}]}))
    res_fail = subprocess.run(
        [sys.executable, str(script_path), "--evidence-dir", str(d), "--date", date],
        capture_output=True,
        text=True
    )
    assert res_fail.returncode == 1
    assert "TAPO evidence artifact validation FAILED" in res_fail.stderr

    # Complete -> Exit code 0
    (d / f"tapo_summary_{date}.json").write_text(json.dumps({"dan_kibble": 20}))
    create_synthetic_mp4(d / f"motion_{date}_062000.mp4")
    res_pass = subprocess.run(
        [sys.executable, str(script_path), "--evidence-dir", str(d), "--date", date],
        capture_output=True,
        text=True
    )
    assert res_pass.returncode == 0
    assert "TAPO evidence artifact bundle validated" in res_pass.stdout


# ── Test 8: Sep-8 Historical Model Evidence Transport Regression ──────────────

def test_sep8_historical_model_evidence_transport(tmp_path):
    """
    Regression transport test using actual recorded Sep-8 production model outputs.

    PURPOSE & SCOPE:
    Proves lossless transport of recorded production model outputs across the CI
    producer -> artifact -> consumer boundary:
    1. Staging summary, timeline, and source clip in /tmp/output/ satisfies the contract.
    2. Consumer receives complete evidence and eliminates the false 'No source footage' state.
    3. House report truthfully reflects the recorded production model outputs:
       - Dan ~24 kibble (71%), Sanbo ~10 kibble (29%), meal finished
       - 7 conflict frames flagged, marking TAPO attribution as contested
       - Sanbo feeding observed at Sanbo feeder
    4. Combined video is composited with live TAPO and Logitech streams.

    IMPORTANT SEMANTIC BOUNDARY:
    - This test verifies transport integrity of model outputs, NOT physical correctness
      of those outputs.
    - Ava's visual inspection confirmed that Dan did not eat 100% of the food, proving
      the previous synthetic 100% test claim was unrepresentative of the real footage.
    - Neither Ava's visual observation nor the model's 71/29 split are treated as a
      physical ground-truth value oracle; rather, the recorded model numbers are used
      as a regression baseline to verify lossless pipeline transport.
    """
    date = "20260908"
    gha_producer_output = tmp_path / "gha_runner_tmp_output"
    gha_producer_output.mkdir()

    # Sep-8 Recorded Production Model Outputs (from morning_report.ipynb Job 101948457341)
    tapo_summary = {
        "date": date,
        "camera": "TAPO",
        "start_time": "06:20:00",
        "end_time": "06:22:03",
        "start_kibble": 34,
        "end_kibble": 0,
        "dan_kibble": 24,
        "sanbo_kibble": 10,
        "dan_percent": 71,
        "sanbo_percent": 29,
        "dan_bowl_seconds": 141.0,
        "sanbo_bowl_seconds": 22.0,
        "dan_first_arrival": "06:20:00",
        "sanbo_first_arrival": "06:22:00",
        "meal_finished": True,
        "has_conflict": True,
        "conflict_frames": 7
    }
    (gha_producer_output / f"tapo_summary_{date}.json").write_text(json.dumps(tapo_summary, indent=2), encoding="utf-8")

    # Timeline feeding phases matching the Sep-8 production timeline
    tapo_timeline = {
        "date": date,
        "camera": "TAPO",
        "generated_at_utc": "2026-09-08T05:45:32+00:00",
        "feeding_phases": [
            {
                "start": "2026-09-08 06:20:00",
                "end": "2026-09-08 06:21:05",
                "cat": "Dan",
                "dan_bowl_seconds": 87.08,
                "sanbo_bowl_seconds": 0.0,
                "conflict_frames": 0,
                "has_conflict": False,
                "exclusion_eligible": True,
                "confidence": 0.95
            },
            {
                "start": "2026-09-08 06:21:05",
                "end": "2026-09-08 06:22:02",
                "cat": "Sanbo",
                "dan_bowl_seconds": 0.0,
                "sanbo_bowl_seconds": 11.5,
                "conflict_frames": 7,
                "has_conflict": True,
                "exclusion_eligible": False,
                "confidence": 0.60
            }
        ]
    }
    (gha_producer_output / f"tapo_timeline_{date}.json").write_text(json.dumps(tapo_timeline, indent=2), encoding="utf-8")

    # Source feeding clip: motion_20260908_062003_2m_30s.mp4
    create_synthetic_mp4(
        gha_producer_output / f"motion_{date}_062003_2m_30s.mp4",
        num_frames=60,
        fps=15.0
    )

    # Step 1: GHA Producer Job Validation
    validation_res = validate_tapo_evidence_bundle(gha_producer_output, date)
    assert validation_res["feeding_detected"] is True
    assert validation_res["has_summary"] is True
    assert validation_res["has_timeline"] is True
    assert validation_res["source_clips_count"] == 1

    # Step 2: GHA Artifact Transfer (actions/upload-artifact -> actions/download-artifact)
    gha_consumer_tapo_dir = tmp_path / "gha_runner_tmp_tapo_evidence"
    shutil.copytree(gha_producer_output, gha_consumer_tapo_dir)

    # Step 3: Logitech VLM Shadow evidence for Sep-8
    gha_consumer_logi_dir = tmp_path / "gha_runner_tmp_logitech_evidence"
    gha_consumer_logi_dir.mkdir()
    logi_summary = {
        "session_start_time": "06:20:10",
        "session_end_time": "06:21:30",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "visibility": "normal",
        "total_duration": "1m 20s"
    }
    (gha_consumer_logi_dir / "summary.json").write_text(json.dumps(logi_summary, indent=2), encoding="utf-8")
    create_synthetic_mp4(
        gha_consumer_logi_dir / f"motion_{date}_062010_1m_20s.mp4",
        num_frames=60,
        fps=15.0
    )

    # Step 4: GHA Unified Delivery Job execution
    delivery_out_dir = tmp_path / f"unified_delivery_{date}"
    delivery_out_dir.mkdir()

    success = deliver_unified_breakfast(
        target_date=date,
        tapo_dir=gha_consumer_tapo_dir,
        logitech_dir=gha_consumer_logi_dir,
        out_dir=delivery_out_dir,
        skip_telegram=True,
        force=True
    )
    assert success is True

    # Step 5: Verify Structured Output
    report_file = delivery_out_dir / f"unified_report_{date}.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    msg = report["telegram_text"]
    # Dan feeder outcome
    assert "Dan feeder" in msg
    assert "Finished ✅" in msg
    assert "~34 kibble consumed" in msg
    # TAPO model attribution with bar & conflict warning
    assert "TAPO model attribution — contested" in msg
    assert "Dan" in msg
    assert "71%" in msg
    assert "(~24)" in msg
    assert "Sanbo" in msg
    assert "29%" in msg
    assert "(~10)" in msg
    assert "7 conflict frames" in msg
    assert "Sanbo feeder: Sanbo feeding observed" in msg
    # Physical reconciliation
    assert "Theft: Not confirmed" in msg
    # Neither camera degraded
    assert "No source footage" not in msg
    assert "artifact incomplete" not in msg
    assert "evidence unavailable" not in msg

    # Combined video rendered and valid
    video_file = delivery_out_dir / f"{date}_combined_breakfast.mp4"
    assert video_file.exists()
    assert video_file.stat().st_size > 0

