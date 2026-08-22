from pathlib import Path


WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "morning-report.yml"


def test_prepare_job_exists_and_exports_target_date():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "prepare-window:" in text
    assert "outputs:" in text
    assert "target_date" in text
    assert "needs: prepare-window" in text
    assert "needs.prepare-window.outputs.target_date" in text
    assert 'echo "target_date=${TARGET_DATE}" >> "$GITHUB_OUTPUT"' in text


def test_prepare_job_timeout_is_large_enough_for_wait_window():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "prepare-window:" in text
    assert "timeout-minutes: 360" in text
    assert "Wait until feeding window is complete" in text

def test_prepare_exports_heartbeat_and_target_date():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'gha_job_started_at_utc' in text
    assert 'gha_schedule_cron' in text
    assert 'gha_scheduled_at_utc' in text
    assert 'gha_schedule_delay_min' in text
    assert 'amsterdam_local_time' in text
    assert 'target_date' in text

    # ensure the embedded Python reads the env SCHEDULE_CRON (exported by workflow env)
    assert "os.getenv('SCHEDULE_CRON'" in text
    # and does not rely on a non-exported shell-only GHA_SCHEDULE_CRON
    assert "os.getenv('GHA_SCHEDULE_CRON'" not in text

def test_analysis_job_uses_single_source_of_truth_for_target_date_and_heartbeat():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'TARGET_DATE: ${{ needs.prepare-window.outputs.target_date }}' in text
    # ensure env wiring to morning-report from prepare-window exists
    assert 'GHA_JOB_STARTED_AT_UTC: ${{ needs.prepare-window.outputs.gha_job_started_at_utc }}' in text
    assert 'GHA_SCHEDULE_CRON: ${{ needs.prepare-window.outputs.gha_schedule_cron }}' in text
    assert 'GHA_SCHEDULE_DELAY_MIN: ${{ needs.prepare-window.outputs.gha_schedule_delay_min }}' in text
    assert '--date "$TARGET_DATE"' in text

def test_replay_step_remains_manual_only():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python scripts/send_tapo_replay_test.py" in text
    assert "if: github.event.inputs.date_override == 'REPLAY_TEST' && matrix.camera == 'TAPO'" in text
    assert "if: matrix.camera == 'TAPO' && github.event.inputs.date_override != 'REPLAY_TEST'" in text


def test_dependency_steps_have_bounded_timeouts():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "timeout-minutes: 20" in text
    assert "Install system dependencies" in text
    assert "Install dependencies" in text


def test_analysis_job_has_fresh_timeout_budget():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "timeout-minutes: 180" in text
    assert "timeout-minutes: 360" in text
    assert "needs: prepare-window" in text
