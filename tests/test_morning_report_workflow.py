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


def test_analysis_job_uses_single_source_of_truth_for_target_date():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'TARGET_DATE: ${{ needs.prepare-window.outputs.target_date }}' in text
    assert 'DATE_OVERRIDE: ${{ github.event.inputs.date_override || needs.prepare-window.outputs.target_date }}' in text
    assert 'TARGET_DATE=${TARGET_DATE}' in text
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
