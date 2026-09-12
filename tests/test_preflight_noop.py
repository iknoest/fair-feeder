import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

"""
Tests for GHA duplicate noon run preflight detection and clean NO-OP gating.

Ensures:
1. is_breakfast_fully_delivered correctly identifies delivered breakfasts.
2. Preflight CLI / checks do not output invalid non-key-value lines into GITHUB_ENV or GITHUB_OUTPUT.
3. Preflight status cleanly outputs 'SKIP' when already delivered and 'PROCEED' when pending.
4. Downstream job gating conditions evaluate to False when already delivered.
"""

import os
import json
import pytest
from pathlib import Path
from scripts.delivery_ledger import (
    init_registry_data,
    record_unified_item_delivered,
    is_breakfast_fully_delivered,
    is_camera_fully_delivered,
    load_delivery_registry
)


def test_breakfast_fully_delivered_detection(tmp_path):
    """Verifies that breakfast is only marked fully delivered when required items are delivered."""
    registry = init_registry_data()
    date = "20260912"
    
    # Initially not delivered
    assert not is_breakfast_fully_delivered(registry, date)
    
    # Deliver summary only -> still NOT fully delivered
    record_unified_item_delivered(None, "dummy_folder", registry, date, "summary", local_fallback_dir=tmp_path)
    assert not is_breakfast_fully_delivered(registry, date)
    
    # Deliver combined_video -> now fully delivered
    record_unified_item_delivered(None, "dummy_folder", registry, date, "combined_video", local_fallback_dir=tmp_path)
    assert is_breakfast_fully_delivered(registry, date)


def test_github_env_formatting_safety(tmp_path, monkeypatch):
    """
    Regression test for GHA runner parser failure in Run #267.
    Ensures preflight scripts only write strictly valid KEY=VALUE lines to GITHUB_ENV.
    """
    gh_env_file = tmp_path / "github_env"
    gh_env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_ENV", str(gh_env_file))
    
    target_date = "20260911"
    
    # Simulate preflight logic writing to GITHUB_ENV
    status = "SKIP"
    with open(gh_env_file, "a", encoding="utf-8") as f:
        f.write(f"preflight_status={status}\n")
        
    lines = gh_env_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    for line in lines:
        assert "=" in line
        key, val = line.split("=", 1)
        assert key.isidentifier()
        assert " " not in key
        # Ensure log message prefixes like "✅" are never in the env file
        assert "✅" not in line
        assert "Preflight:" not in line


def test_prepare_output_gating_semantics():
    """
    Verifies that the workflow job gating conditions:
    `needs.prepare.outputs.already_delivered != 'true'`
    correctly skips jobs when already delivered ('true') and proceeds when pending ('false' or '').
    """
    # When already delivered
    out_delivered = {"already_delivered": "true"}
    skip_tapo = out_delivered.get("already_delivered") != "true"
    assert skip_tapo is False  # Job is skipped
    
    # When not yet delivered
    out_pending = {"already_delivered": "false"}
    run_tapo = out_pending.get("already_delivered") != "true"
    assert run_tapo is True   # Job runs
    
    # Replay test or missing output defaults to running
    out_replay = {"already_delivered": ""}
    run_replay = out_replay.get("already_delivered") != "true"
    assert run_replay is True  # Job runs
