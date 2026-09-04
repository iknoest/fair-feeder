import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_upload_ledger_for_tests(tmp_path, monkeypatch):
    """Ensures each test gets an isolated upload ledger in tmp_path."""
    test_ledger = tmp_path / "test_upload_ledger.json"
    monkeypatch.setenv("UPLOAD_LEDGER_PATH", str(test_ledger))
    yield test_ledger
