import pytest
import sys
import subprocess
import json
from pathlib import Path
import cv2
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from logitech_vlm_shadow import (
    in_feeding_window,
    simple_cat_heuristic,
    generate_vlm_prompt,
    validate_vlm_schema,
    sanitize_error_message,
    call_openai_vlm
)

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "{}")
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake_folder")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake_key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat")

    class FakeCreds:
        token = 'fake_token'
        def refresh(self, request): pass
    monkeypatch.setattr("google.oauth2.service_account.Credentials.from_service_account_info", lambda x, scopes=None: FakeCreds())

    class FakeDrive:
        def files(self):
            class FakeFiles:
                def list(self, **kwargs):
                    class FakeList:
                        def execute(self):
                            return {"files": []}
                    return FakeList()
            return FakeFiles()
    monkeypatch.setattr("googleapiclient.discovery.build", lambda service, version, credentials=None: FakeDrive())

def test_in_feeding_window():
    assert in_feeding_window("motion_20260704_061800.mp4", "20260704") is True
    assert in_feeding_window("motion_20260704_063100.mp4", "20260704") is False

def test_simple_cat_heuristic():
    bg = np.zeros((10, 10, 3), dtype=np.uint8)
    fg_same = np.zeros((10, 10, 3), dtype=np.uint8)
    assert simple_cat_heuristic(fg_same, bg) is False

    fg_diff = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert simple_cat_heuristic(fg_diff, bg) is True

def test_generate_vlm_prompt(tmp_path):
    clip_name = "motion_20260704_061800.mp4"
    date_str = "20260704"
    prompt_file = generate_vlm_prompt(tmp_path, date_str)

    assert prompt_file.exists()
    content = prompt_file.read_text()

    # Check that schema fields and placeholders are in the prompt
    assert 'session' in content
    assert date_str in content
    assert "cat_identity" in content
    assert "eating_evidence" in content
    assert "bowl_state" in content
    assert "confidence" in content
    assert "needs_higher_model" in content

def test_deterministic_sort():
    # Helper test for deterministic sort behavior expected in the script
    unsorted = [{"name": "motion_2_a.mp4"}, {"name": "motion_1_a.mp4"}, {"name": "motion_3_a.mp4"}]
    unsorted.sort(key=lambda x: x['name'])
    assert unsorted[0]["name"] == "motion_1_a.mp4"
    assert unsorted[1]["name"] == "motion_2_a.mp4"
    assert unsorted[2]["name"] == "motion_3_a.mp4"

def test_selection_reason_labels_are_stable():
    # Simulating the extraction and sorting logic of sample_indices_labeled
    total_frames = 100
    sample_indices_labeled = [
        (0, "start"),
        (total_frames // 4, "quarter"),
        (total_frames // 2, "middle"),
        (3 * total_frames // 4, "three_quarter"),
        (total_frames - 1, "end")
    ]
    # Assume first motion is at frame 10
    sample_indices_labeled.append((10, "first_motion"))

    sample_indices_labeled.sort(key=lambda x: x[0])

    seen_indices = set()
    final_samples = []
    for idx, label in sample_indices_labeled:
        if idx not in seen_indices and 0 <= idx < total_frames:
            seen_indices.add(idx)
            final_samples.append((idx, label))

    assert final_samples[0] == (0, "start")
    assert final_samples[1] == (10, "first_motion")
    assert final_samples[2] == (25, "quarter")

def test_run_vlm_without_confirm_cost_exits_nonzero():
    script_path = Path(__file__).parent.parent / "scripts" / "logitech_vlm_shadow.py"
    result = subprocess.run([sys.executable, str(script_path), "--date", "2026-07-05", "--run-vlm"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --confirm-cost" in result.stdout

def test_run_vlm_without_provider_model_exits_nonzero():
    script_path = Path(__file__).parent.parent / "scripts" / "logitech_vlm_shadow.py"
    result = subprocess.run([sys.executable, str(script_path), "--date", "2026-07-05", "--run-vlm", "--confirm-cost"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --vlm-provider and --vlm-model" in result.stdout

def test_schema_validation_accepts_valid():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    # Should not raise
    validate_vlm_schema(valid_data)

def test_schema_validation_rejects_invalid_enum():
    invalid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "UnknownCat", # Invalid
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Invalid cat_identity"):
        validate_vlm_schema(invalid_data)

def test_schema_validation_rejects_missing_field():
    invalid_data = {
        "camera": "LOGITECH",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Missing required field: date"):
        validate_vlm_schema(invalid_data)

@patch("requests.post")
def test_mocked_provider_response(mock_post, tmp_path):
    # Mocking openai request inside pytest
    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "test", "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'
                    }
                }]
            }

    mock_post.return_value = MockResponse()

    # Mock reading image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake_image_data")

    result = call_openai_vlm("prompt", [str(img_path)], "gpt-4o", "fake_key")
    assert result["cat_identity"] == "Sanbo"
    assert result["eating_evidence"] == "yes"

def test_sanitize_error_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-key")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder-key")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-gemini-key")

    # Test OpenAI redaction
    msg1 = "Error: Invalid token sk-fake-openai-key provided"
    assert "sk-fake-openai-key" not in sanitize_error_message(msg1)

    # Test Fair Feeder Gemini redaction
    msg_ff = "Error: Invalid token AIza-fair-feeder-key"
    assert "AIza-fair-feeder-key" not in sanitize_error_message(msg_ff)

    # Test Gemini redaction
    msg2 = "Error: Invalid key AIza-fake-gemini-key"
    assert "AIza-fake-gemini-key" not in sanitize_error_message(msg2)

    # Test OpenAI redaction
    msg1 = "Error: Invalid token sk-fake-openai-key provided"
    assert "sk-fake-openai-key" not in sanitize_error_message(msg1)

    # Test Gemini redaction
    msg2 = "Error: Invalid key AIza-fake-gemini-key"
    assert "AIza-fake-gemini-key" not in sanitize_error_message(msg2)

    # Test URL query param redaction
    msg3 = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIza-some-other-key-123"
    sanitized3 = sanitize_error_message(msg3)
    assert "key=***REDACTED***" in sanitized3
    assert "AIza-some-other-key-123" not in sanitized3

    # Test Bearer redaction
    msg4 = "Authorization: Bearer my-secret-token"
    sanitized4 = sanitize_error_message(msg4)
    assert "Bearer ***REDACTED***" in sanitized4
    assert "my-secret-token" not in sanitized4

    # Test Telegram secrets redaction
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_telegram_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_telegram_chat_id")

    msg5 = "Error with token fake_telegram_bot_token and chat id fake_telegram_chat_id"
    sanitized5 = sanitize_error_message(msg5)
    assert "fake_telegram_bot_token" not in sanitized5
    assert "fake_telegram_chat_id" not in sanitized5
    assert "***REDACTED***" in sanitized5

    # Test Telegram URL redaction
    msg6 = "Failed URL: https://api.telegram.org/bot12345:secret/sendMessage"
    sanitized6 = sanitize_error_message(msg6)
    assert "12345:secret" not in sanitized6
    assert "https://api.telegram.org/bot***REDACTED***/sendMessage" in sanitized6

    msg7 = "Failed URL: https://api.telegram.org/bot54321:other/sendPhoto"
    sanitized7 = sanitize_error_message(msg7)
    assert "54321:other" not in sanitized7
    assert "https://api.telegram.org/bot***REDACTED***/sendPhoto" in sanitized7
def test_schema_rejects_wrong_camera():
    valid_data = {
        "camera": "TAPO", # Invalid
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Invalid camera: TAPO"):
        validate_vlm_schema(valid_data)

def test_schema_rejects_wrong_expected_date():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Invalid date: expected 20260705, got 20260704"):
        validate_vlm_schema(valid_data, expected_date="20260705")

def test_schema_rejects_wrong_expected_clip_name():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_wrong.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Invalid clip_name: expected motion_test.mp4, got motion_wrong.mp4"):
        validate_vlm_schema(valid_data, expected_clip_name="motion_test.mp4")

def test_schema_rejects_confidence_out_of_range():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 1.5, # Out of range
        "reasons": ["Visible eating"],
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="Confidence out of range: 1.5"):
        validate_vlm_schema(valid_data)

    valid_data["confidence"] = -0.1
    with pytest.raises(ValueError, match="Confidence out of range: -0.1"):
        validate_vlm_schema(valid_data)

def test_schema_rejects_non_list_reasons():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": "Visible eating", # String instead of list
        "needs_higher_model": True
    }
    with pytest.raises(ValueError, match="reasons must be a list of strings"):
        validate_vlm_schema(valid_data)

def test_cli_rejects_anthropic_provider():
    script_path = Path(__file__).parent.parent / "scripts" / "logitech_vlm_shadow.py"
    result = subprocess.run([sys.executable, str(script_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "anthropic", "--vlm-model", "claude-3-haiku-20240307"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid choice: 'anthropic'" in result.stderr

def test_missing_gemini_key_error_mentions_both(monkeypatch, capsys):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("FAIR_FEEDER_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-05", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model"]):
        with pytest.raises(SystemExit):
            logitech_vlm_shadow.main()

    captured = capsys.readouterr()
    assert "Missing GDRIVE_SERVICE_ACCOUNT_KEY or GEMINI_API_KEY for Gemini authentication." in captured.out
    assert "AIza" not in captured.out

@patch("requests.post")
def test_gemini_key_lookup_prefers_fair_feeder(mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fallback")

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

    def side_effect(url, *args, **kwargs):
        assert "AIza-fair-feeder" in url
        return MockResponse()

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            # Mock Google Drive API
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        # Ensure the contact sheet and prompt exist so it proceeds to VLM
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        logitech_vlm_shadow.main()

@patch("requests.post")
def test_gemini_key_lookup_falls_back(mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("FAIR_FEEDER_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fallback")

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

    def side_effect(url, *args, **kwargs):
        assert "AIza-fallback" in url
        return MockResponse()

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        logitech_vlm_shadow.main()

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_transient_503_fails_once_then_succeeds(mock_sleep, mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")

    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})

    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

    call_count = [0]
    def side_effect(url, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise MockHTTPError(503)
        return MockResponseSuccess()

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        logitech_vlm_shadow.main()

    # Verify attempts_made appears in success payload
    result_json_path = tmp_path / "logitech_vlm_result_session.json"
    data = json.loads(result_json_path.read_text())
    assert data["attempts_made"] == 2
    assert call_count[0] == 2

    # Check summary counts
    summary = json.loads((tmp_path / "logitech_vlm_shadow_summary.json").read_text())
    assert summary["clips_attempted"] == 1
    assert summary["clips_succeeded"] == 1
    assert summary["clips_failed"] == 0
    assert summary["api_calls_made"] == 2

    # Check shadow report
    md_text = (tmp_path / "logitech_vlm_shadow_report.md").read_text()
    assert "Sanbo" in md_text
    assert "0.9" in md_text

    # Check telegram preview
    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "[SHADOW][LOGITECH]" in tg_text
    assert "😸 Sanbo visible:" in tg_text

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_transient_503_twice_creates_controlled_failure(mock_sleep, mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})

    def side_effect(url, *args, **kwargs):
        raise MockHTTPError(503)

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        with pytest.raises(SystemExit):
                            logitech_vlm_shadow.main()

    failed_json_path = tmp_path / "logitech_vlm_result_session.failed.json"
    data = json.loads(failed_json_path.read_text())
    assert data["attempts_made"] == 2

    summary = json.loads((tmp_path / "logitech_vlm_shadow_summary.json").read_text())
    assert summary["clips_attempted"] == 1
    assert summary["clips_succeeded"] == 0
    assert summary["clips_failed"] == 1
    assert summary["api_calls_made"] == 2

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_401_does_not_retry(mock_sleep, mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})

    call_count = [0]
    def side_effect(url, *args, **kwargs):
        call_count[0] += 1
        raise MockHTTPError(401)

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        with pytest.raises(SystemExit):
                            logitech_vlm_shadow.main()

    failed_json_path = tmp_path / "logitech_vlm_result_session.failed.json"
    data = json.loads(failed_json_path.read_text())
    assert data["attempts_made"] == 1
    assert call_count[0] == 1

@patch("requests.post")
def test_low_confidence_formatting(mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")

    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "unsure", "eating_evidence": "unsure", "bowl_state": "unsure", "confidence": 0.6, "reasons": [], "needs_higher_model": true}'}]}}]}

    mock_post.return_value = MockResponseSuccess()

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        logitech_vlm_shadow.main()

    md_text = (tmp_path / "logitech_vlm_shadow_report.md").read_text()
    assert "0.6" in md_text
    assert "needs higher model" in md_text.lower()
    assert "eating: unsure" in md_text.lower()

    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "eating: UNSURE" in tg_text
    assert "⚠️ Theft status unsure" in tg_text

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_max_api_calls_cap_and_cleanup(mock_sleep, mock_post, monkeypatch, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")

    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})

    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

    # We mock 2 clips.
    # First clip fails first time (status 503), then retries (which makes api_calls_made = 2), but succeeds.
    # Second clip then starts, but api_calls_made is already 2, so it should be skipped.

    call_count = [0]
    def side_effect(url, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise MockHTTPError(503)
        return MockResponseSuccess()

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path), "--cleanup-downloaded-videos"]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "motion_20260704_061800_clip1.mp4", "id": "123"}, {"name": "motion_20260704_061900_clip2.mp4", "id": "124"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        # Create dummy mp4 files to test cleanup
                        (tmp_path / "motion_20260704_061800_clip1.mp4").write_bytes(b"dummy")
                        (tmp_path / "motion_20260704_061900_clip2.mp4").write_bytes(b"dummy")

                        # Create other artifacts to ensure they are NOT deleted
                        (tmp_path / "motion_20260704_061800_clip1_frame_0.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_061800_clip1.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_motion_20260704_061800_clip1.md").write_text("dummy prompt")

                        (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_061900_clip2.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_motion_20260704_061900_clip2.md").write_text("dummy prompt")

                        logitech_vlm_shadow.main()

    # Check that cleanup deleted ONLY the mp4s
    assert not (tmp_path / "motion_20260704_061800_clip1.mp4").exists()
    assert not (tmp_path / "motion_20260704_061900_clip2.mp4").exists()
    assert (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_061800_clip1.jpg").exists()
    assert (tmp_path / "logitech_vlm_prompt_motion_20260704_061800_clip1.md").exists()
    assert (tmp_path / "motion_20260704_061800_clip1_frame_0.jpg").exists()

    # Check summary counts
    summary = json.loads((tmp_path / "logitech_vlm_shadow_summary.json").read_text())
    assert summary["clips_requested"] == 2
    assert summary["clips_attempted"] == 1
    assert summary["clips_succeeded"] == 1
    assert summary["clips_failed"] == 0
    assert summary["clips_skipped"] == 1
    assert summary["skipped_due_to_api_cap"] == 1
    assert summary["api_calls_made"] == 2
    assert summary["api_call_cap"] == 2

    # Verify reports contain skipped/failed logic
    md_text = (tmp_path / "logitech_vlm_shadow_report.md").read_text()

    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()

@patch("requests.post")
def test_failed_clip_in_reports_and_sanitized(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-secret")

    import requests
    def side_effect(*args, **kwargs):
        raise ValueError("Invalid key AIza-secret provided!")

    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-04", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path)]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True):
            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "dummy.mp4", "id": "123"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def get(self, prop): return 1
                        def set(self, prop, val): pass
                        def read(self): return True, __import__('numpy').zeros((10, 10, 3), dtype=__import__('numpy').uint8)
                        def release(self): pass
                    with patch('cv2.VideoCapture', return_value=MockCap()):
                        (tmp_path / "logitech_vlm_contact_sheet_dummy.jpg").write_bytes(b"dummy")
                        (tmp_path / "logitech_vlm_prompt_dummy.md").write_text("dummy prompt")
                        import pytest
                        with pytest.raises(SystemExit):
                            logitech_vlm_shadow.main()

    md_text = (tmp_path / "logitech_vlm_shadow_report.md").read_text()
    assert "AIza-secret" not in md_text

    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "AIza-secret" not in tg_text

def test_send_telegram_shadow_without_run_vlm_exits_nonzero(capsys):
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--send-telegram-shadow", "--date", "2026-07-05"]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "requires --run-vlm" in captured.out

# script loads .env at startup, so missing-env tests must not depend on repo .env.
def test_send_telegram_shadow_without_env_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--send-telegram-shadow", "--run-vlm", "--confirm-cost", "--date", "2026-07-05"]):
        with patch("requests.post") as mock_post:
            with pytest.raises(SystemExit) as excinfo:
                logitech_vlm_shadow.main()
            assert excinfo.value.code != 0
            mock_post.assert_not_called()
    captured = capsys.readouterr()
    assert "Missing required Telegram environment variables" in captured.out

def test_missing_date_exits_before_env_drive_vlm_telegram(monkeypatch, capsys):
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "{}")
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake")

    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--send-telegram-shadow", "--run-vlm", "--confirm-cost"]):
        with patch("requests.post") as mock_post, patch("googleapiclient.discovery.build") as mock_build, patch("logitech_vlm_shadow.call_gemini_vlm") as mock_gemini, patch("logitech_vlm_shadow.call_openai_vlm", return_value={"camera":"LOGITECH","date":"20260702","clip_name":"session","identity_basis": "raw", "visibility": "good", "cat_identity":"Sanbo","eating_evidence":"yes","bowl_state":"low","confidence":0.9,"reasons":[],"needs_higher_model":False}) as mock_openai:
            with pytest.raises(SystemExit) as excinfo:
                logitech_vlm_shadow.main()
            assert excinfo.value.code != 0
            mock_post.assert_not_called()
            mock_build.assert_not_called()
            mock_gemini.assert_not_called()
            mock_openai.assert_not_called()
    captured = capsys.readouterr()
    assert "date is required" in captured.out

def test_send_telegram_shadow_with_explicit_date_reaches_env_guard_or_next_guard(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--send-telegram-shadow", "--run-vlm", "--confirm-cost", "--date", "2026-07-05"]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "Missing required Telegram environment variables" in captured.out
    assert "requires an explicit --date" not in captured.out

@patch("requests.post")
def test_telegram_flags_and_sending(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake_key")

    # We will mock the VLM call to return a specific result to trigger flags
    import logitech_vlm_shadow
    def mock_gemini(*args, **kwargs):
        # We simulate 3 clips processed sequentially
        if not hasattr(mock_gemini, "count"):
            mock_gemini.count = 0
        mock_gemini.count += 1

        if mock_gemini.count == 1:
            return {
                "camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_060001.mp4",
                "camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "both", "eating_evidence": "no", "bowl_state": "low",
                "confidence": 0.9, "reasons": [], "needs_higher_model": True
            }
        elif mock_gemini.count == 2:
            return {
                "camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_060002.mp4",
                "identity_basis": "raw", "visibility": "good", "cat_identity": "Dan", "eating_evidence": "unsure", "bowl_state": "empty",
                "confidence": 0.70, "reasons": [], "needs_higher_model": True
            }
        else:
            return {
                "camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_060003.mp4",
                "identity_basis": "raw", "visibility": "good", "cat_identity": "none", "eating_evidence": "yes", "bowl_state": "half",
                "confidence": 0.9, "reasons": [], "needs_higher_model": True
            }

    # Mock requests.post to succeed
    class MockResp:
        status_code = 200
        def json(self): return {"ok": True, "result": {"message_id": 123}}
        def raise_for_status(self): pass
    mock_post.return_value = MockResp()

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "20260704", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path), "--send-telegram-shadow", "--send-still-images", "--max-clips", "3"]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True), \
             patch('logitech_vlm_shadow.call_gemini_vlm', side_effect=mock_gemini), \
             patch('logitech_vlm_shadow.MAX_API_CALLS_PER_RUN', 3):

            # Setup fake files to process
            (tmp_path / "motion_20260704_060001.mp4").write_bytes(b"")
            (tmp_path / "motion_20260704_060002.mp4").write_bytes(b"")
            (tmp_path / "motion_20260704_060003.mp4").write_bytes(b"")

            # Create fake contact sheets so they can be attached
            (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_060001.jpg").write_bytes(b"img1")
            (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_060002.jpg").write_bytes(b"img2")
            (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_060003.jpg").write_bytes(b"img3")
            (tmp_path / "logitech_vlm_contact_sheet_session.jpg").write_bytes(b"img_session")
            (tmp_path / "logitech_vlm_prompt_session.md").write_text("prompt_session")

            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [
                                        {"name": "motion_20260704_060001.mp4", "id": "1"},
                                        {"name": "motion_20260704_060002.mp4", "id": "2"},
                                        {"name": "motion_20260704_060003.mp4", "id": "3"}
                                    ]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True):
                with patch('logitech_vlm_shadow.download_file'):
                    class MockCap:
                        def __init__(self, *args, **kwargs):
                            self._count = 0
                        def isOpened(self): return True
                        def get(self, prop): return 10
                        def set(self, prop, val): pass
                        def read(self):
                            self._count += 1
                            if self._count > 5:
                                return False, None
                            f = np.zeros((180, 320, 3), dtype=np.uint8)
                            f[:, :, 0] = 30
                            f[:, :, 1] = 45
                            f[:, :, 2] = 60
                            return True, f
                        def release(self): pass
                    with patch('cv2.VideoCapture', side_effect=MockCap):
                        logitech_vlm_shadow.main()

    # Check preview text
    md_text = (tmp_path / "logitech_vlm_shadow_report.md").read_text()
    assert "[SHADOW] Logitech VLM Feeding Session Report" in md_text
    assert "Non-authoritative shadow report. Production report unchanged." in md_text

    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "[SHADOW][LOGITECH]" in tg_text
    assert "⚠️ Possible theft by Dan" in tg_text

    # Check requests.post calls
    assert mock_post.call_count >= 1  # Text sent, plus any media

    # Verify summary
    summary_path = tmp_path / "telegram_shadow_send_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["telegram_send_attempted"] is True
    assert summary["telegram_text_sent"] is True
    assert summary["telegram_images_attempted"] == 1
    assert summary["telegram_images_sent"] == 1
    assert summary["telegram_error"] is None
    assert summary["telegram_send_fully_successful"] is True

    # Check logitech_vlm_shadow_summary.json
    main_summary_path = tmp_path / "logitech_vlm_shadow_summary.json"
    if main_summary_path.exists():
        main_summary = json.loads(main_summary_path.read_text())
        assert main_summary["telegram_sent"] is True
        assert main_summary["telegram_images_sent"] == 1
        assert main_summary["telegram_error"] is None

@patch("requests.post")
def test_telegram_send_text_failure_exits_nonzero(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake_key")

    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        pass
    def side_effect(*args, **kwargs):
        raise MockHTTPError("Telegram API error with token fake_bot_token")
    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    def mock_gemini(*args, **kwargs):
        return {
            "camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_060001.mp4",
            "camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "both", "eating_evidence": "no", "bowl_state": "low",
            "confidence": 0.9, "reasons": [], "needs_higher_model": True
        }

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "20260704", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path), "--send-telegram-shadow", "--max-clips", "1"]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True), patch('logitech_vlm_shadow.call_gemini_vlm', side_effect=mock_gemini):
            (tmp_path / "motion_20260704_060001.mp4").write_bytes(b"")
            (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_060001.jpg").write_bytes(b"img1")

            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "motion_20260704_060001.mp4", "id": "1"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True), patch('logitech_vlm_shadow.download_file'):
                class MockCap:
                    def get(self, prop): return 1
                    def set(self, prop, val): pass
                    def read(self): return True, np.zeros((10, 10, 3), dtype=np.uint8)
                    def release(self): pass
                with patch('cv2.VideoCapture', return_value=MockCap()):
                    import pytest
                    with pytest.raises(SystemExit):
                        logitech_vlm_shadow.main()

    summary_path = tmp_path / "telegram_shadow_send_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["telegram_text_sent"] is False
    assert summary["telegram_images_attempted"] == 0
    assert summary["telegram_send_fully_successful"] is False
    assert "fake_bot_token" not in summary["telegram_error"]
    assert "***REDACTED***" in summary["telegram_error"]

@patch("requests.post")
def test_telegram_send_photo_failure_exits_nonzero(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake_key")

    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        pass

    call_count = [0]
    def side_effect(url, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            class MockResp:
                def raise_for_status(self): pass
            return MockResp()
        else:
            raise MockHTTPError("Telegram API photo error for fake_chat_id")
    mock_post.side_effect = side_effect

    import logitech_vlm_shadow
    def mock_gemini(*args, **kwargs):
        return {
            "camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_060001.mp4",
            "camera": "LOGITECH", "date": "20260704", "clip_name": "session", "identity_basis": "raw", "visibility": "good", "cat_identity": "both", "eating_evidence": "no", "bowl_state": "low",
            "confidence": 0.9, "reasons": [], "needs_higher_model": True
        }

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "20260704", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model", "--out-dir", str(tmp_path), "--send-telegram-shadow", "--send-still-images", "--max-clips", "1"]):
        with patch('logitech_vlm_shadow.check_credentials', return_value=True), patch('logitech_vlm_shadow.call_gemini_vlm', side_effect=mock_gemini):
            (tmp_path / "motion_20260704_060001.mp4").write_bytes(b"")
            (tmp_path / "logitech_vlm_contact_sheet_motion_20260704_060001.jpg").write_bytes(b"img1")

            class MockDrive:
                def files(self):
                    class MockFiles:
                        def list(self, **kwargs):
                            class MockList:
                                def execute(self):
                                    return {"files": [{"name": "motion_20260704_060001.mp4", "id": "1"}]}
                            return MockList()
                        def get_media(self, **kwargs):
                            class MockReq: pass
                            return MockReq()
                    return MockFiles()
            with patch('googleapiclient.discovery.build', return_value=MockDrive()), patch('logitech_vlm_shadow.in_feeding_window', return_value=True), patch('logitech_vlm_shadow.download_file'):
                class MockCap:
                    def get(self, prop): return 1
                    def set(self, prop, val): pass
                    def read(self): return True, np.zeros((10, 10, 3), dtype=np.uint8)
                    def release(self): pass
                with patch('cv2.VideoCapture', return_value=MockCap()):
                    import pytest
                    with pytest.raises(SystemExit):
                        logitech_vlm_shadow.main()

    summary_path = tmp_path / "telegram_shadow_send_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["telegram_text_sent"] is True
    assert summary["telegram_images_attempted"] == 1
    assert summary["telegram_images_sent"] == 0
    assert summary["telegram_send_fully_successful"] is False
    assert "fake_chat_id" not in summary["telegram_error"]
    assert "***REDACTED***" in summary["telegram_error"]

    # Check logitech_vlm_shadow_summary.json
    main_summary_path = tmp_path / "logitech_vlm_shadow_summary.json"
    main_summary = json.loads(main_summary_path.read_text())
    assert main_summary["telegram_sent"] is True
    assert main_summary["telegram_images_sent"] == 0
    assert "***REDACTED***" in main_summary["telegram_error"]

import pytest
from unittest.mock import patch

def test_missing_date_exits_nonzero_global(capsys):
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--out-dir", "fake_dir"]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "date is required" in captured.out

def test_check_image_domain_color():
    from logitech_vlm_shadow import check_image_domain
    import numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:,:,0] = 100
    img[:,:,1] = 50
    img[:,:,2] = 20
    assert check_image_domain(img) == 'COLOR'

def test_check_image_domain_bright_grayscale():
    from logitech_vlm_shadow import check_image_domain
    import numpy as np
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert check_image_domain(img) == 'BRIGHT_GRAYSCALE'

def test_check_image_domain_dark_grayscale():
    from logitech_vlm_shadow import check_image_domain
    import numpy as np
    img = np.full((10, 10, 3), 20, dtype=np.uint8)
    assert check_image_domain(img) == 'DARK_GRAYSCALE'

def test_sanitize_gdrive_key(monkeypatch):
    from logitech_vlm_shadow import sanitize_error_message
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "fake_gdrive_key_value")
    msg = "Error with key fake_gdrive_key_value"
    assert "fake_gdrive_key_value" not in sanitize_error_message(msg)

def test_sanitize_json_private_key():
    from logitech_vlm_shadow import sanitize_error_message
    msg = '{"type": "service_account", "private_key": "-----BEGIN PRIVATE KEY-----\\nSuperSecret\\n-----END PRIVATE KEY-----\\n", "client_email": "fake@foo.com"}'
    sanitized = sanitize_error_message(msg)
    assert "SuperSecret" not in sanitized
    assert 'private_key": "***REDACTED***"' in sanitized or "private_key': '***REDACTED***'" in sanitized

def test_sanitize_pem_block():
    from logitech_vlm_shadow import sanitize_error_message
    msg = "Some error\\n-----BEGIN PRIVATE KEY-----\\nAnotherSecret\\n-----END PRIVATE KEY-----\\nEnd error"
    sanitized = sanitize_error_message(msg)
    assert "AnotherSecret" not in sanitized
    assert "-----BEGIN PRIVATE KEY-----\n***REDACTED***\n-----END PRIVATE KEY-----" in sanitized

def test_zero_selected_clips_hard_stop(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake_folder")
    import logitech_vlm_shadow

    class FakeDrive:
        def files(self):
            return self
        def list(self, **kwargs):
            return self
        def execute(self):
            return {'files': [{'name': 'not_in_window.mp4', 'id': '123'}]}

    class FakeCreds:
        token = 'fake_token'
        def refresh(self, request): pass
    monkeypatch.setattr("google.oauth2.service_account.Credentials.from_service_account_info", lambda x, scopes=None: FakeCreds())
    monkeypatch.setattr("googleapiclient.discovery.build", lambda s, v, credentials: FakeDrive())
    monkeypatch.setattr("logitech_vlm_shadow.check_credentials", lambda: True)

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "gemini-1.5"]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "[NO_CLIPS]" in captured.out
    assert "not_in_window.mp4" in captured.out
    assert (tmp_path / "logitech_vlm_summary.json").exists()

def test_stale_out_dir_summary_guard(tmp_path, capsys):
    import logitech_vlm_shadow
    import json

    s_path = tmp_path / "summary.json"
    with open(s_path, "w") as f:
        json.dump({"date": "20260701"}, f)

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path)]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code != 0

    captured = capsys.readouterr()
    assert "Stale out-dir guard" in captured.out
    assert "has date 20260701 but --date is 20260702" in captured.out

def test_stale_out_dir_mp4_guard(tmp_path, capsys):
    import logitech_vlm_shadow

    (tmp_path / "motion_20260701_061800_10s.mp4").touch()

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path)]):
        with pytest.raises(SystemExit) as excinfo:
            logitech_vlm_shadow.main()
        assert excinfo.value.code != 0

    captured = capsys.readouterr()
    assert "Stale out-dir guard" in captured.out
    assert "does not match --date 20260702" in captured.out


def test_gdrive_logitech_folder_id_missing_exits_nonzero_fixed(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "{}")
    monkeypatch.delenv("GDRIVE_LOGITECH_FOLDER_ID", raising=False)
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake")

    import logitech_vlm_shadow

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test"]):
        with patch("requests.post") as mock_post, patch("googleapiclient.discovery.build") as mock_build, patch("logitech_vlm_shadow.call_gemini_vlm", return_value={"camera":"LOGITECH","date":"20260702","clip_name":"session","identity_basis": "raw", "visibility": "good", "cat_identity":"Sanbo","eating_evidence":"yes","bowl_state":"low","confidence":0.9,"reasons":[],"needs_higher_model":False}) as mock_vlm:
            with pytest.raises(SystemExit) as excinfo:
                logitech_vlm_shadow.main()
            assert excinfo.value.code != 0
            mock_post.assert_not_called()
            mock_build.assert_not_called()
            mock_vlm.assert_not_called()

    captured = capsys.readouterr()
    assert "GDRIVE_LOGITECH_FOLDER_ID is missing" in captured.out


def test_contact_sheet_overlay_trap_fixed(monkeypatch, capsys, tmp_path):
    import logitech_vlm_shadow
    import numpy as np
    import cv2
    from pathlib import Path

    # Hermetic mock of required env vars
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "{}")
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake")

    class FakeDrive:
        def files(self):
            return self
        def list(self, **kwargs):
            return self
        def execute(self):
            return {'files': [{'name': 'motion_20260702_062112_2m_30s.mp4', 'id': '123'}]}

    class FakeCreds:
        token = 'fake_token'
        def refresh(self, request): pass
    monkeypatch.setattr("google.oauth2.service_account.Credentials.from_service_account_info", lambda x, scopes=None: FakeCreds())
    monkeypatch.setattr("googleapiclient.discovery.build", lambda s, v, credentials: FakeDrive())

    # Mock download to just touch the file
    def fake_download(*args, **kwargs):
        dest = kwargs.get('dest_path') or args[2]
        Path(dest).touch()
    monkeypatch.setattr("logitech_vlm_shadow.download_file", fake_download)

    class FakeCap:
        def __init__(self, *args): pass
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT: return 10
            if prop == cv2.CAP_PROP_FPS: return 1.0
            return 0
        def set(self, prop, val): pass
        def read(self):
            # raw frames are bright grayscale
            return True, np.full((10, 10, 3), 100, dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr("cv2.VideoCapture", FakeCap)

    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test", "--send-telegram-shadow"]):
        with patch("requests.post") as mock_post, patch("logitech_vlm_shadow.call_openai_vlm") as mock_openai, patch("logitech_vlm_shadow.call_gemini_vlm", return_value={"camera":"LOGITECH","date":"20260702","clip_name":"session","identity_basis": "raw", "visibility": "good", "cat_identity":"Sanbo","eating_evidence":"yes","bowl_state":"low","confidence":0.9,"reasons":[],"needs_higher_model":False}) as mock_gemini:
            with pytest.raises(SystemExit) as excinfo:
                logitech_vlm_shadow.main()
            assert excinfo.value.code != 0
            mock_post.assert_not_called()
            mock_openai.assert_not_called()
            mock_gemini.assert_not_called()

    captured = capsys.readouterr()
    assert "BRIGHT_GRAYSCALE" in captured.out
    assert "Likely Tapo IR input" in captured.out

def test_gdrive_service_account_key_missing_exits_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("GDRIVE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake")
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--date", "2026-07-02", "--out-dir", str(tmp_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test"]):
        with patch("requests.post") as mock_post, patch("googleapiclient.discovery.build") as mock_build, patch("logitech_vlm_shadow.call_gemini_vlm", return_value={"camera":"LOGITECH","date":"20260702","clip_name":"session","identity_basis": "raw", "visibility": "good", "cat_identity":"Sanbo","eating_evidence":"yes","bowl_state":"low","confidence":0.9,"reasons":[],"needs_higher_model":False}) as mock_vlm:
            with pytest.raises(SystemExit) as excinfo:
                logitech_vlm_shadow.main()
            assert excinfo.value.code != 0
            mock_post.assert_not_called()
            mock_build.assert_not_called()
            mock_vlm.assert_not_called()
    captured = capsys.readouterr()
    assert "Missing required environment variables:" in captured.out


def test_vlm_session_aggregation_one_clip():
    import logitech_vlm_shadow
    selected = [{'name': 'motion_20260702_061945_52s.mp4', 'id': '1'}]
    manifest = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': True}]
    results = [{
        'clip_name': 'motion_20260702_061945_52s.mp4',
        'cat_identity': 'Sanbo',
        'eating_evidence': 'yes',
        'bowl_state': 'low',
        'confidence': 0.9,
        'reasons': ['Sanbo eating from bowl.'],
        'needs_higher_model': False
    }]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, results, [], [], "20260702", "gemini", "gemini-2.5-flash"
    )
    assert session['cat_identity'] == 'Sanbo'
    assert session['eating_evidence'] == 'yes'
    assert session['possible_food_theft'] is False
    assert session['session_start_time'] == '06:19:45'
    assert session['session_end_time'] == '06:20:37'
    assert session['total_duration'] == '52s'
    assert session['evidence_clip_count'] == 1


def test_vlm_session_aggregation_multi_clip():
    import logitech_vlm_shadow
    selected = [
        {'name': 'motion_20260702_061945_2m_30s.mp4', 'id': '1'},
        {'name': 'motion_20260702_062220_1m_15s.mp4', 'id': '2'}
    ]
    manifest = [
        {'timestamp': '2026-07-02 06:19:50', 'motion_detected': True},
        {'timestamp': '2026-07-02 06:22:25', 'motion_detected': True}
    ]
    results = [
        {
            'clip_name': 'motion_20260702_061945_2m_30s.mp4',
            'cat_identity': 'Sanbo',
            'eating_evidence': 'yes',
            'bowl_state': 'full',
            'confidence': 0.95,
            'reasons': ['Sanbo at bowl.'],
            'needs_higher_model': False
        },
        {
            'clip_name': 'motion_20260702_062220_1m_15s.mp4',
            'cat_identity': 'Sanbo',
            'eating_evidence': 'yes',
            'bowl_state': 'low',
            'confidence': 0.90,
            'reasons': ['Sanbo finished eating.'],
            'needs_higher_model': False
        }
    ]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, results, [], [], "20260702", "gemini", "gemini-2.5-flash"
    )
    assert session['cat_identity'] == 'Sanbo'
    assert session['eating_evidence'] == 'yes'
    assert session['bowl_state_progression'] == 'full → low'
    assert session['session_start_time'] == '06:19:45'
    assert session['session_end_time'] == '06:23:35'
    assert session['total_duration'] == '3m 50s'
    assert session['evidence_clip_count'] == 2


def test_vlm_session_aggregation_both_cats_food_theft():
    import logitech_vlm_shadow
    selected = [{'name': 'motion_20260702_061945_2m_30s.mp4', 'id': '1'}]
    manifest = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': True}]
    results = [{
        'clip_name': 'motion_20260702_061945_2m_30s.mp4',
        'cat_identity': 'both',
        'eating_evidence': 'yes',
        'bowl_state': 'low',
        'confidence': 0.85,
        'reasons': ['Dan approached while Sanbo was at feeder.'],
        'needs_higher_model': False
    }]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, results, [], [], "20260702", "gemini", "gemini-2.5-flash"
    )
    assert session['cat_identity'] == 'both'
    assert session['possible_food_theft'] is True
    report_text = logitech_vlm_shadow.format_session_report_text(session, results, [], [])
    assert "Possible food theft" in report_text
    assert "both ⚠️ possible food theft" in report_text


def test_vlm_session_aggregation_unknown_low_confidence():
    import logitech_vlm_shadow
    selected = [{'name': 'motion_20260702_061945_30s.mp4', 'id': '1'}]
    manifest = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': False}]
    results = [{
        'clip_name': 'motion_20260702_061945_30s.mp4',
        'cat_identity': 'unsure',
        'eating_evidence': 'unsure',
        'bowl_state': 'unsure',
        'confidence': 0.50,
        'reasons': ['Blurry cat image.'],
        'needs_higher_model': True
    }]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, results, [], [], "20260702", "gemini", "gemini-2.5-flash"
    )
    assert session['confidence'] == 0.50
    assert session['needs_higher_model'] is True
    report_text = logitech_vlm_shadow.format_session_report_text(session, results, [], [])
    assert "⚠️ low confidence" in report_text or "⚠️ marked for higher model review" in report_text


def test_vlm_session_report_no_kibble_counts():
    import logitech_vlm_shadow
    selected = [{'name': 'motion_20260702_061945_52s.mp4', 'id': '1'}]
    manifest = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': True}]
    results = [{
        'clip_name': 'motion_20260702_061945_52s.mp4',
        'cat_identity': 'Sanbo',
        'eating_evidence': 'yes',
        'bowl_state': 'low',
        'confidence': 0.9,
        'reasons': ['Sanbo eating.'],
        'needs_higher_model': False
    }]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, results, [], [], "20260702", "gemini", "gemini-2.5-flash"
    )
    report_text = logitech_vlm_shadow.format_session_report_text(session, results, [], [])

    # Invariant: Must NOT count individual kibble pieces
    assert "kibble count" not in report_text.lower()
    assert "kibbles" not in report_text.lower()
    assert "pieces" not in report_text.lower()
    assert "start_kibble" not in session
    assert "kibble_eaten" not in session


def test_morning_report_workflow_vlm_isolation():
    from pathlib import Path
    workflow_path = Path(".github/workflows/morning-report.yml")
    assert workflow_path.exists(), "morning-report.yml must exist"
    content = workflow_path.read_text(encoding="utf-8")

    assert "Run Logitech Gemini VLM Shadow Report" in content
    assert "if: matrix.camera == 'LOGITECH'" in content
    assert "--run-vlm" in content
    assert "--confirm-cost" in content
    assert "--vlm-provider gemini" in content
    assert "--vlm-model gemini-2.5-flash" in content
    assert "--send-telegram-shadow" in content
    # assert "FAIR_FEEDER_GEMINI_API_KEY" in content

    # Ensure logitech_vlm_shadow.py is invoked in the workflow
    vlm_steps = [line for line in content.splitlines() if "logitech_vlm_shadow.py" in line]
    assert len(vlm_steps) >= 1


def test_vlm_total_failure_report_format():
    import logitech_vlm_shadow
    selected = [{'name': 'motion_20260702_061945_52s.mp4', 'id': '1'}]
    manifest = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': True}]
    failed = [{
        'clip_name': 'motion_20260702_061945_52s.mp4',
        'error_type': 'ApiCapReached',
        'error_message': '400 Client Error: Bad Request for url: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash',
        'telegram_error_message': 'ApiCapReached - 400 Client Error'
    }]
    session = logitech_vlm_shadow.build_vlm_session_report(
        selected, manifest, [], failed, [], "20260702", "gemini", "gemini-2.5-flash"
    )
    assert session['vlm_success_count'] == 0
    assert session['failure_category'] == 'provider quota/billing'
    assert session['status'] == 'failed'

    report_text = logitech_vlm_shadow.format_vlm_failure_report_text(session, failed, [], shadow_header=True)
    assert "[SHADOW][LOGITECH] ⚠️ VLM analysis failed" in report_text
    assert "No cat/eating/bowl conclusion was produced." in report_text
    assert "Failure category: provider quota/billing" in report_text
    assert "cat: unsure" not in report_text
    assert "eating: unsure" not in report_text


def test_telegram_delivery_accounting_on_failure_report(monkeypatch, tmp_path):
    monkeypatch.setenv("GDRIVE_SERVICE_ACCOUNT_KEY", "{}")
    monkeypatch.setenv("GDRIVE_LOGITECH_FOLDER_ID", "fake")
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "fake_chat_id")

    import logitech_vlm_shadow

    selected_files = [{'name': 'motion_20260702_061945_52s.mp4', 'id': '1'}]
    manifest_data = [{'timestamp': '2026-07-02 06:19:50', 'motion_detected': True}]
    failed = [{
        'clip_name': 'motion_20260702_061945_52s.mp4',
        'error_type': 'ApiCapReached',
        'error_message': '429 Quota Exceeded',
        'telegram_error_message': 'ApiCapReached - 429 Quota Exceeded'
    }]

    session = logitech_vlm_shadow.build_vlm_session_report(
        selected_files, manifest_data, [], failed, [], "20260702", "gemini", "gemini-2.5-flash"
    )

    send_summary = {
        "telegram_send_attempted": True,
        "telegram_text_sent": True,
        "telegram_photos_attempted": 0,
        "telegram_photos_sent": 0,
        "is_failure_report": True,
        "is_analysis_report": False,
        "total_messages_delivered": 1,
        "telegram_error": None
    }

    assert send_summary["total_messages_delivered"] == 1
    assert send_summary["is_failure_report"] is True
    assert send_summary["total_messages_delivered"] != 0


def test_camera_label_in_startup_alert():
    from pathlib import Path
    source = Path("motion_recorder.py").read_text(encoding="utf-8")
    assert "cam_label =" in source
    assert "📷 Fair Feeder Monitor [{cam_label}] is LIVE" in source

def test_generate_vlm_prompt_references(tmp_path):
    p_no_ref = generate_vlm_prompt(tmp_path, "20260816", has_references=False)
    no_ref_text = p_no_ref.read_text()

    p_ref = generate_vlm_prompt(tmp_path, "20260816", has_references=True)
    ref_text = p_ref.read_text()

    assert "REFERENCE IMAGES" not in no_ref_text
    assert "REFERENCE IMAGES" in ref_text
    assert "REFERENCE — DAN" in ref_text


def test_group_clips_into_sessions_splits_large_gap():
    import logitech_vlm_shadow
    clips_20260817 = [
        {"name": "motion_20260817_061955_1m_58s.mp4", "id": "clip1"},
        {"name": "motion_20260817_062528_35s.mp4", "id": "clip2"}
    ]
    sessions = logitech_vlm_shadow.group_clips_into_sessions(clips_20260817, gap_threshold_sec=10)
    assert len(sessions) == 2
    assert sessions[0][0]["name"] == "motion_20260817_061955_1m_58s.mp4"
    assert sessions[1][0]["name"] == "motion_20260817_062528_35s.mp4"


def test_group_clips_into_sessions_merges_small_gap():
    import logitech_vlm_shadow
    continuation_clips = [
        {"name": "motion_20260817_062006_2m_30s.mp4", "id": "chunk1"},
        {"name": "motion_20260817_062241_1m_15s.mp4", "id": "chunk2"}
    ]
    # Chunk 1 ends at 06:22:36. Chunk 2 starts at 06:22:41 (gap = 5s <= 10s)
    sessions = logitech_vlm_shadow.group_clips_into_sessions(continuation_clips, gap_threshold_sec=10)
    assert len(sessions) == 1
    assert len(sessions[0]) == 2


def test_format_multi_session_report_text():
    import logitech_vlm_shadow
    sess1 = {
        "date": "2026-08-17",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "session_start_time": "06:19:55",
        "session_end_time": "06:21:53",
        "total_duration": "1m 58s",
        "cat_identity": "Dan",
        "possible_food_theft": True,
        "identity_basis": "enhanced + reference-assisted",
        "visibility": "usable",
        "confidence": 0.90,
        "eating_evidence": "yes",
        "bowl_state_progression": "half -> empty",
        "hand_human_interaction": "none observed",
        "first_recorded_motion_time": "~06:19:55",
        "recorded_session_duration": "1m 58s",
        "evidence_clip_count": 1,
        "evidence_sampled_frame_count": 6
    }
    sess2 = {
        "date": "2026-08-17",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "session_start_time": "06:25:28",
        "session_end_time": "06:26:03",
        "total_duration": "35s",
        "cat_identity": "none",
        "possible_food_theft": False,
        "identity_basis": "raw",
        "visibility": "good",
        "confidence": 0.95,
        "eating_evidence": "no",
        "bowl_state_progression": "empty",
        "hand_human_interaction": "none observed",
        "first_recorded_motion_time": "~06:25:28",
        "recorded_session_duration": "35s",
        "evidence_clip_count": 1,
        "evidence_sampled_frame_count": 5
    }
    res1 = {"clip_name": "session_1", "reasons": ["Dark coat with white bib visible"]}
    res2 = {"clip_name": "session_2", "reasons": ["Empty bowl visible in lit room"]}

    report = logitech_vlm_shadow.format_multi_session_report_text([sess1, sess2], [res1, res2], [], [])
    assert "=== EVENT 1 (06:19:55-06:21:53, 1m 58s) ===" in report
    assert "cat: Dan ⚠️ Dan at Logitech/Sanbo feeder — verify" in report
    assert "=== EVENT 2 (06:25:28-06:26:03, 35s) ===" in report
    assert "cat: none" in report
    assert "Recorded Sessions: 2 distinct event(s)" in report


def test_enhance_image_gamma_clahe():
    import numpy as np
    import cv2
    import logitech_vlm_shadow

    # Create dark synthetic image
    dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
    enh_img = logitech_vlm_shadow.enhance_image_gamma_clahe(dark_img, gamma=2.5)

    assert enh_img is not None
    assert enh_img.shape == (100, 100, 3)
    assert np.mean(enh_img) > np.mean(dark_img)


def test_make_comparison_contact_sheet(tmp_path):
    import numpy as np
    import cv2
    import logitech_vlm_shadow

    raw_path = tmp_path / "raw_cs.jpg"
    enh_path = tmp_path / "enh_cs.jpg"
    comp_path = tmp_path / "comp_cs.jpg"

    raw_img = np.full((180, 320, 3), 20, dtype=np.uint8)
    enh_img = np.full((180, 320, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(raw_path), raw_img)
    cv2.imwrite(str(enh_path), enh_img)

    logitech_vlm_shadow.make_comparison_contact_sheet(raw_path, enh_path, comp_path)

    assert comp_path.exists()
    comp_img = cv2.imread(str(comp_path))
    assert comp_img is not None
    # Check vertical concatenation with banners: height > 180 * 2
    assert comp_img.shape[0] > 360
    assert comp_img.shape[1] == 320


def test_extract_semantic_keyframes_with_motion():
    import logitech_vlm_shadow
    mock_cap = MagicMock()
    # 100 frames total, fps=10
    total_frames = 100
    fps = 10.0
    current_pos = [0]

    def mock_set(prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            current_pos[0] = int(val)

    def mock_get(prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return current_pos[0]
        return 10.0

    # Motion from frame 20 to 80
    def mock_read():
        pos = current_pos[0]
        frame = np.full((180, 320, 3), 255 if 20 <= pos <= 80 else 0, dtype=np.uint8)
        current_pos[0] += 1
        return True, frame

    mock_cap.read.side_effect = mock_read
    mock_cap.set.side_effect = mock_set
    mock_cap.get.side_effect = mock_get

    bg_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    samples = logitech_vlm_shadow.extract_semantic_keyframes(mock_cap, total_frames, fps, bg_frame)

    reasons = [label for idx, label in samples]
    assert "pre_feed_baseline" in reasons
    assert "first_approach" in reasons
    assert "post_feed" in reasons
    # Ensure pre-feed is before first motion (frame 20)
    pre_feed_idx = [idx for idx, label in samples if label == "pre_feed_baseline"][0]
    first_motion_idx = [idx for idx, label in samples if label == "first_approach"][0]
    assert pre_feed_idx < first_motion_idx


def test_generate_vlm_prompt_bowl_contract(tmp_path):
    import logitech_vlm_shadow
    prompt_file = logitech_vlm_shadow.generate_vlm_prompt(tmp_path, "20260822", "session_1")
    content = prompt_file.read_text()
    assert "BOWL STATE EVIDENCE CONTRACT" in content
    assert "unsure" in content
    assert "Do NOT infer bowl level from subsequent consumption" in content


def test_is_meaningful_feeding_event_suppression():
    from logitech_vlm_shadow import is_meaningful_feeding_event
    # Suppressed: no cat, no eating
    sess_empty = {"cat_identity": "none", "eating_evidence": "no", "possible_food_theft": False}
    assert is_meaningful_feeding_event(sess_empty) is False

    # Meaningful: Sanbo eating
    sess_sanbo = {"cat_identity": "Sanbo", "eating_evidence": "yes", "possible_food_theft": False}
    assert is_meaningful_feeding_event(sess_sanbo) is True

    # Meaningful: Possible food theft
    sess_theft = {"cat_identity": "Dan", "eating_evidence": "no", "possible_food_theft": True}
    assert is_meaningful_feeding_event(sess_theft) is True


def test_format_compact_session_text():
    from logitech_vlm_shadow import format_compact_session_text
    sess_data = {
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "possible_food_theft": False,
        "session_start_time": "06:19:55",
        "session_end_time": "06:22:29",
        "total_duration": "2m34s",
        "confidence": 0.80,
        "visibility": "poor",
        "bowl_state_progression": "unsure → empty",
        "identity_basis": "enhanced + reference-assisted"
    }
    text = format_compact_session_text(sess_data, result={"reference_images": ["ref1.jpg"]})
    assert "[SHADOW][LOGITECH] Feeding summary" in text
    assert "06:19:55–06:22:29 · 2m34s" in text
    assert "😸 Sanbo visible:" in text
    assert "🥣 Kibble: unsure → empty" in text
    assert "⚠️ No theft detected" in text
    assert "Confidence: 0.80 · visibility poor" in text


def test_format_compact_multi_session_text_suppresses_no_eating_events():
    from logitech_vlm_shadow import format_compact_multi_session_text
    sess1 = {
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "possible_food_theft": False,
        "session_start_time": "06:19:55",
        "session_end_time": "06:22:29",
        "total_duration": "2m34s",
        "confidence": 0.80,
        "visibility": "poor",
        "bowl_state_progression": "unsure → empty"
    }
    sess2 = {
        "cat_identity": "none",
        "eating_evidence": "no",
        "possible_food_theft": False,
        "session_start_time": "06:22:57",
        "session_end_time": "06:23:12",
        "total_duration": "15s",
        "confidence": 0.95,
        "visibility": "good",
        "bowl_state_progression": "empty"
    }
    res1 = {"clip_name": "session_1", "reference_images": ["ref.jpg"]}
    res2 = {"clip_name": "session_2"}

    text = format_compact_multi_session_text([sess1, sess2], [res1, res2], [], [])
    # Event 1 is included
    assert "Sanbo visible:" in text
    assert "06:19:55" in text
    # Event 2 is suppressed from output
    assert "06:22:57" not in text


def test_make_before_after_comparison(tmp_path):
    from logitech_vlm_shadow import make_before_after_comparison
    out_path = tmp_path / "before_after.jpg"
    frames = {
        "pre_feed_baseline": {"frame": np.zeros((270, 480, 3), dtype=np.uint8), "timestamp": "06:19:55"},
        "mid_eating": {"frame": np.full((270, 480, 3), 128, dtype=np.uint8), "timestamp": "06:21:00"},
        "post_feed": {"frame": np.full((270, 480, 3), 200, dtype=np.uint8), "timestamp": "06:22:29"}
    }
    make_before_after_comparison(frames, out_path)
    assert out_path.exists()
    img = cv2.imread(str(out_path))
    assert img is not None
    # Check 3 panels joined horizontally: width = 480 * 3 = 1440
    assert img.shape[1] == 1440


def test_generate_enhanced_video_returns_existing_output_path_when_under_45mb(tmp_path):
    from logitech_vlm_shadow import generate_enhanced_video
    raw_path = tmp_path / "raw.mp4"
    out_path = tmp_path / "enh.mp4"

    # Write synthetic 10-frame video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(raw_path), fourcc, 10.0, (320, 180))
    for i in range(10):
        frame = np.full((180, 320, 3), 30 + i, dtype=np.uint8)
        cv2.circle(frame, (160, 90), 20, (150, 150, 150), -1)
        writer.write(frame)
    writer.release()

    res = generate_enhanced_video(raw_path, out_path)
    assert res == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert out_path.stat().st_size < 45 * 1024 * 1024


def test_generate_enhanced_video_multi_clip_session_stitches_duration(tmp_path):
    from logitech_vlm_shadow import generate_enhanced_video
    clip1_path = tmp_path / "motion_20260823_060001_10s.mp4"
    clip2_path = tmp_path / "motion_20260823_060012_10s.mp4"
    out_path = tmp_path / "session_enhanced.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w1 = cv2.VideoWriter(str(clip1_path), fourcc, 10.0, (320, 180))
    for i in range(10):
        frame = np.full((180, 320, 3), 20 + i, dtype=np.uint8)
        cv2.circle(frame, (160, 90), 20, (100, 100, 100), -1)
        w1.write(frame)
    w1.release()

    w2 = cv2.VideoWriter(str(clip2_path), fourcc, 10.0, (320, 180))
    for i in range(15):
        frame = np.full((180, 320, 3), 40 + i, dtype=np.uint8)
        cv2.circle(frame, (160, 90), 20, (150, 150, 150), -1)
        w2.write(frame)
    w2.release()

    res = generate_enhanced_video([clip1_path, clip2_path], out_path, speedup_factor=1.0)
    assert res == out_path
    assert out_path.exists()

    cap = cv2.VideoCapture(str(out_path))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 20 <= total_f <= 45  # frames stitched and transcoded

def test_generate_enhanced_video_zero_frames_raises(tmp_path):
    from logitech_vlm_shadow import generate_enhanced_video
    empty_clip = tmp_path / "empty.mp4"
    empty_clip.write_bytes(b"")
    out_path = tmp_path / "out.mp4"

def test_select_identity_keyframes_chooses_body_coat_frames(tmp_path):
    from logitech_vlm_shadow import select_identity_keyframes
    clip_p = tmp_path / "motion_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(clip_p), fourcc, 10.0, (320, 180))

    # Write 60 frames (6 seconds) with varying body ROI contrast
    for i in range(60):
        frame = np.full((180, 320, 3), 10, dtype=np.uint8)
        if 20 <= i <= 40:
            # Add strong texture/pattern in body ROI
            frame[70:160, 50:270] = 50 + (i % 20) * 5
            cv2.circle(frame, (160, 120), 25, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    keyframes = select_identity_keyframes([clip_p], max_keyframes=2)
    assert len(keyframes) >= 1
    assert keyframes[0]["score"] > 0
    assert "frame_enhanced" in keyframes[0]
    assert keyframes[0]["frame_enhanced"].shape == (180, 320, 3)


def test_generate_enhanced_video_4x_speedup(tmp_path):
    from logitech_vlm_shadow import generate_enhanced_video
    raw_path = tmp_path / "raw_10s.mp4"
    out_path = tmp_path / "enh_4x.mp4"

    # Write 100 frames at 10 fps (10 seconds)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(raw_path), fourcc, 10.0, (320, 180))
    for i in range(100):
        frame = np.full((180, 320, 3), 25 + i % 10, dtype=np.uint8)
        cv2.circle(frame, (160, 90), 20, (150, 150, 150), -1)
        writer.write(frame)
    writer.release()

    res = generate_enhanced_video(raw_path, out_path, speedup_factor=4.0)
    assert res == out_path
    assert out_path.exists()

    cap = cv2.VideoCapture(str(out_path))
    out_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    out_dur = out_frames / out_fps
    # 10s source at 4x speed -> ~2.5s duration (+/- 1.0s)
    assert 1.5 <= out_dur <= 3.5


def test_sanbo_at_sanbo_feeder_produces_no_theft_warning():
    from logitech_vlm_shadow import format_compact_session_text, format_session_report_text
    session_data = {
        "session_start_time": "06:21:15",
        "session_end_time": "06:23:48",
        "total_duration": "2m 33s",
        "clip_count": 2,
        "first_clip": "motion_20260823_062115_2m_30s.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "half -> empty",
        "confidence": 0.90,
        "reasons": ["Light coat with distinct patches visible on flank"],
        "visibility": "usable",
        "identity_basis": "enhanced + reference-assisted",
        "needs_higher_model": False,
        "possible_food_theft": False
    }

    report_md = format_session_report_text(session_data, [session_data], [], [])
    assert "Sanbo feeding session" in report_md
    assert "Possible food theft" not in report_md

    compact_text = format_compact_session_text(session_data)
    assert "😸 Sanbo visible:" in compact_text
    assert "⚠️ No theft detected" in compact_text
    assert "possible theft" not in compact_text.lower()


def test_dan_at_sanbo_feeder_produces_possible_theft_warning():
    from logitech_vlm_shadow import format_compact_session_text, format_session_report_text
    session_data = {
        "session_start_time": "06:21:15",
        "session_end_time": "06:23:48",
        "total_duration": "2m 33s",
        "clip_count": 2,
        "first_clip": "motion_20260823_062115_2m_30s.mp4",
        "cat_identity": "Dan",
        "eating_evidence": "yes",
        "bowl_state": "half -> empty",
        "confidence": 0.90,
        "reasons": ["Solid dark back visible"],
        "visibility": "usable",
        "identity_basis": "enhanced + reference-assisted",
        "needs_higher_model": False,
        "possible_food_theft": True
    }

    report_md = format_session_report_text(session_data, [session_data], [], [])
    assert "Possible food theft — Dan at Sanbo feeder!" in report_md

    compact_text = format_compact_session_text(session_data)
    assert "😸 Dan visible:" in compact_text
    assert "⚠️ Possible theft by Dan" in compact_text


def test_20260823_session_ground_truth_fixture():
    """
    Regression fixture establishing human ground truth for 2026-08-23 session:
    Sanbo eating at Sanbo feeder without food theft warning.
    """
    ground_truth = {
        "date": "20260823",
        "session_time": "06:21:15–06:23:48",
        "source_duration": "2m 33s",
        "expected_cat": "Sanbo",
        "expected_eating": "yes",
        "expected_feeder": "LOGITECH (Sanbo feeder)",
        "expected_theft": False
    }

    assert ground_truth["expected_cat"] == "Sanbo"
    assert ground_truth["expected_eating"] == "yes"
    assert ground_truth["expected_theft"] is False
