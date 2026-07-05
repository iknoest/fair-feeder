import pytest
import sys
import subprocess
import json
from pathlib import Path
import numpy as np
from unittest.mock import patch, mock_open

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
    prompt_file = generate_vlm_prompt(tmp_path, clip_name, date_str)
    
    assert prompt_file.exists()
    content = prompt_file.read_text()
    
    # Check that schema fields and placeholders are in the prompt
    assert clip_name in content
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
    result = subprocess.run([sys.executable, str(script_path), "--run-vlm"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --confirm-cost" in result.stdout

def test_run_vlm_without_provider_model_exits_nonzero():
    script_path = Path(__file__).parent.parent / "scripts" / "logitech_vlm_shadow.py"
    result = subprocess.run([sys.executable, str(script_path), "--run-vlm", "--confirm-cost"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --vlm-provider and --vlm-model" in result.stdout

def test_schema_validation_accepts_valid():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
    }
    # Should not raise
    validate_vlm_schema(valid_data)

def test_schema_validation_rejects_invalid_enum():
    invalid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "cat_identity": "UnknownCat", # Invalid
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
    }
    with pytest.raises(ValueError, match="Invalid cat_identity"):
        validate_vlm_schema(invalid_data)

def test_schema_validation_rejects_missing_field():
    invalid_data = {
        "camera": "LOGITECH",
        "clip_name": "motion_test.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
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
                        "content": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "test", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'
                    }
                }]
            }
            
    mock_post.return_value = MockResponse()
    
    # Mock reading image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake_image_data")
    
    result = call_openai_vlm("prompt", str(img_path), "gpt-4o", "fake_key")
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

def test_schema_rejects_wrong_camera():
    valid_data = {
        "camera": "TAPO", # Invalid
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
    }
    with pytest.raises(ValueError, match="Invalid camera: TAPO"):
        validate_vlm_schema(valid_data)

def test_schema_rejects_wrong_expected_date():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
    }
    with pytest.raises(ValueError, match="Invalid date: expected 20260705, got 20260704"):
        validate_vlm_schema(valid_data, expected_date="20260705")

def test_schema_rejects_wrong_expected_clip_name():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_wrong.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": ["Visible eating"],
        "needs_higher_model": False
    }
    with pytest.raises(ValueError, match="Invalid clip_name: expected motion_test.mp4, got motion_wrong.mp4"):
        validate_vlm_schema(valid_data, expected_clip_name="motion_test.mp4")

def test_schema_rejects_confidence_out_of_range():
    valid_data = {
        "camera": "LOGITECH",
        "date": "20260704",
        "clip_name": "motion_test.mp4",
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 1.5, # Out of range
        "reasons": ["Visible eating"],
        "needs_higher_model": False
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
        "cat_identity": "Sanbo",
        "eating_evidence": "yes",
        "bowl_state": "low",
        "confidence": 0.9,
        "reasons": "Visible eating", # String instead of list
        "needs_higher_model": False
    }
    with pytest.raises(ValueError, match="reasons must be a list of strings"):
        validate_vlm_schema(valid_data)

def test_cli_rejects_anthropic_provider():
    script_path = Path(__file__).parent.parent / "scripts" / "logitech_vlm_shadow.py"
    result = subprocess.run([sys.executable, str(script_path), "--run-vlm", "--confirm-cost", "--vlm-provider", "anthropic", "--vlm-model", "claude-3-haiku-20240307"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid choice: 'anthropic'" in result.stderr

def test_missing_gemini_key_error_mentions_both(monkeypatch, capsys):
    monkeypatch.delenv("FAIR_FEEDER_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import logitech_vlm_shadow
    with patch("sys.argv", ["logitech_vlm_shadow.py", "--run-vlm", "--confirm-cost", "--vlm-provider", "gemini", "--vlm-model", "test-model"]):
        with pytest.raises(SystemExit):
            logitech_vlm_shadow.main()
            
    captured = capsys.readouterr()
    assert "Set FAIR_FEEDER_GEMINI_API_KEY or GEMINI_API_KEY." in captured.out
    assert "AIza" not in captured.out

@patch("requests.post")
def test_gemini_key_lookup_prefers_fair_feeder(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fallback")
    
    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "dummy.mp4", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

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
    monkeypatch.delenv("FAIR_FEEDER_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fallback")
    
    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "dummy.mp4", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

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
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    
    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})
            
    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "dummy.mp4", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

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
    result_json_path = tmp_path / "logitech_vlm_result_dummy.json"
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
    assert "[SHADOW] Logitech VLM" in tg_text
    assert "Production report unchanged" in tg_text

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_transient_503_twice_creates_controlled_failure(mock_sleep, mock_post, monkeypatch, tmp_path):
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
                        
    failed_json_path = tmp_path / "logitech_vlm_result_dummy.failed.json"
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
                        
    failed_json_path = tmp_path / "logitech_vlm_result_dummy.failed.json"
    data = json.loads(failed_json_path.read_text())
    assert data["attempts_made"] == 1
    assert call_count[0] == 1

@patch("requests.post")
def test_low_confidence_formatting(mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    
    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "dummy.mp4", "cat_identity": "unsure", "eating_evidence": "unsure", "bowl_state": "unsure", "confidence": 0.6, "reasons": [], "needs_higher_model": true}'}]}}]}

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
    assert "0.6 (Needs review)" in md_text
    assert "true (Needs higher model)" in md_text
    assert "- Eating evidence: Uncertain" in md_text

    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "Eating: Uncertain" in tg_text

@patch("requests.post")
@patch("time.sleep", return_value=None)
def test_max_api_calls_cap_and_cleanup(mock_sleep, mock_post, monkeypatch, tmp_path):
    monkeypatch.setenv("FAIR_FEEDER_GEMINI_API_KEY", "AIza-fair-feeder")
    
    import requests
    class MockHTTPError(requests.exceptions.HTTPError):
        def __init__(self, status_code):
            self.response = type('obj', (object,), {'status_code': status_code})
            
    class MockResponseSuccess:
        def raise_for_status(self): pass
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": '{"camera": "LOGITECH", "date": "20260704", "clip_name": "motion_20260704_061800_clip1.mp4", "cat_identity": "Sanbo", "eating_evidence": "yes", "bowl_state": "low", "confidence": 0.9, "reasons": [], "needs_higher_model": false}'}]}}]}

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
    assert "Clip: motion_20260704_061900_clip2.mp4" in md_text
    assert "SKIPPED: ApiCapReached" in md_text
    assert "Reason: API call cap reached" in md_text
    
    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "motion_20260704_061900_clip2.mp4 -> SKIPPED: API call cap reached" in tg_text

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
    assert "FAILED: ValueError" in md_text
    assert "AIza-secret" not in md_text
    assert "***REDACTED***" in md_text
    
    tg_text = (tmp_path / "logitech_vlm_shadow_telegram_preview.txt").read_text()
    assert "FAILED: ValueError" in tg_text
    assert "AIza-secret" not in tg_text
