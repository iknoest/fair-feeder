import pytest
import numpy as np
import sys
import subprocess
import cv2
from pathlib import Path

# Add scripts dir to path to import hybrid_spike_offline
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from hybrid_spike_offline import (
    is_probably_ir_frame,
    brightness_score,
    sharpness_score,
    select_representative_frames,
    validate_vlm_result_schema,
    make_contact_sheet
)

def test_is_probably_ir_frame():
    gray_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert is_probably_ir_frame(gray_frame) is True
    
    color_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    color_frame[:, :, 0] = 255
    color_frame[:, :, 1] = 0
    color_frame[:, :, 2] = 0
    assert is_probably_ir_frame(color_frame) is False

def test_brightness_score():
    dark_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert brightness_score(dark_frame) == 0.0
    
    bright_frame = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert abs(brightness_score(bright_frame) - 255.0) < 1.0

def test_sharpness_score():
    flat_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert sharpness_score(flat_frame) == 0.0
    
    np.random.seed(42)
    noise_frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    assert sharpness_score(noise_frame) > 0.0

def test_select_representative_frames():
    frames = [{"id": i} for i in range(10)]
    selected = select_representative_frames(frames, 3)
    assert len(selected) == 3
    assert selected[0]['id'] == 0
    assert selected[1]['id'] == 3
    assert selected[2]['id'] == 6
    
    selected = select_representative_frames(frames, 15)
    assert len(selected) == 10

def test_validate_vlm_result_schema():
    valid = {
        "cat_identity": "Dan",
        "bowl_state": "empty",
        "confidence": 0.9,
        "needs_higher_model": False,
        "reasons": ["Visible tuxedo pattern"]
    }
    assert validate_vlm_result_schema(valid) is True
    
    invalid_missing = valid.copy()
    del invalid_missing["cat_identity"]
    assert validate_vlm_result_schema(invalid_missing) is False
    
    invalid_val = valid.copy()
    invalid_val["cat_identity"] = "Dog"
    assert validate_vlm_result_schema(invalid_val) is False
    
    invalid_type = valid.copy()
    invalid_type["confidence"] = "high"
    assert validate_vlm_result_schema(invalid_type) is False

def test_make_contact_sheet_grid(tmp_path):
    frames_data = []
    for _ in range(7):
        frames_data.append({"frame_data": np.zeros((100, 100, 3), dtype=np.uint8)})
    out_path = tmp_path / "contact.jpg"
    make_contact_sheet(frames_data, out_path, cols=4)
    assert out_path.exists()
    
    # 4 cols wide (4 * 320 = 1280), 2 rows high (2 * 180 = 360) for 7 frames
    img = cv2.imread(str(out_path))
    assert img is not None
    assert img.shape == (360, 1280, 3)

def test_cli_no_args_stops_cleanly():
    result = subprocess.run([sys.executable, str(scripts_dir / "hybrid_spike_offline.py")], capture_output=True, text=True)
    assert "[STOP] Checklist:" in result.stdout
    assert "Stop only when neither directory is provided" in result.stdout
    assert result.returncode == 0

def test_cli_partial_runs(tmp_path):
    tapo_dir = tmp_path / "tapo"
    tapo_dir.mkdir()
    logitech_dir = tmp_path / "logitech"
    logitech_dir.mkdir()
    out_dir = tmp_path / "out"
    
    res_tapo = subprocess.run([sys.executable, str(scripts_dir / "hybrid_spike_offline.py"), "--tapo-dir", str(tapo_dir), "--out-dir", str(out_dir)], capture_output=True, text=True)
    assert "--- Camera Summaries ---" in res_tapo.stdout
    assert "TAPO" in res_tapo.stdout
    
    res_logitech = subprocess.run([sys.executable, str(scripts_dir / "hybrid_spike_offline.py"), "--logitech-dir", str(logitech_dir), "--out-dir", str(out_dir)], capture_output=True, text=True)
    assert "--- Camera Summaries ---" in res_logitech.stdout
    assert "LOGITECH" in res_logitech.stdout

def test_deterministic_sorting(tmp_path):
    for name in ["c.mp4", "a.mp4", "b.mp4"]:
        (tmp_path / name).touch()
    
    files = sorted(list(tmp_path.glob("*.mp4")))
    assert [f.name for f in files] == ["a.mp4", "b.mp4", "c.mp4"]
