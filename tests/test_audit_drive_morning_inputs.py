import pytest
import sys
from pathlib import Path
import numpy as np

scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from audit_drive_morning_inputs import (
    in_feeding_window,
    extract_duration_from_filename,
    brightness_score,
    sharpness_score,
    is_probably_ir_frame
)

def test_in_feeding_window():
    # 06:18 to 06:30
    assert in_feeding_window("motion_20260704_061800.mp4", "20260704")[0] is True
    assert in_feeding_window("motion_20260704_062530_1m_10s.mp4", "20260704")[0] is True
    assert in_feeding_window("motion_20260704_063000.mp4", "20260704")[0] is True
    
    # Outside window
    assert in_feeding_window("motion_20260704_061759.mp4", "20260704")[0] is False
    assert in_feeding_window("motion_20260704_063100.mp4", "20260704")[0] is False
    
    # Date mismatch
    assert in_feeding_window("motion_20260705_062000.mp4", "20260704")[0] is False
    
    # Regex mismatch
    assert in_feeding_window("random_file.mp4", "20260704")[0] is False

def test_extract_duration():
    assert extract_duration_from_filename("motion_20260704_061800_1m_30s.mp4") == 90
    assert extract_duration_from_filename("motion_20260704_061800_45s.mp4") == 45
    assert extract_duration_from_filename("random.mp4") is None

def test_brightness_score():
    dark_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert brightness_score(dark_frame) == 0.0

def test_sharpness_score():
    flat_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert sharpness_score(flat_frame) == 0.0

def test_is_probably_ir_frame():
    gray_frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    assert is_probably_ir_frame(gray_frame) is True
