import pytest
import sys
from pathlib import Path
import numpy as np

scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from audit_morning_report_pipeline import (
    in_feeding_window,
    bbox_iou,
    detect_image_type,
    simple_cat_heuristic
)

def test_in_feeding_window():
    assert in_feeding_window("motion_20260704_061800.mp4", "20260704") is True
    assert in_feeding_window("motion_20260704_063100.mp4", "20260704") is False

def test_bbox_iou():
    box_a = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    box_b = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
    # intersection: 50x50 = 2500
    # area a = 10000, area b = 10000, union = 17500
    assert abs(bbox_iou(box_a, box_b) - (2500 / 17500)) < 1e-4

def test_detect_image_type():
    gray_frame = np.full((10, 10, 3), 128, dtype=np.uint8)
    assert detect_image_type(gray_frame) == "ir"
    
    color_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    color_frame[:, :, 0] = 255
    assert detect_image_type(color_frame) == "color"

def test_simple_cat_heuristic():
    bg = np.zeros((10, 10, 3), dtype=np.uint8)
    fg_same = np.zeros((10, 10, 3), dtype=np.uint8)
    assert simple_cat_heuristic(fg_same, bg) is False
    
    fg_diff = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert simple_cat_heuristic(fg_diff, bg) is True
