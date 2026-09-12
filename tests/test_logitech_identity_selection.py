import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

"""
Tests for Logitech deterministic identity keyframe selection.

Ensures:
1. Empty feeder hardware and bowl rim reflections are rejected.
2. High-contrast body/flank coat patterns are prioritized over empty frames.
3. Keyframes maintain session-wide temporal separation (early, mid, late).
4. Replay validation on Sep-12 production footage rejects Frame 490 (empty bowl)
   and selects genuine cat body frames.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
from scripts.logitech_vlm_shadow import select_identity_keyframes


def test_empty_feeder_frames_rejected_synthetic(tmp_path):
    """
    Verifies that frames with bright empty bowls and dark upper backgrounds are rejected
    in favor of frames where the cat covers the upper body area with high coat variance.
    """
    clip_p = tmp_path / "synthetic_empty_vs_cat.mp4"
    writer = cv2.VideoWriter(str(clip_p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))

    # Frames 0-29: Empty feeder (dark upper background, bright reflective bowl)
    for _ in range(30):
        frame = np.full((180, 320, 3), 2, dtype=np.uint8)
        # Bright bowl rim
        frame[80:150, 100:220] = 60
        writer.write(frame)

    # Frames 30-59: Cat present (high variance in upper body ROI)
    for i in range(30):
        frame = np.full((180, 320, 3), 15, dtype=np.uint8)
        # Upper body coat pattern (e.g. tabby patches)
        frame[20:100, 40:280] = 40 + (i % 5) * 15
        cv2.circle(frame, (160, 60), 20, (180, 180, 180), -1)
        writer.write(frame)

    writer.release()

    keyframes = select_identity_keyframes([clip_p], max_keyframes=2)
    assert len(keyframes) >= 1
    # Ensure every selected keyframe is from the cat phase (frame index >= 30)
    for kf in keyframes:
        assert kf["frame_index"] >= 30
        assert kf["score"] > 5.0
        assert "frame_enhanced" in kf


def test_temporal_separation_enforced(tmp_path):
    """Verifies that keyframes are spread out over time rather than bunched together."""
    clip_p = tmp_path / "synthetic_long.mp4"
    writer = cv2.VideoWriter(str(clip_p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))

    # 300 frames = 30 seconds
    for i in range(300):
        frame = np.full((180, 320, 3), 10, dtype=np.uint8)
        frame[20:100, 40:280] = 30 + (i % 10) * 5
        cv2.circle(frame, (160, 60), 15, (120, 120, 120), -1)
        writer.write(frame)
    writer.release()

    keyframes = select_identity_keyframes([clip_p], max_keyframes=3, min_sep=5.0)
    assert len(keyframes) >= 2
    for i in range(len(keyframes) - 1):
        time_diff = abs(keyframes[i+1]["seconds_from_start"] - keyframes[i]["seconds_from_start"])
        assert time_diff >= 4.5


def test_sep12_production_footage_selection():
    """
    Replay test on real Sep-12 production footage.
    Verifies:
    1. Empty feeder frame 490 (which old algorithm picked) is strictly excluded.
    2. Selected keyframes have substantial body presence and coat variance > 50.
    3. Session spans diverse temporal windows (early, mid, late).
    """
    sep12_clip = Path("scratch/replay_evidence/20260912/motion_20260912_061951_2m_17s.mp4")
    if not sep12_clip.exists():
        pytest.skip("Sep-12 production evidence clip not found in scratch")

    keyframes = select_identity_keyframes([sep12_clip], max_keyframes=3, min_sep=10.0)
    assert len(keyframes) == 3

    selected_indices = [kf["frame_index"] for kf in keyframes]
    # Frame 490 (the empty bowl) MUST NOT be selected
    assert 490 not in selected_indices

    for kf in keyframes:
        assert kf["body_var"] >= 50.0
        assert kf["u_mean"] >= 5.0
        assert kf["frame_enhanced"].shape == (720, 1280, 3)

    # Verify temporal spacing >= 10s
    for i in range(len(keyframes) - 1):
        dt = keyframes[i+1]["seconds_from_start"] - keyframes[i]["seconds_from_start"]
        assert dt >= 10.0
