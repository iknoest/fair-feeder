"""Tests for flagging.py — auto-flag suspicious YOLO detections."""
import pytest
from flagging import flag_detections, FlaggedFrame


def _make_det(cls_name, conf, box=None):
    if box is None:
        box = [100, 100, 200, 200]
    return {'class': cls_name, 'confidence': conf, 'box': box}


def _make_frame(detections, frame_idx=0, jpeg=b'fake-jpeg'):
    return {
        'detections': detections,
        'timestamp': float(frame_idx),
        'jpeg': jpeg,
        'height': 720,
        'width': 1280,
    }


# ---------------------------------------------------------------------------
# Low-confidence
# ---------------------------------------------------------------------------
class TestLowConfidence:
    def test_flags_detection_below_threshold(self):
        frames = [_make_frame([_make_det('Sanbo', 0.31)], frame_idx=0)]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 1
        assert any('low-conf' in t for t in result[0].tags)

    def test_ignores_detection_above_threshold(self):
        frames = [_make_frame([_make_det('Dan', 0.85)], frame_idx=0)]
        result = flag_detections(frames, conf_threshold=0.40)
        # No low-conf tag should appear
        low_tags = [t for r in result for t in r.tags if 'low-conf' in t]
        assert len(low_tags) == 0

    def test_includes_confidence_in_tag(self):
        frames = [_make_frame([_make_det('Sanbo', 0.31)], frame_idx=0)]
        result = flag_detections(frames, conf_threshold=0.40)
        assert 'low-conf-sanbo-31' in result[0].tags

    def test_multiple_low_conf_same_frame(self):
        frames = [_make_frame([
            _make_det('Dan', 0.25),
            _make_det('Kibble', 0.19),
        ], frame_idx=0)]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 1
        assert 'low-conf-dan-25' in result[0].tags
        assert 'low-conf-kibble-19' in result[0].tags


# ---------------------------------------------------------------------------
# Blip detection
# ---------------------------------------------------------------------------
class TestBlipDetection:
    def test_flags_single_frame_appearance(self):
        # Sanbo appears in frame 5 only, absent before and after
        frames = []
        for i in range(15):
            if i == 5:
                frames.append(_make_frame([_make_det('Sanbo', 0.80)], frame_idx=i))
            else:
                frames.append(_make_frame([], frame_idx=i))
        result = flag_detections(frames, blip_max_frames=2, blip_gap_frames=5)
        blip_tags = [t for r in result for t in r.tags if 'blip-' in t]
        assert 'blip-sanbo' in blip_tags

    def test_no_blip_for_sustained_detection(self):
        # Sanbo appears for 10 consecutive frames -- not a blip
        frames = []
        for i in range(20):
            if 3 <= i <= 12:
                frames.append(_make_frame([_make_det('Sanbo', 0.80)], frame_idx=i))
            else:
                frames.append(_make_frame([], frame_idx=i))
        result = flag_detections(frames, blip_max_frames=2, blip_gap_frames=5)
        blip_tags = [t for r in result for t in r.tags if 'blip-' in t]
        assert len(blip_tags) == 0


# ---------------------------------------------------------------------------
# No co-detection
# ---------------------------------------------------------------------------
class TestNoCodetection:
    def test_dan_hand_without_dan_body(self):
        frames = [_make_frame([_make_det('Dan_hand', 0.75)], frame_idx=0)]
        result = flag_detections(frames)
        assert any('no-codetect-dan_hand' in t for r in result for t in r.tags)

    def test_dan_hand_with_dan_body_ok(self):
        frames = [_make_frame([
            _make_det('Dan_hand', 0.75),
            _make_det('Dan', 0.90),
        ], frame_idx=0)]
        result = flag_detections(frames)
        codetect_tags = [t for r in result for t in r.tags if 'no-codetect' in t]
        assert len(codetect_tags) == 0


# ---------------------------------------------------------------------------
# Conflict (overlapping Dan & Sanbo)
# ---------------------------------------------------------------------------
class TestConflict:
    def test_overlapping_dan_sanbo(self):
        # Identical boxes => IoU = 1.0
        frames = [_make_frame([
            _make_det('Dan', 0.90, [100, 100, 300, 300]),
            _make_det('Sanbo', 0.88, [100, 100, 300, 300]),
        ], frame_idx=0)]
        result = flag_detections(frames, iou_conflict=0.50)
        assert any('conflict-dan-sanbo' in t for r in result for t in r.tags)

    def test_no_conflict_when_far_apart(self):
        frames = [_make_frame([
            _make_det('Dan', 0.90, [0, 0, 50, 50]),
            _make_det('Sanbo', 0.88, [500, 500, 600, 600]),
        ], frame_idx=0)]
        result = flag_detections(frames, iou_conflict=0.50)
        conflict_tags = [t for r in result for t in r.tags if 'conflict' in t]
        assert len(conflict_tags) == 0


# ---------------------------------------------------------------------------
# Kibble jump
# ---------------------------------------------------------------------------
class TestKibbleJump:
    def test_large_kibble_count_change(self):
        # Frame 0: 2 kibble, Frame 1: 10 kibble => delta 8
        frames = [
            _make_frame([_make_det('Kibble', 0.90)] * 2, frame_idx=0),
            _make_frame([_make_det('Kibble', 0.90)] * 10, frame_idx=1),
        ]
        result = flag_detections(frames, kibble_jump=5)
        jump_tags = [t for r in result for t in r.tags if 'kibble-jump' in t]
        assert 'kibble-jump-8' in jump_tags


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_adjacent_flagged_frames_deduped(self):
        # Two adjacent frames both with low-conf detections
        frames = [
            _make_frame([_make_det('Dan', 0.20)], frame_idx=0),
            _make_frame([_make_det('Sanbo', 0.25)], frame_idx=1),
        ]
        result = flag_detections(frames, conf_threshold=0.40, dedup_window=3)
        # Should be merged into a single FlaggedFrame
        assert len(result) == 1
        assert 'low-conf-dan-20' in result[0].tags
        assert 'low-conf-sanbo-25' in result[0].tags
