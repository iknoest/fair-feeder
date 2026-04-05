from pathlib import Path
from datetime import datetime, timedelta
import re

# ── Paste the two functions under test here (will match notebook) ──

def _parse_clip_times(filename):
    m = re.match(r'motion_(\d{8})_(\d{6})(?:_(\d+)m)?_(\d+)s', Path(filename).stem)
    if not m:
        return None, None
    start = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
    minutes = int(m.group(3)) if m.group(3) else 0
    seconds = int(m.group(4))
    return start, start + timedelta(minutes=minutes, seconds=seconds)

def _group_by_gap(paths, gap_sec=10):
    if not paths:
        return []
    paths = sorted(paths, key=lambda p: p.name)
    timed = [(p, *_parse_clip_times(p.name)) for p in paths]
    groups, cur = [], [timed[0]]
    for i in range(1, len(timed)):
        prev_end = cur[-1][2]
        curr_start = timed[i][1]
        if prev_end and curr_start:
            gap = (curr_start - prev_end).total_seconds()
        else:
            gap = gap_sec + 1
        if gap <= gap_sec:
            cur.append(timed[i])
        else:
            groups.append([t[0] for t in cur])
            cur = [timed[i]]
    groups.append([t[0] for t in cur])
    return groups

# ── Tests ──

def test_parse_clip_times_with_minutes():
    p = Path("motion_20260322_062004_1m_56s.mp4")
    start, end = _parse_clip_times(p.name)
    assert start == datetime(2026, 3, 22, 6, 20, 4)
    assert end == datetime(2026, 3, 22, 6, 22, 0)

def test_parse_clip_times_seconds_only():
    p = Path("motion_20260322_062540_25s.mp4")
    start, end = _parse_clip_times(p.name)
    assert start == datetime(2026, 3, 22, 6, 25, 40)
    assert end == datetime(2026, 3, 22, 6, 26, 5)

def test_group_far_apart_clips_are_separate_events():
    # Gap = 06:25:40 - 06:22:00 = 220s > 10s → 2 groups
    clips = [
        Path("motion_20260322_062004_1m_56s.mp4"),
        Path("motion_20260322_062540_25s.mp4"),
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 2
    assert groups[0][0].name == "motion_20260322_062004_1m_56s.mp4"
    assert groups[1][0].name == "motion_20260322_062540_25s.mp4"

def test_group_close_clips_are_same_event():
    # Clip 1 ends 06:20:30, Clip 2 starts 06:20:35 → gap = 5s ≤ 10s → 1 group
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062035_20s.mp4"),
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 1
    assert len(groups[0]) == 2

def test_group_exactly_10s_gap_is_same_event():
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062040_20s.mp4"),  # gap = 10s exactly
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 1

def test_group_11s_gap_is_separate():
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062041_20s.mp4"),  # gap = 11s
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 2

def test_empty_input():
    assert _group_by_gap([]) == []

def test_single_clip():
    clips = [Path("motion_20260322_062000_30s.mp4")]
    groups = _group_by_gap(clips)
    assert len(groups) == 1
    assert len(groups[0]) == 1

# ── Minimal FeedingTracker stub for testing attribution logic ──

import numpy as np

class _MockTracker:
    """Minimal tracker stub — only implements what we're testing."""
    def __init__(self, fps, kibble_counts, dan_at_bowl, sanbo_at_bowl):
        self.fps = fps
        self.kibble_counts = kibble_counts
        self.dan_at_bowl = dan_at_bowl
        self.sanbo_at_bowl = sanbo_at_bowl

    def _find_kibble_at_phase_entry(self, phase_start, phase_end, entry_sec=1.0):
        entry_frames = max(1, int(self.fps * entry_sec))
        end_sample = min(phase_start + entry_frames, phase_end + 1)
        counts = self.kibble_counts[phase_start:end_sample]
        return int(np.median(counts)) if counts else None

    def _find_kibble_at_phase_exit(self, phase_start, phase_end, exit_sec=2.0):
        exit_frames = max(1, int(self.fps * exit_sec))
        start_sample = max(phase_start, phase_end - exit_frames + 1)
        counts = self.kibble_counts[start_sample:phase_end + 1]
        return int(np.median(counts)) if counts else None


def test_phase_entry_returns_first_frames_median():
    # kibble=0 before cat, then 8,7,9 when cat arrives, then drops to 2
    t = _MockTracker(fps=2.0,
        kibble_counts=[0,0,0,0,0, 8,7,9,6,5,4,3,2, 0,0],
        dan_at_bowl=[False]*5 + [True]*8 + [False]*2,
        sanbo_at_bowl=[False]*15)
    # Phase: frames 5-12. Entry: first 2 frames (fps=2, entry_sec=1.0 → 2 frames): [8,7]
    # int(np.median([8,7])) = int(7.5) = 7
    result = t._find_kibble_at_phase_entry(5, 12)
    assert result == 7, f"Expected 7, got {result}"


def test_phase_exit_returns_last_frames_median():
    t = _MockTracker(fps=2.0,
        kibble_counts=[0,0,0,0,0, 8,7,9,6,5,4,3,2, 0,0],
        dan_at_bowl=[False]*5 + [True]*8 + [False]*2,
        sanbo_at_bowl=[False]*15)
    # Phase: frames 5-12. Exit: last 4 frames (fps=2, exit_sec=2.0 → 4 frames): [5,4,3,2]
    # sorted: [2,3,4,5], median = 3.5, int(3.5) = 3
    result = t._find_kibble_at_phase_exit(5, 12)
    assert result == 3, f"Expected 3 (median of last 4 frames [5,4,3,2]), got {result}"


def test_phase_entry_single_frame_fallback():
    # fps=0.5 means int(0.5 * 1.0) = 0, max(1, 0) = 1 frame
    t = _MockTracker(fps=0.5,
        kibble_counts=[10, 8, 5, 2],
        dan_at_bowl=[True]*4,
        sanbo_at_bowl=[False]*4)
    result = t._find_kibble_at_phase_entry(0, 3)
    assert result == 10, f"Expected 10, got {result}"


def test_phase_entry_empty_phase():
    t = _MockTracker(fps=2.0,
        kibble_counts=[5, 5, 5],
        dan_at_bowl=[True]*3,
        sanbo_at_bowl=[False]*3)
    # phase_start > phase_end (degenerate)
    result = t._find_kibble_at_phase_entry(5, 3)  # empty slice
    assert result is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
