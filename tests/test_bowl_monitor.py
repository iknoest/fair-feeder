import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch("pathlib.Path.mkdir"):
    import motion_recorder


class _Box:
    def __init__(self, xyxy, cls=45):
        self.cls = [cls]
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    names = {45: "bowl"}

    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, frame, imgsz=640, conf=0.25, verbose=False):
        return [_Result([_Box(box) for box in self.boxes])]


class _Reader:
    def get_latest_frame(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)


def test_bowl_is_ok_when_full_view_is_on_right_side():
    model = _Model([[129, 43, 188, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model)

    status = monitor._detect_bowls(np.zeros((100, 200, 3), dtype=np.uint8))

    assert status["ok"] is True
    assert status["visible_count"] == 1


def test_bowl_is_bad_when_bbox_is_clipped_by_frame_edge():
    model = _Model([[170, 43, 200, 83]])
    monitor = motion_recorder.BowlPositionMonitor(_Reader(), model)

    status = monitor._detect_bowls(np.zeros((100, 200, 3), dtype=np.uint8))

    assert status["ok"] is False
    assert status["reason"] == "not fully visible"


def test_pi_notification_first_lines_use_requested_emojis():
    source = Path(motion_recorder.__file__).read_text(encoding="utf-8")

    assert "📷 Fair Feeder Monitor is LIVE" in source
    assert "✅🥣 Bowl position recovered" in source
    assert "👀? Camera position alert" in source
    assert "🥣? Bowl not detected" in source
