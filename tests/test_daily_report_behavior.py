import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_report_globals():
    nb = json.loads((ROOT / "morning_report.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][9]["source"]).replace("\r", "")
    ns = {
        "np": np,
        "TIMESTAMP_REGEX": re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"),
        "DAN_HAND_GAP_SECONDS": 2.0,
        "DAN_HAND_MIN_SECONDS": 0.5,
        "DAN_HAND_CONF_THRESHOLD": 0.50,
        "OVERLAP_IOU_THRESHOLD": 0.10,
        "EATING_KIBBLE_DROP": 1,
    }

    def bbox_iou(a, b):
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
        area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
        denom = area_a + area_b - inter
        return inter / denom if denom else 0

    def draw_boxes(frame, detections, show_label=False):
        return frame.copy()

    ns["bbox_iou"] = bbox_iou
    ns["draw_boxes"] = draw_boxes
    exec(source, ns)
    return ns


def test_telegram_summary_uses_action_emoji_and_omits_schedule_and_why(monkeypatch):
    ns = _load_report_globals()
    monkeypatch.setenv("GHA_SCHEDULE_CRON", "0 3 * * *")
    monkeypatch.setenv("GHA_JOB_STARTED_AT_UTC", "2026-05-02T05:27:04Z")
    monkeypatch.setenv("GHA_SCHEDULE_DELAY_MIN", "147")

    text = ns["format_feeding_summary"]({
        "duration_sec": 123,
        "start_ts": "2026-05-02 06:20:00",
        "end_ts": "2026-05-02 06:22:03",
        "dan_first_ts": "2026-05-02 06:20:00",
        "sanbo_first_ts": None,
        "start_kibble": 29,
        "dan_kibble_eaten": 25,
        "sanbo_kibble_eaten": 0,
        "dan_bowl_seconds": 118,
        "sanbo_bowl_seconds": 0,
        "hand_episodes": [],
        "flag_summary_text": "Flags: 3 frames -> Roboflow (3 sent)",
    }, "feeding_merged.mp4")

    assert text.splitlines()[0] == "😸 Dan finished breakfast"
    assert "Schedule:" not in text
    assert "Why:" not in text
    assert "bowl from ~06:20:00" in text


def test_telegram_summary_uses_sanbo_action_and_no_activity_actions():
    ns = _load_report_globals()
    sanbo_text = ns["format_feeding_summary"]({
        "duration_sec": 30,
        "start_ts": "2026-05-02 06:20:00",
        "end_ts": "2026-05-02 06:20:30",
        "dan_first_ts": None,
        "sanbo_first_ts": "2026-05-02 06:20:10",
        "start_kibble": 12,
        "dan_kibble_eaten": 0,
        "sanbo_kibble_eaten": 7,
        "dan_bowl_seconds": 0,
        "sanbo_bowl_seconds": 20,
        "hand_episodes": [],
    }, "clip.mp4")
    assert sanbo_text.splitlines()[0] == "😿 Give Dan ~7 kibble"

    no_activity_text = ns["format_feeding_summary"]({
        "duration_sec": 9,
        "start_ts": "2026-05-02 06:28:00",
        "end_ts": "2026-05-02 06:28:09",
        "dan_first_ts": None,
        "sanbo_first_ts": None,
        "start_kibble": 0,
        "dan_kibble_eaten": 0,
        "sanbo_kibble_eaten": 0,
        "dan_bowl_seconds": 0,
        "sanbo_bowl_seconds": 0,
        "hand_episodes": [],
    }, "clip.mp4")
    assert no_activity_text.splitlines()[0] == "🍽️? Feeding machine not working?"


def test_kibble_snapshot_waits_for_stable_clear_count():
    ns = _load_report_globals()
    tracker = ns["FeedingTracker"](fps=2.0)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bowl = {"class_name": "Bowl", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
    dan = {"class_name": "Dan", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
    hand = {"class_name": "Dan_hand", "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}

    tracker.process_frame(0, [bowl, dan, hand], "2026-05-02 06:20:00", frame)
    kibble = {"class_name": "Kibble", "conf": 0.9, "x1": 1, "y1": 1, "x2": 2, "y2": 2}
    tracker.process_frame(1, [bowl, kibble], "2026-05-02 06:20:01", frame)
    assert "kibble_dispensed_ep0" not in tracker.snapshots

    tracker.process_frame(2, [bowl, kibble], "2026-05-02 06:20:02", frame)
    tracker.process_frame(3, [bowl, kibble], "2026-05-02 06:20:03", frame)

    assert "kibble_dispensed_ep0" in tracker.snapshots


def test_phase2_suppresses_later_empty_food_reports():
    nb = json.loads((ROOT / "morning_report.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][12]["source"]).replace("\r", "")

    assert "food_finished_seen = False" in source
    assert "Skipping {vid_name}: food was already finished in an earlier event." in source
    assert "end_kibble is not None and end_kibble <= empty_threshold" in source
