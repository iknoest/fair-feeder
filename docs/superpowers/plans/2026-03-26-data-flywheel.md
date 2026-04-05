# Data Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-flag suspicious YOLO detections and upload them to Roboflow for relabeling, closing the model improvement loop.

**Architecture:** New Python modules `flagging.py` and `roboflow_upload.py` contain the core logic. The `morning_report.ipynb` notebook gets two new cells (Phase 2.5 + 2.6) inserted after Phase 2. Drive video upload is removed from Cell 14. A separate `batch_review.ipynb` notebook reuses the same modules for historical video reprocessing in Colab.

**Tech Stack:** Python, Roboflow SDK (`roboflow` — already in requirements.txt), Ultralytics YOLO cache format (pickle), GitHub Actions, Jupyter/Papermill.

**Spec:** `docs/superpowers/specs/2026-03-26-data-flywheel-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `flagging.py` | Create | Scan detection cache, return list of flagged frames with tags |
| `roboflow_upload.py` | Create | Upload flagged frames to Roboflow via SDK |
| `test_flagging.py` | Create | Unit tests for flagging logic |
| `test_roboflow_upload.py` | Create | Unit tests for upload logic (mocked SDK) |
| `morning_report.ipynb` | Modify | Add Phase 2.5 + 2.6 cells, update Telegram summary, remove Drive video upload |
| `batch_review.ipynb` | Create | Colab notebook for historical video reprocessing |
| `.github/workflows/morning-report.yml` | Modify | Add `ROBOFLOW_API_KEY` env var, add `roboflow` to pip install |
| `fix_add_flagging_cells.py` | Create | Script to inject new cells into morning_report.ipynb |

---

## Task 1: Create `flagging.py` — auto-flag suspicious detections

**Files:**
- Create: `flagging.py`
- Create: `test_flagging.py`

This module scans a detection cache (the pickle format from Phase 1) and returns a list of flagged frames with reasons/tags.

- [ ] **Step 1: Write failing tests for low-confidence flagging**

Create `test_flagging.py`:

```python
"""Tests for flagging.py — auto-flag suspicious YOLO detections."""
import pytest
from flagging import flag_detections, FlaggedFrame


def _make_frame(detections, frame_idx=0, jpeg=b'\xff\xd8fake'):
    """Build a minimal cache frame dict."""
    return {
        'detections': detections,
        'timestamp': frame_idx / 15.0,
        'jpeg': jpeg,
        'height': 720,
        'width': 1280,
    }


def _make_det(cls_name, conf, box=(100, 100, 200, 200)):
    """Build a minimal detection dict."""
    return {'class': cls_name, 'confidence': conf, 'box': list(box)}


class TestLowConfidence:
    def test_flags_detection_below_threshold(self):
        frames = [_make_frame([_make_det('Sanbo', 0.31)])]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 1
        assert 'low-conf-sanbo-31' in result[0].tags

    def test_ignores_detection_above_threshold(self):
        frames = [_make_frame([_make_det('Sanbo', 0.85)])]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 0

    def test_includes_confidence_in_tag(self):
        frames = [_make_frame([_make_det('Dan', 0.28)])]
        result = flag_detections(frames, conf_threshold=0.40)
        assert 'low-conf-dan-28' in result[0].tags

    def test_multiple_low_conf_same_frame(self):
        frames = [_make_frame([
            _make_det('Dan', 0.30),
            _make_det('Kibble', 0.22),
        ])]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 1
        tags = result[0].tags
        assert 'low-conf-dan-30' in tags
        assert 'low-conf-kibble-22' in tags


class TestBlipDetection:
    def test_flags_single_frame_appearance(self):
        frames = [
            _make_frame([], frame_idx=0),
            _make_frame([], frame_idx=1),
            _make_frame([_make_det('Sanbo', 0.60)], frame_idx=2),
            _make_frame([], frame_idx=3),
            _make_frame([], frame_idx=4),
            _make_frame([], frame_idx=5),
            _make_frame([], frame_idx=6),
            _make_frame([], frame_idx=7),
        ]
        result = flag_detections(frames, conf_threshold=0.40,
                                 blip_max_frames=2, blip_gap_frames=5)
        assert len(result) == 1
        assert 'blip-sanbo' in result[0].tags

    def test_no_blip_for_sustained_detection(self):
        frames = [
            _make_frame([_make_det('Sanbo', 0.60)], frame_idx=i)
            for i in range(10)
        ]
        result = flag_detections(frames, conf_threshold=0.40,
                                 blip_max_frames=2, blip_gap_frames=5)
        # No blip tags (may have other flags or none)
        blip_results = [r for r in result if any('blip' in t for t in r.tags)]
        assert len(blip_results) == 0


class TestNoCodetection:
    def test_dan_hand_without_dan_body(self):
        frames = [_make_frame([_make_det('Dan_hand', 0.70)])]
        result = flag_detections(frames, conf_threshold=0.40)
        assert len(result) == 1
        assert 'no-codetect-dan_hand' in result[0].tags

    def test_dan_hand_with_dan_body_ok(self):
        frames = [_make_frame([
            _make_det('Dan_hand', 0.70),
            _make_det('Dan', 0.80),
        ])]
        result = flag_detections(frames, conf_threshold=0.40)
        # Should not have no-codetect tag
        codetect = [r for r in result if any('no-codetect' in t for t in r.tags)]
        assert len(codetect) == 0


class TestConflict:
    def test_overlapping_dan_sanbo(self):
        # Same bounding box = 100% IoU
        box = (100, 100, 300, 300)
        frames = [_make_frame([
            _make_det('Dan', 0.80, box),
            _make_det('Sanbo', 0.75, box),
        ])]
        result = flag_detections(frames, conf_threshold=0.40,
                                 iou_conflict=0.50)
        assert len(result) == 1
        assert 'conflict-dan-sanbo' in result[0].tags

    def test_no_conflict_when_far_apart(self):
        frames = [_make_frame([
            _make_det('Dan', 0.80, (10, 10, 50, 50)),
            _make_det('Sanbo', 0.75, (500, 500, 600, 600)),
        ])]
        result = flag_detections(frames, conf_threshold=0.40,
                                 iou_conflict=0.50)
        conflict = [r for r in result if any('conflict' in t for t in r.tags)]
        assert len(conflict) == 0


class TestKibbleJump:
    def test_large_kibble_count_change(self):
        frames = [
            _make_frame([_make_det('Kibble', 0.9)] * 3, frame_idx=0),
            _make_frame([_make_det('Kibble', 0.9)] * 11, frame_idx=1),
        ]
        result = flag_detections(frames, conf_threshold=0.40,
                                 kibble_jump=5)
        jump_results = [r for r in result if any('kibble-jump' in t for t in r.tags)]
        assert len(jump_results) == 1
        assert 'kibble-jump-8' in jump_results[0].tags


class TestDeduplication:
    def test_adjacent_flagged_frames_deduped(self):
        # 3 consecutive low-conf frames — should keep only the best one
        frames = [
            _make_frame([_make_det('Sanbo', 0.30)], frame_idx=0),
            _make_frame([_make_det('Sanbo', 0.35)], frame_idx=1),
            _make_frame([_make_det('Sanbo', 0.25)], frame_idx=2),
        ]
        result = flag_detections(frames, conf_threshold=0.40,
                                 dedup_window=3)
        assert len(result) == 1
        # Keeps highest-conf frame
        assert result[0].frame_idx == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_flagging.py -v`
Expected: `ModuleNotFoundError: No module named 'flagging'`

- [ ] **Step 3: Implement `flagging.py`**

Create `flagging.py`:

```python
"""Auto-flag suspicious YOLO detections from a detection cache."""
from dataclasses import dataclass, field


@dataclass
class FlaggedFrame:
    frame_idx: int
    jpeg: bytes
    tags: list = field(default_factory=list)
    max_conf: float = 0.0


def _iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _count_class(detections, cls_name):
    """Count detections of a given class."""
    return sum(1 for d in detections if d['class'] == cls_name)


def _find_blips(frames, blip_max_frames, blip_gap_frames):
    """Find single-frame blip appearances for each class."""
    all_classes = set()
    for f in frames:
        for d in f['detections']:
            all_classes.add(d['class'])

    blips = {}  # frame_idx -> set of blip tags
    for cls in all_classes:
        # Build presence array
        presence = [any(d['class'] == cls for d in f['detections']) for f in frames]
        # Find runs of presence
        i = 0
        while i < len(presence):
            if presence[i]:
                run_start = i
                while i < len(presence) and presence[i]:
                    i += 1
                run_end = i  # exclusive
                run_len = run_end - run_start
                if run_len <= blip_max_frames:
                    # Check gap after
                    gap_count = 0
                    j = run_end
                    while j < len(presence) and not presence[j]:
                        gap_count += 1
                        j += 1
                    if gap_count >= blip_gap_frames or run_end == len(presence):
                        # Also check gap before
                        gap_before = 0
                        k = run_start - 1
                        while k >= 0 and not presence[k]:
                            gap_before += 1
                            k -= 1
                        if gap_before >= blip_gap_frames or run_start == 0:
                            tag = f'blip-{cls.lower()}'
                            for fi in range(run_start, run_end):
                                blips.setdefault(fi, set()).add(tag)
            else:
                i += 1
    return blips


def flag_detections(frames, conf_threshold=0.40, blip_max_frames=2,
                    blip_gap_frames=5, iou_conflict=0.50, kibble_jump=5,
                    dedup_window=3):
    """Scan detection cache frames and return flagged frames with tags.

    Args:
        frames: list of cache frame dicts with 'detections', 'jpeg', etc.
        conf_threshold: flag detections below this confidence
        blip_max_frames: max consecutive frames for a blip
        blip_gap_frames: min gap frames after blip to confirm it vanished
        iou_conflict: IoU threshold for Dan/Sanbo overlap conflict
        kibble_jump: flag if kibble count changes by more than this
        dedup_window: merge flagged frames within this window

    Returns:
        list of FlaggedFrame, deduplicated
    """
    flagged = {}  # frame_idx -> FlaggedFrame

    def _ensure(idx, jpeg):
        if idx not in flagged:
            flagged[idx] = FlaggedFrame(frame_idx=idx, jpeg=jpeg)
        return flagged[idx]

    # 1. Low-confidence detections
    for i, frame in enumerate(frames):
        for det in frame['detections']:
            if det['confidence'] < conf_threshold:
                ff = _ensure(i, frame['jpeg'])
                score = int(det['confidence'] * 100)
                ff.tags.append(f"low-conf-{det['class'].lower()}-{score}")
                ff.max_conf = max(ff.max_conf, det['confidence'])

    # 2. Single-frame blips
    blips = _find_blips(frames, blip_max_frames, blip_gap_frames)
    for idx, tags in blips.items():
        ff = _ensure(idx, frames[idx]['jpeg'])
        ff.tags.extend(tags)
        # Update max_conf from this frame's detections
        for det in frames[idx]['detections']:
            ff.max_conf = max(ff.max_conf, det['confidence'])

    # 3. No co-detection (Dan_hand without Dan)
    for i, frame in enumerate(frames):
        has_dan_hand = any(d['class'] == 'Dan_hand' for d in frame['detections'])
        has_dan = any(d['class'] == 'Dan' for d in frame['detections'])
        if has_dan_hand and not has_dan:
            ff = _ensure(i, frame['jpeg'])
            ff.tags.append('no-codetect-dan_hand')
            for det in frame['detections']:
                ff.max_conf = max(ff.max_conf, det['confidence'])

    # 4. High-confidence conflict (Dan + Sanbo overlapping)
    for i, frame in enumerate(frames):
        dan_dets = [d for d in frame['detections'] if d['class'] == 'Dan']
        sanbo_dets = [d for d in frame['detections'] if d['class'] == 'Sanbo']
        for dd in dan_dets:
            for sd in sanbo_dets:
                if _iou(dd['box'], sd['box']) >= iou_conflict:
                    ff = _ensure(i, frame['jpeg'])
                    ff.tags.append('conflict-dan-sanbo')
                    ff.max_conf = max(ff.max_conf, dd['confidence'],
                                      sd['confidence'])

    # 5. Kibble count jump
    for i in range(1, len(frames)):
        prev_count = _count_class(frames[i - 1]['detections'], 'Kibble')
        curr_count = _count_class(frames[i]['detections'], 'Kibble')
        delta = abs(curr_count - prev_count)
        if delta > kibble_jump:
            ff = _ensure(i, frames[i]['jpeg'])
            ff.tags.append(f'kibble-jump-{delta}')
            for det in frames[i]['detections']:
                ff.max_conf = max(ff.max_conf, det['confidence'])

    # 6. Deduplicate adjacent frames
    if not flagged:
        return []

    sorted_indices = sorted(flagged.keys())
    groups = []
    current_group = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        if idx - current_group[-1] <= dedup_window:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    result = []
    for group in groups:
        # Keep frame with highest max_conf
        best = max(group, key=lambda idx: flagged[idx].max_conf)
        # Merge all tags from the group
        all_tags = []
        for idx in group:
            all_tags.extend(flagged[idx].tags)
        ff = flagged[best]
        ff.tags = sorted(set(all_tags))
        result.append(ff)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_flagging.py -v`
Expected: All 14 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flagging.py test_flagging.py
git commit -m "feat: add auto-flagging module for suspicious YOLO detections

Scans detection cache for low-confidence, single-frame blips,
missing co-detections, class conflicts, and kibble count jumps.
Returns deduplicated flagged frames with descriptive tags."
```

---

## Task 2: Create `roboflow_upload.py` — upload flagged frames

**Files:**
- Create: `roboflow_upload.py`
- Create: `test_roboflow_upload.py`

- [ ] **Step 1: Write failing tests**

Create `test_roboflow_upload.py`:

```python
"""Tests for roboflow_upload.py — upload flagged frames to Roboflow."""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import date
from flagging import FlaggedFrame
from roboflow_upload import upload_flagged_frames, UploadResult


class TestUploadFlaggedFrames:
    def _make_flagged(self, frame_idx=0, tags=None):
        return FlaggedFrame(
            frame_idx=frame_idx,
            jpeg=b'\xff\xd8\xff\xe0fake_jpeg_data',
            tags=tags or ['low-conf-sanbo-31'],
            max_conf=0.31,
        )

    @patch('roboflow_upload.Roboflow')
    def test_uploads_frame_to_roboflow(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        flagged = [self._make_flagged()]
        result = upload_flagged_frames(
            flagged, api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='merged_20260326',
        )
        assert result.uploaded == 1
        assert result.failed == 0
        mock_project.upload.assert_called_once()

    @patch('roboflow_upload.Roboflow')
    def test_batch_name_uses_current_month(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        flagged = [self._make_flagged()]
        result = upload_flagged_frames(
            flagged, api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='merged_20260326',
        )
        call_kwargs = mock_project.upload.call_args
        today = date.today()
        expected_batch = f"flagged-{today.strftime('%Y-%m')}"
        assert call_kwargs.kwargs.get('batch_name') == expected_batch

    @patch('roboflow_upload.Roboflow')
    def test_filename_includes_video_stem_and_frame(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        flagged = [self._make_flagged(frame_idx=142)]
        upload_flagged_frames(
            flagged, api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='merged_20260326_0618',
        )
        call_args = mock_project.upload.call_args
        image_path = call_args.kwargs.get('image_path', call_args.args[0] if call_args.args else '')
        assert 'merged_20260326_0618_frame00142' in str(image_path)

    @patch('roboflow_upload.Roboflow')
    def test_handles_upload_failure_gracefully(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_project.upload.side_effect = Exception('Network error')
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        flagged = [self._make_flagged(), self._make_flagged(frame_idx=10)]
        result = upload_flagged_frames(
            flagged, api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='test',
        )
        assert result.uploaded == 0
        assert result.failed == 2

    @patch('roboflow_upload.Roboflow')
    def test_empty_list_returns_zero(self, mock_rf_cls):
        result = upload_flagged_frames(
            [], api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='test',
        )
        assert result.uploaded == 0
        assert result.failed == 0

    @patch('roboflow_upload.Roboflow')
    def test_tags_passed_to_roboflow(self, mock_rf_cls):
        mock_project = MagicMock()
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        flagged = [self._make_flagged(tags=['low-conf-sanbo-31', 'blip-sanbo'])]
        upload_flagged_frames(
            flagged, api_key='test-key', workspace='test-ws',
            project='ir-kibble', video_stem='test',
        )
        call_kwargs = mock_project.upload.call_args.kwargs
        assert set(call_kwargs['tag_names']) == {'low-conf-sanbo-31', 'blip-sanbo'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_roboflow_upload.py -v`
Expected: `ModuleNotFoundError: No module named 'roboflow_upload'`

- [ ] **Step 3: Implement `roboflow_upload.py`**

Create `roboflow_upload.py`:

```python
"""Upload flagged frames to Roboflow for review and relabeling."""
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from roboflow import Roboflow
from flagging import FlaggedFrame


@dataclass
class UploadResult:
    uploaded: int = 0
    failed: int = 0
    tag_counts: dict = None

    def __post_init__(self):
        if self.tag_counts is None:
            self.tag_counts = {}


def upload_flagged_frames(flagged_frames, api_key, workspace, project,
                          video_stem, batch_name=None):
    """Upload flagged frames to Roboflow.

    Args:
        flagged_frames: list of FlaggedFrame from flagging.flag_detections()
        api_key: Roboflow API key
        workspace: Roboflow workspace slug
        project: Roboflow project name (e.g. 'ir-kibble')
        video_stem: video filename stem for naming uploaded images
        batch_name: optional override; defaults to 'flagged-YYYY-MM'

    Returns:
        UploadResult with counts
    """
    result = UploadResult()

    if not flagged_frames:
        return result

    if batch_name is None:
        batch_name = f"flagged-{date.today().strftime('%Y-%m')}"

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)

    for ff in flagged_frames:
        filename = f"{video_stem}_frame{ff.frame_idx:05d}.jpg"

        # Write JPEG bytes to temp file (SDK requires file path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as fh:
                fh.write(ff.jpeg)
                tmp_path = fh.name

            rf_project.upload(
                image_path=tmp_path,
                batch_name=batch_name,
                tag_names=ff.tags,
                num_retry_uploads=2,
            )
            result.uploaded += 1

            # Count tags — group by base (strip trailing number)
            # e.g. low-conf-sanbo-31 -> low-conf-sanbo
            for tag in ff.tags:
                if tag[-1].isdigit() and '-' in tag:
                    base = tag.rsplit('-', 1)[0]
                else:
                    base = tag
                result.tag_counts[base] = result.tag_counts.get(base, 0) + 1

        except Exception:
            result.failed += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return result


def format_telegram_flag_summary(result):
    """Format flag upload result for Telegram message.

    Args:
        result: UploadResult from upload_flagged_frames()

    Returns:
        str: formatted summary line(s) for Telegram
    """
    total = result.uploaded + result.failed
    if total == 0:
        return "No suspicious detections flagged"

    parts = []
    # Sort tag counts by frequency descending
    for tag, count in sorted(result.tag_counts.items(), key=lambda x: -x[1]):
        parts.append(f"{count}x {tag}")
    breakdown = ", ".join(parts)

    if result.failed == 0:
        header = f"Auto-flagged: {result.uploaded} frames -> Roboflow"
    else:
        header = (f"Auto-flagged: {total} frames -> Roboflow "
                  f"({result.uploaded} uploaded, {result.failed} failed)")

    return f"{header}\n   {breakdown}" if breakdown else header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_roboflow_upload.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Write test for `format_telegram_flag_summary`**

Add to `test_roboflow_upload.py`:

```python
class TestFormatTelegramSummary:
    def test_zero_flags(self):
        result = UploadResult(uploaded=0, failed=0)
        assert format_telegram_flag_summary(result) == "No suspicious detections flagged"

    def test_all_uploaded(self):
        result = UploadResult(uploaded=12, failed=0,
                              tag_counts={'low-conf-sanbo': 3, 'blip-kibble': 7, 'kibble-jump': 2})
        text = format_telegram_flag_summary(result)
        assert '12 frames' in text
        assert 'Roboflow' in text
        assert '7x blip-kibble' in text

    def test_partial_failure(self):
        result = UploadResult(uploaded=9, failed=3,
                              tag_counts={'low-conf-sanbo': 9})
        text = format_telegram_flag_summary(result)
        assert '9 uploaded' in text
        assert '3 failed' in text
```

- [ ] **Step 6: Run all upload tests**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_roboflow_upload.py -v`
Expected: All 10 tests PASS

- [ ] **Step 7: Commit**

```bash
git add roboflow_upload.py test_roboflow_upload.py
git commit -m "feat: add Roboflow upload module for flagged frames

Uploads flagged JPEG frames to Roboflow with tags and monthly batch
grouping. Includes Telegram summary formatter."
```

---

## Task 3: Inject Phase 2.5 + 2.6 cells into `morning_report.ipynb`

**Files:**
- Create: `fix_add_flagging_cells.py`
- Modify: `morning_report.ipynb` (via script)

The notebook must be modified programmatically (CLAUDE.md convention). The new cells go after Cell 12 (Phase 2: Analytics) and before Cell 13 (Phase 3: Output).

- [ ] **Step 1: Create the notebook injection script**

Create `fix_add_flagging_cells.py`:

```python
"""Inject Phase 2.5 (auto-flagging) and Phase 2.6 (Roboflow upload) cells
into morning_report.ipynb after the Phase 2 analytics cell."""
import json
import sys

NB_PATH = 'morning_report.ipynb'

PHASE_25_SOURCE = '''# Phase 2.5 — Auto-flag suspicious detections
# Scans the detection cache for low-confidence, blips, conflicts, etc.
# Runs in <1 second (reads from cache only)

from flagging import flag_detections

FLAG_CONF_THRESHOLD = 0.40
FLAG_BLIP_MAX_FRAMES = 2
FLAG_BLIP_GAP_FRAMES = 5
FLAG_IOU_CONFLICT = 0.50
FLAG_KIBBLE_JUMP = 5
FLAG_DEDUP_WINDOW = 3

all_flagged = {}  # vid_stem -> list of FlaggedFrame

for vr in video_results:
    vid_stem = vr['vid_stem']
    cache_path = os.path.join(OUTPUT_DIR, f"{vid_stem}_detections.pkl")
    if not os.path.exists(cache_path):
        print(f"  [skip] No cache for {vid_stem}")
        continue

    import pickle
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    flagged = flag_detections(
        cache['frames'],
        conf_threshold=FLAG_CONF_THRESHOLD,
        blip_max_frames=FLAG_BLIP_MAX_FRAMES,
        blip_gap_frames=FLAG_BLIP_GAP_FRAMES,
        iou_conflict=FLAG_IOU_CONFLICT,
        kibble_jump=FLAG_KIBBLE_JUMP,
        dedup_window=FLAG_DEDUP_WINDOW,
    )
    all_flagged[vid_stem] = flagged
    print(f"  {vid_stem}: {len(flagged)} frames flagged")

total_flagged = sum(len(v) for v in all_flagged.values())
print(f"\\nTotal flagged: {total_flagged} frames")
'''.replace('\r', '')

PHASE_26_SOURCE = '''# Phase 2.6 — Upload flagged frames to Roboflow
# Sends flagged JPEG frames directly to Roboflow for review/relabeling

from roboflow_upload import upload_flagged_frames, format_telegram_flag_summary, UploadResult

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')
ROBOFLOW_WORKSPACE = os.environ.get('ROBOFLOW_WORKSPACE', '')
ROBOFLOW_PROJECT = 'ir-kibble'

combined_result = UploadResult()

if not ROBOFLOW_API_KEY:
    print("ROBOFLOW_API_KEY not set — skipping Roboflow upload")
    flag_summary_text = "Roboflow upload skipped (no API key)"
elif total_flagged == 0:
    print("No frames flagged — nothing to upload")
    flag_summary_text = format_telegram_flag_summary(combined_result)
else:
    for vid_stem, flagged in all_flagged.items():
        if not flagged:
            continue
        print(f"  Uploading {len(flagged)} frames from {vid_stem}...")
        result = upload_flagged_frames(
            flagged,
            api_key=ROBOFLOW_API_KEY,
            workspace=ROBOFLOW_WORKSPACE,
            project=ROBOFLOW_PROJECT,
            video_stem=vid_stem,
        )
        combined_result.uploaded += result.uploaded
        combined_result.failed += result.failed
        for tag, count in result.tag_counts.items():
            combined_result.tag_counts[tag] = combined_result.tag_counts.get(tag, 0) + count

    flag_summary_text = format_telegram_flag_summary(combined_result)
    print(f"\\n{flag_summary_text}")
'''.replace('\r', '')


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\\n' for line in source.rstrip('\\n').split('\\n')]
    }


def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Find Phase 2 cell (Cell 12) — contains "Phase 2" and "FeedingTracker"
    phase2_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if 'Phase 2' in src and 'FeedingTracker' in src and cell['cell_type'] == 'code':
            phase2_idx = i
            break

    if phase2_idx is None:
        # Fallback: find cell with video_results and tracker.summarize
        for i, cell in enumerate(cells):
            src = ''.join(cell.get('source', []))
            if 'video_results' in src and 'summarize' in src and cell['cell_type'] == 'code':
                phase2_idx = i
                break

    if phase2_idx is None:
        print("ERROR: Could not find Phase 2 analytics cell", file=sys.stderr)
        sys.exit(1)

    print(f"Found Phase 2 cell at index {phase2_idx}")

    # Check if already injected
    for cell in cells:
        src = ''.join(cell.get('source', []))
        if 'Phase 2.5' in src:
            print("Phase 2.5 cell already exists — skipping injection")
            sys.exit(0)

    # Insert after Phase 2
    insert_idx = phase2_idx + 1
    cells.insert(insert_idx, make_code_cell(PHASE_25_SOURCE))
    cells.insert(insert_idx + 1, make_code_cell(PHASE_26_SOURCE))

    print(f"Inserted Phase 2.5 at index {insert_idx}")
    print(f"Inserted Phase 2.6 at index {insert_idx + 1}")

    with open(NB_PATH, 'w', encoding='utf-8', newline='\\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\\n')

    print("Done — saved morning_report.ipynb")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the injection script**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python fix_add_flagging_cells.py`
Expected: "Inserted Phase 2.5 at index 13" / "Inserted Phase 2.6 at index 14" / "Done"

- [ ] **Step 3: Verify injection by reading the notebook**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -c "import json; nb=json.load(open('morning_report.ipynb')); [print(f'Cell {i}: {repr(chr(10).join(c[\"source\"][:1])[:80])}') for i,c in enumerate(nb['cells']) if c['cell_type']=='code']"`
Expected: Two new cells visible with "Phase 2.5" and "Phase 2.6" in their first lines

- [ ] **Step 4: Commit**

```bash
git add fix_add_flagging_cells.py morning_report.ipynb
git commit -m "feat: add Phase 2.5/2.6 cells to morning_report.ipynb

Phase 2.5 auto-flags suspicious detections from cache.
Phase 2.6 uploads flagged frames to Roboflow."
```

---

## Task 4: Update Telegram summary to include flag count

**Files:**
- Modify: `morning_report.ipynb` (Cell 13/now Cell 15 — Phase 3 output+telegram)

The Phase 3 cell calls `send_telegram_summary()`. We need to append the flag summary to the Telegram message.

- [ ] **Step 1: Create script to patch the Telegram cell**

Create `fix_add_flag_to_telegram.py`:

```python
"""Append flag_summary_text to the Telegram message in Phase 3."""
import json

NB_PATH = 'morning_report.ipynb'


def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Find Phase 3 cell — contains "Phase 3" and "send_telegram"
    phase3_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if 'Phase 3' in src and 'send_telegram' in src and cell['cell_type'] == 'code':
            phase3_idx = i
            break

    if phase3_idx is None:
        print("ERROR: Could not find Phase 3 cell")
        return

    src = ''.join(cells[phase3_idx]['source'])

    # Check if already patched
    if 'flag_summary_text' in src:
        print("Phase 3 already includes flag summary — skipping")
        return

    # Find the send_telegram_summary call and prepend flag summary to summary_text
    # The summary_text variable is built before the send call
    # We add the flag line after summary_text is assigned
    old = "send_telegram_summary("
    new = """# Append flag summary to Telegram message
    if 'flag_summary_text' in dir():
        summary_text += '\\n\\n' + flag_summary_text

    send_telegram_summary("""

    if old not in src:
        print("ERROR: Could not find send_telegram_summary call")
        return

    src = src.replace(old, new.replace('\r', ''), 1)
    cells[phase3_idx]['source'] = [line + '\n' for line in src.rstrip('\n').split('\n')]

    with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f"Patched Phase 3 cell at index {phase3_idx}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the patch script**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python fix_add_flag_to_telegram.py`
Expected: "Patched Phase 3 cell at index NN"

- [ ] **Step 3: Commit**

```bash
git add fix_add_flag_to_telegram.py morning_report.ipynb
git commit -m "feat: add flag summary to Telegram daily report

Shows count and breakdown of auto-flagged frames sent to Roboflow."
```

---

## Task 5: Remove Drive video upload from Cell 14

**Files:**
- Modify: `morning_report.ipynb` (Cell 14/now Cell 16 — CSV + Drive upload)

Remove the `files().create()` block that uploads annotated videos to Drive. Keep CSV append + `files().update()`.

- [ ] **Step 1: Create script to remove Drive video upload**

Create `fix_remove_drive_video_upload.py`:

```python
"""Remove the annotated video Drive upload from the CSV/upload cell.
Keeps CSV download+append+update logic intact."""
import json
import re

NB_PATH = 'morning_report.ipynb'


def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Find the CSV + upload cell — contains "feeding_log.csv" and "files().create"
    target_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if 'feeding_log.csv' in src and cell['cell_type'] == 'code':
            target_idx = i
            break

    if target_idx is None:
        print("ERROR: Could not find CSV+upload cell")
        return

    src = ''.join(cells[target_idx]['source'])

    # Check if video upload code exists
    if 'video_paths' not in src or '_vmedia' not in src:
        print("Drive video upload already removed — skipping")
        return

    # Remove the video upload block: from "if 'video_paths'" to the end of its try/except
    # Use a regex to match from "if 'video_paths'" to the end of the video upload section
    # The block starts with "if 'video_paths' in globals()" and ends before the next
    # top-level statement or end of cell
    lines = src.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if "video_paths" in line and "globals()" in line:
            skip = True
            new_lines.append("# Drive video upload removed — SA has zero storage quota (Issue #33)")
            new_lines.append("# Annotated videos are delivered via Telegram; archive via Colab.")
            continue
        if skip:
            # Stop skipping at the next top-level statement (no indentation)
            stripped = line.lstrip()
            if stripped and not line.startswith(' ') and not line.startswith('\t'):
                skip = False
                new_lines.append(line)
            # else: skip this line (part of the video upload block)
            continue
        new_lines.append(line)

    new_src = '\n'.join(new_lines)
    cells[target_idx]['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]

    with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(f"Removed Drive video upload from cell {target_idx}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the patch script**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python fix_remove_drive_video_upload.py`
Expected: "Removed Drive video upload from cell NN"

- [ ] **Step 3: Commit**

```bash
git add fix_remove_drive_video_upload.py morning_report.ipynb
git commit -m "fix: remove Drive video upload from CI (SA zero quota)

SA cannot create new files on personal Drive. Annotated videos are
delivered via Telegram. Archive via Colab where Drive is mounted."
```

---

## Task 6: Update GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/morning-report.yml`

Add `ROBOFLOW_API_KEY` and `ROBOFLOW_WORKSPACE` env vars. Add `roboflow` to pip install.

- [ ] **Step 1: Add `roboflow` to pip install step**

In `.github/workflows/morning-report.yml`, find the pip install line and add `roboflow`:

Find:
```yaml
          pip install ultralytics easyocr opencv-python-headless papermill ipykernel google-auth google-api-python-client requests nbformat pytz
```

Replace with:
```yaml
          pip install ultralytics easyocr opencv-python-headless papermill ipykernel google-auth google-api-python-client requests nbformat pytz roboflow
```

- [ ] **Step 2: Add Roboflow env vars to the papermill step**

Find the `env:` block under the papermill step and add:
```yaml
          ROBOFLOW_API_KEY: ${{ secrets.ROBOFLOW_API_KEY }}
          ROBOFLOW_WORKSPACE: ${{ secrets.ROBOFLOW_WORKSPACE }}
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/morning-report.yml
git commit -m "ci: add Roboflow API key and SDK to morning report workflow"
```

---

## Task 7: Create `batch_review.ipynb` for Colab reprocessing

**Files:**
- Create: `batch_review.ipynb`

This is a Colab-only notebook for reprocessing historical Pi-captured videos. It reuses `flagging.py` and `roboflow_upload.py`.

- [ ] **Step 1: Create the notebook programmatically**

Create `create_batch_review.py`:

```python
"""Create batch_review.ipynb for Colab historical video reprocessing."""
import json

NB_PATH = 'batch_review.ipynb'

cells = []


def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source.strip().split('\n')]
    })


def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.strip().split('\n')]
    })


# Cell 0: Title
add_md("""# Batch Review — Historical Video Reprocessing

Reprocess Pi-captured videos through the YOLO pipeline, auto-flag suspicious
detections, and upload flagged frames to Roboflow for relabeling.

**Requires:** Google Colab with GPU runtime (T4 or better)""")

# Cell 1: Mount Drive + Install
add_code("""# Mount Google Drive and install dependencies
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics easyocr roboflow""")

# Cell 2: Config
add_code("""# Configuration — edit these paths and thresholds
import os
from pathlib import Path

# Paths (edit to match your Drive layout)
MODEL_PATH = '/content/drive/MyDrive/Fun Project/Cat monitor/model/fair_feeder_v13_yolov11s.pt'
VIDEO_DIR = '/content/drive/MyDrive/Fun Project/Cat monitor/Test_postmodel'
OUTPUT_DIR = '/content/drive/MyDrive/Fun Project/Cat monitor/Test_postmodel_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO config
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.20
IMGSZ = 1280
FRAME_SKIP = 7

# Flagging thresholds (same as daily pipeline)
FLAG_CONF_THRESHOLD = 0.40
FLAG_BLIP_MAX_FRAMES = 2
FLAG_BLIP_GAP_FRAMES = 5
FLAG_IOU_CONFLICT = 0.50
FLAG_KIBBLE_JUMP = 5
FLAG_DEDUP_WINDOW = 3

# Roboflow — load from Colab Secrets or paste here
# To use Colab Secrets: Runtime > Manage secrets > add ROBOFLOW_API_KEY
try:
    from google.colab import userdata
    ROBOFLOW_API_KEY = userdata.get('ROBOFLOW_API_KEY')
    ROBOFLOW_WORKSPACE = userdata.get('ROBOFLOW_WORKSPACE')
except Exception:
    ROBOFLOW_API_KEY = ''  # paste your key here if secrets unavailable
    ROBOFLOW_WORKSPACE = ''

ROBOFLOW_PROJECT = 'ir-kibble'

print(f"Model: {MODEL_PATH}")
print(f"Videos: {VIDEO_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Roboflow: {'configured' if ROBOFLOW_API_KEY else 'NOT configured'}")""")

# Cell 3: Load model + scan videos
add_code("""# Load model and scan video directory
from ultralytics import YOLO
from pathlib import Path

model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
video_paths = sorted([
    p for p in Path(VIDEO_DIR).iterdir()
    if p.suffix.lower() in VIDEO_EXTENSIONS
])

print(f"Found {len(video_paths)} videos:")
for vp in video_paths:
    size_mb = vp.stat().st_size / 1024 / 1024
    print(f"  {vp.name} ({size_mb:.1f} MB)")""")

# Cell 4: Phase 1 — YOLO inference + cache
add_code("""# Phase 1 — YOLO inference + cache (slow — runs once per video)
import cv2
import pickle
import numpy as np
from tqdm.auto import tqdm

for vp in video_paths:
    vid_stem = vp.stem
    cache_path = os.path.join(OUTPUT_DIR, f"{vid_stem}_detections.pkl")

    if os.path.exists(cache_path):
        print(f"  [cached] {vid_stem}")
        continue

    print(f"  Processing {vid_stem}...")
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_data = []
    frame_idx = 0

    for _ in tqdm(range(total), desc=vid_stem):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                            imgsz=IMGSZ, verbose=False, rect=True)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        'class': model.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'box': box.xyxy[0].tolist(),
                    })

            _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frames_data.append({
                'detections': detections,
                'timestamp': frame_idx / fps,
                'jpeg': jpeg_buf.tobytes(),
                'height': frame.shape[0],
                'width': frame.shape[1],
            })
        frame_idx += 1

    cap.release()

    cache = {'effective_fps': fps / FRAME_SKIP, 'frames': frames_data}
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Cached {len(frames_data)} frames -> {cache_path}")

print(f"\\nPhase 1 complete: {len(video_paths)} videos processed")""")

# Cell 5: Phase 2 — FeedingTracker (import from notebook helpers)
add_code("""# Phase 2 — FeedingTracker analytics
# Note: This cell loads the FeedingTracker from the main notebook.
# If running standalone, copy the FeedingTracker class from morning_report.ipynb Cell 9.
import pickle
import cv2
import numpy as np

# Try importing from the repo (if cloned to Colab)
import sys
sys.path.insert(0, '/content/drive/MyDrive/Fun Project/Cat monitor/fair-feeder')

video_results = []

for vp in video_paths:
    vid_stem = vp.stem
    cache_path = os.path.join(OUTPUT_DIR, f"{vid_stem}_detections.pkl")

    if not os.path.exists(cache_path):
        print(f"  [skip] No cache for {vid_stem}")
        continue

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"  {vid_stem}: {len(cache['frames'])} cached frames")

    video_results.append({
        'vid_name': vp.name,
        'vid_stem': vid_stem,
        'cache': cache,
    })

print(f"\\nLoaded {len(video_results)} video caches")""")

# Cell 6: Phase 2.5 — Auto-flagging
add_code("""# Phase 2.5 — Auto-flag suspicious detections
import sys
sys.path.insert(0, '/content/drive/MyDrive/Fun Project/Cat monitor/fair-feeder')
from flagging import flag_detections

all_flagged = {}

for vr in video_results:
    vid_stem = vr['vid_stem']
    cache = vr['cache']

    flagged = flag_detections(
        cache['frames'],
        conf_threshold=FLAG_CONF_THRESHOLD,
        blip_max_frames=FLAG_BLIP_MAX_FRAMES,
        blip_gap_frames=FLAG_BLIP_GAP_FRAMES,
        iou_conflict=FLAG_IOU_CONFLICT,
        kibble_jump=FLAG_KIBBLE_JUMP,
        dedup_window=FLAG_DEDUP_WINDOW,
    )
    all_flagged[vid_stem] = flagged

    if flagged:
        tag_counts = {}
        for ff in flagged:
            for t in ff.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
        tag_str = ', '.join(f'{c}x {t}' for t, c in top_tags)
        print(f"  {vid_stem}: {len(flagged)} flagged ({tag_str})")
    else:
        print(f"  {vid_stem}: 0 flagged")

total_flagged = sum(len(v) for v in all_flagged.values())
print(f"\\nTotal flagged: {total_flagged} frames across {len(video_results)} videos")""")

# Cell 7: Phase 2.6 — Upload to Roboflow
add_code("""# Phase 2.6 — Upload flagged frames to Roboflow
from roboflow_upload import upload_flagged_frames, format_telegram_flag_summary, UploadResult

if not ROBOFLOW_API_KEY:
    print("ROBOFLOW_API_KEY not set — skipping upload")
    print("Set it in Colab Secrets (Runtime > Manage secrets) or paste in the config cell")
else:
    combined = UploadResult()
    for vid_stem, flagged in all_flagged.items():
        if not flagged:
            continue
        print(f"  Uploading {len(flagged)} frames from {vid_stem}...")
        result = upload_flagged_frames(
            flagged,
            api_key=ROBOFLOW_API_KEY,
            workspace=ROBOFLOW_WORKSPACE,
            project=ROBOFLOW_PROJECT,
            video_stem=vid_stem,
        )
        combined.uploaded += result.uploaded
        combined.failed += result.failed
        for tag, count in result.tag_counts.items():
            combined.tag_counts[tag] = combined.tag_counts.get(tag, 0) + count

    print(f"\\n{format_telegram_flag_summary(combined)}")""")

# Cell 8: Summary
add_code("""# Summary
print("=" * 60)
print("BATCH REVIEW COMPLETE")
print("=" * 60)
print(f"Videos processed:  {len(video_results)}")
print(f"Total flagged:     {total_flagged}")
print()
print("Per-video breakdown:")
for vid_stem, flagged in all_flagged.items():
    print(f"  {vid_stem}: {len(flagged)} frames")
print()
print("Next steps:")
print("1. Open Roboflow -> Annotate tab -> filter by flagged batch")
print("2. Review and correct labels for flagged images")
print("3. Assign corrected images to training split")
print("4. When ~50-100 corrections accumulate: generate new dataset version")
print("5. Retrain with fair_feeder_v14.ipynb")""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "cells": cells,
}

with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f"Created {NB_PATH} with {len(cells)} cells")
```

- [ ] **Step 2: Run the creation script**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python create_batch_review.py`
Expected: "Created batch_review.ipynb with 9 cells"

- [ ] **Step 3: Verify notebook structure**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -c "import json; nb=json.load(open('batch_review.ipynb')); print(f'{len(nb[\"cells\"])} cells'); [print(f'  Cell {i} ({c[\"cell_type\"]}): {repr(chr(10).join(c[\"source\"][:1])[:60])}') for i,c in enumerate(nb['cells'])]"`
Expected: 9 cells listed with correct types and first lines

- [ ] **Step 4: Commit**

```bash
git add create_batch_review.py batch_review.ipynb
git commit -m "feat: add batch_review.ipynb for historical video reprocessing

Colab notebook that runs YOLO on all Pi-captured videos, auto-flags
suspicious detections, and uploads to Roboflow for relabeling."
```

---

## Task 8: User setup — Roboflow API key + GitHub secrets

This task is a manual checklist for the user. No code changes.

- [ ] **Step 1: Get Roboflow API key**

1. Go to https://app.roboflow.com → Settings (gear icon) → Roboflow API → Copy "Private API Key"
2. Note your workspace slug from the URL: `https://app.roboflow.com/<workspace-slug>/ir-kibble`

- [ ] **Step 2: Add GitHub secrets**

Go to the `fair-feeder` repo → Settings → Secrets and variables → Actions:

1. Add secret `ROBOFLOW_API_KEY` → paste your Roboflow API key
2. Add secret `ROBOFLOW_WORKSPACE` → paste your workspace slug

- [ ] **Step 3: Add to Infisical (for Colab)**

Add `ROBOFLOW_API_KEY` to your Infisical project (same place as `TelegramBotToken`).

- [ ] **Step 4: Add to Colab Secrets**

In Google Colab: Runtime → Manage secrets:
1. Add `ROBOFLOW_API_KEY` → paste your API key
2. Add `ROBOFLOW_WORKSPACE` → paste your workspace slug

- [ ] **Step 5: Verify by triggering a manual CI run**

Go to Actions tab → "Morning Kibble Report" → Run workflow → Run

Check:
- Telegram message arrives with flag summary at the bottom
- No 403 Drive upload errors in the workflow log
- Roboflow project shows new images in the `flagged-2026-03` batch

---

## Task 9: End-to-end verification

- [ ] **Step 1: Run all unit tests**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -m pytest test_flagging.py test_roboflow_upload.py tests/legacy_notebook/test_notebook_fixes.py -v`
Expected: All tests pass

- [ ] **Step 2: Verify notebook cells are correctly injected**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -c "import json; nb=json.load(open('morning_report.ipynb')); code=[c for c in nb['cells'] if c['cell_type']=='code']; [print(f'Code cell {i}: {repr(chr(10).join(c[\"source\"][:1])[:80])}') for i,c in enumerate(code)]"`
Expected: Phase 2.5 and Phase 2.6 cells visible in the listing, Drive video upload removed from CSV cell

- [ ] **Step 3: Verify batch_review.ipynb is valid**

Run: `cd C:/Users/AVAVAVA/.gemini/antigravity/scratch/fair-feeder && python -c "import json; nb=json.load(open('batch_review.ipynb')); print(f'Valid notebook: {len(nb[\"cells\"])} cells, nbformat {nb[\"nbformat\"]}')"`
Expected: "Valid notebook: 9 cells, nbformat 4"

- [ ] **Step 4: Final commit with all changes**

```bash
git status
# Ensure all files are committed. If any unstaged:
git add -A
git commit -m "chore: data flywheel implementation complete

- flagging.py: auto-flag suspicious detections (low-conf, blips, conflicts)
- roboflow_upload.py: upload flagged frames to Roboflow
- morning_report.ipynb: Phase 2.5/2.6 cells, flag summary in Telegram, Drive video upload removed
- batch_review.ipynb: Colab notebook for historical video reprocessing
- Tests: 24+ unit tests for flagging and upload logic"
```
