"""Upload flagged frames to Roboflow for review and retraining."""
import io
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image
from roboflow import Roboflow

from flagging import FlaggedFrame

# Class order must match Roboflow project (alphabetical = YOLO default)
_CLASS_ID = {'Bowl': 0, 'Dan': 1, 'Dan_hand': 2, 'Kibble': 3, 'Sanbo': 4}
_LABELMAP = {v: k for k, v in _CLASS_ID.items()}


def _det_cls(det):
    return det.get('class_name') or det.get('class', '')


def _det_box(det):
    if 'x1' in det:
        return [det['x1'], det['y1'], det['x2'], det['y2']]
    return det['box']


def _write_yolo_annotation(detections, width, height, path):
    """Write YOLO-format .txt annotation file from detection dicts."""
    lines = []
    for det in detections:
        cls_name = _det_cls(det)
        if cls_name not in _CLASS_ID:
            continue
        cls_id = _CLASS_ID[cls_name]
        x1, y1, x2, y2 = _det_box(det)
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


@dataclass
class UploadResult:
    uploaded: int = 0
    failed: int = 0
    tag_counts: dict = None

    def __post_init__(self):
        if self.tag_counts is None:
            self.tag_counts = {}


def _strip_trailing_number(tag):
    """Strip trailing -NN from a tag to get the base name.

    e.g. 'low-conf-sanbo-31' -> 'low-conf-sanbo'
         'blip-dan' -> 'blip-dan'
    """
    return re.sub(r'-\d+$', '', tag)


def upload_flagged_frames(flagged_frames, api_key, workspace, project,
                          video_stem, batch_name=None):
    """Upload flagged frames to Roboflow for human review.

    Returns UploadResult with counts of uploaded/failed frames and tag counts.
    """
    result = UploadResult()

    if not flagged_frames:
        return result

    if batch_name is None:
        batch_name = f"flagged-{datetime.now().strftime('%Y-%m')}"

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)

    for ff in flagged_frames:
        tmp_path = None
        ann_path = None
        try:
            filename = f"{video_stem}_frame{ff.frame_idx:05d}.jpg"
            tmp_path = os.path.join(tempfile.gettempdir(), filename)
            with open(tmp_path, 'wb') as f:
                f.write(ff.jpeg)

            # Write YOLO annotation so model predictions appear as pre-labels in Roboflow
            ann_path = tmp_path.replace('.jpg', '.txt')
            img = Image.open(io.BytesIO(ff.jpeg))
            w, h = img.size
            _write_yolo_annotation(ff.detections, w, h, ann_path)

            rf_project.upload(
                image_path=tmp_path,
                annotation_path=ann_path,
                annotation_labelmap=_LABELMAP,
                batch_name=batch_name,
                tag_names=ff.tags,
                num_retry_uploads=2,
                is_prediction=True,
            )

            result.uploaded += 1
            for tag in ff.tags:
                base = _strip_trailing_number(tag)
                result.tag_counts[base] = result.tag_counts.get(base, 0) + 1

        except Exception:
            result.failed += 1

        finally:
            for p in (tmp_path, ann_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    return result


def format_telegram_flag_summary(result):
    """Format an UploadResult into a Telegram-friendly summary string."""
    total = result.uploaded + result.failed
    if total == 0:
        return "No suspicious detections flagged"

    sorted_tags = sorted(result.tag_counts.items(), key=lambda x: x[1], reverse=True)
    tag_str = ", ".join(f"{count}x {tag}" for tag, count in sorted_tags)

    if result.failed == 0:
        header = f"Auto-flagged: {total} frames -> Roboflow"
    else:
        header = (f"Auto-flagged: {total} frames -> Roboflow "
                  f"({result.uploaded} uploaded, {result.failed} failed)")

    if tag_str:
        return f"{header}\n   {tag_str}"
    return header
