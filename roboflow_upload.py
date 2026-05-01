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
    if 'bbox' in det:
        return det['bbox']
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
    skipped: int = 0
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


def _load_tracking(tracking_file):
    """Load set of already-uploaded frame IDs from a tracking file."""
    if tracking_file and os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_tracking(tracking_file, frame_id):
    """Append a frame ID to the tracking file."""
    if tracking_file:
        with open(tracking_file, 'a') as f:
            f.write(frame_id + '\n')


def upload_flagged_frames(flagged_frames, api_key, workspace, project,
                          video_stem, batch_name=None, tracking_file=None):
    """Upload flagged frames to Roboflow for human review.

    tracking_file: path to a text file (on Drive) tracking already-uploaded
    frame IDs. Frames already in this file are skipped to avoid duplicates.

    Returns UploadResult with counts of uploaded/failed frames and tag counts.
    """
    result = UploadResult()

    if not flagged_frames:
        return result

    if batch_name is None:
        batch_name = f"flagged-{datetime.now().strftime('%Y-%m')}"

    already_uploaded = _load_tracking(tracking_file)

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)

    skipped = 0
    for ff in flagged_frames:
        tmp_path = None
        ann_path = None
        frame_id = f"{video_stem}_frame{ff.frame_idx:05d}"
        if frame_id in already_uploaded:
            skipped += 1
            continue
        try:
            filename = f"{frame_id}.jpg"
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
            _save_tracking(tracking_file, frame_id)
            already_uploaded.add(frame_id)
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

    result.skipped = skipped
    return result


def format_telegram_flag_summary(result, flagged_count=None):
    """Format an UploadResult into a Telegram-friendly summary string."""
    total = flagged_count if flagged_count is not None else result.uploaded + result.skipped + result.failed
    if total == 0:
        return "Flags: none"

    sorted_tags = sorted(result.tag_counts.items(), key=lambda x: x[1], reverse=True)
    tag_str = ", ".join(f"{tag} {count}" for tag, count in sorted_tags[:3])

    parts = [f"{result.uploaded} sent"]
    if result.skipped:
        parts.append(f"{result.skipped} skipped")
    if result.failed:
        parts.append(f"{result.failed} failed")
    header = f"Flags: {total} frames -> Roboflow ({', '.join(parts)})"

    if tag_str:
        return f"{header}; top: {tag_str}"
    return header
