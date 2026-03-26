"""Upload flagged frames to Roboflow for review and retraining."""
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime

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
        try:
            filename = f"{video_stem}_frame{ff.frame_idx:05d}.jpg"
            tmp_path = os.path.join(tempfile.gettempdir(), filename)
            with open(tmp_path, 'wb') as f:
                f.write(ff.jpeg)

            rf_project.upload(
                image_path=tmp_path,
                batch_name=batch_name,
                tag_names=ff.tags,
                num_retry_uploads=2,
            )

            result.uploaded += 1
            for tag in ff.tags:
                base = _strip_trailing_number(tag)
                result.tag_counts[base] = result.tag_counts.get(base, 0) + 1

        except Exception:
            result.failed += 1

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
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
