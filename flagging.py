"""Auto-flag suspicious YOLO detections from a detection cache."""
from dataclasses import dataclass, field


@dataclass
class FlaggedFrame:
    frame_idx: int
    jpeg: bytes
    tags: list = field(default_factory=list)
    max_conf: float = 0.0
    detections: list = field(default_factory=list)


def _det_cls(det):
    """Extract class name — handles both old ('class') and new ('class_name') cache formats."""
    return det.get('class_name') or det.get('class', '')


def _det_conf(det):
    """Extract confidence — handles both old ('confidence') and new ('conf') formats."""
    return det.get('conf') if 'conf' in det else det['confidence']


def _det_box(det):
    """Extract [x1,y1,x2,y2] — handles all cache formats:
    - new CI format: x1/y1/x2/y2 keys
    - old morning_report format: 'box' list
    - old batch_review format: 'bbox' list
    """
    if 'x1' in det:
        return [det['x1'], det['y1'], det['x2'], det['y2']]
    if 'bbox' in det:
        return det['bbox']
    return det['box']


def _bbox_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _largest_box(detections, class_name):
    boxes = [d for d in detections if _det_cls(d) == class_name]
    if not boxes:
        return None
    return max(
        boxes,
        key=lambda d: (_det_box(d)[2] - _det_box(d)[0]) * (_det_box(d)[3] - _det_box(d)[1]),
    )


def _clear_bowl_frame(detections):
    bowl = _largest_box(detections, 'Bowl')
    if bowl is None:
        return False
    bowl_box = _det_box(bowl)
    for det in detections:
        if _det_cls(det) in ('Dan', 'Sanbo', 'Dan_hand'):
            if _bbox_iou(_det_box(det), bowl_box) > 0.10:
                return False
    return True


def _find_blips(frames, blip_max_frames, blip_gap_frames):
    """Find classes that appear for <= blip_max_frames then vanish."""
    # Build per-class list of frame indices where detected
    class_frames = {}
    for i, frame in enumerate(frames):
        for det in frame.get('detections', []):
            cls = _det_cls(det)
            class_frames.setdefault(cls, []).append(i)

    # For each class, find runs of consecutive frames
    blip_results = {}  # class -> set of frame indices that are blips
    for cls, indices in class_frames.items():
        indices = sorted(set(indices))
        indices_set = set(indices)  # Compute once, reuse in loops
        # Split into consecutive runs
        runs = []
        run_start = indices[0]
        run_end = indices[0]
        for idx in indices[1:]:
            if idx == run_end + 1:
                run_end = idx
            else:
                runs.append((run_start, run_end))
                run_start = idx
                run_end = idx
        runs.append((run_start, run_end))

        for start, end in runs:
            run_len = end - start + 1
            if run_len > blip_max_frames:
                continue
            # Check gap after: no detection for >= blip_gap_frames
            gap_after = True
            for j in range(end + 1, min(end + 1 + blip_gap_frames, len(frames))):
                if j in indices_set:
                    gap_after = False
                    break
            # If run is at end of video, also count as blip (vanished)
            if end + 1 >= len(frames):
                gap_after = True
            # Check gap before as well
            gap_before = True
            for j in range(max(0, start - blip_gap_frames), start):
                if j in indices_set:
                    gap_before = False
                    break
            if start == 0:
                gap_before = True

            if gap_after and gap_before and run_len <= blip_max_frames:
                blip_results.setdefault(cls, set())
                for fi in range(start, end + 1):
                    blip_results[cls].add(fi)

    return blip_results


def flag_detections(frames, conf_threshold=0.40, blip_max_frames=2,
                    blip_gap_frames=5, iou_conflict=0.50, kibble_jump=5,
                    dedup_window=3):
    """Scan detection cache and return flagged frames with tags."""
    # Per-frame tags: dict of frame_idx -> list of tags
    frame_tags = {}

    def _add_tag(idx, tag):
        frame_tags.setdefault(idx, []).append(tag)

    # 1. Low-confidence
    for i, frame in enumerate(frames):
        for det in frame.get('detections', []):
            if _det_conf(det) < conf_threshold:
                score_int = int(_det_conf(det) * 100)
                cls_lower = _det_cls(det).lower()
                _add_tag(i, f'low-conf-{cls_lower}-{score_int}')

    # 2. Blip detection
    blips = _find_blips(frames, blip_max_frames, blip_gap_frames)
    for cls, blip_indices in blips.items():
        for i in blip_indices:
            _add_tag(i, f'blip-{cls.lower()}')

    # 3. No co-detection (Dan_hand without Dan)
    for i, frame in enumerate(frames):
        dets = frame.get('detections', [])
        classes_in_frame = {_det_cls(d) for d in dets}
        if 'Dan_hand' in classes_in_frame and 'Dan' not in classes_in_frame:
            _add_tag(i, 'no-codetect-dan_hand')

    # 4. High-confidence conflict (Dan & Sanbo overlapping)
    for i, frame in enumerate(frames):
        dets = frame.get('detections', [])
        dan_dets = [d for d in dets if _det_cls(d) == 'Dan']
        sanbo_dets = [d for d in dets if _det_cls(d) == 'Sanbo']
        for dd in dan_dets:
            for sd in sanbo_dets:
                if _bbox_iou(_det_box(dd), _det_box(sd)) >= iou_conflict:
                    _add_tag(i, 'conflict-dan-sanbo')

    # 5. Kibble count jump
    # Only compare clear bowl frames. During eating, cats occlude kibble and
    # normal movement can look like a large count jump.
    prev_kibble = None
    for i, frame in enumerate(frames):
        dets = frame.get('detections', [])
        if not _clear_bowl_frame(dets):
            continue
        kibble_count = sum(1 for d in dets if _det_cls(d) == 'Kibble')
        if prev_kibble is not None:
            delta = abs(kibble_count - prev_kibble)
            if delta > kibble_jump:
                _add_tag(i, f'kibble-jump-{delta}')
        prev_kibble = kibble_count

    # Build FlaggedFrame list (only frames that have tags)
    flagged = []
    for i in sorted(frame_tags.keys()):
        tags = list(dict.fromkeys(frame_tags[i]))  # deduplicate preserving order
        dets = frames[i].get('detections', [])
        max_conf = max((_det_conf(d) for d in dets), default=0.0)
        flagged.append(FlaggedFrame(
            frame_idx=i,
            jpeg=frames[i].get('jpeg', b''),
            tags=tags,
            max_conf=max_conf,
            detections=frames[i].get('detections', []),
        ))

    # Deduplication: merge adjacent flagged frames within dedup_window
    if not flagged:
        return flagged

    merged = [flagged[0]]
    for ff in flagged[1:]:
        prev = merged[-1]
        if ff.frame_idx - prev.frame_idx <= dedup_window:
            # Merge into the one with higher max_conf
            if ff.max_conf > prev.max_conf:
                # Keep ff's frame data, merge tags
                all_tags = list(dict.fromkeys(prev.tags + ff.tags))
                merged[-1] = FlaggedFrame(
                    frame_idx=ff.frame_idx,
                    jpeg=ff.jpeg,
                    tags=all_tags,
                    max_conf=ff.max_conf,
                    detections=ff.detections,
                )
            else:
                # Keep prev's frame data, merge tags
                all_tags = list(dict.fromkeys(prev.tags + ff.tags))
                merged[-1] = FlaggedFrame(
                    frame_idx=prev.frame_idx,
                    jpeg=prev.jpeg,
                    tags=all_tags,
                    max_conf=prev.max_conf,
                    detections=prev.detections,
                )
        else:
            merged.append(ff)

    return merged
