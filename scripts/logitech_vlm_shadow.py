import cv2
import numpy as np

import os
import re
import csv
import json
import argparse
import base64
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import pytz

def load_env_safe():
    env_path = Path('.env')
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    k, v = parts
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

load_env_safe()

REQUIRED_VARS = ["GDRIVE_SERVICE_ACCOUNT_KEY"]

def check_credentials():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print("[STOP] Missing required environment variables:")
        for m in missing:
            print(f"- {m}")
        return False
    return True

FEEDING_WINDOW_START = (6, 18)
FEEDING_WINDOW_END = (6, 30)

MAX_API_CALLS_PER_RUN = 2
REQUEST_TIMEOUT_SECONDS = 60

def in_feeding_window(filename, target_date_str):
    m = re.match(r'motion_(\d{8})_(\d{2})(\d{2})\d{2}', filename)
    if not m:
        return False
    file_date = m.group(1)
    if file_date != target_date_str:
        return False
    file_min = int(m.group(2)) * 60 + int(m.group(3))
    start_min = FEEDING_WINDOW_START[0] * 60 + FEEDING_WINDOW_START[1]
    end_min = FEEDING_WINDOW_END[0] * 60 + FEEDING_WINDOW_END[1]
    return start_min <= file_min <= end_min

def simple_cat_heuristic(frame, bg_frame=None):
    if bg_frame is None:
        return False
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        gray_bg = bg_frame
    diff = cv2.absdiff(gray, gray_bg)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh) / 255
    return float(motion_score) > (frame.shape[0] * frame.shape[1] * 0.05)

def download_file(drive, file_id, dest_path):
    from googleapiclient.http import MediaIoBaseDownload
    if not dest_path.exists():
        req = drive.files().get_media(fileId=file_id)
        with open(dest_path, 'wb') as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
    return dest_path

def extract_timestamp_calc(filename, frame_idx, fps):
    m = re.search(r'(\d{8})_(\d{6})', filename)
    if not m: return ""
    try:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
        current_dt = start_dt + timedelta(seconds=frame_idx / fps)
        return current_dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ""

def make_contact_sheet(frames_data: list, out_path: Path, cols: int = 4):
    if not frames_data:
        return
    images = [data['frame_data'] for data in frames_data]
    target_w, target_h = 320, 180
    resized = [cv2.resize(img, (target_w, target_h)) for img in images]

    rows = (len(resized) + cols - 1) // cols
    total_slots = rows * cols
    for _ in range(total_slots - len(resized)):
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    row_images = []
    for r in range(rows):
        row_images.append(cv2.hconcat(resized[r*cols : (r+1)*cols]))

    contact_sheet = cv2.vconcat(row_images)
    cv2.imwrite(str(out_path), contact_sheet)

def enhance_image_gamma_clahe(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """Apply Gamma 2.5 + CLAHE enhancement to low-light RGB frames."""
    if img is None:
        return None
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(img, table)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def make_comparison_contact_sheet(raw_cs_path: Path, enhanced_cs_path: Path, out_path: Path):
    """Combine RAW and ENHANCED contact sheets into a single comparison image with clear labels."""
    raw_img = cv2.imread(str(raw_cs_path))
    enh_img = cv2.imread(str(enhanced_cs_path))
    if raw_img is None or enh_img is None:
        return

    h, w = raw_img.shape[:2]
    banner_h = 36
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Raw block with top banner
    raw_banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    cv2.putText(raw_banner, "[1/2] RAW LOGITECH FOOTAGE (Low-Light Baseline)", (12, 25), font, font_scale, (200, 200, 200), thickness)
    raw_block = cv2.vconcat([raw_banner, raw_img])

    # Enhanced block with top banner
    enh_banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    cv2.putText(enh_banner, "[2/2] ENHANCED EVIDENCE (Gamma 2.5 + CLAHE - Evaluated by Gemini)", (12, 25), font, font_scale, (0, 255, 255), thickness)
    enh_block = cv2.vconcat([enh_banner, enh_img])

    # Separator line
    separator = np.full((6, w, 3), 128, dtype=np.uint8)

    comparison = cv2.vconcat([raw_block, separator, enh_block])
    cv2.imwrite(str(out_path), comparison)

def extract_semantic_keyframes(cap, total_frames: int, fps: float, bg_frame: np.ndarray = None) -> list:
    """
    Extract semantic keyframes:
    1. pre_feed_baseline: earliest stable bowl frame before motion/cat starts
    2. first_approach: first detected motion frame
    3. early_eating: ~25% into the motion window
    4. mid_eating: ~50% into the motion window
    5. late_eating: ~75% into the motion window
    6. post_feed: post-feeding/exit frame after eating
    """
    if total_frames <= 0:
        return []

    if bg_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, bg_frame = cap.read()

    step = max(1, int(fps))
    motion_indices = []
    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and simple_cat_heuristic(frame, bg_frame):
            motion_indices.append(idx)

    labeled_samples = []

    if motion_indices:
        first_motion_idx = motion_indices[0]
        last_motion_idx = motion_indices[-1]

        # 1. Pre-feed baseline: before first motion starts (e.g. frame 0 or max(0, first_motion - 2s))
        pre_feed_idx = 0
        if first_motion_idx > int(fps):
            pre_feed_idx = max(0, first_motion_idx - int(fps * 1.5))
        labeled_samples.append((pre_feed_idx, "pre_feed_baseline"))

        # 2. First approach
        labeled_samples.append((first_motion_idx, "first_approach"))

        # Eating window: between first motion and last motion
        motion_span = max(1, last_motion_idx - first_motion_idx)
        if motion_span > int(fps * 2):
            # 3. Early eating (~25%)
            early_idx = min(total_frames - 1, first_motion_idx + int(motion_span * 0.25))
            labeled_samples.append((early_idx, "early_eating"))

            # 4. Mid eating (~50%)
            mid_idx = min(total_frames - 1, first_motion_idx + int(motion_span * 0.50))
            labeled_samples.append((mid_idx, "mid_eating"))

            # 5. Late eating (~75%)
            late_idx = min(total_frames - 1, first_motion_idx + int(motion_span * 0.75))
            labeled_samples.append((late_idx, "late_eating"))
        else:
            mid_idx = min(total_frames - 1, first_motion_idx + motion_span // 2)
            labeled_samples.append((mid_idx, "mid_eating"))

        # 6. Post-feed: end of clip or after last motion
        post_feed_idx = total_frames - 1
        labeled_samples.append((post_feed_idx, "post_feed"))

    else:
        # No motion detected - fallback to baseline, early, mid, late, post
        labeled_samples = [
            (0, "pre_feed_baseline"),
            (total_frames // 4, "early_eating"),
            (total_frames // 2, "mid_eating"),
            (3 * total_frames // 4, "late_eating"),
            (total_frames - 1, "post_feed")
        ]

    # Deduplicate and sort by frame index
    labeled_samples.sort(key=lambda x: x[0])
    seen = set()
    final_samples = []
    for idx, label in labeled_samples:
        clamped_idx = max(0, min(total_frames - 1, idx))
        if clamped_idx not in seen:
            seen.add(clamped_idx)
            final_samples.append((clamped_idx, label))

    return final_samples

def make_before_after_comparison(frames_by_reason: dict, out_path: Path):
    """
    Create a compact, high-signal before/after comparison image:
    [PRE-FEED] [EATING (optional)] [POST-FEED]
    """
    pre_frame = frames_by_reason.get("pre_feed_baseline")
    post_frame = frames_by_reason.get("post_feed")
    eating_frame = (
        frames_by_reason.get("mid_eating")
        or frames_by_reason.get("early_eating")
        or frames_by_reason.get("late_eating")
        or frames_by_reason.get("first_approach")
    )

    target_w, target_h = 480, 270
    banner_h = 32
    font = cv2.FONT_HERSHEY_SIMPLEX

    items = []
    if pre_frame is not None:
        items.append(("PRE-FEED BASELINE", pre_frame))
    if eating_frame is not None:
        items.append(("FEEDING ACTIVITY", eating_frame))
    if post_frame is not None:
        items.append(("POST-FEED", post_frame))

    if not items:
        return None

    rendered_blocks = []
    for title, img_data in items:
        raw_img = img_data['frame']
        ts = img_data.get('timestamp', '')
        # Apply enhancement
        enh = enhance_image_gamma_clahe(raw_img, gamma=2.5)
        resized = cv2.resize(enh, (target_w, target_h))

        banner = np.zeros((banner_h, target_w, 3), dtype=np.uint8)
        header_text = f"{title} ({ts})" if ts else title
        cv2.putText(banner, header_text, (10, 22), font, 0.50, (0, 255, 255), 1)
        block = cv2.vconcat([banner, resized])
        rendered_blocks.append(block)

    comparison_img = cv2.hconcat(rendered_blocks)
    cv2.imwrite(str(out_path), comparison_img)
    return out_path

def generate_enhanced_video(raw_mp4_path: Path, output_mp4_path: Path, gamma: float = 2.5) -> Path:
    """
    Applies Gamma 2.5 + CLAHE frame-by-frame on raw video on CI/runner.
    Verifies size using telegram_video_guard.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from telegram_video_guard import compress_video_for_telegram
    except ImportError:
        try:
            from scripts.telegram_video_guard import compress_video_for_telegram
        except ImportError:
            compress_video_for_telegram = None

    raw_mp4_path = Path(raw_mp4_path)
    output_mp4_path = Path(output_mp4_path)

    cap = cv2.VideoCapture(str(raw_mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    temp_raw_enh = output_mp4_path.with_name(f"temp_raw_{output_mp4_path.name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_raw_enh), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000000
    frames_read = 0
    while frames_read < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        enh_frame = enhance_image_gamma_clahe(frame, gamma=gamma)
        writer.write(enh_frame)
        frames_read += 1

    cap.release()
    writer.release()

    if not temp_raw_enh.exists() or temp_raw_enh.stat().st_size == 0:
        return output_mp4_path

    if compress_video_for_telegram:
        success, final_path, size_bytes = compress_video_for_telegram(temp_raw_enh, output_path=output_mp4_path)
        if temp_raw_enh.exists() and final_path != temp_raw_enh:
            temp_raw_enh.unlink()
        return final_path
    else:
        if temp_raw_enh.exists():
            if output_mp4_path.exists():
                output_mp4_path.unlink()
            temp_raw_enh.rename(output_mp4_path)
        return output_mp4_path

def generate_vlm_prompt(out_dir: Path, date_str: str, session_name: str = "session", has_references: bool = False):
    ref_text = ""
    if has_references:
        ref_text = """
The first few images provided are REFERENCE IMAGES, organized as:
REFERENCE — DAN
REFERENCE — SANBO
Followed by:
SESSION EVIDENCE (the contact sheet of the actual feeding session)

Reference images are ONLY for identity comparison.
They MUST NOT be used to infer eating, bowl state, hand presence, or food theft in the current session.
Those behaviors must come ONLY from the SESSION EVIDENCE frames.
"""

    prompt = f"""You are an expert feline behavior and feeding monitor. Your task is to analyze frames from a top-down RGB camera (Logitech) looking at a cat feeding bowl.

You are evaluating a complete feeding session consisting of sampled frames.
Date: {date_str}
{ref_text}
Rules:
1. Use only visible evidence from the provided frames.
2. BOWL STATE EVIDENCE CONTRACT: ONLY report bowl level ('full', 'half', 'low', 'empty') if clearly visible in the frame. If the initial/pre-feed bowl is obscured or unclear, you MUST report 'unsure' or 'UNKNOWN' for that phase (e.g. 'unsure -> empty'). Do NOT infer bowl level from subsequent consumption, cat duration, or assumptions.
3. Do not count individual kibble pieces. Provide a general bowl state progression grounded strictly in visible frames.
4. Do not claim machine failure or say "feeding machine not working".
5. If the cat identity is ambiguous or obstructed due to darkness, return `unsure`.
6. If the bowl state is obstructed, return `unsure`.
7. Sanbo is the light-colored cat with dark spots (cow/calico). Dan is the dark-colored cat with white markings (tuxedo).
8. Logitech is a top-down RGB/ambient view. Only rely on visual evidence.
9. Set 'identity_basis' to 'enhanced + reference-assisted' if reference images are present, otherwise 'raw'.
10. Set 'visibility' to 'poor', 'usable', or 'good'. Note that if the underlying capture is extremely dark (even if pre-processing/enhancement recovers usable contrast), visibility should not be labeled 'good'.
11. Calibrate your confidence carefully. A low-light, reference-assisted result should rarely reach 1.0 certainty, even if pre-processing makes markings easier to inspect.

Output ONLY valid JSON matching the exact expected schema below.

Expected JSON schema:
```json
{{
  "camera": "LOGITECH",
  "date": "{date_str}",
  "clip_name": "{session_name}",
  "cat_identity": "Dan | Sanbo | both | none | unsure",
  "identity_basis": "raw / enhanced + reference-assisted",
  "visibility": "poor | usable | good",
  "eating_evidence": "yes | no | unsure",
  "bowl_state": "empty | low | half | full | unsure | (e.g. unsure -> empty)",
  "confidence": 0.0,
  "reasons": ["short visual reasons..."],
  "needs_higher_model": true/false
}}
```"""
    prompt_file = out_dir / f"logitech_vlm_prompt_{session_name}.md"
    with open(prompt_file, "w") as f:
        f.write(prompt)
    if session_name != "session":
        with open(out_dir / "logitech_vlm_prompt_session.md", "w") as f:
            f.write(prompt)
    return prompt_file

def generate_vlm_schema(out_dir: Path):
    schema = {
        "camera": "LOGITECH",
        "date": "YYYYMMDD",
        "clip_name": "...",
        "cat_identity": "Dan | Sanbo | both | none | unsure",
        "identity_basis": "raw / enhanced + reference-assisted",
        "visibility": "poor / usable / good",
        "eating_evidence": "yes | no | unsure",
        "bowl_state": "empty | low | half | full | unsure",
        "confidence": 0.0,
        "reasons": ["short visual reasons"],
        "needs_higher_model": True
    }
    with open(out_dir / "logitech_vlm_expected_schema.json", "w") as f:
        json.dump(schema, f, indent=2)



def sanitize_error_message(message: str) -> str:
    s = str(message)
    if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
        s = s.replace(os.environ['OPENAI_API_KEY'], "***REDACTED***")
    if 'FAIR_FEEDER_GEMINI_API_KEY' in os.environ and os.environ['FAIR_FEEDER_GEMINI_API_KEY']:
        s = s.replace(os.environ['FAIR_FEEDER_GEMINI_API_KEY'], "***REDACTED***")
    if 'GEMINI_API_KEY' in os.environ and os.environ['GEMINI_API_KEY']:
        s = s.replace(os.environ['GEMINI_API_KEY'], "***REDACTED***")

    if 'TELEGRAM_BOT_TOKEN' in os.environ and os.environ['TELEGRAM_BOT_TOKEN']:
        s = s.replace(os.environ['TELEGRAM_BOT_TOKEN'], "***REDACTED***")
    if 'TELEGRAM_CHAT_ID' in os.environ and os.environ['TELEGRAM_CHAT_ID']:
        s = s.replace(os.environ['TELEGRAM_CHAT_ID'], "***REDACTED***")

    if 'GDRIVE_SERVICE_ACCOUNT_KEY' in os.environ and os.environ['GDRIVE_SERVICE_ACCOUNT_KEY']:
        s = s.replace(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'], "***REDACTED***")

    # Redact URL query param key=...
    s = re.sub(r'key=[^&\s]+', 'key=***REDACTED***', s)
    # Redact Authorization bearer token
    s = re.sub(r'(?i)bearer\s+[^\s]+', 'Bearer ***REDACTED***', s)
    # Redact Telegram bot URL tokens
    s = re.sub(r'https://api\.telegram\.org/bot[^/]+/', 'https://api.telegram.org/bot***REDACTED***/', s)
    # Redact JSON private_key field
    s = re.sub(r'("|\')private_key\1\s*:\s*("|\')(?:(?!\2).)*\2', r'\1private_key\1: \2***REDACTED***\2', s)
    # Redact PEM block independent of env
    s = re.sub(r'-----BEGIN (?:RSA )?PRIVATE KEY-----.*?-----END (?:RSA )?PRIVATE KEY-----', '-----BEGIN PRIVATE KEY-----\n***REDACTED***\n-----END PRIVATE KEY-----', s, flags=re.DOTALL)
    return s

def extract_duration_from_filename(filename):
    m = re.search(r'motion_\d{8}_\d{6}(?:_(\d+)m)?_(\d+)s', str(filename))
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        secs = int(m.group(2))
        return mins * 60 + secs
    return 0

def format_duration_str(seconds: float) -> str:
    secs = int(round(seconds))
    m, s = divmod(secs, 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def group_clips_into_sessions(selected_files, gap_threshold_sec=10):
    """
    Groups video clips into distinct feeding sessions based on time gaps.
    If gap between clip[i-1] end and clip[i] start <= gap_threshold_sec, they belong to the same session.
    Otherwise, they start a new session.
    """
    if not selected_files:
        return []

    from datetime import timedelta

    timed_clips = []
    for f in selected_files:
        name = f['name'] if isinstance(f, dict) else str(f)
        dur = extract_duration_from_filename(name)
        m = re.search(r'motion_(\d{8})_(\d{6})', name)
        if m:
            try:
                start_dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
                end_dt = start_dt + timedelta(seconds=dur)
                timed_clips.append({'file': f, 'start': start_dt, 'end': end_dt, 'duration': dur, 'name': name})
            except Exception:
                pass

    if not timed_clips:
        return [[f] for f in selected_files]

    timed_clips.sort(key=lambda x: x['start'])

    sessions = []
    current_session = [timed_clips[0]]

    for i in range(1, len(timed_clips)):
        prev_end = current_session[-1]['end']
        curr_start = timed_clips[i]['start']
        gap_sec = (curr_start - prev_end).total_seconds()

        if gap_sec <= gap_threshold_sec:
            current_session.append(timed_clips[i])
        else:
            sessions.append([item['file'] for item in current_session])
            current_session = [timed_clips[i]]

    if current_session:
        sessions.append([item['file'] for item in current_session])

    return sessions

def build_vlm_session_report(selected_files, manifest_data, all_results, all_failed, all_skipped, search_date, provider, model):
    """
    Aggregates clip-level VLM results and video metadata into a session-level feeding report.
    Guarantees: VLM MUST NOT count individual kibble pieces.
    """
    from datetime import timedelta

    # 1. Deterministic Session Timestamps & Duration
    clip_times = []
    total_active_video_sec = 0.0
    for f in selected_files:
        name = f['name'] if isinstance(f, dict) else str(f)
        m = re.search(r'motion_(\d{8})_(\d{6})', name)
        dur = extract_duration_from_filename(name)
        total_active_video_sec += dur
        if m:
            try:
                dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
                clip_times.append((dt, dt + timedelta(seconds=dur)))
            except Exception:
                pass

    if clip_times:
        clip_times.sort(key=lambda x: x[0])
        session_start_dt = clip_times[0][0]
        session_end_dt = clip_times[-1][1]
        session_span_sec = max(0.0, (session_end_dt - session_start_dt).total_seconds())
        start_time_str = session_start_dt.strftime("%H:%M:%S")
        end_time_str = session_end_dt.strftime("%H:%M:%S")
        span_str = format_duration_str(session_span_sec)
    else:
        start_time_str = "unknown"
        end_time_str = "unknown"
        span_str = "0s"

    clean_date = str(search_date).replace("-", "")
    if len(clean_date) == 8 and clean_date.isdigit():
        formatted_date = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:]}"
    else:
        formatted_date = str(search_date)

    # 2. Cat Identity Aggregation
    identities = [r.get("cat_identity") for r in all_results if r.get("cat_identity")]
    if any(i in ["both"] for i in identities) or ("Dan" in identities and "Sanbo" in identities):
        cat_identity = "both"
    elif identities and all(i == "Sanbo" for i in identities):
        cat_identity = "Sanbo"
    elif identities and all(i == "Dan" for i in identities):
        cat_identity = "Dan"
    elif identities and all(i == "none" for i in identities):
        cat_identity = "none"
    elif identities:
        cat_identity = identities[0]
    else:
        cat_identity = "unsure"

    # 3. Eating Evidence Aggregation
    eating_evidences = [r.get("eating_evidence") for r in all_results if r.get("eating_evidence")]
    if "yes" in eating_evidences:
        eating_evidence = "yes"
    elif "unsure" in eating_evidences:
        eating_evidence = "unsure"
    elif eating_evidences:
        eating_evidence = "no"
    else:
        eating_evidence = "unsure"

    # 4. Bowl State Progression (Qualitative start -> end)
    bowl_states = [r.get("bowl_state") for r in all_results if r.get("bowl_state")]
    dedup_states = []
    for bs in bowl_states:
        if not dedup_states or dedup_states[-1] != bs:
            dedup_states.append(bs)
    if len(dedup_states) > 1:
        bowl_state_progression = " → ".join(dedup_states)
    elif dedup_states:
        bowl_state_progression = dedup_states[0]
    else:
        bowl_state_progression = "unsure"

    # 5. First Bowl Interaction Time & Interaction Duration
    motion_timestamps = [m["timestamp"] for m in manifest_data if m.get("timestamp") and m.get("motion_detected")]
    if motion_timestamps:
        first_seen = f"~{motion_timestamps[0].split()[-1]}"
    elif manifest_data and manifest_data[0].get("timestamp"):
        first_seen = f"~{manifest_data[0]['timestamp'].split()[-1]}"
    else:
        first_seen = f"~{start_time_str}"

    bowl_interaction_duration = format_duration_str(total_active_video_sec)

    # 6. Hand / Human Interaction Detection
    reasons_text = " ".join([reason for r in all_results for reason in r.get("reasons", [])]).lower()
    hand_keywords = ["hand", "human", "person", "dispens", "refill", "finger"]
    if any(kw in reasons_text for kw in hand_keywords):
        hand_interaction = "1 ep (human hand/interaction observed)"
    else:
        hand_interaction = "none observed"

    # 7. Food Theft Flag
    possible_food_theft = cat_identity in ["both", "Dan"]

    # 8. Confidence & Higher Model Flag
    confidences = [r.get("confidence", 0.0) for r in all_results if isinstance(r.get("confidence"), (int, float))]
    confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0
    needs_higher_model = any(r.get("needs_higher_model", False) for r in all_results) or (confidence < 0.75)

    # 9. Failure Category (when failures occur)
    all_error_text = " ".join([str(f.get("error_message", "")) + " " + str(f.get("error_type", "")) for f in all_failed + all_skipped]).lower()
    if any(k in all_error_text for k in ["429", "quota", "billing", "prepayment", "apicapreached"]):
        failure_category = "provider quota/billing"
    elif any(k in all_error_text for k in ["401", "403", "credential", "unauthorized", "key"]):
        failure_category = "credentials/authentication"
    elif any(k in all_error_text for k in ["timeout", "connect", "network"]):
        failure_category = "network/timeout"
    elif all_failed or all_skipped:
        failure_category = "provider API error"
    else:
        failure_category = None

    session_data = {
        "date": formatted_date,
        "session_start_time": start_time_str,
        "session_end_time": end_time_str,
        "total_duration": span_str,
        "cat_identity": cat_identity,
        "identity_basis": all_results[0].get("identity_basis", "unknown") if all_results else "unknown",
        "visibility": all_results[0].get("visibility", "unknown") if all_results else "unknown",
        "eating_evidence": eating_evidence,
        "bowl_state_progression": bowl_state_progression,
        "first_recorded_motion_time": first_seen,
        "recorded_session_duration": bowl_interaction_duration,
        "hand_human_interaction": hand_interaction,
        "possible_food_theft": possible_food_theft,
        "confidence": confidence,
        "needs_higher_model": needs_higher_model,
        "evidence_clip_count": len(selected_files),
        "evidence_sampled_frame_count": len(manifest_data),
        "vlm_success_count": len(all_results),
        "vlm_failure_count": len(all_failed),
        "vlm_skipped_count": len(all_skipped),
        "failure_category": failure_category,
        "provider": provider,
        "model": model,
        "status": "completed" if all_results else "failed"
    }

    return session_data

def format_vlm_failure_report_text(session_data, all_failed, all_skipped, shadow_header=True):
    lines = []
    if shadow_header:
        lines.append("[SHADOW][LOGITECH] ⚠️ VLM analysis failed")
        lines.append("Non-authoritative shadow report. Production report unchanged.")
    lines.append(f"Date: {session_data.get('date', 'unknown')}")
    lines.append(f"Time: {session_data.get('session_start_time', '')}-{session_data.get('session_end_time', '')} ({session_data.get('total_duration', '')})")
    lines.append(f"Provider/model: {session_data.get('provider', '')} / {session_data.get('model', '')}")
    lines.append("")
    lines.append(f"Evidence prepared: {session_data.get('evidence_clip_count', 0)} clip(s) / {session_data.get('evidence_sampled_frame_count', 0)} frame(s)")
    lines.append(f"VLM analysis: FAILED (0/{session_data.get('evidence_clip_count', 0)} clips succeeded)")
    lines.append("")
    lines.append("No cat/eating/bowl conclusion was produced.")
    lines.append("Production report unchanged.")
    lines.append(f"Failure category: {session_data.get('failure_category') or 'provider API error'}")
    lines.append("")
    lines.append("Failures:")
    if all_failed:
        for f in all_failed:
            clip = f.get("clip_name", "unknown")
            err_type = f.get("error_type", "Error")
            msg = f.get("telegram_error_message", f.get("error_message", ""))
            lines.append(f"- {clip}: {err_type} - {msg}")
    if all_skipped:
        for s in all_skipped:
            clip = s.get("clip_name", "unknown")
            err_type = s.get("error_type", "Skipped")
            msg = s.get("telegram_error_message", s.get("error_message", ""))
            lines.append(f"- {clip}: {err_type} - {msg}")
    if not all_failed and not all_skipped:
        lines.append("- No clip results processed.")

    return "\n".join(lines)

def format_session_report_text(session_data, all_results, all_failed, all_skipped, shadow_header=True, custom_header=None):
    lines = []
    if custom_header:
        lines.append(custom_header)
    elif shadow_header:
        lines.append("[SHADOW] Logitech VLM Feeding Session Report")
        lines.append("Non-authoritative shadow report. Production report unchanged.")
    lines.append(f"Date: {session_data['date']}")
    lines.append(f"Time: {session_data['session_start_time']}-{session_data['session_end_time']} ({session_data['total_duration']})")
    lines.append(f"Provider/model: {session_data['provider']} / {session_data['model']}")
    lines.append("")
    lines.append("--- VLM VISUAL CONCLUSIONS ---")

    cat_id = session_data["cat_identity"]
    has_refs = bool(all_results and all_results[0].get('reference_images'))

    basis = session_data.get('identity_basis', 'enhanced + reference-assisted' if has_refs else 'raw')
    visibility = session_data.get('visibility', 'unknown')
    conf = session_data.get('confidence', 0.0)

    if session_data["possible_food_theft"]:
        if cat_id == "both":
            title = "😿 Possible food theft — Dan & Sanbo at bowl!"
            cat_line = f"      cat: both ⚠️ possible food theft — verify"
        else:
            title = f"😿 Possible food theft — {cat_id} at Sanbo feeder!"
            cat_line = f"      cat: {cat_id} ⚠️ Dan at Logitech/Sanbo feeder — verify"
    elif cat_id == "Sanbo":
        title = "😸 Sanbo feeding session"
        cat_line = f"      cat: Sanbo"
    elif cat_id == "Dan":
        title = "😸 Dan feeding session"
        cat_line = f"      cat: Dan"
    else:
        title = f"🐱 {cat_id} feeding session"
        cat_line = f"      cat: {cat_id}"

    lines.append(title)
    lines.append(cat_line)
    lines.append(f"      identity basis: {basis}")
    lines.append(f"      visibility: {visibility}")
    lines.append(f"      confidence: {conf}")

    ee = session_data["eating_evidence"]
    ee_flag = " ⚠️ eating uncertain" if ee == "unsure" else (" ⚠️ no eating evidence" if ee == "no" else "")
    lines.append(f"   eating: {ee}{ee_flag}")
    lines.append(f"     bowl: {session_data['bowl_state_progression']}")
    lines.append(f"     hand: {session_data['hand_human_interaction']}")

    lines.append("")
    lines.append("--- RECORDED MOTION METADATA ---")
    lines.append(f"motion start: {session_data['first_recorded_motion_time']}")
    lines.append(f"motion duration: ~{session_data['recorded_session_duration']}")
    lines.append(f"evidence: {session_data['evidence_clip_count']} clip(s) ({session_data['evidence_sampled_frame_count']} frames)")

    lines.append("")
    conf_flag = " ⚠️ low confidence" if session_data["confidence"] < 0.75 else ""
    lines.append(f"confidence: {session_data['confidence']}{conf_flag}")

    if session_data["needs_higher_model"]:
        lines.append("     model: ⚠️ needs higher model review")

    lines.append("")
    lines.append("Key Observations:")
    reasons = [reason for r in all_results for reason in r.get("reasons", [])]
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("- No VLM visual observations recorded.")

    lines.append("")
    lines.append("Verification / Flags:")
    flags = []
    if session_data["possible_food_theft"]:
        flags.append("⚠️ possible food theft — verify")
    if session_data["confidence"] < 0.75:
        flags.append("⚠️ low confidence score — manual review recommended")
    if session_data["needs_higher_model"]:
        flags.append("⚠️ marked for higher model review")
    if all_failed:
        flags.append(f"⚠️ {len(all_failed)} clip(s) failed API processing")
    if all_skipped:
        flags.append(f"⚠️ {len(all_skipped)} clip(s) skipped")

    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("- None (clean session)")

    return "\n".join(lines)

def format_multi_session_report_text(all_session_data, all_results, all_failed, all_skipped, shadow_header=True, custom_header=None):
    if len(all_session_data) == 1:
        return format_session_report_text(all_session_data[0], all_results, all_failed, all_skipped, shadow_header=shadow_header, custom_header=custom_header)

    lines = []
    if custom_header:
        lines.append(custom_header)
    elif shadow_header:
        lines.append("[SHADOW] Logitech VLM Feeding Report")
        lines.append("Non-authoritative shadow report. Production report unchanged.")
    if all_session_data:
        date = all_session_data[0]["date"]
        provider = all_session_data[0]["provider"]
        model = all_session_data[0]["model"]
        lines.append(f"Date: {date}")
        lines.append(f"Provider/model: {provider} / {model}")
        lines.append(f"Recorded Sessions: {len(all_session_data)} distinct event(s)")
        lines.append("")

        for i, session_data in enumerate(all_session_data, 1):
            matching_results = [r for r in all_results if r.get("clip_name") in [f"session_{i}", "session", str(i)]]
            res = matching_results[0] if matching_results else (all_results[i-1] if i-1 < len(all_results) else {})

            lines.append(f"=== EVENT {i} ({session_data['session_start_time']}-{session_data['session_end_time']}, {session_data['total_duration']}) ===")
            lines.append("--- VLM VISUAL CONCLUSIONS ---")

            cat_id = session_data["cat_identity"]
            has_refs = bool(res and res.get('reference_images'))
            basis = session_data.get('identity_basis', 'enhanced + reference-assisted' if has_refs else 'raw')
            visibility = session_data.get('visibility', 'unknown')
            conf = session_data.get('confidence', 0.0)

            if session_data["possible_food_theft"]:
                if cat_id == "both":
                    title = "😿 Possible food theft — Dan & Sanbo at bowl!"
                    cat_line = f"      cat: both ⚠️ possible food theft — verify"
                else:
                    title = f"😿 Possible food theft — {cat_id} at Sanbo feeder!"
                    cat_line = f"      cat: {cat_id} ⚠️ Dan at Logitech/Sanbo feeder — verify"
            elif cat_id == "Sanbo":
                title = "😸 Sanbo feeding session"
                cat_line = f"      cat: Sanbo"
            elif cat_id == "Dan":
                title = "😸 Dan feeding session"
                cat_line = f"      cat: Dan"
            else:
                title = f"🐱 {cat_id} feeding session"
                cat_line = f"      cat: {cat_id}"

            lines.append(title)
            lines.append(cat_line)
            lines.append(f"      identity basis: {basis}")
            lines.append(f"      visibility: {visibility}")
            lines.append(f"      confidence: {conf}")

            ee = session_data["eating_evidence"]
            ee_flag = " ⚠️ eating uncertain" if ee == "unsure" else (" ⚠️ no eating evidence" if ee == "no" else "")
            lines.append(f"   eating: {ee}{ee_flag}")
            lines.append(f"     bowl: {session_data['bowl_state_progression']}")
            lines.append(f"     hand: {session_data['hand_human_interaction']}")

            lines.append("")
            lines.append("--- RECORDED MOTION METADATA ---")
            lines.append(f"motion start: {session_data['first_recorded_motion_time']}")
            lines.append(f"motion duration: ~{session_data['recorded_session_duration']}")
            lines.append(f"evidence: {session_data['evidence_clip_count']} clip(s) ({session_data['evidence_sampled_frame_count']} frames)")
            lines.append("")

            reasons = res.get("reasons", []) if isinstance(res, dict) else []
            if reasons:
                lines.append("Observations:")
                for r in reasons:
                    lines.append(f"- {r}")
                lines.append("")

    return "\n".join(lines).strip()

def is_meaningful_feeding_event(session_data, result=None) -> bool:
    """
    Determines whether a feeding event has real user-facing feeding value.
    Returns False when cat_identity is 'none' AND eating_evidence is 'no' (and no anomaly/theft/human interaction).
    """
    cat_id = str(session_data.get("cat_identity", "none")).strip().lower()
    eating = str(session_data.get("eating_evidence", "no")).strip().lower()
    theft = bool(session_data.get("possible_food_theft", False))
    hand = str(session_data.get("hand_human_interaction", "none")).strip().lower()

    if theft:
        return True
    if hand not in ["none", "no", "none observed", ""]:
        return True
    if cat_id in ["none", "no"] and eating in ["no", "none"]:
        return False
    return True

def format_compact_session_text(session_data, result=None, shadow_header=True, custom_header=None) -> str:
    """Compact, high-signal Telegram report format."""
    lines = []
    cat_id = session_data.get("cat_identity", "unknown")
    eating = session_data.get("eating_evidence", "unknown")
    theft = session_data.get("possible_food_theft", False)
    bowl = session_data.get("bowl_state_progression", "unsure")
    conf = session_data.get("confidence", 0.0)
    vis = session_data.get("visibility", "poor")
    has_refs = bool(result and result.get('reference_images'))
    basis = session_data.get('identity_basis', 'enhanced + reference-assisted' if has_refs else 'raw')

    # Title line
    if custom_header:
        lines.append(custom_header)
    elif shadow_header:
        if theft:
            lines.append(f"[SHADOW][LOGITECH] 😿 Possible food theft at Sanbo feeder!")
        elif cat_id == "Sanbo" and eating == "yes":
            lines.append(f"[SHADOW][LOGITECH] 😸 Sanbo ate at Sanbo feeder")
        elif cat_id == "Dan" and eating == "yes":
            lines.append(f"[SHADOW][LOGITECH] 😸 Dan ate at Sanbo feeder")
        elif eating == "yes":
            lines.append(f"[SHADOW][LOGITECH] 😸 {cat_id} ate at Sanbo feeder")
        else:
            lines.append(f"[SHADOW][LOGITECH] 🐱 {cat_id} activity at Sanbo feeder")

    start_t = session_data.get("session_start_time", "")
    end_t = session_data.get("session_end_time", "")
    dur_t = session_data.get("total_duration", "")
    lines.append(f"{start_t}–{end_t} · {dur_t}")
    lines.append(f"Confidence: {conf:.2f} · visibility {vis}")
    lines.append(f"Bowl: {bowl}")
    lines.append(f"Evidence: {basis}")

    if theft:
        lines.append("⚠️ Possible food theft")
    if cat_id in ["unsure", "unknown"]:
        lines.append("⚠️ Identity unsure")
    if eating == "unsure":
        lines.append("⚠️ Eating uncertain")
    hand = str(session_data.get("hand_human_interaction", "")).lower()
    if hand and hand not in ["none", "no", "none observed"]:
        lines.append("⚠️ Human interaction")

    return "\n".join(lines)

def format_compact_multi_session_text(all_session_data, all_results, all_failed, all_skipped, shadow_header=True, custom_header=None) -> str:
    """Formats multiple events compactly, suppressing non-meaningful events."""
    meaningful_pairs = []
    for i, sess in enumerate(all_session_data, 1):
        matching_results = [r for r in all_results if r.get("clip_name") in [f"session_{i}", "session", str(i)]]
        res = matching_results[0] if matching_results else (all_results[i-1] if i-1 < len(all_results) else {})
        if is_meaningful_feeding_event(sess, res):
            meaningful_pairs.append((sess, res))

    if not meaningful_pairs:
        return ""

    if len(meaningful_pairs) == 1:
        sess, res = meaningful_pairs[0]
        return format_compact_session_text(sess, res, shadow_header=shadow_header, custom_header=custom_header)

    lines = []
    if custom_header:
        lines.append(custom_header)
    elif shadow_header:
        lines.append("[SHADOW][LOGITECH] Multiple feeding sessions detected")

    for sess, res in meaningful_pairs:
        lines.append("")
        lines.append(format_compact_session_text(sess, res, shadow_header=False))

    return "\n".join(lines).strip()

def check_image_domain(img) -> str:
    if len(img.shape) < 3 or img.shape[2] != 3:
        mean_lum = np.mean(img)
        if mean_lum > 30:
            return 'BRIGHT_GRAYSCALE'
        return 'DARK_GRAYSCALE'
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    diff_rg = np.mean(np.abs(r.astype(int) - g.astype(int)))
    diff_gb = np.mean(np.abs(g.astype(int) - b.astype(int)))
    diff_rb = np.mean(np.abs(r.astype(int) - b.astype(int)))
    max_diff = max(diff_rg, diff_gb, diff_rb)
    if max_diff > 5:
        return 'COLOR'
    mean_lum = np.mean(img)
    if mean_lum > 30:
        return 'BRIGHT_GRAYSCALE'
    return 'DARK_GRAYSCALE'

def validate_vlm_schema(data, expected_date=None, expected_clip_name=None):
    required_fields = [
        "camera", "date", "clip_name", "cat_identity",
        "identity_basis", "visibility",
        "eating_evidence", "bowl_state", "confidence",
        "reasons", "needs_higher_model"
    ]
    for rf in required_fields:
        if rf not in data:
            raise ValueError(f"Missing required field: {rf}")

    if data["camera"] != "LOGITECH":
        raise ValueError(f"Invalid camera: {data['camera']}")
    if expected_date and data["date"] != expected_date:
        raise ValueError(f"Invalid date: expected {expected_date}, got {data['date']}")
    if expected_clip_name:
        allowed_names = [expected_clip_name] if isinstance(expected_clip_name, str) else list(expected_clip_name)
        if data["clip_name"] not in allowed_names:
            raise ValueError(f"Invalid clip_name: expected {expected_clip_name}, got {data['clip_name']}")

    if data["cat_identity"] not in ["Dan", "Sanbo", "both", "none", "unsure"]:
        raise ValueError(f"Invalid cat_identity: {data['cat_identity']}")
    if data["eating_evidence"] not in ["yes", "no", "unsure"]:
        raise ValueError(f"Invalid eating_evidence: {data['eating_evidence']}")
    if data["bowl_state"] not in ["empty", "low", "half", "full", "unsure"]:
        pass

    if not isinstance(data["confidence"], (int, float)):
        raise ValueError(f"Invalid confidence type: {type(data['confidence'])}")
    if not (0.0 <= data["confidence"] <= 1.0):
        raise ValueError(f"Confidence out of range: {data['confidence']}")

    if not isinstance(data["reasons"], list) or not all(isinstance(x, str) for x in data["reasons"]):
        raise ValueError("reasons must be a list of strings")

    if not isinstance(data["needs_higher_model"], bool):
        raise ValueError("needs_higher_model must be a boolean")

def call_openai_vlm(prompt_text, image_paths, model_name, api_key):
    image_path = image_paths[-1] # fallback to just the contact sheet for now
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "response_format": { "type": "json_object" },
        "temperature": 0.0
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    try:
        text_resp = data['choices'][0]['message']['content']
        return json.loads(text_resp)
    except Exception as e:
        raise ValueError(f"Failed to parse OpenAI response: {resp.text}") from e

def call_gemini_vlm(prompt_text, image_paths, model_name, api_key):
    import requests
    if isinstance(api_key, tuple):
        token, project_id = api_key
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model_name}:generateContent"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

    parts = [{"text": prompt_text}]

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_data = f.read()
        b64_img = base64.b64encode(img_data).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64_img
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "response_mime_type": "application/json"
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    try:
        text_resp = data['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_resp)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response: {resp.text}") from e

def main():
    parser = argparse.ArgumentParser(description="Logitech VLM Shadow Scaffold")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-vlm", action="store_true", help="Attempt to run VLM API if key is present")
    parser.add_argument("--confirm-cost", action="store_true", help="Explicitly confirm real VLM API execution costs")
    parser.add_argument("--vlm-provider", type=str, choices=['gemini', 'openai'], help="VLM Provider")
    parser.add_argument("--vlm-model", type=str, help="VLM Model name")
    parser.add_argument("--max-clips", type=int, default=2, help="Max clips to process in VLM API")
    parser.add_argument("--cleanup-downloaded-videos", action="store_true", help="Remove downloaded mp4 files from the out-dir after result generation")
    parser.add_argument("--send-telegram-shadow", action="store_true", help="Send a shadow Telegram report")
    parser.add_argument("--reference-dir", type=str, default=None, help="Path to private reference image directory")
    parser.add_argument("--custom-header", type=str, default=None, help="Custom header text for shadow report")
    args = parser.parse_args()

    if args.date is None:
        print("[STOP] --date is required to avoid date rollover mistakes.")
        sys.exit(1)

    args.date = args.date.replace("-", "")

    if args.out_dir is None:
        args.out_dir = f".agent/artifacts/logitech_vlm_shadow_{args.date}"

    telegram_token = None
    telegram_chat_id = None
    if args.send_telegram_shadow:
        if not args.run_vlm or not args.confirm_cost:
            print("[STOP] --send-telegram-shadow requires --run-vlm and --confirm-cost.")
            sys.exit(1)
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not telegram_token or not telegram_chat_id:
            print("[STOP] Missing required Telegram environment variables (TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID).")
            sys.exit(1)

    if args.run_vlm:
        if not args.confirm_cost:
            print("[STOP] --run-vlm requires --confirm-cost to explicitly acknowledge API charges.")
            sys.exit(1)
        if not args.vlm_provider or not args.vlm_model:
            print("[STOP] --run-vlm requires --vlm-provider and --vlm-model.")
            sys.exit(1)

        if args.vlm_provider == 'gemini':
            gdrive_creds_json = os.environ.get('GDRIVE_SERVICE_ACCOUNT_KEY')
            if gdrive_creds_json:
                try:
                    from google.oauth2 import service_account
                    from google.auth.transport.requests import Request
                    creds_dict = json.loads(gdrive_creds_json)
                    creds = service_account.Credentials.from_service_account_info(
                        creds_dict,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    creds.refresh(Request())
                    api_key = (creds.token, creds_dict.get('project_id'))
                except Exception as e:
                    print(f"[STOP] Failed to obtain Vertex AI token: {e}")
                    sys.exit(1)
            else:
                api_key = os.environ.get('FAIR_FEEDER_GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')
                if not api_key:
                    print("[STOP] Missing GDRIVE_SERVICE_ACCOUNT_KEY or GEMINI_API_KEY for Gemini authentication.")
                    sys.exit(1)
        elif args.vlm_provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("[STOP] Missing required API key. Set OPENAI_API_KEY.")
                sys.exit(1)
        else:
            print(f"[STOP] Unsupported provider: {args.vlm_provider}")
            sys.exit(1)

    if not check_credentials():
        sys.exit(1)

    folder_id = os.environ.get('GDRIVE_LOGITECH_FOLDER_ID')
    if not folder_id:
        print("[STOP] GDRIVE_LOGITECH_FOLDER_ID is missing from environment.")
        sys.exit(1)

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    try:
        key_dict = json.loads(os.environ.get('GDRIVE_SERVICE_ACCOUNT_KEY', '{}'))
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        drive = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"[STOP] Failed to connect to Drive: {sanitize_error_message(str(e))}")
        sys.exit(1)

    out_dir = Path(args.out_dir)

    if out_dir.exists():
        for summary_file in ["summary.json", "logitech_vlm_shadow_summary.json"]:
            s_path = out_dir / summary_file
            if s_path.exists():
                try:
                    with open(s_path, "r") as f:
                        s_data = json.load(f)
                    if s_data.get("date") and s_data.get("date") != args.date:
                        print(f"[STOP] Stale out-dir guard: {summary_file} has date {s_data.get('date')} but --date is {args.date}.")
                        sys.exit(1)
                except Exception:
                    pass
        for mp4_file in out_dir.glob("motion_*.mp4"):
            if f"_{args.date}_" not in mp4_file.name:
                print(f"[STOP] Stale out-dir guard: {mp4_file.name} does not match --date {args.date}.")
                sys.exit(1)

    frames_dir = out_dir / "logitech_vlm_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    generate_vlm_schema(out_dir)

    search_date = args.date.replace("-", "")

    q = f"'{folder_id}' in parents and mimeType='video/mp4' and name contains '{search_date}' and trashed=false"
    results = drive.files().list(pageSize=1000, q=q, fields='files(id, name)').execute()
    all_files = results.get('files', [])

    selected_files = [f for f in all_files if in_feeding_window(f['name'], search_date)]
    selected_files.sort(key=lambda x: x['name'])

    if not selected_files:
        if not all_files:
            print(f"[NO_CLIPS] No mp4 video files found in Drive folder for date {search_date}.")
            print(f"[DIAGNOSTIC] Possible causes: No cat visited the Sanbo feeder, or Pi sync was delayed.")
            print(f"[STATUS] NO FEEDING EVENT recorded for {args.date}. Exiting cleanly.")
        else:
            print(f"[NO_CLIPS] Found {len(all_files)} file(s) for date {search_date}, but none within feeding window 06:18-06:30:")
            for f in all_files:
                print(f"  - {f['name']}")
            print(f"[STATUS] NO FEEDING EVENT within 06:18-06:30 for {args.date}. Exiting cleanly.")

        no_clips_summary = {
            "date": search_date,
            "selected_clip_names": [],
            "session_count": 0,
            "extracted_frames_count": 0,
            "frames_with_motion_count": 0,
            "status": "NO_CLIPS",
            "total_drive_files_for_date": len(all_files),
            "note": "No feeding event detected in window; exited cleanly without VLM invocation"
        }
        with open(out_dir / "logitech_vlm_summary.json", "w") as jf:
            json.dump(no_clips_summary, jf, indent=2)
        sys.exit(0)

    sessions = group_clips_into_sessions(selected_files, gap_threshold_sec=10)
    print(f"ℹ️ {len(selected_files)} clip(s) grouped into {len(sessions)} feeding session(s) (gap threshold: 10s)")

    manifest_data = []
    clip_domains = {}
    session_manifests = []
    session_contact_sheets = []

    for s_idx, session_clips in enumerate(sessions, 1):
        s_name = f"session_{s_idx}" if len(sessions) > 1 else "session"
        session_manifest = []
        contact_sheet_frames = []
        session_frames_by_reason = {}

        for f in session_clips:
            dest_path = out_dir / f['name']
            download_file(drive, f['id'], dest_path)

            cap = cv2.VideoCapture(str(dest_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            bg_frame = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = cap.read()

            final_samples = extract_semantic_keyframes(cap, total_frames, fps, bg_frame)

            m_filename_time = re.search(r'(\d{8})_(\d{6})', f['name'])
            clip_start_time_str = f"{m_filename_time.group(1)} {m_filename_time.group(2)}" if m_filename_time else ""

            for idx, selection_reason in final_samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue

                ts = extract_timestamp_calc(f['name'], idx, fps)
                heuristic_cat = simple_cat_heuristic(frame, bg_frame)

                if f['name'] not in clip_domains:
                    clip_domains[f['name']] = []
                clip_domains[f['name']].append(check_image_domain(frame))

                frame_filename = f"{f['name']}_frame_{idx}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                seconds_from_start = round(idx / fps, 2)

                row = {
                    "session_id": s_name,
                    "clip_name": f['name'],
                    "frame_filename": frame_filename,
                    "timestamp": ts,
                    "frame_index": idx,
                    "motion_detected": heuristic_cat,
                    "selection_reason": selection_reason,
                    "clip_start_time_from_filename": clip_start_time_str,
                    "seconds_from_clip_start": seconds_from_start,
                    "source_drive_file_id": f['id']
                }
                manifest_data.append(row)
                session_manifest.append(row)
                session_frames_by_reason[selection_reason] = {"frame": frame.copy(), "timestamp": ts, "reason": selection_reason}

                # Put timestamp on frame for contact sheet
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Clip: {f['name']}", (10, 30), font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {seconds_from_start}s ({ts})", (10, 60), font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Reason: {selection_reason}", (10, 90), font, 0.7, (0, 255, 0), 2)
                motion_str = "MOTION: YES" if heuristic_cat else "MOTION: NO"
                motion_color = (0, 0, 255) if heuristic_cat else (255, 0, 0)
                cv2.putText(frame, motion_str, (10, 120), font, 0.7, motion_color, 2)

                contact_sheet_frames.append({"frame_data": frame, "name": str(idx)})

            cap.release()

        session_manifests.append(session_manifest)

        # Generate compact before/after comparison image for user
        ba_path = out_dir / f"logitech_vlm_before_after_{s_name}.jpg"
        make_before_after_comparison(session_frames_by_reason, ba_path)
        if s_idx == 1:
            make_before_after_comparison(session_frames_by_reason, out_dir / "logitech_vlm_before_after_session.jpg")

        cs_path = out_dir / f"logitech_vlm_contact_sheet_{s_name}.jpg"
        if contact_sheet_frames:
            if len(contact_sheet_frames) > 16:
                indices = np.linspace(0, len(contact_sheet_frames) - 1, 16, dtype=int)
                sampled_cs_frames = [contact_sheet_frames[i] for i in indices]
            else:
                sampled_cs_frames = contact_sheet_frames
            make_contact_sheet(sampled_cs_frames, cs_path)
            raw_cs_img = cv2.imread(str(cs_path))
            if raw_cs_img is not None:
                enhanced_img = enhance_image_gamma_clahe(raw_cs_img, gamma=2.5)
                enh_cs_path = cs_path.with_name(cs_path.name.replace(".jpg", "_enhanced.jpg"))
                cv2.imwrite(str(enh_cs_path), enhanced_img)
                comp_cs_path = cs_path.with_name(cs_path.name.replace(".jpg", "_comparison.jpg"))
                make_comparison_contact_sheet(cs_path, enh_cs_path, comp_cs_path)

            if s_idx == 1:
                sess_raw = out_dir / "logitech_vlm_contact_sheet_session.jpg"
                make_contact_sheet(sampled_cs_frames, sess_raw)
                sess_raw_img = cv2.imread(str(sess_raw))
                if sess_raw_img is not None:
                    sess_enh_img = enhance_image_gamma_clahe(sess_raw_img, gamma=2.5)
                    sess_enh = out_dir / "logitech_vlm_contact_sheet_session_enhanced.jpg"
                    cv2.imwrite(str(sess_enh), sess_enh_img)
                    sess_comp = out_dir / "logitech_vlm_contact_sheet_session_comparison.jpg"
                    make_comparison_contact_sheet(sess_raw, sess_enh, sess_comp)
            session_contact_sheets.append(cs_path)

        generate_vlm_prompt(out_dir, search_date, session_name=s_name, has_references=bool(args.reference_dir))

    manifest_path = out_dir / "logitech_vlm_manifest.csv"
    with open(manifest_path, "w", newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "session_id", "clip_name", "frame_filename", "timestamp", "frame_index", "motion_detected",
            "selection_reason", "clip_start_time_from_filename", "seconds_from_clip_start", "source_drive_file_id"
        ])
        writer.writeheader()
        writer.writerows(manifest_data)

    summary = {
        "date": search_date,
        "selected_clip_names": [f['name'] for f in selected_files],
        "session_count": len(sessions),
        "extracted_frames_count": len(manifest_data),
        "frames_with_motion_count": sum(1 for row in manifest_data if row["motion_detected"]),
        "schema_path": str(out_dir / "logitech_vlm_expected_schema.json"),
        "note": "prepare-only mode does not call VLM"
    }

    if args.run_vlm:
        print(f"[VLM Shadow] Starting real VLM API execution across {len(sessions)} session(s)...")
        api_calls_made = 0
        all_results = []
        all_failed = []
        all_skipped = []
        all_session_data = []

        clips_requested = len(sessions)
        clips_attempted = 0
        clips_succeeded = 0
        clips_failed = 0
        clips_skipped = 0
        skipped_due_to_api_cap = 0
        had_failures = False
        import time
        import requests

        for s_idx, (session_clips, session_manifest) in enumerate(zip(sessions, session_manifests), 1):
            s_name = f"session_{s_idx}" if len(sessions) > 1 else "session"
            clip_name = s_name
            stem = s_name
            contact_sheet_path = out_dir / f"logitech_vlm_contact_sheet_{s_name}.jpg"
            if not contact_sheet_path.exists():
                contact_sheet_path = out_dir / "logitech_vlm_contact_sheet_session.jpg"
            prompt_path = out_dir / f"logitech_vlm_prompt_{s_name}.md"
            if not prompt_path.exists():
                prompt_path = out_dir / "logitech_vlm_prompt_session.md"

            if not contact_sheet_path.exists() or not prompt_path.exists():
                continue

            prompt_text = prompt_path.read_text()
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()

            if api_calls_made >= MAX_API_CALLS_PER_RUN:
                print(f"[VLM] Reached max API calls per run ({MAX_API_CALLS_PER_RUN}). Skipping {clip_name}.")
                failed_json = {
                    "clip_name": clip_name,
                    "provider": args.vlm_provider,
                    "model": args.vlm_model,
                    "error_type": "ApiCapReached",
                    "error_message": "API call cap reached",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_hash": prompt_hash,
                    "attempts_made": 0
                }
                out_path = out_dir / f"logitech_vlm_result_{stem}.failed.json"
                with open(out_path, "w") as jf:
                    json.dump(failed_json, jf, indent=2)
                all_skipped.append(failed_json)
                clips_skipped += 1
                skipped_due_to_api_cap += 1
                continue

            print(f"[VLM] Processing {clip_name} ({len(session_clips)} clip(s)) with {args.vlm_provider}...")
            clips_attempted += 1

            # Check all domains across this session's clips
            session_domains = []
            for f in session_clips:
                session_domains.extend(clip_domains.get(f['name'], []))

            if session_domains:
                if 'COLOR' in session_domains:
                    pass
                elif all(d == 'BRIGHT_GRAYSCALE' for d in session_domains):
                    print(f"[STOP] RGB/IR domain guard: {clip_name} is BRIGHT_GRAYSCALE. Likely Tapo IR input routed to Logitech. Aborting.")
                    sys.exit(1)
                elif all(d == 'DARK_GRAYSCALE' for d in session_domains):
                    print(f"[WARN] RGB/IR domain guard: {clip_name} is DARK_GRAYSCALE. Proceeding as dark RGB morning.")
                else:
                    print(f"[STOP] RGB/IR domain guard: {clip_name} has mixed grayscale frames. Aborting conservatively.")
                    sys.exit(1)

            attempts = 0
            max_attempts = 2
            success = False
            session_results = []
            session_failed = []
            session_skipped = []

            while attempts < max_attempts:
                attempts += 1
                api_calls_made += 1
                try:
                    image_paths = []
                    ref_metadata = []
                    if args.reference_dir:
                        ref_dir = Path(args.reference_dir)
                        for cat in ['dan', 'sanbo']:
                            cat_dir = ref_dir / cat
                            if cat_dir.exists():
                                for img in sorted(cat_dir.glob("*.jpg")):
                                    image_paths.append(str(img))
                                    with open(img, "rb") as f:
                                        h = hashlib.sha256(f.read()).hexdigest()
                                    ref_metadata.append({"cat": cat, "basename": img.name, "sha256": h})
                    # Bounded Low-Light VLM Preprocessing (Gamma + CLAHE)
                    # The original contact sheet remains available for audit
                    enhanced_path = contact_sheet_path.with_name(contact_sheet_path.name.replace(".jpg", "_enhanced.jpg"))
                    if not enhanced_path.exists():
                        img = cv2.imread(str(contact_sheet_path))
                        if img is not None:
                            gamma = 2.5
                            invGamma = 1.0 / gamma
                            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                            gamma_corrected = cv2.LUT(img, table)
                            lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
                            l_channel, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                            cl = clahe.apply(l_channel)
                            limg = cv2.merge((cl,a,b))
                            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                            cv2.imwrite(str(enhanced_path), enhanced)

                    image_paths.append(str(enhanced_path))

                    if args.vlm_provider == 'openai':
                        result_json = call_openai_vlm(prompt_text, image_paths, args.vlm_model, api_key)
                    elif args.vlm_provider == 'gemini':
                        result_json = call_gemini_vlm(prompt_text, image_paths, args.vlm_model, api_key)
                    else:
                        raise NotImplementedError(f"Provider {args.vlm_provider} not supported.")

                    validate_vlm_schema(result_json, expected_date=search_date, expected_clip_name=[clip_name, "session"])

                    result_json["provider"] = args.vlm_provider
                    result_json["model"] = args.vlm_model
                    result_json["prompt_hash"] = prompt_hash
                    result_json["created_at_utc"] = datetime.now(timezone.utc).isoformat()
                    result_json["source_contact_sheet"] = str(contact_sheet_path.name)
                    result_json["reference_images"] = ref_metadata
                    result_json["raw_response_saved"] = False
                    result_json["attempts_made"] = attempts
                    result_json["session_index"] = s_idx

                    out_path = out_dir / f"logitech_vlm_result_{stem}.json"
                    with open(out_path, "w") as jf:
                        json.dump(result_json, jf, indent=2)
                    all_results.append(result_json)
                    session_results.append(result_json)
                    print(f"[VLM] Success for {clip_name}.")
                    clips_succeeded += 1
                    success = True
                    break

                except Exception as e:
                    should_retry = False
                    if isinstance(e, requests.exceptions.HTTPError):
                        status = e.response.status_code
                        if status in [429, 500, 502, 503, 504]:
                            should_retry = True

                    sanitized_msg = sanitize_error_message(str(e))

                    if attempts < max_attempts and should_retry:
                        if api_calls_made >= MAX_API_CALLS_PER_RUN:
                            print(f"[VLM] Transient error {status} for {clip_name}, but API call cap reached. Skipping retry.")
                            failed_json = {
                                "clip_name": clip_name,
                                "provider": args.vlm_provider,
                                "model": args.vlm_model,
                                "error_type": "ApiCapReached",
                                "error_message": f"API call cap reached. Last error: {sanitized_msg}",
                                "telegram_error_message": "ApiCapReached - API call cap reached.",
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                "prompt_hash": prompt_hash,
                                "attempts_made": attempts
                            }
                            out_path = out_dir / f"logitech_vlm_result_{stem}.failed.json"
                            with open(out_path, "w") as jf:
                                json.dump(failed_json, jf, indent=2)
                            all_skipped.append(failed_json)
                            session_skipped.append(failed_json)
                            clips_skipped += 1
                            skipped_due_to_api_cap += 1
                            break

                        print(f"[VLM] Transient error {status} for {clip_name}. Retrying in 2 seconds...")
                        time.sleep(2)
                        continue

                    print(f"[VLM] Failed for {clip_name}: {sanitized_msg}")

                    telegram_bound_msg = sanitized_msg
                    if "parse" in sanitized_msg.lower() or "schema" in sanitized_msg.lower() or "json" in sanitized_msg.lower() or isinstance(e, ValueError):
                        telegram_bound_msg = f"{type(e).__name__} - provider response could not be parsed; see local artifact"
                    else:
                        if len(telegram_bound_msg) > 200:
                            telegram_bound_msg = telegram_bound_msg[:197] + "..."

                    failed_json = {
                        "clip_name": clip_name,
                        "provider": args.vlm_provider,
                        "model": args.vlm_model,
                        "error_type": type(e).__name__,
                        "error_message": sanitized_msg[:1000],
                        "telegram_error_message": telegram_bound_msg,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "prompt_hash": prompt_hash,
                        "attempts_made": attempts
                    }
                    out_path = out_dir / f"logitech_vlm_result_{stem}.failed.json"
                    with open(out_path, "w") as jf:
                        json.dump(failed_json, jf, indent=2)
                    all_failed.append(failed_json)
                    session_failed.append(failed_json)
                    clips_failed += 1
                    had_failures = True
                    break

            sess_data = build_vlm_session_report(
                session_clips, session_manifest, session_results, session_failed, session_skipped,
                search_date, args.vlm_provider, args.vlm_model
            )
            sess_data["session_index"] = s_idx
            all_session_data.append(sess_data)
            with open(out_dir / f"logitech_vlm_session_{s_idx}_summary.json", "w") as f_s:
                json.dump(sess_data, f_s, indent=2)

        # Save aggregates
        with open(out_dir / "logitech_vlm_results.json", "w") as jf:
            json.dump(all_results, jf, indent=2)

        if all_results:
            keys = all_results[0].keys()
            with open(out_dir / "logitech_vlm_results.csv", "w", newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_results)

        summary["vlm_completed"] = True
        summary["api_calls_made"] = api_calls_made
        summary["api_call_cap"] = MAX_API_CALLS_PER_RUN
        summary["clips_requested"] = clips_requested
        summary["clips_attempted"] = clips_attempted
        summary["clips_succeeded"] = clips_succeeded
        summary["clips_failed"] = clips_failed
        summary["clips_skipped"] = clips_skipped
        summary["skipped_due_to_api_cap"] = skipped_due_to_api_cap
        summary["provider"] = args.vlm_provider
        summary["model"] = args.vlm_model
        summary["production_side_effects"] = "none"
        summary["production_report_changed"] = False
        summary["telegram_sent"] = False
        summary["baseline"] = "no baseline"
        summary["note"] = f"real VLM API execution with {args.vlm_provider}"
        summary["sessions"] = all_session_data
        if all_session_data:
            summary["session"] = all_session_data[0]

        with open(out_dir / "logitech_vlm_shadow_summary.json", "w") as f_sum:
            json.dump(summary, f_sum, indent=2)

        if all_session_data:
            with open(out_dir / "logitech_vlm_session_summary.json", "w") as f_sess:
                json.dump(all_session_data[0], f_sess, indent=2)

        if len(all_results) > 0:
            if len(all_session_data) > 1:
                report_text = format_multi_session_report_text(all_session_data, all_results, all_failed, all_skipped, shadow_header=True, custom_header=args.custom_header)
            else:
                report_text = format_session_report_text(all_session_data[0], all_results, all_failed, all_skipped, shadow_header=True, custom_header=args.custom_header)
        else:
            first_sess = all_session_data[0] if all_session_data else {}
            report_text = format_vlm_failure_report_text(first_sess, all_failed, all_skipped, shadow_header=True)

        (out_dir / "logitech_vlm_shadow_report.md").write_text(report_text)

        # Prepare compact Telegram preview text
        if len(all_results) > 0:
            tg_text = format_compact_multi_session_text(all_session_data, all_results, all_failed, all_skipped, shadow_header=True, custom_header=args.custom_header)
        else:
            first_sess = all_session_data[0] if all_session_data else {}
            tg_text = format_vlm_failure_report_text(first_sess, all_failed, all_skipped, shadow_header=True)

        (out_dir / "logitech_vlm_shadow_telegram_preview.txt").write_text(tg_text)

        if args.send_telegram_shadow:
            send_summary = {
                "telegram_send_attempted": True,
                "telegram_text_sent": False,
                "telegram_images_attempted": 0,
                "telegram_images_sent": 0,
                "telegram_videos_attempted": 0,
                "telegram_videos_sent": 0,
                "attached_media": [],
                "delivery_evidence": [],
                "suppressed_no_feeding": False,
                "is_failure_report": len(all_results) == 0,
                "is_analysis_report": len(all_results) > 0,
                "total_messages_delivered": 0,
                "telegram_error": None,
                "production_report_changed": False,
                "original_report_changed": False,
                "telegram_is_shadow": True,
                "message_starts_with_shadow": tg_text.startswith("[SHADOW]"),
                "telegram_send_fully_successful": False
            }

            # Check if there are meaningful sessions
            meaningful_sessions = []
            for i, sess in enumerate(all_session_data, 1):
                matching_results = [r for r in all_results if r.get("clip_name") in [f"session_{i}", "session", str(i)]]
                res = matching_results[0] if matching_results else (all_results[i-1] if i-1 < len(all_results) else {})
                if is_meaningful_feeding_event(sess, res):
                    meaningful_sessions.append((i, sess, res))

            if len(all_results) > 0 and (not meaningful_sessions or not tg_text.strip()):
                print("[VLM] No meaningful feeding activity detected (e.g. cat=none & eating=no). Suppressing Telegram transmission.")
                send_summary["suppressed_no_feeding"] = True
                send_summary["telegram_send_fully_successful"] = True
                with open(out_dir / "telegram_shadow_send_summary.json", "w") as jf:
                    json.dump(send_summary, jf, indent=2)
            else:
                import requests
                try:
                    # 1. Send text
                    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                    resp = requests.post(url, data={"chat_id": telegram_chat_id, "text": tg_text}, timeout=20)

                    try:
                        r_json = resp.json()
                        send_summary["delivery_evidence"].append({
                            "type": "text",
                            "status": resp.status_code,
                            "ok": r_json.get("ok"),
                            "message_id": r_json.get("result", {}).get("message_id")
                        })
                    except Exception:
                        pass

                    resp.raise_for_status()
                    send_summary["telegram_text_sent"] = True

                    # 2. Send compact before/after images and enhanced videos for meaningful sessions
                    for (i, sess, res) in meaningful_sessions:
                        s_name = f"session_{i}" if len(all_session_data) > 1 else "session"
                        ba_path = out_dir / f"logitech_vlm_before_after_{s_name}.jpg"
                        if not ba_path.exists():
                            ba_path = out_dir / "logitech_vlm_before_after_session.jpg"
                        if not ba_path.exists():
                            ba_path = out_dir / f"logitech_vlm_contact_sheet_{s_name}_comparison.jpg"
                        if not ba_path.exists():
                            ba_path = out_dir / f"logitech_vlm_contact_sheet_{s_name}.jpg"

                        if ba_path.exists():
                            send_summary["telegram_images_attempted"] += 1
                            photo_url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
                            iso_date = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}"
                            caption = f"[SHADOW] {iso_date} {s_name} Pre-feed vs Post-feed (Enhanced)"
                            with open(ba_path, "rb") as photo_f:
                                p_resp = requests.post(photo_url, data={"chat_id": telegram_chat_id, "caption": caption}, files={"photo": photo_f}, timeout=30)
                                try:
                                    pr_json = p_resp.json()
                                    send_summary["delivery_evidence"].append({
                                        "type": "photo",
                                        "status": p_resp.status_code,
                                        "ok": pr_json.get("ok"),
                                        "message_id": pr_json.get("result", {}).get("message_id")
                                    })
                                except Exception:
                                    pass
                                p_resp.raise_for_status()
                                send_summary["telegram_images_sent"] += 1
                                send_summary["attached_media"].append(ba_path.name)

                        # Generate runner-side enhanced video and send
                        session_clips = sessions[i-1] if i-1 < len(sessions) else []
                        if session_clips:
                            raw_clip_path = out_dir / session_clips[0]['name']
                            if raw_clip_path.exists():
                                enh_video_path = out_dir / f"logitech_vlm_{s_name}_enhanced.mp4"
                                if not enh_video_path.exists():
                                    print(f"[VLM] Generating runner-side enhanced video for {raw_clip_path.name}...")
                                    generate_enhanced_video(raw_clip_path, enh_video_path)

                                if enh_video_path.exists() and enh_video_path.stat().st_size > 0:
                                    send_summary["telegram_videos_attempted"] += 1
                                    vid_url = f"https://api.telegram.org/bot{telegram_token}/sendVideo"
                                    vid_caption = f"[SHADOW] {sess.get('session_start_time','')}-{sess.get('session_end_time','')} Enhanced Video"
                                    with open(enh_video_path, "rb") as vid_f:
                                        v_resp = requests.post(vid_url, data={"chat_id": telegram_chat_id, "caption": vid_caption, "supports_streaming": True}, files={"video": vid_f}, timeout=60)
                                        try:
                                            vr_json = v_resp.json()
                                            send_summary["delivery_evidence"].append({
                                                "type": "video",
                                                "status": v_resp.status_code,
                                                "ok": vr_json.get("ok"),
                                                "message_id": vr_json.get("result", {}).get("message_id")
                                            })
                                        except Exception:
                                            pass
                                        v_resp.raise_for_status()
                                        send_summary["telegram_videos_sent"] += 1
                                        send_summary["attached_media"].append(enh_video_path.name)

                    send_summary["total_messages_delivered"] = (1 if send_summary["telegram_text_sent"] else 0) + send_summary["telegram_images_sent"] + send_summary["telegram_videos_sent"]
                    send_summary["telegram_send_fully_successful"] = True

                    # Print delivery evidence for GitHub Actions logs
                    print("[VLM] Telegram delivery evidence:")
                    for ev in send_summary["delivery_evidence"]:
                        print(f"  - type: {ev.get('type')}, status: {ev.get('status')}, ok: {ev.get('ok')}, message_id: {ev.get('message_id')}")

                except Exception as e:
                    send_summary["telegram_error"] = sanitize_error_message(str(e))
                    send_summary["total_messages_delivered"] = (1 if send_summary["telegram_text_sent"] else 0) + send_summary.get("telegram_images_sent", 0) + send_summary.get("telegram_videos_sent", 0)
                    print(f"[VLM] Telegram send error: {send_summary['telegram_error']}")
                    had_failures = True

                with open(out_dir / "telegram_shadow_send_summary.json", "w") as jf:
                    json.dump(send_summary, jf, indent=2)

            # Update logitech_vlm_shadow_summary.json
            summary_path = out_dir / "logitech_vlm_shadow_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f_sum:
                    main_sum = json.load(f_sum)
                main_sum["telegram_sent"] = send_summary["telegram_text_sent"]
                main_sum["telegram_images_sent"] = send_summary["telegram_images_sent"]
                main_sum["telegram_videos_sent"] = send_summary["telegram_videos_sent"]
                main_sum["total_messages_delivered"] = send_summary["total_messages_delivered"]
                main_sum["telegram_error"] = send_summary["telegram_error"]
                main_sum["suppressed_no_feeding"] = send_summary.get("suppressed_no_feeding", False)
                with open(summary_path, "w") as f_sum:
                    json.dump(main_sum, f_sum, indent=2)

        if args.cleanup_downloaded_videos:
            for mp4_file in out_dir.glob("motion_*.mp4"):
                mp4_file.unlink()
                print(f"[VLM] Cleaned up downloaded video: {mp4_file.name}")

        if had_failures:
            sys.exit(1)
    else:
        with open(out_dir / "summary.json", "w") as f_sum:
            json.dump(summary, f_sum, indent=2)
        print(f"✅ Created prepare-only artifacts in {out_dir}")
        print(f"  - logitech_vlm_frames/ (extracted frames)")
        print(f"  - logitech_vlm_contact_sheet_*.jpg (per clip)")
        print(f"  - logitech_vlm_prompt_*.md (per clip)")
        print(f"  - logitech_vlm_manifest.csv")
        print(f"  - logitech_vlm_expected_schema.json")
        print(f"  - summary.json")

if __name__ == "__main__":
    main()
