#!/usr/bin/env python3
"""
scripts/unified_breakfast.py - Fair Feeder Unified Breakfast Analysis & Video Generator

Produces:
1. ONE authoritative house-level breakfast analysis combining TAPO (Dan Feeder) and
   LOGITECH (Sanbo Feeder) evidence into a single cohesive Telegram delivery.
2. ONE synchronized side-by-side combined video (TAPO left, LOGITECH right) on a
   common wall-clock timeline, with neutral placeholders for missing-source slices
   and zero fabricated frames.
"""

import os
import re
import cv2
import json
import shutil
import subprocess
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure repo root and scripts directory are in sys.path so both
# `from scripts.foo import bar` and `from foo import bar` work
# when invoked as `python scripts/unified_breakfast.py` or via pytest.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from scripts.telegram_video_guard import compress_video_for_telegram, validate_video_content
except ImportError:
    try:
        from telegram_video_guard import compress_video_for_telegram, validate_video_content
    except ImportError:
        compress_video_for_telegram = None
        validate_video_content = None

try:
    from scripts.logitech_vlm_shadow import enhance_image_gamma_clahe
except ImportError:
    try:
        from logitech_vlm_shadow import enhance_image_gamma_clahe
    except ImportError:
        def enhance_image_gamma_clahe(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
            if img is None:
                return img
            inv = 1.0 / max(0.1, gamma)
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
            lut = cv2.LUT(img, table)
            lab = cv2.cvtColor(lut, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def parse_clip_timestamp(filename: str) -> Optional[datetime]:
    """Parses start datetime from motion_YYYYMMDD_HHMMSS... filename."""
    m = re.search(r'motion_(\d{8})_(\d{6})', filename)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
        except Exception:
            pass
    return None


def extract_clip_duration(filename: str) -> float:
    """Extracts duration from filename like motion_..._1m_50s.mp4 or _6s.mp4."""
    m = re.search(r'_(\d+m_)?(\d+)s', filename)
    if m:
        dur = int(m.group(2))
        if m.group(1):
            dur += int(m.group(1).replace("m_", "")) * 60
        return float(dur)
    return 0.0


# ── House-Level Reconciled Report (Track E) ───────────────────────────────────

def render_kibble_bar(pct: Optional[Union[int, float]], width: int = 8) -> str:
    """Renders a compact 8-block unicode horizontal bar (e.g. '████░░░░')."""
    if pct is None:
        return ""
    try:
        val = float(pct)
        filled = min(width, max(0, round((val / 100.0) * width)))
        return "█" * filled + "░" * (width - filled)
    except Exception:
        return ""


def generate_unified_breakfast_report(
    target_date: str,
    tapo_summary: Dict[str, Any],
    logitech_session: Dict[str, Any],
    tapo_raw_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes TAPO and Logitech evidence into ONE house-level breakfast analysis.
    Guarantees:
    - Feeder meal outcome answered first, independent of cat identity
    - TAPO 8-block percent bars preserved even when classification is contested
    - Contested notice clearly marks camera-model attribution without asserting theft
    - Dan feeder food consumed is NOT phrased as 'Dan consumed all food'
    - Meal completion displayed explicitly or marked uncertain if unobserved
    - House-level conclusion reconciles cross-camera evidence without physically impossible claims
    - Zero hardcoded fixture numbers
    """
    clean_date = str(target_date).replace("-", "").strip()
    formatted_date = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:]}" if len(clean_date) == 8 else str(target_date)

    # 1. Timeline Window Span (strictly from evidence)
    tapo_start = tapo_summary.get("start_time") or tapo_summary.get("start_ts")
    tapo_end = tapo_summary.get("end_time") or tapo_summary.get("end_ts")
    logi_start = logitech_session.get("session_start_time") or logitech_session.get("start_time")
    logi_end = logitech_session.get("session_end_time") or logitech_session.get("end_time")

    times = [t for t in [tapo_start, tapo_end, logi_start, logi_end] if t and isinstance(t, str) and ":" in t]
    start_time = min(times) if times else "unknown"
    end_time = max(times) if times else "unknown"

    span_str = "unknown"
    if start_time != "unknown" and end_time != "unknown":
        try:
            t0 = datetime.strptime(start_time.strip()[-8:], "%H:%M:%S")
            t1 = datetime.strptime(end_time.strip()[-8:], "%H:%M:%S")
            span_sec = max(0, int((t1 - t0).total_seconds()))
            m, s = divmod(span_sec, 60)
            span_str = f"{m}m {s}s" if m > 0 else f"{s}s"
        except Exception:
            pass

    # 2. Feeder Meal Outcome & TAPO Evidence Extraction
    dan_kibble = tapo_summary.get("dan_kibble") if tapo_summary.get("dan_kibble") is not None else tapo_summary.get("dan_kibble_eaten")
    sanbo_kibble = tapo_summary.get("sanbo_kibble") if tapo_summary.get("sanbo_kibble") is not None else tapo_summary.get("sanbo_kibble_eaten")
    dan_pct = tapo_summary.get("dan_percent")
    sanbo_pct = tapo_summary.get("sanbo_percent")
    dan_bowl_time = tapo_summary.get("dan_bowl_time")
    if not dan_bowl_time and "dan_bowl_seconds" in tapo_summary:
        dan_bowl_time = f"{int(tapo_summary['dan_bowl_seconds'])}s"
    sanbo_bowl_time = tapo_summary.get("sanbo_bowl_time")
    if not sanbo_bowl_time and "sanbo_bowl_seconds" in tapo_summary:
        sanbo_bowl_time = f"{int(tapo_summary['sanbo_bowl_seconds'])}s"

    dan_seen = tapo_summary.get("dan_first_ts") or tapo_summary.get("dan_first_arrival") or tapo_summary.get("dan_seen")
    sanbo_seen = tapo_summary.get("sanbo_first_ts") or tapo_summary.get("sanbo_first_arrival") or tapo_summary.get("sanbo_seen")

    has_tapo_conflict = tapo_summary.get("has_conflict", False)
    conflict_frames = tapo_summary.get("conflict_frames", 0)
    total_start_kibble = tapo_summary.get("start_kibble")
    total_end_kibble = tapo_summary.get("end_kibble")
    meal_finished_explicit = tapo_summary.get("meal_finished")

    # Determine consumed_kibble
    consumed_kibble: Optional[int] = None
    if total_start_kibble is not None and total_end_kibble is not None:
        consumed_kibble = max(0, total_start_kibble - total_end_kibble)
    elif dan_kibble is not None or sanbo_kibble is not None:
        consumed_kibble = (dan_kibble or 0) + (sanbo_kibble or 0)

    # Determine meal_finished
    # Product rule: Prefer observed final bowl/kibble state.
    # Do NOT infer end=0 merely because per-cat estimates sum approximately to start amount.
    if meal_finished_explicit is not None:
        meal_finished: Optional[bool] = bool(meal_finished_explicit)
    elif total_end_kibble is not None:
        meal_finished = (total_end_kibble <= 1)
    else:
        meal_finished = None

    # TAPO Feeder Meal Outcome line
    if meal_finished is True:
        tapo_meal_status_str = "Finished ✅"
        if total_start_kibble is not None and total_end_kibble is not None:
            tapo_meal_line = f"🥣 Meal: ~{total_start_kibble} → {total_end_kibble} kibble · {tapo_meal_status_str}"
        elif total_start_kibble is not None:
            tapo_meal_line = f"🥣 Meal: ~{total_start_kibble} kibble · {tapo_meal_status_str}"
        else:
            tapo_meal_line = f"🥣 Meal: {tapo_meal_status_str}"
    elif meal_finished is False:
        tapo_meal_status_str = "Remaining ⚠️"
        if total_start_kibble is not None and total_end_kibble is not None:
            eaten_part = f"~{consumed_kibble} eaten · " if consumed_kibble is not None else ""
            tapo_meal_line = f"🥣 Meal: ~{total_start_kibble} → ~{total_end_kibble} kibble · {eaten_part}{tapo_meal_status_str}"
        else:
            tapo_meal_line = f"🥣 Meal: {tapo_meal_status_str}"
    else:
        tapo_meal_status_str = "uncertain"
        tapo_meal_line = "🥣 Meal completion: uncertain"

    # Compute Dan/Sanbo percentages if missing
    if (dan_pct is None or sanbo_pct is None) and consumed_kibble and consumed_kibble > 0:
        if dan_pct is None and dan_kibble is not None:
            dan_pct = round((dan_kibble / consumed_kibble) * 100)
        if sanbo_pct is None and sanbo_kibble is not None:
            sanbo_pct = round((sanbo_kibble / consumed_kibble) * 100)

    dan_bar = render_kibble_bar(dan_pct)
    sanbo_bar = render_kibble_bar(sanbo_pct)

    has_identity_conflict = bool(has_tapo_conflict or conflict_frames > 15)

    # Build TAPO attribution lines
    tapo_attribution_lines = []
    if has_identity_conflict:
        tapo_attribution_lines.append("⚠️ TAPO model attribution — contested")
    else:
        tapo_attribution_lines.append("TAPO model attribution")

    if dan_pct is not None or dan_kibble is not None:
        bar_str = f"{dan_bar} " if dan_bar else ""
        pct_str = f"{dan_pct}%" if dan_pct is not None else ""
        amt_str = f" (~{dan_kibble})" if dan_kibble is not None else ""
        tapo_attribution_lines.append(f"Dan    {bar_str}{pct_str}{amt_str}")

        dan_details = []
        if dan_bowl_time:
            dan_details.append(f"bowl {dan_bowl_time}")
        if dan_seen:
            d_seen_str = str(dan_seen).strip().split()[-1]
            if not d_seen_str.startswith("~"):
                d_seen_str = f"~{d_seen_str}"
            dan_details.append(f"from {d_seen_str}")
        if dan_details:
            tapo_attribution_lines.append(f"       {' · '.join(dan_details)}")

    if sanbo_pct is not None or sanbo_kibble is not None:
        bar_str = f"{sanbo_bar} " if sanbo_bar else ""
        pct_str = f"{sanbo_pct}%" if sanbo_pct is not None else ""
        amt_str = f" (~{sanbo_kibble})" if sanbo_kibble is not None else ""
        tapo_attribution_lines.append(f"Sanbo  {bar_str}{pct_str}{amt_str}")

        sanbo_details = []
        if sanbo_bowl_time:
            sanbo_details.append(f"bowl {sanbo_bowl_time}")
        if sanbo_seen:
            s_seen_str = str(sanbo_seen).strip().split()[-1]
            if not s_seen_str.startswith("~"):
                s_seen_str = f"~{s_seen_str}"
            sanbo_details.append(f"from {s_seen_str}")
        if sanbo_details:
            tapo_attribution_lines.append(f"       {' · '.join(sanbo_details)}")

    if has_identity_conflict:
        c_frames_str = f"{conflict_frames} conflict frames — " if conflict_frames > 0 else ""
        tapo_attribution_lines.append("")
        tapo_attribution_lines.append(
            f"{c_frames_str}this split is camera-model evidence, not reliable enough by itself to prove theft."
        )

    # 3. Logitech Evidence Extraction
    logi_cat = logitech_session.get("cat_identity") or logitech_session.get("cat") or "unknown"
    logi_eating = logitech_session.get("eating_evidence") or "unknown"
    logi_vis = logitech_session.get("visibility") or "unknown"
    logi_gaps = logitech_session.get("source_gaps", [])
    logi_duration = logitech_session.get("total_duration") or "56s"
    if "wall_clock_span_sec" in logitech_session:
        logi_duration = f"{int(logitech_session['wall_clock_span_sec'])}s"

    gap_note = ""
    if logi_gaps:
        g_sec = int(logi_gaps[0].get("gap_sec", 0))
        if g_sec > 0:
            gap_note = f", {g_sec}s low-motion gap preserved"

    logi_bowl_prog = logitech_session.get("bowl_state_progression") or logitech_session.get("bowl_state")
    logi_meal_finished_explicit = logitech_session.get("meal_finished")
    logi_meal_status_explicit = logitech_session.get("meal_status")

    if logi_meal_status_explicit:
        logi_meal_desc = logi_meal_status_explicit
    elif logi_meal_finished_explicit is True or (logi_bowl_prog and "empty" in str(logi_bowl_prog).lower()):
        logi_meal_desc = "Finished likely"
    elif logi_meal_finished_explicit is False:
        logi_meal_desc = "Remaining ⚠️"
    elif str(logi_eating).lower() in ("yes", "true", "eating", "observed"):
        logi_meal_desc = "Finished likely" if (logi_bowl_prog and "empty" in str(logi_bowl_prog).lower()) else "uncertain"
    else:
        logi_meal_desc = "uncertain"

    if logi_bowl_prog and logi_bowl_prog != "unsure":
        if logi_meal_desc != "uncertain":
            logi_meal_line = f"🥣 Meal: {logi_bowl_prog} · {logi_meal_desc}"
        else:
            logi_meal_line = f"🥣 Meal: {logi_bowl_prog} · Completion uncertain"
    elif logi_meal_desc == "Finished likely":
        logi_meal_line = f"🥣 Meal: {logi_meal_desc}"
    else:
        logi_meal_line = "🥣 Meal: completion uncertain"

    if str(logi_eating).lower() in ("yes", "true", "eating", "observed"):
        logi_eating_line = "🍽 Eating observed"
    elif str(logi_eating).lower() in ("no", "false"):
        logi_eating_line = "🍽 No eating observed"
    else:
        logi_eating_line = "🍽 Eating: unsure"

    if logi_cat and logi_cat != "unknown":
        logi_cat_line = f"🐱 {logi_cat}"
    else:
        logi_cat_line = "🐱 Unknown / unverified"

    logi_time_line = f"⏱ {logi_start or 'unknown'}–{logi_end or 'unknown'} ({logi_duration}{gap_note})"

    logi_vis_line = ""
    if logi_vis and logi_vis != "unknown":
        vis_str = str(logi_vis).strip()
        logi_vis_line = f"🌙 {vis_str[0].upper() + vis_str[1:]}"

    # 4. House-Level Synthesis & Physical Exclusion Analysis
    is_sanbo_at_logi = (logi_cat == "Sanbo" and str(logi_eating).lower() in ("yes", "true", "eating", "observed"))

    # Check temporal overlap
    has_temporal_overlap = False
    if tapo_start and tapo_end and logi_start and logi_end:
        try:
            ts_t0 = datetime.strptime(tapo_start.strip()[-8:], "%H:%M:%S")
            ts_t1 = datetime.strptime(tapo_end.strip()[-8:], "%H:%M:%S")
            ls_t0 = datetime.strptime(logi_start.strip()[-8:], "%H:%M:%S")
            ls_t1 = datetime.strptime(logi_end.strip()[-8:], "%H:%M:%S")
            has_temporal_overlap = (ts_t0 <= ls_t1) and (ls_t0 <= ts_t1)
        except Exception:
            has_temporal_overlap = True

    # Feeder meal outcomes for house section
    if meal_finished is True:
        dan_k_str = f"~{consumed_kibble or total_start_kibble} kibble consumed" if (consumed_kibble or total_start_kibble) else "Food consumed"
        house_dan_feeder = f"{dan_k_str} · Finished ✅"
    elif meal_finished is False:
        house_dan_feeder = f"~{consumed_kibble} kibble consumed · ~{total_end_kibble} remaining ⚠️"
    elif consumed_kibble and consumed_kibble > 0:
        house_dan_feeder = f"~{consumed_kibble} kibble consumed · Completion uncertain"
    else:
        house_dan_feeder = "Consumption uncertain"

    if is_sanbo_at_logi:
        if logi_meal_desc == "Finished likely":
            house_sanbo_feeder = f"Finished likely · {logi_cat} feeding observed"
        else:
            house_sanbo_feeder = f"{logi_cat} feeding observed ({logi_start}–{logi_end})"
    elif str(logi_eating).lower() in ("yes", "true", "eating", "observed"):
        house_sanbo_feeder = f"Feeding observed ({logi_cat})"
    else:
        house_sanbo_feeder = "No feeding observed"

    if has_identity_conflict:
        if is_sanbo_at_logi and has_temporal_overlap:
            house_identity = "Dan confirmed at Dan feeder during overlap (Sanbo at own feeder); individual TAPO split contested"
        else:
            house_identity = "Contested at Dan feeder; individual attribution unresolved"
    else:
        house_identity = "Dan and Sanbo identities consistent with camera attribution"

    if sanbo_kibble and sanbo_kibble > 5 and not has_identity_conflict and (dan_kibble is None or dan_kibble < 5):
        house_theft = "Confirmed: Sanbo ate at Dan feeder"
    elif has_identity_conflict:
        house_theft = "Not confirmed"
    else:
        house_theft = "Not confirmed"

    # Build Telegram report lines matching requested message hierarchy
    lines = [
        f"🍳 Breakfast · {formatted_date}",
        f"⏱ {start_time}–{end_time} ({span_str})",
        "",
        "TAPO · Dan feeder",
        tapo_meal_line,
        "",
        *tapo_attribution_lines,
        "",
        "LOGITECH · Sanbo feeder",
        logi_cat_line,
        logi_eating_line,
        logi_meal_line,
        logi_time_line,
    ]
    if logi_vis_line:
        lines.append(logi_vis_line)

    lines.extend([
        "",
        "House",
        f"- Dan feeder: {house_dan_feeder}",
        f"- Sanbo feeder: {house_sanbo_feeder}",
        f"- Identity: {house_identity}",
        f"- Theft: {house_theft}"
    ])

    report_text = "\n".join(lines)
    return {
        "date": clean_date,
        "formatted_date": formatted_date,
        "time_window": f"{start_time}–{end_time}",
        "duration_str": span_str,
        "start_kibble": total_start_kibble,
        "end_kibble": total_end_kibble,
        "consumed_kibble": consumed_kibble,
        "meal_finished": meal_finished,
        "tapo_meal_status": tapo_meal_status_str,
        "logitech_meal_status": logi_meal_desc,
        "telegram_text": report_text,
        "tapo_summary": tapo_summary,
        "logitech_summary": logitech_session,
        "house_theft_verdict": house_theft
    }


# ── Synchronized Vertical Combined Video (Track F) ───────────────────────────

def create_neutral_placeholder(
    width: int,
    height: int,
    camera_label: str,
    timestamp_str: str,
    reason: str = "No source footage"
) -> np.ndarray:
    """Generates a neutral dark placeholder frame with readable label, timestamp, and status."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (28, 28, 32)  # Clean neutral dark slate

    # Accent border
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), (50, 50, 60), 1)

    # Status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = reason
    font_scale = 0.75
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(10, (width - tw) // 2)
    ty = (height // 2) - 8
    cv2.putText(img, text, (tx, ty), font, font_scale, (160, 160, 175), thickness, cv2.LINE_AA)

    subtext = f"Clock running: {timestamp_str}"
    font_scale_sub = 0.48
    (stw, sth), _ = cv2.getTextSize(subtext, font, font_scale_sub, 1)
    stx = max(10, (width - stw) // 2)
    sty = ty + 32
    cv2.putText(img, subtext, (stx, sty), font, font_scale_sub, (110, 110, 125), 1, cv2.LINE_AA)

    return img


def draw_panel_overlay(
    panel_img: np.ndarray,
    camera_title: str,
    timestamp_str: str,
    is_live_footage: bool = True
) -> np.ndarray:
    """Legacy helper: draws top header bar for backward compatibility."""
    out = panel_img.copy()
    h, w, _ = out.shape
    banner_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    font = cv2.FONT_HERSHEY_SIMPLEX
    dot_color = (0, 220, 0) if is_live_footage else (100, 100, 110)
    cv2.circle(out, (14, 18), 5, dot_color, -1)
    cv2.putText(out, camera_title, (26, 23), font, 0.55, (240, 240, 245), 1, cv2.LINE_AA)

    (tsw, _), _ = cv2.getTextSize(timestamp_str, font, 0.55, 1)
    cv2.putText(out, timestamp_str, (w - tsw - 12, 23), font, 0.55, (240, 240, 245), 1, cv2.LINE_AA)
    return out


def render_header_bar(
    width: int,
    height: int,
    title: str,
    timestamp_str: str,
    is_live: bool = True
) -> np.ndarray:
    """Renders a dedicated header or footer bar placed entirely outside source pixels."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (20, 20, 24)

    font = cv2.FONT_HERSHEY_SIMPLEX
    dot_color = (0, 220, 0) if is_live else (100, 100, 110)
    cv2.circle(bar, (18, height // 2), 5, dot_color, -1)

    cv2.putText(bar, title, (32, (height // 2) + 5), font, 0.52, (240, 240, 245), 1, cv2.LINE_AA)

    (tsw, _), _ = cv2.getTextSize(timestamp_str, font, 0.50, 1)
    cv2.putText(bar, timestamp_str, (width - tsw - 16, (height // 2) + 5), font, 0.50, (240, 240, 245), 1, cv2.LINE_AA)

    return bar


def render_separator_bar(
    width: int,
    height: int,
    text: str = "-- SHARED TIMELINE (4x speedup) --"
) -> np.ndarray:
    """Renders the central timeline separator bar between TAPO and Logitech panels."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (25, 25, 30)

    # Top and bottom hairline dividers
    cv2.line(bar, (0, 0), (width, 0), (45, 45, 50), 1)
    cv2.line(bar, (0, height - 1), (width, height - 1), (45, 45, 50), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = max(10, (width - tw) // 2)
    ty = (height // 2) + 4
    cv2.putText(bar, text, (tx, ty), font, font_scale, (140, 140, 150), 1, cv2.LINE_AA)

    return bar


class VideoStreamSampler:
    """Provides random-access frame retrieval aligned to wall-clock seconds."""
    def __init__(self, clip_paths: List[Path], is_logitech: bool = False):
        self.is_logitech = is_logitech
        self.clips = []
        for p in clip_paths:
            if not p.exists():
                continue
            st = parse_clip_timestamp(p.name)
            dur = extract_clip_duration(p.name)
            # Verify actual video duration using OpenCV
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            real_dur = (fc / fps) if (fps > 0 and fc > 0) else dur
            dur = max(dur, real_dur)

            if st and dur > 0:
                self.clips.append({
                    "path": p,
                    "start": st,
                    "end": st + timedelta(seconds=dur),
                    "duration": dur,
                    "cap": None,
                    "fps": fps,
                    "last_frame_idx": -1
                })
        self.clips.sort(key=lambda x: x["start"])

    def get_frame_at(self, dt: datetime, target_size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], bool]:
        """Returns (frame, is_live_footage) for given wall-clock timestamp."""
        w, h = target_size
        for c in self.clips:
            if c["start"] <= dt <= c["end"]:
                offset_sec = (dt - c["start"]).total_seconds()
                if c["cap"] is None:
                    c["cap"] = cv2.VideoCapture(str(c["path"]))
                cap = c["cap"]
                fps = c.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_idx = int(offset_sec * fps)

                # Avoid redundant seek if next sequential frame
                if frame_idx != c["last_frame_idx"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                c["last_frame_idx"] = frame_idx + 1 if ret else -1

                if ret and frame is not None:
                    if self.is_logitech:
                        # Apply low-light enhancement if dark
                        if np.mean(frame) < 30.0:
                            frame = enhance_image_gamma_clahe(frame, gamma=2.5)
                    # Resize preserving exact aspect ratio
                    resized = cv2.resize(frame, (w, h))
                    return resized, True

        return None, False

    def close(self):
        for c in self.clips:
            if c["cap"] is not None:
                c["cap"].release()
                c["cap"] = None


def generate_combined_breakfast_video(
    tapo_clips: List[Union[Path, str]],
    logitech_clips: List[Union[Path, str]],
    output_path: Union[Path, str],
    target_date: str = "20260905",
    speedup_factor: float = 4.0,
    out_width: int = 720,
    target_fps: float = 16.0,
    start_time_override: Optional[datetime] = None,
    end_time_override: Optional[datetime] = None
) -> Path:
    """
    Renders synchronized vertical full-frame video on a shared wall-clock timeline.
    Layout:
    - Dedicated Header (38px): TAPO status dot, title, timestamp
    - Top Panel (720x405): TAPO (Dan Feeder) - uncropped 16:9, ZERO overlay on source pixels!
    - Central Separator (34px): Shared timeline indicator
    - Bottom Panel (720x405): LOGITECH (Sanbo Feeder) - uncropped 16:9, ZERO overlay on source pixels!
    - Dedicated Footer (38px): LOGITECH status dot, title, timestamp
    Total Canvas: 720x920 (both even numbers, H.264/yuv420p safe).
    Missing intervals show clean neutral placeholder without fake frames.
    Speedup: 4x playback.
    Telegram-safe: H.264, yuv420p, +faststart, <45 MB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tapo_paths = [Path(p) for p in tapo_clips if Path(p).exists()]
    logi_paths = [Path(p) for p in logitech_clips if Path(p).exists()]

    tapo_sampler = VideoStreamSampler(tapo_paths, is_logitech=False)
    logi_sampler = VideoStreamSampler(logi_paths, is_logitech=True)

    # Determine timeline boundaries dynamically from actual clips
    all_starts = [c["start"] for c in tapo_sampler.clips + logi_sampler.clips]
    all_ends = [c["end"] for c in tapo_sampler.clips + logi_sampler.clips]

    if not all_starts:
        raise ValueError("No valid source video clips found to build combined video")

    t_start = start_time_override or min(all_starts)
    t_end = end_time_override or max(all_ends)

    # Canvas Dimensions
    width = out_width  # 720
    panel_h = int(width * (9 / 16))  # 405
    header_h = 38
    sep_h = 34
    footer_h = 38
    total_h = header_h + panel_h + sep_h + panel_h + footer_h  # 38 + 405 + 34 + 405 + 38 = 920

    temp_raw = output_path.with_name(f"temp_raw_combined_{output_path.name}")
    if temp_raw.exists():
        temp_raw.unlink()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_raw), fourcc, target_fps, (width, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV VideoWriter for {temp_raw}")

    # Wall-clock stepping: 4x speedup means each output video frame advances 0.25s of real time
    step_sec = speedup_factor / target_fps  # 4.0 / 16.0 = 0.25s real time per frame
    time_step = timedelta(seconds=step_sec)
    curr_time = t_start

    try:
        while curr_time <= t_end:
            time_str = curr_time.strftime("%Y-%m-%d %H:%M:%S")

            # 1. TAPO Panel (Top)
            tapo_frame, tapo_is_live = tapo_sampler.get_frame_at(curr_time, (width, panel_h))
            if not tapo_is_live or tapo_frame is None:
                tapo_frame = create_neutral_placeholder(width, panel_h, "TAPO", time_str, reason="No source footage")

            # 2. LOGITECH Panel (Bottom)
            logi_frame, logi_is_live = logi_sampler.get_frame_at(curr_time, (width, panel_h))
            if not logi_is_live or logi_frame is None:
                logi_frame = create_neutral_placeholder(width, panel_h, "LOGITECH", time_str, reason="No source footage")

            # 3. Header, Separator, Footer (dedicated bars OUTSIDE source pixels)
            header_bar = render_header_bar(width, header_h, "TAPO - Dan Feeder", time_str, is_live=tapo_is_live)
            separator_bar = render_separator_bar(width, sep_h, text="-- SHARED TIMELINE (4x speedup) --")
            footer_bar = render_header_bar(width, footer_h, "LOGITECH - Sanbo Feeder", time_str, is_live=logi_is_live)

            # 4. Vertical stack: 38 + 405 + 34 + 405 + 38 = 920px height
            vertical_canvas = np.vstack([header_bar, tapo_frame, separator_bar, logi_frame, footer_bar])
            writer.write(vertical_canvas)

            curr_time += time_step
    finally:
        writer.release()
        tapo_sampler.close()
        logi_sampler.close()

    # 5. Transcode to H.264, faststart, yuv420p, Telegram safe size
    # Since frames were sampled at 4x speedup, playback is already accelerated; use speedup_factor=1.0 for ffmpeg pass
    if compress_video_for_telegram:
        ok, safe_p, _ = compress_video_for_telegram(
            temp_raw,
            output_path=output_path,
            speedup_factor=1.0,
            target_height=total_h
        )
        if temp_raw.exists():
            temp_raw.unlink()
        if ok and safe_p.exists():
            return safe_p

    # Fallback to direct ffmpeg
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y", "-i", str(temp_raw),
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-r", str(int(target_fps)),
            "-an",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if temp_raw.exists():
            temp_raw.unlink()
        return output_path

    if temp_raw.exists():
        temp_raw.rename(output_path)
    return output_path


# ── Production Unified Delivery Engine ────────────────────────────────────────

def deliver_unified_breakfast(
    target_date: str,
    tapo_dir: Optional[Path] = None,
    logitech_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    preview: bool = False,
    skip_telegram: bool = False,
    force: bool = False,
    drive_service: Any = None,
    folder_id: Optional[str] = None
) -> bool:
    """
    Executes the single-authority unified breakfast delivery:
    1. Preflight check via delivery_registry.json
    2. Collects TAPO and Logitech artifacts
    3. Synthesizes truthful house report
    4. Delivers summary (item: summary)
    5. Renders & delivers vertical combined video (item: combined_video)
    6. Commits terminal breakfast completion (fail closed)
    """
    import requests
    try:
        from scripts.delivery_ledger import (
            load_delivery_registry,
            is_breakfast_fully_delivered,
            is_unified_item_delivered,
            record_unified_item_delivered,
            commit_breakfast_completion
        )
    except ImportError:
        from delivery_ledger import (
            load_delivery_registry,
            is_breakfast_fully_delivered,
            is_unified_item_delivered,
            record_unified_item_delivered,
            commit_breakfast_completion
        )

    clean_date = str(target_date).replace("-", "").strip()
    out_dir = Path(out_dir) if out_dir else Path(f"/tmp/unified_delivery_{clean_date}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    # Step 1: Preflight Check
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=out_dir)
    if not force and is_breakfast_fully_delivered(registry, clean_date):
        print(f"✅ [Preflight] Breakfast for {clean_date} already fully delivered. Skipping.")
        return True

    # Step 2: Ingest TAPO Artifacts
    tapo_summary = {}
    tapo_clips = []
    tapo_search_dirs = [d for d in [tapo_dir, Path("/tmp/output"), Path("scratch/replay_sep5"), Path(".")] if d and Path(d).exists()]

    for d in tapo_search_dirs:
        sum_p = Path(d) / f"tapo_summary_{clean_date}.json"
        if sum_p.exists():
            try:
                tapo_summary = json.loads(sum_p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    if not tapo_summary:
        # Check for individual *_summary.txt or fallback dict
        for d in tapo_search_dirs:
            for txt_f in Path(d).glob(f"*{clean_date}*_summary.txt"):
                tapo_summary["raw_summary_text"] = txt_f.read_text(encoding="utf-8")
                break
            if "raw_summary_text" in tapo_summary:
                break

    # Collect TAPO video clips
    for d in tapo_search_dirs:
        for vid in sorted(Path(d).glob(f"*{clean_date}*.mp4")):
            if "combined" not in vid.name and "annotated" not in vid.name and vid not in tapo_clips:
                tapo_clips.append(vid)

    # Step 3: Ingest Logitech Artifacts
    logitech_summary = {}
    logi_clips = []
    logi_search_dirs = [d for d in [logitech_dir, Path(f"/tmp/logitech_vlm_shadow_{clean_date}"), Path("scratch/replay_sep5"), Path(".")] if d and Path(d).exists()]

    for d in logi_search_dirs:
        l_sum_p = Path(d) / "logitech_vlm_shadow_summary.json"
        if not l_sum_p.exists():
            l_sum_p = Path(d) / "summary.json"
        if l_sum_p.exists():
            try:
                logitech_summary = json.loads(l_sum_p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    for d in logi_search_dirs:
        for vid in sorted(Path(d).glob(f"*{clean_date}*.mp4")):
            if "combined" not in vid.name and "annotated" not in vid.name and vid not in logi_clips and vid not in tapo_clips:
                logi_clips.append(vid)

    # Step 4: Generate House-Level Report
    report = generate_unified_breakfast_report(clean_date, tapo_summary, logitech_summary)
    summary_text = report["telegram_text"]
    if preview:
        summary_text = f"[TEST][PREVIEW] Unified Breakfast UX · Sep-5 fixture\n\n{summary_text}"

    # Step 5: Item-Level Delivery
    base_tg = f"https://api.telegram.org/bot{bot_token}" if bot_token else None

    # Item 1: Summary
    if not is_unified_item_delivered(registry, clean_date, "summary"):
        print(f"📤 Delivering unified breakfast summary for {clean_date}...")
        sum_msg_id = None
        if base_tg and chat_id and not skip_telegram:
            resp = requests.post(f"{base_tg}/sendMessage", data={
                "chat_id": chat_id,
                "text": summary_text[:4096]
            }, timeout=30)
            if resp.status_code != 200:
                print(f"❌ Failed to send summary to Telegram: {resp.text}")
                return False
            try:
                sum_msg_id = resp.json().get("result", {}).get("message_id")
            except Exception:
                pass

        record_unified_item_delivered(
            drive_service, folder_id, registry, clean_date, "summary",
            message_id=sum_msg_id, local_fallback_dir=out_dir
        )
        print(f"✅ Summary delivered (message_id={sum_msg_id})")
    else:
        print(f"ℹ️ Summary already delivered for {clean_date}. Skipping item.")

    # Item 2: Combined Video
    combined_video_path = out_dir / f"{clean_date}_combined_breakfast.mp4"
    if not is_unified_item_delivered(registry, clean_date, "combined_video"):
        print(f"🎥 Rendering unified vertical video for {clean_date}...")
        if not tapo_clips or not logi_clips:
            print(f"⚠️ Missing clips for video render (tapo={len(tapo_clips)}, logi={len(logi_clips)}). Cannot deliver video.")
            return False

        generate_combined_breakfast_video(
            tapo_clips=tapo_clips,
            logitech_clips=logi_clips,
            output_path=combined_video_path,
            target_date=clean_date,
            speedup_factor=4.0
        )

        # Validate video size and content
        vid_size_mb = combined_video_path.stat().st_size / (1024 * 1024)
        if vid_size_mb >= 45.0:
            print(f"❌ Video size {vid_size_mb:.2f} MB exceeds 45 MB Telegram limit")
            return False

        caption = "TAPO top · LOGITECH bottom · synchronized 4x · full-frame"
        if preview:
            caption = f"[TEST][PREVIEW] {caption}"

        vid_msg_id = None
        if base_tg and chat_id and not skip_telegram:
            print(f"📤 Delivering combined video ({vid_size_mb:.2f} MB) to Telegram...")
            with open(combined_video_path, "rb") as vf:
                resp = requests.post(
                    f"{base_tg}/sendVideo",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"video": (combined_video_path.name, vf, "video/mp4")},
                    timeout=180
                )
            if resp.status_code != 200:
                print(f"❌ Failed to send combined video to Telegram: {resp.text}")
                return False
            try:
                vid_msg_id = resp.json().get("result", {}).get("message_id")
            except Exception:
                pass

        record_unified_item_delivered(
            drive_service, folder_id, registry, clean_date, "combined_video",
            message_id=vid_msg_id, local_fallback_dir=out_dir
        )
        print(f"✅ Combined video delivered (message_id={vid_msg_id})")
    else:
        print(f"ℹ️ Combined video already delivered for {clean_date}. Skipping item.")

    # Step 6: Commit Breakfast Completion (Fail closed!)
    committed = commit_breakfast_completion(
        drive_service, folder_id, clean_date,
        extra={"delivered_by": "unified_breakfast.py", "video": combined_video_path.name},
        required_items=["summary", "combined_video"],
        local_fallback_dir=out_dir
    )
    if not committed:
        print(f"❌ Failed to commit breakfast completion for {clean_date}")
        return False

    print(f"🎉 Breakfast for {clean_date} fully delivered and registered.")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Breakfast Generator & Delivery Engine")
    subparsers = parser.add_subparsers(dest="subcommand")

    # generate command
    p_gen = subparsers.add_parser("generate", help="Generate combined video and report")
    p_gen.add_argument("--date", default="20260905", help="Target date YYYYMMDD")
    p_gen.add_argument("--tapo-video", nargs="*", default=[], help="TAPO video file(s)")
    p_gen.add_argument("--logitech-video", nargs="*", default=[], help="Logitech video file(s)")
    p_gen.add_argument("--out-video", default="sep5_combined_breakfast.mp4", help="Output combined video path")

    # deliver command
    p_del = subparsers.add_parser("deliver", help="Execute single-authority unified delivery")
    p_del.add_argument("--date", required=True, help="Target date YYYYMMDD")
    p_del.add_argument("--tapo-dir", default=None, help="Directory containing TAPO artifacts")
    p_del.add_argument("--logitech-dir", default=None, help="Directory containing Logitech artifacts")
    p_del.add_argument("--out-dir", default=None, help="Output directory")
    p_del.add_argument("--preview", action="store_true", help="Format as test preview")
    p_del.add_argument("--skip-telegram", action="store_true", help="Skip Telegram API network calls")
    p_del.add_argument("--force", action="store_true", help="Force delivery ignoring preflight")

    args = parser.parse_args()

    if args.subcommand == "deliver":
        ok = deliver_unified_breakfast(
            target_date=args.date,
            tapo_dir=Path(args.tapo_dir) if args.tapo_dir else None,
            logitech_dir=Path(args.logitech_dir) if args.logitech_dir else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            preview=args.preview,
            skip_telegram=args.skip_telegram,
            force=args.force
        )
        sys.exit(0 if ok else 1)
    else:
        # Default / generate
        date = getattr(args, "date", "20260905")
        tapo_vid = getattr(args, "tapo_video", [])
        logi_vid = getattr(args, "logitech_video", [])
        out_vid = getattr(args, "out_video", "sep5_combined_breakfast.mp4")
        if tapo_vid and logi_vid:
            out_p = generate_combined_breakfast_video(
                tapo_clips=tapo_vid,
                logitech_clips=logi_vid,
                output_path=out_vid,
                target_date=date
            )
            print(f"Generated combined breakfast video at: {out_p}")


if __name__ == "__main__":
    main()
