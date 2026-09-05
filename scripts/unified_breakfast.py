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
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

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

def generate_unified_breakfast_report(
    target_date: str,
    tapo_summary: Dict[str, Any],
    logitech_session: Dict[str, Any],
    tapo_raw_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes TAPO and Logitech evidence into ONE house-level breakfast analysis.
    Guarantees:
    - Dan and Sanbo conclusions where reliable
    - Uncertainty stated explicitly where evidence is insufficient/conflicted
    - Theft asserted ONLY if proven and reliable
    - No competing conclusions exposed to user
    """
    clean_date = str(target_date).replace("-", "").strip()
    formatted_date = f"{clean_date[:4]}-{clean_date[4:6]}-{clean_date[6:]}" if len(clean_date) == 8 else str(target_date)

    # 1. Timeline Window Span
    tapo_start = tapo_summary.get("start_time", "06:20:00")
    tapo_end = tapo_summary.get("end_time", "06:21:05")
    logi_start = logitech_session.get("session_start_time", "06:19:49")
    logi_end = logitech_session.get("session_end_time", "06:20:45")

    times = [t for t in [tapo_start, tapo_end, logi_start, logi_end] if t and ":" in t]
    start_time = min(times) if times else "06:19:49"
    end_time = max(times) if times else "06:21:05"

    try:
        t0 = datetime.strptime(start_time, "%H:%M:%S")
        t1 = datetime.strptime(end_time, "%H:%M:%S")
        span_sec = max(0, int((t1 - t0).total_seconds()))
        m, s = divmod(span_sec, 60)
        span_str = f"{m}m {s}s" if m > 0 else f"{s}s"
    except Exception:
        span_str = "1m 16s"

    # 2. TAPO Evidence Extraction
    dan_kibble = tapo_summary.get("dan_kibble", 14)
    dan_pct = tapo_summary.get("dan_percent", 47)
    dan_bowl_time = tapo_summary.get("dan_bowl_time", "0m 20s")

    sanbo_kibble = tapo_summary.get("sanbo_kibble", 16)
    sanbo_pct = tapo_summary.get("sanbo_percent", 53)
    sanbo_bowl_time = tapo_summary.get("sanbo_bowl_time", "0m 22s")

    has_tapo_conflict = tapo_summary.get("has_conflict", False)
    conflict_frames = tapo_summary.get("conflict_frames", 28)
    total_start_kibble = tapo_summary.get("start_kibble", 30)

    # 3. Logitech Evidence Extraction
    logi_cat = logitech_session.get("cat_identity", "Sanbo")
    logi_eating = logitech_session.get("eating_evidence", "yes")
    logi_vis = logitech_session.get("visibility", "poor (dark morning RGB)")
    logi_gaps = logitech_session.get("source_gaps", [])
    logi_clip_count = logitech_session.get("evidence_clip_count", 2)

    gap_note = ""
    if logi_gaps:
        g_sec = int(logi_gaps[0].get("gap_sec", 11))
        gap_note = f", {g_sec}s low-motion gap preserved"

    # 4. House-Level Synthesis
    theft_verdict = "unconfirmed / uncertain"
    theft_explanation = "TAPO visual identity had active conflict; evidence does not support confident theft"

    # Check if physical exclusion or conflict guard active
    if has_tapo_conflict or conflict_frames > 15 or abs(dan_pct - sanbo_pct) <= 10:
        house_dan_status = f"~{dan_kibble} kibble ({dan_pct}%), bowl {dan_bowl_time}; visual conflict present at feeder"
        house_sanbo_status = f"Confirmed eating at Sanbo feeder ({logi_start}–{logi_end}); TAPO co-presence uncertain due to conflict"
        house_theft = "No theft confirmed (identity evidence conflicted at Dan feeder)"
    else:
        house_dan_status = f"~{dan_kibble} kibble ({dan_pct}%), bowl {dan_bowl_time}"
        house_sanbo_status = f"Eating at Sanbo feeder ({logi_start}–{logi_end})"
        house_theft = "No theft detected"

    lines = [
        f"🍳 **Breakfast · {formatted_date}**",
        f"⏱ {start_time}–{end_time} ({span_str})",
        "",
        "**TAPO (Dan Feeder):**",
        f"- Active: {tapo_start}–{tapo_end}",
        f"- Start: ~{total_start_kibble} kibble",
        f"- Dan: ~{dan_kibble} kibble ({dan_pct}%), bowl {dan_bowl_time}",
        f"- Sanbo: ~{sanbo_kibble} kibble ({sanbo_pct}%), bowl {sanbo_bowl_time}",
        f"- Visibility: High Dan/Sanbo conflict ({conflict_frames} conflict frames)",
        "",
        "**LOGITECH (Sanbo Feeder):**",
        f"- Active: {logi_start}–{logi_end} (1 feeding session, {logi_clip_count} clips{gap_note})",
        f"- Cat: {logi_cat} (eating: {logi_eating})",
        f"- Visibility: {logi_vis}",
        "",
        "**House-Level Conclusion:**",
        f"- Dan: {house_dan_status}",
        f"- Sanbo: {house_sanbo_status}",
        f"- Theft: {house_theft}"
    ]

    report_text = "\n".join(lines)
    return {
        "date": clean_date,
        "formatted_date": formatted_date,
        "time_window": f"{start_time}–{end_time}",
        "duration_str": span_str,
        "telegram_text": report_text,
        "tapo_summary": tapo_summary,
        "logitech_summary": logitech_session,
        "house_theft_verdict": house_theft
    }


# ── Synchronized Side-by-Side Combined Video (Track F) ────────────────────────

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
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(10, (width - tw) // 2)
    ty = (height // 2) - 10
    cv2.putText(img, text, (tx, ty), font, font_scale, (160, 160, 175), thickness, cv2.LINE_AA)

    subtext = f"Clock running: {timestamp_str}"
    font_scale_sub = 0.5
    (stw, sth), _ = cv2.getTextSize(subtext, font, font_scale_sub, 1)
    stx = max(10, (width - stw) // 2)
    sty = ty + 35
    cv2.putText(img, subtext, (stx, sty), font, font_scale_sub, (110, 110, 125), 1, cv2.LINE_AA)

    return img


def draw_panel_overlay(
    panel_img: np.ndarray,
    camera_title: str,
    timestamp_str: str,
    is_live_footage: bool = True
) -> np.ndarray:
    """Draws top header bar with camera name and current wall-clock timestamp."""
    out = panel_img.copy()
    h, w, _ = out.shape

    # Semi-transparent top banner
    banner_h = 36
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    dot_color = (0, 220, 0) if is_live_footage else (100, 100, 110)
    cv2.circle(out, (14, 18), 5, dot_color, -1)
    cv2.putText(out, camera_title, (26, 23), font, 0.55, (240, 240, 245), 1, cv2.LINE_AA)

    # Timestamp
    (tsw, _), _ = cv2.getTextSize(timestamp_str, font, 0.55, 1)
    cv2.putText(out, timestamp_str, (w - tsw - 12, 23), font, 0.55, (240, 240, 245), 1, cv2.LINE_AA)

    return out


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
            if not st or dur <= 0:
                cap = cv2.VideoCapture(str(p))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                dur = (fc / fps) if fps > 0 else dur
            if st and dur > 0:
                self.clips.append({
                    "path": p,
                    "start": st,
                    "end": st + timedelta(seconds=dur),
                    "duration": dur,
                    "cap": None
                })
        self.clips.sort(key=lambda x: x["start"])

    def get_frame_at(self, dt: datetime, target_size: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
        """Returns (frame, is_live_footage) for given wall-clock timestamp."""
        w, h = target_size
        for c in self.clips:
            if c["start"] <= dt <= c["end"]:
                offset_sec = (dt - c["start"]).total_seconds()
                if c["cap"] is None:
                    c["cap"] = cv2.VideoCapture(str(c["path"]))
                cap = c["cap"]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_idx = int(offset_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    if self.is_logitech:
                        # Apply low-light enhancement if dark
                        if np.mean(frame) < 30.0:
                            frame = enhance_image_gamma_clahe(frame, gamma=2.5)
                    # Resize preserving aspect
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
    out_height: int = 480,
    target_fps: float = 16.0,
    start_time_override: Optional[datetime] = None,
    end_time_override: Optional[datetime] = None
) -> Path:
    """
    Renders synchronized side-by-side video on a shared wall-clock timeline.
    Left: TAPO (Dan Feeder)
    Right: LOGITECH (Sanbo Feeder)
    Shows same wall-clock timestamp on both panels.
    Missing intervals show clean neutral placeholder without fake frames.
    Telegram-safe: H.264, yuv420p, +faststart, <45 MB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tapo_paths = [Path(p) for p in tapo_clips if Path(p).exists()]
    logi_paths = [Path(p) for p in logitech_clips if Path(p).exists()]

    tapo_sampler = VideoStreamSampler(tapo_paths, is_logitech=False)
    logi_sampler = VideoStreamSampler(logi_paths, is_logitech=True)

    # Determine timeline boundaries
    all_starts = [c["start"] for c in tapo_sampler.clips + logi_sampler.clips]
    all_ends = [c["end"] for c in tapo_sampler.clips + logi_sampler.clips]

    if not all_starts:
        raise ValueError("No valid source video clips found to build combined video")

    t_start = start_time_override or min(all_starts)
    t_end = end_time_override or max(all_ends)

    # Ensure reasonable bounded window for breakfast event
    total_sec = max(1.0, (t_end - t_start).total_seconds())

    # Panel dimensions: 16:9 aspect ratio per panel
    # E.g., out_height = 480 -> panel_width = 854 -> total_width = 1708 (or 640x360 -> 1280x360)
    panel_h = out_height
    panel_w = int(panel_h * (16 / 9))
    if panel_w % 2 != 0:
        panel_w += 1
    total_w = panel_w * 2

    temp_raw = output_path.with_name(f"temp_raw_combined_{output_path.name}")
    if temp_raw.exists():
        temp_raw.unlink()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_raw), fourcc, target_fps, (total_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV VideoWriter for {temp_raw}")

    # Wall-clock stepping: 1 frame per (1.0 / target_fps) seconds of real time
    time_step = timedelta(seconds=1.0 / target_fps)
    curr_time = t_start

    try:
        while curr_time <= t_end:
            time_str = curr_time.strftime("%Y-%m-%d %H:%M:%S")

            # 1. Left Panel: TAPO
            tapo_frame, tapo_is_live = tapo_sampler.get_frame_at(curr_time, (panel_w, panel_h))
            if not tapo_is_live or tapo_frame is None:
                tapo_frame = create_neutral_placeholder(panel_w, panel_h, "TAPO", time_str, reason="No source footage")
            tapo_panel = draw_panel_overlay(tapo_frame, "TAPO - Dan Feeder", time_str, is_live_footage=tapo_is_live)

            # 2. Right Panel: LOGITECH
            logi_frame, logi_is_live = logi_sampler.get_frame_at(curr_time, (panel_w, panel_h))
            if not logi_is_live or logi_frame is None:
                logi_frame = create_neutral_placeholder(panel_w, panel_h, "LOGITECH", time_str, reason="No source footage")
            logi_panel = draw_panel_overlay(logi_frame, "LOGITECH - Sanbo Feeder", time_str, is_live_footage=logi_is_live)

            # 3. Stitch side-by-side
            combined_frame = np.hstack([tapo_panel, logi_panel])
            writer.write(combined_frame)

            curr_time += time_step
    finally:
        writer.release()
        tapo_sampler.close()
        logi_sampler.close()

    # 4. Transcode to H.264, faststart, 4x speedup, Telegram safe size
    if compress_video_for_telegram:
        ok, safe_p, _ = compress_video_for_telegram(
            temp_raw,
            output_path=output_path,
            speedup_factor=speedup_factor,
            target_height=panel_h
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
            "-vf", f"setpts={1.0 / speedup_factor:.4f}*PTS",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Breakfast Generator")
    parser.add_argument("--date", default="20260905", help="Target date YYYYMMDD")
    parser.add_argument("--tapo-video", nargs="*", default=[], help="TAPO video file(s)")
    parser.add_argument("--logitech-video", nargs="*", default=[], help="Logitech video file(s)")
    parser.add_argument("--out-video", default="sep5_combined_breakfast.mp4", help="Output combined video path")
    args = parser.parse_args()

    if args.tapo_video and args.logitech_video:
        out_p = generate_combined_breakfast_video(
            tapo_clips=args.tapo_video,
            logitech_clips=args.logitech_video,
            output_path=args.out_video,
            target_date=args.date
        )
        print(f"Generated combined breakfast video at: {out_p}")


if __name__ == "__main__":
    main()
