import argparse
import os
import cv2
import numpy as np
import csv
import json
import shutil
from pathlib import Path

def is_probably_ir_frame(frame: np.ndarray) -> bool:
    """Check if a frame is likely an IR (grayscale) frame by comparing R, G, B channels."""
    if len(frame.shape) < 3 or frame.shape[2] != 3:
        return True
    b, g, r = cv2.split(frame)
    diff_rg = cv2.absdiff(r, g)
    diff_gb = cv2.absdiff(g, b)
    return float(np.mean(diff_rg)) < 5.0 and float(np.mean(diff_gb)) < 5.0

def brightness_score(frame: np.ndarray) -> float:
    """Calculate average brightness."""
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))

def sharpness_score(frame: np.ndarray) -> float:
    """Calculate variance of Laplacian as a proxy for sharpness."""
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def select_representative_frames(frames_data: list, max_frames: int) -> list:
    """
    Select representative frames from a list.
    Interval sampling ensures time spread.
    """
    if not frames_data:
        return []
    if len(frames_data) <= max_frames:
        return frames_data
    
    step = len(frames_data) / max_frames
    selected = []
    for i in range(max_frames):
        idx = int(i * step)
        selected.append(frames_data[idx])
    return selected

def validate_vlm_result_schema(result_dict: dict) -> bool:
    """Validate VLM output against expected JSON schema."""
    required_keys = {"cat_identity", "bowl_state", "confidence", "reasons", "needs_higher_model"}
    if not required_keys.issubset(result_dict.keys()):
        return False
    if result_dict.get("cat_identity") not in ["Dan", "Sanbo", "both", "none", "unsure"]:
        return False
    if result_dict.get("bowl_state") not in ["empty", "low", "half", "full", "unsure"]:
        return False
    if not isinstance(result_dict.get("confidence"), (float, int)):
        return False
    if not isinstance(result_dict.get("needs_higher_model"), bool):
        return False
    if not isinstance(result_dict.get("reasons"), list):
        return False
    return True

def process_video(video_path: Path, camera_name: str, out_dir: Path, max_frames: int):
    """Extract frames from a video, score them, and return selected keyframes."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return []

    frames_data = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample 1 frame per second to avoid over-processing
        if frame_idx % int(fps) == 0:
            b_score = brightness_score(frame)
            s_score = sharpness_score(frame)
            ir_flag = is_probably_ir_frame(frame)
            
            frames_data.append({
                'camera': camera_name,
                'video_path': str(video_path),
                'frame_idx': frame_idx,
                'timestamp_sec': frame_idx / fps,
                'brightness_mean': b_score,
                'grayscale_score': ir_flag,
                'sharpness_score': s_score,
                'frame_data': frame # Keep in memory for contact sheet
            })
        frame_idx += 1
    
    cap.release()
    
    selected = select_representative_frames(frames_data, max_frames)
    
    # Save selected frames
    for i, data in enumerate(selected):
        frame_name = f"{camera_name}_{video_path.stem}_keyframe_{i}.jpg"
        frame_path = out_dir / frame_name
        cv2.imwrite(str(frame_path), data['frame_data'])
        data['frame_path'] = str(frame_path)
        data['selected_reason'] = "interval_sampling"
        
    return selected

def make_contact_sheet(frames_data: list, out_path: Path, cols: int = 4):
    """Create a grid contact sheet from a list of frames."""
    if not frames_data:
        return
    
    images = [data['frame_data'] for data in frames_data]
    target_w, target_h = 320, 180
    resized = [cv2.resize(img, (target_w, target_h)) for img in images]
    
    # Pad images if they don't fill the last row
    rows = (len(resized) + cols - 1) // cols
    total_slots = rows * cols
    for _ in range(total_slots - len(resized)):
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
        
    # Build rows
    row_images = []
    for r in range(rows):
        row_images.append(cv2.hconcat(resized[r*cols : (r+1)*cols]))
        
    # Build final grid
    contact_sheet = cv2.vconcat(row_images)
    cv2.imwrite(str(out_path), contact_sheet)

def generate_prompt_template(out_dir: Path):
    """Write the VLM prompt template to the output directory."""
    template_content = '''# VLM Prompt Template

You are an expert at identifying cats and checking their food bowls in low-light and ambient light conditions.

Look at the provided keyframes from the feeding session.

Respond ONLY with a JSON object matching this schema:
```json
{
  "cat_identity": "Dan | Sanbo | both | none | unsure",
  "bowl_state": "empty | low | half | full | unsure",
  "confidence": 0.95,
  "reasons": [
    "Identified Dan by his black and white tuxedo pattern.",
    "Bowl is clearly visible and full of kibble."
  ],
  "needs_higher_model": false
}
```

## References:
- Dan: black and white tuxedo cat.
- Sanbo: calico cat (white, orange, black patches).

## Notes:
- If the image is completely dark and you cannot make out any features, output "unsure" for both and set "needs_higher_model" to true if you suspect a larger model could enhance the image.
'''
    template_path = out_dir / "vlm_prompt_template.md"
    template_path.write_text(template_content, encoding='utf-8')

def compute_summary(frames_data: list) -> dict:
    if not frames_data:
        return {}
    brightnesses = [d['brightness_mean'] for d in frames_data]
    sharpnesses = [d['sharpness_score'] for d in frames_data]
    ir_flags = [d['grayscale_score'] for d in frames_data]
    
    return {
        "frames_selected": len(frames_data),
        "brightness_min": float(np.min(brightnesses)),
        "brightness_median": float(np.median(brightnesses)),
        "brightness_max": float(np.max(brightnesses)),
        "sharpness_median": float(np.median(sharpnesses)),
        "ir_like_percentage": float(np.mean(ir_flags) * 100)
    }

def main():
    parser = argparse.ArgumentParser(description="Hybrid Spike Offline Frame Extractor")
    parser.add_argument("--tapo-dir", type=str, help="Local directory with Tapo clips")
    parser.add_argument("--logitech-dir", type=str, help="Local directory with Logitech clips")
    parser.add_argument("--out-dir", type=str, default=".agent/artifacts/hybrid_spike")
    parser.add_argument("--max-tapo-windows", type=int, default=3)
    parser.add_argument("--max-logitech-windows", type=int, default=5)
    args = parser.parse_args()

    print("--- Fair-Feeder Phase 1A Offline Spike ---")

    if not args.tapo_dir and not args.logitech_dir:
        print("\n[STOP] Checklist:")
        print("- No directories provided. Stop only when neither directory is provided.")
        print("- Provide --logitech-dir, --tapo-dir, or both.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    summary_path = out_dir / "summary.json"

    generate_prompt_template(out_dir)

    all_selected_frames = []
    summary_data = {}

    # Process Logitech
    if args.logitech_dir:
        logitech_dir = Path(args.logitech_dir)
        logitech_videos = sorted(list(logitech_dir.glob("*.mp4")))[:args.max_logitech_windows]
        logitech_frames = []
        for vid in logitech_videos:
            print(f"Processing Logitech video: {vid}")
            frames = process_video(vid, "LOGITECH", out_dir, 5)
            logitech_frames.extend(frames)
            all_selected_frames.extend(frames)
        
        if logitech_frames:
            make_contact_sheet(logitech_frames, out_dir / "logitech_contact_sheet.jpg", cols=4)
        
        summary_data["LOGITECH"] = {
            "videos_processed": len(logitech_videos),
            **compute_summary(logitech_frames)
        }

    # Process Tapo
    if args.tapo_dir:
        tapo_dir = Path(args.tapo_dir)
        tapo_videos = sorted(list(tapo_dir.glob("*.mp4")))[:args.max_tapo_windows]
        tapo_frames = []
        for vid in tapo_videos:
            print(f"Processing Tapo video: {vid}")
            frames = process_video(vid, "TAPO", out_dir, 3)
            tapo_frames.extend(frames)
            all_selected_frames.extend(frames)
            
        if tapo_frames:
            make_contact_sheet(tapo_frames, out_dir / "tapo_contact_sheet.jpg", cols=4)
            
        summary_data["TAPO"] = {
            "videos_processed": len(tapo_videos),
            **compute_summary(tapo_frames)
        }

    # Print and Write Summary
    print("\n--- Camera Summaries ---")
    print(json.dumps(summary_data, indent=2))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    # Write Manifest
    if all_selected_frames:
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['camera', 'video_path', 'frame_path', 'timestamp_sec', 'brightness_mean', 'grayscale_score', 'sharpness_score', 'selected_reason'])
            for d in all_selected_frames:
                writer.writerow([
                    d['camera'], d['video_path'], d.get('frame_path', ''),
                    f"{d['timestamp_sec']:.2f}", f"{d['brightness_mean']:.2f}",
                    d['grayscale_score'], f"{d['sharpness_score']:.2f}",
                    d.get('selected_reason', '')
                ])
        print(f"Successfully wrote manifest to {manifest_path}")

if __name__ == "__main__":
    main()
