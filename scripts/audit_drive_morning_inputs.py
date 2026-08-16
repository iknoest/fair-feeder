import os
import re
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
import pytz

# Try to safely load .env without printing it
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

# Do not print or expose values of these env vars
REQUIRED_VARS = ["GDRIVE_SERVICE_ACCOUNT_KEY"]

def check_credentials():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        print("[STOP] Missing required environment variables:")
        for m in missing:
            print(f"- {m}")
        print("Please ensure they are set in the environment or in the local .env file.")
        print("(Do not print or share the actual secret values!)")
        return False
    return True

FEEDING_WINDOW_START = (6, 18)
FEEDING_WINDOW_END = (6, 30)

def in_feeding_window(filename, target_date_str):
    m = re.match(r'motion_(\d{8})_(\d{2})(\d{2})\d{2}', filename)
    if not m:
        return False, "Regex mismatch"
    
    file_date = m.group(1)
    if file_date != target_date_str:
        return False, f"Date mismatch ({file_date} != {target_date_str})"
    
    file_min = int(m.group(2)) * 60 + int(m.group(3))
    start_min = FEEDING_WINDOW_START[0] * 60 + FEEDING_WINDOW_START[1]
    end_min = FEEDING_WINDOW_END[0] * 60 + FEEDING_WINDOW_END[1]
    
    if start_min <= file_min <= end_min:
        return True, "In feeding window"
    else:
        return False, f"Time outside window ({m.group(2)}:{m.group(3)})"

def extract_duration_from_filename(filename):
    m = re.match(r'motion_\d{8}_\d{6}(?:_(\d+)m)?_(\d+)s', filename)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        secs = int(m.group(2))
        return mins * 60 + secs
    return None

import cv2
import numpy as np

def brightness_score(frame: np.ndarray) -> float:
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))

def sharpness_score(frame: np.ndarray) -> float:
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def is_probably_ir_frame(frame: np.ndarray) -> bool:
    if len(frame.shape) < 3 or frame.shape[2] != 3:
        return True
    b, g, r = cv2.split(frame)
    diff_rg = cv2.absdiff(r, g)
    diff_gb = cv2.absdiff(g, b)
    return float(np.mean(diff_rg)) < 5.0 and float(np.mean(diff_gb)) < 5.0

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

def main():
    parser = argparse.ArgumentParser(description="Drive Input Audit Script")
    parser.add_argument("--camera", type=str, choices=["TAPO", "LOGITECH", "BOTH"], default="BOTH")
    parser.add_argument("--date", type=str, help="YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--out-dir", type=str, default=".agent/artifacts/drive_input_audit")
    args = parser.parse_args()

    if not check_credentials():
        return

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io

    try:
        key_dict = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        drive = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"[STOP] Failed to parse or use GDRIVE_SERVICE_ACCOUNT_KEY: {type(e).__name__}")
        return

    cet_tz = pytz.timezone('Europe/Amsterdam')
    if args.date:
        search_date = args.date.replace("-", "")
    else:
        search_date = datetime.now(cet_tz).strftime('%Y%m%d')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_to_process = ["TAPO", "LOGITECH"] if args.camera == "BOTH" else [args.camera]

    for cam in cameras_to_process:
        folder_env_var = 'GDRIVE_UPLOAD_FOLDER_ID' if cam == "TAPO" else 'GDRIVE_LOGITECH_FOLDER_ID'
        folder_id = os.environ.get(folder_env_var)
        if not folder_id:
            print(f"Skipping {cam} because {folder_env_var} is not set in environment.")
            continue

        print(f"\n--- Auditing {cam} for {search_date} ---")
        q = f"'{folder_id}' in parents and mimeType='video/mp4' and name contains '{search_date}' and trashed=false"
        try:
            results = drive.files().list(
                pageSize=1000, q=q, fields='files(id, name, modifiedTime, size)', orderBy='name'
            ).execute()
        except Exception as e:
            print(f"Failed to list files for {cam}: {e}")
            continue
            
        all_files = results.get('files', [])
        print(f"Found {len(all_files)} total candidate mp4 clips in Drive.")

        selected_files = []
        for f in all_files:
            in_window, reason = in_feeding_window(f['name'], search_date)
            f['in_window'] = in_window
            f['reason'] = reason
            if in_window:
                selected_files.append(f)

        print(f"Selected {len(selected_files)} clips matching the feeding window ({FEEDING_WINDOW_START[0]:02d}:{FEEDING_WINDOW_START[1]:02d}-{FEEDING_WINDOW_END[0]:02d}:{FEEDING_WINDOW_END[1]:02d}).")

        csv_path = out_dir / f"{cam}_{search_date}_selected_clips.csv"
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['name', 'id', 'modifiedTime', 'size', 'duration_sec', 'reason'])
            for f in all_files:
                dur = extract_duration_from_filename(f['name'])
                writer.writerow([f['name'], f['id'], f.get('modifiedTime',''), f.get('size',''), dur if dur else '', f['reason']])

        # Download and process selected clips
        sampled_frames = []
        frame_metrics = []
        possible_cat_activity = False

        for f in selected_files:
            print(f"Downloading {f['name']} for frame sampling...")
            dest_path = out_dir / f['name']
            if not dest_path.exists():
                req = drive.files().get_media(fileId=f['id'])
                with open(dest_path, 'wb') as fh:
                    dl = MediaIoBaseDownload(fh, req)
                    done = False
                    while not done:
                        _, done = dl.next_chunk()

            cap = cv2.VideoCapture(str(dest_path))
            if not cap.isOpened():
                print(f"Failed to open {f['name']} with cv2")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Sample 1 frame per second
                if frame_idx % int(fps) == 0:
                    b_score = brightness_score(frame)
                    s_score = sharpness_score(frame)
                    ir_flag = is_probably_ir_frame(frame)
                    
                    frame_metrics.append({
                        'camera': cam,
                        'video_name': f['name'],
                        'timestamp_sec': frame_idx / fps,
                        'brightness': b_score,
                        'sharpness': s_score,
                        'is_ir': ir_flag
                    })
                    
                    # Very crude heuristic for cat motion check - sharp changes in brightness over time might mean cat
                    # We will just collect all frames for the contact sheet if we sample at low rate
                    if len(sampled_frames) < 30: # Limit contact sheet size
                        sampled_frames.append({
                            'frame_data': frame,
                            'name': f"{f['name']}_{frame_idx}"
                        })
                frame_idx += 1
            cap.release()

        if sampled_frames:
            make_contact_sheet(sampled_frames, out_dir / f"{cam}_{search_date}_contact_sheet.jpg")

        metrics_csv_path = out_dir / f"{cam}_{search_date}_frame_metrics.csv"
        with open(metrics_csv_path, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['camera', 'video_name', 'timestamp_sec', 'brightness', 'sharpness', 'is_ir'])
            writer.writeheader()
            for m in frame_metrics:
                writer.writerow(m)

        summary = {
            "camera": cam,
            "date": search_date,
            "folder_env_var": folder_env_var,
            "total_candidates": len(all_files),
            "selected_clips": len(selected_files),
            "selected_clip_names": [f['name'] for f in selected_files],
            "warning": "No cat-like motion could be confidently inferred by crude local logic. Please check the contact sheets."
        }
        
        with open(out_dir / f"{cam}_{search_date}_summary.json", 'w') as f_json:
            json.dump(summary, f_json, indent=2)

    print(f"\nAudit complete. Artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()
