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
import cv2
import numpy as np

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

def generate_vlm_prompt(out_dir: Path, clip_name: str, date_str: str):
    prompt = f"""You are an expert feline behavior and feeding monitor. Your task is to analyze frames from a top-down RGB camera (Logitech) looking at a cat feeding bowl.

You are evaluating the clip: {clip_name}
Date: {date_str}

Rules:
1. Use only visible evidence from the provided frames.
2. Do not count individual kibble pieces. Provide a general bowl state (empty, low, half, full, unsure).
3. Do not claim machine failure or say "feeding machine not working".
4. If the cat identity is ambiguous or obstructed, return `unsure`.
5. If the bowl state is obstructed (e.g. cat head in the way), return `unsure`.
6. Sanbo is the calico cat. Dan is the black-and-white tuxedo cat.
7. Logitech is a top-down RGB/ambient view. Only rely on visual evidence.

Output ONLY valid JSON matching the exact expected schema below.

Expected JSON schema:
```json
{{
  "camera": "LOGITECH",
  "date": "{date_str}",
  "clip_name": "{clip_name}",
  "cat_identity": "Dan | Sanbo | both | none | unsure",
  "eating_evidence": "yes | no | unsure",
  "bowl_state": "empty | low | half | full | unsure",
  "confidence": 0.0,
  "reasons": ["short visual reasons"],
  "needs_higher_model": true
}}
```
"""
    prompt_path = out_dir / f"logitech_vlm_prompt_{Path(clip_name).stem}.md"
    prompt_path.write_text(prompt)
    return prompt_path

def generate_vlm_schema(out_dir: Path):
    schema = {
        "camera": "LOGITECH",
        "date": "YYYYMMDD",
        "clip_name": "...",
        "cat_identity": "Dan | Sanbo | both | none | unsure",
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
        
    # Redact URL query param key=...
    s = re.sub(r'key=[^&\s]+', 'key=***REDACTED***', s)
    # Redact Authorization bearer token
    s = re.sub(r'(?i)bearer\s+[^\s]+', 'Bearer ***REDACTED***', s)
    return s

def validate_vlm_schema(data, expected_date=None, expected_clip_name=None):
    required_fields = [
        "camera", "date", "clip_name", "cat_identity", 
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
    if expected_clip_name and data["clip_name"] != expected_clip_name:
        raise ValueError(f"Invalid clip_name: expected {expected_clip_name}, got {data['clip_name']}")
            
    if data["cat_identity"] not in ["Dan", "Sanbo", "both", "none", "unsure"]:
        raise ValueError(f"Invalid cat_identity: {data['cat_identity']}")
    if data["eating_evidence"] not in ["yes", "no", "unsure"]:
        raise ValueError(f"Invalid eating_evidence: {data['eating_evidence']}")
    if data["bowl_state"] not in ["empty", "low", "half", "full", "unsure"]:
        raise ValueError(f"Invalid bowl_state: {data['bowl_state']}")
        
    if not isinstance(data["confidence"], (int, float)):
        raise ValueError(f"Invalid confidence type: {type(data['confidence'])}")
    if not (0.0 <= data["confidence"] <= 1.0):
        raise ValueError(f"Confidence out of range: {data['confidence']}")
        
    if not isinstance(data["reasons"], list) or not all(isinstance(x, str) for x in data["reasons"]):
        raise ValueError("reasons must be a list of strings")
        
    if not isinstance(data["needs_higher_model"], bool):
        raise ValueError("needs_higher_model must be a boolean")

def call_openai_vlm(prompt_text, image_path, model_name, api_key):
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

def call_gemini_vlm(prompt_text, image_path, model_name, api_key):
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode("utf-8")
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_img
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "response_mime_type": "application/json"
        }
    }
    
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    
    try:
        text_resp = data['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_resp)
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response: {resp.text}") from e

def main():
    parser = argparse.ArgumentParser(description="Logitech VLM Shadow Scaffold")
    parser.add_argument("--date", type=str, default="20260704")
    parser.add_argument("--out-dir", type=str, default=".agent/artifacts/logitech_vlm_shadow_20260704")
    parser.add_argument("--run-vlm", action="store_true", help="Attempt to run VLM API if key is present")
    parser.add_argument("--confirm-cost", action="store_true", help="Explicitly confirm real VLM API execution costs")
    parser.add_argument("--vlm-provider", type=str, choices=['gemini', 'openai'], help="VLM Provider")
    parser.add_argument("--vlm-model", type=str, help="VLM Model name")
    parser.add_argument("--max-clips", type=int, default=2, help="Max clips to process in VLM API")
    parser.add_argument("--cleanup-downloaded-videos", action="store_true", help="Remove downloaded mp4 files from the out-dir after result generation")
    args = parser.parse_args()

    if args.run_vlm:
        if not args.confirm_cost:
            print("[STOP] --run-vlm requires --confirm-cost to explicitly acknowledge API charges.")
            sys.exit(1)
        if not args.vlm_provider or not args.vlm_model:
            print("[STOP] --run-vlm requires --vlm-provider and --vlm-model.")
            sys.exit(1)
            
        if args.vlm_provider == 'gemini':
            api_key = os.environ.get('FAIR_FEEDER_GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                print("[STOP] Missing required API key. Set FAIR_FEEDER_GEMINI_API_KEY or GEMINI_API_KEY.")
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
        return

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    try:
        key_dict = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        drive = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"[STOP] Failed to connect to Drive: {e}")
        return

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "logitech_vlm_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    generate_vlm_schema(out_dir)

    search_date = args.date.replace("-", "")
    folder_id = os.environ.get('GDRIVE_LOGITECH_FOLDER_ID')
    
    if not folder_id:
        print("[STOP] GDRIVE_LOGITECH_FOLDER_ID is missing from environment.")
        return

    q = f"'{folder_id}' in parents and mimeType='video/mp4' and name contains '{search_date}' and trashed=false"
    results = drive.files().list(pageSize=1000, q=q, fields='files(id, name)').execute()
    all_files = results.get('files', [])

    selected_files = [f for f in all_files if in_feeding_window(f['name'], search_date)]
    selected_files.sort(key=lambda x: x['name'])
    
    manifest_data = []
    for f in selected_files:
        contact_sheet_frames = []
        dest_path = out_dir / f['name']
        download_file(drive, f['id'], dest_path)
        
        cap = cv2.VideoCapture(str(dest_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Sample frames
        sample_indices_labeled = [
            (0, "start"),
            (total_frames // 4, "quarter"),
            (total_frames // 2, "middle"),
            (3 * total_frames // 4, "three_quarter"),
            (total_frames - 1, "end")
        ]
        
        bg_frame = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, bg_frame = cap.read()
        
        first_cat_idx = None
        for idx in range(1, total_frames, int(fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and simple_cat_heuristic(frame, bg_frame):
                first_cat_idx = idx
                break
        
        if first_cat_idx is not None and first_cat_idx not in [x[0] for x in sample_indices_labeled]:
            sample_indices_labeled.append((first_cat_idx, "first_motion"))
            
        sample_indices_labeled.sort(key=lambda x: x[0])
        
        # Deduplicate by frame index, keeping the first label found
        seen_indices = set()
        final_samples = []
        for idx, label in sample_indices_labeled:
            if idx not in seen_indices and 0 <= idx < total_frames:
                seen_indices.add(idx)
                final_samples.append((idx, label))
        
        m_filename_time = re.search(r'(\d{8})_(\d{6})', f['name'])
        clip_start_time_str = f"{m_filename_time.group(1)} {m_filename_time.group(2)}" if m_filename_time else ""
        
        for idx, selection_reason in final_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            ts = extract_timestamp_calc(f['name'], idx, fps)
            heuristic_cat = simple_cat_heuristic(frame, bg_frame)
            
            frame_filename = f"{f['name']}_frame_{idx}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            seconds_from_start = round(idx / fps, 2)
            
            manifest_data.append({
                "clip_name": f['name'],
                "frame_filename": frame_filename,
                "timestamp": ts,
                "frame_index": idx,
                "motion_detected": heuristic_cat,
                "selection_reason": selection_reason,
                "clip_start_time_from_filename": clip_start_time_str,
                "seconds_from_clip_start": seconds_from_start,
                "source_drive_file_id": f['id']
            })
            
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
        
        if contact_sheet_frames:
            make_contact_sheet(contact_sheet_frames, out_dir / f"logitech_vlm_contact_sheet_{Path(f['name']).stem}.jpg")
            
        generate_vlm_prompt(out_dir, f['name'], search_date)
        
    manifest_path = out_dir / "logitech_vlm_manifest.csv"
    with open(manifest_path, "w", newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "clip_name", "frame_filename", "timestamp", "frame_index", "motion_detected",
            "selection_reason", "clip_start_time_from_filename", "seconds_from_clip_start", "source_drive_file_id"
        ])
        writer.writeheader()
        writer.writerows(manifest_data)
        
    summary = {
        "date": search_date,
        "selected_clip_names": [f['name'] for f in selected_files],
        "extracted_frames_count": len(manifest_data),
        "frames_with_motion_count": sum(1 for row in manifest_data if row["motion_detected"]),
        "schema_path": str(out_dir / "logitech_vlm_expected_schema.json"),
        "note": "prepare-only mode does not call VLM"
    }

    if args.run_vlm:
        print("[VLM Shadow] Starting real VLM API execution...")
        api_calls_made = 0
        all_results = []
        all_failed = []
        all_skipped = []
        
        clips_requested = len(selected_files[:args.max_clips])
        clips_attempted = 0
        clips_succeeded = 0
        clips_failed = 0
        clips_skipped = 0
        skipped_due_to_api_cap = 0
        had_failures = False
        import time
        import requests
        
        for f in selected_files[:args.max_clips]:
            clip_name = f['name']
            stem = Path(clip_name).stem
            contact_sheet_path = out_dir / f"logitech_vlm_contact_sheet_{stem}.jpg"
            prompt_path = out_dir / f"logitech_vlm_prompt_{stem}.md"
            
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
            
            print(f"[VLM] Processing {clip_name} with {args.vlm_provider}...")
            clips_attempted += 1
            
            attempts = 0
            max_attempts = 2
            success = False
            
            while attempts < max_attempts:
                attempts += 1
                api_calls_made += 1
                try:
                    if args.vlm_provider == 'openai':
                        result_json = call_openai_vlm(prompt_text, contact_sheet_path, args.vlm_model, api_key)
                    elif args.vlm_provider == 'gemini':
                        result_json = call_gemini_vlm(prompt_text, contact_sheet_path, args.vlm_model, api_key)
                    else:
                        raise NotImplementedError(f"Provider {args.vlm_provider} not supported.")
                        
                    validate_vlm_schema(result_json, expected_date=search_date, expected_clip_name=clip_name)
                    
                    result_json["provider"] = args.vlm_provider
                    result_json["model"] = args.vlm_model
                    result_json["prompt_hash"] = prompt_hash
                    result_json["created_at_utc"] = datetime.now(timezone.utc).isoformat()
                    result_json["source_contact_sheet"] = str(contact_sheet_path.name)
                    result_json["raw_response_saved"] = False
                    result_json["attempts_made"] = attempts
                    
                    out_path = out_dir / f"logitech_vlm_result_{stem}.json"
                    with open(out_path, "w") as jf:
                        json.dump(result_json, jf, indent=2)
                    all_results.append(result_json)
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
                                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                "prompt_hash": prompt_hash,
                                "attempts_made": attempts
                            }
                            out_path = out_dir / f"logitech_vlm_result_{stem}.failed.json"
                            with open(out_path, "w") as jf:
                                json.dump(failed_json, jf, indent=2)
                            all_skipped.append(failed_json)
                            clips_skipped += 1
                            skipped_due_to_api_cap += 1
                            break
                            
                        print(f"[VLM] Transient error {status} for {clip_name}. Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                        
                    print(f"[VLM] Failed for {clip_name}: {sanitized_msg}")
                    failed_json = {
                        "clip_name": clip_name,
                        "provider": args.vlm_provider,
                        "model": args.vlm_model,
                        "error_type": type(e).__name__,
                        "error_message": sanitized_msg,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "prompt_hash": prompt_hash,
                        "attempts_made": attempts
                    }
                    out_path = out_dir / f"logitech_vlm_result_{stem}.failed.json"
                    with open(out_path, "w") as jf:
                        json.dump(failed_json, jf, indent=2)
                    all_failed.append(failed_json)
                    clips_failed += 1
                    had_failures = True
                    break
                    
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
        
        with open(out_dir / "logitech_vlm_shadow_summary.json", "w") as f_sum:
            json.dump(summary, f_sum, indent=2)
            
        # Human-readable report
        report_lines = [
            "[SHADOW] Logitech VLM / Sanbo feeder",
            f"Date: {search_date}",
            f"Provider/model: {args.vlm_provider} / {args.vlm_model}",
            "Production report changed: no",
            ""
        ]
        for r in all_results:
            report_lines.append(f"Clip: {r['clip_name']}")
            report_lines.append(f"- Cat identity: {r['cat_identity']}")
            ee = r['eating_evidence']
            if ee == "unsure":
                ee = "Uncertain"
            report_lines.append(f"- Eating evidence: {ee}")
            report_lines.append(f"- Bowl state: {r['bowl_state']}")
            conf = r['confidence']
            conf_str = str(conf)
            if conf < 0.75:
                conf_str += " (Needs review)"
            report_lines.append(f"- Confidence: {conf_str}")
            nhm = r['needs_higher_model']
            nhm_str = str(nhm).lower()
            if nhm:
                nhm_str += " (Needs higher model)"
            report_lines.append(f"- Needs higher model: {nhm_str}")
            report_lines.append(f"- Contact sheet: {r['source_contact_sheet']}")
            report_lines.append("- Reasons:")
            for reason in r.get('reasons', []):
                report_lines.append(f"  - {reason}")
            report_lines.append("")
            
        for r in all_failed:
            report_lines.append(f"Clip: {r['clip_name']}")
            report_lines.append(f"- FAILED: {r['error_type']}")
            report_lines.append(f"- Error: {r['error_message']}")
            report_lines.append("")
            
        for r in all_skipped:
            report_lines.append(f"Clip: {r['clip_name']}")
            report_lines.append(f"- SKIPPED: {r['error_type']}")
            report_lines.append(f"- Reason: {r['error_message']}")
            report_lines.append("")
            
        (out_dir / "logitech_vlm_shadow_report.md").write_text("\n".join(report_lines))
        
        # Telegram preview
        tg_lines = [
            "[SHADOW] Logitech VLM",
            "Production report unchanged."
        ]
        for r in all_results:
            ee = "Uncertain" if r['eating_evidence'] == "unsure" else r['eating_evidence']
            tg_lines.append(f"Clip: {r['clip_name']} -> Cat: {r['cat_identity']} Eating: {ee} Bowl: {r['bowl_state']} Conf: {r['confidence']}")
            
        for r in all_failed:
            tg_lines.append(f"Clip: {r['clip_name']} -> FAILED: {r['error_type']}")
            
        for r in all_skipped:
            tg_lines.append(f"Clip: {r['clip_name']} -> SKIPPED: API call cap reached")
            
        (out_dir / "logitech_vlm_shadow_telegram_preview.txt").write_text("\n".join(tg_lines))
        
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
