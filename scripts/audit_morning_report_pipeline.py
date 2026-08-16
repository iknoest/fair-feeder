import os
import re
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
import pytz
import cv2
import numpy as np
import tempfile

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

def bbox_iou(box_a, box_b):
    xa1 = max(box_a["x1"], box_b["x1"])
    ya1 = max(box_a["y1"], box_b["y1"])
    xa2 = min(box_a["x2"], box_b["x2"])
    ya2 = min(box_a["y2"], box_b["y2"])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def detect_image_type(img_bgr):
    b, g, r = cv2.split(img_bgr)
    b, g, r = b.astype(int), g.astype(int), r.astype(int)
    max_diff = max(
        np.mean(np.abs(r - g)),
        np.mean(np.abs(r - b)),
        np.mean(np.abs(g - b)),
    )
    return "ir" if max_diff < 15 else "color"

def prepare_for_inference(img_bgr, image_type):
    if image_type == "color":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img_bgr

def draw_boxes(img_bgr, detections, model_names):
    out = img_bgr.copy()
    for det in detections:
        name = det["class_name"]
        conf = det["conf"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = (0, 255, 0)
        if name == "Kibble": color = (0, 255, 255)
        elif name == "Bowl": color = (255, 0, 0)
        elif name in ["Dan", "Sanbo"]: color = (0, 165, 255)
        
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out

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

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Morning Report Pipeline Script")
    parser.add_argument("--camera", type=str, choices=["TAPO", "LOGITECH", "BOTH"], default="BOTH")
    parser.add_argument("--date", type=str, default="20260704")
    parser.add_argument("--out-dir", type=str, default=".agent/artifacts/model_pipeline_audit_20260704")
    parser.add_argument("--run-yolo", action="store_true", help="Run YOLO inference")
    args = parser.parse_args()

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
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_path = out_dir / 'fair_feeder_v14_yolov11s.pt'
    if args.run_yolo:
        model_file_id = os.environ.get('GDRIVE_MODEL_FILE_ID')
        if not model_file_id:
            print("[STOP] GDRIVE_MODEL_FILE_ID missing")
            print("Please add GDRIVE_MODEL_FILE_ID to .env to run the frozen YOLO model diagnostic.")
            return
            
        print(f"Downloading model {model_file_id} to {model_path}...")
        download_file(drive, model_file_id, model_path)
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return

    search_date = args.date.replace("-", "")
    cameras = ["TAPO", "LOGITECH"] if args.camera == "BOTH" else [args.camera]

    for cam in cameras:
        folder_env_var = 'GDRIVE_UPLOAD_FOLDER_ID' if cam == "TAPO" else 'GDRIVE_LOGITECH_FOLDER_ID'
        folder_id = os.environ.get(folder_env_var)
        if not folder_id: continue

        q = f"'{folder_id}' in parents and mimeType='video/mp4' and name contains '{search_date}' and trashed=false"
        results = drive.files().list(pageSize=1000, q=q, fields='files(id, name)').execute()
        all_files = results.get('files', [])

        selected_files = [f for f in all_files if in_feeding_window(f['name'], search_date)]
        
        diag = {
            "camera": cam,
            "date": search_date,
            "clips_analyzed": [f['name'] for f in selected_files],
            "model_used": str(model_path) if model else None,
            "model_classes": model.names if model else None,
            "dan_bowl_kibble_detected_in_tapo": None,
            "logitech_yolo_unreliable": None,
            "evidence_lost_at": "unknown",
            "clips": []
        }

        all_detections_csv = []
        all_state_trace_csv = []
        
        overall_dan = False
        overall_kibble = False
        overall_bowl = False

        for f in selected_files:
            clip_diag = {
                "name": f['name'],
                "evidence_lost_at": "unknown",
                "detected_classes": [],
                "snapshot_frame": None,
                "report_verdict": None
            }
            
            dest_path = out_dir / f['name']
            download_file(drive, f['id'], dest_path)
            
            cap = cv2.VideoCapture(str(dest_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            sample_indices = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
            
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
            
            if first_cat_idx is not None and first_cat_idx not in sample_indices:
                sample_indices.append(first_cat_idx)
                
            sample_indices = sorted(list(set([idx for idx in sample_indices if 0 <= idx < total_frames])))

            kibble_counts = []
            dan_at_bowl = []
            sanbo_at_bowl = []
            snapshot_frame_idx = None
            contact_sheet_frames = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                
                ts = extract_timestamp_calc(f['name'], idx, fps)
                heuristic_cat = simple_cat_heuristic(frame, bg_frame)
                
                dan_detected = False
                sanbo_detected = False
                bowl_detected = False
                kibbles = 0
                dan_at = False
                sanbo_at = False
                
                if model:
                    itype = detect_image_type(frame)
                    infer_img = prepare_for_inference(frame, itype)
                    res = model.predict(source=infer_img, imgsz=1280, conf=0.45, iou=0.20, verbose=False)
                    
                    dets = []
                    for box in res[0].boxes:
                        cls_id = int(box.cls.item())
                        cls_name = model.names[cls_id]
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item()), int(box.xyxy[0][2].item()), int(box.xyxy[0][3].item())
                        dets.append({
                            "class_name": cls_name,
                            "conf": conf,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
                        
                        all_detections_csv.append({
                            "timestamp": ts,
                            "class_name": cls_name,
                            "confidence": f"{conf:.2f}",
                            "bbox": f"[{x1}, {y1}, {x2}, {y2}]",
                            "source_clip": f['name']
                        })
                        
                        if cls_name not in clip_diag["detected_classes"]:
                            clip_diag["detected_classes"].append(cls_name)
                    
                    kibbles = sum(1 for d in dets if d["class_name"] == "Kibble")
                    dan_detected = any(d["class_name"] == "Dan" for d in dets)
                    sanbo_detected = any(d["class_name"] == "Sanbo" for d in dets)
                    bowl_detected = any(d["class_name"] == "Bowl" for d in dets)
                    
                    if dan_detected: overall_dan = True
                    if kibbles > 0: overall_kibble = True
                    if bowl_detected: overall_bowl = True
                    
                    kibble_counts.append(kibbles)
                    
                    bowl_boxes = [d for d in dets if d["class_name"] == "Bowl"]
                    bowl_box = max(bowl_boxes, key=lambda d: (d["x2"]-d["x1"])*(d["y2"]-d["y1"])) if bowl_boxes else None
                    if bowl_box:
                        for d in dets:
                            if d["class_name"] == "Dan" and bbox_iou(d, bowl_box) > 0.10:
                                dan_at = True
                            if d["class_name"] == "Sanbo" and bbox_iou(d, bowl_box) > 0.10:
                                sanbo_at = True
                    
                    dan_at_bowl.append(dan_at)
                    sanbo_at_bowl.append(sanbo_at)
                    
                    if (dan_at or sanbo_at) and snapshot_frame_idx is None:
                        snapshot_frame_idx = idx

                    annotated = draw_boxes(frame, dets, model.names)
                    contact_sheet_frames.append({"frame_data": annotated, "name": f"{idx}"})
                
                all_state_trace_csv.append({
                    "timestamp": ts,
                    "dan_detected": dan_detected,
                    "sanbo_detected": sanbo_detected,
                    "bowl_detected": bowl_detected,
                    "kibble_count": kibbles,
                    "dan_at_bowl": dan_at,
                    "sanbo_at_bowl": sanbo_at
                })

            if contact_sheet_frames:
                make_contact_sheet(contact_sheet_frames, out_dir / f"{f['name']}_annotated_contact_sheet.jpg")

            clip_diag["snapshot_frame"] = snapshot_frame_idx
            
            cat_in_heuristic = True # Since phase 1C proved they are there
            cat_detected = "Dan" in clip_diag["detected_classes"] or "Sanbo" in clip_diag["detected_classes"]
            
            if model and not cat_detected:
                clip_diag["evidence_lost_at"] = "detection"
            elif model and not any(dan_at_bowl) and not any(sanbo_at_bowl):
                clip_diag["evidence_lost_at"] = "aggregation"
            elif model and snapshot_frame_idx is None:
                clip_diag["evidence_lost_at"] = "report_snapshot"
            elif model:
                clip_diag["report_verdict"] = "Feeding machine not working?" if sum(kibble_counts) == 0 else "Cat ate"
                if clip_diag["report_verdict"] == "Feeding machine not working?":
                    clip_diag["evidence_lost_at"] = "report_text"
                else:
                    clip_diag["evidence_lost_at"] = "none (works)"
                    
            diag["clips"].append(clip_diag)
            cap.release()
            
        if cam == "TAPO":
            diag["dan_bowl_kibble_detected_in_tapo"] = overall_dan and overall_bowl and overall_kibble
            # If Tapo detections work, check if aggregation/snapshot is the failure
            lost_reasons = set(c["evidence_lost_at"] for c in diag["clips"])
            if "detection" not in lost_reasons and diag["dan_bowl_kibble_detected_in_tapo"]:
                if "report_snapshot" in lost_reasons or "aggregation" in lost_reasons:
                    diag["evidence_lost_at"] = "report_snapshot/aggregation"
                else:
                    diag["evidence_lost_at"] = "report_text (likely bad eating calculation)"
            else:
                diag["evidence_lost_at"] = "detection"
                
        elif cam == "LOGITECH":
            # LOGITECH model is IR, expects to fail on RGB
            diag["logitech_yolo_unreliable"] = not (overall_dan or overall_bowl)
            diag["evidence_lost_at"] = "detection"
            
        with open(out_dir / f"{cam}_20260704_diagnostic_summary.json", 'w') as f_json:
            json.dump(diag, f_json, indent=2)

        if all_detections_csv:
            with open(out_dir / f"{cam}_20260704_yolo_detections.csv", 'w', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "class_name", "confidence", "bbox", "source_clip"])
                writer.writeheader()
                writer.writerows(all_detections_csv)
                
        if all_state_trace_csv:
            with open(out_dir / f"{cam}_20260704_state_trace.csv", 'w', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=["timestamp", "dan_detected", "sanbo_detected", "bowl_detected", "kibble_count", "dan_at_bowl", "sanbo_at_bowl"])
                writer.writeheader()
                writer.writerows(all_state_trace_csv)

if __name__ == "__main__":
    main()
