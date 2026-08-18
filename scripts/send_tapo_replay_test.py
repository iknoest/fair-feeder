import os
import sys
import json
import time
import requests
import cv2
from pathlib import Path

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[STOP] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable.", flush=True)
        sys.exit(1)

    video_path = Path("scratch/replay_acceptance/TAPO_REPLAY_ACCEPTANCE_20260818.mp4")
    
    # If not present locally (e.g. in CI runner), generate it using replay harness
    if not video_path.exists():
        print(f"[REPLAY] Generating {video_path} from real source footage...", flush=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download real source from Drive if not present
        src_path = Path("scratch/replay_acceptance/motion_20260818_062005_2m_30s.mp4")
        if not src_path.exists():
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            gdrive_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
            folder_id = os.environ.get("GDRIVE_UPLOAD_FOLDER_ID") or os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
            creds = service_account.Credentials.from_service_account_info(
                json.loads(gdrive_json),
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            service = build('drive', 'v3', credentials=creds)
            query = "name contains '20260818_062005' and trashed = false"
            res = service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, size)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            files = res.get('files', [])
            if not files:
                raise FileNotFoundError("Could not find motion_20260818_062005 in Google Drive!")
            file_id = files[0]['id']
            print(f"Found Drive file: {files[0]['name']} (ID: {file_id})", flush=True)
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(src_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download {int(status.progress() * 100)}%", flush=True)
            print(f"Downloaded real source {src_path.name} ({src_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
            
        # Run replay stitch
        cap = cv2.VideoCapture(str(src_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        
        # 150s + 20s extension loop
        extra_frames = frames[-int(20*fps):] if len(frames) >= int(20*fps) else frames
        all_frames = frames + extra_frames
        
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in all_frames:
            writer.write(f)
        writer.release()
        print(f"Generated replay video: {video_path} ({len(all_frames)} frames)", flush=True)

    # Verify duration
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = frame_count / fps
    cap.release()
    
    print(f"\n[VERIFY] TAPO Replay Video:", flush=True)
    print(f"  - File: {video_path.name}", flush=True)
    print(f"  - Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB", flush=True)
    print(f"  - Duration: {duration:.1f}s ({int(duration//60)}m {int(duration%60)}s)", flush=True)
    print(f"  - Exceeds 150s (2m30): {'YES' if duration > 150.0 else 'NO'}", flush=True)
    
    if duration <= 150.0:
        print("[ERROR] Test video does not exceed 150 seconds! Aborting send.", flush=True)
        sys.exit(1)

    caption = (
        "[TEST][TAPO] 150s continuation replay acceptance\n\n"
        "Source: 2026-08-18 real Dan breakfast\n"
        "Replay-only validation\n"
        "Old cutoff: 2m30\n"
        "Test video: ~2m50\n"
        "Please verify playback continues past 2m30."
    )

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    print(f"\n[SEND] Delivering test video to Telegram feeder group...", flush=True)
    with open(video_path, "rb") as vf:
        files = {"video": (video_path.name, vf, "video/mp4")}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "supports_streaming": True
        }
        resp = requests.post(url, data=data, files=files, timeout=120)
        
    try:
        r_json = resp.json()
    except Exception:
        r_json = {}
        
    status_code = resp.status_code
    ok = r_json.get("ok", False)
    message_id = r_json.get("result", {}).get("message_id")
    
    print(f"[DELIVERY EVIDENCE]", flush=True)
    print(f"  - HTTP Status: {status_code}", flush=True)
    print(f"  - ok: {ok}", flush=True)
    print(f"  - message_id: {message_id}", flush=True)
    
    if not ok or status_code != 200:
        print(f"[ERROR] Telegram delivery failed: {resp.text}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
