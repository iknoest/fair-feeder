#!/usr/bin/env python3
"""
Fair Feeder Delivery Ledger

Provides durable item-level delivery idempotency across Google Drive and local runners.
Tracks individual user-visible Telegram artifacts (summary, photos, videos) per (TARGET_DATE, CAMERA).
Ensures that partial-delivery failures resume item-by-item without sending duplicate Telegram messages.
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from googleapiclient.http import MediaIoBaseDownload, MediaInMemoryUpload
except ImportError:
    MediaIoBaseDownload = None
    MediaInMemoryUpload = None


def get_ledger_filename(target_date: str, camera: str) -> str:
    clean_date = str(target_date).replace("-", "").strip()
    clean_cam = str(camera).upper().strip()
    return f"delivery_ledger_{clean_date}_{clean_cam}.json"


def init_ledger_data(target_date: str, camera: str) -> Dict[str, Any]:
    clean_date = str(target_date).replace("-", "").strip()
    clean_cam = str(camera).upper().strip()
    return {
        "date": clean_date,
        "camera": clean_cam,
        "analysis_completed": False,
        "camera_fully_delivered": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": {}
    }


def _find_drive_file_id(drive_service: Any, folder_id: str, filename: str) -> Optional[str]:
    if not drive_service or not folder_id:
        return None
    try:
        q = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
        res = drive_service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
    except Exception as e:
        print(f"[DeliveryLedger] Warning searching Drive for {filename}: {e}")
    return None


def load_durable_artifact(drive_service: Any, folder_id: str, filename: str, local_fallback_dir: Optional[Path] = None) -> Optional[bytes]:
    """Loads a file from Google Drive, falling back to local directory if provided."""
    if drive_service and folder_id:
        file_id = _find_drive_file_id(drive_service, folder_id, filename)
        if file_id:
            try:
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, drive_service.files().get_media(fileId=file_id))
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return buf.getvalue()
            except Exception as e:
                print(f"[DeliveryLedger] Warning downloading {filename} from Drive: {e}")

    if local_fallback_dir:
        local_path = Path(local_fallback_dir) / filename
        if local_path.exists():
            try:
                return local_path.read_bytes()
            except Exception as e:
                print(f"[DeliveryLedger] Warning reading local file {local_path}: {e}")

    return None


def save_durable_artifact(drive_service: Any, folder_id: str, filename: str, content_bytes: bytes, mime_type: str = "application/json", local_fallback_dir: Optional[Path] = None) -> Optional[str]:
    """Saves or updates a file in Google Drive, and saves to local directory if provided."""
    file_id = None
    if drive_service and folder_id and MediaInMemoryUpload:
        try:
            existing_id = _find_drive_file_id(drive_service, folder_id, filename)
            media = MediaInMemoryUpload(content_bytes, mimetype=mime_type, resumable=False)
            if existing_id:
                res = drive_service.files().update(fileId=existing_id, media_body=media, fields="id").execute()
                file_id = res.get("id")
            else:
                body = {"name": filename, "parents": [folder_id]}
                res = drive_service.files().create(body=body, media_body=media, fields="id").execute()
                file_id = res.get("id")
        except Exception as e:
            print(f"[DeliveryLedger] Warning saving {filename} to Drive: {e}")

    if local_fallback_dir:
        try:
            local_path = Path(local_fallback_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content_bytes)
        except Exception as e:
            print(f"[DeliveryLedger] Warning saving local file {filename}: {e}")

    return file_id


def load_delivery_ledger(drive_service: Any, folder_id: str, target_date: str, camera: str, local_fallback_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Loads existing delivery ledger for (target_date, camera), or initializes empty template."""
    filename = get_ledger_filename(target_date, camera)
    raw_bytes = load_durable_artifact(drive_service, folder_id, filename, local_fallback_dir=local_fallback_dir)
    if raw_bytes:
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(data, dict) and "items" in data:
                return data
        except Exception as e:
            print(f"[DeliveryLedger] Error parsing ledger {filename}: {e}")

    return init_ledger_data(target_date, camera)


def save_delivery_ledger(drive_service: Any, folder_id: str, ledger_data: Dict[str, Any], local_fallback_dir: Optional[Path] = None) -> Optional[str]:
    """Persists delivery ledger to Google Drive and optional local fallback."""
    target_date = ledger_data.get("date", "")
    camera = ledger_data.get("camera", "")
    filename = get_ledger_filename(target_date, camera)
    ledger_data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    content_bytes = json.dumps(ledger_data, indent=2).encode("utf-8")
    return save_durable_artifact(drive_service, folder_id, filename, content_bytes, mime_type="application/json", local_fallback_dir=local_fallback_dir)


def is_item_delivered(ledger_data: Dict[str, Any], item_key: str) -> bool:
    """Checks whether a specific artifact has already been delivered to Telegram."""
    items = ledger_data.get("items", {})
    item_info = items.get(item_key, {})
    return bool(item_info.get("delivered", False))


def record_item_delivered(
    drive_service: Any,
    folder_id: str,
    ledger_data: Dict[str, Any],
    item_key: str,
    message_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    local_fallback_dir: Optional[Path] = None
) -> None:
    """Marks an item as delivered with its message_id and immediately commits to Drive."""
    now_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "delivered": True,
        "delivered_at_utc": now_utc,
        "message_id": message_id
    }
    if extra:
        entry.update(extra)

    if "items" not in ledger_data:
        ledger_data["items"] = {}
    ledger_data["items"][item_key] = entry
    save_delivery_ledger(drive_service, folder_id, ledger_data, local_fallback_dir=local_fallback_dir)


def is_camera_fully_delivered(ledger_data: Dict[str, Any]) -> bool:
    """Returns True if all required delivery items for this camera have completed."""
    return bool(ledger_data.get("camera_fully_delivered", False))


def commit_camera_completion(
    drive_service: Any,
    folder_id: str,
    ledger_data: Dict[str, Any],
    required_items: Optional[List[str]] = None,
    local_fallback_dir: Optional[Path] = None
) -> bool:
    """
    Evaluates required items. If all delivered, sets camera_fully_delivered=True
    and commits to Drive. Returns True if fully delivered.
    """
    items = ledger_data.get("items", {})
    if required_items:
        all_ok = all(items.get(k, {}).get("delivered", False) for k in required_items)
    else:
        # Default: if at least summary is delivered and all recorded items are delivered
        all_ok = items.get("summary", {}).get("delivered", False) and all(v.get("delivered", False) for v in items.values())

    if all_ok:
        ledger_data["camera_fully_delivered"] = True
        ledger_data["analysis_completed"] = True
        save_delivery_ledger(drive_service, folder_id, ledger_data, local_fallback_dir=local_fallback_dir)
        return True
    return False


def export_tapo_timeline(
    drive_service: Any,
    folder_id: str,
    target_date: str,
    feeding_phases: List[Dict[str, Any]],
    local_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Exports durable TAPO feeding timeline using existing accepted FeedingTracker phase semantics.
    Must be persisted BEFORE TAPO reaches terminal delivery completion.
    """
    clean_date = str(target_date).replace("-", "").strip()
    filename = f"tapo_timeline_{clean_date}.json"
    timeline_data = {
        "date": clean_date,
        "camera": "TAPO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feeding_phases": feeding_phases
    }
    content_bytes = json.dumps(timeline_data, indent=2).encode("utf-8")
    save_durable_artifact(drive_service, folder_id, filename, content_bytes, mime_type="application/json", local_fallback_dir=local_dir)
    return timeline_data


def load_tapo_timeline(
    drive_service: Any,
    folder_id: str,
    target_date: str,
    local_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Loads durable TAPO timeline from Google Drive or local directory."""
    clean_date = str(target_date).replace("-", "").strip()
    filename = f"tapo_timeline_{clean_date}.json"
    raw_bytes = load_durable_artifact(drive_service, folder_id, filename, local_fallback_dir=local_dir)
    if raw_bytes:
        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            print(f"[DeliveryLedger] Error parsing {filename}: {e}")
    return None


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Delivery Ledger CLI & Preflight Guard")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: check-preflight
    p_check = subparsers.add_parser("check-preflight", help="Check if camera is already fully delivered")
    p_check.add_argument("--date", required=True, help="Target date YYYYMMDD")
    p_check.add_argument("--camera", required=True, choices=["TAPO", "LOGITECH"], help="Camera name")
    p_check.add_argument("--local-dir", default=None, help="Local directory fallback")

    args = parser.parse_args()

    if args.subcommand == "check-preflight":
        drive_service = None
        folder_id = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID") or os.environ.get("GDRIVE_LOGITECH_FOLDER_ID")
        gdrive_key = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
        if gdrive_key:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                creds = service_account.Credentials.from_service_account_info(
                    json.loads(gdrive_key), scopes=["https://www.googleapis.com/auth/drive"]
                )
                drive_service = build("drive", "v3", credentials=creds)
            except Exception as e:
                print(f"[Preflight] Warning initializing Drive service: {e}")

        ledger = load_delivery_ledger(drive_service, folder_id, args.date, args.camera, local_fallback_dir=Path(args.local_dir) if args.local_dir else None)
        is_delivered = is_camera_fully_delivered(ledger)
        print(f"[Preflight] date={args.date} camera={args.camera} fully_delivered={is_delivered}")
        if is_delivered:
            print(f"ALREADY_DELIVERED=true")
            sys.exit(0)
        else:
            print(f"ALREADY_DELIVERED=false")
            sys.exit(1)


if __name__ == "__main__":
    main()

