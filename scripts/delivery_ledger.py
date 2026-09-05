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


DELIVERY_REGISTRY_FILENAME = "delivery_registry.json"


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


def init_registry_data() -> Dict[str, Any]:
    return {
        "version": 2,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dates": {}
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


def load_delivery_registry(
    drive_service: Any,
    folder_id: str,
    local_fallback_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Loads the durable delivery registry from Google Drive or local fallback."""
    raw_bytes = load_durable_artifact(drive_service, folder_id, DELIVERY_REGISTRY_FILENAME, local_fallback_dir=local_fallback_dir)
    if raw_bytes:
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(data, dict) and "dates" in data:
                return data
        except Exception as e:
            print(f"[DeliveryLedger] Error parsing {DELIVERY_REGISTRY_FILENAME}: {e}")
    return init_registry_data()


def save_delivery_registry(
    drive_service: Any,
    folder_id: str,
    registry_data: Dict[str, Any],
    local_fallback_dir: Optional[Path] = None
) -> Optional[str]:
    """Persists the durable delivery registry to Google Drive and optional local fallback."""
    registry_data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    content_bytes = json.dumps(registry_data, indent=2).encode("utf-8")
    return save_durable_artifact(
        drive_service,
        folder_id,
        DELIVERY_REGISTRY_FILENAME,
        content_bytes,
        mime_type="application/json",
        local_fallback_dir=local_fallback_dir
    )


def is_breakfast_fully_delivered(
    arg1: Any,
    arg2: Any,
    target_date: Optional[str] = None,
    local_fallback_dir: Optional[Path] = None
) -> bool:
    """
    Returns True if the entire breakfast for target_date is fully delivered.
    Supports two calling signatures:
    1. is_breakfast_fully_delivered(registry_data, target_date)
    2. is_breakfast_fully_delivered(drive_service, folder_id, target_date, local_fallback_dir=...)
    """
    if isinstance(arg1, dict):
        registry_data = arg1
        clean_date = str(arg2).replace("-", "").strip()
    else:
        drive_service = arg1
        folder_id = arg2
        clean_date = str(target_date).replace("-", "").strip()
        registry_data = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_fallback_dir)

    date_info = registry_data.get("dates", {}).get(clean_date, {})
    if date_info.get("breakfast_fully_delivered", False):
        return True
    unified = date_info.get("unified", {})
    if unified.get("fully_delivered", False):
        return True
    if unified.get("items", {}).get("summary", {}).get("delivered", False):
        return True
    cameras = date_info.get("cameras", {})
    tapo_ok = cameras.get("TAPO", {}).get("camera_fully_delivered", False)
    logi_ok = cameras.get("LOGITECH", {}).get("camera_fully_delivered", False)
    return bool(tapo_ok and logi_ok)


def commit_breakfast_completion(
    drive_service: Any,
    folder_id: str,
    target_date: str,
    extra: Optional[Dict[str, Any]] = None,
    local_fallback_dir: Optional[Path] = None
) -> bool:
    """Marks entire breakfast as completed in the registry and commits to Drive."""
    clean_date = str(target_date).replace("-", "").strip()
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_fallback_dir)
    if "dates" not in registry:
        registry["dates"] = {}
    if clean_date not in registry["dates"]:
        registry["dates"][clean_date] = {"cameras": {}, "items": {}}

    registry["dates"][clean_date]["breakfast_fully_delivered"] = True
    registry["dates"][clean_date]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if extra:
        registry["dates"][clean_date].update(extra)

    save_delivery_registry(drive_service, folder_id, registry, local_fallback_dir=local_fallback_dir)
    return True


def load_delivery_ledger(
    drive_service: Any,
    folder_id: str,
    target_date: str,
    camera: str,
    local_fallback_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Loads delivery ledger for (target_date, camera), checking registry first."""
    clean_date = str(target_date).replace("-", "").strip()
    clean_cam = str(camera).upper().strip()

    # 1. Check registry
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_fallback_dir)
    dates = registry.get("dates", {})
    if clean_date in dates:
        date_entry = dates[clean_date]
        if date_entry.get("breakfast_fully_delivered", False):
            cam_entry = date_entry.get("cameras", {}).get(clean_cam, {})
            cam_entry.setdefault("date", clean_date)
            cam_entry.setdefault("camera", clean_cam)
            cam_entry["camera_fully_delivered"] = True
            cam_entry["analysis_completed"] = True
            return cam_entry

        cam_data = date_entry.get("cameras", {}).get(clean_cam)
        if cam_data and isinstance(cam_data, dict):
            cam_data.setdefault("date", clean_date)
            cam_data.setdefault("camera", clean_cam)
            return cam_data

    # 2. Check local/individual ledger file
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


def save_delivery_ledger(
    drive_service: Any,
    folder_id: str,
    ledger_data: Dict[str, Any],
    local_fallback_dir: Optional[Path] = None
) -> Optional[str]:
    """Persists delivery ledger into registry and local file."""
    target_date = ledger_data.get("date", "")
    camera = ledger_data.get("camera", "")
    clean_date = str(target_date).replace("-", "").strip()
    clean_cam = str(camera).upper().strip()

    # 1. Update registry
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_fallback_dir)
    if "dates" not in registry:
        registry["dates"] = {}
    if clean_date not in registry["dates"]:
        registry["dates"][clean_date] = {
            "breakfast_fully_delivered": False,
            "cameras": {},
            "items": {}
        }
    registry["dates"][clean_date].setdefault("cameras", {})
    registry["dates"][clean_date]["cameras"][clean_cam] = ledger_data

    res_reg = save_delivery_registry(drive_service, folder_id, registry, local_fallback_dir=local_fallback_dir)

    # 2. Also save individual file locally for backward compatibility
    filename = get_ledger_filename(target_date, camera)
    ledger_data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    content_bytes = json.dumps(ledger_data, indent=2).encode("utf-8")
    save_durable_artifact(drive_service, folder_id, filename, content_bytes, mime_type="application/json", local_fallback_dir=local_fallback_dir)

    return res_reg


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
    Writes to deterministic paths (/tmp/output/, local_dir) and registry.
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

    # 1. Update registry
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_dir)
    if "dates" not in registry:
        registry["dates"] = {}
    if clean_date not in registry["dates"]:
        registry["dates"][clean_date] = {"cameras": {}, "items": {}}
    registry["dates"][clean_date]["tapo_timeline"] = timeline_data
    save_delivery_registry(drive_service, folder_id, registry, local_fallback_dir=local_dir)

    # 2. Write to deterministic /tmp/output/ if possible
    tmp_out = Path("/tmp/output")
    try:
        tmp_out.mkdir(parents=True, exist_ok=True)
        (tmp_out / filename).write_bytes(content_bytes)
    except Exception:
        pass

    # 3. Write to local_dir
    if local_dir:
        try:
            p = Path(local_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / filename).write_bytes(content_bytes)
        except Exception as e:
            print(f"[DeliveryLedger] Warning writing local timeline: {e}")

    # 4. Also write to repo root (.) for papermill cwd compatibility
    try:
        Path(filename).write_bytes(content_bytes)
    except Exception:
        pass

    return timeline_data


def load_tapo_timeline(
    drive_service: Any,
    folder_id: str,
    target_date: str,
    local_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Loads durable TAPO timeline from local artifact paths, registry, or Google Drive."""
    clean_date = str(target_date).replace("-", "").strip()
    filename = f"tapo_timeline_{clean_date}.json"

    # Search paths in order
    search_dirs = []
    if local_dir:
        search_dirs.append(Path(local_dir))
    search_dirs.extend([
        Path("/tmp/tapo_timeline_artifact"),
        Path("/tmp/output"),
        Path(".")
    ])
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[DeliveryLedger] Error reading local timeline {candidate}: {e}")

    # Fall back to registry
    registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_dir)
    date_info = registry.get("dates", {}).get(clean_date, {})
    if "tapo_timeline" in date_info:
        return date_info["tapo_timeline"]

    # Direct Drive download fallback
    raw_bytes = load_durable_artifact(drive_service, folder_id, filename, local_fallback_dir=local_dir)
    if raw_bytes:
        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            pass

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Delivery Ledger & Registry CLI Preflight Guard")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_check = subparsers.add_parser("check-preflight", help="Check if delivery is already completed")
    p_check.add_argument("--date", required=True, help="Target date YYYYMMDD")
    p_check.add_argument("--camera", default=None, choices=["TAPO", "LOGITECH"], help="Camera name")
    p_check.add_argument("--breakfast", action="store_true", help="Check entire breakfast delivery status")
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

        clean_date = str(args.date).replace("-", "").strip()
        local_fallback = Path(args.local_dir) if args.local_dir else None

        registry = load_delivery_registry(drive_service, folder_id, local_fallback_dir=local_fallback)
        is_breakfast_done = is_breakfast_fully_delivered(registry, clean_date)

        if args.breakfast or not args.camera:
            is_delivered = is_breakfast_done
        else:
            ledger = load_delivery_ledger(drive_service, folder_id, clean_date, args.camera, local_fallback_dir=local_fallback)
            is_delivered = is_breakfast_done or is_camera_fully_delivered(ledger)

        print(f"[Preflight] date={clean_date} camera={args.camera} breakfast_done={is_breakfast_done} is_delivered={is_delivered}")
        if is_delivered:
            print("ALREADY_DELIVERED=true")
            sys.exit(0)
        else:
            print("ALREADY_DELIVERED=false")
            sys.exit(1)


if __name__ == "__main__":
    main()

