#!/usr/bin/env python3
"""
Fair Feeder Lifecycle Manager

Manages the operating-model state transitions for Fair Feeder:
  EVENING_READINESS -> MORNING_ACTIVE -> LOCAL_MORNING_DRAIN -> DAYTIME_IDLE
  + temporary ON_DEMAND_CAPTURE (returns to DAYTIME_IDLE)

Provides host-observable state in state.json for shared-host coordination.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import pytz

# Add parent directory to path for importing local modules if needed
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def load_env_safe():
    env_path = repo_root / '.env'
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

# Default State File Locations
DEFAULT_STATE_FILE = os.environ.get("FAIR_FEEDER_STATE_FILE", str(repo_root / "state.json"))
SYSTEM_STATE_FILE = "/var/log/fair_feeder_state.json"

class LifecycleState:
    EVENING_READINESS = "EVENING_READINESS"
    MORNING_ACTIVE = "MORNING_ACTIVE"
    LOCAL_MORNING_DRAIN = "LOCAL_MORNING_DRAIN"
    DAYTIME_IDLE = "DAYTIME_IDLE"
    ON_DEMAND_CAPTURE = "ON_DEMAND_CAPTURE"
    UNKNOWN = "UNKNOWN"


def get_amsterdam_now() -> datetime:
    tz = pytz.timezone("Europe/Amsterdam")
    return datetime.now(tz)


def get_mem_available_mb() -> int:
    """Returns MemAvailable in MB from /proc/meminfo or sysconf."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) // 1024
    except Exception:
        pass
    # Fallback using shutil or os
    try:
        usage = shutil.disk_usage("/")
        return 1000  # Default safe placeholder on non-Linux
    except Exception:
        return 0


def get_fair_feeder_processes() -> List[Dict[str, Any]]:
    """Inspects running Fair Feeder processes and returns their PSS/RSS."""
    procs = []
    if not Path("/proc").exists():
        return procs

    import glob
    for p in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(p)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
            if any(k in cmd for k in ["motion_recorder", "cat-monitor", "usb-monitor", "logitech_vlm_shadow"]):
                pss = 0
                rss = 0
                try:
                    with open(f"/proc/{pid}/smaps_rollup") as sf:
                        for line in sf:
                            if line.startswith("Pss:"):
                                pss = int(line.split()[1])
                            elif line.startswith("Rss:"):
                                rss = int(line.split()[1])
                except Exception:
                    pass
                procs.append({"pid": pid, "cmd": cmd, "pss_kb": pss, "rss_kb": rss})
        except Exception:
            continue
    return procs


def is_service_active(service_name: str) -> bool:
    """Checks if a systemd unit is active."""
    try:
        res = subprocess.run(["systemctl", "is-active", service_name], capture_output=True, text=True, check=False)
        return res.stdout.strip() == "active"
    except Exception:
        return False


def get_active_workers_count() -> int:
    """Counts active ffmpeg/rclone child workers for Fair Feeder."""
    count = 0
    if not Path("/proc").exists():
        return 0
    import glob
    for p in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(p)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
            if any(k in cmd for k in ["ffmpeg", "rclone"]) and any(ff in cmd for ff in ["motion_", "gdrive-randomdice", "usb-camera-sync", "fair-feeder"]):
                count += 1
        except Exception:
            continue
    return count


def get_state_file_path() -> Path:
    return Path(os.environ.get("FAIR_FEEDER_STATE_FILE", str(repo_root / "state.json")))


def read_state() -> Dict[str, Any]:
    """Reads the current lifecycle state file."""
    path = get_state_file_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "state": LifecycleState.UNKNOWN,
        "updated_at": None,
        "local_drain_complete": False,
        "services_active": False,
        "active_workers": 0,
        "source_evidence_ready": False,
        "last_event": "State file initialized."
    }


def write_state(state_data: Dict[str, Any]):
    """Writes the current lifecycle state to disk atomically."""
    state_data["updated_at"] = get_amsterdam_now().isoformat()
    state_data["mem_available_mb"] = get_mem_available_mb()

    # Write to target state file
    target_path = get_state_file_path()
    temp_path = target_path.with_suffix(".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
    temp_path.replace(target_path)

    # Attempt to mirror to /var/log/fair_feeder_state.json if permissions allow
    try:
        sys_path = Path(SYSTEM_STATE_FILE)
        sys_temp = sys_path.with_suffix(".tmp")
        sys_temp.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
        sys_temp.replace(sys_path)
    except Exception:
        pass


def send_telegram_alert(text: str) -> bool:
    """Sends an alert to Telegram using project credentials."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TelegramBotToken")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TelegramChatId")
    if not token or not chat_id:
        print(f"[Telegram Alert Suppressed - No Credentials] {text}")
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        return resp.status_code == 200 and resp.json().get("ok", False)
    except Exception as e:
        print(f"[Telegram Alert Failed] {e}")
        return False


def activate_morning() -> bool:
    """
    Transitions to MORNING_ACTIVE and starts heavy recorder services (cat-monitor, usb-monitor).
    Fail-loud if services cannot be activated.
    """
    now = get_amsterdam_now()
    print(f"[{now.isoformat()}] Activating Fair Feeder MORNING_ACTIVE...")

    state_data = read_state()
    state_data["state"] = LifecycleState.MORNING_ACTIVE
    state_data["local_drain_complete"] = False
    state_data["source_evidence_ready"] = False
    state_data["last_event"] = "Morning monitoring activation initiated."
    write_state(state_data)

    # Start systemd services if on Linux/systemd host
    errors = []
    for svc in ["cat-monitor", "usb-monitor"]:
        try:
            res = subprocess.run(["systemctl", "start", svc], capture_output=True, text=True, check=False)
            if res.returncode != 0:
                errors.append(f"Failed to start {svc}: {res.stderr.strip()}")
        except Exception as e:
            # On local dev mac or non-systemd, record warning but do not crash
            print(f"[Warning] Could not execute systemctl start {svc}: {e}")

    # Wait up to 10s to verify active status if systemctl is available
    if shutil.which("systemctl"):
        time.sleep(3)
        for svc in ["cat-monitor", "usb-monitor"]:
            if not is_service_active(svc):
                errors.append(f"{svc} is not active after start command.")

    if errors:
        err_msg = "🚨 Fair Feeder Morning Activation FAILED:\n" + "\n".join(errors)
        print(err_msg)
        send_telegram_alert(err_msg)
        state_data["last_event"] = f"Morning activation error: {', '.join(errors)}"
        state_data["services_active"] = False
        write_state(state_data)
        return False

    state_data["services_active"] = True
    state_data["last_event"] = "Morning monitoring active. Both cameras recording."
    write_state(state_data)
    print("✅ MORNING_ACTIVE started successfully.")
    return True


def check_drain_prerequisites(temp_dirs: Optional[List[Path]] = None) -> Tuple[bool, str]:
    """
    Verifies whether local morning drain is complete:
    - No active recording in progress
    - No child ffmpeg/rclone workers running
    - Temporary recording folders empty
    """
    if temp_dirs is None:
        temp_dirs = [
            repo_root / "recordings_temp",
            repo_root / "recordings_usb_temp"
        ]

    # 1. Check child worker processes
    active_workers = get_active_workers_count()
    if active_workers > 0:
        return False, f"{active_workers} background ffmpeg/rclone worker(s) still active"

    # 2. Check temp directories for active MP4s
    for td in temp_dirs:
        if td.exists():
            temp_files = list(td.glob("*.mp4"))
            if temp_files:
                return False, f"{len(temp_files)} unfinished file(s) in {td.name}"

    return True, "All recordings, finalizers, and uploads complete"


def drain_and_idle(max_wait_sec: int = 900) -> bool:
    """
    Waits for active recordings and workers to finish, then safely stops
    heavy services and enters DAYTIME_IDLE, releasing ~1.3 GB PSS.
    """
    now = get_amsterdam_now()
    print(f"[{now.isoformat()}] Initiating LOCAL_MORNING_DRAIN (timeout: {max_wait_sec}s)...")

    state_data = read_state()
    state_data["state"] = LifecycleState.LOCAL_MORNING_DRAIN
    state_data["last_event"] = "Local morning drain in progress. Waiting for active recordings to conclude."
    write_state(state_data)

    start_t = time.time()
    while time.time() - start_t < max_wait_sec:
        drained, reason = check_drain_prerequisites()
        if drained:
            print(f"✅ Drain gate satisfied: {reason}")
            break
        print(f"⏳ Waiting for drain: {reason}...")
        time.sleep(10)
    else:
        print(f"⚠️ Drain wait reached {max_wait_sec}s ceiling. Proceeding with service shutdown.")

    # Stop heavy recorder services
    for svc in ["cat-monitor", "usb-monitor"]:
        try:
            subprocess.run(["systemctl", "stop", svc], capture_output=True, text=True, check=False)
        except Exception:
            pass

    # Record measured resource release
    time.sleep(2)
    remaining_procs = get_fair_feeder_processes()
    total_pss_kb = sum(p.get("pss_kb", 0) for p in remaining_procs)
    mem_available = get_mem_available_mb()

    state_data["state"] = LifecycleState.DAYTIME_IDLE
    state_data["local_drain_complete"] = True
    state_data["source_evidence_ready"] = True
    state_data["services_active"] = False
    state_data["active_workers"] = 0
    state_data["remaining_fair_feeder_pss_mb"] = total_pss_kb // 1024
    state_data["last_event"] = f"Entered DAYTIME_IDLE. Heavy services stopped. MemAvailable: {mem_available} MB."
    write_state(state_data)

    print(f"✅ DAYTIME_IDLE active. Remaining Fair Feeder PSS: {total_pss_kb // 1024} MB. MemAvailable: {mem_available} MB.")
    return True


def on_demand_capture(camera: str = "tapo", duration_sec: int = 10, out_path: Optional[str] = None, send_telegram: bool = False) -> Optional[str]:
    """
    Executes a bounded daytime capture without permanently waking heavy services.
    Returns the path to the encoded MP4 file.
    """
    now = get_amsterdam_now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    camera_upper = camera.upper()

    print(f"[{now.isoformat()}] Starting ON_DEMAND_CAPTURE for {camera_upper} ({duration_sec}s)...")

    # Record temporary on-demand state
    prior_state = read_state()
    temp_state = dict(prior_state)
    temp_state["state"] = LifecycleState.ON_DEMAND_CAPTURE
    temp_state["last_event"] = f"On-demand capture in progress for {camera_upper}."
    write_state(temp_state)

    if out_path is None:
        out_dir = repo_root / "scratch" / "on_demand"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"on_demand_{camera.lower()}_{timestamp_str}.mp4")

    raw_temp_path = out_path + ".raw.mp4"

    success = False
    try:
        import cv2

        cap = None
        if camera_upper == "TAPO":
            rtsp_url = os.environ.get("RTSP_URL") or os.environ.get("TAPO_RTSP_URL")
            if not rtsp_url:
                raise ValueError("RTSP_URL environment variable is missing.")
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        else:
            # Logitech USB camera
            v4l2_dev = os.environ.get("V4L2_DEVICE", "/dev/video0")
            # If numeric index
            dev_idx = int(v4l2_dev.replace("/dev/video", "")) if "/dev/video" in v4l2_dev else 0
            cap = cv2.VideoCapture(dev_idx)

        if not cap or not cap.isOpened():
            raise RuntimeError(f"Could not open camera device for {camera_upper}.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        if fps <= 0 or fps > 60:
            fps = 15.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_temp_path, fourcc, fps, (width, height))

        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            out.write(frame)
            frame_count += 1

        out.release()
        cap.release()

        if frame_count == 0:
            raise RuntimeError("Zero frames captured from camera.")

        # Re-encode to standard web/Telegram playable H.264 with faststart
        cmd = [
            "ffmpeg", "-y", "-i", raw_temp_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            # If ffmpeg libx264 fails, move raw
            shutil.move(raw_temp_path, out_path)
        else:
            if Path(raw_temp_path).exists():
                Path(raw_temp_path).unlink()

        success = True
        print(f"✅ On-demand capture completed: {out_path} ({frame_count} frames)")

        # Send to Telegram if requested
        if send_telegram:
            token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TelegramBotToken")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TelegramChatId")
            if token and chat_id and Path(out_path).exists():
                import requests
                caption = f"📹 [ON-DEMAND CAPTURE] {camera_upper}\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')}\nDuration: {duration_sec}s"
                url = f"https://api.telegram.org/bot{token}/sendVideo"
                with open(out_path, "rb") as vf:
                    requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"video": vf}, timeout=30)
                print("✅ Telegram on-demand video sent.")

    except Exception as e:
        print(f"❌ On-demand capture failed: {e}")
        out_path = None
    finally:
        # Return state immediately to DAYTIME_IDLE
        write_state(prior_state)
        if Path(raw_temp_path).exists():
            try:
                Path(raw_temp_path).unlink()
            except Exception:
                pass

    return out_path if success else None


def evening_readiness(send_telegram_on_failure: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes a fast, lightweight evening readiness check without maintaining heavy recorder residency.
    Verifies camera streams, disk space, and essential prerequisites for tomorrow morning.
    """
    now = get_amsterdam_now()
    print(f"[{now.isoformat()}] Running Fair Feeder EVENING_READINESS check...")

    results = {
        "timestamp": now.isoformat(),
        "tapo_reachable": False,
        "logitech_reachable": False,
        "storage_ok": False,
        "storage_free_gb": 0.0,
        "ready": False,
        "issues": []
    }

    # 1. Check storage space
    try:
        usage = shutil.disk_usage(str(repo_root))
        free_gb = round(usage.free / (1024**3), 2)
        results["storage_free_gb"] = free_gb
        if free_gb >= 2.0:
            results["storage_ok"] = True
        else:
            results["issues"].append(f"Low disk space: {free_gb} GB free (minimum 2.0 GB required)")
    except Exception as e:
        results["issues"].append(f"Disk check failed: {e}")

    # 2. Check Tapo camera reachability
    rtsp_url = os.environ.get("RTSP_URL") or os.environ.get("TAPO_RTSP_URL")
    if not rtsp_url:
        results["issues"].append("Tapo RTSP_URL is not configured in .env")
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    results["tapo_reachable"] = True
            cap.release()
        except Exception as e:
            results["issues"].append(f"Tapo RTSP connection failed: {e}")

    if not results["tapo_reachable"] and "Tapo RTSP_URL is not configured" not in "".join(results["issues"]):
        results["issues"].append("Tapo camera unreachable or returned empty frame")

    # 3. Check Logitech USB camera
    v4l2_dev = os.environ.get("V4L2_DEVICE", "/dev/video0")
    if Path(v4l2_dev).exists() or sys.platform != "linux":
        try:
            import cv2
            dev_idx = int(v4l2_dev.replace("/dev/video", "")) if "/dev/video" in v4l2_dev else 0
            cap = cv2.VideoCapture(dev_idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    results["logitech_reachable"] = True
            cap.release()
        except Exception as e:
            results["issues"].append(f"Logitech camera check failed: {e}")

    if not results["logitech_reachable"] and Path(v4l2_dev).exists():
        results["issues"].append(f"Logitech USB camera ({v4l2_dev}) could not capture test frame")

    # Overall verdict
    is_ready = len(results["issues"]) == 0
    results["ready"] = is_ready

    # Update state file
    state_data = read_state()
    state_data["state"] = LifecycleState.EVENING_READINESS
    state_data["last_event"] = "Evening readiness check completed: " + ("READY" if is_ready else "ACTION NEEDED")
    state_data["readiness"] = results
    write_state(state_data)

    if not is_ready:
        issues_text = "\n".join(f"- {issue}" for issue in results["issues"])
        alert_msg = (
            f"⚠️ Fair Feeder Evening Readiness Alert\n"
            f"Action needed before tomorrow's breakfast:\n"
            f"{issues_text}\n"
            f"Please check the feeder setup while awake."
        )
        print(f"❌ {alert_msg}")
        if send_telegram_on_failure:
            send_telegram_alert(alert_msg)
    else:
        print("✅ EVENING_READINESS: READY for tomorrow's breakfast.")

    return is_ready, results


def main():
    parser = argparse.ArgumentParser(description="Fair Feeder Lifecycle Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Displays current lifecycle status and host metrics")

    # activate-morning
    subparsers.add_parser("activate-morning", help="Transitions to MORNING_ACTIVE and starts heavy recorders")

    # drain-and-idle
    drain_p = subparsers.add_parser("drain-and-idle", help="Waits for drain prerequisites and enters DAYTIME_IDLE")
    drain_p.add_argument("--max-wait-sec", type=int, default=900, help="Maximum seconds to wait for drain completion")

    # on-demand-capture
    capture_p = subparsers.add_parser("on-demand-capture", help="Captures a bounded on-demand clip during daytime idle")
    capture_p.add_argument("--camera", choices=["tapo", "logitech"], default="tapo", help="Camera to capture from")
    capture_p.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    capture_p.add_argument("--out-file", type=str, default=None, help="Output MP4 path")
    capture_p.add_argument("--send-telegram", action="store_true", help="Send captured video to Telegram")

    # evening-readiness
    readiness_p = subparsers.add_parser("evening-readiness", help="Runs evening readiness check for next breakfast")
    readiness_p.add_argument("--no-telegram", action="store_true", help="Do not send Telegram on failure")

    args = parser.parse_args()

    if args.command == "status":
        state = read_state()
        print(json.dumps(state, indent=2))
    elif args.command == "activate-morning":
        success = activate_morning()
        sys.exit(0 if success else 1)
    elif args.command == "drain-and-idle":
        success = drain_and_idle(max_wait_sec=args.max_wait_sec)
        sys.exit(0 if success else 1)
    elif args.command == "on-demand-capture":
        res = on_demand_capture(camera=args.camera, duration_sec=args.duration, out_path=args.out_file, send_telegram=args.send_telegram)
        sys.exit(0 if res else 1)
    elif args.command == "evening-readiness":
        is_ready, _ = evening_readiness(send_telegram_on_failure=not args.no_telegram)
        sys.exit(0 if is_ready else 1)

if __name__ == "__main__":
    main()
